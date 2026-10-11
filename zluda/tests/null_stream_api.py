#!/usr/bin/env python3
"""Native CUDA API regression fixtures; run only on a machine with HIP/GPU.

python3 zluda/tests/null_stream_api.py --library /absolute/path/libnvcuda.so
Each case gets a fresh process, all ZLUDA_* variables removed, and a timeout.
The polling kernel also has a GPU-side deadline, so a regression does not
leave a permanently spinning kernel behind when the host process is killed.
"""
import argparse
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


PTX = b""".version 6.5
.target sm_30
.address_size 64
.visible .entry produce(.param .u64 flag) {
 .reg .u64 addr;
 .reg .u32 old;
 ld.param.u64 addr, [flag];
 atom.global.exch.b32 old, [addr], 1;
 ret;
}
.visible .entry poll(.param .u64 flag, .param .u64 output) {
 .reg .u64 addr, dst;
 .reg .u32 seen, start, now, elapsed;
 .reg .pred ready, expired;
 ld.param.u64 addr, [flag];
 ld.param.u64 dst, [output];
 mov.u32 start, %clock;
loop:
 atom.global.add.u32 seen, [addr], 0;
 setp.ne.u32 ready, seen, 0;
 @ready bra done;
 mov.u32 now, %clock;
 sub.u32 elapsed, now, start;
 setp.ge.u32 expired, elapsed, 3000000000;
 @!expired bra loop;
done:
 st.global.u32 [dst], seen;
 ret;
}
.visible .entry step(.param .u64 input, .param .u64 output, .param .u64 delay) {
 .reg .u64 src, dst, cycles, elapsed64;
 .reg .u32 value, start, now, elapsed;
 .reg .pred waiting;
 ld.param.u64 src, [input];
 ld.param.u64 dst, [output];
 ld.param.u64 cycles, [delay];
 mov.u32 start, %clock;
wait:
 mov.u32 now, %clock;
 sub.u32 elapsed, now, start;
 cvt.u64.u32 elapsed64, elapsed;
 setp.lt.u64 waiting, elapsed64, cycles;
 @waiting bra wait;
 ld.global.u32 value, [src];
 add.u32 value, value, 1;
 st.global.u32 [dst], value;
 ret;
}
"""

CASES = (
    "flags", "nonblocking_before_legacy", "nonblocking_after_legacy",
    "blocking_order", "cross_thread_null", "event_order",
    "per_thread_handle", "per_thread_entry_point",
)


class Driver:
    def __init__(self, library):
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.call("cuInit", [C.c_uint], 0)
        self.context = C.c_void_p()
        self.call("cuCtxCreate_v2", [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
                  C.byref(self.context), 0, 0)
        self.module = C.c_void_p()
        text = C.create_string_buffer(PTX)
        self.call("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p],
                  C.byref(self.module), C.cast(text, C.c_void_p))
        self.functions = {}
        for name in ("produce", "poll", "step"):
            function = C.c_void_p()
            self.call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                      C.byref(function), self.module, name.encode())
            self.functions[name] = function
        self.allocations, self.streams, self.events = [], [], []
        # Warm compilation before a deadline-sensitive poll. No gate is held.
        flag, output = self.alloc(), self.alloc()
        self.launch("produce", None, flag)
        self.launch("poll", None, flag, output)
        self.launch("step", None, flag, output, 0)
        self.sync(None)

    def call(self, name, types, *args):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = types, C.c_int
        result = fn(*args)
        if result:
            raise RuntimeError(f"{name} returned {result}")

    def alloc(self):
        pointer = C.c_uint64()
        self.call("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(pointer), 4)
        self.allocations.append(pointer)
        self.memset(pointer, 0, None)
        return pointer

    def stream(self, flags, priority=False):
        stream = C.c_void_p()
        types, args = [C.POINTER(C.c_void_p), C.c_uint], [C.byref(stream), flags]
        if priority:
            types.append(C.c_int)
            args.append(0)
        self.call("cuStreamCreateWithPriority" if priority else "cuStreamCreate", types, *args)
        self.streams.append(stream)
        return stream

    def flags(self, stream):
        actual = C.c_uint(0xFFFFFFFF)
        self.call("cuStreamGetFlags", [C.c_void_p, C.POINTER(C.c_uint)], stream, C.byref(actual))
        return actual.value

    def event(self):
        event = C.c_void_p()
        self.call("cuEventCreate", [C.POINTER(C.c_void_p), C.c_uint], C.byref(event), 2)
        self.events.append(event)
        return event

    def launch(self, name, stream, *values, ptsz=False):
        values = [v if isinstance(v, C.c_uint64) else C.c_uint64(v) for v in values]
        params = (C.c_void_p * len(values))(*[C.cast(C.byref(v), C.c_void_p).value for v in values])
        self.call("cuLaunchKernel_ptsz" if ptsz else "cuLaunchKernel",
                  [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p],
                  self.functions[name], 1, 1, 1, 1, 1, 1, 0, stream, params, None)

    def sync(self, stream):
        self.call("cuStreamSynchronize", [C.c_void_p], stream)

    def memset(self, pointer, value, stream):
        self.call("cuMemsetD32Async", [C.c_uint64, C.c_uint, C.c_size_t, C.c_void_p],
                  pointer, value, 1, stream)

    def copy(self, dst, src, stream):
        self.call("cuMemcpyDtoDAsync_v2", [C.c_uint64, C.c_uint64, C.c_size_t, C.c_void_p],
                  dst, src, 4, stream)

    def read(self, pointer):
        output = C.c_uint()
        self.call("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t],
                  C.byref(output), pointer, 4)
        return output.value

    def record(self, event, stream):
        self.call("cuEventRecord", [C.c_void_p, C.c_void_p], event, stream)

    def wait(self, stream, event):
        self.call("cuStreamWaitEvent", [C.c_void_p, C.c_void_p, C.c_uint], stream, event, 0)

    def close(self):
        self.call("cuCtxSynchronize", [])
        for event in self.events:
            self.call("cuEventDestroy_v2", [C.c_void_p], event)
        for stream in self.streams:
            self.call("cuStreamDestroy_v2", [C.c_void_p], stream)
        for pointer in self.allocations:
            self.call("cuMemFree_v2", [C.c_uint64], pointer)
        self.call("cuModuleUnload", [C.c_void_p], self.module)
        self.call("cuCtxDestroy_v2", [C.c_void_p], self.context)


def run_case(d, case):
    if case == "flags":
        for priority in (False, True):
            for flags in (0, 1):
                assert d.flags(d.stream(flags, priority)) == flags
    elif case.startswith("nonblocking_"):
        for priority in (False, True):
            stream = d.stream(1, priority)
            # Test both implicit zero and the explicit CU_STREAM_LEGACY handle.
            for legacy in (None, 1):
                flag, output = d.alloc(), d.alloc()
                d.sync(None)
                before = case == "nonblocking_before_legacy"
                d.launch("poll", stream if before else legacy, flag, output)
                d.launch("produce", legacy if before else stream, flag)
                d.sync(stream)
                d.sync(legacy)
                assert d.read(output) == 1, "nonblocking stream was included in a legacy barrier"
    elif case == "blocking_order":
        for priority in (False, True):
            stream = d.stream(0, priority)
            for legacy in (None, 1):
                a, b, c, result = [d.alloc() for _ in range(4)]
                d.sync(None)
                d.memset(a, 40, stream)
                d.launch("step", stream, a, b, 10000000)
                # Null-stream D2D must follow blocking work, and the later
                # blocking launch must follow the null-stream copy.
                d.copy(c, b, legacy)
                d.launch("step", stream, c, result, 0)
                d.sync(stream)
                assert d.read(result) == 42
                d.launch("step", stream, a, b, 10000000)
                d.launch("step", legacy, b, c, 0)
                d.copy(result, c, stream)
                d.sync(stream)
                assert d.read(result) == 42
                d.launch("step", stream, a, b, 10000000)
                d.memset(b, 42, legacy)
                d.copy(result, b, stream)
                d.sync(stream)
                assert d.read(result) == 42
                d.launch("step", stream, a, b, 10000000)
                d.call("cuMemcpyDtoD_v2", [C.c_uint64, C.c_uint64, C.c_size_t], c, b, 4)
                d.launch("step", stream, c, result, 0)
                d.sync(stream)
                assert d.read(result) == 42
    elif case == "cross_thread_null":
        a, b, c = [d.alloc() for _ in range(3)]
        stream = d.stream(0)
        d.sync(None)

        def submit(kind, value):
            d.call("cuCtxSetCurrent", [C.c_void_p], d.context)
            if kind == "memset":
                d.memset(a, value, None)
            elif kind == "kernel":
                d.launch("step", None, a, b, 0)
                d.copy(c, b, None)
            elif kind == "blocking":
                d.memset(a, value, stream)
                d.launch("step", stream, a, b, 10000000)
            elif kind == "null":
                d.launch("step", None, b, c, 0)
            elif kind == "after_null":
                d.launch("step", stream, c, a, 0)

        # API-return ordering across two persistent host threads, without GPU
        # waits in either thread, reproduces the April memset/launch pattern.
        with ThreadPoolExecutor(1) as producer, ThreadPoolExecutor(1) as consumer:
            for value in range(32):
                producer.submit(submit, "memset", value).result()
                consumer.submit(submit, "kernel", value).result()
                d.sync(None)
                assert d.read(c) == value + 1
            producer.submit(submit, "blocking", 40).result()
            consumer.submit(submit, "null", 0).result()
            producer.submit(submit, "after_null", 0).result()
            d.sync(stream)
            assert d.read(a) == 43
    elif case == "event_order":
        blocking, nonblocking = d.stream(0), d.stream(1)
        a, b, c = [d.alloc() for _ in range(3)]
        event = d.event()
        # Initialize the existing SYSTEM-scope scratch before the actual chain.
        d.record(event, None)
        d.sync(None)
        d.memset(a, 40, blocking)
        d.launch("step", blocking, a, b, 10000000)
        d.record(event, None)
        d.wait(nonblocking, event)
        d.copy(c, b, nonblocking)
        d.sync(nonblocking)
        assert d.read(c) == 41
        d.record(event, nonblocking)
        d.wait(None, event)
        d.launch("step", blocking, c, a, 0)
        d.sync(blocking)
        assert d.read(a) == 42
    elif case in ("per_thread_handle", "per_thread_entry_point"):
        stream = d.stream(0)
        flag, output = d.alloc(), d.alloc()
        d.sync(None)
        # PTDS is an ordinary blocking stream: independent of this explicit
        # stream until a legacy operation is submitted. A mistaken legacy zero
        # here would make the producer wait for the poll's deadline.
        ptsz = case == "per_thread_entry_point"
        d.launch("poll", None if ptsz else 2, flag, output, ptsz=ptsz)
        d.launch("produce", stream, flag)
        d.sync(2)
        d.sync(stream)
        assert d.read(output) == 1, "per-thread stream zero was treated as legacy"
        # PTDS must still synchronize with the legacy stream in both directions.
        a, b, c, result = [d.alloc() for _ in range(4)]
        d.sync(None)
        d.memset(a, 40, 2)
        d.launch("step", None if ptsz else 2, a, b, 10000000, ptsz=ptsz)
        d.copy(c, b, None)
        d.launch("step", None if ptsz else 2, c, result, 0, ptsz=ptsz)
        d.sync(2)
        assert d.read(result) == 42
    else:
        raise ValueError(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=90, help="seconds per isolated case")
    parser.add_argument("--case", choices=CASES, action="append", help="default: all cases")
    parser.add_argument("--child-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_case:
        assert not any(key.startswith("ZLUDA_") for key in os.environ), "ZLUDA switches must be unset"
        driver = Driver(args.library)
        run_case(driver, args.child_case)
        driver.close()
        print(f"PASS {args.child_case} (all ZLUDA_* unset)", flush=True)
        return
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ZLUDA_")}
    for case in args.case or CASES:
        with tempfile.TemporaryDirectory(prefix="zluda-null-stream-") as cache:
            environment["XDG_CACHE_HOME"] = cache
            print(f"RUN {case} (all ZLUDA_* unset)", flush=True)
            try:
                subprocess.run([sys.executable, __file__, "--library", str(args.library.resolve()),
                                "--child-case", case], env=environment, check=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"FAIL {case}: exceeded {args.timeout:g}s") from None
    print("PASS native null-stream fixtures", flush=True)


if __name__ == "__main__":
    main()

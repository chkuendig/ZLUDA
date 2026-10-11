#!/usr/bin/env python3
"""Real CUDA-path counter and production-dtor integration; Linux x86-64 GPU gate.

Run in a fresh, clean process. This intentionally enables recording and must
never be included in timed benchmarks. Native reader does all atomic loads.

This gates quiescent physical unload and recording cleanup. Full driver reload
is a separate diagnostic: the baseline also returns UNKNOWN on reinitialization.
"""
import argparse
import ctypes as C
import _ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import null_stream_api as fixture


def require(value, message):
    if not value: raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--reader', type=Path, required=True)
    p.add_argument('--build', required=True)
    p.add_argument('--evidence', type=Path, required=True)
    args = p.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    require(os.environ.get('ZLUDA_RECORDING') == '1', 'recording must be explicitly enabled')
    require(not any(k.startswith('ZLUDA_') and k != 'ZLUDA_RECORDING' for k in os.environ), 'diagnostic environment is not clean')
    require('PYTHONOPTIMIZE' not in os.environ and 'LD_PRELOAD' not in os.environ, 'unclean environment')
    library = args.library.resolve()
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    identity = library.stat()
    calling_tids = set()

    def capture_maps(name):
        text = Path('/proc/self/maps').read_text()
        (args.evidence/name).write_text(text)
        for line in text.splitlines():
            fields = line.split(None, 5)
            if len(fields) < 5: continue
            major, minor = (int(x, 16) for x in fields[3].split(':'))
            if os.makedev(major, minor) == identity.st_dev and int(fields[4]) == identity.st_ino:
                return True
        return False
    executable = Path(Path('/proc/self/cmdline').read_bytes().split(b'\0')[0].decode()).name
    with tempfile.TemporaryDirectory(prefix='zluda-recording-api-', dir='/dev/shm') as temp:
        os.environ['XDG_RUNTIME_DIR'] = temp
        os.environ['SteamAppId'] = '0'
        directory = Path(temp)/'zluda-recording'

        def total():
            return int(subprocess.check_output([str(args.reader), 'total', str(os.getpid()), '0', args.build,
                executable, str(library), digest, '-', temp], text=True, timeout=10))

        def exercise():
            calling_tids.add(threading.get_native_id())
            d = fixture.Driver(library, primary_context=True)
            require(total() == 3, 'warmup launch count differs')
            flag = d.alloc()
            for _ in range(7): d.launch('produce', None, flag)
            for _ in range(11): d.launch('produce', None, flag, ptsz=True)
            streams = [d.stream(1), d.stream(1)]
            barrier = threading.Barrier(2)

            def worker(stream):
                calling_tids.add(threading.get_native_id())
                d.call('cuCtxSetCurrent', [C.c_void_p], d.context)
                try:
                    barrier.wait(timeout=10)
                    for i in range(40): d.launch('produce', stream, flag, ptsz=bool(i%2))
                    d.sync(stream)
                finally: d.call('cuCtxSetCurrent', [C.c_void_p], None)

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(worker, streams))
            d.sync(None)
            require(total() == 101, 'normal/ptsz/concurrent launches were not counted exactly once')
            require(d.read(flag) == 1, 'produce kernel output wrong')
            # A zero grid is rejected by HIP without submitting a kernel.
            fn = d.lib.cuLaunchKernel
            fn.argtypes = [C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
            fn.restype = C.c_int
            value = C.c_uint64(flag.value)
            params = (C.c_void_p*1)(C.cast(C.byref(value), C.c_void_p).value)
            failed = fn(d.functions['produce'],0,1,1,1,1,1,0,None,params,None)
            require(failed != 0, 'expected a rejected zero-grid launch')
            require(total() == 101, 'failed HIP launch changed counter')
            print('PASS recording CUDA: 101 successful submissions, normal/ptsz/concurrent; failed launch excluded', flush=True)
            d.close()
            return d.lib._handle

        # CUDA context stacks register Rust TLS destructors. Join every calling
        # thread before requiring physical unload, rather than dlclose residency.
        with ThreadPoolExecutor(max_workers=1) as owner:
            handle = owner.submit(exercise).result()
        record, = directory.glob('*.rec')
        threads_gone = all(not Path('/proc/self/task', str(tid)).exists() for tid in calling_tids)
        require(threads_gone, 'CUDA-calling native thread still exists')
        require(total() == 101, 'resident recording lost its final count')
        before = capture_maps('before-unload.maps')
        require(before, 'candidate device/inode not mapped before close')
        _ctypes.dlclose(handle)
        after = capture_maps('after-unload.maps')
        result = dict(source=args.build, library_sha256=digest,
            calling_native_tids=sorted(calling_tids), threads_gone=threads_gone,
            mapped_before=before, mapped_after=after, recording_removed=not record.exists(), total=101)
        (args.evidence/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        require(not after, 'DSO remained resident after all CUDA calling threads exited')
        require(not record.exists(), 'production unload did not remove recording instance')
        print('PASS recording production quiescent unload and file removal', flush=True)


if __name__ == '__main__': main()

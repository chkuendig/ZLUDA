#!/usr/bin/env python3
"""Real CUDA-path counter and production-dtor integration; Linux x86-64 GPU gate.

Run in a fresh, clean process. This intentionally enables recording and must
never be included in timed benchmarks. Native reader does all atomic loads.
"""
import argparse
import ctypes as C
import _ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
import null_stream_api as fixture


def require(value, message):
    if not value: raise RuntimeError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--reader', type=Path, required=True)
    p.add_argument('--build', required=True)
    args = p.parse_args()
    require(os.environ.get('ZLUDA_RECORDING') == '1', 'recording must be explicitly enabled')
    require(not any(k.startswith('ZLUDA_') and k != 'ZLUDA_RECORDING' for k in os.environ), 'diagnostic environment is not clean')
    require('PYTHONOPTIMIZE' not in os.environ and 'LD_PRELOAD' not in os.environ, 'unclean environment')
    library = args.library.resolve()
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    executable = Path(Path('/proc/self/cmdline').read_bytes().split(b'\0')[0].decode()).name
    with tempfile.TemporaryDirectory(prefix='zluda-recording-api-', dir='/dev/shm') as temp:
        os.environ['XDG_RUNTIME_DIR'] = temp
        os.environ['SteamAppId'] = '0'
        directory = Path(temp)/'zluda-recording'

        def total():
            return int(subprocess.check_output([str(args.reader), 'total', str(os.getpid()), '0', args.build,
                executable, str(library), digest, '-', temp], text=True, timeout=10))

        d = fixture.Driver(library, primary_context=True)
        require(total() == 3, 'warmup launch count differs')
        flag = d.alloc()
        for _ in range(7): d.launch('produce', None, flag)
        for _ in range(11): d.launch('produce', None, flag, ptsz=True)
        streams = [d.stream(1), d.stream(1)]

        def worker(stream):
            d.call('cuCtxSetCurrent', [C.c_void_p], d.context)
            try:
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
        record, = directory.glob('*.rec')
        old_name = record.name
        handle = d.lib._handle
        del d
        _ctypes.dlclose(handle)
        require(not record.exists(), 'production dlclose did not remove recording instance')
        lib = C.CDLL(str(library))
        lib.cuInit.argtypes, lib.cuInit.restype = [C.c_uint], C.c_int
        require(lib.cuInit(0) == 0, 'reload cuInit failed')
        new, = directory.glob('*.rec')
        require(new.name != old_name and total() == 0, 'reload did not publish a fresh zero-count instance')
        _ctypes.dlclose(lib._handle)
        require(not new.exists(), 'reload dlclose left recording active')
        print('PASS recording production dlclose/reload', flush=True)


if __name__ == '__main__': main()

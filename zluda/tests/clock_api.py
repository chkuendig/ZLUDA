#!/usr/bin/env python3
"""Fresh-process gfx1030 %clock advancement check in both FP emission modes."""
import argparse
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import null_stream_api as fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--child-mode', choices=('normal', 'constrained'))
    args = parser.parse_args()
    if args.child_mode:
        assert not any(k.startswith('ZLUDA_') for k in os.environ)
        root = Path(__file__).parent
        fixture.PTX += b'\n' + b'.visible' + (root/'clock_advance.ptx').read_bytes().split(b'.visible', 1)[1]
        if args.child_mode == 'constrained':
            fixture.PTX += b'\n' + (root/'clock_constrained_probe.ptx').read_bytes()
        d = fixture.Driver(args.library)
        fn = C.c_void_p()
        d.call('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
               C.byref(fn), d.module, b'clock_advance')
        d.functions['clock_advance'] = fn
        output = d.alloc()
        d.sync(None)
        d.launch('clock_advance', None, output)
        d.sync(None)
        delta = d.read(output)
        assert delta != 0, 'cycle counter failed to advance within the bounded loop'
        d.close()
        print(f'PASS clock {args.child_mode}: unsigned delta={delta}', flush=True)
        return
    env = {k:v for k,v in os.environ.items() if not k.startswith('ZLUDA_') and k not in ('PYTHONOPTIMIZE', 'LD_PRELOAD')}
    for mode in ('normal', 'constrained'):
        with tempfile.TemporaryDirectory(prefix='zluda-clock-') as cache:
            env['XDG_CACHE_HOME'] = cache
            subprocess.run([sys.executable, __file__, '--library', str(args.library.resolve()),
                            '--child-mode', mode], env=env, check=True, timeout=90)
    print('PASS clock advancement (both FP modes; all ZLUDA_* unset)', flush=True)


if __name__ == '__main__':
    main()

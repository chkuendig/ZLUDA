#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Run the zluda32 -> zluda64_server -> SveSop nvcuda chain in GE-Proton Wine (WoW64).

  run-chain.py --ge GE_DIR --inputs DIR --work DIR [--iterations N]

DIR layout (from the workflow): msvc/{nvcuda.dll,zluda64_server.exe}, gnu/{...},
probe/{probe32.exe,probe64.exe}, svesop/{stock,patched}/nvcuda.dll.so,
stub/libcuda.so.1, and optionally zluda-linux/libcuda.so.1 (real ZLUDA).
Writes WORK/RESULT.json; each arm gets a fresh WoW64 prefix copy.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--ge', type=Path, required=True)
ap.add_argument('--inputs', type=Path, required=True)
ap.add_argument('--work', type=Path, required=True)
ap.add_argument('--iterations', type=int, default=20000)
args = ap.parse_args()
ge, inp, work = args.ge.resolve(), args.inputs.resolve(), args.work.resolve()
work.mkdir(parents=True, exist_ok=True)
files = ge / 'files'


def base_env(prefix, sync, libcuda_dir, extra_ld=()):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('WINE', 'PROTON_', 'ZLUDA', 'LD_'))}
    ld = [str(libcuda_dir)] if libcuda_dir else []
    ld += list(extra_ld) + [str(files / 'lib/wine/x86_64-unix'), str(files / 'lib/x86_64-linux-gnu'),
                            str(files / 'lib/i386-linux-gnu')]
    env.update(WINEPREFIX=str(prefix), WINEARCH='wow64', WINELOADER=str(files / 'bin/wine'),
               WINESERVER=str(files / 'bin/wineserver'), WINEDLLPATH=str(files / 'lib/wine'),
               LD_LIBRARY_PATH=':'.join(ld), WINEDLLOVERRIDES='nvcuda=n,b;winemenubuilder.exe=d;mscoree,mshtml=',
               WINEDEBUG='err+all,fixme+nvcuda,-fixme+all', ZLUDA64_PATH='C:\\zluda',
               WINEESYNC='1' if sync == 'esync' else '0', WINEFSYNC='1' if sync == 'fsync' else '0',
               WINENTSYNC='1' if sync == 'ntsync' else '0', LC_ALL='C',
               PATH=str(files / 'bin') + ':' + os.environ.get('PATH', '/usr/bin:/bin'))
    env.pop('DISPLAY', None)
    return env


def wine(env, argv, timeout):
    t0 = time.monotonic()
    try:
        r = subprocess.run([str(files / 'bin/wine')] + argv, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = 'timeout', e.stdout or b'', e.stderr or b''
    return code, out.decode(errors='replace'), err.decode(errors='replace'), round(time.monotonic() - t0, 2)


def processes():
    r = subprocess.run(['ps', '-eo', 'pid,args'], stdout=subprocess.PIPE, text=True)
    return [l.strip() for l in r.stdout.splitlines() if 'zluda64_server' in l or 'probe' in l.lower()]


def kill(env):
    subprocess.run([env['WINESERVER'], '-k'], env=env, timeout=60, check=False)
    subprocess.run([env['WINESERVER'], '-w'], env=env, timeout=60, check=False)


template = work / 'prefix-template'
if not template.exists():
    env = base_env(template, 'none', None)
    code, out, err, secs = wine(env, ['wineboot', '-i'], 600)
    kill(env)
    (work / 'wineboot.stderr').write_text(err[-20000:])
    print(f'wineboot exit={code} seconds={secs}', flush=True)


def arm(name, build, svesop, libcuda, sync, probe='probe32.exe', timeout=120, extra_ld=()):
    prefix = work / ('prefix-' + name)
    shutil.rmtree(prefix, ignore_errors=True)
    shutil.copytree(template, prefix, symlinks=True)
    c = prefix / 'drive_c'
    # GE copies the winelib nvcuda.dll.so to system32/nvcuda.dll the same way.
    shutil.copy(inp / 'svesop' / svesop / 'nvcuda.dll.so', c / 'windows/system32/nvcuda.dll')
    syswow = c / 'windows/syswow64/nvcuda.dll'
    if syswow.exists() or syswow.is_symlink():
        syswow.unlink()
    shutil.copy(inp / build / 'nvcuda.dll', syswow)
    (c / 'zluda').mkdir()
    shutil.copy(inp / build / 'zluda64_server.exe', c / 'zluda/zluda64_server.exe')
    (c / 'probe').mkdir()
    shutil.copy(inp / 'probe' / probe, c / 'probe' / probe)
    env = base_env(prefix, sync, libcuda, extra_ld)
    # GE's Wine takes ntsync whenever /dev/ntsync opens, whatever WINENTSYNC says.
    if Path('/dev/ntsync').exists():
        subprocess.run(['sudo', 'chmod', '666' if sync == 'ntsync' else '000', '/dev/ntsync'], check=False)
    code, out, err, secs = wine(env, ['C:\\probe\\' + probe, str(args.iterations)], timeout)
    procs = processes()
    kill(env)
    rows = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    logs = work / 'logs'
    logs.mkdir(exist_ok=True)
    (logs / f'{name}.stdout').write_text(out)
    (logs / f'{name}.stderr').write_text(err[-200000:])
    sync_lines = [l for l in err.splitlines() if 'sync' in l.lower() and ('running' in l or 'not' in l)]
    res = {'arm': name, 'build': build, 'svesop': svesop, 'libcuda': str(libcuda), 'sync': sync,
           'probe': probe, 'exit': code, 'seconds': secs, 'rows': rows,
           'processes_at_end': procs, 'sync_messages': sync_lines[:5],
           'stub_lines': [l for l in err.splitlines() if l.startswith('STUB:')][:40],
           'tail': err.splitlines()[-25:],
           'nvcuda_lines': [l for l in err.splitlines() if 'nvcuda' in l][:20]}
    print(json.dumps({k: res[k] for k in ('arm', 'exit', 'seconds', 'sync_messages')}), flush=True)
    return res


stub = inp / 'stub'
results = []
results.append(arm('stock-stub-msvc', 'msvc', 'stock', stub, 'ntsync', timeout=60))
for sync in ('ntsync', 'fsync', 'esync', 'server'):
    results.append(arm(f'patched-stub-msvc-{sync}', 'msvc', 'patched', stub, sync))
results.append(arm('patched-stub-gnu-ntsync', 'gnu', 'patched', stub, 'ntsync'))
results.append(arm('patched-stub-direct64-ntsync', 'msvc', 'patched', stub, 'fsync', probe='probe64.exe'))
zl = inp / 'zluda-linux'
if (zl / 'libcuda.so.1').exists():
    results.append(arm('patched-zluda-norocm', 'msvc', 'patched', zl, 'ntsync', timeout=60))
    rocm = Path(os.environ.get('ROCM_LIB', '/opt/rocm/lib'))
    if (rocm / 'libamdhip64.so.7').exists() or (rocm / 'libamdhip64.so').exists():
        results.append(arm('patched-zluda-rocm', 'msvc', 'patched', zl, 'ntsync', timeout=60, extra_ld=[str(rocm)]))
(work / 'RESULT.json').write_text(json.dumps(results, indent=1))

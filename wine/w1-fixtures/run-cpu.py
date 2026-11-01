#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Hosted CPU validation of the W1 fixtures in real Wine (no GPU, fake/absent backend).

  run-cpu.py --tool TOOL --fixtures BUILD_OUT --work FRESH_DIR

TOOL is an assembled Proton tool with the bridge under
files/lib/wine-cuda/{zluda,stub} (e.g. the 42b557a2 iteration tool from
proton-ge-custom run 41115896989). Runs, per case, one disposable Wine process
with a deadline and records stdout JSONL, stderr and the exit code. The
expectations below are what a fixture must do without a GPU: terminate in time,
print parseable start/finish records, and, for the PE32 stack fixture, detect
the 24 historical cleanup widths on the 42b557a2 bridge (negative control).
Writes WORK/CPU-RESULT.json and exits 1 if any expectation is not met.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

BACKENDS = {'absent': '/nonexistent/w1/libcuda.so.1', 'bundled': None}
STACK_CASES = [(name,int(correct),int(old)) for name,correct,old in re.findall(
    r'X\((\w+),\s*(\d+),\s*(\d+),',
    (Path(__file__).resolve().parent/'src/pe32_widths.h').read_text())]
STACK_TOTAL = len(STACK_CASES)
STACK_UNCHANGED = sum(correct==old for _,correct,old in STACK_CASES)
assert STACK_TOTAL==25 and STACK_TOTAL-STACK_UNCHANGED==24



def environment(tool, work, arch, root, enabled, backend):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('WINE', 'PROTON_', 'ZLUDA_', 'W1_')) and k not in ('LD_PRELOAD', 'DISPLAY')}
    prefix = work / ('prefix-' + arch)
    for sub in ('tmp', 'runtime'):
        (work / sub).mkdir(mode=0o700, exist_ok=True)
    env.update(LD_LIBRARY_PATH=':'.join(str(tool / 'files/lib' / a)
                                        for a in ['wine/x86_64-unix', 'x86_64-linux-gnu', 'i386-linux-gnu']),
               WINEPREFIX=str(prefix), WINEARCH=arch, WINEDEBUG='-all',
               WINELOADER=str(tool / 'files/bin/wine'), WINESERVER=str(tool / 'files/bin/wineserver'),
               WINEESYNC='0', WINEFSYNC='0', WINEDLLPATH=str(tool / 'files/lib/wine-cuda' / root),
               WINEDLLOVERRIDES='nvcuda=b;winemenubuilder.exe=d',
               PROTON_ENABLE_ZLUDA='1' if enabled else '0',
               PATH=str(tool / 'files/bin') + ':' + env.get('PATH', '/usr/bin:/bin'),
               LC_ALL='C', TMPDIR=str(work / 'tmp'), XDG_RUNTIME_DIR=str(work / 'runtime'))
    if enabled and backend:
        env['ZLUDA_WINE_BACKEND_PATH'] = backend
    return env


def run(name, argv, env, work, timeout, extra_env=None):
    env = dict(env, **(extra_env or {}))
    logs = work / 'logs'
    logs.mkdir(exist_ok=True)
    start = time.monotonic()
    try:
        r = subprocess.run([str(a) for a in argv], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = 'timeout', e.stdout or b'', e.stderr or b''
    (logs / (name + '.jsonl')).write_bytes(out)
    (logs / (name + '.stderr')).write_bytes(err[-65536:])
    rows = []
    for line in out.decode(errors='replace').splitlines():
        if line.startswith('{'):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    subprocess.run([env['WINESERVER'], '-k'], env=env, timeout=30, check=False)
    subprocess.run([env['WINESERVER'], '-w'], env=env, timeout=30, check=False)
    return {'name': name, 'exit': code, 'seconds': round(time.monotonic() - start, 2), 'rows': rows}


def checks(result):
    return {r['check']: r['pass'] for r in result['rows'] if 'check' in r}


def events(result, kind):
    return [r for r in result['rows'] if r.get('event') == kind]


def terminated(result):
    """Ran to its own finish record (any verdict) instead of hanging/crashing."""
    return result['exit'] in (0, 1, 2) and bool(events(result, 'start')) and (
        bool(events(result, 'finish')) or result['exit'] == 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    work = a.work
    work.mkdir(parents=True)
    fx = work / 'fixtures'
    shutil.copytree(a.fixtures, fx)
    x64, pe32 = fx / 'x64', fx / 'pe32'
    results, expectations = [], []

    def expect(label, ok, detail=None):
        expectations.append({'expectation': label, 'pass': bool(ok), 'detail': detail})

    wine = a.tool / 'files/bin/wine'
    for arch in ('win64', 'wow64'):
        env = environment(a.tool, work, arch, 'stub', False, None)
        results.append(run('boot-' + arch, [wine, 'wineboot', '-u'], env, work, 600))

    # 1. PE32 stack-balance fixture: self-test oracles, then the 42b557a2 bridge
    wow_off = environment(a.tool, work, 'wow64', 'stub', False, None)
    st = pe32 / 'pe32_stack.exe'
    good = run('pe32-stack-stub-correct', [wine, st, '--expect', 'correct'], wow_off, work, 120,
               {'W1_NVCUDA': 'w1stub_correct.dll'})
    old = run('pe32-stack-stub-old-expect-old', [wine, st, '--expect', 'old'], wow_off, work, 120,
              {'W1_NVCUDA': 'w1stub_old.dll'})
    detect = run('pe32-stack-stub-old-expect-correct', [wine, st, '--expect', 'correct'], wow_off, work, 120,
                 {'W1_NVCUDA': 'w1stub_old.dll'})
    results += [good, old, detect]
    summary = lambda r: (events(r, 'summary') or [{}])[0]
    expect('self-test: correct stub balanced 25/25', good['exit'] == 0 and summary(good).get('matched') == STACK_TOTAL)
    expect('self-test: 24 historical displacements and unchanged LUID cleanup',
           old['exit'] == 0 and summary(old).get('matched') == STACK_TOTAL)
    expect('self-test: old stub FAILS the correct-width gate on all 24',
           detect['exit'] == 1 and summary(detect).get('matched') == STACK_UNCHANGED)
    negative = {}
    for backend, path in BACKENDS.items():
        env = environment(a.tool, work, 'wow64', 'zluda', True, path)
        neg = run(f'pe32-stack-42b557a2-{backend}-expect-correct', [wine, st, '--expect', 'correct'], env, work, 120)
        exact = run(f'pe32-stack-42b557a2-{backend}-expect-old', [wine, st, '--expect', 'old'], env, work, 120)
        results += [neg, exact]
        negative[backend] = {'loaded': any(r.get('label') == 'nvcuda' for r in events(neg, 'module')),
                             'correct_matched': summary(neg).get('matched'),
                             'old_matched': summary(exact).get('matched')}
    loaded = [b for b, v in negative.items() if v['loaded']]
    expect('negative control: 42b557a2 PE32 nvcuda loaded in at least one backend mode', loaded, negative)
    expect('negative control: 42b557a2 FAILS 24/24 with the exact historical displacement',
           loaded and all(negative[b]['correct_matched'] == STACK_UNCHANGED and negative[b]['old_matched'] == STACK_TOTAL for b in loaded),
           negative)

    # 2. identity: off = GE empty stub; on = bridge surface (788 / 690 names)
    for arch, d, n in (('win64', x64, 788), ('wow64', pe32, 690)):
        off = run(f'identity-off-{arch}', [wine, d / 'identity.exe', '--expect', 'off'],
                  environment(a.tool, work, arch, 'stub', False, None), work, 120)
        on = run(f'identity-on-{arch}', [wine, d / 'identity.exe', '--expect', 'on', '--exports', str(n)],
                 environment(a.tool, work, arch, 'zluda', True, BACKENDS['absent']), work, 120)
        results += [off, on]
        expect(f'identity off {arch}: GE stub, no CUDA exports', off['exit'] == 0, checks(off))
        expect(f'identity on {arch}: {n} exports, no GPU -> cuInit fails, terminates',
               checks(on).get('export-count') is True and checks(on).get('cuInit') is False and terminated(on),
               checks(on))

    # 3. smoke: every fixture terminates deterministically without a GPU
    smoke = [('realgpu', ['realgpu.exe']), ('churn', ['churn.exe', '--cycles', '1000']),
             ('modload', ['modload.exe', '--self']), ('interop-data', ['interop.exe', 'data'])]
    smoke += [('lifecycle-' + c, ['lifecycle.exe', c]) for c in
              ('cls-basic', 'cls-overwrite', 'cls-two', 'cls-drained', 'tls-notify', 'freelibrary', 'exitprocess',
               'no-user32', 'reentry')]
    for arch, d in (('win64', x64), ('wow64', pe32)):
        env = environment(a.tool, work, arch, 'zluda', True, BACKENDS['absent'])
        for name, argv in smoke:
            r = run(f'smoke-{name}-{arch}', [wine, d / argv[0], *argv[1:]], env, work, 180)
            results.append(r)
            expect(f'smoke {name} {arch}: terminates with start/finish, no hang', terminated(r),
                   {'exit': r['exit']})
            if name == 'modload':
                expect(f'modload {arch}: PE stack-paint calibration', checks(r).get('pe-paint-calibration') is True,
                       checks(r))

    record = {'tool': str(a.tool), 'expectations': expectations,
              'results': [{k: v for k, v in r.items() if k != 'rows'} | {'checks': checks(r)} for r in results]}
    (work / 'CPU-RESULT.json').write_text(json.dumps(record, indent=2) + '\n')
    failed = [e for e in expectations if not e['pass']]
    for e in expectations:
        print(('PASS ' if e['pass'] else 'FAIL ') + e['expectation'])
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

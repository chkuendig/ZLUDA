"""Isolated exact-source CPU/build/format checks; never packages or admits a capsule."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess as S

PLAN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('fold', PLAN / 'check-fold.py')
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)
REF = 'refs/heads/ci/independent-pr-validation-20261111'


def table():
    value = f.load(PLAN / 'independent-commits.json')
    f.require(value['base'] == 'f4df215edbc3b6698a1c61ef7a8688fdff3872be', 'Unexpected base')
    f.require(len(value['commits']) == 8, 'Expected six legacy and two standalone commits')
    f.require(len({r['commit'] for r in value['commits']}) == 8, 'Duplicate source')
    for row in value['commits']:
        f.require(len(row['commit']) == 40 and len(row['tree']) == 40, 'Invalid source identity')
    f.require(value['initial'] in value['commits'], 'Initial source not in table')
    return value


f.table = table
f.REF = REF


def executable(metadata, target):
    found = []
    for line in Path(metadata).read_text().splitlines():
        row = json.loads(line)
        if row.get('reason') == 'compiler-artifact' and row['target']['name'] == target and row.get('executable'):
            found.append(row['executable'])
    f.require(len(found) == 1, 'Expected exactly one test binary for ' + target)
    return found[0]


def selected_names(listing, suffixes):
    available = [l.removesuffix(': test') for l in listing.splitlines() if l.endswith(': test')]
    names = []
    for suffix in suffixes:
        matches = [n for n in available if n == suffix or n.endswith('::' + suffix)]
        f.require(len(matches) == 1, 'Missing/ambiguous test: ' + suffix)
        names.extend(matches)
    f.require(len(set(names)) == len(suffixes), 'Duplicate selected tests')
    return names


def focused(binary, suffixes, out):
    out.mkdir(exist_ok=False)
    listing = S.check_output([binary, '--list'], text=True, timeout=60)
    (out / 'TESTS.txt').write_text(listing)
    names = selected_names(listing, suffixes)
    results = []
    for i, name in enumerate(names):
        row = f.command([binary, name, '--exact', '--nocapture', '--test-threads=1'], out, str(i), cwd=Path('/src'))
        text = (out / (str(i) + '.stdout')).read_text()
        row.update(name=name, executed_one='1 passed; 0 failed;' in text)
        results.append(row)
    receipt = {'tests': results, 'passed': all(r['returncode'] == 0 and r['executed_one'] for r in results)}
    f.write(out / 'RESULT.json', receipt)
    return receipt


def initial():
    source = Path('/src')
    proof = f.source_proof(source, table()['initial'], table())
    tests = focused(executable('/out/diagnostics/test-build.json', 'ptx'), [
        'kernel_rooted_calls', 'unreachable_call_does_not_keep_callee',
        'reachable_declarations_keep_source_order', 'test::_Z9vectorAddPKfS0_Pfi_ptx',
    ], Path('/out/diagnostics/independent-initial'))
    f.require(proof == f.source_proof(source, table()['initial'], table()), 'Source changed')
    # Compilation phase succeeds independently of the recorded tests. The final
    # aggregate checks tests.passed and fails the workflow if any test failed.
    f.write('/out/diagnostics/INDEPENDENT-INITIAL.json', {'source': proof, 'tests': tests,
            'not_capsule_qualification': True})


def inside(index):
    f.inside(index)
    row = table()['commits'][index]
    if row['label'] == 'context':
        out = Path('/out/diagnostics/fold-check') / str(index)
        argv = ['cargo', '+1.99.0', 'test', '--locked', '--offline', '-p', 'zluda', '--lib', '--no-run', '--message-format=json']
        command = f.command(argv, out, 'context-build', cwd=Path('/src'))
        f.require(command['returncode'] == 0, 'Context test build failed')
        tests = focused(executable(out / 'context-build.stdout', 'nvcuda'), [
            'destroy_context_clears_current_and_duplicate_handles',
            'destroy_context_preserves_unrelated_current',
            'destroy_context_rejects_primary_without_changing_stack',
        ], out / 'context-cpu')
        f.require(tests['passed'], 'Context CPU test failed')


def run():
    f.require(os.environ['GITHUB_REF'] == REF, 'Wrong ref')
    value = table()
    workspace = Path(os.environ['GITHUB_WORKSPACE'])
    source = workspace / 'source'
    work = Path(os.environ['RUNNER_TEMP']) / 'p400-jc2'
    dist = workspace / 'dist/jc2'
    output = dist / 'diagnostics/fold-check'
    output.mkdir(exist_ok=False)
    first = f.load(dist / 'diagnostics/INDEPENDENT-INITIAL.json')
    f.require(first['source']['commit'] == value['initial']['commit'], 'Wrong initial test source')
    f.require(not (dist / 'compiler-cache/IDENTITY.json').exists(), 'Foreign-ref cache must remain disabled')
    image = 'p400-jc2:' + os.environ['GITHUB_RUN_ID'] + '-' + os.environ['GITHUB_RUN_ATTEMPT']
    args = ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--cpus=3', '--memory=10g', '--memory-swap=10g',
            '-v', str(source) + ':/src:ro', '-v', str(PLAN) + ':/plan:ro',
            '-v', str(work / 'tools') + ':/tools', '-v', str(work / 'target') + ':/target', '-v', str(dist) + ':/out', '-w', '/src']
    env = {'PATH': '/tools/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
           'RUSTUP_HOME': '/tools/rustup', 'CARGO_HOME': '/tools/cargo', 'CARGO_TARGET_DIR': '/target',
           'CARGO_INCREMENTAL': '0', 'CARGO_BUILD_JOBS': '3', 'CMAKE_BUILD_PARALLEL_LEVEL': '3', 'LD_LIBRARY_PATH': '/opt/rocm/lib'}
    for i, path in enumerate(('/src', '/src/ext/llvm-project', '/src/ext/HiGHS')):
        env['GIT_CONFIG_KEY_' + str(i)] = 'safe.directory'; env['GIT_CONFIG_VALUE_' + str(i)] = path
    env['GIT_CONFIG_COUNT'] = '3'
    for key, val in env.items(): args.extend(['-e', key + '=' + val])
    S.run(args + ['--network', 'bridge', image, 'rustup', 'component', 'add', 'rustfmt', '--toolchain', '1.99.0'], check=True)
    S.run(['git', '-C', str(source), 'fetch', '--depth=2', 'origin', value['base'], *[r['commit'] for r in value['commits']]], check=True)
    results = []
    try:
        for i, row in enumerate(value['commits']):
            S.run(['git', '-C', str(source), '-c', 'core.hooksPath=/dev/null', 'checkout', '--detach', row['commit']], check=True)
            f.source_proof(source, row, value)
            rc = S.run(args + ['--network', 'none', image, 'python3', '/plan/check-independent.py', 'inside', '--index', str(i)]).returncode
            result = f.load(output / str(i) / 'RESULT.json')
            results.append({'commit': row['commit'], 'label': row['label'], 'returncode': rc,
                            'status': result['status'], 'passed': rc == 0 and result['status'] == 'PASS_COMPILE_AND_NO_NEW_FORMATTING'})
    finally:
        S.run(['git', '-C', str(source), '-c', 'core.hooksPath=/dev/null', 'checkout', '--detach', value['initial']['commit']], check=True)
        f.source_proof(source, value['initial'], value)
        passed = len(results) == 8 and all(r['passed'] for r in results) and first['tests']['passed']
        f.write(output / 'RESULT.json', {'passed': passed, 'commits': results, 'initial_tests': first['tests'],
                                        'no_capsule_built_or_admitted': True, 'no_gpu_tests_run': True})
    f.require(passed, 'Independent validation failed; inspect individual receipts')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['initial', 'run', 'inside']); p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.mode == 'initial': initial()
    elif a.mode == 'run': run()
    else:
        f.require(a.index is not None and 0 <= a.index < 8, 'Bad index')
        inside(a.index)

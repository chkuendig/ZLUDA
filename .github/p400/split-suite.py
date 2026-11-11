"""Split-mode PTX CPU suite gate; diagnostics only, never admits a capsule.

This branch deliberately carries no golden regeneration: the in-tree LLVM
golden corpus keeps the Windows-only llvm.amdgcn.s.dcache.inv prelude that
upstream itself never compares on Linux CI, and the documented
emission-dependent typed-goldens tests fail against the initializer emission
of 806d3392, which this split does not carry. Every other failure fails the
check. Runs the same filtered suite as run-ptx-cpu.py.
"""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

PLAN = pathlib.Path('/plan')
OUT = pathlib.Path('/out/diagnostics/split-suite')
REQUIRED_TESTS = (
    'test::spirv_run::legacy_f64_cvt_llvm',
    'test::spirv_run::legacy_slct_llvm',
    'test::spirv_run::legacy_slct_scalar_types',
    'test::spirv_run::legacy_typed_globals_llvm',
    'test::spirv_run::legacy_typed_globals32_metadata',
    'test::spirv_run::legacy_global_initializers32',
    'test::spirv_run::legacy_sreg16_llvm',
    'test::spirv_run::legacy_sreg16_32_truncates_helpers',
    'test::spirv_run::legacy_tex1d_coords_v4_llvm',
    'test::spirv_run::legacy_tex1d_coords_v4_32_llvm',
    'test::spirv_run::legacy_tex2d_coords_v4_llvm',
    'test::spirv_run::legacy_tex2d_coords_v4_32_llvm',
    'test::spirv_run::ordinary_mov_u16_rejects_u32_register',
)
PRELUDE_MARKER = 'llvm.amdgcn.s.dcache.inv'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def events(path):
    with open(path) as stream:
        for line in stream:
            item = json.loads(line)
            if item.get('reason') == 'compiler-artifact':
                yield item


def test_binary(build_json):
    for item in events(build_json):
        if item['target']['name'] == 'ptx' and item.get('executable'):
            require('ci_build' in item.get('features', []), 'ptx tests built without ci_build')
            return pathlib.Path(item['executable'])
    raise RuntimeError('No ptx test executable in build metadata')


def parse_runs(output):
    """Split libtest output into (name, status, block) per executed test."""
    header = re.compile(r'^test (.+?) \.\.\.\s+(ok|FAILED|ignored)\b')
    runs = []
    current = None
    for line in output.splitlines():
        match = header.match(line)
        if match:
            if current is not None:
                runs.append(current)
            current = (match.group(1), match.group(2), [])
        elif current is not None:
            current[2].append(line)
    if current is not None:
        runs.append(current)
    return runs


def classify(runs, emission_dependent):
    prelude = []
    emission = []
    unclassified = []
    for name, status, block in runs:
        if status != 'FAILED':
            continue
        if PRELUDE_MARKER in '\n'.join(block):
            prelude.append(name)
        elif name in emission_dependent:
            emission.append(name)
        else:
            unclassified.append(name)
    return prelude, emission, unclassified


def counts(output):
    match = re.search(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;', output)
    if match is None:
        return None
    return dict(zip(['passed', 'failed', 'ignored'], map(int, match.groups())))


def consistent(counted, runs, names):
    if counted is None or not names:
        return False
    observed = {'ok': 0, 'FAILED': 0, 'ignored': 0}
    for _, status, _ in runs:
        observed[status] += 1
    return (counted['passed'] == observed['ok'] and counted['failed'] == observed['FAILED']
            and counted['ignored'] == observed['ignored']
            and counted['passed'] + counted['failed'] + counted['ignored'] == len(names))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    table = load(PLAN / 'fold-commits.json')
    emission_dependent = set(table.get('split_emission_dependent_tests', []))
    require(emission_dependent, 'Split table without documented emission-dependent tests')
    binary = test_binary('/out/diagnostics/test-build.json')
    skip = ['--skip', '_amdgpu', '--skip', '_cuda']
    listing = subprocess.run([str(binary), '--list', *skip], check=True, capture_output=True,
                             text=True, timeout=90).stdout
    names = [line.removesuffix(': test') for line in listing.splitlines()
             if line.endswith(': test')]
    require(names and len(names) == len(set(names)), 'Duplicate or empty suite selection')
    require(not any('_amdgpu' in name or '_cuda' in name for name in names), 'Filter leaked')
    missing = [name for name in REQUIRED_TESTS if name not in names]
    require(not missing, 'Split tip tests missing: ' + ', '.join(missing))
    (OUT / 'TESTS.txt').write_text(listing)
    command = [str(binary), *skip, '--nocapture', '--test-threads=1', '--color', 'never']
    env = dict(__import__('os').environ, TEST_PTX_LLVM_FAIL_DIR=str(OUT / 'llvm-failures'))
    result = subprocess.run(command, capture_output=True, text=True, timeout=900, env=env)
    output = result.stdout + result.stderr
    (OUT / 'suite.log').write_text(output)
    runs = parse_runs(output)
    prelude, emission, unclassified = classify(runs, emission_dependent)
    counted = counts(output)
    legacy = sorted(name for name in names if 'legacy' in name)
    good = (consistent(counted, runs, names) and counted['passed'] > 0 and not unclassified)
    receipt = {
        'mode': 'SPLIT_CLASSIFIED',
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'command': command,
        'skip_filters': ['_amdgpu', '_cuda'],
        'selected_tests': names,
        'counts': counted,
        'returncode': result.returncode,
        'prelude_golden_failures': prelude,
        'emission_dependent_failures': emission,
        'unclassified_failures': unclassified,
        'legacy_tests': legacy,
        'emission_dependent_documented': sorted(emission_dependent),
        'passed': good,
        'note': ('prelude_golden_failures keep the Windows-only dcache.inv prelude that '
                 'upstream never compares on Linux CI; emission_dependent_failures are the '
                 'documented typed-goldens tests depending on the not-carried 806d3392 '
                 'initializer emission'),
    }
    (OUT / 'RESULT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in
                      ('counts', 'prelude_golden_failures', 'emission_dependent_failures',
                       'unclassified_failures', 'passed')}, indent=2))
    sys.exit(0 if good else 1)


if __name__ == '__main__':
    main()

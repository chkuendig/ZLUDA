"""Opt-in, public-source fold checks. Never admits or publishes a capsule."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PLAN = Path(__file__).resolve().parent
REF = 'refs/heads/ci/legacy-ptx-split-control-20261111'
# Split mode: this branch carries no golden regeneration, so the tip CPU suite
# is classified by split-suite.py instead of required to be fully green. The
# only tolerated failures are the Windows-only dcache.inv prelude goldens
# (upstream never compares them on Linux CI) and the documented
# emission-dependent typed-goldens tests of the not-carried 806d3392.


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def table():
    value = load(PLAN / 'fold-commits.json')
    control = bool(value.get('control'))
    parent = value['base']
    for row in value['commits']:
        require(re.fullmatch('[0-9a-f]{40}', row['commit']) is not None, 'Invalid SHA')
        require(row['parent'] == parent, 'Invalid parent chain')
        parent = row['commit']
    if control:
        require(len(value['commits']) == 1, 'Control checks exactly one commit')
    else:
        require(len(value['commits']) >= 2, 'Expected a native tip and a diagnostic child')
        require(value['commits'][-1]['parent'] == value['commits'][-2]['commit'],
                'Diagnostic child must build on the native tip')
    emission = value.get('split_emission_dependent_tests')
    require(isinstance(emission, list)
            and (control or emission)
            and all(re.fullmatch(r'[a-z0-9_:]+', name) for name in emission),
            'Split table without documented emission-dependent tests')
    return value


def pin():
    require(os.environ.get('GITHUB_REF') == REF, 'Fold mode requires the isolated CI ref')
    require(os.environ.get('FOLD_TARGET') == 'jc2', 'Fold mode requires target jc2')
    require(os.environ.get('FOLD_ORACLES', '').lower() == 'false', 'No oracle collection')
    value = table()
    tip = value['commits'][-1] if value.get('control') else value['commits'][-2]
    path = PLAN / 'pins.json'
    require(digest(path) == value['original_pins_sha256'], 'Original pins differ')
    pins = load(path)
    original = json.loads(json.dumps(pins))
    pins['jc2']['commit'] = tip['commit']
    pins['jc2']['tree'] = tip['tree']
    changed = [(group, key) for group in pins for key in pins[group]
               if pins[group][key] != original[group][key]]
    require(changed == [('jc2', 'commit'), ('jc2', 'tree')], 'Unexpected pin override')
    write(path, pins)
    write(PLAN.parent.parent / 'FOLD-PIN-OVERRIDE.json', {
        'original_sha256': value['original_pins_sha256'], 'actual_sha256': digest(path),
        'changed_fields': changed, 'check_only': True, 'ref': REF})


def added_lines(diff):
    result = {}
    name = None
    for line in diff.splitlines():
        if line.startswith('+++ b/'):
            name = line[6:]
        match = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
        if match:
            require(name is not None, 'Hunk has no path')
            start, count = int(match[1]), int(match[2] or 1)
            result.setdefault(name, set()).update(range(start, start + count))
    return result


def classify_fmt(text, source, added):
    """Validate each rustfmt diff against source before classifying '-' lines."""
    headers = list(re.finditer(r'^Diff in (.+):(\d+):\n', text, re.M))
    require(headers and headers[0].start() == 0, 'Expected rustfmt diff, not an error')
    rows = []
    for index, match in enumerate(headers):
        path = Path(match[1])
        path = (path if path.is_absolute() else source / path).resolve()
        require(path.is_relative_to(source.resolve()), 'Formatter path escaped source')
        relative = path.relative_to(source).as_posix()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        lines = text[match.end():end].splitlines(keepends=True)
        # rustfmt prints a separator after an EOF hunk; it is not source context.
        while lines and lines[-1] == '\n':
            lines.pop()
        require(lines and all(line[:1] in (' ', '+', '-') for line in lines),
                'Unexpected formatter output')
        original = path.read_text().splitlines(keepends=True)
        start = int(match[2]) - 1
        # Stable rustfmt emits one empty context line beyond EOF as a separator.
        while (lines and lines[-1] == ' \n'
               and sum(line[0] != '+' for line in lines) > len(original) - start):
            lines.pop()
        old = [line[1:] for line in lines if line[0] != '+']
        require(original[start:start + len(old)] == old, 'Diff does not match source')
        cursor = start + 1
        removed = []
        for line in lines:
            if line[0] == '-':
                removed.append(cursor)
            if line[0] != '+':
                cursor += 1
        new = sorted(set(removed) & added.get(relative, set()))
        rows.append({'path': relative, 'start': start + 1, 'removed_lines': removed,
                     'commit_added_lines_changed': new,
                     'classification': 'NEW_FAIL' if new else 'BASELINE_LISTED'})
    return rows


def command(argv, out, name, *, cwd):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out / (name + '.stdout')).open('x') as stdout, \
            (out / (name + '.stderr')).open('x') as stderr:
        result = subprocess.run(argv, cwd=cwd, stdout=stdout, stderr=stderr)
    return {'argv': argv, 'returncode': result.returncode, 'started': started,
            'finished': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def source_proof(source, row, value):
    require(git(source, 'rev-parse', 'HEAD') == row['commit'], 'Source HEAD differs')
    require(git(source, 'rev-parse', 'HEAD^{tree}') == row['tree'], 'Source tree differs')
    require(git(source, 'rev-parse', 'HEAD^') == row['parent'], 'Source parent differs')
    require(not git(source, 'status', '--porcelain', '--untracked-files=all'), 'Dirty source')
    for path, oid in value['invariant_objects'].items():
        require(git(source, 'rev-parse', 'HEAD:' + path) == oid, 'Changed invariant: ' + path)
    for path in ('ext/llvm-project', 'ext/HiGHS'):
        require(git(source / path, 'rev-parse', 'HEAD') == value['invariant_objects'][path],
                'Submodule checkout differs')
        require(not git(source / path, 'status', '--porcelain'), 'Dirty submodule')
    return {'commit': row['commit'], 'tree': row['tree'], 'source_clean': True,
            'invariant_objects': value['invariant_objects']}


def edition(source, path):
    for directory in [path.parent, *path.parent.parents]:
        if directory != source and source not in directory.parents:
            break
        manifest = directory / 'Cargo.toml'
        if manifest.is_file():
            matches = re.findall(r'^edition\s*=\s*"(2015|2018|2021|2024)"',
                                 manifest.read_text(), re.M)
            require(len(matches) <= 1, 'Ambiguous Rust edition')
            require('edition.workspace' not in manifest.read_text(), 'Resolve inherited edition')
            return matches[0] if matches else '2015'
    raise RuntimeError('No owning Cargo.toml')


def inside(index):
    value = table()
    commits = value['commits']
    row = commits[index]
    source, out = Path('/src'), Path('/out/diagnostics/fold-check') / str(index)
    out.mkdir(exist_ok=False)
    result = {'status': 'FAIL', 'source_before': source_proof(source, row, value),
              'commands': [], 'format': []}
    try:
        clean = ['cargo', '+1.99.0', 'clean', '-p', 'ptx_parser', '-p', 'ptx', '-p', 'zluda']
        check = ['cargo', '+1.99.0', 'check', '--locked', '--offline', '--tests',
                 '-p', 'ptx_parser', '-p', 'ptx', '-p', 'zluda', '--features', 'ptx/ci_build',
                 '--message-format=json']
        commands = [('clean', clean), ('check', check)]
        if index == len(commits) - 1 and not table().get('control'):
            commands += [('clean-tool', ['cargo', '+1.99.0', 'clean', '-p', 'compiler']),
                         ('check-tool', ['cargo', '+1.99.0', 'check', '--locked', '--offline',
                                         '-p', 'compiler', '--bin', 'zoc', '--message-format=json'])]
        version = command(['rustfmt', '+1.99.0', '--version'], out, 'rustfmt-version', cwd=source)
        require(version['returncode'] == 0, 'Formatter unavailable')
        result['formatter_version'] = (out / 'rustfmt-version.stdout').read_text().strip()
        require(result['formatter_version'].startswith('rustfmt 1.10.0-stable'),
                'Unexpected pinned formatter version')
        for name, argv in commands:
            receipt = command(argv, out, name, cwd=source)
            result['commands'].append(receipt)
            require(receipt['returncode'] == 0, name + ' failed')
            if name in ('check', 'check-tool'):
                fresh = []
                wanted = {'ptx_parser', 'ptx', 'nvcuda'} if name == 'check' else {'zoc'}
                for line in (out / (name + '.stdout')).read_text().splitlines():
                    item = json.loads(line)
                    if item.get('reason') == 'compiler-artifact':
                        target = item['target']['name']
                        if target in wanted:
                            require(item['fresh'] is False, 'Source check unexpectedly reused: ' + target)
                            fresh.append(target)
                require(set(fresh) == wanted, 'Missing source check artifacts')
                receipt['fresh_source_metadata_targets'] = sorted(set(fresh))
        changed = added_lines(git(source, 'diff', row['parent'], row['commit'], '-U0'))
        paths = git(source, 'diff', '--name-only', row['parent'], row['commit']).splitlines()
        for number, name in enumerate(path for path in paths if path.endswith('.rs')):
            path = source / name
            if not path.exists():
                result['format'].append({'root': name, 'deleted': True})
                continue
            receipt = command(['rustfmt', '+1.99.0', '--check', '--edition', edition(source, path),
                               str(path)], out, 'fmt-' + str(number), cwd=source)
            text = (out / ('fmt-' + str(number) + '.stdout')).read_text()
            errors = (out / ('fmt-' + str(number) + '.stderr')).read_text()
            require(not errors, 'Formatter warning or error; not a diff')
            require(receipt['returncode'] in (0, 1), 'Formatter execution failed')
            hunks = classify_fmt(text, source, changed) if receipt['returncode'] == 1 else []
            require(receipt['returncode'] != 0 or not text, 'Unexpected successful formatter output')
            result['format'].append({'root': name, 'command': receipt, 'hunks': hunks})
            require(not any(h['classification'] == 'NEW_FAIL' for h in hunks), 'New formatting drift')
        result['source_after'] = source_proof(source, row, value)
        require(result['source_before'] == result['source_after'], 'Source changed during checks')
        result['status'] = 'PASS_COMPILE_AND_NO_NEW_FORMATTING'
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        write(out / 'RESULT.json', result)


def run():
    require(os.environ.get('GITHUB_REF') == REF, 'Wrong fold CI ref')
    value = table()
    tip = value['commits'][-1] if value.get('control') else value['commits'][-2]
    workspace = Path(os.environ['GITHUB_WORKSPACE'])
    source = workspace / 'source'
    work = Path(os.environ['RUNNER_TEMP']) / 'p400-jc2'
    dist = workspace / 'dist/jc2'
    output = dist / 'diagnostics/fold-check'
    output.mkdir(exist_ok=False)
    # Foreign-ref cache guard must stay active; do not impersonate the production ref.
    require(not (dist / 'compiler-cache/IDENTITY.json').exists(), 'Unexpected cache identity')
    require(git(source, 'rev-parse', 'HEAD') == tip['commit'], 'Wrong initial tip')
    # Split mode: no capsule is packaged or admitted. The build step ran the
    # classified split suite instead of the fully-green capsule qualification.
    suite = load(dist / 'diagnostics/split-suite/RESULT.json')
    require(suite['mode'] == 'SPLIT_CLASSIFIED', 'Unexpected suite receipt')
    require(suite['skip_filters'] == ['_amdgpu', '_cuda'], 'Changed suite filters')
    require(suite['passed'] is True, 'Tip suite has unclassified failures')
    require(not suite['unclassified_failures'] and suite['counts']['passed'] > 0,
            'Tip suite classification failed')
    write(output / 'TIP-SUITE.json', suite)
    image = 'p400-jc2:' + os.environ['GITHUB_RUN_ID'] + '-' + os.environ['GITHUB_RUN_ATTEMPT']
    args = ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--cpus=4',
            '--memory=10g', '--memory-swap=10g', '-v', str(source) + ':/src:ro',
            '-v', str(PLAN) + ':/plan:ro', '-v', str(work / 'tools') + ':/tools',
            '-v', str(work / 'target') + ':/target', '-v', str(dist) + ':/out', '-w', '/src',
            '-e', 'PATH=/tools/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            '-e', 'RUSTUP_HOME=/tools/rustup', '-e', 'CARGO_HOME=/tools/cargo',
            '-e', 'CARGO_TARGET_DIR=/target', '-e', 'CARGO_INCREMENTAL=0', '-e', 'CARGO_BUILD_JOBS=4',
            '-e', 'CMAKE_BUILD_PARALLEL_LEVEL=4', '-e', 'LD_LIBRARY_PATH=/opt/rocm/lib',
            '-e', 'GIT_CONFIG_COUNT=3', '-e', 'GIT_CONFIG_KEY_0=safe.directory', '-e', 'GIT_CONFIG_VALUE_0=/src',
            '-e', 'GIT_CONFIG_KEY_1=safe.directory', '-e', 'GIT_CONFIG_VALUE_1=/src/ext/llvm-project',
            '-e', 'GIT_CONFIG_KEY_2=safe.directory', '-e', 'GIT_CONFIG_VALUE_2=/src/ext/HiGHS']
    subprocess.run(args + ['--network', 'bridge', image, 'rustup', 'component', 'add',
                           'rustfmt', '--toolchain', '1.99.0'], check=True)
    subprocess.run(['git', '-C', str(source), 'fetch', '--depth=2', 'origin',
                    value['base'], *[r['commit'] for r in value['commits']]], check=True)
    results = []
    try:
        for index, row in enumerate(value['commits']):
            subprocess.run(['git', '-C', str(source), '-c', 'core.hooksPath=/dev/null',
                            'checkout', '--detach', row['commit']], check=True)
            source_proof(source, row, value)
            subprocess.run(args + ['--network', 'none', image, 'python3', '/plan/check-fold.py',
                                   'inside', '--index', str(index)], check=True)
            result = load(output / str(index) / 'RESULT.json')
            require(result['status'] == 'PASS_COMPILE_AND_NO_NEW_FORMATTING', 'Commit failed')
            results.append(result)
    finally:
        subprocess.run(['git', '-C', str(source), '-c', 'core.hooksPath=/dev/null', 'checkout',
                        '--detach', tip['commit']], check=True)
        source_proof(source, tip, value)
        write(output / 'RESULT.json', {'status': 'PASS' if len(results) == len(value['commits'])
              else 'FAIL',
              'commits': [r['source_before'] for r in results], 'no_admitted_capsule': True,
              'capsule_not_built_in_split_mode': True,
              'cache': 'FOREIGN_REF_COLD_NO_RESTORE_OR_SAVE',
              'tip_full_ptx_counts': suite['counts'], 'tip_skip_filters': suite['skip_filters'],
              'tip_suite_classification': {
                  'prelude_golden_failures': len(suite['prelude_golden_failures']),
                  'emission_dependent_failures': suite['emission_dependent_failures'],
                  'unclassified_failures': suite['unclassified_failures']}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pin', 'run', 'inside'])
    parser.add_argument('--index', type=int)
    options = parser.parse_args()
    if options.mode == 'pin':
        pin()
    elif options.mode == 'run':
        run()
    else:
        require(options.index is not None and 0 <= options.index < len(table()['commits']),
                'Invalid commit index')
        inside(options.index)

import json, os, pathlib, re, subprocess, sys
from capsule import PINS, sha, write

events_path, source, out, cases_path = map(pathlib.Path, sys.argv[1:])
out.mkdir()
cases = json.loads(cases_path.read_text())
assert len(cases) == 8 and len({c['name'] for c in cases}) == 8
oracles = {c['oracle'] for c in cases if 'oracle' in c}
assert len(oracles) == 4 and sum(c['expected_returncode'] == 0 for c in cases) == 4
for name in oracles:
    assert pathlib.Path(name).name == name and name.endswith('.ll')
    golden = source / 'ptx/src/test/ll' / name
    assert golden.is_file() and not golden.is_symlink() and golden.stat().st_size == 0, name

events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
artifacts = [e for e in events if e['reason'] == 'compiler-artifact']
assert artifacts and all(e['fresh'] is False for e in artifacts)
llvm_events = [e for e in events if e['reason'] == 'build-script-executed' and '/llvm_zluda' in e['package_id']]
llvm_scripts = [e for e in artifacts if e['target']['src_path'] == '/src/llvm_zluda/build.rs']
assert llvm_events and llvm_scripts
freshness = dict(compiler_artifacts=len(artifacts), fresh_true=0, llvm_events=llvm_events,
                 llvm_scripts=[dict(path=p, sha256=sha(p)) for e in llvm_scripts for p in e['filenames']])
binaries = {}
for name in sorted({c['binary'] for c in cases}):
    matches = [e for e in artifacts if e['profile']['test'] and e.get('executable') and e['target']['name'] == name]
    assert len(matches) == 1, (name, len(matches))
    event = matches[0]
    if name == 'ptx':
        assert 'ci_build' in event['features']
    binary = pathlib.Path(event['executable'])
    binaries[name] = dict(path=str(binary), sha256=sha(binary), features=event['features'])
    listing = subprocess.run([str(binary), '--list'], check=True, capture_output=True, text=True, timeout=90).stdout
    (out / (name + '-TESTS.txt')).write_text(listing)
    names = [line.removesuffix(': test') for line in listing.splitlines() if line.endswith(': test')]
    for case in cases:
        if case['binary'] == name:
            assert names.count(case['name']) == 1, case['name']

results = []
for case in cases:
    directory = out / case['name'].rsplit('::', 1)[-1]
    directory.mkdir()
    llvm_dir = directory / 'llvm'
    command = [binaries[case['binary']]['path'], case['name'], '--exact', '--nocapture',
               '--test-threads=1', '--color', 'never']
    environment = dict(os.environ, TEST_PTX_LLVM_FAIL_DIR=str(llvm_dir))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, env=environment)
        code, output = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        code, output = 124, decode(error.stdout) + decode(error.stderr) + '\nTIMEOUT after 90 seconds\n'
    (directory / 'test.log').write_text(output)
    match = re.search(r'test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;', output)
    counts = dict(zip(['passed', 'failed', 'ignored'], map(int, match.groups()[1:]))) if match else None
    row = dict(case, command=command, returncode=code, counts=counts, timeout_seconds=90, matched=False)
    try:
        assert code == case['expected_returncode'], 'Unexpected return code'
        expected = dict(passed=0, failed=1, ignored=0) if 'oracle' in case else dict(passed=1, failed=0, ignored=0)
        assert counts == expected, 'Expected exactly one test'
        files = sorted(p.relative_to(llvm_dir).as_posix() for p in llvm_dir.rglob('*')) if llvm_dir.exists() else []
        if 'oracle' in case:
            assert 'assertion failed: `(left == right)`' in output, 'Not the LLVM comparison failure'
            assert files == [case['oracle']], 'Wrong LLVM outputs'
            path = llvm_dir / case['oracle']
            assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
            assert '\x00' not in path.read_text(encoding='utf-8')
            row['output'] = dict(path=path.relative_to(out).as_posix(), sha256=sha(path), bytes=path.stat().st_size)
        else:
            assert not files, 'Passing case emitted a mismatch'
        row['matched'] = True
    except (AssertionError, UnicodeError) as error:
        row['error'] = str(error)
    results.append(row)

source_receipt = json.loads((out.parent / 'SOURCE.json').read_text())
assert source_receipt['commit'] == PINS['jc2']['commit'] and source_receipt['tree'] == PINS['jc2']['tree']
good = all(row['matched'] for row in results)
receipt = dict(status='EXPECTED_ORACLE_FAILURES_COLLECTED' if good else 'COLLECTION_FAILED',
               qualified=False, capsule=False, source=source_receipt, freshness=freshness,
               binaries=binaries, cases_sha256=sha(cases_path), results=results,
               rustc=(out.parent / 'RUSTC.txt').read_text())
write(out / 'COLLECTION.json', receipt)
(out / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.relative_to(out).as_posix() + '\n'
                                      for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
print(json.dumps(receipt, indent=2))
sys.exit(0 if good else 1)

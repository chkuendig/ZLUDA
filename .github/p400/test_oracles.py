import hashlib, json, os, pathlib, subprocess, sys, tempfile, unittest
from capsule import HERE, PINS, write


class OracleCollection(unittest.TestCase):
    def run_collection(self, mode):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, diagnostics = root / 'source', root / 'diagnostics'
            goldens = source / 'ptx/src/test/ll'
            goldens.mkdir(parents=True)
            diagnostics.mkdir()
            cases = [c for c in json.loads((HERE / 'oracle-cases.json').read_text())
                     if 'oracle' not in c or c.get('golden_state') != 'existing_windows_prelude']
            if mode in ['existing-golden', 'wrong-golden']:
                case = next(c for c in cases if 'oracle' in c)
                case.update(golden_state='existing_windows_prelude', c7_golden_sha256=hashlib.sha256(b'; pinned public golden\n').hexdigest())
            cases_path = root / 'cases.json'
            write(cases_path, cases)
            for case in cases:
                if 'oracle' in case:
                    text = '; pinned public golden\n' if case.get('golden_state') == 'existing_windows_prelude' else ''
                    if text and mode == 'wrong-golden':
                        text = '; changed public golden\n'
                    (goldens / case['oracle']).write_text(text)
            golden_before = {p.name: p.read_bytes() for p in goldens.iterdir()}
            write(diagnostics / 'SOURCE.json', dict(commit=PINS['jc2']['commit'], tree=PINS['jc2']['tree']))
            (diagnostics / 'RUSTC.txt').write_text('public synthetic toolchain\n')
            script = '''import json, os, pathlib, sys
cases = json.loads(CASES)
if '--list' in sys.argv:
    print(''.join(c['name'] + ': test\\n' for c in cases)); sys.exit(0)
assert '--exact' in sys.argv and '--test-threads=1' in sys.argv
case = next(c for c in cases if c['name'] == sys.argv[1])
mode = os.environ['ORACLE_FAKE_MODE']
if 'oracle' in case:
    if mode != 'translation-error':
        directory = pathlib.Path(os.environ['TEST_PTX_LLVM_FAIL_DIR']); directory.mkdir()
        (directory / case['oracle']).write_text('; public synthetic LLVM IR\\n')
        if mode == 'extra-output': (directory / '_attributes.ll').write_text('; unexpected\\n')
        if mode == 'cache-prelude': (directory / case['oracle']).write_text('call void @llvm.amdgcn.s.dcache.inv()\\n')
        print('assertion failed: `(left == right)`')
    else: print('Todo before LLVM comparison')
    print('test result: FAILED. 0 passed; 1 failed; 0 ignored;'); sys.exit(101)
count = 0 if mode == 'zero-tests' else 1
print(f'test result: ok. {count} passed; 0 failed; 0 ignored;')
'''.replace('CASES', repr(json.dumps(cases)))
            artifacts = []
            for name in ['ptx_parser', 'ptx']:
                binary = root / name
                binary.write_text('#!' + sys.executable + '\n' + script)
                binary.chmod(0o755)
                artifacts.append(dict(reason='compiler-artifact', fresh=mode == 'stale-artifact',
                                      profile=dict(test=True), target=dict(name=name, src_path='/src/' + name + '/src/lib.rs'),
                                      executable=str(binary), features=['ci_build'] if mode != 'no-ci-feature' else []))
            compiled_script = root / 'llvm-build-script'
            compiled_script.write_text('public synthetic build-script artifact')
            artifacts += [dict(reason='compiler-artifact', fresh=False, profile=dict(test=False),
                               target=dict(name='build-script-build', src_path='/src/llvm_zluda/build.rs'),
                               filenames=[str(compiled_script)]),
                          dict(reason='build-script-executed', package_id='path+file:///src/llvm_zluda#0.0.0')]
            events = diagnostics / 'events.json'
            events.write_text(''.join(json.dumps(e) + '\n' for e in artifacts))
            out = diagnostics / 'oracle-collection'
            result = subprocess.run([sys.executable, str(HERE / 'collect-oracles.py'), str(events), str(source), str(out), str(cases_path)],
                                    capture_output=True, text=True, env=dict(os.environ, ORACLE_FAKE_MODE=mode, PYTHONDONTWRITEBYTECODE='1'))
            receipt = json.loads((out / 'COLLECTION.json').read_text()) if (out / 'COLLECTION.json').exists() else None
            self.assertEqual({p.name: p.read_bytes() for p in goldens.iterdir()}, golden_before)
            self.assertFalse(list(root.rglob('capsule.tar.gz')))
            return result, receipt

    def test_four_expected_failures_and_four_passes_are_collection_only(self):
        result, receipt = self.run_collection('expected')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(receipt['qualified'])
        self.assertFalse(receipt['capsule'])
        self.assertTrue(all(r['matched'] for r in receipt['results']))
        self.assertEqual(sum('output' in r for r in receipt['results']), 4)

    def test_translation_panic_is_not_an_oracle_failure(self):
        result, receipt = self.run_collection('translation-error')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt['status'], 'COLLECTION_FAILED')
        self.assertEqual(len(receipt['results']), 8)

    def test_existing_golden_must_match_its_pin(self):
        result, receipt = self.run_collection('existing-golden')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(r['matched'] for r in receipt['results']))
        result, receipt = self.run_collection('wrong-golden')
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(receipt)

    def test_extra_ir_and_zero_tests_are_rejected(self):
        for mode in ['extra-output', 'zero-tests', 'cache-prelude']:
            with self.subTest(mode=mode):
                result, receipt = self.run_collection(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(all(r['matched'] for r in receipt['results']))

    def test_stale_build_and_missing_embedded_fixtures_are_rejected(self):
        for mode in ['stale-artifact', 'no-ci-feature']:
            with self.subTest(mode=mode):
                result, receipt = self.run_collection(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(receipt)


if __name__ == '__main__':
    unittest.main()

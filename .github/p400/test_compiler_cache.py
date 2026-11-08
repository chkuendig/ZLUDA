"""Small filesystem fixtures, not a compiler/build/runner invocation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compiler_cache import cache_receipt, encoded, identity, qualification, sha

BRANCH = 'refs/heads/ci/p400-capsules-20261107'

class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.plan, self.source = self.root / 'plan', self.root / 'source'
        self.plan.mkdir()
        self.source.mkdir()
        self.pin = {'llvm_built': True, 'llvm_gitlink': 'a' * 40, 'llvm_build_rs_blob': 'b' * 40,
                    'llvm_manifest_blob': 'c' * 40, 'cargo_lock_blob': 'd' * 40, 'commit': 'e' * 40}
        self.pins = {name: dict(self.pin) for name in ('jc2', 'reference', 'zoc')}
        self.put(self.plan / 'pins.json', self.pins)
        self.put(self.plan / 'images.json', {'ubuntu': 'ubuntu@sha256:123', 'rust': 'rust@sha256:456', 'rocm_version': '6.3.4'})
        for name in ('cloud-build.sh', 'build-in-container.sh', 'compiler_cache.py'):
            (self.plan / name).write_text('fixture recipe ' + name)
        for name in ('.github/workflows/rocm_setup_build.sh', 'llvm_zluda/build.rs', 'llvm_zluda/Cargo.toml'):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture source ' + name)
        self.toolchain = {name: 'exact fixture ' + name for name in (
            'compiler_binaries', 'compiler_versions', 'packages', 'rustc', 'cmake', 'ninja', 'ccache')}

    def put(self, path, data):
        Path(path).write_bytes(encoded(data))

    def key(self, target='jc2', branch=BRANCH):
        return identity(self.plan, self.source, self.toolchain, target, branch)['namespace']

    def test_native_only_commit_change_preserves_llvm_object_namespace(self):
        before = self.key()
        self.pins['jc2']['commit'] = 'f' * 40
        self.put(self.plan / 'pins.json', self.pins)
        self.assertEqual(before, self.key())

    def test_llvm_source_or_configuration_pin_invalidates_cache(self):
        before = self.key()
        for field in ('llvm_gitlink', 'llvm_build_rs_blob', 'llvm_manifest_blob', 'cargo_lock_blob'):
            original = self.pins['jc2'][field]
            self.pins['jc2'][field] = 'f' * 40
            self.put(self.plan / 'pins.json', self.pins)
            self.assertNotEqual(before, self.key(), field)
            self.pins['jc2'][field] = original
        self.put(self.plan / 'pins.json', self.pins)

    def test_toolchain_package_or_compiler_content_changes_invalidate_cache(self):
        before = self.key()
        for field in self.toolchain:
            original = self.toolchain[field]
            self.toolchain[field] = 'changed'
            self.assertNotEqual(before, self.key(), field)
            self.toolchain[field] = original

    def test_build_recipe_and_rocm_setup_changes_invalidate_cache(self):
        before = self.key()
        for path in (self.plan / 'build-in-container.sh', self.source / '.github/workflows/rocm_setup_build.sh'):
            old = path.read_bytes()
            path.write_bytes(b'changed recipe')
            self.assertNotEqual(before, self.key())
            path.write_bytes(old)

    def test_release_and_test_profile_are_separate(self):
        self.assertNotEqual(self.key('jc2'), self.key('zoc'))
        self.assertEqual(self.key('jc2'), self.key('reference'))

    def test_foreign_or_pr_cache_branch_is_rejected(self):
        for ref in ('refs/heads/main', 'refs/pull/1/merge', 'refs/heads/other'):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.key(branch=ref)

    def test_missing_toolchain_identity_refuses_cache_selection(self):
        del self.toolchain['compiler_binaries']
        with self.assertRaises(ValueError):
            self.key()


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.identity = {'namespace': 'p400-llvm-objects-v1-' + BRANCH + '-exact-context-'}
        self.before = 'direct_cache_hit\t0\npreprocessed_cache_hit\t0\ncache_miss\t0\n'
        self.after = 'direct_cache_hit\t100\npreprocessed_cache_hit\t2\ncache_miss\t3\n'

    def test_warm_hits_and_matched_key_are_recorded(self):
        key = self.identity['namespace'] + '77-1'
        row = cache_receipt(self.identity, key, self.before, self.after)
        self.assertEqual(row['cache_matched_key'], key)
        self.assertEqual(row['total_hits'], 102)
        self.assertTrue(row['restored'])

    def test_full_hit_restore_skips_redundant_cache_save(self):
        row = cache_receipt(self.identity, self.identity['namespace'] + '77-1', self.before,
                            'direct_cache_hit\t100\npreprocessed_cache_hit\t2\ncache_miss\t0\n')
        self.assertFalse(row['save_needed'])
        row = cache_receipt(self.identity, self.identity['namespace'] + '77-1', self.before, self.after)
        self.assertTrue(row['save_needed'])

    def test_mismatched_restore_key_refuses(self):
        with self.assertRaises(ValueError):
            cache_receipt(self.identity, 'different-branch-or-key', self.before, self.after)

    def test_cold_cache_and_missing_counters_are_distinct(self):
        self.assertFalse(cache_receipt(self.identity, '', self.before, self.after)['restored'])
        with self.assertRaises(ValueError):
            cache_receipt(self.identity, '', '', self.after)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cache = {'namespace': 'exact-reviewed-cache-namespace'}
        for name, data in (('bin/ptx', b'fresh test harness'), ('runtime/lib/libc.so.6', b'exact libc'),
                           ('lib/extra.so', b'exact extra DSO')):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.cpu = {'harness_sha256': sha(self.root / 'bin/ptx'),
                    'results': [{'test': 'required_cpu', 'returncode': 0, 'one_test_passed': True}]}
        self.suite = {'harness_sha256': self.cpu['harness_sha256'], 'passed': True, 'returncode': 0,
                      'counts': {'passed': 231, 'failed': 0, 'ignored': 0}}
        self.freshness = {'target_initially_empty': True, 'fresh_true': 0, 'LLVM_BUILT': True,
                          'llvm_build_script_events': [{'real': True}],
                          'llvm_compiled_build_script': [{'path': '/target/exact-build-script', 'sha256': 'pinned'}]}
        self.source = {'commit': 'a' * 40, 'source_clean': True}
        self.manifest = {'target': 'jc2', 'source': self.source, 'freshness': self.freshness,
                         'executables': {'bin/ptx': {'sha256': self.cpu['harness_sha256'], 'fresh': False}},
                         'runtime': {'libc.so.6': {'sha256': sha(self.root / 'runtime/lib/libc.so.6'), 'aliases': [], 'needed': []}},
                         'loader': 'runtime/lib/libc.so.6', 'cpu': {'ptx': self.cpu}, 'ptx_cpu_suite': self.suite,
                         'files': {name: {'kind': 'file', 'sha256': sha(self.root / name)} for name in
                                   ('bin/ptx', 'runtime/lib/libc.so.6', 'lib/extra.so')}}
        for name, data in (('SOURCE.json', self.source), ('receipts/FRESHNESS.json', self.freshness),
                           ('receipts/cpu-ptx/RESULT.json', self.cpu), ('receipts/cpu-ptx-full/RESULT.json', self.suite)):
            self.put(name, data)
        self.commit()

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(data))

    def commit(self):
        self.put('MANIFEST.json', self.manifest)

    def qualified(self):
        return qualification(self.root, self.cache)

    def test_same_actual_receipts_and_bytes_yield_byte_identical_comparison(self):
        a = encoded(self.qualified())
        self.assertEqual(a, encoded(self.qualified()))
        self.assertEqual(set(self.qualified()['payload_files_sha256']),
                         {'bin/ptx', 'runtime/lib/libc.so.6', 'lib/extra.so'})

    def test_each_changed_executable_or_dso_blocks_equivalence(self):
        for name in ('bin/ptx', 'runtime/lib/libc.so.6', 'lib/extra.so'):
            path = self.root / name
            old = path.read_bytes()
            path.write_bytes(b'different')
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualified()
            path.write_bytes(old)

    def test_cargo_fresh_or_missing_llvm_event_cannot_be_qualified(self):
        original = copy.deepcopy(self.manifest)
        for field, value in (('fresh_true', 1), ('llvm_build_script_events', []),
                             ('llvm_compiled_build_script', []), ('target_initially_empty', False)):
            self.manifest['freshness'][field] = value
            self.commit()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.qualified()
            self.manifest = copy.deepcopy(original)

    def test_failed_focused_or_full_cpu_result_cannot_be_qualified(self):
        self.manifest['cpu']['ptx']['results'][0]['one_test_passed'] = False
        self.commit()
        with self.assertRaises(ValueError): self.qualified()
        self.manifest['cpu']['ptx']['results'][0]['one_test_passed'] = True
        self.manifest['ptx_cpu_suite']['counts']['failed'] = 1
        self.commit()
        with self.assertRaises(ValueError): self.qualified()

    def test_changed_raw_cpu_receipt_is_not_normalised_away(self):
        before = encoded(self.qualified())
        self.cpu['results'][0]['extra_actual_evidence'] = 'changed'
        self.manifest['cpu']['ptx'] = self.cpu
        self.put('receipts/cpu-ptx/RESULT.json', self.cpu)
        self.commit()
        self.assertNotEqual(before, encoded(self.qualified()))

    def test_receipt_manifest_disagreement_refuses_comparison(self):
        self.put('receipts/cpu-ptx-full/RESULT.json', {'passed': True})
        with self.assertRaises(ValueError): self.qualified()


if __name__ == '__main__':
    unittest.main()

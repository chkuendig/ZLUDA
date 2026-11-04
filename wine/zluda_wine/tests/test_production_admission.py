# SPDX-License-Identifier: MIT OR Apache-2.0
"""The build consumes a pinned attestation, never its Wine source inputs."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('admit_wine', HERE / 'tools/admit_wine.py')
ADMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADMIT)


class ProductionAdmission(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.provider = self.root / 'sdk/lib/ntdll.so'
        self.provider.parent.mkdir(parents=True)
        self.provider.write_bytes(b'fixture SDK provider')
        sources = ['include/ntuser.h', 'include/winternl.h', 'include/winnt.h',
                   'dlls/user32/user_main.c', 'dlls/ntdll/signal_x86_64.c',
                   'dlls/ntdll/unix/signal_x86_64.c', 'dlls/ntdll/loader.c',
                   'dlls/ntdll/unix/thread.c', 'dlls/ntdll/unix/virtual.c',
                   'dlls/ntdll/unix/unix_private.h']
        self.receipt = {
            'contract': 'zluda-wine-prepared-abi-2',
            'wine_pin': '46b29104e3741fe23bf5e2547196a253aab88c89',
            'verified': dict.fromkeys(['callback_id_zero', 'same_thread_destructor',
                'process_exit', 'thread_exit', 'loader_lock_refusal',
                'locally_scoped_strong_provider', 'eager_binding'], True),
            'prepared_files': dict.fromkeys(sources, '0' * 64),
            'sdk_files': {'lib/ntdll.so': hashlib.sha256(self.provider.read_bytes()).hexdigest()},
        }

    def admit(self, pinned=None):
        path = self.root / 'receipt.json'
        path.write_text(json.dumps(self.receipt))
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if pinned is None else pinned
        with patch.object(sys, 'argv', ['admit_wine.py', str(path), str(self.root / 'sdk')]), \
             patch.dict(os.environ, {'ZLUDA_WINE_ABI_RECEIPT_SHA256': digest}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            os.environ.pop('WINE_SOURCE_ROOT', None)
            ADMIT.main()
        return output.getvalue()

    def test_valid_attestation_needs_no_wine_source_tree(self):
        self.assertEqual(self.admit().splitlines(), ['cargo:rerun-if-changed=' + str(self.provider)])

    def test_wrong_receipt_pin_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.admit('f' * 64)

    def test_changed_provider_is_refused(self):
        self.provider.write_bytes(b'changed provider')
        with self.assertRaisesRegex(ValueError, 'SDK bytes differ'):
            self.admit()

    def test_unverified_lifetime_fact_is_refused(self):
        self.receipt['verified']['same_thread_destructor'] = False
        with self.assertRaisesRegex(ValueError, 'same_thread_destructor'):
            self.admit()

    def test_source_file_in_sdk_extent_is_refused_without_reading_it(self):
        self.receipt['sdk_files']['include/ntuser.h'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'reviewer attestation metadata'):
            self.admit()

    def test_source_metadata_paths_and_digests_are_validated(self):
        self.receipt['prepared_files']['../escape.c'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'receipt member'):
            self.admit()
        del self.receipt['prepared_files']['../escape.c']
        self.receipt['prepared_files']['include/ntuser.h'] = 'not a hash'
        with self.assertRaisesRegex(ValueError, 'receipt digest'):
            self.admit()


if __name__ == '__main__':
    unittest.main()

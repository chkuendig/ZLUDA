# SPDX-License-Identifier: MIT OR Apache-2.0
"""Backend source admission rejects absent, relative and changed declarations."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

CRATE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRATE / 'tools'))
from backend_source import backend_source


class BackendSource(unittest.TestCase):
    def test_missing_and_relative_roots_are_rejected(self):
        for value in ['', 'relative/backend']:
            with patch.dict(os.environ, ZLUDA_WINE_BACKEND_SOURCE=value):
                with self.assertRaisesRegex(ValueError, 'absolute pinned'):
                    backend_source()

    def test_moved_exact_inputs_pass_and_changed_input_is_rejected(self):
        original = backend_source()
        contract = json.loads((CRATE.parents[1] / 'BACKEND-SOURCE.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in contract['source_inputs']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original / name, path)
            with patch.dict(os.environ, ZLUDA_WINE_BACKEND_SOURCE=str(root)):
                self.assertEqual(backend_source(), root)
                path = root / next(iter(contract['source_inputs']))
                path.write_bytes(path.read_bytes() + b'\n// changed declaration input\n')
                with self.assertRaisesRegex(ValueError, 'Pinned backend source differs'):
                    backend_source()

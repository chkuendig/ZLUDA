# SPDX-License-Identifier: MIT OR Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('candidate_import', Path(__file__).resolve().parents[1] / 'tools/import_candidate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidateImport(unittest.TestCase):
    def test_documents_and_unreviewed_paths_fail_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crate = root / 'crate'
            candidate = root / 'candidate'
            (crate / 'LICENSES').mkdir(parents=True)
            candidate.mkdir()
            (crate / 'LICENSES/PRODUCTION-INPUTS.json').write_text(json.dumps({'source_inputs': {'wine/zluda_wine/src/a.rs': ''}}))
            for name in ['QUESTIONS.md', 'IMPLEMENTATION-NOTES.md', 'src/extra.rs', 'LICENSES/PROVENANCE.md', 'tests/a.rs']:
                path = candidate / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('unreviewed')
                with self.assertRaises(ValueError):
                    module.plan(candidate, crate, {})
                self.assertFalse((crate / name).exists())
                path.unlink()

    def test_unchanged_scaffold_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'LICENSES').mkdir()
            (root / 'candidate/src').mkdir(parents=True)
            old = root / 'candidate/src/old.rs'
            new = root / 'candidate/src/new.rs'
            old.write_text('old scaffold')
            new.write_text('reviewed change')
            (root / 'LICENSES/PRODUCTION-INPUTS.json').write_text(json.dumps({'source_inputs': {'wine/zluda_wine/src/old.rs': '', 'wine/zluda_wine/src/new.rs': ''}}))
            changes = module.plan(root / 'candidate', root, {'wine/zluda_wine/src/old.rs': hashlib.sha256(old.read_bytes()).hexdigest()})
            self.assertEqual([name for name, _, _ in changes], ['src/new.rs'])

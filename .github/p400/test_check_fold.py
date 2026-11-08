import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("fold", Path(__file__).with_name("check-fold.py"))
fold = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fold)


class Formatting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "src" / "lib.rs"
        self.path.parent.mkdir()
        self.path.write_text("fn f() {\n    let x =  1;\n}\n")

    def tearDown(self):
        self.tmp.cleanup()

    def diff(self):
        return (f"Diff in {self.path}:1:\n fn f() {{\n-    let x =  1;\n"
                "+    let x = 1;\n }\n \n\n")

    def test_commit_line_fails(self):
        result = fold.classify_fmt(self.diff(), self.root, {"src/lib.rs": {2}})
        self.assertEqual(result[0]["classification"], "NEW_FAIL")
        self.assertEqual(result[0]["commit_added_lines_changed"], [2])

    def test_only_removed_lines_count(self):
        result = fold.classify_fmt(self.diff(), self.root, {"src/lib.rs": {1, 3}})
        self.assertEqual(result[0]["classification"], "BASELINE_LISTED")

    def test_child_path_is_classified(self):
        result = fold.classify_fmt(self.diff(), self.root, {})
        self.assertEqual(result[0]["path"], "src/lib.rs")
        self.assertEqual(result[0]["removed_lines"], [2])

    def test_error_is_not_a_diff(self):
        with self.assertRaises(RuntimeError):
            fold.classify_fmt("error: parser failed\n", self.root, {})

    def test_mismatched_source_fails(self):
        with self.assertRaises(RuntimeError):
            fold.classify_fmt(self.diff().replace("let x =  1", "let x =  2"), self.root, {})

    def test_escaped_path_fails(self):
        escaped = self.path.parent / ".." / ".." / "outside.rs"
        with self.assertRaises(RuntimeError):
            fold.classify_fmt(self.diff().replace(str(self.path), str(escaped)), self.root, {})

    def test_real_eof_removal_is_not_stripped(self):
        self.path.write_text("fn f() {}\n\n")
        text = f"Diff in {self.path}:1:\n fn f() {{}}\n-\n"
        result = fold.classify_fmt(text, self.root, {"src/lib.rs": {2}})
        self.assertEqual(result[0]["classification"], "NEW_FAIL")

    def test_multiple_headers_keep_old_line_numbers(self):
        self.path.write_text("a\nb\nc\nd\ne\n")
        text = (f"Diff in {self.path}:1:\n-a\n+A\n b\n"
                f"Diff in {self.path}:4:\n-d\n+D\n e\n")
        result = fold.classify_fmt(text, self.root, {"src/lib.rs": {4}})
        self.assertEqual([r["classification"] for r in result], ["BASELINE_LISTED", "NEW_FAIL"])

    def test_u0_added_ranges(self):
        text = "+++ b/a.rs\n@@ -2,0 +3,2 @@\n+x\n+y\n@@ -6,2 +8,0 @@\n-x\n-y\n"
        self.assertEqual(fold.added_lines(text), {"a.rs": {3, 4}})


class BoundTable(unittest.TestCase):
    def test_exact_six_commit_chain(self):
        value = fold.table()
        self.assertEqual(len(value["commits"]), 6)
        self.assertEqual(value["commits"][4]["commit"], "ec365cf8331707b6c606bfb17f12517072d6f64b")
        self.assertEqual(value["commits"][5]["commit"], "2c958eb8af3581a4701eecedfdf8432427bdd655")


class RuntimePin(unittest.TestCase):
    def test_changes_only_two_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "control/.github/p400"
            plan.mkdir(parents=True)
            for name in ("fold-commits.json", "pins.json"):
                (plan / name).write_bytes((fold.PLAN / name).read_bytes())
            before = fold.load(plan / "pins.json")
            with patch.object(fold, "PLAN", plan), patch.dict(fold.os.environ, {
                    "GITHUB_REF": fold.REF, "FOLD_TARGET": "jc2", "FOLD_ORACLES": "false"}):
                fold.pin()
            after = fold.load(plan / "pins.json")
            changed = [(g, k) for g in before for k in before[g] if before[g][k] != after[g][k]]
            self.assertEqual(changed, [("jc2", "commit"), ("jc2", "tree")])
            self.assertTrue((plan.parent.parent / "FOLD-PIN-OVERRIDE.json").is_file())

    def test_foreign_target_and_oracles_refused(self):
        for target, oracle, ref in (("zoc", "false", fold.REF),
                                    ("jc2", "true", fold.REF),
                                    ("jc2", "false", "refs/heads/main")):
            with patch.dict(fold.os.environ, {"GITHUB_REF": ref, "FOLD_TARGET": target,
                                              "FOLD_ORACLES": oracle}):
                with self.assertRaises(RuntimeError):
                    fold.pin()


if __name__ == "__main__":
    unittest.main()

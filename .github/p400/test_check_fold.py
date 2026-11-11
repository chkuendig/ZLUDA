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
    def modified_table(self, mutate):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "control/.github/p400"
            plan.mkdir(parents=True)
            value = json.loads((fold.PLAN / "fold-commits.json").read_text())
            mutate(value)
            (plan / "fold-commits.json").write_text(json.dumps(value, indent=2) + "\n")
            with patch.object(fold, "PLAN", plan):
                return fold.table()

    def test_control_checks_exactly_upstream_base(self):
        value = fold.table()
        self.assertTrue(value["control"])
        self.assertEqual(len(value["commits"]), 1)
        self.assertEqual(value["commits"][0]["commit"], "f4df215edbc3b6698a1c61ef7a8688fdff3872be")
        self.assertEqual(value["commits"][0]["parent"], value["base"])
        self.assertEqual(value["invariant_objects"]["ext/llvm-project"],
                         "ca1f2d04a41ceab20cc9e0bb8214b3e6be873a0b")

    def test_control_may_have_empty_emission_list(self):
        value = fold.table()
        self.assertEqual(value["split_emission_dependent_tests"], [])

    def test_control_requires_single_commit(self):
        def mutate(value):
            value["commits"] = value["commits"] * 2
        with self.assertRaises(RuntimeError):
            self.modified_table(mutate)

    def test_control_requires_flag_for_single_commit(self):
        def mutate(value):
            del value["control"]
        with self.assertRaises(RuntimeError):
            self.modified_table(mutate)

    def test_no_hardcoded_green_suite(self):
        source = Path(fold.__file__).read_text()
        self.assertNotIn("TIP_SUITE", source)
        self.assertIn("SPLIT_CLASSIFIED", source)


class SplitSuite(unittest.TestCase):
    SPEC = importlib.util.spec_from_file_location(
        "split", Path(__file__).with_name("split-suite.py"))
    split = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(split)

    def test_golden_path(self):
        self.assertEqual(self.split.golden_path("test::spirv_run::add_llvm"),
                         "ptx/src/test/ll/add.ll")
        self.assertIsNone(self.split.golden_path("test::_Z9vectorAddPKfS0_Pfi_ptx"))
        self.assertIsNone(self.split.golden_path("test::spirv_run::legacy_typed_globals32_metadata"))

    def test_prelude_classified_from_golden_file(self):
        reader = {"ptx/src/test/ll/add.ll": "call void @llvm.amdgcn.s.dcache.inv()\n"}.get
        prelude, emission, unclassified = self.split.classify(
            ["test::spirv_run::add_llvm"], reader, set())
        self.assertEqual(prelude, ["test::spirv_run::add_llvm"])
        self.assertEqual(unclassified, [])

    def test_non_prelude_golden_is_unclassified(self):
        reader = {"ptx/src/test/ll/x.ll": "define void @x() {}\n"}.get
        _, _, unclassified = self.split.classify(
            ["test::spirv_run::x_llvm"], reader, set())
        self.assertEqual(unclassified, ["test::spirv_run::x_llvm"])

    def test_emission_dependent_without_prelude(self):
        reader = {"ptx/src/test/ll/legacy_typed_globals.ll": "@zero = global float 0.0\n"}.get
        _, emission, _ = self.split.classify(
            ["test::spirv_run::legacy_typed_globals_llvm"], reader,
            {"test::spirv_run::legacy_typed_globals_llvm"})
        self.assertEqual(emission, ["test::spirv_run::legacy_typed_globals_llvm"])

    def test_non_golden_test_is_unclassified(self):
        prelude, emission, unclassified = self.split.classify(
            ["test::_Z9vectorAddPKfS0_Pfi_ptx"], lambda path: None, set())
        self.assertEqual(unclassified, ["test::_Z9vectorAddPKfS0_Pfi_ptx"])

    def test_counts_consistency(self):
        text = ("test a ... ok\ntest b ... FAILED\ntest c ... ignored\n"
                "test result: FAILED. 1 passed; 1 failed; 1 ignored;\n")
        runs = self.split.parse_runs(text)
        counted = self.split.counts(text)
        self.assertTrue(self.split.consistent(counted, runs, ["a", "b", "c"]))
        self.assertFalse(self.split.consistent(counted, runs, ["a", "b"]))
        self.assertFalse(self.split.consistent(None, runs, ["a", "b", "c"]))



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
            self.assertEqual(after["jc2"]["commit"], "f4df215edbc3b6698a1c61ef7a8688fdff3872be")
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

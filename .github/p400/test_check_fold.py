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

    def test_seven_commit_chain(self):
        value = fold.table()
        self.assertEqual(len(value["commits"]), 7)
        self.assertEqual(value["commits"][-2]["commit"], "7af79191c602f61eafd9d1a174ce09efef7d782a")
        self.assertEqual(value["commits"][-1]["commit"], "128ffa5cebc4ee84d413eb246f08eae021203259")

    def test_base_is_upstream_master(self):
        value = fold.table()
        self.assertEqual(value["base"], "f4df215edbc3b6698a1c61ef7a8688fdff3872be")
        self.assertEqual(value["invariant_objects"]["ext/llvm-project"],
                         "ca1f2d04a41ceab20cc9e0bb8214b3e6be873a0b")

    def test_no_hardcoded_green_suite(self):
        # Split mode classifies failures; a fully-green tip count cannot be promised.
        source = Path(fold.__file__).read_text()
        self.assertNotIn("TIP_SUITE", source)
        self.assertIn("SPLIT_CLASSIFIED", source)

    def test_tip_is_second_to_last(self):
        value = fold.table()
        tip, diag = value["commits"][-2], value["commits"][-1]
        self.assertEqual(diag["parent"], tip["commit"])
        self.assertEqual(tip["parent"], "5e47e6b5f8b57bab986289974557b9616a4c08bd")

    def test_wrong_diag_parent_refused(self):
        def mutate(value):
            value["commits"][-1]["parent"] = value["commits"][-3]["commit"]
        with self.assertRaises(RuntimeError):
            self.modified_table(mutate)

    def test_broken_parent_chain_refused(self):
        def mutate(value):
            value["commits"][3]["parent"] = "0" * 40
        with self.assertRaises(RuntimeError):
            self.modified_table(mutate)

    def test_tip_only_table_refused(self):
        def mutate(value):
            value["commits"] = value["commits"][:1]
        with self.assertRaises(RuntimeError):
            self.modified_table(mutate)

    def test_missing_emission_list_refused(self):
        def mutate(value):
            del value["split_emission_dependent_tests"]
        with self.assertRaises(RuntimeError):
            self.modified_table(mutate)

    def test_bad_emission_name_refused(self):
        def mutate(value):
            value["split_emission_dependent_tests"] = ["not a test name!"]
        with self.assertRaises(RuntimeError):
            self.modified_table(mutate)


class SplitSuite(unittest.TestCase):
    SPEC = importlib.util.spec_from_file_location(
        "split", Path(__file__).with_name("split-suite.py"))
    split = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(split)

    def output(self, *blocks):
        return "".join(blocks)

    def test_prelude_failure_classified(self):
        text = self.output(
            "test test::spirv_run::abs_llvm ... FAILED\n",
            " Left: call void @llvm.amdgcn.s.dcache.inv()\n",
            "test test::empty ... ok\n",
            "test result: FAILED. 1 passed; 1 failed; 0 ignored;\n")
        runs = self.split.parse_runs(text)
        prelude, emission, unclassified = self.split.classify(runs, set())
        self.assertEqual(prelude, ["test::spirv_run::abs_llvm"])
        self.assertEqual(emission, [])
        self.assertEqual(unclassified, [])

    def test_emission_dependent_failure_classified(self):
        text = self.output(
            "test test::spirv_run::legacy_typed_globals_llvm ... FAILED\n",
            " Left: @zero = addrspace(1) externally_initialized global float 0.0\n",
            "test result: FAILED. 0 passed; 1 failed; 0 ignored;\n")
        runs = self.split.parse_runs(text)
        prelude, emission, unclassified = self.split.classify(
            runs, {"test::spirv_run::legacy_typed_globals_llvm"})
        self.assertEqual(emission, ["test::spirv_run::legacy_typed_globals_llvm"])
        self.assertEqual(unclassified, [])

    def test_other_failure_is_unclassified(self):
        text = self.output(
            "test test::spirv_run::legacy_sreg16_llvm ... FAILED\n",
            " mismatch\n",
            "test result: FAILED. 0 passed; 1 failed; 0 ignored;\n")
        runs = self.split.parse_runs(text)
        _, _, unclassified = self.split.classify(
            runs, {"test::spirv_run::legacy_typed_globals_llvm"})
        self.assertEqual(unclassified, ["test::spirv_run::legacy_sreg16_llvm"])

    def test_counts_consistency(self):
        text = self.output(
            "test a ... ok\ntest b ... FAILED\ntest c ... ignored\n",
            "test result: FAILED. 1 passed; 1 failed; 1 ignored;\n")
        runs = self.split.parse_runs(text)
        counted = self.split.counts(text)
        self.assertTrue(self.split.consistent(counted, runs, ["a", "b", "c"]))
        self.assertFalse(self.split.consistent(counted, runs, ["a", "b"]))
        self.assertFalse(self.split.consistent(None, runs, ["a", "b", "c"]))

    def test_failure_summary_lines_are_not_tests(self):
        text = self.output(
            "test a ... FAILED\n",
            "failures:\n",
            "    a\n",
            "test result: FAILED. 0 passed; 1 failed; 0 ignored;\n")
        runs = self.split.parse_runs(text)
        self.assertEqual([name for name, _, _ in runs], ["a"])



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

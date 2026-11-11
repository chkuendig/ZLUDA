import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('independent', Path(__file__).with_name('check-independent.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FailClosedSelection(unittest.TestCase):
    def test_only_exact_suffixes_are_selected(self):
        self.assertEqual(m.selected_names('a::foo: test\na::foo_cuda: test\n1 test\n', ['foo']), ['a::foo'])

    def test_missing_case_is_not_a_successful_zero_test_run(self):
        with self.assertRaises(RuntimeError):
            m.selected_names('a::other: test\n', ['foo'])

    def test_ambiguous_case_refuses(self):
        with self.assertRaises(RuntimeError):
            m.selected_names('a::foo: test\nb::foo: test\n', ['foo'])

    def test_missing_or_duplicate_build_artifacts_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'build.json'
            p.write_text('{"reason":"compiler-artifact","target":{"name":"other"},"executable":"/tmp/a"}\n')
            with self.assertRaises(RuntimeError): m.executable(p, 'ptx')
            line = '{"reason":"compiler-artifact","target":{"name":"ptx"},"executable":"/tmp/a"}\n'
            p.write_text(line + line)
            with self.assertRaises(RuntimeError): m.executable(p, 'ptx')


if __name__ == '__main__': unittest.main()

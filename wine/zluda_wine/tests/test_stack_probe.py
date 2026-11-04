import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('stack_measure',Path(__file__).parent/'stack/measure.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class StackMeasurementAdmission(unittest.TestCase):
    def test_requires_same_thread_guarded_complete_compilation(self):
        sample={'reserve_bytes':1048576,'low_guard_bytes':4096,'upper_guard_bytes':4096,
                'usable_bytes':1044480,'touched_high_water_bytes':66560,'rounded_observed_bound_bytes':69632,
                'initial_tid':123,'compile_tid':123,'same_thread':True,'main_status':0,'seed':91}
        self.assertEqual(probe.validate(sample),sample)
        for bad in [dict(sample,same_thread=1),dict(sample,compile_tid=124),dict(sample,main_status=1),
            dict(sample,initial_tid=0,compile_tid=0),dict(sample,low_guard_bytes=0,upper_guard_bytes=0,usable_bytes=1048576),
            dict(sample,rounded_observed_bound_bytes=66560),dict(sample,touched_high_water_bytes=0),
            dict(sample,reserve_bytes=64*1048576),dict(sample,rounded_observed_bound_bytes=1048576)]:
            with self.subTest(value=bad):
                with self.assertRaises(RuntimeError):probe.validate(bad)

import importlib.util
from pathlib import Path
import unittest
import json

spec=importlib.util.spec_from_file_location('lifecycle_runner',Path(__file__).parent/'abi/run-lifecycle.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LifecycleReceipts(unittest.TestCase):
    def test_original_masked_failure_and_unarmed_setup_failure_remain_failures(self):
        native=['backend-loaded','cuInit','native-dark-query']
        with self.assertRaisesRegex(RuntimeError,'rc=93'):
            runner.validate_case('free-library',93,
                'lifecycle failure55: TlsSetValue(state.tls,state.value) && !setup(&state) (0)\n',
                native,['detach-begin','loader-worker-refused'])
        with self.assertRaisesRegex(RuntimeError,'storage-set=0x0000000000000321'):
            runner.validate_case('free-library',1,
                'LIFECYCLE storage-set=0x0000000000000321\nLIFECYCLE setup-returned=0x0000000000000321\n',
                native,[])

    def test_timing_keeps_raw_samples_and_rejects_missing_or_invalid_fields(self):
        value={'iterations':200000,'frequency_hz':10000000,'samples_ticks':[300000]*11,'baseline_ticks':[10000]*11}
        output='TRANSITIONS: '+json.dumps(value)+'\nPASS: timing\n'
        receipt=runner.timing_receipt(0,output)
        self.assertAlmostEqual(receipt['median_ns_per_call'],150)
        self.assertEqual(receipt['samples_ticks'],value['samples_ticks'])
        for changed in [dict(value,iterations=True),dict(value,frequency_hz=0),dict(value,samples_ticks=[1]*10),
                        dict(value,baseline_ticks=[-1]*11),dict(value,unexpected=1)]:
            with self.assertRaises(RuntimeError):runner.timing_receipt(0,'TRANSITIONS: '+json.dumps(changed)+'\nPASS:')
        with self.assertRaises(RuntimeError):runner.timing_receipt(124,output)
    def test_each_actual_exit_path_needs_complete_ordered_receipt(self):
        for mode in runner.MODES:
            events=[event for event,count in runner.EXPECTED_EVENTS[mode].items() for _ in range(count)]
            pe=runner.EXPECTED_PE_EVENTS[mode]
            runner.validate_case(mode,0,'PASS: lifecycle',events,pe)
            for code,output,native,windows in ((124,'FIXTURE TIMEOUT',events,pe),
                (0,'',events,pe),(0,'PASS:',events[:-1],pe),
                (0,'PASS:',events+['BAD-callback'],pe),(0,'PASS:',events,pe[:-1]),
                (0,'PASS:',events,list(reversed(pe)))):
                with self.subTest(mode=mode,code=code,native=native,windows=windows):
                    with self.assertRaises(RuntimeError):runner.validate_case(mode,code,output,native,windows)

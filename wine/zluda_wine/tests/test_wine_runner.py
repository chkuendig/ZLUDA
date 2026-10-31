"""Selection/harness control fixtures only; never execute Wine or a PE module."""
import argparse
import fcntl
from contextlib import redirect_stdout
import io
import importlib.util
import os
import json
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent/'abi'
spec = importlib.util.spec_from_file_location('wine_runner',HERE/'run-fixtures.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def probe_image(bits=64):
    image=bytearray(256);image[:2]=b'MZ';struct.pack_into('<I',image,60,128)
    image[128:132]=b'PE\0\0';struct.pack_into('<H',image,132,0x14c if bits==32 else 0x8664)
    struct.pack_into('<H',image,152,0x10b if bits==32 else 0x20b)
    return image


def backend_image():
    image=bytearray(64);image[:6]=b'\x7fELF\x02\x01';struct.pack_into('<HH',image,16,3,62)
    return image


class WineRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.args=argparse.Namespace(**{key:root/key for key in
            ('wine','wineserver','dll_dir','on_root','off_root','backend','probe','prefix','results')})
        self.args.dll_dir.mkdir()
        for key in ('wine','wineserver','backend','probe'):
            getattr(self.args,key).write_bytes(b'test-only input')
        self.args.backend.write_bytes(backend_image());self.args.probe.write_bytes(probe_image())
        for name in runner.ON_FILES:
            path=self.args.on_root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test-only on module')
        for name in runner.GE_STUBS:
            path=self.args.off_root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
        self.fake_stubs={name:runner.sha(self.args.off_root/name) for name in runner.GE_STUBS}

    def test_launch_environment_has_exactly_one_root_and_backend_only_when_on(self):
        with patch.dict(os.environ,{'WINEDLLPATH':'/poison/a:/poison/b',
             'ZLUDA_WINE_BACKEND_PATH':'/poison/libcuda.so.1','PROTON_ENABLE_ZLUDA':'1'}):
            for mode in runner.MODES:
                env=runner.environment(self.args,mode,Path('/audit'))
                self.assertEqual(env['WINEDLLPATH'],str(self.args.on_root if mode.startswith('on') else self.args.off_root))
                self.assertNotIn(':',env['WINEDLLPATH'])
                if mode=='on': self.assertEqual(env['ZLUDA_WINE_BACKEND_PATH'],str(self.args.backend))
                elif mode=='on-relative-backend': self.assertEqual(env['ZLUDA_WINE_BACKEND_PATH'],'backend')
                else: self.assertNotIn('ZLUDA_WINE_BACKEND_PATH',env)
                if mode=='off-zero': self.assertEqual(env['PROTON_ENABLE_ZLUDA'],'0')
                elif mode.startswith('off'): self.assertNotIn('PROTON_ENABLE_ZLUDA',env)

    def test_canonical_module_and_root_symlinks_fail_before_execution(self):
        stale=self.args.dll_dir/'i386-windows/nvcuda.dll';stale.parent.mkdir();stale.symlink_to('/missing/module')
        with self.assertRaisesRegex(ValueError,'canonical dll_dir'):
            runner.run(self.args)
        stale.unlink()
        link=self.args.on_root/'x86_64-unix/nvcuda.so';link.unlink();link.symlink_to('/missing/helper')
        with self.assertRaisesRegex(ValueError,'regular files'):
            runner.run(self.args)

    def test_only_exact_off_stub_hashes_admitted(self):
        with patch.object(runner,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'exact pinned GE stubs'):
                runner.run(self.args)
            execute.assert_not_called()

    def test_probe_profile_and_backend_machine_fail_before_execution(self):
        self.args.winearch='wow64'
        with patch.object(runner,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'machine does not match'):
                runner.run(self.args)
            self.args.probe.write_bytes(probe_image(32))
            self.args.backend.write_bytes(b'not an ELF64 shared object')
            with self.assertRaisesRegex(ValueError,'ELF64 x86_64'):
                runner.run(self.args)
            execute.assert_not_called()

    def test_wow64_profile_requires_real_machine_receipts_and_bounded_native_witness(self):
        self.args.winearch='wow64';self.args.probe.write_bytes(probe_image(32))
        calls=[]
        def execute(command,env,timeout):
            calls.append((command,env.copy()))
            self.assertEqual(env['WINEARCH'],'wow64')
            if command[1]=='wineboot.exe':self.args.prefix.mkdir()
            output='PASS: simulated control only\n'
            if command[1]==str(self.args.probe):
                mode='off' if '--expect-stub' in command else 'on'
                receipt={'process_machine':'0x014c','native_machine':'0x8664','pointer_bits':32,
                    'module_machine':'0x014c','module_bits':32,'cu_deviceptr_bits':32,'exports':0 if mode=='off' else 690}
                output='ARCHITECTURE: '+json.dumps(receipt)+'\n'+output
                if len(command)==2:
                    Path(env['ZLUDA_WINE_FIXTURE_AUDIT']).write_text(''.join((event+'\n')*count for event,count in runner.WOW64_EVENTS.items()))
                    witness={'native_bits':64,'native_pid':123,'unixlib_path':str(self.args.on_root/'x86_64-unix/nvcuda.so'),
                        'unixlib_mapped':True,'i386_unix_cuda_mapped':False,'unexpected_cuda_mapping':False,
                        'pool_reserved_bytes':512*1024*1024,'pool_native_base':0x100000000,
                        'htod_offset':260,'dtoh_offset':260,'pool_released':True,'context_released':True}
                    Path(env['ZLUDA_WINE_FIXTURE_WITNESS']).write_text(json.dumps(witness))
            return (1,'') if command[1]=='-k0' else (0,output)
        with patch.object(runner,'GE_STUBS',self.fake_stubs),patch.object(runner,'execute',side_effect=execute),redirect_stdout(io.StringIO()):
            runner.run(self.args)
        result=json.loads((self.args.results/'RESULT.json').read_text())
        self.assertEqual(result['qualification'],'CPU-Wine-PE32-WoW64-pass; GPU-W1-W2-pending')
        self.assertEqual(result['architecture'],runner.architecture(self.args))
        self.assertEqual(len(result['cases']),6)
        self.assertEqual(result['cases'][1]['wow64_witness']['htod_offset'],260)
        self.assertEqual(len([command for command,_ in calls if command[1]==str(self.args.probe)]),6)
        receipt=result['cases'][1]['architecture']
        self.assertEqual(runner.wow64_receipt('ARCHITECTURE: '+json.dumps(receipt),'on'),receipt)
        for bad in ('', 'ARCHITECTURE: '+json.dumps(dict(receipt,pointer_bits=64))):
            with self.assertRaisesRegex(RuntimeError,'architecture receipt'):runner.wow64_receipt(bad,'on')
        path=self.args.results/'on-backend.json';witness=json.loads(path.read_text())
        for change in ({'i386_unix_cuda_mapped':True},{'unixlib_mapped':False},
                       {'htod_offset':256},{'pool_native_base':0x1000},{'context_released':False},
                       {'unixlib_mapped':1},{'native_pid':True}):
            path.write_text(json.dumps(dict(witness,**change)))
            with self.assertRaisesRegex(RuntimeError,'WoW64'):runner.wow64_witness(path,self.args)

    def test_wow64_missing_receipt_fails_and_still_cleans_its_server(self):
        self.args.winearch='wow64';self.args.probe.write_bytes(probe_image(32))
        def execute(command,env,timeout):
            if command[1]=='wineboot.exe':self.args.prefix.mkdir()
            return (1,'') if command[1]=='-k0' else (0,'PASS: simulated but no real architecture receipt\n')
        with patch.object(runner,'GE_STUBS',self.fake_stubs),patch.object(runner,'execute',side_effect=execute) as execute:
            with self.assertRaisesRegex(RuntimeError,'architecture receipt'):runner.run(self.args)
        self.assertEqual(execute.call_args.args[0],[str(self.args.wineserver),'-k0'])

    def test_same_prefix_off_on_off_records_cleanup_and_keeps_stub_bytes(self):
        calls=[]
        def execute(command,env,timeout):
            calls.append((command,env.copy(),timeout))
            if command[1]=='wineboot.exe': self.args.prefix.mkdir()
            if command[1]==str(self.args.probe) and len(command)==2:
                Path(env['ZLUDA_WINE_FIXTURE_AUDIT']).write_text(''.join(
                    (event+'\n')*count for event,count in runner.REQUIRED_EVENTS.items()))
            return (1,'') if command[1]=='-k0' else (0,'PASS: simulated control fixture only\n')
        with patch.object(runner,'GE_STUBS',self.fake_stubs),patch.object(runner,'execute',side_effect=execute),redirect_stdout(io.StringIO()):
            runner.run(self.args)
        probes=[env for command,env,_ in calls if command[1]==str(self.args.probe)]
        self.assertEqual(len(probes),6)
        self.assertEqual([probes[i]['WINEDLLPATH'] for i in (0,1,2)],
            [str(self.args.off_root),str(self.args.on_root),str(self.args.off_root)])
        self.assertEqual({env['WINEPREFIX'] for env in probes},{str(self.args.prefix)})
        self.assertEqual([call[0][1] for call in calls[-3:]],['-k','-w','-k0'])
        self.assertEqual(runner.sha(self.args.prefix/'drive_c/windows/system32/nvcuda.dll'),
                         self.fake_stubs['x86_64-windows/nvcuda.dll'])

    def test_backend_use_while_off_is_a_failure_and_server_is_cleaned_up(self):
        def execute(command,env,timeout):
            if command[1]=='wineboot.exe': self.args.prefix.mkdir()
            elif command[1]==str(self.args.probe):
                Path(env['ZLUDA_WINE_FIXTURE_AUDIT']).write_text('backend-loaded\n')
            return (1,'') if command[1]=='-k0' else (0,'PASS: simulated control fixture only\n')
        with patch.object(runner,'GE_STUBS',self.fake_stubs),patch.object(runner,'execute',side_effect=execute) as execute:
            with self.assertRaisesRegex(RuntimeError,'off/refused case'):
                runner.run(self.args)
        self.assertEqual(execute.call_args.args[0],[str(self.args.wineserver),'-k0'])

    def test_cleanup_accepts_killed_and_already_exited_servers_only_after_wait_and_verification(self):
        self.args.results.mkdir()
        for kill in (0,1):
            with self.subTest(kill=kill),patch.object(runner,'execute',side_effect=[(kill,''),(0,''),(1,'')]) as execute:
                result=runner.cleanup_server(self.args,{'WINEPREFIX':str(self.args.prefix)})
            self.assertTrue(result['verified_absent'])
            self.assertEqual([call.args[0][1] for call in execute.call_args_list],['-k','-w','-k0'])
            self.assertEqual(result['lock']['reason'],'prefix-absent')
            self.assertTrue(all((self.args.results/step['log']).is_file() for step in result['steps']))

    def test_cleanup_rejects_survivor_timeout_diagnostics_and_lock_probe_errors(self):
        self.args.results.mkdir()
        cases=[[(1,''),(0,''),(0,'')],[(0,''),(124,'timeout'),(1,'')],
               [(1,'fcntl failed'),(0,''),(1,'')],[(0,''),(0,''),(1,'probe error')],
               [(2,''),(0,''),(1,'')]]
        for replies in cases:
            with self.subTest(replies=replies),patch.object(runner,'execute',side_effect=replies):
                self.assertFalse(runner.cleanup_server(self.args,{})['verified_absent'])
        with patch.object(runner,'execute',side_effect=[(1,''),(0,''),(1,'')]),\
             patch.object(runner,'verify_server_lock_absent',side_effect=PermissionError('lock inspection failed')):
            self.assertFalse(runner.cleanup_server(self.args,{})['verified_absent'])

    def test_prefix_lock_probe_uses_dev_inode_and_detects_actual_foreign_lock(self):
        self.args.prefix.mkdir()
        info=self.args.prefix.stat()
        base=Path(self.tmp.name)/'private-server-base'
        directory=base/f'server-{info.st_dev:x}-{info.st_ino:x}'
        directory.mkdir(parents=True)
        lock=directory/'lock';lock.write_bytes(b'unchanged')
        self.assertTrue(runner.verify_server_lock_absent(self.args.prefix,base)['absent'])
        # POSIX record locks are process-owned. A separate short-lived Python
        # child holds Wine's exact byte range; no Wine/server is executed.
        ready_read,ready_write=os.pipe();stop_read,stop_write=os.pipe()
        pid=os.fork()
        if pid==0:
            try:
                os.close(ready_read);os.close(stop_write)
                with lock.open('r+b') as stream:
                    fcntl.lockf(stream,fcntl.LOCK_EX,1,0,os.SEEK_SET)
                    os.write(ready_write,b'1');os.read(stop_read,1)
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(ready_write);os.close(stop_read)
        try:
            self.assertEqual(os.read(ready_read,1),b'1')
            result=runner.verify_server_lock_absent(self.args.prefix,base)
            self.assertFalse(result['absent']);self.assertEqual(result['reason'],'server-lock-still-held')
        finally:
            os.write(stop_write,b'1');os.close(stop_write);os.close(ready_read)
            _,status=os.waitpid(pid,0)
        self.assertEqual(status,0)
        self.assertTrue(runner.verify_server_lock_absent(self.args.prefix,base)['absent'])
        self.assertEqual(lock.read_bytes(),b'unchanged')
        lock.unlink();lock.symlink_to('/missing/unsafe-lock')
        with self.assertRaises(OSError):runner.verify_server_lock_absent(self.args.prefix,base)

"""Offline PE metadata regressions; fixtures are never executable modules."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE/'tools'))
import pe_artifact


def fixture(arch, dependency='KERNEL32.dll'):
    b = bytearray(0x1400)
    b[:2] = b'MZ'
    b[0x40:0x50] = b'Wine builtin DLL'
    struct.pack_into('<I', b, 0x3c, 0x80)
    b[0x80:0x84] = b'PE\0\0'
    machine, magic, optsize = (0x8664, 0x20b, 240) if arch=='x86_64' else (0x14c, 0x10b, 224)
    struct.pack_into('<HHIIIHH', b, 0x84, machine, 1, 0, 0, 0, optsize, 0x2022)
    optional = 0x98
    struct.pack_into('<H', b, optional, magic)
    directory = optional+(112 if arch=='x86_64' else 96)
    struct.pack_into('<I',b,directory-4,16)
    struct.pack_into('<IIII', b, directory, 0x1000, 0x180, 0x1300, 40)
    struct.pack_into('<8sIIII', b, optional+optsize, b'.rdata', 0x1000, 0x1000, 0x1000, 0x200)
    struct.pack_into('<IIIIII', b, 0x210, 1, 2, 2, 0x1040, 0x1050, 0x1060)
    struct.pack_into('<II', b, 0x240, 0x1800, 0x1810)
    struct.pack_into('<II', b, 0x250, 0x1080, 0x1090)
    struct.pack_into('<HH', b, 0x260, 0, 1)
    for start,name in [(0x280,'cuAlpha'),(0x290,'cuBeta'),(0x580,dependency)]:
        b[start:start+len(name)+1] = name.encode()+b'\0'
    struct.pack_into('<IIIII', b, 0x500, 0x1410, 0, 0, 0x1380, 0x1410)
    return bytes(b)


class PEContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.oracle = self.root/'oracle.tsv'
        self.oracle.write_text('cuAlpha\tstdcall\t1\t0\t1\ncuBeta\tstdcall\t2\t0\t2\n')

    def validate(self, arch, data):
        p = self.root/'nvcuda.dll'
        p.write_bytes(data)
        return pe_artifact.validate(p, arch, oracle=self.oracle)

    def test_both_machines_and_api_set_dependencies(self):
        for arch in ['x86_64','i686']:
            for library in ['KERNEL32.dll','api-ms-win-core-synch-l1-2-0.dll','ext-ms-win-test.dll','ucrtbase.dll']:
                with self.subTest(arch=arch, library=library):
                    info = self.validate(arch,fixture(arch,library))
                    self.assertEqual(info['exports'],[(1,'cuAlpha'),(2,'cuBeta')])
                    self.assertEqual(info['imports'],[library])

    def test_non_system_runtime_dlls_rejected_for_both_arches(self):
        for arch in ['x86_64','i686']:
            for library in ['libgcc_s_seh-1.dll','libgcc_s_dw2-1.dll','libwinpthread-1.dll','nvcuda.dll','amdhip64.dll','unknown.dll','api-ms-win-/evil.dll','ext-ms-win-..\\evil.dll','api-ms-win-test.dll:evil']:
                with self.subTest(arch=arch, library=library):
                    with self.assertRaisesRegex(ValueError,'unexpected PE runtime dependency'):
                        self.validate(arch,fixture(arch,library))

    def test_corrupt_or_wrong_surface_cannot_pass(self):
        cases=[]
        b=bytearray(fixture('x86_64')); b[0x40]=0; cases.append(b)
        b=bytearray(fixture('x86_64')); struct.pack_into('<I',b,0x250,0xfffffff0); cases.append(b)
        b=bytearray(fixture('x86_64')); struct.pack_into('<H',b,0x262,2); cases.append(b)
        b=bytearray(fixture('x86_64')); b[0x290:0x298]=b'cuAlpha\0'; cases.append(b)
        b=bytearray(fixture('x86_64')); struct.pack_into('<I',b,0x240,0x1080); cases.append(b)
        for data in cases:
            with self.assertRaises(ValueError): self.validate('x86_64',bytes(data))
        with self.assertRaises(ValueError): self.validate('i686',fixture('x86_64'))

    def test_delayed_or_hidden_import_directories_cannot_bypass_gate(self):
        for arch in ['x86_64','i686']:
            directory=0x98+(112 if arch=='x86_64' else 96)
            b=bytearray(fixture(arch));struct.pack_into('<II',b,directory+13*8,0x1450,32)
            with self.assertRaisesRegex(ValueError,'delayed import'):
                self.validate(arch,bytes(b))
            for count in [0,1,17,0xffffffff]:
                b=bytearray(fixture(arch));struct.pack_into('<I',b,directory-4,count)
                with self.assertRaisesRegex(ValueError,'directory extent'):
                    self.validate(arch,bytes(b))

    def test_final_link_is_static_libgcc_and_exact_def_for_each_arch(self):
        archive=self.root/'Rust archive.a'; archive.write_bytes(b'fixture only')
        log=self.root/'native-libs.log'; log.write_text('note: native-static-libs: -lkernel32 -lntdll -lkernel32\n')
        for arch,count in [('x86_64',788),('i686',690)]:
            args=[sys.executable,str(HERE/'tools/link_pe.py'),'--triplet',arch,
                  '--archive',str(archive),'--native-libs-log',str(log),
                  '--output',str(self.root/'nvcuda.dll'),'--dry-run']
            result=subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE)
            command=json.loads(result.stdout)
            self.assertIn('-static-libgcc',command)
            self.assertIn('-Wl,--exclude-all-symbols',command)
            self.assertEqual(command[-5:],['-lkernel32','-lntdll','-lkernel32','-o',str(self.root/'nvcuda.dll')])
            lines=archive.with_suffix('.exports.def').read_text().splitlines()[2:]
            self.assertEqual(len(lines),count)
            self.assertEqual('-Wl,--kill-at' in command,arch=='i686')


if __name__=='__main__': unittest.main()

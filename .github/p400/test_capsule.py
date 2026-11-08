import io,json,os,pathlib,shutil,stat,subprocess,sys,tarfile,tempfile,unittest,zipfile
from capsule import PINS,IMAGES,COMPAT,CPU,sha,write,inventory,HERE
WORKFLOW="1"*40
class Capsules(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);self.dest=self.root/"download";self.dest.mkdir()
 def tearDown(self):self.tmp.cleanup()
 def api(self):
  run=dict(id=7,repository=dict(id=3,full_name="chkuendig/ZLUDA"),head_repository=dict(full_name="chkuendig/ZLUDA"),head_branch="ci/p400-capsules-20261107",head_sha=WORKFLOW,path=".github/workflows/p400-capsules.yml",event="push",status="completed",conclusion="success",run_attempt=1)
  artifact=dict(id=9,name=f"p400-reference-{PINS['reference']['commit']}-7-1",expired=False,size_in_bytes=1,digest="sha256:"+"2"*64,workflow_run=dict(id=7,head_sha=WORKFLOW,repository_id=3))
  jobs=dict(jobs=[dict(id=8,run_id=7,name="build (reference, pinned, llvm)",status="completed",conclusion="success",labels=["ubuntu-22.04"])])
  for n,v in [("RUN.json",run),("ARTIFACT.json",artifact),("JOBS.json",jobs)]:write(self.dest/n,v)
  return artifact
 def run_script(self,name,*args,free=100*2**30):
  command=[sys.executable,str(HERE/name),str(self.dest),*args]
  if name=="verify-extract.py":
   wrapper="import runpy,sys,types;from unittest.mock import patch;sys.path.insert(0,"+repr(str(HERE))+");sys.argv="+repr(command[1:])+"\nwith patch('shutil.disk_usage',return_value=types.SimpleNamespace(free="+str(free)+")):runpy.run_path(sys.argv[0],run_name='__main__')"
   command=[sys.executable,"-c",wrapper]
  return subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
 def archive(self,mutation=None,extra=None):
  root=self.root/"payload";root.mkdir();(root/"bin").mkdir();(root/"runtime/lib").mkdir(parents=True);(root/"receipts/cpu-nvcuda").mkdir(parents=True)
  pin=PINS["reference"];source=dict(commit=pin["commit"],tree=pin["tree"],source_clean=True,LLVM_BUILT=True,lfs={"public.bc":"0"*64},gitlinks={"ext/llvm-project":dict(commit=pin["llvm_gitlink"]),"ext/HiGHS":dict(commit=pin["highs_gitlink"])})
  provenance=dict(workflow_commit=WORKFLOW,run_id=7,run_attempt=1,repository="chkuendig/ZLUDA",ref="refs/heads/ci/p400-capsules-20261107",images=IMAGES,profile="debug",rustc="release: 1.99.0\n",helper_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file()})
  binary=root/"bin/nvcuda";binary.write_bytes(b"public synthetic ELF fixture");binary.chmod(0o755)
  freshness=dict(target_initially_empty=True,fresh_true=0,LLVM_BUILT=True,llvm_build_script_events=[{"public":True}],llvm_compiled_build_script=[{"public":True}])
  cpu=dict(harness_sha256=sha(binary),results=[dict(test=n,returncode=0,one_test_passed=True) for n in CPU["reference"]["nvcuda"]])
  for path,value in [("SOURCE.json",source),("PROVENANCE.json",provenance),("receipts/FRESHNESS.json",freshness),("receipts/cpu-nvcuda/RESULT.json",cpu)]:write(root/path,value)
  runtime={}
  for name in ["ld-linux-x86-64.so.2",*COMPAT]:
   f=root/"runtime/lib"/name;f.write_bytes(("synthetic "+name).encode());f.chmod(0o755);runtime[name]=dict(sha256=sha(f),needed=[])
  manifest=dict(schema=1,target="reference",source=source,provenance=provenance,files=inventory(root),freshness=freshness,executables={"bin/nvcuda":dict(fresh=False,sha256=sha(binary))},cpu={"nvcuda":cpu},loader="runtime/lib/ld-linux-x86-64.so.2",runtime=runtime,glibc_compatibility=dict(included={n:{} for n in COMPAT}))
  write(root/"MANIFEST.json",manifest)
  (root/"SHA256SUMS").write_text("".join(sha(p)+"  "+p.relative_to(root).as_posix()+"\n" for p in sorted(root.rglob("*")) if p.is_file() and p.name!="SHA256SUMS"))
  if mutation:mutation(root)
  tar_path=self.root/"capsule.tar.gz"
  with tarfile.open(tar_path,"w:gz") as tar:
   for p in sorted(root.rglob("*")):tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
   if extra:extra(tar)
  zip_path=self.dest/"artifact.zip"
  with zipfile.ZipFile(zip_path,"w") as z:z.write(tar_path,"capsule.tar.gz")
  a=self.api();a.update(size_in_bytes=zip_path.stat().st_size,digest="sha256:"+sha(zip_path));write(self.dest/"ARTIFACT.json",a)
  rc=self.run_script("verify-api.py","reference",WORKFLOW);self.assertEqual(rc.returncode,0,rc.stderr)
 def test_valid_capsule_and_cpu_identity(self):
  self.archive();p=self.run_script("verify-extract.py","reference");self.assertEqual(p.returncode,0,p.stderr);self.assertTrue((self.dest/"ADMISSION.json").exists())
 def test_low_space_rejected_before_extract(self):
  self.archive();p=self.run_script("verify-extract.py","reference",free=0);self.assertNotEqual(p.returncode,0);self.assertIn("30GiB projected floor",p.stderr);self.assertFalse((self.dest/"capsule").exists())
 def test_wrong_workflow_and_expired_artifact(self):
  a=self.api();self.assertNotEqual(self.run_script("verify-api.py","reference","f"*40).returncode,0)
  a["expired"]=True;write(self.dest/"ARTIFACT.json",a);self.assertNotEqual(self.run_script("verify-api.py","reference",WORKFLOW).returncode,0)
 def test_bad_outer_digest_stops_before_tar_write(self):
  self.archive();p=self.dest/"artifact.zip";p.write_bytes(p.read_bytes()+b"tamper")
  self.assertNotEqual(self.run_script("verify-extract.py","reference").returncode,0);self.assertFalse((self.dest/"capsule.tar.gz").exists())
 def test_payload_tamper_fails_manifest_hash(self):
  self.archive(lambda r:(r/"bin/nvcuda").write_bytes(b"different public fixture"));self.assertNotEqual(self.run_script("verify-extract.py","reference").returncode,0);self.assertFalse((self.dest/"ADMISSION.json").exists())
 def test_parent_traversal_rejected_before_extract(self):
  def extra(t):
   m=tarfile.TarInfo("../escape");m.size=1;t.addfile(m,io.BytesIO(b"x"))
  self.archive(extra=extra);self.assertNotEqual(self.run_script("verify-extract.py","reference").returncode,0);self.assertFalse((self.dest/"capsule").exists())
 def test_duplicate_member_rejected(self):
  def extra(t):
   m=tarfile.TarInfo("bin/nvcuda");m.size=1;t.addfile(m,io.BytesIO(b"x"))
  self.archive(extra=extra);self.assertNotEqual(self.run_script("verify-extract.py","reference").returncode,0)
 def test_symlink_parent_rejected_before_extract(self):
  def extra(t):
   m=tarfile.TarInfo("runtime/link");m.type=tarfile.SYMTYPE;m.linkname="lib";t.addfile(m)
   m=tarfile.TarInfo("runtime/link/injected");m.size=1;t.addfile(m,io.BytesIO(b"x"))
  self.archive(extra=extra);self.assertNotEqual(self.run_script("verify-extract.py","reference").returncode,0);self.assertFalse((self.dest/"capsule").exists())
 def cpu_suite(self,mode):
  binary=self.root/"ptx";output=self.root/("suite-"+mode)
  binary.write_text("#!"+sys.executable+"\n"+'''import os,pathlib,sys
assert sys.argv.count('--skip')==2 and '_amdgpu' in sys.argv and '_cuda' in sys.argv
if '--list' in sys.argv:
 print('test::spirv_run::legacy_slct_scalar_types: test\\nsecond_cpu: test');sys.exit(0)
assert '--test-threads=1' in sys.argv
mode=os.environ['PTX_FAKE_MODE']
if mode=='fail':
 p=pathlib.Path(os.environ['TEST_PTX_LLVM_FAIL_DIR']);p.mkdir(parents=True);(p/'public.ll').write_text('; public synthetic failure\\n')
passed,failed=(2,0) if mode=='ok' else (0,2) if mode=='fail' else (0,0)
print(f'test result: {"FAILED" if failed else "ok"}. {passed} passed; {failed} failed; 0 ignored;')
sys.exit(101 if failed else 0)
''')
  binary.chmod(0o755)
  result=subprocess.run([sys.executable,str(HERE/"run-ptx-cpu.py"),str(binary),str(output)],capture_output=True,text=True,env=dict(os.environ,PTX_FAKE_MODE=mode,PYTHONDONTWRITEBYTECODE="1"))
  return result,json.loads((output/"RESULT.json").read_text()),output
 def test_cpu_suite_filters_and_records_pass(self):
  result,receipt,_=self.cpu_suite('ok');self.assertEqual(result.returncode,0,result.stderr)
  self.assertTrue(receipt['passed']);self.assertEqual(receipt['counts']['passed'],2)
 def test_cpu_suite_preserves_failure_and_rejects_zero_tests(self):
  result,receipt,out=self.cpu_suite('fail');self.assertNotEqual(result.returncode,0);self.assertFalse(receipt['passed'])
  self.assertEqual(receipt['returncode'],101);self.assertTrue((out/'llvm-failures/public.ll').exists())
  result,receipt,_=self.cpu_suite('zero');self.assertNotEqual(result.returncode,0);self.assertFalse(receipt['passed'])
 def zoc_smoke(self,mode):
  binary=self.root/'zoc';loader=self.root/'loader';runtime=self.root/'runtime';runtime.mkdir();out=self.root/'zoc-smoke'
  loader.write_text('#!'+sys.executable+'\nimport os,sys\nassert sys.argv[1]=="--library-path"\nos.execv(sys.argv[3],sys.argv[3:])\n');loader.chmod(0o755)
  binary.write_text('#!'+sys.executable+'\n'+'''import os,pathlib,sys
assert sys.argv[sys.argv.index('--arch')+1]=='gfx1030'
source=pathlib.Path(sys.argv[-1]);out=pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1]);out.mkdir()
if source.stem=='unknown-symbol':
 print('Error: PTX TranslateError::UnknownSymbol',file=sys.stderr)
 if os.environ['ZOC_FAKE_MODE']=='ok':print('Caused by: Unknown symbol: "zoc_smoke_missing_symbol"',file=sys.stderr)
 sys.exit(1)
header=bytearray(64);header[:6]=b'\\x7fELF\\x02\\x01';header[18:20]=(224).to_bytes(2,'little')
(out/(source.stem+'.elf')).write_bytes(header);(out/(source.stem+'.ll')).write_text('; public synthetic IR\\n')
print('Pass "compile_bitcode" took 1ms')
''');binary.chmod(0o755)
  result=subprocess.run([sys.executable,str(HERE/'zoc-smoke.py'),str(binary),str(out),str(loader),str(runtime)],capture_output=True,text=True,env=dict(os.environ,ZOC_FAKE_MODE=mode,PYTHONDONTWRITEBYTECODE='1'))
  return result,json.loads((out/'RESULT.json').read_text())
 def test_zoc_public_smoke_uses_packaged_loader_and_both_widths(self):
  result,receipt=self.zoc_smoke('ok');self.assertEqual(result.returncode,0,result.stderr)
  self.assertEqual([r['case'] for r in receipt['results']],['ptx32','ptx64','unknown-symbol']);self.assertTrue(all(r['passed'] for r in receipt['results']))
 def test_zoc_smoke_rejects_erased_inner_error(self):
  result,receipt=self.zoc_smoke('outer-only');self.assertNotEqual(result.returncode,0);self.assertFalse(receipt['results'][-1]['passed'])
if __name__=="__main__":unittest.main()

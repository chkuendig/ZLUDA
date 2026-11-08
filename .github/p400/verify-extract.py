import hashlib,os,pathlib,re,shutil,stat,sys,tarfile,zipfile
from capsule import PINS,IMAGES,CPU,COMPAT,sha,load,write,safe_name,inventory
root=pathlib.Path(sys.argv[1]);target=sys.argv[2];pin=PINS[target]
api=load(root/"API-ADMISSION.json");a=load(root/"ARTIFACT.json");zip_path=root/"artifact.zip"
assert api["target"]==target and api["artifact"]==a["id"] and api["digest"]==a["digest"]
assert zip_path.stat().st_size==a["size_in_bytes"] and "sha256:"+sha(zip_path)==a["digest"]
with zipfile.ZipFile(zip_path) as z:
 members=z.infolist();assert len(members)==1 and members[0].filename=="capsule.tar.gz"
 m=members[0];assert not m.is_dir() and not m.flag_bits&1 and 0<m.file_size<=2*2**30
 mode=m.external_attr>>16;assert stat.S_IFMT(mode) in [0,stat.S_IFREG]
 tar_path=root/"capsule.tar.gz"
 with z.open(m) as source,tar_path.open("xb") as dest:
  copied=0
  for block in iter(lambda:source.read(1024*1024),b""):
   copied+=len(block);assert copied<=m.file_size;dest.write(block)
 assert copied==m.file_size
capsule=root/"capsule";assert not capsule.exists()
with tarfile.open(tar_path,"r:gz") as tar:
 members=tar.getmembers();assert len(members)<=20000
 names={};size=0
 for m in members:
  path=safe_name(m.name);assert m.name not in names
  assert m.isfile() or m.isdir() or m.issym();assert not m.islnk() and not m.sparse
  assert m.mode&~0o777==0,"Special permission bits"
  assert not any(k.startswith("GNU.sparse") for k in m.pax_headers)
  if m.issym():
   assert m.linkname and not pathlib.PurePosixPath(m.linkname).is_absolute()
   parts=[]
   for part in (*path.parent.parts,*pathlib.PurePosixPath(m.linkname).parts):
    if part=="..":assert parts;parts.pop()
    elif part!=".":parts.append(part)
  if m.isfile():assert m.size>=0;size+=m.size
  names[m.name]=m
 assert size<=4*2**30
 assert zip_path.stat().st_size+tar_path.stat().st_size+size<=6*2**30,"6GiB download/unpack budget"
 assert shutil.disk_usage(root).free-size>=30*2**30,"30GiB projected floor"
 for name,m in names.items():
  for parent in pathlib.PurePosixPath(name).parents:
   if str(parent)!=".":assert str(parent) in names and names[str(parent)].isdir(),"Missing or symlink parent"
 assert names["MANIFEST.json"].isfile() and names["MANIFEST.json"].size<=8*2**20
 assert names["SHA256SUMS"].isfile() and names["SHA256SUMS"].size<=8*2**20
 capsule.mkdir()
 for m in sorted(members,key=lambda x:(len(pathlib.PurePosixPath(x.name).parts),not x.isdir(),x.name)):
  dest=capsule/m.name
  if m.isdir():dest.mkdir(exist_ok=True)
  elif m.issym():dest.symlink_to(m.linkname)
  else:
   with tar.extractfile(m) as source,dest.open("xb") as output:
    copied=0
    for block in iter(lambda:source.read(1024*1024),b""):
     copied+=len(block);assert copied<=m.size;output.write(block)
   assert copied==m.size
  if not m.issym():dest.chmod(m.mode)
manifest=load(capsule/"MANIFEST.json");assert manifest["schema"]==1 and manifest["target"]==target
assert manifest["files"]==inventory(capsule),"File/link/mode/hash coverage mismatch"
assert set(names)==set(manifest["files"])|{"MANIFEST.json","SHA256SUMS"}
for name,row in manifest["files"].items():
 if row["kind"]=="symlink":assert (capsule/name).resolve(strict=True).is_relative_to(capsule.resolve())
sums={}
for line in (capsule/"SHA256SUMS").read_text().splitlines():
 digest,name=line.split("  ",1);safe_name(name);assert name not in sums and re.fullmatch(r"[a-f0-9]{64}",digest);sums[name]=digest
files={n for n,r in manifest["files"].items() if r["kind"]=="file"}|{"MANIFEST.json"}
assert set(sums)==files and all(sha(capsule/n)==digest for n,digest in sums.items())
s=manifest["source"];assert s==load(capsule/"SOURCE.json") and s["commit"]==pin["commit"] and s["tree"]==pin["tree"] and s["source_clean"]
assert s["LLVM_BUILT"]==pin["llvm_built"] and s["gitlinks"]["ext/llvm-project"]["commit"]==pin["llvm_gitlink"] and s["gitlinks"]["ext/HiGHS"]["commit"]==pin["highs_gitlink"]
assert bool(s["lfs"])==pin["llvm_built"]
p=manifest["provenance"];assert p==load(capsule/"PROVENANCE.json")
assert p["workflow_commit"]==api["workflow_commit"] and p["run_id"]==api["run"] and p["run_attempt"]==1
assert p["repository"]=="chkuendig/ZLUDA" and p["ref"]=="refs/heads/ci/p400-capsules-20261107"
assert p["images"]==IMAGES and p["profile"]==("release" if target=="zoc" else "debug") and "release: 1.99.0\n" in p["rustc"]
assert set(p["helper_sha256"])=={h.name for h in pathlib.Path(__file__).parent.iterdir() if h.is_file()}
for helper,digest in p["helper_sha256"].items():assert sha(pathlib.Path(__file__).parent/helper)==digest,helper
fresh=manifest["freshness"];assert fresh==load(capsule/"receipts/FRESHNESS.json") and fresh["target_initially_empty"] and fresh["fresh_true"]==0 and fresh["LLVM_BUILT"]==pin["llvm_built"]
assert bool(fresh["llvm_build_script_events"])==pin["llvm_built"] and bool(fresh["llvm_compiled_build_script"])==pin["llvm_built"]
expected={"jc2":{"bin/nvcuda","bin/ptx","bin/ptx_parser"},"reference":{"bin/nvcuda"},"trace":{"bin/zluda_trace-tests","lib/libzluda_trace.so"},"zoc":{"bin/zoc"}}[target]
assert set(manifest["executables"])==expected
for name,row in manifest["executables"].items():assert row["fresh"] is False and row["sha256"]==sha(capsule/name)
assert manifest["cpu"].keys()==CPU[target].keys()
for name,cases in CPU[target].items():
 receipt=load(capsule/("receipts/cpu-"+name+"/RESULT.json"));assert receipt==manifest["cpu"][name]
 assert receipt["harness_sha256"]==sha(capsule/"bin"/name)
 assert [r["test"] for r in receipt["results"]]==cases
 assert all(r["returncode"]==0 and r["one_test_passed"] is True for r in receipt["results"])
if target=="jc2":
 assert "ci_build" in manifest["executables"]["bin/ptx"]["features"]
 suite=load(capsule/"receipts/cpu-ptx-full/RESULT.json");assert suite==manifest["ptx_cpu_suite"]
 assert suite["harness_sha256"]==sha(capsule/"bin/ptx") and suite["returncode"]==0 and suite["passed"] is True
 assert suite["skip_filters"]==["_amdgpu","_cuda"] and suite["selected_tests"]
 assert "test::spirv_run::legacy_slct_scalar_types" in suite["selected_tests"]
 assert not any("_amdgpu" in name or "_cuda" in name for name in suite["selected_tests"])
 assert suite["counts"]["passed"]>0 and suite["counts"]["failed"]==0
 assert suite["counts"]["passed"]+suite["counts"]["ignored"]==len(suite["selected_tests"])
loader=capsule/manifest["loader"];assert loader.is_file() and manifest["loader"].startswith("runtime/lib/")
assert set(manifest["glibc_compatibility"]["included"])==set(COMPAT)
for name,row in manifest["runtime"].items():
 assert row["sha256"]==sha(capsule/"runtime/lib"/name)
 assert not re.match(r"^(libcuda(?:\.|$)|libnvidia|libcudart|libnvcuvid)",name)
 assert all((capsule/"runtime/lib"/n).is_file() for n in row["needed"])
if target=="trace":
 assert fresh["trace_cdylib_fresh"] is False
 assert manifest["trace_smoke"]["status"]=="PASS_SYNTHETIC_COMPLETION_LOGGING_ONLY"
 assert manifest["trace_smoke"]["downstream_census"]=="PARTIAL"
 assert not s["LLVM_BUILT"]
if target=="zoc":
 assert s["backend_base"]==pin["backend_base"] and s["diagnostic_only_paths"]==["compiler/src/main.rs"]
 profile=manifest["executables"]["bin/zoc"]["profile"];assert profile["opt_level"]=="3" and profile["debug_assertions"] is False and profile["test"] is False
 smoke=load(capsule/"receipts/zoc-smoke/RESULT.json");assert smoke==manifest["zoc_smoke"]
 assert smoke["tool_sha256"]==sha(capsule/"bin/zoc") and smoke["loader_sha256"]==sha(loader)
 assert smoke["status"]=="PASS_PUBLIC_CPU_COMPILER_SMOKES" and smoke["gpu_execution"] is False and smoke["private_inputs"] is False
 assert [r["case"] for r in smoke["results"]]==["ptx32","ptx64","unknown-symbol"] and all(r["passed"] is True for r in smoke["results"])
 assert [r["returncode"] for r in smoke["results"]]==[0,0,1]
write(root/"ADMISSION.json",dict(status="ADMITTED_PUBLIC_CAPSULE_NOT_GPU_RESULT",target=target,source_commit=s["commit"],llvm_gitlink=pin["llvm_gitlink"],workflow_commit=p["workflow_commit"],run_id=p["run_id"],artifact=a["id"],zip_sha256=sha(zip_path),tar_sha256=sha(tar_path),manifest_sha256=sha(capsule/"MANIFEST.json"),capsule=str(capsule),payload_bytes=size,loader=manifest["loader"],PE32="NOT_ADMITTED",local_gpu_execution="requires lane clearance and GO; any pause VOID"))

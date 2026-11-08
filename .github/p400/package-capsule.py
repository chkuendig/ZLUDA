import os,pathlib,re,shutil,stat,subprocess,tarfile
from capsule import PINS,IMAGES,CPU,COMPAT,sha,load,write,inventory
out=pathlib.Path("/out/diagnostics");root=pathlib.Path("/out/capsule");root.mkdir()
(root/"bin").mkdir();(root/"lib").mkdir();runtime=root/"runtime/lib";runtime.mkdir(parents=True)
target=os.environ["TARGET"];pin=PINS[target]
source=load(out/"SOURCE.json");assert source["commit"]==pin["commit"] and source["LLVM_BUILT"]==pin["llvm_built"]
def command(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=90)
def events(name):return [__import__("json").loads(l) for l in (out/name).read_text().splitlines() if l.strip()]
test_events=events("build.json" if target=="zoc" else "test-build.json");artifacts=[e for e in test_events if e["reason"]=="compiler-artifact"]
assert artifacts and all(e["fresh"] is False for e in artifacts),"Fresh target required"
llvm_events=[e for e in test_events if e["reason"]=="build-script-executed" and "/llvm_zluda" in e["package_id"]]
llvm_scripts=[e for e in artifacts if e["target"]["src_path"]=="/src/llvm_zluda/build.rs"]
if pin["llvm_built"]:
 assert llvm_events and llvm_scripts and all(not e["fresh"] for e in llvm_scripts)
 for e in llvm_scripts:
  for p in e["filenames"]:assert pathlib.Path(p).is_file()
else:assert not llvm_events and not llvm_scripts
fresh={"target_initially_empty":True,"compiler_artifacts":len(artifacts),"fresh_true":0,"LLVM_BUILT":pin["llvm_built"],"llvm_build_script_events":llvm_events,"llvm_compiled_build_script":[dict(path=p,sha256=sha(p)) for e in llvm_scripts for p in e["filenames"]]}
selected={};executables=[]
expected={"jc2":["nvcuda","ptx","ptx_parser"],"reference":["nvcuda"],"trace":["zluda_trace"],"zoc":["zoc"]}[target]
for name in expected:
 matches=[e for e in artifacts if e["profile"]["test"]==(target!="zoc") and e.get("executable") and e["target"]["name"]==name]
 assert len(matches)==1,(name,len(matches))
 e=matches[0];original=pathlib.Path(e["executable"]);dest=root/"bin"/("zluda_trace-tests" if target=="trace" else name)
 if target=="jc2" and name=="ptx":assert "ci_build" in e["features"]
 shutil.copy2(original,dest);assert sha(original)==sha(dest)
 selected[str(dest.relative_to(root))]=dict(cargo_path=str(original),sha256=sha(original),fresh=e["fresh"],target=name,features=e["features"])
 if target=="zoc":
  assert e["target"]["kind"]==["bin"] and e["profile"]["opt_level"]=="3" and e["profile"]["debug_assertions"] is False
  selected[str(dest.relative_to(root))]["profile"]=e["profile"]
 executables.append(dest)
if target=="trace":
 tree=(out/"CARGO-TREE.txt").read_text();assert not re.search(r"\b(?:llvm_zluda|ptx|highs-sys) v",tree)
 builds=[e for e in events("build.json") if e["reason"]=="compiler-artifact" and e["target"]["name"]=="zluda_trace" and "cdylib" in e["target"]["kind"]]
 assert len(builds)==1 and builds[0]["fresh"] is False
 paths=[pathlib.Path(p) for p in builds[0]["filenames"] if p.endswith(".so")];assert len(paths)==1
 dest=root/"lib/libzluda_trace.so";shutil.copy2(paths[0],dest);assert sha(paths[0])==sha(dest)
 selected["lib/libzluda_trace.so"]=dict(cargo_path=str(paths[0]),sha256=sha(dest),fresh=False,target="zluda_trace")
 executables.append(dest);fresh["trace_cdylib_fresh"]=False
write(out/"FRESHNESS.json",fresh)
cpu_ok=True
for name,cases in CPU[target].items():
 env=dict(os.environ,TEST_PTX_LLVM_FAIL_DIR=str(out/("cpu-"+name)/"llvm-failures"))
 cpu_ok &= subprocess.run(["python3","/plan/run-exact.py",str(root/"bin"/name),str(out/("cpu-"+name)),*cases],env=env).returncode==0
if target=="jc2":
 cpu_ok &= subprocess.run(["python3","/plan/run-ptx-cpu.py",str(root/"bin/ptx"),str(out/"cpu-ptx-full")]).returncode==0
assert cpu_ok,"CPU qualification failed; retain diagnostics and LLVM failure outputs"
if target=="trace":
 binary=root/"bin/zluda_trace-tests";listing=command([str(binary),"--list"]);(out/"TESTS-zluda_trace.txt").write_text(listing)
 p=subprocess.run([str(binary),"--test-threads=1"],capture_output=True,text=True,timeout=90)
 (out/"TRACE-UNIT.log").write_text(p.stdout+p.stderr);assert p.returncode==0
 write(out/"TRACE-UNIT.json",dict(returncode=p.returncode,listed_tests=len(re.findall(r": test$",listing,re.M)),zero_tests_are_runtime_evidence=False))
 shutil.copy2(root/"lib/libzluda_trace.so",out/"libzluda_trace.so")
 (out/"fake-cuda.c").write_text("int cuInit(unsigned int f){(void)f;return 801;}\nint cuDriverGetVersion(int *v){*v=12345;return 0;}\n")
 subprocess.run(["cc","-shared","-fPIC","-O2","-o",str(out/"libfake-cuda.so"),str(out/"fake-cuda.c")],check=True)
 (out/"smoke-log").mkdir()
 env=dict(os.environ,ZLUDA_CUDA_LIB=str(out/"libfake-cuda.so"),ZLUDA_LOG_DIR=str(out/"smoke-log"));env.pop("ZLUDA_OVERRIDE_COMPUTE_CAPABILITY",None)
 p=subprocess.run(["python3","/plan/trace-smoke.py"],capture_output=True,text=True,timeout=90,env=env)
 (out/"smoke.stdout").write_text(p.stdout);(out/"smoke.stderr").write_text(p.stderr);assert p.returncode==0
 smoke=load(out/"SMOKE.json");smoke.update(environment="cloud Ubuntu22.04/glibc2.35; not retained SDK2.41 or Deck admission",downstream_census="PARTIAL",current_path_hold="UNCHANGED");write(out/"SMOKE.json",smoke)
closure={};interpreter=None
reject=re.compile(r"^(?:libcuda(?:\.|$)|libnvidia|libcudart|libnvcuvid)")
def dynamic(path,label):
 text=command(["readelf","-dW",str(path)]);(out/("DYNAMIC-"+label+".txt")).write_text(text)
 needed=re.findall(r"\(NEEDED\).*?\[(.*?)\]",text)
 assert all("/" not in n and not reject.match(n) for n in needed),needed
 paths=re.findall(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]",text)
 assert not any("/target" in p or "/src" in p for p in paths),paths
 if target=="trace" and path==root/"lib/libzluda_trace.so":assert not paths and not any(re.search(r"hip|hsa|comgr|rocm|LLVM|amd",n,re.I) for n in needed),needed
 versions=command(["readelf","-VW",str(path)]);(out/("VERSIONS-"+label+".txt")).write_text(versions)
 if target=="trace" and path==root/"lib/libzluda_trace.so":
  assert "GLIBC_PRIVATE" not in versions
  assert all(tuple(map(int,v.split(".")))<=(2,35) for v in re.findall(r"GLIBC_(\d+\.\d+(?:\.\d+)?)",versions))
 (out/("ELF-"+label+".txt")).write_text(command(["readelf","-hW",str(path)]))
 return needed
def library(path,alias=None):
 path=pathlib.Path(path);real=path.resolve(strict=True);name=real.name;assert not reject.match(name)
 dest=runtime/name
 if dest.exists():assert sha(dest)==sha(real),(name,"Conflicting runtime providers")
 else:shutil.copy2(real,dest)
 aliases={path.name}
 soname=re.findall(r"\(SONAME\).*?\[(.*?)\]",command(["readelf","-dW",str(real)]));aliases.update(soname)
 if alias:aliases.add(alias)
 for a in aliases:
  assert a==pathlib.Path(a).name and not reject.match(a)
  link=runtime/a
  if a!=name:
   if link.is_symlink():assert os.readlink(link)==name
   elif link.exists():assert sha(link)==sha(real)
   else:link.symlink_to(name)
 closure[name]=dict(source=str(real),sha256=sha(real),aliases=sorted(aliases),needed=re.findall(r"\(NEEDED\).*?\[(.*?)\]",command(["readelf","-dW",str(real)])))
for binary in executables:
 label=binary.name;needed=dynamic(binary,label)
 header=command(["readelf","-lW",str(binary)]);m=re.search(r"Requesting program interpreter: (.*?)\]",header)
 if m:
  if interpreter:assert interpreter==m[1]
  interpreter=m[1];library(interpreter)
 text=command(["ldd",str(binary)]);(out/("LOADER-original-"+label+".txt")).write_text(text);assert "not found" not in text
 for line in text.splitlines():
  match=re.match(r"\s*(\S+) => (/\S+)",line)
  if match:library(match[2],match[1])
  else:
   match=re.match(r"\s*(/\S+) ",line)
   if match:library(match[1])
 assert all((runtime/n).exists() for n in needed)
assert interpreter
cache=command(["/sbin/ldconfig","-p"])
compat={}
for name in COMPAT:
 matches=re.findall(r"\s+"+re.escape(name)+r" \(libc6,x86-64[^)]*\) => (\S+)",cache);assert matches,name
 path=pathlib.Path(matches[0]);owner=command(["dpkg-query","-S",str(path)]).strip();assert owner.startswith("libc6:amd64:"),owner
 library(path,name);compat[name]=dict(source=str(path),sha256=sha(path),package=owner)
# Verify the closure of copied DSOs, including driver compatibility libraries.
for name,row in list(closure.items()):
 assert all((runtime/n).exists() for n in row["needed"]),(name,row["needed"])
loader=runtime/pathlib.Path(interpreter).name;assert loader.exists()
for binary in executables:
 text=command([str(loader),"--library-path",str(runtime),"--list",str(binary)]);(out/("LOADER-packaged-"+binary.name+".txt")).write_text(text);assert "not found" not in text
 for line in text.splitlines():
  m=re.match(r"\s*(\S+) => (/\S+)",line)
  if m:assert pathlib.Path(m[2]).resolve().is_relative_to(runtime.resolve()),line
  else:
   m=re.match(r"\s*(/\S+) ",line)
   if m:assert pathlib.Path(m[1]).resolve().is_relative_to(runtime.resolve()),line
if target=="zoc":subprocess.run(["python3","/plan/zoc-smoke.py",str(root/"bin/zoc"),str(out/"zoc-smoke"),str(loader),str(runtime)],check=True)
build_argv="cargo +1.99.0 build --locked --offline --release -p compiler --bin zoc -j4 --message-format=json" if target=="zoc" else "cargo +1.99.0 test --locked --offline [allowlisted packages]"+(" --features ptx/ci_build" if target=="jc2" else "")+" --lib -j4 --no-run --message-format=json"
provenance=dict(repository=os.environ["GITHUB_REPOSITORY"],workflow_commit=os.environ["GITHUB_SHA"],run_id=int(os.environ["GITHUB_RUN_ID"]),run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),job=os.environ["GITHUB_JOB"],ref=os.environ["GITHUB_REF"],images=IMAGES,actual_images=load(out/"IMAGE.json"),packages=dict(line.split("\t",1) for line in (out/"PACKAGES.txt").read_text().splitlines()),profile="release" if target=="zoc" else "debug",rustc=(out/"RUSTC.txt").read_text(),cargo=(out/"CARGO.txt").read_text(),rocm="6.3.4",build_env={k:os.environ[k] for k in ["RUSTUP_HOME","CARGO_HOME","CARGO_TARGET_DIR","CARGO_BUILD_JOBS","CMAKE_BUILD_PARALLEL_LEVEL","CARGO_INCREMENTAL","LD_LIBRARY_PATH"]},build_argv=build_argv,helper_sha256={p.name:sha(p) for p in pathlib.Path("/plan").iterdir() if p.is_file()})
write(root/"SOURCE.json",source);write(root/"PROVENANCE.json",provenance)
receipts=root/"receipts";receipts.mkdir()
for p in out.iterdir():
 if p.is_file() and p.suffix in [".json",".log",".txt",".stdout",".stderr",".c"]:shutil.copy2(p,receipts/p.name)
 elif p.is_dir() and (p.name.startswith("cpu-") or p.name in ["smoke-log","zoc-smoke"]):shutil.copytree(p,receipts/p.name)
manifest=dict(schema=1,target=target,source=source,provenance=provenance,executables=selected,loader=str(loader.relative_to(root)),runtime=closure,glibc_compatibility=dict(included=compat,reason="Same-build glibc compatibility DSOs for the separately injected driver; no NVIDIA bytes",libc_package=command(["dpkg-query","-W","-f=${Version}","libc6:amd64"])),freshness=fresh,cpu={name:load(out/("cpu-"+name)/"RESULT.json") for name in CPU[target]},trace_smoke=load(out/"SMOKE.json") if target=="trace" else None,files=inventory(root))
if target=="jc2":manifest["ptx_cpu_suite"]=load(out/"cpu-ptx-full/RESULT.json")
if target=="zoc":manifest["zoc_smoke"]=load(out/"zoc-smoke/RESULT.json")
write(root/"MANIFEST.json",manifest)
with (root/"SHA256SUMS").open("w") as f:
 for p in sorted(root.rglob("*")):
  if stat.S_ISREG(p.lstat().st_mode) and p.name!="SHA256SUMS":f.write(sha(p)+"  "+p.relative_to(root).as_posix()+"\n")
archive=pathlib.Path("/out/capsule.tar.gz")
assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())<=4*2**30
with tarfile.open(archive,"w:gz") as tar:
 for p in sorted(root.rglob("*")):
  info=tar.gettarinfo(str(p),arcname=p.relative_to(root).as_posix());assert not info.islnk()
  info.uid=info.gid=0;info.uname=info.gname="";info.mtime=0
  if info.isfile():
   with p.open("rb") as stream:tar.addfile(info,stream)
  else:tar.addfile(info)
assert archive.stat().st_size<=2*2**30
write(out/"CAPSULE.json",dict(sha256=sha(archive),bytes=archive.stat().st_size,manifest_sha256=sha(root/"MANIFEST.json")))

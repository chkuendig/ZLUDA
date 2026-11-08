import hashlib,json,os,pathlib,subprocess,sys,re
root=pathlib.Path(sys.argv[1]).resolve();target=sys.argv[2];out=pathlib.Path(sys.argv[3])
pin=json.loads(pathlib.Path(__file__).with_name("pins.json").read_text())[target]
def git(*args):return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
assert git("rev-parse","HEAD")==pin["commit"]
assert git("rev-parse","HEAD^{tree}")==pin["tree"]
for path,key in [("Cargo.lock","cargo_lock_blob"),("llvm_zluda/build.rs","llvm_build_rs_blob"),("llvm_zluda/Cargo.toml","llvm_manifest_blob")]:assert git("rev-parse","HEAD:"+path)==pin[key]
links={}
for path,key in [("ext/llvm-project","llvm_gitlink"),("ext/HiGHS","highs_gitlink")]:
 assert git("ls-tree","HEAD",path).split()[2]==pin[key]
 if pin["llvm_built"]:
  assert git("-C",path,"rev-parse","HEAD")==pin[key]
  status=subprocess.check_output(["git","-C",str(root),"submodule","status","--",path],text=True)
  assert status.startswith(" "+pin[key]),status
  assert not git("-C",path,"status","--porcelain","--untracked-files=no")
  links[path]=dict(commit=pin[key],status="INITIALIZED_CLEAN")
 else:
  assert not (root/path/".git").exists()
  links[path]=dict(commit=pin[key],status="NOT_BUILT")
assert not git("status","--porcelain","--untracked-files=no"),"Source must remain unmodified"
lfs={}
if pin["llvm_built"]:
 paths=[root/"llvm_zluda/src/device-libs/ockl.bc",*sorted((root/"ptx/lib").glob("*.bc"))]
 assert len(paths)>1
 for path in paths:
  with path.open("rb") as f:assert f.read(2)==b"BC",str(path)+" is not hydrated bitcode"
  name=str(path.relative_to(root));digest=hashlib.sha256(path.read_bytes()).hexdigest()
  pointer=subprocess.check_output(["git","-C",str(root),"show","HEAD:"+name]);oid=re.search(rb"oid sha256:([a-f0-9]{64})",pointer)
  assert oid and oid[1].decode()==digest,"LFS object identity mismatch"
  lfs[name]=digest
value=dict(target=target,commit=pin["commit"],tree=pin["tree"],gitlinks=links,source_clean=True,lfs=lfs,LLVM_BUILT=pin["llvm_built"])
if target=="zoc":
 base=pin["backend_base"];assert git("rev-parse","HEAD^")==base
 assert git("diff","--name-only",base,"HEAD")=="compiler/src/main.rs"
 trees={path:git("rev-parse","HEAD:"+path) for path in ["ptx","ptx_parser","llvm_zluda","zluda","Cargo.lock"]}
 assert all(tree==git("rev-parse",base+":"+path) for path,tree in trees.items())
 value.update(backend_base=base,backend_trees=trees,diagnostic_only_paths=["compiler/src/main.rs"])
if out.exists():assert json.loads(out.read_text())==value,"Source changed after build"
else:out.write_text(json.dumps(value,indent=2)+"\n")
print(json.dumps(value,indent=2))

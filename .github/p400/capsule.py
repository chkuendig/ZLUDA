import hashlib,json,os,pathlib,stat
HERE=pathlib.Path(__file__).resolve().parent
PINS=json.loads((HERE/"pins.json").read_text())
IMAGES=json.loads((HERE/"images.json").read_text())
COMPAT=["libm.so.6","libdl.so.2","libpthread.so.0","librt.so.1","libutil.so.1"]
CPU={
 "jc2":{
 "ptx_parser":["tests::target_keeps_double_instructions_and_eight_byte_storage","tests::cvt_keeps_double_precision_rounding_modes","tests::legacy_texture_declaration_and_signed_selection","tests::legacy_typed_globals32","tests::legacy_sreg16_reads"],
 "ptx":["test::spirv_run::legacy_f64_cvt_llvm","test::spirv_run::legacy_slct_llvm","test::spirv_run::legacy_typed_globals_llvm","test::spirv_run::legacy_typed_globals32_metadata","test::spirv_run::legacy_global_initializers32","test::spirv_run::legacy_sreg16_llvm","test::spirv_run::legacy_sreg16_32_truncates_helpers","test::spirv_run::ordinary_mov_u16_rejects_u32_register","pass::test::remove_unreachable_basic_blocks::kernel_rooted_calls","pass::test::remove_unreachable_basic_blocks::unreachable_call_does_not_keep_callee","pass::test::remove_unreachable_basic_blocks::reachable_declarations_keep_source_order","test::_Z9vectorAddPKfS0_Pfi_ptx"],
 "nvcuda":["r#impl::tex::pitched_tests::pitched_texture_shape_checks_rows_channels_and_overflow","r#impl::driver::tests::get_allocation"]},
 "reference":{"nvcuda":["r#impl::driver::tests::get_allocation"]},"trace":{},"zoc":{}}
def sha(path):
 with pathlib.Path(path).open("rb") as f:
  h=hashlib.sha256()
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def load(path):
 def unique(pairs):
  d={}
  for k,v in pairs:
   assert k not in d,"Duplicate JSON key: "+k
   d[k]=v
  return d
 return json.loads(pathlib.Path(path).read_text(),object_pairs_hook=unique)
def write(path,value):pathlib.Path(path).write_text(json.dumps(value,indent=2)+"\n")
def safe_name(name):
 p=pathlib.PurePosixPath(name)
 assert name and "\x00" not in name and "\\" not in name and not any(ord(c)<32 for c in name)
 assert name and not p.is_absolute() and all(x not in ("",".","..") for x in name.split("/")),name
 return p
def inventory(root):
 result={}
 for path in sorted(root.rglob("*")):
  name=path.relative_to(root).as_posix()
  if name in ["MANIFEST.json","SHA256SUMS"]:continue
  safe_name(name);s=path.lstat();row={"mode":stat.S_IMODE(s.st_mode)}
  if stat.S_ISLNK(s.st_mode):
   target=os.readlink(path);assert not pathlib.PurePosixPath(target).is_absolute()
   assert path.resolve().is_relative_to(root.resolve())
   row.update(kind="symlink",target=target)
  elif stat.S_ISREG(s.st_mode):row.update(kind="file",bytes=s.st_size,sha256=sha(path))
  elif stat.S_ISDIR(s.st_mode):row.update(kind="directory")
  else:raise AssertionError("Special payload file: "+name)
  result[name]=row
 return result

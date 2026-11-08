import os,pathlib,re,subprocess,sys
from capsule import sha,write

binary=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]);out.mkdir()
loader=pathlib.Path(sys.argv[3]).resolve();runtime=pathlib.Path(sys.argv[4]).resolve()
prefix=[str(loader),"--library-path",str(runtime),str(binary)]
env=dict(os.environ,RUST_BACKTRACE="1");env.pop("LD_PRELOAD",None);env.pop("LD_LIBRARY_PATH",None)
results=[]
for width in [32,64]:
 source=out/("public"+str(width)+".ptx")
 source.write_text(f""".version 2.3
.target sm_10
.address_size {width}
.visible .entry zoc_public{width}(.param .u{width} output)
{{
    .reg .u{width} address;
    ld.param.u{width} address, [output];
    st.global.u32 [address], 0x12345678;
    ret;
}}
""")
 output=out/("output"+str(width));argv=prefix+["--arch","gfx1030","--output-dir",str(output),str(source)]
 p=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=180)
 (out/(str(width)+".stdout")).write_text(p.stdout);(out/(str(width)+".stderr")).write_text(p.stderr)
 elf=output/(source.stem+".elf");ir=output/(source.stem+".ll")
 header=elf.read_bytes()[:20] if elf.is_file() else b""
 valid_elf=header[:6]==b"\x7fELF\x02\x01" and int.from_bytes(header[18:20],"little")==224
 passes=re.findall(r'^Pass "([^"]+)" took ',p.stdout,re.M)
 results.append(dict(case="ptx"+str(width),argv=argv,returncode=p.returncode,input_sha256=sha(source),
  passes=passes,elf_sha256=sha(elf) if elf.is_file() else None,
  passed=p.returncode==0 and valid_elf and ir.is_file() and "compile_bitcode" in passes))
source=out/"unknown-symbol.ptx"
source.write_text(""".version 6.5
.target sm_30
.address_size 64
.visible .entry zoc_error_chain_smoke()
{
    .reg .u32 result;
    mov.u32 result, zoc_smoke_missing_symbol;
    ret;
}
""")
argv=prefix+["--arch","gfx1030","--output-dir",str(out/"error-output"),str(source)]
p=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=30)
(out/"error.stdout").write_text(p.stdout);(out/"error.stderr").write_text(p.stderr)
details=['Error: PTX TranslateError::UnknownSymbol','Caused by: Unknown symbol: "zoc_smoke_missing_symbol"']
results.append(dict(case="unknown-symbol",argv=argv,returncode=p.returncode,input_sha256=sha(source),
 required_stderr=details,passed=p.returncode==1 and all(line in p.stderr for line in details)))
receipt=dict(status="PASS_PUBLIC_CPU_COMPILER_SMOKES" if all(r["passed"] for r in results) else "FAIL_PUBLIC_CPU_COMPILER_SMOKES",
 tool_sha256=sha(binary),loader_sha256=sha(loader),arch="gfx1030",gpu_execution=False,private_inputs=False,results=results)
write(out/"RESULT.json",receipt)
assert all(r["passed"] for r in results),"Public zoc CPU smoke failed; retain all diagnostics"

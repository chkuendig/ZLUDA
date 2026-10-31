// SPDX-License-Identifier: MIT OR Apache-2.0
use quote::{format_ident, quote, ToTokens};
use std::{collections::{BTreeMap, BTreeSet}, env, fs, path::PathBuf, process::Command};
#[path = "generate_dark.rs"] mod generate_dark;
#[path = "generate_wow64.rs"] mod generate_wow64;

fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |value, byte| {
        (value ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn declarations(source: &str, into: &mut BTreeMap<String, syn::Signature>) {
    for item in syn::parse_file(source).unwrap().items {
        if let syn::Item::ForeignMod(module) = item {
            for item in module.items {
                if let syn::ForeignItem::Fn(mut item) = item {
                    item.sig.inputs.pop_punct();
                    assert!(into.insert(item.sig.ident.to_string(), item.sig).is_none(),
                            "duplicate canonical/compatibility declaration");
                }
            }
        }
    }
}

// These APIs cannot execute a callback in the packaged release backend. Do
// not manufacture callback ownership for them or forward a Windows function
// pointer. An implementation in the backend must force an ownership review.
fn backend_status_guard(root: &std::path::Path, names: &BTreeSet<&str>, canonical: &BTreeMap<String, syn::Signature>, policy: &BTreeMap<&str,&str>) {
    let expected = BTreeSet::from(["cuGraphAddHostNode","cuGraphHostNodeGetParams","cuGraphHostNodeSetParams",
        "cuGraphExecHostNodeSetParams","cuGraphAddNode","cuGraphAddNode_v2","cuGraphNodeSetParams",
        "cuGraphExecNodeSetParams","cuGraphNodeGetParams","cuLaunchHostFunc","cuLaunchHostFunc_ptsz",
        "cuLaunchHostFunc_v2","cuLaunchHostFunc_v2_ptsz","cuUserObjectCreate"]);
    assert_eq!(names,&expected,"status-only routes must be reviewed individually");
    assert_eq!(names.iter().filter(|name| !canonical.contains_key(**name)).copied().collect::<BTreeSet<_>>(),
        BTreeSet::from(["cuGraphNodeGetParams","cuLaunchHostFunc_v2","cuLaunchHostFunc_v2_ptsz"]),
        "canonical/compatibility status-only split changed; review before calling");
    let path = root.join("zluda/src/lib.rs");
    println!("cargo:rerun-if-changed={}",path.display());
    let source = fs::read_to_string(path).expect("backend status guard needs zluda/src/lib.rs");
    let file = syn::parse_file(&source).unwrap();
    let declaration = file.items.iter().filter_map(|item| match item {
        syn::Item::Macro(item) if item.mac.path.segments.last().is_some_and(|part|part.ident=="cuda_function_declarations") => Some(item), _=>None
    }).collect::<Vec<_>>();
    assert_eq!(declaration.len(),1);
    let tokens = declaration[0].mac.tokens.to_string();
    assert!(tokens.starts_with("unimplemented ,"),"backend default implementation changed");
    for name in ["cuGraphDestroy","cuGraphExecDestroy","cuGraphExecUpdate_v2","cuGraphGetNodes",
                 "cuGraphInstantiateWithFlags","cuGraphLaunch"] {
        assert!(tokens.split(|c: char| !c.is_ascii_alphanumeric() && c!='_').any(|word| word==name),
            "implemented graph forwarding inventory changed: {name}");
        assert_eq!(policy.get(name).copied().unwrap_or("forwarded"),"forwarded",
            "implemented graph API must remain ordinary typed forwarding: {name}");
    }
    for name in names {
        assert!(!tokens.split(|c: char| !c.is_ascii_alphanumeric() && c!='_').any(|word| word==*name),
            "{name}: backend now implements this API; review graph/exec callback ownership first");
        // Compatibility names absent from canonical CUDA are deliberately
        // probed as optional symbols. They must also remain absent here.
        if !canonical.contains_key(*name) {
            assert!(!source.contains(name),"{name}: compatibility backend implementation needs review");
        }
    }
    let path = root.join("zluda/src/impl/mod.rs");
    println!("cargo:rerun-if-changed={}",path.display());
    let source = fs::read_to_string(path).expect("backend status guard needs zluda/src/impl/mod.rs");
    let file = syn::parse_file(&source).unwrap();
    let release = file.items.iter().filter_map(|item| match item {
        syn::Item::Fn(item) if item.sig.ident=="unimplemented" && item.attrs.iter().any(|attr|
            attr.to_token_stream().to_string().contains("not (debug_assertions)")) => Some(item), _=>None
    }).collect::<Vec<_>>();
    assert_eq!(release.len(),1);
    assert_eq!(release[0].block.to_token_stream().to_string(),"{ CUerror :: NOT_SUPPORTED }",
        "release backend unsupported status changed; review status-only routes");
    let expected_body: syn::ItemMacro=syn::parse_str(r#"
        macro_rules! unimplemented {
            ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
                $(
                    #[cfg_attr(not(test), no_mangle)]
                    #[allow(improper_ctypes)]
                    #[allow(improper_ctypes_definitions)]
                    #[allow(unused_variables)]
                    pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                        Err(r#impl::unimplemented())
                    }
                )*
            };
        }
    "#).unwrap();
    let actual=syn::parse_file(&fs::read_to_string(root.join("zluda/src/lib.rs")).unwrap()).unwrap();
    let definition=actual.items.iter().find_map(|item| match item {
        syn::Item::Macro(item) if item.ident.as_ref().is_some_and(|name| name=="unimplemented")=>Some(item),_=>None
    }).unwrap();
    assert_eq!(definition.mac.tokens.to_string(),expected_body.mac.tokens.to_string(),
        "default backend body changed: prove input-independent refusal before status-only calls");
}


static GENERATION_PHASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn phase(value:usize) {GENERATION_PHASE.store(value,std::sync::atomic::Ordering::Relaxed);}
fn main() {
    println!("cargo:rustc-check-cfg=cfg(zluda_wine_generation_failed)");
    for n in 0..=7 {println!("cargo:rustc-check-cfg=cfg(zluda_wine_generation_phase_{n})");}
    // A failure stays fatal, but exposes a source-free phase at a candidate location.
    if std::panic::catch_unwind(||generate(PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()),
            PathBuf::from(env::var_os("OUT_DIR").unwrap()))).is_err() {
        println!("cargo:rustc-cfg=zluda_wine_generation_failed");
        println!("cargo:rustc-cfg=zluda_wine_generation_phase_{}",GENERATION_PHASE.load(std::sync::atomic::Ordering::Relaxed));
    }
}

pub fn generate(here: PathBuf, out: PathBuf) {
    println!("cargo:rustc-check-cfg=cfg(zluda_wine_oracles)");
    let root = here.join("../..");
    println!("cargo:rustc-check-cfg=cfg(zluda_wine_check_only)");
    println!("cargo:rerun-if-env-changed=ZLUDA_WINE_DECLARATIONS_ONLY");
    let check_only = env::var("ZLUDA_WINE_DECLARATIONS_ONLY").as_deref() == Ok("1");
    if check_only { println!("cargo:rustc-cfg=zluda_wine_check_only"); }
    println!("cargo:rustc-check-cfg=cfg(zluda_wine_native_sdk)");
    println!("cargo:rerun-if-env-changed=WINE_SDK_ROOT");
    if !check_only && env::var("CARGO_CFG_TARGET_OS").unwrap()=="linux" {
        if let Some(sdk)=env::var_os("WINE_SDK_ROOT") {
            let ntdll=PathBuf::from(sdk).join("lib/ntdll.so");
            assert!(ntdll.is_absolute() && ntdll.is_file(),"matching SDK lib/ntdll.so is required");
            println!("cargo:rerun-if-changed={}",ntdll.display());
            println!("cargo:rustc-cfg=zluda_wine_native_sdk");
            println!("cargo:rustc-link-arg-cdylib={}",ntdll.display());
            println!("cargo:rustc-link-arg-cdylib=-Wl,-z,defs");
            println!("cargo:rustc-link-arg-cdylib=-Wl,-z,now");
        }
    }
    for name in ["ZLUDA_WINE_ABI_RECEIPT","ZLUDA_WINE_ABI_RECEIPT_SHA256"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if !check_only {
        assert!(env::var_os("WINE_SDK_ROOT").is_some(), "shipping requires matching prepared Wine SDK");
        let receipt=env::var_os("ZLUDA_WINE_ABI_RECEIPT").expect("shipping requires reviewer prepared-Wine attestation receipt");
        println!("cargo:rerun-if-changed={}",PathBuf::from(&receipt).display());
        println!("cargo:rerun-if-changed={}",here.join("tools/admit_wine.py").display());
        assert!(Command::new("python3").arg(here.join("tools/admit_wine.py"))
            .arg(receipt).arg(env::var_os("WINE_SDK_ROOT").unwrap()).status().unwrap().success());
    }
    phase(1);
    let types_path = root.join("cuda_types/src/cuda.rs");
    let declarations_path = root.join("cuda_macros/src/cuda.rs");
    for path in [&types_path, &declarations_path, &here.join("api/legacy.rs"),
                 &here.join("api/exports.tsv"), &here.join("api/wrappers.tsv"),
                 &here.join("api/proc-address.tsv"), &here.join("wine_adapter.c")] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    // cuda_types currently also builds HIP/math bindings. A generated view of
    // its canonical CUDA module avoids linking HIP/LLVM into the thin PE DLL.
    // Remove only the HIP error conversion, never a type or CUDA declaration.
    let types_source = fs::read_to_string(&types_path).unwrap();
    let mut types = syn::parse_file(&types_source).unwrap();
    let before = types.items.len();
    types.items.retain(|item| !matches!(item, syn::Item::Impl(item)
        if item.to_token_stream().to_string().contains("hip_runtime_sys")));
    assert_eq!(before - types.items.len(), 1, "re-audit the CUDA-only type view");
    let items = types.items;
    fs::write(out.join("cuda_types.rs"), quote! {
        #[allow(warnings)] pub mod cuda { #(#items)* }
    }.to_string()).unwrap();
    let dark_types_path = root.join("cuda_types/src/dark_api.rs");
    println!("cargo:rerun-if-changed={}", dark_types_path.display());
    println!("cargo:rerun-if-changed={}", root.join("dark_api/src/lib.rs").display());
    let dark_types_source = fs::read_to_string(dark_types_path).unwrap();
    let dark_types: Vec<_> = syn::parse_file(&dark_types_source).unwrap().items.into_iter()
        .filter(|item| matches!(item, syn::Item::Struct(item)
            if item.ident == "FatbincWrapper" || item.ident == "FatbinHeader")).collect();
    assert_eq!(dark_types.len(), 2);
    fs::write(out.join("dark_types.rs"), quote! {
        #[allow(warnings)] pub mod dark_api {
            use std::ffi::{c_uint, c_ulonglong, c_ushort, c_void};
            #(#dark_types)*
        }
    }.to_string()).unwrap();
    phase(2);
    // This verifier reads only the cleared canonical Rust dark declarations.
    println!("cargo:rerun-if-changed={}",here.join("tools/import_native_dark.py").display());
    assert!(Command::new("python3").arg(here.join("tools/import_native_dark.py"))
        .arg("--check").status().unwrap().success());
    let dark_schema = generate_dark::generate(&here, &out);
    let callback_path = here.join("src/callback_protocol.rs");
    println!("cargo:rerun-if-changed={}", callback_path.display());
    let callback_schema = fs::read_to_string(callback_path).unwrap();
    let mut occupancy_slots = Vec::new();
    let mut occupancy_functions = Vec::new();
    for i in 0..128usize {
        let name = format_ident!("occupancy_slot_{i}");
        occupancy_functions.push(name.clone());
        occupancy_slots.push(quote! {
            unsafe extern "C" fn #name(block: i32) -> usize {
                let token = OCCUPANCY[#i].load(Ordering::Acquire);
                if token == 0 { return 0; }
                unix_callbacks::emit(token, protocol::OCCUPANCY, [unixlib::pack(block),0,0,0,0,0]) as usize
            }
        });
    }
    fs::write(out.join("occupancy_slots.rs"), quote! {
        #(#occupancy_slots)*
        const OCCUPANCY_FUNCTIONS: [OccupancyCallback; POOL_SIZE] = [#(#occupancy_functions),*];
    }.to_string()).unwrap();
    phase(3);
    let wow64_schema = generate_wow64::generate(&here, &out, &root);
    // Reuse the canonical pure integrity algorithm with PE-supplied Windows
    // identities/addresses. Exclude its macros, serialization and driver code.
    phase(4);
    let integrity_names = ["integrity_check", "pass7", "pass6", "zero_result", "pass5",
        "pass2", "pass3", "pass4", "hash_pass", "integrity_check_single_pass"];
    let algorithm: Vec<_> = syn::parse_file(&fs::read_to_string(root.join("dark_api/src/lib.rs")).unwrap())
        .unwrap().items.into_iter().filter(|item| match item {
            syn::Item::Fn(item) => integrity_names.iter().any(|name| item.sig.ident == *name),
            syn::Item::Struct(item) => item.ident == "Pass3Input" || item.ident == "DeviceHashinfo",
            _ => false,
        }).collect();
    assert_eq!(algorithm.len(), integrity_names.len() + 2);
    fs::write(out.join("integrity_algorithm.rs"), quote! {
        use std::ffi::c_void;
        use crate::cuda::CUuuid;
        #(#algorithm)*
    }.to_string()).unwrap();

    phase(5);
    let mut signatures = BTreeMap::new();
    declarations(&fs::read_to_string(declarations_path).unwrap(), &mut signatures);
    let canonical = signatures.clone();
    let legacy = fs::read_to_string(here.join("api/legacy.rs")).unwrap();
    declarations(&legacy, &mut signatures);
    let symbol_names=signatures.keys().map(String::as_str).chain(["hipHostRegister","hipHostUnregister"])
        .collect::<BTreeSet<_>>();
    let symbols=symbol_names.iter().map(|name| {
        let bytes=syn::LitByteStr::new(format!("{name}\0").as_bytes(),format_ident!("_native_symbol").span());
        quote! { (#bytes,OnceLock::new()), }
    }).collect::<Vec<_>>();
    let symbol_count=symbols.len();
    fs::write(out.join("native_symbols.rs"),quote! {
        // Sorted canonical/compatibility names. Static storage, no map/lock or
        // allocation per symbol lookup; the backend is retained for life.
        static NATIVE_SYMBOLS:[(&[u8],OnceLock<Result<usize,CUerror>>);#symbol_count]=[#(#symbols)*];
    }.to_string()).unwrap();
    let surface = fs::read_to_string(here.join("api/exports.tsv")).unwrap();
    let entries: Vec<Vec<_>> = surface.lines().filter(|line| !line.starts_with('#'))
        .map(|line| line.split('\t').collect()).collect();
    assert_eq!(entries.len(), 788);
    assert_eq!(entries.iter().filter(|row| row[1] == "stdcall").count(), 690);
    let policy_source = fs::read_to_string(here.join("api/wrappers.tsv")).unwrap();
    let mut policy = BTreeMap::new();
    for line in policy_source.lines().filter(|line| !line.starts_with('#')) {
        let row: Vec<_> = line.split('\t').collect();
        assert_eq!(row.len(), 3);
        assert!(entries.iter().any(|entry| entry[0] == row[0] && entry[1] == "stdcall"),
                "wrapper policy names an absent export: {}", row[0]);
        assert!(policy.insert(row[0], row[1]).is_none());
    }
    let status_names = policy.iter().filter(|(_,mode)| **mode=="backend_status").map(|(name,_)| *name).collect();
    backend_status_guard(&root,&status_names,&canonical,&policy);
    let proc_source = fs::read_to_string(here.join("api/proc-address.tsv")).unwrap();
    let proc_rows: Vec<Vec<_>> = proc_source.lines().filter(|line| !line.starts_with('#'))
        .map(|line| line.split('\t').collect()).collect();
    assert_eq!(proc_rows.len(), 703);
    assert_eq!(proc_rows.iter().map(|r| (&r[0],&r[1],&r[2])).collect::<BTreeSet<_>>().len(),703);
    let private = ["cuLaunchCooperativeKernel_ptsz", "cuStreamWriteValue64_v2_ptsz"];
    let mut names: Vec<_> = entries.iter().filter(|row| row[1] == "stdcall")
        .map(|row| row[0]).chain(private).collect();
    names.sort_unstable();
    assert_eq!(names.len(), 692);
    phase(6);
    let mut ids: BTreeSet<u64> = (0..256).collect();
    let mut groups: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    let mut native_signatures = Vec::new();
    let mut status_signatures = Vec::new();
    let mut status_fixture_wrappers = Vec::new();
    let mut status_fixture_pointers = Vec::new();
    let mut pointers = Vec::new();
    let mut schema = types_source;
    schema.push_str(&dark_types_source);
    schema.push_str(&dark_schema);
    schema.push_str(&wow64_schema);
    schema.push_str(&legacy);
    schema.push_str(&surface);
    schema.push_str(&policy_source);
    let mut layout_checks = Vec::new();
    let mut coverage = String::from("# name\tx64-adapter\n");
    for name in names {
        assert!(ids.insert(hash(name.as_bytes())), "duplicate function or ID collision");
        let signature = signatures.get(name).unwrap_or_else(|| panic!("{name}: no CUDA declaration"));
        assert!(signature.inputs.len() <= 16);
        let mut layouts = Vec::new();
        for argument in &signature.inputs {
            let syn::FnArg::Typed(argument) = argument else { panic!("unexpected receiver") };
            let ty = &argument.ty;
            layouts.push(quote! { std::mem::size_of::<#ty>() });
        }
        let mode = policy.get(name).copied().unwrap_or("forwarded");
        assert!(["forwarded", "manual", "adapted", "pending", "backend_status"].contains(&mode));
        let abi = if private.contains(&name) { "private" } else { "system" };
        groups.entry(mode).or_default().push(quote! { #abi #signature; });
        if mode=="backend_status" {
            status_signatures.push(quote! { #abi #signature; });
        } else if mode == "forwarded" || mode == "adapted" {
            native_signatures.push(quote! { #abi #signature; });
        }
        // All current by-value inputs are <=64 bytes, with at most one >8-byte
        // aggregate per call. A CUDA update must explicitly revise the wire ABI.
        layout_checks.push(quote! {
            const _: () = {
                let sizes: &[usize] = &[#(#layouts),*];
                let mut i = 0;
                let mut aggregates = 0;
                while i < sizes.len() {
                    assert!(sizes[i] <= 64);
                    if sizes[i] > 8 { aggregates += 1; }
                    i += 1;
                }
                assert!(aggregates <= 1);
            };
        });
        let id = hash(name.as_bytes());
        let ident = &signature.ident;
        if mode=="backend_status" || name=="cuGraphRetainUserObject" {
            let inputs=&signature.inputs;
            let output=&signature.output;
            let args=signature.inputs.iter().map(|arg| match arg {
                syn::FnArg::Typed(arg)=>&arg.pat,_=>unreachable!()
            }).collect::<Vec<_>>();
            status_fixture_wrappers.push(quote! {
                unsafe extern "win64" fn #ident(#inputs) #output { crate::pe_thunks::#ident(#(#args),*) }
            });
            status_fixture_pointers.push(quote! { #id=>#ident as *const (), });
        }
        let target = if mode == "manual" || mode == "adapted" { quote!(crate::pe_wrappers::#ident) }
                     else { quote!(#ident) };
        pointers.push(quote! { #id => Some(#target as *const () as *mut c_void), });
        coverage.push_str(&format!("{name}\t{mode}\n"));
        schema.push_str(&signature.to_token_stream().to_string());
    }
    phase(7);
    let blocks: Vec<_> = groups.into_iter().map(|(mode, signatures)| {
        let mode = format_ident!("{}", mode);
        quote! { #mode! { #(#signatures)* } }
    }).collect();
    fs::write(out.join("pe_declarations.rs"), quote! {
        #[allow(unused_imports)] use crate::{cuda::*, legacy::*};
        #(#layout_checks)*
        #(#blocks)*
    }.to_string()).unwrap();
    fs::write(out.join("unix_declarations.rs"), quote! {
        #[allow(unused_imports)] use crate::{cuda::*, legacy::*};
        #(#layout_checks)*
        forwarded! { #(#native_signatures)* }
        backend_status! { #(#status_signatures)* }
    }.to_string()).unwrap();
    fs::write(out.join("pe_pointers.rs"), quote! {
        pub fn pointer(id: u64) -> Option<*mut c_void> {
            match id { #(#pointers)* _ => None }
        }
    }.to_string()).unwrap();
    fs::write(out.join("status_fixture_wrappers.rs"),quote! {
        #[allow(unused_imports)]
        use crate::{cuda::*,legacy::*};
        use std::ffi::c_void;
        #(#status_fixture_wrappers)*
        fn pointer(id:u64)->*const () { match id {#(#status_fixture_pointers)* _=>panic!("absent status fixture API") } }
    }.to_string()).unwrap();
    let stubs: Vec<_> = entries.iter().filter(|row| row[1] == "stub").map(|row| {
        let name = row[0];
        let ident = format_ident!("wine_stub_{}", name);
        quote! {
            #[cfg_attr(windows, export_name = #name)]
            pub unsafe extern "system" fn #ident() -> ! {
                crate::pe_wrappers::unimplemented(concat!(#name, "\0").as_bytes())
            }
        }
    }).collect();
    fs::write(out.join("pe_stubs.rs"), quote! { #(#stubs)* }.to_string()).unwrap();
    let mappings: Vec<_> = proc_rows.iter().map(|row| {
        let name = row[0].as_bytes();
        let name = syn::LitByteStr::new(name, format_ident!("_cuda_string").span());
        let version: i32 = row[1].parse().unwrap();
        let flags: u64 = row[2].parse().unwrap();
        let id = hash(row[3].as_bytes());
        quote! { (#name, #version, #flags, #id) }
    }).collect();
    fs::write(out.join("proc_address.rs"), quote! {
        pub const MAPPINGS: &[(&[u8], i32, u64, u64)] = &[#(#mappings),*];
    }.to_string()).unwrap();
    fs::write(out.join("coverage.tsv"), coverage).unwrap();
    schema.push_str(&callback_schema);
    let luid_path = here.join("src/luid_protocol.rs");
    println!("cargo:rerun-if-changed={}",luid_path.display());
    schema.push_str(&fs::read_to_string(luid_path).unwrap());
    let interop_path=here.join("src/interop_protocol.rs");
    println!("cargo:rerun-if-changed={}",interop_path.display());
    schema.push_str(&fs::read_to_string(interop_path).unwrap());
    let context_path=here.join("src/context_protocol.rs");
    println!("cargo:rerun-if-changed={}",context_path.display());
    schema.push_str(&fs::read_to_string(context_path).unwrap());
    let api_hash = hash(schema.as_bytes());
    fs::write(out.join("schema.rs"), format!("pub const API_HASH: u64 = {api_hash}u64;\n")).unwrap();

    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
        // Distinguish the transport library from ZLUDA's libnvcuda backend.
        println!("cargo:rustc-link-arg=-Wl,-soname,nvcuda.so");
    }

    if !check_only && env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let triplet = if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "x86" { "i686" } else { "x86_64" };
        let cc = env::var("ZLUDA_WINE_PE_CC").unwrap_or_else(|_| format!("{triplet}-w64-mingw32-gcc"));
        let ar = env::var("ZLUDA_WINE_PE_AR").unwrap_or_else(|_| format!("{triplet}-w64-mingw32-ar"));
        for key in ["ZLUDA_WINE_PE_CC", "ZLUDA_WINE_PE_AR"] {
            println!("cargo:rerun-if-env-changed={key}");
        }
        let object = out.join("wine_adapter.o");
        assert!(Command::new(cc).args(["-O2", "-Wall", "-Wextra", "-Werror", "-c"])
            .arg(here.join("wine_adapter.c"))
            .arg("-o").arg(&object).status().unwrap().success());
        assert!(Command::new(ar).arg("crs").arg(out.join("libwine_adapter.a"))
            .arg(object).status().unwrap().success());
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=wine_adapter");
        println!("cargo:rustc-link-lib=kernel32");
        if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "x86_64" {
            println!("cargo:rustc-link-lib=user32");
        }
    }
}

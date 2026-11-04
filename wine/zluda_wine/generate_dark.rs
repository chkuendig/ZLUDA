// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generate separate Windows and SysV entries from canonical Rust dark contracts.
use quote::{format_ident, quote};
use std::{collections::BTreeMap, fs, path::Path};

pub fn generate(here: &Path, out: &Path) -> String {
    let source = fs::read_to_string(here.join("api/native-dark.tsv")).unwrap();
    println!("cargo:rerun-if-changed={}", here.join("api/native-dark.tsv").display());
    println!("cargo:rerun-if-changed={}", here.join("generate_dark.rs").display());
    let mut indices = BTreeMap::new();
    let mut meta = Vec::new();
    let mut native = Vec::new();
    let mut pe = Vec::new();
    let mut pointers = Vec::new();
    for row in source.lines().filter(|line| !line.starts_with('#')) {
        let row: Vec<_> = row.split('\t').collect();
        assert_eq!(row.len(), 7);
        let uuid: Vec<_> = row[0].as_bytes().chunks_exact(2)
            .map(|bytes| u8::from_str_radix(std::str::from_utf8(bytes).unwrap(), 16).unwrap()).collect();
        assert_eq!(uuid.len(), 16);
        let name = row[1];
        let words: usize = row[2].parse().unwrap();
        let next = indices.len();
        let index = *indices.entry(name.to_owned()).or_insert(next);
        if index == meta.len() {
            let header = name != "CONTEXT_LOCAL_STORAGE_INTERFACE_V0301" && name != "ZLUDA32_INTERNAL";
            let windows_words = if ["TOOLS_TLS", "CONTEXT_LOCAL_STORAGE_INTERFACE_V0301"].contains(&name) { 3 } else { words };
            let callable_mask = source.lines().filter(|line| !line.starts_with('#')).map(|line| line.split('\t').collect::<Vec<_>>())
                .filter(|row| row[1] == name && row[5] != "-")
                .fold(0u64, |mask, row| mask | (1u64 << row[3].parse::<u32>().unwrap()));
            meta.push(quote! { TableSpec { uuid: [#(#uuid),*], words: #words, header: #header,
                name: #name, windows_words: #windows_words, callable_mask: #callable_mask } });
        }
        if row[5] == "-" { continue; }
        let slot: usize = row[3].parse().unwrap();
        let token = index as u64 + 1;
        let signature: syn::Signature = syn::parse_str(row[5]).unwrap();
        // These APIs need PE-owned buffers, callback/lifetime records or a
        // Windows loader. Never let generic code expose their native outputs.
        if ["CONTEXT_LOCAL_STORAGE_INTERFACE_V0301", "HEAP_ACCESS", "INTEGRITY_CHECK",
            "HYBRID_CUDART", "ZLUDA32_INTERNAL"].contains(&name) { continue; }
        let mut types = Vec::new();
        let mut args = Vec::new();
        for arg in &signature.inputs {
            let syn::FnArg::Typed(arg) = arg else { panic!("dark receiver") };
            let syn::Pat::Ident(pat) = arg.pat.as_ref() else { panic!("dark argument pattern") };
            types.push(&arg.ty);
            args.push(&pat.ident);
        }
        assert!(args.len() + 2 <= 15);
        let syn::ReturnType::Type(_, ret) = &signature.output else { panic!("dark return") };
        let return_kind = if matches!(ret.as_ref(), syn::Type::Tuple(tuple) if tuple.elems.is_empty()) { 0 }
                          else if matches!(ret.as_ref(), syn::Type::Path(path) if path.path.is_ident("u32")) { 1 }
                          else { assert!(row[5].ends_with("::CUresult")); 2 };
        let ret_native = match return_kind {
            0 => quote! { native(#(#args),*); call.result = 0; },
            1 => quote! { call.args[15] = u64::from(native(#(#args),*)); call.result = 0; },
            _ => quote! { call.result = crate::unixlib::code(native(#(#args),*)); },
        };
        let fields: Vec<_> = (2..args.len() + 2).collect();
        native.push(quote! {
            (#token, #slot) => {
                let address = crate::dark_unix::native_slot(#token, #slot)?;
                let native: unsafe extern "C" fn(#(#types),*) -> #ret = std::mem::transmute(address);
                #(let #args: #types = call.get(#fields)?;)*
                #ret_native
                Ok(())
            }
        });
        let mut pe_signature = signature.clone();
        pe_signature.ident = format_ident!("dark_{}_{}", name, slot);
        pe_signature.abi = Some(syn::parse_quote!(extern "win64"));
        pe_signature.unsafety = Some(syn::parse_quote!(unsafe));
        let ident = &pe_signature.ident;
        let ret_pe = match return_kind {
            0 => quote! { if crate::dark_protocol::submit(&mut call).is_err() {
                crate::dark_protocol::void_failure();
            } },
            1 => quote! {
                if crate::dark_protocol::submit(&mut call).is_err() { crate::dark_protocol::void_failure(); }
                call.args[15] as u32
            },
            _ => quote! { crate::dark_protocol::submit(&mut call) },
        };
        pe.push(quote! {
            pub #pe_signature {
                let mut call = crate::unixlib::Call::new(crate::dark_protocol::CALL);
                call.args[0] = #token;
                call.args[1] = #slot as u64;
                #(call.put(#fields, #args);)*
                #ret_pe
            }
        });
        pointers.push(quote! { (#token, #slot) => Some(#ident as *const () as *mut std::ffi::c_void), });
    }
    assert_eq!(meta.len(), 11);
    println!("cargo:rerun-if-changed={}", here.join("api/legacy-uuid.txt").display());
    let c_source = fs::read_to_string(here.join("api/legacy-uuid.txt")).unwrap();
    let old: Vec<_> = c_source.lines().map(|id| {
        let uuid: Vec<_> = id.as_bytes().chunks_exact(2)
            .map(|bytes| u8::from_str_radix(std::str::from_utf8(bytes).unwrap(), 16).unwrap()).collect();
        assert_eq!(uuid.len(),16);
        quote! { LegacySpec { uuid: [#(#uuid),*] } }
    }).collect();
    assert_eq!(old.len(),4);
    fs::write(out.join("dark_meta.rs"), quote! {
        pub const TABLES: &[TableSpec] = &[#(#meta),*];
        pub const LEGACY: &[LegacySpec] = &[#(#old),*];
    }.to_string()).unwrap();
    fs::write(out.join("dark_unix_thunks.rs"), quote! {
        pub unsafe fn typed_call(call: &mut crate::unixlib::Call) -> Result<(), crate::cuda::CUerror> {
            if call.args[0] == 0 { return Err(crate::cuda::CUerror::INVALID_VALUE); }
            match (call.args[0], call.args[1] as usize) {
                #(#native)*
                _ => Err(crate::cuda::CUerror::NOT_SUPPORTED)
            }
        }
    }.to_string()).unwrap();
    fs::write(out.join("dark_pe_thunks.rs"), quote! {
        #(#pe)*
        pub fn pointer(token: u64, slot: usize) -> Option<*mut std::ffi::c_void> {
            match (token, slot) { #(#pointers)* _ => None }
        }
    }.to_string()).unwrap();
    format!("{source}{c_source}")
}

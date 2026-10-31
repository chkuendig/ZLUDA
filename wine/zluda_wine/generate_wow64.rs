// SPDX-License-Identifier: MIT OR Apache-2.0
//! Canonical PE32 parameter types with target-evaluated ABI assertions (§17).
use quote::{format_ident,quote};
use std::{collections::BTreeMap,fs,path::Path};
pub fn generate(here:&Path,out:&Path,root:&Path)->String {
    for name in ["wow64-opcodes.tsv","wow64-required.txt","pe32-implemented.txt","exports.tsv","legacy.rs"] {
        println!("cargo:rerun-if-changed={}",here.join("api").join(name).display());
    }
    println!("cargo:rerun-if-changed={}",here.join("generate_wow64.rs").display());
    let operations=fs::read_to_string(here.join("api/wow64-opcodes.tsv")).unwrap();
    let required=fs::read_to_string(here.join("api/wow64-required.txt")).unwrap();
    let ops:Vec<_>=operations.lines().filter(|s|!s.starts_with('#')).map(|s|{let (n,k)=s.split_once('\t').unwrap();let n:u32=n.parse().unwrap();let k=format_ident!("{k}");quote!(pub const #k:u32=#n;)}).collect();
    assert_eq!(ops.len(),60);
    let symbols:Vec<_>=required.lines().filter(|s|!s.starts_with('#')).map(|n|syn::LitByteStr::new(format!("{n}\0").as_bytes(),format_ident!("_symbol").span())).collect();
    assert_eq!(symbols.len(),58);
    fs::write(out.join("wow64_contract.rs"),quote!(#(#ops)* pub const REQUIRED:&[&[u8]]=&[#(#symbols),*];).to_string()).unwrap();
    let mut signatures=BTreeMap::new();
    super::declarations(&fs::read_to_string(root.join("cuda_macros/src/cuda.rs")).unwrap(),&mut signatures);
    super::declarations(&fs::read_to_string(here.join("api/legacy.rs")).unwrap(),&mut signatures);
    let implemented=fs::read_to_string(here.join("api/pe32-implemented.txt")).unwrap();
    let surface=fs::read_to_string(here.join("api/exports.tsv")).unwrap();
    let mut stubs=Vec::new();let mut real=Vec::new();let mut checks=Vec::new();
    for row in surface.lines().filter(|s|!s.starts_with('#')).map(|s|s.split('\t').collect::<Vec<_>>()).filter(|r|r[1]=="stdcall") {
        let name=format_ident!("{}",row[0]);let sig=&signatures[row[0]];let stack:usize=row[3].parse().unwrap();
        let expected:Vec<usize>=row[5].split_whitespace().map(|s|s.parse().unwrap()).collect();
        assert_eq!(expected.len(),sig.inputs.len(),"PE32 parameter count: {}",row[0]);
        assert_eq!(expected.iter().sum::<usize>(),stack,"PE32 reviewed width sum: {}",row[0]);
        let mut args=Vec::new();let mut sizes=Vec::new();
        for (i,arg) in sig.inputs.iter().enumerate() {
            let syn::FnArg::Typed(arg)=arg else{panic!("receiver")};let ty=&arg.ty;let ident=format_ident!("_arg{i}");
            args.push(quote!(#ident:#ty));sizes.push(quote!((std::mem::size_of::<#ty>()+3)&!3usize));
        }
        checks.push(quote! {
            #[cfg(all(windows,target_arch="x86"))]
            const _:()={
                let actual:&[usize]=&[#(#sizes),*];let expected:&[usize]=&[#(#expected),*];
                let mut i=0;let mut sum=0;while i<actual.len(){assert!(actual[i]==expected[i]);sum+=actual[i];i+=1;}assert!(sum==#stack);
            };
        });
        if implemented.lines().any(|s|s==row[0]) {real.push(quote!(const _:*const ()=super::wow64_pe::#name as *const ();));}
        else {stubs.push(quote!(#[cfg_attr(all(windows,target_arch="x86"),no_mangle)] pub unsafe extern "system" fn #name(#(#args),*)->i32 {801}));}
    }
    assert_eq!(real.len(),88);assert_eq!(stubs.len(),602);
    fs::write(out.join("wow64_stubs.rs"),quote!(#[allow(unused_imports)] use crate::{cuda::*,legacy::*}; #(#checks)* #(#stubs)* #(#real)*).to_string()).unwrap();
    // Include the packet/version and new-operation argument contract in the
    // shared PE/Unix schema, in addition to operation numbers and exports.
    let protocol_path=here.join("src/wow64_protocol.rs");
    println!("cargo:rerun-if-changed={}",protocol_path.display());
    let protocol=fs::read_to_string(protocol_path).unwrap();
    format!("{operations}{required}{implemented}{surface}{protocol}")
}

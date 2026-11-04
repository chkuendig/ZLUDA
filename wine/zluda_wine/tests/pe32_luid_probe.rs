// SPDX-License-Identifier: MIT OR Apache-2.0
//! Actual PE32 generated LUID stub, attribute caller, config hints and call
//! trace; fake transport only.
#![allow(non_snake_case,dead_code,unused_imports,improper_ctypes_definitions)]
extern crate self as cuda_types;
pub enum FILE {}
include!(concat!(env!("OUT_DIR"), "/cuda_types.rs"));
#[path="../src/legacy.rs"] mod legacy;
#[path="../src/wow64_protocol.rs"] mod wow64_protocol;
#[path="../src/wow64_trace.rs"] mod wow64_trace;
#[path="../src/pe_sync.rs"] mod pe_sync;
#[path="../src/wow64_pe.rs"] mod wow64_pe;
mod wow64_stubs {include!(concat!(env!("OUT_DIR"),"/wow64_stubs.rs"));}
mod wow64_local {
    pub unsafe fn storage_destroy(_: *mut std::ffi::c_void) {}
    pub fn cuGetExportTable() {}
}
use std::{ffi::c_void,sync::atomic::{AtomicU32,Ordering}};
use wow64_protocol::*;
static SERVER_VERSION:AtomicU32=AtomicU32::new(4);
static CALLS:AtomicU32=AtomicU32::new(0);
static CONTEXT_CALLS:AtomicU32=AtomicU32::new(0);
#[no_mangle] pub extern "C" fn zluda_wine_attach(_: *mut c_void)->i32 {1}
#[no_mangle] pub unsafe extern "C" fn zluda_wine_call(p:*mut c_void)->i32 {
    let c=&mut *p.cast::<Call32>();
    if !c.valid() || c.version!=SERVER_VERSION.load(Ordering::Relaxed){return 0xc000000du32 as i32;}
    if c.opcode==HELLO{c.result=0;return 0;}
    if c.opcode==C32_CTX_CURRENT{CONTEXT_CALLS.fetch_add(1,Ordering::Relaxed);ptr::<u32>(c.args[0]).write(0x40000001);c.result=0;return 0;}
    assert_eq!(c.opcode,C32_DEVICE_ATTR);
    assert!(matches!(c.args[1],50|33|34));
    CALLS.fetch_add(1,Ordering::Relaxed);
    if (c.args[2] as i32)<0{c.result=101;}
    else if c.args[0]==0{c.result=1;}
    else {ptr::<i32>(c.args[0]).write(c.args[1] as i32+100*c.args[2] as i32);c.result=0;}
    0
}
std::arch::global_asm!(include_str!("pe32_luid_stack.S"),options(att_syntax));
extern "C" {fn w1_raw_stdcall(f:*const (),words:*const u32,count:u32,delta:*mut i32,pad:*mut u32)->i32;}
unsafe fn raw(function:*const (),words:&[u32])->i32 {
    let mut delta=i32::MAX;let mut pad=[0u32;16];
    let status=w1_raw_stdcall(function,words.as_ptr(),words.len() as u32,&mut delta,pad.as_mut_ptr());
    assert_eq!(delta,0,"PE32 callee must clean exactly {} argument bytes",4*words.len());
    for (i,value) in pad.into_iter().enumerate(){assert_eq!(value,0x5a5a0000^((i+1) as u32));}
    status
}
unsafe fn invoke(out:*mut i8,node:*mut u32,device:i32)->i32 {
    let args=[out as usize as u32,node as usize as u32,device as u32];
    let mut delta=i32::MAX;let mut pad=[0u32;16];
    let status=w1_raw_stdcall(wow64_stubs::cuDeviceGetLuid as *const (),args.as_ptr(),3,&mut delta,pad.as_mut_ptr());
    assert_eq!(delta,0,"PE32 callee must clean exactly 12 argument bytes");
    for (i,value) in pad.into_iter().enumerate(){assert_eq!(value,0x5a5a0000^((i+1) as u32));}
    status
}
fn main(){unsafe{
    assert_eq!(std::mem::size_of::<usize>(),4);
    assert_eq!(VERSION,4);assert_eq!(C32_RESERVED_59,59);
    std::env::set_var("PROTON_ENABLE_ZLUDA","1");
    // The trace opens on the first traced export; this is a Windows path, so
    // the frontend uses CreateFileA rather than Wine's Unix-path conversion.
    let trace=std::env::temp_dir().join(format!("zluda-pe32-calls-{}",std::process::id()));
    std::env::set_var("ZLUDA_WINE_CALL_LOG",&trace);
    assert_eq!(wow64_pe::DllMain(std::ptr::null_mut(),1,std::ptr::null_mut()),1);
    let mut out=[0xa5u8;24];let mut nodes=[0x11223344u32,0xdeadbeef,0xaabbccdd];
    for device in [0,1,-1] {
        assert_eq!(invoke(out.as_mut_ptr().add(8).cast(),nodes.as_mut_ptr().add(1),device),801);
        assert_eq!(out,[0xa5;24]);assert_eq!(nodes,[0x11223344,0xdeadbeef,0xaabbccdd]);
    }
    assert_eq!(invoke(std::ptr::null_mut(),nodes.as_mut_ptr(),0),801);
    assert_eq!(invoke(out.as_mut_ptr().cast(),std::ptr::null_mut(),0),801);
    assert_eq!(CALLS.load(Ordering::Relaxed),0,"LUID stub must not submit a request");
    for attr in [50,33,34] {for device in [0,1] {
        let mut value=-1;
        assert_eq!(wow64_pe::cuDeviceGetAttribute(&mut value,attr,device),0);
        assert_eq!(value,attr+100*device);
    }}
    let mut value=0x12345678;
    SERVER_VERSION.store(3,Ordering::Relaxed);
    assert_eq!(wow64_pe::cuDeviceGetAttribute(&mut value,50,0),3);
    assert_eq!(value,0x12345678);
    SERVER_VERSION.store(4,Ordering::Relaxed);
    for opcode in [C32_DEVICE_ATTR,C32_RESERVED_59] {
        let mut old=Call32::new(opcode,&[0xdeadbeef,50,0]);old.version=3;
        assert!(!old.valid(),"v4 receiver must reject a v3 packet");
        assert_eq!(zluda_wine_call((&mut old as *mut Call32).cast()),0xc000000du32 as i32);
        assert_eq!(old.result,801);
    }
    // Cache/shared-memory hints: exact stdcall cleanup, CUDA ranges, no backend op.
    assert_eq!(raw(wow64_pe::cuCtxSetCacheConfig as *const (),&[1]),0);
    assert_eq!(raw(wow64_pe::cuCtxSetCacheConfig as *const (),&[4]),1);
    assert_eq!(raw(wow64_pe::cuCtxSetSharedMemConfig as *const (),&[2]),0);
    let mut config=-1i32;
    assert_eq!(raw(wow64_pe::cuCtxGetCacheConfig as *const (),&[&mut config as *mut i32 as usize as u32]),0);
    assert_eq!(config,0);config=-1;
    assert_eq!(raw(wow64_pe::cuCtxGetSharedMemConfig as *const (),&[&mut config as *mut i32 as usize as u32]),0);
    assert_eq!(config,0);
    assert_eq!(raw(wow64_pe::cuFuncSetCacheConfig as *const (),&[0x40000010,2]),0);
    assert_eq!(raw(wow64_pe::cuFuncSetCacheConfig as *const (),&[0,2]),400);
    assert_eq!(raw(wow64_pe::cuFuncSetSharedMemConfig as *const (),&[0x40000010,3]),1);
    assert_eq!(CONTEXT_CALLS.load(Ordering::Relaxed),4,"only the four context hints query the current context");
    let path=std::path::PathBuf::from(format!("{}.{}",trace.display(),std::process::id()));
    let log=std::fs::read_to_string(&path).expect("PE32 call trace file");
    let _=std::fs::remove_file(&path);
    let rows:Vec<(String,i32,String)>=log.lines().map(|line|{
        let fields:Vec<&str>=line.split(' ').collect();
        assert!(fields.len()==5||fields.len()==6,"trace line shape: {line}");
        fields[0].parse::<u64>().unwrap();u32::from_str_radix(fields[1],16).unwrap();
        assert!(fields[4].starts_with("dt=")&&fields[4][3..].parse::<u64>().is_ok(),"{line}");
        (fields[2].to_string(),fields[3].strip_prefix("r=").unwrap().parse().unwrap(),fields.get(5).unwrap_or(&"").to_string())
    }).collect();
    let count=|name:&str,status:i32|rows.iter().filter(|r|r.0==name&&r.1==status).count();
    assert_eq!(count("cuDeviceGetLuid",801),5);
    assert_eq!(count("cuDeviceGetAttribute",0),6);
    assert_eq!(count("cuDeviceGetAttribute",3),1);
    assert!(rows.iter().any(|r|r.0=="cuDeviceGetAttribute"&&r.1==0&&r.2.ends_with(",22,1")),"attribute argument words");
    assert_eq!(count("cuCtxSetCacheConfig",0),1);assert_eq!(count("cuCtxSetCacheConfig",1),1);
    assert_eq!(count("cuFuncSetCacheConfig",400),1);assert_eq!(count("cuFuncSetSharedMemConfig",1),1);
    assert_eq!(rows.len(),5+7+8,"one line per traced export call");
    println!("PE32 SCOPE PASS: LUID=801 argument_bytes=12 stack_delta=0 no-LUID-transport; PCI50/33/34 forwarding and v3/v4 refusal passed; config hints and call trace passed");
}}

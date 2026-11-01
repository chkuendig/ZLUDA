// SPDX-License-Identifier: MIT OR Apache-2.0
//! Actual PE32 generated LUID stub and attribute caller; fake transport only.
#![allow(non_snake_case,dead_code,unused_imports,improper_ctypes_definitions)]
extern crate self as cuda_types;
pub enum FILE {}
include!(concat!(env!("OUT_DIR"), "/cuda_types.rs"));
#[path="../src/legacy.rs"] mod legacy;
#[path="../src/wow64_protocol.rs"] mod wow64_protocol;
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
#[no_mangle] pub extern "C" fn zluda_wine_attach(_: *mut c_void)->i32 {1}
#[no_mangle] pub unsafe extern "C" fn zluda_wine_call(p:*mut c_void)->i32 {
    let c=&mut *p.cast::<Call32>();
    if !c.valid() || c.version!=SERVER_VERSION.load(Ordering::Relaxed){return 0xc000000du32 as i32;}
    if c.opcode==HELLO{c.result=0;return 0;}
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
    let mut old=Call32::new(C32_DEVICE_ATTR,&[0xdeadbeef,50,0]);old.version=3;
    assert!(!old.valid(),"v4 receiver must reject a v3 packet");
    SERVER_VERSION.store(4,Ordering::Relaxed);
    assert_eq!(zluda_wine_call((&mut old as *mut Call32).cast()),0xc000000du32 as i32);
    assert_eq!(old.result,801);
    println!("PE32 SCOPE PASS: LUID=801 argument_bytes=12 stack_delta=0 no-LUID-transport; PCI50/33/34 forwarding and v3/v4 refusal passed");
}}

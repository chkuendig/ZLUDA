// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native 32-bit Windows caller of the submitted PE frontend; fake transport only.
#![allow(non_snake_case,dead_code,unused_imports,improper_ctypes_definitions)]
extern crate self as cuda_types;
pub enum FILE {}
include!(concat!(env!("OUT_DIR"), "/cuda_types.rs"));
#[path="../src/wow64_protocol.rs"] mod wow64_protocol;
#[path="../src/pe_platform.rs"] mod pe_platform;
#[path="../src/pe_sync.rs"] mod pe_sync;
#[path="../src/pe_luid.rs"] mod pe_luid;
#[path="../src/wow64_pe.rs"] mod wow64_pe;
mod wow64_local {
    pub unsafe fn storage_destroy(_: *mut std::ffi::c_void) {}
    pub fn cuGetExportTable() {}
}
use std::{ffi::c_void,sync::atomic::{AtomicBool,AtomicU32,Ordering}};
use wow64_protocol::*;
static MISMATCH:AtomicBool=AtomicBool::new(false);
static CALLS:AtomicU32=AtomicU32::new(0);
const UUID:[u8;16]=[0x80,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0xff];
#[no_mangle] pub extern "C" fn zluda_wine_attach(_: *mut c_void)->i32 {1}
#[no_mangle] pub unsafe extern "C" fn zluda_wine_call(p:*mut c_void)->i32 {
    let c=&mut *p.cast::<Call32>();
    if MISMATCH.load(Ordering::Relaxed){c.version=2;}
    if !c.valid(){return 0xc000000du32 as i32;}
    if c.opcode==HELLO{c.result=0;return 0;}
    assert_eq!(c.opcode,59);
    CALLS.fetch_add(1,Ordering::Relaxed);
    if c.args[1] as i32!=0{c.result=101;return 0;}
    std::ptr::copy_nonoverlapping(UUID.as_ptr(),ptr(c.args[0]),16);
    ptr::<u32>(c.args[2]).write(1);
    c.result=0;0
}
std::arch::global_asm!(include_str!("pe32_luid_stack.S"),options(att_syntax));
extern "C" {fn w1_raw_stdcall(f:*const (),words:*const u32,count:u32,delta:*mut i32,pad:*mut u32)->i32;}
unsafe fn invoke(out:*mut i8,node:*mut u32,device:i32)->i32 {
    let args=[out as usize as u32,node as usize as u32,device as u32];
    let mut delta=i32::MAX;let mut pad=[0u32;16];
    let status=w1_raw_stdcall(wow64_pe::cuDeviceGetLuid as *const (),args.as_ptr(),3,&mut delta,pad.as_mut_ptr());
    assert_eq!(delta,0,"PE32 callee must clean exactly 12 argument bytes");
    for (i,value) in pad.into_iter().enumerate(){assert_eq!(value,0x5a5a0000^((i+1) as u32));}
    status
}
fn main(){unsafe{
    assert_eq!(std::mem::size_of::<usize>(),4);
    assert_eq!(VERSION,3);assert_eq!(C32_DEVICE_UUID,59);
    std::env::set_var("PROTON_ENABLE_ZLUDA","1");
    assert_eq!(wow64_pe::DllMain(std::ptr::null_mut(),1,std::ptr::null_mut()),1);
    let mut out=[0xa5u8;24];let mut nodes=[0x11223344u32,0,0xaabbccdd];
    assert_eq!(invoke(out.as_mut_ptr().add(8).cast(),nodes.as_mut_ptr().add(1),0),0);
    let expected=UUID.iter().fold(0u64,|h,b|h.wrapping_mul(31).wrapping_add(*b as i8 as i64 as u64));
    assert_eq!(&out[8..16],&expected.to_le_bytes());assert_eq!(nodes[1],1);
    assert_eq!(&out[..8],&[0xa5;8]);assert_eq!(&out[16..],&[0xa5;8]);
    assert_eq!(nodes[0],0x11223344);assert_eq!(nodes[2],0xaabbccdd);
    let calls=CALLS.load(Ordering::Relaxed);
    assert_eq!(invoke(std::ptr::null_mut(),nodes.as_mut_ptr(),0),1);
    assert_eq!(invoke(out.as_mut_ptr().cast(),std::ptr::null_mut(),0),1);
    assert_eq!(CALLS.load(Ordering::Relaxed),calls);
    out.fill(0xa5);nodes[1]=0xdeadbeef;
    assert_eq!(invoke(out.as_mut_ptr().add(8).cast(),nodes.as_mut_ptr().add(1),-1),101);
    assert_eq!(out,[0xa5;24]);assert_eq!(nodes[1],0xdeadbeef);
    MISMATCH.store(true,Ordering::Relaxed);
    assert_eq!(invoke(out.as_mut_ptr().add(8).cast(),nodes.as_mut_ptr().add(1),0),3);
    assert_eq!(out,[0xa5;24]);assert_eq!(nodes[1],0xdeadbeef);
    println!("PE32 LUID PASS: opcode=59 version=3 argument_bytes=12 stack_delta=0; canaries/null/invalid-device/version-mismatch passed");
}}

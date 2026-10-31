// SPDX-License-Identifier: MIT OR Apache-2.0
//! Strong imported synchronous Wine callback; no worker or global symbol lookup.
use crate::{cuda::*,unixlib::{self,Call},context_protocol as protocol};
use std::{ffi::c_void,sync::atomic::{AtomicU64,Ordering}};
#[cfg(all(not(zluda_wine_native_sdk),not(zluda_wine_check_only)))]
compile_error!("shipping ELF bridge requires the matching prepared ntdll.so SDK");
extern "C" {fn KeUserModeCallback(id:u32,data:*const c_void,len:u32,out:*mut *mut c_void,out_len:*mut u32)->i32;}
static ENTRY:AtomicU64=AtomicU64::new(0);
pub unsafe fn initialize(call:&mut Call)->CUresult {
    if call.args[0]==0 || call.args[1..].iter().any(|v|*v!=0) || call.aggregate!=[0;64] {return Err(CUerror::INVALID_VALUE);}
    match ENTRY.compare_exchange(0,call.args[0],Ordering::AcqRel,Ordering::Acquire) {Ok(_)=>{},Err(v) if v==call.args[0]=>{},Err(_)=>return Err(CUerror::INVALID_VALUE)}
    call.result=0;Ok(())
}
unsafe extern "C" fn destructor(context:CUcontext,key:*mut c_void,value:*mut c_void) {
    let entry=ENTRY.load(Ordering::Acquire);
    if entry==0 || !crate::unix_wrappers::inside_call(){std::process::abort();}
    let event=protocol::Event{callback:entry,magic:protocol::MAGIC,api_hash:unixlib::API_HASH,token:value as usize as u64,context:context.0 as usize as u64,key:key as usize as u64};
    let mut out=std::ptr::null_mut();let mut len=0;
    if KeUserModeCallback(0,(&event as *const protocol::Event).cast(),48,&mut out,&mut len)!=0 {std::process::abort();}
}
pub unsafe fn dispatch(call:&mut Call)->Option<CUresult> {
    if crate::dark_protocol::TABLES.get(call.args[0].wrapping_sub(1) as usize)?.name!="CONTEXT_LOCAL_STORAGE_INTERFACE_V0301" {return None;}
    Some((||{match call.args[1] {
        0=> {if ENTRY.load(Ordering::Acquire)==0{return Err(CUerror::NOT_INITIALIZED);}if call.args[4]==0 || call.args[4]>i64::MAX as u64 || call.args[5]!=0{return Err(CUerror::INVALID_VALUE);}
            let f:unsafe extern "C" fn(CUcontext,*mut c_void,*mut c_void,Option<unsafe extern "C" fn(CUcontext,*mut c_void,*mut c_void)>)->CUresult=std::mem::transmute(crate::dark_unix::native_slot(call.args[0],0)?);
            call.result=unixlib::code(f(call.get(2)?,call.get(3)?,call.args[4] as usize as *mut c_void,Some(destructor)));},
        1=>{let f:unsafe extern "C" fn(CUcontext,*mut c_void)->CUresult=std::mem::transmute(crate::dark_unix::native_slot(call.args[0],1)?);call.result=unixlib::code(f(call.get(2)?,call.get(3)?));},
        2=>{let f:unsafe extern "C" fn(*mut *mut c_void,CUcontext,*mut c_void)->CUresult=std::mem::transmute(crate::dark_unix::native_slot(call.args[0],2)?);call.result=unixlib::code(f(call.get(2)?,call.get(3)?,call.get(4)?));},
        _=>return Err(CUerror::NOT_SUPPORTED),
    }Ok(())})())
}

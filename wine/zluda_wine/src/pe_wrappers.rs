// SPDX-License-Identifier: MIT OR Apache-2.0
//! Windows policy. Native calls always use the cleared packet transport.
use crate::{cuda::*,unixlib::{self,Call},pe_platform as os};
use std::{ffi::{c_char,c_void,CStr},sync::{atomic::{AtomicBool,Ordering}}};
use crate::pe_sync::Once as OnceLock;
pub use crate::pe_callback_api::*;
pub use crate::pe_external::*;
pub use crate::pe_interop::*;
pub use crate::pe_luid::cuDeviceGetLuid;
pub use crate::pe_dark_tables::cuGetExportTable;
static ATTACHED:AtomicBool=AtomicBool::new(false);
static HELLO:OnceLock<bool>=OnceLock::new();
#[no_mangle]
pub unsafe extern "system" fn DllMain(module:*mut c_void,reason:u32,_reserved:*mut c_void)->i32 {
    match reason {
        1=> {if std::env::var("PROTON_ENABLE_ZLUDA").as_deref()!=Ok("1") {return 0;}
            let ready=os::zluda_wine_attach(module)!=0;ATTACHED.store(ready,Ordering::Release);i32::from(ready)},
        3=> {crate::pe_dark_owned::thread_detach();crate::pe_callbacks::thread_detach();1},_=>1,
    }
}
pub unsafe fn direct(call:&mut Call)->i32 {os::zluda_wine_call((call as *mut Call).cast())}
pub unsafe fn submit(call:&mut Call)->CUresult {
    if !ATTACHED.load(Ordering::Acquire) {return Err(CUerror::NOT_INITIALIZED);}
    if !*HELLO.get_or_init(||{let mut c=Call::new(0);direct(&mut c)==0 && c.result==0}) {return Err(CUerror::NOT_INITIALIZED);}
    let status=if call.function>crate::context_protocol::INITIALIZE || matches!(call.function,2|3|4|10|11) {crate::pe_callbacks::route(call)}else {direct(call)};
    if status!=0 {return Err(CUerror::NOT_INITIALIZED);} unixlib::result(call.result)
}
pub unsafe fn raw(name:&str,args:&[u64])->CUresult {let mut c=Call::new(unixlib::function_id(name));c.args[..args.len()].copy_from_slice(args);submit(&mut c)}
pub unsafe fn available(name:&[u8])->CUresult {let mut c=Call::new(unixlib::HAS_SYMBOL);c.put(0,name.as_ptr());submit(&mut c)}
pub unsafe fn unimplemented(name:&[u8])->! {let p=[c"nvcuda.dll".as_ptr() as usize,name.as_ptr() as usize];os::RaiseException(0x80000100,1,2,p.as_ptr());std::process::abort()}
#[no_mangle]
pub unsafe extern "system" fn cuGetProcAddress(name:*const c_char,out:*mut *mut c_void,version:i32,flags:u64)->CUresult {lookup(name,out,version,flags,std::ptr::null_mut(),false)}
#[no_mangle]
pub unsafe extern "system" fn cuGetProcAddress_v2(name:*const c_char,out:*mut *mut c_void,version:i32,flags:u64,query:*mut CUdriverProcAddressQueryResult)->CUresult {lookup(name,out,version,flags,query,true)}
unsafe fn lookup(name:*const c_char,out:*mut *mut c_void,version:i32,flags:u64,query:*mut CUdriverProcAddressQueryResult,v2:bool)->CUresult {
    if name.is_null() || out.is_null() {return Err(CUerror::INVALID_VALUE);} *out=std::ptr::null_mut();
    if v2 {available(b"cuGetProcAddress_v2\0")?;}
    let name=CStr::from_ptr(name).to_bytes();
    let selected=crate::proc_address::MAPPINGS.iter().find(|r|r.0==name && r.1==version && r.2==flags);
    if let Some(row)=selected {*out=crate::pe_thunks::pointer(row.3).ok_or(CUerror::NOT_SUPPORTED)?;}
    if !query.is_null() {std::ptr::write(query.cast::<u32>(),u32::from(selected.is_none()));} Ok(())
}
#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetAttribute(out:*mut i32,attribute:CUdevice_attribute,device:CUdevice)->CUresult {
    let r=raw("cuDeviceGetAttribute",&[unixlib::pack(out),unixlib::pack(attribute),unixlib::pack(device)]);
    if r==Err(CUerror::INVALID_VALUE) && attribute.0==0x20080402 && device==0 && !out.is_null() {*out=0;Ok(())}else {r}
}
macro_rules! uuid {($n:ident)=>{#[no_mangle] pub unsafe extern "system" fn $n(out:*mut CUuuid,device:CUdevice)->CUresult {raw(stringify!($n),&[unixlib::pack(out),unixlib::pack(device)]).map_err(|_|CUerror::INVALID_VALUE)}};}
uuid!(cuDeviceGetUuid);uuid!(cuDeviceGetUuid_v2);
#[no_mangle]
pub unsafe extern "system" fn cuModuleLoad(out:*mut CUmodule,name:*const c_char)->CUresult {
    if name.is_null() {return Err(CUerror::INVALID_VALUE);}let mut wide=[0u16;260];
    if os::MultiByteToWideChar(0,0,name,-1,wide.as_mut_ptr(),260)==0 {return Err(CUerror::INVALID_VALUE);}
    let module=os::GetModuleHandleA(c"kernel32.dll".as_ptr());let address=os::GetProcAddress(module,c"wine_get_unix_file_name".as_ptr());
    if address.is_null() {return Err(CUerror::NOT_SUPPORTED);}let convert:unsafe extern "C" fn(*const u16)->*mut c_char=std::mem::transmute(address);
    let unix=convert(wide.as_ptr());let r=raw("cuModuleLoad",&[unixlib::pack(out),unixlib::pack(unix)]);os::HeapFree(os::GetProcessHeap(),0,unix.cast());r
}
#[no_mangle]
pub unsafe extern "system" fn cuWGLGetDevice(out:*mut CUdevice,_graphics:*mut c_void)->CUresult {let mut dev=0;raw("cuDeviceGet",&[unixlib::pack(&mut dev as *mut CUdevice),0])?;if !out.is_null(){*out=dev;}Ok(())}
macro_rules! device {($name:ident,$ty:ty)=>{#[no_mangle] pub unsafe extern "system" fn $name(out:*mut CUdevice,_adapter:*mut $ty)->CUresult {raw("cuDeviceGet",&[unixlib::pack(out),0])}};}
#[no_mangle]
pub unsafe extern "system" fn cuD3D9GetDevice(out:*mut CUdevice,_adapter:*const c_char)->CUresult {raw("cuDeviceGet",&[unixlib::pack(out),0])}
device!(cuD3D10GetDevice,IDXGIAdapter);device!(cuD3D11GetDevice,IDXGIAdapter);
#[no_mangle]
pub unsafe extern "system" fn cuD3D9CtxCreate(out:*mut CUcontext,device:*mut CUdevice,flags:u32,_graphics:*mut IDirect3DDevice9)->CUresult {
    let mut d=0;raw("cuDeviceGet",&[unixlib::pack(&mut d as *mut CUdevice),0])?;raw("cuCtxCreate",&[unixlib::pack(out),u64::from(flags),unixlib::pack(d)])?;if !device.is_null(){*device=d;}Ok(())
}
macro_rules! context {($name:ident,$native:literal)=>{#[no_mangle] pub unsafe extern "system" fn $name(out:*mut CUcontext,device:*mut CUdevice,flags:u32,_graphics:*mut ID3D11Device)->CUresult {if device.is_null(){return Err(CUerror::INVALID_VALUE);}raw("cuDeviceGet",&[unixlib::pack(device),0])?;raw($native,&[unixlib::pack(out),u64::from(flags),unixlib::pack(*device)])}};}
context!(cuD3D11CtxCreate,"cuCtxCreate");context!(cuD3D11CtxCreate_v2,"cuCtxCreate_v2");
macro_rules! graphics {($name:ident,$ty:ty)=>{#[no_mangle] pub unsafe extern "system" fn $name(_out:*mut CUgraphicsResource,_resource:*mut $ty,flags:u32)->CUresult {Err(if flags!=0 {CUerror::INVALID_VALUE}else{CUerror::UNKNOWN})}};}
graphics!(cuGraphicsD3D9RegisterResource,IDirect3DResource9);graphics!(cuGraphicsD3D10RegisterResource,ID3D10Resource);
#[no_mangle]
pub unsafe extern "system" fn cuDriverGetGpuCodeIsaVersion(_a:*mut c_void,_b:*mut c_void)->CUresult {available(b"cuDriverGetGpuCodeIsaVersion\0")?;Err(CUerror::UNKNOWN)}

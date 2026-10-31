// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{cuda::*,legacy::*,unixlib,callback_protocol as p,pe_callbacks as workers,pe_wrappers::raw};
use std::ffi::c_void;
macro_rules! stream {($name:ident)=>{#[no_mangle] pub unsafe extern "system" fn $name(stream:CUstream,callback:CUstreamCallback,data:*mut c_void,flags:u32)->CUresult {
    let callback=callback.ok_or(CUerror::INVALID_VALUE)?;let token=workers::allocate(p::STREAM,callback as usize,data as usize,true,0)?;
    let r=raw(stringify!($name),&[unixlib::pack(stream),token,0,u64::from(flags)]);if r.is_err(){workers::cancel(token)?;}r
}};}
stream!(cuStreamAddCallback);stream!(cuStreamAddCallback_ptsz);
unsafe fn occupancy(name:&str,grid:*mut i32,block:*mut i32,function:CUfunction,callback:CUoccupancyB2DSize,bytes:usize,limit:i32,flags:u32)->CUresult {
    let token=match callback{Some(f)=>workers::allocate(p::OCCUPANCY,f as usize,0,false,0)?,None=>0};
    let r=raw(name,&[unixlib::pack(grid),unixlib::pack(block),unixlib::pack(function),token,bytes as u64,unixlib::pack(limit),u64::from(flags)]);
    if token!=0{workers::cancel(token)?;}r
}
#[no_mangle]
pub unsafe extern "system" fn cuOccupancyMaxPotentialBlockSize(grid:*mut i32,block:*mut i32,function:CUfunction,callback:CUoccupancyB2DSize,bytes:usize,limit:i32)->CUresult {occupancy("cuOccupancyMaxPotentialBlockSize",grid,block,function,callback,bytes,limit,0)}
#[no_mangle]
pub unsafe extern "system" fn cuOccupancyMaxPotentialBlockSizeWithFlags(grid:*mut i32,block:*mut i32,function:CUfunction,callback:CUoccupancyB2DSize,bytes:usize,limit:i32,flags:u32)->CUresult {occupancy("cuOccupancyMaxPotentialBlockSizeWithFlags",grid,block,function,callback,bytes,limit,flags)}
#[no_mangle]
pub unsafe extern "system" fn cuDeviceRegisterAsyncNotification(device:CUdevice,callback:CUasyncCallback,data:*mut c_void,out:*mut CUasyncCallbackHandle)->CUresult {
    if workers::in_device(){return Err(CUerror::NOT_PERMITTED);}let f=callback.ok_or(CUerror::INVALID_VALUE)?;
    let token=workers::allocate(p::DEVICE,f as usize,data as usize,false,device as u32)?;
    let r=raw("cuDeviceRegisterAsyncNotification",&[unixlib::pack(device),token,0,unixlib::pack(out)]);
    if r.is_err(){workers::cancel(token)?;}else if !out.is_null(){workers::bind(token,*out as usize);}r
}
#[no_mangle]
pub unsafe extern "system" fn cuDeviceUnregisterAsyncNotification(device:CUdevice,handle:CUasyncCallbackHandle)->CUresult {
    if workers::in_device(){return Err(CUerror::NOT_PERMITTED);}raw("cuDeviceUnregisterAsyncNotification",&[unixlib::pack(device),unixlib::pack(handle)])?;workers::retire(p::DEVICE,device as u32,handle as usize)
}
#[no_mangle]
pub unsafe extern "system" fn cuLogsRegisterCallback(callback:CUlogsCallback,data:*mut c_void,out:*mut CUlogsCallbackHandle)->CUresult {
    let f=callback.ok_or(CUerror::INVALID_VALUE)?;let token=workers::allocate(p::LOG,f as usize,data as usize,false,0)?;
    let r=raw("cuLogsRegisterCallback",&[token,0,unixlib::pack(out)]);if r.is_err(){workers::cancel(token)?;}else if !out.is_null(){workers::bind(token,*out as usize);}r
}
#[no_mangle]
pub unsafe extern "system" fn cuLogsUnregisterCallback(handle:CUlogsCallbackHandle)->CUresult {raw("cuLogsUnregisterCallback",&[unixlib::pack(handle)])?;workers::retire(p::LOG,0,handle as usize)}
macro_rules! coredump {($reg:ident,$unreg:ident,$domain:expr)=>{
    #[no_mangle] pub unsafe extern "system" fn $reg(callback:CUcoredumpStatusCallback,data:*mut c_void,out:*mut CUcoredumpCallbackHandle)->CUresult {
        let f=callback.ok_or(CUerror::INVALID_VALUE)?;let token=workers::allocate(p::COREDUMP,f as usize,data as usize,false,$domain)?;
        let r=raw(stringify!($reg),&[token,0,unixlib::pack(out)]);if r.is_err(){workers::cancel(token)?;}else if !out.is_null(){workers::bind(token,*out as usize);}r
    }
    #[no_mangle] pub unsafe extern "system" fn $unreg(handle:CUcoredumpCallbackHandle)->CUresult {raw(stringify!($unreg),&[unixlib::pack(handle)])?;workers::retire(p::COREDUMP,$domain,handle as usize)}
};}
coredump!(cuCoredumpRegisterStartCallback,cuCoredumpDeregisterStartCallback,0);
coredump!(cuCoredumpRegisterCompleteCallback,cuCoredumpDeregisterCompleteCallback,1);

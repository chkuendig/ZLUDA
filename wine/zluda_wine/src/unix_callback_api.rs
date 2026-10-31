// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native CUDA sees SysV shims and opaque tokens only.
use crate::{cuda::*,unixlib::{self,Call},callback_protocol as protocol,unix_callbacks};
use std::{ffi::{c_char,c_void},sync::atomic::{AtomicU64,Ordering}};
unsafe fn symbol(name:&[u8])->Result<*mut c_void,CUerror>{crate::unix_wrappers::symbol(name)}
unsafe extern "C" fn stream(stream:CUstream,status:CUresult,data:*mut c_void){unix_callbacks::emit(data as usize as u64,protocol::STREAM,[unixlib::pack(stream),u64::from(unixlib::code(status)),0,0,0,0]);}
unsafe extern "C" fn device(info:*mut CUasyncNotificationInfo,data:*mut c_void,handle:CUasyncCallbackHandle){unix_callbacks::emit(data as usize as u64,protocol::DEVICE,[unixlib::pack(info),unixlib::pack(handle),0,0,0,0]);}
unsafe extern "C" fn logs(data:*mut c_void,level:CUlogLevel,message:*mut c_char,len:usize){unix_callbacks::emit(data as usize as u64,protocol::LOG,[unixlib::pack(level),unixlib::pack(message),len as u64,0,0,0]);}
unsafe extern "C" fn coredump(data:*mut c_void,process:i32,device:CUdevice){unix_callbacks::emit(data as usize as u64,protocol::COREDUMP,[unixlib::pack(process),unixlib::pack(device),0,0,0,0]);}
type OccupancyCallback=unsafe extern "C" fn(i32)->usize;
const POOL_SIZE:usize=128;
static OCCUPANCY:[AtomicU64;POOL_SIZE]=[const{AtomicU64::new(0)};POOL_SIZE];
include!(concat!(env!("OUT_DIR"),"/occupancy_slots.rs"));
struct Slot(usize);impl Drop for Slot {fn drop(&mut self){OCCUPANCY[self.0].store(0,Ordering::Release);}}
unsafe fn invoke(call:&mut Call)->CUresult {
    let id=call.function;
    if ["cuStreamAddCallback","cuStreamAddCallback_ptsz"].iter().any(|n|unixlib::function_id(n)==id) {
        if call.args[1]==0 || call.args[2]!=0 {return Err(CUerror::INVALID_VALUE);}
        let name=if id==unixlib::function_id("cuStreamAddCallback"){b"cuStreamAddCallback\0".as_slice()}else{b"cuStreamAddCallback_ptsz\0".as_slice()};
        let f:unsafe extern "C" fn(CUstream,Option<unsafe extern "C" fn(CUstream,CUresult,*mut c_void)>,*mut c_void,u32)->CUresult=std::mem::transmute(symbol(name)?);
        return f(call.get(0)?,Some(stream),call.args[1] as usize as *mut c_void,call.get(3)?);
    }
    if ["cuOccupancyMaxPotentialBlockSize","cuOccupancyMaxPotentialBlockSizeWithFlags"].iter().any(|n|unixlib::function_id(n)==id) {
        let token=call.args[3];let slot=if token==0 {None}else {Some(Slot(OCCUPANCY.iter().position(|s|s.compare_exchange(0,token,Ordering::AcqRel,Ordering::Acquire).is_ok()).ok_or(CUerror::OUT_OF_MEMORY)?))};
        let callback=slot.as_ref().map(|s|OCCUPANCY_FUNCTIONS[s.0]);
        if id==unixlib::function_id("cuOccupancyMaxPotentialBlockSize") {
            let f:unsafe extern "C" fn(*mut i32,*mut i32,CUfunction,Option<OccupancyCallback>,usize,i32)->CUresult=std::mem::transmute(symbol(b"cuOccupancyMaxPotentialBlockSize\0")?);return f(call.get(0)?,call.get(1)?,call.get(2)?,callback,call.get(4)?,call.get(5)?);
        }
        let f:unsafe extern "C" fn(*mut i32,*mut i32,CUfunction,Option<OccupancyCallback>,usize,i32,u32)->CUresult=std::mem::transmute(symbol(b"cuOccupancyMaxPotentialBlockSizeWithFlags\0")?);return f(call.get(0)?,call.get(1)?,call.get(2)?,callback,call.get(4)?,call.get(5)?,call.get(6)?);
    }
    if id==unixlib::function_id("cuDeviceRegisterAsyncNotification") {
        if call.args[1]==0 || call.args[2]!=0{return Err(CUerror::INVALID_VALUE);}
        let f:unsafe extern "C" fn(CUdevice,Option<unsafe extern "C" fn(*mut CUasyncNotificationInfo,*mut c_void,CUasyncCallbackHandle)>,*mut c_void,*mut CUasyncCallbackHandle)->CUresult=std::mem::transmute(symbol(b"cuDeviceRegisterAsyncNotification\0")?);
        return f(call.get(0)?,Some(device),call.args[1] as usize as *mut c_void,call.get(3)?);
    }
    if id==unixlib::function_id("cuDeviceUnregisterAsyncNotification") {let f:unsafe extern "C" fn(CUdevice,CUasyncCallbackHandle)->CUresult=std::mem::transmute(symbol(b"cuDeviceUnregisterAsyncNotification\0")?);return f(call.get(0)?,call.get(1)?);}
    if id==unixlib::function_id("cuLogsRegisterCallback") {
        if call.args[0]==0 || call.args[1]!=0{return Err(CUerror::INVALID_VALUE);}
        let f:unsafe extern "C" fn(Option<unsafe extern "C" fn(*mut c_void,CUlogLevel,*mut c_char,usize)>,*mut c_void,*mut CUlogsCallbackHandle)->CUresult=std::mem::transmute(symbol(b"cuLogsRegisterCallback\0")?);return f(Some(logs),call.args[0] as usize as *mut c_void,call.get(2)?);
    }
    if id==unixlib::function_id("cuLogsUnregisterCallback") {let f:unsafe extern "C" fn(CUlogsCallbackHandle)->CUresult=std::mem::transmute(symbol(b"cuLogsUnregisterCallback\0")?);return f(call.get(0)?);}
    for (name,register) in [("cuCoredumpRegisterStartCallback",true),("cuCoredumpRegisterCompleteCallback",true),("cuCoredumpDeregisterStartCallback",false),("cuCoredumpDeregisterCompleteCallback",false)] {
        if id!=unixlib::function_id(name){continue;}
        let bytes=match name {"cuCoredumpRegisterStartCallback"=>b"cuCoredumpRegisterStartCallback\0".as_slice(),"cuCoredumpRegisterCompleteCallback"=>b"cuCoredumpRegisterCompleteCallback\0".as_slice(),"cuCoredumpDeregisterStartCallback"=>b"cuCoredumpDeregisterStartCallback\0".as_slice(),_=>b"cuCoredumpDeregisterCompleteCallback\0".as_slice()};
        if register {if call.args[0]==0 || call.args[1]!=0{return Err(CUerror::INVALID_VALUE);}let f:unsafe extern "C" fn(Option<unsafe extern "C" fn(*mut c_void,i32,CUdevice)>,*mut c_void,*mut crate::legacy::CUcoredumpCallbackHandle)->CUresult=std::mem::transmute(symbol(bytes)?);return f(Some(coredump),call.args[0] as usize as *mut c_void,call.get(2)?);}
        let f:unsafe extern "C" fn(crate::legacy::CUcoredumpCallbackHandle)->CUresult=std::mem::transmute(symbol(bytes)?);return f(call.get(0)?);
    }Err(CUerror::NOT_SUPPORTED)
}
pub unsafe fn dispatch(call:&mut Call)->Option<CUresult> {
    if !["cuStreamAddCallback","cuStreamAddCallback_ptsz","cuOccupancyMaxPotentialBlockSize","cuOccupancyMaxPotentialBlockSizeWithFlags","cuDeviceRegisterAsyncNotification","cuDeviceUnregisterAsyncNotification","cuLogsRegisterCallback","cuLogsUnregisterCallback","cuCoredumpRegisterStartCallback","cuCoredumpRegisterCompleteCallback","cuCoredumpDeregisterStartCallback","cuCoredumpDeregisterCompleteCallback"].iter().any(|n|unixlib::function_id(n)==call.function){return None;}
    call.result=unixlib::code(invoke(call));Some(Ok(()))
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Byte-preserving external-handle adaptation; specification §11.
use crate::{cuda::*,unixlib,pe_platform as os,pe_wrappers::raw};
use std::ffi::c_void;
use crate::pe_sync::Mutex;
unsafe fn query(handle:*mut c_void,code:u32)->Result<u32,()> {
    let mut output=0u32;let mut io=os::IoStatus{status:0,information:0};
    if os::NtDeviceIoControlFile(handle,std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),&mut io,code,std::ptr::null(),0,(&mut output as *mut u32).cast(),4)==0 {Ok(output)}else{Err(())}
}
fn server_handle(value:u32)->*mut c_void {(value as i32 as isize) as *mut c_void}
unsafe fn fd(mut handle:*mut c_void,kmt:bool)->i32 {
    if kmt {handle=query(handle,0x234008).map(server_handle).unwrap_or(usize::MAX as *mut c_void);}
    let mut duplicate=std::ptr::null_mut();let process=usize::MAX as *mut c_void;
    if os::NtDuplicateObject(process,handle,process,&mut duplicate,0,0,2)!=0{return -1;}
    let Ok(resource)=query(duplicate,0x23400c) else {return -1;};let resource=server_handle(resource);
    let module=os::GetModuleHandleA(c"ntdll.dll".as_ptr());let address=os::GetProcAddress(module,c"wine_server_handle_to_fd".as_ptr());
    if address.is_null(){return -1;}
    let convert:unsafe extern "C" fn(*mut c_void,u32,*mut i32,*mut u32)->i32=std::mem::transmute(address);
    let mut output=-1;if convert(resource,1,&mut output,std::ptr::null_mut())!=0 || output<0 {return -1;}
    os::NtClose(resource);output
}
#[repr(align(8))] struct Bytes<const N:usize>([u8;N]);
unsafe fn copy_descriptor<const N:usize>(input:*const c_void,semaphore:bool)->Bytes<N> {
    let mut bytes=[0u8;N];std::ptr::copy_nonoverlapping(input.cast::<u8>(),bytes.as_mut_ptr(),N);
    let kind=u32::from_le_bytes(bytes[..4].try_into().unwrap());
    if kind==2 || kind==3 || (semaphore && kind==10) {
        let handle=usize::from_le_bytes(bytes[8..16].try_into().unwrap()) as *mut c_void;
        let descriptor=fd(handle,kind==3);bytes[..4].copy_from_slice(&(if kind==10 {9u32}else{1u32}).to_le_bytes());bytes[8..12].copy_from_slice(&descriptor.to_le_bytes());
    }Bytes(bytes)
}
#[no_mangle]
pub unsafe extern "system" fn cuImportExternalMemory(out:*mut CUexternalMemory,descriptor:*const CUDA_EXTERNAL_MEMORY_HANDLE_DESC)->CUresult {
    if out.is_null() || descriptor.is_null(){return Err(CUerror::INVALID_VALUE);}
    let bytes=copy_descriptor::<104>(descriptor.cast(),false);
    raw("cuImportExternalMemory",&[unixlib::pack(out),unixlib::pack(bytes.0.as_ptr())])
}
static HISTORY:Mutex<[usize;4]>=Mutex::new([0;4]);
#[no_mangle]
pub unsafe extern "system" fn cuImportExternalSemaphore(out:*mut CUexternalSemaphore,descriptor:*const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC)->CUresult {
    if out.is_null() || descriptor.is_null(){return Err(CUerror::INVALID_VALUE);}
    let bytes=copy_descriptor::<96>(descriptor.cast(),true);
    raw("cuImportExternalSemaphore",&[unixlib::pack(out),unixlib::pack(bytes.0.as_ptr())])?;
    let oldest={let mut history=HISTORY.lock();let first=history[0];history.rotate_left(1);history[3]=*out as usize;first};
    if oldest!=0 {let _=raw("cuDestroyExternalSemaphore",&[oldest as u64]);}Ok(())
}

const _:()={
    assert!(std::mem::size_of::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>()==104);
    assert!(std::mem::align_of::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>()==8);
    assert!(std::mem::size_of::<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>()==96);
    assert!(std::mem::align_of::<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>()==8);
};

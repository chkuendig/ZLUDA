// SPDX-License-Identifier: MIT OR Apache-2.0
//! Windows ABI projection from specification §§10–14. No header-inline code.
use std::ffi::{c_char,c_void};
pub type Handle=*mut c_void;

// Specification §12.2 supplies the Win64 TEB self pointer at GS:0x30. It
// remains available through PE detach and adds no OS call to warmed routing.
#[cfg(target_arch="x86_64")]
#[inline]
pub fn thread_identity()->usize {
    #[cfg(windows)] {
        let identity:usize;
        unsafe {std::arch::asm!("mov {}, gs:[0x30]",out(reg) identity,options(nostack,readonly,preserves_flags));}
        identity
    }
    #[cfg(not(windows))] {HOSTED_THREAD_ID.with(|cell|cell as *const _ as usize)}
}

// The PE projection is also compiled by native Linux CPU fixtures. This
// const, Drop-free Linux cell is independent of Windows std TLS; the
// coordinator explicitly confirmed its lifetime through PE teardown.
#[cfg(not(windows))]
thread_local! {static HOSTED_THREAD_ID:std::cell::Cell<u8>=const{std::cell::Cell::new(0)};}

#[cfg_attr(windows,link(name="kernel32"))]
extern "system" {
    pub fn GetModuleHandleA(name:*const c_char)->Handle;
    pub fn GetProcAddress(module:Handle,name:*const c_char)->*mut c_void;
    pub fn GetModuleHandleExW(flags:u32,name:*const u16,out:*mut Handle)->i32;
    pub fn LoadLibraryExW(name:*const u16,reserved:Handle,flags:u32)->Handle;
    pub fn FreeLibrary(module:Handle)->i32;
    pub fn GetProcessHeap()->Handle;
    pub fn HeapAlloc(heap:Handle,flags:u32,size:usize)->*mut c_void;
    pub fn HeapFree(heap:Handle,flags:u32,memory:*mut c_void)->i32;
    pub fn MultiByteToWideChar(page:u32,flags:u32,source:*const c_char,len:i32,out:*mut u16,capacity:i32)->i32;
    pub fn GetCurrentProcessId()->u32;
    pub fn GetCurrentThreadId()->u32;
    pub fn Sleep(ms:u32);
    pub fn GetTickCount64()->u64;
    pub fn RaiseException(code:u32,flags:u32,count:u32,parameters:*const usize);
}
#[repr(C)] pub struct IoStatus {pub status:usize,pub information:usize}
#[repr(C)] pub struct Unicode {pub len:u16,pub max:u16,#[cfg(target_pointer_width="64")] pub padding:u32,pub buffer:*const u16}
#[repr(C)] pub struct Object {pub len:u32,#[cfg(target_pointer_width="64")] pub padding:u32,pub root:Handle,pub name:*const Unicode,pub attributes:u32,#[cfg(target_pointer_width="64")] pub padding2:u32,pub security:Handle,pub qos:Handle}
#[cfg_attr(windows,link(name="ntdll"))]
extern "system" {
    pub fn NtClose(handle:Handle)->i32;
    pub fn NtDuplicateObject(source_process:Handle,source:Handle,target_process:Handle,out:*mut Handle,access:u32,attributes:u32,options:u32)->i32;
    pub fn NtDeviceIoControlFile(file:Handle,event:Handle,apc:Handle,context:Handle,status:*mut IoStatus,code:u32,input:*const c_void,input_bytes:u32,output:*mut c_void,output_bytes:u32)->i32;
    pub fn NtOpenKeyEx(out:*mut Handle,access:u32,object:*const Object,options:u32)->i32;
    pub fn NtEnumerateKey(key:Handle,index:u32,class:u32,out:*mut c_void,capacity:u32,returned:*mut u32)->i32;
    pub fn NtQueryValueKey(key:Handle,name:*const Unicode,class:u32,out:*mut c_void,capacity:u32,returned:*mut u32)->i32;
}
extern "C" {
    pub fn zluda_wine_attach(module:Handle)->i32;
    pub fn zluda_wine_call(packet:*mut c_void)->i32;
    pub fn zluda_wine_callback_ready()->i32;
    pub fn zluda_wine_worker_allowed()->i32;
}
pub unsafe fn pin(address:*const ())->bool {let mut module=std::ptr::null_mut();GetModuleHandleExW(5,address.cast(),&mut module)!=0}

#[cfg(target_pointer_width="64")]
const _:()={assert!(std::mem::size_of::<Unicode>()==16);assert!(std::mem::size_of::<Object>()==48);assert!(std::mem::size_of::<IoStatus>()==16);};

#[cfg(target_pointer_width="64")]
const _:()={
    assert!(std::mem::offset_of!(Unicode,buffer)==8);
    assert!(std::mem::offset_of!(Object,root)==8);
    assert!(std::mem::offset_of!(Object,name)==16);
    assert!(std::mem::offset_of!(Object,attributes)==24);
    assert!(std::mem::offset_of!(Object,security)==32);
    assert!(std::mem::offset_of!(Object,qos)==40);
    assert!(std::mem::offset_of!(IoStatus,information)==8);
};

// Specification §14 supplies both NT data models. Only explicit PE64 padding
// fields are omitted on PE32; pointer fields retain the target pointer width.
#[cfg(target_pointer_width="32")]
const _:()={
    assert!(std::mem::size_of::<Unicode>()==8);
    assert!(std::mem::align_of::<Unicode>()==4);
    assert!(std::mem::offset_of!(Unicode,buffer)==4);
    assert!(std::mem::size_of::<Object>()==24);
    assert!(std::mem::align_of::<Object>()==4);
    assert!(std::mem::offset_of!(Object,root)==4);
    assert!(std::mem::offset_of!(Object,name)==8);
    assert!(std::mem::offset_of!(Object,attributes)==12);
    assert!(std::mem::offset_of!(Object,security)==16);
    assert!(std::mem::offset_of!(Object,qos)==20);
    assert!(std::mem::size_of::<IoStatus>()==8);
    assert!(std::mem::offset_of!(IoStatus,information)==4);
};

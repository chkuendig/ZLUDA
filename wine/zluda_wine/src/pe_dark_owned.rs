// SPDX-License-Identifier: MIT OR Apache-2.0
//! Windows heap/TLS, hybrid loading, and canonical integrity arithmetic.
use crate::{cuda::*,pe_platform as os,integrity_algorithm::{self,DeviceHashinfo}};
use std::ffi::{c_char,c_void};
use crate::pe_sync::Mutex;
#[repr(C)] struct Prefix {callback:usize,data:usize}
struct Entry {pointer:usize,registered:bool,running:usize}
struct State {owner:Option<usize>,depth:usize,entries:Vec<Entry>}
static TLS:Mutex<State>=Mutex::new(State{owner:None,depth:0,entries:Vec::new()});
struct Guard;
impl Guard {
    fn enter()->Self {
        let id=os::thread_identity();
        loop {
            let mut s=TLS.lock();
            if s.owner.is_none() || s.owner==Some(id){s.owner=Some(id);s.depth+=1;return Self;}
            drop(s);crate::pe_sync::wait();
        }
    }
}
impl Drop for Guard {fn drop(&mut self){let mut s=TLS.lock();s.depth-=1;if s.depth==0{s.owner=None;}}}
unsafe extern "win64" fn register(out:*mut *const c_void,callback:usize,data:usize)->CUresult {
    if out.is_null(){return Err(CUerror::INVALID_VALUE);}let _guard=Guard::enter();
    {TLS.lock().entries.try_reserve(1).map_err(|_|CUerror::OUT_OF_MEMORY)?;}
    let pointer=os::HeapAlloc(os::GetProcessHeap(),0,std::mem::size_of::<Prefix>()).cast::<Prefix>();if pointer.is_null(){return Err(CUerror::OUT_OF_MEMORY);}pointer.write(Prefix{callback,data});
    TLS.lock().entries.push(Entry{pointer:pointer as usize,registered:true,running:0});*out=pointer.cast();Ok(())
}
unsafe extern "win64" fn remove(pointer:*const c_void,_ignored:*mut usize)->CUresult {
    if pointer.is_null(){return Err(CUerror::INVALID_VALUE);}let _guard=Guard::enter();
    let release={let mut s=TLS.lock();let e=s.entries.iter_mut().find(|e|e.pointer==pointer as usize && e.registered).ok_or(CUerror::INVALID_VALUE)?;e.registered=false;if e.running==0{e.pointer=0;true}else{false}};
    if release{os::HeapFree(os::GetProcessHeap(),0,pointer.cast_mut());}Ok(())
}
pub unsafe fn thread_detach() {
    let _guard=Guard::enter();let mut index=0;
    loop {
        let pointer={let mut s=TLS.lock();let Some(e)=s.entries.get_mut(index)else{break;};index+=1;if !e.registered{continue;}e.running+=1;e.pointer};
        let record=&*(pointer as *const Prefix);
        if record.callback!=0{let f:unsafe extern "win64" fn(u32,*mut c_void)=std::mem::transmute(record.callback);let _scope=crate::pe_callbacks::enter_callback();f(0,record.data as *mut c_void);}
        let release={let mut s=TLS.lock();let e=&mut s.entries[index-1];e.running-=1;if !e.registered && e.running==0{e.pointer=0;true}else{false}};
        if release{os::HeapFree(os::GetProcessHeap(),0,pointer as *mut c_void);}
    }
}
unsafe extern "win64" fn hybrid_load(_name:*const c_char,function:*mut *const c_void,token:*mut usize)->CUresult {
    if function.is_null() || token.is_null(){return Err(CUerror::INVALID_VALUE);}
    let name:Vec<_>="nvcudart_hybrid64.dll\0".encode_utf16().collect();let module=os::LoadLibraryExW(name.as_ptr(),std::ptr::null_mut(),0x800);if module.is_null(){return Err(CUerror::INVALID_HANDLE);}
    let entry=os::GetProcAddress(module,c"__cudaGetProcAddress".as_ptr());if entry.is_null(){return Err(CUerror::INVALID_HANDLE);}*function=entry;*token=module as usize;Ok(())
}
unsafe extern "win64" fn hybrid_free(token:usize)->CUresult {if token==0{return Err(CUerror::INVALID_HANDLE);}os::FreeLibrary(token as *mut c_void);Ok(())}
unsafe extern "win64" fn integrity(version:u32,seconds:u64,out:*mut [u64;2])->CUresult {
    let integrity_index=crate::dark_protocol::TABLES.iter().position(|s|s.name=="INTEGRITY_CHECK").unwrap();
    let integrity_table=crate::pe_dark_tables::obtain(integrity_index)?;let cudart_table=crate::pe_dark_tables::obtain(0)?;
    if out.is_null(){return Err(CUerror::INVALID_VALUE);}
    let mut driver=0;
    crate::pe_thunks::cuDriverGetVersion(&mut driver).map_err(|_|CUerror::UNKNOWN)?;
    let mut devices=Vec::new();
    if version%10>=2 {
        let mut count=0;if crate::pe_thunks::cuDeviceGetCount(&mut count).is_ok() && count>0 {
            devices.try_reserve_exact(count as usize).map_err(|_|CUerror::OUT_OF_MEMORY)?;
            for ordinal in 0..count {
                let mut device=0;let mut record=DeviceHashinfo{guid:CUuuid{bytes:[0;16]},pci_domain:0,pci_bus:0,pci_device:0};
                if crate::pe_thunks::cuDeviceGet(&mut device,ordinal).is_err() || crate::pe_wrappers::cuDeviceGetUuid(&mut record.guid,device).is_err()
                    || crate::pe_wrappers::cuDeviceGetAttribute(&mut record.pci_bus,CUdevice_attribute(33),device).is_err()
                    || crate::pe_wrappers::cuDeviceGetAttribute(&mut record.pci_device,CUdevice_attribute(34),device).is_err()
                    || crate::pe_wrappers::cuDeviceGetAttribute(&mut record.pci_domain,CUdevice_attribute(50),device).is_err(){break;}
                devices.push(record);
            }
        }
    }
    let result=integrity_algorithm::integrity_check(version,seconds,driver as u32,os::GetCurrentProcessId(),os::GetCurrentThreadId(),integrity_table,cudart_table,integrity as *const () as *const c_void,devices.len() as u32,|i|devices[i as usize]);*out=result;Ok(())
}
pub fn pointer(name:&str,slot:usize)->Option<*mut c_void> {
    let f=match (name,slot) {
        ("HEAP_ACCESS",1)=>register as *const (),("HEAP_ACCESS",2)=>remove as *const (),
        ("HYBRID_CUDART",1)=>hybrid_load as *const (),("HYBRID_CUDART",2)=>hybrid_free as *const (),
        ("INTEGRITY_CHECK",1)=>integrity as *const (),
        ("CONTEXT_LOCAL_STORAGE_INTERFACE_V0301",0)=>crate::pe_context_storage::put as *const (),
        ("CONTEXT_LOCAL_STORAGE_INTERFACE_V0301",1)=>crate::pe_context_storage::remove as *const (),
        ("CONTEXT_LOCAL_STORAGE_INTERFACE_V0301",2)=>crate::pe_context_storage::get as *const (),_=>return None,
    };Some(f.cast_mut().cast())
}

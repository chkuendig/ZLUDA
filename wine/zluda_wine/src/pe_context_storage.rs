// SPDX-License-Identifier: MIT OR Apache-2.0
//! Immutable retained associations make warm storage gets lock-free.
use crate::{cuda::*,unixlib::{self,Call},context_protocol as protocol};
use std::{ffi::c_void,sync::atomic::{AtomicPtr,AtomicBool,AtomicU64,Ordering}};
type Destructor=unsafe extern "win64" fn(CUcontext,*mut c_void,*mut c_void);
struct Record {next:*mut Record,token:u64,context:usize,key:usize,value:usize,destructor:Option<Destructor>,active:AtomicBool}
static RECORDS:AtomicPtr<Record>=AtomicPtr::new(std::ptr::null_mut());
static NEXT:AtomicU64=AtomicU64::new(1);
fn token()->Result<u64,CUerror> {NEXT.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|v|if v<i64::MAX as u64{Some(v+1)}else{None}).map_err(|_|CUerror::OUT_OF_MEMORY)}
unsafe fn find(token:u64)->Option<&'static Record> {let mut p=RECORDS.load(Ordering::Acquire);while !p.is_null(){let r=&*p;if r.token==token{return Some(r);}p=r.next;}None}
fn table()->u64 {crate::dark_protocol::TABLES.iter().position(|s|s.name=="CONTEXT_LOCAL_STORAGE_INTERFACE_V0301").unwrap() as u64+1}
unsafe fn ready()->CUresult {
    if crate::pe_platform::zluda_wine_callback_ready()==0{return Err(CUerror::NOT_SUPPORTED);}
    let mut c=Call::new(protocol::INITIALIZE);c.args[0]=entry as *const () as usize as u64;crate::pe_wrappers::submit(&mut c)
}
unsafe extern "win64" fn entry(data:*mut c_void,bytes:u32)->i32 {
    if data.is_null() || bytes!=48 || (data as usize)%8!=0 {return 0xc000000du32 as i32;}
    let e=&*data.cast::<protocol::Event>();if e.magic!=protocol::MAGIC || e.api_hash!=unixlib::API_HASH{return 0xc000000du32 as i32;}
    if let Some(r)=find(e.token) {if r.context as u64==e.context && r.key as u64==e.key && r.active.swap(false,Ordering::AcqRel) {if let Some(f)=r.destructor{let _scope=crate::pe_callbacks::enter_callback();f(unixlib::unpack(e.context),e.key as usize as *mut c_void,r.value as *mut c_void);}}}0
}
unsafe fn context(context:&mut CUcontext)->CUresult {if context.0.is_null(){crate::pe_thunks::cuCtxGetCurrent(context)?;}Ok(())}
pub unsafe extern "win64" fn put(mut ctx:CUcontext,key:*mut c_void,value:*mut c_void,destructor:Option<Destructor>)->CUresult {
    ready()?;context(&mut ctx)?;let token=token()?;
    let p=std::alloc::alloc(std::alloc::Layout::new::<Record>()).cast::<Record>();if p.is_null(){return Err(CUerror::OUT_OF_MEMORY);}
    p.write(Record{next:std::ptr::null_mut(),token,context:ctx.0 as usize,key:key as usize,value:value as usize,destructor,active:AtomicBool::new(true)});
    let mut head=RECORDS.load(Ordering::Acquire);loop{(*p).next=head;match RECORDS.compare_exchange_weak(head,p,Ordering::Release,Ordering::Acquire){Ok(_)=>break,Err(next)=>head=next}}
    let mut c=Call::new(crate::dark_protocol::CALL);c.args[..6].copy_from_slice(&[table(),0,ctx.0 as usize as u64,key as usize as u64,token,0]);
    let result=crate::pe_wrappers::submit(&mut c);if result.is_err(){(*p).active.store(false,Ordering::Release);}result
}
pub unsafe extern "win64" fn get(out:*mut *mut c_void,ctx:CUcontext,key:*mut c_void)->CUresult {
    if out.is_null(){return Err(CUerror::INVALID_VALUE);}let mut token=std::ptr::null_mut::<c_void>();let mut c=Call::new(crate::dark_protocol::CALL);
    c.args[..5].copy_from_slice(&[table(),2,(&mut token as *mut *mut c_void) as usize as u64,ctx.0 as usize as u64,key as usize as u64]);crate::pe_wrappers::submit(&mut c)?;
    let r=find(token as usize as u64).filter(|r|r.active.load(Ordering::Acquire) && r.key==key as usize && (ctx.0.is_null() || r.context==ctx.0 as usize)).ok_or(CUerror::INVALID_HANDLE)?;
    *out=r.value as *mut c_void;Ok(())
}
pub unsafe extern "win64" fn remove(mut ctx:CUcontext,key:*mut c_void)->CUresult {
    context(&mut ctx)?;let mut token=std::ptr::null_mut::<c_void>();let mut c=Call::new(crate::dark_protocol::CALL);
    c.args[..5].copy_from_slice(&[table(),2,(&mut token as *mut *mut c_void) as usize as u64,ctx.0 as usize as u64,key as usize as u64]);let _=crate::pe_wrappers::submit(&mut c);
    c=Call::new(crate::dark_protocol::CALL);c.args[..4].copy_from_slice(&[table(),1,ctx.0 as usize as u64,key as usize as u64]);crate::pe_wrappers::submit(&mut c)?;
    if let Some(r)=find(token as usize as u64){if r.context==ctx.0 as usize && r.key==key as usize {r.active.store(false,Ordering::Release);}}Ok(())
}

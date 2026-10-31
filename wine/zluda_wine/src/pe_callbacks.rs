// SPDX-License-Identifier: MIT OR Apache-2.0
//! Windows workers and registration ownership. Never execute PE code on pthreads.
use crate::{cuda::*,unixlib::{self,Call},callback_protocol as protocol,pe_platform as os};
use std::{ffi::c_void,sync::{atomic::{AtomicBool,AtomicPtr,AtomicU64,AtomicUsize,Ordering}}};
use crate::pe_sync::Once as OnceLock;
struct Record {next:*mut Record,token:u64,kind:u64,callback:usize,data:usize,once:bool,domain:u32,handle:AtomicU64,active:AtomicBool}
static RECORDS:AtomicPtr<Record>=AtomicPtr::new(std::ptr::null_mut());
static WORKER:OnceLock<CUresult>=OnceLock::new();

// Callback values belong to the thread identified by `identity`. Atomic
// fields also make subsequent reuse of a TEB identity race-safe. Records
// live in process-owned heap storage, never in Rust's Windows TLS runtime.
struct ThreadState {next:*mut ThreadState,identity:usize,job:AtomicU64,token:AtomicU64,kind:AtomicU64,depth:AtomicUsize}
impl ThreadState {
    fn job(&self)->(u64,u64,u64) {(self.job.load(Ordering::Acquire),self.token.load(Ordering::Relaxed),self.kind.load(Ordering::Relaxed))}
    fn set_job(&self,job:(u64,u64,u64)) {self.token.store(job.1,Ordering::Relaxed);self.kind.store(job.2,Ordering::Relaxed);self.job.store(job.0,Ordering::Release);}
}
static THREADS:AtomicPtr<ThreadState>=AtomicPtr::new(std::ptr::null_mut());
unsafe fn find_thread(identity:usize)->Option<&'static ThreadState> {
    let mut p=THREADS.load(Ordering::Acquire);
    while !p.is_null(){let state=&*p;if state.identity==identity{return Some(state);}p=state.next;}
    None
}
fn thread_state(create:bool)->Option<&'static ThreadState> {
    let identity=os::thread_identity();
    unsafe {
        if let Some(state)=find_thread(identity){return Some(state);}
        if !create{return None;}
        let layout=std::alloc::Layout::new::<ThreadState>();
        let p=std::alloc::alloc(layout).cast::<ThreadState>();
        if p.is_null(){return None;}
        p.write(ThreadState{next:std::ptr::null_mut(),identity,job:AtomicU64::new(0),token:AtomicU64::new(0),kind:AtomicU64::new(0),depth:AtomicUsize::new(0)});
        let mut head=THREADS.load(Ordering::Acquire);
        loop {
            (*p).next=head;
            match THREADS.compare_exchange_weak(head,p,Ordering::Release,Ordering::Acquire) {
                Ok(_)=>return Some(&*p),
                Err(next)=>{
                    // A reentrant first publication may already have supplied
                    // this identity. Discard only our unpublished allocation.
                    if let Some(state)=find_thread(identity){std::alloc::dealloc(p.cast(),layout);return Some(state);}
                    head=next;
                }
            }
        }
    }
}
fn current_job()->(u64,u64,u64) {thread_state(false).map_or((0,0,0),|state|state.job())}
pub struct CallbackScope {state:&'static ThreadState,previous:usize}
pub fn enter_callback()->CallbackScope {
    let state=thread_state(true).unwrap_or_else(||std::process::abort());
    let previous=state.depth.load(Ordering::Relaxed);state.depth.store(previous.checked_add(1).unwrap_or_else(||std::process::abort()),Ordering::Relaxed);
    CallbackScope{state,previous}
}
impl Drop for CallbackScope {fn drop(&mut self){self.state.depth.store(self.previous,Ordering::Relaxed);}}
struct JobScope {state:&'static ThreadState,previous:(u64,u64,u64)}
impl JobScope {
    fn enter(job:(u64,u64,u64))->Self {
        let state=thread_state(true).unwrap_or_else(||std::process::abort());
        let previous=state.job();state.set_job(job);Self{state,previous}
    }
}
impl Drop for JobScope {fn drop(&mut self){self.state.set_job(self.previous);}}
pub fn inside()->bool {thread_state(false).is_some_and(|state|state.job().0!=0 || state.depth.load(Ordering::Relaxed)!=0)}
pub fn in_device()->bool {current_job().2==protocol::DEVICE}
pub fn thread_detach() {
    // The TEB identity can be reused by a later Windows thread. Keep the
    // immutable node for lock-free readers, but leave no callback state in it.
    if let Some(state)=thread_state(false){state.set_job((0,0,0));state.depth.store(0,Ordering::Relaxed);}
}
pub unsafe fn route(call:&mut Call)->i32 {
    let (job,_,_)=current_job();if job==0{return crate::pe_wrappers::direct(call);}
    let mut c=Call::new(protocol::REENTER);c.args[0]=job;c.put(1,call as *mut Call);
    let status=crate::pe_wrappers::direct(&mut c);if status!=0 {return status;}
    if c.result!=0 {call.result=c.result;return 0;}c.args[0] as u32 as i32
}
unsafe fn control(call:&mut Call)->CUresult {crate::pe_wrappers::submit(call)}
unsafe fn ensure_worker()->CUresult {
    if os::zluda_wine_worker_allowed()==0{return Err(CUerror::NOT_SUPPORTED);}
    *WORKER.get_or_init(|| {
        if !os::pin(dispatcher as *const ()){return Err(CUerror::NOT_SUPPORTED);}
        std::thread::Builder::new().spawn(||unsafe{dispatcher()}).map(|_|()).map_err(|_|CUerror::OUT_OF_MEMORY)
    })
}
unsafe fn find(token:u64)->Option<&'static Record> {let mut p=RECORDS.load(Ordering::Acquire);while !p.is_null(){let r=&*p;if r.token==token{return Some(r);}p=r.next;}None}
pub unsafe fn allocate(kind:u64,callback:usize,data:usize,once:bool,domain:u32)->Result<u64,CUerror> {
    if callback==0{return Err(CUerror::INVALID_VALUE);}ensure_worker()?;
    let p=std::alloc::alloc(std::alloc::Layout::new::<Record>()).cast::<Record>();if p.is_null(){return Err(CUerror::OUT_OF_MEMORY);}
    let mut c=Call::new(protocol::ALLOC);c.args[0]=kind;c.args[1]=u64::from(once);
    if let Err(e)=control(&mut c){std::alloc::dealloc(p.cast(),std::alloc::Layout::new::<Record>());return Err(e);}
    let token=c.args[0];p.write(Record{next:std::ptr::null_mut(),token,kind,callback,data,once,domain,handle:AtomicU64::new(0),active:AtomicBool::new(true)});
    let mut head=RECORDS.load(Ordering::Acquire);loop{(*p).next=head;match RECORDS.compare_exchange_weak(head,p,Ordering::Release,Ordering::Acquire){Ok(_)=>break,Err(n)=>head=n}}
    Ok(token)
}
pub unsafe fn bind(token:u64,handle:usize){if let Some(r)=find(token){r.handle.store(handle as u64,Ordering::Release);}}
pub unsafe fn cancel(token:u64)->CUresult {
    if let Some(r)=find(token){r.active.store(false,Ordering::Release);}
    let (job,current,_)=current_job();let mut c=Call::new(protocol::CANCEL);c.args[0]=token;c.args[1]=if current==token {job}else{0};control(&mut c)
}
pub unsafe fn retire(kind:u64,domain:u32,handle:usize)->CUresult {
    let mut p=RECORDS.load(Ordering::Acquire);while !p.is_null(){let r=&*p;if r.kind==kind && r.domain==domain && r.handle.load(Ordering::Acquire)==handle as u64 && r.active.load(Ordering::Acquire){cancel(r.token)?;}p=r.next;}Ok(())
}
unsafe fn dispatcher()->! {
    loop {
        let mut c=Call::new(protocol::DEQUEUE);if control(&mut c).is_err(){std::process::abort();}
        let payload=c.args;
        if std::thread::Builder::new().spawn(move||unsafe{deliver(payload)}).is_err(){std::process::abort();}
    }
}
unsafe fn deliver(args:[u64;16]) {
    let (job,token,kind)=(args[0],args[1],args[2]);let _job=JobScope::enter((job,token,kind));
    let mut result=0;
    if let Some(r)=find(token).filter(|r|r.kind==kind && r.active.load(Ordering::Acquire)) {
        let data=r.data as *mut c_void;
        match kind {
            protocol::STREAM=>{let f:unsafe extern "win64" fn(CUstream,CUresult,*mut c_void)=std::mem::transmute(r.callback);f(unixlib::unpack(args[3]),unixlib::result(args[4] as u32),data);},
            protocol::HOST=>{
                // Internal reverse-call kind (§12.1); public host APIs remain status-only.
                let f:unsafe extern "win64" fn(*mut c_void)=std::mem::transmute(r.callback);
                f(data);
            },
            protocol::OCCUPANCY=>{let f:unsafe extern "win64" fn(i32)->usize=std::mem::transmute(r.callback);result=f(args[3] as u32 as i32) as u64;},
            protocol::DEVICE=>{let f:unsafe extern "win64" fn(*mut CUasyncNotificationInfo,*mut c_void,CUasyncCallbackHandle)=std::mem::transmute(r.callback);f(unixlib::unpack(args[3]),data,unixlib::unpack(args[4]));},
            protocol::LOG=>{let f:unsafe extern "win64" fn(*mut c_void,CUlogLevel,*mut std::ffi::c_char,usize)=std::mem::transmute(r.callback);f(data,unixlib::unpack(args[3]),unixlib::unpack(args[4]),args[5] as usize);},
            protocol::COREDUMP=>{let f:unsafe extern "win64" fn(*mut c_void,i32,CUdevice)=std::mem::transmute(r.callback);f(data,args[3] as u32 as i32,unixlib::unpack(args[4]));},_=>std::process::abort(),
        }
        if r.once {r.active.store(false,Ordering::Release);}
    }
    let mut c=Call::new(protocol::ACK);c.args[..3].copy_from_slice(&[job,token,result]);if control(&mut c).is_err(){std::process::abort();}
}

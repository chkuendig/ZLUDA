// SPDX-License-Identifier: MIT OR Apache-2.0
//! Borrowed reverse calls and native-thread nested execution.
use crate::{cuda::*,unixlib::{self,Call},callback_protocol as protocol};
use std::{collections::{HashMap,VecDeque},sync::{Arc,Mutex,Condvar,OnceLock}};
struct Registration {kind:u64,once:bool,claimed:bool}
struct Request {pointer:usize,result:Mutex<Option<i32>>,ready:Condvar}
#[derive(Clone,Copy,PartialEq)] enum Phase {Queued,Dispatched,Done}
struct Delivery {phase:Phase,result:u64,nested:VecDeque<Arc<Request>>}
struct Job {id:u64,token:u64,kind:u64,payload:[u64;6],state:Mutex<Delivery>,ready:Condvar}
struct State {token:u64,job:u64,registrations:HashMap<u64,Registration>,jobs:Vec<Arc<Job>>,queue:VecDeque<Arc<Job>>}
static SERVICE:OnceLock<(Mutex<State>,Condvar)>=OnceLock::new();
fn state()->&'static (Mutex<State>,Condvar) {SERVICE.get_or_init(||(Mutex::new(State{token:1,job:1,registrations:HashMap::new(),jobs:Vec::new(),queue:VecDeque::new()}),Condvar::new()))}
fn new_id(next:&mut u64)->Result<u64,CUerror> {if *next>=i64::MAX as u64{return Err(CUerror::OUT_OF_MEMORY);}let n=*next;*next+=1;Ok(n)}
fn abort_on_error<T,E>(r:Result<T,E>)->T {match r{Ok(v)=>v,Err(_)=>std::process::abort()}}
pub unsafe fn emit(token:u64,kind:u64,payload:[u64;6])->u64 {
    let (mutex,changed)=state();
    let job={let mut s=abort_on_error(mutex.lock());
        let Some(reg)=s.registrations.get_mut(&token)else{return 0;};
        if reg.kind!=kind || (reg.once && reg.claimed) {return 0;}reg.claimed=true;
        let id=abort_on_error(new_id(&mut s.job));
        let job=Arc::new(Job{id,token,kind,payload,state:Mutex::new(Delivery{phase:Phase::Queued,result:0,nested:VecDeque::new()}),ready:Condvar::new()});
        s.jobs.push(job.clone());s.queue.push_back(job.clone());changed.notify_all();job};
    let result=loop {
        let mut d=abort_on_error(job.state.lock());
        if let Some(request)=d.nested.pop_front() {drop(d);let status=crate::unix_wrappers::dispatch(request.pointer as *mut std::ffi::c_void);*abort_on_error(request.result.lock())=Some(status);request.ready.notify_all();continue;}
        if d.phase==Phase::Done {break d.result;}
        drop(abort_on_error(job.ready.wait(d)));
    };
    let mut s=abort_on_error(mutex.lock());s.jobs.retain(|j|j.id!=job.id);
    if s.registrations.get(&token).is_some_and(|r|r.once) {s.registrations.remove(&token);}
    changed.notify_all();result
}
fn public(function:u64)->bool {function>12 || matches!(function,2|3|4|10|11)}
pub unsafe fn service(call:&mut Call)->CUresult {
    let (mutex,changed)=state();
    match call.function {
        protocol::ALLOC=> {
            let kind=call.args[0];if !(1..=6).contains(&kind) || call.args[1]>1 {return Err(CUerror::INVALID_VALUE);}
            let mut s=mutex.lock().map_err(|_|CUerror::UNKNOWN)?;s.registrations.try_reserve(1).map_err(|_|CUerror::OUT_OF_MEMORY)?;
            let token=new_id(&mut s.token)?;s.registrations.insert(token,Registration{kind,once:call.args[1]==1,claimed:false});call.args[0]=token;
        },
        protocol::CANCEL=> {
            let token=call.args[0];let exclude=call.args[1];let mut s=mutex.lock().map_err(|_|CUerror::UNKNOWN)?;s.registrations.remove(&token);
            s.queue.retain(|j|{if j.token!=token{return true;}let mut d=abort_on_error(j.state.lock());d.phase=Phase::Done;j.ready.notify_all();false});
            while s.jobs.iter().any(|j|j.token==token && j.id!=exclude && abort_on_error(j.state.lock()).phase!=Phase::Done) {s=changed.wait(s).map_err(|_|CUerror::UNKNOWN)?;}
        },
        protocol::DEQUEUE=> {
            let mut s=mutex.lock().map_err(|_|CUerror::UNKNOWN)?;
            let job=loop {if let Some(j)=s.queue.pop_front(){break j;}s=changed.wait(s).map_err(|_|CUerror::UNKNOWN)?;};
            abort_on_error(job.state.lock()).phase=Phase::Dispatched;
            call.args=[0;16];call.args[..3].copy_from_slice(&[job.id,job.token,job.kind]);call.args[3..9].copy_from_slice(&job.payload);
        },
        protocol::ACK=> {
            let s=mutex.lock().map_err(|_|CUerror::UNKNOWN)?;let j=s.jobs.iter().find(|j|j.id==call.args[0] && j.token==call.args[1]).ok_or(CUerror::INVALID_HANDLE)?;
            let mut d=j.state.lock().map_err(|_|CUerror::UNKNOWN)?;if d.phase!=Phase::Dispatched{return Err(CUerror::INVALID_VALUE);}d.result=call.args[2];d.phase=Phase::Done;j.ready.notify_all();changed.notify_all();
        },
        protocol::REENTER=> {
            let pointer=call.args[1] as usize;if pointer==0 || pointer%std::mem::align_of::<Call>()!=0 || pointer==call as *mut Call as usize {return Err(CUerror::INVALID_VALUE);}
            // Validate the envelope's fixed prefix before accessing arguments.
            let p=pointer as *const u8;
            if std::ptr::read(p.cast::<u64>())!=unixlib::MAGIC || std::ptr::read(p.add(8).cast::<u32>())!=unixlib::VERSION || std::ptr::read(p.add(12).cast::<u32>()) as usize!=std::mem::size_of::<Call>() {return Err(CUerror::INVALID_VALUE);}
            let inner=&*(pointer as *const Call);if !inner.valid() || !public(inner.function){return Err(CUerror::INVALID_VALUE);}
            let request=Arc::new(Request{pointer,result:Mutex::new(None),ready:Condvar::new()});
            {let s=mutex.lock().map_err(|_|CUerror::UNKNOWN)?;let job=s.jobs.iter().find(|j|j.id==call.args[0]).ok_or(CUerror::INVALID_HANDLE)?;
                let mut d=job.state.lock().map_err(|_|CUerror::UNKNOWN)?;if d.phase!=Phase::Dispatched{return Err(CUerror::INVALID_VALUE);}d.nested.push_back(request.clone());job.ready.notify_all();}
            let mut r=request.result.lock().map_err(|_|CUerror::UNKNOWN)?;while r.is_none(){r=request.ready.wait(r).map_err(|_|CUerror::UNKNOWN)?;}call.args[0]=r.unwrap() as u32 as u64;
        },_=>return Err(CUerror::INVALID_VALUE),
    }call.result=0;Ok(())
}

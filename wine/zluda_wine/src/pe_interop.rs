// SPDX-License-Identifier: MIT OR Apache-2.0
//! ZLUDA CPU-copy design: authorized owner d3d11_cpu_copy.h and specification §15.
//! A logical operation lease serializes ownership; no mutex spans COM or CUDA.
use crate::{cuda::*,unixlib::{self,Call},d3d11 as com,pe_platform as os,pe_wrappers::{raw,available}};
use std::{ffi::c_void,cell::UnsafeCell,sync::atomic::{AtomicBool,Ordering}};
use crate::pe_sync::Mutex;
#[derive(Default)]
struct Resource {texture:usize,staging:usize,context:usize,multithread:usize,ready:usize,identity:usize,cuda:usize,array:usize,pixels:usize,width:u32,height:u32,event:usize,mapped:bool,pending:bool,poisoned:bool,primed:bool,flags:u32}
struct Node {next:usize,data:UnsafeCell<Resource>}
struct State {head:usize,owner:Option<usize>}
static REGISTRY:Mutex<State>=Mutex::new(State{head:0,owner:None});
static ACTIVE:AtomicBool=AtomicBool::new(false);
struct Lease;
impl Lease {
    fn acquire()->Result<Self,CUerror> {
        let id=os::thread_identity();
        loop {
            let mut s=REGISTRY.lock();
            if s.owner.is_none(){s.owner=Some(id);return Ok(Self);}
            if s.owner==Some(id) || crate::pe_callbacks::inside(){return Err(CUerror::NOT_PERMITTED);}
            drop(s);crate::pe_sync::wait();
        }
    }
}
impl Drop for Lease {fn drop(&mut self){REGISTRY.lock().owner=None;}}
unsafe fn lookup(s:&State,handle:usize)->*mut Node {let mut n=s.head;while n!=0 {if n==handle{return n as *mut Node;}n=(*(n as *const Node)).next;}std::ptr::null_mut()}
unsafe fn find(handle:CUgraphicsResource)->*mut Node {lookup(&REGISTRY.lock(),handle as usize)}
unsafe fn value(node:*mut Node)->&'static mut Resource {&mut *(*node).data.get()}
unsafe fn environment()->Result<u64,CUerror> {let mut c=Call::new(crate::interop_protocol::ENVIRONMENT);crate::pe_wrappers::submit(&mut c)?;Ok(c.args[0])}
unsafe fn current(t:&Resource)->CUresult {let mut context=CUcontext(std::ptr::null_mut());raw("cuCtxGetCurrent",&[unixlib::pack(&mut context as *mut CUcontext)])?;if context.0 as usize==t.cuda {Ok(())}else{Err(CUerror::INVALID_CONTEXT)}}
unsafe fn synchronize(stream:CUstream,ptsz:bool)->CUresult {raw(if ptsz{"cuStreamSynchronize_ptsz"}else{"cuStreamSynchronize"},&[unixlib::pack(stream)])}
unsafe fn drain(t:&mut Resource)->CUresult {if t.poisoned{return Err(CUerror::UNKNOWN);}if t.pending {raw("cuEventSynchronize",&[t.event as u64])?;t.pending=false;}Ok(())}
unsafe fn cleanup(t:&Resource) {
    if t.pixels!=0{let _=raw("cuMemFreeHost",&[t.pixels as u64]);}if t.event!=0{let _=raw("cuEventDestroy_v2",&[t.event as u64]);}
    for o in [t.ready,t.staging,t.texture,t.identity,t.multithread,t.context] {com::release(o);}
}
unsafe fn registration(t:&mut Resource,resource:usize,enable_mt:bool)->CUresult {
    let hr=com::query(resource,&com::TEXTURE,&mut t.texture);if hr<0{return Err(CUerror::NOT_SUPPORTED);}if t.texture==0{return Err(CUerror::UNKNOWN);}
    if com::query(t.texture,&com::UNKNOWN,&mut t.identity)<0 || t.identity==0{return Err(CUerror::UNKNOWN);}
    let mut desc=com::Description::default();com::description(t.texture,&mut desc);
    if desc.format!=2 || desc.mips!=1 || desc.arrays!=1 || desc.samples!=1{return Err(CUerror::NOT_SUPPORTED);}
    t.width=desc.width;t.height=desc.height;
    raw("cuCtxGetCurrent",&[unixlib::pack(&mut t.cuda as *mut usize)])?;if t.cuda==0{return Err(CUerror::INVALID_CONTEXT);}
    let bytes=(t.width as usize).checked_mul(t.height as usize).and_then(|n|n.checked_mul(16)).filter(|n|*n!=0).ok_or(CUerror::INVALID_VALUE)?;
    raw("cuMemHostAlloc",&[unixlib::pack(&mut t.pixels as *mut usize),bytes as u64,0])?;
    raw("cuEventCreate",&[unixlib::pack(&mut t.event as *mut usize),2])?;
    let mut device=0usize;com::device(resource,&mut device);if device==0{return Err(CUerror::UNKNOWN);}
    let result=(||{
        if com::creation(device)&1!=0{return Err(CUerror::NOT_SUPPORTED);}
        com::immediate(device,&mut t.context);if t.context==0{return Err(CUerror::UNKNOWN);}
        let hr=com::query(device,&com::MULTITHREAD,&mut t.multithread);
        if hr<0{return Err(CUerror::NOT_SUPPORTED);}if t.multithread==0{return Err(CUerror::UNKNOWN);}
        if !com::protected(t.multithread) && enable_mt{com::protect(t.multithread);}
        if !com::protected(t.multithread){return Err(CUerror::NOT_SUPPORTED);}
        desc.usage=3;desc.bind=0;desc.misc=0;desc.cpu=0x30000;
        if com::texture(device,&desc,&mut t.staging)<0 || t.staging==0{return Err(CUerror::UNKNOWN);}
        if com::create_query(device,&mut t.ready)<0 || t.ready==0{return Err(CUerror::UNKNOWN);}Ok(())
    })();com::release(device);result?;
    let descriptor=CUDA_ARRAY_DESCRIPTOR{Width:t.width as usize,Height:t.height as usize,Format:CUarray_format(0x20),NumChannels:4};
    // Failed array creation does not establish array ownership.
    let mut array=0usize;raw("cuArrayCreate_v2",&[unixlib::pack(&mut array as *mut usize),unixlib::pack(&descriptor as *const CUDA_ARRAY_DESCRIPTOR)])?;t.array=array;Ok(())
}
#[no_mangle]
pub unsafe extern "system" fn cuGraphicsD3D11RegisterResource(out:*mut CUgraphicsResource,resource:*mut ID3D11Resource,flags:u32)->CUresult {
    let env=environment()?;if env&crate::interop_protocol::COPY==0{return Err(if flags!=0{CUerror::INVALID_VALUE}else{CUerror::UNKNOWN});}
    if out.is_null() || resource.is_null(){return Err(CUerror::INVALID_VALUE);}if flags!=0{return Err(CUerror::NOT_SUPPORTED);}let _lease=Lease::acquire()?;
    for name in [b"cuMemHostAlloc\0".as_slice(),b"cuMemFreeHost\0",b"cuMemcpy2DAsync_v2\0",b"cuMemcpy2DAsync_v2_ptsz\0",b"cuEventCreate\0",b"cuEventRecord\0",b"cuEventRecord_ptsz\0",b"cuEventSynchronize\0",b"cuEventDestroy_v2\0"] {available(name)?;}
    let mut t=Resource::default();if let Err(e)=registration(&mut t,resource as usize,env&crate::interop_protocol::ENABLE_MT!=0){cleanup(&t);return Err(e);}
    let duplicate={let state=REGISTRY.lock();let mut p=state.head;let mut duplicate=false;while p!=0{if value(p as *mut Node).identity==t.identity{duplicate=true;break;}p=(*(p as *const Node)).next;}duplicate};
    if duplicate {let r=raw("cuArrayDestroy",&[t.array as u64]);cleanup(&t);return r.and(Err(CUerror::INVALID_HANDLE));}
    let node=std::alloc::alloc(std::alloc::Layout::new::<Node>()).cast::<Node>();if node.is_null(){let _=raw("cuArrayDestroy",&[t.array as u64]);cleanup(&t);return Err(CUerror::OUT_OF_MEMORY);}
    let mut state=REGISTRY.lock();node.write(Node{next:state.head,data:UnsafeCell::new(t)});state.head=node as usize;ACTIVE.store(true,Ordering::Release);*out=node.cast();Ok(())
}
unsafe fn protected<T>(t:&Resource,f:impl FnOnce()->T)->T {com::enter(t.multithread);let r=f();com::leave(t.multithread);r}
unsafe fn snapshot(t:&Resource,input:bool){protected(t,||com::copy(t.context,if input{t.staging}else{t.texture},if input{t.texture}else{t.staging}));}
unsafe fn stage(t:&Resource,input:bool)->Result<com::Mapped,CUerror> {
    let start=os::GetTickCount64();let mut attempt=0;
    loop {let mut mapped=com::Mapped::default();let hr=protected(t,||com::map(t.context,t.staging,input,&mut mapped));
        if hr as u32!=0x887a000a {if hr<0{return Err(CUerror::UNKNOWN);}if mapped.data.is_null() || (mapped.row as usize)<t.width as usize*16 {unstage(t);return Err(CUerror::UNKNOWN);}return Ok(mapped);}
        if os::GetTickCount64().wrapping_sub(start)>=30000{return Err(CUerror::UNKNOWN);}os::Sleep(if attempt<4{0}else{1});attempt+=1;
    }
}
unsafe fn unstage(t:&Resource){protected(t,||com::unmap(t.context,t.staging));}
unsafe fn transfer(t:&Resource,stream:CUstream,ptsz:bool,input:bool)->CUresult {
    #[repr(align(8))] struct CopyBytes([u8;128]);
    let mut storage=CopyBytes([0;128]);let copy=&mut *storage.0.as_mut_ptr().cast::<CUDA_MEMCPY2D>();copy.WidthInBytes=t.width as usize*16;copy.Height=t.height as usize;
    if input {copy.srcMemoryType=CUmemorytype(1);copy.srcHost=t.pixels as *const c_void;copy.srcPitch=copy.WidthInBytes;copy.dstMemoryType=CUmemorytype(3);copy.dstArray=t.array as CUarray;}
    else {copy.srcMemoryType=CUmemorytype(3);copy.srcArray=t.array as CUarray;copy.dstMemoryType=CUmemorytype(1);copy.dstHost=t.pixels as *mut c_void;copy.dstPitch=copy.WidthInBytes;}
    raw(if ptsz{"cuMemcpy2DAsync_v2_ptsz"}else{"cuMemcpy2DAsync_v2"},&[unixlib::pack(copy as *const CUDA_MEMCPY2D),unixlib::pack(stream)])
}
unsafe fn upload(t:&mut Resource,stream:CUstream,ptsz:bool)->CUresult {
    let mapped=stage(t,true)?;let row=t.width as usize*16;
    for y in 0..t.height as usize {std::ptr::copy_nonoverlapping(mapped.data.add(y*mapped.row as usize),(t.pixels as *mut u8).add(y*row),row);}unstage(t);
    let r=transfer(t,stream,ptsz,true).and_then(|()|raw(if ptsz{"cuEventRecord_ptsz"}else{"cuEventRecord"},&[t.event as u64,unixlib::pack(stream)]));
    if r.is_ok(){t.pending=true;}else if synchronize(stream,ptsz).is_err(){t.poisoned=true;}r
}
unsafe fn publish(t:&Resource)->CUresult {let mapped=stage(t,false)?;let row=t.width as usize*16;for y in 0..t.height as usize{std::ptr::copy_nonoverlapping((t.pixels as *const u8).add(y*row),mapped.data.add(y*mapped.row as usize),row);}unstage(t);snapshot(t,false);Ok(())}
fn needs_input(t:&Resource)->bool {t.flags!=2 || !t.primed}
unsafe fn last(count:u32,handles:*mut CUgraphicsResource,i:u32)->bool {let t=value(find(*handles.add(i as usize)));if !needs_input(t){return false;}for j in i+1..count{let u=value(find(*handles.add(j as usize)));if needs_input(u) && t.context==u.context{return false;}}true}
unsafe fn wait_batch(count:u32,handles:*mut CUgraphicsResource)->CUresult {
    for i in 0..count {if last(count,handles,i){let t=value(find(*handles.add(i as usize)));protected(t,||com::end(t.context,t.ready));protected(t,||com::flush(t.context));}}
    for i in 0..count {if last(count,handles,i){let t=value(find(*handles.add(i as usize)));let start=os::GetTickCount64();let mut attempt=0;
        loop {let hr=protected(t,||com::data(t.context,t.ready));if hr==0{break;}if hr!=1 || os::GetTickCount64().wrapping_sub(start)>=30000{return Err(CUerror::UNKNOWN);}os::Sleep(if attempt<4{0}else{1});attempt+=1;}
    }}Ok(())
}
unsafe fn batch(count:u32,handles:*mut CUgraphicsResource,stream:CUstream,unmap:bool,ptsz:bool)->Option<CUresult> {
    if !ACTIVE.load(Ordering::Acquire) || handles.is_null(){return None;}
    let found={let s=REGISTRY.lock();(0..count).filter(|i|!lookup(&s,*handles.add(*i as usize) as usize).is_null()).count()};
    if found==0{return None;}if found!=count as usize{return Some(Err(CUerror::NOT_SUPPORTED));}
    Some((||{
        let _lease=Lease::acquire()?;
        for i in 0..count {let handle=*handles.add(i as usize);for j in 0..i{if *handles.add(j as usize)==handle{return Err(CUerror::INVALID_VALUE);}}
            let node=find(handle);if node.is_null(){return Err(CUerror::INVALID_HANDLE);}let t=value(node);current(t)?;drain(t)?;if t.mapped!=unmap{return Err(if unmap{CUerror::NOT_MAPPED}else{CUerror::ALREADY_MAPPED});}
        }
        if !unmap {
            if (0..count).any(|i|needs_input(value(find(*handles.add(i as usize))))) {
                synchronize(stream,ptsz)?;
                for i in 0..count {let t=value(find(*handles.add(i as usize)));if needs_input(t){snapshot(t,true);}}
                wait_batch(count,handles)?;
                for i in 0..count {let t=value(find(*handles.add(i as usize)));if !needs_input(t){continue;}
                    if let Err(e)=upload(t,stream,ptsz){for j in 0..i{let _=drain(value(find(*handles.add(j as usize))));}return Err(e);}t.primed=true;
                }
            }
        }else {
            let mut queued=Ok(());for i in 0..count {let t=value(find(*handles.add(i as usize)));if t.flags!=1{queued=transfer(t,stream,ptsz,false);if queued.is_err(){break;}}}
            let drained=synchronize(stream,ptsz);if drained.is_err(){for i in 0..count{value(find(*handles.add(i as usize))).poisoned=true;}}queued?;drained?;
            for i in 0..count{let t=value(find(*handles.add(i as usize)));if t.flags!=1{publish(t)?;}}
        }
        for i in 0..count{value(find(*handles.add(i as usize))).mapped=!unmap;}Ok(())
    })())
}
macro_rules! batches {($name:ident,$unmap:expr,$ptsz:expr)=>{#[no_mangle] pub unsafe extern "system" fn $name(count:u32,handles:*mut CUgraphicsResource,stream:CUstream)->CUresult {batch(count,handles,stream,$unmap,$ptsz).unwrap_or_else(||raw(stringify!($name),&[u64::from(count),unixlib::pack(handles),unixlib::pack(stream)]))}};}
batches!(cuGraphicsMapResources,false,false);batches!(cuGraphicsMapResources_ptsz,false,true);batches!(cuGraphicsUnmapResources,true,false);batches!(cuGraphicsUnmapResources_ptsz,true,true);
unsafe fn resource(handle:CUgraphicsResource,operation:u32,out:*mut c_void,a:u32,b:u32)->Option<CUresult> {
    if !ACTIVE.load(Ordering::Acquire) || find(handle).is_null(){return None;}
    Some((||{let _lease=Lease::acquire()?;let node=find(handle);if node.is_null(){return Err(CUerror::INVALID_HANDLE);}let t=value(node);current(t)?;
        match operation {
            0=>{if out.is_null() || a!=0 || b!=0{return Err(CUerror::INVALID_VALUE);}if !t.mapped{return Err(CUerror::NOT_MAPPED);}out.cast::<CUarray>().write(t.array as CUarray);},
            1=>return Err(if t.mapped{CUerror::NOT_MAPPED_AS_POINTER}else{CUerror::NOT_MAPPED}),
            2=>{if t.mapped{return Err(CUerror::ALREADY_MAPPED);}if a>2{return Err(CUerror::INVALID_VALUE);}t.flags=a;},
            3=>return Err(CUerror::NOT_SUPPORTED),
            4=>{if t.mapped{return Err(CUerror::ALREADY_MAPPED);}drain(t)?;raw("cuArrayDestroy",&[t.array as u64])?;
                {let mut s=REGISTRY.lock();let mut p=&mut s.head as *mut usize;while *p!=node as usize{p=&mut (*(*p as *mut Node)).next;}*p=(*node).next;}
                cleanup(t);std::alloc::dealloc(node.cast(),std::alloc::Layout::new::<Node>());
            },_=>return Err(CUerror::INVALID_VALUE),
        }Ok(())})())
}
#[no_mangle] pub unsafe extern "system" fn cuGraphicsSubResourceGetMappedArray(out:*mut CUarray,handle:CUgraphicsResource,array:u32,mip:u32)->CUresult {resource(handle,0,out.cast(),array,mip).unwrap_or_else(||raw("cuGraphicsSubResourceGetMappedArray",&[unixlib::pack(out),unixlib::pack(handle),array as u64,mip as u64]))}
macro_rules! pointer {($name:ident,$bytes:ty,$device:ty)=>{#[no_mangle] pub unsafe extern "system" fn $name(out:*mut $device,bytes:*mut $bytes,handle:CUgraphicsResource)->CUresult {resource(handle,1,std::ptr::null_mut(),0,0).unwrap_or_else(||raw(stringify!($name),&[unixlib::pack(out),unixlib::pack(bytes),unixlib::pack(handle)]))}};}
pointer!(cuGraphicsResourceGetMappedPointer,u32,CUdeviceptr_v1);pointer!(cuGraphicsResourceGetMappedPointer_v2,usize,CUdeviceptr);
macro_rules! flags {($name:ident)=>{#[no_mangle] pub unsafe extern "system" fn $name(handle:CUgraphicsResource,flags:u32)->CUresult {resource(handle,2,std::ptr::null_mut(),flags,0).unwrap_or_else(||raw(stringify!($name),&[unixlib::pack(handle),flags as u64]))}};}
flags!(cuGraphicsResourceSetMapFlags);flags!(cuGraphicsResourceSetMapFlags_v2);
#[no_mangle] pub unsafe extern "system" fn cuGraphicsResourceGetMappedMipmappedArray(out:*mut CUmipmappedArray,handle:CUgraphicsResource)->CUresult {resource(handle,3,out.cast(),0,0).unwrap_or_else(||raw("cuGraphicsResourceGetMappedMipmappedArray",&[unixlib::pack(out),unixlib::pack(handle)]))}
#[no_mangle] pub unsafe extern "system" fn cuGraphicsUnregisterResource(handle:CUgraphicsResource)->CUresult {resource(handle,4,std::ptr::null_mut(),0,0).unwrap_or_else(||raw("cuGraphicsUnregisterResource",&[unixlib::pack(handle)]))}

const _:()={
    assert!(std::mem::size_of::<CUDA_MEMCPY2D>()==128);
    assert!(std::mem::align_of::<CUDA_MEMCPY2D>()==8);
    assert!(std::mem::size_of::<CUDA_ARRAY_DESCRIPTOR>()==24);
    assert!(std::mem::offset_of!(CUDA_MEMCPY2D,srcHost)==24);
    assert!(std::mem::offset_of!(CUDA_MEMCPY2D,dstHost)==80);
    assert!(std::mem::offset_of!(CUDA_MEMCPY2D,WidthInBytes)==112);
};

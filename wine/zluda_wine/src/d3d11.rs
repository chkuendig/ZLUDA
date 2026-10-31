// SPDX-License-Identifier: MIT OR Apache-2.0
//! COM projection from reviewed specification §15, independently authored.
use std::ffi::c_void;
#[repr(C)] pub struct Guid {a:u32,b:u16,c:u16,d:[u8;8]}
pub const TEXTURE:Guid=Guid{a:0x6f15aaf2,b:0xd208,c:0x4e89,d:[0x9a,0xb4,0x48,0x95,0x35,0xd3,0x4f,0x9c]};
pub const UNKNOWN:Guid=Guid{a:0,b:0,c:0,d:[0xc0,0,0,0,0,0,0,0x46]};
pub const MULTITHREAD:Guid=Guid{a:0x9b7e4e00,b:0x342c,c:0x4106,d:[0xa1,0x9f,0x4f,0x27,0x04,0xf6,0x89,0xf0]};
#[derive(Default)] #[repr(C)] pub struct Description {pub width:u32,pub height:u32,pub mips:u32,pub arrays:u32,pub format:u32,pub samples:u32,pub quality:u32,pub usage:u32,pub bind:u32,pub cpu:u32,pub misc:u32}
#[repr(C)] pub struct Query {pub kind:u32,pub misc:u32}
#[repr(C)] pub struct Mapped {pub data:*mut u8,pub row:u32,pub depth:u32}
impl Default for Mapped {fn default()->Self {Self{data:std::ptr::null_mut(),row:0,depth:0}}}
unsafe fn slot(object:usize,index:usize)->*const c_void {let vtable=*(object as *const *const usize);*vtable.add(index) as *const c_void}
macro_rules! invoke {($object:expr,$slot:expr,($($ty:ty),*)->$ret:ty,$($arg:expr),*)=>{{let f:unsafe extern "win64" fn(*mut c_void,$($ty),*)->$ret=std::mem::transmute(slot($object,$slot));f($object as *mut c_void,$($arg),*)}};}
pub unsafe fn query(object:usize,iid:&Guid,out:*mut usize)->i32 {invoke!(object,0,(*const Guid,*mut usize)->i32,iid,out)}
pub unsafe fn release(object:usize){if object!=0{invoke!(object,2,()->u32,);}}
pub unsafe fn device(object:usize,out:*mut usize){invoke!(object,3,(*mut usize)->(),out)}
pub unsafe fn description(object:usize,out:*mut Description){invoke!(object,10,(*mut Description)->(),out)}
pub unsafe fn creation(object:usize)->u32 {invoke!(object,38,()->u32,)}
pub unsafe fn immediate(object:usize,out:*mut usize){invoke!(object,40,(*mut usize)->(),out)}
pub unsafe fn texture(object:usize,desc:&Description,out:*mut usize)->i32 {invoke!(object,5,(*const Description,*const c_void,*mut usize)->i32,desc,std::ptr::null(),out)}
pub unsafe fn create_query(object:usize,out:*mut usize)->i32 {let desc=Query{kind:0,misc:0};invoke!(object,24,(*const Query,*mut usize)->i32,&desc,out)}
pub unsafe fn protected(object:usize)->bool {invoke!(object,6,()->i32,)!=0}
pub unsafe fn protect(object:usize){invoke!(object,5,(i32)->i32,1);}
pub unsafe fn enter(object:usize){invoke!(object,3,()->(),)}
pub unsafe fn leave(object:usize){invoke!(object,4,()->(),)}
pub unsafe fn copy(object:usize,dst:usize,src:usize){invoke!(object,47,(*mut c_void,*mut c_void)->(),dst as *mut c_void,src as *mut c_void)}
pub unsafe fn map(object:usize,resource:usize,read:bool,out:*mut Mapped)->i32 {invoke!(object,14,(*mut c_void,u32,u32,u32,*mut Mapped)->i32,resource as *mut c_void,0,if read{1}else{2},0x100000,out)}
pub unsafe fn unmap(object:usize,resource:usize){invoke!(object,15,(*mut c_void,u32)->(),resource as *mut c_void,0)}
pub unsafe fn end(object:usize,query:usize){invoke!(object,28,(*mut c_void)->(),query as *mut c_void)}
pub unsafe fn flush(object:usize){invoke!(object,111,()->(),)}
pub unsafe fn data(object:usize,query:usize)->i32 {invoke!(object,29,(*mut c_void,*mut c_void,u32,u32)->i32,query as *mut c_void,std::ptr::null_mut(),0,1)}
const _:()={assert!(std::mem::size_of::<Description>()==44);assert!(std::mem::size_of::<Mapped>()==16);assert!(std::mem::size_of::<Guid>()==16);};

const _:()={
    assert!(std::mem::align_of::<Guid>()==4);
    assert!(std::mem::offset_of!(Guid,d)==8);
    assert!(std::mem::align_of::<Description>()==4);
    assert!(std::mem::offset_of!(Description,usage)==28);
    assert!(std::mem::size_of::<Query>()==8);
    assert!(std::mem::align_of::<Query>()==4);
    assert!(std::mem::align_of::<Mapped>()==8);
    assert!(std::mem::offset_of!(Mapped,row)==8);
    assert!(std::mem::offset_of!(Mapped,depth)==12);
};

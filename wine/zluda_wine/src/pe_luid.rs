// SPDX-License-Identifier: MIT OR Apache-2.0
//! Registry selection and close order from specification §14.
use crate::{cuda::*,pe_platform as os};
#[cfg(target_arch="x86_64")] use crate::unixlib::Call;
use std::ffi::c_void;
#[repr(align(8))] struct Buffer([u8;1024]);
struct Key(*mut c_void);
impl Drop for Key {fn drop(&mut self){unsafe{os::NtClose(self.0);}}}
unsafe fn open(root:*mut c_void,name:&[u16])->Option<Key> {
    let bytes=u16::try_from(name.len().checked_mul(2)?).ok()?;let string=os::Unicode{len:bytes,max:bytes,#[cfg(target_pointer_width="64")] padding:0,buffer:name.as_ptr()};
    let mut object:os::Object=std::mem::zeroed();object.len=std::mem::size_of::<os::Object>() as u32;object.root=root;object.name=&string;
    let mut handle=std::ptr::null_mut();if os::NtOpenKeyEx(&mut handle,0x02000000,&object,0)==0 && !handle.is_null(){Some(Key(handle))}else{None}
}
unsafe fn child(key:&Key,index:u32)->Option<Vec<u16>> {
    let mut buffer=Buffer([0;1024]);let mut returned=0;
    if os::NtEnumerateKey(key.0,index,0,buffer.0.as_mut_ptr().cast(),1024,&mut returned)!=0 {return None;}
    let len=u32::from_le_bytes(buffer.0[12..16].try_into().unwrap()) as usize;if len%2!=0 || len>1008{return None;}
    Some(buffer.0[16..16+len].chunks_exact(2).map(|p|u16::from_le_bytes([p[0],p[1]])).collect())
}
unsafe fn property<const N:usize>(key:&Key,name:&str)->Option<[u8;N]> {
    let wide:Vec<_>=name.encode_utf16().collect();let property=open(key.0,&wide)?;
    let mut buffer=Buffer([0;1024]);let mut returned=0;let empty=os::Unicode{len:0,max:0,#[cfg(target_pointer_width="64")] padding:0,buffer:std::ptr::null()};
    if os::NtQueryValueKey(property.0,&empty,2,buffer.0.as_mut_ptr().cast(),1024,&mut returned)!=0 || returned as usize!=12+N{return None;}
    Some(buffer.0[12..12+N].try_into().unwrap())
}
#[cfg(target_arch="x86_64")]
#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetLuid(out:*mut c_char,node:*mut u32,device:CUdevice)->CUresult {
    if out.is_null() || node.is_null(){return Err(CUerror::INVALID_VALUE);}
    let mut uuid=CUuuid{bytes:[0;16]};let mut request=Call::new(crate::luid_protocol::UUID);request.put(0,&mut uuid as *mut CUuuid);request.put(1,device);
    crate::pe_wrappers::submit(&mut request)?;
    translate(out,node,&uuid,request.args[2]!=0)
}
// Both PE widths share the complete Windows translation and output timing.
// Callers validate null outputs and finish the native UUID request first.
pub(super) unsafe fn translate(out:*mut c_char,node:*mut u32,uuid:&CUuuid,fake:bool)->CUresult {
    if fake {let hash=uuid.bytes.iter().fold(0u64,|h,b|h.wrapping_mul(31).wrapping_add(*b as i8 as i64 as u64));std::ptr::copy_nonoverlapping(hash.to_le_bytes().as_ptr(),out.cast(),8);*node=1;return Ok(());}
    let path:Vec<_>="\\Registry\\Machine\\System\\CurrentControlSet\\Enum\\PCI".encode_utf16().collect();
    let root=open(std::ptr::null_mut(),&path).ok_or(CUerror::UNKNOWN)?;let mut found=false;let mut i=0;
    while let Some(name)=child(&root,i) {
        if let Some(vendor)=open(root.0,&name) {let mut j=0;while let Some(name)=child(&vendor,j) {
            if let Some(instance)=open(vendor.0,&name) {
                if property::<16>(&instance,"Properties\\{233A9EF3-AFC4-4ABD-B564-C32F21F1535C}\\0002")==Some(uuid.bytes) {
                    if let Some(luid)=property::<8>(&instance,"Properties\\{60B193CB-5276-4D0F-96FC-F173ABAD3EC6}\\0002") {std::ptr::copy_nonoverlapping(luid.as_ptr(),out.cast(),8);*node=1;found=true;}
                }
            }
            let Some(next)=j.checked_add(1)else{break;};j=next;
        }}
        let Some(next)=i.checked_add(1)else{break;};i=next;
    }
    if found {Ok(())}else{Err(CUerror::UNKNOWN)}
}
use std::ffi::c_char;

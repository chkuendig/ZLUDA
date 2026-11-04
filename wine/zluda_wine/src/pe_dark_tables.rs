// SPDX-License-Identifier: MIT OR Apache-2.0
//! Immutable, process-lifetime PE table publication.
use crate::{cuda::*,unixlib::Call,dark_protocol::{self,TABLES}};
use std::ffi::c_void;
use crate::pe_sync::Once as OnceLock;
static PUBLISHED:[OnceLock<Box<[usize]>>;11]=[const{OnceLock::new()};11];
pub unsafe fn obtain(index:usize)->Result<*const c_void,CUerror> {
    let spec=TABLES.get(index).ok_or(CUerror::UNKNOWN)?;
    if spec.name=="ZLUDA32_INTERNAL"{return Err(CUerror::UNKNOWN);}
    if let Some(table)=PUBLISHED[index].get(){return Ok(table.as_ptr().cast());}
    let token=index as u64+1;let mut c=Call::new(dark_protocol::GET);c.args[0]=token;
    let queried=crate::pe_wrappers::submit(&mut c);
    let local=matches!(spec.name,"HEAP_ACCESS"|"INTEGRITY_CHECK");
    let words=if local {spec.windows_words}else {
        queried?;let extent=usize::try_from(c.args[1]).map_err(|_|CUerror::NOT_SUPPORTED)?;
        if extent==0 || extent>spec.words || (extent<64 && c.args[2]>>extent!=0){return Err(CUerror::NOT_SUPPORTED);}
        let visible=extent.min(spec.windows_words);let mask=if visible==64{u64::MAX}else{(1u64<<visible)-1};
        if (spec.callable_mask & mask) & !c.args[2]!=0{return Err(CUerror::NOT_SUPPORTED);}visible
    };
    let mut table=Vec::new();table.try_reserve_exact(words).map_err(|_|CUerror::OUT_OF_MEMORY)?;table.resize(words,0usize);
    if spec.header {table[0]=words*8;}
    for (slot,entry) in table.iter_mut().enumerate().skip(usize::from(spec.header)) {
        if let Some(address)=crate::pe_dark_owned::pointer(spec.name,slot){*entry=address as usize;}
        else if spec.callable_mask & (1u64<<slot)!=0 {*entry=crate::dark_pe_thunks::pointer(token,slot).ok_or(CUerror::NOT_SUPPORTED)? as usize;}
    }
    // A published table and its PE callables are process-lifetime objects.
    if !crate::pe_platform::pin(cuGetExportTable as *const ()) {return Err(CUerror::NOT_SUPPORTED);}
    // All native calls precede publication. Losing contenders drop only their unpublished allocation.
    let _=PUBLISHED[index].set(table.into_boxed_slice());Ok(PUBLISHED[index].get().unwrap().as_ptr().cast())
}
#[no_mangle]
pub unsafe extern "system" fn cuGetExportTable(out:*mut *const c_void,uuid:*const CUuuid)->CUresult {
    if out.is_null() || uuid.is_null(){return Err(CUerror::INVALID_VALUE);}*out=std::ptr::null();let bytes=std::ptr::read_unaligned(uuid.cast::<[u8;16]>());
    if let Some(index)=TABLES.iter().position(|s|s.uuid==bytes){*out=obtain(index)?;return Ok(());}
    if let Some(index)=dark_protocol::LEGACY.iter().position(|s|s.uuid==bytes){let mut c=Call::new(dark_protocol::LEGACY_QUERY);c.args[0]=index as u64;crate::pe_wrappers::submit(&mut c)?;return Err(CUerror::NOT_SUPPORTED);}Err(CUerror::UNKNOWN)
}

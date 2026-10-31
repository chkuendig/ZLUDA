// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lifetime-pinned native table registry. Native code pointers stay here.
use crate::{cuda::*, dark_protocol::TABLES, unixlib::Call};
use std::{ffi::c_void, sync::{Arc, Mutex, OnceLock}};

struct NativeTable { address: usize, words: usize, capabilities: u64 }
static REGISTRY: OnceLock<Mutex<Vec<Option<Arc<NativeTable>>>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<Option<Arc<NativeTable>>>> {
    REGISTRY.get_or_init(|| Mutex::new((0..TABLES.len()).map(|_| None).collect()))
}

unsafe fn obtain(index: usize) -> Result<Arc<NativeTable>, CUerror> {
    let spec = TABLES.get(index).ok_or(CUerror::UNKNOWN)?;
    if let Some(record) = registry().lock().map_err(|_| CUerror::UNKNOWN)?[index].clone() {
        return Ok(record);
    }
    let function: unsafe extern "C" fn(*mut *const c_void, *const CUuuid) -> CUresult =
        std::mem::transmute(crate::unix_wrappers::symbol(b"cuGetExportTable\0")?);
    let mut address = std::ptr::null();
    function(&mut address, &CUuuid { bytes: spec.uuid })?;
    if address.is_null() { return Err(CUerror::UNKNOWN); }
    if address as usize % std::mem::align_of::<usize>() != 0 { return Err(CUerror::NOT_SUPPORTED); }
    let words = if spec.header {
        let bytes = *address.cast::<usize>();
        if bytes < 8 || bytes % 8 != 0 || bytes / 8 > spec.words { return Err(CUerror::NOT_SUPPORTED); }
        bytes / 8
    } else { spec.words };
    let mut capabilities = 0;
    for slot in usize::from(spec.header)..words {
        if *address.cast::<usize>().add(slot) != 0 { capabilities |= 1u64 << slot; }
    }
    let record = Arc::new(NativeTable { address: address as usize, words, capabilities });
    let mut entries = registry().lock().map_err(|_| CUerror::UNKNOWN)?;
    // Concurrent successful lookups publish one immutable process-lifetime
    // table. No registry lock spans a native call or a possible callback.
    Ok(entries[index].get_or_insert(record).clone())
}

pub unsafe fn get(call: &mut Call) -> Result<(), CUerror> {
    let token = call.args[0];
    let index = usize::try_from(token.checked_sub(1).ok_or(CUerror::UNKNOWN)?).map_err(|_| CUerror::UNKNOWN)?;
    let record = obtain(index)?;
    call.args[1] = record.words as u64;
    call.args[2] = record.capabilities;
    call.result = 0;
    Ok(())
}

pub unsafe fn legacy_query(call: &mut Call) -> Result<(), CUerror> {
    let index = usize::try_from(call.args[0]).map_err(|_| CUerror::INVALID_VALUE)?;
    let spec = crate::dark_protocol::LEGACY.get(index).ok_or(CUerror::INVALID_VALUE)?;
    let function: unsafe extern "C" fn(*mut *const c_void, *const CUuuid) -> CUresult =
        std::mem::transmute(crate::unix_wrappers::symbol(b"cuGetExportTable\0")?);
    let mut discarded = std::ptr::null();
    call.result = crate::unixlib::code(function(&mut discarded, &CUuuid { bytes: spec.uuid }));
    // Only the native CUDA status crosses. Unknown native layouts/code
    // addresses are never handed to a Windows caller.
    Ok(())
}

#[cfg(test)]
thread_local! {
    static TEST_SLOT: std::cell::Cell<Option<(u64, usize, usize)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub fn with_test_slot<T>(token: u64, slot: usize, address: usize, action: impl FnOnce() -> T) -> T {
    TEST_SLOT.with(|value| {
        assert!(value.replace(Some((token, slot, address))).is_none());
        struct Restore<'a>(&'a std::cell::Cell<Option<(u64, usize, usize)>>);
        impl Drop for Restore<'_> { fn drop(&mut self) { self.0.set(None); } }
        let _restore = Restore(value);
        action()
    })
}

pub unsafe fn native_slot(token: u64, slot: usize) -> Result<*mut c_void, CUerror> {
    #[cfg(test)]
    if let Some((known, known_slot, address)) = TEST_SLOT.with(|value| value.get()) {
        return if known == token && known_slot == slot && address != 0 { Ok(address as *mut c_void) }
               else { Err(CUerror::NOT_SUPPORTED) };
    }
    let index = usize::try_from(token.checked_sub(1).ok_or(CUerror::INVALID_VALUE)?)
        .map_err(|_| CUerror::INVALID_VALUE)?;
    let entries = registry().lock().map_err(|_| CUerror::UNKNOWN)?;
    let record = entries.get(index).and_then(Option::as_ref).ok_or(CUerror::NOT_INITIALIZED)?.clone();
    drop(entries);
    if slot >= record.words || record.capabilities & (1u64 << slot) == 0 {
        return Err(CUerror::NOT_SUPPORTED);
    }
    Ok(*(record.address as *const *mut c_void).add(slot))
}

include!(concat!(env!("OUT_DIR"), "/dark_unix_thunks.rs"));

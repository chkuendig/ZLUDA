// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixed-width, same-process x64 submission protocol; no native function
//! pointer crosses it. CUDA pointers refer to the same process address space.
use crate::cuda::{CUerror, CUresult};
use std::{ffi::c_void, mem::{size_of, MaybeUninit}};

include!(concat!(env!("OUT_DIR"), "/schema.rs"));
pub const MAGIC: u64 = 0x31454e4957414443;
pub const VERSION: u32 = 2;
pub const HAS_SYMBOL: u64 = 1;
pub const NOT_SUPPORTED: u32 = 801;

#[repr(C)]
pub struct Call {
    pub magic: u64,
    pub version: u32,
    pub size: u32,
    pub api_hash: u64,
    pub function: u64,
    pub result: u32,
    pub reserved: u32,
    pub args: [u64; 16],
    pub aggregate: [u8; 64],
}

impl Call {
    pub fn new(function: u64) -> Self {
        Self { magic: MAGIC, version: VERSION, size: size_of::<Self>() as u32,
               api_hash: API_HASH, function, result: NOT_SUPPORTED, reserved: 0,
               args: [0; 16], aggregate: [0; 64] }
    }
    pub fn valid(&self) -> bool {
        self.magic == MAGIC && self.version == VERSION && self.size as usize == size_of::<Self>()
            && self.api_hash == API_HASH && self.reserved == 0
    }
    // Code generation verifies a maximum of one by-value aggregate per API.
    // Its bytes travel inside the packet; no stack temporary pointer or native
    // ABI aggregate classification crosses the PE/Unix boundary.
    pub unsafe fn put<T: Copy>(&mut self, index: usize, value: T) {
        if size_of::<T>() <= 8 {
            self.args[index] = pack(value);
        } else {
            assert!(size_of::<T>() <= self.aggregate.len());
            std::ptr::copy_nonoverlapping((&value as *const T).cast::<u8>(),
                                         self.aggregate.as_mut_ptr(), size_of::<T>());
            self.args[index] = 0x4147475245474154;
        }
    }
    pub unsafe fn get<T: Copy>(&self, index: usize) -> Result<T, CUerror> {
        if size_of::<T>() <= 8 { return Ok(unpack(self.args[index])); }
        if size_of::<T>() > self.aggregate.len() || self.args[index] != 0x4147475245474154 {
            return Err(CUerror::INVALID_VALUE);
        }
        let mut result = MaybeUninit::<T>::uninit();
        std::ptr::copy_nonoverlapping(self.aggregate.as_ptr(),
                                     result.as_mut_ptr().cast::<u8>(), size_of::<T>());
        Ok(result.assume_init())
    }
}

pub const fn function_id(name: &str) -> u64 {
    let bytes = name.as_bytes();
    let mut value = 0xcbf29ce484222325u64;
    let mut i = 0;
    while i < bytes.len() {
        value = (value ^ bytes[i] as u64).wrapping_mul(0x100000001b3);
        i += 1;
    }
    value
}

// Raw bit copies for scalar/transparent inputs. By-value aggregates use the
// separate in-packet storage, and callbacks require an explicit adapter.
pub unsafe fn pack<T: Copy>(value: T) -> u64 {
    assert!(size_of::<T>() <= size_of::<u64>());
    let mut result = 0u64;
    std::ptr::copy_nonoverlapping((&value as *const T).cast::<u8>(),
                                 (&mut result as *mut u64).cast::<u8>(), size_of::<T>());
    result
}
pub unsafe fn unpack<T: Copy>(value: u64) -> T {
    assert!(size_of::<T>() <= size_of::<u64>());
    let mut result = MaybeUninit::<T>::uninit();
    std::ptr::copy_nonoverlapping((&value as *const u64).cast::<u8>(),
                                 result.as_mut_ptr().cast::<u8>(), size_of::<T>());
    result.assume_init()
}
pub fn code(value: CUresult) -> u32 {
    match value { Ok(()) => 0, Err(error) => error.0.get() }
}
pub fn result(value: u32) -> CUresult {
    match std::num::NonZeroU32::new(value) { None => Ok(()), Some(error) => Err(CUerror(error)) }
}
pub type UnixEntry = unsafe extern "C" fn(*mut c_void) -> i32;
const _: () = assert!(size_of::<Call>() == 232);
const _: () = assert!(std::mem::offset_of!(Call, args) == 40);
const _: () = assert!(size_of::<crate::cuda::CUdeviceptr>() == 8);
const _: () = assert!(size_of::<crate::cuda::CUdevice>() == 4);
const _: () = assert!(size_of::<crate::cuda::CUdevice_attribute>() == 4);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_protocol_and_type_view_mismatches_before_a_call() {
        let mut call = Call::new(function_id("cuInit"));
        assert!(call.valid());
        call.api_hash ^= 1;
        assert!(!call.valid());
        call.api_hash ^= 1;
        call.size -= 8;
        assert!(!call.valid());
    }
    #[test]
    fn preserves_signed_scalars_and_full_width_device_addresses() {
        unsafe {
            assert_eq!(unpack::<i32>(pack(-17i32)), -17);
            assert_eq!(unpack::<u64>(pack(0xfedcba9876543210u64)), 0xfedcba9876543210);
            let pointer = 0x100000001usize as *mut c_void;
            assert_eq!(unpack::<*mut c_void>(pack(pointer)), pointer);
        }
    }
    #[test]
    fn transports_ipc_handle_bytes_without_native_aggregate_abi() {
        let handle = std::array::from_fn::<_, 64, _>(|i| (i as u8).wrapping_mul(7));
        let mut call = Call::new(function_id("cuIpcOpenMemHandle"));
        unsafe {
            call.put(1, handle);
            assert_eq!(call.get::<[u8; 64]>(1).unwrap(), handle);
            call.args[1] ^= 1;
            assert_eq!(call.get::<[u8; 64]>(1), Err(CUerror::INVALID_VALUE));
        }
    }
}

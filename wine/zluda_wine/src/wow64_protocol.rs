// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixed-width PE32 submission, including a schema check before interpretation.
//! Pointers in args are zero-extended PE addresses; native handles never cross.
#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/wow64_contract.rs"));
include!(concat!(env!("OUT_DIR"), "/schema.rs"));
pub const MAGIC: u64 = 0x32454e4957414443;
pub const VERSION: u32 = 3;
// Owner protocol extension rule: append operations and increment VERSION.
// C32_DEVICE_UUID (59): args[0] = writable 16-byte CUuuid PE32 pointer;
// args[1] = signed device low 32 bits; args[2] = writable u32 fake-mode PE32
// pointer. Pointer words are zero-extended; CUDA status is Call32.result.
pub const HELLO: u32 = u32::MAX;
pub const INVALID: i32 = 1;
pub const OOM: i32 = 2;
pub const INVALID_CONTEXT: i32 = 201;
pub const INVALID_HANDLE: i32 = 400;
pub const UNSUPPORTED: i32 = 801;

#[repr(C, align(8))]
pub struct Call32 {
    pub magic: u64,
    pub version: u32,
    pub size: u32,
    pub api_hash: u64,
    pub opcode: u32,
    pub result: i32,
    pub args: [u32; 16],
}
impl Call32 {
    pub fn new(opcode: u32, args: &[u32]) -> Self {
        let mut request = Self { magic: MAGIC, version: VERSION,
            size: std::mem::size_of::<Self>() as u32, api_hash: API_HASH,
            opcode, result: UNSUPPORTED, args: [0; 16] };
        request.args[..args.len()].copy_from_slice(args);
        request
    }
    pub fn valid(&self) -> bool {
        self.magic == MAGIC && self.version == VERSION &&
            self.size as usize == std::mem::size_of::<Self>() && self.api_hash == API_HASH
    }
}
const _: () = assert!(std::mem::size_of::<Call32>() == 96);
const _: () = assert!(std::mem::offset_of!(Call32, args) == 32);
pub fn pointer<T>(value: *const T) -> u32 { value as usize as u32 }
pub fn ptr<T>(value: u32) -> *mut T { value as usize as *mut T }

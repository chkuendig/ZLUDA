// SPDX-License-Identifier: MIT OR Apache-2.0
//! Same-thread Wine user callback. The only PE address is the fixed bridge
//! entry; native CUDA still stores an opaque token, never caller code/data.
pub const INITIALIZE:u64=12;
pub const MAGIC:u64=0x3158544357414443;
#[repr(C)] pub struct Event {pub callback:u64,pub magic:u64,pub api_hash:u64,pub token:u64,pub context:u64,pub key:u64}
const _:()={
    assert!(std::mem::size_of::<Event>()==48);
    assert!(std::mem::offset_of!(Event,callback)==0);
    assert!(std::mem::offset_of!(Event,magic)==8);
    assert!(std::mem::align_of::<Event>()==8);
};

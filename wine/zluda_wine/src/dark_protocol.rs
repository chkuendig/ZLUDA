// SPDX-License-Identifier: MIT OR Apache-2.0
//! Private table tokens and separate typed PE/Unix dark-table calls.
pub const GET: u64 = 2;
pub const CALL: u64 = 3;
pub const LEGACY_QUERY: u64 = 4;
pub struct LegacySpec { pub uuid: [u8; 16] }
pub struct TableSpec {
    pub uuid: [u8; 16],
    pub words: usize,
    pub header: bool,
    pub name: &'static str,
    pub windows_words: usize,
    pub callable_mask: u64,
}
include!(concat!(env!("OUT_DIR"), "/dark_meta.rs"));

#[cfg(any(windows, test))]
pub unsafe fn submit(call: &mut crate::unixlib::Call) -> crate::cuda::CUresult {
    #[cfg(windows)] { crate::pe_wrappers::submit(call) }
    #[cfg(all(test, not(windows)))] {
        match crate::dark_unix::typed_call(call) {
            Ok(()) => crate::unixlib::result(call.result), Err(error) => Err(error)
        }
    }
}

#[cfg(any(windows, test))]
pub fn void_failure() -> ! {
    // A void API has no CUDA status channel. Never fabricate success on a
    // transport failure. Publication must have checked all callable slots.
    std::process::abort()
}

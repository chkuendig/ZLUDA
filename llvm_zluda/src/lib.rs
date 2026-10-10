#![allow(non_upper_case_globals)]

pub use llvm_sys::*;

pub mod compile;
mod ffi;
pub mod utils;

pub use ffi::*;

pub use compile::compile;

/// Legacy fork mitigations are retained for comparison, not PTX requirements.
/// Explicit `0` retires a mitigation; compiler settings must enter the cache key.
pub fn legacy_workaround_enabled(name: &str) -> bool {
    std::env::var(name).map_or(true, |value| value != "0")
}

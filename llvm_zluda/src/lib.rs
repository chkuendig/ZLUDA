#![allow(non_upper_case_globals)]

pub use llvm_sys::*;

pub mod compile;
mod ffi;
pub mod utils;

pub use ffi::*;

pub use compile::compile;

#[cfg(test)]
mod image_wrappers_test;

/// Legacy fork mitigations are retained for comparison, not PTX requirements.
/// They are retired by default (the configuration validated in games);
/// `1` re-enables one. Compiler settings must enter the cache key.
pub fn legacy_workaround_enabled(name: &str) -> bool {
    std::env::var(name).map_or(false, |value| value == "1")
}

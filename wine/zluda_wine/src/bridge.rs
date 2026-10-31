// SPDX-License-Identifier: MIT OR Apache-2.0
// Independent opt-in Wine CUDA frontend; see IMPLEMENTATION-NOTES.md.


#[cfg(not(any(target_arch = "x86_64", all(windows, target_arch = "x86"))))]
compile_error!("zluda_wine supports PE64, PE32 and a single ELF64 Unix library");
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
compile_error!("zluda_wine supports Windows x64 PE and Linux ELF64 only");

extern crate self as cuda_types;
pub enum FILE {}
include!(concat!(env!("OUT_DIR"), "/cuda_types.rs"));
#[cfg(target_arch = "x86_64")]
include!(concat!(env!("OUT_DIR"), "/dark_types.rs"));

#[cfg(target_arch = "x86_64")]
mod unixlib;
mod legacy;
#[cfg(target_arch = "x86_64")] mod luid_protocol;
#[cfg(target_arch = "x86_64")] mod interop_protocol;
#[cfg(target_arch = "x86_64")]
mod proc_address;
#[cfg(target_arch = "x86_64")]
mod dark_protocol;
#[cfg(target_arch = "x86_64")]
mod integrity_algorithm { include!(concat!(env!("OUT_DIR"), "/integrity_algorithm.rs")); }
#[cfg(target_arch = "x86_64")] mod callback_protocol;
#[cfg(target_arch = "x86_64")] mod context_protocol;
#[cfg(target_os = "linux")] mod unix_callbacks;
#[cfg(target_os = "linux")] mod unix_callback_api;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_callback_api;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_callbacks;
#[cfg(target_os = "linux")] mod dark_unix;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod dark_pe_thunks;
#[cfg(all(test, zluda_wine_oracles, target_os = "linux"))] mod cpu_fixtures;
#[cfg(all(test, zluda_wine_oracles, target_os = "linux"))] mod callback_fixtures;
#[cfg(all(test, zluda_wine_oracles, target_os = "linux"))] mod graph_status_fixtures;
#[cfg(all(test, zluda_wine_oracles, target_os = "linux"))] mod interop_fixtures;
// CPU tests also type-check the full PE sources. Only Windows builds export
// them; Windows API and module-load behavior still need the real SDK/runtime.
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_thunks;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_wrappers;
#[cfg(any(windows, all(test,zluda_wine_oracles)))] mod pe_platform;
#[cfg(any(windows, all(test,zluda_wine_oracles)))] mod pe_sync;
#[cfg(any(windows, all(test,zluda_wine_oracles)))] mod pe_luid;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_external;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod d3d11;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_interop;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_context_storage;
#[cfg(target_os = "linux")] mod unix_context_storage;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_dark_owned;
#[cfg(any(all(windows, target_arch = "x86_64"), all(test,zluda_wine_oracles)))] mod pe_dark_tables;
#[cfg(target_os = "linux")] mod unix_thunks;
#[cfg(target_os = "linux")] mod unix_wrappers;
mod wow64_protocol;
#[cfg(target_os = "linux")] mod wow64_unix;
#[cfg(any(all(windows, target_arch = "x86"), all(test,zluda_wine_oracles)))] mod wow64_pe;
#[cfg(any(all(windows, target_arch = "x86"), all(test,zluda_wine_oracles)))] mod wow64_local;
#[cfg(any(all(windows, target_arch = "x86"), all(test,zluda_wine_oracles)))]
#[allow(dead_code)]
mod wow64_stubs { include!(concat!(env!("OUT_DIR"), "/wow64_stubs.rs")); }

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reverse-call protocol: native code carries opaque registrations, never PE
//! function addresses. Borrowed native payloads live until acknowledgement.
pub const ALLOC: u64 = 5;
pub const CANCEL: u64 = 6;
pub const DEQUEUE: u64 = 7;
pub const ACK: u64 = 8;
pub const REENTER: u64 = 9;
pub const STREAM: u64 = 1;
pub const HOST: u64 = 2;
pub const OCCUPANCY: u64 = 3;
pub const DEVICE: u64 = 4;
pub const LOG: u64 = 5;
pub const COREDUMP: u64 = 6;
// Context events carry (CUcontext,key); the PE record owns value/destructor.
// Native dark put uses args[4]=token and args[5]=reserved zero, never PE data.
// Kind7 was the retired context-storage worker route. It must stay rejected;
// synchronous context destructors use the separate fixed user32 callback ABI.

pub unsafe fn submit(call: &mut crate::unixlib::Call) -> crate::cuda::CUresult {
    #[cfg(windows)] { crate::pe_wrappers::submit(call) }
    #[cfg(all(test, not(windows)))] {
        crate::unix_callbacks::service(call)?;
        crate::unixlib::result(call.result)
    }
    #[cfg(all(not(test), not(windows)))] {
        let _ = call;
        Err(crate::cuda::CUerror::NOT_SUPPORTED)
    }
}

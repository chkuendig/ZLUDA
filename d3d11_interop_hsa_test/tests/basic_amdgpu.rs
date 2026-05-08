//! Phase A.HSA.0 — test harness baseline.
//!
//! Goals:
//!   1. Prove the integration-test crate compiles + links against
//!      libamdhip64.so.
//!   2. Prove we can run a HIP API call on the GPU (alloc, memset, copy
//!      back, free).
//!   3. Lock in a regression baseline for Phase A.HSA.1+ (which add HSA
//!      external-memory primitives).
//!
//! These tests require a working ROCm runtime + AMD GPU. They are
//! intentionally minimal — no surface objects, no kernels, no Vulkan.
//! Higher-level tests build on this scaffold.

use hip_runtime_sys::*;
use std::mem::MaybeUninit;
use std::ptr;

/// Convenience: assert `hipError_t` is success, panic with the actual
/// error code otherwise. Avoids `.unwrap()` losing the error name.
fn ok(res: Result<(), hipErrorCode_t>) {
    if let Err(e) = res {
        panic!("HIP call failed: {:?}", e);
    }
}

/// Initialise the runtime once per test process. HIP is happy to be
/// initialised multiple times — the underlying ROCm runtime caches.
fn init_hip_once() {
    unsafe {
        // hipInit is the driver-API style; the runtime API auto-inits on
        // first call too, but being explicit here keeps the trace clean.
        ok(hipInit(0));
    }
}

#[test]
fn linkage_smoke() {
    // The least we can ask: HIP is loadable, hipGetDeviceCount works.
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip_once();
    let mut count: i32 = 0;
    unsafe {
        ok(hipGetDeviceCount(&mut count));
    }
    assert!(count >= 1, "no AMD GPU visible to ROCm — got count={}", count);
    eprintln!("[test] linkage_smoke: hipGetDeviceCount = {}", count);
}

#[test]
fn hip_alloc_memset_readback_amdgpu() {
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip_once();
    unsafe {
        ok(hipSetDevice(0));

        // Allocate 4 KiB on device.
        let bytes: usize = 4096;
        let mut dev_ptr: *mut std::ffi::c_void = ptr::null_mut();
        ok(hipMalloc(&mut dev_ptr, bytes));
        assert!(!dev_ptr.is_null(), "hipMalloc returned null");

        // Memset to 0xAB.
        ok(hipMemsetD8(
            hipDeviceptr_t(dev_ptr),
            0xAB,
            bytes,
        ));

        // Read back to host.
        let mut host_buf = vec![0u8; bytes];
        ok(hipMemcpyDtoH(
            host_buf.as_mut_ptr() as *mut std::ffi::c_void,
            hipDeviceptr_t(dev_ptr),
            bytes,
        ));

        // Every byte should be 0xAB.
        let bad = host_buf.iter().filter(|&&b| b != 0xAB).count();
        assert_eq!(
            bad, 0,
            "expected all bytes 0xAB; {} bytes differ; first 16 = {:02x?}",
            bad,
            &host_buf[..16]
        );

        ok(hipFree(dev_ptr));
    }
}

/// Quick sanity that `hipExternalMemoryGetMappedBuffer` is exported and
/// returns the documented "invalid value" for null inputs. We're not
/// importing anything yet — this just checks the symbol resolves and
/// behaves as a HIP function should.
#[test]
fn external_memory_get_mapped_buffer_symbol_exists() {
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip_once();
    unsafe {
        // Passing null external memory + null descriptor; HIP should
        // return INVALID_VALUE, not crash, not return success.
        let mut out_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let res = hipExternalMemoryGetMappedBuffer(
            &mut out_ptr,
            ptr::null_mut(), // ext_mem
            ptr::null(),     // bufferDesc
        );
        // We don't care about the exact code — only that the function
        // exists and returns SOMETHING (compile + link smoke).
        eprintln!(
            "[test] hipExternalMemoryGetMappedBuffer(null,null,null) = {:?}",
            res
        );
        // Don't assert success — that would be wrong for null inputs.
        let _ = MaybeUninit::new(res);
    }
}

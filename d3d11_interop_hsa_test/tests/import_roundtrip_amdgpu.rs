//! Phase A.HSA.1 — dma_buf round-trip test.
//!
//! Locks in the half of the interop path that ALREADY works:
//!   1. Allocate device memory via HIP.
//!   2. Export it as a dma_buf via HSA's `hsa_amd_portable_export_dmabuf`.
//!   3. Re-import the dma_buf via `hipImportExternalMemory` +
//!      `hipExternalMemoryGetMappedBuffer`.
//!   4. Verify the imported pointer aliases the original — bytes written
//!      via `hipMemset` on the original are visible through the imported
//!      pointer.
//!
//! This is a stable regression baseline. Phase A.HSA.2 will add the
//! hipArray-fabrication test that's the actual unsolved bit.

use hip_runtime_sys::*;
use std::os::raw::{c_int, c_void};
use std::ptr;

// Direct FFI to libhsa-runtime64.so. Just the one function we need.
// Bound at runtime via the HSA library that HIP already loads.
type HsaStatus = u32;
const HSA_STATUS_SUCCESS: HsaStatus = 0;

#[link(name = "hsa-runtime64")]
extern "C" {
    fn hsa_amd_portable_export_dmabuf(
        ptr: *const c_void,
        size: usize,
        dmabuf: *mut c_int,
        offset: *mut u64,
    ) -> HsaStatus;
}

fn ok(res: Result<(), hipErrorCode_t>) {
    if let Err(e) = res {
        panic!("HIP call failed: {:?}", e);
    }
}

fn init_hip() {
    // Don't call hipInit — it errors when called twice in the same
    // process (cargo test parallelism). hipSetDevice auto-initialises
    // the runtime on first call and is idempotent afterwards.
    unsafe {
        ok(hipSetDevice(0));
    }
}

#[test]
fn dma_buf_import_aliases_original_amdgpu() {
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip();

    const N: usize = 64 * 1024; // 64 KiB
    const PATTERN: u8 = 0xAB;

    unsafe {
        // 1. Allocate + fill with pattern.
        let mut dev_ptr_a: *mut c_void = ptr::null_mut();
        ok(hipMalloc(&mut dev_ptr_a, N));
        ok(hipMemsetD8(hipDeviceptr_t(dev_ptr_a), PATTERN, N));
        // Make sure the memset committed.
        ok(hipDeviceSynchronize());

        // 2. Export as dma_buf.
        let mut fd: c_int = -1;
        let mut offset: u64 = u64::MAX;
        let st = hsa_amd_portable_export_dmabuf(
            dev_ptr_a as *const c_void,
            N,
            &mut fd,
            &mut offset,
        );
        assert_eq!(
            st, HSA_STATUS_SUCCESS,
            "hsa_amd_portable_export_dmabuf returned {} (non-zero)", st
        );
        assert!(fd >= 0, "exported dmabuf fd is invalid: {}", fd);
        eprintln!("[test] exported dmabuf fd={} offset={}", fd, offset);

        // 3. Re-import via HIP external memory API.
        let mut handle_desc: hipExternalMemoryHandleDesc = std::mem::zeroed();
        handle_desc.type_ =
            hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueFd;
        handle_desc.handle.fd = fd;
        handle_desc.size = N as u64;
        handle_desc.flags = 0;

        let mut ext_mem: hipExternalMemory_t = ptr::null_mut();
        ok(hipImportExternalMemory(&mut ext_mem, &handle_desc));
        assert!(!ext_mem.is_null(), "hipImportExternalMemory returned null ext_mem");

        // 4. Get device pointer aliasing the import.
        let mut buffer_desc: hipExternalMemoryBufferDesc = std::mem::zeroed();
        buffer_desc.offset = offset;
        buffer_desc.size = N as u64;
        buffer_desc.flags = 0;

        let mut dev_ptr_b: *mut c_void = ptr::null_mut();
        ok(hipExternalMemoryGetMappedBuffer(
            &mut dev_ptr_b,
            ext_mem,
            &buffer_desc,
        ));
        assert!(!dev_ptr_b.is_null(), "imported dev_ptr is null");
        eprintln!("[test] dev_ptr_a={:p}, dev_ptr_b={:p}", dev_ptr_a, dev_ptr_b);

        // 5. Read back through the imported pointer.
        let mut host_buf = vec![0u8; N];
        ok(hipMemcpyDtoH(
            host_buf.as_mut_ptr() as *mut c_void,
            hipDeviceptr_t(dev_ptr_b),
            N,
        ));

        // Every byte should be PATTERN — the import aliases the original.
        let bad = host_buf.iter().filter(|&&b| b != PATTERN).count();
        assert_eq!(
            bad, 0,
            "{}/{} bytes mismatch through imported alias; first 16 = {:02x?}",
            bad,
            N,
            &host_buf[..16]
        );

        // Cleanup.
        ok(hipDestroyExternalMemory(ext_mem));
        // hipDestroyExternalMemory closes the dmabuf fd internally.
        ok(hipFree(dev_ptr_a));
    }
}

#[test]
fn imported_alias_writes_back_to_original_amdgpu() {
    // Symmetric direction: write through the imported pointer, observe
    // the bytes through the original. Confirms the alias is bidirectional
    // (necessary for the eventual surface-write → D3D11-read use case).
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip();

    const N: usize = 4096;
    const PATTERN: u8 = 0x5A;

    unsafe {
        let mut dev_a: *mut c_void = ptr::null_mut();
        ok(hipMalloc(&mut dev_a, N));
        ok(hipMemsetD8(hipDeviceptr_t(dev_a), 0, N));
        ok(hipDeviceSynchronize());

        let mut fd: c_int = -1;
        let mut offset: u64 = 0;
        let st = hsa_amd_portable_export_dmabuf(dev_a, N, &mut fd, &mut offset);
        assert_eq!(st, HSA_STATUS_SUCCESS);

        let mut handle_desc: hipExternalMemoryHandleDesc = std::mem::zeroed();
        handle_desc.type_ =
            hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueFd;
        handle_desc.handle.fd = fd;
        handle_desc.size = N as u64;
        let mut ext_mem: hipExternalMemory_t = ptr::null_mut();
        ok(hipImportExternalMemory(&mut ext_mem, &handle_desc));

        let mut buf_desc: hipExternalMemoryBufferDesc = std::mem::zeroed();
        buf_desc.offset = offset;
        buf_desc.size = N as u64;
        let mut dev_b: *mut c_void = ptr::null_mut();
        ok(hipExternalMemoryGetMappedBuffer(&mut dev_b, ext_mem, &buf_desc));

        // Write through the IMPORTED pointer.
        ok(hipMemsetD8(hipDeviceptr_t(dev_b), PATTERN, N));
        ok(hipDeviceSynchronize());

        // Read back through the ORIGINAL pointer.
        let mut host = vec![0u8; N];
        ok(hipMemcpyDtoH(host.as_mut_ptr() as *mut c_void, hipDeviceptr_t(dev_a), N));

        let bad = host.iter().filter(|&&b| b != PATTERN).count();
        assert_eq!(
            bad, 0,
            "writes through imported ptr did not appear in original; {} bytes mismatch",
            bad
        );

        ok(hipDestroyExternalMemory(ext_mem));
        ok(hipFree(dev_a));
    }
}

use crate::r#impl::{
    context,
    driver::{self},
};
use cuda_types::cuda::{CUerror, CUresult, CUresultConsts};
use hip_runtime_sys::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{mem, ptr};
use zluda_common::FromCuda;

// Global HIP allocation range tracking for pointer validation
pub(crate) static ALLOC_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
pub(crate) static ALLOC_MAX: AtomicU64 = AtomicU64::new(0);

/// Table of host allocations created via `cuMemAllocHost` / `cuMemHostAlloc`.
/// Each entry is `(host_ptr, device_ptr, size)`. The MS_4 dump matches
/// kernel arg device pointers against `[device_ptr, device_ptr + size)`
/// to identify which tracked buffer a kernel reads from, then reads
/// contents via the host pointer (no GPU ops, no ProfilingSignal race).
pub(crate) static SMALL_HOST_ALLOCS: std::sync::Mutex<Vec<(usize, usize, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// Tracker for the specific 1040000-byte cuMemHostAlloc'd buffers we believe
/// PhysX/FleX use as actor lists. Plan Phase 3a Path 1b: log first 32 bytes
/// at allocation time and periodically thereafter, then diff PASS vs FAIL
/// to identify the buffer whose first slot stays NULL on the failing path.
/// Each entry is just `host_ptr` (the size==1040000 invariant identifies
/// the buffer; use the index in this vec as a stable label across log
/// messages). Gated by `ZLUDA_HOSTBUF_SAMPLE=1`.
pub(crate) static HOSTBUF_TRACKER: std::sync::Mutex<Vec<usize>> =
    std::sync::Mutex::new(Vec::new());

pub(crate) fn hostbuf_sample_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_HOSTBUF_SAMPLE").map_or(false, |v| v == "1"))
}

unsafe fn track_hostbuf(host: *mut ::core::ffi::c_void, size: usize) {
    if !hostbuf_sample_enabled() || size != 1040000 {
        return;
    }
    if let Ok(mut t) = HOSTBUF_TRACKER.lock() {
        let idx = t.len();
        let p = host as *const u64;
        let v0 = p.read_unaligned();
        let v1 = p.add(1).read_unaligned();
        let v2 = p.add(2).read_unaligned();
        let v3 = p.add(3).read_unaligned();
        super::driver::zluda_log_err(&format!(
            "[HOSTBUF_TRACK #{:02}] ptr=0x{:x} q0=0x{:016x} q1=0x{:016x} q2=0x{:016x} q3=0x{:016x}",
            idx, host as usize, v0, v1, v2, v3
        ));
        t.push(host as usize);
    }
}

fn untrack_hostbuf(host: *mut ::core::ffi::c_void) {
    if !hostbuf_sample_enabled() {
        return;
    }
    if let Ok(mut t) = HOSTBUF_TRACKER.lock() {
        t.retain(|p| *p != host as usize);
    }
}

/// Snapshot all tracked 1040000 buffers' first 32 bytes. Call from rare
/// events (alloc completion, every Nth kernel launch). The output line is
/// one per call, with all buffers' q0 values inline so a single grep finds
/// the timeline. q0 alone usually suffices because PhysX iterates `array[0]`
/// as a pointer; q1..q3 provided for context.
pub(crate) unsafe fn sample_hostbufs(reason: &str) {
    if !hostbuf_sample_enabled() {
        return;
    }
    if let Ok(t) = HOSTBUF_TRACKER.lock() {
        if t.is_empty() {
            return;
        }
        let mut s = String::with_capacity(t.len() * 32 + 32);
        s.push_str(&format!("[HOSTBUF_SAMPLE {}] ", reason));
        for (i, &p) in t.iter().enumerate() {
            let q0 = (p as *const u64).read_unaligned();
            s.push_str(&format!("#{:02}=0x{:016x} ", i, q0));
        }
        super::driver::zluda_log_err(&s);
    }
}

fn track_host_alloc(host: *mut ::core::ffi::c_void, dev: *mut ::core::ffi::c_void, size: usize) {
    if size == 0 {
        return;
    }
    if let Ok(mut t) = SMALL_HOST_ALLOCS.lock() {
        t.push((host as usize, dev as usize, size));
    }
}

fn untrack_host_alloc(host: *mut ::core::ffi::c_void) {
    if let Ok(mut t) = SMALL_HOST_ALLOCS.lock() {
        t.retain(|(p, _, _)| *p != host as usize);
    }
}

pub(crate) fn update_alloc_range(addr: u64, size: u64) {
    ALLOC_MIN.fetch_min(addr, Ordering::Relaxed);
    ALLOC_MAX.fetch_max(addr + size, Ordering::Relaxed);
}

/// Check whether VRAM placement diagnostics are enabled. Gated behind
/// `ZLUDA_VRAM_LOG=1` because the sysfs read on every allocation has a
/// non-negligible runtime cost we don't want to pay in normal operation.
fn vram_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_VRAM_LOG").map_or(false, |v| v == "1"))
}

/// Check whether DtoH-copy diagnostics are enabled (`ZLUDA_DTOH_LOG=1`).
/// When ON, logs every cuMemcpyDtoH{,Async} with dst/src/size and the
/// first up-to-16 dwords of the destination buffer AFTER the copy
/// completes. For Mode-B-SIGSEGV investigation: PhysX reads back small
/// scalars/structs from GPU; this surfaces the value (count, range,
/// status flag) that drives downstream CPU-side allocation/iteration.
fn dtoh_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_DTOH_LOG").map_or(false, |v| v == "1"))
}

/// Format up to 16 dwords (64 bytes) at byte offset `off` from `dst_host`.
/// Caller must guarantee the slice is mapped + readable.
unsafe fn format_dwords_at(dst_host: *const u8, off: usize, n_bytes: usize) -> String {
    let n_dwords = n_bytes / 4;
    let mut s = String::with_capacity(n_dwords * 9);
    for i in 0..n_dwords {
        let v = (dst_host.add(off) as *const u32).add(i).read_unaligned();
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{:08x}", v));
    }
    s
}

/// Format dwords for DtoH logging.
///
/// - Small buffers (≤ 256 bytes): one window at offset 0, up to 16 dwords.
/// - Big buffers (> 256 bytes): four 16-byte windows at offsets 0,
///   1/4·size, 1/2·size, 3/4·size — disambiguates "all-zero" from
///   "zero-prefix only with data later in the buffer".
unsafe fn format_dtoh_dwords(dst_host: *const u8, byte_count: usize) -> String {
    if byte_count <= 256 {
        let n = byte_count.min(64);
        let dwords = format_dwords_at(dst_host, 0, n);
        return format!("[{}]", dwords);
    }
    // Big buffer: 4 sample windows × 16 bytes (4 dwords) each
    let q1 = (byte_count / 4) & !0x3;       // align to 4
    let q2 = (byte_count / 2) & !0x3;
    let q3 = (3 * byte_count / 4) & !0x3;
    let w0 = format_dwords_at(dst_host, 0, 16);
    let w1 = format_dwords_at(dst_host, q1, 16);
    let w2 = format_dwords_at(dst_host, q2, 16);
    let w3 = format_dwords_at(dst_host, q3, 16);
    format!(
        "[@0:{} @{:#x}:{} @{:#x}:{} @{:#x}:{}]",
        w0, q1, w1, q2, w2, q3, w3
    )
}

/// Read current VRAM usage from sysfs. Returns bytes used, or 0 on error
/// OR when `ZLUDA_VRAM_LOG` is unset. The gate is here (not at the call
/// sites) so callers can freely `let vram_before = vram_used_bytes();`
/// without conditional bracketing — when logging is off this is a single
/// atomic-load + early return, so the allocation hot path pays almost
/// nothing.
pub(crate) fn vram_used_bytes() -> u64 {
    if !vram_log_enabled() {
        return 0;
    }
    std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_used")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Log VRAM delta for an allocation. Returns (vram_before, vram_after).
/// No-op when `ZLUDA_VRAM_LOG` is unset.
pub(crate) fn log_vram_delta(label: &str, vram_before: u64, vram_after: u64, size: usize) {
    if !vram_log_enabled() {
        return;
    }
    let delta = vram_after as i64 - vram_before as i64;
    let placement = if delta > 1024 { "VRAM" } else { "GTT/other" };
    super::driver::zluda_log_err(&format!(
        "{} size={} KiB vram_delta={:+} KiB total_vram={} MiB ({})",
        label,
        size / 1024,
        delta / 1024,
        vram_after / (1024 * 1024),
        placement
    ));
}

struct DropGuard<F: FnMut()>(F);

impl<F: FnMut()> Drop for DropGuard<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

pub(crate) unsafe fn alloc_v2(dptr: &mut hipDeviceptr_t, bytesize: usize) -> CUresult {
    super::driver::zluda_log(&format!("[ZLUDA] cuMemAlloc_v2({} bytes)", bytesize));
    let cu_context = context::get_current_context()?;
    let context: &context::Context = FromCuda::<_, CUerror>::from_cuda(&cu_context)?;
    let vram_before = vram_used_bytes();
    // Round up to 16-byte boundary to provide headroom for vectorized over-reads.
    // CUDA kernels (e.g. PhysX radix sort) use ld.global.v4.u32 to load 4 elements
    // (16 bytes) at a time. When the element count isn't a multiple of 4, the last
    // load reads 1-3 elements past the allocation. On NVIDIA, 2MB page granularity
    // makes this safe. On AMD with 4KB pages, it can cross into unmapped memory.
    // 16 bytes covers exactly one v4.u32 vector load — the previous 256-byte pad
    // was 16x the minimum and is being bisected as a possible suspect for the
    // TurbulenceFS-zeros-counter cross-allocation corruption (see plan B.2).
    let alloc_size = (bytesize + 15) & !15;
    hipMalloc(ptr::from_mut(dptr).cast(), alloc_size)?;
    log_vram_delta("hipMalloc", vram_before, vram_used_bytes(), alloc_size);
    fill_with_zero_and_register(cu_context, context, dptr, alloc_size)?;
    Ok(())
}

unsafe fn fill_with_zero_and_register(
    cu_context: cuda_types::cuda::CUcontext,
    context: &context::Context,
    dptr: &mut hipDeviceptr_t,
    bytesize: usize,
) -> Result<(), CUerror> {
    let drop_guard = DropGuard(|| {
        unsafe { hipFree(dptr.0) }.ok();
    });
    let mut capturing_status = mem::zeroed();
    hipStreamIsCapturing(hipStream_t(ptr::null_mut()), &mut capturing_status)?;
    if capturing_status == hipStreamCaptureStatus::hipStreamCaptureStatusNone {
        hipMemsetD8Async(*dptr, 0, bytesize, context.alloc_stream)?;
        hipStreamSynchronize(context.alloc_stream)?;
    }
    super::driver::zluda_log(&format!("[ZLUDA] cuMemAlloc_v2 -> {:?} OK", dptr.0));
    super::driver::zluda_log_err(&format!("ALLOC {:?} size={}", dptr.0, bytesize));
    update_alloc_range(dptr.0 as u64, bytesize as u64);
    add_allocation(dptr.0, bytesize, cu_context)?;
    mem::forget(drop_guard);
    Ok(())
}

pub(crate) unsafe fn free_v2(dptr: hipDeviceptr_t) -> CUresult {
    super::driver::zluda_log(&format!("[ZLUDA] cuMemFree_v2({:?})", dptr.0));
    super::driver::zluda_log_err(&format!("FREE {:?}", dptr.0));
    let hip_result = hipFree(dptr.0);
    remove_allocation(dptr.0)?;
    Ok(hip_result?)
}

pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    let result = unsafe { hipMemcpyDtoH(dst_host, src_device, byte_count) };
    if result.is_ok() && dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(dst_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[DTOH/sync] dst={:p} src={:p} size={} bytes={}",
            dst_host, src_device.0, byte_count, dwords
        ));
    }
    result
}

pub(crate) fn copy_hto_d_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    if dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(src_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[HTOD/sync] dst={:p} src={:p} size={} bytes={}",
            dst_device.0, src_host, byte_count, dwords
        ));
    }
    // Log gParams writes (496 bytes) — validate texture handles and device pointers
    if byte_count == 496 {
        let data = src_host as *const u64;
        // Texture object handles at known offsets in gParams (from CollideConvexes PTX)
        let tex_offsets: &[(usize, &str)] = &[
            (40, "320:positions"),
            (41, "328:SDFTex"),
            (47, "376:grid"),
            (48, "384:convexIdx"),
            (49, "392:convexPlanes1"),
            (50, "400:convexPlanes2"),
            (51, "408:convexNormals"),
            (52, "416:convexPos"),
            (53, "424:convexRot"),
        ];
        let alloc_min = ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
        let alloc_max = ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
        let mut bad_ptrs = Vec::new();
        for &(idx, label) in tex_offsets {
            let val = unsafe { *data.add(idx) };
            // Strip top 3 bits (texture shift encoding) for range check
            let stripped = (val << 3) >> 3;
            if stripped != 0 && (stripped < alloc_min || stripped > alloc_max) {
                bad_ptrs.push(format!("{}=0x{:x}(OOB)", label, val));
            }
        }
        let num_convexes = unsafe { *(src_host as *const u32).add(32) }; // offset 128 = u32 index 32
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoD gParams dst={:p} numConvex={} bad_ptrs={}",
            dst_device.0,
            num_convexes,
            if bad_ptrs.is_empty() {
                "none".to_string()
            } else {
                bad_ptrs.join(", ")
            }
        ));
    }
    let result = unsafe { hipMemcpyHtoD(dst_device, src_host.cast_mut(), byte_count) };
    if result.is_err() {
        // hipMemcpyHtoD failed — try hipMemcpy with Default kind which
        // handles unified/host-mapped pointers correctly via the runtime.
        // NEVER fall back to raw memcpy: casting a device pointer to a CPU
        // pointer bypasses GPU page tables and can corrupt GPU memory,
        // causing surface descriptor corruption and TCP page faults.
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoD_v2 dst={:p} src={:p} size={} -> {:?}, trying hipMemcpy default",
            dst_device.0, src_host, byte_count, result
        ));
        unsafe {
            hip_runtime_sys::hipMemcpy(
                dst_device.0,
                src_host.cast_mut(),
                byte_count,
                hipMemcpyKind::hipMemcpyDefault,
            )
        }?;
    }
    Ok(())
}

pub(crate) fn copy_hto_d_v2_ptds(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    unsafe {
        hipMemcpy_spt(
            dst_device.0.cast(),
            src_host.cast_mut(),
            byte_count,
            hipMemcpyKind::hipMemcpyHostToDevice,
        )
    }
}

pub(crate) fn copy_dto_h_v2_ptds(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    let result = unsafe {
        hipMemcpy_spt(
            dst_host.cast(),
            src_device.0.cast(),
            byte_count,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        )
    };
    if result.is_ok() && dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(dst_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[DTOH/ptds] dst={:p} src={:p} size={} bytes={}",
            dst_host, src_device.0, byte_count, dwords
        ));
    }
    result
}

pub(crate) fn get_address_range_v2(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    unsafe { hipMemGetAddressRange(pbase, psize, dptr) }
}

/// Log a cuMemset* call when ZLUDA_DTOH_LOG=1. Used to track which device
/// pointers get initialized via memset (vs HtoD or kernel writes).
fn log_memset(kind: &str, dst: u64, value: u64, n: usize) {
    if dtoh_log_enabled() {
        super::driver::zluda_log(&format!(
            "[MEMSET/{}] dst=0x{:x} value=0x{:x} n={} bytes={}",
            kind, dst, value, n,
            // Approximate total bytes: D8=n, D16=n*2, D32=n*4
            match kind {
                "d8" | "d8a" => n,
                "d16" | "d16a" => n * 2,
                "d32" | "d32a" => n * 4,
                _ => n,
            }
        ));
    }
}

pub(crate) fn set_d8_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uchar, n: usize) -> hipError_t {
    log_memset("d8", dst.0 as u64, value as u64, n);
    unsafe { hipMemsetD8(dst, value, n) }
}

pub(crate) fn set_d8_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_uchar,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    log_memset("d8a", dst.0 as u64, value as u64, n);
    let _guard = super::stream::null_stream_sync(stream);
    unsafe { hipMemsetD8Async(dst, value, n, stream) }
}

pub(crate) fn set_d16_v2(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_ushort,
    n: usize,
) -> hipError_t {
    log_memset("d16", dst.0 as u64, value as u64, n);
    unsafe { hipMemsetD16(dst, value, n) }
}

pub(crate) fn set_d16_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_ushort,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    log_memset("d16a", dst.0 as u64, value as u64, n);
    let _guard = super::stream::null_stream_sync(stream);
    unsafe { hipMemsetD16Async(dst, value, n, stream) }
}

pub(crate) fn set_d32_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uint, n: usize) -> hipError_t {
    log_memset("d32", dst.0 as u64, value as u64, n);
    unsafe { hipMemsetD32(dst, value as _, n) }
}

pub(crate) fn set_d32_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_uint,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    log_memset("d32a", dst.0 as u64, value as u64, n);
    let _guard = super::stream::null_stream_sync(stream);
    unsafe { hipMemsetD32Async(dst, value as _, n, stream) }
}

pub(crate) fn get_info_v2(free: *mut usize, total: *mut usize) -> hipError_t {
    let result = unsafe { hipMemGetInfo(free, total) };
    if let Ok(()) = result {
        let hip_total = unsafe { *total };
        let hip_free = unsafe { *free };
        // Match cuDeviceTotalMem (hip_total/2). hipMemGetInfo returns
        // VRAM+GTT combined; cap to match what we report as total.
        let reported_total = hip_total / 2;
        let reported_free = hip_free.min(reported_total);
        unsafe {
            *total = reported_total;
            *free = reported_free;
        };
        super::driver::zluda_log_err(&format!(
            "cuMemGetInfo -> free={} MiB, total={} MiB (hip_free={}, hip_total={})",
            reported_free / (1024 * 1024),
            reported_total / (1024 * 1024),
            hip_free / (1024 * 1024),
            hip_total / (1024 * 1024)
        ));
    }
    result
}

pub(crate) unsafe fn free_host(ptr: *mut ::core::ffi::c_void) -> CUresult {
    let hip_result = hipFreeHost(ptr);
    untrack_host_alloc(ptr);
    untrack_hostbuf(ptr);
    remove_allocation(ptr)?;
    Ok(hip_result?)
}

pub(crate) unsafe fn alloc_host_v2(pp: &mut *mut ::core::ffi::c_void, bytesize: usize) -> CUresult {
    host_alloc(pp, bytesize, 0)
}

pub(crate) unsafe fn host_alloc(
    pp: &mut *mut ::core::ffi::c_void,
    bytesize: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    let context = context::get_current_context()?;
    // B.3 bisection: pass flags through unchanged. Previously we OR'd in
    // hipHostMallocMapped (0x02) on every host allocation, justifying it as
    // "PhysX requires it for pinned-memory spinWait." That force-mapping
    // changes GTT/VRAM placement for every host alloc and is a candidate
    // suspect for the TurbulenceFS-zeros-counter cross-allocation corruption.
    // If PhysX truly needs mapping for spinWait it will pass the flag itself
    // (or fail with a clear error pointing back here).
    let hip_flags = flags;
    // Round up for vectorized over-read headroom (same as alloc_v2 — 16 bytes
    // covers one ld.global.v4.u32 over-read; reduced from 256 bytes as part of
    // plan B.2 bisection).
    let alloc_size = (bytesize + 15) & !15;
    hipHostMalloc(pp, alloc_size, hip_flags)?;
    // Log host alloc addresses — Flex may pass these directly as kernel args
    // expecting CUDA UVA (host pointer = device pointer). On HIP, the device
    // pointer for mapped memory is different from the host pointer.
    let mut dev_ptr: *mut ::core::ffi::c_void = ptr::null_mut();
    let _ = hipHostGetDevicePointer(&mut dev_ptr, *pp, 0);
    super::driver::zluda_log_err(&format!(
        "cuMemHostAlloc size={} host={:p} dev={:p} same={}",
        bytesize,
        *pp,
        dev_ptr,
        *pp == dev_ptr
    ));
    track_host_alloc(*pp, dev_ptr, bytesize);
    track_hostbuf(*pp, bytesize);
    sample_hostbufs("post_alloc");
    add_allocation(*pp, bytesize, context)?;
    Ok(())
}

pub(crate) unsafe fn host_get_device_pointer_v2(
    pdptr: &mut hipDeviceptr_t,
    p: *mut ::core::ffi::c_void,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    if p.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // HIP equivalent of cuMemHostGetDevicePointer_v2
    hipHostGetDevicePointer(std::ptr::from_mut(pdptr).cast(), p, flags)?;
    if dtoh_log_enabled() {
        super::driver::zluda_log(&format!(
            "[HOSTGETDEV] host=0x{:x} -> dev=0x{:x} flags=0x{:x}",
            p as usize, pdptr.0 as usize, flags
        ));
    }
    Ok(())
}

fn add_allocation(
    dptr: *mut ::core::ffi::c_void,
    bytesize: usize,
    context: cuda_types::cuda::CUcontext,
) -> Result<(), CUerror> {
    let global_state = driver::global_state()?;
    let mut allocations = global_state
        .allocations
        .lock()
        .map_err(|_| CUerror::UNKNOWN)?;
    allocations.insert(dptr as usize, bytesize, context);
    Ok(())
}

fn remove_allocation(ptr: *mut std::ffi::c_void) -> Result<(), CUerror> {
    let global_state = driver::global_state()?;
    let mut allocations = global_state
        .allocations
        .lock()
        .map_err(|_| CUerror::UNKNOWN)?;
    allocations.remove(ptr as usize);
    Ok(())
}

pub(crate) unsafe fn retain_allocation_handle(
    _handle: *mut cuda_types::cuda::CUmemGenericAllocationHandle,
    _addr: *mut ::core::ffi::c_void,
) -> CUresult {
    CUresult::ERROR_NOT_SUPPORTED
}

pub(crate) unsafe fn copy_hto_d_async_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    if dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(src_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[HTOD/async] dst={:p} src={:p} size={} stream={:p} bytes={}",
            dst_device.0, src_host, byte_count, stream.0, dwords
        ));
    }
    // Log gParams writes (496 bytes = Flex solver params) — validate texture handles
    if byte_count == 496 {
        let data = src_host as *const u64;
        let alloc_min = ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
        let alloc_max = ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
        // Check texture object handles at known gParams offsets
        let tex_offsets: &[(usize, &str)] = &[
            (40, "320:pos"),
            (41, "328:sdf"),
            (47, "376:grid"),
            (48, "384:cidx"),
            (49, "392:cp1"),
            (50, "400:cp2"),
        ];
        let mut bad = Vec::new();
        for &(idx, label) in tex_offsets {
            let val = *data.add(idx);
            let stripped = (val << 3) >> 3;
            if stripped != 0 && (stripped < alloc_min || stripped > alloc_max) {
                bad.push(format!("{}=0x{:x}", label, val));
            }
        }
        let num_convex = *(src_host as *const u32).add(32); // offset 128
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoDAsync gParams dst={:p} nConvex={} bad={}",
            dst_device.0,
            num_convex,
            if bad.is_empty() {
                "none".to_string()
            } else {
                bad.join(",")
            }
        ));
    }
    let _guard = super::stream::null_stream_sync(stream);
    let result = hipMemcpyHtoDAsync(dst_device, src_host.cast_mut(), byte_count, stream);
    if result.is_err() {
        // hipMemcpyHtoDAsync failed — try hipMemcpyAsync with Default kind.
        // NEVER fall back to raw memcpy: casting a device pointer to a CPU
        // pointer bypasses GPU page tables and can corrupt GPU memory.
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoDAsync_v2 dst={:p} src={:p} size={} -> {:?}, trying hipMemcpyAsync default",
            dst_device.0, src_host, byte_count, result
        ));
        hipMemcpyAsync(
            dst_device.0,
            src_host.cast_mut(),
            byte_count,
            hipMemcpyKind::hipMemcpyDefault,
            stream,
        )?;
    }
    Ok(())
}

pub(crate) unsafe fn copy_dto_h_async_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(stream);
    let result = hipMemcpyDtoHAsync(dst_host, src_device, byte_count, stream);
    if result.is_ok() && dtoh_log_enabled() {
        // The async copy may not have completed by the time the call
        // returns. Sync the stream so the host buffer reflects what the
        // GPU actually wrote — otherwise we'd log stale pre-copy bytes.
        let _ = hipStreamSynchronize(stream);
        let dwords = format_dtoh_dwords(dst_host as *const u8, byte_count);
        super::driver::zluda_log(&format!(
            "[DTOH/async] dst={:p} src={:p} size={} stream={:p} bytes={}",
            dst_host, src_device.0, byte_count, stream.0, dwords
        ));
    }
    result
}

pub(crate) unsafe fn copy_dto_d_v2(
    dst_device: hipDeviceptr_t,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    super::function::maybe_log_creategrid_sorted_keys_copy(
        dst_device.0 as u64,
        src_device.0 as u64,
        byte_count,
        hipStream_t(std::ptr::null_mut()),
    );
    hipMemcpyDtoD(dst_device, src_device, byte_count)
}

pub(crate) unsafe fn copy_dto_d_async_v2(
    dst_device: hipDeviceptr_t,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    super::function::maybe_log_creategrid_sorted_keys_copy(
        dst_device.0 as u64,
        src_device.0 as u64,
        byte_count,
        stream,
    );
    let _guard = super::stream::null_stream_sync(stream);
    hipMemcpyDtoDAsync(dst_device, src_device, byte_count, stream)
}

pub(crate) unsafe fn copy_async(
    dst: hipDeviceptr_t,
    src: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(stream);
    hipMemcpyAsync(
        dst.0,
        src.0,
        byte_count,
        hipMemcpyKind::hipMemcpyDefault,
        stream,
    )
}

pub(crate) fn get_allocation_granularity(
    _granularity: &mut usize,
    _property: &cuda_types::cuda::CUmemAllocationProp,
    _option: cuda_types::cuda::CUmemAllocationGranularity_flags,
) -> CUresult {
    CUresult::ERROR_NOT_SUPPORTED
}

pub(crate) unsafe fn alloc_pitch_v2(
    dptr: *mut hipDeviceptr_t,
    p_pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: ::core::ffi::c_uint,
) -> CUresult {
    let vram_before = vram_used_bytes();
    hipMemAllocPitch(
        dptr.cast(),
        p_pitch,
        width_in_bytes,
        height,
        element_size_bytes,
    )?;
    let size = width_in_bytes * height;
    log_vram_delta("hipMemAllocPitch", vram_before, vram_used_bytes(), size);
    super::driver::zluda_log_err(&format!(
        "ALLOC_PITCH {:?} {}x{} elem={} ~{} KiB",
        unsafe { *dptr },
        width_in_bytes,
        height,
        element_size_bytes,
        size / 1024
    ));
    Ok(())
}

pub(crate) unsafe fn copy_2d_v2(memcpy: hip_Memcpy2D) -> CUresult {
    hipMemcpyParam2D(&memcpy)?;
    Ok(())
}

pub(crate) unsafe fn copy_2d_async_v2(
    memcpy: hip_Memcpy2D,
    stream: hipStream_t,
) -> CUresult {
    // 2026-05-09: APEX TurbulenceFS calls cuMemcpy2DAsync_v2 many
    // times per frame after we returned its hipArray sentinel. We
    // weren't implementing it; the resulting NOT_SUPPORTED was
    // (we believe) breaking the smoke data flow. The simple
    // pass-through to hipMemcpyParam2DAsync should cover linear
    // copies; if APEX uses our D3D11 sentinel as src/dst, the
    // downstream HIP would treat its `data` as cl_mem and crash —
    // we'd need an additional sentinel-aware path then.
    hipMemcpyParam2DAsync(&memcpy, stream)?;
    Ok(())
}

pub(crate) unsafe fn copy_2d_unaligned_v2(memcpy: hip_Memcpy2D) -> CUresult {
    hipDrvMemcpy2DUnaligned(&memcpy)?;
    Ok(())
}

pub(crate) unsafe fn set_d_2d32_v2(
    dst_device: hipDeviceptr_t,
    dst_pitch: usize,
    value: ::core::ffi::c_uint,
    width: usize,
    height: usize,
) -> hipError_t {
    hipMemset2D(dst_device.0, dst_pitch, value as _, width, height)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Test that cuMemcpyHtoD correctly copies data to device memory
    /// that a GPU kernel can read. Previously, a raw memcpy fallback
    /// bypassed GPU page tables and could corrupt data, causing
    /// TurbulenceFS surface descriptor corruption and TCP page faults.
    #[test]
    fn memcpy_htod_data_integrity() {
        unsafe {
            if hipInit(0).is_err() {
                return;
            } // Skip if no GPU

            // Allocate device memory
            let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            hipMalloc(&mut dev_ptr, 256).unwrap();

            // Write a known pattern via hipMemcpyHtoD
            let host_data: Vec<u64> = (0..32).map(|i| 0xDEAD_BEEF_0000_0000u64 + i).collect();
            hipMemcpyHtoD(hipDeviceptr_t(dev_ptr), host_data.as_ptr() as *mut _, 256).unwrap();

            // Read back and verify
            let mut readback = vec![0u64; 32];
            hipMemcpyDtoH(
                readback.as_mut_ptr() as *mut _,
                hipDeviceptr_t(dev_ptr),
                256,
            )
            .unwrap();

            for i in 0..32 {
                assert_eq!(
                    readback[i], host_data[i],
                    "mismatch at index {}: got 0x{:016x}, expected 0x{:016x}",
                    i, readback[i], host_data[i]
                );
            }

            hipFree(dev_ptr).unwrap();
        }
    }
}

pub(crate) unsafe fn set_d_2d32_async(
    dst_device: hipDeviceptr_t,
    dst_pitch: usize,
    value: ::core::ffi::c_uint,
    width: usize,
    height: usize,
    stream: hipStream_t,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(stream);
    hipMemset2DAsync(dst_device.0, dst_pitch, value as _, width, height, stream)
}

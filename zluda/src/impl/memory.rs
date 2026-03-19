use crate::r#impl::{
    context,
    driver::{self, global_state},
};
use cuda_types::cuda::{CUerror, CUresult, CUresultConsts};
use hip_runtime_sys::*;
use std::{mem, ptr};

pub(crate) unsafe fn alloc_v2(dptr: &mut hipDeviceptr_t, bytesize: usize) -> CUresult {
    super::driver::zluda_log(&format!("[ZLUDA] cuMemAlloc_v2({} bytes)", bytesize));
    let global = global_state()?;
    let context = context::get_current_context()?;
    hipMalloc(ptr::from_mut(dptr).cast(), bytesize)?;
    add_allocation(dptr.0, bytesize, context)?;
    let mut status = mem::zeroed();
    hipStreamIsCapturing(hipStream_t(ptr::null_mut()), &mut status)?;
    if global.should_zero_allocations
        && status == hipStreamCaptureStatus::hipStreamCaptureStatusNone
    {
        hipMemsetD8(*dptr, 0, bytesize)?;
    }
    super::driver::zluda_log(&format!("[ZLUDA] cuMemAlloc_v2 -> {:?} OK", dptr.0));
    Ok(())
}

pub(crate) unsafe fn free_v2(dptr: hipDeviceptr_t) -> CUresult {
    super::driver::zluda_log(&format!("[ZLUDA] cuMemFree_v2({:?})", dptr.0));
    let hip_result = hipFree(dptr.0);
    remove_allocation(dptr.0)?;
    Ok(hip_result?)
}

pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    unsafe { hipMemcpyDtoH(dst_host, src_device, byte_count) }
}

pub(crate) fn copy_hto_d_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    // Log gParams writes (496 bytes) to see texture handle at offset 328
    if byte_count == 496 {
        let data = src_host as *const u64;
        let tex_handle = unsafe { *data.add(41) }; // offset 328 = u64 index 41
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoD gParams? dst={:p} size={} tex_handle@328=0x{:x}",
            dst_device.0, byte_count, tex_handle
        ));
        // Dump u64 values around offset 320-352
        let vals: Vec<String> = (40..44).map(|i| {
            let v = unsafe { *data.add(i) };
            format!("[{}]=0x{:x}", i*8, v)
        }).collect();
        super::driver::zluda_log_err(&format!("  gParams dump: {}", vals.join(" ")));
    }
    let result = unsafe { hipMemcpyHtoD(dst_device, src_host.cast_mut(), byte_count) };
    if result.is_err() {
        // CUDA unified addressing allows cuMemcpyHtoD to host-mapped pointers.
        // HIP rejects these. Fallback to memcpy for host→host copies.
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoD_v2 dst={:p} src={:p} size={} -> {:?}, falling back to memcpy",
            dst_device.0, src_host, byte_count, result
        ));
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_host.cast::<u8>(),
                dst_device.0.cast::<u8>(),
                byte_count,
            );
        }
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
    unsafe {
        hipMemcpy_spt(
            dst_host.cast(),
            src_device.0.cast(),
            byte_count,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        )
    }
}

pub(crate) fn get_address_range_v2(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    unsafe { hipMemGetAddressRange(pbase, psize, dptr) }
}

pub(crate) fn set_d8_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uchar, n: usize) -> hipError_t {
    unsafe { hipMemsetD8(dst, value, n) }
}

pub(crate) fn set_d8_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_uchar,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    unsafe { hipMemsetD8Async(dst, value, n, stream) }
}

pub(crate) fn set_d16_v2(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_ushort,
    n: usize,
) -> hipError_t {
    unsafe { hipMemsetD16(dst, value, n) }
}

pub(crate) fn set_d16_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_ushort,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    unsafe { hipMemsetD16Async(dst, value, n, stream) }
}

pub(crate) fn set_d32_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uint, n: usize) -> hipError_t {
    unsafe { hipMemsetD32(dst, value as _, n) }
}

pub(crate) fn set_d32_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_uint,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    unsafe { hipMemsetD32Async(dst, value as _, n, stream) }
}

pub(crate) fn get_info_v2(free: *mut usize, total: *mut usize) -> hipError_t {
    unsafe { hipMemGetInfo(free, total) }
}

pub(crate) unsafe fn free_host(ptr: *mut ::core::ffi::c_void) -> CUresult {
    let hip_result = hipFreeHost(ptr);
    remove_allocation(ptr)?;
    Ok(hip_result?)
}

pub(crate) unsafe fn alloc_host_v2(
    pp: &mut *mut ::core::ffi::c_void,
    bytesize: usize,
) -> CUresult {
    host_alloc(pp, bytesize, 0)
}

pub(crate) unsafe fn host_alloc(
    pp: &mut *mut ::core::ffi::c_void,
    bytesize: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    let context = context::get_current_context()?;
    // In CUDA, all page-locked host memory is implicitly device-mapped.
    // In HIP, hipHostMallocMapped must be set explicitly for
    // hipHostGetDevicePointer to work. PhysX requires this for its
    // GPU→CPU signaling pattern (pinned memory spinWait).
    let hip_flags = flags | 0x02; // hipHostMallocMapped
    hipHostMalloc(pp, bytesize, hip_flags)?;
    // Log host alloc addresses — Flex may pass these directly as kernel args
    // expecting CUDA UVA (host pointer = device pointer). On HIP, the device
    // pointer for mapped memory is different from the host pointer.
    let mut dev_ptr: *mut ::core::ffi::c_void = ptr::null_mut();
    let _ = hipHostGetDevicePointer(&mut dev_ptr, *pp, 0);
    super::driver::zluda_log_err(&format!(
        "cuMemHostAlloc size={} host={:p} dev={:p} same={}",
        bytesize, *pp, dev_ptr, *pp == dev_ptr
    ));
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
    // Log gParams writes for diagnosis (496 bytes = Flex solver params)
    if byte_count == 496 {
        let data = src_host as *const u64;
        let tex_handle = unsafe { *data.add(41) }; // offset 328 = u64 index 41
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoDAsync gParams dst={:p} tex@328=0x{:x}",
            dst_device.0, tex_handle
        ));
    }
    let result = hipMemcpyHtoDAsync(dst_device, src_host.cast_mut(), byte_count, stream);
    if result.is_err() {
        // CUDA unified addressing allows cuMemcpyHtoD to host-mapped pointers
        // (e.g. from cuMemHostAlloc with hipHostMallocMapped). HIP rejects
        // these because the pointer is in host address space, not device.
        // Try hipMemcpy with hipMemcpyDefault which handles unified pointers.
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoDAsync_v2 dst={:p} src={:p} size={} -> {:?}, trying hipMemcpy default",
            dst_device.0, src_host, byte_count, result
        ));
        let result2 = hipMemcpyAsync(
            dst_device.0,
            src_host.cast_mut(),
            byte_count,
            hipMemcpyKind::hipMemcpyDefault,
            stream,
        );
        if result2.is_err() {
            // Last resort: plain memcpy (host→host)
            super::driver::zluda_log_err(&format!(
                "cuMemcpyHtoDAsync_v2 hipMemcpy also failed ({:?}), using memcpy",
                result2
            ));
            std::ptr::copy_nonoverlapping(
                src_host.cast::<u8>(),
                dst_device.0.cast::<u8>(),
                byte_count,
            );
        }
    }
    Ok(())
}

pub(crate) unsafe fn copy_dto_h_async_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemcpyDtoHAsync(dst_host, src_device, byte_count, stream)
}

pub(crate) unsafe fn copy_dto_d_async_v2(
    dst_device: hipDeviceptr_t,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemcpyDtoDAsync(dst_device, src_device, byte_count, stream)
}

pub(crate) unsafe fn copy_async(
    dst: hipDeviceptr_t,
    src: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
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
    hipMemAllocPitch(
        dptr.cast(),
        p_pitch,
        width_in_bytes,
        height,
        element_size_bytes,
    )?;
    Ok(())
}

pub(crate) unsafe fn copy_2d_v2(memcpy: hip_Memcpy2D) -> CUresult {
    hipMemcpyParam2D(&memcpy)?;
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
pub(crate) unsafe fn set_d_2d32_async(
    dst_device: hipDeviceptr_t,
    dst_pitch: usize,
    value: ::core::ffi::c_uint,
    width: usize,
    height: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemset2DAsync(dst_device.0, dst_pitch, value as _, width, height, stream)
}

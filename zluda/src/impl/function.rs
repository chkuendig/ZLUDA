use cuda_types::cuda::{CUfunction, CUfunction_attribute, CUkernel};
use hip_runtime_sys::*;
use std::mem;
use std::collections::HashMap;
use std::sync::Mutex;

static FUNC_NAMES: std::sync::LazyLock<Mutex<HashMap<usize, String>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn register_name(f: hipFunction_t, name: &str) {
    if let Ok(mut map) = FUNC_NAMES.lock() {
        map.insert(f.0 as usize, name.to_string());
    }
}

fn lookup_name(f: hipFunction_t) -> String {
    FUNC_NAMES
        .lock()
        .ok()
        .and_then(|map| map.get(&(f.0 as usize)).cloned())
        .unwrap_or_default()
}

pub(crate) struct Function {
    pub(crate) base: hipFunction_t,
    pub(crate) sm_version: u32,
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, CUfunction, E>
    for &'a Function
{
    fn from_cuda(cu_func: &'a CUfunction) -> Result<Self, E> {
        Ok(unsafe { mem::transmute(*cu_func) })
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, CUkernel, E> for &'a Function {
    fn from_cuda(cu_func: &'a CUkernel) -> Result<Self, E> {
        Ok(unsafe { mem::transmute(*cu_func) })
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, *mut CUfunction, E>
    for &'a mut &'a Function
{
    fn from_cuda(cu_func: &'a *mut CUfunction) -> Result<Self, E> {
        let cu_func = unsafe { cu_func.as_mut() }.ok_or_else(|| E::INVALID_VALUE)?;
        Ok(unsafe { mem::transmute(cu_func) })
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, *mut CUkernel, E>
    for &'a mut &'a Function
{
    fn from_cuda(cu_func: &'a *mut CUkernel) -> Result<Self, E> {
        let cu_func = unsafe { cu_func.as_mut() }.ok_or_else(|| E::INVALID_VALUE)?;
        Ok(unsafe { mem::transmute(cu_func) })
    }
}

pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: CUfunction_attribute,
    func: &Function,
) -> hipError_t {
    match cu_attrib {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION => {
            *pi = func.sm_version as i32;
            return Ok(());
        }
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => {
            *pi = 120;
            return Ok(());
        }
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE => {
            *pi = 0;
            return Ok(());
        }
        _ => {}
    }
    let result = unsafe { hipFuncGetAttribute(pi, mem::transmute(cu_attrib), func.base) };
    if let Ok(()) = result {
        if cu_attrib == CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS {
            *pi = (*pi).max(1);
        }
    }
    result
}

pub(crate) fn launch_kernel(
    f: &Function,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: hipStream_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> hipError_t {
    // Log kernel name for crash diagnosis (lightweight — no sync, no data reads)
    let name = lookup_name(f.base);
    super::driver::zluda_log_err(&format!(
        "cuLaunchKernel '{}' grid=[{},{},{}] block=[{},{},{}] shared={}",
        name, grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes));
    // Diagnostic validation removed — the hipMemcpyDtoH calls in the validators
    // trigger the ROCm ProfilingSignal pool race condition (same as the surf.rs bug).
    // The shift encoding fix (cuTexObjectCreate adds top-3-bit shift) is verified by tests.
    // CU_LAUNCH_PARAM_END=0x0, HIP_LAUNCH_PARAM_END=0x3 — translate if using extra
    let hip_launch_param_end = 0x3usize as *mut ::core::ffi::c_void;
    let cu_launch_param_buffer_pointer = 0x1usize as *mut ::core::ffi::c_void;
    let cu_launch_param_buffer_size = 0x2usize as *mut ::core::ffi::c_void;
    let result = if !extra.is_null() && kernel_params.is_null() {
        // Translate CU_LAUNCH_PARAM_END (0) -> HIP_LAUNCH_PARAM_END (3)
        // extra format: [BUFFER_POINTER, ptr, BUFFER_SIZE, size_ptr, END]
        let extra_params = unsafe { std::slice::from_raw_parts(extra, 5) };
        if extra_params[0] == cu_launch_param_buffer_pointer
            && extra_params[2] == cu_launch_param_buffer_size
        {
            let mut patched = [
                extra_params[0],
                extra_params[1],
                extra_params[2],
                extra_params[3],
                hip_launch_param_end,
            ];
            unsafe {
                hipModuleLaunchKernel(
                    f.base, grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z,
                    shared_mem_bytes, stream,
                    std::ptr::null_mut(),
                    patched.as_mut_ptr(),
                )
            }
        } else {
            unsafe {
                hipModuleLaunchKernel(
                    f.base, grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z,
                    shared_mem_bytes, stream, kernel_params, extra,
                )
            }
        }
    } else {
        unsafe {
            hipModuleLaunchKernel(
                f.base, grid_dim_x, grid_dim_y, grid_dim_z,
                block_dim_x, block_dim_y, block_dim_z,
                shared_mem_bytes, stream, kernel_params, extra,
            )
        }
    };
    // Translate HIP error 719 (LaunchFailure) → LaunchOutOfResources.
    // On NVIDIA, launching with too many threads returns CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (701),
    // which CUDA apps handle by reducing block size and retrying. HIP returns 719 (LAUNCH_FAILURE)
    // which many apps treat as fatal. This translation makes ZLUDA match NVIDIA behavior.
    let result = match result {
        Err(code) if code == hipErrorCode_t::LaunchFailure => {
            hipError_t::ErrorLaunchOutOfResources
        }
        other => other,
    };
    // Track last kernel name for SIGABRT handler
    if !name.is_empty() {
        super::driver::set_last_kernel(name.clone());
    }
    // Log launch failures with full context for diagnosis
    if result.is_err() {
        super::driver::zluda_log_err(&format!(
            "cuLaunchKernel FAILED '{}' func={:p} grid=[{},{},{}] block=[{},{},{}] shared={} -> {:?}",
            name, f.base.0, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, result));
    }
    // ZLUDA_SYNC_LAUNCHES=1: sync after every kernel to find the exact faulting one.
    // This serializes GPU execution and is slow, but pinpoints the crash.
    if result.is_ok() {
        static SYNC_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sync_all = *SYNC_ALL.get_or_init(|| {
            std::env::var("ZLUDA_SYNC_LAUNCHES").map_or(false, |v| v == "1")
        });
        if sync_all {
            static LAUNCH_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = LAUNCH_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let sync_result = unsafe { hip_runtime_sys::hipStreamSynchronize(stream) };
            if sync_result.is_err() {
                super::driver::zluda_log_err(&format!(
                    "*** GPU_FAULT #{} '{}' grid=[{},{},{}] block=[{},{},{}] sync={:?} stream={:?}",
                    n, name, grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z, sync_result, stream));
            }
        }
    }
    result
}

// _ptsz = per-thread default stream variant of cuLaunchKernel.
// cuGetProcAddress returns this for cuLaunchKernel with certain version/flags.
// Behavior is identical to launch_kernel — the stream difference is already
// handled by HIP's stream semantics.
pub(crate) fn launch_kernel_ptsz(
    f: &Function,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: hipStream_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> hipError_t {
    launch_kernel(
        f, grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z,
        shared_mem_bytes, stream, kernel_params, extra,
    )
}

pub(crate) unsafe fn set_attribute(
    func: &Function,
    attribute: CUfunction_attribute,
    value: i32,
) -> hipError_t {
    match attribute {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => {
            return hipError_t::ErrorNotSupported;
        }
        _ => {}
    }
    hipFuncSetAttribute(func.base.0.cast(), hipFuncAttribute(attribute.0), value)
}

/// Pre-launch diagnostic for ParticleDensityAtomicToTexKernel.
/// Reads the surface handle from the 456-byte param struct (at offset 368)
/// and validates the surface descriptor it points to.
unsafe fn validate_surface_kernel_inputs(
    extra: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
    name: &str,
) {
    use hip_runtime_sys::*;

    let _ = hipStreamSynchronize(stream);

    // extra format: [CU_LAUNCH_PARAM_BUFFER_POINTER, ptr, CU_LAUNCH_PARAM_BUFFER_SIZE, size_ptr, END]
    let extra_params = std::slice::from_raw_parts(extra, 5);
    if extra_params[0] != (0x1usize as *mut std::ffi::c_void) {
        return; // Not the expected extra format
    }
    let buffer = extra_params[1] as *const u8;
    let buf_size = *(extra_params[3] as *const usize);

    if buf_size < 376 {
        return; // Too small to contain the surface handle at offset 368
    }

    // Surface handle is a u64 at offset 368 in the param struct
    let surf_handle = *(buffer.add(368) as *const u64);

    // Decode: top 3 bits = pixel shift, remaining bits = descriptor pointer
    let shift = surf_handle >> 61;
    let desc_ptr = (surf_handle << 3) >> 3;

    // Read the 32-byte surface descriptor from device memory
    let mut desc_words = [0u32; 8];
    let copy_result = hipMemcpyDtoH(
        desc_words.as_mut_ptr() as _,
        hipDeviceptr_t(desc_ptr as *mut std::ffi::c_void),
        32,
    );

    static SURF_CALL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = SURF_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE_SURF {} #{}: handle=0x{:016x} desc_ptr=0x{:016x} shift={} READ FAILED {:?}",
            name, call_n, surf_handle, desc_ptr, shift, copy_result));
        return;
    }

    // Descriptor word 0-1: base address (48 bits)
    let base = (desc_words[0] as u64) | ((desc_words[1] as u64) << 32);
    // Check if base address looks valid (in device memory range)
    let alloc_min = super::memory::ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
    let alloc_max = super::memory::ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
    let base_in_range = base >= alloc_min && base <= alloc_max;

    // Log first 5, then every 10th, or if base looks bad
    if call_n < 5 || call_n % 10 == 0 || !base_in_range {
        super::driver::zluda_log_err(&format!(
            "VALIDATE_SURF {} #{}: handle=0x{:016x} shift={} base=0x{:012x} in_range={} words=[{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}]",
            name, call_n, surf_handle, shift, base, base_in_range,
            desc_words[0], desc_words[1], desc_words[2], desc_words[3],
            desc_words[4], desc_words[5], desc_words[6], desc_words[7]));
    }

    if !base_in_range && base != 0 {
        super::driver::zluda_log_err(&format!(
            "*** CORRUPT SURFACE DESCRIPTOR #{}: base=0x{:012x} outside alloc range [{:012x}..{:012x}]",
            call_n, base, alloc_min, alloc_max));
    }
}

/// Pre-launch diagnostic for CreateParticleShapesIncremental_MultiStage_4.
/// Reads back the flags array (param_2) and index array (param_5) from device memory
/// and checks for out-of-range indices or corrupt flags that would cause wild scatter writes.
///
/// MultiStage_4 kernel params (via kernel_params pointer array):
///   [0] param_0: u64 (unused)
///   [1] param_1: u16 (unused)
///   [2] param_2: u64 → flags buffer (u16 per particle, stride 4 bytes)
///   [3] param_3: u32 → numParticles
///   [4] param_4: u64 → output scatter buffer
///   [5] param_5: u64 → index array (u32 per particle)
///   [6] param_6: u64 → counter base array
///   [7] param_7: u64 → atomic counter array
unsafe fn validate_multistage4_inputs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
    name: &str,
) {
    use hip_runtime_sys::*;

    // Sync the stream first so all prior writes are visible
    let _ = hipStreamSynchronize(stream);

    let params = std::slice::from_raw_parts(kernel_params, 8);
    let flags_ptr = *(params[2] as *const u64) as *mut std::ffi::c_void;
    let num_particles = *(params[3] as *const u32);
    let index_ptr = *(params[5] as *const u64) as *mut std::ffi::c_void;

    if num_particles == 0 || num_particles > 100_000 {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {}: numParticles={} (suspicious)", name, num_particles));
        return;
    }

    let n = num_particles as usize;

    // Read flags (u16 per entry, but stride is 4 bytes based on PTX: mul.wide.u32 %rd8, %r1, 4)
    // So flags[i] is at offset i*4, reading 2 bytes
    let flags_byte_size = n * 4; // 4 bytes stride per entry
    let mut h_flags = vec![0u8; flags_byte_size];
    let flags_result = hipMemcpyDtoH(
        h_flags.as_mut_ptr() as _,
        hipDeviceptr_t(flags_ptr),
        flags_byte_size,
    );
    if flags_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {}: flags read FAILED {:?} ptr={:?} size={}",
            name, flags_result, flags_ptr, flags_byte_size));
        return;
    }

    // Read index array (u32 per entry, stride 4 bytes)
    let mut h_indices = vec![0u32; n];
    let idx_result = hipMemcpyDtoH(
        h_indices.as_mut_ptr() as _,
        hipDeviceptr_t(index_ptr),
        n * 4,
    );
    if idx_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {}: index read FAILED {:?} ptr={:?} size={}",
            name, idx_result, index_ptr, n * 4));
        return;
    }

    // Check: which flags indicate "process this particle"?
    // Stage_4 checks: (flags[i] & 0x11) == 1
    let mut active_count = 0u32;
    let mut bad_indices = Vec::new();
    let mut max_index = 0u32;

    for i in 0..n {
        // Flag is u16 at offset i*4
        let flag = u16::from_ne_bytes([h_flags[i * 4], h_flags[i * 4 + 1]]);
        let is_active = (flag & 0x11) == 1;
        if is_active {
            active_count += 1;
            let idx = h_indices[i];
            if idx > max_index {
                max_index = idx;
            }
            // Check for obviously bad indices (> 1M or bit 31 set)
            if idx >= 1_000_000 || (idx & 0x80000000) != 0 {
                bad_indices.push((i, idx, flag));
            }
        }
    }

    static CALL_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Log summary (always for first 10, then every 10th, or if bad indices found)
    if call_n < 10 || call_n % 10 == 0 || !bad_indices.is_empty() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: n={} active={} max_idx={} bad={}",
            name, call_n, n, active_count, max_index, bad_indices.len()));
    }

    if !bad_indices.is_empty() {
        let show = bad_indices.len().min(10);
        for &(i, idx, flag) in &bad_indices[..show] {
            super::driver::zluda_log_err(&format!(
                "  BAD [{}]: flag=0x{:04x} index=0x{:08x} ({})",
                i, flag, idx, idx as i32));
        }
        if bad_indices.len() > 10 {
            super::driver::zluda_log_err(&format!(
                "  ... and {} more bad indices", bad_indices.len() - 10));
        }
    }
}

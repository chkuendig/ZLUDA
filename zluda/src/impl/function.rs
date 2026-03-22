use cuda_types::cuda::CUfunction_attribute;
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

pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: CUfunction_attribute,
    func: hipFunction_t,
) -> hipError_t {
    // TODO: implement HIP_FUNC_ATTRIBUTE_PTX_VERSION
    // TODO: implement HIP_FUNC_ATTRIBUTE_BINARY_VERSION
    match cu_attrib {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => {
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
    let result = unsafe { hipFuncGetAttribute(pi, mem::transmute(cu_attrib), func) };
    if let Ok(()) = result {
        if cu_attrib == CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS {
            *pi = (*pi).max(1);
        }
    }
    result
}

pub(crate) fn launch_kernel(
    f: hipFunction_t,
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
                    f, grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z,
                    shared_mem_bytes, stream,
                    std::ptr::null_mut(),
                    patched.as_mut_ptr(),
                )
            }
        } else {
            unsafe {
                hipModuleLaunchKernel(
                    f, grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z,
                    shared_mem_bytes, stream, kernel_params, extra,
                )
            }
        }
    } else {
        unsafe {
            hipModuleLaunchKernel(
                f, grid_dim_x, grid_dim_y, grid_dim_z,
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
    // Log launch failures with full context for diagnosis
    if result.is_err() {
        let name = lookup_name(f);
        super::driver::zluda_log_err(&format!(
            "cuLaunchKernel FAILED '{}' func={:p} grid=[{},{},{}] block=[{},{},{}] shared={} -> {:?}",
            name, f.0, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, result));
    }
    // ZLUDA_SYNC_LAUNCHES: synchronize after every kernel launch to pinpoint GPU faults.
    // When enabled, any GPU fault will be reported immediately after the faulting kernel
    // instead of being detected asynchronously later.
    if result.is_ok() {
        static SYNC_LAUNCHES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sync = *SYNC_LAUNCHES.get_or_init(|| {
            std::env::var("ZLUDA_SYNC_LAUNCHES").map_or(false, |v| v == "1")
        });
        if sync {
            let name = lookup_name(f);
            // Log before sync — HSA may kill the process during hipStreamSynchronize on fault
            super::driver::zluda_log_err(&format!(
                "SYNC '{}' func={:p} grid=[{},{},{}] block=[{},{},{}] shared={}",
                name, f.0, grid_dim_x, grid_dim_y, grid_dim_z,
                block_dim_x, block_dim_y, block_dim_z,
                shared_mem_bytes));
            let sync_result = unsafe { hipStreamSynchronize(stream) };
            if sync_result.is_err() {
                super::driver::zluda_log_err(&format!(
                    "SYNC FAULT '{}' func={:p} -> {:?}",
                    name, f.0, sync_result));
                return sync_result;
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
    f: hipFunction_t,
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
    func: hipFunction_t,
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
    hipFuncSetAttribute(func.0.cast(), hipFuncAttribute(attribute.0), value)
}

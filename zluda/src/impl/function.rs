use cuda_types::cuda::CUfunction_attribute;
use hip_runtime_sys::*;
use std::mem;

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
            super::driver::zluda_log(&format!("[ZLUDA] cuFuncGetAttribute({:?}) -> {} (hardcoded)", cu_attrib, *pi));
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
    // Always log cuFuncGetAttribute to track what PhysX sees
    super::driver::zluda_log_err(&format!(
        "cuFuncGetAttribute({:?}, func={:?}) -> val={}, result={:?}",
        cu_attrib, func, *pi, result
    ));
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
    // Skip ALL kernels through index N to find faulting kernel.
    // If skipping 0-N prevents fault, kernel N+1 is the culprit.
    // Set to 99 to disable. Start with 5 (skip all Flex, run only sort).
    const SKIP_THROUGH: u32 = 99; // Skip ALL kernels
    static LAUNCH_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = LAUNCH_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n <= SKIP_THROUGH {
        super::driver::zluda_log_err(&format!("SKIPPING kernel #{}", n));
        return Ok(());
    }
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
    // Log kernel launches with param values for crash diagnosis
    {
        let mut param_info = String::new();
        if !extra.is_null() && kernel_params.is_null() {
            let extra_params = unsafe { std::slice::from_raw_parts(extra, 5) };
            if extra_params[0] as usize == 1 && extra_params[2] as usize == 2 {
                let buf = extra_params[1] as *const u64;
                let size = unsafe { *(extra_params[3] as *const usize) };
                let num_vals = (size / 8).min(20);
                let has_null = (0..num_vals).any(|i| unsafe { *buf.add(i) } == 0);
                if has_null {
                    let ptrs: Vec<String> = (0..num_vals)
                        .map(|i| {
                            let val = unsafe { *buf.add(i) };
                            if val == 0 { "NULL".to_string() } else { format!("0x{:x}", val) }
                        })
                        .collect();
                    param_info = format!(" NULLS_IN_PARAMS[{}B]=[{}]", size, ptrs.join(","));
                }
            }
        }
        super::driver::zluda_log_err(&format!(
            "launch func={:p} grid=[{},{},{}] block=[{},{},{}] shared={}{}-> {:?}",
            f.0, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, param_info, result));
    }
    result
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

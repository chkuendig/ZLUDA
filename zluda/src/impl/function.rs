use cuda_types::cuda::CUfunction_attribute;
use hip_runtime_sys::*;
use std::mem;
use std::sync::OnceLock;

/// Dummy device buffer (zeroed) used to replace NULL pointers in kernel params.
/// On NVIDIA, NULL device pointers in kernel params are handled gracefully
/// (trap page or similar). On AMD/ROCm, address 0 is unmapped and causes a
/// GPU page fault. We replace NULLs with this dummy buffer so kernels that
/// conditionally access optional buffers (like Flex's velocities) read zeros
/// instead of faulting.
static NULL_GUARD_BUFFER: OnceLock<u64> = OnceLock::new();

/// Get (or lazily allocate) a device pointer to a zeroed guard buffer.
fn get_null_guard_ptr() -> u64 {
    *NULL_GUARD_BUFFER.get_or_init(|| {
        let size = 64 * 1024; // 64KB — large enough for any NULL-accessed buffer
        let mut ptr: *mut ::core::ffi::c_void = std::ptr::null_mut();
        unsafe {
            if hipMalloc(&mut ptr, size).is_ok() && !ptr.is_null() {
                let _ = hipMemsetD8(hipDeviceptr_t(ptr), 0, size);
                return ptr as u64;
            }
        }
        0 // If allocation fails, can't do anything — will still fault
    })
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
            // Replace NULL device pointers in the param buffer with a guard buffer.
            // NVIDIA GPUs handle NULL gracefully (trap page); AMD faults.
            // See: Flex Predict() param 4, SolveContacts() param 7 pass NULL
            // for optional buffers that the kernel conditionally accesses.
            let buf = extra_params[1] as *mut u64;
            let size = unsafe { *(extra_params[3] as *const usize) };
            let guard = get_null_guard_ptr();
            if guard != 0 {
                let num_slots = size / 8;
                for i in 0..num_slots {
                    let val = unsafe { *buf.add(i) };
                    if val == 0 {
                        unsafe { *buf.add(i) = guard; }
                    }
                }
            }
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
    // Log all kernel launches with function handle, name, and param pointers
    {
        let name = super::module::get_kernel_name(f)
            .unwrap_or_else(|| format!("unknown@{:p}", f.0));
        // Dump kernel parameter pointers to find NULLs causing GPU page faults
        let mut param_info = String::new();
        unsafe {
            if !kernel_params.is_null() {
                // kernel_params is array of pointers to individual params
                // Each param is a pointer to the actual value
                // For device pointers, the value itself is a u64 device address
                let mut ptrs = Vec::new();
                let mut i = 0;
                while i < 20 {
                    let param_ptr = *kernel_params.add(i);
                    if param_ptr.is_null() { break; }
                    let val = *(param_ptr as *const u64);
                    ptrs.push(if val == 0 { format!("NULL") } else { format!("0x{:x}", val) });
                    i += 1;
                }
                param_info = format!(" kparams=[{}]", ptrs.join(","));
            } else if !extra.is_null() {
                // extra format: [BUFFER_POINTER(1), ptr, BUFFER_SIZE(2), size_ptr, END(0)]
                let tag = *extra as usize;
                if tag == 1 {
                    let buf = *extra.add(1) as *const u64;
                    let size = *(*extra.add(3) as *const usize);
                    let num_vals = (size / 8).min(16);
                    let ptrs: Vec<String> = (0..num_vals)
                        .map(|i| {
                            let val = *buf.add(i);
                            if val == 0 { format!("NULL") } else { format!("0x{:x}", val) }
                        })
                        .collect();
                    param_info = format!(" extra[{}B]=[{}]", size, ptrs.join(","));
                }
            }
        }
        super::driver::zluda_log_err(&format!(
            "launch func={:p} '{}' grid=[{},{},{}] block=[{},{},{}] shared={}{} -> {:?}",
            f.0, name,
            grid_dim_x, grid_dim_y, grid_dim_z,
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

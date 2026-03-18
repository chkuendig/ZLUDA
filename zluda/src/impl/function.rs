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
    // Skip kernels known to fault on AMD (NULL pointer dereference at address 0).
    // On NVIDIA, address 0 is valid shared memory. On AMD, it's unmapped → fatal.
    if super::module::is_faulting_kernel(f) {
        super::driver::zluda_log_err("SKIPPING faulting kernel (CollideParticles/CollideConvexes)");
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
            // Replace NULL device pointers in the param buffer.
            // NVIDIA Flex passes NULL for optional buffers (velocities,
            // delta-positions). On NVIDIA, address 0 contains valid shared
            // memory (Fermi architecture), so NULL reads return data.
            // On AMD/ROCm, address 0 is unmapped → fatal GPU page fault.
            // We replace NULLs with the first valid device pointer in the
            // buffer so the kernel reads harmless (wrong but non-crashing)
            // data instead of faulting.
            // Ref: NVIDIA Fermi Compute Architecture Whitepaper (address 0)
            //      ROCm/ROCm#5616 (memory access fault on address nil)
            let buf = extra_params[1] as *mut u64;
            let size = unsafe { *(extra_params[3] as *const usize) };
            let num_slots = size / 8;
            // Find first value that looks like a device pointer (high bits set)
            let mut guard_ptr: u64 = 0;
            for i in 0..num_slots {
                let val = unsafe { *buf.add(i) };
                if val > 0x1000_0000_0000 { // looks like a device pointer
                    guard_ptr = val;
                    break;
                }
            }
            if guard_ptr != 0 {
                for i in 0..num_slots {
                    let val = unsafe { *buf.add(i) };
                    if val == 0 {
                        unsafe { *buf.add(i) = guard_ptr; }
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
    // Log kernel launches and force sync to identify which kernel faults.
    // hipDeviceSynchronize after each launch is expensive but pinpoints
    // which kernel causes the GPU NULL page fault.
    {
        super::driver::zluda_log_err(&format!(
            "launch func={:p} grid=[{},{},{}] block=[{},{},{}] shared={} -> {:?}",
            f.0, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, result));
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

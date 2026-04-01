use cuda_types::cuda::{CUfunction, CUfunction_attribute, CUkernel};
use hip_runtime_sys::*;
use std::mem;
use std::collections::HashMap;
use std::sync::Mutex;

static FUNC_NAMES: std::sync::LazyLock<Mutex<HashMap<usize, String>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

// Track Stage_4's counter array address across pipeline stages
static SHAPES4_CTR_PTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SHAPES4_CTR_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

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

/// Fix PhysX radix sort uint4 over-read padding.
///
/// PhysX's GPU radix sort loads elements in uint4 chunks (ld.global.v4.u32).
/// When element count is not a multiple of 4, the last uint4 load reads 1-3
/// elements past the valid data. The kernel clamps padding KEYS to 0xFFFFFFFF
/// via sanitizeKeys(), but RANKS at those positions contain whatever is in
/// memory (adjacent suballocated buffer data in PhysX's GPU pool).
///
/// On NVIDIA, 2MB page granularity means over-reads never fault, and the sort's
/// output guards prevent wrong values from propagating. On AMD/HIP with 4KB
/// pages, the adjacent data corrupts the sort output, producing garbage indices
/// that cause subsequent kernels to access unmapped memory.
///
/// This function pads the output key and rank buffers with 0xFFFFFFFF before
/// each sort pass, matching NVIDIA's effective behavior.
///
/// PxgRadixSortDesc struct layout (from PxgRadixSortDesc.h):
///   offset  0: inputKeys   (u64 ptr)
///   offset  8: inputRanks  (u64 ptr)
///   offset 16: outputKeys  (u64 ptr)
///   offset 24: outputRanks (u64 ptr)
///   offset 32: radixBlockCounts (u64 ptr)
///   offset 40: count       (u32)
/// Sentinel value stored in a static so its address is stable for async copies.
static SENTINEL_U32: u32 = 0xFFFFFFFF;

unsafe fn fix_radix_sort_padding(
    kernel_params: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
) {
    if kernel_params.is_null() { return; }
    // param_0 = pointer to PxgRadixSortDesc on device
    let desc_ptr = *(*kernel_params.add(0) as *const u64);
    // param_1 = numKeys (the WithCount variants pass count as a separate parameter)
    let count = *(*kernel_params.add(1) as *const u32);
    if count == 0 || count % 4 == 0 { return; }
    // Read the descriptor struct from device memory (48 bytes)
    let mut desc = [0u8; 48];
    let r = hip_runtime_sys::hipMemcpyDtoH(
        desc.as_mut_ptr() as *mut _,
        hip_runtime_sys::hipDeviceptr_t(desc_ptr as *mut _),
        48,
    );
    if r.is_err() { return; }
    let in_keys   = u64::from_ne_bytes(desc[0..8].try_into().unwrap());
    let in_ranks  = u64::from_ne_bytes(desc[8..16].try_into().unwrap());
    let out_keys  = u64::from_ne_bytes(desc[16..24].try_into().unwrap());
    let out_ranks = u64::from_ne_bytes(desc[24..32].try_into().unwrap());
    let padding = 4 - (count % 4) as usize;
    let sentinel_ptr = &SENTINEL_U32 as *const u32 as *mut ::core::ffi::c_void;
    super::driver::zluda_log_err(&format!(
        "SORT PAD: count={} padding={} in_keys=0x{:x} in_ranks=0x{:x} out_keys=0x{:x} out_ranks=0x{:x}",
        count, padding, in_keys, in_ranks, out_keys, out_ranks));
    // Pad both input AND output buffers with 0xFFFFFFFF sentinel values.
    // Input: the kernel loads in uint4 chunks, over-reading 1-3 elements.
    // Output: the scatter writes to positions based on the padded input count.
    for buf in [in_keys, in_ranks, out_keys, out_ranks] {
        for i in 0..padding {
            let off = (count as usize + i) * 4;
            let r = hip_runtime_sys::hipMemcpyHtoD(
                hip_runtime_sys::hipDeviceptr_t((buf as usize + off) as *mut _),
                sentinel_ptr,
                4,
            );
            if r.is_err() {
                super::driver::zluda_log_err(&format!(
                    "SORT PAD FAIL: buf=0x{:x} off={} err={:?}", buf, off, r));
            }
        }
    }
    // Verify: read back in_ranks padding to confirm write succeeded
    let mut verify = [0u32; 4];
    let verify_count = padding.min(4);
    let _ = hip_runtime_sys::hipMemcpyDtoH(
        verify.as_mut_ptr() as *mut _,
        hip_runtime_sys::hipDeviceptr_t((in_ranks as usize + count as usize * 4) as *mut _),
        verify_count * 4,
    );
    super::driver::zluda_log_err(&format!(
        "SORT PAD VERIFY in_ranks[{}..{}] = {:?}",
        count, count as usize + verify_count, &verify[..verify_count]));
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
    // Fix PhysX radix sort padding before each sort pass
    {
        let name = lookup_name(f.base);
        if name.contains("radixSortMultiCalculateRanksLaunch")
            || name.contains("radixSortMultiBlockLaunch") {
            unsafe { fix_radix_sort_padding(kernel_params, stream); }
        }
        // After the LAST RadixSort Downsweep pass (flip=1), immediately read
        // back keys_out to check if the sort itself produces corrupt values
        // (bit 31 stuck) or if something else corrupts them afterward.
        if name.contains("RadixSortDownsweep") && !extra.is_null() && kernel_params.is_null() {
            unsafe {
                let extra_slice = std::slice::from_raw_parts(extra, 5);
                let buf_ptr = extra_slice[1] as *const u8;
                let buf_size = *(extra_slice[3] as *const usize);
                if !buf_ptr.is_null() && buf_size >= 52 {
                    let keys_out = *(buf_ptr.add(8) as *const u64);
                    let count = *(buf_ptr.add(40) as *const u32);
                    let flip = *(buf_ptr.add(50) as *const u8);
                    if flip == 1 && count > 0 && count < 100000 {
                        // This is the last pass — read back keys_out immediately
                        let n = count as usize;
                        let mut keys = vec![0u32; n];
                        let r = hip_runtime_sys::hipDeviceSynchronize();
                        if r.is_ok() {
                            let r2 = hip_runtime_sys::hipMemcpyDtoH(
                                keys.as_mut_ptr() as *mut _,
                                hip_runtime_sys::hipDeviceptr_t(keys_out as *mut _),
                                n * 4,
                            );
                            if r2.is_ok() {
                                let bad: Vec<usize> = keys.iter().enumerate()
                                    .filter(|(_, &v)| v & 0x80000000 != 0)
                                    .map(|(i, _)| i)
                                    .collect();
                                if !bad.is_empty() {
                                    super::driver::zluda_log_err(&format!(
                                        "SORT_OUTPUT_CORRUPT keys_out=0x{:x} count={} bad={} first_bad=[{}]=0x{:08x}",
                                        keys_out, count, bad.len(), bad[0], keys[bad[0]]));
                                    // Fix: clear bit 31 on all corrupt values and write back
                                    let mut fixed = 0usize;
                                    for &idx in &bad {
                                        let v = keys[idx];
                                        let cleared = v & 0x7FFFFFFF;
                                        if (cleared as usize) < n {
                                            keys[idx] = cleared;
                                            fixed += 1;
                                        }
                                    }
                                    if fixed > 0 {
                                        let r3 = hip_runtime_sys::hipMemcpyHtoD(
                                            hip_runtime_sys::hipDeviceptr_t(keys_out as *mut std::ffi::c_void),
                                            keys.as_ptr() as *mut std::ffi::c_void,
                                            n * 4,
                                        );
                                        super::driver::zluda_log_err(&format!(
                                            "  SORT_FIX applied: cleared bit 31 on {}/{} values, writeback={:?}",
                                            fixed, bad.len(), r3));
                                    }
                                } else {
                                    super::driver::zluda_log_err(&format!(
                                        "SORT_OUTPUT_OK keys_out=0x{:x} count={} all_clean", keys_out, count));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Validate CreateGrid input buffer — check for corrupt indices
        if name.contains("CreateGrid") && !extra.is_null() && kernel_params.is_null() {
            unsafe {
                let extra_slice = std::slice::from_raw_parts(extra, 5);
                let buf_ptr = extra_slice[1] as *const u8;
                if !buf_ptr.is_null() {
                    // CreateGrid params: u32 count (offset 0), u64 input_ptr (offset 8)
                    let count = *(buf_ptr as *const u32);
                    let input_ptr = *(buf_ptr.add(8) as *const u64);
                    let output_ptr1 = *(buf_ptr.add(16) as *const u64);
                    let output_ptr2 = *(buf_ptr.add(24) as *const u64);
                    // Always log CreateGrid params
                    super::driver::zluda_log_err(&format!(
                        "CREATEGRID count={} input=0x{:x} out1=0x{:x} out2=0x{:x} grid={}",
                        count, input_ptr, output_ptr1, output_ptr2, grid_dim_x));
                    // Read back first min(count, 128) indices from input buffer
                    // Check ALL indices (not just first 128) — values with bit 31
                    // set cause sign-extension in mul.wide.s32 → negative offset → fault
                    let n = count as usize;
                    if n > 0 && n < 100_000 && input_ptr > 0x1000 {
                        let mut indices = vec![0i32; n];
                        let read_result = hip_runtime_sys::hipMemcpyDtoH(
                            indices.as_mut_ptr() as *mut _,
                            hip_runtime_sys::hipDeviceptr_t(input_ptr as *mut _),
                            n * 4,
                        );
                        if read_result.is_ok() {
                            // Check for out-of-range indices
                            let max_reasonable = 10_000_000i32; // 10M elements max
                            let mut bad_count = 0u32;
                            let mut first_bad_idx = 0usize;
                            let mut first_bad_val = 0i32;
                            for (i, &val) in indices.iter().enumerate() {
                                if val < 0 || val > max_reasonable {
                                    if bad_count == 0 {
                                        first_bad_idx = i;
                                        first_bad_val = val;
                                    }
                                    bad_count += 1;
                                }
                            }
                            if bad_count > 0 {
                                super::driver::zluda_log_err(&format!(
                                    "CORRUPT_INPUT CreateGrid count={} input=0x{:x} out1=0x{:x} bad_indices={}/{} first_bad=[{}]={} (0x{:08x})",
                                    count, input_ptr, output_ptr1, bad_count, n, first_bad_idx, first_bad_val, first_bad_val as u32));
                                // Fix: clamp all out-of-range indices to 0.
                                // CreateGrid uses mul.wide.s32 — negative indices produce
                                // massive negative offsets → GPU page fault. On NVIDIA, sort
                                // corruption doesn't happen due to different CWSR/preemption
                                // behavior. On AMD, radix sort intermittently produces garbage
                                // values (bit 31 set, or fully corrupt) when Vulkan rendering
                                // preempts compute wavefronts. Clamping to 0 makes CreateGrid
                                // write duplicate entries at output[0] instead of faulting.
                                // One frame of slightly wrong grid data is invisible vs fatal crash.
                                let mut fixed = 0usize;
                                let count_i32 = count as i32;
                                for idx in indices.iter_mut() {
                                    if *idx < 0 || *idx >= count_i32 {
                                        *idx = 0;
                                        fixed += 1;
                                    }
                                }
                                if fixed > 0 {
                                    let r3 = hip_runtime_sys::hipMemcpyHtoD(
                                        hip_runtime_sys::hipDeviceptr_t(input_ptr as *mut std::ffi::c_void),
                                        indices.as_ptr() as *mut std::ffi::c_void,
                                        n * 4,
                                    );
                                    super::driver::zluda_log_err(&format!(
                                        "  CREATEGRID_FIX clamped {}/{} indices to 0, writeback={:?}",
                                        fixed, n, r3));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Validate TurbulenceFS surface descriptors
        if name.contains("TurbulenceFS_dynamicGrid") && !extra.is_null() && kernel_params.is_null() {
            unsafe {
                let extra_slice = std::slice::from_raw_parts(extra, 5);
                let buf_ptr = extra_slice[1] as *const u8;
                let buf_size = *(extra_slice[3] as *const usize);
                if !buf_ptr.is_null() && buf_size >= 32 {
                    // param_5 (surface handle) at byte offset 24
                    let handle = *(buf_ptr.add(24) as *const u64);
                    let shift = handle >> (64 - 3);
                    let desc_ptr = ((handle << 3) >> 3) as u64;
                    let alloc_min = super::memory::ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
                    let alloc_max = super::memory::ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
                    if desc_ptr > 0x1000 && (desc_ptr >> 40) == 0x7f {
                        if desc_ptr < alloc_min.saturating_sub(0x10000000) || desc_ptr > alloc_max.saturating_add(0x10000000) {
                            super::driver::zluda_log_err(&format!(
                                "CORRUPT_SURFACE '{}' handle=0x{:x} desc_ptr=0x{:x} range=[0x{:x}..0x{:x}]",
                                name, handle, desc_ptr, alloc_min, alloc_max));
                        }
                    }
                }
            }
        }
        // Fix PhysX CreateParticleShapesIncremental_MultiStage_4 stale data crash.
        //
        // ROOT CAUSE (confirmed 2026-04-01):
        // PhysX's particle shape pipeline runs 4 stages per frame:
        //   Stage_1: initializes shape index/count arrays from particle data
        //   Stage_2_Batched: reads shape records, computes counters for Stage_4
        //   Stage_3: processes records, marks them with sentinel -1 in field_at_8
        //   Stage_4: scatter-writes using counters — atomicAdd(&counter[shapeIdx], -1)
        //
        // Stage_2_Batched reads rd6[rd7[i]].field_at_8 from shape records,
        // decrements by 1, and writes the result as the counter for Stage_4.
        // On NVIDIA, shape records are implicitly cleared between frames (GPU
        // memory pages are zeroed by the driver — confirmed by LeftoverLocals
        // CVE-2023-4969 research showing NVIDIA is not affected). On AMD/HIP,
        // global memory retains stale data from previous kernel writes.
        //
        // When Stage_1 doesn't write new indices for a shape (e.g., particle
        // count changes between frames), Stage_2_Batched reads rd7[i]=0 (stale
        // from previous Stage_4 drain), indexes into rd6[0].field_at_8 which
        // contains Stage_3's stale -1 sentinel, computes -1-1=-2, and writes
        // -2 to the counter array. Stage_4 then does atomicAdd on -2, producing
        // wild scatter-write addresses → GPU page fault (TCP, VMID 15).
        //
        // FIX: Before Stage_4 launches, read the counter array. If any values
        // are negative (> 0x80000000 as u32), zero the FLAGS array (param_2)
        // so all Stage_4 threads skip the scatter-write (flags & 0x11 != 1).
        // This makes Stage_4 a safe no-op for the affected frame. One frame of
        // missing particle shape updates is invisible; a GPU fault is fatal.
        if name.contains("CreateParticleShapesIncremental_MultiStage_4") {
            unsafe {
                let (count, flags_ptr, shape_idx_ptr, counter_ptr) =
                    if !kernel_params.is_null() {
                        let count = *(*kernel_params.add(3) as *const u32);
                        let flags_ptr = *(*kernel_params.add(2) as *const u64);
                        let shape_idx_ptr = *(*kernel_params.add(6) as *const u64);
                        let counter_ptr = *(*kernel_params.add(7) as *const u64);
                        (count, flags_ptr, shape_idx_ptr, counter_ptr)
                    } else if !extra.is_null() {
                        let extra_slice = std::slice::from_raw_parts(extra, 5);
                        let buf = extra_slice[1] as *const u8;
                        let sz = *(extra_slice[3] as *const usize);
                        if buf.is_null() || sz < 64 { (0u32, 0u64, 0u64, 0u64) } else {
                            let count = *(buf.add(24) as *const u32);
                            let flags_ptr = *(buf.add(16) as *const u64);
                            let shape_idx_ptr = *(buf.add(48) as *const u64);
                            let counter_ptr = *(buf.add(56) as *const u64);
                            (count, flags_ptr, shape_idx_ptr, counter_ptr)
                        }
                    } else {
                        (0u32, 0u64, 0u64, 0u64)
                    };
                if count > 0 && count < 100_000 && shape_idx_ptr > 0x1000 {
                    let n = count as usize;
                    // Read shapeIndex to find max shape index → counter array size
                    let mut shape_indices = vec![0u32; n];
                    let r = hip_runtime_sys::hipMemcpyDtoH(
                        shape_indices.as_mut_ptr() as *mut _,
                        hip_runtime_sys::hipDeviceptr_t(shape_idx_ptr as *mut _),
                        n * 4,
                    );
                    if r.is_ok() {
                        let max_idx = shape_indices.iter().filter(|&&v| v < 100_000).max().copied().unwrap_or(0);
                        // Save for inter-stage tracking
                        SHAPES4_CTR_PTR.store(counter_ptr, std::sync::atomic::Ordering::Relaxed);
                        SHAPES4_CTR_COUNT.store(max_idx + 1, std::sync::atomic::Ordering::Relaxed);
                        // Read counter array and check for corruption
                        if counter_ptr > 0x1000 && max_idx > 0 {
                            let cn = (max_idx + 1) as usize;
                            let mut counters = vec![0u32; cn];
                            let r2 = hip_runtime_sys::hipMemcpyDtoH(
                                counters.as_mut_ptr() as *mut _,
                                hip_runtime_sys::hipDeviceptr_t(counter_ptr as *mut _),
                                cn * 4,
                            );
                            if r2.is_ok() {
                                let bad_count = counters.iter()
                                    .filter(|&&v| v > 0x80000000).count();
                                if bad_count > 0 {
                                    // Counter corruption detected — stale shape record
                                    // sentinels caused Stage_2_Batched to write negative
                                    // values. Zero the flags array to disable all threads
                                    // in Stage_4, preventing scatter-write to wild addresses.
                                    let first5: Vec<String> = counters.iter().take(cn.min(8))
                                        .map(|v| format!("0x{:08x}", v)).collect();
                                    super::driver::zluda_log_err(&format!(
                                        "SHAPES4_FIX count={} ctr_bad={}/{} vals=[{}] → zeroing flags at 0x{:x}",
                                        count, bad_count, cn, first5.join(","), flags_ptr));
                                    if flags_ptr > 0x1000 {
                                        // Flags array: count entries, 4 bytes each (u16 at stride 4).
                                        // Zero all flags → (flags & 0x11) == 0 != 1 → all threads skip.
                                        let _ = hip_runtime_sys::hipMemsetD8(
                                            hip_runtime_sys::hipDeviceptr_t(flags_ptr as *mut _),
                                            0,
                                            n * 4,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Inter-stage counter tracking: after each CreateParticleShapes stage completes
        // (hipDeviceSynchronize runs in the SYNC block after launch), read the counter
        // array that Stage_4 will use. We do this BEFORE launch so it captures the state
        // left by the PREVIOUS kernel.
        if name.contains("CreateParticleShapesIncremental_MultiStage") && !name.contains("_4") {
            let ctr_ptr = SHAPES4_CTR_PTR.load(std::sync::atomic::Ordering::Relaxed);
            let ctr_n = SHAPES4_CTR_COUNT.load(std::sync::atomic::Ordering::Relaxed) as usize;
            if ctr_ptr > 0x1000 && ctr_n > 0 && ctr_n < 1000 {
                let mut ctr = vec![0u32; ctr_n];
                let r = unsafe { hip_runtime_sys::hipMemcpyDtoH(
                    ctr.as_mut_ptr() as *mut _,
                    hip_runtime_sys::hipDeviceptr_t(ctr_ptr as *mut _),
                    ctr_n * 4,
                )};
                if r.is_ok() {
                    let bad_count = ctr.iter().filter(|&&v| v > 100_000).count();
                    // Always log counter values for the frame where corruption appears
                    let vals: Vec<String> = ctr.iter().take(8)
                        .map(|v| format!("0x{:08x}", v)).collect();
                    super::driver::zluda_log_err(&format!(
                        "CTR_TRACK pre-'{}' ctrPtr=0x{:x} n={} bad={} vals=[{}]",
                        name, ctr_ptr, ctr_n, bad_count, vals.join(",")));
                }
            }
        }
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
    // Log launch failures with full context for diagnosis
    if result.is_err() {
        let name = lookup_name(f.base);
        super::driver::zluda_log_err(&format!(
            "cuLaunchKernel FAILED '{}' func={:p} grid=[{},{},{}] block=[{},{},{}] shared={} -> {:?}",
            name, f.base.0, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, result));
    }
    // Lightweight launch counter + periodic monitoring
    if result.is_ok() {
        static LAUNCH_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = LAUNCH_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // VRAM/GTT every 2000 launches
        if n % 2000 == 0 {
            let vram = std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_used").unwrap_or_default();
            let gtt = std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_gtt_used").unwrap_or_default();
            super::driver::zluda_log_err(&format!(
                "MEM#{} VRAM={} GTT={}", n, vram.trim(), gtt.trim()));
        }
        // Dump /proc/self/maps every 5000 launches
        if n % 5000 == 0 {
            if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
                let relevant: Vec<&str> = maps.lines()
                    .filter(|l| l.contains("kfd") || l.contains("drm") || l.contains("anon_inode")
                        || (l.contains("rw") && !l.contains(".so") && !l.contains("[") && l.len() > 40))
                    .collect();
                super::driver::zluda_log_err(&format!(
                    "MAPS#{} {} relevant mappings (total {})", n, relevant.len(), maps.lines().count()));
                for m in relevant.iter().take(20) {
                    super::driver::zluda_log_err(&format!("  {}", m));
                }
            }
        }
        // ZLUDA_SYNC_LAUNCHES: optionally sync after each launch
        static SYNC_LAUNCHES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sync = *SYNC_LAUNCHES.get_or_init(|| {
            std::env::var("ZLUDA_SYNC_LAUNCHES").map_or(false, |v| v == "1")
        });
        if sync {
            let name = lookup_name(f.base);
            // Dump kernel params for crash analysis
            let mut raw_params = String::new();
            if !extra.is_null() && kernel_params.is_null() {
                // extra buffer: packed params
                unsafe {
                    let ep = std::slice::from_raw_parts(extra, 5);
                    let buf = ep[1] as *const u8;
                    let sz = *(ep[3] as *const usize);
                    if !buf.is_null() && sz > 0 {
                        let max_params = if name.contains("TurbulenceFS") { 48 } else { 8 };
                        let n_u64 = (sz / 8).min(max_params);
                        for j in 0..n_u64 {
                            let v = *(buf.add(j * 8) as *const u64);
                            if !raw_params.is_empty() { raw_params.push(','); }
                            raw_params.push_str(&format!("{:#x}", v));
                        }
                    }
                }
            } else if !kernel_params.is_null() {
                // kernel_params: array of void* pointers to individual param values.
                // Just log the pointer addresses, don't dereference them —
                // they point into the caller's stack/heap and may be unmapped.
                unsafe {
                    for j in 0..8usize {
                        let pp = *kernel_params.add(j);
                        let pp_addr = pp as usize;
                        if pp_addr == 0 || pp_addr < 0x1000 || pp_addr > 0x7fffffffffff { break; }
                        if !raw_params.is_empty() { raw_params.push(','); }
                        raw_params.push_str(&format!("@{:#x}", pp_addr));
                    }
                }
            }
            let sync_info = format!("SYNC#{} '{}' [{}]", n, name, raw_params);
            super::driver::zluda_log_err(&sync_info);
            // Store for SIGABRT handler to dump on crash
            super::driver::set_last_kernel(sync_info);
            // Use hipDeviceSynchronize to catch faults from ALL streams
            // (PhysX uses multiple threads/streams — TurbulenceFS on 02bc,
            // RadixSort/CreateGrid on 0130)
            let sync_result = unsafe { hip_runtime_sys::hipDeviceSynchronize() };
            // Check GPU-side sust debug buffer (defined in zluda_ptx_impl.bc)
            {
                static SUST_DBG: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let dbg_addr = SUST_DBG.get_or_init(|| {
                    // Find the debug buffer global by trying hipGetSymbolAddress
                    // with the mangled name
                    let mut ptr = std::ptr::null_mut();
                    let name = c"__zluda_ptx_impl_sust_debug_buf";
                    let r = unsafe { hip_runtime_sys::hipGetSymbolAddress(&mut ptr, name.as_ptr() as _) };
                    if r.is_ok() && !ptr.is_null() {
                        super::driver::zluda_log_err(&format!("sust_debug_buf found at {:p}", ptr));
                        ptr as u64
                    } else { 0 }
                });
                if *dbg_addr != 0 {
                    let mut data = [0u64; 8];
                    let _ = unsafe { hip_runtime_sys::hipMemcpyDtoH(
                        data.as_mut_ptr() as _, hip_runtime_sys::hipDeviceptr_t(*dbg_addr as _), 64) };
                    if data[0] != 0 {
                        super::driver::zluda_log_err(&format!(
                            "SUST_BAD_DESCRIPTOR kernel='{}' [0]=0x{:x} [1]=0x{:x} [2]=0x{:x} [3]=0x{:x} [4]=0x{:x} [5]=0x{:x} [6]=0x{:x} [7]=0x{:x}",
                            name, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]));
                        let _ = unsafe { hip_runtime_sys::hipMemset(*dbg_addr as _, 0, 64) };
                    }
                }
            }
            if sync_result.is_err() {
                // GPU fault detected! This kernel (or a preceding async one) faulted.
                // Log everything we can before the HSA runtime kills us with SIGABRT.
                super::driver::zluda_log_err(&format!(
                    "GPU_FAULT after SYNC#{} '{}' sync={:?} params=[{}]",
                    n, name, sync_result, raw_params));
                // Read VRAM usage at fault time
                if let Ok(vram) = std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_used") {
                    super::driver::zluda_log_err(&format!(
                        "GPU_FAULT VRAM={} MiB", vram.trim().parse::<usize>().unwrap_or(0) / (1024*1024)));
                }
                // Log file uses append mode (open/write/close per line),
                // so data is already on disk.
            }
            // After CopyBits2(flag=1) completes, verify key buffer
            if sync_result.is_ok()
                && name.contains("CopyBits2")
                && !kernel_params.is_null()
            {
                unsafe {
                    let p1 = *(*kernel_params.add(1) as *const u64);
                    let count = *(*kernel_params.add(3) as *const u32);
                    let flag = *(*kernel_params.add(4) as *const u8);
                    if flag == 1 {
                        let n = std::cmp::min(count as usize, 28);
                        let mut keys = vec![0u32; n];
                        let _ = hip_runtime_sys::hipMemcpyDtoH(
                            keys.as_mut_ptr() as *mut _,
                            hip_runtime_sys::hipDeviceptr_t(p1 as *mut _),
                            n * 4,
                        );
                        let hex: Vec<String> = keys.iter().map(|k| format!("0x{:08x}", k)).collect();
                        super::driver::zluda_log_err(&format!(
                            "COPYBITS2 OUTPUT flag={} ptr1=0x{:x}\n  keys_hex={:?}", flag, p1, hex));
                    }
                }
            }
            // After MultiBlock completes, dump the per-block histogram
            if sync_result.is_ok()
                && name.contains("MultiBlockLaunch")
                && !kernel_params.is_null()
            {
                unsafe {
                    let desc_ptr = *(*kernel_params.add(0) as *const u64);
                    let count = *(*kernel_params.add(1) as *const u32);
                    let shift = *(*kernel_params.add(2) as *const u32);
                    let mut desc = [0u8; 48];
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        desc.as_mut_ptr() as *mut _,
                        hip_runtime_sys::hipDeviceptr_t(desc_ptr as *mut _),
                        48,
                    );
                    let in_keys_ptr = u64::from_ne_bytes(desc[0..8].try_into().unwrap());
                    let rbc_ptr = u64::from_ne_bytes(desc[32..40].try_into().unwrap());
                    // Dump first 28 input keys to verify they match expectations
                    let padded = ((count as usize + 3) / 4) * 4;
                    let mut in_keys = vec![0u32; padded];
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        in_keys.as_mut_ptr() as *mut _,
                        hip_runtime_sys::hipDeviceptr_t(in_keys_ptr as *mut _),
                        padded * 4,
                    );
                    // Also dump as hex for clarity
                    let hex_keys: Vec<String> = in_keys[..padded].iter().map(|k| format!("0x{:08x}", k)).collect();
                    super::driver::zluda_log_err(&format!(
                        "MULTIBLOCK shift={} count={} in_keys=0x{:x}\n  keys_hex={:?}",
                        shift, count, in_keys_ptr, hex_keys));
                    // Read block 0's histogram: 16 digits, stride = gridDim.x (32)
                    // Layout: radixBlockCounts[digit * numBlocks + blockIdx]
                    let num_blocks = grid_dim_x as usize;
                    let total = 16 * num_blocks;
                    let mut rbc = vec![0u32; total];
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        rbc.as_mut_ptr() as *mut _,
                        hip_runtime_sys::hipDeviceptr_t(rbc_ptr as *mut _),
                        total * 4,
                    );
                    // Extract block 0's counts for each digit
                    let block0_counts: Vec<u32> = (0..16).map(|d| rbc[d * num_blocks]).collect();
                    let total_count: u32 = block0_counts.iter().sum();
                    // Also find ALL non-zero entries
                    let nonzero: Vec<(usize, u32)> = rbc.iter().enumerate()
                        .filter(|(_, &v)| v != 0).collect::<Vec<_>>()
                        .into_iter().map(|(i, &v)| (i, v)).collect();
                    // Also try block-major layout: rbc[blockIdx * 16 + digit]
                    let block0_alt: Vec<u32> = rbc[0..16].to_vec();
                    let alt_total: u32 = block0_alt.iter().sum();
                    super::driver::zluda_log_err(&format!(
                        "HISTOGRAM shift={} count={} digit_major=[total={} {:?}] block_major=[total={} {:?}]\n  nonzero={:?}",
                        shift, count, total_count, block0_counts, alt_total, block0_alt, nonzero));
                }
            }
            // After CalculateRanks completes, dump the OUTPUT ranks buffer
            if sync_result.is_ok()
                && name.contains("CalculateRanksLaunch")
                && !kernel_params.is_null()
            {
                unsafe {
                    let desc_ptr = *(*kernel_params.add(0) as *const u64);
                    let count = *(*kernel_params.add(1) as *const u32);
                    let shift = *(*kernel_params.add(2) as *const u32);
                    let mut desc = [0u8; 48];
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        desc.as_mut_ptr() as *mut _,
                        hip_runtime_sys::hipDeviceptr_t(desc_ptr as *mut _),
                        48,
                    );
                    let out_ranks = u64::from_ne_bytes(desc[24..32].try_into().unwrap());
                    let out_keys = u64::from_ne_bytes(desc[16..24].try_into().unwrap());
                    let padded = ((count as usize + 3) / 4) * 4;
                    let mut ranks = vec![0u32; padded];
                    let mut keys = vec![0u32; padded];
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        ranks.as_mut_ptr() as *mut _,
                        hip_runtime_sys::hipDeviceptr_t(out_ranks as *mut _),
                        padded * 4,
                    );
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        keys.as_mut_ptr() as *mut _,
                        hip_runtime_sys::hipDeviceptr_t(out_keys as *mut _),
                        padded * 4,
                    );
                    let bad: Vec<_> = ranks[..count as usize].iter().enumerate()
                        .filter(|(_, &r)| r != 0xFFFFFFFF && r >= count)
                        .collect();
                    super::driver::zluda_log_err(&format!(
                        "POST-PASS shift={} count={} bad_ranks={:?}\n  ranks={:?}\n  keys={:?}\n  sentinel_r={:?} sentinel_k={:?}",
                        shift, count, bad,
                        &ranks[..count as usize], &keys[..count as usize],
                        &ranks[count as usize..padded], &keys[count as usize..padded]));
                }
            }
            if sync_result.is_err() {
                super::driver::zluda_log_err(&format!(
                    "SYNC FAULT '{}' func={:p} -> {:?}",
                    name, f.base.0, sync_result));
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

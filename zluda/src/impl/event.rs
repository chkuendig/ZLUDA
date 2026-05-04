use hip_runtime_sys::*;

// hipEventElapsedTime unit history:
//   ROCm 6.2: returns realistic ms values including host-side latency.
//   ROCm 6.4: returns only GPU-side execution time, near-zero when no kernels
//             ran between the two events (effectively a unit/scale change for
//             PhysX's purposes — frames look like sub-microsecond intervals).
//
// PhysX uses cuEventElapsedTime as its frame timer; near-zero values cause it
// to never trigger physics simulation steps. We previously worked around this
// on 6.4 by substituting a wall-clock delta when hip_ms < 1.0. That workaround
// was REMOVED on 2026-04-09 because we're now pinned to ROCm 6.2.4
// (proton-ge-custom scripts/fetch-rocm.sh, see commit 26273fe0 and the comment in
// that script). On 6.2 the workaround was CLOBBERING valid sub-ms values with
// stale wall-clock deltas — e.g. a 74 µs operation between two events would
// be returned as 2959 ms (the elapsed wall-clock since the previous call),
// causing PhysX to think a single physics frame took 3 seconds and corrupt
// its actor state on the very first MS_2_Batched launch.
//
// **If we ever upgrade back to ROCm 6.4+**, this workaround MUST be reinstated
// (and ideally gated on the runtime hipRuntimeGetVersion or a cargo feature).
// See `feedback_*` memory note about 6.4 timer scaling.

pub(crate) unsafe fn create(event: *mut hipEvent_t, flags: ::core::ffi::c_uint) -> hipError_t {
    // Flag values are compatible between CUDA and HIP for 0,1,2,4
    let result = hipEventCreateWithFlags(event, flags);
    super::driver::zluda_log_err(&format!(
        "cuEventCreate(flags={}) -> {:?} (event={:p})",
        flags,
        result,
        if event.is_null() {
            std::ptr::null()
        } else {
            *event as *const _
        }
    ));
    result
}

pub(crate) unsafe fn elapsed_time(
    milliseconds: *mut ::core::ffi::c_float,
    start: hipEvent_t,
    end: hipEvent_t,
) -> hipError_t {
    // Pass-through on ROCm 6.2.4 — see file-header comment for the 6.4 history.
    let result = hipEventElapsedTime(milliseconds, start, end);
    if result.is_ok() && !milliseconds.is_null() {
        super::driver::zluda_log_err(&format!(
            "cuEventElapsedTime -> {:.6} ms (result={:?})",
            *milliseconds, result
        ));
    }
    result
}

pub(crate) unsafe fn elapsed_time_v2(
    milliseconds: *mut ::core::ffi::c_float,
    start: hipEvent_t,
    end: hipEvent_t,
) -> hipError_t {
    elapsed_time(milliseconds, start, end)
}

pub(crate) unsafe fn query(event: hipEvent_t) -> hipError_t {
    let result = hipEventQuery(event);
    super::driver::zluda_log_err(&format!("cuEventQuery(event={:?}) -> {:?}", event, result));
    result
}

pub(crate) unsafe fn destroy_v2(event: hipEvent_t) -> hipError_t {
    super::driver::zluda_log_err(&format!("cuEventDestroy_v2(event={:?})", event));
    hipEventDestroy(event)
}

pub(crate) unsafe fn record(event: hipEvent_t, stream: hipStream_t) -> hipError_t {
    let result = hipEventRecord(event, stream);
    super::driver::zluda_log_err(&format!(
        "cuEventRecord(event={:?}, stream={:?}) -> {:?}",
        event, stream, result
    ));
    result
}

pub(crate) unsafe fn record_with_flags(
    event: hipEvent_t,
    stream: hipStream_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    // Flag values are compatible between CUDA and HIP for 0,1

    // The ROCm 6.4.0 headers have a declaration for hipEventRecordWithFlags, but the library has
    // no implementation. The implementation was added in ROCm 6.4.2. We only support the default flag for now.
    if flags != hipEventRecordDefault {
        return hipError_t::ErrorInvalidValue;
    }

    hipEventRecord(event, stream)
}

pub(crate) unsafe fn synchronize(event: hipEvent_t) -> hipError_t {
    super::driver::zluda_log_err(&format!("cuEventSynchronize(event={:?}) ENTER", event));
    let result = hipEventSynchronize(event);
    super::driver::zluda_log_err(&format!(
        "cuEventSynchronize(event={:?}) -> {:?}",
        event, result
    ));
    result
}

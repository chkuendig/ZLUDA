use hip_runtime_sys::*;
use std::sync::Mutex;
use std::time::Instant;

// Track wall-clock time between cuEventElapsedTime calls.
// ROCm 6.4 changed hipEventElapsedTime to return only GPU-side execution time
// between two events (nearly zero when no kernels run between records).
// ROCm 6.2 included host-side latency, returning realistic frame intervals.
// PhysX uses cuEventElapsedTime as its frame timer — near-zero values cause
// it to never trigger physics simulation steps. When HIP returns < 1ms but
// wall-clock time is > 1ms, substitute the wall-clock value.
static ELAPSED_TIME_LAST_CALL: Mutex<Option<Instant>> = Mutex::new(None);

pub(crate) unsafe fn create(event: *mut hipEvent_t, flags: ::core::ffi::c_uint) -> hipError_t {
    // Flag values are compatible between CUDA and HIP for 0,1,2,4
    let result = hipEventCreateWithFlags(event, flags);
    super::driver::zluda_log_err(&format!(
        "cuEventCreate(flags={}) -> {:?} (event={:p})",
        flags, result, if event.is_null() { std::ptr::null() } else { *event as *const _ }
    ));
    result
}

pub(crate) unsafe fn elapsed_time(
    milliseconds: *mut ::core::ffi::c_float,
    start: hipEvent_t,
    end: hipEvent_t,
) -> hipError_t {
    let result = hipEventElapsedTime(milliseconds, start, end);
    if result.is_ok() && !milliseconds.is_null() {
        let hip_ms = *milliseconds;
        // ROCm 6.4 returns near-zero GPU-side time when no kernels run between
        // event records. PhysX uses this as a frame timer and needs realistic
        // wall-clock intervals to trigger physics steps. When HIP returns < 1ms,
        // substitute wall-clock time if it's larger.
        if hip_ms < 1.0 {
            if let Ok(mut last) = ELAPSED_TIME_LAST_CALL.lock() {
                let now = Instant::now();
                if let Some(prev) = last.as_ref() {
                    let wall_ms = prev.elapsed().as_secs_f32() * 1000.0;
                    if wall_ms > 1.0 {
                        *milliseconds = wall_ms;
                    }
                }
                *last = Some(now);
            }
        }
        super::driver::zluda_log_err(&format!(
            "cuEventElapsedTime -> {:.6} ms (hip={:.6} ms, result={:?})",
            *milliseconds, hip_ms, result
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
    super::driver::zluda_log_err(&format!(
        "cuEventQuery(event={:?}) -> {:?}", event, result
    ));
    result
}

pub(crate) unsafe fn destroy_v2(event: hipEvent_t) -> hipError_t {
    super::driver::zluda_log_err(&format!("cuEventDestroy_v2(event={:?})", event));
    hipEventDestroy(event)
}

pub(crate) unsafe fn record(event: hipEvent_t, stream: hipStream_t) -> hipError_t {
    let result = hipEventRecord(event, stream);
    super::driver::zluda_log_err(&format!(
        "cuEventRecord(event={:?}, stream={:?}) -> {:?}", event, stream, result
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
    super::driver::zluda_log_err(&format!("cuEventSynchronize(event={:?}) -> {:?}", event, result));
    result
}

use hip_runtime_sys::*;

pub(crate) fn synchronize(stream: hipStream_t) -> hipError_t {
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamSynchronize({:?})", stream));
    let result = unsafe { hipStreamSynchronize(stream) };
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamSynchronize -> {:?}", result));
    result
}

pub(crate) fn create(stream: *mut hipStream_t, flags: ::core::ffi::c_uint) -> hipError_t {
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamCreate(flags={})", flags));
    let result = unsafe { hipStreamCreateWithFlags(stream, flags) };
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamCreate -> {:?}", result));
    result
}

pub(crate) fn create_with_priority(
    stream: *mut hipStream_t,
    flags: ::core::ffi::c_uint,
    priority: ::core::ffi::c_int,
) -> hipError_t {
    unsafe { hipStreamCreateWithPriority(stream, flags, priority) }
}

pub(crate) fn destroy_v2(stream: hipStream_t) -> hipError_t {
    unsafe { hipStreamDestroy(stream) }
}

pub(crate) fn begin_capture_v2(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t {
    unsafe { hipStreamBeginCapture(stream, mode) }
}

pub(crate) fn end_capture(stream: hipStream_t, graph: *mut hipGraph_t) -> hipError_t {
    unsafe { hipStreamEndCapture(stream, graph) }
}

pub(crate) fn is_capturing(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
) -> hipError_t {
    unsafe { hipStreamIsCapturing(stream, capture_status) }
}

pub(crate) fn get_capture_info_v2(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
    id: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status,
            id,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

pub(crate) fn get_capture_info_v3(
    stream: hipStream_t,
    capture_status_out: *mut hipStreamCaptureStatus,
    id_out: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    edge_data_out: *mut *const cuda_types::cuda::CUgraphEdgeData,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    if !edge_data_out.is_null() {
        return hipError_t::ErrorNotSupported;
    }
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status_out,
            id_out,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

static QUERY_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) fn query(stream: hipStream_t) -> hipError_t {
    let result = unsafe { hipStreamQuery(stream) };
    // PhysX uses cuStreamQuery(NULL) as a push-buffer flush / dispatch heartbeat.
    // On real NVIDIA, this returns NOT_READY when there's pending GPU work.
    // Our dispatch loop runs too fast (no real kernels), causing PhysX to
    // decide "no GPU work needed" and tear down GPU before APEX creates actors.
    // Always return NOT_READY for NULL stream to keep the dispatch loop alive.
    let result = if stream.0.is_null() {
        Err(hipErrorCode_t::NotReady)
    } else {
        result
    };
    let count = QUERY_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Log first 100, then only when ZLUDA_LOG_ALL is set
    if count < 100 || crate::is_log_all_enabled() {
        super::driver::zluda_log_err(&format!(
            "cuStreamQuery(stream={:?}) -> {:?} [#{}]",
            stream, result, count
        ));
    }
    result
}

pub(crate) fn wait_event(
    stream: hipStream_t,
    event: hipEvent_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    unsafe { hipStreamWaitEvent(stream, event, flags) }
}

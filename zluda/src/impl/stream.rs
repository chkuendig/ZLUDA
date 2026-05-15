use hip_runtime_sys::*;
use std::sync::Mutex;

/// Global lock implementing CUDA's NULL-stream implicit synchronization.
///
/// CUDA semantics: the NULL stream (stream 0) serializes with ALL other streams.
/// An operation on the NULL stream cannot begin until all prior operations on all
/// streams complete, and no operation on any stream can begin until the NULL-stream
/// operation completes.
///
/// HIP/ROCm 6.2.4 does NOT implement this. We enforce it with:
/// - Before any NULL-stream operation (kernel launch, memcpy, memset): acquire this
///   lock + hipDeviceSynchronize (drain all pending work)
/// - The lock is held until the operation is submitted, preventing another thread
///   from submitting work between our sync and our operation
///
/// This lock is shared between kernel launches (function.rs) and memory operations
/// (memory.rs) so that ALL NULL-stream operations serialize with each other.
pub(crate) static NULL_STREAM_LOCK: Mutex<()> = Mutex::new(());

fn strict_null_stream_barrier_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var("ZLUDA_NULL_STREAM_STRICT").ok();
        let v = raw.as_deref().map_or(false, |s| s == "1");
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] ZLUDA_NULL_STREAM_STRICT env={:?} → strict={}",
            raw, v
        ));
        v
    })
}

fn null_stream_uses_global_lock(stream: hipStream_t, strict: bool) -> bool {
    strict || stream.0.is_null()
}

fn null_stream_needs_pre_sync(stream: hipStream_t) -> bool {
    stream.0.is_null()
}

fn null_stream_needs_post_sync(stream: hipStream_t, strict: bool) -> bool {
    strict && stream.0.is_null()
}

pub(crate) struct NullStreamGuard {
    guard: Option<std::sync::MutexGuard<'static, ()>>,
    post_sync: bool,
}

impl Drop for NullStreamGuard {
    fn drop(&mut self) {
        if self.post_sync {
            let _ = unsafe { hipDeviceSynchronize() };
        }
    }
}

/// Synchronize for CUDA NULL-stream semantics before performing a GPU operation.
/// Call this before hipModuleLaunchKernel, hipMemcpyAsync, hipMemsetAsync, etc.
///
/// If `stream` is NULL (0x0): acquires the global lock and waits for ALL pending
/// work on all streams. Returns a guard that must be held until the HIP operation
/// is submitted — this prevents another thread from submitting work in between.
///
/// If `ZLUDA_NULL_STREAM_STRICT=1` is set, we also enforce the missing postlaunch
/// half of CUDA's NULL-stream barrier by:
/// - keeping the lock held until the NULL-stream operation completes, and
/// - briefly taking the same lock for non-NULL submissions so they cannot race
///   ahead of an in-flight NULL-stream operation.
/// This gate is kept opt-in because it fixed Batman's earlier late `n=8827`
/// CreateGrid corruption but still exposes a different later `n=4688` family.
pub(crate) fn null_stream_sync(stream: hipStream_t) -> NullStreamGuard {
    let strict = strict_null_stream_barrier_enabled();
    let guard = if null_stream_uses_global_lock(stream, strict) {
        Some(NULL_STREAM_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    } else {
        None
    };
    if null_stream_needs_pre_sync(stream) {
        let _ = unsafe { hipDeviceSynchronize() };
    }
    NullStreamGuard {
        guard,
        post_sync: null_stream_needs_post_sync(stream, strict),
    }
}

/// 2026-05-15 Plan v6 Step A: NULL-stream barrier for cuEventRecord and
/// cuStreamWaitEvent. Per NVIDIA docs, NULL-stream actions (including
/// kernel launches, cuEventRecord, cuStreamWaitEvent) first wait on all
/// blocking streams. HIP doesn't enforce this. We provide the
/// pre-action sync via the same null_stream_sync mechanism.
///
/// Env-disable with `ZLUDA_NULL_STREAM_BARRIER=0` for A/B testing.
/// Default ON (matches CUDA spec).
pub(crate) fn null_stream_sync_for_event(stream: hipStream_t) -> NullStreamGuard {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let disabled = *DISABLED
        .get_or_init(|| std::env::var("ZLUDA_NULL_STREAM_BARRIER").map_or(false, |v| v == "0"));
    if disabled || !stream.0.is_null() {
        return NullStreamGuard {
            guard: None,
            post_sync: false,
        };
    }
    null_stream_sync(stream)
}

pub(crate) fn synchronize(stream: hipStream_t) -> hipError_t {
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamSynchronize({:?})", stream));
    let result = unsafe { hipStreamSynchronize(stream) };
    // 2026-04-29 Phase 3g: HDP flush hypothesis test. Mode B SIGSEGVs continue
    // even after our LDS-visibility patches eliminated all GPU page faults.
    // Hypothesis: the GPU writes to host-pinned (zero-copy) memory complete
    // on the GPU side, but the writes sit in the HDP write buffer and aren't
    // visible to a subsequent CPU read. NVIDIA cuStreamSynchronize implicitly
    // ensures host visibility; HIP hipStreamSynchronize on AMDGPU may not
    // include HDP flush semantics. Forcing a hipDeviceSynchronize after the
    // stream sync acts as an HDP flush via the driver's full-device wait
    // path. Heavy-handed (~adds latency to every sync) but a clean test of
    // the hypothesis. Env-gate ZLUDA_HDP_FLUSH_ON_SYNC=1 (default ON).
    static HDP_FLUSH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let hdp_flush = *HDP_FLUSH
        .get_or_init(|| std::env::var("ZLUDA_HDP_FLUSH_ON_SYNC").map_or(true, |v| v != "0"));
    if hdp_flush {
        let _ = unsafe { hipDeviceSynchronize() };
        // Also issue a CPU-side memory fence to invalidate any stale CPU
        // caches that might be holding pre-write values.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamSynchronize -> {:?}", result));
    result
}

pub(crate) fn create(stream: *mut hipStream_t, flags: ::core::ffi::c_uint) -> hipError_t {
    let result = unsafe { hipStreamCreateWithFlags(stream, flags) };
    super::driver::zluda_log_err(&format!(
        "cuStreamCreate(flags={:#x}) -> {:?} stream={:?}",
        flags,
        result,
        unsafe { *stream }
    ));
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

/// 2026-05-15 Plan v6 Step C: track whether PhysX has launched its first real
/// kernel. Before the first launch, return NotReady for cuStreamQuery(NULL)
/// to keep PhysX's dispatch loop alive during startup (the original masking
/// rationale). AFTER first launch, pass through the real hipStreamQuery
/// result — matches NVIDIA CUDA spec ("all operations in the stream have
/// completed"; for NULL legacy stream, that means all blocking-stream work).
pub(crate) static FIRST_REAL_LAUNCH_SEEN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn query(stream: hipStream_t) -> hipError_t {
    let hip_result = unsafe { hipStreamQuery(stream) };
    // PhysX uses cuStreamQuery(NULL) as a push-buffer flush / dispatch heartbeat.
    // On real NVIDIA, this returns NOT_READY when there's pending GPU work and
    // SUCCESS when idle. Our previous behavior was hardcoded NotReady (poisoned
    // PhysX dispatch logic). Now: NotReady ONLY before first real launch
    // (PhysX startup race), then real result.
    //
    // Env overrides:
    //   ZLUDA_STREAM_QUERY_LEGACY=1 → restore old always-NotReady behavior
    //   ZLUDA_STREAM_QUERY_REAL=1 → always pass through real result (no startup mask)
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        if std::env::var("ZLUDA_STREAM_QUERY_REAL").map_or(false, |v| v == "1") {
            2 // always-real
        } else if std::env::var("ZLUDA_STREAM_QUERY_LEGACY").map_or(false, |v| v == "1") {
            0 // always-NotReady (old behavior)
        } else {
            1 // refined: NotReady before first launch, real after
        }
    });
    let returned = if stream.0.is_null() {
        match mode {
            0 => Err(hipErrorCode_t::NotReady),
            2 => hip_result,
            _ => {
                if !FIRST_REAL_LAUNCH_SEEN.load(std::sync::atomic::Ordering::Acquire) {
                    Err(hipErrorCode_t::NotReady)
                } else {
                    hip_result
                }
            }
        }
    } else {
        hip_result
    };
    let count = QUERY_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Log first 200, then every 100th, or always when ZLUDA_LOG_ALL is set
    if count < 200 || count % 100 == 0 || crate::is_log_all_enabled() {
        // Show both real HIP result and what we return, to detect masking
        if stream.0.is_null() {
            super::driver::zluda_log_err(&format!(
                "cuStreamQuery(NULL) hip={:?} returned=NOT_READY [#{}]",
                hip_result, count
            ));
        } else {
            super::driver::zluda_log_err(&format!(
                "cuStreamQuery(stream={:?}) -> {:?} [#{}]",
                stream, returned, count
            ));
        }
    }
    returned
}

pub(crate) fn wait_event(
    stream: hipStream_t,
    event: hipEvent_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    // CUDA NULL-stream semantics (NVIDIA docs verified 2026-05-15):
    // "When an action is taken in the legacy stream such as a kernel launch
    // or cudaStreamWaitEvent(), the legacy stream first waits on all blocking
    // streams, the action is queued in the legacy stream, and then all
    // blocking streams wait on the legacy stream."
    //
    // HIP doesn't enforce the pre-action global barrier. Use null_stream_sync
    // (env-disable via ZLUDA_NULL_STREAM_BARRIER=0).
    let _guard = null_stream_sync_for_event(stream);
    let r = unsafe { hipStreamWaitEvent(stream, event, flags) };
    super::driver::zluda_log_err(&format!(
        "cuStreamWaitEvent(stream={:?}, event={:?}, flags={:#x}) -> {:?}",
        stream, event, flags, r
    ));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_stream_policy_default_mode_only_locks_null_stream() {
        let null = hipStream_t(std::ptr::null_mut());
        let other = hipStream_t(1usize as *mut _);
        let strict = false;

        assert!(null_stream_uses_global_lock(null, strict));
        assert!(!null_stream_uses_global_lock(other, strict));
        assert!(null_stream_needs_pre_sync(null));
        assert!(!null_stream_needs_pre_sync(other));
        assert!(!null_stream_needs_post_sync(null, strict));
        assert!(!null_stream_needs_post_sync(other, strict));
    }

    #[test]
    fn null_stream_policy_strict_mode_blocks_all_submissions_and_postsyncs_null() {
        let null = hipStream_t(std::ptr::null_mut());
        let other = hipStream_t(1usize as *mut _);
        let strict = true;

        assert!(null_stream_uses_global_lock(null, strict));
        assert!(null_stream_uses_global_lock(other, strict));
        assert!(null_stream_needs_pre_sync(null));
        assert!(!null_stream_needs_pre_sync(other));
        assert!(null_stream_needs_post_sync(null, strict));
        assert!(!null_stream_needs_post_sync(other, strict));
    }
}

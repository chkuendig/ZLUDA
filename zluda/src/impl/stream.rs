use cuda_types::cuda::CUresult;
use hip_runtime_sys::*;
use std::sync::Mutex;

// ROCm 7.2.4 implements legacy default-stream ordering in hip::getStream:
// null/legacy submissions enqueue waits on prior blocking streams, and blocking
// streams enqueue waits on prior legacy work. Nonblocking streams are exempt.
// Use those queued dependencies for kernels, copies, memsets and events. A
// hipDeviceSynchronize before/after a legacy operation can deadlock when an
// exempt stream waits for a value that the legacy operation will produce.
// The legacy queue is shared across host threads on the selected HIP device.
// Distinct CUDA contexts on one HIP device already share that queue; context
// isolation requires a separate context/stream mapping change.

/// Cross-kernel L1 visibility fix ("Bug A") — default ON.
///
/// On gfx10 RDNA, rocclr's default AQL `release_fence_scope = AGENT` does NOT
/// trigger the Command Processor's inter-dispatch L1 invalidation. A reader
/// kernel on a different CU can therefore read STALE L1 lines for data a writer
/// kernel stored on another stream — reproduced by `bug_a_xkernel_coherency`
/// (5/50k stale reads). In PhysX this surfaces as wild indices/counters →
/// `(u32)-N × 4` wild VA → GPU page fault.
///
/// The CP inter-dispatch L1 invalidation only runs at `HSA_FENCE_SCOPE_SYSTEM`,
/// which rocclr sets after memcpy/image ops (`gpu().addSystemScope()` in
/// rocblit.cpp). So a tiny `hipMemcpyDtoDAsync` on a stream forces the NEXT AQL
/// packet on that stream to SYSTEM scope. We inject it at the cross-stream
/// dependency points (`cuEventRecord` producer-release, `cuStreamWaitEvent`
/// consumer-acquire) so PhysX's producer→consumer hand-offs become coherent.
///
/// Validated by a decisive same-session A/B (2026-05-28): fix OFF crashed with
/// 10 GPU page faults; fix ON ran 3/3 clean (~1.5M kernels, 0 faults).
///
/// Default ON. Set `ZLUDA_XKERNEL_SYSTEM_SCOPE=0` to disable (e.g. for A/B).
/// NOTE: the D2D-to-force-SYSTEM-scope is a pragmatic mechanism; see the active
/// plan for a less hacky replacement (direct AQL fence scope / cache invalidate).
fn xkernel_system_scope_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ZLUDA_XKERNEL_SYSTEM_SCOPE").map_or(true, |v| v != "0")
    })
}

/// Lazily-allocated 8-byte device scratch for the system-scope D2D.
/// Stored as usize so the OnceLock is Send+Sync.
fn system_scope_scratch() -> Option<hipDeviceptr_t> {
    static SCRATCH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let p = *SCRATCH.get_or_init(|| {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        if unsafe { hipMalloc(&mut ptr, 8) }.is_ok() {
            ptr as usize
        } else {
            0
        }
    });
    (p != 0).then(|| hipDeviceptr_t(p as *mut std::ffi::c_void))
}

/// Force the next AQL packet on `stream` to SYSTEM fence scope via a 4-byte
/// device-to-device copy (routes through rocblit → `gpu().addSystemScope()`).
/// Active by default; no-op if `ZLUDA_XKERNEL_SYSTEM_SCOPE=0`. See `xkernel_system_scope_enabled`.
pub(crate) fn force_system_scope(stream: hipStream_t) {
    if !xkernel_system_scope_enabled() {
        return;
    }
    if let Some(scratch) = system_scope_scratch() {
        let dst = hipDeviceptr_t((scratch.0 as usize + 4) as *mut std::ffi::c_void);
        let _ = unsafe { hipMemcpyDtoDAsync(dst, scratch, 4, stream) };
    }
}

/// Resolve stream zero for the implemented CUDA _ptsz entry points.
/// Explicit legacy (1), per-thread (2) and user-created handles are preserved.
/// HIP's headers define hipStreamPerThread as the same pointer sentinel (2).
pub(crate) fn per_thread_stream(stream: hipStream_t) -> hipStream_t {
    if stream.0.is_null() {
        hipStream_t(2usize as *mut _)
    } else {
        stream
    }
}

// Track the flags passed to both creation APIs. HIP keeps the same flags for
// its dependency filtering; the registry also serves cuStreamGetFlags without
// dereferencing a possibly stale HIP handle after a successful destroy.
#[derive(Default)]
struct StreamFlags {
    streams: std::collections::HashMap<usize, (u32, u64)>,
    generation: u64,
}

impl StreamFlags {
    fn created(&mut self, stream: hipStream_t, flags: u32) {
        self.generation += 1;
        self.streams
            .insert(stream.0 as usize, (flags, self.generation));
    }

    fn destroyed(&mut self, stream: hipStream_t, generation: Option<u64>) {
        if generation.is_some() && self.generation_of(stream) == generation {
            self.streams.remove(&(stream.0 as usize));
        }
    }

    fn get(&self, stream: hipStream_t) -> Option<u32> {
        self.streams
            .get(&(stream.0 as usize))
            .map(|&(flags, _)| flags)
    }

    fn generation_of(&self, stream: hipStream_t) -> Option<u64> {
        self.streams
            .get(&(stream.0 as usize))
            .map(|&(_, generation)| generation)
    }
}

static STREAM_FLAGS: std::sync::LazyLock<Mutex<StreamFlags>> =
    std::sync::LazyLock::new(|| Mutex::new(StreamFlags::default()));

pub(crate) fn synchronize(stream: hipStream_t) -> hipError_t {
    super::driver::sync_timeline_log("stream", stream.0 as usize);
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamSynchronize({:?})", stream));
    // 2026-06-03: MS_1 rd5+22176 reset drain. Host-thread safe DtoD;
    // no-op when ZLUDA_MS1_RD5_RESET=0. Placed before the actual
    // hipStreamSynchronize so deferred resets land before downstream
    // launches see the "synchronized" state.
    super::memory::drain_ms1_rd5_reset_queue();
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
    // the hypothesis. Env-gate ZLUDA_HDP_FLUSH_ON_SYNC=1 (off by default since the retirement profile).
    static HDP_FLUSH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let hdp_flush = *HDP_FLUSH
        .get_or_init(|| std::env::var("ZLUDA_HDP_FLUSH_ON_SYNC").map_or(false, |v| v == "1"));
    if hdp_flush {
        let _ = unsafe { hipDeviceSynchronize() };
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    super::driver::zluda_log(&format!("[ZLUDA] cuStreamSynchronize -> {:?}", result));
    result
}

pub(crate) fn create(stream: *mut hipStream_t, flags: ::core::ffi::c_uint) -> CUresult {
    let result = super::signal::with_preserved_state(|| {
        unsafe { hipStreamCreateWithFlags(stream, flags) }?;
        Ok(())
    });
    if result.is_ok() {
        STREAM_FLAGS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .created(unsafe { *stream }, flags);
    }
    super::driver::zluda_log_err(&format!(
        "cuStreamCreate(flags={:#x}) -> {:?} stream={:?}",
        flags,
        result,
        if result.is_ok() {
            unsafe { *stream }
        } else {
            hipStream_t(std::ptr::null_mut())
        }
    ));
    result
}

pub(crate) fn create_with_priority(
    stream: *mut hipStream_t,
    flags: ::core::ffi::c_uint,
    priority: ::core::ffi::c_int,
) -> CUresult {
    let result = super::signal::with_preserved_state(|| {
        unsafe { hipStreamCreateWithPriority(stream, flags, priority) }?;
        Ok(())
    });
    if result.is_ok() {
        STREAM_FLAGS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .created(unsafe { *stream }, flags);
    }
    result
}

pub(crate) fn destroy_v2(stream: hipStream_t) -> hipError_t {
    let generation = STREAM_FLAGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .generation_of(stream);
    // Do not hold our metadata lock across a HIP call. A new stream may reuse
    // this address before hipStreamDestroy returns; remove only this generation.
    let result = unsafe { hipStreamDestroy(stream) };
    if result.is_ok() {
        STREAM_FLAGS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .destroyed(stream, generation);
    }
    result
}

// The generated CUDA wrapper converts this output pointer to a reference and
// rejects null with CUDA_ERROR_INVALID_VALUE before entering this function.
pub(crate) fn get_flags(stream: hipStream_t, flags: &mut u32) -> hipError_t {
    if let Some(value) = STREAM_FLAGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(stream)
    {
        *flags = value;
        return Ok(());
    }
    // The per-thread default stream is created lazily inside HIP.
    if stream.0 as usize == 2 {
        return unsafe { hipStreamGetFlags(stream, flags) };
    }
    hipError_t::ErrorInvalidHandle
}

pub(crate) fn get_flags_ptsz(stream: hipStream_t, flags: &mut u32) -> hipError_t {
    get_flags(per_thread_stream(stream), flags)
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
    let hip_result = unsafe { hipStreamQuery(stream) };
    // HIP queries the legacy queue after queuing dependencies on blocking
    // streams only. Pass that result through by default; retain the old
    // always-NotReady override solely for diagnostic comparisons.
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
            1 // default: pass through HIP
        }
    });
    let returned = if stream.0.is_null() {
        match mode {
            0 => Err(hipErrorCode_t::NotReady),
            // Default (mode 1) and REAL (mode 2): always pass through the real HIP result.
            // Evidence from 2026-06-04: masking queries as NotReady causes PhysX to take
            // the wrong dispatch path and crash at ~152k launches (Class B). Passing through
            // the real result lets PhysX get Ok() early (when HIP is actually idle at startup)
            // and take the correct "proceed" path. LEGACY override (mode 0) retained for A/B.
            _ => hip_result,
        }
    } else {
        hip_result
    };
    // 2026-06-05 Class B residual investigation: PhysX uses cuStreamQuery as
    // an UN-SYNCED "is this kernel done?" poll. Live evidence: 6,015 query
    // calls per Class B bench run, vs 51k cuEventSync calls. When the query
    // returns SUCCESS, PhysX reads mapped-host memory directly without an
    // explicit sync — but on AMD, HDP write-combine can still be holding
    // the kernel's writes. CPU sees stale data → inner_array[0]=NULL →
    // canonical Class B host crash. NVIDIA has no HDP equivalent, so this
    // works there. Mirror the Phase 2 / MEMCPY_HDP_FLUSH pattern: when the
    // query returns SUCCESS (stream is idle), trigger HDP flush via
    // hipDeviceSynchronize. On an idle stream this is fast — just an AQL
    // barrier-AND packet round-trip whose release fence does the flush.
    // Env-gate ZLUDA_QUERY_HDP_FLUSH: off by default (retirement profile), =1 enables.
    if returned == Ok(()) {
        static QUERY_HDP_FLUSH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *QUERY_HDP_FLUSH
            .get_or_init(|| std::env::var("ZLUDA_QUERY_HDP_FLUSH").map_or(false, |v| v == "1"))
        {
            unsafe {
                let _ = hipDeviceSynchronize();
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            }
        }
    }
    let count = QUERY_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Log first 200, then every 100th, or always when ZLUDA_LOG_ALL is set
    if count < 200 || count % 100 == 0 || crate::is_log_all_enabled() {
        // Show both real HIP result and what we return, to detect masking
        if stream.0.is_null() {
            super::driver::zluda_log_err(&format!(
                "cuStreamQuery(NULL) hip={:?} returned={:?} [#{}]",
                hip_result, returned, count
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
    let result = unsafe { hipStreamWaitEvent(stream, event, flags) };
    super::driver::zluda_log_err(&format!(
        "cuStreamWaitEvent(stream={:?}, event={:?}, flags={:#x}) -> {:?}",
        stream, event, flags, result
    ));
    // Consumer-acquire side of the Bug A fix (cherry-picked from 49414873):
    // after waiting on a producer stream's event, force SYSTEM scope so the
    // consumer's next kernel invalidates stale L1 lines and sees the
    // producer's stores. Env-gated ZLUDA_XKERNEL_SYSTEM_SCOPE (default-ON
    // per a9427b04 — fixes the IOFX_actorRange / TFS wild-VA TCP faults).
    force_system_scope(stream);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_thread_entry_points_resolve_only_stream_zero() {
        let resolved = per_thread_stream(hipStream_t(std::ptr::null_mut()));
        assert_eq!(resolved.0 as usize, 2);
        for handle in [1, 2, 0x1000] {
            let resolved = per_thread_stream(hipStream_t(handle as *mut _));
            assert_eq!(resolved.0 as usize, handle);
        }
    }

    #[test]
    fn stream_flags_distinguish_blocking_and_nonblocking_streams() {
        let blocking = hipStream_t(0x1000usize as *mut _);
        let nonblocking = hipStream_t(0x2000usize as *mut _);
        let mut flags = StreamFlags::default();
        flags.created(blocking, hipStreamDefault);
        flags.created(nonblocking, hipStreamNonBlocking);
        assert_eq!(flags.get(blocking), Some(0));
        assert_eq!(flags.get(nonblocking), Some(1));
        let generation = flags.generation_of(nonblocking);
        flags.destroyed(nonblocking, generation);
        assert_eq!(flags.get(nonblocking), None);
        // A reused HIP address must carry the new stream's flags.
        flags.created(nonblocking, hipStreamDefault);
        assert_eq!(flags.get(nonblocking), Some(0));
        assert_eq!(flags.get(blocking), Some(0));
    }

    #[test]
    fn delayed_destroy_does_not_forget_reused_stream_flags() {
        let stream = hipStream_t(0x1000usize as *mut _);
        let mut flags = StreamFlags::default();
        flags.created(stream, hipStreamDefault);
        let destroyed_generation = flags.generation_of(stream);
        flags.created(stream, hipStreamNonBlocking);
        flags.destroyed(stream, destroyed_generation);
        assert_eq!(flags.get(stream), Some(hipStreamNonBlocking));
        flags.destroyed(stream, None);
        assert_eq!(flags.get(stream), Some(hipStreamNonBlocking));
    }

    #[test]
    fn stream_flags_reject_invalid_arguments_without_entering_hip() {
        assert_eq!(
            unsafe { crate::cuStreamGetFlags(std::ptr::null_mut(), std::ptr::null_mut()) },
            Err(cuda_types::cuda::CUerror::INVALID_VALUE)
        );
        assert_eq!(
            unsafe { crate::cuStreamGetFlags_ptsz(std::ptr::null_mut(), std::ptr::null_mut()) },
            Err(cuda_types::cuda::CUerror::INVALID_VALUE)
        );
        let mut flags = 0;
        assert_eq!(
            get_flags(hipStream_t(0x1234usize as *mut _), &mut flags),
            hipError_t::ErrorInvalidHandle
        );
    }
}

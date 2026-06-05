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
    // CUDA NULL-stream semantics (verified against NVIDIA docs 2026-05-15):
    // "When an action is taken in the legacy stream such as a kernel launch
    // or cudaStreamWaitEvent(), the legacy stream first waits on all blocking
    // streams, the action is queued in the legacy stream, and then all
    // blocking streams wait on the legacy stream." cuEventRecord on NULL
    // qualifies as an action on the legacy stream.
    //
    // HIP does NOT enforce this. We reuse null_stream_sync (which
    // hipDeviceSynchronize's on NULL stream) to add the pre-action barrier.
    // Env-disable with ZLUDA_NULL_STREAM_BARRIER=0 for A/B testing.
    let _guard = super::stream::null_stream_sync_for_event(stream);
    // Plan v25 Phase B.2: drain DEV5/6/7 dump queue. cuEventRecord fires
    // ~91k times/run (vs ~120k for cuEventSynchronize); adding the drain
    // here shrinks the worst-case "MS_4 launched but its dump hasn't run"
    // gap from ~134 ms to ~10-20 ms. Drain is a no-op when ZLUDA_DEV5_DUMP=0.
    super::memory::drain_dev5_dump_queue();
    // MS2 batched descriptor/value drain (host-thread safe DtoH).
    super::memory::drain_ms2_dump_queue();
    // Plan v26 Phase B.3: drain IOFX_actorRange param_4 dump queue. Same
    // safety properties as the DEV5 drain; no-op when ZLUDA_ACTORRANGE_DUMP=0.
    super::memory::drain_iofx_ar_dump_queue();
    // 2026-06-03: drain MS1_RD5_RESET queue. Defers the per-cycle
    // hipMemsetD32(rd5+22176, 0, 1) out of launch_kernel (where it
    // races with the ROCm 6.2.4 ProfilingSignal pool on Wine 10.0) onto
    // this host-thread call site. No-op when ZLUDA_MS1_RD5_RESET=0.
    super::memory::drain_ms1_rd5_reset_queue();
    // Producer-release side of the Bug A fix (cherry-picked from 49414873):
    // force SYSTEM scope before recording the event so the producer stream's
    // stores are flushed to the system-visible level before downstream
    // waiters observe completion. Env-gated (ZLUDA_XKERNEL_SYSTEM_SCOPE=1
    // default-ON per a9427b04 — 3/3 runs reached 1.5M clean with this).
    super::stream::force_system_scope(stream);
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

    // Same NULL-stream barrier as cuEventRecord above.
    let _guard = super::stream::null_stream_sync_for_event(stream);
    super::stream::force_system_scope(stream);
    hipEventRecord(event, stream)
}

pub(crate) unsafe fn synchronize(event: hipEvent_t) -> hipError_t {
    super::driver::sync_timeline_log("event", event as usize);
    super::driver::zluda_log_err(&format!("cuEventSynchronize(event={:?}) ENTER", event));
    // Plan v24 Phase B.2: drain DEV5 dump queue. cuEventSynchronize is a
    // host-thread call (safe for DtoH per feedback_no_d2d_from_launch_kernel),
    // fires very frequently from PhysX (120k+ calls per run). Drain is a
    // no-op if ZLUDA_DEV5_DUMP=0 or queue is empty.
    super::memory::drain_dev5_dump_queue();
    super::memory::drain_ms2_dump_queue();
    // Plan v26 Phase B.3: drain IOFX_actorRange dump queue (sibling).
    super::memory::drain_iofx_ar_dump_queue();
    // 2026-06-03: MS_1 rd5+22176 reset drain (sibling).
    super::memory::drain_ms1_rd5_reset_queue();
    let result = hipEventSynchronize(event);
    // 2026-06-03: Class B PhysX-host vtable-iteration crash hardening.
    // Ghidra-decoded crash at PhysX3_x64.dll RVA 0x20ced9 is `MOV RAX,[RCX]`
    // first deref of a pointer pulled from a paired-iterator output array
    // returned by `[obj+0x60].vmethod[2]()`. Three callers iterate the
    // canonical PhysX `[+0xa0]=ptr_array,[+0xa8]=count` Ps::Array<T*>.
    // The crash is deterministic at ~158k–160k launches with capture probes
    // (project_class_b_physx_vtable_crash_decoded_2026_06_03.md).
    //
    // Working hypothesis: cuEventSynchronize returns once the GPU event
    // fires, but GPU writes destined for zero-copy/host-pinned memory may
    // still sit in the AMD HDP write buffer (not yet visible to CPU reads).
    // PhysX then iterates a snapshot of an actor/shape array that still
    // holds stale pointers and dereferences a freed entry. The symmetric
    // workaround in cuStreamSynchronize (ZLUDA_HDP_FLUSH_ON_SYNC, default-ON)
    // already does hipDeviceSynchronize + dirty-flag clear; cuEventSynchronize
    // did NOT, leaving the event-based completion path uncovered.
    //
    // Env-gate ZLUDA_EVENT_HDP_FLUSH. 2026-06-05: n=3 A/B with default-ON shows
    // mean 521k vs v5 baseline mean 294k (+77%), median 353k vs 214k (+65%), and
    // 0/3 Class B canonical NULL-deref vs 3/5 in baseline. Class B (host-side
    // NULL deref at PhysX3.dll+0x20ced9) is effectively eliminated. Residual
    // crash class is Class A (GPU TCP wild-VA, separate IOFX/TFS texref race).
    // Default flipped to ON; set =0 to opt out for A/B.
    static HDP_FLUSH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let hdp_flush = *HDP_FLUSH
        .get_or_init(|| std::env::var("ZLUDA_EVENT_HDP_FLUSH").map_or(true, |v| v != "0"));
    if hdp_flush {
        let _ = hipDeviceSynchronize();
        super::stream::clear_dirty_flag_after_device_sync();
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    super::driver::zluda_log_err(&format!(
        "cuEventSynchronize(event={:?}) -> {:?}",
        event, result
    ));
    result
}

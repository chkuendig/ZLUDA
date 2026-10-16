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
    // Flag values are compatible between CUDA and HIP for 0,1,2,4:
    //   0 = CU_EVENT_DEFAULT       (generic, timing + sync)
    //   1 = CU_EVENT_BLOCKING_SYNC (CPU blocks in syscall instead of spin)
    //   2 = CU_EVENT_DISABLE_TIMING (sync only, no timing)
    //   4 = CU_EVENT_INTERPROCESS  (cross-process shared-mem semantics)
    //
    // HIP-only extension (no CUDA equivalent):
    //   0x80000000 = hipEventReleaseToSystem  — explicitly documented to
    //   "Perform a system-scope release operation when the event is recorded"
    //   (HIP Coherence Control docs, ROCm 6.x/7.x:
    //   https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/
    //   hip_runtime_api/memory_management/coherence_control.html)
    //
    // 2026-06-06 Class B residual fix: OR-in `hipEventReleaseToSystem` on
    // events that PhysX will use for HOST-VISIBLE sync.
    //
    // **What's documented**:
    // - HIP: cuEventRecord on a default-flag event uses agent-scope release.
    //   `hipEventReleaseToSystem` is the explicit opt-in to system-scope
    //   release that makes mapped-host writes visible to CPU readers.
    //   (HIP Coherence Control docs, URL above.)
    // - CUDA C++ Memory Model: CUDA exposes `__threadfence_system()` and
    //   the `cuda::thread_scope_system` scope for system-coherent fences
    //   (NVIDIA CUDA Programming Guide § 5.7 CUDA C++ Memory Model:
    //   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/
    //   cuda-cpp-memory-model.html). But NVIDIA does NOT document any
    //   automatic system-scope fence at cuEventRecord boundaries — events
    //   are described as "places an event into a stream; the device records
    //   a timestamp when it reaches that event" (CUDA Runtime API docs:
    //   https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).
    //
    // **What's inferred (not formally documented)**:
    // - PhysX (compiled for NVIDIA, observably never sets a HIP-style
    //   system-release flag because none exists in CUDA's API) DOES work on
    //   NVIDIA hardware without explicit system fences around event-record
    //   patterns where host subsequently reads mapped-pinned memory.
    // - The most likely explanations: (a) NVIDIA GPUs lack the HDP
    //   write-combine buffer that AMD GPUs have, so writes that reached L2
    //   become host-visible without an explicit MMIO flush; (b) NVIDIA's
    //   default fence behavior at completion barriers may be stronger than
    //   HIP's agent-scope default. We don't have a primary NVIDIA-doc
    //   source for either; this is inference from observable behavior on
    //   gfx10 + ROCm 6.2.4 + Wine 10.0.
    //
    // **What's empirically validated** (this branch, 2026-06-06):
    // - Adding hipEventReleaseToSystem on cuEventCreate (every event PhysX
    //   creates → every cuEventRecord becomes system-scope release →
    //   HDP-flushed) eliminates Class B host NULL deref in n=2 (0/2 vs
    //   prior 14% incidence). Run2 reached 1.200M+ launches with 0 amdgpu
    //   faults — second-ever clean completion on this branch.
    //
    // **Selective rather than universal**: empirical PhysX flag distribution
    // is 31× flag=0, 5× flag=2, 1× flag=1, 0× flag=4 (INTERPROCESS).
    // INTERPROCESS events have shared-mem semantics where combining with a
    // system fence is poorly specified; PhysX never uses them anyway, so
    // we skip those. All other patterns ({0,1,2}) get the flag.
    //
    // Env-gate ZLUDA_EVENT_RELEASE_SYSTEM default-ON, opt-out with =0.
    static EVENT_RELEASE_SYSTEM: std::sync::OnceLock<bool> =
        std::sync::OnceLock::new();
    let release_system = *EVENT_RELEASE_SYSTEM
        .get_or_init(|| std::env::var("ZLUDA_EVENT_RELEASE_SYSTEM").map_or(true, |v| v != "0"));
    let is_interprocess = (flags & 4) != 0;
    let effective_flags = if release_system && !is_interprocess {
        flags | hipEventReleaseToSystem
    } else {
        flags
    };
    let result = hipEventCreateWithFlags(event, effective_flags);
    super::driver::zluda_log_err(&format!(
        "cuEventCreate(flags={} effective=0x{:x}) -> {:?} (event={:p})",
        flags,
        effective_flags,
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
    // Default flipped to ON, then retired (off by default); =1 enables.
    static HDP_FLUSH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let hdp_flush = *HDP_FLUSH
        .get_or_init(|| std::env::var("ZLUDA_EVENT_HDP_FLUSH").map_or(false, |v| v == "1"));
    if hdp_flush {
        let _ = hipDeviceSynchronize();
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    super::driver::zluda_log_err(&format!(
        "cuEventSynchronize(event={:?}) -> {:?}",
        event, result
    ));
    result
}

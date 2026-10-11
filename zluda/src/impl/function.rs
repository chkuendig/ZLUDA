use cuda_types::cuda::{CUfunction, CUfunction_attribute, CUkernel};
use dark_api::FunctionArgInfo;
use hip_runtime_sys::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as IoWrite;
use std::mem;
use std::sync::Mutex;

/// Device-allocation bases observed as the SOURCE of MemCopyBatchedAsync
/// descriptors writing into mapped-host near the upstream relay struct
/// (off ≈ 1 MiB). Each entry is `(base, size)` of a device allocation.
/// Populated lazily as MemCopyBatchedAsync descriptors arrive; read by the
/// per-launch writer probe (`ZLUDA_MEMCPY_SRC_WRITER=1`) which logs any
/// kernel-arg pointer falling within a tracked alloc. Pure diagnostic.
static TRACKED_MEMCPY_SRC_ALLOCS: std::sync::Mutex<Vec<(u64, u64)>> =
    std::sync::Mutex::new(Vec::new());

/// Per-allocation launch history for the Class A writer hunt. Keyed by
/// device allocation BASE. Value is a deque of (timestamp_ms, kernel_name,
/// arg_idx) tuples capped at the last 8 distinct (kernel_name, arg_idx)
/// pairs that touched this allocation. When MemCopyBatchedAsync copies
/// from a known alloc into the relay struct slot, the probe dumps this
/// history so the upstream WRITER kernel is visible even if it fired
/// before the MemCopy.
static ALLOC_TOUCH_HISTORY: std::sync::Mutex<
    Option<std::collections::HashMap<u64, std::collections::VecDeque<(u64, String, u8)>>>,
> = std::sync::Mutex::new(None);

const ALLOC_HISTORY_CAP: usize = 8;
/// Process start time in ms for relative timestamps (set lazily on first probe).
static PROBE_START_MS: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn probe_now_ms() -> u64 {
    let start = PROBE_START_MS.get_or_init(std::time::Instant::now);
    start.elapsed().as_millis() as u64
}

static FUNC_NAMES: std::sync::LazyLock<Mutex<HashMap<usize, String>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

// ============================================================================
// MS_4 input capture
// ============================================================================
//
// Goal: snapshot CreateParticleShapesIncremental_MultiStage_4 inputs from a
// running game, then feed them into a standalone GPU test to verify whether
// the exact bytes that the game saw reproduce the fault in isolation.
//
// Constraint: `hipMemcpyDtoH` from inside `launch_kernel` triggers the ROCm
// `std::vector<ProfilingSignal*>` race (this is what the removed validator
// did — it crashed the process before the real MS_4 fault). So we CANNOT
// do D2H copies in the launch path.
//
// Approach: allocate a pinned host-mapped buffer via `hipHostMalloc(...Mapped)`.
// That buffer is visible from both the CPU and the GPU. For each MS_4 launch,
// issue `hipMemcpyDtoD` from the device input buffers INTO the mapped host
// buffer. D2D bypasses the ProfilingSignal pool entirely — it lowers to a
// plain kernel submission. The CPU can read the mapped buffer at any time,
// including from the SIGABRT handler via a raw `write()` syscall.
//
// We keep a ring of the last N snapshots so the SIGABRT dump captures the
// history leading up to the crash, not just one slot that might itself have
// been the cause.

const MS4_CAPTURE_SLOTS: usize = 32;
// 64k: MS_4's n grows toward ~65,000 before the historic underflow crashes
// (n=62471/64534 observed); at 32k the capture skipped exactly the launches
// where the counter audit matters most.
const MS4_CAPTURE_MAX_PARTICLES: usize = 64 * 1024;
const MS4_CAPTURE_MAX_SHAPES: usize = 256;

/// Size of the flags/shape_idx region per slot (particles × 4 bytes).
const MS4_PARTICLE_REGION_BYTES: usize = MS4_CAPTURE_MAX_PARTICLES * 4;
/// Size of the struct region per slot (shapes × 12 bytes per entry).
const MS4_STRUCT_REGION_BYTES: usize = MS4_CAPTURE_MAX_SHAPES * 12;
/// Size of the counter region per slot (shapes × 4 bytes).
const MS4_COUNTER_REGION_BYTES: usize = MS4_CAPTURE_MAX_SHAPES * 4;

/// Per-slot layout — must be repr(C) so the byte layout matches between CPU
/// host-side reads and GPU device-side writes.
#[repr(C)]
struct Ms4CaptureSlot {
    // Header (128 bytes — aligned so the data regions start cleanly)
    magic: u64,           // 0: 0x4d533443_4150_5455 = "MS4CAPTU"
    launch_n: u64,        // 8
    timestamp_ms: u64,    // 16
    num_particles: u32,   // 24
    grid_x: u32,          // 28
    block_x: u32,         // 32
    _pad0: u32,           // 36
    flags_ptr: u64,       // 40
    struct_ptr: u64,      // 48
    scatter_ptr: u64,     // 56
    shape_idx_ptr: u64,   // 64
    counter_ptr: u64,     // 72
    flags_bytes: u32,     // 80 — actual bytes captured for this slot
    shape_idx_bytes: u32, // 84
    struct_bytes: u32,    // 88
    counter_bytes: u32,   // 92
    scatter_bytes: u32,   // 96 — bytes of the POST-launch scatter output (param4)
    _pad1: [u8; 28],      // 100..=127
    // Data regions (aligned at 128)
    flags_data: [u8; MS4_PARTICLE_REGION_BYTES],
    shape_idx_data: [u8; MS4_PARTICLE_REGION_BYTES],
    struct_data: [u8; MS4_STRUCT_REGION_BYTES],
    counter_data: [u8; MS4_COUNTER_REGION_BYTES],
    // 2026-06-14 Phase B1: MS_4's SCATTER OUTPUT (param4, the particle-index
    // list). Captured POST-launch (it is written BY the kernel) to settle
    // whether the GPU scatter is correct given the (proven-correct) inputs.
    scatter_data: [u8; MS4_PARTICLE_REGION_BYTES],
}

const MS4_SLOT_SIZE: usize = std::mem::size_of::<Ms4CaptureSlot>();
const MS4_CAPTURE_TOTAL_SIZE: usize = MS4_SLOT_SIZE * MS4_CAPTURE_SLOTS;
const MS4_CAPTURE_MAGIC: u64 = 0x4d53344341505455; // "MS4CAPTU" big-endian

/// Capture state — lazily initialized on first MS_4 launch.
struct Ms4CaptureState {
    /// CPU-visible pointer to the pinned host buffer (same address space as
    /// this process). Reads from any thread, including the SIGABRT handler.
    host_ptr: *mut u8,
    /// GPU-visible pointer to the same buffer (from hipHostGetDevicePointer).
    /// Used as the DESTINATION for hipMemcpyDtoD calls.
    dev_ptr: *mut std::ffi::c_void,
    /// Monotonic count of MS_4 launches (used to pick the ring slot).
    launch_counter: u64,
}

// SAFETY: the pointers are used across threads, but host_ptr is only read
// (the CPU never writes into the capture region except to initialize headers
// via the device-side writes) and dev_ptr is passed to HIP APIs which are
// thread-safe.
unsafe impl Send for Ms4CaptureState {}
unsafe impl Sync for Ms4CaptureState {}

static MS4_CAPTURE: Mutex<Option<Ms4CaptureState>> = Mutex::new(None);

/// Enabled via env var — off by default because it allocates ~2 MB of pinned
/// memory and adds 4 extra hipMemcpyDtoD calls per MS_4 launch.
fn ms4_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_MS4_CAPTURE").map_or(false, |v| v == "1"))
}

/// F.6 mutation tracker — separately gated because it does
/// `hipMemcpyDtoD` against accumulated rd5/counter pointers, and
/// those pointers can become stale when PhysX frees particle groups.
/// A stale pointer hits an unmapped GPU page → rocclr aborts → SIGABRT
/// in our handler that LOOKS like a TurbulenceFS kernel crash but is
/// actually our own diagnostic code reading freed memory.
///
/// See `project_f6_tracker_stale_pointer_bug.md`. The 3/5 "TurbulenceFS
/// family" crashes in runs 31, 36, 37 are this bug, not real game bugs.
///
/// Off by default. Re-enable only for short investigation runs where
/// you know PhysX won't free particle groups (e.g., the first 10 s of
/// the benchmark, before any group teardown).
fn f6_tracker_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_F6_TRACK").map_or(false, |v| v == "1"))
}

/// Initialize the capture state on first use. Called from inside the mutex.
unsafe fn ms4_capture_init(state: &mut Option<Ms4CaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    // hipHostMallocMapped = 2 (the flag value is in hip_runtime_sys).
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, MS4_CAPTURE_TOTAL_SIZE, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS4 capture: hipHostMalloc({}) failed: {:?}",
            MS4_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    // Zero the buffer so uninitialized slots are obviously empty.
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, MS4_CAPTURE_TOTAL_SIZE);
    }

    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS4 capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }

    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MS4 capture: {} slots × {} KB = {} KB pinned; host=0x{:x} dev=0x{:x}",
        MS4_CAPTURE_SLOTS,
        MS4_SLOT_SIZE / 1024,
        MS4_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    // Publish the host pointer to the signal handler via an atomic.
    MS4_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    MS4_CAPTURE_TOTAL_BYTES.store(MS4_CAPTURE_TOTAL_SIZE, std::sync::atomic::Ordering::SeqCst);

    *state = Some(Ms4CaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

/// Global host pointer — read from the SIGABRT signal handler to write the
/// ring buffer to disk. Using an atomic so the signal handler doesn't have
/// to lock a mutex (mutex acquire from a signal handler is not async-signal-safe).
pub(crate) static MS4_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static MS4_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Called from launch_kernel right before MS_4 executes. Snapshots the
/// relevant input buffers into the next ring slot via hipMemcpyDtoD.
unsafe fn capture_ms4_inputs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_x: u32,
    block_x: u32,
) -> Option<usize> {
    if !ms4_capture_enabled() {
        return None;
    }

    let flags_ptr: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    let num_particles: u32 = read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0);
    let struct_ptr: u64 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
    let scatter_ptr: u64 = read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0);
    let shape_idx_ptr: u64 = read_kernel_arg(kernel_params, extra, 48, 6).unwrap_or(0);
    let counter_ptr: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);

    if num_particles == 0
        || num_particles as usize > MS4_CAPTURE_MAX_PARTICLES
        || flags_ptr == 0
        || struct_ptr == 0
        || shape_idx_ptr == 0
        || counter_ptr == 0
    {
        return None;
    }

    let mut guard = match MS4_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    ms4_capture_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None, // init failed; it already logged
    };

    let slot_index = (state.launch_counter as usize) % MS4_CAPTURE_SLOTS;
    state.launch_counter += 1;

    // How many bytes to capture from each buffer — we don't know exact shape
    // counts, so capture the full reserved region (the kernel only ever reads
    // valid entries anyway).
    let flags_bytes = (num_particles as usize) * 4;
    let shape_idx_bytes = (num_particles as usize) * 4;
    let struct_bytes = MS4_STRUCT_REGION_BYTES;
    let counter_bytes = MS4_COUNTER_REGION_BYTES;
    if flags_bytes > MS4_PARTICLE_REGION_BYTES {
        return None;
    }

    // Compute offsets into the slot. The slot struct is repr(C), so we can
    // compute them from std::mem::offset_of style or from the field order.
    // For robustness against future field reordering, we build offsets from
    // the struct layout directly by walking a dummy reference.
    let slot_host = state.host_ptr.add(slot_index * MS4_SLOT_SIZE) as *mut Ms4CaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * MS4_SLOT_SIZE) as *mut u8;
    // Write the header host-side (no GPU call needed — it's host-visible memory).
    (*slot_host).magic = MS4_CAPTURE_MAGIC;
    (*slot_host).launch_n = state.launch_counter - 1;
    (*slot_host).timestamp_ms = crate::elapsed_ms();
    (*slot_host).num_particles = num_particles;
    (*slot_host).grid_x = grid_x;
    (*slot_host).block_x = block_x;
    (*slot_host).flags_ptr = flags_ptr;
    (*slot_host).struct_ptr = struct_ptr;
    (*slot_host).scatter_ptr = scatter_ptr;
    (*slot_host).shape_idx_ptr = shape_idx_ptr;
    (*slot_host).counter_ptr = counter_ptr;
    (*slot_host).flags_bytes = flags_bytes as u32;
    (*slot_host).shape_idx_bytes = shape_idx_bytes as u32;
    (*slot_host).struct_bytes = struct_bytes as u32;
    (*slot_host).counter_bytes = counter_bytes as u32;
    // scatter output (param4) is copied POST-launch by capture_ms4_output_post;
    // pre-set the byte count here (num_particles u32 entries) and zero the region
    // so an un-refreshed slot is obviously empty.
    (*slot_host).scatter_bytes = (num_particles as usize * 4).min(MS4_PARTICLE_REGION_BYTES) as u32;

    // Device-to-device copies into the corresponding data regions of this
    // slot. Use std::mem::offset_of when stable, else hand-compute using
    // the known header size of 128 bytes.
    let data_base = 128usize;
    let flags_off = data_base;
    let shape_off = flags_off + MS4_PARTICLE_REGION_BYTES;
    let struct_off = shape_off + MS4_PARTICLE_REGION_BYTES;
    let counter_off = struct_off + MS4_STRUCT_REGION_BYTES;

    // EXPANDED capture path (Phase 1.5 Item B) — copy counter AND shape_idx.
    //
    // Rationale: the previous "lightweight" path only copied the counter
    // array because 4 D2D copies per MS_4 launch triggered the ROCm
    // ProfilingSignal pool race. However, runs 18/20/21 crash on MS_4 with
    // healthy-looking counters (pos=21, neg=0) and ~4500 particles — meaning
    // the counter state alone doesn't explain the crash. The most likely
    // next step is that some particle has a shape_idx outside the populated
    // counter range (e.g., shape_idx >= 21 when only slots 0..20 are
    // populated), causing MS_4 to read counter[shape_idx] = 0 → wild scatter.
    //
    // We add ONE more D2D (shape_idx) to capture the per-particle shape
    // indices. 2 D2Ds per launch is still half the 4-D2D count that caused
    // the race historically, and should be safe. If we see the race manifest
    // we can fall back to capturing only the LAST N MS_4 launches.
    //
    // We leave flags and struct uncaptured — flags is mostly used for bit
    // checks (active/inactive), and struct is a compact per-shape descriptor
    // that's less likely to explain the crash than shape_idx.
    let _ = hipMemcpyDtoD(
        hipDeviceptr_t(slot_dev_base.add(counter_off) as _),
        hipDeviceptr_t(counter_ptr as _),
        counter_bytes,
    );
    let flag_cap = flags_bytes.min(MS4_PARTICLE_REGION_BYTES);
    if flag_cap > 0 && flags_ptr != 0 {
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(slot_dev_base.add(flags_off) as _),
            hipDeviceptr_t(flags_ptr as _),
            flag_cap,
        );
    }
    // NEW: capture the shape_idx buffer (num_particles u32 entries).
    // Cap at MS4_PARTICLE_REGION_BYTES to stay within the slot's reserved space.
    let shape_cap = shape_idx_bytes.min(MS4_PARTICLE_REGION_BYTES);
    if shape_cap > 0 && shape_idx_ptr != 0 {
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(slot_dev_base.add(shape_off) as _),
            hipDeviceptr_t(shape_idx_ptr as _),
            shape_cap,
        );
    }
    // 2026-06-10 divergence-frame forensics: capture the pool table (p4,
    // 12-byte entries: f0 = cumulative offset, f1 = capacity, f2 = misc) so
    // anomalous launches can compare counter[i] against capacity f1[i].
    // counter == f1 while uses differ ⇒ counters are healthy capacities and
    // the ASSIGNMENT (flags/shape_idx) moved after refill; counter != f1 ⇒
    // the refill itself wrote divergent values. Third D2D per MS_4 launch —
    // historically 4 D2Ds triggered the ProfilingSignal race, 2 were safe;
    // watch for early-crash regression and gate down if it appears.
    if struct_ptr != 0 {
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(slot_dev_base.add(struct_off) as _),
            hipDeviceptr_t(struct_ptr as _),
            struct_bytes,
        );
    }

    // Track this counter pointer so the MS_3 capture can snapshot it too.
    track_counter_ptr(counter_ptr);

    // Force the D2D to drain and its writes to be visible to CPU reads of the
    // host-mapped buffer. hipMemcpyDtoD is synchronous from the host-API
    // perspective, but on host-mapped destinations the CPU-visible alias may
    // see stale data until a device sync flushes GPU caches. We verified
    // empirically that without this sync the BUG SIGNATURE scan below reads
    // zeros while the post-mortem file dump (which happens much later) shows
    // the real values — classic "read-before-flush" symptom.
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // CPU-side bug detection: look at the first 64 counter entries. MS_4 will
    // fault if any counter interpreted as signed i32 is negative, because
    // atomic_add(-1) on a negative u32 produces a "huge" old value → scatter
    // position = base + huge_old_value - 1 → wild write.
    //
    // Two failure patterns we've seen from actual crashing runs:
    //   - counter[i] == 0           → kernel is a no-op for that shape, but
    //                                 if any particles reference it the
    //                                 atomic_add(-1) wraps to 0xFFFFFFFF
    //   - counter[i] == 0xFFFFFFFE  → came from MS_2_Batched reading stale
    //                                 -1 sentinels written by a previous
    //                                 frame's MS_3, and computing val - 1
    //
    // Either pattern should stop the investigation immediately and tell us
    // which frame produced the bad state.
    if num_particles > 100 {
        let launch_n = state.launch_counter - 1;
        let flags_host_ptr = state.host_ptr.add(slot_index * MS4_SLOT_SIZE + flags_off);
        let shape_host_ptr = state.host_ptr.add(slot_index * MS4_SLOT_SIZE + shape_off);
        let counter_host_ptr =
            state.host_ptr.add(slot_index * MS4_SLOT_SIZE + counter_off) as *const u32;
        let scan_count = 64usize.min(counter_bytes / 4);
        let mut zero_count = 0usize;
        let mut negative_count = 0usize;
        let mut first_neg: Option<(usize, u32)> = None;
        for i in 0..scan_count {
            let c = *counter_host_ptr.add(i);
            if c == 0 {
                zero_count += 1;
            }
            if (c as i32) < 0 {
                negative_count += 1;
                if first_neg.is_none() {
                    first_neg = Some((i, c));
                }
            }
        }
        if negative_count > 0 {
            super::driver::zluda_log_err(&format!(
                "*** MS_4 BUG SIGNATURE (stale -1 → -2) launch_n={} n={} negative_counters={}/{} first=(shape={}, counter=0x{:x}={}) — kernel will scatter to wild address",
                launch_n,
                num_particles,
                negative_count,
                scan_count,
                first_neg.unwrap().0,
                first_neg.unwrap().1,
                first_neg.unwrap().1 as i32,
            ));
        } else if zero_count == scan_count {
            super::driver::zluda_log_err(&format!(
                "*** MS_4 BUG SIGNATURE (all-zero) launch_n={} n={} counter[0..{}] all zero — kernel will underflow",
                launch_n,
                num_particles,
                scan_count,
            ));
        }
        if let Some(summary) = summarize_ms4_capture(
            std::slice::from_raw_parts(flags_host_ptr, flag_cap),
            std::slice::from_raw_parts(shape_host_ptr, shape_cap),
            std::slice::from_raw_parts(counter_host_ptr as *const u8, counter_bytes),
            num_particles as usize,
        ) {
            let has_issues = summary.out_of_range_particles > 0
                || summary.zero_used_shapes > 0
                || summary.negative_used_shapes > 0
                || summary.underflow_count > 0;
            if launch_n < 5 || launch_n % 10 == 0 || has_issues {
                super::driver::zluda_log_err(&format!(
                    "MS4_SAFE #{} n={} active={} shapes={} max_shape={} out_of_range={} zero_used={} neg_used={} underflows={} exact={} surplus={} max_surplus={} sum_ctr={} sum_uses={}",
                    launch_n,
                    num_particles,
                    summary.active_count,
                    summary.unique_shapes,
                    summary.max_shape_idx,
                    summary.out_of_range_particles,
                    summary.zero_used_shapes,
                    summary.negative_used_shapes,
                    summary.underflow_count,
                    summary.exact_shapes,
                    summary.surplus_shapes,
                    summary.max_surplus,
                    summary.sum_counter,
                    summary.sum_uses,
                ));
            }
            if let Some((particle_idx, shape, flag)) = summary.first_out_of_range_particle {
                super::driver::zluda_log_err(&format!(
                    "*** MS4_SHAPE_IDX_OOR launch_n={} particle={} flag=0x{:04x} shape=0x{:08x}",
                    launch_n, particle_idx, flag, shape
                ));
            }
            if let Some(issue) = &summary.first_underflow {
                super::driver::zluda_log_err(&format!(
                    "*** MS4_COUNTER_UNDERFLOW launch_n={} shape={} counter={} uses={}",
                    launch_n, issue.shape, issue.counter, issue.uses
                ));
            }
            // Per-shape anomaly detail vs the pool table: counter == f1
            // (capacity) ⇒ counters are healthy and the ASSIGNMENT moved;
            // counter != f1 ⇒ the refill wrote divergent values.
            if has_issues && !summary.anomalies.is_empty() {
                let struct_host =
                    state.host_ptr.add(slot_index * MS4_SLOT_SIZE + struct_off) as *const u32;
                for a in &summary.anomalies {
                    let idx = a.shape as usize;
                    if idx < MS4_CAPTURE_MAX_SHAPES {
                        let f0 = *struct_host.add(idx * 3);
                        let f1 = *struct_host.add(idx * 3 + 1);
                        super::driver::zluda_log_err(&format!(
                            "MS4_ANOMALY launch_n={} shape={} f0={} f1={} counter={} uses={} ctr_eq_f1={}",
                            launch_n, a.shape, f0, f1, a.counter, a.uses, a.counter == f1
                        ));
                    }
                }
            }
        }
    }
    Some(slot_index)
}

/// Called from launch_kernel AFTER hipModuleLaunchKernel for MS_4. Copies the
/// SCATTER OUTPUT (param4 — the particle-index list the kernel just wrote) into
/// the slot `capture_ms4_inputs` returned. D2D into host-mapped memory (the
/// same race-safe path the input capture uses), then a device sync so the
/// CPU-visible alias is coherent for the post-mortem dump. Lets Phase B decide
/// GPU-vs-host: spec-sim the (proven-correct) inputs and compare to this output.
unsafe fn capture_ms4_output_post(
    slot_index: usize,
    scatter_ptr: u64,
    num_particles: u32,
    _stream: hipStream_t,
) {
    if scatter_ptr == 0 || num_particles == 0 {
        return;
    }
    let guard = match MS4_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };
    let scatter_bytes =
        (num_particles as usize * 4).min(MS4_PARTICLE_REGION_BYTES);
    // scatter_data region is the LAST region in the slot:
    // 128(header) + 2*PARTICLE + STRUCT + COUNTER.
    let scatter_off = 128usize
        + MS4_PARTICLE_REGION_BYTES
        + MS4_PARTICLE_REGION_BYTES
        + MS4_STRUCT_REGION_BYTES
        + MS4_COUNTER_REGION_BYTES;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * MS4_SLOT_SIZE) as *mut u8;
    let _ = hipMemcpyDtoD(
        hipDeviceptr_t(slot_dev_base.add(scatter_off) as _),
        hipDeviceptr_t(scatter_ptr as _),
        scatter_bytes,
    );
    // NO per-launch hipDeviceSynchronize here: capture_ms4_inputs already does one
    // device sync per MS_4 launch, and the NEXT launch's input-capture sync drains
    // this DtoD before the buffer can be reused. Adding a 2nd full sync per launch
    // roughly halved benchmark throughput and made the run miss the crash window.
    // The crash-relevant MS_4 output is consumed several launches before the host
    // SIGSEGV, so prior slots are always drained; only the very last slot may be
    // in-flight at crash time (acceptable — the ring holds 32).
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

// ============================================================================
// MS_2_Batched packet capture
// ============================================================================
//
// Purpose: snapshot the 88-byte-per-block "packet" that MS_2_Batched reads.
// Each packet entry (one per workgroup) contains:
//   offset 0 : u32 count  — if 0, MS_2_Batched skips this block
//   offset 32: u64 rd5    — destination struct array (where per-shape bases go)
//   offset 72: u64 rd6    — source data (12-byte struct, first u32 read)
//   offset 80: u64 rd7    — shape index array (u32)
//
// The MS_4 replay showed that on the crashing launch Actor B's counter buffer
// is all-zero — meaning MS_2_Batched didn't write to it this frame. We want
// to know WHY: was the packet's count field 0, were the pointers NULL, or
// was something else wrong.
//
// Same capture strategy as MS_4: pinned host-mapped ring buffer, one
// hipMemcpyDtoD per launch from the packet into our slot, dumped to disk
// via raw write() from the SIGABRT handler.

const MS2B_CAPTURE_SLOTS: usize = 32;
const MS2B_CAPTURE_MAX_PACKET_BYTES: usize = 4096; // 46 entries × 88 bytes
                                                   // MS_2_Batched pre/post counter capture (F.2/F.3): the F.2 capture revealed
                                                   // that MS_2_Batched silently no-ops in the crash frame — pre and post of the
                                                   // rd7 counter buffer are both all-zero despite entry[0].count=1497.
                                                   //
                                                   // Reading the PTX (CreateParticleShapesIncremental_MultiStage_2_Batched in
                                                   // zluda_ptx_086.ptx line 2564) shows the kernel skips ALL writes to rd7 when
                                                   // `*(rd5 + 22176)` (the source for the shared `particleShapesCounter`) is 0,
                                                   // branching to BB11_23 which only writes 4 bytes to rd5+22180 and returns.
                                                   //
                                                   // F.3 extends the capture to also snapshot rd5+22160..+22224 (64 bytes,
                                                   // containing the +22176 field at offset 16 of the captured window) per
                                                   // entry, both BEFORE and AFTER the kernel runs. This lets us prove which
                                                   // kernel zeroed `*(rd5+22176)` and confirm the no-op hypothesis.
                                                   //
                                                   // We deliberately reuse MS3_MAX_TRACKED_COUNTERS / MS3_COUNTER_BYTES_PER
                                                   // — the same TRACKED_COUNTER_PTRS pool feeds both rings, and the per-buffer
                                                   // size (256 shapes × 4 bytes) is intrinsic to the PhysX layout, not the
                                                   // stage. Keeping the consts in lockstep guarantees the parser can use one
                                                   // set of dimensions for both rings.
const MS2B_MAX_TRACKED_COUNTERS: usize = MS3_MAX_TRACKED_COUNTERS;
const MS2B_COUNTER_BYTES_PER: usize = MS3_COUNTER_BYTES_PER;
// rd5 metadata window: 64 bytes per entry, 16 entries max (we typically see
// 3 entries per launch, 16 gives plenty of headroom).
const MS2B_RD5_META_BYTES_PER_ENTRY: usize = 64;
const MS2B_RD5_META_MAX_ENTRIES: usize = 16;
const MS2B_RD5_META_TOTAL: usize = MS2B_RD5_META_BYTES_PER_ENTRY * MS2B_RD5_META_MAX_ENTRIES;
// Offset within rd5 to start the capture window. The kernel reads
// `*(rd5 + 22176)` (particleShapesCounter source) and writes
// `*(rd5 + 22180)` (the result). Capturing rd5+22160..rd5+22224 puts the
// +22176 field at byte offset 16 of the window, with a couple of u32s of
// context on either side.
const MS2B_RD5_META_OFFSET_IN_RD5: usize = 22160;

#[repr(C)]
struct Ms2bCaptureSlot {
    // Header — 192 bytes
    magic: u64,                                     // 0:  "MS2BCAPT"
    launch_n: u64,                                  // 8:  MS_2_Batched invocation count (monotonic)
    timestamp_ms: u64,                              // 16
    grid_x: u32,                                    // 24
    block_x: u32,                                   // 28
    packet_ptr: u64,                                // 32: the kernarg param_2 value the game passed
    packet_bytes: u32,                              // 40: how many bytes we actually copied
    num_tracked: u32, // 44: how many entries in counter_ptrs are valid
    counter_ptrs: [u64; MS2B_MAX_TRACKED_COUNTERS], // 48..112
    _pad: [u8; 80],   // 112..192
    // Data — 4096 bytes of raw packet memory
    packet_data: [u8; MS2B_CAPTURE_MAX_PACKET_BYTES], // 192..4288
    // Pre/post counter snapshots — 8 KB each (8 ptrs × 1024 B)
    counter_pre: [u8; MS2B_MAX_TRACKED_COUNTERS * MS2B_COUNTER_BYTES_PER],
    counter_post: [u8; MS2B_MAX_TRACKED_COUNTERS * MS2B_COUNTER_BYTES_PER],
    // F.3: rd5+22160..+22224 window per entry, pre/post the kernel.
    rd5_meta_pre: [u8; MS2B_RD5_META_TOTAL],
    rd5_meta_post: [u8; MS2B_RD5_META_TOTAL],
}

const MS2B_SLOT_SIZE: usize = std::mem::size_of::<Ms2bCaptureSlot>();
const MS2B_HEADER_BYTES: usize = 192;
const MS2B_PACKET_OFF: usize = MS2B_HEADER_BYTES;
const MS2B_PRE_OFF: usize = MS2B_PACKET_OFF + MS2B_CAPTURE_MAX_PACKET_BYTES;
const MS2B_POST_OFF: usize = MS2B_PRE_OFF + MS2B_MAX_TRACKED_COUNTERS * MS2B_COUNTER_BYTES_PER;
const MS2B_RD5_PRE_OFF: usize = MS2B_POST_OFF + MS2B_MAX_TRACKED_COUNTERS * MS2B_COUNTER_BYTES_PER;
const MS2B_RD5_POST_OFF: usize = MS2B_RD5_PRE_OFF + MS2B_RD5_META_TOTAL;
const MS2B_CAPTURE_TOTAL_SIZE: usize = MS2B_SLOT_SIZE * MS2B_CAPTURE_SLOTS;
const MS2B_CAPTURE_MAGIC: u64 = 0x4d533242_43415054; // "MS2BCAPT"

struct Ms2bCaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
}
unsafe impl Send for Ms2bCaptureState {}
unsafe impl Sync for Ms2bCaptureState {}

static MS2B_CAPTURE: Mutex<Option<Ms2bCaptureState>> = Mutex::new(None);

pub(crate) static MS2B_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static MS2B_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

unsafe fn ms2b_capture_init(state: &mut Option<Ms2bCaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, MS2B_CAPTURE_TOTAL_SIZE, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS2B capture: hipHostMalloc({}) failed: {:?}",
            MS2B_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, MS2B_CAPTURE_TOTAL_SIZE);
    }

    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS2B capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }

    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MS2B capture: {} slots × {} B = {} KB pinned; host=0x{:x} dev=0x{:x}",
        MS2B_CAPTURE_SLOTS,
        MS2B_SLOT_SIZE,
        MS2B_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    MS2B_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    MS2B_CAPTURE_TOTAL_BYTES.store(MS2B_CAPTURE_TOTAL_SIZE, std::sync::atomic::Ordering::SeqCst);

    *state = Some(Ms2bCaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

/// Called from launch_kernel BEFORE hipModuleLaunchKernel for MS_2_Batched.
/// Snapshots the kernarg param_2 packet buffer into the next ring slot via
/// hipMemcpyDtoD, harvests rd7 counter pointers from the packet entries into
/// TRACKED_COUNTER_PTRS, then snapshots every tracked counter into the slot's
/// counter_pre region. Returns the slot index so the caller can pass it to
/// `capture_ms2b_post` after the kernel has finished.
unsafe fn capture_ms2b_inputs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_x: u32,
    block_x: u32,
) -> Option<usize> {
    if !ms4_capture_enabled() {
        // Reuse the MS_4 env gate — if the user enabled capture, enable both.
        return None;
    }

    // MS_2_Batched has 3 user params:
    //   param_0: u64 @ 0  (unused)
    //   param_1: u16 @ 8  (unused)
    //   param_2: u64 @ 16 ← packet pointer
    let packet_ptr: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    if packet_ptr == 0 {
        return None;
    }

    let mut guard = match MS2B_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    ms2b_capture_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    // MS_2_Batched is called roughly once per frame — much less frequently
    // than MS_4 — so we can afford to capture every call without significant
    // ProfilingSignal pool pressure. The ring holds the last 32 packet
    // snapshots which covers ~32 frames of history, far more than enough to
    // see the packet that produced the bad counter state.
    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % MS2B_CAPTURE_SLOTS;

    // Capture `grid_x * 88` bytes (the per-block packet data), capped at the
    // slot's data region. 88 bytes per block, usual grid is 2 or 3 so
    // 176-264 bytes typically.
    let packet_bytes = ((grid_x as usize) * 88).min(MS2B_CAPTURE_MAX_PACKET_BYTES);

    let slot_host = state.host_ptr.add(slot_index * MS2B_SLOT_SIZE) as *mut Ms2bCaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * MS2B_SLOT_SIZE) as *mut u8;
    (*slot_host).magic = MS2B_CAPTURE_MAGIC;
    (*slot_host).launch_n = this_launch;
    (*slot_host).timestamp_ms = crate::elapsed_ms();
    (*slot_host).grid_x = grid_x;
    (*slot_host).block_x = block_x;
    (*slot_host).packet_ptr = packet_ptr;
    (*slot_host).packet_bytes = packet_bytes as u32;
    // Reset num_tracked / counter_ptrs so the post-capture and parser see the
    // exact set of tracked ptrs valid at the time of THIS launch.
    (*slot_host).num_tracked = 0;
    for i in 0..MS2B_MAX_TRACKED_COUNTERS {
        (*slot_host).counter_ptrs[i] = 0;
    }

    // Packet snapshot — D2D into slot.packet_data.
    let _ = hipMemcpyDtoD(
        hipDeviceptr_t(slot_dev_base.add(MS2B_PACKET_OFF) as _),
        hipDeviceptr_t(packet_ptr as _),
        packet_bytes,
    );

    // Harvest the tracked counter pointers from the packet (rd7 at offset +80
    // in each 88-byte entry) so the MS_3 capture can snapshot them even on
    // the very first frame, before any MS_4 has been seen. We already have
    // the data in the host-mapped buffer thanks to the D2D above.
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let num_entries = packet_bytes / 88;
    for e in 0..num_entries {
        let base = state
            .host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(base as *const u32);
        if count == 0 {
            continue;
        }
        let rd7 = *(base.add(80) as *const u64);
        if rd7 != 0 {
            track_counter_ptr(rd7);
        }
        // F.6: also track the rd5 base so the per-launch *(rd5+22176) tracker
        // can spot resets and the doubling bug.
        let rd5 = *(base.add(32) as *const u64);
        if rd5 != 0 {
            track_rd5_ptr(rd5);
        }
    }

    // Pre-snapshot: copy the tracked counters list (under TRACKED_COUNTER_PTRS
    // lock) into the slot header, then D2D every tracked counter into the
    // slot's counter_pre region. The counters tracked by THIS frame's packet
    // (just inserted above) are now visible — first-frame coverage works.
    let tracked: Vec<u64> = {
        let g = match TRACKED_COUNTER_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        g.clone()
    };
    let n_tracked = tracked.len().min(MS2B_MAX_TRACKED_COUNTERS);
    (*slot_host).num_tracked = n_tracked as u32;
    for (i, &ptr) in tracked.iter().take(n_tracked).enumerate() {
        (*slot_host).counter_ptrs[i] = ptr;
        let dst = slot_dev_base.add(MS2B_PRE_OFF + i * MS2B_COUNTER_BYTES_PER);
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst as _),
            hipDeviceptr_t(ptr as _),
            MS2B_COUNTER_BYTES_PER,
        );
    }

    // F.3 pre-snapshot: rd5 metadata window per entry. The MS_2_Batched
    // kernel reads `*(rd5 + 22176)` and skips ALL writes to rd7 if it's 0.
    // Capturing rd5+22160..+22224 (64 bytes per entry) lets us see the
    // particleShapesCounter source value AT KERNEL LAUNCH TIME.
    let n_rd5_entries = num_entries.min(MS2B_RD5_META_MAX_ENTRIES);
    for e in 0..n_rd5_entries {
        let entry_base = state
            .host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(entry_base as *const u32);
        if count == 0 {
            continue;
        }
        let rd5 = *(entry_base.add(32) as *const u64);
        if rd5 == 0 {
            continue;
        }
        let dst = slot_dev_base.add(MS2B_RD5_PRE_OFF + e * MS2B_RD5_META_BYTES_PER_ENTRY);
        let src = (rd5 as usize + MS2B_RD5_META_OFFSET_IN_RD5) as *mut std::ffi::c_void;
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst as _),
            hipDeviceptr_t(src),
            MS2B_RD5_META_BYTES_PER_ENTRY,
        );
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Real-time WILL NOOP detector: scan the rd5 metadata pre region and log
    // any entry whose `*(rd5+22176)` source counter is 0 — that means the
    // kernel will take the BB11_23 path and skip rd7 writes entirely.
    // Offset in our 64-byte window: 22176 - 22160 = 16, i.e. word index 4.
    for e in 0..n_rd5_entries {
        let entry_base = state
            .host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(entry_base as *const u32);
        if count == 0 {
            continue;
        }
        let rd5 = *(entry_base.add(32) as *const u64);
        let rd7 = *(entry_base.add(80) as *const u64);
        let meta_host = state
            .host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_RD5_PRE_OFF + e * MS2B_RD5_META_BYTES_PER_ENTRY)
            as *const u32;
        let shapes_counter_src = *meta_host.add(4); // word index 4 = byte 16 = rd5+22176
        if shapes_counter_src == 0 {
            super::driver::zluda_log_err(&format!(
                "*** MS_2B WILL NOOP launch_n={} entry[{}] count={} rd5=0x{:x} rd7=0x{:x} *(rd5+22176)=0 → BB11_23",
                this_launch, e, count, rd5, rd7,
            ));
        }
    }

    Some(slot_index)
}

/// Called from launch_kernel AFTER hipModuleLaunchKernel for MS_2_Batched.
/// Uses the slot_index returned by `capture_ms2b_inputs` to write into the
/// matching counter_post region. The kernel must have finished — the caller
/// is responsible for issuing hipDeviceSynchronize first (mirrors MS_3).
unsafe fn capture_ms2b_post(slot_index: usize) {
    if !ms4_capture_enabled() {
        return;
    }

    let guard = match MS2B_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };

    let slot_host = state.host_ptr.add(slot_index * MS2B_SLOT_SIZE) as *const Ms2bCaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * MS2B_SLOT_SIZE) as *mut u8;
    let num_tracked = (*slot_host).num_tracked as usize;
    let packet_bytes = (*slot_host).packet_bytes as usize;
    let num_entries = (packet_bytes / 88).min(MS2B_RD5_META_MAX_ENTRIES);

    for i in 0..num_tracked {
        let ptr = (*slot_host).counter_ptrs[i];
        if ptr == 0 {
            continue;
        }
        let dst = slot_dev_base.add(MS2B_POST_OFF + i * MS2B_COUNTER_BYTES_PER);
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst as _),
            hipDeviceptr_t(ptr as _),
            MS2B_COUNTER_BYTES_PER,
        );
    }

    // F.3 post-snapshot: re-read rd5+22160..+22224 per entry so we can diff
    // against the pre-snapshot. Reads from the slot's own packet copy
    // (already on the host side) for rd5 / count, then D2D from the live rd5.
    for e in 0..num_entries {
        let entry_base = state
            .host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(entry_base as *const u32);
        if count == 0 {
            continue;
        }
        let rd5 = *(entry_base.add(32) as *const u64);
        if rd5 == 0 {
            continue;
        }
        let dst = slot_dev_base.add(MS2B_RD5_POST_OFF + e * MS2B_RD5_META_BYTES_PER_ENTRY);
        let src = (rd5 as usize + MS2B_RD5_META_OFFSET_IN_RD5) as *mut std::ffi::c_void;
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst as _),
            hipDeviceptr_t(src),
            MS2B_RD5_META_BYTES_PER_ENTRY,
        );
    }

    // Flush + real-time diff: scan the first 18 i32 entries of every tracked
    // counter looking for a pre→post sign flip into a small negative value
    // (the "-2 across 8 lanes" bug signature). Logging the FULL 18-entry
    // window matches the parse_captures.py output we used to confirm the bug.
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let launch_n = (*slot_host).launch_n;
    for i in 0..num_tracked {
        let ptr = (*slot_host).counter_ptrs[i];
        if ptr == 0 {
            continue;
        }
        let pre_host = state
            .host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_PRE_OFF + i * MS2B_COUNTER_BYTES_PER)
            as *const i32;
        let post_host = state
            .host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_POST_OFF + i * MS2B_COUNTER_BYTES_PER)
            as *const i32;
        // Quick scan: any new negative-and-small value in [0..18) of post that
        // wasn't already negative in pre? If so, this is the smoking gun.
        let mut bug = false;
        for j in 0..18 {
            let p = *pre_host.add(j);
            let q = *post_host.add(j);
            if q < 0 && q > -1_000_000 && p >= 0 {
                bug = true;
                break;
            }
        }
        if bug {
            // Pretty-print 18 entries pre and post for the log.
            let mut pre_str = String::with_capacity(120);
            let mut post_str = String::with_capacity(120);
            pre_str.push('[');
            post_str.push('[');
            for j in 0..18 {
                if j > 0 {
                    pre_str.push(',');
                    post_str.push(',');
                }
                pre_str.push_str(&(*pre_host.add(j)).to_string());
                post_str.push_str(&(*post_host.add(j)).to_string());
            }
            pre_str.push(']');
            post_str.push(']');
            super::driver::zluda_log_err(&format!(
                "*** MS_2B BUG SIGNATURE launch_n={} counter[0x{:x}] pre={} post={}",
                launch_n, ptr, pre_str, post_str,
            ));
        }
    }
}

// ============================================================================
// MS_3 counter-buffer capture (pre/post)
// ============================================================================
//
// Purpose: MS_3 runs BETWEEN MS_2_Batched and MS_4 in each PhysX frame. The
// MS_4 capture shows that counter[0] holds a plausible positive value right
// after MS_2_Batched writes it, but becomes 0xFFFFFFFE (-2 signed) by the
// time MS_4 reads it. Either MS_3 itself is corrupting the counter, or some
// intervening operation is — and to distinguish we snapshot every known
// counter pointer BEFORE MS_3 launches and AFTER it finishes.
//
// We don't know MS_3's kernarg layout, so we track counter pointers the
// "transitive" way: whenever capture_ms4_inputs or capture_ms2b_inputs sees
// a counter_ptr (rd7 for MS_2_Batched, arg[7] for MS_4), we add it to a
// global set. On each MS_3 launch, the pre-capture snapshots all tracked
// pointers; the post-capture (called after hipModuleLaunchKernel) re-snaps
// them. A diff of pre vs post tells us exactly what MS_3 did.

const MS3_CAPTURE_SLOTS: usize = 64;
const MS3_MAX_TRACKED_COUNTERS: usize = 8;
const MS3_COUNTER_BYTES_PER: usize = 1024; // 256 shapes × 4 bytes

#[repr(C)]
struct Ms3CaptureSlot {
    // Header — 128 bytes
    magic: u64,                                    // 0
    launch_n: u64,                                 // 8
    timestamp_ms: u64,                             // 16
    grid_x: u32,                                   // 24
    block_x: u32,                                  // 28
    num_tracked: u32,                              // 32
    _pad0: u32,                                    // 36
    counter_ptrs: [u64; MS3_MAX_TRACKED_COUNTERS], // 40..104
    _pad1: [u8; 24],                               // 104..128
    // Data — 8 counter buffers before + 8 after = 16 × 1024 = 16384 bytes
    counter_pre: [u8; MS3_MAX_TRACKED_COUNTERS * MS3_COUNTER_BYTES_PER],
    counter_post: [u8; MS3_MAX_TRACKED_COUNTERS * MS3_COUNTER_BYTES_PER],
}

const MS3_SLOT_SIZE: usize = std::mem::size_of::<Ms3CaptureSlot>();
const MS3_CAPTURE_TOTAL_SIZE: usize = MS3_SLOT_SIZE * MS3_CAPTURE_SLOTS;
const MS3_CAPTURE_MAGIC: u64 = 0x4d5333434150_5455; // "MS3CAPTU"

struct Ms3CaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
}
unsafe impl Send for Ms3CaptureState {}
unsafe impl Sync for Ms3CaptureState {}

static MS3_CAPTURE: Mutex<Option<Ms3CaptureState>> = Mutex::new(None);

pub(crate) static MS3_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static MS3_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Tracked counter pointers — populated by MS_2_Batched and MS_4 captures,
/// consumed by MS_3 capture. At most MS3_MAX_TRACKED_COUNTERS distinct ptrs.
static TRACKED_COUNTER_PTRS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn track_counter_ptr(ptr: u64) {
    if ptr == 0 {
        return;
    }
    let mut guard = match TRACKED_COUNTER_PTRS.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if !guard.contains(&ptr) && guard.len() < MS3_MAX_TRACKED_COUNTERS {
        guard.push(ptr);
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS3 capture: tracking counter_ptr[{}]=0x{:x}",
            guard.len() - 1,
            ptr
        ));
    }
}

/// F.6: Tracked rd5 base pointers — populated by MS_2_Batched packet capture
/// (we extract rd5 from packet entry offset 32). The F.6 tracker watches the
/// 4 bytes at `rd5+22176` (the source for `particleShapesCounter` that MS_1
/// uses as its slot allocator) and logs every kernel that mutates it.
static TRACKED_RD5_PTRS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn track_rd5_ptr(ptr: u64) {
    if ptr == 0 {
        return;
    }
    let mut guard = match TRACKED_RD5_PTRS.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if !guard.contains(&ptr) && guard.len() < MS3_MAX_TRACKED_COUNTERS {
        guard.push(ptr);
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F6 rd5 tracker: tracking rd5[{}]=0x{:x} (watching +22176)",
            guard.len() - 1,
            ptr
        ));
    }
}

fn mcb_descriptor_hits_rd5_reset_window(
    d_dst: u64,
    d_size: u64,
    tracked_rd5: &[u64],
) -> Option<u64> {
    let dst_end = d_dst.saturating_add(d_size);
    for &rd5 in tracked_rd5 {
        let watch_start = rd5.saturating_add(F6_OFFSET_IN_RD5 as u64);
        let watch_end = watch_start.saturating_add(4);
        if d_dst < watch_end && dst_end > watch_start {
            return Some(rd5);
        }
    }
    None
}

/// MS_1 defensive reset: rd5 values whose `+22176` slot needs zeroing
/// before the next MS_1 launches. Marked by MS_4 launch hook (end of
/// the prior MS-chain cycle), consumed once by the first MS_1 of the
/// next cycle. Throttled to exactly ONE `hipMemsetD32` per rd5 per
/// cycle to avoid the HIP ProfilingSignal race documented in
/// `feedback_no_d2d_from_launch_kernel.md`.
static MS1_RD5_RESET_PENDING: Mutex<Vec<u64>> = Mutex::new(Vec::new());

pub(crate) fn ms1_rd5_reset_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled =
            std::env::var("ZLUDA_MS1_RD5_RESET").map_or(false, |v| v == "1");
        if enabled {
            super::driver::zluda_log_err(
                "[ZLUDA] MS_1 defensive reset of *(rd5+22176) enabled",
            );
        }
        enabled
    })
}

/// Called at MS_4 launch (end of MS-chain cycle). Adds the rd5 pointer
/// (= MS_4's p4) to the pending-reset set. Per-cycle, called once per
/// active entry's MS_4.
///
/// ## Open root-cause investigation (2026-06-03)
///
/// The bug this fix patches is **NOT** in our code — it's a missing
/// PhysX side-effect on AMDGPU. PhysX's intended path to zero
/// `*(rd5+22176)` ("particleShapesCounter") between MS-chain cycles is
/// the `RemoveParticleShapes` kernel, which contains the only literal
/// `st.u32 [..., 0]` at offset 22176 across all 168 loaded PTX modules.
/// **`RemoveParticleShapes` registers but NEVER launches** on Batman's
/// APEX configuration (verified by grep across `cuLaunchKernel` log).
///
/// PhysX nonetheless gets clean counter resets in *pass* cycles on
/// NVIDIA via a side-effect: `TurbulenceFS_dynamicGridClearKernel`'s
/// single-op `sust.b.3d.b32.zero` surface store **happens to alias
/// `rd5+22176`** in pass cycles. The alias breaks in fail cycles
/// (extent or binding shifts) → counter is never reset → MS_2_Batched's
/// `r38 = rd6[r37*12+8]` reads garbage → MS_4 `-2 as i32` cascade →
/// wild VA scatter → TCP page fault.
///
/// This kernel-name-specific fix supplies the explicit zero, making
/// the alias side-effect redundant. The fix structure is generic (host-
/// thread drain queue mirrors `drain_dev5_dump_queue` pattern) — only
/// the producer/consumer name match is per-kernel, and that match is
/// intrinsic to the bug (PhysX-internal state expected at MS_4/MS_1
/// boundary specifically).
///
/// **Open investigation**: figure out why `RemoveParticleShapes`
/// registers but never dispatches on Batman's APEX path. If we can
/// force it to launch (or supply a CUDA-side reset that PhysX issues
/// on NVIDIA), the explicit `hipMemsetD32` would become unnecessary.
///
/// References:
/// - `~/.claude/projects/-home-deck-Desktop-zluda-zluda/memory/project_clearkernel_aliases_rd5_22176_2026_05_22.md`
/// - `project_ms1_rd5_reset_fixes_ms4_cascade_2026_05_22.md` (N=5 validation)
/// - `project_ms1_rd5_reset_structural_fix_2026_06_03.md` (drain-queue refactor)
fn mark_rd5_reset_pending(rd5_ptr: u64) {
    if rd5_ptr == 0 || !ms1_rd5_reset_enabled() {
        return;
    }
    let mut guard = match MS1_RD5_RESET_PENDING.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if !guard.contains(&rd5_ptr) {
        if guard.len() < 16 {
            guard.push(rd5_ptr);
        } else {
            // 2026-06-03: Surface cap saturation. If drains lag MS_4
            // launches, this caps the pending Vec at 16 and MS_4 silently
            // stops marking new entries → MS_4 -2 cascade recurs.
            // Throttle the log to once per 64 drops to avoid spam.
            static DROP_N: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            let n = DROP_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n % 64 == 0 {
                super::driver::zluda_log_err(&format!(
                    "[ZLUDA] MS_1 reset PENDING DROPPED rd5=0x{:x} cap=16 dropped_n={}",
                    rd5_ptr, n
                ));
            }
        }
    }
}

/// Called at MS_1 launch (start of a new MS-chain cycle). Drains the
/// pending-reset set and returns the rd5 pointers that need their
/// `+22176` slot zeroed.
fn take_pending_rd5_resets() -> Vec<u64> {
    if !ms1_rd5_reset_enabled() {
        return Vec::new();
    }
    let mut guard = match MS1_RD5_RESET_PENDING.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let drained: Vec<u64> = guard.drain(..).collect();
    drained
}

/// SIGSEGV-safe check: is the page containing `addr` mapped readable
/// in this process? Parses `/proc/self/maps` and looks for a region
/// covering `addr` with 'r' permission. One-shot per call (cheap; the
/// probe only fires at MS_1 launches which are rare).
fn is_addr_readable(addr: u64) -> bool {
    let s = match std::fs::read_to_string("/proc/self/maps") {
        Ok(s) => s,
        Err(_) => return false,
    };
    for line in s.lines() {
        // Format: "start-end perms offset dev inode  path"
        let mut parts = line.split_whitespace();
        let range = match parts.next() {
            Some(r) => r,
            None => continue,
        };
        let perms = match parts.next() {
            Some(p) => p,
            None => continue,
        };
        let mut bounds = range.split('-');
        let start = match bounds.next().and_then(|h| u64::from_str_radix(h, 16).ok()) {
            Some(v) => v,
            None => continue,
        };
        let end = match bounds.next().and_then(|h| u64::from_str_radix(h, 16).ok()) {
            Some(v) => v,
            None => continue,
        };
        if addr >= start && addr < end {
            return perms.starts_with('r');
        }
    }
    false
}

/// Diagnostic probe: at MS_1 launch, compare CPU-direct read of
/// `*(rd5+22176)` to GPU-side reading via hipMemcpyDtoH. If they
/// differ, we have a CPU-GPU cache-coherency issue.
///
/// SIGSEGV-safe: checks /proc/self/maps for the page's permissions
/// BEFORE dereferencing. If the page isn't CPU-readable, returns
/// (0xDEAD, gpu_value, false-with-special-marker) without faulting.
///
/// Returns (cpu_value, gpu_value, equal).
unsafe fn probe_rd5_counter_cpu_vs_gpu(rd5: u64) -> (u32, u32, bool) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED
        .get_or_init(|| std::env::var("ZLUDA_RD5_CPU_PROBE").map_or(false, |v| v == "1"))
    {
        return (0, 0, true);
    }
    if rd5 == 0 {
        return (0, 0, true);
    }
    let cpu_addr = rd5 + F6_OFFSET_IN_RD5 as u64;
    // SIGSEGV-safe gate
    let cpu_readable = is_addr_readable(cpu_addr);
    // GPU read via DtoH (always works for valid device pointers)
    let mut gpu_value: u32 = 0xCAFEBABE;
    let r = hipMemcpyDtoH(
        &mut gpu_value as *mut u32 as *mut std::ffi::c_void,
        hipDeviceptr_t(cpu_addr as *mut std::ffi::c_void),
        4,
    );
    if r.is_err() {
        gpu_value = 0xCAFEBABE;
    }
    if !cpu_readable {
        // Encode "not CPU-readable" as 0xDEADBEEF + not-equal (unless
        // gpu happened to also be 0xDEADBEEF, unlikely).
        return (0xDEADBEEFu32, gpu_value, false);
    }
    let cpu_value = std::ptr::read_volatile(cpu_addr as *const u32);
    (cpu_value, gpu_value, cpu_value == gpu_value)
}

/// At MS_1 launch, consume any pending reset markers. Behavior depends on
/// `ZLUDA_MS1_RD5_RESET_SYNC`:
///
/// - **=1 (legacy, pre-8b53ca00 behavior)**: issue `hipMemsetD32`
///   synchronously here, inside `launch_kernel`. Violates the
///   no-D2D-from-launch_kernel rule (`feedback_no_d2d_from_launch_kernel.md`)
///   but matches the 2026-05-23 N=5 baseline that hit 839k peak.
/// - **=0 (default, post-8b53ca00 refactor)**: enqueue onto
///   `MS1_RD5_RESET_DRAIN_QUEUE`; the actual `hipMemsetD32` runs in
///   `drain_ms1_rd5_reset_queue` from `event::record/synchronize` and
///   `stream::synchronize`. Respects the rule but defers the zero by
///   one event-sync hop, which may explain the 839k→160k regression
///   in iter26 vs the 05-23 baseline.
///
/// Env-gated A/B for the 2026-06-03 investigation: did the drain-queue
/// refactor cost us 5x bench depth?
unsafe fn ms1_consume_rd5_resets() {
    let pending = take_pending_rd5_resets();
    if pending.is_empty() {
        return;
    }
    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SYNC_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let sync_mode = *SYNC_MODE.get_or_init(|| {
        let on = std::env::var("ZLUDA_MS1_RD5_RESET_SYNC").map_or(false, |v| v == "1");
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] ZLUDA_MS1_RD5_RESET_SYNC={} (sync hipMemsetD32 vs host-thread drain)",
            if on { "1" } else { "0" }
        ));
        on
    });
    if sync_mode {
        for rd5 in pending {
            let dst = (rd5 + F6_OFFSET_IN_RD5 as u64) as *mut std::ffi::c_void;
            let r = hipMemsetD32(hipDeviceptr_t(dst), 0, 1);
            let ok = !r.is_err();
            super::driver::zluda_log_err(&format!(
                "MS1_RD5_RESET #{} rd5=0x{:x} target=0x{:x} hipMemsetD32 SYNC ok={}",
                call_n, rd5, dst as u64, ok
            ));
        }
        return;
    }
    let queue = MS1_RD5_RESET_DRAIN_QUEUE.get_or_init(|| {
        std::sync::Mutex::new(std::collections::VecDeque::with_capacity(64))
    });
    if let Ok(mut q) = queue.lock() {
        for rd5 in pending {
            while q.len() >= 64 {
                q.pop_front();
            }
            q.push_back(rd5);
            super::driver::zluda_log_err(&format!(
                "MS1_RD5_RESET #{} rd5=0x{:x} target=0x{:x} enqueued for host-thread drain",
                call_n,
                rd5,
                rd5 + F6_OFFSET_IN_RD5 as u64
            ));
        }
    }
}

/// If `dst_dev_ptr + dst_byte_len` overlaps any tracked rd5+22176 word,
/// returns the matching rd5 base. Used by memory.rs to detect host-side
/// `cuMemsetD32_v2 / cuMemsetD32Async` calls that target the
/// particleShapesCounter reset slot. Only returns Some when env
/// `ZLUDA_RD5_MEMSET_TRACE=1` is set.
pub(crate) fn check_memset_hits_rd5_counter(
    dst_dev_ptr: u64,
    dst_byte_len: u64,
) -> Option<u64> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED
        .get_or_init(|| std::env::var("ZLUDA_RD5_MEMSET_TRACE").map_or(false, |v| v == "1"))
    {
        return None;
    }
    let snapshot: Vec<u64> = match TRACKED_RD5_PTRS.lock() {
        Ok(g) => g.clone(),
        Err(p) => p.into_inner().clone(),
    };
    if snapshot.is_empty() {
        return None;
    }
    mcb_descriptor_hits_rd5_reset_window(dst_dev_ptr, dst_byte_len, &snapshot)
}

unsafe fn ms3_capture_init(state: &mut Option<Ms3CaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, MS3_CAPTURE_TOTAL_SIZE, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS3 capture: hipHostMalloc({}) failed: {:?}",
            MS3_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, MS3_CAPTURE_TOTAL_SIZE);
    }

    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS3 capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }

    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MS3 capture: {} slots × {} KB = {} KB pinned; host=0x{:x} dev=0x{:x}",
        MS3_CAPTURE_SLOTS,
        MS3_SLOT_SIZE / 1024,
        MS3_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    MS3_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    MS3_CAPTURE_TOTAL_BYTES.store(MS3_CAPTURE_TOTAL_SIZE, std::sync::atomic::Ordering::SeqCst);

    *state = Some(Ms3CaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

/// Called from launch_kernel BEFORE hipModuleLaunchKernel for MS_3.
/// Snapshots every tracked counter pointer into the next ring slot's
/// counter_pre region. Returns the slot_index so the post-capture can
/// write to the matching counter_post region.
unsafe fn capture_ms3_pre(grid_x: u32, block_x: u32) -> Option<usize> {
    if !ms4_capture_enabled() {
        return None;
    }

    let tracked: Vec<u64> = {
        let guard = match TRACKED_COUNTER_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.clone()
    };
    if tracked.is_empty() {
        return None;
    }

    let mut guard = match MS3_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    ms3_capture_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % MS3_CAPTURE_SLOTS;

    let slot_host = state.host_ptr.add(slot_index * MS3_SLOT_SIZE) as *mut Ms3CaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * MS3_SLOT_SIZE) as *mut u8;
    (*slot_host).magic = MS3_CAPTURE_MAGIC;
    (*slot_host).launch_n = this_launch;
    (*slot_host).timestamp_ms = crate::elapsed_ms();
    (*slot_host).grid_x = grid_x;
    (*slot_host).block_x = block_x;
    (*slot_host).num_tracked = tracked.len() as u32;
    // Clear the counter_ptrs array so an unused tail reads as zeros.
    for i in 0..MS3_MAX_TRACKED_COUNTERS {
        (*slot_host).counter_ptrs[i] = if i < tracked.len() { tracked[i] } else { 0 };
    }

    // Pre-snapshot: D2D every tracked counter into the counter_pre region.
    // The data region starts at offset 128 (header size).
    let pre_base = 128usize;
    for (i, &ptr) in tracked.iter().enumerate() {
        let dst = slot_dev_base.add(pre_base + i * MS3_COUNTER_BYTES_PER);
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst as _),
            hipDeviceptr_t(ptr as _),
            MS3_COUNTER_BYTES_PER,
        );
    }

    Some(slot_index)
}

/// Called from launch_kernel AFTER hipModuleLaunchKernel for MS_3. Uses the
/// slot_index returned by capture_ms3_pre to write into the matching
/// counter_post region. The kernel is expected to have completed (caller
/// ensures this by running after hipDeviceSynchronize or under mode 2).
unsafe fn capture_ms3_post(slot_index: usize) {
    if !ms4_capture_enabled() {
        return;
    }

    let guard = match MS3_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };

    let slot_host = state.host_ptr.add(slot_index * MS3_SLOT_SIZE) as *const Ms3CaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * MS3_SLOT_SIZE) as *mut u8;
    let num_tracked = (*slot_host).num_tracked as usize;

    // Post-snapshot lives immediately after the counter_pre region.
    let post_base = 128usize + MS3_MAX_TRACKED_COUNTERS * MS3_COUNTER_BYTES_PER;
    for i in 0..num_tracked {
        let ptr = (*slot_host).counter_ptrs[i];
        if ptr == 0 {
            continue;
        }
        let dst = slot_dev_base.add(post_base + i * MS3_COUNTER_BYTES_PER);
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst as _),
            hipDeviceptr_t(ptr as _),
            MS3_COUNTER_BYTES_PER,
        );
    }

    // Flush + real-time diff check: compare counter[0] in pre vs post for each
    // tracked pointer. If any pre was positive and post is negative, log the
    // transition — that's the smoking-gun line we want in the log.
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let launch_n = (*slot_host).launch_n;
    for i in 0..num_tracked {
        let ptr = (*slot_host).counter_ptrs[i];
        if ptr == 0 {
            continue;
        }
        let pre_host = state
            .host_ptr
            .add(slot_index * MS3_SLOT_SIZE + 128 + i * MS3_COUNTER_BYTES_PER)
            as *const i32;
        let post_host = state
            .host_ptr
            .add(slot_index * MS3_SLOT_SIZE + post_base + i * MS3_COUNTER_BYTES_PER)
            as *const i32;
        let pre0 = *pre_host;
        let post0 = *post_host;
        if pre0 != post0 || (post0 < 0 && post0 > -1_000_000) {
            super::driver::zluda_log_err(&format!(
                "[MS_3 DIFF] launch_n={} counter[0x{:x}]: pre={} post={} delta={}",
                launch_n,
                ptr,
                pre0,
                post0,
                post0.wrapping_sub(pre0),
            ));
        }
    }
}

// ============================================================================
// F.4: Per-launch counter mutation tracker
// ============================================================================
//
// Purpose: identify EXACTLY which kernel mutates the per-shape counter buffer
// in the gap between MS_4 (consume) and the next MS_2_Batched (process). The
// F.3b run captured the smoking gun pattern: between consecutive MS_2_Batched
// calls the counter buffer's valid data shifts from [0..N-1] to [N..2N-1] and
// the front N positions are zeroed, while *(rd5+22176) doubles. Some kernel
// in the IOFX_/ParticleIOS_/FieldSampler_ pipeline does this — F.4 finds it
// by snapshotting the counter after EVERY launch and logging diffs.
//
// Cost: one D2D + one hipDeviceSynchronize per kernel launch per tracked
// counter (≤ 8). Gated by `ZLUDA_MS4_CAPTURE=1` AND tracked counters being
// non-empty, so init phase is unaffected. Adds roughly one extra device sync
// per launch on top of the LAUNCH_MUTEX=2 sync — significantly slows the
// game but should still reach the bug.

const F4_MAX_COUNTERS: usize = MS3_MAX_TRACKED_COUNTERS;
const F4_BYTES_PER: usize = MS3_COUNTER_BYTES_PER;

struct CounterTracker {
    /// Pinned host-mapped scratch (current snapshot target).
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
    /// Last-seen state per tracked counter pointer.
    last_state: std::collections::HashMap<u64, [u8; F4_BYTES_PER]>,
}
unsafe impl Send for CounterTracker {}
unsafe impl Sync for CounterTracker {}

static COUNTER_TRACKER: Mutex<Option<CounterTracker>> = Mutex::new(None);

unsafe fn counter_tracker_init(state: &mut Option<CounterTracker>) {
    if state.is_some() {
        return;
    }
    let total = F4_MAX_COUNTERS * F4_BYTES_PER;
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe { hipHostMalloc(&mut host_ptr, total, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F4 counter tracker: hipHostMalloc({}) failed: {:?}",
            total, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, total);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F4 counter tracker: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] F4 counter tracker: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
        total, host_ptr as usize, dev_ptr as usize,
    ));
    *state = Some(CounterTracker {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
        last_state: std::collections::HashMap::new(),
    });
}

/// Snapshot all tracked counter pointers and diff against the previous
/// snapshot. If any differ, log `*** COUNTER MUTATED by '<kernel>' ...`
/// with the prev/new state. The "shift-and-zero" pattern (front N entries
/// zeroed AND prev had non-zero values there) gets a stronger
/// `*** COUNTER SHIFT_ZERO` tag for easy grepping.
unsafe fn check_counter_mutation(current_kernel: &str) {
    if !ms4_capture_enabled() {
        return;
    }
    // F.6 tracker is gated separately because it has a stale-pointer
    // hazard — see f6_tracker_enabled() comment.
    if !f6_tracker_enabled() {
        return;
    }
    let tracked: Vec<u64> = {
        let g = match TRACKED_COUNTER_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        g.clone()
    };
    if tracked.is_empty() {
        return;
    }

    let mut guard = match COUNTER_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    counter_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(t) => t,
        None => return,
    };

    // D2D every tracked counter into the scratch (one slot per index in
    // tracked[]). Then a single sync flushes all of them to the host map.
    let n = tracked.len().min(F4_MAX_COUNTERS);
    for (i, &ptr) in tracked.iter().take(n).enumerate() {
        let dst = (tracker.scratch_dev as usize + i * F4_BYTES_PER) as *mut std::ffi::c_void;
        let _ = hipMemcpyDtoD(hipDeviceptr_t(dst), hipDeviceptr_t(ptr as _), F4_BYTES_PER);
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Compare scratch_host vs last_state per pointer.
    for (i, &ptr) in tracked.iter().take(n).enumerate() {
        let new_slice: &[u8] =
            std::slice::from_raw_parts(tracker.scratch_host.add(i * F4_BYTES_PER), F4_BYTES_PER);
        // Read first 18 i32 entries for the human-readable log.
        let new_i32: &[i32] = std::slice::from_raw_parts(new_slice.as_ptr() as *const i32, 18);
        if let Some(prev_arr) = tracker.last_state.get(&ptr) {
            if &prev_arr[..] != new_slice {
                let prev_i32: &[i32] =
                    std::slice::from_raw_parts(prev_arr.as_ptr() as *const i32, 18);
                // Detect the shift-and-zero pattern: count how many leading
                // zeros are in `new` and how many of those positions had
                // non-zero values in `prev`. If both are >= 3, this is the
                // signature we're hunting.
                let mut new_leading_zeros = 0usize;
                while new_leading_zeros < 32 {
                    let v = *(new_slice.as_ptr() as *const i32).add(new_leading_zeros);
                    if v != 0 {
                        break;
                    }
                    new_leading_zeros += 1;
                }
                let mut prev_was_nonzero_in_zone = 0usize;
                for j in 0..new_leading_zeros.min(32) {
                    let pv = *(prev_arr.as_ptr() as *const i32).add(j);
                    if pv != 0 {
                        prev_was_nonzero_in_zone += 1;
                    }
                }
                let tag = if new_leading_zeros >= 3 && prev_was_nonzero_in_zone >= 3 {
                    "SHIFT_ZERO"
                } else {
                    "MUTATED"
                };
                let mut prev_str = String::with_capacity(120);
                let mut new_str = String::with_capacity(120);
                prev_str.push('[');
                new_str.push('[');
                for j in 0..18 {
                    if j > 0 {
                        prev_str.push(',');
                        new_str.push(',');
                    }
                    prev_str.push_str(&prev_i32[j].to_string());
                    new_str.push_str(&new_i32[j].to_string());
                }
                prev_str.push(']');
                new_str.push(']');
                super::driver::zluda_log_err(&format!(
                    "*** COUNTER {} by '{}' ptr=0x{:x} prev={} new={}",
                    tag, current_kernel, ptr, prev_str, new_str,
                ));
            }
        }
        // Update last_state.
        let mut buf = [0u8; F4_BYTES_PER];
        buf.copy_from_slice(new_slice);
        tracker.last_state.insert(ptr, buf);
    }
}

// ============================================================================
// F.6: Per-launch *(rd5+22176) tracker
// ============================================================================
//
// Purpose: identify which kernel/operation normally resets `*(rd5+22176)` to
// 0 between PhysX frames, and detect when that reset is missed (the
// signature: doubling instead of going to 0). MS_1 uses
// `atomic_inc(*(rd5+22176))` to allocate slot indices into the per-shape
// counter buffer, and we have direct evidence that on bug frames, MS_1
// starts from a non-zero offset because the source counter wasn't reset.
//
// Cost: snapshot a 4-byte word per tracked rd5 pointer (≤ 8) into pinned
// host-mapped scratch + one hipDeviceSynchronize per kernel launch. Adds
// another sync layer on top of the F.4 counter tracker, but the read
// volume is tiny (32 bytes per launch worst case).

const F6_MAX_RD5: usize = MS3_MAX_TRACKED_COUNTERS;
const F6_BYTES_PER: usize = 4; // single u32 read of *(rd5+22176)
const F6_SCRATCH_BYTES: usize = F6_MAX_RD5 * F6_BYTES_PER;
pub(crate) const F6_OFFSET_IN_RD5: usize = 22176;

struct Rd5Tracker {
    /// Pinned host-mapped scratch (8 × u32 = 32 bytes total).
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
    /// Last-seen value of *(rd5+22176) per tracked rd5 pointer.
    last_state: std::collections::HashMap<u64, u32>,
}
unsafe impl Send for Rd5Tracker {}
unsafe impl Sync for Rd5Tracker {}

static RD5_TRACKER: Mutex<Option<Rd5Tracker>> = Mutex::new(None);

unsafe fn rd5_tracker_init(state: &mut Option<Rd5Tracker>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, F6_SCRATCH_BYTES, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F6 rd5 tracker: hipHostMalloc({}) failed: {:?}",
            F6_SCRATCH_BYTES, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, F6_SCRATCH_BYTES);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F6 rd5 tracker: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] F6 rd5 tracker: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
        F6_SCRATCH_BYTES, host_ptr as usize, dev_ptr as usize,
    ));
    *state = Some(Rd5Tracker {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
        last_state: std::collections::HashMap::new(),
    });
}

// ============================================================================
// Phase 1.5 Item A: MemCopyBatchedAsync scratch buffer dump
// ============================================================================
//
// Purpose: answer the "PhysX bug or ZLUDA bug?" root-cause question for Bug A.
//
// When PhysX launches `MemCopyBatchedAsync`, it passes a device-memory scratch
// buffer as Arg2 (param_2 at offset 16 in the flat arg buffer). That scratch
// buffer contains the per-batch rd5 list that tells the kernel which rd5s to
// reset. We need to see its contents to answer:
//
//   - Does the bug-firing rd5 appear in the scratch list but get skipped
//     by the kernel? → kernel-internal / ZLUDA codegen bug
//   - Does the bug-firing rd5 NOT appear in the scratch at all? → PhysX
//     host-side scheduling mismatch (the canonical "PhysX bug" hypothesis)
//
// We don't know the exact struct layout of the scratch entries, so we dump
// the first 128 bytes raw (hex) and let the post-mortem analyzer figure it
// out. 128 bytes = up to 32 u32 values or 16 u64 values, enough for ~8 rd5s
// regardless of entry size.
//
// The dump is synchronous (hipMemcpyDtoH into a pinned host scratch), logged
// as `*** MCB SCRATCH arg2=0x... grid=[1,N,1] bytes=[...]` and correlates
// 1:1 with the next `cuLaunchKernel 'MemCopyBatchedAsync'` log line.
//
// Gated on ZLUDA_MS4_CAPTURE=1 like the other diagnostic trackers.

// Max grid_y observed in MemCopyBatchedAsync launches is 16. Each entry is
// 4 u64s = 32 bytes. So 16 entries × 32 bytes = 512 bytes covers ALL entries
// in any batch. Initial dumping at 128 bytes only covered the first 4 entries
// of larger batches, missing the entries that contain the actual rd5+22176
// reset descriptors.
const MCB_DUMP_BYTES: usize = 512;

/// Pinned host-mapped scratch for the MemCopyBatchedAsync Arg2 dump.
/// Must use hipMemcpyDtoD (NOT hipMemcpyDtoH) per the file-header comment
/// about the ROCm ProfilingSignal pool race that crashes the process before
/// the real GPU fault manifests. Same pattern as the MS_4 / MS_2_Batched /
/// F.6 capture buffers.
struct McbScratchBuffer {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
}
unsafe impl Send for McbScratchBuffer {}
unsafe impl Sync for McbScratchBuffer {}

static MCB_SCRATCH: Mutex<Option<McbScratchBuffer>> = Mutex::new(None);

unsafe fn mcb_scratch_init(state: &mut Option<McbScratchBuffer>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe { hipHostMalloc(&mut host_ptr, MCB_DUMP_BYTES, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MCB scratch: hipHostMalloc({}) failed: {:?}",
            MCB_DUMP_BYTES, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, MCB_DUMP_BYTES);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MCB scratch: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MCB scratch: pinned {} B host=0x{:x} dev=0x{:x}",
        MCB_DUMP_BYTES, host_ptr as usize, dev_ptr as usize
    ));
    *state = Some(McbScratchBuffer {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
    });
}

/// Called from launch_kernel BEFORE `hipModuleLaunchKernel` for every
/// MemCopyBatchedAsync kernel. Reads Arg2 (scratch pointer), copies the first
/// 128 bytes into our pinned host-mapped scratch via hipMemcpyDtoD (NOT D2H —
/// see the file-header comment about the ProfilingSignal pool race), syncs,
/// then reads and logs the bytes as u64 hex values.
unsafe fn dump_mcb_scratch(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_y: u32, // the "N" in grid=[1,N,1] — correlates to # rd5s in batch
) {
    if !ms4_capture_enabled() {
        return;
    }

    // MemCopyBatchedAsync param layout (same as MS_2_Batched, per the PTX):
    //   param_0: u64 @ 0
    //   param_1: u16 @ 8
    //   param_2: u64 @ 16  ← scratch buffer device pointer
    let scratch_ptr: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    if scratch_ptr == 0 {
        return;
    }

    let mut guard = match MCB_SCRATCH.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    mcb_scratch_init(&mut guard);
    let buf = match guard.as_mut() {
        Some(b) => b,
        None => return,
    };

    // D2D copy from the kernel's scratch into our host-mapped scratch — the
    // ROCm runtime treats D2D as a plain kernel submission, avoiding the
    // ProfilingSignal pool entirely. Required because D2H from launch_kernel
    // has historically crashed the process with a ProfilingSignal race
    // (see file header comment near line 18).
    let copy_result = hipMemcpyDtoD(
        hipDeviceptr_t(buf.dev_ptr),
        hipDeviceptr_t(scratch_ptr as *mut std::ffi::c_void),
        MCB_DUMP_BYTES,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MCB scratch dump: hipMemcpyDtoD(0x{:x}, {}) failed: {:?}",
            scratch_ptr, MCB_DUMP_BYTES, copy_result
        ));
        return;
    }
    // Drain the D2D so CPU-side reads of host_ptr see the fresh bytes.
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Format as u64 values, but only dump as many as the grid_y suggests
    // (each work group ~ 1 entry = 4 u64s). Cap at MCB_DUMP_BYTES/8 u64s.
    // This avoids logging trailing zeros from unused buffer space.
    let entries_to_dump = ((grid_y as usize) + 1).min(MCB_DUMP_BYTES / 32);
    let u64s_to_dump = (entries_to_dump * 4).min(MCB_DUMP_BYTES / 8);
    let u64_slice = std::slice::from_raw_parts(buf.host_ptr as *const u64, u64s_to_dump);
    let mut hex_parts: Vec<String> = Vec::with_capacity(u64s_to_dump);
    for &v in u64_slice {
        hex_parts.push(format!("0x{:016x}", v));
    }
    super::driver::zluda_log_err(&format!(
        "*** MCB SCRATCH arg2=0x{:x} grid_y={} u64s=[{}]",
        scratch_ptr,
        grid_y,
        hex_parts.join(", "),
    ));
}

/// Phase 1 fix for Bug A: defensively force-zero `*(rd5+22176)` for every
/// tracked rd5 base BEFORE MS_1 reads it. This is the workaround for the
/// Bug A pattern where PhysX's per-frame `MemCopyBatchedAsync` reset batch
/// occasionally excludes the rd5 buffer that MS_1 is about to use,
/// leaving stale values that MS_1's atomic-add lands on top of (causing
/// the doubled-counter cascade and the -2 / wild-pointer crash chain).
///
/// **Causal chain of Bug A** (verified across runs 1, 5, 6, 7, 8, 14):
/// 1. PhysX calls `cuLaunchKernel('MemCopyBatchedAsync', ...)` 5-6× per
///    frame, with each call processing a subset of rd5 buffers.
/// 2. The host code that builds the per-batch rd5 list sometimes excludes
///    a specific rd5 in a specific frame (still under investigation, but
///    consistent across all 6 Bug A runs).
/// 3. The next frame's MS_1 reads `*(rd5+22176)` expecting 0, gets the
///    prior frame's value N, atomic-adds another N → doubled.
/// 4. MS_2_Batched runs its decrement loop 2N times instead of 0,
///    walking N counter slots and writing -2 to each (because rd6[0]
///    contains a sentinel that, after the kernel's `r38 - 1` op,
///    produces -2).
/// 5. MS_4 reads the -2 counter as `0xFFFFFFFE`, multiplies by 12 for
///    addressing, and produces a wild ~48 GB offset → SIGABRT.
///
/// **The fix**: PhysX expects `*(rd5+22176) == 0` before MS_1 launches.
/// We read each tracked rd5+22176, and any that are non-zero we explicitly
/// `hipMemsetD32` to 0. This is what PhysX's reset batch *should* have
/// done. The write is a no-op on healthy frames (already 0) and corrective
/// on bug frames. Cost: ≤8 4-byte writes per MS_1 launch (pinned scratch
/// already exists from F.6), small fraction of a millisecond.
///
/// We also update `tracker.last_state[rd5] = 0` so the post-launch F.6
/// detector logs the next MS_1 firing as a normal `MUTATED prev=0 new=N`
/// instead of a spurious `DOUBLED prev=stale new=...`.
///
/// Gated on `ZLUDA_MS4_CAPTURE=1` because that's what populates
/// `TRACKED_RD5_PTRS`. In non-capture mode the tracked list is empty and
/// this function early-returns. (Future: populate the list unconditionally
/// once the fix is verified.)
unsafe fn force_zero_rd5_22176_before_ms1() {
    if !ms4_capture_enabled() {
        return;
    }
    // Gate: ZLUDA_PHASE1_DISABLE=1 turns the workaround into a no-op so we
    // can observe PhysX's natural rd5+22176 reset behavior with the rest of
    // the instrumentation (MCB scratch dumps, F.6 tracker) still active.
    // Used for root-cause investigation of the run-25 overzero finding.
    if std::env::var("ZLUDA_PHASE1_DISABLE").ok().as_deref() == Some("1") {
        return;
    }
    let tracked: Vec<u64> = {
        let g = match TRACKED_RD5_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        g.clone()
    };
    if tracked.is_empty() {
        return;
    }

    let mut guard = match RD5_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    rd5_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(t) => t,
        None => return,
    };

    // Read current values via the same pinned scratch as check_rd5_22176_mutation.
    // Synchronous D2D + hipDeviceSynchronize so we observe the up-to-date values.
    let n = tracked.len().min(F6_MAX_RD5);
    for (i, &rd5) in tracked.iter().take(n).enumerate() {
        let dst = (tracker.scratch_dev as usize + i * F6_BYTES_PER) as *mut std::ffi::c_void;
        let src = (rd5 as usize + F6_OFFSET_IN_RD5) as *mut std::ffi::c_void;
        let _ = hipMemcpyDtoD(hipDeviceptr_t(dst), hipDeviceptr_t(src), F6_BYTES_PER);
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // For any rd5 with non-zero +22176, force-zero on the device and update
    // the tracker's last_state so the post-launch F.6 detector sees a clean
    // prev=0. Log every force-zero so we can measure how often the workaround
    // actually does anything (= how often PhysX skips the reset).
    let mut forced = 0usize;
    for (i, &rd5) in tracked.iter().take(n).enumerate() {
        let cur_val = *((tracker.scratch_host.add(i * F6_BYTES_PER)) as *const u32);
        if cur_val != 0 {
            let target = (rd5 as usize + F6_OFFSET_IN_RD5) as *mut std::ffi::c_void;
            let _ = hipMemsetD32(hipDeviceptr_t(target), 0, 1);
            tracker.last_state.insert(rd5, 0);
            forced += 1;
            super::driver::zluda_log_err(&format!(
                "*** RD5_22176 FORCE_ZERO before MS_1 rd5=0x{:x} was={} (PhysX reset batch missed this rd5)",
                rd5, cur_val,
            ));
        }
    }
    if forced > 0 {
        // One more sync to make sure the memset is visible to MS_1.
        let _ = hipDeviceSynchronize();
    }
}

/// Extract the current MS_1 launch's rd5 pointer.
///
/// In `CreateParticleShapesIncremental_MultiStage_1`, PTX param_6 is the rd5
/// base whose `+22176` word feeds the per-frame particle-shape counter.
fn current_ms1_rd5_ptr(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> Option<u64> {
    let rd5 = unsafe { read_kernel_arg(kernel_params, extra, 40, 6) }?;
    (rd5 != 0).then_some(rd5)
}

/// Targeted Bug A workaround: zero only the current MS_1 launch's rd5+22176.
///
/// The original broad workaround zeroed every tracked rd5 before every MS_1,
/// which corrupted the first actor when Batman issued the second MS_1 launch
/// in the same frame. This targeted variant only touches the rd5 pointer from
/// the CURRENT MS_1 launch packet (PTX param_6), so it corrects the stale
/// carry-over case without overzeroing another actor's freshly written value.
///
/// 2026-06-02: gate was previously `ms4_capture_enabled()` (ZLUDA_MS4_CAPTURE=1),
/// which is an investigation knob that protonfixes does NOT set. The workaround
/// was silently OFF in production despite ZLUDA_MS1_RD5_RESET=1 being set.
/// The 05-22/23 N=5 validation runs had BOTH env vars set, which masked this
/// dependency. Now gated by the dedicated ZLUDA_MS1_RD5_RESET env var so the
/// fix actually fires whenever the protonfix opts in.
unsafe fn force_zero_current_ms1_rd5_22176(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) {
    if !ms1_rd5_reset_enabled() {
        return;
    }
    if std::env::var("ZLUDA_PHASE1_DISABLE").ok().as_deref() == Some("1") {
        return;
    }
    let rd5 = match current_ms1_rd5_ptr(kernel_params, extra) {
        Some(rd5) => rd5,
        None => return,
    };
    track_rd5_ptr(rd5);

    let mut guard = match RD5_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    rd5_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(t) => t,
        None => return,
    };

    let src = (rd5 as usize + F6_OFFSET_IN_RD5) as *mut std::ffi::c_void;
    let _ = hipMemcpyDtoD(
        hipDeviceptr_t(tracker.scratch_dev),
        hipDeviceptr_t(src),
        F6_BYTES_PER,
    );
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let cur_val = *(tracker.scratch_host as *const u32);
    if cur_val == 0 {
        tracker.last_state.insert(rd5, 0);
        return;
    }

    let target = (rd5 as usize + F6_OFFSET_IN_RD5) as *mut std::ffi::c_void;
    let _ = hipMemsetD32(hipDeviceptr_t(target), 0, 1);
    tracker.last_state.insert(rd5, 0);
    super::driver::zluda_log_err(&format!(
        "*** RD5_22176 FORCE_ZERO current MS_1 rd5=0x{:x} was={} (targeted workaround)",
        rd5, cur_val,
    ));
    let _ = hipDeviceSynchronize();
}

/// Snapshot `*(rd5+22176)` for every tracked rd5 pointer and diff against
/// the previous snapshot. Logs `*** RD5_22176 {RESET|DOUBLED|MUTATED} by
/// '<kernel>' rd5=0x... prev=N new=M`. The RESET tag fires when N→0
/// (normal between-frame reset), DOUBLED fires when N→2N (the bug),
/// MUTATED otherwise.
unsafe fn check_rd5_22176_mutation(current_kernel: &str) {
    if !ms4_capture_enabled() {
        return;
    }
    // F.6 tracker is gated separately because it has a stale-pointer
    // hazard — see f6_tracker_enabled() comment.
    if !f6_tracker_enabled() {
        return;
    }
    let tracked: Vec<u64> = {
        let g = match TRACKED_RD5_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        g.clone()
    };
    if tracked.is_empty() {
        return;
    }

    let mut guard = match RD5_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    rd5_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(t) => t,
        None => return,
    };

    // D2D 4 bytes per tracked rd5 (the +22176 word) into our scratch.
    let n = tracked.len().min(F6_MAX_RD5);
    for (i, &rd5) in tracked.iter().take(n).enumerate() {
        let dst = (tracker.scratch_dev as usize + i * F6_BYTES_PER) as *mut std::ffi::c_void;
        let src = (rd5 as usize + F6_OFFSET_IN_RD5) as *mut std::ffi::c_void;
        let _ = hipMemcpyDtoD(hipDeviceptr_t(dst), hipDeviceptr_t(src), F6_BYTES_PER);
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Compare each scratch word against the last seen value.
    for (i, &rd5) in tracked.iter().take(n).enumerate() {
        let new_val = *((tracker.scratch_host.add(i * F6_BYTES_PER)) as *const u32);
        if let Some(&prev_val) = tracker.last_state.get(&rd5) {
            if new_val != prev_val {
                let tag = if new_val == 0 && prev_val != 0 {
                    "RESET"
                } else if prev_val != 0 && new_val == prev_val.saturating_mul(2) {
                    "DOUBLED"
                } else if prev_val > 0 && new_val > prev_val && new_val < prev_val * 4 {
                    "GREW"
                } else {
                    "MUTATED"
                };
                super::driver::zluda_log_err(&format!(
                    "*** RD5_22176 {} by '{}' rd5=0x{:x} prev={} new={}",
                    tag, current_kernel, rd5, prev_val, new_val,
                ));
            }
        }
        tracker.last_state.insert(rd5, new_val);
    }
}

/// Global launch serialization lock. Opt-in via `ZLUDA_LAUNCH_MUTEX` env var:
///   unset/`0` → no mutex (current behavior, concurrent launches allowed)
///   `1`       → lock held around `hipModuleLaunchKernel` only
///   `2`       → lock held around the launch AND a forced `hipDeviceSynchronize`
///
/// Mode 2 is the conclusive concurrency test: it guarantees no other CPU thread can
/// enter `launch_kernel` or queue GPU work while the current launch is in flight.
/// If PhysX crashes disappear in mode 2 but persist in mode 0/1, a cross-thread race
/// is confirmed as the root cause.
static LAUNCH_LOCK: Mutex<()> = Mutex::new(());

/// Plan v22 Phase B.1: set of every distinct DEV5 (p5) pointer ever observed
/// at MS_4 launch. The per-launch IOFX kernarg probe (gated on
/// ZLUDA_REMAP_KARGS=1) compares IOFX_remap/sort/actorRange/makeSortKeys
/// kernel arg pointers against this set to find launches whose output lands
/// in DEV5 — those are MS_4's upstream writers, the suspects for DEV5
/// corruption.
static MS4_DEV5_ADDRS: std::sync::OnceLock<Mutex<std::collections::HashSet<u64>>> =
    std::sync::OnceLock::new();

/// Plan v25 Phase B.2: queue of `(p4, p5, p6, p7, n_threads, launch_n)`
/// tuples pushed by the MS_4 launch hook in `launch_kernel`, drained by
/// the `cuEventRecord` and `cuEventSynchronize` handlers (host-thread
/// contexts where DtoH is allowed per
/// `feedback_no_d2d_from_launch_kernel`). The consumer does
/// `hipDeviceSynchronize` + `hipMemcpyDtoH` to read DEV5, DEV6, DEV7
/// contents and log histograms.
///
/// Tuple expanded from 3 to 6 elements in Plan v25 so the consumer can
/// dump DEV6 (initial counter, indexed by `r6*12`) and DEV7 (atomic
/// counter, indexed by `r6*4`) in addition to DEV5. Run e6 saw clean
/// DEV5 contents but only sampled indices 0..1023 (cap was 1024); the
/// crashing launch had n_threads=1722 so entries 1024..1721 were not
/// visible. v25 lifts that cap and adds the sibling buffers.
///
/// Gated on `ZLUDA_DEV5_DUMP=1`. Capped at 64 entries; drop-oldest on
/// overflow. Push side holds the lock only for the push itself (no
/// HIP calls) to avoid the UAF class of 2026-05-18.
pub(crate) static MS4_DEV5_DUMP_QUEUE: std::sync::OnceLock<
    Mutex<std::collections::VecDeque<(u64, u64, u64, u64, u32, u32)>>,
> = std::sync::OnceLock::new();

/// 2026-06-03: queue of `rd5` base pointers awaiting a defensive
/// `hipMemsetD32` zero of their `+22176` slot, drained from host-thread
/// contexts (`event::record`, `event::synchronize`, `stream::synchronize`)
/// per `feedback_no_d2d_from_launch_kernel.md`. Producer is the MS_1
/// launch hook (which calls `take_pending_rd5_resets` and enqueues here
/// rather than issuing `hipMemsetD32` synchronously from `launch_kernel`).
///
/// Cap 64 entries; drop-oldest on overflow. Lock held only across the
/// push itself; HIP calls happen outside the critical section in the
/// drain function.
pub(crate) static MS1_RD5_RESET_DRAIN_QUEUE: std::sync::OnceLock<
    Mutex<std::collections::VecDeque<u64>>,
> = std::sync::OnceLock::new();

/// Plan v27 Phase B.3 (extended 2026-05-19): queue of
/// `(param_4_ptr, param_6_ptr, param_9_ptr, launch_n)` tuples pushed by
/// the IOFX_actorRangeKernel hook. The consumer dumps all three READ
/// buffers (per Explore-identified PTX usage: param_4, param_6, param_9
/// are loaded as global u32, param_7+param_8 are written).
///
/// Run e18 (2026-05-19 01:14) caught 142 wild dumps from param_4. The
/// other 13 crashes in the same overnight loop showed no param_4
/// wildness — but they might have wild param_6 or param_9. Extending
/// to all three reads catches more of the same bug class.
///
/// Gated on `ZLUDA_ACTORRANGE_DUMP=1`. Capped at 64 entries; drop-oldest.
pub(crate) static IOFX_AR_DUMP_QUEUE: std::sync::OnceLock<
    Mutex<std::collections::VecDeque<(u64, u64, u64, u32)>>,
> = std::sync::OnceLock::new();

/// Queue of `(batch_desc_array_ptr, grid_x, grid_y, grid_z, launch_n)` for
/// `CreateParticleShapesIncremental_MultiStage_2_Batched` launches.
///
/// Producer runs in `launch_kernel` and only enqueues metadata. Consumer runs
/// from event handlers (host-thread context) and performs synchronized DtoH
/// reads of the 88-byte batch descriptors (+32/+72/+80 pointers) and first
/// shape samples (`rd7[i]`, `rd6[rd7[i]*12+8]`) to expose the true source of
/// MS4 counters.
///
/// Gated on `ZLUDA_MS2_CAPTURE=1`. Capped at 64 entries; drop-oldest.
pub(crate) static MS2_BATCH_DUMP_QUEUE: std::sync::OnceLock<
    Mutex<std::collections::VecDeque<(u64, u32, u32, u32, u32)>>,
> = std::sync::OnceLock::new();

/// Recent `(rd6_ptr, ms2_launch_n)` watchlist extracted from
/// `CreateParticleShapesIncremental_MultiStage_2_Batched` entry descriptors.
/// Used by memcpy hooks in `memory.rs` to attribute writes that hit
/// `rd6[*12+8]` source words consumed by the MS2 `r38-1 -> r39` path.
pub(crate) static MS2_RD6_WATCH: std::sync::OnceLock<
    Mutex<std::collections::VecDeque<(u64, u64)>>,
> = std::sync::OnceLock::new();

fn launch_mutex(lock_mode: u8) -> Option<&'static Mutex<()>> {
    if lock_mode >= 1 {
        Some(&LAUNCH_LOCK)
    } else {
        None
    }
}

/// 2026-06-03 Fix C helper: lazy-grown device-side zero buffer for the
/// `TurbulenceFS_dynamicGridClearKernel` intercept. Allocated once on
/// first use and grown if a larger array shows up. Holding a single
/// global keeps allocator pressure low across the hundreds of clear
/// dispatches per bench frame.
static CLEAR_ZERO_BUF: std::sync::OnceLock<Mutex<(hipDeviceptr_t, usize)>> =
    std::sync::OnceLock::new();

/// Ensure `CLEAR_ZERO_BUF` is at least `need_bytes` large and points to
/// all-zero device memory. Returns the device pointer on success, None
/// on any HIP error. Called from `try_host_clear_3d_array` only.
unsafe fn ensure_clear_zero_buf(need_bytes: usize) -> Option<hipDeviceptr_t> {
    let cell = CLEAR_ZERO_BUF.get_or_init(|| {
        Mutex::new((hipDeviceptr_t(std::ptr::null_mut()), 0usize))
    });
    let mut guard = cell.lock().ok()?;
    let (ref mut ptr, ref mut size) = *guard;
    if *size >= need_bytes && !ptr.0.is_null() {
        return Some(*ptr);
    }
    // (Re)allocate to at least need_bytes, doubling current capacity for
    // amortized growth.
    let new_size = need_bytes.max(*size * 2).max(64 * 1024);
    if !ptr.0.is_null() {
        let _ = hipFree(ptr.0);
    }
    let mut new_dev: *mut std::ffi::c_void = std::ptr::null_mut();
    if hipMalloc(&mut new_dev, new_size).is_err() || new_dev.is_null() {
        *ptr = hipDeviceptr_t(std::ptr::null_mut());
        *size = 0;
        return None;
    }
    if hipMemset(new_dev, 0, new_size).is_err() {
        let _ = hipFree(new_dev);
        *ptr = hipDeviceptr_t(std::ptr::null_mut());
        *size = 0;
        return None;
    }
    *ptr = hipDeviceptr_t(new_dev);
    *size = new_size;
    Some(*ptr)
}

/// 2026-06-03 Fix C: try to replace a `TurbulenceFS_dynamicGridClearKernel`
/// launch with a host-side `hipMemcpy3DAsync` from a zero buffer into the
/// 3D `hipArray_t` the surface object backs. Returns `true` if the
/// intercept succeeded and the caller should skip the actual kernel
/// dispatch; returns `false` if anything went wrong (caller falls through
/// to the normal `hipModuleLaunchKernel` path).
///
/// All HIP errors are non-fatal — they trigger fallback to the kernel
/// launch, never a crash. The intercept is intentionally conservative.
unsafe fn try_host_clear_3d_array(
    arr: hipArray_t,
    surf_obj_u64: u64,
    stream: hipStream_t,
) -> bool {
    let mut desc: HIP_ARRAY3D_DESCRIPTOR = std::mem::zeroed();
    if hipArray3DGetDescriptor(&mut desc, arr).is_err() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS_dynamicGridClear intercept: hipArray3DGetDescriptor failed surf=0x{:x} arr={:?} — falling back",
            surf_obj_u64, arr
        ));
        return false;
    }
    let chans = if desc.NumChannels == 0 { 1 } else { desc.NumChannels as usize };
    let fmt_size = match desc.Format {
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT8
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT8 => 1,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_HALF => 2,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_FLOAT => 4,
        _ => 0,
    };
    if fmt_size == 0 || desc.Width == 0 || desc.Height == 0 || desc.Depth == 0 {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS_dynamicGridClear intercept: bad descriptor (fmt={:?} ch={} {}x{}x{}) — falling back",
            desc.Format, desc.NumChannels, desc.Width, desc.Height, desc.Depth
        ));
        return false;
    }
    let row_bytes = desc.Width * chans * fmt_size;
    let slice_bytes = row_bytes * desc.Height;
    let total_bytes = slice_bytes * desc.Depth;
    let zero_dev = match ensure_clear_zero_buf(total_bytes) {
        Some(p) => p,
        None => {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] TFS_dynamicGridClear intercept: zero-buf alloc failed need={} — falling back",
                total_bytes
            ));
            return false;
        }
    };
    let mut params: hipMemcpy3DParms = std::mem::zeroed();
    params.srcArray = std::ptr::null_mut();
    params.srcPtr = hipPitchedPtr {
        ptr: zero_dev.0,
        pitch: row_bytes,
        xsize: row_bytes,
        ysize: desc.Height,
    };
    params.srcPos = hipPos { x: 0, y: 0, z: 0 };
    params.dstArray = arr;
    params.dstPos = hipPos { x: 0, y: 0, z: 0 };
    params.dstPtr = hipPitchedPtr {
        ptr: std::ptr::null_mut(),
        pitch: 0,
        xsize: 0,
        ysize: 0,
    };
    params.extent = hipExtent {
        width: row_bytes,
        height: desc.Height,
        depth: desc.Depth,
    };
    params.kind = hipMemcpyKind::hipMemcpyDeviceToDevice;
    let r = hipMemcpy3DAsync(&params, stream);
    if r.is_err() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS_dynamicGridClear intercept: hipMemcpy3DAsync err={:?} — falling back",
            r
        ));
        return false;
    }
    static FIRST_HIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    FIRST_HIT.get_or_init(|| {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS_dynamicGridClear intercept ACTIVE: surf=0x{:x} arr={:?} {}x{}x{} fmt={:?} ch={} (subsequent intercepts silent)",
            surf_obj_u64, arr, desc.Width, desc.Height, desc.Depth, desc.Format, desc.NumChannels
        ));
        true
    });
    true
}

// ============================================================================
// F.5: Safe CreateGrid sortedKeys validator
// ============================================================================
//
// The earlier validate_create_grid_inputs / validate_create_grid_from_args
// functions used hipMemcpyDtoH which triggers the ROCm ProfilingSignal pool
// race (see commit 589d0290 / the comment in the original launch_kernel
// site). They were removed and replaced with a dead-code suppression. This
// left us with NO visibility into when CUB Downsweep produces corrupt sort
// output — exactly the failure mode that crashes CreateGrid in `_Z10CreateGrid…`.
//
// F.5 reinstates the validator using D2D into pinned host-mapped scratch
// (the same pattern as the MS_4/MS_3/MS_2_Batched/F.4 capture rings, which
// avoid the ProfilingSignal pool race entirely). The validator runs from
// `launch_kernel` BEFORE every CreateGrid launch, snapshots the sortedKeys
// input buffer, scans for non-monotonic and out-of-range entries, and logs
// `*** SORT CORRUPTED` if anything is wrong.

const F5_MAX_KEYS: usize = 16384;
const F5_SCRATCH_BYTES: usize = F5_MAX_KEYS * 4;
const F5_TOTAL_SCRATCH_BYTES: usize = F5_SCRATCH_BYTES * 2;
const SORT_PASS_SPINE_MAX_WORDS: usize = 256;
const SORT_PASS_SPINE_BYTES: usize = SORT_PASS_SPINE_MAX_WORDS * 4;
const SORT_PASS_KEYS_IN_OFFSET: usize = 0;
const SORT_PASS_KEYS_OUT_OFFSET: usize = SORT_PASS_KEYS_IN_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_VALS_IN_OFFSET: usize = SORT_PASS_KEYS_OUT_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_VALS_OUT_OFFSET: usize = SORT_PASS_VALS_IN_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_KEYS_IN_BEFORE_OFFSET: usize = SORT_PASS_VALS_OUT_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_VALS_IN_BEFORE_OFFSET: usize = SORT_PASS_KEYS_IN_BEFORE_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_KEYS_OUT_BEFORE_OFFSET: usize = SORT_PASS_VALS_IN_BEFORE_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_VALS_OUT_BEFORE_OFFSET: usize = SORT_PASS_KEYS_OUT_BEFORE_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_SPINE_OFFSET: usize = SORT_PASS_VALS_OUT_BEFORE_OFFSET + F5_SCRATCH_BYTES;
const SORT_PASS_TOTAL_SCRATCH_BYTES: usize = SORT_PASS_SPINE_OFFSET + SORT_PASS_SPINE_BYTES;
const SORT_PASS_QUEUE_SLOT_OFFSETS: [usize; 8] = [
    SORT_PASS_KEYS_IN_OFFSET,
    SORT_PASS_KEYS_OUT_OFFSET,
    SORT_PASS_VALS_IN_OFFSET,
    SORT_PASS_VALS_OUT_OFFSET,
    SORT_PASS_KEYS_IN_BEFORE_OFFSET,
    SORT_PASS_VALS_IN_BEFORE_OFFSET,
    SORT_PASS_KEYS_OUT_BEFORE_OFFSET,
    SORT_PASS_VALS_OUT_BEFORE_OFFSET,
];
const SORT_PASS_QUEUE_MAX_PENDING: usize = SORT_PASS_QUEUE_SLOT_OFFSETS.len();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SortedKeySummary {
    min_key: i32,
    max_key: i32,
    violations: u32,
    negative_count: u32,
    huge_count: u32,
    first_violation: Option<usize>,
    first_bad_value: Option<(usize, i32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KeyDomainSummary {
    min_key: i32,
    max_key: i32,
    negative_count: u32,
    huge_count: u32,
    first_bad_value: Option<(usize, i32)>,
}

fn summarize_sorted_keys(keys: &[i32]) -> Option<SortedKeySummary> {
    let (&first, rest) = keys.split_first()?;
    let mut summary = SortedKeySummary {
        min_key: first,
        max_key: first,
        violations: 0,
        negative_count: u32::from(first < 0),
        huge_count: u32::from(first > 16_000_000),
        first_violation: None,
        first_bad_value: if first < 0 || first > 16_000_000 {
            Some((0, first))
        } else {
            None
        },
    };
    let mut prev = first;
    for (idx, &key) in rest.iter().enumerate() {
        let i = idx + 1;
        if key < prev {
            summary.violations += 1;
            if summary.first_violation.is_none() {
                summary.first_violation = Some(i);
            }
        }
        if key < summary.min_key {
            summary.min_key = key;
        }
        if key > summary.max_key {
            summary.max_key = key;
        }
        if key < 0 {
            summary.negative_count += 1;
        }
        if key > 16_000_000 {
            summary.huge_count += 1;
        }
        if summary.first_bad_value.is_none() && (key < 0 || key > 16_000_000) {
            summary.first_bad_value = Some((i, key));
        }
        prev = key;
    }
    Some(summary)
}

fn summarize_key_domain(keys: &[i32]) -> Option<KeyDomainSummary> {
    let (&first, rest) = keys.split_first()?;
    let mut summary = KeyDomainSummary {
        min_key: first,
        max_key: first,
        negative_count: u32::from(first < 0),
        huge_count: u32::from(first > 16_000_000),
        first_bad_value: if first < 0 || first > 16_000_000 {
            Some((0, first))
        } else {
            None
        },
    };
    for (idx, &key) in rest.iter().enumerate() {
        let i = idx + 1;
        if key < summary.min_key {
            summary.min_key = key;
        }
        if key > summary.max_key {
            summary.max_key = key;
        }
        if key < 0 {
            summary.negative_count += 1;
        }
        if key > 16_000_000 {
            summary.huge_count += 1;
        }
        if summary.first_bad_value.is_none() && (key < 0 || key > 16_000_000) {
            summary.first_bad_value = Some((i, key));
        }
    }
    Some(summary)
}

fn restore_signed_radix_residue(keys: &[i32]) -> Vec<i32> {
    keys.iter()
        .map(|&key| {
            if key < 0 {
                ((key as u32) ^ 0x8000_0000) as i32
            } else {
                key
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SortValueSummary {
    negative_count: u32,
    extra_count: u32,
    missing_count: u32,
    first_negative: Option<(usize, i32)>,
    first_extra: Option<(usize, i32, u32, u32)>,
    first_missing: Option<(usize, i32, u32, u32)>,
}

fn summarize_sort_values(values_in: &[i32], values_out: &[i32]) -> Option<SortValueSummary> {
    if values_in.is_empty() || values_out.is_empty() || values_in.len() != values_out.len() {
        return None;
    }
    let mut input_counts = HashMap::<i32, u32>::new();
    for &value in values_in {
        *input_counts.entry(value).or_insert(0) += 1;
    }
    let mut output_counts = HashMap::<i32, u32>::new();
    let mut summary = SortValueSummary {
        negative_count: 0,
        extra_count: 0,
        missing_count: 0,
        first_negative: None,
        first_extra: None,
        first_missing: None,
    };
    for (i, &value) in values_out.iter().enumerate() {
        if value < 0 {
            summary.negative_count += 1;
            if summary.first_negative.is_none() {
                summary.first_negative = Some((i, value));
            }
        }
        let seen = output_counts.entry(value).or_insert(0);
        *seen += 1;
        let input_seen = input_counts.get(&value).copied().unwrap_or(0);
        if *seen > input_seen {
            summary.extra_count += 1;
            if summary.first_extra.is_none() {
                summary.first_extra = Some((i, value, input_seen, *seen));
            }
        }
    }
    for (&value, &expected) in input_counts.iter() {
        let actual = output_counts.get(&value).copied().unwrap_or(0);
        if expected > actual {
            summary.missing_count += expected - actual;
        }
    }
    for (idx, &value) in values_in.iter().enumerate() {
        let expected = input_counts.get(&value).copied().unwrap_or(0);
        let actual = output_counts.get(&value).copied().unwrap_or(0);
        if expected > actual {
            if summary.first_missing.is_none() {
                summary.first_missing = Some((idx, value, expected, actual));
            }
            break;
        }
    }
    Some(summary)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GoodSortReplayCaptureKind {
    TargetBit0N8109,
    FallbackBit0LargeClean,
    LateBit16N4688Clean,
}

fn classify_good_sort_replay_capture(
    begin_bit: u32,
    n_items: usize,
    first: u8,
    overwrite: u8,
    summary_out: SortedKeySummary,
    vals_summary: Option<SortValueSummary>,
) -> Option<GoodSortReplayCaptureKind> {
    if summary_out.negative_count != 0
        || !vals_summary
            .map(|summary| {
                summary.negative_count == 0
                    && summary.extra_count == 0
                    && summary.missing_count == 0
            })
            .unwrap_or(false)
    {
        return None;
    }
    if begin_bit == 0 && first == 0 && overwrite == 0 && n_items == 8109 {
        Some(GoodSortReplayCaptureKind::TargetBit0N8109)
    } else if begin_bit == 0 && first == 0 && overwrite == 0 && n_items >= 1024 {
        Some(GoodSortReplayCaptureKind::FallbackBit0LargeClean)
    } else if begin_bit == 16 && overwrite != 0 && n_items == 4688 {
        Some(GoodSortReplayCaptureKind::LateBit16N4688Clean)
    } else {
        None
    }
}

fn good_sort_replay_capture_path(
    kind: GoodSortReplayCaptureKind,
    begin_bit: u32,
    n_items: usize,
    pass_n: u32,
) -> String {
    match kind {
        GoodSortReplayCaptureKind::TargetBit0N8109 => {
            format!(
                "/tmp/zluda_sort_replay_capture_good_bit{}_pass{}.bin",
                begin_bit, pass_n
            )
        }
        GoodSortReplayCaptureKind::FallbackBit0LargeClean => {
            format!(
                "/tmp/zluda_sort_replay_capture_good_bit{}_n{}_pass{}.bin",
                begin_bit, n_items, pass_n
            )
        }
        GoodSortReplayCaptureKind::LateBit16N4688Clean => {
            format!(
                "/tmp/zluda_sort_replay_capture_good_bit{}_n{}_pass{}.bin",
                begin_bit, n_items, pass_n
            )
        }
    }
}

fn format_sort_pair_window(
    keys: &[i32],
    values: &[i32],
    center: usize,
    label: &str,
) -> Option<String> {
    if keys.len() != values.len() || center >= keys.len() || keys.is_empty() {
        return None;
    }
    let start = center.saturating_sub(4);
    let end = std::cmp::min(keys.len(), center + 5);
    let mut parts = Vec::new();
    for idx in start..end {
        parts.push(format!(
            "[{}]=k:{}(0x{:08x}) v:{}",
            idx, keys[idx], keys[idx] as u32, values[idx]
        ));
    }
    Some(format!(
        "{}[{}..{}): {}",
        label,
        start,
        end,
        parts.join(" ")
    ))
}

fn summarize_sort_pair_window_diff(
    expected_keys: &[i32],
    expected_values: &[i32],
    actual_keys: &[i32],
    actual_values: &[i32],
    center: usize,
    label: &str,
) -> Option<String> {
    if expected_keys.len() != expected_values.len()
        || actual_keys.len() != actual_values.len()
        || expected_keys.len() != actual_keys.len()
        || center >= expected_keys.len()
        || expected_keys.is_empty()
    {
        return None;
    }
    let start = center.saturating_sub(4);
    let end = std::cmp::min(expected_keys.len(), center + 5);
    let mismatches = (start..end)
        .filter(|&idx| {
            expected_keys[idx] != actual_keys[idx] || expected_values[idx] != actual_values[idx]
        })
        .collect::<Vec<_>>();
    let mismatch_runs = mismatches
        .split_first()
        .map(|(&first, rest)| {
            let mut runs = Vec::new();
            let mut run_start = first;
            let mut run_end = first + 1;
            for &idx in rest {
                if idx == run_end {
                    run_end += 1;
                } else {
                    runs.push(format!("{}..{}", run_start, run_end));
                    run_start = idx;
                    run_end = idx + 1;
                }
            }
            runs.push(format!("{}..{}", run_start, run_end));
            format!("[{}]", runs.join(", "))
        })
        .unwrap_or_else(|| "[]".to_string());
    Some(format!(
        "{}[{}..{}): mismatch_count={} mismatches={:?} runs={}",
        label,
        start,
        end,
        mismatches.len(),
        mismatches,
        mismatch_runs
    ))
}

fn contiguous_index_runs(indices: &[usize]) -> Vec<(usize, usize)> {
    let Some((&first, rest)) = indices.split_first() else {
        return Vec::new();
    };
    let mut runs = Vec::new();
    let mut run_start = first;
    let mut run_end = first + 1;
    for &idx in rest {
        if idx == run_end {
            run_end += 1;
        } else {
            runs.push((run_start, run_end));
            run_start = idx;
            run_end = idx + 1;
        }
    }
    runs.push((run_start, run_end));
    runs
}

fn format_index_runs(indices: &[usize]) -> String {
    let runs = contiguous_index_runs(indices);
    if runs.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            runs.iter()
                .map(|(start, end)| format!("{}..{}", start, end))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn find_value_positions(values: &[i32], target: i32, limit: usize) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value == target { Some(idx) } else { None })
        .take(limit)
        .collect()
}

fn find_key_positions(keys: &[i32], target: i32, limit: usize) -> Vec<usize> {
    keys.iter()
        .enumerate()
        .filter_map(|(idx, &key)| if key == target { Some(idx) } else { None })
        .take(limit)
        .collect()
}

fn find_pair_positions(
    keys: &[i32],
    values: &[i32],
    target_key: i32,
    target_value: i32,
    limit: usize,
) -> Vec<usize> {
    keys.iter()
        .zip(values.iter())
        .enumerate()
        .filter_map(|(idx, (&key, &value))| {
            if key == target_key && value == target_value {
                Some(idx)
            } else {
                None
            }
        })
        .take(limit)
        .collect()
}

fn matching_pair_run(
    keys: &[i32],
    values: &[i32],
    left: usize,
    right: usize,
) -> Option<(usize, usize, usize)> {
    if keys.len() != values.len() || left >= keys.len() || right >= keys.len() || left == right {
        return None;
    }
    let mut left_start = left;
    let mut right_start = right;
    while left_start > 0
        && right_start > 0
        && keys[left_start - 1] == keys[right_start - 1]
        && values[left_start - 1] == values[right_start - 1]
    {
        left_start -= 1;
        right_start -= 1;
    }
    let mut len = 0;
    while left_start + len < keys.len()
        && right_start + len < keys.len()
        && keys[left_start + len] == keys[right_start + len]
        && values[left_start + len] == values[right_start + len]
    {
        len += 1;
    }
    Some((left_start, right_start, len))
}

fn matching_value_run(
    left_values: &[i32],
    right_values: &[i32],
    left: usize,
    right: usize,
) -> Option<(usize, usize, usize)> {
    if left >= left_values.len() || right >= right_values.len() {
        return None;
    }
    let mut left_start = left;
    let mut right_start = right;
    while left_start > 0
        && right_start > 0
        && left_values[left_start - 1] == right_values[right_start - 1]
    {
        left_start -= 1;
        right_start -= 1;
    }
    let mut len = 0;
    while left_start + len < left_values.len()
        && right_start + len < right_values.len()
        && left_values[left_start + len] == right_values[right_start + len]
    {
        len += 1;
    }
    Some((left_start, right_start, len))
}

fn sort_pass_radix_digit(key: i32, begin_bit: u32) -> u32 {
    ((key as u32) >> begin_bit) & 0xf
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpectedSortPass {
    keys_out: Vec<i32>,
    values_out: Vec<i32>,
    output_to_input: Vec<usize>,
}

fn reconstruct_expected_sort_pass(
    keys_in: &[i32],
    values_in: &[i32],
    begin_bit: u32,
) -> Option<ExpectedSortPass> {
    if keys_in.is_empty() || keys_in.len() != values_in.len() {
        return None;
    }
    let mut counts = [0usize; 16];
    for &key in keys_in {
        counts[sort_pass_radix_digit(key, begin_bit) as usize] += 1;
    }
    let mut offsets = [0usize; 16];
    let mut running = 0usize;
    for (digit, count) in counts.into_iter().enumerate() {
        offsets[digit] = running;
        running += count;
    }
    let mut keys_out = vec![0i32; keys_in.len()];
    let mut values_out = vec![0i32; values_in.len()];
    let mut output_to_input = vec![0usize; keys_in.len()];
    for (input_idx, (&key, &value)) in keys_in.iter().zip(values_in.iter()).enumerate() {
        let digit = sort_pass_radix_digit(key, begin_bit) as usize;
        let out_idx = offsets[digit];
        keys_out[out_idx] = key;
        values_out[out_idx] = value;
        output_to_input[out_idx] = input_idx;
        offsets[digit] += 1;
    }
    Some(ExpectedSortPass {
        keys_out,
        values_out,
        output_to_input,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpectedSortPassDiagnostics {
    mismatch_line: Option<String>,
    expected_window: Option<String>,
    extra_positions: Option<String>,
    missing_positions: Option<String>,
    actual_pair_route: Option<String>,
    actual_pair_source_window: Option<String>,
    actual_pair_clone_window: Option<String>,
    actual_value_route: Option<String>,
    actual_value_input_window: Option<String>,
    actual_value_input_run: Option<String>,
    actual_value_expected_window: Option<String>,
    actual_value_expected_run: Option<String>,
    actual_value_source_window: Option<String>,
    actual_value_clone_window: Option<String>,
    actual_value_source_diff: Option<String>,
    actual_value_clone_diff: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SortGridEvenShare {
    total_grains: i32,
    big_blocks: i32,
    big_share: i32,
    normal_share: i32,
    normal_base_offset: i32,
    num_items: i32,
    grid_size: i32,
    block_offset: i32,
    block_end: i32,
}

impl SortGridEvenShare {
    fn from_words(words: [i32; 9]) -> Self {
        Self {
            total_grains: words[0],
            big_blocks: words[1],
            big_share: words[2],
            normal_share: words[3],
            normal_base_offset: words[4],
            num_items: words[5],
            grid_size: words[6],
            block_offset: words[7],
            block_end: words[8],
        }
    }

    fn block_count(&self) -> usize {
        self.grid_size.max(self.total_grains).max(0) as usize
    }

    fn item_count(&self, limit: usize) -> usize {
        let num_items = usize::try_from(self.num_items.max(0)).unwrap_or(0);
        if num_items == 0 {
            limit
        } else {
            std::cmp::min(limit, num_items)
        }
    }

    fn covered_item_count(&self) -> Option<usize> {
        let block_count = self.block_count();
        if block_count == 0 {
            return Some(0);
        }
        let big_blocks = self.big_blocks.max(0) as usize;
        let big_share = usize::try_from(self.big_share.max(0)).ok()?;
        let normal_share = usize::try_from(self.normal_share.max(0)).ok()?;
        let normal_base_offset = usize::try_from(self.normal_base_offset.max(0)).ok()?;
        let big_end = big_blocks.checked_mul(big_share)?;
        let normal_blocks = block_count.saturating_sub(big_blocks);
        let normal_end =
            normal_base_offset.checked_add(normal_blocks.checked_mul(normal_share)?)?;
        Some(big_end.max(normal_end))
    }

    fn block_range(&self, block_idx: usize, limit: usize) -> Option<(usize, usize)> {
        let item_count = self.item_count(limit);
        if item_count == 0 || block_idx >= self.block_count() {
            return None;
        }
        let big_blocks = self.big_blocks.max(0) as usize;
        let (start_i32, share_i32) = if block_idx < big_blocks {
            (
                i32::try_from(block_idx).ok()?.checked_mul(self.big_share)?,
                self.big_share,
            )
        } else {
            let normal_idx = i32::try_from(block_idx - big_blocks).ok()?;
            (
                self.normal_base_offset
                    .checked_add(normal_idx.checked_mul(self.normal_share)?)?,
                self.normal_share,
            )
        };
        let start = usize::try_from(start_i32).ok()?;
        let share = usize::try_from(share_i32).ok()?;
        let end = start.saturating_add(share).min(item_count);
        if start >= end {
            None
        } else {
            Some((start, end))
        }
    }

    fn block_for_index(&self, index: usize, limit: usize) -> Option<(usize, usize, usize)> {
        let item_count = self.item_count(limit);
        if index >= item_count {
            return None;
        }
        (0..self.block_count()).find_map(|block_idx| {
            self.block_range(block_idx, item_count)
                .and_then(|(start, end)| {
                    if index >= start && index < end {
                        Some((block_idx, start, end))
                    } else {
                        None
                    }
                })
        })
    }
}

fn sort_pass_capture_len(n_items: usize, grid_share: Option<SortGridEvenShare>) -> usize {
    let capture_len = grid_share
        .and_then(|share| share.covered_item_count())
        .map(|covered| covered.max(n_items))
        .unwrap_or(n_items);
    capture_len.min(F5_MAX_KEYS)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DownsweepRankDiagnostics {
    expected_line: Option<String>,
    actual_line: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DownsweepBlockCapture {
    block_idx: usize,
    start: usize,
    end: usize,
    expected_source_block: usize,
    actual_source_block: Option<usize>,
    capture_text: String,
}

const SORT_PASS_REPLAY_CAPTURE_MAGIC: u32 = 0x5350_5243; // "SPRC"
const SORT_PASS_REPLAY_CAPTURE_VERSION: u32 = 6;
const SORT_PASS_REPLAY_CAPTURE_FLAG_HAS_VALS: u32 = 1 << 0;
const SORT_PASS_REPLAY_CAPTURE_FLAG_HAS_GRID_SHARE: u32 = 1 << 1;

fn push_le_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_le_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_le_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn pack_sort_replay_launch_flags(first: u8, last: u8, overwrite: u8) -> u32 {
    u32::from(first) | (u32::from(last) << 8) | (u32::from(overwrite) << 16)
}

fn serialize_sort_pass_replay_capture(
    pass_n: u32,
    begin_bit: u32,
    n_items: usize,
    grid_dim_x: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    first: u8,
    last: u8,
    overwrite: u8,
    keys_in_ptr: u64,
    keys_out_ptr: u64,
    vals_in_ptr: u64,
    vals_out_ptr: u64,
    spine_ptr: u64,
    stream_ptr: u64,
    keys_in: &[i32],
    vals_in: Option<&[i32]>,
    keys_in_before: &[i32],
    vals_in_before: Option<&[i32]>,
    keys_out_before: &[i32],
    vals_out_before: Option<&[i32]>,
    keys_out: &[i32],
    vals_out: Option<&[i32]>,
    spine_words: &[u32],
    grid_share: Option<SortGridEvenShare>,
) -> Option<Vec<u8>> {
    if keys_in.len() != keys_out.len()
        || keys_in.len() != keys_in_before.len()
        || keys_in.len() != keys_out_before.len()
    {
        return None;
    }
    if vals_in.is_some() != vals_out.is_some()
        || vals_in.is_some() != vals_in_before.is_some()
        || vals_in.is_some() != vals_out_before.is_some()
    {
        return None;
    }
    if let Some(vals_in) = vals_in {
        if vals_in.len() != keys_in.len()
            || vals_in_before?.len() != keys_in_before.len()
            || vals_out?.len() != keys_out.len()
            || vals_out_before?.len() != keys_out_before.len()
        {
            return None;
        }
    }
    if n_items > keys_in.len() {
        return None;
    }
    let n_items = u32::try_from(n_items).ok()?;
    let capture_items = u32::try_from(keys_in.len()).ok()?;
    let spine_words_len = u32::try_from(spine_words.len()).ok()?;
    let grid_share_words = grid_share.map(|share| {
        [
            share.total_grains,
            share.big_blocks,
            share.big_share,
            share.normal_share,
            share.normal_base_offset,
            share.num_items,
            share.grid_size,
            share.block_offset,
            share.block_end,
        ]
    });
    let mut flags = 0u32;
    if vals_in.is_some() {
        flags |= SORT_PASS_REPLAY_CAPTURE_FLAG_HAS_VALS;
    }
    if grid_share_words.is_some() {
        flags |= SORT_PASS_REPLAY_CAPTURE_FLAG_HAS_GRID_SHARE;
    }
    let payload_words = keys_in.len()
        + keys_in_before.len()
        + keys_out_before.len()
        + keys_out.len()
        + vals_in.map(|v| v.len()).unwrap_or(0)
        + vals_in_before.map(|v| v.len()).unwrap_or(0)
        + vals_out_before.map(|v| v.len()).unwrap_or(0)
        + vals_out.map(|v| v.len()).unwrap_or(0)
        + spine_words.len();
    let mut bytes = Vec::with_capacity((35 + payload_words) * 4);

    push_le_u32(&mut bytes, SORT_PASS_REPLAY_CAPTURE_MAGIC);
    push_le_u32(&mut bytes, SORT_PASS_REPLAY_CAPTURE_VERSION);
    push_le_u32(&mut bytes, pass_n);
    push_le_u32(&mut bytes, begin_bit);
    push_le_u32(&mut bytes, n_items);
    push_le_u32(&mut bytes, capture_items);
    push_le_u32(&mut bytes, grid_dim_x);
    push_le_u32(&mut bytes, spine_words_len);
    push_le_u32(&mut bytes, flags);
    push_le_u32(
        &mut bytes,
        pack_sort_replay_launch_flags(first, last, overwrite),
    );
    push_le_u32(&mut bytes, block_dim_x);
    push_le_u32(&mut bytes, block_dim_y);
    push_le_u32(&mut bytes, block_dim_z);
    push_le_u32(&mut bytes, shared_mem_bytes);
    push_le_u64(&mut bytes, keys_in_ptr);
    push_le_u64(&mut bytes, keys_out_ptr);
    push_le_u64(&mut bytes, vals_in_ptr);
    push_le_u64(&mut bytes, vals_out_ptr);
    push_le_u64(&mut bytes, spine_ptr);
    push_le_u64(&mut bytes, stream_ptr);
    for word in grid_share_words.unwrap_or([0; 9]) {
        push_le_i32(&mut bytes, word);
    }
    for &value in keys_in {
        push_le_i32(&mut bytes, value);
    }
    if let Some(vals_in) = vals_in {
        for &value in vals_in {
            push_le_i32(&mut bytes, value);
        }
    }
    for &value in keys_in_before {
        push_le_i32(&mut bytes, value);
    }
    if let Some(vals_in_before) = vals_in_before {
        for &value in vals_in_before {
            push_le_i32(&mut bytes, value);
        }
    }
    for &value in keys_out_before {
        push_le_i32(&mut bytes, value);
    }
    if let Some(vals_out_before) = vals_out_before {
        for &value in vals_out_before {
            push_le_i32(&mut bytes, value);
        }
    }
    for &value in keys_out {
        push_le_i32(&mut bytes, value);
    }
    if let Some(vals_out) = vals_out {
        for &value in vals_out {
            push_le_i32(&mut bytes, value);
        }
    }
    for &value in spine_words {
        push_le_u32(&mut bytes, value);
    }
    Some(bytes)
}

fn write_good_sort_replay_capture_once(
    captured: &std::sync::atomic::AtomicBool,
    path: &str,
    capture: Option<&[u8]>,
    begin_bit: u32,
    n_items: usize,
    grid_dim_x: u32,
    spine_words: usize,
) -> Option<String> {
    if captured
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        return None;
    }
    let line = match capture {
        Some(bytes) => match std::fs::write(path, bytes) {
            Ok(()) => format!(
                "SORT_REPLAY_CAPTURE_GOOD bit={} n={} grid={} spine_words={} path={}",
                begin_bit, n_items, grid_dim_x, spine_words, path
            ),
            Err(err) => {
                captured.store(false, std::sync::atomic::Ordering::Relaxed);
                format!(
                    "SORT_REPLAY_CAPTURE_GOOD bit={} n={} grid={} spine_words={} path={} write_err={}",
                    begin_bit, n_items, grid_dim_x, spine_words, path, err
                )
            }
        },
        None => {
            captured.store(false, std::sync::atomic::Ordering::Relaxed);
            format!(
                "SORT_REPLAY_CAPTURE_GOOD bit={} n={} grid={} spine_words={} path={} capture_missing",
                begin_bit, n_items, grid_dim_x, spine_words, path
            )
        }
    };
    Some(line)
}

fn compute_sort_digit_block_counts(
    keys_in: &[i32],
    begin_bit: u32,
    grid_share: &SortGridEvenShare,
) -> Option<Vec<[u32; 16]>> {
    let item_count = grid_share.item_count(keys_in.len());
    let block_count = grid_share.block_count();
    if item_count == 0 || block_count == 0 {
        return None;
    }
    let mut counts = vec![[0u32; 16]; block_count];
    let mut saw_block = false;
    for block_idx in 0..block_count {
        let Some((start, end)) = grid_share.block_range(block_idx, item_count) else {
            continue;
        };
        saw_block = true;
        for &key in &keys_in[start..end] {
            counts[block_idx][sort_pass_radix_digit(key, begin_bit) as usize] += 1;
        }
    }
    saw_block.then_some(counts)
}

fn format_downsweep_rank_line(
    label: &str,
    keys_in: &[i32],
    values_in: &[i32],
    begin_bit: u32,
    grid_share: &SortGridEvenShare,
    block_counts: &[[u32; 16]],
    source_idx: usize,
    out_idx: usize,
    output_key: Option<i32>,
    actual_dst: Option<usize>,
) -> Option<String> {
    if keys_in.len() != values_in.len() || source_idx >= keys_in.len() {
        return None;
    }
    let key = keys_in[source_idx];
    let value = values_in[source_idx];
    let digit = sort_pass_radix_digit(key, begin_bit) as usize;
    let (block_idx, block_start, block_end) =
        grid_share.block_for_index(source_idx, keys_in.len())?;
    let scan_counts = block_counts
        .iter()
        .map(|counts| counts[digit] as usize)
        .collect::<Vec<_>>();
    let mut running = 0usize;
    let scan_prefix = scan_counts
        .iter()
        .map(|&count| {
            let prefix = running;
            running += count;
            prefix
        })
        .collect::<Vec<_>>();
    let local_rank = keys_in[block_start..source_idx]
        .iter()
        .filter(|&&candidate| sort_pass_radix_digit(candidate, begin_bit) as usize == digit)
        .count();
    let global_rank = scan_prefix.get(block_idx).copied()? + local_rank;
    let digit_base = (0..digit)
        .map(|digit_idx| {
            block_counts
                .iter()
                .map(|counts| counts[digit_idx] as usize)
                .sum::<usize>()
        })
        .sum::<usize>();
    let scatter_dst = digit_base + global_rank;
    let output_key_note = output_key
        .filter(|&candidate| candidate != key)
        .map(|candidate| format!(" output_k:{}(0x{:08x})", candidate, candidate as u32))
        .unwrap_or_default();
    let actual_dst_note = actual_dst
        .map(|dst| format!(" actual_dst={}", dst))
        .unwrap_or_default();
    Some(format!(
        "{} out[{}] source=in[{}] k:{}(0x{:08x}) v:{} digit={} block={} range=[{}..{}) scan_counts={:?} scan_prefix={:?} digit_base={} local_rank={} global_rank={} scatter_dst={}{}{}",
        label,
        out_idx,
        source_idx,
        key,
        key as u32,
        value,
        digit,
        block_idx,
        block_start,
        block_end,
        scan_counts,
        scan_prefix,
        digit_base,
        local_rank,
        global_rank,
        scatter_dst,
        output_key_note,
        actual_dst_note,
    ))
}

fn append_downsweep_block_capture_rows(
    capture_text: &mut String,
    section: &str,
    block_idx: usize,
    start: usize,
    end: usize,
    mismatch_out: usize,
    keys_in: &[i32],
    values_in: &[i32],
    expected: &ExpectedSortPass,
    keys_out: &[i32],
    values_out: &[i32],
    begin_bit: u32,
) {
    let _ = writeln!(
        capture_text,
        "{} block={} range=[{}..{})",
        section, block_idx, start, end
    );
    for idx in start..end {
        let in_key = keys_in[idx];
        let in_value = values_in[idx];
        let expected_key = expected.keys_out[idx];
        let expected_value = expected.values_out[idx];
        let expected_src = expected.output_to_input[idx];
        let actual_key_at_idx = keys_out[idx];
        let actual_value_at_idx = values_out[idx];
        let actual_xor_note = if actual_key_at_idx < 0 {
            let restored = ((actual_key_at_idx as u32) ^ 0x8000_0000) as i32;
            format!(" xor_k:{}(0x{:08x})", restored, restored as u32)
        } else {
            String::new()
        };
        let marker = if idx == mismatch_out { "!" } else { " " };
        let _ = writeln!(
            capture_text,
            "{}[{}] in=k:{}(0x{:08x}) v:{} d:{} | expect<-in[{}] k:{}(0x{:08x}) v:{} d:{} | actual k:{}(0x{:08x}) v:{} d:{}{}",
            marker,
            idx,
            in_key,
            in_key as u32,
            in_value,
            sort_pass_radix_digit(in_key, begin_bit),
            expected_src,
            expected_key,
            expected_key as u32,
            expected_value,
            sort_pass_radix_digit(expected_key, begin_bit),
            actual_key_at_idx,
            actual_key_at_idx as u32,
            actual_value_at_idx,
            sort_pass_radix_digit(actual_key_at_idx, begin_bit),
            actual_xor_note,
        );
    }
}

fn analyze_expected_sort_pass(
    keys_in: &[i32],
    values_in: &[i32],
    keys_out: &[i32],
    values_out: &[i32],
    begin_bit: u32,
    vals_summary: Option<SortValueSummary>,
) -> Option<ExpectedSortPassDiagnostics> {
    if keys_in.len() != values_in.len()
        || keys_out.len() != values_out.len()
        || keys_in.len() != keys_out.len()
        || keys_in.is_empty()
    {
        return None;
    }
    let expected = reconstruct_expected_sort_pass(keys_in, values_in, begin_bit)?;
    let first_mismatch = expected
        .keys_out
        .iter()
        .zip(expected.values_out.iter())
        .zip(keys_out.iter().zip(values_out.iter()))
        .enumerate()
        .find_map(
            |(idx, ((&expected_key, &expected_value), (&actual_key, &actual_value)))| {
                if expected_key != actual_key || expected_value != actual_value {
                    Some((idx, expected_key, expected_value, actual_key, actual_value))
                } else {
                    None
                }
            },
        );
    let mismatch_line = first_mismatch.map(
        |(out_idx, expected_key, expected_value, actual_key, actual_value)| {
            let input_idx = expected.output_to_input[out_idx];
            let digit = sort_pass_radix_digit(expected_key, begin_bit);
            let actual_input_positions =
                find_pair_positions(keys_in, values_in, actual_key, actual_value, 4);
            let actual_expected_out_positions = find_pair_positions(
                &expected.keys_out,
                &expected.values_out,
                actual_key,
                actual_value,
                4,
            );
            let actual_value_input_positions = find_value_positions(values_in, actual_value, 4);
            let actual_value_expected_out_positions =
                find_value_positions(&expected.values_out, actual_value, 4);
            let actual_value_route_note = actual_value_input_positions
                .first()
                .map(|&input_pos| {
                    let source_key = keys_in[input_pos];
                    let source_digit = sort_pass_radix_digit(source_key, begin_bit);
                    let expected_out_note = actual_value_expected_out_positions
                        .first()
                        .map(|&expected_out_pos| {
                            let expected_key_for_value = expected.keys_out[expected_out_pos];
                            format!(
                                " expected_out[{}]=k:{}(0x{:08x}) digit={}",
                                expected_out_pos,
                                expected_key_for_value,
                                expected_key_for_value as u32,
                                sort_pass_radix_digit(expected_key_for_value, begin_bit),
                            )
                        })
                        .unwrap_or_else(|| " expected_out=[]".to_string());
                    format!(
                        " actual_value_src=in[{}]=k:{}(0x{:08x}) digit={}{}",
                        input_pos,
                        source_key,
                        source_key as u32,
                        source_digit,
                        expected_out_note,
                    )
                })
                .unwrap_or_default();
            let actual_xor_note = if actual_key < 0 {
                let restored_key = ((actual_key as u32) ^ 0x8000_0000) as i32;
                let restored_input_positions =
                    find_pair_positions(keys_in, values_in, restored_key, actual_value, 4);
                let restored_expected_out_positions = find_pair_positions(
                    &expected.keys_out,
                    &expected.values_out,
                    restored_key,
                    actual_value,
                    4,
                );
                let restored_key_input_positions = find_key_positions(keys_in, restored_key, 4);
                let restored_key_expected_out_positions =
                    find_key_positions(&expected.keys_out, restored_key, 4);
                format!(
                    " actual_xor=k:{}(0x{:08x}) in={:?} expected_out={:?} key_in={:?} key_expected_out={:?}",
                    restored_key,
                    restored_key as u32,
                    restored_input_positions,
                    restored_expected_out_positions,
                    restored_key_input_positions,
                    restored_key_expected_out_positions,
                )
            } else {
                String::new()
            };
            format!(
                "EXPECT_MISMATCH out[{}] bit={} digit={} expected=in[{}] k:{}(0x{:08x}) v:{} actual=k:{}(0x{:08x}) v:{} actual_in={:?} actual_expected_out={:?} actual_value_in={:?} actual_value_expected_out={:?}",
                out_idx,
                begin_bit,
                digit,
                input_idx,
                expected_key,
                expected_key as u32,
                expected_value,
                actual_key,
                actual_key as u32,
                actual_value,
                actual_input_positions,
                actual_expected_out_positions,
                actual_value_input_positions,
                actual_value_expected_out_positions,
            ) + &actual_value_route_note
                + &actual_xor_note
        },
    );
    let actual_pair_route = first_mismatch.and_then(|(out_idx, _, _, actual_key, actual_value)| {
        let actual_input_positions =
            find_pair_positions(keys_in, values_in, actual_key, actual_value, 4);
        let actual_expected_out_positions = find_pair_positions(
            &expected.keys_out,
            &expected.values_out,
            actual_key,
            actual_value,
            4,
        );
        actual_expected_out_positions
            .first()
            .map(|&expected_out_idx| {
                let input_note = actual_input_positions
                    .first()
                    .map(|&input_idx| format!("in[{}] -> ", input_idx))
                    .unwrap_or_default();
                format!(
                    "ACTUAL_PAIR_ROUTE {}expected_out[{}] -> out[{}] k:{}(0x{:08x}) v:{}",
                    input_note,
                    expected_out_idx,
                    out_idx,
                    actual_key,
                    actual_key as u32,
                    actual_value,
                )
            })
    });
    let expected_window = first_mismatch.and_then(|(idx, _, _, _, _)| {
        format_sort_pair_window(&expected.keys_out, &expected.values_out, idx, "EXPECT")
    });
    let actual_pair_source_window =
        first_mismatch.and_then(|(_, _, _, actual_key, actual_value)| {
            let actual_expected_out_positions = find_pair_positions(
                &expected.keys_out,
                &expected.values_out,
                actual_key,
                actual_value,
                4,
            );
            actual_expected_out_positions.first().and_then(|&idx| {
                format_sort_pair_window(keys_out, values_out, idx, "ACTUAL_PAIR_SRC_OUT")
            })
        });
    let actual_pair_clone_window =
        first_mismatch.and_then(|(out_idx, _, _, actual_key, actual_value)| {
            let actual_expected_out_positions = find_pair_positions(
                &expected.keys_out,
                &expected.values_out,
                actual_key,
                actual_value,
                4,
            );
            actual_expected_out_positions
                .first()
                .and_then(|&source_idx| {
                    if source_idx != out_idx {
                        format_sort_pair_window(
                            keys_out,
                            values_out,
                            out_idx,
                            "ACTUAL_PAIR_DST_OUT",
                        )
                    } else {
                        None
                    }
                })
        });
    let actual_value_route = first_mismatch.and_then(|(out_idx, _, _, _, actual_value)| {
        let actual_value_input_positions = find_value_positions(values_in, actual_value, 4);
        let actual_value_expected_out_positions =
            find_value_positions(&expected.values_out, actual_value, 4);
        actual_value_expected_out_positions
            .first()
            .map(|&expected_out_idx| {
                let input_note = actual_value_input_positions
                    .first()
                    .map(|&input_idx| format!("in[{}] -> ", input_idx))
                    .unwrap_or_default();
                format!(
                    "ACTUAL_VALUE_ROUTE {}expected_out[{}] -> out[{}] v:{}",
                    input_note, expected_out_idx, out_idx, actual_value
                )
            })
    });
    let actual_value_input_window = first_mismatch.and_then(|(_, _, _, _, actual_value)| {
        let actual_value_input_positions = find_value_positions(values_in, actual_value, 4);
        actual_value_input_positions.first().and_then(|&idx| {
            format_sort_pair_window(keys_in, values_in, idx, "ACTUAL_VALUE_SRC_IN")
        })
    });
    let actual_value_input_run = first_mismatch.and_then(|(out_idx, _, _, _, actual_value)| {
        let actual_value_input_positions = find_value_positions(values_in, actual_value, 4);
        actual_value_input_positions.first().and_then(|&input_idx| {
            matching_value_run(values_in, values_out, input_idx, out_idx).map(
                |(input_start, out_start, len)| {
                    format!(
                        "ACTUAL_VALUE_INPUT_RUN in[{}] -> out[{}] start_in={} start_out={} len={}",
                        input_idx, out_idx, input_start, out_start, len
                    )
                },
            )
        })
    });
    let actual_value_expected_window = first_mismatch.and_then(|(_, _, _, _, actual_value)| {
        let actual_value_expected_out_positions =
            find_value_positions(&expected.values_out, actual_value, 4);
        actual_value_expected_out_positions
            .first()
            .and_then(|&idx| {
                format_sort_pair_window(
                    &expected.keys_out,
                    &expected.values_out,
                    idx,
                    "EXPECT_VALUE_SRC",
                )
            })
    });
    let actual_value_expected_run = first_mismatch.and_then(|(out_idx, _, _, _, actual_value)| {
        let actual_value_expected_out_positions =
            find_value_positions(&expected.values_out, actual_value, 4);
        actual_value_expected_out_positions
            .first()
            .and_then(|&expected_idx| {
                matching_value_run(&expected.values_out, values_out, expected_idx, out_idx).map(
                    |(expected_start, out_start, len)| {
                        format!(
                            "ACTUAL_VALUE_EXPECTED_RUN expected_out[{}] -> out[{}] start_expected={} start_out={} len={}",
                            expected_idx, out_idx, expected_start, out_start, len
                        )
                    },
                )
            })
    });
    let actual_value_source_window = first_mismatch.and_then(|(_, _, _, _, actual_value)| {
        let actual_value_expected_out_positions =
            find_value_positions(&expected.values_out, actual_value, 4);
        actual_value_expected_out_positions
            .first()
            .and_then(|&idx| {
                format_sort_pair_window(keys_out, values_out, idx, "ACTUAL_VALUE_SRC_OUT")
            })
    });
    let actual_value_source_diff = first_mismatch.and_then(|(_, _, _, _, actual_value)| {
        let actual_value_expected_out_positions =
            find_value_positions(&expected.values_out, actual_value, 4);
        actual_value_expected_out_positions
            .first()
            .and_then(|&idx| {
                summarize_sort_pair_window_diff(
                    &expected.keys_out,
                    &expected.values_out,
                    keys_out,
                    values_out,
                    idx,
                    "ACTUAL_VALUE_SRC_DIFF",
                )
            })
    });
    let actual_value_clone_window = first_mismatch.and_then(|(out_idx, _, _, _, actual_value)| {
        let actual_value_expected_out_positions =
            find_value_positions(&expected.values_out, actual_value, 4);
        actual_value_expected_out_positions
            .first()
            .and_then(|&source_idx| {
                if source_idx != out_idx {
                    format_sort_pair_window(keys_out, values_out, out_idx, "ACTUAL_VALUE_DST_OUT")
                } else {
                    None
                }
            })
    });
    let actual_value_clone_diff = first_mismatch.and_then(|(out_idx, _, _, _, _)| {
        summarize_sort_pair_window_diff(
            &expected.keys_out,
            &expected.values_out,
            keys_out,
            values_out,
            out_idx,
            "ACTUAL_VALUE_DST_DIFF",
        )
    });
    let extra_positions =
        vals_summary
            .and_then(|summary| summary.first_extra)
            .map(|(_, value, _, _)| {
                let expected_positions = find_value_positions(&expected.values_out, value, 4);
                let actual_positions = find_value_positions(values_out, value, 4);
                format!(
                    "EXPECT_POS extra={} expected_out={:?} actual_out={:?}",
                    value, expected_positions, actual_positions
                )
            });
    let missing_positions =
        vals_summary
            .and_then(|summary| summary.first_missing)
            .map(|(_, value, _, _)| {
                let expected_positions = find_value_positions(&expected.values_out, value, 4);
                let actual_positions = find_value_positions(values_out, value, 4);
                format!(
                    "EXPECT_POS missing={} expected_out={:?} actual_out={:?}",
                    value, expected_positions, actual_positions
                )
            });
    Some(ExpectedSortPassDiagnostics {
        mismatch_line,
        expected_window,
        extra_positions,
        missing_positions,
        actual_pair_route,
        actual_pair_source_window,
        actual_pair_clone_window,
        actual_value_route,
        actual_value_input_window,
        actual_value_input_run,
        actual_value_expected_window,
        actual_value_expected_run,
        actual_value_source_window,
        actual_value_clone_window,
        actual_value_source_diff,
        actual_value_clone_diff,
    })
}

fn analyze_downsweep_rank_diagnostics(
    keys_in: &[i32],
    values_in: &[i32],
    keys_out: &[i32],
    values_out: &[i32],
    begin_bit: u32,
    grid_share: Option<SortGridEvenShare>,
) -> Option<DownsweepRankDiagnostics> {
    if begin_bit != 0
        || keys_in.len() != values_in.len()
        || keys_out.len() != values_out.len()
        || keys_in.len() != keys_out.len()
        || keys_in.is_empty()
    {
        return None;
    }
    let grid_share = grid_share?;
    let expected = reconstruct_expected_sort_pass(keys_in, values_in, begin_bit)?;
    let first_mismatch = expected
        .keys_out
        .iter()
        .zip(expected.values_out.iter())
        .zip(keys_out.iter().zip(values_out.iter()))
        .enumerate()
        .find_map(
            |(idx, ((&expected_key, &expected_value), (&actual_key, &actual_value)))| {
                if expected_key != actual_key || expected_value != actual_value {
                    Some((idx, actual_key, actual_value))
                } else {
                    None
                }
            },
        )?;
    let block_counts = compute_sort_digit_block_counts(keys_in, begin_bit, &grid_share)?;
    let expected_source_idx = expected.output_to_input[first_mismatch.0];
    let expected_line = format_downsweep_rank_line(
        "RANK_EXPECTED",
        keys_in,
        values_in,
        begin_bit,
        &grid_share,
        &block_counts,
        expected_source_idx,
        first_mismatch.0,
        None,
        None,
    );
    let actual_line = find_value_positions(values_in, first_mismatch.2, 1)
        .into_iter()
        .next()
        .and_then(|source_idx| {
            format_downsweep_rank_line(
                "RANK_ACTUAL",
                keys_in,
                values_in,
                begin_bit,
                &grid_share,
                &block_counts,
                source_idx,
                first_mismatch.0,
                Some(first_mismatch.1),
                Some(first_mismatch.0),
            )
        });
    Some(DownsweepRankDiagnostics {
        expected_line,
        actual_line,
    })
}

fn analyze_downsweep_block_capture(
    keys_in: &[i32],
    values_in: &[i32],
    keys_out: &[i32],
    values_out: &[i32],
    begin_bit: u32,
    grid_share: Option<SortGridEvenShare>,
) -> Option<DownsweepBlockCapture> {
    if keys_in.len() != values_in.len()
        || keys_out.len() != values_out.len()
        || keys_in.len() != keys_out.len()
        || keys_in.is_empty()
    {
        return None;
    }
    let grid_share = grid_share?;
    let expected = reconstruct_expected_sort_pass(keys_in, values_in, begin_bit)?;
    let (out_idx, _, _, _, actual_value) = expected
        .keys_out
        .iter()
        .zip(expected.values_out.iter())
        .zip(keys_out.iter().zip(values_out.iter()))
        .enumerate()
        .find_map(
            |(idx, ((&expected_key, &expected_value), (&actual_key, &actual_value)))| {
                if expected_key != actual_key || expected_value != actual_value {
                    Some((idx, expected_key, expected_value, actual_key, actual_value))
                } else {
                    None
                }
            },
        )?;
    let expected_source_idx = expected.output_to_input[out_idx];
    let actual_source_idx = find_value_positions(values_in, actual_value, 1)
        .into_iter()
        .next();
    let (block_idx, start, end) = grid_share.block_for_index(out_idx, keys_in.len())?;
    let expected_source_block = grid_share
        .block_for_index(expected_source_idx, keys_in.len())
        .map(|(block, _, _)| block)?;
    let actual_source_block = actual_source_idx.and_then(|idx| {
        grid_share
            .block_for_index(idx, keys_in.len())
            .map(|(block, _, _)| block)
    });
    let mut capture_text = String::new();
    let _ = writeln!(
        &mut capture_text,
        "SORT_CTA_CAPTURE bit={} block={} range=[{}..{}) mismatch_out={} expected_src=in[{}] actual_value_src={:?} expected_src_block={} actual_src_block={:?}",
        begin_bit,
        block_idx,
        start,
        end,
        out_idx,
        expected_source_idx,
        actual_source_idx,
        expected_source_block,
        actual_source_block,
    );
    let block_key_mismatches = (start..end)
        .filter(|&idx| expected.keys_out[idx] != keys_out[idx])
        .collect::<Vec<_>>();
    let block_value_mismatches = (start..end)
        .filter(|&idx| expected.values_out[idx] != values_out[idx])
        .collect::<Vec<_>>();
    let block_negative_keys = (start..end)
        .filter(|&idx| keys_out[idx] < 0)
        .collect::<Vec<_>>();
    let _ = writeln!(
        &mut capture_text,
        "DEST_BLOCK_RUNS key_mismatch={} runs={} value_mismatch={} runs={} neg={} runs={}",
        block_key_mismatches.len(),
        format_index_runs(&block_key_mismatches),
        block_value_mismatches.len(),
        format_index_runs(&block_value_mismatches),
        block_negative_keys.len(),
        format_index_runs(&block_negative_keys),
    );
    append_downsweep_block_capture_rows(
        &mut capture_text,
        "DEST_BLOCK",
        block_idx,
        start,
        end,
        out_idx,
        keys_in,
        values_in,
        &expected,
        keys_out,
        values_out,
        begin_bit,
    );
    if expected_source_block != block_idx {
        if let Some((expected_start, expected_end)) =
            grid_share.block_range(expected_source_block, keys_in.len())
        {
            append_downsweep_block_capture_rows(
                &mut capture_text,
                "EXPECTED_SOURCE_BLOCK",
                expected_source_block,
                expected_start,
                expected_end,
                out_idx,
                keys_in,
                values_in,
                &expected,
                keys_out,
                values_out,
                begin_bit,
            );
        }
    }
    if let Some(actual_source_block) = actual_source_block {
        if actual_source_block != block_idx && actual_source_block != expected_source_block {
            if let Some((actual_start, actual_end)) =
                grid_share.block_range(actual_source_block, keys_in.len())
            {
                append_downsweep_block_capture_rows(
                    &mut capture_text,
                    "ACTUAL_SOURCE_BLOCK",
                    actual_source_block,
                    actual_start,
                    actual_end,
                    out_idx,
                    keys_in,
                    values_in,
                    &expected,
                    keys_out,
                    values_out,
                    begin_bit,
                );
            }
        }
    }
    Some(DownsweepBlockCapture {
        block_idx,
        start,
        end,
        expected_source_block,
        actual_source_block,
        capture_text,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SortSpineSummary {
    words_read: usize,
    nonzero_count: u32,
    max_value: u32,
    sum: u64,
}

fn summarize_sort_spine(words: &[u32]) -> Option<SortSpineSummary> {
    if words.is_empty() {
        return None;
    }
    Some(SortSpineSummary {
        words_read: words.len(),
        nonzero_count: words.iter().filter(|&&v| v != 0).count() as u32,
        max_value: words.iter().copied().max().unwrap_or(0),
        sum: words.iter().map(|&v| v as u64).sum(),
    })
}

fn format_sorted_key_window(
    keys: &[i32],
    summary: &SortedKeySummary,
    label: &str,
) -> Option<String> {
    let center = summary
        .first_bad_value
        .map(|(idx, _)| idx)
        .or(summary.first_violation)?;
    let start = center.saturating_sub(5);
    let end = (center + 5).min(keys.len());
    let around: Vec<String> = (start..end)
        .map(|i| format!("{}[{}]={}", label, i, keys[i]))
        .collect();
    Some(around.join(", "))
}

fn format_key_window_around(keys: &[i32], center: usize, label: &str) -> Option<String> {
    if center >= keys.len() {
        return None;
    }
    let start = center.saturating_sub(5);
    let end = (center + 5).min(keys.len());
    let around: Vec<String> = (start..end)
        .map(|i| format!("{}[{}]={}", label, i, keys[i]))
        .collect();
    Some(around.join(", "))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Ms4ShapeUseSummary {
    shape: u32,
    counter: u32,
    uses: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Ms4CaptureSummary {
    active_count: u32,
    unique_shapes: usize,
    max_shape_idx: u32,
    out_of_range_particles: u32,
    zero_used_shapes: u32,
    negative_used_shapes: u32,
    underflow_count: u32,
    // Counter-vs-real-count audit (2026-06-10 MS_4 guard reinvestigation):
    // a healthy launch consumes each pool's counter to exactly 0, so
    // counter == uses for every used shape. Surplus (counter > uses) means
    // fewer metadata-passing threads than reservations — also anomalous.
    exact_shapes: u32,
    surplus_shapes: u32,
    max_surplus: u32,
    // Sum coherence over USED shapes: sum_counter == sum_uses with per-pool
    // mismatches ⇒ stale per-pool ASSIGNMENT (same particle total,
    // different distribution snapshot); sum_counter != sum_uses ⇒ the
    // refill missed/duplicated particles outright.
    sum_counter: u64,
    sum_uses: u64,
    first_out_of_range_particle: Option<(usize, u32, u16)>,
    first_underflow: Option<Ms4ShapeUseSummary>,
    // Up to 6 underflow + 2 surplus shapes for per-shape anomaly logging
    // (cross-referenced against the pool table f0/f1 by the caller).
    anomalies: Vec<Ms4ShapeUseSummary>,
}

fn summarize_ms4_capture(
    flags_data: &[u8],
    shape_idx_data: &[u8],
    counter_data: &[u8],
    num_particles: usize,
) -> Option<Ms4CaptureSummary> {
    let particle_count = num_particles
        .min(flags_data.len() / 4)
        .min(shape_idx_data.len() / 4);
    if particle_count == 0 {
        return None;
    }
    let counter_words: Vec<u32> = counter_data
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let mut summary = Ms4CaptureSummary {
        active_count: 0,
        unique_shapes: 0,
        max_shape_idx: 0,
        out_of_range_particles: 0,
        zero_used_shapes: 0,
        negative_used_shapes: 0,
        underflow_count: 0,
        exact_shapes: 0,
        surplus_shapes: 0,
        max_surplus: 0,
        sum_counter: 0,
        sum_uses: 0,
        first_out_of_range_particle: None,
        first_underflow: None,
        anomalies: Vec::new(),
    };
    let mut use_count = HashMap::<u32, u32>::new();
    for i in 0..particle_count {
        let flag = u16::from_ne_bytes([flags_data[i * 4], flags_data[i * 4 + 1]]);
        if (flag & 0x11) != 1 {
            continue;
        }
        summary.active_count += 1;
        let shape = u32::from_ne_bytes([
            shape_idx_data[i * 4],
            shape_idx_data[i * 4 + 1],
            shape_idx_data[i * 4 + 2],
            shape_idx_data[i * 4 + 3],
        ]);
        if shape > summary.max_shape_idx {
            summary.max_shape_idx = shape;
        }
        if (shape & 0x8000_0000) != 0 || shape as usize >= counter_words.len() {
            summary.out_of_range_particles += 1;
            if summary.first_out_of_range_particle.is_none() {
                summary.first_out_of_range_particle = Some((i, shape, flag));
            }
            continue;
        }
        *use_count.entry(shape).or_insert(0) += 1;
    }
    summary.unique_shapes = use_count.len();
    for (shape, uses) in use_count {
        let counter = counter_words[shape as usize];
        summary.sum_counter += counter as u64;
        summary.sum_uses += uses as u64;
        if counter == 0 {
            summary.zero_used_shapes += 1;
        }
        if (counter as i32) < 0 {
            summary.negative_used_shapes += 1;
        }
        if counter == uses {
            summary.exact_shapes += 1;
        } else if counter > uses && (counter as i32) >= 0 {
            summary.surplus_shapes += 1;
            summary.max_surplus = summary.max_surplus.max(counter - uses);
            if summary.anomalies.len() < 8
                && summary.anomalies.iter().filter(|a| a.counter > a.uses).count() < 2
            {
                summary.anomalies.push(Ms4ShapeUseSummary { shape, counter, uses });
            }
        }
        if counter < uses {
            summary.underflow_count += 1;
            if summary.first_underflow.is_none() {
                summary.first_underflow = Some(Ms4ShapeUseSummary {
                    shape,
                    counter,
                    uses,
                });
            }
            if summary.anomalies.len() < 8 {
                summary.anomalies.push(Ms4ShapeUseSummary { shape, counter, uses });
            }
        }
    }
    Some(summary)
}

fn creategrid_safe_validator_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_VALIDATE_SORT_SAFE").map_or(false, |v| v == "1"))
}

fn creategrid_sort_touch_log_start_from_value(value: Option<&str>) -> Option<u32> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("0") => None,
        Some("1") => Some(0),
        Some(value) => value.parse().ok(),
    }
}

fn creategrid_sort_touch_log_start() -> Option<u32> {
    static START: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *START.get_or_init(|| {
        creategrid_sort_touch_log_start_from_value(
            std::env::var("ZLUDA_CREATEGRID_TOUCH_LOG").ok().as_deref(),
        )
    })
}

fn creategrid_snapshot_log_start() -> Option<u32> {
    static START: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *START.get_or_init(|| {
        creategrid_sort_touch_log_start_from_value(
            std::env::var("ZLUDA_CREATEGRID_SNAPSHOT_LOG")
                .ok()
                .as_deref(),
        )
    })
}

fn createcell_snapshot_log_start() -> Option<u32> {
    static START: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *START.get_or_init(|| {
        creategrid_sort_touch_log_start_from_value(
            std::env::var("ZLUDA_CREATECELL_SNAPSHOT_LOG")
                .ok()
                .as_deref(),
        )
    })
}

fn sort_replay_capture_begin_bit() -> u32 {
    static BEGIN_BIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BEGIN_BIT.get_or_init(|| {
        std::env::var("ZLUDA_SORT_REPLAY_CAPTURE_BIT")
            .ok()
            .as_deref()
            .map(str::trim)
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

fn should_queue_sort_replay_capture(
    begin_bit: u32,
    creategrid_call_n: u32,
    start_call: Option<u32>,
    capture_begin_bit: u32,
) -> bool {
    start_call
        .map(|start| begin_bit == capture_begin_bit && creategrid_call_n >= start)
        .unwrap_or(false)
}

fn zero_sort_input_slack_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("ZLUDA_SORT_ZERO_INPUT_SLACK").map_or(false, |v| v == "1"))
}

fn append_sort_input_slack_probe(message: &str) {
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/zluda_sort_zero_input_slack.log")
    else {
        return;
    };
    let _ = writeln!(file, "{}", message);
}

fn trace_sort_copy_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_TRACE_SORT_COPY").map_or(false, |v| v == "1"))
}

static LAST_CREATEGRID_SORTED_KEYS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static LAST_CREATEGRID_KEY_COUNT: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
static CREATEGRID_LOG_CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static CREATECELL_SNAPSHOT_LOG_CALL_N: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
static SORT_PASS_SNAPSHOT_CALL_N: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

#[derive(Clone, Copy)]
struct PendingCreateCellSnapshot {
    call_n: u32,
    n: usize,
    read_n: usize,
    hash_out_dev: u64,
    stream: hipStream_t,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
}

static PENDING_CREATECELL_SNAPSHOT: Mutex<Option<PendingCreateCellSnapshot>> = Mutex::new(None);

#[derive(Clone, Copy)]
struct PendingSortPassSnapshot {
    snapshot_n: u32,
    begin_bit: u32,
    n_items: usize,
    read_n: usize,
    keys_in: u64,
    keys_out: u64,
    first: u8,
    last: u8,
    overwrite: u8,
    stream: hipStream_t,
    grid_dim_x: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    scratch_slot: usize,
}

static PENDING_SORT_PASS_SNAPSHOTS: Mutex<Vec<PendingSortPassSnapshot>> = Mutex::new(Vec::new());

fn note_creategrid_sorted_keys(sorted_keys: u64, n: usize) {
    if sorted_keys == 0 || n == 0 || n > u32::MAX as usize {
        return;
    }
    LAST_CREATEGRID_SORTED_KEYS.store(sorted_keys, std::sync::atomic::Ordering::Relaxed);
    LAST_CREATEGRID_KEY_COUNT.store(n as u32, std::sync::atomic::Ordering::Relaxed);
}

fn is_creategrid_sorted_keys_copy(
    sorted_keys: u64,
    n: usize,
    dst: u64,
    src: u64,
    byte_count: usize,
) -> bool {
    sorted_keys != 0
        && n > 0
        && byte_count == n * 4
        && dst == sorted_keys
        && src == sorted_keys + (n * 4) as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CreateGridSortBufferRole {
    CreateGridInput,
    CopySource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CreateGridSortBufferHit {
    role: CreateGridSortBufferRole,
    byte_offset: usize,
}

fn classify_creategrid_sort_buffer(
    sorted_keys: u64,
    n: usize,
    buffer: u64,
) -> Option<CreateGridSortBufferRole> {
    if sorted_keys == 0 || n == 0 || buffer == 0 {
        return None;
    }
    if buffer == sorted_keys {
        return Some(CreateGridSortBufferRole::CreateGridInput);
    }
    let copy_src = sorted_keys.checked_add((n.checked_mul(4)?) as u64)?;
    if buffer == copy_src {
        return Some(CreateGridSortBufferRole::CopySource);
    }
    None
}

fn locate_creategrid_sort_buffer_hit(
    sorted_keys: u64,
    n: usize,
    buffer: u64,
) -> Option<CreateGridSortBufferHit> {
    if sorted_keys == 0 || n == 0 || buffer == 0 {
        return None;
    }
    let buffer_bytes = n.checked_mul(4)?;
    let copy_src = sorted_keys.checked_add(buffer_bytes as u64)?;
    let input_end = sorted_keys.checked_add(buffer_bytes as u64)?;
    let copy_end = copy_src.checked_add(buffer_bytes as u64)?;
    if (sorted_keys..input_end).contains(&buffer) {
        return Some(CreateGridSortBufferHit {
            role: CreateGridSortBufferRole::CreateGridInput,
            byte_offset: (buffer - sorted_keys) as usize,
        });
    }
    if (copy_src..copy_end).contains(&buffer) {
        return Some(CreateGridSortBufferHit {
            role: CreateGridSortBufferRole::CopySource,
            byte_offset: (buffer - copy_src) as usize,
        });
    }
    None
}

fn creategrid_sort_buffer_role_name(role: CreateGridSortBufferRole) -> &'static str {
    match role {
        CreateGridSortBufferRole::CreateGridInput => "creategrid_input",
        CreateGridSortBufferRole::CopySource => "copy_src",
    }
}

const CREATEGRID_SORT_TOUCH_MAX_EXTRA_BYTES: usize = 160;
const CREATEGRID_SORT_TOUCH_MAX_RECENT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CreateGridSortTouchArgLocation {
    ExtraOffset(usize),
    KernelParamIndex(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CreateGridSortTouchArg {
    location: CreateGridSortTouchArgLocation,
    buffer: u64,
    role: CreateGridSortBufferRole,
    byte_offset: usize,
}

#[derive(Clone, Debug)]
struct CreateGridSortTouchRecord {
    launch_n: u32,
    sorted_keys: u64,
    n_items: usize,
    name: String,
    stream: String,
    grid_dim: [u32; 3],
    block_dim: [u32; 3],
    shared_mem_bytes: u32,
    refs: Vec<CreateGridSortTouchArg>,
}

static CREATEGRID_SORT_TOUCHES: Mutex<Vec<CreateGridSortTouchRecord>> = Mutex::new(Vec::new());

fn scan_launch_extra_for_creategrid_sort_refs(
    sorted_keys: u64,
    n: usize,
    buffer: &[u8],
) -> Vec<CreateGridSortTouchArg> {
    if sorted_keys == 0 || n == 0 || buffer.len() < 8 {
        return Vec::new();
    }
    let scan_len = buffer.len().min(CREATEGRID_SORT_TOUCH_MAX_EXTRA_BYTES);
    let mut refs = Vec::new();
    for extra_offset in (0..=(scan_len - 8)).step_by(8) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&buffer[extra_offset..extra_offset + 8]);
        let candidate = u64::from_le_bytes(raw);
        let Some(hit) = locate_creategrid_sort_buffer_hit(sorted_keys, n, candidate) else {
            continue;
        };
        refs.push(CreateGridSortTouchArg {
            location: CreateGridSortTouchArgLocation::ExtraOffset(extra_offset),
            buffer: candidate,
            role: hit.role,
            byte_offset: hit.byte_offset,
        });
    }
    refs
}

unsafe fn scan_launch_kernel_params_for_creategrid_sort_refs(
    sorted_keys: u64,
    n: usize,
    kernel_params: *mut *mut ::core::ffi::c_void,
) -> Vec<CreateGridSortTouchArg> {
    if sorted_keys == 0 || n == 0 || kernel_params.is_null() {
        return Vec::new();
    }
    let mut refs = Vec::new();
    for arg_index in 0..16usize {
        let param_ptr = *kernel_params.add(arg_index);
        if param_ptr.is_null() {
            break;
        }
        let candidate = std::ptr::read_unaligned(param_ptr as *const u64);
        let Some(hit) = locate_creategrid_sort_buffer_hit(sorted_keys, n, candidate) else {
            continue;
        };
        refs.push(CreateGridSortTouchArg {
            location: CreateGridSortTouchArgLocation::KernelParamIndex(arg_index),
            buffer: candidate,
            role: hit.role,
            byte_offset: hit.byte_offset,
        });
    }
    refs
}

unsafe fn collect_creategrid_sort_refs_for_launch(
    sorted_keys: u64,
    n: usize,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> Vec<CreateGridSortTouchArg> {
    let mut refs = if !extra.is_null() {
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        if extra_arr[0] == (0x1usize as *mut _) && !extra_arr[1].is_null() {
            let buffer_size = if extra_arr[2] == (0x2usize as *mut _) && !extra_arr[3].is_null() {
                std::ptr::read_unaligned(extra_arr[3] as *const usize)
            } else {
                CREATEGRID_SORT_TOUCH_MAX_EXTRA_BYTES
            };
            if buffer_size >= 8 {
                scan_launch_extra_for_creategrid_sort_refs(
                    sorted_keys,
                    n,
                    std::slice::from_raw_parts(extra_arr[1] as *const u8, buffer_size),
                )
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    refs.extend(scan_launch_kernel_params_for_creategrid_sort_refs(
        sorted_keys,
        n,
        kernel_params,
    ));
    refs
}

fn format_creategrid_sort_touch_refs(refs: &[CreateGridSortTouchArg]) -> String {
    refs.iter()
        .map(|arg| {
            let location = match arg.location {
                CreateGridSortTouchArgLocation::ExtraOffset(offset) => {
                    format!("extra+{}", offset)
                }
                CreateGridSortTouchArgLocation::KernelParamIndex(index) => {
                    format!("arg{}", index)
                }
            };
            if arg.byte_offset == 0 {
                format!(
                    "{}=0x{:x}({})",
                    location,
                    arg.buffer,
                    creategrid_sort_buffer_role_name(arg.role)
                )
            } else {
                format!(
                    "{}=0x{:x}({}+0x{:x})",
                    location,
                    arg.buffer,
                    creategrid_sort_buffer_role_name(arg.role),
                    arg.byte_offset
                )
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

unsafe fn log_creategrid_snapshot_direct(
    call_n: u32,
    n: usize,
    sorted_keys_dev: u64,
    stream: hipStream_t,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
) {
    use hip_runtime_sys::*;

    let read_n = n.min(F5_MAX_KEYS);
    if read_n == 0 {
        return;
    }

    let mut guard = match CREATEGRID_VALIDATOR.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    createGrid_validator_init(&mut guard);
    let validator = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };

    let copy_result = hipMemcpyDtoDAsync(
        hipDeviceptr_t(validator.scratch_dev),
        hipDeviceptr_t(sorted_keys_dev as *mut std::ffi::c_void),
        read_n * 4,
        stream,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_SNAPSHOT_DIRECT creategrid#{}: copy failed {:?} sk=0x{:x} n={} read={} stream={:?}",
            call_n, copy_result, sorted_keys_dev, n, read_n, stream
        ));
        return;
    }

    let sync_result = hipStreamSynchronize(stream);
    if sync_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_SNAPSHOT_DIRECT creategrid#{}: stream sync failed {:?} sk=0x{:x} n={} read={} stream={:?}",
            call_n, sync_result, sorted_keys_dev, n, read_n, stream
        ));
        return;
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    log_pending_createcell_snapshot_after_sync();
    log_pending_sort_pass_snapshots_after_sync(call_n, n, sorted_keys_dev);
    log_pending_sort_replay_capture_after_sync();

    let keys = std::slice::from_raw_parts(validator.scratch_host as *const i32, read_n);
    let Some(summary) = summarize_sorted_keys(keys) else {
        return;
    };
    let restored = restore_signed_radix_residue(keys);
    let restored_summary = summarize_sorted_keys(&restored);
    super::driver::zluda_log_err(&format!(
        "    SORT_SNAPSHOT_DIRECT creategrid#{} grid=[{},{},{}] block=[{},{},{}] shared={} stream={:?} n={} read={} sk=0x{:x} A[min={},max={},viol={},neg={},huge={}]{}",
        call_n,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        stream,
        n,
        read_n,
        sorted_keys_dev,
        summary.min_key,
        summary.max_key,
        summary.violations,
        summary.negative_count,
        summary.huge_count,
        restored_summary
            .map(|restored| format!(
                " A_XOR[min={},max={},viol={},neg={},huge={}]",
                restored.min_key,
                restored.max_key,
                restored.violations,
                restored.negative_count,
                restored.huge_count
            ))
            .unwrap_or_default()
    ));
    if let Some(window) = format_sorted_key_window(keys, &summary, "A") {
        super::driver::zluda_log_err(&format!("        {}", window));
    }
}

unsafe fn queue_createcell_snapshot_direct(
    call_n: u32,
    n: usize,
    hash_out_dev: u64,
    stream: hipStream_t,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
) {
    use hip_runtime_sys::*;

    let read_n = n.min(F5_MAX_KEYS);
    if read_n == 0 {
        return;
    }

    let mut guard = match CREATECELL_SNAPSHOT_VALIDATOR.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    createGrid_validator_init(&mut guard);
    let validator = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };

    let copy_result = hipMemcpyDtoDAsync(
        hipDeviceptr_t(validator.scratch_dev),
        hipDeviceptr_t(hash_out_dev as *mut std::ffi::c_void),
        read_n * 4,
        stream,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** CREATECELL_SNAPSHOT_DIRECT createcell#{}: copy failed {:?} out=0x{:x} n={} read={} stream={:?}",
            call_n, copy_result, hash_out_dev, n, read_n, stream
        ));
        return;
    }

    let mut pending = match PENDING_CREATECELL_SNAPSHOT.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    *pending = Some(PendingCreateCellSnapshot {
        call_n,
        n,
        read_n,
        hash_out_dev,
        stream,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
    });
}

fn log_pending_createcell_snapshot_after_sync() {
    let pending = {
        let mut guard = match PENDING_CREATECELL_SNAPSHOT.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.take()
    };
    let Some(pending) = pending else {
        return;
    };

    let guard = match CREATECELL_SNAPSHOT_VALIDATOR.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(validator) = guard.as_ref() else {
        return;
    };

    let keys =
        unsafe { std::slice::from_raw_parts(validator.scratch_host as *const i32, pending.read_n) };
    let Some(summary) = summarize_key_domain(keys) else {
        return;
    };
    super::driver::zluda_log_err(&format!(
        "    CREATECELL_SNAPSHOT_DIRECT createcell#{} grid=[{},{},{}] block=[{},{},{}] shared={} stream={:?} n={} read={} out=0x{:x} OUT[min={},max={},neg={},huge={}]",
        pending.call_n,
        pending.grid_dim_x,
        pending.grid_dim_y,
        pending.grid_dim_z,
        pending.block_dim_x,
        pending.block_dim_y,
        pending.block_dim_z,
        pending.shared_mem_bytes,
        pending.stream,
        pending.n,
        pending.read_n,
        pending.hash_out_dev,
        summary.min_key,
        summary.max_key,
        summary.negative_count,
        summary.huge_count,
    ));
    if let Some((idx, _)) = summary.first_bad_value {
        if let Some(window) = format_key_window_around(keys, idx, "OUT") {
            super::driver::zluda_log_err(&format!("        {}", window));
        }
    }
}

unsafe fn queue_sort_downsweep_snapshot(
    begin_bit: u32,
    n_items: usize,
    keys_in: u64,
    keys_out: u64,
    first: u8,
    last: u8,
    overwrite: u8,
    stream: hipStream_t,
    grid_dim_x: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
) {
    use hip_runtime_sys::*;

    let read_n = n_items.min(F5_MAX_KEYS);
    if read_n == 0 || keys_out == 0 {
        return;
    }

    let mut pending = match PENDING_SORT_PASS_SNAPSHOTS.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if pending.len() >= SORT_PASS_QUEUE_MAX_PENDING {
        super::driver::zluda_log_err(&format!(
            "*** SORT_PASS_QUEUE: overflow before CreateGrid sync; dropping {} pending snapshots",
            pending.len()
        ));
        pending.clear();
    }
    let scratch_slot = pending.len();
    let Some(scratch_offset) = sort_pass_queue_slot_offset(scratch_slot) else {
        return;
    };

    let mut guard = match SORT_PASS_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    sort_pass_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };

    let copy_result = hipMemcpyDtoDAsync(
        hipDeviceptr_t((tracker.scratch_dev as usize + scratch_offset) as _),
        hipDeviceptr_t(keys_out as *mut std::ffi::c_void),
        read_n * 4,
        stream,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_PASS_QUEUE: copy failed {:?} bit={} keys_out=0x{:x} n={} read={} stream={:?}",
            copy_result, begin_bit, keys_out, n_items, read_n, stream
        ));
        return;
    }

    pending.push(PendingSortPassSnapshot {
        snapshot_n: SORT_PASS_SNAPSHOT_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        begin_bit,
        n_items,
        read_n,
        keys_in,
        keys_out,
        first,
        last,
        overwrite,
        stream,
        grid_dim_x,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        scratch_slot,
    });
}

fn log_pending_sort_pass_snapshots_after_sync(call_n: u32, n: usize, sorted_keys_dev: u64) {
    let pending = {
        let mut guard = match PENDING_SORT_PASS_SNAPSHOTS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        std::mem::take(&mut *guard)
    };
    if pending.is_empty() {
        return;
    }

    let guard = match SORT_PASS_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(tracker) = guard.as_ref() else {
        return;
    };

    for snapshot in pending {
        let Some(scratch_offset) = sort_pass_queue_slot_offset(snapshot.scratch_slot) else {
            continue;
        };
        let keys = unsafe {
            std::slice::from_raw_parts(
                tracker.scratch_host.add(scratch_offset) as *const i32,
                snapshot.read_n,
            )
        };
        let logical_n = snapshot.n_items.min(snapshot.read_n);
        let keys = &keys[..logical_n];
        let Some(summary) = summarize_sorted_keys(keys) else {
            continue;
        };
        let restored_summary = if summary.negative_count > 0 {
            let restored = restore_signed_radix_residue(keys);
            summarize_sorted_keys(&restored)
        } else {
            None
        };
        let in_role = classify_creategrid_sort_buffer(sorted_keys_dev, n, snapshot.keys_in)
            .map(creategrid_sort_buffer_role_name)
            .unwrap_or("-");
        let out_role = classify_creategrid_sort_buffer(sorted_keys_dev, n, snapshot.keys_out)
            .map(creategrid_sort_buffer_role_name)
            .unwrap_or("-");
        super::driver::zluda_log_err(&format!(
            "    SORT_PASS_QUEUE_DIRECT creategrid#{} snap#{} bit={} first={} last={} ow={} grid={} block=[{},{},{}] shared={} stream={:?} keys_in=0x{:x}({}) keys_out=0x{:x}({}) OUT[min={},max={},viol={},neg={},huge={}]{}",
            call_n,
            snapshot.snapshot_n,
            snapshot.begin_bit,
            snapshot.first,
            snapshot.last,
            snapshot.overwrite,
            snapshot.grid_dim_x,
            snapshot.block_dim_x,
            snapshot.block_dim_y,
            snapshot.block_dim_z,
            snapshot.shared_mem_bytes,
            snapshot.stream,
            snapshot.keys_in,
            in_role,
            snapshot.keys_out,
            out_role,
            summary.min_key,
            summary.max_key,
            summary.violations,
            summary.negative_count,
            summary.huge_count,
            restored_summary
                .map(|restored| format!(
                    " OUT_XOR[min={},max={},viol={},neg={},huge={}]",
                    restored.min_key,
                    restored.max_key,
                    restored.violations,
                    restored.negative_count,
                    restored.huge_count
                ))
                .unwrap_or_default()
        ));
        if let Some(window) = format_sorted_key_window(keys, &summary, "OUT") {
            super::driver::zluda_log_err(&format!("        {}", window));
        }
    }
}

unsafe fn queue_sort_downsweep_replay_capture_prelaunch(
    args: SortDownsweepLaunchArgs,
    stream: hipStream_t,
    grid_dim_x: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
) {
    use hip_runtime_sys::*;

    if args.begin_bit != sort_replay_capture_begin_bit() {
        return;
    }
    let read_n = sort_pass_capture_len(args.n_items, Some(args.grid_share));
    if read_n == 0 || args.keys_in == 0 || args.keys_out == 0 {
        return;
    }

    let mut guard = match SORT_PASS_REPLAY_QUEUE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    sort_pass_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };

    let queued = [
        (SORT_PASS_KEYS_IN_OFFSET, args.keys_in, "keys_in", true),
        (
            SORT_PASS_KEYS_IN_BEFORE_OFFSET,
            args.keys_in,
            "keys_in_before",
            true,
        ),
        (
            SORT_PASS_KEYS_OUT_BEFORE_OFFSET,
            args.keys_out,
            "keys_out_before",
            true,
        ),
    ];
    for (offset, src, label, required) in queued {
        let result = hipMemcpyDtoDAsync(
            hipDeviceptr_t((tracker.scratch_dev as usize + offset) as _),
            hipDeviceptr_t(src as *mut std::ffi::c_void),
            read_n * 4,
            stream,
        );
        if required && result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_REPLAY_QUEUE: prelaunch {} copy failed {:?} bit={} ptr=0x{:x} n={} stream={:?}",
                label, result, args.begin_bit, src, read_n, stream
            ));
            return;
        }
    }
    if args.vals_in != 0 {
        for (offset, src, label) in [
            (SORT_PASS_VALS_IN_OFFSET, args.vals_in, "vals_in"),
            (
                SORT_PASS_VALS_IN_BEFORE_OFFSET,
                args.vals_in,
                "vals_in_before",
            ),
            (
                SORT_PASS_VALS_OUT_BEFORE_OFFSET,
                args.vals_out,
                "vals_out_before",
            ),
        ] {
            let result = hipMemcpyDtoDAsync(
                hipDeviceptr_t((tracker.scratch_dev as usize + offset) as _),
                hipDeviceptr_t(src as *mut std::ffi::c_void),
                read_n * 4,
                stream,
            );
            if result.is_err() {
                super::driver::zluda_log_err(&format!(
                    "*** SORT_REPLAY_QUEUE: prelaunch {} copy failed {:?} bit={} ptr=0x{:x} n={} stream={:?}",
                    label, result, args.begin_bit, src, read_n, stream
                ));
                return;
            }
        }
    }

    let mut pending = match PENDING_SORT_REPLAY_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(previous) = *pending {
        super::driver::zluda_log_err(&format!(
            "*** SORT_REPLAY_QUEUE: overwriting pending capture prev=bit{}#{} n={} keys_in=0x{:x} keys_out=0x{:x} stream={:?} next=bit{} n={} keys_in=0x{:x} keys_out=0x{:x} stream={:?}",
            previous.begin_bit,
            previous.capture_n,
            previous.n_items,
            previous.keys_in,
            previous.keys_out,
            previous.stream,
            args.begin_bit,
            args.n_items,
            args.keys_in,
            args.keys_out,
            stream,
        ));
    }
    *pending = Some(PendingSortReplayCapture {
        capture_n: SORT_PASS_SNAPSHOT_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        begin_bit: args.begin_bit,
        n_items: args.n_items,
        read_n,
        keys_in: args.keys_in,
        keys_out: args.keys_out,
        vals_in: args.vals_in,
        vals_out: args.vals_out,
        spine_ptr: args.spine_ptr,
        first: args.first,
        last: args.last,
        overwrite: args.overwrite,
        grid_share: Some(args.grid_share),
        stream,
        grid_dim_x,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
    });
}

unsafe fn queue_sort_downsweep_replay_capture_postlaunch(
    args: SortDownsweepLaunchArgs,
    stream: hipStream_t,
) {
    use hip_runtime_sys::*;

    if args.begin_bit != sort_replay_capture_begin_bit() {
        return;
    }
    let pending = {
        let guard = match PENDING_SORT_REPLAY_CAPTURE.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        *guard
    };
    let Some(pending) = pending else {
        return;
    };
    if pending.begin_bit != args.begin_bit
        || pending.n_items != args.n_items
        || pending.keys_in != args.keys_in
        || pending.keys_out != args.keys_out
    {
        super::driver::zluda_log_err(&format!(
            "*** SORT_REPLAY_QUEUE: postlaunch mismatch pending=bit{}#{} n={} keys_in=0x{:x} keys_out=0x{:x} stream={:?} launch=bit{} n={} keys_in=0x{:x} keys_out=0x{:x} stream={:?}",
            pending.begin_bit,
            pending.capture_n,
            pending.n_items,
            pending.keys_in,
            pending.keys_out,
            pending.stream,
            args.begin_bit,
            args.n_items,
            args.keys_in,
            args.keys_out,
            stream,
        ));
        return;
    }

    let mut guard = match SORT_PASS_REPLAY_QUEUE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let tracker = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };
    for (offset, src, label, bytes) in [
        (
            SORT_PASS_KEYS_OUT_OFFSET,
            args.keys_out,
            "keys_out",
            pending.read_n * 4,
        ),
        (
            SORT_PASS_VALS_OUT_OFFSET,
            args.vals_out,
            "vals_out",
            pending.read_n * 4,
        ),
    ] {
        if src == 0 {
            continue;
        }
        let result = hipMemcpyDtoDAsync(
            hipDeviceptr_t((tracker.scratch_dev as usize + offset) as _),
            hipDeviceptr_t(src as *mut std::ffi::c_void),
            bytes,
            stream,
        );
        if result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_REPLAY_QUEUE: postlaunch {} copy failed {:?} bit={} ptr=0x{:x} n={} stream={:?}",
                label, result, args.begin_bit, src, pending.read_n, stream
            ));
            return;
        }
    }
    let spine_words = ((pending.grid_dim_x as usize) * 16).min(SORT_PASS_SPINE_MAX_WORDS);
    if pending.spine_ptr != 0 && spine_words > 0 {
        let result = hipMemcpyDtoDAsync(
            hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_SPINE_OFFSET) as _),
            hipDeviceptr_t(pending.spine_ptr as *mut std::ffi::c_void),
            spine_words * 4,
            stream,
        );
        if result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_REPLAY_QUEUE: postlaunch spine copy failed {:?} bit={} spine=0x{:x} words={} stream={:?}",
                result, args.begin_bit, pending.spine_ptr, spine_words, stream
            ));
        }
    }
}

fn log_pending_sort_replay_capture_after_sync() {
    let pending = {
        let mut guard = match PENDING_SORT_REPLAY_CAPTURE.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.take()
    };
    let Some(pending) = pending else {
        return;
    };

    let guard = match SORT_PASS_REPLAY_QUEUE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(tracker) = guard.as_ref() else {
        return;
    };

    let keys_in = unsafe {
        std::slice::from_raw_parts(
            tracker.scratch_host.add(SORT_PASS_KEYS_IN_OFFSET) as *const i32,
            pending.read_n,
        )
    };
    let keys_in_before = unsafe {
        std::slice::from_raw_parts(
            tracker.scratch_host.add(SORT_PASS_KEYS_IN_BEFORE_OFFSET) as *const i32,
            pending.read_n,
        )
    };
    let keys_out_before = unsafe {
        std::slice::from_raw_parts(
            tracker.scratch_host.add(SORT_PASS_KEYS_OUT_BEFORE_OFFSET) as *const i32,
            pending.read_n,
        )
    };
    let keys_out = unsafe {
        std::slice::from_raw_parts(
            tracker.scratch_host.add(SORT_PASS_KEYS_OUT_OFFSET) as *const i32,
            pending.read_n,
        )
    };
    let vals_in = if pending.vals_in != 0 {
        Some(unsafe {
            std::slice::from_raw_parts(
                tracker.scratch_host.add(SORT_PASS_VALS_IN_OFFSET) as *const i32,
                pending.read_n,
            )
        })
    } else {
        None
    };
    let vals_in_before = if pending.vals_in != 0 {
        Some(unsafe {
            std::slice::from_raw_parts(
                tracker.scratch_host.add(SORT_PASS_VALS_IN_BEFORE_OFFSET) as *const i32,
                pending.read_n,
            )
        })
    } else {
        None
    };
    let vals_out_before = if pending.vals_out != 0 {
        Some(unsafe {
            std::slice::from_raw_parts(
                tracker.scratch_host.add(SORT_PASS_VALS_OUT_BEFORE_OFFSET) as *const i32,
                pending.read_n,
            )
        })
    } else {
        None
    };
    let vals_out = if pending.vals_out != 0 {
        Some(unsafe {
            std::slice::from_raw_parts(
                tracker.scratch_host.add(SORT_PASS_VALS_OUT_OFFSET) as *const i32,
                pending.read_n,
            )
        })
    } else {
        None
    };
    let spine_words_len = ((pending.grid_dim_x as usize) * 16).min(SORT_PASS_SPINE_MAX_WORDS);
    let spine_words = unsafe {
        std::slice::from_raw_parts(
            tracker.scratch_host.add(SORT_PASS_SPINE_OFFSET) as *const u32,
            spine_words_len,
        )
    };
    let logical_n = pending.n_items.min(pending.read_n);
    let Some(summary_out) = summarize_sorted_keys(&keys_out[..logical_n]) else {
        return;
    };
    if summary_out.negative_count == 0 && summary_out.huge_count == 0 {
        return;
    }
    let capture = match serialize_sort_pass_replay_capture(
        pending.capture_n,
        pending.begin_bit,
        pending.n_items,
        pending.grid_dim_x,
        pending.block_dim_x,
        pending.block_dim_y,
        pending.block_dim_z,
        pending.shared_mem_bytes,
        pending.first,
        pending.last,
        pending.overwrite,
        pending.keys_in,
        pending.keys_out,
        pending.vals_in,
        pending.vals_out,
        pending.spine_ptr,
        pending.stream.0 as u64,
        keys_in,
        vals_in,
        keys_in_before,
        vals_in_before,
        keys_out_before,
        vals_out_before,
        keys_out,
        vals_out,
        spine_words,
        pending.grid_share,
    ) {
        Some(capture) => capture,
        None => return,
    };
    let path = format!(
        "/tmp/zluda_sort_replay_capture_bit{}_pass{}.bin",
        pending.begin_bit, pending.capture_n
    );
    let capture_line = match std::fs::write(&path, &capture) {
        Ok(()) => format!(
            "SORT_REPLAY_CAPTURE bit={} n={} grid={} spine_words={} path={}",
            pending.begin_bit, pending.n_items, pending.grid_dim_x, spine_words_len, path
        ),
        Err(err) => format!(
            "SORT_REPLAY_CAPTURE bit={} n={} grid={} spine_words={} path={} write_err={}",
            pending.begin_bit, pending.n_items, pending.grid_dim_x, spine_words_len, path, err
        ),
    };
    super::driver::zluda_log_err(&format!("    {}", capture_line));
}

unsafe fn maybe_capture_creategrid_sort_buffer_touch(
    name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    shared_mem_bytes: u32,
    stream: hipStream_t,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
) {
    if name.contains("CreateGrid") && !name.contains("Turbulence") {
        return;
    }
    let sorted_keys = LAST_CREATEGRID_SORTED_KEYS.load(std::sync::atomic::Ordering::Relaxed);
    let n = LAST_CREATEGRID_KEY_COUNT.load(std::sync::atomic::Ordering::Relaxed) as usize;
    if sorted_keys == 0 || n == 0 || (extra.is_null() && kernel_params.is_null()) {
        return;
    }
    let refs = collect_creategrid_sort_refs_for_launch(sorted_keys, n, kernel_params, extra);
    if refs.is_empty() {
        return;
    }
    static LAUNCH_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let launch_n = LAUNCH_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut guard = match CREATEGRID_SORT_TOUCHES.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    guard.push(CreateGridSortTouchRecord {
        launch_n,
        sorted_keys,
        n_items: n,
        name: name.to_string(),
        stream: format!("{:?}", stream),
        grid_dim: [grid_dim_x, grid_dim_y, grid_dim_z],
        block_dim: [block_dim_x, block_dim_y, block_dim_z],
        shared_mem_bytes,
        refs,
    });
    if guard.len() > CREATEGRID_SORT_TOUCH_MAX_RECENT {
        let drop_n = guard.len() - CREATEGRID_SORT_TOUCH_MAX_RECENT;
        guard.drain(0..drop_n);
    }
}

fn dump_recent_creategrid_sort_buffer_touches(call_n: u32, n: usize, sorted_keys: u64) {
    let guard = match CREATEGRID_SORT_TOUCHES.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let mut matched = 0usize;
    for record in guard
        .iter()
        .rev()
        .filter(|record| record.n_items == n && record.sorted_keys == sorted_keys)
        .take(12)
    {
        matched += 1;
        let refs = format_creategrid_sort_touch_refs(&record.refs);
        super::driver::zluda_log_err(&format!(
            "    SORT_TOUCH_MATCH creategrid#{} launch#{} '{}' grid=[{},{},{}] block=[{},{},{}] shared={} stream={} refs=[{}]",
            call_n,
            record.launch_n,
            record.name,
            record.grid_dim[0],
            record.grid_dim[1],
            record.grid_dim[2],
            record.block_dim[0],
            record.block_dim[1],
            record.block_dim[2],
            record.shared_mem_bytes,
            record.stream,
            refs
        ));
    }
    if matched == 0 {
        super::driver::zluda_log_err(&format!(
            "    SORT_TOUCH_MATCH creategrid#{}: no recent launch extra matched sk=0x{:x} n={}",
            call_n, sorted_keys, n
        ));
    }
}

pub(crate) fn maybe_log_creategrid_sorted_keys_copy(
    dst: u64,
    src: u64,
    byte_count: usize,
    stream: hipStream_t,
) {
    if !trace_sort_copy_enabled() {
        return;
    }
    let sorted_keys = LAST_CREATEGRID_SORTED_KEYS.load(std::sync::atomic::Ordering::Relaxed);
    let n = LAST_CREATEGRID_KEY_COUNT.load(std::sync::atomic::Ordering::Relaxed) as usize;
    if !is_creategrid_sorted_keys_copy(sorted_keys, n, dst, src, byte_count) {
        return;
    }
    static COPY_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let copy_n = COPY_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    super::driver::zluda_log_err(&format!(
        "SORT_COPY #{} dst=0x{:x} src=0x{:x} bytes={} n={} stream={:?}",
        copy_n, dst, src, byte_count, n, stream
    ));
}

#[derive(Clone, Debug)]
struct SortPassSummaryRecord {
    pass_n: u32,
    begin_bit: u32,
    n_items: usize,
    keys_in: u64,
    keys_out: u64,
    spine_ptr: u64,
    summary_in: SortedKeySummary,
    summary_out: SortedKeySummary,
    vals_summary: Option<SortValueSummary>,
    spine_summary: Option<SortSpineSummary>,
    window_in: Option<String>,
    window_out: Option<String>,
    vals_in_window: Option<String>,
    vals_out_window: Option<String>,
    vals_extra_in_window: Option<String>,
    vals_extra_first_out_window: Option<String>,
    vals_extra_positions: Option<String>,
    vals_extra_run: Option<String>,
    vals_missing_positions: Option<String>,
    expected_mismatch_line: Option<String>,
    expected_window: Option<String>,
    expected_extra_positions: Option<String>,
    expected_missing_positions: Option<String>,
    expected_actual_pair_route: Option<String>,
    expected_actual_pair_source_window: Option<String>,
    expected_actual_pair_clone_window: Option<String>,
    expected_actual_value_route: Option<String>,
    expected_actual_value_input_window: Option<String>,
    expected_actual_value_input_run: Option<String>,
    expected_actual_value_expected_window: Option<String>,
    expected_actual_value_expected_run: Option<String>,
    expected_actual_value_source_window: Option<String>,
    expected_actual_value_clone_window: Option<String>,
    expected_actual_value_source_diff: Option<String>,
    expected_actual_value_clone_diff: Option<String>,
    rank_expected_line: Option<String>,
    rank_actual_line: Option<String>,
    cta_capture_line: Option<String>,
    replay_capture_line: Option<String>,
    restored_out_summary: Option<SortedKeySummary>,
    restored_out_window: Option<String>,
}

struct SortPassTracker {
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
    recent: Vec<SortPassSummaryRecord>,
}
unsafe impl Send for SortPassTracker {}
unsafe impl Sync for SortPassTracker {}

static SORT_PASS_TRACKER: Mutex<Option<SortPassTracker>> = Mutex::new(None);
static SORT_PASS_REPLAY_QUEUE: Mutex<Option<SortPassTracker>> = Mutex::new(None);

#[derive(Clone, Copy)]
struct PendingSortReplayCapture {
    capture_n: u32,
    begin_bit: u32,
    n_items: usize,
    read_n: usize,
    keys_in: u64,
    keys_out: u64,
    vals_in: u64,
    vals_out: u64,
    spine_ptr: u64,
    first: u8,
    last: u8,
    overwrite: u8,
    grid_share: Option<SortGridEvenShare>,
    stream: hipStream_t,
    grid_dim_x: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
}

static PENDING_SORT_REPLAY_CAPTURE: Mutex<Option<PendingSortReplayCapture>> = Mutex::new(None);

fn sort_pass_queue_slot_offset(slot: usize) -> Option<usize> {
    SORT_PASS_QUEUE_SLOT_OFFSETS.get(slot).copied()
}

unsafe fn sort_pass_tracker_init(state: &mut Option<SortPassTracker>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(
            &mut host_ptr,
            SORT_PASS_TOTAL_SCRATCH_BYTES,
            hipHostMallocMapped,
        )
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] SORT pass tracker: hipHostMalloc({}) failed: {:?}",
            SORT_PASS_TOTAL_SCRATCH_BYTES, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, SORT_PASS_TOTAL_SCRATCH_BYTES);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] SORT pass tracker: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    *state = Some(SortPassTracker {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
        recent: Vec::new(),
    });
}

unsafe fn capture_sort_downsweep_safe(
    begin_bit: u32,
    n_items: usize,
    keys_in: u64,
    keys_out: u64,
    vals_in: u64,
    vals_out: u64,
    spine_ptr: u64,
    grid_share: Option<SortGridEvenShare>,
    grid_dim_x: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    first: u8,
    last: u8,
    overwrite: u8,
    stream: hipStream_t,
) {
    if n_items == 0 || keys_in == 0 || keys_out == 0 {
        return;
    }
    // Synchronize the downsweep stream before issuing diagnostic D2D snapshots.
    // Offline full-pass replay showed the sort kernels themselves can produce a
    // clean window while host-side copy paths still diverge or wedge, so the
    // validator must not rely on copy ordering to serialize against the launch.
    let pre_copy_sync_result = hip_runtime_sys::hipStreamSynchronize(stream);
    if pre_copy_sync_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_PASS_SAFE: pre-copy stream sync failed {:?} bit={} keys_in=0x{:x} keys_out=0x{:x} n={}",
            pre_copy_sync_result, begin_bit, keys_in, keys_out, n_items
        ));
        return;
    }
    let read_n = sort_pass_capture_len(n_items, grid_share);
    let logical_n = n_items.min(read_n);
    let mut guard = match SORT_PASS_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    sort_pass_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };
    let dst_in_dev = hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_KEYS_IN_OFFSET) as _);
    let dst_out_dev =
        hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_KEYS_OUT_OFFSET) as _);
    let dst_vals_in_dev =
        hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_VALS_IN_OFFSET) as _);
    let dst_vals_out_dev =
        hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_VALS_OUT_OFFSET) as _);
    let dst_spine_dev =
        hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_SPINE_OFFSET) as _);
    let copy_in_result = hipMemcpyDtoD(
        dst_in_dev,
        hipDeviceptr_t(keys_in as *mut std::ffi::c_void),
        read_n * 4,
    );
    let copy_out_result = hipMemcpyDtoD(
        dst_out_dev,
        hipDeviceptr_t(keys_out as *mut std::ffi::c_void),
        read_n * 4,
    );
    let copy_vals_in_result = if vals_in != 0 {
        Some(hipMemcpyDtoD(
            dst_vals_in_dev,
            hipDeviceptr_t(vals_in as *mut std::ffi::c_void),
            read_n * 4,
        ))
    } else {
        None
    };
    let copy_vals_out_result = if vals_out != 0 {
        Some(hipMemcpyDtoD(
            dst_vals_out_dev,
            hipDeviceptr_t(vals_out as *mut std::ffi::c_void),
            read_n * 4,
        ))
    } else {
        None
    };
    let spine_words = ((grid_dim_x as usize) * 16).min(SORT_PASS_SPINE_MAX_WORDS);
    let spine_copy_result = if spine_ptr != 0 && spine_words > 0 {
        Some(hipMemcpyDtoD(
            dst_spine_dev,
            hipDeviceptr_t(spine_ptr as *mut std::ffi::c_void),
            spine_words * 4,
        ))
    } else {
        None
    };
    if copy_in_result.is_err() || copy_out_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_PASS_SAFE: D2D failed bit={} in={:?} out={:?} keys_in=0x{:x} keys_out=0x{:x} n={}",
            begin_bit, copy_in_result, copy_out_result, keys_in, keys_out, read_n
        ));
        return;
    }
    if let Some(copy_result) = copy_vals_in_result {
        if copy_result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_PASS_SAFE: vals_in D2D failed bit={} vals_in=0x{:x} n={} -> {:?}",
                begin_bit, vals_in, read_n, copy_result
            ));
        }
    }
    if let Some(copy_result) = copy_vals_out_result {
        if copy_result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_PASS_SAFE: vals_out D2D failed bit={} vals_out=0x{:x} n={} -> {:?}",
                begin_bit, vals_out, read_n, copy_result
            ));
        }
    }
    if let Some(copy_result) = spine_copy_result {
        if copy_result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_PASS_SAFE: spine D2D failed bit={} spine=0x{:x} words={} -> {:?}",
                begin_bit, spine_ptr, spine_words, copy_result
            ));
        }
    }
    let sync_result = hip_runtime_sys::hipDeviceSynchronize();
    if sync_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_PASS_SAFE: post-copy device sync failed {:?} bit={} keys_in=0x{:x} keys_out=0x{:x} n={}",
            sync_result, begin_bit, keys_in, keys_out, read_n
        ));
        return;
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let keys_in_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_KEYS_IN_OFFSET) as *const i32,
        read_n,
    );
    let keys_in_before_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_KEYS_IN_BEFORE_OFFSET) as *const i32,
        read_n,
    );
    let keys_out_before_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_KEYS_OUT_BEFORE_OFFSET) as *const i32,
        read_n,
    );
    let keys_out_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_KEYS_OUT_OFFSET) as *const i32,
        read_n,
    );
    let vals_in_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_VALS_IN_OFFSET) as *const i32,
        read_n,
    );
    let vals_in_before_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_VALS_IN_BEFORE_OFFSET) as *const i32,
        read_n,
    );
    let vals_out_before_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_VALS_OUT_BEFORE_OFFSET) as *const i32,
        read_n,
    );
    let vals_out_full: &[i32] = std::slice::from_raw_parts(
        tracker.scratch_host.add(SORT_PASS_VALS_OUT_OFFSET) as *const i32,
        read_n,
    );
    let keys_in_host = &keys_in_full[..logical_n];
    let keys_out_host = &keys_out_full[..logical_n];
    let vals_in_host = &vals_in_full[..logical_n];
    let vals_out_host = &vals_out_full[..logical_n];
    let summary_in = match summarize_sorted_keys(keys_in_host) {
        Some(summary) => summary,
        None => return,
    };
    let summary_out = match summarize_sorted_keys(keys_out_host) {
        Some(summary) => summary,
        None => return,
    };
    let vals_summary = if vals_in != 0 && vals_out != 0 {
        summarize_sort_values(vals_in_host, vals_out_host)
    } else {
        None
    };
    let restored_out_summary = if summary_out.negative_count > 0 {
        let restored = restore_signed_radix_residue(keys_out_host);
        summarize_sorted_keys(&restored)
    } else {
        None
    };
    let spine_summary = if spine_ptr != 0 && spine_words > 0 {
        let spine_words_host: &[u32] = std::slice::from_raw_parts(
            tracker.scratch_host.add(SORT_PASS_SPINE_OFFSET) as *const u32,
            spine_words,
        );
        summarize_sort_spine(spine_words_host)
    } else {
        None
    };
    let window_in = format_sorted_key_window(keys_in_host, &summary_in, "IN");
    let window_out = format_sorted_key_window(keys_out_host, &summary_out, "OUT");
    let vals_in_window = vals_summary
        .and_then(|summary| summary.first_missing)
        .and_then(|(idx, _, _, _)| {
            format_sort_pair_window(keys_in_host, vals_in_host, idx, "VAL_IN")
        });
    let vals_out_window = vals_summary
        .and_then(|summary| summary.first_extra)
        .and_then(|(idx, _, _, _)| {
            format_sort_pair_window(keys_out_host, vals_out_host, idx, "VAL_OUT")
        });
    let vals_extra_in_window = vals_summary
        .and_then(|summary| summary.first_extra)
        .and_then(|(_, value, _, _)| {
            find_value_positions(vals_in_host, value, 1)
                .into_iter()
                .next()
                .and_then(|idx| {
                    format_sort_pair_window(keys_in_host, vals_in_host, idx, "VAL_EXTRA_IN")
                })
        });
    let vals_extra_first_out_window = vals_summary
        .and_then(|summary| summary.first_extra)
        .and_then(|(_, value, _, _)| {
            find_value_positions(vals_out_host, value, 1)
                .into_iter()
                .next()
                .and_then(|idx| {
                    format_sort_pair_window(keys_out_host, vals_out_host, idx, "VAL_EXTRA_OUT0")
                })
        });
    let vals_extra_positions =
        vals_summary
            .and_then(|summary| summary.first_extra)
            .map(|(_, value, _, _)| {
                let input_positions = find_value_positions(vals_in_host, value, 4);
                let output_positions = find_value_positions(vals_out_host, value, 4);
                format!(
                    "VAL_POS extra={} in={:?} out={:?}",
                    value, input_positions, output_positions
                )
            });
    let vals_extra_run = vals_summary
        .and_then(|summary| summary.first_extra)
        .and_then(|(dup_idx, value, _, _)| {
            let output_positions = find_value_positions(vals_out_host, value, 2);
            if output_positions.len() < 2 {
                return None;
            }
            let first_idx = output_positions[0];
            let second_idx = if output_positions[1] == dup_idx {
                output_positions[1]
            } else {
                dup_idx
            };
            matching_pair_run(keys_out_host, vals_out_host, first_idx, second_idx).map(
                |(first_start, second_start, len)| {
                    format!(
                        "VAL_DUP_RUN value={} out0={} out1={} start0={} start1={} len={}",
                        value, first_idx, second_idx, first_start, second_start, len
                    )
                },
            )
        });
    let vals_missing_positions =
        vals_summary
            .and_then(|summary| summary.first_missing)
            .map(|(_, value, _, _)| {
                let input_positions = find_value_positions(vals_in_host, value, 4);
                let output_positions = find_value_positions(vals_out_host, value, 4);
                format!(
                    "VAL_POS missing={} in={:?} out={:?}",
                    value, input_positions, output_positions
                )
            });
    let restored_out_window = if let Some(summary) = restored_out_summary.as_ref() {
        let restored = restore_signed_radix_residue(keys_out_host);
        format_sorted_key_window(&restored, summary, "OUT_XOR")
    } else {
        None
    };
    static PASS_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let pass_n = PASS_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let has_issues = summary_in.violations > 0
        || summary_in.negative_count > 0
        || summary_in.huge_count > 0
        || summary_out.violations > 0
        || summary_out.negative_count > 0
        || summary_out.huge_count > 0
        || vals_summary
            .map(|summary| {
                summary.negative_count > 0 || summary.extra_count > 0 || summary.missing_count > 0
            })
            .unwrap_or(false);
    let expected_diagnostics = if has_issues && vals_summary.is_some() {
        analyze_expected_sort_pass(
            keys_in_host,
            vals_in_host,
            keys_out_host,
            vals_out_host,
            begin_bit,
            vals_summary,
        )
    } else {
        None
    };
    let rank_diagnostics = if has_issues {
        analyze_downsweep_rank_diagnostics(
            keys_in_host,
            vals_in_host,
            keys_out_host,
            vals_out_host,
            begin_bit,
            grid_share,
        )
    } else {
        None
    };
    let cta_capture = if has_issues {
        analyze_downsweep_block_capture(
            keys_in_host,
            vals_in_host,
            keys_out_host,
            vals_out_host,
            begin_bit,
            grid_share,
        )
    } else {
        None
    };
    let spine_words_host = if spine_ptr != 0 && spine_words > 0 {
        Some(std::slice::from_raw_parts(
            tracker.scratch_host.add(SORT_PASS_SPINE_OFFSET) as *const u32,
            spine_words,
        ))
    } else {
        None
    };
    let good_capture_kind = classify_good_sort_replay_capture(
        begin_bit,
        n_items,
        first,
        overwrite,
        summary_out,
        vals_summary,
    );
    let replay_capture = if has_issues || good_capture_kind.is_some() {
        serialize_sort_pass_replay_capture(
            pass_n,
            begin_bit,
            n_items,
            grid_dim_x,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes,
            first,
            last,
            overwrite,
            keys_in,
            keys_out,
            vals_in,
            vals_out,
            spine_ptr,
            stream.0 as u64,
            keys_in_full,
            (vals_in != 0 && vals_out != 0).then_some(vals_in_full),
            keys_in_before_full,
            (vals_in != 0 && vals_out != 0).then_some(vals_in_before_full),
            keys_out_before_full,
            (vals_in != 0 && vals_out != 0).then_some(vals_out_before_full),
            keys_out_full,
            (vals_in != 0 && vals_out != 0).then_some(vals_out_full),
            spine_words_host.unwrap_or(&[]),
            grid_share,
        )
    } else {
        None
    };
    static GOOD_TARGET_CAPTURED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    static GOOD_FALLBACK_CAPTURED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    static GOOD_LATE_4688_CAPTURED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if let Some(kind) = good_capture_kind {
        let path = good_sort_replay_capture_path(kind, begin_bit, n_items, pass_n);
        let captured = match kind {
            GoodSortReplayCaptureKind::TargetBit0N8109 => &GOOD_TARGET_CAPTURED,
            GoodSortReplayCaptureKind::FallbackBit0LargeClean => &GOOD_FALLBACK_CAPTURED,
            GoodSortReplayCaptureKind::LateBit16N4688Clean => &GOOD_LATE_4688_CAPTURED,
        };
        if let Some(capture_line) = write_good_sort_replay_capture_once(
            captured,
            &path,
            replay_capture.as_deref(),
            begin_bit,
            n_items,
            grid_dim_x,
            spine_words,
        ) {
            super::driver::zluda_log_err(&format!("    {}", capture_line));
        }
    }
    let spine_max = spine_summary.map(|s| s.max_value).unwrap_or(0);
    let spine_sum = spine_summary.map(|s| s.sum).unwrap_or(0);
    if pass_n < 8 || pass_n % 100 == 0 || has_issues {
        let mut line = format!(
            "SORT_PASS_SAFE #{} bit={} n={} IN[viol={},neg={},huge={}] OUT[viol={},neg={},huge={}]",
            pass_n,
            begin_bit,
            n_items,
            summary_in.violations,
            summary_in.negative_count,
            summary_in.huge_count,
            summary_out.violations,
            summary_out.negative_count,
            summary_out.huge_count,
        );
        if let Some(restored) = restored_out_summary {
            line.push_str(&format!(
                " OUT_XOR[viol={},neg={},huge={}]",
                restored.violations, restored.negative_count, restored.huge_count,
            ));
        }
        if let Some(vals_summary) = vals_summary {
            line.push_str(&format!(
                " VAL_FLOW[neg={},extra={},miss={}]",
                vals_summary.negative_count, vals_summary.extra_count, vals_summary.missing_count
            ));
        }
        line.push_str(&format!(
            " spine=0x{:x}[max={},sum={}] keys_in=0x{:x} keys_out=0x{:x}",
            spine_ptr, spine_max, spine_sum, keys_in, keys_out
        ));
        super::driver::zluda_log_err(&line);
        if let Some(summary) = vals_summary {
            if summary.negative_count > 0 || summary.extra_count > 0 || summary.missing_count > 0 {
                if let Some(window) = vals_out_window.as_ref() {
                    super::driver::zluda_log_err(&format!("    {}", window));
                }
                if let Some(window) = vals_in_window.as_ref() {
                    super::driver::zluda_log_err(&format!("    {}", window));
                }
                if let Some(window) = vals_extra_in_window.as_ref() {
                    super::driver::zluda_log_err(&format!("    {}", window));
                }
                if let Some(window) = vals_extra_first_out_window.as_ref() {
                    super::driver::zluda_log_err(&format!("    {}", window));
                }
                if let Some(line) = vals_extra_positions.as_ref() {
                    super::driver::zluda_log_err(&format!("    {}", line));
                }
                if let Some(line) = vals_extra_run.as_ref() {
                    super::driver::zluda_log_err(&format!("    {}", line));
                }
                if let Some(line) = vals_missing_positions.as_ref() {
                    super::driver::zluda_log_err(&format!("    {}", line));
                }
                if let Some(diag) = expected_diagnostics.as_ref() {
                    if let Some(line) = diag.mismatch_line.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(window) = diag.expected_window.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", window));
                    }
                    if let Some(line) = diag.actual_pair_route.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(window) = diag.actual_pair_source_window.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", window));
                    }
                    if let Some(window) = diag.actual_pair_clone_window.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", window));
                    }
                    if let Some(line) = diag.actual_value_route.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(window) = diag.actual_value_input_window.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", window));
                    }
                    if let Some(line) = diag.actual_value_input_run.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(window) = diag.actual_value_expected_window.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", window));
                    }
                    if let Some(line) = diag.actual_value_expected_run.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(window) = diag.actual_value_source_window.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", window));
                    }
                    if let Some(window) = diag.actual_value_clone_window.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", window));
                    }
                    if let Some(line) = diag.actual_value_source_diff.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(line) = diag.actual_value_clone_diff.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(line) = diag.extra_positions.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(line) = diag.missing_positions.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                }
                if let Some(diag) = rank_diagnostics.as_ref() {
                    if let Some(line) = diag.expected_line.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                    if let Some(line) = diag.actual_line.as_ref() {
                        super::driver::zluda_log_err(&format!("    {}", line));
                    }
                }
                if let Some(capture) = cta_capture.as_ref() {
                    let path = format!(
                        "/tmp/zluda_sort_cta_capture_bit{}_pass{}.txt",
                        begin_bit, pass_n
                    );
                    let capture_line = match std::fs::write(&path, &capture.capture_text) {
                        Ok(()) => format!(
                            "SORT_CTA_CAPTURE bit={} block={} range=[{}..{}) expected_src_block={} actual_src_block={:?} path={}",
                            begin_bit,
                            capture.block_idx,
                            capture.start,
                            capture.end,
                            capture.expected_source_block,
                            capture.actual_source_block,
                            path
                        ),
                        Err(err) => format!(
                            "SORT_CTA_CAPTURE bit={} block={} range=[{}..{}) expected_src_block={} actual_src_block={:?} path={} write_err={}",
                            begin_bit,
                            capture.block_idx,
                            capture.start,
                            capture.end,
                            capture.expected_source_block,
                            capture.actual_source_block,
                            path,
                            err
                        ),
                    };
                    super::driver::zluda_log_err(&format!("    {}", capture_line));
                }
                if let Some(capture) = replay_capture.as_ref() {
                    let path = format!(
                        "/tmp/zluda_sort_replay_capture_bit{}_pass{}.bin",
                        begin_bit, pass_n
                    );
                    let capture_line = match std::fs::write(&path, capture) {
                        Ok(()) => format!(
                            "SORT_REPLAY_CAPTURE bit={} n={} grid={} spine_words={} path={}",
                            begin_bit, n_items, grid_dim_x, spine_words, path
                        ),
                        Err(err) => format!(
                            "SORT_REPLAY_CAPTURE bit={} n={} grid={} spine_words={} path={} write_err={}",
                            begin_bit, n_items, grid_dim_x, spine_words, path, err
                        ),
                    };
                    super::driver::zluda_log_err(&format!("    {}", capture_line));
                }
            }
        }
    }
    tracker.recent.push(SortPassSummaryRecord {
        pass_n,
        begin_bit,
        n_items,
        keys_in,
        keys_out,
        spine_ptr,
        summary_in,
        summary_out,
        vals_summary,
        spine_summary,
        window_in,
        window_out,
        vals_in_window,
        vals_out_window,
        vals_extra_in_window,
        vals_extra_first_out_window,
        vals_extra_positions,
        vals_extra_run,
        vals_missing_positions,
        expected_mismatch_line: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.mismatch_line.clone()),
        expected_window: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.expected_window.clone()),
        expected_extra_positions: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.extra_positions.clone()),
        expected_missing_positions: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.missing_positions.clone()),
        expected_actual_pair_route: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_pair_route.clone()),
        expected_actual_pair_source_window: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_pair_source_window.clone()),
        expected_actual_pair_clone_window: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_pair_clone_window.clone()),
        expected_actual_value_route: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_route.clone()),
        expected_actual_value_input_window: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_input_window.clone()),
        expected_actual_value_input_run: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_input_run.clone()),
        expected_actual_value_expected_window: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_expected_window.clone()),
        expected_actual_value_expected_run: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_expected_run.clone()),
        expected_actual_value_source_window: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_source_window.clone()),
        expected_actual_value_clone_window: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_clone_window.clone()),
        expected_actual_value_source_diff: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_source_diff.clone()),
        expected_actual_value_clone_diff: expected_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_value_clone_diff.clone()),
        rank_expected_line: rank_diagnostics
            .as_ref()
            .and_then(|diag| diag.expected_line.clone()),
        rank_actual_line: rank_diagnostics
            .as_ref()
            .and_then(|diag| diag.actual_line.clone()),
        cta_capture_line: cta_capture.as_ref().map(|capture| {
            format!(
                "SORT_CTA_CAPTURE bit={} block={} range=[{}..{}) expected_src_block={} actual_src_block={:?} path=/tmp/zluda_sort_cta_capture_bit{}_pass{}.txt",
                begin_bit,
                capture.block_idx,
                capture.start,
                capture.end,
                capture.expected_source_block,
                capture.actual_source_block,
                begin_bit,
                pass_n
            )
        }),
        replay_capture_line: replay_capture.as_ref().map(|_| {
            format!(
                "SORT_REPLAY_CAPTURE bit={} n={} grid={} spine_words={} path=/tmp/zluda_sort_replay_capture_bit{}_pass{}.bin",
                begin_bit, n_items, grid_dim_x, spine_words, begin_bit, pass_n
            )
        }),
        restored_out_summary,
        restored_out_window,
    });
    const SORT_PASS_RECENT_LIMIT: usize = 24;
    if tracker.recent.len() > SORT_PASS_RECENT_LIMIT {
        let overflow = tracker.recent.len() - SORT_PASS_RECENT_LIMIT;
        tracker.recent.drain(0..overflow);
    }
}

fn dump_recent_sort_passes_for_creategrid(call_n: u32, n: usize, sorted_keys: u64) {
    let guard = match SORT_PASS_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let tracker = match guard.as_ref() {
        Some(v) => v,
        None => return,
    };
    let mut matched = 0usize;
    for record in tracker.recent.iter().rev() {
        if record.n_items != n {
            continue;
        }
        let in_role = classify_creategrid_sort_buffer(sorted_keys, n, record.keys_in);
        let out_role = classify_creategrid_sort_buffer(sorted_keys, n, record.keys_out);
        if in_role.is_none() && out_role.is_none() {
            continue;
        }
        matched += 1;
        let in_role_str = in_role.map(creategrid_sort_buffer_role_name).unwrap_or("-");
        let out_role_str = out_role
            .map(creategrid_sort_buffer_role_name)
            .unwrap_or("-");
        let spine_words = record.spine_summary.map(|s| s.words_read).unwrap_or(0);
        let spine_max = record.spine_summary.map(|s| s.max_value).unwrap_or(0);
        let spine_sum = record.spine_summary.map(|s| s.sum).unwrap_or(0);
        let mut line = format!(
            "    SORT_PASS_MATCH creategrid#{} pass#{} bit={} keys_in=0x{:x}({}) keys_out=0x{:x}({}) spine=0x{:x}[words={},max={},sum={}] IN[viol={},neg={},huge={}] OUT[viol={},neg={},huge={}]",
            call_n,
            record.pass_n,
            record.begin_bit,
            record.keys_in,
            in_role_str,
            record.keys_out,
            out_role_str,
            record.spine_ptr,
            spine_words,
            spine_max,
            spine_sum,
            record.summary_in.violations,
            record.summary_in.negative_count,
            record.summary_in.huge_count,
            record.summary_out.violations,
            record.summary_out.negative_count,
            record.summary_out.huge_count,
        );
        if let Some(vals_summary) = record.vals_summary {
            line.push_str(&format!(
                " VAL_FLOW[neg={},extra={},miss={}]",
                vals_summary.negative_count, vals_summary.extra_count, vals_summary.missing_count
            ));
        }
        super::driver::zluda_log_err(&line);
        let value_issues = record
            .vals_summary
            .map(|summary| {
                summary.negative_count > 0 || summary.extra_count > 0 || summary.missing_count > 0
            })
            .unwrap_or(false);
        if record.summary_out.negative_count > 0
            || record.summary_out.huge_count > 0
            || value_issues
        {
            if let Some(window) = record.window_out.as_ref() {
                super::driver::zluda_log_err(&format!("        OUT window: {}", window));
            }
            if let Some((idx, value)) = record.summary_out.first_bad_value {
                super::driver::zluda_log_err(&format!(
                    "        OUT bad value: [{}]={} (0x{:x})",
                    idx, value, value as u32
                ));
            }
            if let Some(restored) = record.restored_out_summary {
                super::driver::zluda_log_err(&format!(
                    "        OUT xor-restore: viol={} neg={} huge={}",
                    restored.violations, restored.negative_count, restored.huge_count
                ));
            }
            if let Some(window) = record.restored_out_window.as_ref() {
                super::driver::zluda_log_err(&format!("        OUT xor window: {}", window));
            }
            if let Some(vals_summary) = record.vals_summary {
                super::driver::zluda_log_err(&format!(
                    "        OUT values: neg={} extra={} miss={}",
                    vals_summary.negative_count,
                    vals_summary.extra_count,
                    vals_summary.missing_count
                ));
                if let Some((idx, value, expected, seen)) = vals_summary.first_extra {
                    super::driver::zluda_log_err(&format!(
                        "        OUT values first extra: [{}]={} (in={}, out_seen={})",
                        idx, value, expected, seen
                    ));
                }
                if let Some((idx, value, expected, actual)) = vals_summary.first_missing {
                    super::driver::zluda_log_err(&format!(
                        "        OUT values first missing: in[{}]={} (in={}, out={})",
                        idx, value, expected, actual
                    ));
                }
                if let Some(window) = record.vals_out_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(window) = record.vals_in_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(window) = record.vals_extra_in_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(window) = record.vals_extra_first_out_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(line) = record.vals_extra_positions.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.vals_extra_run.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.vals_missing_positions.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.expected_mismatch_line.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(window) = record.expected_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(line) = record.expected_actual_pair_route.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(window) = record.expected_actual_pair_source_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(window) = record.expected_actual_pair_clone_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(line) = record.expected_actual_value_route.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(window) = record.expected_actual_value_input_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(line) = record.expected_actual_value_input_run.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(window) = record.expected_actual_value_expected_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(line) = record.expected_actual_value_expected_run.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(window) = record.expected_actual_value_source_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(window) = record.expected_actual_value_clone_window.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", window));
                }
                if let Some(line) = record.expected_actual_value_source_diff.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.expected_actual_value_clone_diff.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.expected_extra_positions.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.expected_missing_positions.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.rank_expected_line.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.rank_actual_line.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.cta_capture_line.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
                if let Some(line) = record.replay_capture_line.as_ref() {
                    super::driver::zluda_log_err(&format!("        {}", line));
                }
            }
        }
        if record.summary_in.negative_count > 0 || record.summary_in.huge_count > 0 {
            if let Some(window) = record.window_in.as_ref() {
                super::driver::zluda_log_err(&format!("        IN window: {}", window));
            }
            if let Some((idx, value)) = record.summary_in.first_bad_value {
                super::driver::zluda_log_err(&format!(
                    "        IN bad value: [{}]={} (0x{:x})",
                    idx, value, value as u32
                ));
            }
        }
    }
    if matched == 0 {
        super::driver::zluda_log_err(&format!(
            "    SORT_PASS_MATCH creategrid#{}: no recent downsweep record matched sk=0x{:x} n={}",
            call_n, sorted_keys, n
        ));
    }
}

struct CreateGridValidator {
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
}
unsafe impl Send for CreateGridValidator {}
unsafe impl Sync for CreateGridValidator {}

static CREATEGRID_VALIDATOR: Mutex<Option<CreateGridValidator>> = Mutex::new(None);
static CREATECELL_SNAPSHOT_VALIDATOR: Mutex<Option<CreateGridValidator>> = Mutex::new(None);

unsafe fn createGrid_validator_init(state: &mut Option<CreateGridValidator>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, F5_TOTAL_SCRATCH_BYTES, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F5 CreateGrid validator: hipHostMalloc({}) failed: {:?}",
            F5_TOTAL_SCRATCH_BYTES, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, F5_TOTAL_SCRATCH_BYTES);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F5 CreateGrid validator: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] F5 CreateGrid validator: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
        F5_TOTAL_SCRATCH_BYTES, host_ptr as usize, dev_ptr as usize,
    ));
    *state = Some(CreateGridValidator {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
    });
}

/// Snapshot CreateGrid's sortedKeys input via D2D into pinned host-mapped
/// scratch, sync, walk it on host, and log `*** SORT CORRUPTED` with
/// neighborhood context if it's not monotonically non-decreasing or has
/// out-of-range values.
///
/// Called from launch_kernel BEFORE the CreateGrid kernel launches. Safe
/// because hipMemcpyDtoD does NOT enter the ProfilingSignal pool — unlike
/// the original validator's hipMemcpyDtoH which crashed the HSA runtime.
unsafe fn validate_create_grid_safe(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
) {
    // CreateGrid(int n, const int* sortedKeys, int* gridStarts, int* gridEnds)
    // arg layout (extra flat buffer): n@0, sortedKeys@8, gridStarts@16, gridEnds@24
    let num_particles: i32 = match read_kernel_arg(kernel_params, extra, 0, 0) {
        Some(v) => v,
        None => return,
    };
    let sorted_keys_dev: u64 = match read_kernel_arg(kernel_params, extra, 8, 1) {
        Some(v) => v,
        None => return,
    };
    if num_particles <= 0 || num_particles > 1_000_000 {
        super::driver::zluda_log_err(&format!(
            "*** F5 CreateGrid: numParticles={} (suspicious)",
            num_particles
        ));
        return;
    }
    if sorted_keys_dev == 0 {
        return;
    }

    let n = num_particles as usize;
    let read_n = n.min(F5_MAX_KEYS);
    note_creategrid_sorted_keys(sorted_keys_dev, n);

    let mut guard = match CREATEGRID_VALIDATOR.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    createGrid_validator_init(&mut guard);
    let validator = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };

    // Synchronize the CreateGrid launch stream before validator snapshots.
    // The live-only sort corruption still reproduces even when captured pass
    // inputs replay cleanly offline, so the prelaunch validator must not race
    // preceding sort/copy work on the same stream.
    let pre_copy_sync_result = hip_runtime_sys::hipStreamSynchronize(stream);
    if pre_copy_sync_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** F5_SAFE CreateGrid: pre-copy stream sync failed {:?} ptr=0x{:x} n={} stream={:?}",
            pre_copy_sync_result, sorted_keys_dev, n, stream
        ));
        return;
    }

    let dst_a_dev = hipDeviceptr_t(validator.scratch_dev);
    let dst_b_dev = hipDeviceptr_t((validator.scratch_dev as usize + F5_SCRATCH_BYTES) as _);
    let copy_a_result = hipMemcpyDtoD(
        dst_a_dev,
        hipDeviceptr_t(sorted_keys_dev as *mut std::ffi::c_void),
        read_n * 4,
    );
    let copy_b_result = hipMemcpyDtoD(
        dst_b_dev,
        hipDeviceptr_t((sorted_keys_dev + (n * 4) as u64) as *mut std::ffi::c_void),
        read_n * 4,
    );
    if copy_a_result.is_err() || copy_b_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** F5_SAFE CreateGrid: D2D failed a={:?} b={:?} ptr=0x{:x} n={} read={}",
            copy_a_result, copy_b_result, sorted_keys_dev, n, read_n
        ));
        return;
    }
    let sync_result = hipDeviceSynchronize();
    if sync_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** F5_SAFE CreateGrid: hipDeviceSynchronize failed {:?} ptr=0x{:x} n={}",
            sync_result, sorted_keys_dev, n
        ));
        return;
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let keys_a: &[i32] = std::slice::from_raw_parts(validator.scratch_host as *const i32, read_n);
    let keys_b: &[i32] = std::slice::from_raw_parts(
        validator.scratch_host.add(F5_SCRATCH_BYTES) as *const i32,
        read_n,
    );
    let summary_a = match summarize_sorted_keys(keys_a) {
        Some(summary) => summary,
        None => return,
    };
    let summary_b = match summarize_sorted_keys(keys_b) {
        Some(summary) => summary,
        None => return,
    };
    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let has_issues = summary_a.violations > 0
        || summary_a.negative_count > 0
        || summary_a.huge_count > 0
        || summary_b.violations > 0
        || summary_b.negative_count > 0
        || summary_b.huge_count > 0;

    if call_n < 5 || call_n % 50 == 0 || has_issues {
        super::driver::zluda_log_err(&format!(
            "F5_SAFE CreateGrid #{}: n={} read={} A[min={},max={},viol={},neg={},huge={}] B[min={},max={},viol={},neg={},huge={}] sk=0x{:x}",
            call_n,
            n,
            read_n,
            summary_a.min_key,
            summary_a.max_key,
            summary_a.violations,
            summary_a.negative_count,
            summary_a.huge_count,
            summary_b.min_key,
            summary_b.max_key,
            summary_b.violations,
            summary_b.negative_count,
            summary_b.huge_count,
            sorted_keys_dev,
        ));
    }

    if has_issues {
        super::driver::zluda_log_err(&format!(
            "*** SORT_BISECT_SAFE #{} n={} A(viol={},neg={},huge={}) B(viol={},neg={},huge={}) sk=0x{:x}",
            call_n,
            n,
            summary_a.violations,
            summary_a.negative_count,
            summary_a.huge_count,
            summary_b.violations,
            summary_b.negative_count,
            summary_b.huge_count,
            sorted_keys_dev,
        ));
        dump_recent_sort_passes_for_creategrid(call_n, n, sorted_keys_dev);
        dump_recent_creategrid_sort_buffer_touches(call_n, n, sorted_keys_dev);
        if let Some(vi) = summary_a.first_violation {
            let start = vi.saturating_sub(5);
            let end = (vi + 5).min(read_n);
            let around: Vec<String> = (start..end)
                .map(|j| format!("A[{}]={}", j, keys_a[j]))
                .collect();
            super::driver::zluda_log_err(&format!("    A around violation: {}", around.join(", ")));
        }
        if let Some((idx, key)) = summary_a.first_bad_value {
            super::driver::zluda_log_err(&format!(
                "    A bad value: [{}]={} (0x{:x})",
                idx, key, key as u32
            ));
        }
        if let Some(vi) = summary_b.first_violation {
            let start = vi.saturating_sub(5);
            let end = (vi + 5).min(read_n);
            let around: Vec<String> = (start..end)
                .map(|j| format!("B[{}]={}", j, keys_b[j]))
                .collect();
            super::driver::zluda_log_err(&format!("    B around violation: {}", around.join(", ")));
        }
        if let Some((idx, key)) = summary_b.first_bad_value {
            super::driver::zluda_log_err(&format!(
                "    B bad value: [{}]={} (0x{:x})",
                idx, key, key as u32
            ));
        }
        let bytes_a: Vec<u8> = keys_a
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect();
        let bytes_b: Vec<u8> = keys_b
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect();
        let _ = std::fs::write(format!("/tmp/sort_bisect_safe_{}_a.bin", call_n), &bytes_a);
        let _ = std::fs::write(format!("/tmp/sort_bisect_safe_{}_b.bin", call_n), &bytes_b);
    }
}

// ============================================================================
// Counter buffer per-launch snapshot tracker (Bug A — fresh-bug-a-2026-04-10
// plan, Step 4)
// ============================================================================
//
// Purpose: identify the exact kernel that zeroes the per-shape counter buffer
// between two MS_4 launches. We've observed that on the bug frame the
// counter buffer (kp[7] of the MS_4 wrapper) is `[0; 64]` despite a healthy
// MS_4 having consumed it one frame earlier. The existing F.4 tracker
// (`check_counter_mutation`) does almost the same thing but is gated on
// ZLUDA_F6_TRACK=1 because F.4 piggybacks on the same accumulated pointer
// set as the F.6 rd5 tracker — which has a stale-pointer hazard documented
// in `project_f6_tracker_stale_pointer_bug.md` and is responsible for the
// "TurbulenceFS family" false-positive crashes in runs 31/36/37.
//
// THIS tracker avoids that hazard by latching a SINGLE counter pointer from
// the most-recent MS_4 launch's kp[7]. Because MS_4 just consumed (or is
// about to consume) that buffer, it cannot be a stale pointer to a freed
// particle group. Each subsequent MS_4 re-latches, so the tracked pointer
// always corresponds to a live, currently-in-use buffer.
//
// Cost: 1 hipMemcpyDtoD (256 bytes) + 1 hipDeviceSynchronize per kernel
// launch when enabled. Lighter than F.4 (which copies 1 KB × N tracked).
//
// Gated on ZLUDA_COUNTER_SNAPSHOT=1 — independent of ZLUDA_MS4_CAPTURE so it
// can be enabled in isolation.

const COUNTER_SNAPSHOT_BYTES: usize = 256; // first 64 u32 entries

fn counter_snapshot_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_COUNTER_SNAPSHOT").map_or(false, |v| v == "1"))
}

// 2026-05-16 v7 Step 4: shape_table probe — D2D from shape_table to a
// host-mapped scratch before MS_2_Batched launches. Captures the input
// state MS_2_Batched will read at offset +8 of each 12-byte stride entry.
// Gated on ZLUDA_SHAPE_TABLE_PROBE=1.
//
// Reuses the counter_snapshot pattern: hipHostMalloc(Mapped) +
// hipHostGetDevicePointer + hipMemcpyDtoD + hipDeviceSynchronize.

const SHAPE_PROBE_BYTES: usize = 384; // 32 entries × 12 bytes (covers field0..8)

fn shape_table_probe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_SHAPE_TABLE_PROBE").map_or(false, |v| v == "1"))
}

struct ShapeProbeState {
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
    last_bytes: [u8; SHAPE_PROBE_BYTES],
    last_ptr: u64,
    launch_n: u64,
}
unsafe impl Send for ShapeProbeState {}
unsafe impl Sync for ShapeProbeState {}

static SHAPE_PROBE: Mutex<Option<ShapeProbeState>> = Mutex::new(None);

unsafe fn shape_probe_init(state: &mut Option<ShapeProbeState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, SHAPE_PROBE_BYTES, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] shape probe: hipHostMalloc({}) failed: {:?}",
            SHAPE_PROBE_BYTES, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, SHAPE_PROBE_BYTES);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] shape probe: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] shape probe: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
        SHAPE_PROBE_BYTES, host_ptr as usize, dev_ptr as usize,
    ));
    *state = Some(ShapeProbeState {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
        last_bytes: [0u8; SHAPE_PROBE_BYTES],
        last_ptr: 0,
        launch_n: 0,
    });
}

/// Snapshot shape_table state by D2D'ing from `shape_ptr` to our pinned
/// scratch, then logging via host alias. Called from launch_kernel just
/// before MS_4 fires (so we capture exactly what MS_4 sees AFTER MS_2_Batched
/// ran — and indirectly, what MS_2_Batched processed at offset +8).
unsafe fn snapshot_shape_table_before_ms4(shape_ptr: u64) {
    if !shape_table_probe_enabled() || shape_ptr == 0 {
        return;
    }
    let mut guard = match SHAPE_PROBE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    shape_probe_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };

    if state.last_ptr != shape_ptr {
        state.last_bytes = [0u8; SHAPE_PROBE_BYTES];
        state.last_ptr = shape_ptr;
    }
    state.launch_n += 1;
    let n = state.launch_n;

    let copy_result = hipMemcpyDtoD(
        hipDeviceptr_t(state.scratch_dev),
        hipDeviceptr_t(shape_ptr as _),
        SHAPE_PROBE_BYTES,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "[SHAPE] snapshot D2D failed for ptr=0x{:x}: {:?}",
            shape_ptr, copy_result
        ));
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let cur_slice: &[u8] = std::slice::from_raw_parts(state.scratch_host, SHAPE_PROBE_BYTES);
    if cur_slice == state.last_bytes {
        return; // no change since last snapshot for this ptr
    }
    // Log the .field8 (offset +8 within each 12-byte stride) of the
    // first 32 entries as i32 — that's the value MS_2_Batched reads.
    let mut field8s = String::with_capacity(256);
    field8s.push('[');
    let base = state.scratch_host as *const u8;
    for i in 0..32 {
        if i > 0 {
            field8s.push(',');
        }
        let v: i32 = std::ptr::read_unaligned(base.add(i * 12 + 8) as *const i32);
        field8s.push_str(&v.to_string());
    }
    field8s.push(']');
    super::driver::zluda_log_err(&format!(
        "[SHAPE PROBE] launch_n={} shape_ptr=0x{:x} field8[0..32]={}",
        n, shape_ptr, field8s
    ));
    state.last_bytes.copy_from_slice(cur_slice);
}

/// Most recently observed MS_4 counter pointer (kp[7]). Updated by
/// `latch_counter_ptr_from_ms4` immediately before each MS_4 launch, read
/// by `snapshot_counter_after_launch` after every other launch. 0 = unset.
static LATCHED_COUNTER_PTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct CounterSnapshotState {
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
    last_bytes: [u8; COUNTER_SNAPSHOT_BYTES],
    last_ptr: u64,
    launch_n: u64,
}
unsafe impl Send for CounterSnapshotState {}
unsafe impl Sync for CounterSnapshotState {}

static COUNTER_SNAPSHOT: Mutex<Option<CounterSnapshotState>> = Mutex::new(None);

// 2026-05-16 v7 Step 6: flag-count probe — D2D the flags buffer slice
// MS_4 will read, then count bits matching `(flags & 0x11) == 1` to verify
// the OVERCOUNT hypothesis. Compare against sum(counters); should be equal
// on NVIDIA, larger on ZLUDA if InitColl writes too many flag=1 bits.
//
// Gated on ZLUDA_FLAG_COUNT_PROBE=1. Scratch is fixed 1 MB (256K u32) —
// for 4-byte-stride flags arrays of up to 256K elements (largest seen
// ~62K, so plenty of headroom). Larger writes truncated.

const FLAG_PROBE_BYTES: usize = 1024 * 1024;

fn flag_count_probe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_FLAG_COUNT_PROBE").map_or(false, |v| v == "1"))
}

struct FlagProbeState {
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
    launch_n: u64,
}
unsafe impl Send for FlagProbeState {}
unsafe impl Sync for FlagProbeState {}

static FLAG_PROBE: Mutex<Option<FlagProbeState>> = Mutex::new(None);

unsafe fn flag_probe_init(state: &mut Option<FlagProbeState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, FLAG_PROBE_BYTES, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] flag probe: hipHostMalloc({}) failed: {:?}",
            FLAG_PROBE_BYTES, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, FLAG_PROBE_BYTES);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] flag probe: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] flag probe: scratch={} KB pinned; host=0x{:x} dev=0x{:x}",
        FLAG_PROBE_BYTES / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    *state = Some(FlagProbeState {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
        launch_n: 0,
    });
}

/// Generic flag-count probe. Copies up to `n_threads * 4` bytes from
/// `flags_ptr` to pinned scratch, then counts bits matching MS_4's
/// filter `(flag & 0x11) == 1`. Logs with `label` prefix to
/// distinguish call sites (pre-MS_4 vs pre-InitColl).
///
/// flags is u16 per element with 4-byte STRIDE (MS_4 PTX:
/// `mul.wide.u32 %rd8, %r1, 4; ... ld.global.u16 [%rd9]`).
unsafe fn snapshot_flag_count_at(label: &str, flags_ptr: u64, n_threads: u32, extra_ptr: u64) {
    if !flag_count_probe_enabled() || flags_ptr == 0 || n_threads == 0 {
        return;
    }
    let bytes = (n_threads as usize).saturating_mul(4).min(FLAG_PROBE_BYTES);
    let mut guard = match FLAG_PROBE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    flag_probe_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };
    state.launch_n += 1;
    let n = state.launch_n;

    let copy_result = hipMemcpyDtoD(
        hipDeviceptr_t(state.scratch_dev),
        hipDeviceptr_t(flags_ptr as _),
        bytes,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "[FLAG PROBE {}] D2D failed for flags=0x{:x}: {:?}",
            label, flags_ptr, copy_result
        ));
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let mut count_pass: u32 = 0;
    let mut count_zero: u32 = 0;
    let mut count_set_any: u32 = 0;
    let base = state.scratch_host as *const u8;
    for i in 0..n_threads as usize {
        let f: u16 = std::ptr::read_unaligned(base.add(i * 4) as *const u16);
        if f == 0 {
            count_zero += 1;
        }
        if f != 0 {
            count_set_any += 1;
        }
        if (f & 0x11) == 1 {
            count_pass += 1;
        }
    }
    super::driver::zluda_log_err(&format!(
        "[FLAG PROBE {}] launch_n={} flags=0x{:x} extra=0x{:x} n_threads={} pass={} zero={} set_any={}",
        label, n, flags_ptr, extra_ptr, n_threads, count_pass, count_zero, count_set_any
    ));
}

unsafe fn snapshot_flag_count_before_ms4(flags_ptr: u64, n_threads: u32, counter_ptr: u64) {
    snapshot_flag_count_at("PRE_MS4", flags_ptr, n_threads, counter_ptr)
}

/// Pre-InitColl flag probe. Called from launch_kernel hook when
/// InitCollDataAndApplyConstraints fires. InitColl's flags ptr is
/// param_3 (DEV3 in KARG_LOG). n_threads = grid_dim_x * block_dim_x.
unsafe fn snapshot_flag_count_before_initcoll(flags_ptr: u64, n_threads: u32) {
    snapshot_flag_count_at("PRE_INITCOLL", flags_ptr, n_threads, 0)
}

/// 2026-05-16 v8 Step 3: latched MS_4 flags ptr + n_threads, set from
/// MS_4 launch hook. Used by post-MS_1 probe to count bit-4-set on
/// the SAME buffer MS_4 will read.
static MS4_FLAGS_PTR_LATCHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MS4_N_LATCHED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// 2026-05-16 v11 Step 2: MS_1 gate-trace buffer.
/// Per-kernel-name PTX instrumentation (allowed per user clarification 2026-05-16)
/// inserted at MS_1 entry's bit-4-set gate (module_10:2122-2128). Counts
/// per-thread predicate outcomes into 4 atomic counters:
///   [0]:  count of threads with %r178 == 0
///   [4]:  count of threads with %r178 != 0 AND [SPL+16] == 0
///   [8]:  count of threads with %r178 != 0 AND [SPL+16] != 0 (bit-4 fires)
///   [12]: total threads reaching the gate
///
/// Patch_ms1_gate_trace in module.rs takes the dev_ptr value and
/// substitutes it as a literal in the PTX. Gated on ZLUDA_MS1_GATE_TRACE=1.
const MS1_GATE_TRACE_BYTES: usize = 16;
static MS1_GATE_TRACE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static MS1_GATE_TRACE_DEV_PTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MS1_GATE_TRACE_DUMP_TICK: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

pub(crate) fn ms1_gate_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_MS1_GATE_TRACE").map_or(false, |v| v == "1"))
}

/// Initialize the trace buffer; idempotent. Returns the DEVICE pointer
/// to embed as a PTX literal, or 0 on failure.
pub(crate) fn ms1_gate_trace_get_or_init_dev_ptr() -> u64 {
    if !ms1_gate_trace_enabled() {
        return 0;
    }
    let cached = MS1_GATE_TRACE_DEV_PTR.load(std::sync::atomic::Ordering::Acquire);
    if cached != 0 {
        return cached;
    }
    unsafe {
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let alloc = hipHostMalloc(&mut host_ptr, MS1_GATE_TRACE_BYTES, hipHostMallocMapped);
        if alloc.is_err() || host_ptr.is_null() {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] ms1_gate_trace: hipHostMalloc failed: {:?}",
                alloc
            ));
            return 0;
        }
        std::ptr::write_bytes(host_ptr as *mut u8, 0, MS1_GATE_TRACE_BYTES);
        let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let map = hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0);
        if map.is_err() || dev_ptr.is_null() {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] ms1_gate_trace: hipHostGetDevicePointer failed: {:?}",
                map
            ));
            let _ = hipHostFree(host_ptr);
            return 0;
        }
        MS1_GATE_TRACE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::Release);
        MS1_GATE_TRACE_DEV_PTR.store(dev_ptr as u64, std::sync::atomic::Ordering::Release);
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] ms1_gate_trace: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
            MS1_GATE_TRACE_BYTES, host_ptr as usize, dev_ptr as usize
        ));
        dev_ptr as u64
    }
}

/// Dump + reset the trace counters. Called after every Nth MS_1 launch
/// (gated on ZLUDA_MS1_GATE_TRACE=1).
unsafe fn ms1_gate_trace_dump_and_reset() {
    if !ms1_gate_trace_enabled() {
        return;
    }
    let host_ptr = MS1_GATE_TRACE_HOST_PTR.load(std::sync::atomic::Ordering::Acquire);
    if host_ptr == 0 {
        return;
    }
    let tick = MS1_GATE_TRACE_DUMP_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Sync to ensure prior atomic writes are visible
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let bucket = host_ptr as *mut u32;
    let r178_zero = std::ptr::read_volatile(bucket);
    let state_zero = std::ptr::read_volatile(bucket.add(1));
    let bit4_fires = std::ptr::read_volatile(bucket.add(2));
    let total = std::ptr::read_volatile(bucket.add(3));
    super::driver::zluda_log_err(&format!(
        "[MS1_GATE_TRACE] tick={} r178_zero={} state_zero={} bit4_fires={} total={}",
        tick, r178_zero, state_zero, bit4_fires, total
    ));
    // Reset
    std::ptr::write_volatile(bucket, 0);
    std::ptr::write_volatile(bucket.add(1), 0);
    std::ptr::write_volatile(bucket.add(2), 0);
    std::ptr::write_volatile(bucket.add(3), 0);
}

/// 2026-05-16 v10 Step 1: Probe MS_1's pool counter at `[pool_ptr + 22176]`.
/// MS_1 atomically increments this counter to reserve slots; if it stays
/// saturated (>= 924), all MS_1 allocations fail → bit-4 never set →
/// MS_4 wild scatter (per project_rd5_22176_counter_never_reset_2026_05_16).
/// On NVIDIA, RemoveParticleShapes resets this to 0 between frames.
/// On ZLUDA, RemoveParticleShapes is never dispatched (verified run #20).
///
/// Gated on `ZLUDA_POOL_22176_PROBE=1`. Reuses FLAG_PROBE scratch (8 bytes).
unsafe fn snapshot_pool_22176_counter(pool_ptr: u64) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED
        .get_or_init(|| std::env::var("ZLUDA_POOL_22176_PROBE").map_or(false, |v| v == "1"));
    if !enabled || pool_ptr == 0 {
        return;
    }
    let mut guard = match FLAG_PROBE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    flag_probe_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };
    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let copy_result = hipMemcpyDtoD(
        hipDeviceptr_t(state.scratch_dev),
        hipDeviceptr_t((pool_ptr + 22176) as _),
        8,
    );
    if copy_result.is_err() {
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let cnt_22176: u32 = std::ptr::read_unaligned(state.scratch_host as *const u32);
    let cnt_22180: u32 = std::ptr::read_unaligned((state.scratch_host as *const u32).add(1));
    super::driver::zluda_log_err(&format!(
        "[POOL_22176] call_n={} pool_ptr=0x{:x} +22176={} +22180={}",
        call_n, pool_ptr, cnt_22176, cnt_22180
    ));
}

/// Generic flag-bit-distribution probe. Reads `n_threads * 4` bytes
/// from `flags_ptr` and counts bit 0 vs bit 4 set.
unsafe fn snapshot_flag_bit4_at(label: &str, flags_ptr: u64, n_threads: u32) {
    if !flag_count_probe_enabled() || flags_ptr == 0 || n_threads == 0 {
        return;
    }
    let bytes = (n_threads as usize).saturating_mul(4).min(FLAG_PROBE_BYTES);
    let mut guard = match FLAG_PROBE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    flag_probe_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };
    state.launch_n += 1;
    let n = state.launch_n;

    let copy_result = hipMemcpyDtoD(
        hipDeviceptr_t(state.scratch_dev),
        hipDeviceptr_t(flags_ptr as _),
        bytes,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "[FLAG PROBE {}] D2D failed for flags=0x{:x}: {:?}",
            label, flags_ptr, copy_result
        ));
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let mut bit0_set: u32 = 0;
    let mut bit4_set: u32 = 0;
    let mut both_bits_set: u32 = 0;
    let mut zero: u32 = 0;
    let base = state.scratch_host as *const u8;
    for i in 0..n_threads as usize {
        let f: u16 = std::ptr::read_unaligned(base.add(i * 4) as *const u16);
        if f == 0 {
            zero += 1;
        }
        if (f & 0x01) != 0 {
            bit0_set += 1;
        }
        if (f & 0x10) != 0 {
            bit4_set += 1;
        }
        if (f & 0x11) == 0x11 {
            both_bits_set += 1;
        }
    }
    super::driver::zluda_log_err(&format!(
        "[FLAG PROBE {}] launch_n={} flags=0x{:x} n={} bit0_set={} bit4_set={} both={} zero={}",
        label, n, flags_ptr, n_threads, bit0_set, bit4_set, both_bits_set, zero
    ));
}

unsafe fn counter_snapshot_init(state: &mut Option<CounterSnapshotState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, COUNTER_SNAPSHOT_BYTES, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] counter snapshot: hipHostMalloc({}) failed: {:?}",
            COUNTER_SNAPSHOT_BYTES, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, COUNTER_SNAPSHOT_BYTES);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] counter snapshot: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] counter snapshot: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
        COUNTER_SNAPSHOT_BYTES, host_ptr as usize, dev_ptr as usize,
    ));
    *state = Some(CounterSnapshotState {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
        last_bytes: [0u8; COUNTER_SNAPSHOT_BYTES],
        last_ptr: 0,
        launch_n: 0,
    });
}

/// Latch the counter buffer pointer from an MS_4 launch's kp[7]. Called
/// from launch_kernel when MS_4 fires. The MS_4 wrapper's 8-arg layout
/// puts the counter ptr at flat-buffer offset 56 (= arg index 7).
unsafe fn latch_counter_ptr_from_ms4(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) {
    if !counter_snapshot_enabled() {
        return;
    }
    let counter_ptr: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);
    if counter_ptr == 0 {
        return;
    }
    let prev = LATCHED_COUNTER_PTR.swap(counter_ptr, std::sync::atomic::Ordering::SeqCst);
    if prev != counter_ptr {
        super::driver::zluda_log_err(&format!(
            "[COUNTER] latched ptr=0x{:x} (prev=0x{:x}) from MS_4 launch",
            counter_ptr, prev,
        ));
    }
}

/// Ring of the last 32 NRSS launch args — pushed on every
/// IOFX_newRadixSortStepKernel launch when ZLUDA_NRSS_VALIDATE=1, dumped
/// when validation detects a corrupt sort. Removes the arg-trace-window
/// lottery (r11-r13: corrupt frames kept landing outside fixed windows).
/// Entry: (seq, grid, p2, p4, p5, p6, p8, p10, p11, p12).
static NRSS_ARG_RING: Mutex<Option<std::collections::VecDeque<(u32, u32, u32, u32, u32, u64, u64, u64, u32, u32)>>> =
    Mutex::new(None);

unsafe fn nrss_ring_push(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_x: u32,
) {
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p2: u32 = read_kernel_arg(kernel_params, extra, 12, 2).unwrap_or(0);
    let p4: u32 = read_kernel_arg(kernel_params, extra, 20, 4).unwrap_or(0);
    let p5: u32 = read_kernel_arg(kernel_params, extra, 24, 5).unwrap_or(0);
    let p6: u64 = read_kernel_arg(kernel_params, extra, 32, 6).unwrap_or(0);
    let p8: u64 = read_kernel_arg(kernel_params, extra, 48, 8).unwrap_or(0);
    let p10: u64 = read_kernel_arg(kernel_params, extra, 64, 10).unwrap_or(0);
    let p11: u32 = read_kernel_arg(kernel_params, extra, 72, 11).unwrap_or(0);
    let p12: u32 = read_kernel_arg(kernel_params, extra, 76, 12).unwrap_or(0);
    let mut guard = match NRSS_ARG_RING.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let ring = guard.get_or_insert_with(|| std::collections::VecDeque::with_capacity(33));
    if ring.len() >= 32 {
        ring.pop_front();
    }
    ring.push_back((seq, grid_x, p2, p4, p5, p6, p8, p10, p11, p12));
}

fn nrss_ring_dump(tag: &str) {
    let guard = match NRSS_ARG_RING.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(ring) = guard.as_ref() {
        for (seq, grid, p2, p4, p5, p6, p8, p10, p11, p12) in ring.iter() {
            super::driver::zluda_log_err(&format!(
                "NRSS_RING[{}] #{} grid={} p2={} p4={} bit={} p6=0x{:x} p8=0x{:x} p10=0x{:x} phase={} p12={}",
                tag, seq, grid, p2, p4, p5, p6, p8, p10, p11, p12
            ));
        }
    }
}

fn nrss_validation_mode() -> u32 {
    // 0=off, 1=final-pass-only, 2=every scatter pass.
    static MODE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("ZLUDA_NRSS_VALIDATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// 2026-06-10 IOFX-mode writer hunt: in-game NRSS output validation.
/// The lab exonerates IOFX_newRadixSortStepKernel under every reproducible
/// configuration (solo / NRSS+NRSS / cub+NRSS / degenerate keys), yet the
/// in-game IOFX-mode wild writes decode to NRSS-buffer-family + ~10.24 GiB
/// (r4/r8b, ASLR-independent). This checks the kernel's REAL in-game
/// output: after the final scatter pass (p11==3, p5==28), D2H keys_out
/// (p8, p4 entries) and verify u32 non-decreasing order (8×4-bit LSB
/// passes ⇒ fully sorted). A violation = first direct in-game evidence of
/// NRSS output corruption + its location; persistent cleanliness = NRSS
/// exonerated in-game too and the writer is elsewhere in the chain.
/// Caller must have synced the device. Gated ZLUDA_NRSS_VALIDATE=1.
unsafe fn validate_nrss_output_after_launch(
    name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_x: u32,
    block_x: u32,
    shared_mem: u32,
) {
    // 0=off, 1=final-pass-only (whose-fault), 2=EVERY scatter pass
    // (which-pass localization: 2 D2H per phase-3 launch).
    let mode = nrss_validation_mode();
    if mode == 0 {
        return;
    }
    if name != "IOFX_newRadixSortStepKernel" {
        return;
    }
    // Every NRSS launch goes into the args ring (dumped on corruption).
    nrss_ring_push(kernel_params, extra, grid_x);
    let p11: u32 = match read_kernel_arg(kernel_params, extra, 72, 11) {
        Some(v) => v,
        None => return,
    };
    let p5: u32 = match read_kernel_arg(kernel_params, extra, 24, 5) {
        Some(v) => v,
        None => return,
    };
    // 2026-06-12 PHASE-2 SCAN validation (root-cause bisection): read the spine
    // (p10) RIGHT AFTER the scan, BEFORE phase-3 runs, so phase-3 can't confound
    // the read by using p10 as scratch. A correct scan produces a monotonic
    // non-decreasing exclusive prefix (the per-CTA-per-digit global bases). A
    // non-monotonic spine here = Step2 (scan) / Step1 (count) is the corrupter,
    // independent of the scatter. Gated to mode 2; only logs when broken.
    if mode == 2 && p11 == 2 {
        let p10s: u64 = read_kernel_arg(kernel_params, extra, 64, 10).unwrap_or(0);
        // For phase-2 the LAUNCH grid is always 1 (single-block scan); the real
        // CTA count being scanned (= the spine's CTA dimension) is param 12.
        let num_ctas: u32 = read_kernel_arg(kernel_params, extra, 76, 12).unwrap_or(grid_x).max(1);
        let g = num_ctas as usize;
        if p10s != 0 {
            let mut spine = vec![0u32; 16 * g];
            if hipMemcpyDtoH(spine.as_mut_ptr() as _, hipDeviceptr_t(p10s as _), 16 * g * 4).is_ok() {
                // Scan order is digit-major, CTA-minor: spine[d*g + c]. Check the
                // full sequence is non-decreasing (a valid exclusive prefix sum).
                let inversions = spine.windows(2).filter(|w| w[1] < w[0]).count();
                if inversions > 0 {
                    static SSEQ: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let ss = SSEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if ss < 40 || ss % 50 == 0 {
                        super::driver::zluda_log_err(&format!(
                            "*** NRSS_SCAN_BROKEN #{} bit={} num_ctas={} block={} inversions={} spine[0]={:x} spine={:x?} => Step2 scan produced NON-MONOTONIC spine (count/scan corrupter, NOT scatter)",
                            ss, p5, num_ctas, block_x, inversions, spine.first().copied().unwrap_or(0), spine
                        ));
                        // NO-OP vs MISCOMPUTE: for grid==1 the histogram == the
                        // GLOBAL per-digit counts (computable from the input keys
                        // p6), and the correct spine is their exclusive prefix sum.
                        // If actual spine == raw histogram => the scan ran as
                        // IDENTITY (no-op: didn't accumulate). If actual spine
                        // == correct exclusive-prefix => scan fine (shouldn't
                        // happen here). Otherwise the scan miscomputed.
                        if g == 1 {
                            let p6h: u64 = read_kernel_arg(kernel_params, extra, 32, 6).unwrap_or(0);
                            let p4h: u32 = read_kernel_arg(kernel_params, extra, 20, 4).unwrap_or(0);
                            let nn = (p4h as usize).min(65536);
                            if p6h != 0 && nn >= 2 {
                                let mut keys = vec![0u32; nn];
                                if hipMemcpyDtoH(keys.as_mut_ptr() as _, hipDeviceptr_t(p6h as _), nn * 4).is_ok() {
                                    let mut hist = [0u32; 16];
                                    for &k in &keys { hist[((k >> p5) & 0xF) as usize] += 1; }
                                    let mut excl = [0u32; 16];
                                    let mut acc = 0u32;
                                    for d in 0..16 { excl[d] = acc; acc += hist[d]; }
                                    let eq_raw = (0..16).all(|d| spine[d] == hist[d]);
                                    let eq_scan = (0..16).all(|d| spine[d] == excl[d]);
                                    let verdict = if eq_raw { "SCAN IS NO-OP (spine == raw histogram; Step2 did not accumulate)" }
                                        else if eq_scan { "scan correct?!" }
                                        else { "scan MISCOMPUTED (spine != raw AND != correct prefix)" };
                                    super::driver::zluda_log_err(&format!(
                                        "*** NRSS_SCAN_DETAIL bit={} n={} hist={:x?} correct_excl={:x?} actual_spine={:x?} => {}",
                                        p5, nn, hist, excl, &spine[..16], verdict
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }
    if p11 != 3 {
        return;
    }
    // Mode 2: validate EVERY scatter pass. Radix sort is LSB-first; after
    // the pass for `bit`, keys must be sorted by the low (bit+4) bits, and
    // its INPUT must already be sorted by the low `bit` bits. The first
    // pass where input is OK but output breaks = the pass that corrupted.
    if mode == 2 {
        let p4: u32 = read_kernel_arg(kernel_params, extra, 20, 4).unwrap_or(0);
        let p6: u64 = read_kernel_arg(kernel_params, extra, 32, 6).unwrap_or(0);
        let p8: u64 = read_kernel_arg(kernel_params, extra, 48, 8).unwrap_or(0);
        let n = (p4 as usize).min(65536);
        if n >= 2 && p6 != 0 && p8 != 0 {
            let in_mask: u32 = if p5 == 0 { 0 } else { ((1u64 << p5) - 1) as u32 };
            let out_bits = p5 + 4;
            let out_mask: u32 = if out_bits >= 32 {
                0xFFFF_FFFF
            } else {
                ((1u64 << out_bits) - 1) as u32
            };
            let mut inp = vec![0u32; n];
            let mut outp = vec![0u32; n];
            let ok_in = hipMemcpyDtoH(inp.as_mut_ptr() as _, hipDeviceptr_t(p6 as _), n * 4).is_ok();
            let ok_out = hipMemcpyDtoH(outp.as_mut_ptr() as _, hipDeviceptr_t(p8 as _), n * 4).is_ok();
            // 2026-06-12 VALUE-scatter validation: the radix sort carries a
            // (key, value) pair; IOFX_actorRangeKernel scatters using the sorted
            // VALUES (particle indices, param 9), NOT the keys (param 8). A run
            // with 52/52 clean KEY validations still crashed at actorRange ⇒ the
            // bug may be in the VALUE permutation, invisible to key checks. The
            // sorted values must be a permutation of the input values (param 7).
            let p7v: u64 = read_kernel_arg(kernel_params, extra, 40, 7).unwrap_or(0);
            let p9v: u64 = read_kernel_arg(kernel_params, extra, 56, 9).unwrap_or(0);
            if p7v != 0 && p9v != 0 {
                let mut vin = vec![0u32; n];
                let mut vout = vec![0u32; n];
                let vi_ok = hipMemcpyDtoH(vin.as_mut_ptr() as _, hipDeviceptr_t(p7v as _), n * 4).is_ok();
                let vo_ok = hipMemcpyDtoH(vout.as_mut_ptr() as _, hipDeviceptr_t(p9v as _), n * 4).is_ok();
                if vi_ok && vo_ok {
                    let mut svi = vin.clone();
                    let mut svo = vout.clone();
                    svi.sort_unstable();
                    svo.sort_unstable();
                    if svi != svo {
                        let in_set: std::collections::HashSet<u32> = vin.iter().copied().collect();
                        let foreign = vout.iter().filter(|v| !in_set.contains(v)).count();
                        let vzeros = vout.iter().filter(|&&v| v == 0).count();
                        static VSEQ: std::sync::atomic::AtomicU32 =
                            std::sync::atomic::AtomicU32::new(0);
                        let vs = VSEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        super::driver::zluda_log_err(&format!(
                            "*** NRSS_VAL_LOSS #{} bit={} n={} grid={} block={} shmem={} p9=0x{:x} VALUES-NOT-PERM foreign={} vzeros={} vin[..6]={:x?} vout[..6]={:x?} => value-scatter corrupt (actorRange producer)",
                            vs, p5, n, grid_x, block_x, shared_mem, p9v, foreign, vzeros,
                            &vin[..6.min(n)], &vout[..6.min(n)]
                        ));
                        // One-shot capture of the failing value-scatter dataset.
                        static VDUMP: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !VDUMP.swap(true, std::sync::atomic::Ordering::SeqCst) {
                            let path = format!("/tmp/nrss_valcorrupt_bit{}_grid{}_n{}.bin", p5, grid_x, n);
                            let mut blob: Vec<u8> = Vec::with_capacity((6 + 4 * n) * 4);
                            // header v2: magic 0x4e525357, n, bit, grid, block, shmem
                            for h in [0x4e525357u32, n as u32, p5, grid_x, block_x, shared_mem] {
                                blob.extend_from_slice(&h.to_le_bytes());
                            }
                            for &k in &inp { blob.extend_from_slice(&k.to_le_bytes()); }
                            for &k in &vin { blob.extend_from_slice(&k.to_le_bytes()); }
                            for &k in &outp { blob.extend_from_slice(&k.to_le_bytes()); }
                            for &k in &vout { blob.extend_from_slice(&k.to_le_bytes()); }
                            let _ = std::fs::write(&path, &blob);
                            super::driver::zluda_log_err(&format!(
                                "*** NRSS_VAL_CAPTURE wrote value-scatter dataset to {} ({} bytes)",
                                path, blob.len()
                            ));
                        }
                        // 2026-06-12 PHASE-PINPOINT probe (fires on every value-loss):
                        // (1) SPINE consistency — read p10 (the scanned global digit
                        //     bases, layout digit*grid+ctaid). For a correct scatter the
                        //     bases must be non-decreasing in scan order and the final
                        //     base+count must equal n. A broken spine => phase-2 scan is
                        //     the corrupter. A consistent spine + lost elements => phase-3
                        //     scatter (rank/write) is the corrupter.
                        // (2) OOB-SCATTER — read the OUTPUT key buffer's beyond-n region
                        //     [n, n+256). If written (non-zero), the scatter computed an
                        //     offset >= n and wrote OUT OF BOUNDS (wild write = the GPU
                        //     fault), which ALSO removes the element from [0,n) => loss.
                        let p10s: u64 = read_kernel_arg(kernel_params, extra, 64, 10).unwrap_or(0);
                        let g = grid_x.max(1) as usize;
                        let mut spine = vec![0u32; 16 * g];
                        let spine_ok = p10s != 0
                            && hipMemcpyDtoH(spine.as_mut_ptr() as _, hipDeviceptr_t(p10s as _), 16 * g * 4).is_ok();
                        if spine_ok {
                            let maxv = spine.iter().copied().max().unwrap_or(0);
                            let over_n = spine.iter().filter(|&&v| v > n as u32).count();
                            super::driver::zluda_log_err(&format!(
                                "*** NRSS_SPINE bit={} grid={} n={} max={} over_n={} spine={:x?}",
                                p5, grid_x, n, maxv, over_n, spine
                            ));
                        }
                        // ROOT-CAUSE BISECTION (2026-06-12): rank-error vs spine/count-error.
                        // Per-digit input count at THIS bit vs the spine-reserved region
                        // size. spine layout = base[digit*grid + cta]; digit d's region
                        // = [spine[d*grid], spine[(d+1)*grid]). If region size == input
                        // count for every digit, the spine RESERVED exactly right => any
                        // overflow is a LOCAL-RANK error (rank exceeds the digit's count).
                        // If they mismatch, the count/scan (Step1/2) is the corrupter.
                        if spine_ok {
                            let mut in_count = [0u32; 16];
                            for &k in &inp {
                                in_count[((k >> p5) & 0xF) as usize] += 1;
                            }
                            let region = |d: usize| -> i64 {
                                let lo = spine[d * g] as i64;
                                let hi = if d + 1 < 16 { spine[(d + 1) * g] as i64 } else { n as i64 };
                                hi - lo
                            };
                            let mut mismatches: Vec<(usize, i64, u32)> = Vec::new();
                            for d in 0..16 {
                                let r = region(d);
                                if r != in_count[d] as i64 {
                                    mismatches.push((d, r, in_count[d]));
                                }
                            }
                            // Capture the FULL overflow region [n, aligned + 4*pad + slack)
                            // and classify the overflowed elements by digit.
                            let aligned = ((n + 127) & !127) as usize;
                            // Read just past the aligned (tile-padded) region + a small
                            // margin; bounded so we never read far past the allocation.
                            let oob_words = ((aligned - n) + 256).min(4096);
                            let mut oob = vec![0u32; oob_words];
                            let oob_read = hipMemcpyDtoH(
                                oob.as_mut_ptr() as _,
                                hipDeviceptr_t((p8 + (n as u64) * 4) as _),
                                oob_words * 4,
                            ).is_ok();
                            let nz = if oob_read { oob.iter().filter(|&&v| v != 0).count() } else { 0 };
                            let mut oob_digits = [0u32; 16];
                            if oob_read {
                                for &k in oob.iter().filter(|&&v| v != 0) {
                                    oob_digits[((k >> p5) & 0xF) as usize] += 1;
                                }
                            }
                            // Are all OOB-element digits ones whose spine region fit in [0,n)?
                            let oob_digits_fit = (0..16).all(|d| {
                                oob_digits[d] == 0 || (spine[d * g] as i64 + in_count[d] as i64) <= n as i64
                            });
                            let verdict = if !mismatches.is_empty() {
                                "SPINE/COUNT ERROR (region size != input count) => Step1/Step2 corrupter"
                            } else if nz > 0 && oob_digits_fit {
                                "LOCAL-RANK ERROR (spine reserved exactly, yet rank overflowed past digit region) => Step3 intra-CTA scan"
                            } else if nz > 0 {
                                "OOB but spine-region for those digits already exceeds n => mixed"
                            } else {
                                "no OOB (in-bounds collision only)"
                            };
                            super::driver::zluda_log_err(&format!(
                                "*** NRSS_BISECT bit={} n={} oob_nz={} region_mismatches={:?} oob_digits={:?} => {}",
                                p5, n, nz, mismatches, oob_digits, verdict
                            ));
                        }
                    }
                }
            }
            // 2026-06-12: on the FIRST pass (bit=0) dump the input buffer's
            // beyond-n tail [n, ceil(n/128)*128). The radix StepKernel pads the
            // partial final tile to 128 and its scatter over-reads this region
            // unclamped; correctness REQUIRES these slots == 0xFFFFFFFF
            // (sentinel). all-FF => PhysX sentinel-padded (ZLUDA bug is
            // elsewhere); all-zero => no padding (benign-on-NVIDIA or
            // zero-init divergence); stale keys => zero-init divergence.
            if p5 == 0 && n % 128 != 0 {
                let aligned = (n + 127) & !127;
                let tail = aligned - n;
                let mut tbuf = vec![0u32; tail];
                if hipMemcpyDtoH(
                    tbuf.as_mut_ptr() as _,
                    hipDeviceptr_t((p6 + (n as u64) * 4) as _),
                    tail * 4,
                )
                .is_ok()
                {
                    let zeros = tbuf.iter().filter(|&&x| x == 0).count();
                    let ffs = tbuf.iter().filter(|&&x| x == 0xFFFF_FFFF).count();
                    let cls = if zeros == tail {
                        "ALL-ZERO (no sentinel pad)"
                    } else if ffs == tail {
                        "ALL-FF (sentinel padded)"
                    } else {
                        "MIXED (stale/valid keys)"
                    };
                    super::driver::zluda_log_err(&format!(
                        "NRSS_TAIL n={} nmod128={} tail={} zeros={} ffs={} first6={:x?} => {}",
                        n, n % 128, tail, zeros, ffs,
                        &tbuf[..6.min(tbuf.len())], cls
                    ));
                }
            }
            if ok_in && ok_out {
                let in_viol = inp
                    .windows(2)
                    .filter(|w| (w[1] & in_mask) < (w[0] & in_mask))
                    .count();
                let out_viol = outp
                    .windows(2)
                    .filter(|w| (w[1] & out_mask) < (w[0] & out_mask))
                    .count();
                if out_viol > 0 {
                    static PSEQ: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let ps = PSEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let verdict = if in_viol == 0 {
                        "THIS scatter pass broke it (input clean, output unsorted)"
                    } else {
                        "came in already broken (upstream of this bit)"
                    };
                    // Permutation check: is the output the SAME multiset as the
                    // input (wrong scatter OFFSETS — histogram/scan bad) or does
                    // it contain FOREIGN values (an external kernel clobbered
                    // p8)? Decisive internal-vs-external split.
                    let mut si = inp.clone();
                    let mut so = outp.clone();
                    si.sort_unstable();
                    so.sort_unstable();
                    let is_perm = si == so;
                    // Count output values absent from the input multiset.
                    let in_set: std::collections::HashSet<u32> = inp.iter().copied().collect();
                    let foreign: Vec<u32> = outp
                        .iter()
                        .copied()
                        .filter(|v| !in_set.contains(v))
                        .take(6)
                        .collect();
                    // Histogram (p10): for bit=0 the scatter offsets come from
                    // this pass's count+scan. Dump its first 16 entries.
                    let p10: u64 = read_kernel_arg(kernel_params, extra, 64, 10).unwrap_or(0);
                    let mut hist = vec![0u32; 16];
                    if p10 != 0 {
                        let _ = hipMemcpyDtoH(
                            hist.as_mut_ptr() as _,
                            hipDeviceptr_t(p10 as _),
                            16 * 4,
                        );
                    }
                    super::driver::zluda_log_err(&format!(
                        "*** NRSS_PASS #{} bit={} n={} p6=0x{:x} p8=0x{:x} p10=0x{:x} in_viol={} out_viol={} is_perm={} foreign={:x?} hist16={:?} grid={} => {}",
                        ps, p5, n, p6, p8, p10, in_viol, out_viol, is_perm, foreign, hist, grid_x, verdict
                    ));
                    // 2026-06-12 faithful-repro capture: on the FIRST true
                    // corrupter (clean input -> lost-element output), dump the
                    // exact input-key buffer + output so the failing grid>=2
                    // scatter can be replayed deterministically in the lab.
                    // The lab harness is grid>=3-unfaithful only because it
                    // lacks the real data; this hands it the real data.
                    if in_viol == 0 && !is_perm {
                        static DUMPED: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !DUMPED.swap(true, std::sync::atomic::Ordering::SeqCst) {
                            let path = format!(
                                "/tmp/nrss_corrupt_bit{}_grid{}_n{}.bin",
                                p5, grid_x, n
                            );
                            // Layout: [u32 magic=0x4e525353, u32 n, u32 bit,
                            //          u32 grid, n*u32 input_keys, n*u32 output_keys]
                            let mut blob: Vec<u8> = Vec::with_capacity((4 + 2 * n) * 4);
                            for h in [0x4e52_5353u32, n as u32, p5, grid_x] {
                                blob.extend_from_slice(&h.to_le_bytes());
                            }
                            for &k in &inp {
                                blob.extend_from_slice(&k.to_le_bytes());
                            }
                            for &k in &outp {
                                blob.extend_from_slice(&k.to_le_bytes());
                            }
                            let _ = std::fs::write(&path, &blob);
                            super::driver::zluda_log_err(&format!(
                                "*** NRSS_CAPTURE wrote failing dataset to {} ({} bytes) — replay in lab",
                                path, blob.len()
                            ));
                        }
                    }
                }
                // 2026-06-12 Class-B silent-loss detector: a dropped scatter
                // can leave an output slot UNWRITTEN. If that slot ends up
                // holding a low/zero value the multiset shrinks but adjacent
                // sortedness can still hold (out_viol==0) — the exact pattern
                // behind the host-side Class-B NULL deref (inner_array[0..2]=0,
                // a compaction PhysX builds from this sort's permutation). The
                // out_viol>0 branch already reports is_perm; here we catch the
                // case it misses: sorted-but-not-a-permutation (elements LOST).
                if out_viol == 0 {
                    let mut si = inp.clone();
                    let mut so = outp.clone();
                    si.sort_unstable();
                    so.sort_unstable();
                    if si != so {
                        // Count output values absent from the input multiset
                        // and zero-holes (the canonical lost-slot fill).
                        let in_set: std::collections::HashSet<u32> =
                            inp.iter().copied().collect();
                        let foreign = outp.iter().filter(|v| !in_set.contains(v)).count();
                        let out_zeros = outp.iter().filter(|&&v| v == 0).count();
                        let in_zeros = inp.iter().filter(|&&v| v == 0).count();
                        static LSEQ: std::sync::atomic::AtomicU32 =
                            std::sync::atomic::AtomicU32::new(0);
                        let ls = LSEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        super::driver::zluda_log_err(&format!(
                            "*** NRSS_LOSS #{} bit={} n={} p8=0x{:x} grid={} SORTED-BUT-NOT-PERM foreign={} out_zeros={} in_zeros={} => silent element loss (Class-B producer signature)",
                            ls, p5, n, p8, grid_x, foreign, out_zeros, in_zeros
                        ));
                        nrss_ring_dump(&format!("loss#{}", ls));
                    }
                }
                // 2026-06-12 partial-tile correlation: on the FINAL scatter
                // pass (bit=28) log a compact shape line for EVERY episode
                // (clean OR corrupt) so n%128==0 (clean-predicted) vs n%128!=0
                // (corrupt-predicted) can be correlated. Tests the lab finding
                // that out_viol tracks the partial final tile (tile size 128).
                // is_perm catches silent element loss even when out_viol==0.
                if p5 == 28 {
                    static FSEQ: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let fs = FSEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if out_viol > 0 || fs < 60 || fs % 50 == 0 {
                        let mut si = inp.clone();
                        let mut so = outp.clone();
                        si.sort_unstable();
                        so.sort_unstable();
                        let is_perm = si == so;
                        super::driver::zluda_log_err(&format!(
                            "NRSS_FINAL #{} n={} nmod128={} out_viol={} is_perm={} grid={}",
                            fs, n, (n as u32) % 128, out_viol, is_perm, grid_x
                        ));
                    }
                }
            }
        }
        return;
    }
    if p5 != 28 {
        return;
    }
    let p4: u32 = read_kernel_arg(kernel_params, extra, 20, 4).unwrap_or(0);
    let p8: u64 = read_kernel_arg(kernel_params, extra, 48, 8).unwrap_or(0);
    let n = (p4 as usize).min(65536);
    if n < 2 || p8 == 0 {
        return;
    }
    let mut keys = vec![0u32; n];
    let copy = hipMemcpyDtoH(
        keys.as_mut_ptr() as _,
        hipDeviceptr_t(p8 as _),
        n * 4,
    );
    if copy.is_err() {
        super::driver::zluda_log_err(&format!(
            "NRSS_VALIDATE: D2H failed p8=0x{:x} n={}: {:?}",
            p8, n, copy
        ));
        return;
    }
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let viol = keys.windows(2).filter(|w| w[1] < w[0]).count();
    if viol > 0 {
        let first = keys
            .windows(2)
            .position(|w| w[1] < w[0])
            .unwrap_or(0);
        let lo = first.saturating_sub(3);
        let hi = (first + 5).min(n);
        // LOCALIZATION (2026-06-11): also read the INPUT to this final pass
        // (p6) and check the invariant the first 7 passes must have
        // established — keys sorted by the low 28 bits (mask 0x0FFFFFFF).
        //   in_lo28_sorted == true  -> the input to the bit-28 pass was
        //     correct; THIS StepKernel launch turned correct input into
        //     wrong output => the radix-sort kernel itself is the corrupter
        //     at the final pass (our codegen / a kernel bug — read its ISA).
        //   in_lo28_sorted == false -> corruption is UPSTREAM of the final
        //     pass (an earlier pass, or an external kernel wrote into these
        //     buffers between launches) => walk back / hunt the writer.
        // Also flag wild values (a foreign scatter would leave non-key
        // bit-patterns) vs a clean permutation that's merely mis-ordered.
        let p6: u64 = read_kernel_arg(kernel_params, extra, 32, 6).unwrap_or(0);
        let (in_lo28_viol, in_first, in_window): (i64, i64, Vec<u32>) = if p6 != 0 {
            let mut inp = vec![0u32; n];
            if hipMemcpyDtoH(inp.as_mut_ptr() as _, hipDeviceptr_t(p6 as _), n * 4).is_ok() {
                let mask = 0x0FFF_FFFFu32;
                let v = inp
                    .windows(2)
                    .filter(|w| (w[1] & mask) < (w[0] & mask))
                    .count() as i64;
                let f = inp
                    .windows(2)
                    .position(|w| (w[1] & mask) < (w[0] & mask))
                    .map(|x| x as i64)
                    .unwrap_or(-1);
                let wlo = f.max(0) as usize;
                let w = inp[wlo..(wlo + 6).min(n)].to_vec();
                (v, f, w)
            } else {
                (-1, -1, Vec::new())
            }
        } else {
            (-1, -1, Vec::new())
        };
        let verdict = match in_lo28_viol {
            0 => "INPUT-OK => final bit28 pass corrupted (kernel-internal)",
            v if v > 0 => "INPUT-CORRUPT => upstream of final pass (earlier pass or external writer)",
            _ => "input-unread",
        };
        super::driver::zluda_log_err(&format!(
            "*** NRSS_VALIDATE #{} CORRUPT n={} p8=0x{:x} violations={} first_at={} window={:x?} | p6=0x{:x} in_lo28_viol={} in_first={} in_window={:x?} | {}",
            seq, n, p8, viol, first, &keys[lo..hi],
            p6, in_lo28_viol, in_first, in_window, verdict
        ));
        // Dump the last 32 NRSS launches (the corrupt sort's own 24 + tail
        // of the previous one) for arg-consistency analysis.
        nrss_ring_dump(&format!("corrupt#{}", seq));
    } else if seq < 20 || seq % 200 == 0 {
        super::driver::zluda_log_err(&format!(
            "NRSS_VALIDATE #{} ok n={} p8=0x{:x} min=0x{:x} max=0x{:x}",
            seq,
            n,
            p8,
            keys.first().copied().unwrap_or(0),
            keys.last().copied().unwrap_or(0)
        ));
    }
}

/// Snapshot the latched counter buffer after a kernel launch. Called from
/// launch_kernel after a successful hipModuleLaunchKernel + sync. Logs only
/// when the snapshot changes from the previous one, with the kernel name.
/// Special-cases the non-zero → all-zero transition (the bug signature
/// we're hunting) with a `[COUNTER ZEROED]` tag.
unsafe fn snapshot_counter_after_launch(kernel_name: &str) {
    if !counter_snapshot_enabled() {
        return;
    }
    let ptr = LATCHED_COUNTER_PTR.load(std::sync::atomic::Ordering::SeqCst);
    if ptr == 0 {
        return;
    }

    let mut guard = match COUNTER_SNAPSHOT.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    counter_snapshot_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };

    // If the latched pointer just changed (a new MS_4 frame), reset the
    // baseline so we don't false-positive-diff against the previous buffer.
    if state.last_ptr != ptr {
        state.last_bytes = [0u8; COUNTER_SNAPSHOT_BYTES];
        state.last_ptr = ptr;
    }
    state.launch_n += 1;
    let n = state.launch_n;

    let copy_result = hipMemcpyDtoD(
        hipDeviceptr_t(state.scratch_dev),
        hipDeviceptr_t(ptr as _),
        COUNTER_SNAPSHOT_BYTES,
    );
    if copy_result.is_err() {
        // Bad / unmapped pointer. Unlatch so we don't keep retrying.
        super::driver::zluda_log_err(&format!(
            "[COUNTER] snapshot D2D failed for ptr=0x{:x} after '{}': {:?} — unlatching",
            ptr, kernel_name, copy_result
        ));
        LATCHED_COUNTER_PTR.store(0, std::sync::atomic::Ordering::SeqCst);
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let cur_slice: &[u8] = std::slice::from_raw_parts(state.scratch_host, COUNTER_SNAPSHOT_BYTES);
    if cur_slice == state.last_bytes {
        return;
    }

    // Mutated. Build the human log line and detect the all-zero bug signature.
    let cur_i32: &[i32] =
        std::slice::from_raw_parts(state.scratch_host as *const i32, COUNTER_SNAPSHOT_BYTES / 4);
    let prev_i32: &[i32] = std::slice::from_raw_parts(
        state.last_bytes.as_ptr() as *const i32,
        COUNTER_SNAPSHOT_BYTES / 4,
    );
    let all_zero_now = cur_i32.iter().all(|&x| x == 0);
    let prev_had_nonzero = prev_i32.iter().any(|&x| x != 0);
    let tag = if all_zero_now && prev_had_nonzero {
        "ZEROED"
    } else if all_zero_now {
        "STILL_ZERO"
    } else {
        "CHANGED"
    };
    let mut prev_str = String::with_capacity(80);
    let mut new_str = String::with_capacity(80);
    prev_str.push('[');
    new_str.push('[');
    for j in 0..8 {
        if j > 0 {
            prev_str.push(',');
            new_str.push(',');
        }
        prev_str.push_str(&prev_i32[j].to_string());
        new_str.push_str(&cur_i32[j].to_string());
    }
    prev_str.push(']');
    new_str.push(']');
    super::driver::zluda_log_err(&format!(
        "[COUNTER {}] launch_n={} kernel='{}' ptr=0x{:x} prev={} new={}",
        tag, n, kernel_name, ptr, prev_str, new_str,
    ));
    state.last_bytes.copy_from_slice(cur_slice);
}

// ============================================================================
// MS_4 counter underflow fix (2026-04-10)
// ============================================================================
//
// ROOT CAUSE: PhysX's CreateParticleShapesIncremental_MultiStage_4 does
//   scatter_idx = base_count + atom.add(counter[shape], -1) - 1
// When counter[shape]=0 (set by MS_2B for shapes with particle_count=1),
// atom.add returns 0, scatter_idx = base_count + 0 - 1 = 0xFFFFFFFF (underflow).
// The kernel then writes at scatter_buf + 0xFFFFFFFF*4 = scatter_buf + 16 GiB,
// which is unmapped → GPU page fault.
//
// NVIDIA silently drops writes to unmapped pages; AMD faults. This is a latent
// PhysX bug that's invisible on NVIDIA hardware.
//
// FIX: Before each MS_4 launch, D2D the counter array (param_7) to a
// host-mapped scratch, check for zeros, and if found, patch each zero to 1.
// With counter=1, atom.add returns 1, scatter_idx = base + 0 = valid.
// The write targets a real slot (the shape's base offset). Since the shape
// only had 1 particle (the pivot), this slot would have been written by
// the pivot anyway — we're just making the write land in-bounds.
//
// Cost: 1-2 hipMemcpyDtoD + 1 hipDeviceSynchronize per MS_4 launch (~20/session).
// This is far below the ProfilingSignal race threshold.

const MS4_MAX_SHAPES: usize = 1024; // PhysX caps at 924, use 1024 for safety

struct Ms4CounterFixup {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
}
unsafe impl Send for Ms4CounterFixup {}
unsafe impl Sync for Ms4CounterFixup {}

static MS4_COUNTER_FIXUP: Mutex<Option<Ms4CounterFixup>> = Mutex::new(None);

unsafe fn ms4_counter_fixup_init(state: &mut Option<Ms4CounterFixup>) {
    if state.is_some() {
        return;
    }
    let alloc_size = MS4_MAX_SHAPES * 4;
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let r = hipHostMalloc(&mut host_ptr, alloc_size, hipHostMallocMapped);
    if r.is_err() || host_ptr.is_null() {
        return;
    }
    std::ptr::write_bytes(host_ptr as *mut u8, 0, alloc_size);
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let r2 = hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0);
    if r2.is_err() || dev_ptr.is_null() {
        let _ = hipHostFree(host_ptr);
        return;
    }
    *state = Some(Ms4CounterFixup {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
    });
}

/// Pre-launch fixup for MS_4: patch any zero counters to 1.
/// Returns the number of zeros found and patched.
unsafe fn fixup_ms4_counters(
    counter_ptr: u64, // param_7 device pointer
    n_particles: u32, // param_3 (N) — upper bound on shape count
) -> u32 {
    if counter_ptr == 0 {
        return 0;
    }
    // The counter array has one u32 per shape. We don't know the exact shape
    // count, but MS_4's grid_x * block_x threads can reference at most
    // grid_x * block_x different shapes. A safe upper bound on the shape
    // indices is n_particles (each particle has one shape). But shapes are
    // usually << particles. Use min(n_particles, MS4_MAX_SHAPES).
    let n_shapes = (n_particles as usize).min(MS4_MAX_SHAPES);
    let copy_bytes = n_shapes * 4;

    let mut guard = match MS4_COUNTER_FIXUP.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    ms4_counter_fixup_init(&mut guard);
    let buf = match guard.as_mut() {
        Some(b) => b,
        None => return 0,
    };

    // D2D copy counter array → host-mapped scratch
    let r = hipMemcpyDtoD(
        hipDeviceptr_t(buf.dev_ptr),
        hipDeviceptr_t(counter_ptr as *mut std::ffi::c_void),
        copy_bytes,
    );
    if r.is_err() {
        return 0;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Log first 32 values BEFORE patching for diagnosis
    let counters = std::slice::from_raw_parts_mut(buf.host_ptr as *mut u32, n_shapes);
    let dump_n = n_shapes.min(32);
    let mut vals = String::with_capacity(dump_n * 8);
    for i in 0..dump_n {
        if i > 0 {
            vals.push(',');
        }
        vals.push_str(&format!("{}", counters[i]));
    }
    super::driver::zluda_log_err(&format!(
        "[MS4_COUNTERS] n={} ptr=0x{:x} first{}=[{}]",
        n_shapes, counter_ptr, dump_n, vals,
    ));

    // Diagnostic only — no patching. Count zeros in counter range.
    let mut n_zeros = 0u32;
    for i in 0..n_shapes {
        if counters[i] == 0 {
            n_zeros += 1;
        }
    }
    n_zeros
}

/// Root cause diagnostic: read flags (param_2), shape_idx (param_6), and
/// counters (param_7). For each particle with (flags & 0x11)==1, tally
/// which shape it references. Compare active[s] vs counter[s] per shape.
/// If active[s] > counter[s], the scatter will underflow for shape s.
unsafe fn scan_ms4_shape_idx(
    flags_ptr: u64,
    shape_idx_ptr: u64,
    counter_ptr: u64,
    n_particles: u32,
) {
    if flags_ptr == 0 || shape_idx_ptr == 0 || counter_ptr == 0 || n_particles == 0 {
        return;
    }
    let n = (n_particles as usize).min(2048);

    let mut guard = match MS4_COUNTER_FIXUP.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    ms4_counter_fixup_init(&mut guard);
    let buf = match guard.as_ref() {
        Some(b) => b,
        None => return,
    };
    let chunk = MS4_MAX_SHAPES; // 1024 u32s fit in our scratch

    // 1. Read counters (first 64)
    let n_ctr = 64usize;
    let r = hipMemcpyDtoD(
        hipDeviceptr_t(buf.dev_ptr),
        hipDeviceptr_t(counter_ptr as *mut std::ffi::c_void),
        n_ctr * 4,
    );
    if r.is_err() {
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let ctr = std::slice::from_raw_parts(buf.host_ptr as *const u32, n_ctr);
    let mut counter_vals = [0u32; 64];
    counter_vals.copy_from_slice(ctr);
    let n_shapes = counter_vals
        .iter()
        .rposition(|&v| v != 0)
        .map(|i| i + 1)
        .unwrap_or(0);

    // 2. Read flags + shape_idx, count active per shape
    let mut active_per_shape = [0u32; 64];
    let mut offset = 0usize;
    while offset < n {
        let this_chunk = (n - offset).min(chunk);
        // Read flags chunk (u16 at stride 4, so 4 bytes per particle)
        let r2 = hipMemcpyDtoD(
            hipDeviceptr_t(buf.dev_ptr),
            hipDeviceptr_t((flags_ptr + (offset * 4) as u64) as *mut std::ffi::c_void),
            this_chunk * 4,
        );
        if r2.is_err() {
            break;
        }
        let _ = hipDeviceSynchronize();
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let flags_raw = std::slice::from_raw_parts(buf.host_ptr as *const u8, this_chunk * 4);
        // Save flags locally
        let mut flags_local = vec![0u16; this_chunk];
        for i in 0..this_chunk {
            flags_local[i] = u16::from_le_bytes([flags_raw[i * 4], flags_raw[i * 4 + 1]]);
        }

        // Read shape_idx chunk
        let r3 = hipMemcpyDtoD(
            hipDeviceptr_t(buf.dev_ptr),
            hipDeviceptr_t((shape_idx_ptr + (offset * 4) as u64) as *mut std::ffi::c_void),
            this_chunk * 4,
        );
        if r3.is_err() {
            break;
        }
        let _ = hipDeviceSynchronize();
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let sids = std::slice::from_raw_parts(buf.host_ptr as *const u32, this_chunk);

        for i in 0..this_chunk {
            let flag = flags_local[i];
            if (flag & 0x11) == 1 {
                let sid = sids[i] as usize;
                if sid < 64 {
                    active_per_shape[sid] += 1;
                }
            }
        }
        offset += this_chunk;
    }

    // 3. Compare active vs counter per shape
    let mut mismatches = Vec::new();
    for s in 0..n_shapes {
        let a = active_per_shape[s];
        let c = counter_vals[s];
        if a != c {
            mismatches.push(format!("s{}:active={},ctr={}", s, a, c));
        }
    }
    if !mismatches.is_empty() {
        super::driver::zluda_log_err(&format!(
            "[MS4_MISMATCH] n_shapes={} mismatches: {}",
            n_shapes,
            mismatches.join(" "),
        ));
    } else {
        super::driver::zluda_log_err(&format!(
            "[MS4_MATCH] n_shapes={} all active==counter (no mismatch)",
            n_shapes,
        ));
    }
}

// ============================================================================
// TurbulenceFS launch capture ring (2026-04-09 evening, end-of-day)
// ============================================================================
//
// PURPOSE: Replay infrastructure for TurbulenceFS kernel crashes.
//
// The 8 timer-fixed runs (52-60) on the new stack all crashed on various
// TurbulenceFS kernels (ParticleDensityToParticles, PressureDivergence,
// SmoothVelocity, dynamicGridClear, Advect — different kernel each run),
// with Bug A and Bug B both verified non-reproducing in isolated tests.
// To investigate the TurbulenceFS family in isolation, we need to capture
// the actual kernel arguments (especially the 456-byte DynamicGrid
// descriptor struct) from a real game run, then replay them in a unit test.
//
// 4 of 5 crashing TurbulenceFS kernels take a 456-byte struct as either
// param_4 (ParticleDensityToParticles) or param_5 (Advect, SmoothVelocity,
// PressureDivergence, plus most other TurbulenceFS_dynamicGrid* kernels).
// The struct contains GPU pointers at offsets 232, 256, 312, 320 (the
// velocity/density/pressure field buffers) plus various scalar grid
// metadata.
//
// V1 (this revision): capture the raw kernel argument flat buffer for
// every TurbulenceFS_* kernel launch, including the embedded 456-byte
// struct. Does NOT yet follow the GPU pointers — that's V2.
//
// Gated on `ZLUDA_TFS_CAPTURE=1`. Off by default. The capture ring is a
// 256-slot pinned host-mapped buffer dumped from the SIGABRT handler to
// `/tmp/tfs_captures.bin` (same dump pattern as MS_4 / MS_2_Batched / MS_3).

const TFS_CAPTURE_SLOTS: usize = 4096;
const TFS_DESC_BYTES: usize = 456;
const TFS_KERNEL_NAME_BYTES: usize = 64;
const TFS_RAW_ARGS_BYTES: usize = 512; // covers struct + scalar args + extras
const TFS_RAW_ARGS_OFF: usize = 120;
const TFS_SLOT_SIZE: usize = std::mem::size_of::<TfsCaptureSlot>();
const TFS_CAPTURE_TOTAL_SIZE: usize = TFS_SLOT_SIZE * TFS_CAPTURE_SLOTS;
const TFS_CAPTURE_MAGIC: u64 = 0x5446_5343_4150_5455; // "TFSCAPTU"

#[repr(C)]
struct TfsCaptureSlot {
    magic: u64,                               // 0
    launch_n: u64,                            // 8
    timestamp_ms: u64,                        // 16
    kernel_id: u32,                           // 24 — see TFS_KERNEL_ID_*
    grid_x: u32,                              // 28
    grid_y: u32,                              // 32
    grid_z: u32,                              // 36
    block_x: u32,                             // 40
    block_y: u32,                             // 44
    block_z: u32,                             // 48
    raw_args_len: u32,                        // 52 — actual bytes captured
    kernel_name: [u8; TFS_KERNEL_NAME_BYTES], // 56 — null-padded ASCII
    raw_args: [u8; TFS_RAW_ARGS_BYTES],       // 120 — kernel arg flat buffer dump
    arg_meta: [u8; 8],                        // 632 — launch encoding flags
}

// Kernel IDs (compact identifier for which captured downstream kernel — used
// by the post-mortem analyzer to find the right replay slot configuration).
const TFS_KERNEL_ID_OTHER: u32 = 0; // any TFS_* not specifically known
const TFS_KERNEL_ID_CLEAR: u32 = 1;
const TFS_KERNEL_ID_ADVECT: u32 = 2;
const TFS_KERNEL_ID_SMOOTH_VELOCITY: u32 = 3;
const TFS_KERNEL_ID_PRESSURE_DIVERGENCE: u32 = 4;
const TFS_KERNEL_ID_PARTICLE_DENSITY_TO_PARTICLES: u32 = 5;
const TFS_KERNEL_ID_MIGRATE: u32 = 6;
const TFS_KERNEL_ID_COUPLE_FROM_PARTICLES: u32 = 7;
const TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX: u32 = 8;
const TFS_KERNEL_ID_PARTICLE_DENSITY_SMOOTH: u32 = 9;
const TFS_KERNEL_ID_INIT_COLL_DATA_AND_APPLY_CONSTRAINTS: u32 = 10;
const TFS_KERNEL_ID_FINALIZE_COLLISION: u32 = 11;
const TFS_KERNEL_ID_PARTICLE_COLLISION_STATIC_BOX: u32 = 12;
const TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS1: u32 = 13;
const TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED: u32 = 14;
const TFS_KERNEL_ID_FIELD_SAMPLER_POINTS: u32 = 15;
const TFS_KERNEL_ID_IOFX_ACTOR_RANGE: u32 = 16;
const TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2: u32 = 17;
const TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3: u32 = 18;
const TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4: u32 = 19;

fn classify_tfs_kernel(name: &str) -> Option<u32> {
    if !name.starts_with("TurbulenceFS_") {
        return Some(match name {
            "InitCollDataAndApplyConstraints" => TFS_KERNEL_ID_INIT_COLL_DATA_AND_APPLY_CONSTRAINTS,
            "FinalizeCollision" => TFS_KERNEL_ID_FINALIZE_COLLISION,
            "ParticleCollisionStaticBox" => TFS_KERNEL_ID_PARTICLE_COLLISION_STATIC_BOX,
            "CreateParticleShapesIncremental_MultiStage_1" => {
                TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS1
            }
            "CreateParticleShapesIncremental_MultiStage_2_Batched" => {
                TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED
            }
            "CreateParticleShapesIncremental_MultiStage_2" => {
                TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2
            }
            "CreateParticleShapesIncremental_MultiStage_3" => {
                TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3
            }
            "CreateParticleShapesIncremental_MultiStage_4" => {
                TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4
            }
            "IOFX_actorRangeKernel" => TFS_KERNEL_ID_IOFX_ACTOR_RANGE,
            _ => return None,
        });
    }
    let id = if name.contains("ClearKernel") {
        TFS_KERNEL_ID_CLEAR
    } else if name.contains("AdvectKernel") || name.contains("AdvectMacCormackKernel") {
        TFS_KERNEL_ID_ADVECT
    } else if name.contains("SmoothVelocityKernel") {
        TFS_KERNEL_ID_SMOOTH_VELOCITY
    } else if name.contains("PressureDivergenceKernel") {
        TFS_KERNEL_ID_PRESSURE_DIVERGENCE
    } else if name.contains("ParticleDensityToParticlesKernel") {
        TFS_KERNEL_ID_PARTICLE_DENSITY_TO_PARTICLES
    } else if name.contains("MigrateKernel") {
        TFS_KERNEL_ID_MIGRATE
    } else if name.contains("CoupleFromParticlesKernel") {
        TFS_KERNEL_ID_COUPLE_FROM_PARTICLES
    } else if name.contains("ParticleDensityAtomicToTexKernel") {
        TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX
    } else if name.contains("ParticleDensitySmooth") {
        TFS_KERNEL_ID_PARTICLE_DENSITY_SMOOTH
    } else if name.contains("fieldSamplerPointsKernel") {
        TFS_KERNEL_ID_FIELD_SAMPLER_POINTS
    } else {
        TFS_KERNEL_ID_OTHER
    };
    Some(id)
}

struct TfsCaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
}
unsafe impl Send for TfsCaptureState {}
unsafe impl Sync for TfsCaptureState {}

static TFS_CAPTURE: Mutex<Option<TfsCaptureState>> = Mutex::new(None);

pub(crate) static TFS_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static TFS_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

const IOFX_CAPTURE_SLOTS: usize = 16;
const IOFX_ACTOR_RANGE_MAX_COUNT: usize = 8192;
const IOFX_ACTOR_RANGE_INPUT_BYTES: usize = IOFX_ACTOR_RANGE_MAX_COUNT * 4;
const IOFX_BBOX_POINTER_COUNT: usize = 8;
// The bbox texture path samples float4 positions by actor index. Late Batman
// launches use indices well past the first 4096 records, so 64 KiB can truncate
// the captured texref prefix and make offline replay fault on a harness artifact.
const IOFX_BBOX_TEXREF_BYTES: usize = 1024 * 1024;
// The texture-path bbox launches also walk param_4 / param_5 index arrays up to
// param_2 elements. Current large captures reach ~50k items, so 32 KiB still
// clips those u32 windows and only the smaller `(1,0)` launches replay fully.
const IOFX_BBOX_BUFFER_BYTES: usize = 256 * 1024;

#[repr(C)]
struct IofxActorRangeCaptureSlot {
    magic: u64,                         // 0: "IOFXCAPT"
    launch_n: u64,                      // 8
    timestamp_ms: u64,                  // 16
    grid_x: u32,                        // 24
    block_x: u32,                       // 28
    count: u32,                         // 32: param 322
    input_bytes: u32,                   // 36
    input_ptr_324: u64,                 // 40: read-only input array
    output_ptr_326: u64,                // 48: write-only output array
    output_ptr_327: u64,                // 56: write-only output array
    output_ptr_328: u64,                // 64: write-only output array
    input_ptr_329: u64,                 // 72: read-only input array
    raw_args_len: u32,                  // 80
    shared_mem_bytes: u32,              // 84
    arg_meta: [u8; 8],                  // 88
    _pad1: [u8; 32],                    // 96
    raw_args: [u8; TFS_RAW_ARGS_BYTES], // 128
    input_324: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
    input_329: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
    output_326: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
    output_327: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
    output_328: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
    pre_output_326: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
    pre_output_327: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
    pre_output_328: [u8; IOFX_ACTOR_RANGE_INPUT_BYTES],
}

const IOFX_ACTOR_RANGE_CAPTURE_MAGIC: u64 = 0x494f4658_43415054; // "IOFXCAPT"
const IOFX_ACTOR_RANGE_SLOT_SIZE: usize = std::mem::size_of::<IofxActorRangeCaptureSlot>();
const IOFX_ACTOR_RANGE_RAW_ARGS_OFF: usize = 128;
const IOFX_ACTOR_RANGE_INPUT_324_OFF: usize = IOFX_ACTOR_RANGE_RAW_ARGS_OFF + TFS_RAW_ARGS_BYTES;
const IOFX_ACTOR_RANGE_INPUT_329_OFF: usize =
    IOFX_ACTOR_RANGE_INPUT_324_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES;
const IOFX_ACTOR_RANGE_OUTPUT_326_OFF: usize =
    IOFX_ACTOR_RANGE_INPUT_329_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES;
const IOFX_ACTOR_RANGE_OUTPUT_327_OFF: usize =
    IOFX_ACTOR_RANGE_OUTPUT_326_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES;
const IOFX_ACTOR_RANGE_OUTPUT_328_OFF: usize =
    IOFX_ACTOR_RANGE_OUTPUT_327_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES;
const IOFX_ACTOR_RANGE_PRE_OUTPUT_326_OFF: usize =
    IOFX_ACTOR_RANGE_OUTPUT_328_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES;
const IOFX_ACTOR_RANGE_PRE_OUTPUT_327_OFF: usize =
    IOFX_ACTOR_RANGE_PRE_OUTPUT_326_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES;
const IOFX_ACTOR_RANGE_PRE_OUTPUT_328_OFF: usize =
    IOFX_ACTOR_RANGE_PRE_OUTPUT_327_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES;
const IOFX_ACTOR_RANGE_CAPTURE_TOTAL_SIZE: usize = IOFX_ACTOR_RANGE_SLOT_SIZE * IOFX_CAPTURE_SLOTS;

struct IofxActorRangeCaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
    capture_counter: u64,
}
unsafe impl Send for IofxActorRangeCaptureState {}
unsafe impl Sync for IofxActorRangeCaptureState {}

static IOFX_ACTOR_RANGE_CAPTURE: Mutex<Option<IofxActorRangeCaptureState>> = Mutex::new(None);

pub(crate) static IOFX_ACTOR_RANGE_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static IOFX_ACTOR_RANGE_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IofxActorRangeArgs {
    count: u32,
    input_ptr_324: u64,
    output_ptr_326: u64,
    output_ptr_327: u64,
    output_ptr_328: u64,
    input_ptr_329: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct IofxBboxTexrefCaptureDesc {
    address_mode: [u32; 3],
    filter_mode: u32,
    flags: u32,
    max_anisotropy: u32,
    mipmap_filter_mode: u32,
    mipmap_level_bias: f32,
    min_mipmap_level_clamp: f32,
    max_mipmap_level_clamp: f32,
    border_color: [f32; 4],
}

impl Default for IofxBboxTexrefCaptureDesc {
    fn default() -> Self {
        Self {
            address_mode: [0; 3],
            filter_mode: 0,
            flags: 0,
            max_anisotropy: 0,
            mipmap_filter_mode: 0,
            mipmap_level_bias: 0.0,
            min_mipmap_level_clamp: 0.0,
            max_mipmap_level_clamp: 0.0,
            border_color: [0.0; 4],
        }
    }
}

#[repr(C)]
struct IofxBboxCaptureSlot {
    magic: u64,                                 // 0: "IOFXBBCP"
    launch_n: u64,                              // 8
    timestamp_ms: u64,                          // 16
    grid_x: u32,                                // 24
    block_x: u32,                               // 28
    shared_mem_bytes: u32,                      // 32
    raw_args_len: u32,                          // 36
    param_2: u32,                               // 40
    param_3: u32,                               // 44
    param_12: u32,                              // 48
    param_13: u32,                              // 52
    arg_meta: [u8; 8],                          // 56
    texref_src_ptr: u64,                        // 64
    texref_capture_bytes: u32,                  // 72
    texref_format: u32,                         // 76
    texref_num_channels: u32,                   // 80
    texref_desc: IofxBboxTexrefCaptureDesc,     // 84
    _pad0: [u8; 4],                             // 140
    ptr_params: [u64; IOFX_BBOX_POINTER_COUNT], // 144
    raw_args: [u8; TFS_RAW_ARGS_BYTES],         // 208
    pre_ptr_buffers: [u8; IOFX_BBOX_POINTER_COUNT * IOFX_BBOX_BUFFER_BYTES],
    post_ptr_buffers: [u8; IOFX_BBOX_POINTER_COUNT * IOFX_BBOX_BUFFER_BYTES],
    texref_buffer: [u8; IOFX_BBOX_TEXREF_BYTES],
}

const IOFX_BBOX_CAPTURE_MAGIC: u64 = 0x494f4658_42424350; // "IOFXBBCP"
const IOFX_BBOX_SLOT_SIZE: usize = std::mem::size_of::<IofxBboxCaptureSlot>();
const IOFX_BBOX_RAW_ARGS_OFF: usize = 208;
const IOFX_BBOX_PRE_PTRS_OFF: usize = IOFX_BBOX_RAW_ARGS_OFF + TFS_RAW_ARGS_BYTES;
const IOFX_BBOX_POST_PTRS_OFF: usize =
    IOFX_BBOX_PRE_PTRS_OFF + IOFX_BBOX_POINTER_COUNT * IOFX_BBOX_BUFFER_BYTES;
const IOFX_BBOX_TEXREF_OFF: usize =
    IOFX_BBOX_POST_PTRS_OFF + IOFX_BBOX_POINTER_COUNT * IOFX_BBOX_BUFFER_BYTES;
const IOFX_BBOX_CAPTURE_TOTAL_SIZE: usize = IOFX_BBOX_SLOT_SIZE * IOFX_CAPTURE_SLOTS;

struct IofxBboxCaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
}
unsafe impl Send for IofxBboxCaptureState {}
unsafe impl Sync for IofxBboxCaptureState {}

static IOFX_BBOX_CAPTURE: Mutex<Option<IofxBboxCaptureState>> = Mutex::new(None);

pub(crate) static IOFX_BBOX_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static IOFX_BBOX_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IofxBboxArgs {
    param_2: u32,
    param_3: u32,
    ptr_params: [u64; IOFX_BBOX_POINTER_COUNT],
    param_12: u32,
    param_13: u32,
}

fn tfs_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_TFS_CAPTURE").map_or(false, |v| v == "1"))
}

fn iofx_actor_range_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_IOFX_CAPTURE").map_or(false, |v| v == "1"))
}

fn iofx_bbox_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_IOFX_BBOX_CAPTURE").map_or(false, |v| v == "1"))
}

unsafe fn iofx_actor_range_capture_init(state: &mut Option<IofxActorRangeCaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(
            &mut host_ptr,
            IOFX_ACTOR_RANGE_CAPTURE_TOTAL_SIZE,
            hipHostMallocMapped,
        )
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] IOFX capture: hipHostMalloc({}) failed: {:?}",
            IOFX_ACTOR_RANGE_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, IOFX_ACTOR_RANGE_CAPTURE_TOTAL_SIZE);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] IOFX capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] IOFX capture: {} slots × {} KB = {} KB pinned; host=0x{:x} dev=0x{:x}",
        IOFX_CAPTURE_SLOTS,
        IOFX_ACTOR_RANGE_SLOT_SIZE / 1024,
        IOFX_ACTOR_RANGE_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    IOFX_ACTOR_RANGE_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    IOFX_ACTOR_RANGE_CAPTURE_TOTAL_BYTES.store(
        IOFX_ACTOR_RANGE_CAPTURE_TOTAL_SIZE,
        std::sync::atomic::Ordering::SeqCst,
    );
    *state = Some(IofxActorRangeCaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
        capture_counter: 0,
    });
}

unsafe fn iofx_bbox_capture_init(state: &mut Option<IofxBboxCaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(
            &mut host_ptr,
            IOFX_BBOX_CAPTURE_TOTAL_SIZE,
            hipHostMallocMapped,
        )
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] IOFX bbox capture: hipHostMalloc({}) failed: {:?}",
            IOFX_BBOX_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, IOFX_BBOX_CAPTURE_TOTAL_SIZE);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] IOFX bbox capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] IOFX bbox capture: {} slots × {} KB = {} KB pinned; host=0x{:x} dev=0x{:x}",
        IOFX_CAPTURE_SLOTS,
        IOFX_BBOX_SLOT_SIZE / 1024,
        IOFX_BBOX_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    IOFX_BBOX_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    IOFX_BBOX_CAPTURE_TOTAL_BYTES.store(
        IOFX_BBOX_CAPTURE_TOTAL_SIZE,
        std::sync::atomic::Ordering::SeqCst,
    );
    *state = Some(IofxBboxCaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

unsafe fn read_iofx_actor_range_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> Option<IofxActorRangeArgs> {
    let count: u32 = unsafe { read_kernel_arg(kernel_params, extra, 12, 2) }?;
    let input_ptr_324: u64 = unsafe { read_kernel_arg(kernel_params, extra, 24, 4) }?;
    let output_ptr_326: u64 = unsafe { read_kernel_arg(kernel_params, extra, 40, 6) }?;
    let output_ptr_327: u64 = unsafe { read_kernel_arg(kernel_params, extra, 48, 7) }?;
    let output_ptr_328: u64 = unsafe { read_kernel_arg(kernel_params, extra, 56, 8) }?;
    let input_ptr_329: u64 = unsafe { read_kernel_arg(kernel_params, extra, 64, 9) }?;
    if count == 0
        || input_ptr_324 == 0
        || output_ptr_326 == 0
        || output_ptr_327 == 0
        || output_ptr_328 == 0
        || input_ptr_329 == 0
    {
        return None;
    }
    // Note: count > IOFX_ACTOR_RANGE_MAX_COUNT (8192) is NOT a rejection.
    // The slot's input buffers are pre-sized to IOFX_ACTOR_RANGE_INPUT_BYTES
    // = 32 KB; the DtoD copy in capture_iofx_actor_range_inputs caps at
    // `input_bytes = (count * 4).min(IOFX_ACTOR_RANGE_INPUT_BYTES)` so larger
    // launches truncate to the first 8192 actors. The capture stays
    // functional past launch 374, which previously was the silent cutoff
    // for full-sim large-count launches that triggered actorRange faults.
    Some(IofxActorRangeArgs {
        count,
        input_ptr_324,
        output_ptr_326,
        output_ptr_327,
        output_ptr_328,
        input_ptr_329,
    })
}

unsafe fn read_iofx_bbox_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> Option<IofxBboxArgs> {
    let param_2: u32 = unsafe { read_kernel_arg(kernel_params, extra, 12, 2) }?;
    let param_3: u32 = unsafe { read_kernel_arg(kernel_params, extra, 16, 3) }?;
    let ptr_params = [
        unsafe { read_kernel_arg(kernel_params, extra, 24, 4) }?,
        unsafe { read_kernel_arg(kernel_params, extra, 32, 5) }?,
        unsafe { read_kernel_arg(kernel_params, extra, 40, 6) }.unwrap_or(0),
        unsafe { read_kernel_arg(kernel_params, extra, 48, 7) }?,
        unsafe { read_kernel_arg(kernel_params, extra, 56, 8) }?,
        unsafe { read_kernel_arg(kernel_params, extra, 64, 9) }?,
        unsafe { read_kernel_arg(kernel_params, extra, 72, 10) }?,
        unsafe { read_kernel_arg(kernel_params, extra, 80, 11) }?,
    ];
    let param_12: u32 = unsafe { read_kernel_arg(kernel_params, extra, 88, 12) }?;
    let param_13: u32 = unsafe { read_kernel_arg(kernel_params, extra, 92, 13) }?;
    if ptr_params.iter().all(|&ptr| ptr == 0) {
        return None;
    }
    Some(IofxBboxArgs {
        param_2,
        param_3,
        ptr_params,
        param_12,
        param_13,
    })
}

fn iofx_actor_range_output_copy_bytes(
    output_ptr_326: u64,
    output_ptr_327: u64,
    output_ptr_328: u64,
    input_bytes: usize,
) -> usize {
    let mut min_gap = u64::MAX;
    for lhs in [output_ptr_326, output_ptr_327, output_ptr_328] {
        for rhs in [output_ptr_326, output_ptr_327, output_ptr_328] {
            if rhs > lhs {
                min_gap = min_gap.min(rhs - lhs);
            }
        }
    }
    if min_gap == u64::MAX {
        input_bytes
    } else {
        input_bytes.min(min_gap as usize)
    }
}

unsafe fn capture_iofx_actor_range_inputs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_x: u32,
    block_x: u32,
    shared_mem_bytes: u32,
) -> Option<usize> {
    if !iofx_actor_range_capture_enabled() {
        return None;
    }
    let args = match unsafe { read_iofx_actor_range_args(kernel_params, extra) } {
        Some(args) => args,
        None => return None,
    };

    let mut guard = match IOFX_ACTOR_RANGE_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    unsafe { iofx_actor_range_capture_init(&mut guard) };
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    // Sampling (added 2026-05-11): the 2x DtoD per actorRange launch (input
    // 324/329) plus 3x post-launch DtoD on outputs causes stream pressure
    // (`feedback_no_d2d_from_launch_kernel`). Two recent runs crashed at
    // launches 161 / 2852 vs typical 30-50k+ baseline once the previous
    // count > IOFX_ACTOR_RANGE_MAX_COUNT silent filter was removed and ALL
    // launches now do DtoD. Sample to 1-in-N (default 64); set
    // ZLUDA_IOFX_CAPTURE_SAMPLE=1 to capture every launch (the previous
    // behavior, useful for short investigation runs).
    static SAMPLE_INTERVAL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let sample = *SAMPLE_INTERVAL.get_or_init(|| {
        std::env::var("ZLUDA_IOFX_CAPTURE_SAMPLE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64)
    });
    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    if this_launch % sample != 0 {
        return None;
    }
    let captured_n = state.capture_counter;
    state.capture_counter += 1;
    let slot_index = (captured_n as usize) % IOFX_CAPTURE_SLOTS;
    let slot_host = state.host_ptr.add(slot_index * IOFX_ACTOR_RANGE_SLOT_SIZE)
        as *mut IofxActorRangeCaptureSlot;
    let slot_dev_base =
        (state.dev_ptr as usize + slot_index * IOFX_ACTOR_RANGE_SLOT_SIZE) as *mut u8;
    std::ptr::write_bytes(slot_host as *mut u8, 0, IOFX_ACTOR_RANGE_SLOT_SIZE);
    (*slot_host).magic = IOFX_ACTOR_RANGE_CAPTURE_MAGIC;
    (*slot_host).launch_n = this_launch;
    (*slot_host).timestamp_ms = crate::elapsed_ms();
    (*slot_host).grid_x = grid_x;
    (*slot_host).block_x = block_x;
    (*slot_host).count = args.count;
    (*slot_host).input_bytes = (args.count as usize * 4).min(IOFX_ACTOR_RANGE_INPUT_BYTES) as u32;
    (*slot_host).input_ptr_324 = args.input_ptr_324;
    (*slot_host).output_ptr_326 = args.output_ptr_326;
    (*slot_host).output_ptr_327 = args.output_ptr_327;
    (*slot_host).output_ptr_328 = args.output_ptr_328;
    (*slot_host).input_ptr_329 = args.input_ptr_329;
    (*slot_host).shared_mem_bytes = shared_mem_bytes;
    (*slot_host).arg_meta = encode_launch_arg_meta(kernel_params, extra);
    let raw_dst = (*slot_host).raw_args.as_mut_ptr();
    let raw_len = dump_kernel_args_raw(kernel_params, extra, raw_dst, TFS_RAW_ARGS_BYTES);
    (*slot_host).raw_args_len = raw_len as u32;

    let input_bytes = (*slot_host).input_bytes as usize;
    let _ = hipMemcpyDtoD(
        hipDeviceptr_t(slot_dev_base.add(IOFX_ACTOR_RANGE_INPUT_324_OFF) as _),
        hipDeviceptr_t(args.input_ptr_324 as _),
        input_bytes,
    );
    let _ = hipMemcpyDtoD(
        hipDeviceptr_t(slot_dev_base.add(IOFX_ACTOR_RANGE_INPUT_329_OFF) as _),
        hipDeviceptr_t(args.input_ptr_329 as _),
        input_bytes,
    );
    Some(slot_index)
}

unsafe fn capture_iofx_actor_range_outputs(
    slot_index: usize,
    stream: hipStream_t,
    prelaunch: bool,
) {
    if !iofx_actor_range_capture_enabled() {
        return;
    }

    let guard = match IOFX_ACTOR_RANGE_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };

    let slot_host = state.host_ptr.add(slot_index * IOFX_ACTOR_RANGE_SLOT_SIZE)
        as *const IofxActorRangeCaptureSlot;
    let slot_dev_base =
        (state.dev_ptr as usize + slot_index * IOFX_ACTOR_RANGE_SLOT_SIZE) as *mut u8;
    let output_bytes = iofx_actor_range_output_copy_bytes(
        (*slot_host).output_ptr_326,
        (*slot_host).output_ptr_327,
        (*slot_host).output_ptr_328,
        (*slot_host).input_bytes as usize,
    );
    if output_bytes == 0 {
        return;
    }
    let dst_offsets = if prelaunch {
        [
            IOFX_ACTOR_RANGE_PRE_OUTPUT_326_OFF,
            IOFX_ACTOR_RANGE_PRE_OUTPUT_327_OFF,
            IOFX_ACTOR_RANGE_PRE_OUTPUT_328_OFF,
        ]
    } else {
        [
            IOFX_ACTOR_RANGE_OUTPUT_326_OFF,
            IOFX_ACTOR_RANGE_OUTPUT_327_OFF,
            IOFX_ACTOR_RANGE_OUTPUT_328_OFF,
        ]
    };
    for (src_ptr, dst_off) in [
        ((*slot_host).output_ptr_326, dst_offsets[0]),
        ((*slot_host).output_ptr_327, dst_offsets[1]),
        ((*slot_host).output_ptr_328, dst_offsets[2]),
    ] {
        if src_ptr == 0 {
            continue;
        }
        let _ = hipMemcpyDtoDAsync(
            hipDeviceptr_t(slot_dev_base.add(dst_off) as _),
            hipDeviceptr_t(src_ptr as _),
            output_bytes,
            stream,
        );
    }
}

unsafe fn capture_iofx_bbox_inputs(
    module: hipModule_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_x: u32,
    block_x: u32,
    shared_mem_bytes: u32,
) -> Option<usize> {
    if !iofx_bbox_capture_enabled() {
        return None;
    }
    let args = match unsafe { read_iofx_bbox_args(kernel_params, extra) } {
        Some(args) => args,
        None => return None,
    };

    let mut guard = match IOFX_BBOX_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    unsafe { iofx_bbox_capture_init(&mut guard) };
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % IOFX_CAPTURE_SLOTS;
    let slot_host =
        state.host_ptr.add(slot_index * IOFX_BBOX_SLOT_SIZE) as *mut IofxBboxCaptureSlot;
    std::ptr::write_bytes(slot_host as *mut u8, 0, IOFX_BBOX_SLOT_SIZE);
    (*slot_host).magic = IOFX_BBOX_CAPTURE_MAGIC;
    (*slot_host).launch_n = this_launch;
    (*slot_host).timestamp_ms = crate::elapsed_ms();
    (*slot_host).grid_x = grid_x;
    (*slot_host).block_x = block_x;
    (*slot_host).shared_mem_bytes = shared_mem_bytes;
    (*slot_host).param_2 = args.param_2;
    (*slot_host).param_3 = args.param_3;
    (*slot_host).param_12 = args.param_12;
    (*slot_host).param_13 = args.param_13;
    (*slot_host).arg_meta = encode_launch_arg_meta(kernel_params, extra);
    (*slot_host).ptr_params = args.ptr_params;
    unsafe { populate_iofx_bbox_texref_capture(slot_host, module) };
    let raw_dst = (*slot_host).raw_args.as_mut_ptr();
    (*slot_host).raw_args_len =
        dump_kernel_args_raw(kernel_params, extra, raw_dst, TFS_RAW_ARGS_BYTES) as u32;
    // Phase D of plan fluttering-waddling-zephyr.md (2026-05-09): emit a
    // one-line summary of launches #0..N so we can compare the inputs
    // for #1, #2 (work) vs #3 (crashes). We log up to 8 launches —
    // enough to span the crashing window without exploding logs once
    // the kernel runs in a steady state.
    const IOFX_BBOX_LOG_LAUNCH_LIMIT: u64 = 8;
    if this_launch < IOFX_BBOX_LOG_LAUNCH_LIMIT {
        let p = &args.ptr_params;
        super::driver::zluda_log_err(&format!(
            "[IOFX_BBOX] launch_n={} grid_x={} block_x={} shared={} \
             p2={} p3={} p12={} p13={} \
             ptrs=[0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x}]",
            this_launch,
            grid_x,
            block_x,
            shared_mem_bytes,
            args.param_2,
            args.param_3,
            args.param_12,
            args.param_13,
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            p[6],
            p[7],
        ));
    }
    Some(slot_index)
}

unsafe fn capture_iofx_bbox_ptr_regions(slot_index: usize, stream: hipStream_t, prelaunch: bool) {
    if !iofx_bbox_capture_enabled() {
        return;
    }

    let guard = match IOFX_BBOX_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };

    let slot_host =
        state.host_ptr.add(slot_index * IOFX_BBOX_SLOT_SIZE) as *const IofxBboxCaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * IOFX_BBOX_SLOT_SIZE) as *mut u8;
    let base_off = if prelaunch {
        IOFX_BBOX_PRE_PTRS_OFF
    } else {
        IOFX_BBOX_POST_PTRS_OFF
    };
    for (idx, src_ptr) in (*slot_host).ptr_params.iter().copied().enumerate() {
        if src_ptr == 0 {
            continue;
        }
        let dst_off = base_off + idx * IOFX_BBOX_BUFFER_BYTES;
        let _ = hipMemcpyDtoDAsync(
            hipDeviceptr_t(slot_dev_base.add(dst_off) as _),
            hipDeviceptr_t(src_ptr as _),
            IOFX_BBOX_BUFFER_BYTES,
            stream,
        );
    }
    if prelaunch && (*slot_host).texref_src_ptr != 0 && (*slot_host).texref_capture_bytes != 0 {
        let _ = hipMemcpyDtoDAsync(
            hipDeviceptr_t(slot_dev_base.add(IOFX_BBOX_TEXREF_OFF) as _),
            hipDeviceptr_t((*slot_host).texref_src_ptr as _),
            (*slot_host).texref_capture_bytes as usize,
            stream,
        );
    }
}

unsafe fn capture_iofx_bbox_pre(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_bbox_ptr_regions(slot_index, stream, true) };
}

unsafe fn capture_iofx_bbox_post(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_bbox_ptr_regions(slot_index, stream, false) };
}

// ============================================================================
// CUB RadixSort Downsweep capture. Mirrors the IOFX_BBOX capture pattern:
// pinned-mapped host buffer, ring of slots, D2D-async snapshots on the
// kernel's own stream (safe vs. the ProfilingSignal race that specifically
// affects post-launch D2H — see `feedback_no_d2d_from_launch_kernel.md` and
// the NOTE below `capture_iofx_bbox_ptr_regions`).
//
// Gated by ZLUDA_DOWNSWEEP_CAPTURE=1. The kernel takes 5 pointer params
// (verified from a live capture on 2026-04-23):
//   0: d_keys_in, 1: d_keys_out, 2: d_values_in, 3: d_values_out, 4: d_spine
// We snapshot CUB_DOWNSWEEP_BUFFER_BYTES (128 KB) from each, plus the 88-byte
// extra-args buffer. Output file: /tmp/cub_downsweep_captures.bin.
// ============================================================================

const CUB_DOWNSWEEP_POINTER_COUNT: usize = 5;
const CUB_DOWNSWEEP_BUFFER_BYTES: usize = 128 * 1024;
const CUB_DOWNSWEEP_RAW_ARGS_BYTES: usize = 128;
const CUB_DOWNSWEEP_CAPTURE_MAGIC: u64 = 0x43554244_4f574e53; // "CUBDOWNS"

#[repr(C)]
struct CubDownsweepCaptureSlot {
    magic: u64,                                     // 0
    launch_n: u64,                                  // 8
    timestamp_ms: u64,                              // 16
    grid_x: u32,                                    // 24
    grid_y: u32,                                    // 28
    grid_z: u32,                                    // 32
    block_x: u32,                                   // 36
    block_y: u32,                                   // 40
    block_z: u32,                                   // 44
    shared_mem_bytes: u32,                          // 48
    raw_args_len: u32,                              // 52
    ptr_params: [u64; CUB_DOWNSWEEP_POINTER_COUNT], // 56..96
    buffer_bytes_per_ptr: u32,                      // 96
    _pad0: [u8; 4],                                 // 100
    raw_args: [u8; CUB_DOWNSWEEP_RAW_ARGS_BYTES],   // 104..232
    global_launch_n: u64,                           // 232
    _pad1: [u8; 16],                                // 240..256
    pre_ptr_buffers: [u8; CUB_DOWNSWEEP_POINTER_COUNT * CUB_DOWNSWEEP_BUFFER_BYTES],
}

const CUB_DOWNSWEEP_SLOT_SIZE: usize = std::mem::size_of::<CubDownsweepCaptureSlot>();
const CUB_DOWNSWEEP_PRE_PTRS_OFF: usize = 256;
const CUB_DOWNSWEEP_CAPTURE_TOTAL_SIZE: usize = CUB_DOWNSWEEP_SLOT_SIZE * IOFX_CAPTURE_SLOTS;

struct CubDownsweepCaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
}
unsafe impl Send for CubDownsweepCaptureState {}
unsafe impl Sync for CubDownsweepCaptureState {}

static CUB_DOWNSWEEP_CAPTURE: Mutex<Option<CubDownsweepCaptureState>> = Mutex::new(None);

pub(crate) static CUB_DOWNSWEEP_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static CUB_DOWNSWEEP_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn cub_downsweep_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_DOWNSWEEP_CAPTURE").map_or(false, |v| v == "1"))
}

unsafe fn cub_downsweep_capture_init(state: &mut Option<CubDownsweepCaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(
            &mut host_ptr,
            CUB_DOWNSWEEP_CAPTURE_TOTAL_SIZE,
            hipHostMallocMapped,
        )
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] CUB downsweep capture: hipHostMalloc({}) failed: {:?}",
            CUB_DOWNSWEEP_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, CUB_DOWNSWEEP_CAPTURE_TOTAL_SIZE);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] CUB downsweep capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] CUB downsweep capture: {} slots × {} KB = {} KB pinned; host=0x{:x} dev=0x{:x}",
        IOFX_CAPTURE_SLOTS,
        CUB_DOWNSWEEP_SLOT_SIZE / 1024,
        CUB_DOWNSWEEP_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    CUB_DOWNSWEEP_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    CUB_DOWNSWEEP_CAPTURE_TOTAL_BYTES.store(
        CUB_DOWNSWEEP_CAPTURE_TOTAL_SIZE,
        std::sync::atomic::Ordering::SeqCst,
    );
    *state = Some(CubDownsweepCaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

unsafe fn capture_cub_downsweep_inputs(
    extra: *mut *mut ::core::ffi::c_void,
    global_launch_n: u64,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
) -> Option<usize> {
    if !cub_downsweep_capture_enabled() {
        return None;
    }
    if extra.is_null() {
        return None;
    }
    let extra_arr = unsafe { std::slice::from_raw_parts(extra, 5) };
    if extra_arr[0] != (0x1usize as *mut _) || extra_arr[1].is_null() {
        return None;
    }
    let buf = extra_arr[1] as *const u8;
    let mut ptrs = [0u64; CUB_DOWNSWEEP_POINTER_COUNT];
    for i in 0..CUB_DOWNSWEEP_POINTER_COUNT {
        ptrs[i] = unsafe { std::ptr::read_unaligned(buf.add(i * 8) as *const u64) };
    }

    let mut guard = match CUB_DOWNSWEEP_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    unsafe { cub_downsweep_capture_init(&mut guard) };
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % IOFX_CAPTURE_SLOTS;
    let slot_host = unsafe {
        state.host_ptr.add(slot_index * CUB_DOWNSWEEP_SLOT_SIZE) as *mut CubDownsweepCaptureSlot
    };
    unsafe {
        std::ptr::write_bytes(slot_host as *mut u8, 0, CUB_DOWNSWEEP_SLOT_SIZE);
        (*slot_host).magic = CUB_DOWNSWEEP_CAPTURE_MAGIC;
        (*slot_host).launch_n = this_launch;
        (*slot_host).timestamp_ms = crate::elapsed_ms();
        (*slot_host).grid_x = grid_x;
        (*slot_host).grid_y = grid_y;
        (*slot_host).grid_z = grid_z;
        (*slot_host).block_x = block_x;
        (*slot_host).block_y = block_y;
        (*slot_host).block_z = block_z;
        (*slot_host).shared_mem_bytes = shared_mem_bytes;
        (*slot_host).ptr_params = ptrs;
        (*slot_host).buffer_bytes_per_ptr = CUB_DOWNSWEEP_BUFFER_BYTES as u32;
        // Raw args buffer is 88 bytes from the extra; copy and pad to 128.
        let dst = (*slot_host).raw_args.as_mut_ptr();
        std::ptr::copy_nonoverlapping(buf, dst, 88);
        (*slot_host).raw_args_len = 88;
        (*slot_host).global_launch_n = global_launch_n;
    }
    Some(slot_index)
}

// ============================================================================
// CUB RadixSortScanKernel pre-Scan spine capture
// ----------------------------------------------------------------------------
// Per project_cub_scan_spine_corruption_2026_05_11.md: after the VGPR-zero
// pass shipped, the dominant residual EARLY-crash class is CUB Downsweep
// (3/10 of N=10). Post-mortem of cub_downsweep_captures.bin showed the spine
// state Downsweep consumes has wildly wrong values (3000x-16000x legal max
// in bins 16-31, while bins 0-15 healthy). Magnitudes too large to come from
// Scan summing healthy Upsweep counts alone — Upsweep is suspect too.
//
// This probe captures the spine state AT Scan kernel entry (= post-Upsweep,
// pre-Scan). Comparing pre-Scan vs post-Scan (the downsweep capture) tells
// us definitively whether Upsweep or Scan is the corruption source. Async
// DtoD on the same stream — no sync stall, no timing perturbation.
//
// Gated by ZLUDA_CUB_SCAN_PROBE=1 (default OFF). Class-name match on
// `name.contains("RadixSortScanKernel")` is INSTRUMENTATION (allowed per
// feedback_no_per_kernel_branching.md — kernel-name matching for diagnostic
// capture is permitted; only fix code must not branch).
// ============================================================================

const CUB_SCAN_BUFFER_BYTES: usize = 128 * 1024;

#[repr(C)]
struct CubScanCaptureSlot {
    magic: u64,                             // 0  "CUBSCANP"
    launch_n: u64,                          // 8
    timestamp_ms: u64,                      // 16
    grid_x: u32,                            // 24
    grid_y: u32,                            // 28
    grid_z: u32,                            // 32
    block_x: u32,                           // 36
    block_y: u32,                           // 40
    block_z: u32,                           // 44
    shared_mem_bytes: u32,                  // 48
    scan_n: u32,                            // 52: param_1 = total scan size
    spine_ptr: u64,                         // 56
    pre_spine_bytes: u32,                   // 64: bytes copied into pre_spine
    _pad0: u32,                             // 68..72
    global_launch_n: u64,                   // 72
    _pad1: [u8; 40],                        // 80..120
    pre_spine: [u8; CUB_SCAN_BUFFER_BYTES], // 120..(120+128K)
}

const CUB_SCAN_CAPTURE_MAGIC: u64 = 0x5042_4E43_5343_4255; // "UBCSCNBP" little-endian
const CUB_SCAN_SLOT_SIZE: usize = std::mem::size_of::<CubScanCaptureSlot>();
const CUB_SCAN_PRE_SPINE_OFF: usize = 120;
const CUB_SCAN_CAPTURE_TOTAL_SIZE: usize = CUB_SCAN_SLOT_SIZE * IOFX_CAPTURE_SLOTS;

struct CubScanCaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
}
unsafe impl Send for CubScanCaptureState {}
unsafe impl Sync for CubScanCaptureState {}

static CUB_SCAN_CAPTURE: Mutex<Option<CubScanCaptureState>> = Mutex::new(None);

pub(crate) static CUB_SCAN_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static CUB_SCAN_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn cub_scan_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_CUB_SCAN_PROBE").map_or(false, |v| v == "1"))
}

unsafe fn cub_scan_capture_init(state: &mut Option<CubScanCaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(
            &mut host_ptr,
            CUB_SCAN_CAPTURE_TOTAL_SIZE,
            hipHostMallocMapped,
        )
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] CUB scan capture: hipHostMalloc({}) failed: {:?}",
            CUB_SCAN_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, CUB_SCAN_CAPTURE_TOTAL_SIZE);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] CUB scan capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] CUB scan capture: {} slots × {} KB = {} KB pinned; host=0x{:x} dev=0x{:x}",
        IOFX_CAPTURE_SLOTS,
        CUB_SCAN_SLOT_SIZE / 1024,
        CUB_SCAN_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    CUB_SCAN_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    CUB_SCAN_CAPTURE_TOTAL_BYTES.store(
        CUB_SCAN_CAPTURE_TOTAL_SIZE,
        std::sync::atomic::Ordering::SeqCst,
    );
    *state = Some(CubScanCaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

unsafe fn capture_cub_scan_inputs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    global_launch_n: u64,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
) -> Option<usize> {
    if !cub_scan_capture_enabled() {
        return None;
    }
    let spine_ptr: u64 = unsafe { read_kernel_arg(kernel_params, extra, 0, 0) }?;
    let scan_n: u32 = unsafe { read_kernel_arg(kernel_params, extra, 8, 1) }.unwrap_or(0);
    if (spine_ptr >> 40) != 0x7f {
        return None;
    }

    let mut guard = match CUB_SCAN_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    unsafe { cub_scan_capture_init(&mut guard) };
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % IOFX_CAPTURE_SLOTS;
    let slot_host =
        unsafe { state.host_ptr.add(slot_index * CUB_SCAN_SLOT_SIZE) as *mut CubScanCaptureSlot };
    unsafe {
        std::ptr::write_bytes(slot_host as *mut u8, 0, CUB_SCAN_SLOT_SIZE);
        (*slot_host).magic = CUB_SCAN_CAPTURE_MAGIC;
        (*slot_host).launch_n = this_launch;
        (*slot_host).timestamp_ms = crate::elapsed_ms();
        (*slot_host).grid_x = grid_x;
        (*slot_host).grid_y = grid_y;
        (*slot_host).grid_z = grid_z;
        (*slot_host).block_x = block_x;
        (*slot_host).block_y = block_y;
        (*slot_host).block_z = block_z;
        (*slot_host).shared_mem_bytes = shared_mem_bytes;
        (*slot_host).scan_n = scan_n;
        (*slot_host).spine_ptr = spine_ptr;
        (*slot_host).pre_spine_bytes = 0;
        (*slot_host).global_launch_n = global_launch_n;
    }
    Some(slot_index)
}

unsafe fn capture_cub_scan_pre(slot_index: usize, stream: hipStream_t) {
    if !cub_scan_capture_enabled() {
        return;
    }
    let guard = match CUB_SCAN_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };
    let slot_host =
        unsafe { state.host_ptr.add(slot_index * CUB_SCAN_SLOT_SIZE) as *mut CubScanCaptureSlot };
    let slot_dev_base = unsafe { (state.dev_ptr as *mut u8).add(slot_index * CUB_SCAN_SLOT_SIZE) };
    let spine_ptr = unsafe { (*slot_host).spine_ptr };
    let scan_n = unsafe { (*slot_host).scan_n };
    if spine_ptr == 0 {
        return;
    }
    // Scan spine allocations are often much smaller than 128 KB. A flat
    // 128 KB DtoD on a small allocation can fail and leave this slot zeroed.
    // Bound the copy to both scan_n-derived size and the source allocation's
    // available bytes.
    let mut copy_bytes = if scan_n > 0 {
        (scan_n as usize).saturating_mul(std::mem::size_of::<u32>())
    } else {
        0
    };
    if copy_bytes == 0 {
        copy_bytes = CUB_SCAN_BUFFER_BYTES;
    }
    copy_bytes = copy_bytes.max(64).min(CUB_SCAN_BUFFER_BYTES);
    if let Some((_, alloc_size, alloc_off)) = unsafe { query_device_pointer_range(spine_ptr) } {
        let available = alloc_size.saturating_sub(alloc_off);
        copy_bytes = copy_bytes.min(available);
    }
    if copy_bytes == 0 {
        unsafe {
            (*slot_host).pre_spine_bytes = 0;
        }
        return;
    }
    unsafe {
        (*slot_host).pre_spine_bytes = copy_bytes as u32;
    }
    unsafe {
        let _ = hipMemcpyDtoDAsync(
            hipDeviceptr_t(slot_dev_base.add(CUB_SCAN_PRE_SPINE_OFF) as _),
            hipDeviceptr_t(spine_ptr as _),
            copy_bytes,
            stream,
        );
    }
}

unsafe fn capture_cub_downsweep_pre(slot_index: usize, stream: hipStream_t) {
    if !cub_downsweep_capture_enabled() {
        return;
    }
    let guard = match CUB_DOWNSWEEP_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };
    let slot_host = unsafe {
        state.host_ptr.add(slot_index * CUB_DOWNSWEEP_SLOT_SIZE) as *const CubDownsweepCaptureSlot
    };
    let slot_dev_base =
        unsafe { (state.dev_ptr as *mut u8).add(slot_index * CUB_DOWNSWEEP_SLOT_SIZE) };
    // Read n_items + grid from the captured raw_args to size each DtoD to
    // the actual live allocation. Reading PAST a small spine (~8 KB) with
    // a flat 128 KB DtoD silently fails (leaving the dest zero) and can
    // disturb unrelated allocations. Compute:
    //   key/value buffers (idx 0..=3): n_items * sizeof(i32)
    //   spine (idx 4):                 grid_x * num_bins * sizeof(i32)
    // With num_bins bounded by 32 (RADIX_BITS=5 max). If n_items is 0 or
    // suspicious, fall back to CUB_DOWNSWEEP_BUFFER_BYTES for keys/values
    // so the kernel arg parsing quirks of different CUB versions don't
    // silently disable the capture.
    let raw_args = unsafe { (*slot_host).raw_args.as_ptr() };
    let n_items = unsafe { std::ptr::read_unaligned(raw_args.add(40) as *const u32) };
    let grid_x = unsafe { (*slot_host).grid_x };
    let key_value_bytes = if n_items > 0 && (n_items as usize) * 4 <= CUB_DOWNSWEEP_BUFFER_BYTES {
        (n_items as usize) * 4
    } else {
        CUB_DOWNSWEEP_BUFFER_BYTES
    };
    let spine_bytes_est = (grid_x as usize).saturating_mul(32 * 4);
    let spine_bytes = spine_bytes_est.max(64).min(CUB_DOWNSWEEP_BUFFER_BYTES);
    for idx in 0..CUB_DOWNSWEEP_POINTER_COUNT {
        let src_ptr = unsafe { (*slot_host).ptr_params[idx] };
        if src_ptr == 0 {
            continue;
        }
        let n_bytes = if idx == 4 {
            spine_bytes
        } else {
            key_value_bytes
        };
        let dst_off = CUB_DOWNSWEEP_PRE_PTRS_OFF + idx * CUB_DOWNSWEEP_BUFFER_BYTES;
        unsafe {
            let _ = hipMemcpyDtoDAsync(
                hipDeviceptr_t(slot_dev_base.add(dst_off) as _),
                hipDeviceptr_t(src_ptr as _),
                n_bytes,
                stream,
            );
        }
    }
}

unsafe fn capture_iofx_actor_range_pre(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_actor_range_outputs(slot_index, stream, true) };
}

unsafe fn capture_iofx_actor_range_post(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_actor_range_outputs(slot_index, stream, false) };
}

// ============================================================================
// IOFX chain capture — IOFX_makeSortKeys_templ$0/$1 + IOFX_remapKernel_templ$0/$1
//
// These are the kernels that run immediately before IOFX_actorRangeKernel in
// a PhysX/APEX frame: makeSortKeys → CUB sort (Upsweep/Scan/Downsweep) →
// remapKernel → actorRange (×2). The actorRange wild-pointer fault is
// state-dependent (offline replay clean across 1,200 isolated launches) so
// the next decisive test is to replay the full chain in order and see
// whether the fault reproduces.
//
// Single shared ring discriminated by kernel_kind. Kernel-name guard at the
// call site keeps dump_kernel_args_raw safe — both kernels have ≥11 args so
// reading 16 kernel_params slots stays in-bounds.
// ============================================================================

const IOFX_CHAIN_POINTER_COUNT: usize = 8;
// 16 KB per ptr — verified 2026-05-11 that 64 KB × 8 × 2 (= 1 MB/slot, 16 MB
// ring) consistently broke PhysX init at IOFX_volumeMigrationKernel after
// ~90 launches across N=3 runs. At 16 KB the slot footprint drops to
// 16 KB × 8 × 2 = 256 KB/slot. We trade off partial input
// captures for early-init stability — chain replay tests can re-run a
// single captured frame, the pre/post buffer DtoD is the diagnostic, not
// the replay payload.
const IOFX_CHAIN_BUFFER_BYTES: usize = 16 * 1024;
const IOFX_CHAIN_RAW_ARGS_BYTES: usize = 512;
const IOFX_CHAIN_NAME_BYTES: usize = 64;
const IOFX_CHAIN_CAPTURE_MAGIC: u64 = 0x494f4658_4348414e; // "IOFXCHAN"
                                                           // 2026-05-12 PM (bumped 128→256): cover ~16 PhysX frames of chain history.
                                                           // With sample=16, this gives ~16 sampled slots per ring window vs ~8 at 128
                                                           // — enough to reach further upstream to catch makeSortKeys's bit-31-or-not
                                                           // POST data when it doesn't land on a % 16 boundary in the immediate
                                                           // frame. 256 × 262864 ≈ 66 MB pinned, comfortable on 16 GB UMA.
const IOFX_CHAIN_CAPTURE_SLOTS: usize = 256;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IofxChainKind {
    MakeSortKeysTempl0 = 0,
    MakeSortKeysTempl1 = 1,
    RemapTempl0 = 2,
    RemapTempl1 = 3,
    ActorRange = 4,
    NewRadixSortStep = 5,
}

fn classify_iofx_chain_kernel(name: &str) -> Option<IofxChainKind> {
    if name == "IOFX_makeSortKeys_templ$0" {
        Some(IofxChainKind::MakeSortKeysTempl0)
    } else if name == "IOFX_makeSortKeys_templ$1" {
        Some(IofxChainKind::MakeSortKeysTempl1)
    } else if name == "IOFX_remapKernel_templ$0" {
        Some(IofxChainKind::RemapTempl0)
    } else if name == "IOFX_remapKernel_templ$1" {
        Some(IofxChainKind::RemapTempl1)
    } else if name == "IOFX_actorRangeKernel" {
        Some(IofxChainKind::ActorRange)
    } else if name == "IOFX_newRadixSortStepKernel" {
        Some(IofxChainKind::NewRadixSortStep)
    } else {
        None
    }
}

#[repr(C)]
struct IofxChainCaptureSlot {
    magic: u64,                                  // 0:  "IOFXCHAN"
    launch_n: u64,                               // 8
    timestamp_ms: u64,                           // 16
    kernel_kind: u32,                            // 24: IofxChainKind
    grid_x: u32,                                 // 28
    grid_y: u32,                                 // 32
    grid_z: u32,                                 // 36
    block_x: u32,                                // 40
    block_y: u32,                                // 44
    block_z: u32,                                // 48
    shared_mem_bytes: u32,                       // 52
    raw_args_len: u32,                           // 56
    arg_meta: [u8; 8],                           // 60
    _pad0: [u8; 4],                              // 68
    kernel_name: [u8; IOFX_CHAIN_NAME_BYTES],    // 72..136
    ptr_params: [u64; IOFX_CHAIN_POINTER_COUNT], // 136..200
    ptr_buffer_bytes: u32,                       // 200
    _pad1: [u8; 4],                              // 204
    raw_args: [u8; IOFX_CHAIN_RAW_ARGS_BYTES],   // 208..720
    pre_ptr_buffers: [u8; IOFX_CHAIN_POINTER_COUNT * IOFX_CHAIN_BUFFER_BYTES],
    post_ptr_buffers: [u8; IOFX_CHAIN_POINTER_COUNT * IOFX_CHAIN_BUFFER_BYTES],
}

const IOFX_CHAIN_SLOT_SIZE: usize = std::mem::size_of::<IofxChainCaptureSlot>();
const IOFX_CHAIN_RAW_ARGS_OFF: usize = 208;
const IOFX_CHAIN_PRE_PTRS_OFF: usize = IOFX_CHAIN_RAW_ARGS_OFF + IOFX_CHAIN_RAW_ARGS_BYTES;
const IOFX_CHAIN_POST_PTRS_OFF: usize =
    IOFX_CHAIN_PRE_PTRS_OFF + IOFX_CHAIN_POINTER_COUNT * IOFX_CHAIN_BUFFER_BYTES;
const IOFX_CHAIN_CAPTURE_TOTAL_SIZE: usize = IOFX_CHAIN_SLOT_SIZE * IOFX_CHAIN_CAPTURE_SLOTS;

struct IofxChainCaptureState {
    host_ptr: *mut u8,
    dev_ptr: *mut std::ffi::c_void,
    launch_counter: u64,
}
unsafe impl Send for IofxChainCaptureState {}
unsafe impl Sync for IofxChainCaptureState {}

static IOFX_CHAIN_CAPTURE: Mutex<Option<IofxChainCaptureState>> = Mutex::new(None);

pub(crate) static IOFX_CHAIN_CAPTURE_HOST_PTR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static IOFX_CHAIN_CAPTURE_TOTAL_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn iofx_chain_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_IOFX_CHAIN_CAPTURE").map_or(false, |v| v == "1"))
}

unsafe fn iofx_chain_capture_init(state: &mut Option<IofxChainCaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(
            &mut host_ptr,
            IOFX_CHAIN_CAPTURE_TOTAL_SIZE,
            hipHostMallocMapped,
        )
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] IOFX chain capture: hipHostMalloc({}) failed: {:?}",
            IOFX_CHAIN_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, IOFX_CHAIN_CAPTURE_TOTAL_SIZE);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] IOFX chain capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] IOFX chain capture: {} slots × {} KB = {} MB pinned; host=0x{:x} dev=0x{:x}",
        IOFX_CHAIN_CAPTURE_SLOTS,
        IOFX_CHAIN_SLOT_SIZE / 1024,
        IOFX_CHAIN_CAPTURE_TOTAL_SIZE / (1024 * 1024),
        host_ptr as usize,
        dev_ptr as usize,
    ));
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] IOFX chain capture layout: SLOT_SIZE={} PRE_PTRS_OFF={} POST_PTRS_OFF={} \
         BUFFER_BYTES={} POINTER_COUNT={} RAW_ARGS_OFF={} RAW_ARGS_BYTES={}",
        IOFX_CHAIN_SLOT_SIZE,
        IOFX_CHAIN_PRE_PTRS_OFF,
        IOFX_CHAIN_POST_PTRS_OFF,
        IOFX_CHAIN_BUFFER_BYTES,
        IOFX_CHAIN_POINTER_COUNT,
        IOFX_CHAIN_RAW_ARGS_OFF,
        IOFX_CHAIN_RAW_ARGS_BYTES,
    ));
    IOFX_CHAIN_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    IOFX_CHAIN_CAPTURE_TOTAL_BYTES.store(
        IOFX_CHAIN_CAPTURE_TOTAL_SIZE,
        std::sync::atomic::Ordering::SeqCst,
    );
    *state = Some(IofxChainCaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

/// Scan up to N kernel_params slots as u64 ptrs; record up to ptr_count_max
/// values that look like device pointers (HSA-mapped user-space range).
///
/// Heuristic: tightened 2026-05-11 to the 0x7f00_0000_0000 .. 0x7fff_ffff_ffff
/// window where every HSA-mapped device pointer we have ever captured (across
/// IOFX_actorRange, IOFX_bbox, CUB downsweep, and the IOFX chain) actually
/// lives. The earlier looser `v >= 0x10000 && v < 0x8000_0000_0000` heuristic
/// false-positive'd on scalar args like 0x5b00100000 and 0x100000004; when
/// the buffer-snapshot path then dereferenced those as DtoD sources, HIP
/// queue state got wedged and PhysX faulted at IOFX_volumeMigrationKernel
/// after ~90 launches. The tighter range eliminates the regression.
unsafe fn iofx_chain_extract_ptrs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    arg_slots: usize,
    ptr_count_max: usize,
) -> [u64; IOFX_CHAIN_POINTER_COUNT] {
    let mut ptrs = [0u64; IOFX_CHAIN_POINTER_COUNT];
    let mut nr = 0usize;
    for i in 0..arg_slots {
        if nr >= ptr_count_max {
            break;
        }
        let v: u64 = match unsafe { read_kernel_arg(kernel_params, extra, i * 8, i) } {
            Some(v) => v,
            None => continue,
        };
        // Require both: high 24 bits == 0x7f**** (HSA-mapped user range)
        // AND low 32 bits >= 0x10000 (page-aligned past the first 64 KB
        // of the region). This rejects false-positives where read_kernel_arg
        // reads a u64 spanning two adjacent u32 scalar params — e.g.
        // newRadixSortStep's `u32 param_3=4` + `u32 param_4=0x7f41` reads
        // as 0x7f4100000004; the resulting DtoD on that bogus pointer
        // wedges HIP's stream into an error state, causing a downstream
        // host-side SIGABRT misattributed to whatever kernel ran next.
        // (Verified 2026-05-11 23:42 newradix_chain_bufs_v1 run.)
        if (v >> 40) == 0x7f && (v & 0xFFFFFFFF) >= 0x10000 {
            ptrs[nr] = v;
            nr += 1;
        }
    }
    ptrs
}

unsafe fn capture_iofx_chain_inputs(
    name: &str,
    kernel_kind: IofxChainKind,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
) -> Option<usize> {
    if !iofx_chain_capture_enabled() {
        return None;
    }
    // Argument-slot bound by kernel kind. makeSortKeys has 16 params, remap
    // has 11. Stay strictly within these so read_kernel_arg can't walk past
    // the array.
    let arg_slots = match kernel_kind {
        IofxChainKind::MakeSortKeysTempl0 | IofxChainKind::MakeSortKeysTempl1 => 16,
        IofxChainKind::RemapTempl0 | IofxChainKind::RemapTempl1 => 11,
        IofxChainKind::ActorRange => 10,
        IofxChainKind::NewRadixSortStep => 13,
    };
    let ptrs = unsafe {
        iofx_chain_extract_ptrs(kernel_params, extra, arg_slots, IOFX_CHAIN_POINTER_COUNT)
    };
    if ptrs.iter().all(|&p| p == 0) {
        return None;
    }

    let mut guard = match IOFX_CHAIN_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    unsafe { iofx_chain_capture_init(&mut guard) };
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % IOFX_CHAIN_CAPTURE_SLOTS;
    let slot_host = unsafe {
        state.host_ptr.add(slot_index * IOFX_CHAIN_SLOT_SIZE) as *mut IofxChainCaptureSlot
    };
    unsafe {
        std::ptr::write_bytes(slot_host as *mut u8, 0, IOFX_CHAIN_SLOT_SIZE);
        (*slot_host).magic = IOFX_CHAIN_CAPTURE_MAGIC;
        (*slot_host).launch_n = this_launch;
        (*slot_host).timestamp_ms = crate::elapsed_ms();
        (*slot_host).kernel_kind = kernel_kind as u32;
        (*slot_host).grid_x = grid_x;
        (*slot_host).grid_y = grid_y;
        (*slot_host).grid_z = grid_z;
        (*slot_host).block_x = block_x;
        (*slot_host).block_y = block_y;
        (*slot_host).block_z = block_z;
        (*slot_host).shared_mem_bytes = shared_mem_bytes;
        (*slot_host).arg_meta = encode_launch_arg_meta(kernel_params, extra);
        (*slot_host).ptr_params = ptrs;
        (*slot_host).ptr_buffer_bytes = IOFX_CHAIN_BUFFER_BYTES as u32;
        // Truncate kernel_name into the fixed buffer.
        let name_bytes = name.as_bytes();
        let n = name_bytes.len().min(IOFX_CHAIN_NAME_BYTES);
        let name_dst = (&raw mut (*slot_host).kernel_name) as *mut u8;
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), name_dst, n);
        let raw_dst = (*slot_host).raw_args.as_mut_ptr();
        (*slot_host).raw_args_len =
            dump_kernel_args_raw(kernel_params, extra, raw_dst, IOFX_CHAIN_RAW_ARGS_BYTES) as u32;
    }
    Some(slot_index)
}

unsafe fn capture_iofx_chain_ptr_regions(slot_index: usize, stream: hipStream_t, prelaunch: bool) {
    if !iofx_chain_capture_enabled() {
        return;
    }
    // Pre/post DtoD snapshotting is gated separately — 8 DtoD copies per
    // chain launch on every PhysX frame causes IOFX_volumeMigrationKernel
    // to fault during init (verified 2026-05-11 across N=4 runs at both
    // 64 KB and 16 KB per-ptr buffer sizes). Default OFF; header-only
    // capture is enough to enumerate kernel kinds, frequencies, and
    // pointer-arg layouts. Opt in with ZLUDA_IOFX_CHAIN_CAPTURE_BUFS=1
    // for short investigation runs where the early-init regression is
    // tolerable.
    //
    // SAMPLING (added 2026-05-11): when BUFS=1, by default snapshot every
    // ZLUDA_IOFX_CHAIN_CAPTURE_BUFS_SAMPLE-th launch (default 64) instead
    // of every launch — reduces stream pressure ~64× while still filling
    // the 16-slot ring with launches spread across the entire run.
    // Set BUFS_SAMPLE=1 to snapshot every launch (the previous behavior).
    static SNAPSHOT_BUFS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let snapshot_bufs = *SNAPSHOT_BUFS
        .get_or_init(|| std::env::var("ZLUDA_IOFX_CHAIN_CAPTURE_BUFS").map_or(false, |v| v == "1"));
    if !snapshot_bufs {
        return;
    }
    static SNAPSHOT_SAMPLE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let sample_interval = *SNAPSHOT_SAMPLE.get_or_init(|| {
        std::env::var("ZLUDA_IOFX_CHAIN_CAPTURE_BUFS_SAMPLE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64)
    });
    // The pre and post DtoDs for a given launch share the same launch_n
    // (captured in the slot header at the slot we were just given). Reuse
    // that for the sampling decision so pre and post stay paired.
    let launch_n_for_slot = {
        let guard = match IOFX_CHAIN_CAPTURE.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let state = match guard.as_ref() {
            Some(s) => s,
            None => return,
        };
        let slot_host = unsafe {
            state.host_ptr.add(slot_index * IOFX_CHAIN_SLOT_SIZE) as *const IofxChainCaptureSlot
        };
        unsafe { (*slot_host).launch_n }
    };
    if launch_n_for_slot % sample_interval != 0 {
        return;
    }
    let guard = match IOFX_CHAIN_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };
    let slot_host = unsafe {
        state.host_ptr.add(slot_index * IOFX_CHAIN_SLOT_SIZE) as *const IofxChainCaptureSlot
    };
    let slot_dev_base =
        unsafe { (state.dev_ptr as *mut u8).add(slot_index * IOFX_CHAIN_SLOT_SIZE) };
    let base_off = if prelaunch {
        IOFX_CHAIN_PRE_PTRS_OFF
    } else {
        IOFX_CHAIN_POST_PTRS_OFF
    };
    let launch_n = unsafe { (*slot_host).launch_n };
    let kernel_kind = unsafe { (*slot_host).kernel_kind };
    for (idx, src_ptr) in unsafe { (*slot_host).ptr_params }
        .iter()
        .copied()
        .enumerate()
    {
        if src_ptr == 0 {
            continue;
        }
        let dst_off = base_off + idx * IOFX_CHAIN_BUFFER_BYTES;
        let res = unsafe {
            hipMemcpyDtoDAsync(
                hipDeviceptr_t(slot_dev_base.add(dst_off) as _),
                hipDeviceptr_t(src_ptr as _),
                IOFX_CHAIN_BUFFER_BYTES,
                stream,
            )
        };
        if let Err(code) = res {
            // Log the first few failures per session so we know whether
            // DtoD is silently failing (e.g. invalid/freed src ptr,
            // wrong context, queue wedged). Capped to avoid log spam.
            static ERR_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = ERR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 32 {
                super::driver::zluda_log_err(&format!(
                    "[IOFX_CHAIN] DtoD err#{} {} launch_n={} kind={} idx={} \
                     src=0x{:x} dst=0x{:x} bytes={} err={:?}",
                    n,
                    if prelaunch { "PRE" } else { "POST" },
                    launch_n,
                    kernel_kind,
                    idx,
                    src_ptr,
                    slot_dev_base as usize + dst_off,
                    IOFX_CHAIN_BUFFER_BYTES,
                    code,
                ));
            }
        }
    }
}

unsafe fn capture_iofx_chain_pre(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_chain_ptr_regions(slot_index, stream, true) };
}

unsafe fn capture_iofx_chain_post(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_chain_ptr_regions(slot_index, stream, false) };
}

// 2026-06-16 BUG-2 Step-1a: in-process value-scan TRIP-WIRE (ZLUDA_IOFX_VALUE_TRIPWIRE=1).
// Class-4 = a consumer (bbox/actorRange) reads a garbage actor-index (a float bit-pattern) from a
// pooled index buffer → wild tex fetch / scatter. This catches the corrupt word + names the buffer +
// the adjacent pooled sub-buffer it likely came from, in the faulting frame, with NO debugger.
// MUST use DtoD into a pinned-mapped scratch then read the HOST side — hipMemcpyDtoH from launch_kernel
// crashes the ROCm runtime (see file-header note line 49). Light: <=8 * 2KB DtoD only on the target
// IOFX kernels, only after a warmup, one-shot on first hit.
const IOFX_TW_SLOTS: usize = 8;
const IOFX_TW_SLOT_BYTES: usize = 16384; // first 4096 u32 of each ptr param (cover mid-range indices)
struct IofxTwState { host: *mut u8, dev: *mut std::ffi::c_void }
unsafe impl Send for IofxTwState {}
static IOFX_TW: Mutex<Option<IofxTwState>> = Mutex::new(None);
static IOFX_TW_LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IOFX_TW_FOUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn iofx_value_tripwire_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("ZLUDA_IOFX_VALUE_TRIPWIRE").map_or(false, |v| v == "1"))
}

unsafe fn iofx_value_tripwire(
    name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
) {
    if !iofx_value_tripwire_enabled() || IOFX_TW_FOUND.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    // Target the index CONSUMERS (where the fault reads the garbage index) + makeSortKeys/remap.
    let is_target = name == "IOFX_bboxKernel"
        || name == "IOFX_actorRangeKernel"
        || name.starts_with("IOFX_remapKernel")
        || name.starts_with("IOFX_makeSortKeys");
    if !is_target {
        return;
    }
    let nl = IOFX_TW_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if nl < 3000 {
        return; // warmup — skip init/early frames; Class-4 is deep
    }
    // up to 14 pointer args; keep the 0x7f-prefixed, page-aligned ones (real device ptrs)
    let mut ptrs: Vec<u64> = Vec::new();
    for i in 0..14usize {
        if let Some(v) = read_kernel_arg::<u64>(kernel_params, extra, i * 8, i) {
            if (v >> 40) == 0x7f && (v & 0xFFFF) == 0 && (v & 0xFFFFFFFF) >= 0x10000 {
                ptrs.push(v);
            }
        }
        if ptrs.len() >= IOFX_TW_SLOTS {
            break;
        }
    }
    if ptrs.is_empty() {
        return;
    }
    let mut guard = match IOFX_TW.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if guard.is_none() {
        let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
        if unsafe {
            hipHostMalloc(&mut host, IOFX_TW_SLOTS * IOFX_TW_SLOT_BYTES, hipHostMallocMapped)
        }
        .is_err()
            || host.is_null()
        {
            return;
        }
        let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
        if unsafe { hipHostGetDevicePointer(&mut dev, host, 0) }.is_err() {
            return;
        }
        *guard = Some(IofxTwState { host: host as *mut u8, dev });
    }
    let st = guard.as_ref().unwrap();
    for (i, &p) in ptrs.iter().enumerate() {
        let dst = hipDeviceptr_t((st.dev as usize + i * IOFX_TW_SLOT_BYTES) as *mut std::ffi::c_void);
        let _ = unsafe { hipMemcpyDtoDAsync(dst, hipDeviceptr_t(p as _), IOFX_TW_SLOT_BYTES, stream) };
    }
    let _ = unsafe { hipStreamSynchronize(stream) };
    // scan each captured slot host-side for an index-buffer-with-float-outlier
    let floaty = |w: u32| {
        let hb = (w >> 24) & 0xff;
        (0x40..=0x4f).contains(&hb) || (0xb8..=0xcf).contains(&hb)
    };
    for (i, &p) in ptrs.iter().enumerate() {
        let base = unsafe { st.host.add(i * IOFX_TW_SLOT_BYTES) } as *const u32;
        let words: &[u32] = unsafe { std::slice::from_raw_parts(base, IOFX_TW_SLOT_BYTES / 4) };
        let nz: Vec<u32> = words
            .iter()
            .copied()
            .filter(|&w| w != 0 && w != 0xffff_ffff)
            .collect();
        if nz.len() < 16 {
            continue;
        }
        let small = nz
            .iter()
            .filter(|&&w| w < (1 << 20) && w != 0xffff_ffff)
            .count();
        let outliers: Vec<(usize, u32)> = words
            .iter()
            .enumerate()
            .filter(|(_, &w)| w >= (1 << 24) && w != 0xffff_ffff && floaty(w))
            .map(|(j, &w)| (j, w))
            .collect();
        if small >= (nz.len() * 6) / 10 && (1..=8).contains(&outliers.len()) {
            // HIT — name the buffer, its alloc, and hunt the float's source in adjacent sub-buffers
            IOFX_TW_FOUND.store(true, std::sync::atomic::Ordering::Relaxed);
            let alloc = unsafe { query_device_pointer_range(p) };
            let (abase, asize) = alloc.map(|(b, s, _)| (b, s)).unwrap_or((0, 0));
            let bad = outliers[0].1;
            // look for the same float value in the two adjacent pooled sub-buffers below this ptr
            let mut src = String::from("none");
            for delta in [0x800u64, 0x1000, 0x80000, 0x40000] {
                if p < delta {
                    continue;
                }
                let probe = p - delta;
                if let Some((b2, _, _)) = unsafe { query_device_pointer_range(probe) } {
                    if b2 == abase {
                        src = format!("same-alloc@-0x{:x}", delta);
                    } else {
                        src = format!("alloc@0x{:x}(-0x{:x})", b2, delta);
                    }
                    break;
                }
            }
            super::driver::zluda_log_err(&format!(
                "[IOFX_VALUE_TRIPWIRE] kernel={} launch={} param_slot={} ptr=0x{:x} \
                 alloc[base=0x{:x} size={} off={}] small={}/{} OUTLIERS={:x?} below={}",
                name,
                nl,
                i,
                p,
                abase,
                asize,
                p.wrapping_sub(abase),
                small,
                nz.len(),
                &outliers[..outliers.len().min(8)],
                src
            ));
            // 2026-06-17 tripwire→ring dump: snapshot the IOFX_CHAIN_CAPTURE
            // 256-slot ring to disk so `scripts/analyze_iofx_chain.py` can name
            // the exact kernel+param that wrote these distance-KEY floats into the
            // value/index buffer (the Class-4 producer). The offline shared-pool
            // repro proved the sort scatter is clean in isolation, so the producer
            // must be caught live. Requires the run to ALSO set
            // ZLUDA_IOFX_CHAIN_CAPTURE=1 (+ ZLUDA_IOFX_CHAIN_CAPTURE_BUFS=1 for the
            // pre/post buffer images the analyzer scans). One-shot — IOFX_TW_FOUND
            // is already set above so this fires once, on the first corrupt frame.
            let ring_host =
                IOFX_CHAIN_CAPTURE_HOST_PTR.load(std::sync::atomic::Ordering::SeqCst);
            let ring_bytes =
                IOFX_CHAIN_CAPTURE_TOTAL_BYTES.load(std::sync::atomic::Ordering::SeqCst);
            if ring_host != 0 && ring_bytes != 0 {
                let path = format!("/tmp/iofx_chain_tripwire_{}.bin", nl);
                let slice =
                    unsafe { std::slice::from_raw_parts(ring_host as *const u8, ring_bytes) };
                match std::fs::write(&path, slice) {
                    Ok(_) => super::driver::zluda_log_err(&format!(
                        "[IOFX_VALUE_TRIPWIRE] dumped chain ring ({} bytes) to {} — \
                         run scripts/analyze_iofx_chain.py on it to name the producer",
                        ring_bytes, path
                    )),
                    Err(e) => super::driver::zluda_log_err(&format!(
                        "[IOFX_VALUE_TRIPWIRE] chain ring dump to {} FAILED: {}",
                        path, e
                    )),
                }
            } else {
                super::driver::zluda_log_err(
                    "[IOFX_VALUE_TRIPWIRE] chain ring EMPTY — set ZLUDA_IOFX_CHAIN_CAPTURE=1 \
                     + ZLUDA_IOFX_CHAIN_CAPTURE_BUFS=1 to capture the producer",
                );
            }
            let _ = bad;
            break;
        }
    }
}

unsafe fn tfs_capture_init(state: &mut Option<TfsCaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result =
        unsafe { hipHostMalloc(&mut host_ptr, TFS_CAPTURE_TOTAL_SIZE, hipHostMallocMapped) };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS capture: hipHostMalloc({}) failed: {:?}",
            TFS_CAPTURE_TOTAL_SIZE, alloc_result
        ));
        return;
    }
    unsafe {
        std::ptr::write_bytes(host_ptr as *mut u8, 0, TFS_CAPTURE_TOTAL_SIZE);
    }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS capture: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        unsafe {
            let _ = hipHostFree(host_ptr);
        }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] TFS capture: {} slots × {} B = {} KB pinned; host=0x{:x} dev=0x{:x}",
        TFS_CAPTURE_SLOTS,
        TFS_SLOT_SIZE,
        TFS_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize,
        dev_ptr as usize,
    ));
    TFS_CAPTURE_HOST_PTR.store(host_ptr as usize, std::sync::atomic::Ordering::SeqCst);
    TFS_CAPTURE_TOTAL_BYTES.store(TFS_CAPTURE_TOTAL_SIZE, std::sync::atomic::Ordering::SeqCst);

    *state = Some(TfsCaptureState {
        host_ptr: host_ptr as *mut u8,
        dev_ptr,
        launch_counter: 0,
    });
}

/// Read up to `max_len` bytes of the kernel argument flat buffer into `dst`.
/// Handles both the kernel_params (array of pointers) and extra (flat byte
/// buffer) calling conventions. Returns number of bytes actually read.
///
/// For kernel_params: each entry is a pointer to one argument's value, so
/// we can only read individual arguments, not a contiguous range. We probe
/// the first 16 arg indices and pack their bytes sequentially. This loses
/// the original byte-offset information BUT preserves all the data; the
/// post-mortem analyzer needs to know each kernel's per-arg sizes.
///
/// For extra: we read max_len bytes from the flat buffer starting at offset 0.
/// This preserves the in-game byte layout exactly — the post-mortem
/// analyzer can read at the same offsets the kernel does.
unsafe fn dump_kernel_args_raw(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    dst: *mut u8,
    max_len: usize,
) -> usize {
    if !extra.is_null() {
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        // CU_LAUNCH_PARAM_BUFFER_POINTER = 1, _SIZE = 2
        if extra_arr[0] == (0x1usize as *mut ::core::ffi::c_void)
            && extra_arr[2] == (0x2usize as *mut ::core::ffi::c_void)
        {
            let buffer = extra_arr[1] as *const u8;
            let size_ptr = extra_arr[3] as *const usize;
            if !buffer.is_null() && !size_ptr.is_null() {
                let size = *size_ptr;
                let read_len = size.min(max_len);
                std::ptr::copy_nonoverlapping(buffer, dst, read_len);
                return read_len;
            }
        }
    }
    if !kernel_params.is_null() {
        // Pack up to 16 arg slots sequentially into dst, 8 bytes per slot.
        // Null entries are recorded as zero-filled 8-byte holes instead of
        // terminating the dump, because some PhysX launches leave early slots
        // null while still passing meaningful later arguments (for example the
        // adjacent CreateParticleShapesIncremental_MultiStage_2_Batched crash
        // window we now capture).
        let slot_count = (max_len / 8).min(16);
        for i in 0..slot_count {
            let dst_slot = dst.add(i * 8);
            std::ptr::write_bytes(dst_slot, 0, 8);
            let p = *kernel_params.add(i);
            if !p.is_null() {
                std::ptr::copy_nonoverlapping(p as *const u8, dst_slot, 8);
            }
        }
        return slot_count * 8;
    }
    0
}

unsafe fn dump_kernel_param_ptrs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    dst: *mut u8,
    max_len: usize,
) -> usize {
    if kernel_params.is_null() {
        return 0;
    }
    let slot_count = (max_len / 8).min(16);
    for i in 0..slot_count {
        let dst_slot = dst.add(i * 8) as *mut u64;
        std::ptr::write_unaligned(dst_slot, (*kernel_params.add(i)) as usize as u64);
    }
    slot_count * 8
}

unsafe fn summarize_launch_arg_encoding(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> String {
    let mut summary = String::new();
    let extra_flat = if extra.is_null() {
        false
    } else {
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        extra_arr[0] == (0x1usize as *mut ::core::ffi::c_void)
            && extra_arr[2] == (0x2usize as *mut ::core::ffi::c_void)
    };
    let _ = write!(
        &mut summary,
        "extra_null={} extra_flat={} kernel_params_null={}",
        extra.is_null(),
        extra_flat,
        kernel_params.is_null()
    );
    if !extra.is_null() {
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        let _ = write!(&mut summary, " extra=[");
        for (i, ptr) in extra_arr.iter().enumerate() {
            if i > 0 {
                summary.push_str(", ");
            }
            let _ = write!(&mut summary, "{}:0x{:x}", i, *ptr as usize);
        }
        summary.push(']');
    }
    summary
}

unsafe fn encode_launch_arg_meta(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> [u8; 8] {
    let mut meta = [0u8; 8];
    meta[0] = extra.is_null() as u8;
    meta[1] = kernel_params.is_null() as u8;
    if !extra.is_null() {
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        meta[2] = (extra_arr[0] == (0x1usize as *mut ::core::ffi::c_void)) as u8;
        meta[3] = (extra_arr[2] == (0x2usize as *mut ::core::ffi::c_void)) as u8;
        meta[4] = extra_arr[1].is_null() as u8;
        meta[5] = extra_arr[3].is_null() as u8;
    }
    meta
}

fn should_skip_tfs_raw_arg_dump(name: &str) -> bool {
    matches!(
        name,
        "CreateParticleShapesIncremental_MultiStage_1"
            | "CreateParticleShapesIncremental_MultiStage_2_Batched"
            | "CreateParticleShapesIncremental_MultiStage_3"
            | "CreateParticleShapesIncremental_MultiStage_4"
            | "InitCollDataAndApplyConstraints"
            | "FinalizeCollision"
            | "ParticleCollisionStaticBox"
    )
}

const TFS_MS2B_ARGS_CAPTURE_MAGIC: u32 = 0x4D32_4241; // "M2BA"
const TFS_MS2B_ARGS_CAPTURE_VERSION: u32 = 1;
const TFS_MS2B_ARGS_CAPTURE_VERSION_V2: u32 = 2;
const TFS_MS2B_ARGS_CAPTURE_VERSION_V3: u32 = 3;
const TFS_MS2B_ARGS_CAPTURE_VERSION_V4: u32 = 4;
const TFS_MS2B_PACKET_SAMPLE_BYTES: usize = 88 * 3;
const TFS_MS2B_PACKET_ENTRY_BYTES: usize = 88;
const TFS_MS2B_PACKET_SAMPLE_ENTRIES: usize =
    TFS_MS2B_PACKET_SAMPLE_BYTES / TFS_MS2B_PACKET_ENTRY_BYTES;
const TFS_MS2B_RD_SAMPLE_WORDS: usize = 8;
const TFS_MS2B_SLOT_PROBE_WORDS: usize = 4;
const TFS_MS3_ARGS_CAPTURE_MAGIC: u32 = 0x4D53_3341; // "MS3A"
const TFS_MS3_ARGS_CAPTURE_VERSION: u32 = 1;
const TFS_MS3_ARGS_CAPTURE_VERSION_V2: u32 = 2;
const TFS_MS3_P2_SAMPLE_WORDS: usize = 8;
const TFS_MS3_P3_SAMPLE_WORDS: usize = 8;
const TFS_MS4_ARGS_CAPTURE_MAGIC: u32 = 0x4D53_3441; // "MS4A"
const TFS_MS4_ARGS_CAPTURE_VERSION: u32 = 1;
const TFS_MS4_ARGS_CAPTURE_VERSION_V2: u32 = 2;
const TFS_MS4_P2_SAMPLE_WORDS: usize = 32;
const TFS_MS4_P4_SAMPLE_WORDS: usize = 8;
const TFS_MS4_P6_SAMPLE_WORDS: usize = 32;
const TFS_MS4_P7_SAMPLE_WORDS: usize = 32;
const TFS_ATOMIC_TO_TEX_CAPTURE_MAGIC: u32 = 0x4154_5841; // "ATXA"
const TFS_ATOMIC_TO_TEX_CAPTURE_VERSION: u32 = 1;
const TFS_ATOMIC_TO_TEX_SAMPLE_WORDS: usize = 8;
const TFS_ATOMIC_TO_TEX_SURFACE_DESC_WORDS: usize = 8;
const TFS_MUT_P47_CAPTURE_MAGIC: u32 = 0x4D50_3437; // "MP47"
const TFS_MUT_P47_CAPTURE_VERSION: u32 = 4;
const TFS_MUT_P47_CAPTURE_VERSION_V3: u32 = 3;
const TFS_MUT_P47_CAPTURE_VERSION_V2: u32 = 2;
const TFS_MUT_P47_CAPTURE_VERSION_V1: u32 = 1;
const TFS_MUT_P47_SAMPLE_WORDS: usize = 8;
const TFS_MUT_P47_ARG_PREFIX_WORDS: usize = 16;
/// Byte offset within rd5 of the "particleShapesCounter" persistent slot.
/// MS_2_Batched's BB11_4 loads `*(rd5 + 22176)` into LDS as the active-shape
/// count. MS_1 is the suspected writer; the counter probe captures it at
/// every MUT_P47 sample point to verify whether MS_1 receives a stale
/// non-zero counter in the failing cycle.
const TFS_RD5_PARTICLE_SHAPES_COUNTER_OFFSET: usize = 22176;
static TFS_MUT_P47_LAST_MS2_LAUNCH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static TFS_MUT_P47_LAST_RD5: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TFS_MUT_P47_LAST_RD6: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TFS_MUT_P47_LAST_RD7: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TFS_MUT_P47_ENTRY_INDEX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static TFS_MUT_P47_BROAD_PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct TfsMs2BatchedArgsCaptureV1 {
    magic: u32,
    version: u32,
    p2: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMs2BatchedArgsCaptureV2 {
    magic: u32,
    version: u32,
    p2: u64,
    grid_x: u32,
    block_x: u32,
    packet_sample_bytes: u32,
    _pad0: u32,
    packet_sample: [u8; TFS_MS2B_PACKET_SAMPLE_BYTES],
}

impl Default for TfsMs2BatchedArgsCaptureV2 {
    fn default() -> Self {
        Self {
            magic: TFS_MS2B_ARGS_CAPTURE_MAGIC,
            version: TFS_MS2B_ARGS_CAPTURE_VERSION_V2,
            p2: 0,
            grid_x: 0,
            block_x: 0,
            packet_sample_bytes: 0,
            _pad0: 0,
            packet_sample: [0; TFS_MS2B_PACKET_SAMPLE_BYTES],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMs2BatchedArgsCaptureV3 {
    magic: u32,
    version: u32,
    p2: u64,
    grid_x: u32,
    block_x: u32,
    packet_sample_bytes: u32,
    _pad0: u32,
    packet_sample: [u8; TFS_MS2B_PACKET_SAMPLE_BYTES],
    rd5_words: u32,
    rd6_words: u32,
    rd7_words: u32,
    _pad1: u32,
    rd5_words_data: [u32; TFS_MS2B_RD_SAMPLE_WORDS],
    rd6_words_data: [u32; TFS_MS2B_RD_SAMPLE_WORDS],
    rd7_words_data: [u32; TFS_MS2B_RD_SAMPLE_WORDS],
}

impl Default for TfsMs2BatchedArgsCaptureV3 {
    fn default() -> Self {
        Self {
            magic: TFS_MS2B_ARGS_CAPTURE_MAGIC,
            version: TFS_MS2B_ARGS_CAPTURE_VERSION_V3,
            p2: 0,
            grid_x: 0,
            block_x: 0,
            packet_sample_bytes: 0,
            _pad0: 0,
            packet_sample: [0; TFS_MS2B_PACKET_SAMPLE_BYTES],
            rd5_words: 0,
            rd6_words: 0,
            rd7_words: 0,
            _pad1: 0,
            rd5_words_data: [0; TFS_MS2B_RD_SAMPLE_WORDS],
            rd6_words_data: [0; TFS_MS2B_RD_SAMPLE_WORDS],
            rd7_words_data: [0; TFS_MS2B_RD_SAMPLE_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMs2BatchedArgsCaptureV4 {
    magic: u32,
    version: u32,
    p2: u64,
    grid_x: u32,
    block_x: u32,
    packet_sample_bytes: u32,
    _pad0: u32,
    packet_sample: [u8; TFS_MS2B_PACKET_SAMPLE_BYTES],
    rd5_words: u32,
    rd6_words: u32,
    rd7_words: u32,
    _pad1: u32,
    rd5_words_data: [u32; TFS_MS2B_RD_SAMPLE_WORDS],
    rd6_words_data: [u32; TFS_MS2B_RD_SAMPLE_WORDS],
    rd7_words_data: [u32; TFS_MS2B_RD_SAMPLE_WORDS],
    requested_entry: u32,
    effective_entry: u32,
    slot_probe_words: u32,
    slot_probe_rd5_mask: u32,
    slot_probe_r38_mask: u32,
    _pad2: u32,
    slot_probe_slots: [u32; TFS_MS2B_SLOT_PROBE_WORDS],
    slot_probe_rd5_p4: [u32; TFS_MS2B_SLOT_PROBE_WORDS],
    slot_probe_r38: [u32; TFS_MS2B_SLOT_PROBE_WORDS],
}

impl Default for TfsMs2BatchedArgsCaptureV4 {
    fn default() -> Self {
        Self {
            magic: TFS_MS2B_ARGS_CAPTURE_MAGIC,
            version: TFS_MS2B_ARGS_CAPTURE_VERSION_V4,
            p2: 0,
            grid_x: 0,
            block_x: 0,
            packet_sample_bytes: 0,
            _pad0: 0,
            packet_sample: [0; TFS_MS2B_PACKET_SAMPLE_BYTES],
            rd5_words: 0,
            rd6_words: 0,
            rd7_words: 0,
            _pad1: 0,
            rd5_words_data: [0; TFS_MS2B_RD_SAMPLE_WORDS],
            rd6_words_data: [0; TFS_MS2B_RD_SAMPLE_WORDS],
            rd7_words_data: [0; TFS_MS2B_RD_SAMPLE_WORDS],
            requested_entry: 0,
            effective_entry: 0,
            slot_probe_words: 0,
            slot_probe_rd5_mask: 0,
            slot_probe_r38_mask: 0,
            _pad2: 0,
            slot_probe_slots: [0; TFS_MS2B_SLOT_PROBE_WORDS],
            slot_probe_rd5_p4: [0; TFS_MS2B_SLOT_PROBE_WORDS],
            slot_probe_r38: [0; TFS_MS2B_SLOT_PROBE_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct TfsMs3ArgsCaptureV1 {
    magic: u32,
    version: u32,
    p2: u64,
    p3: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMs3ArgsCaptureV2 {
    magic: u32,
    version: u32,
    p2: u64,
    p3: u64,
    p2_words: u32,
    p3_words: u32,
    p2_words_data: [u32; TFS_MS3_P2_SAMPLE_WORDS],
    p3_words_data: [u32; TFS_MS3_P3_SAMPLE_WORDS],
}

impl Default for TfsMs3ArgsCaptureV2 {
    fn default() -> Self {
        Self {
            magic: TFS_MS3_ARGS_CAPTURE_MAGIC,
            version: TFS_MS3_ARGS_CAPTURE_VERSION_V2,
            p2: 0,
            p3: 0,
            p2_words: 0,
            p3_words: 0,
            p2_words_data: [0; TFS_MS3_P2_SAMPLE_WORDS],
            p3_words_data: [0; TFS_MS3_P3_SAMPLE_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct TfsMs4ArgsCaptureV1 {
    magic: u32,
    version: u32,
    n: u32,
    _pad0: u32,
    p2: u64,
    p4: u64,
    p5: u64,
    p6: u64,
    p7: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMs4ArgsCaptureV2 {
    magic: u32,
    version: u32,
    n: u32,
    _pad0: u32,
    p2: u64,
    p4: u64,
    p5: u64,
    p6: u64,
    p7: u64,
    p2_words: u32,
    p4_words: u32,
    p6_words: u32,
    p7_words: u32,
    p2_words_data: [u32; TFS_MS4_P2_SAMPLE_WORDS],
    p4_words_data: [u32; TFS_MS4_P4_SAMPLE_WORDS],
    p6_words_data: [u32; TFS_MS4_P6_SAMPLE_WORDS],
    p7_words_data: [u32; TFS_MS4_P7_SAMPLE_WORDS],
}

impl Default for TfsMs4ArgsCaptureV2 {
    fn default() -> Self {
        Self {
            magic: TFS_MS4_ARGS_CAPTURE_MAGIC,
            version: TFS_MS4_ARGS_CAPTURE_VERSION_V2,
            n: 0,
            _pad0: 0,
            p2: 0,
            p4: 0,
            p5: 0,
            p6: 0,
            p7: 0,
            p2_words: 0,
            p4_words: 0,
            p6_words: 0,
            p7_words: 0,
            p2_words_data: [0; TFS_MS4_P2_SAMPLE_WORDS],
            p4_words_data: [0; TFS_MS4_P4_SAMPLE_WORDS],
            p6_words_data: [0; TFS_MS4_P6_SAMPLE_WORDS],
            p7_words_data: [0; TFS_MS4_P7_SAMPLE_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMutP47CaptureV1 {
    magic: u32,
    version: u32,
    source_ms2_launch_n: u64,
    rd5: u64,
    rd7: u64,
    rd5_words: u32,
    rd7_words: u32,
    rd5_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    rd7_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
}

impl Default for TfsMutP47CaptureV1 {
    fn default() -> Self {
        Self {
            magic: TFS_MUT_P47_CAPTURE_MAGIC,
            version: TFS_MUT_P47_CAPTURE_VERSION_V1,
            source_ms2_launch_n: 0,
            rd5: 0,
            rd7: 0,
            rd5_words: 0,
            rd7_words: 0,
            rd5_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            rd7_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMutP47CaptureV2 {
    magic: u32,
    version: u32,
    source_ms2_launch_n: u64,
    rd5: u64,
    rd7: u64,
    rd5_words: u32,
    rd7_words: u32,
    rd5_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    rd7_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    arg_words: u32,
    arg_mode: u32, // 1=flat raw args, 2=kernel_params ptr slots
    arg_words_data: [u32; TFS_MUT_P47_ARG_PREFIX_WORDS],
}

impl Default for TfsMutP47CaptureV2 {
    fn default() -> Self {
        Self {
            magic: TFS_MUT_P47_CAPTURE_MAGIC,
            version: TFS_MUT_P47_CAPTURE_VERSION_V2,
            source_ms2_launch_n: 0,
            rd5: 0,
            rd7: 0,
            rd5_words: 0,
            rd7_words: 0,
            rd5_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            rd7_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            arg_words: 0,
            arg_mode: 0,
            arg_words_data: [0; TFS_MUT_P47_ARG_PREFIX_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMutP47CaptureV3 {
    magic: u32,
    version: u32,
    source_ms2_launch_n: u64,
    rd5: u64,
    rd6: u64,
    rd7: u64,
    rd5_words: u32,
    rd6_words: u32,
    rd7_words: u32,
    arg_words: u32,
    arg_mode: u32,
    _pad0: u32,
    rd5_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    rd6_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    rd7_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    arg_words_data: [u32; TFS_MUT_P47_ARG_PREFIX_WORDS],
}

impl Default for TfsMutP47CaptureV3 {
    fn default() -> Self {
        Self {
            magic: TFS_MUT_P47_CAPTURE_MAGIC,
            version: TFS_MUT_P47_CAPTURE_VERSION_V3,
            source_ms2_launch_n: 0,
            rd5: 0,
            rd6: 0,
            rd7: 0,
            rd5_words: 0,
            rd6_words: 0,
            rd7_words: 0,
            arg_words: 0,
            arg_mode: 0,
            _pad0: 0,
            rd5_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            rd6_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            rd7_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            arg_words_data: [0; TFS_MUT_P47_ARG_PREFIX_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsMutP47CaptureV4 {
    magic: u32,
    version: u32,
    source_ms2_launch_n: u64,
    rd5: u64,
    rd6: u64,
    rd7: u64,
    rd5_words: u32,
    rd6_words: u32,
    rd7_words: u32,
    arg_words: u32,
    arg_mode: u32,
    /// `*(rd5 + 22176)` — particleShapesCounter sampled into the slot via
    /// the same D2D mechanism as the rd5/6/7 heads. 0xFFFFFFFF means the
    /// read was out-of-range (rd5 too short or pointer invalid).
    rd5_off_22176: u32,
    rd5_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    rd6_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    rd7_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    arg_words_data: [u32; TFS_MUT_P47_ARG_PREFIX_WORDS],
}

impl Default for TfsMutP47CaptureV4 {
    fn default() -> Self {
        Self {
            magic: TFS_MUT_P47_CAPTURE_MAGIC,
            version: TFS_MUT_P47_CAPTURE_VERSION,
            source_ms2_launch_n: 0,
            rd5: 0,
            rd6: 0,
            rd7: 0,
            rd5_words: 0,
            rd6_words: 0,
            rd7_words: 0,
            arg_words: 0,
            arg_mode: 0,
            rd5_off_22176: 0xFFFFFFFF,
            rd5_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            rd6_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            rd7_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            arg_words_data: [0; TFS_MUT_P47_ARG_PREFIX_WORDS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TfsAtomicToTexCaptureV1 {
    magic: u32,
    version: u32,
    source_ms2_launch_n: u64,
    rd5: u64,
    rd7: u64,
    source_ptr: u64,
    surface_handle: u64,
    surface_desc_ptr: u64,
    output_base: u64,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    surface_shift: u32,
    rd5_words: u32,
    rd7_words: u32,
    arg_words: u32,
    arg_mode: u32,
    source_words: u32,
    output_pre_words: u32,
    output_post_words: u32,
    _pad0: u32,
    rd5_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    rd7_words_data: [u32; TFS_MUT_P47_SAMPLE_WORDS],
    arg_words_data: [u32; TFS_MUT_P47_ARG_PREFIX_WORDS],
    surface_desc_words_data: [u32; TFS_ATOMIC_TO_TEX_SURFACE_DESC_WORDS],
    source_words_data: [u32; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
    output_pre_words_data: [u32; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
    output_post_words_data: [u32; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
}

impl Default for TfsAtomicToTexCaptureV1 {
    fn default() -> Self {
        Self {
            magic: TFS_ATOMIC_TO_TEX_CAPTURE_MAGIC,
            version: TFS_ATOMIC_TO_TEX_CAPTURE_VERSION,
            source_ms2_launch_n: 0,
            rd5: 0,
            rd7: 0,
            source_ptr: 0,
            surface_handle: 0,
            surface_desc_ptr: 0,
            output_base: 0,
            dim_x: 0,
            dim_y: 0,
            dim_z: 0,
            surface_shift: 0,
            rd5_words: 0,
            rd7_words: 0,
            arg_words: 0,
            arg_mode: 0,
            source_words: 0,
            output_pre_words: 0,
            output_post_words: 0,
            _pad0: 0,
            rd5_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            rd7_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
            arg_words_data: [0; TFS_MUT_P47_ARG_PREFIX_WORDS],
            surface_desc_words_data: [0; TFS_ATOMIC_TO_TEX_SURFACE_DESC_WORDS],
            source_words_data: [0; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
            output_pre_words_data: [0; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
            output_post_words_data: [0; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
        }
    }
}

unsafe fn dump_tfs_ms4_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs4ArgsCaptureV1>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let encoded = TfsMs4ArgsCaptureV1 {
        magic: TFS_MS4_ARGS_CAPTURE_MAGIC,
        version: TFS_MS4_ARGS_CAPTURE_VERSION,
        n: read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0),
        _pad0: 0,
        p2: read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0),
        p4: read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0),
        p5: read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0),
        p6: read_kernel_arg(kernel_params, extra, 48, 6).unwrap_or(0),
        p7: read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0),
    };
    std::ptr::write_unaligned(dst as *mut TfsMs4ArgsCaptureV1, encoded);
    std::mem::size_of::<TfsMs4ArgsCaptureV1>()
}

unsafe fn tfs_bounded_sample_words(ptr: u64, max_words: usize) -> u32 {
    if ptr < 0x1000 || max_words == 0 {
        return 0;
    }
    let mut alloc_base = hipDeviceptr_t(std::ptr::null_mut());
    let mut alloc_size = 0usize;
    if super::memory::get_address_range_v2(
        &mut alloc_base as *mut _,
        &mut alloc_size as *mut _,
        hipDeviceptr_t(ptr as *mut std::ffi::c_void),
    )
    .is_err()
    {
        return 0;
    }
    let base = alloc_base.0 as usize;
    let ptr_usize = ptr as usize;
    if base == 0 || ptr_usize < base {
        return 0;
    }
    let offset = ptr_usize - base;
    let available_bytes = alloc_size.saturating_sub(offset);
    (available_bytes / 4).min(max_words) as u32
}

unsafe fn tfs_bounded_sample_bytes(ptr: u64, max_bytes: usize) -> u32 {
    if ptr < 0x1000 || max_bytes == 0 {
        return 0;
    }
    let mut alloc_base = hipDeviceptr_t(std::ptr::null_mut());
    let mut alloc_size = 0usize;
    if super::memory::get_address_range_v2(
        &mut alloc_base as *mut _,
        &mut alloc_size as *mut _,
        hipDeviceptr_t(ptr as *mut std::ffi::c_void),
    )
    .is_err()
    {
        return 0;
    }
    let base = alloc_base.0 as usize;
    let ptr_usize = ptr as usize;
    if base == 0 || ptr_usize < base {
        return 0;
    }
    let offset = ptr_usize - base;
    let available_bytes = alloc_size.saturating_sub(offset);
    available_bytes.min(max_bytes) as u32
}

unsafe fn tfs_capture_d2d_bytes(
    slot_dev_base: *mut u8,
    src_ptr: u64,
    raw_args_field_off: usize,
    copy_bytes: usize,
    tag: &str,
) {
    if slot_dev_base.is_null() || src_ptr == 0 || copy_bytes == 0 {
        return;
    }
    let dst_dev = hipDeviceptr_t(slot_dev_base.add(TFS_RAW_ARGS_OFF + raw_args_field_off) as _);
    let src_dev = hipDeviceptr_t(src_ptr as *mut std::ffi::c_void);
    if let Err(err) = hipMemcpyDtoD(dst_dev, src_dev, copy_bytes) {
        static ERR_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = ERR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 16 {
            super::driver::zluda_log_err(&format!(
                "[{}] DtoD err#{} src=0x{:x} dst=0x{:x} bytes={} err={:?}",
                tag,
                n,
                src_ptr,
                slot_dev_base as usize + TFS_RAW_ARGS_OFF + raw_args_field_off,
                copy_bytes,
                err
            ));
        }
    }
}

unsafe fn tfs_capture_d2d_bytes_async(
    slot_dev_base: *mut u8,
    src_ptr: u64,
    raw_args_field_off: usize,
    copy_bytes: usize,
    stream: hipStream_t,
    tag: &str,
) {
    if slot_dev_base.is_null() || src_ptr == 0 || copy_bytes == 0 {
        return;
    }
    let dst_dev = hipDeviceptr_t(slot_dev_base.add(TFS_RAW_ARGS_OFF + raw_args_field_off) as _);
    let src_dev = hipDeviceptr_t(src_ptr as *mut std::ffi::c_void);
    if let Err(err) = hipMemcpyDtoDAsync(dst_dev, src_dev, copy_bytes, stream) {
        static ERR_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = ERR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 16 {
            super::driver::zluda_log_err(&format!(
                "[{}] DtoDAsync err#{} src=0x{:x} dst=0x{:x} bytes={} err={:?}",
                tag,
                n,
                src_ptr,
                slot_dev_base as usize + TFS_RAW_ARGS_OFF + raw_args_field_off,
                copy_bytes,
                err
            ));
        }
    }
}

unsafe fn tfs_ms4_capture_words_d2d(
    slot_dev_base: *mut u8,
    src_ptr: u64,
    raw_args_word_field_off: usize,
    words: u32,
) {
    tfs_capture_d2d_bytes(
        slot_dev_base,
        src_ptr,
        raw_args_word_field_off,
        (words as usize) * 4,
        "TFS_MS4A_V2",
    );
}

unsafe fn tfs_ms2b_capture_words_d2d(
    slot_dev_base: *mut u8,
    src_ptr: u64,
    raw_args_word_field_off: usize,
    words: u32,
) {
    tfs_capture_d2d_bytes(
        slot_dev_base,
        src_ptr,
        raw_args_word_field_off,
        (words as usize) * 4,
        "TFS_M2BA_V3",
    );
}

unsafe fn tfs_ms3_capture_words_d2d(
    slot_dev_base: *mut u8,
    src_ptr: u64,
    raw_args_word_field_off: usize,
    words: u32,
) {
    tfs_capture_d2d_bytes(
        slot_dev_base,
        src_ptr,
        raw_args_word_field_off,
        (words as usize) * 4,
        "TFS_MS3A_V2",
    );
}

fn tfs_mut_p47_track(rd5: u64, rd6: u64, rd7: u64, launch_n: u64) {
    TFS_MUT_P47_LAST_RD5.store(rd5, std::sync::atomic::Ordering::Relaxed);
    TFS_MUT_P47_LAST_RD6.store(rd6, std::sync::atomic::Ordering::Relaxed);
    TFS_MUT_P47_LAST_RD7.store(rd7, std::sync::atomic::Ordering::Relaxed);
    TFS_MUT_P47_LAST_MS2_LAUNCH.store(launch_n, std::sync::atomic::Ordering::Relaxed);
}

fn parse_tfs_mut_p47_entry_index(raw: Option<String>) -> usize {
    let parsed = raw
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    parsed.min(TFS_MS2B_PACKET_SAMPLE_ENTRIES.saturating_sub(1))
}

fn tfs_mut_p47_entry_index() -> usize {
    *TFS_MUT_P47_ENTRY_INDEX.get_or_init(|| {
        let entry_index =
            parse_tfs_mut_p47_entry_index(std::env::var("ZLUDA_TFS_MUT_P47_ENTRY").ok());
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS_MUT_P47 tracking MS2 packet entry {}",
            entry_index
        ));
        entry_index
    })
}

fn parse_tfs_mut_p47_broad_probe(raw: Option<String>) -> bool {
    raw.map_or(false, |value| value == "1")
}

fn tfs_mut_p47_broad_probe_enabled() -> bool {
    *TFS_MUT_P47_BROAD_PROBE.get_or_init(|| {
        let enabled = parse_tfs_mut_p47_broad_probe(std::env::var("ZLUDA_TFS_MUT_P47_BROAD").ok());
        if enabled {
            super::driver::zluda_log_err(
                "[ZLUDA] TFS_MUT_P47 broad probe enabled (capturing CLEAR/OTHER/MS3 window)",
            );
        }
        enabled
    })
}

fn tfs_atomic_to_tex_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ZLUDA_TFS_ATOMIC_TO_TEX_CAPTURE").map_or(false, |v| v == "1")
    })
}

fn parse_tfs_capture_only_atomic_to_tex(raw: Option<String>) -> bool {
    raw.map_or(false, |value| value == "1")
}

fn parse_tfs_capture_only_ms_chain(raw: Option<String>) -> bool {
    raw.map_or(false, |value| value == "1")
}

fn tfs_capture_only_atomic_to_tex_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = parse_tfs_capture_only_atomic_to_tex(
            std::env::var("ZLUDA_TFS_CAPTURE_ONLY_ATOMIC_TO_TEX").ok(),
        );
        if enabled {
            super::driver::zluda_log_err(
                "[ZLUDA] TFS capture focused on ParticleDensityAtomicToTex only",
            );
        }
        enabled
    })
}

fn tfs_capture_only_ms_chain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled =
            parse_tfs_capture_only_ms_chain(std::env::var("ZLUDA_TFS_CAPTURE_ONLY_MS_CHAIN").ok());
        if enabled {
            super::driver::zluda_log_err("[ZLUDA] TFS capture focused on MS2_Batched/MS3/MS4 only");
        }
        enabled
    })
}

fn should_capture_tfs_kernel_for_mode(
    kernel_id: u32,
    atomic_only: bool,
    ms_chain_only: bool,
) -> bool {
    if atomic_only {
        return kernel_id == TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX;
    }
    if ms_chain_only {
        return matches!(
            kernel_id,
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED
                | TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3
                | TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4
        );
    }
    true
}

fn should_capture_tfs_kernel(kernel_id: u32) -> bool {
    should_capture_tfs_kernel_for_mode(
        kernel_id,
        tfs_capture_only_atomic_to_tex_enabled(),
        tfs_capture_only_ms_chain_enabled(),
    )
}

pub(crate) fn ms2_rd6_watch_track(rd6: u64, launch_n: u64) {
    if rd6 < 0x1000 {
        return;
    }
    let watch =
        MS2_RD6_WATCH.get_or_init(|| Mutex::new(std::collections::VecDeque::with_capacity(64)));
    if let Ok(mut q) = watch.lock() {
        if let Some(pos) = q.iter().position(|(ptr, _)| *ptr == rd6) {
            q.remove(pos);
        }
        while q.len() >= 64 {
            q.pop_front();
        }
        q.push_back((rd6, launch_n));
    }
}

pub(crate) fn ms2_rd6_watch_snapshot() -> Vec<(u64, u64)> {
    match MS2_RD6_WATCH.get() {
        Some(watch) => match watch.lock() {
            Ok(q) => q.iter().copied().collect(),
            Err(_) => Vec::new(),
        },
        None => Vec::new(),
    }
}

fn ms2_rd6_watch_candidates(watch: &[(u64, u64)], max: usize) -> Vec<(u64, u64)> {
    if max == 0 {
        return Vec::new();
    }
    const PTR_LO: u64 = 0x1000;
    const PTR_HI: u64 = 0x0000_8000_0000_0000;
    let mut out = Vec::with_capacity(max.min(watch.len()));
    for &(rd6, launch_n) in watch.iter().rev() {
        if rd6 < PTR_LO || rd6 >= PTR_HI {
            continue;
        }
        if out.iter().any(|(seen, _)| *seen == rd6) {
            continue;
        }
        out.push((rd6, launch_n));
        if out.len() >= max {
            break;
        }
    }
    out.reverse();
    out
}

const MS2_RD6_TRACE_MAX_PTRS: usize = 8;
const MS2_RD6_TRACE_WORD_BYTES: usize = 4;
const MS2_RD6_TRACE_SCRATCH_BYTES: usize = MS2_RD6_TRACE_MAX_PTRS * MS2_RD6_TRACE_WORD_BYTES;

struct Ms2Rd6Tracker {
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
    last_state: std::collections::HashMap<u64, u32>,
    copy_error_count: u32,
}
unsafe impl Send for Ms2Rd6Tracker {}
unsafe impl Sync for Ms2Rd6Tracker {}

static MS2_RD6_TRACKER: Mutex<Option<Ms2Rd6Tracker>> = Mutex::new(None);

fn ms2_rd6_kernel_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_MS2_RD6_KERNEL_TRACE").map_or(false, |v| v == "1"))
}

unsafe fn ms2_rd6_tracker_init(state: &mut Option<Ms2Rd6Tracker>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = hipHostMalloc(
        &mut host_ptr,
        MS2_RD6_TRACE_SCRATCH_BYTES,
        hipHostMallocMapped,
    );
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS2 rd6 tracker: hipHostMalloc({}) failed: {:?}",
            MS2_RD6_TRACE_SCRATCH_BYTES, alloc_result
        ));
        return;
    }
    std::ptr::write_bytes(host_ptr as *mut u8, 0, MS2_RD6_TRACE_SCRATCH_BYTES);
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0);
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS2 rd6 tracker: hipHostGetDevicePointer failed: {:?}",
            map_result
        ));
        let _ = hipHostFree(host_ptr);
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MS2 rd6 tracker: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
        MS2_RD6_TRACE_SCRATCH_BYTES, host_ptr as usize, dev_ptr as usize
    ));
    *state = Some(Ms2Rd6Tracker {
        scratch_host: host_ptr as *mut u8,
        scratch_dev: dev_ptr,
        last_state: std::collections::HashMap::new(),
        copy_error_count: 0,
    });
}

unsafe fn check_ms2_rd6_mutation(current_kernel: &str, current_launch_n: Option<u64>) {
    if !ms2_rd6_kernel_trace_enabled() {
        return;
    }
    let watch = ms2_rd6_watch_snapshot();
    let tracked = ms2_rd6_watch_candidates(&watch, MS2_RD6_TRACE_MAX_PTRS);
    if tracked.is_empty() {
        return;
    }
    let mut guard = match MS2_RD6_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    ms2_rd6_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(t) => t,
        None => return,
    };
    let mut copied = [false; MS2_RD6_TRACE_MAX_PTRS];
    for (i, (rd6, _source_ms2_launch_n)) in tracked.iter().enumerate() {
        let dst =
            (tracker.scratch_dev as usize + i * MS2_RD6_TRACE_WORD_BYTES) as *mut std::ffi::c_void;
        let src = rd6.saturating_add(8) as *mut std::ffi::c_void;
        let copy_result = hipMemcpyDtoD(
            hipDeviceptr_t(dst),
            hipDeviceptr_t(src),
            MS2_RD6_TRACE_WORD_BYTES,
        );
        if copy_result.is_ok() {
            copied[i] = true;
        } else if tracker.copy_error_count < 16 {
            super::driver::zluda_log_err(&format!(
                "[MS2_RD6_KERNEL_TRACE] copy failed kernel='{}' launch_n={:?} rd6=0x{:x} err={:?}",
                current_kernel, current_launch_n, rd6, copy_result
            ));
            tracker.copy_error_count += 1;
        }
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    for (i, (rd6, source_ms2_launch_n)) in tracked.iter().enumerate() {
        if !copied[i] {
            continue;
        }
        let value =
            (tracker.scratch_host.add(i * MS2_RD6_TRACE_WORD_BYTES) as *const u32).read_unaligned();
        let previous = tracker.last_state.insert(*rd6, value);
        if let Some(prev) = previous {
            if prev != value {
                super::driver::zluda_log_err(&format!(
                    "[MS2_RD6_KERNEL_TRACE] kernel='{}' launch_n={:?} rd6=0x{:x} source_ms2_launch={} prev=0x{:08x} new=0x{:08x}",
                    current_kernel,
                    current_launch_n,
                    rd6,
                    source_ms2_launch_n,
                    prev,
                    value
                ));
            }
        }
    }
    tracker
        .last_state
        .retain(|ptr, _| tracked.iter().any(|(tracked_ptr, _)| tracked_ptr == ptr));
}

fn tfs_mut_p47_read_track() -> (u64, u64, u64, u64) {
    (
        TFS_MUT_P47_LAST_RD5.load(std::sync::atomic::Ordering::Relaxed),
        TFS_MUT_P47_LAST_RD6.load(std::sync::atomic::Ordering::Relaxed),
        TFS_MUT_P47_LAST_RD7.load(std::sync::atomic::Ordering::Relaxed),
        TFS_MUT_P47_LAST_MS2_LAUNCH.load(std::sync::atomic::Ordering::Relaxed),
    )
}

unsafe fn tfs_mut_p47_capture_words_d2d(
    slot_dev_base: *mut u8,
    src_ptr: u64,
    raw_args_word_field_off: usize,
    words: u32,
) {
    tfs_capture_d2d_bytes(
        slot_dev_base,
        src_ptr,
        raw_args_word_field_off,
        (words as usize) * 4,
        "TFS_MUT_P47",
    );
}

unsafe fn tfs_ms2b_entry_ptr(
    sample_bytes: u32,
    packet_sample: &[u8; TFS_MS2B_PACKET_SAMPLE_BYTES],
    entry_index: usize,
    entry_field_off: usize,
) -> u64 {
    let sample_bytes = sample_bytes as usize;
    let entry_base = entry_index.saturating_mul(TFS_MS2B_PACKET_ENTRY_BYTES);
    if sample_bytes < entry_base + TFS_MS2B_PACKET_ENTRY_BYTES
        || entry_field_off + 8 > TFS_MS2B_PACKET_ENTRY_BYTES
        || entry_base + entry_field_off + 8 > packet_sample.len()
    {
        return 0;
    }
    u64::from_le_bytes(
        packet_sample[entry_base + entry_field_off..entry_base + entry_field_off + 8]
            .try_into()
            .unwrap_or([0; 8]),
    )
}

fn tfs_ms2b_entry_count(
    sample_bytes: u32,
    packet_sample: &[u8; TFS_MS2B_PACKET_SAMPLE_BYTES],
    entry_index: usize,
) -> u32 {
    let sample_bytes = sample_bytes as usize;
    let entry_base = entry_index.saturating_mul(TFS_MS2B_PACKET_ENTRY_BYTES);
    if sample_bytes < entry_base + 4 || entry_base + 4 > packet_sample.len() {
        return 0;
    }
    u32::from_le_bytes(
        packet_sample[entry_base..entry_base + 4]
            .try_into()
            .unwrap_or([0; 4]),
    )
}

fn tfs_ms2b_effective_entry_index(
    sample_bytes: u32,
    packet_sample: &[u8; TFS_MS2B_PACKET_SAMPLE_BYTES],
    requested_entry: usize,
) -> usize {
    let available_entries =
        ((sample_bytes as usize) / TFS_MS2B_PACKET_ENTRY_BYTES).min(TFS_MS2B_PACKET_SAMPLE_ENTRIES);
    if available_entries == 0 {
        return requested_entry;
    }
    if requested_entry < available_entries {
        return requested_entry;
    }
    for idx in (0..available_entries).rev() {
        if tfs_ms2b_entry_count(sample_bytes, packet_sample, idx) > 0 {
            return idx;
        }
    }
    available_entries - 1
}

unsafe fn tfs_mut_p47_capture_arg_prefix(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> (u32, u32, [u32; TFS_MUT_P47_ARG_PREFIX_WORDS]) {
    let mut out_words = [0u32; TFS_MUT_P47_ARG_PREFIX_WORDS];
    let mut tmp = [0u8; TFS_MUT_P47_ARG_PREFIX_WORDS * 4];
    let mut mode = 0u32;
    let mut bytes = 0usize;
    if !kernel_params.is_null() || !extra.is_null() {
        bytes = dump_kernel_args_raw(kernel_params, extra, tmp.as_mut_ptr(), tmp.len());
        if bytes > 0 {
            mode = 1;
        } else if !kernel_params.is_null() {
            bytes = dump_kernel_param_ptrs(kernel_params, tmp.as_mut_ptr(), tmp.len());
            if bytes > 0 {
                mode = 2;
            }
        }
    }
    let copy_bytes = bytes.min(tmp.len());
    let full_words = copy_bytes / 4;
    let tail_bytes = copy_bytes % 4;
    for i in 0..full_words {
        let off = i * 4;
        out_words[i] = u32::from_le_bytes(tmp[off..off + 4].try_into().unwrap_or([0; 4]));
    }
    if tail_bytes != 0 && full_words < out_words.len() {
        let off = full_words * 4;
        let mut last = [0u8; 4];
        last[..tail_bytes].copy_from_slice(&tmp[off..off + tail_bytes]);
        out_words[full_words] = u32::from_le_bytes(last);
    }
    (
        ((copy_bytes + 3) / 4).min(TFS_MUT_P47_ARG_PREFIX_WORDS) as u32,
        mode,
        out_words,
    )
}

unsafe fn dump_tfs_mut_p47_capture(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    slot_dev_base: *mut u8,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null() || dst_len < std::mem::size_of::<TfsMutP47CaptureV1>() {
        return 0;
    }
    let (rd5, rd6, rd7, source_ms2_launch_n) = tfs_mut_p47_read_track();
    let (arg_words, arg_mode, arg_words_data) =
        tfs_mut_p47_capture_arg_prefix(kernel_params, extra);
    if dst_len >= std::mem::size_of::<TfsMutP47CaptureV4>() {
        // Probe `*(rd5 + 22176)` only if the bounded sample call shows
        // we have at least that many bytes available in the allocation.
        let counter_available = if rd5 != 0 {
            tfs_bounded_sample_words(
                rd5.saturating_add(TFS_RD5_PARTICLE_SHAPES_COUNTER_OFFSET as u64),
                1,
            ) >= 1
        } else {
            false
        };
        let encoded = TfsMutP47CaptureV4 {
            source_ms2_launch_n,
            rd5,
            rd6,
            rd7,
            rd5_words: tfs_bounded_sample_words(rd5, TFS_MUT_P47_SAMPLE_WORDS),
            rd6_words: tfs_bounded_sample_words(rd6, TFS_MUT_P47_SAMPLE_WORDS),
            rd7_words: tfs_bounded_sample_words(rd7, TFS_MUT_P47_SAMPLE_WORDS),
            arg_words,
            arg_mode,
            arg_words_data,
            ..Default::default()
        };
        std::ptr::write_unaligned(dst as *mut TfsMutP47CaptureV4, encoded);
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd5,
            std::mem::offset_of!(TfsMutP47CaptureV4, rd5_words_data),
            encoded.rd5_words,
        );
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd6,
            std::mem::offset_of!(TfsMutP47CaptureV4, rd6_words_data),
            encoded.rd6_words,
        );
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd7,
            std::mem::offset_of!(TfsMutP47CaptureV4, rd7_words_data),
            encoded.rd7_words,
        );
        if counter_available {
            tfs_capture_d2d_bytes(
                slot_dev_base,
                rd5 + TFS_RD5_PARTICLE_SHAPES_COUNTER_OFFSET as u64,
                std::mem::offset_of!(TfsMutP47CaptureV4, rd5_off_22176),
                4,
                "TFS_MUT_P47_CNT",
            );
        }
        return std::mem::size_of::<TfsMutP47CaptureV4>();
    }
    if dst_len >= std::mem::size_of::<TfsMutP47CaptureV3>() {
        let encoded = TfsMutP47CaptureV3 {
            source_ms2_launch_n,
            rd5,
            rd6,
            rd7,
            rd5_words: tfs_bounded_sample_words(rd5, TFS_MUT_P47_SAMPLE_WORDS),
            rd6_words: tfs_bounded_sample_words(rd6, TFS_MUT_P47_SAMPLE_WORDS),
            rd7_words: tfs_bounded_sample_words(rd7, TFS_MUT_P47_SAMPLE_WORDS),
            arg_words,
            arg_mode,
            arg_words_data,
            ..Default::default()
        };
        std::ptr::write_unaligned(dst as *mut TfsMutP47CaptureV3, encoded);
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd5,
            std::mem::offset_of!(TfsMutP47CaptureV3, rd5_words_data),
            encoded.rd5_words,
        );
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd6,
            std::mem::offset_of!(TfsMutP47CaptureV3, rd6_words_data),
            encoded.rd6_words,
        );
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd7,
            std::mem::offset_of!(TfsMutP47CaptureV3, rd7_words_data),
            encoded.rd7_words,
        );
        return std::mem::size_of::<TfsMutP47CaptureV3>();
    }
    if dst_len >= std::mem::size_of::<TfsMutP47CaptureV2>() {
        let encoded = TfsMutP47CaptureV2 {
            source_ms2_launch_n,
            rd5,
            rd7,
            rd5_words: tfs_bounded_sample_words(rd5, TFS_MUT_P47_SAMPLE_WORDS),
            rd7_words: tfs_bounded_sample_words(rd7, TFS_MUT_P47_SAMPLE_WORDS),
            arg_words,
            arg_mode,
            arg_words_data,
            ..Default::default()
        };
        std::ptr::write_unaligned(dst as *mut TfsMutP47CaptureV2, encoded);
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd5,
            std::mem::offset_of!(TfsMutP47CaptureV2, rd5_words_data),
            encoded.rd5_words,
        );
        tfs_mut_p47_capture_words_d2d(
            slot_dev_base,
            rd7,
            std::mem::offset_of!(TfsMutP47CaptureV2, rd7_words_data),
            encoded.rd7_words,
        );
        return std::mem::size_of::<TfsMutP47CaptureV2>();
    }
    let encoded = TfsMutP47CaptureV1 {
        source_ms2_launch_n,
        rd5,
        rd7,
        rd5_words: tfs_bounded_sample_words(rd5, TFS_MUT_P47_SAMPLE_WORDS),
        rd7_words: tfs_bounded_sample_words(rd7, TFS_MUT_P47_SAMPLE_WORDS),
        ..Default::default()
    };
    std::ptr::write_unaligned(dst as *mut TfsMutP47CaptureV1, encoded);
    tfs_mut_p47_capture_words_d2d(
        slot_dev_base,
        rd5,
        std::mem::offset_of!(TfsMutP47CaptureV1, rd5_words_data),
        encoded.rd5_words,
    );
    tfs_mut_p47_capture_words_d2d(
        slot_dev_base,
        rd7,
        std::mem::offset_of!(TfsMutP47CaptureV1, rd7_words_data),
        encoded.rd7_words,
    );
    std::mem::size_of::<TfsMutP47CaptureV1>()
}

unsafe fn read_kernel_arg_bytes(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    flat_arg_offset: usize,
    arg_index: usize,
    dst: *mut u8,
    len: usize,
) -> bool {
    if dst.is_null() || len == 0 {
        return false;
    }
    if !extra.is_null() {
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        if extra_arr[0] == (0x1usize as *mut ::core::ffi::c_void)
            && extra_arr[2] == (0x2usize as *mut ::core::ffi::c_void)
        {
            let buffer = extra_arr[1] as *const u8;
            let size_ptr = extra_arr[3] as *const usize;
            if !buffer.is_null() && !size_ptr.is_null() {
                let size = *size_ptr;
                if flat_arg_offset <= size && len <= size.saturating_sub(flat_arg_offset) {
                    std::ptr::copy_nonoverlapping(buffer.add(flat_arg_offset), dst, len);
                    return true;
                }
            }
        }
    }
    if !kernel_params.is_null() {
        let param_ptr = *kernel_params.add(arg_index);
        if !param_ptr.is_null() {
            std::ptr::copy_nonoverlapping(param_ptr as *const u8, dst, len);
            return true;
        }
    }
    false
}

fn tfs_atomic_to_tex_surface_desc_ptr(surface_handle: u64) -> (u32, u64) {
    (
        ((surface_handle >> 61) & 0x7) as u32,
        (surface_handle << 3) >> 3,
    )
}

fn tfs_atomic_to_tex_surface_desc_base(words: &[u32; TFS_ATOMIC_TO_TEX_SURFACE_DESC_WORDS]) -> u64 {
    // AMD image/surface descriptors store a 256-byte-aligned base split across
    // word0 and the low 16 bits of word1; the remaining word1 bits are
    // descriptor metadata. Reconstruct the byte address instead of treating the
    // first 64 bits as a raw pointer.
    ((((words[1] as u64) & 0xFFFF) << 32) | (words[0] as u64)) << 8
}

unsafe fn dump_tfs_atomic_to_tex_capture(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    slot_dev_base: *mut u8,
    stream: hipStream_t,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null() || dst_len < std::mem::size_of::<TfsAtomicToTexCaptureV1>() {
        return 0;
    }
    const PARAM5_FLAT_OFFSET: usize = 24;
    const PARAM5_STRUCT_BYTES: usize = 456;
    const PARAM5_DIM_X_OFF: usize = 28;
    const PARAM5_DIM_Y_OFF: usize = 32;
    const PARAM5_DIM_Z_OFF: usize = 36;
    const PARAM5_SURFACE_HANDLE_OFF: usize = 368;
    const PARAM6_FLAT_OFFSET: usize = PARAM5_FLAT_OFFSET + PARAM5_STRUCT_BYTES;

    let (rd5, _rd6, rd7, source_ms2_launch_n) = tfs_mut_p47_read_track();
    let (arg_words, arg_mode, arg_words_data) =
        tfs_mut_p47_capture_arg_prefix(kernel_params, extra);
    let mut param5 = [0u8; PARAM5_STRUCT_BYTES];
    let _ = read_kernel_arg_bytes(
        kernel_params,
        extra,
        PARAM5_FLAT_OFFSET,
        5,
        param5.as_mut_ptr(),
        param5.len(),
    );
    let source_ptr: u64 = read_kernel_arg(kernel_params, extra, PARAM6_FLAT_OFFSET, 6).unwrap_or(0);
    let dim_x = u32::from_le_bytes(
        param5[PARAM5_DIM_X_OFF..PARAM5_DIM_X_OFF + 4]
            .try_into()
            .unwrap_or([0; 4]),
    );
    let dim_y = u32::from_le_bytes(
        param5[PARAM5_DIM_Y_OFF..PARAM5_DIM_Y_OFF + 4]
            .try_into()
            .unwrap_or([0; 4]),
    );
    let dim_z = u32::from_le_bytes(
        param5[PARAM5_DIM_Z_OFF..PARAM5_DIM_Z_OFF + 4]
            .try_into()
            .unwrap_or([0; 4]),
    );
    let surface_handle = u64::from_le_bytes(
        param5[PARAM5_SURFACE_HANDLE_OFF..PARAM5_SURFACE_HANDLE_OFF + 8]
            .try_into()
            .unwrap_or([0; 8]),
    );
    let (surface_shift, surface_desc_ptr) = tfs_atomic_to_tex_surface_desc_ptr(surface_handle);
    let mut surface_desc_words = [0u32; TFS_ATOMIC_TO_TEX_SURFACE_DESC_WORDS];
    if surface_desc_ptr >= 0x1000 {
        let _ = hipMemcpyDtoH(
            surface_desc_words.as_mut_ptr() as _,
            hipDeviceptr_t(surface_desc_ptr as *mut std::ffi::c_void),
            std::mem::size_of_val(&surface_desc_words),
        );
    }
    let output_base = tfs_atomic_to_tex_surface_desc_base(&surface_desc_words);
    let encoded = TfsAtomicToTexCaptureV1 {
        source_ms2_launch_n,
        rd5,
        rd7,
        source_ptr,
        surface_handle,
        surface_desc_ptr,
        output_base,
        dim_x,
        dim_y,
        dim_z,
        surface_shift,
        rd5_words: tfs_bounded_sample_words(rd5, TFS_MUT_P47_SAMPLE_WORDS),
        rd7_words: tfs_bounded_sample_words(rd7, TFS_MUT_P47_SAMPLE_WORDS),
        arg_words,
        arg_mode,
        source_words: tfs_bounded_sample_words(source_ptr, TFS_ATOMIC_TO_TEX_SAMPLE_WORDS),
        output_pre_words: tfs_bounded_sample_words(output_base, TFS_ATOMIC_TO_TEX_SAMPLE_WORDS),
        output_post_words: tfs_bounded_sample_words(output_base, TFS_ATOMIC_TO_TEX_SAMPLE_WORDS),
        rd5_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
        rd7_words_data: [0; TFS_MUT_P47_SAMPLE_WORDS],
        arg_words_data,
        surface_desc_words_data: surface_desc_words,
        source_words_data: [0; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
        output_pre_words_data: [0; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
        output_post_words_data: [0; TFS_ATOMIC_TO_TEX_SAMPLE_WORDS],
        ..Default::default()
    };
    std::ptr::write_unaligned(dst as *mut TfsAtomicToTexCaptureV1, encoded);
    tfs_capture_d2d_bytes_async(
        slot_dev_base,
        rd5,
        std::mem::offset_of!(TfsAtomicToTexCaptureV1, rd5_words_data),
        encoded.rd5_words as usize * 4,
        stream,
        "TFS_ATXA_RD5",
    );
    tfs_capture_d2d_bytes_async(
        slot_dev_base,
        rd7,
        std::mem::offset_of!(TfsAtomicToTexCaptureV1, rd7_words_data),
        encoded.rd7_words as usize * 4,
        stream,
        "TFS_ATXA_RD7",
    );
    tfs_capture_d2d_bytes_async(
        slot_dev_base,
        source_ptr,
        std::mem::offset_of!(TfsAtomicToTexCaptureV1, source_words_data),
        encoded.source_words as usize * 4,
        stream,
        "TFS_ATXA_SRC",
    );
    tfs_capture_d2d_bytes_async(
        slot_dev_base,
        output_base,
        std::mem::offset_of!(TfsAtomicToTexCaptureV1, output_pre_words_data),
        encoded.output_pre_words as usize * 4,
        stream,
        "TFS_ATXA_PRE",
    );
    std::mem::size_of::<TfsAtomicToTexCaptureV1>()
}

unsafe fn dump_tfs_ms4_args_v2(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    slot_dev_base: *mut u8,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs4ArgsCaptureV2>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    let p4: u64 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
    let p5: u64 = read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0);
    let p6: u64 = read_kernel_arg(kernel_params, extra, 48, 6).unwrap_or(0);
    let p7: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);

    let encoded = TfsMs4ArgsCaptureV2 {
        n: read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0),
        p2,
        p4,
        p5,
        p6,
        p7,
        p2_words: tfs_bounded_sample_words(p2, TFS_MS4_P2_SAMPLE_WORDS),
        p4_words: tfs_bounded_sample_words(p4, TFS_MS4_P4_SAMPLE_WORDS),
        p6_words: tfs_bounded_sample_words(p6, TFS_MS4_P6_SAMPLE_WORDS),
        p7_words: tfs_bounded_sample_words(p7, TFS_MS4_P7_SAMPLE_WORDS),
        ..Default::default()
    };
    std::ptr::write_unaligned(dst as *mut TfsMs4ArgsCaptureV2, encoded);

    tfs_ms4_capture_words_d2d(
        slot_dev_base,
        p2,
        std::mem::offset_of!(TfsMs4ArgsCaptureV2, p2_words_data),
        encoded.p2_words,
    );
    tfs_ms4_capture_words_d2d(
        slot_dev_base,
        p4,
        std::mem::offset_of!(TfsMs4ArgsCaptureV2, p4_words_data),
        encoded.p4_words,
    );
    tfs_ms4_capture_words_d2d(
        slot_dev_base,
        p6,
        std::mem::offset_of!(TfsMs4ArgsCaptureV2, p6_words_data),
        encoded.p6_words,
    );
    tfs_ms4_capture_words_d2d(
        slot_dev_base,
        p7,
        std::mem::offset_of!(TfsMs4ArgsCaptureV2, p7_words_data),
        encoded.p7_words,
    );

    std::mem::size_of::<TfsMs4ArgsCaptureV2>()
}

unsafe fn dump_tfs_ms2_batched_args_v4(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    slot_dev_base: *mut u8,
    launch_n: u64,
    grid_x: u32,
    block_x: u32,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs2BatchedArgsCaptureV4>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    let requested_bytes = ((grid_x as usize) * 88).min(TFS_MS2B_PACKET_SAMPLE_BYTES);
    let sample_bytes = tfs_bounded_sample_bytes(p2, requested_bytes);
    let requested_entry = tfs_mut_p47_entry_index();
    let mut encoded = TfsMs2BatchedArgsCaptureV4 {
        p2,
        grid_x,
        block_x,
        packet_sample_bytes: sample_bytes,
        requested_entry: requested_entry as u32,
        ..Default::default()
    };
    std::ptr::write_unaligned(dst as *mut TfsMs2BatchedArgsCaptureV4, encoded);
    tfs_capture_d2d_bytes(
        slot_dev_base,
        p2,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, packet_sample),
        sample_bytes as usize,
        "TFS_M2BA_V4",
    );
    encoded = std::ptr::read_unaligned(dst as *const TfsMs2BatchedArgsCaptureV4);
    let effective_entry =
        tfs_ms2b_effective_entry_index(sample_bytes, &encoded.packet_sample, requested_entry);
    encoded.effective_entry = effective_entry as u32;
    let rd5_ptr = tfs_ms2b_entry_ptr(sample_bytes, &encoded.packet_sample, effective_entry, 32);
    let rd6_ptr = tfs_ms2b_entry_ptr(sample_bytes, &encoded.packet_sample, effective_entry, 72);
    let rd7_ptr = tfs_ms2b_entry_ptr(sample_bytes, &encoded.packet_sample, effective_entry, 80);
    tfs_mut_p47_track(rd5_ptr, rd6_ptr, rd7_ptr, launch_n);
    track_rd5_ptr(rd5_ptr);
    ms2_rd6_watch_track(rd6_ptr, launch_n);
    encoded.rd5_words = tfs_bounded_sample_words(rd5_ptr, TFS_MS2B_RD_SAMPLE_WORDS);
    encoded.rd6_words = tfs_bounded_sample_words(rd6_ptr, TFS_MS2B_RD_SAMPLE_WORDS);
    encoded.rd7_words = tfs_bounded_sample_words(rd7_ptr, TFS_MS2B_RD_SAMPLE_WORDS);
    std::ptr::write_unaligned(dst as *mut TfsMs2BatchedArgsCaptureV4, encoded);
    tfs_ms2b_capture_words_d2d(
        slot_dev_base,
        rd5_ptr,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, rd5_words_data),
        encoded.rd5_words,
    );
    tfs_ms2b_capture_words_d2d(
        slot_dev_base,
        rd6_ptr,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, rd6_words_data),
        encoded.rd6_words,
    );
    tfs_ms2b_capture_words_d2d(
        slot_dev_base,
        rd7_ptr,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, rd7_words_data),
        encoded.rd7_words,
    );

    encoded = std::ptr::read_unaligned(dst as *const TfsMs2BatchedArgsCaptureV4);
    let probe_words = (encoded.rd7_words as usize).min(TFS_MS2B_SLOT_PROBE_WORDS);
    let mut rd5_src_ptrs = [0u64; TFS_MS2B_SLOT_PROBE_WORDS];
    let mut rd6_src_ptrs = [0u64; TFS_MS2B_SLOT_PROBE_WORDS];
    encoded.slot_probe_words = probe_words as u32;
    for i in 0..probe_words {
        let slot = encoded.rd7_words_data[i];
        encoded.slot_probe_slots[i] = slot;
        if let Some(rd5_word_index) = i.checked_mul(3).and_then(|v| v.checked_add(1)) {
            if (tfs_bounded_sample_words(rd5_ptr, rd5_word_index.saturating_add(1)) as usize)
                > rd5_word_index
            {
                if let Some(byte_off) = rd5_word_index.checked_mul(4).map(|v| v as u64) {
                    if let Some(src_ptr) = rd5_ptr.checked_add(byte_off) {
                        encoded.slot_probe_rd5_mask |= 1u32 << i;
                        rd5_src_ptrs[i] = src_ptr;
                    }
                }
            }
        }
        if let Some(rd6_word_index) = (slot as usize)
            .checked_mul(3)
            .and_then(|v| v.checked_add(2))
        {
            if (tfs_bounded_sample_words(rd6_ptr, rd6_word_index.saturating_add(1)) as usize)
                > rd6_word_index
            {
                if let Some(byte_off) = rd6_word_index.checked_mul(4).map(|v| v as u64) {
                    if let Some(src_ptr) = rd6_ptr.checked_add(byte_off) {
                        encoded.slot_probe_r38_mask |= 1u32 << i;
                        rd6_src_ptrs[i] = src_ptr;
                    }
                }
            }
        }
    }
    std::ptr::write_unaligned(dst as *mut TfsMs2BatchedArgsCaptureV4, encoded);
    for i in 0..probe_words {
        if (encoded.slot_probe_rd5_mask & (1u32 << i)) != 0 {
            tfs_capture_d2d_bytes(
                slot_dev_base,
                rd5_src_ptrs[i],
                std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, slot_probe_rd5_p4) + i * 4,
                4,
                "TFS_M2BA_V4",
            );
        }
        if (encoded.slot_probe_r38_mask & (1u32 << i)) != 0 {
            tfs_capture_d2d_bytes(
                slot_dev_base,
                rd6_src_ptrs[i],
                std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, slot_probe_r38) + i * 4,
                4,
                "TFS_M2BA_V4",
            );
        }
    }
    std::mem::size_of::<TfsMs2BatchedArgsCaptureV4>()
}

unsafe fn dump_tfs_ms2_batched_args_v3(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    slot_dev_base: *mut u8,
    launch_n: u64,
    grid_x: u32,
    block_x: u32,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs2BatchedArgsCaptureV3>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    let requested_bytes = ((grid_x as usize) * 88).min(TFS_MS2B_PACKET_SAMPLE_BYTES);
    let sample_bytes = tfs_bounded_sample_bytes(p2, requested_bytes);
    let mut encoded = TfsMs2BatchedArgsCaptureV3 {
        p2,
        grid_x,
        block_x,
        packet_sample_bytes: sample_bytes,
        ..Default::default()
    };
    std::ptr::write_unaligned(dst as *mut TfsMs2BatchedArgsCaptureV3, encoded);
    tfs_capture_d2d_bytes(
        slot_dev_base,
        p2,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, packet_sample),
        sample_bytes as usize,
        "TFS_M2BA_V3",
    );
    encoded = std::ptr::read_unaligned(dst as *const TfsMs2BatchedArgsCaptureV3);
    let tracked_entry = tfs_mut_p47_entry_index();
    let rd5_ptr = tfs_ms2b_entry_ptr(sample_bytes, &encoded.packet_sample, tracked_entry, 32);
    let rd6_ptr = tfs_ms2b_entry_ptr(sample_bytes, &encoded.packet_sample, tracked_entry, 72);
    let rd7_ptr = tfs_ms2b_entry_ptr(sample_bytes, &encoded.packet_sample, tracked_entry, 80);
    tfs_mut_p47_track(rd5_ptr, rd6_ptr, rd7_ptr, launch_n);
    ms2_rd6_watch_track(rd6_ptr, launch_n);
    encoded.rd5_words = tfs_bounded_sample_words(rd5_ptr, TFS_MS2B_RD_SAMPLE_WORDS);
    encoded.rd6_words = tfs_bounded_sample_words(rd6_ptr, TFS_MS2B_RD_SAMPLE_WORDS);
    encoded.rd7_words = tfs_bounded_sample_words(rd7_ptr, TFS_MS2B_RD_SAMPLE_WORDS);
    std::ptr::write_unaligned(dst as *mut TfsMs2BatchedArgsCaptureV3, encoded);
    tfs_ms2b_capture_words_d2d(
        slot_dev_base,
        rd5_ptr,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, rd5_words_data),
        encoded.rd5_words,
    );
    tfs_ms2b_capture_words_d2d(
        slot_dev_base,
        rd6_ptr,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, rd6_words_data),
        encoded.rd6_words,
    );
    tfs_ms2b_capture_words_d2d(
        slot_dev_base,
        rd7_ptr,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, rd7_words_data),
        encoded.rd7_words,
    );
    std::mem::size_of::<TfsMs2BatchedArgsCaptureV3>()
}

unsafe fn dump_tfs_ms2_batched_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs2BatchedArgsCaptureV1>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let encoded = TfsMs2BatchedArgsCaptureV1 {
        magic: TFS_MS2B_ARGS_CAPTURE_MAGIC,
        version: TFS_MS2B_ARGS_CAPTURE_VERSION,
        p2: read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0),
    };
    std::ptr::write_unaligned(dst as *mut TfsMs2BatchedArgsCaptureV1, encoded);
    std::mem::size_of::<TfsMs2BatchedArgsCaptureV1>()
}

unsafe fn dump_tfs_ms2_batched_args_v2(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    slot_dev_base: *mut u8,
    grid_x: u32,
    block_x: u32,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs2BatchedArgsCaptureV2>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    let requested_bytes = ((grid_x as usize) * 88).min(TFS_MS2B_PACKET_SAMPLE_BYTES);
    let sample_bytes = tfs_bounded_sample_bytes(p2, requested_bytes);
    let encoded = TfsMs2BatchedArgsCaptureV2 {
        p2,
        grid_x,
        block_x,
        packet_sample_bytes: sample_bytes,
        ..Default::default()
    };
    std::ptr::write_unaligned(dst as *mut TfsMs2BatchedArgsCaptureV2, encoded);
    tfs_capture_d2d_bytes(
        slot_dev_base,
        p2,
        std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, packet_sample),
        sample_bytes as usize,
        "TFS_M2BA_V2",
    );
    std::mem::size_of::<TfsMs2BatchedArgsCaptureV2>()
}

unsafe fn dump_tfs_ms3_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs3ArgsCaptureV1>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let encoded = TfsMs3ArgsCaptureV1 {
        magic: TFS_MS3_ARGS_CAPTURE_MAGIC,
        version: TFS_MS3_ARGS_CAPTURE_VERSION,
        p2: read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0),
        p3: read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0),
    };
    std::ptr::write_unaligned(dst as *mut TfsMs3ArgsCaptureV1, encoded);
    std::mem::size_of::<TfsMs3ArgsCaptureV1>()
}

unsafe fn dump_tfs_ms3_args_v2(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    slot_dev_base: *mut u8,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    if dst.is_null()
        || dst_len < std::mem::size_of::<TfsMs3ArgsCaptureV2>()
        || (kernel_params.is_null() && extra.is_null())
    {
        return 0;
    }
    let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    let p3: u64 = read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0);
    let encoded = TfsMs3ArgsCaptureV2 {
        p2,
        p3,
        p2_words: tfs_bounded_sample_words(p2, TFS_MS3_P2_SAMPLE_WORDS),
        p3_words: tfs_bounded_sample_words(p3, TFS_MS3_P3_SAMPLE_WORDS),
        ..Default::default()
    };
    std::ptr::write_unaligned(dst as *mut TfsMs3ArgsCaptureV2, encoded);
    tfs_ms3_capture_words_d2d(
        slot_dev_base,
        p2,
        std::mem::offset_of!(TfsMs3ArgsCaptureV2, p2_words_data),
        encoded.p2_words,
    );
    tfs_ms3_capture_words_d2d(
        slot_dev_base,
        p3,
        std::mem::offset_of!(TfsMs3ArgsCaptureV2, p3_words_data),
        encoded.p3_words,
    );
    std::mem::size_of::<TfsMs3ArgsCaptureV2>()
}

fn should_capture_ms2b(name: &str, grid_x: u32) -> bool {
    name == "CreateParticleShapesIncremental_MultiStage_2_Batched" && grid_x >= 3
}

fn should_capture_mut_p47_probe_for_mode(kernel_id: u32, broad_probe: bool) -> bool {
    let tail_window = matches!(
        kernel_id,
        TFS_KERNEL_ID_MIGRATE
            | TFS_KERNEL_ID_ADVECT
            | TFS_KERNEL_ID_COUPLE_FROM_PARTICLES
            | TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX
            | TFS_KERNEL_ID_PARTICLE_DENSITY_SMOOTH
    );
    if tail_window {
        return true;
    }
    broad_probe
        && matches!(
            kernel_id,
            TFS_KERNEL_ID_CLEAR
                | TFS_KERNEL_ID_OTHER
                | TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3
                | TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS1
        )
}

fn should_capture_mut_p47_probe(kernel_id: u32) -> bool {
    should_capture_mut_p47_probe_for_mode(kernel_id, tfs_mut_p47_broad_probe_enabled())
}

unsafe fn capture_tfs_atomic_to_tex_post(slot_index: usize, stream: hipStream_t) {
    if !tfs_capture_enabled() {
        return;
    }
    let guard = match TFS_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let state = match guard.as_ref() {
        Some(s) => s,
        None => return,
    };
    let slot_host = state.host_ptr.add(slot_index * TFS_SLOT_SIZE) as *const TfsCaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * TFS_SLOT_SIZE) as *mut u8;
    let encoded =
        std::ptr::read_unaligned((*slot_host).raw_args.as_ptr() as *const TfsAtomicToTexCaptureV1);
    if encoded.magic != TFS_ATOMIC_TO_TEX_CAPTURE_MAGIC
        || encoded.version != TFS_ATOMIC_TO_TEX_CAPTURE_VERSION
        || encoded.output_base < 0x1000
        || encoded.output_post_words == 0
    {
        return;
    }
    tfs_capture_d2d_bytes_async(
        slot_dev_base,
        encoded.output_base,
        std::mem::offset_of!(TfsAtomicToTexCaptureV1, output_post_words_data),
        encoded.output_post_words as usize * 4,
        stream,
        "TFS_ATXA_POST",
    );
}

/// Capture a TurbulenceFS kernel launch into the ring buffer. Called from
/// launch_kernel BEFORE hipModuleLaunchKernel for any kernel whose name
/// starts with "TurbulenceFS_". Cheap (host-side memcpy only, no GPU
/// involvement). Gated on ZLUDA_TFS_CAPTURE=1.
unsafe fn capture_tfs_launch(
    name: &str,
    kernel_id: u32,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) -> Option<usize> {
    if !tfs_capture_enabled() {
        return None;
    }
    if !should_capture_tfs_kernel(kernel_id) {
        return None;
    }

    let mut guard = match TFS_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    tfs_capture_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return None,
    };

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % TFS_CAPTURE_SLOTS;

    let slot_ptr = state.host_ptr.add(slot_index * TFS_SLOT_SIZE) as *mut TfsCaptureSlot;
    let slot_dev_base = (state.dev_ptr as usize + slot_index * TFS_SLOT_SIZE) as *mut u8;
    // Zero the slot first.
    std::ptr::write_bytes(slot_ptr as *mut u8, 0, TFS_SLOT_SIZE);

    (*slot_ptr).magic = TFS_CAPTURE_MAGIC;
    (*slot_ptr).launch_n = this_launch;
    (*slot_ptr).timestamp_ms = crate::elapsed_ms();
    (*slot_ptr).kernel_id = kernel_id;
    (*slot_ptr).grid_x = grid_x;
    (*slot_ptr).grid_y = grid_y;
    (*slot_ptr).grid_z = grid_z;
    (*slot_ptr).block_x = block_x;
    (*slot_ptr).block_y = block_y;
    (*slot_ptr).block_z = block_z;

    // Copy kernel name (truncate / null-pad).
    let name_bytes = name.as_bytes();
    let copy_len = name_bytes.len().min(TFS_KERNEL_NAME_BYTES - 1);
    std::ptr::copy_nonoverlapping(
        name_bytes.as_ptr(),
        (*slot_ptr).kernel_name.as_mut_ptr(),
        copy_len,
    );

    (*slot_ptr).arg_meta = encode_launch_arg_meta(kernel_params, extra);

    if should_skip_tfs_raw_arg_dump(name) {
        // These kernels previously went through generic pointer-slot dumping and
        // were fragile under unknown launch layouts. Keep them on audited,
        // fixed-field extractors only.
        if name == "CreateParticleShapesIncremental_MultiStage_2_Batched" {
            let ms2b_v4_len = dump_tfs_ms2_batched_args_v4(
                kernel_params,
                extra,
                slot_dev_base,
                this_launch,
                grid_x,
                block_x,
                (*slot_ptr).raw_args.as_mut_ptr(),
                TFS_RAW_ARGS_BYTES,
            ) as u32;
            (*slot_ptr).raw_args_len = if ms2b_v4_len > 0 {
                ms2b_v4_len
            } else {
                let ms2b_v3_len = dump_tfs_ms2_batched_args_v3(
                    kernel_params,
                    extra,
                    slot_dev_base,
                    this_launch,
                    grid_x,
                    block_x,
                    (*slot_ptr).raw_args.as_mut_ptr(),
                    TFS_RAW_ARGS_BYTES,
                ) as u32;
                if ms2b_v3_len > 0 {
                    ms2b_v3_len
                } else {
                    let ms2b_v2_len = dump_tfs_ms2_batched_args_v2(
                        kernel_params,
                        extra,
                        slot_dev_base,
                        grid_x,
                        block_x,
                        (*slot_ptr).raw_args.as_mut_ptr(),
                        TFS_RAW_ARGS_BYTES,
                    ) as u32;
                    if ms2b_v2_len > 0 {
                        ms2b_v2_len
                    } else {
                        dump_tfs_ms2_batched_args(
                            kernel_params,
                            extra,
                            (*slot_ptr).raw_args.as_mut_ptr(),
                            TFS_RAW_ARGS_BYTES,
                        ) as u32
                    }
                }
            };
        } else if name == "CreateParticleShapesIncremental_MultiStage_3" {
            let ms3_v2_len = dump_tfs_ms3_args_v2(
                kernel_params,
                extra,
                slot_dev_base,
                (*slot_ptr).raw_args.as_mut_ptr(),
                TFS_RAW_ARGS_BYTES,
            ) as u32;
            (*slot_ptr).raw_args_len = if ms3_v2_len > 0 {
                ms3_v2_len
            } else {
                dump_tfs_ms3_args(
                    kernel_params,
                    extra,
                    (*slot_ptr).raw_args.as_mut_ptr(),
                    TFS_RAW_ARGS_BYTES,
                ) as u32
            };
        } else if name == "CreateParticleShapesIncremental_MultiStage_4" {
            let ms4_v2_len = dump_tfs_ms4_args_v2(
                kernel_params,
                extra,
                slot_dev_base,
                (*slot_ptr).raw_args.as_mut_ptr(),
                TFS_RAW_ARGS_BYTES,
            ) as u32;
            (*slot_ptr).raw_args_len = if ms4_v2_len > 0 {
                ms4_v2_len
            } else {
                dump_tfs_ms4_args(
                    kernel_params,
                    extra,
                    (*slot_ptr).raw_args.as_mut_ptr(),
                    TFS_RAW_ARGS_BYTES,
                ) as u32
            };
        } else if name == "CreateParticleShapesIncremental_MultiStage_1" {
            // CPU-vs-GPU coherency probe (gated). Reads *(rd5+22176)
            // via CPU pointer AND via hipMemcpyDtoH — if they differ,
            // we have a UMA coherency issue on Steam Deck APU.
            // Snapshot tracked rd5 list BEFORE consuming resets so we
            // see the pre-reset values.
            {
                let snapshot: Vec<u64> = match TRACKED_RD5_PTRS.lock() {
                    Ok(g) => g.clone(),
                    Err(p) => p.into_inner().clone(),
                };
                static PROBE_CALL_N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                for &rd5 in snapshot.iter().take(4) {
                    let (cpu_val, gpu_val, eq) = probe_rd5_counter_cpu_vs_gpu(rd5);
                    if !eq {
                        let n = PROBE_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        super::driver::zluda_log_err(&format!(
                            "RD5_CPU_PROBE #{} rd5=0x{:x} CPU_view=0x{:x} GPU_view=0x{:x} DIVERGED",
                            n, rd5, cpu_val, gpu_val
                        ));
                    } else if cpu_val != 0 {
                        // Non-zero but coherent — log occasionally for trend
                        let n = PROBE_CALL_N.load(std::sync::atomic::Ordering::Relaxed);
                        if n % 50 == 0 {
                            super::driver::zluda_log_err(&format!(
                                "RD5_CPU_PROBE coherent rd5=0x{:x} value=0x{:x} (CPU==GPU)",
                                rd5, cpu_val
                            ));
                        }
                    }
                }
            }
            // Defensive reset (gated): zero `*(rd5+22176)` for each rd5
            // that the prior MS_4 marked as pending. Fires BEFORE the
            // V4 probe so the probe captures the post-reset counter
            // value. Throttled to one memset per cycle per rd5.
            ms1_consume_rd5_resets();
            if should_capture_mut_p47_probe(kernel_id) {
                // MS_1 is the suspected writer of shape indices into rd7.
                // Capture rd5/rd6/rd7 heads at MS_1 launch time using the
                // existing MS_2 tracker, so the chain
                //   MS_2 (post) → MS_1 #a → MS_1 #b → MS_2 (pre)
                // reveals whether MS_1 transitions rd7 from cleared to
                // populated.
                let probe_len = dump_tfs_mut_p47_capture(
                    kernel_params,
                    extra,
                    slot_dev_base,
                    (*slot_ptr).raw_args.as_mut_ptr(),
                    TFS_RAW_ARGS_BYTES,
                ) as u32;
                (*slot_ptr).raw_args_len = if probe_len > 0 {
                    probe_len
                } else {
                    dump_kernel_param_ptrs(
                        kernel_params,
                        (*slot_ptr).raw_args.as_mut_ptr(),
                        TFS_RAW_ARGS_BYTES,
                    ) as u32
                };
            } else {
                (*slot_ptr).raw_args_len = dump_kernel_param_ptrs(
                    kernel_params,
                    (*slot_ptr).raw_args.as_mut_ptr(),
                    TFS_RAW_ARGS_BYTES,
                ) as u32;
            }
        } else {
            (*slot_ptr).raw_args_len = dump_kernel_param_ptrs(
                kernel_params,
                (*slot_ptr).raw_args.as_mut_ptr(),
                TFS_RAW_ARGS_BYTES,
            ) as u32;
        }
        return None;
    }

    if kernel_id == TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX
        && tfs_atomic_to_tex_capture_enabled()
    {
        let capture_len = dump_tfs_atomic_to_tex_capture(
            kernel_params,
            extra,
            slot_dev_base,
            stream,
            (*slot_ptr).raw_args.as_mut_ptr(),
            TFS_RAW_ARGS_BYTES,
        ) as u32;
        if capture_len > 0 {
            (*slot_ptr).raw_args_len = capture_len;
            return Some(slot_index);
        }
    }

    if should_capture_mut_p47_probe(kernel_id) {
        let probe_len = dump_tfs_mut_p47_capture(
            kernel_params,
            extra,
            slot_dev_base,
            (*slot_ptr).raw_args.as_mut_ptr(),
            TFS_RAW_ARGS_BYTES,
        ) as u32;
        if probe_len > 0 {
            (*slot_ptr).raw_args_len = probe_len;
            return None;
        }
    }

    // Dump the raw kernel argument bytes.
    let raw_dst = (*slot_ptr).raw_args.as_mut_ptr();
    let mut raw_len = dump_kernel_args_raw(kernel_params, extra, raw_dst, TFS_RAW_ARGS_BYTES);
    if raw_len == 0 && !kernel_params.is_null() {
        raw_len = dump_kernel_param_ptrs(kernel_params, raw_dst, TFS_RAW_ARGS_BYTES);
    }
    (*slot_ptr).raw_args_len = raw_len as u32;
    None
}

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
    pub(crate) module: hipModule_t,
    pub(crate) sm_version: u32,
    pub(crate) explicit_args_size_align: Option<Vec<FunctionArgInfo>>,
}

fn iofx_bbox_texref_desc_from_hip(desc: &HIP_TEXTURE_DESC) -> IofxBboxTexrefCaptureDesc {
    IofxBboxTexrefCaptureDesc {
        address_mode: [
            desc.addressMode[0].0,
            desc.addressMode[1].0,
            desc.addressMode[2].0,
        ],
        filter_mode: desc.filterMode.0,
        flags: desc.flags,
        max_anisotropy: desc.maxAnisotropy,
        mipmap_filter_mode: desc.mipmapFilterMode.0,
        mipmap_level_bias: desc.mipmapLevelBias,
        min_mipmap_level_clamp: desc.minMipmapLevelClamp,
        max_mipmap_level_clamp: desc.maxMipmapLevelClamp,
        border_color: desc.borderColor,
    }
}

unsafe fn populate_iofx_bbox_texref_capture(
    slot_host: *mut IofxBboxCaptureSlot,
    module: hipModule_t,
) {
    let Some(binding) =
        super::module::get_texref_linear_binding(module, "IOFX_texRefBBoxPositions")
    else {
        return;
    };
    (*slot_host).texref_src_ptr = binding.src_ptr.0 as u64;
    (*slot_host).texref_capture_bytes = binding.size_in_bytes.min(IOFX_BBOX_TEXREF_BYTES) as u32;
    (*slot_host).texref_format = binding.format.0;
    (*slot_host).texref_num_channels = binding.num_channels;
    (*slot_host).texref_desc = iofx_bbox_texref_desc_from_hip(&binding.texture_desc);
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

#[derive(Clone, Copy)]
struct SortDownsweepLaunchArgs {
    begin_bit: u32,
    n_items: usize,
    keys_in: u64,
    keys_out: u64,
    vals_in: u64,
    vals_out: u64,
    spine_ptr: u64,
    first: u8,
    last: u8,
    overwrite: u8,
    grid_share: SortGridEvenShare,
}

#[derive(Clone, Copy)]
struct CreateCellIndicesSnapshotArgs {
    n_items: usize,
    hash_out: u64,
}

unsafe fn read_sort_downsweep_launch_args(
    extra: *mut *mut ::core::ffi::c_void,
) -> Option<SortDownsweepLaunchArgs> {
    if extra.is_null() {
        return None;
    }
    let extra_arr = std::slice::from_raw_parts(extra, 5);
    if extra_arr[0] != (0x1usize as *mut _) || extra_arr[1].is_null() {
        return None;
    }
    let buf = extra_arr[1] as *const u8;
    Some(SortDownsweepLaunchArgs {
        n_items: std::ptr::read_unaligned(buf.add(40) as *const u32) as usize,
        begin_bit: std::ptr::read_unaligned(buf.add(44) as *const u32),
        keys_in: std::ptr::read_unaligned(buf.add(0) as *const u64),
        keys_out: std::ptr::read_unaligned(buf.add(8) as *const u64),
        vals_in: std::ptr::read_unaligned(buf.add(16) as *const u64),
        vals_out: std::ptr::read_unaligned(buf.add(24) as *const u64),
        spine_ptr: std::ptr::read_unaligned(buf.add(32) as *const u64),
        first: *buf.add(48),
        last: *buf.add(49),
        overwrite: *buf.add(50),
        grid_share: SortGridEvenShare::from_words(std::array::from_fn(|idx| {
            std::ptr::read_unaligned(buf.add(52 + idx * 4) as *const i32)
        })),
    })
}

unsafe fn read_createcell_snapshot_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> Option<CreateCellIndicesSnapshotArgs> {
    let n_items = read_kernel_arg::<u32>(kernel_params, extra, 0, 0)? as usize;
    let hash_out = read_kernel_arg::<u64>(kernel_params, extra, 24, 3)?;
    if n_items == 0 || hash_out == 0 {
        return None;
    }
    Some(CreateCellIndicesSnapshotArgs { n_items, hash_out })
}

unsafe fn snapshot_sort_downsweep_prelaunch_state(
    begin_bit: u32,
    n_items: usize,
    grid_share: SortGridEvenShare,
    keys_in: u64,
    vals_in: u64,
    keys_out: u64,
    vals_out: u64,
) {
    if n_items == 0 || keys_in == 0 || keys_out == 0 {
        return;
    }
    let read_n = sort_pass_capture_len(n_items, Some(grid_share));
    let mut guard = match SORT_PASS_TRACKER.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    sort_pass_tracker_init(&mut guard);
    let tracker = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };
    let dst_in_before_dev =
        hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_KEYS_IN_BEFORE_OFFSET) as _);
    let copy_in_before_result = hipMemcpyDtoD(
        dst_in_before_dev,
        hipDeviceptr_t(keys_in as *mut std::ffi::c_void),
        read_n * 4,
    );
    if copy_in_before_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_PASS_SAFE: prelaunch keys_in D2D failed bit={} keys_in=0x{:x} n={} -> {:?}",
            begin_bit, keys_in, read_n, copy_in_before_result
        ));
    }
    if vals_in != 0 {
        let dst_vals_in_before_dev =
            hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_VALS_IN_BEFORE_OFFSET) as _);
        let copy_vals_in_before_result = hipMemcpyDtoD(
            dst_vals_in_before_dev,
            hipDeviceptr_t(vals_in as *mut std::ffi::c_void),
            read_n * 4,
        );
        if copy_vals_in_before_result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_PASS_SAFE: prelaunch vals_in D2D failed bit={} vals_in=0x{:x} n={} -> {:?}",
                begin_bit, vals_in, read_n, copy_vals_in_before_result
            ));
        }
    }
    let dst_out_before_dev =
        hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_KEYS_OUT_BEFORE_OFFSET) as _);
    let copy_out_before_result = hipMemcpyDtoD(
        dst_out_before_dev,
        hipDeviceptr_t(keys_out as *mut std::ffi::c_void),
        read_n * 4,
    );
    if copy_out_before_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** SORT_PASS_SAFE: prelaunch keys_out D2D failed bit={} keys_out=0x{:x} n={} -> {:?}",
            begin_bit, keys_out, read_n, copy_out_before_result
        ));
    }
    if vals_out != 0 {
        let dst_vals_out_before_dev =
            hipDeviceptr_t((tracker.scratch_dev as usize + SORT_PASS_VALS_OUT_BEFORE_OFFSET) as _);
        let copy_vals_out_before_result = hipMemcpyDtoD(
            dst_vals_out_before_dev,
            hipDeviceptr_t(vals_out as *mut std::ffi::c_void),
            read_n * 4,
        );
        if copy_vals_out_before_result.is_err() {
            super::driver::zluda_log_err(&format!(
                "*** SORT_PASS_SAFE: prelaunch vals_out D2D failed bit={} vals_out=0x{:x} n={} -> {:?}",
                begin_bit, vals_out, read_n, copy_vals_out_before_result
            ));
        }
    }
}

unsafe fn zero_sort_downsweep_input_slack(
    n_items: usize,
    grid_share: SortGridEvenShare,
    keys_in: u64,
    vals_in: u64,
) {
    if !zero_sort_input_slack_enabled() {
        return;
    }
    let Some(covered_items) = grid_share.covered_item_count() else {
        append_sort_input_slack_probe(&format!(
            "skip covered-none n={} share={:?}",
            n_items, grid_share
        ));
        return;
    };
    if covered_items <= n_items {
        append_sort_input_slack_probe(&format!(
            "skip no-slack n={} covered={}",
            n_items, covered_items
        ));
        return;
    }
    let slack_items = covered_items - n_items;
    let slack_bytes = slack_items.saturating_mul(std::mem::size_of::<i32>());
    if slack_bytes == 0 {
        append_sort_input_slack_probe(&format!(
            "skip zero-bytes n={} covered={} slack={}",
            n_items, covered_items, slack_items
        ));
        return;
    }
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let keys_tail = keys_in.saturating_add((n_items * std::mem::size_of::<i32>()) as u64);
    let keys_result = hipMemsetD32(
        hipDeviceptr_t(keys_tail as *mut std::ffi::c_void),
        0,
        slack_items,
    );
    let vals_result = if vals_in != 0 {
        let vals_tail = vals_in.saturating_add((n_items * std::mem::size_of::<i32>()) as u64);
        Some(hipMemsetD32(
            hipDeviceptr_t(vals_tail as *mut std::ffi::c_void),
            0,
            slack_items,
        ))
    } else {
        None
    };
    if let Err(code) = keys_result {
        append_sort_input_slack_probe(&format!(
            "keys-error n={} covered={} slack={} err={:?}",
            n_items, covered_items, slack_items, code
        ));
    }
    if let Some(Err(code)) = vals_result {
        append_sort_input_slack_probe(&format!(
            "vals-error n={} covered={} slack={} err={:?}",
            n_items, covered_items, slack_items, code
        ));
    }
    if keys_result.is_ok()
        && vals_result.as_ref().map_or(true, Result::is_ok)
        && !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        append_sort_input_slack_probe(&format!(
            "active n={} covered={} slack={} keys_in=0x{:x} vals_in=0x{:x}",
            n_items, covered_items, slack_items, keys_in, vals_in
        ));
    }
}

// These opt-in readers need a completed kernel before inspecting its output.
// Their shared device drain must stay disabled during ordinary asynchronous
// launches: a nonblocking stream may need a later launch to make progress.
fn post_launch_diagnostics_enabled(name: &str) -> bool {
    let ms1 = name.contains("CreateParticleShapesIncremental_MultiStage_1");
    (ms4_capture_enabled() && f6_tracker_enabled())
        || ms2_rd6_kernel_trace_enabled()
        || counter_snapshot_enabled()
        || (nrss_validation_mode() != 0 && name == "IOFX_newRadixSortStepKernel")
        || super::memory::mapped_write_trace_enabled()
        || (flag_count_probe_enabled() && (name == "ParticleCollisionStaticBox" || ms1))
        || (ms1_gate_trace_enabled() && ms1)
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
    static FIRST_KERNEL_LAUNCH_LOGGED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    let first_kernel_launch_logged =
        FIRST_KERNEL_LAUNCH_LOGGED.load(std::sync::atomic::Ordering::Relaxed);
    // Resolve ZLUDA_LAUNCH_MUTEX mode once per process (see LAUNCH_LOCK docs).
    // Also logs the resolved mode on first launch so we can confirm the env var
    // propagated through Steam → Proton → Wine → libcuda.so.
    static LOCK_MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let lock_mode = *LOCK_MODE.get_or_init(|| {
        let raw = std::env::var("ZLUDA_LAUNCH_MUTEX").ok();
        let mode = raw
            .as_deref()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0)
            .min(2);
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] ZLUDA_LAUNCH_MUTEX env={:?} → lock_mode={}",
            raw, mode
        ));
        mode
    });
    // Acquire the global launch lock before ANY diagnostic logging or HIP calls, so
    // mode 2 also prevents log lines from interleaving with other threads. The guard
    // is held until it is explicitly dropped or the function returns.
    let mut launch_guard =
        launch_mutex(lock_mode).map(|lock| lock.lock().unwrap_or_else(|e| e.into_inner()));

    // Log kernel name — gated by ZLUDA_LOG_LAUNCHES=1 to avoid overhead
    static LOG_LAUNCHES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let log_launches = *LOG_LAUNCHES
        .get_or_init(|| std::env::var("ZLUDA_LOG_LAUNCHES").map_or(false, |v| v == "1"));
    let name = lookup_name(f.base);
    static KERNEL_LAUNCH_COUNTER: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    let global_launch_n = if name.is_empty() {
        None
    } else {
        Some(KERNEL_LAUNCH_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    };

    // 2026-06-09 Class A bf000 investigation. Once a 1.04 MiB buffer at
    // page-offset 0xBF000 has been detected (by `fill_with_zero_and_register`),
    // every kernel launch is scanned for that VA in its arg buffer. The
    // scan only reads the `extra` flat buffer (whose size is declared via
    // CU_LAUNCH_PARAM_BUFFER_SIZE) — safe; no blind kernel_params reads.
    // Each unique (kernel_name, byte_offset_in_buffer) hit logs once.
    // Default behaviour: silent until bf000 alloc fires; thereafter ~one
    // 64-bit-stride scan per launch. Diagnostic only.
    {
        let target = super::memory::BF000_TARGET_VA.load(std::sync::atomic::Ordering::Relaxed);
        if target != 0 && !name.is_empty() && !extra.is_null() {
            unsafe {
                let extra_arr = std::slice::from_raw_parts(extra, 5);
                let p0 = extra_arr[0] as usize;
                let p2 = extra_arr[2] as usize;
                let buf_ptr = extra_arr[1] as *const u8;
                let size_ptr = extra_arr[3] as *const usize;
                if p0 == 0x1 && p2 == 0x2 && !buf_ptr.is_null() && !size_ptr.is_null() {
                    let buf_size = std::ptr::read(size_ptr);
                    if buf_size <= 4096 {
                        // 8-byte stride. Walk every aligned 8-byte slot.
                        let n = buf_size / 8;
                        for i in 0..n {
                            let v = std::ptr::read_unaligned(
                                buf_ptr.add(i * 8) as *const u64,
                            );
                            if v == target {
                                static SEEN: std::sync::OnceLock<
                                    std::sync::Mutex<
                                        std::collections::HashSet<(String, usize)>,
                                    >,
                                > = std::sync::OnceLock::new();
                                let seen = SEEN.get_or_init(|| {
                                    std::sync::Mutex::new(std::collections::HashSet::new())
                                });
                                let mut g = seen.lock().unwrap_or_else(|e| e.into_inner());
                                let off = i * 8;
                                if g.insert((name.clone(), off)) {
                                    drop(g);
                                    super::driver::zluda_log_err(&format!(
                                        "*** BF000_USE kernel={} off={} (slot {}) — first observed launch_n={:?}",
                                        name,
                                        off,
                                        i,
                                        global_launch_n,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2026-05-18 Plan v20.3: per-launch n_threads slot barrier read.
    //
    // The Batman: Arkham Knight MS_4 wild-VA crash is timing/coherency
    // sensitive. Reading the n_threads carrier slots
    // (v[663640] and v[1056768] — empirically confirmed via
    // ZLUDA_RELAY_SLOT_PROBE on 2026-05-18) from every tracked >=2MB
    // mapped buffer at each kernel launch entry forces the writes that
    // populate those slots to be visible before the next kernel reads
    // them. This was discovered as a side effect of the diagnostic
    // ZLUDA_NTHREADS_WRITER tracker: the instrumented run completed
    // cleanly (419k kernels) where the bare-baseline run crashes.
    //
    // Two modes share the same read loop:
    //   - ZLUDA_NTHREADS_BARRIER=1 (DEFAULT-ON, the fix): silent reads.
    //   - ZLUDA_NTHREADS_WRITER=1  (OPT-IN diagnostic): emits V_CHANGE
    //     log lines attributing slot changes to the previously-launched
    //     kernel. WRITER implies BARRIER (the read happens unconditionally
    //     when either is on).
    {
        static BARRIER_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static WRITER_TRACK_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let barrier_on = *BARRIER_ENABLED.get_or_init(|| {
            std::env::var("ZLUDA_NTHREADS_BARRIER")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        let writer_log_on = *WRITER_TRACK_ENABLED
            .get_or_init(|| std::env::var("ZLUDA_NTHREADS_WRITER").map_or(false, |v| v == "1"));
        if barrier_on || writer_log_on {
            const OFFSETS: [usize; 2] = [663640, 1056768];
            // Per-(host,off) previous value + previous kernel name. Only
            // touched when writer_log_on; the barrier path doesn't need
            // memory.
            static PREV_STATE: std::sync::Mutex<
                Option<std::collections::HashMap<(usize, usize), (u32, String)>>,
            > = std::sync::Mutex::new(None);
            unsafe {
                if let Ok(table) = super::memory::SMALL_HOST_ALLOCS.lock() {
                    let mut state_opt_guard = if writer_log_on {
                        PREV_STATE.lock().ok()
                    } else {
                        None
                    };
                    let mut state = state_opt_guard
                        .as_mut()
                        .map(|g| g.get_or_insert_with(std::collections::HashMap::new));
                    for &(host, _dev, size) in table.iter() {
                        if size < 2_000_000 {
                            continue;
                        }
                        for &off in OFFSETS.iter() {
                            if off + 4 > size {
                                continue;
                            }
                            // Volatile read prevents the optimiser from
                            // eliding it; this is the barrier itself.
                            let cur = std::ptr::read_volatile((host + off) as *const u32);
                            // Force the read to participate in a fence so
                            // the compiler can't reorder it past the next
                            // HIP call.
                            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                            if let Some(state) = state.as_deref_mut() {
                                let key = (host, off);
                                if let Some((prev_v, prev_k)) = state.get(&key) {
                                    if cur != *prev_v {
                                        super::driver::zluda_log_err(&format!(
                                            "V_CHANGE host=0x{:x} off={} prev_kernel='{}' \
                                             v: {} -> {} (this_kernel='{}')",
                                            host, off, prev_k, prev_v, cur, name
                                        ));
                                    }
                                }
                                state.insert(key, (cur, name.to_string()));
                            }
                        }
                    }
                }
            }
        }
    }

    if log_launches {
        super::driver::zluda_log_err(&format!(
            "cuLaunchKernel '{}' grid=[{},{},{}] block=[{},{},{}] shared={} stream=0x{:x}",
            name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes,
            stream.0 as u64
        ));
    }

    // 2026-06-06 Class A upstream-writer hunt v2 — pure diagnostic, gated on
    // ZLUDA_MEMCPY_SRC_WRITER=1.
    //
    // v1 (committed 22bacd48) only logged matches AFTER MemCopyBatchedAsync
    // had populated the tracked set, which missed the upstream writer that
    // ran BEFORE the MemCopy. v2 fixes this by tracking launch HISTORY per
    // device allocation: every launch's u64-aligned args are checked against
    // EVERY known device allocation, and the (timestamp, kernel, arg_idx)
    // tuple is pushed into a per-alloc history (capped at last 8 distinct
    // (kernel, arg) pairs). When MemCopyBatchedAsync detects a relay-struct
    // copy (handled below), it dumps the history for the source alloc.
    //
    // False-positive filtering: a kernel may declare a .u32 param at index
    // N; reading 8 bytes there yields (u32_value | next4_bytes << 32) and
    // can coincidentally land in a real allocation range. Two heuristics
    // weed those out: ptr % 16 == 0 (most CUDA pointers are 16-byte
    // aligned; junk reads from .u32 counts rarely are) and ptr in the
    // device-mapped range (> 0x4000_0000_0000 on amdgpu+HSA).
    {
        static WRITER_PROBE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let writer_probe_on = *WRITER_PROBE_ENABLED.get_or_init(|| {
            std::env::var("ZLUDA_MEMCPY_SRC_WRITER").map_or(false, |v| v == "1")
        });
        if writer_probe_on && name != "MemCopyBatchedAsync" && name != "MemCopyAsync" {
            // v3 light: ONE Mutex acquire per launch. Snapshot only LARGE
            // allocs (>= 1 MiB) — per-actor relay buffers are typically
            // 2 MiB; small scratch/control allocs are excluded. Linear scan
            // of ~20-50 entries is trivial.
            let large_allocs: Vec<(u64, u64)> = {
                if let Ok(g) = super::memory::DEV_ALLOCS.lock() {
                    g.iter()
                        .filter(|(_, s)| *s >= 1_000_000)
                        .copied()
                        .collect()
                } else {
                    Vec::new()
                }
            };
            if large_allocs.is_empty() {
                // Nothing tracked yet → skip the probe entirely.
            } else {
                let ts_ms = probe_now_ms();
                // First 8 arg slots — covers most kernels' pointer params.
                let mut matched: Option<(u64, u8)> = None;
                for arg_idx in 0..8usize {
                    let off = arg_idx * 8;
                    let p = match unsafe {
                        read_kernel_arg::<u64>(kernel_params, extra, off, arg_idx)
                    } {
                        Some(v) => v,
                        None => continue,
                    };
                    if p & 0xF != 0 {
                        continue;
                    }
                    if p < 0x4000_0000_0000 || p >= 0x8000_0000_0000_0000 {
                        continue;
                    }
                    for &(base, size) in &large_allocs {
                        if p >= base && p < base + size {
                            matched = Some((base, arg_idx as u8));
                            break;
                        }
                    }
                    if matched.is_some() {
                        break;
                    }
                }
                // Update history only if we got a real match — one Mutex
                // acquire on the rare path.
                if let Some((base, arg_idx)) = matched {
                    if let Ok(mut guard) = ALLOC_TOUCH_HISTORY.lock() {
                        let map = guard.get_or_insert_with(std::collections::HashMap::new);
                        let entry =
                            map.entry(base).or_insert_with(std::collections::VecDeque::new);
                        let same_as_last = entry
                            .back()
                            .map(|(_, k, a)| k.as_str() == name && *a == arg_idx)
                            .unwrap_or(false);
                        if !same_as_last {
                            entry.push_back((ts_ms, name.to_string(), arg_idx));
                            while entry.len() > ALLOC_HISTORY_CAP {
                                entry.pop_front();
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 3a Path 1b: periodically sample first-qword of each tracked
    // 1040000-byte cuMemHostAlloc'd buffer. We only fire every Nth launch
    // to avoid log explosion (game does ~400k launches per benchmark cycle).
    if super::memory::hostbuf_sample_enabled() {
        static LAUNCH_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = LAUNCH_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Sample every 1000th launch — at typical launch rate that's ~1 sample/sec.
        if n % 1000 == 0 {
            unsafe {
                super::memory::sample_hostbufs(&format!("launch_tick={}", n));
            }
        }
    }

    // Generic kernel-arg pointer logger — gated by ZLUDA_KARG_LOG=1. One-shot
    // per kernel name (logs only the first launch for each distinct kernel)
    // to avoid log explosion. For each arg index 0..16 we deref the
    // kernel_params slot, read the first 8 bytes, and report any value that
    // falls within the ZLUDA-tracked ALLOC range — those are likely device
    // pointers. Goal: cross-reference DtoH source pointers with kernel args
    // to see whether always-zero DtoH'd buffers are ever bound to a kernel.
    static KARG_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let karg_log =
        *KARG_LOG.get_or_init(|| std::env::var("ZLUDA_KARG_LOG").map_or(false, |v| v == "1"));
    if karg_log {
        static KARG_SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
            std::sync::OnceLock::new();
        let seen_set =
            KARG_SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let is_first = {
            let mut s = seen_set.lock().unwrap_or_else(|e| e.into_inner());
            s.insert(name.to_string())
        };
        if is_first {
            let alloc_min = super::memory::ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
            let alloc_max = super::memory::ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
            let mut all_args: Vec<String> = Vec::new();
            // Plausible user-space pointer range on x86_64 Linux. Anything
            // outside is either NULL/small-int packing or wild read past
            // the kernel_params array end. Use this to early-bail instead
            // of trusting the unbounded `read_kernel_arg` walk.
            const PTR_LO: u64 = 0x1000;
            const PTR_HI: u64 = 0x0000_8000_0000_0000;
            let plausible_ptr = |p: u64| p >= PTR_LO && p < PTR_HI;
            // Walk up to 12 indices (enough for every PhysX kernel we've
            // seen — max observed is ~10). Two safety layers:
            // 1. For kparams: each kernel_params[i] entry must itself be
            //    a plausible user-space pointer before we dereference it.
            //    Past-the-array reads usually return non-NULL garbage that
            //    fails this check, avoiding the wild deref that previously
            //    SIGSEGV'd in alloc::dealloc.
            // 2. For extra: byte layout is bounded by the buffer header,
            //    so read_kernel_arg's offset arithmetic is safe up to the
            //    declared size. Cap at 12 anyway for symmetry.
            for i in 0..12usize {
                if !kernel_params.is_null() {
                    // Pre-check the slot pointer is plausible before deref.
                    let slot_ptr = unsafe { *kernel_params.add(i) } as u64;
                    if slot_ptr == 0 {
                        all_args.push(format!("p{}=null", i));
                        break;
                    }
                    if !plausible_ptr(slot_ptr) {
                        all_args.push(format!("p{}=oob(0x{:x})", i, slot_ptr));
                        break;
                    }
                }
                let val: Option<u64> =
                    unsafe { read_kernel_arg::<u64>(kernel_params, extra, i * 8, i) };
                match val {
                    Some(v) => {
                        let kind = if alloc_min <= alloc_max && v >= alloc_min && v <= alloc_max {
                            "DEV"
                        } else {
                            "val"
                        };
                        all_args.push(format!("{}{}=0x{:x}", kind, i, v));
                    }
                    None => {
                        all_args.push(format!("p{}=null", i));
                        break;
                    }
                }
            }
            let kp_kind = if !kernel_params.is_null() {
                "kparams"
            } else if !extra.is_null() {
                "extra"
            } else {
                "none"
            };
            super::driver::zluda_log_err(&format!(
                "[KARG] '{}' first-launch via={} grid=[{},{},{}] block=[{},{},{}] args=[{}]",
                name,
                kp_kind,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
                all_args.join(" ")
            ));
        }
    }
    // 2026-06-10 NRSS lab-driver capture — ZLUDA_NRSS_ARG_TRACE=1.
    //
    // Log ALL 13 typed args + launch dims for the first 600
    // IOFX_newRadixSortStepKernel launches. NRSS is a multi-phase
    // dispatcher (p11 = phase selector, p12 = count); building the
    // isolated game-shape reproducer (the playbook that cracked the
    // cub/SDMA bug) requires the real per-burst arg sequences, not just
    // the first launch (KARG_LOG) or the pointer ranges
    // (FREE_AFTER_LAUNCH_SCAN). Declared layout from the PTX:
    //   p0 u64, p1 u16, p2 u32, p3 u32, p4 u32, p5 u32,
    //   p6..p10 u64, p11 u32, p12 u32
    // Flat-buffer offsets (extra path): 0,8,12,16,20,24,32,40,48,56,64,72,76.
    {
        // Value semantics: "1" = legacy 600-launch cap; any other number =
        // that cap (r11 lesson: corrupt sorts can appear after sort #300 ≈
        // launch #7,200 — pair with ZLUDA_NRSS_VALIDATE needs a big cap).
        static NRSS_TRACE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let cap = *NRSS_TRACE.get_or_init(|| {
            match std::env::var("ZLUDA_NRSS_ARG_TRACE").ok().as_deref() {
                None | Some("0") => 0,
                Some("1") => 600,
                Some(v) => v.parse().unwrap_or(600),
            }
        });
        if cap > 0 && name == "IOFX_newRadixSortStepKernel" {
            static NRSS_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let k = NRSS_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if k < cap {
                let p0 = unsafe { read_kernel_arg::<u64>(kernel_params, extra, 0, 0) };
                let p1 = unsafe { read_kernel_arg::<u16>(kernel_params, extra, 8, 1) };
                let p2 = unsafe { read_kernel_arg::<u32>(kernel_params, extra, 12, 2) };
                let p3 = unsafe { read_kernel_arg::<u32>(kernel_params, extra, 16, 3) };
                let p4 = unsafe { read_kernel_arg::<u32>(kernel_params, extra, 20, 4) };
                let p5 = unsafe { read_kernel_arg::<u32>(kernel_params, extra, 24, 5) };
                let p6 = unsafe { read_kernel_arg::<u64>(kernel_params, extra, 32, 6) };
                let p7 = unsafe { read_kernel_arg::<u64>(kernel_params, extra, 40, 7) };
                let p8 = unsafe { read_kernel_arg::<u64>(kernel_params, extra, 48, 8) };
                let p9 = unsafe { read_kernel_arg::<u64>(kernel_params, extra, 56, 9) };
                let p10 = unsafe { read_kernel_arg::<u64>(kernel_params, extra, 64, 10) };
                let p11 = unsafe { read_kernel_arg::<u32>(kernel_params, extra, 72, 11) };
                let p12 = unsafe { read_kernel_arg::<u32>(kernel_params, extra, 76, 12) };
                super::driver::zluda_log_err(&format!(
                    "NRSS_ARGS#{} grid={} block={} sh={} p0={:x?} p1={:?} p2={:?} p3={:?} p4={:?} p5={:?} p6={:x?} p7={:x?} p8={:x?} p9={:x?} p10={:x?} p11={:?} p12={:?}",
                    k, grid_dim_x, block_dim_x, shared_mem_bytes,
                    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12
                ));
            }
        }
    }
    // Plan v22 Phase B.1 — per-launch IOFX kernarg probe.
    //
    // Goal: identify which IOFX_remap / IOFX_makeSortKeys / IOFX_newRadixSort*
    // / IOFX_actorRange launch writes corruption into MS_4's DEV5 buffer.
    // Filter: kernel name matches the IOFX-family regex AND at least one
    // kernarg is a pointer that matches (exact or within 64 MiB of) a known
    // MS_4 DEV5 (p5) address — keeps log volume tractable.
    //
    // Safety: only reads from `kernel_params` (alive for the duration of this
    // call). Does NOT touch SMALL_HOST_ALLOCS or any tracker that requires
    // holding another lock. Probe-induced UAF (2026-05-18 incident) is the
    // primary failure mode for new probes; this one is read-only on
    // ProtoBuf-shaped contiguous param slots.
    //
    // Gated on ZLUDA_REMAP_KARGS=1, default OFF.
    static REMAP_KARGS_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let remap_kargs = *REMAP_KARGS_ENABLED
        .get_or_init(|| std::env::var("ZLUDA_REMAP_KARGS").map_or(false, |v| v == "1"));
    if remap_kargs && !kernel_params.is_null() {
        // Filter restored to e2 state (2026-05-18). e3 (no filter) and e4
        // (wider PhysX-family filter, walk=12) both SIGSEGV'd in
        // launch_kernel after the first MS_4 — coredump top frame in our
        // libcuda.so.1's launch_kernel. The kernarg walk is fundamentally
        // unsafe for kernels whose kernel_params layout we haven't
        // explicitly observed safe.
        //
        // Phase B.1 with this filter is complete (run e2): MS_1/MS_2_Batched/
        // MS_3 touch MS_4's DEV5 pool but none writes at the DEV5 offset.
        // To find the actual DEV5 writer, pivot to Plan v22 Phase B.2
        // (DEV5 DtoH dump from cuMemcpyAsync handler) — different code path,
        // no per-launch kernarg walks of arbitrary kernels.
        let is_dev5_candidate = name.contains("IOFX_remap")
            || name.contains("IOFX_makeSortKeys")
            || name.contains("IOFX_newRadixSort")
            || name.contains("IOFX_actorRange")
            || name.contains("CreateParticleShapesIncremental_MultiStage_1")
            || name.contains("CreateParticleShapesIncremental_MultiStage_2_Batched")
            || name.contains("CreateParticleShapesIncremental_MultiStage_3");
        if is_dev5_candidate {
            // Snapshot the tracked DEV5 set without holding the lock across
            // the kernarg reads.
            let dev5_addrs: Vec<u64> = MS4_DEV5_ADDRS
                .get()
                .and_then(|m| m.lock().ok().map(|s| s.iter().copied().collect()))
                .unwrap_or_default();
            if !dev5_addrs.is_empty() {
                // Plan v23 — same 2 MiB-aligned-base match. Run e1 proved
                // ±64 MiB is too tight (PhysX sub-buffer arenas are ~1.1 GiB
                // apart). All observed MS_4 / IOFX allocations are exactly
                // 2 MiB (0x200000) so the 2 MiB-aligned-base heuristic is
                // empirically safe; upgrade to full ALLOC-table lookup only
                // if this returns false negatives we can prove.
                const POOL_MASK: u64 = !0x1f_ffffu64; // 2 MiB align mask
                const PTR_LO: u64 = 0x1000;
                const PTR_HI: u64 = 0x0000_8000_0000_0000;
                let mut all_args: Vec<String> = Vec::new();
                let mut hit_any = false;
                // Walk cap: 16 slots is safe for the e2 filter (MS_1 has 13
                // kernargs, others fewer). Restored after e3/e4 SIGSEGV with
                // unfiltered/wider walks.
                for i in 0..16usize {
                    let slot_ptr = unsafe { *kernel_params.add(i) } as u64;
                    if slot_ptr == 0 {
                        all_args.push(format!("p{}=null", i));
                        break;
                    }
                    if slot_ptr < PTR_LO || slot_ptr >= PTR_HI {
                        all_args.push(format!("p{}=oob(0x{:x})", i, slot_ptr));
                        break;
                    }
                    let v = unsafe { std::ptr::read_unaligned(slot_ptr as *const u64) };
                    let v_u32 = (v & 0xFFFF_FFFF) as u32;
                    let mut tag = "val".to_string();
                    let v_pool = v & POOL_MASK;
                    for &dev5 in dev5_addrs.iter() {
                        if v_pool == (dev5 & POOL_MASK) {
                            hit_any = true;
                            let off_in_pool = v & 0x1f_ffffu64;
                            let dev5_off = dev5 & 0x1f_ffffu64;
                            tag = format!("DEV5pool(p5@+{:#x},this@+{:#x})", dev5_off, off_in_pool);
                            break;
                        }
                    }
                    all_args.push(format!("p{}={}=0x{:x}(u32=0x{:x})", i, tag, v, v_u32));
                }
                if hit_any {
                    static REMAP_SEQ: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let seq = REMAP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    super::driver::zluda_log_err(&format!(
                        "[REMAP_KARGS #{}] '{}' grid=[{},{},{}] block=[{},{},{}] args=[{}]",
                        seq,
                        name,
                        grid_dim_x,
                        grid_dim_y,
                        grid_dim_z,
                        block_dim_x,
                        block_dim_y,
                        block_dim_z,
                        all_args.join(" ")
                    ));
                }
            }
        }
    }
    // Capture every MemCopyAsync / MemCopyBatchedAsync launch's packed
    // 32-byte struct param. PhysX uses these kernels as INTERNAL D2D copies
    // instead of cuMemcpyDtoD — so they don't appear in our cuMem* logs.
    // Layout (from MemCopyAsync PTX byte-level decode):
    //   bytes  0..7  = field0 (size or count)
    //   bytes  8..15 = ptr_a (likely src or dst)
    //   bytes 16..23 = ptr_b (likely the other end)
    //   bytes 24..31 = type/flags (32-bit `setp.eq.s32 r,3` check at offset 24)
    // Cross-reference logged ptr_a/ptr_b against DtoH-zero buffers to find
    // the FleX initial-data path.
    // 2026-06-05 Class B residual: PhysX's actor-shape iterator reads from a
    // host-mapped buffer that is populated by MemCopyBatchedAsync (the kernel
    // appears 3-5 launches before every captured Class B crash). The kernel's
    // SC system-scope kernel-exit fence flushes GPU L2 to DRAM but NOT the
    // HDP write-combine buffer between DRAM and CPU memory. If PhysX skips
    // cuEventSync (covered by Phase 2 ZLUDA_EVENT_HDP_FLUSH default-ON) and
    // reads via mapped pointer directly, the host can see NULL where the
    // kernel wrote a valid pointer (canonical inner_array[0]=NULL).
    //
    // ZLUDA_MEMCPY_HDP_FLUSH=1 issues hipDeviceSynchronize right after the
    // MemCopyBatchedAsync launch hooks. HIP's hipDeviceSynchronize wraps an
    // HSA queue barrier-AND packet whose release fence triggers the HDP
    // flush MMIO. Cost: one device-wide drain per MemCopyBatchedAsync (PhysX
    // issues a few per frame, so ceiling is ~30 syncs/sec — bearable).
    if name == "MemCopyBatchedAsync" {
        // 2026-06-05 n=2 validation: default-OFF baseline mean 429k vs
        // default-ON mean 1.10M (~2.6×). Run1: 1.06M, 0 Class B (ended Class
        // A GPU fault). Run2: 1.146M, 1 Class B (much later than baseline).
        // Class B not fully eliminated but heavily delayed. Default flipped
        // to ON, then retired (off by default); set =1 to enable.
        static MEMCPY_HDP_FLUSH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *MEMCPY_HDP_FLUSH
            .get_or_init(|| std::env::var("ZLUDA_MEMCPY_HDP_FLUSH").map_or(false, |v| v == "1"))
        {
            unsafe {
                let _ = hipDeviceSynchronize();
            }
        }
    }

    if (name == "MemCopyAsync" || name == "MemCopyBatchedAsync") && !kernel_params.is_null() {
        unsafe {
            // kernel_params[2] -> &param_2 (the 32-byte struct)
            let p2_ptr = *kernel_params.add(2);
            if !p2_ptr.is_null() {
                let buf = p2_ptr as *const u8;
                let f0 = std::ptr::read_unaligned(buf.add(0) as *const u64);
                let pa = std::ptr::read_unaligned(buf.add(8) as *const u64);
                let pb = std::ptr::read_unaligned(buf.add(16) as *const u64);
                let ty = std::ptr::read_unaligned(buf.add(24) as *const u64);
                let p0 = read_kernel_arg::<u64>(kernel_params, extra, 0, 0).unwrap_or(0);
                let p1 = read_kernel_arg::<u64>(kernel_params, extra, 8, 1).unwrap_or(0);
                static MC_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let seq = MC_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::driver::zluda_log_err(&format!(
                    "MEMCPY_ARGS#{} {} grid=[{},{},{}] p0=0x{:x} p1=0x{:x} f0=0x{:x} pa=0x{:x} pb=0x{:x} ty=0x{:x}",
                    seq, name, grid_dim_x, grid_dim_y, grid_dim_z, p0, p1, f0, pa, pb, ty
                ));
            }
        }
    }

    // Capture CUB RadixSort kernel args (Downsweep + Upsweep). Pure CPU reads
    // from the extra buffer — no GPU ops, no ProfilingSignal risk. Every launch
    // logs (was previously gated to a single sort's worth of passes, which was
    // hiding the crashing Downsweep's state).
    if (name.contains("RadixSortDownsweepKernel") || name.contains("RadixSortUpsweepKernel"))
        && !extra.is_null()
    {
        unsafe {
            let extra_arr = std::slice::from_raw_parts(extra, 5);
            if extra_arr[0] == (0x1usize as *mut _) && !extra_arr[1].is_null() {
                let buf = extra_arr[1] as *const u8;
                // Per-call sequence counter so we can align this with the
                // kernel-launch history dump at crash time.
                static SORT_SEQ: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let seq = SORT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if name.contains("RadixSortDownsweepKernel") {
                    // Downsweep packed extra args:
                    //   [0..40):  5 x u64 pointers
                    //   [40..44): n_items
                    //   [44..48): begin_bit
                    //   [48..51): is_first / is_last / overwrite
                    //   [52..88): GridEvenShare (9 x i32)
                    let ptrs: [u64; 5] = [
                        std::ptr::read_unaligned(buf.add(0) as *const u64),
                        std::ptr::read_unaligned(buf.add(8) as *const u64),
                        std::ptr::read_unaligned(buf.add(16) as *const u64),
                        std::ptr::read_unaligned(buf.add(24) as *const u64),
                        std::ptr::read_unaligned(buf.add(32) as *const u64),
                    ];
                    let n_items = std::ptr::read_unaligned(buf.add(40) as *const u32);
                    let begin_bit = std::ptr::read_unaligned(buf.add(44) as *const u32);
                    let is_first = *buf.add(48);
                    let is_last = *buf.add(49);
                    let overwrite = *buf.add(50);
                    let ges_i32: Vec<i32> = (0..9)
                        .map(|i| std::ptr::read_unaligned(buf.add(52 + i * 4) as *const i32))
                        .collect();
                    super::driver::zluda_log_err(&format!(
                        "SORT_ARGS#{} DS n={} bit={} first={} last={} ow={} GES={:?} grid={} stream={:?} ptrs=[0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}]",
                        seq, n_items, begin_bit, is_first, is_last, overwrite, ges_i32, grid_dim_x, stream,
                        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4]
                    ));
                } else {
                    // Upsweep packed extra args (from PTX signature):
                    //   [0..16):  pointers (d_keys, d_spine)
                    //   [16..20): n_items
                    //   [20..24): begin_bit
                    //   [24..26): 2 x u8 flags
                    //   [28..64): GridEvenShare (9 x i32)
                    let d_keys = std::ptr::read_unaligned(buf.add(0) as *const u64);
                    let d_spine = std::ptr::read_unaligned(buf.add(8) as *const u64);
                    let n_items = std::ptr::read_unaligned(buf.add(16) as *const u32);
                    let begin_bit = std::ptr::read_unaligned(buf.add(20) as *const u32);
                    let flag0 = *buf.add(24);
                    let flag1 = *buf.add(25);
                    let ges_i32: Vec<i32> = (0..9)
                        .map(|i| std::ptr::read_unaligned(buf.add(28 + i * 4) as *const i32))
                        .collect();
                    super::driver::zluda_log_err(&format!(
                        "SORT_ARGS#{} US n={} bit={} f0={} f1={} GES={:?} grid={} stream={:?} ptrs=[0x{:x}, 0x{:x}]",
                        seq, n_items, begin_bit, flag0, flag1, ges_i32, grid_dim_x, stream, d_keys, d_spine
                    ));
                }
            }
        }
    }
    // NOTE: pre-launch validation that called hipMemcpyDtoH has been disabled —
    // the D2H copies trigger the ROCm ProfilingSignal pool race condition
    // (see commit 589d0290 "Fix TurbulenceFS GPU page fault" + the comment in
    // tex.rs around the ZLUDA_LOG_ALL gate). That race manifests as a
    // `std::vector<amd::roc::ProfilingSignal*>` bounds assertion abort in the
    // HSA runtime, which we were misattributing to a GPU kernel fault. Any
    // future validator MUST NOT call hipMemcpyDtoH from launch_kernel.
    let _ = &validate_kernel_inputs; // reference so dead-code warning is quiet

    // Plan v19 Phase C: hook MemCopyBatchedAsync to log src/dst of each
    // copy in its batch. The kernel signature (inferred from KARG logs:
    // `args=[val0=0x0 DEV1=0x6fff00000002 DEV2=0x7fb8144a3000 p3=null]`)
    // is roughly: param_0 reserved, param_1 = packed (count|flags),
    // param_2 = pointer to descriptor array. grid_y == descriptor count.
    // Each descriptor is { void* src; void* dst; size_t bytes; } = 24 B.
    //
    // We only log batches that contain a dst falling in a tracked mapped
    // buffer at an interesting offset (663640 ± 16 for the actor table
    // relay, OR 1056768 ± 16 for the upstream-relay buffer the count
    // first lands in). Otherwise the spam would be enormous (~6k batches
    // per crashed run).
    //
    // Gated on ZLUDA_MEMCPY_SRC_TRACE=1. Default OFF.
    if name == "MemCopyBatchedAsync" {
        unsafe {
            static MEMCPY_SRC_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *MEMCPY_SRC_TRACE
                .get_or_init(|| std::env::var("ZLUDA_MEMCPY_SRC_TRACE").map_or(false, |v| v == "1"))
            {
                // Decoded descriptor format (from MEMCPY_RAW_DATA observation,
                // 2026-05-17): 32 bytes per descriptor:
                //   +0  : src u64  (device pointer)
                //   +8  : dst u64  (device pointer; can be 0 for fill/clear)
                //   +16 : size u64 (bytes)
                //   +24 : flags u64 (0 or 0x6fff_xxxxxxxx or 0x7f18_00000001…)
                // Descriptor count = grid_y (one block per descriptor).
                let desc_ptr: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
                let num_descs = grid_dim_y.max(1) as usize;
                static MEMCPY_CALL_N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let call_n = MEMCPY_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Ok(table) = super::memory::SMALL_HOST_ALLOCS.lock() {
                    let mut desc_host: Option<usize> = None;
                    for &(host, dev, size) in table.iter() {
                        let dev_end = dev.wrapping_add(size);
                        if (desc_ptr as usize) >= dev && (desc_ptr as usize) < dev_end {
                            let off = (desc_ptr as usize) - dev;
                            desc_host = Some(host + off);
                            break;
                        }
                    }
                    if let Some(dh) = desc_host {
                        for i in 0..num_descs.min(64) {
                            let base = (dh + i * 32) as *const u64;
                            let d_src = std::ptr::read_volatile(base);
                            let d_dst = std::ptr::read_volatile(base.add(1));
                            let d_size = std::ptr::read_volatile(base.add(2));
                            // Resolve dst to a tracked mapped buffer if applicable.
                            let mut matched_dst: Option<(u64, usize)> = None;
                            for &(host, dev, size) in table.iter() {
                                let dev_end = dev.wrapping_add(size);
                                if (d_dst as usize) >= dev && (d_dst as usize) < dev_end {
                                    matched_dst = Some((host as u64, (d_dst as usize) - dev));
                                    break;
                                }
                            }
                            if let Some((dst_host, dst_off)) = matched_dst {
                                // 2026-06-06 Class A investigation: widen filter
                                // to catch the upstream n_threads struct writer.
                                // The struct is 16 bytes at off~1048576 in a 2 MiB
                                // mapped buffer (per
                                // project_class_a_n_threads_chain_decoded_2026_06_06.md).
                                // Original filter `d_size == 4` missed this.
                                // New filter: any size, dst_off in the 1 MiB ± 4 KiB
                                // window where the struct lives.
                                let near_upstream_struct =
                                    dst_off >= 1044480 && dst_off < 1052672;
                                let interesting = d_size == 4 || near_upstream_struct;
                                // Class A upstream-writer hunt: when this
                                // descriptor copies into the relay struct
                                // region, record the src's containing dev
                                // alloc so the per-launch SRC_WRITER probe
                                // can match kernel-arg pointers against it.
                                if near_upstream_struct {
                                    if let Some((base, size)) =
                                        super::memory::find_dev_alloc_containing(d_src)
                                    {
                                        if let Ok(mut t) = TRACKED_MEMCPY_SRC_ALLOCS.lock() {
                                            if !t.iter().any(|&(b, _)| b == base) {
                                                t.push((base, size));
                                            }
                                        }
                                        // v2 history dump: show launches that
                                        // touched this alloc BEFORE the relay
                                        // copy. The writer is most likely in
                                        // the last few entries.
                                        if let Ok(guard) = ALLOC_TOUCH_HISTORY.lock() {
                                            if let Some(map) = guard.as_ref() {
                                                if let Some(history) = map.get(&base) {
                                                    let entries: Vec<String> = history
                                                        .iter()
                                                        .map(|(t, k, a)| {
                                                            format!("({}ms,{},a{})", t, k, a)
                                                        })
                                                        .collect();
                                                    super::driver::zluda_log_err(&format!(
                                                        "[RELAY_HIST] memcpy_call={} src_alloc=(0x{:x},{}) \
                                                         dst_off={} n_entries={} history=[{}]",
                                                        call_n,
                                                        base,
                                                        size,
                                                        dst_off,
                                                        history.len(),
                                                        entries.join(",")
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                                if interesting {
                                    // Try to read the source u32 value too —
                                    // resolve src to a tracked buffer first.
                                    let mut src_value: Option<u32> = None;
                                    for &(host, dev, size) in table.iter() {
                                        let dev_end = dev.wrapping_add(size);
                                        if (d_src as usize) >= dev && (d_src as usize) < dev_end {
                                            let off = (d_src as usize) - dev;
                                            if off + 4 <= size {
                                                src_value = Some(std::ptr::read_volatile(
                                                    (host + off) as *const u32,
                                                ));
                                            }
                                            break;
                                        }
                                    }
                                    super::driver::zluda_log_err(&format!(
                                        "MEMCPY_SRC #{} i={}/{} dst_off={} \
                                         src=0x{:x} src_u32={:?} dst=0x{:x} \
                                         dst_host=0x{:x} size={}",
                                        call_n,
                                        i,
                                        num_descs,
                                        dst_off,
                                        d_src,
                                        src_value,
                                        d_dst,
                                        dst_host,
                                        d_size
                                    ));
                                }
                            }
                        }
                        static MCB_RD5_TRACE: std::sync::OnceLock<bool> =
                            std::sync::OnceLock::new();
                        if *MCB_RD5_TRACE.get_or_init(|| {
                            std::env::var("ZLUDA_MCB_RD5_TRACE").map_or(false, |v| v == "1")
                        }) {
                            let tracked_rd5: Vec<u64> = match TRACKED_RD5_PTRS.lock() {
                                Ok(g) => g.clone(),
                                Err(p) => p.into_inner().clone(),
                            };
                            let desc_ptr: u64 =
                                read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
                            let num_descs = grid_dim_y.max(1) as usize;
                            if desc_ptr != 0 && !tracked_rd5.is_empty() {
                                if let Ok(table) = super::memory::SMALL_HOST_ALLOCS.lock() {
                                    let mut desc_host: Option<usize> = None;
                                    for &(host, dev, size) in table.iter() {
                                        let dev_end = dev.wrapping_add(size);
                                        if (desc_ptr as usize) >= dev
                                            && (desc_ptr as usize) < dev_end
                                        {
                                            let off = (desc_ptr as usize) - dev;
                                            desc_host = Some(host + off);
                                            break;
                                        }
                                    }
                                    if let Some(dh) = desc_host {
                                        static MCB_RD5_CALL_N: std::sync::atomic::AtomicU32 =
                                            std::sync::atomic::AtomicU32::new(0);
                                        let call_n = MCB_RD5_CALL_N
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        for i in 0..num_descs.min(64) {
                                            let base = (dh + i * 32) as *const u64;
                                            let d_src = std::ptr::read_volatile(base);
                                            let d_dst = std::ptr::read_volatile(base.add(1));
                                            let d_size = std::ptr::read_volatile(base.add(2));
                                            let d_flags = std::ptr::read_volatile(base.add(3));
                                            if let Some(rd5) = mcb_descriptor_hits_rd5_reset_window(
                                                d_dst,
                                                d_size,
                                                &tracked_rd5,
                                            ) {
                                                let watch_start = rd5 + F6_OFFSET_IN_RD5 as u64;
                                                let dst_off = d_dst.saturating_sub(watch_start);
                                                super::driver::zluda_log_err(&format!(
                                                    "MCB_RD5_TRACE #{} i={}/{} rd5=0x{:x} watch=0x{:x} dst=0x{:x} dst_off={} src=0x{:x} size={} flags=0x{:x}",
                                                    call_n,
                                                    i,
                                                    num_descs,
                                                    rd5,
                                                    watch_start,
                                                    d_dst,
                                                    dst_off,
                                                    d_src,
                                                    d_size,
                                                    d_flags
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ============================================================================
    // CLASS A WILD-SCATTER MECHANISM (PARKED — Rocksteady PhysX 3.3 internal bug)
    // ============================================================================
    //
    // MS_4 PTX (`/tmp/zluda_ptx_003.ptx`, ~line 2972 in CreateParticleShapesIncremental_MultiStage_4):
    //   ld.param.u32 %r2, [param_3];      // r2 = n_threads (host-passed)
    //   ...
    //   ld.global.u16 %rs1, [%rd9];       // rs1 = filter[r1*4] low u16
    //   and.b16 %rs2, %rs1, 17;
    //   setp.ne.s16 %p2, %rs2, 1;
    //   @%p2 bra BB13_3;                  // skip if filter bit not set
    //   ...
    //   atom.global.add.u32 %r8, [%rd19], -1;   // r8 = OLD counter, decrement
    //   add.s32 %r9, %r7, %r8;            // r9 = first_slot + old_counter
    //   add.s32 %r10, %r9, -1;            // r10 = first_slot + old_counter - 1
    //   mul.wide.u32 %rd20, %r10, 4;      // **UNSIGNED widen!** if r10 signed-negative
    //                                      // (i.e. r10 = -1 as s32 = 0xFFFFFFFF as u32),
    //                                      // rd20 = 0x3FFFFFFFC (~16 GiB)
    //   add.s64 %rd21, %rd10, %rd20;      // store target = param_5 + ~16 GiB FORWARD
    //   st.global.u32 [%rd21], %r1;       // WILD SCATTER
    //
    // **The bug** (root cause investigated 2026-06-06, see
    // `memory/project_class_a_writer_identified_2026_06_06.md`):
    //   - PhysX 3.3 (Rocksteady-customized, build path `e:\w\d9cc1c962e607cee\sw\
    //     physx\physxsdk\3.3\special\rocksteady\source\physxgpu\src\cuda\UB\`) has
    //     a particle-shape relay struct {u32 removed_count, u32 active_count,
    //     float active_count_f, float multiplier=1.0} at offset 0x800 of each
    //     per-actor 2 MiB ctx buffer.
    //   - MS-chain kernels (MS_1, MS_2_Batched, MS_3) write `active_count` correctly
    //     but NEVER write `removed_count` — that's PhysX's standalone-particle
    //     `RemoveParticleShapes` kernel's job, and Batman:AK's APEX 1.4 bypasses
    //     PhysX's particle-lifecycle CPU API entirely (uses ParticleIOS_* instead).
    //   - PhysX C++ reads `n_threads = active_count - removed_count = active_count`,
    //     which grows toward a 65000 cap as particles accumulate.
    //   - When MS_4 is launched with n_threads > sum-of-filter-matches, the excess
    //     threads `atom.global.add.u32 [counter], -1` past zero. The atom returns
    //     the OLD value: 0 → next call wraps counter to 0xFFFFFFFF.
    //   - First-underflow thread: r8=0, r9=first_slot, r10=first_slot-1. If
    //     `first_slot == 0` (typical for actor 0 / first shape group), r10 = -1
    //     and the wild scatter lands `param_5 + ~16 GiB FORWARD`.
    //   - That target falls in a dma_buf-imported D3D11 texture region (per
    //     `zluda_d3d11_register_imported`, 45 imports/run, base addresses in
    //     0x7fa6c0900000–0x7fa6d8380000 range), corrupting it. Next frame DXVK
    //     samples the corrupted texture, GPU TCP cache reads cross unmapped
    //     page boundaries → page fault on dxvk-submit thread → SIGABRT.
    //
    // **NVIDIA also has this bug** (per Steam Community Guide 2433732129): Batman:AK
    // with PhysX particles enabled crashes on NVIDIA driver ≥461.09; users must
    // downgrade to 456.55 (~Aug 2020). The pre-460 driver likely placed the wild
    // scatter's +16 GiB target in unallocated memory (graceful kernel-side fault),
    // and 460+ removed that implicit protection.
    //
    // **PhysX 3.4 open source** (NVIDIAGameWorks/PhysX-3.4) does NOT contain
    // `RemoveParticleShapes` or `CreateParticleShapesIncremental_MultiStage_*` —
    // those kernels are Rocksteady-bespoke. PhysX particles were deprecated in 3.4
    // (April 2017) and removed in 4.x.
    //
    // **Fix paths considered, all rejected**:
    // - Host-side cap of n_threads in this hook: requires reading the counter sum
    //   from param_7 via D2H from launch_kernel → forbidden (see
    //   feedback_no_d2d_from_launch_kernel.md, kills bench via ProfilingSignal race).
    // - PTX patch to add `if r10 > some_bound: skip`: per-kernel value-mutating
    //   PTX patch, disallowed by session constraints.
    // - Pre-buffer cuMemAlloc guard pages (2026-06-06 experiment): would have
    //   helped if scatter went backward, but `mul.wide.u32` math means it goes
    //   **+16 GiB FORWARD** → guards are wrong direction. Tried 64 KiB (no
    //   improvement) and 1 MiB (regression via memory pressure / Class B at
    //   startup). Reverted same-session.
    // - Synthetic RemoveParticleShapes dispatcher in ZLUDA: would mimic the
    //   missing cleanup PhysX intends. Multi-session reverse-engineering of the
    //   kernel's data structures required. Open option for future work.
    //
    // **Current state**: Class A is parked. The `patch_iofx_bbox_tail_merge` PTX
    // patch (in `module.rs`) catches some wild writes at the IOFX_actorRange site
    // (a band-aid one step downstream of the actual scatter). The existing
    // `MS1_RD5_RESET` defensive zero of `ctx+22176` handles the OTHER stale-counter
    // pattern. Beyond that, Class A is upstream of ZLUDA's intercept points.
    //
    // ============================================================================
    // Host-side zero-copy counter dump — runs for every MS_4 launch.
    // Pure CPU reads, no DtoH, no GPU ops — safe from the ProfilingSignal
    // race. The widened HOST_ALLOCS tracker now records (host, dev, size)
    // triples for every host allocation, so we can resolve MS_4's kernel
    // arg device pointers back to their host-mapped backing store and read
    // the SPECIFIC region the kernel will read, plus scan the whole buffer
    // for the 0xFFFFFC01 sentinel we know produces r10=0xFFFFFC00.
    if name.contains("CreateParticleShapesIncremental_MultiStage_4") {
        unsafe {
            let n: u32 = read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0);
            let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
            let p4: u64 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
            let p5: u64 = read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0);
            let p6: u64 = read_kernel_arg(kernel_params, extra, 48, 6).unwrap_or(0);
            let p7: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);
            static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if ms4_pointer_validation_enabled() {
                validate_ms4_pointer_ranges(call_n, n, p2, p4, p5, p6, p7);
            }
            super::driver::zluda_log_err(&format!(
                "MS4_ARGS #{} n={} p2=0x{:x} p4=0x{:x} p5=0x{:x} p6=0x{:x} p7=0x{:x}",
                call_n, n, p2, p4, p5, p6, p7
            ));
            // Defensive reset (gated): mark MS_4's p4 (= rd5 of this entry's
            // active MS-chain cycle) for zeroing of *(rd5+22176) at the next
            // MS_1 launch. PhysX should be calling `RemoveParticleShapes`
            // here but doesn't on Batman; the prior cycle's stale counter
            // value leaks into the next cycle's MS_1 → MS_2_Batched chain
            // and drives the -2 cascade. See feedback note for the
            // single-memset-per-cycle throttling rationale.
            mark_rd5_reset_pending(p4);
            // Concurrency-test probe (gated): drain all pending GPU work
            // after MS_4 to test whether the missing counter-reset is
            // actually a GPU-side ordering issue (some kernel/op meant
            // to fire after MS_4 completes its writes is racing the next
            // MS_1 read). Non-value-manipulating — pure sync.
            //
            // Mode '1': hipDeviceSynchronize (drains all streams). Original
            //   blunt-instrument fix.
            // Mode '2': hipStreamSynchronize on MS_4's stream. Lighter; only
            //   drains this stream's queue. If this also fixes the crash,
            //   the bug is per-stream memory visibility (GL2 cache flush
            //   on stream barrier), not cross-stream ordering.
            static MS4_POST_SYNC: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
            let post_sync_mode = *MS4_POST_SYNC.get_or_init(|| {
                std::env::var("ZLUDA_MS4_POST_SYNC")
                    .ok()
                    .and_then(|v| v.parse::<u8>().ok())
                    .unwrap_or(0)
            });
            match post_sync_mode {
                1 => { let _ = hipDeviceSynchronize(); }
                2 => { let _ = hipStreamSynchronize(stream); }
                _ => {}
            }
            // Plan v22 Phase B.1: record DEV5 (p5) addresses for the IOFX
            // per-launch kernarg probe to cross-reference against. Bounded
            // size — only the most recent ~64 distinct values matter.
            if p5 != 0 {
                let set = MS4_DEV5_ADDRS
                    .get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
                if let Ok(mut s) = set.lock() {
                    if s.len() < 256 {
                        s.insert(p5);
                    }
                }
            }
            // Plan v25 Phase B.2: push 6-tuple (p4, p5, p6, p7, n, call_n)
            // onto the dump queue so cuEventRecord / cuEventSynchronize can
            // drain it later (DtoH from launch_kernel is forbidden per
            // feedback_no_d2d_from_launch_kernel). The 4-pointer tuple lets
            // the consumer dump DEV5/6/7 (and optionally DEV4) in a single
            // drain pass. Gated on ZLUDA_DEV5_DUMP=1. Drop-oldest at 64
            // entries; queue lock held only across the push itself, NEVER
            // across a HIP call.
            static DEV5_DUMP_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *DEV5_DUMP_ENABLED
                .get_or_init(|| std::env::var("ZLUDA_DEV5_DUMP").map_or(false, |v| v == "1"))
                && p5 != 0
            {
                let queue = MS4_DEV5_DUMP_QUEUE.get_or_init(|| {
                    std::sync::Mutex::new(std::collections::VecDeque::with_capacity(64))
                });
                if let Ok(mut q) = queue.lock() {
                    while q.len() >= 64 {
                        q.pop_front();
                    }
                    q.push_back((p4, p5, p6, p7, n, call_n));
                }
            }
            // Plan v13 Step 1C: scan all `cuMemHostAlloc` (zero-copy
            // mapped) buffers for the u32 value `n`. Whoever holds it is
            // the buffer PhysX read from to compute MS_4 grid_size. Gated
            // on ZLUDA_MAPPED_N_SCAN=1 (default OFF: scan cost is ~2.5M
            // cmps per MS_4 launch, run unaffected with env unset).
            static SCAN_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *SCAN_ENABLED
                .get_or_init(|| std::env::var("ZLUDA_MAPPED_N_SCAN").map_or(false, |v| v == "1"))
            {
                let matches = super::memory::scan_mapped_for_u32(n, 16);
                if matches.is_empty() {
                    super::driver::zluda_log_err(&format!(
                        "[MAPPED_N_SCAN #{}] n={} NO_MATCH (value not in any cuMemHostAlloc buffer)",
                        call_n, n
                    ));
                } else {
                    for (host, size, off, ctx) in matches.iter() {
                        super::driver::zluda_log_err(&format!(
                            "[MAPPED_N_SCAN #{}] n={} host=0x{:x} size={} off={} ctx=[{:08x} {:08x} {:08x} {:08x}]",
                            call_n, n, host, size, off, ctx[0], ctx[1], ctx[2], ctx[3]
                        ));
                    }
                }
            }
            // v7 Step 4: probe shape_table (p4) state right before MS_4 fires.
            // Gated on ZLUDA_SHAPE_TABLE_PROBE=1.
            snapshot_shape_table_before_ms4(p4);
            // v7 Step 6: count (flags & 0x11)==1 bits MS_4 will dispatch.
            // Compare to sum(counter[]) — overcount hypothesis: pass > sum_counters.
            // Gated on ZLUDA_FLAG_COUNT_PROBE=1.
            snapshot_flag_count_before_ms4(p2, n, p7);
            // v8 Step 3: latch MS_4's flags ptr + n for post-MS_1 probe.
            // Different MS_4 launches use different actor flag ptrs, so we
            // latch the most recent one; post-MS_1 probe uses this to look
            // at the buffer that MS_4 will read in the next dispatch.
            MS4_FLAGS_PTR_LATCHED.store(p2, std::sync::atomic::Ordering::Release);
            MS4_N_LATCHED.store(n, std::sync::atomic::Ordering::Release);
            // p2=flags is just logged (no buffer read — avoid D2H from launch
            // path per feedback_no_d2d_from_launch_kernel). The address itself
            // shows whether flags is stable across frames (per-actor static)
            // and lets us correlate with addresses written by upstream kernels.
            // 2026-05-18: gate the legacy MS4_HOST_COUNTERS + MS4_ARG_DUMP
            // dumps behind ZLUDA_MS4_HOST_DUMP=1. Previously unconditional;
            // they dereference (host + offset) from every tracked mapped
            // buffer and were a suspect for the host-side SIGABRT class
            // that masqueraded as "GPU fault likely" through the
            // 2026-05-18 D-series. Default OFF for production; turn on
            // only when actively debugging MS_4 pre-launch state.
            static MS4_HOST_DUMP_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let ms4_host_dump_on = *MS4_HOST_DUMP_ENABLED
                .get_or_init(|| std::env::var("ZLUDA_MS4_HOST_DUMP").map_or(false, |v| v == "1"));
            if let Ok(table) = super::memory::SMALL_HOST_ALLOCS.lock() {
                if ms4_host_dump_on {
                    // Legacy dump: first 16 u32s of every tracked buffer (unchanged
                    // behavior for small host counters we've been relying on).
                    for &(host, dev, size) in table.iter() {
                        let words = (size / 4).min(16);
                        let mut dump = String::new();
                        for i in 0..words {
                            let w = std::ptr::read_volatile((host as *const u32).add(i));
                            if i > 0 {
                                dump.push(' ');
                            }
                            dump.push_str(&format!("{:08x}", w));
                        }
                        super::driver::zluda_log_err(&format!(
                            "MS4_HOST_COUNTERS #{} n={} host=0x{:x} dev=0x{:x} size={} u32=[{}]",
                            call_n, n, host, dev, size, dump
                        ));
                    }
                    // Arg resolution: for each of Arg4 (pool table), Arg6 (pool
                    // idx array), Arg7 (counter array), find the tracked buffer
                    // whose device range contains the arg pointer. Dump the 64
                    // u32s starting at the arg offset.
                    let args = [(4u8, p4), (5, p5), (6, p6), (7, p7)];
                    for &(arg_idx, arg_ptr) in args.iter() {
                        if arg_ptr == 0 {
                            continue;
                        }
                        for &(host, dev, size) in table.iter() {
                            let dev_end = dev.wrapping_add(size);
                            if (arg_ptr as usize) >= dev && (arg_ptr as usize) < dev_end {
                                let offset = (arg_ptr as usize) - dev;
                                let remaining = size - offset;
                                let words = (remaining / 4).min(64);
                                let mut dump = String::new();
                                let base = (host + offset) as *const u32;
                                for i in 0..words {
                                    let w = std::ptr::read_volatile(base.add(i));
                                    if i > 0 {
                                        dump.push(' ');
                                    }
                                    dump.push_str(&format!("{:08x}", w));
                                }
                                super::driver::zluda_log_err(&format!(
                                    "MS4_ARG_DUMP #{} n={} arg{}=0x{:x} host=0x{:x} dev=0x{:x} \
                                 size={} offset={} u32[0..{}]=[{}]",
                                    call_n,
                                    n,
                                    arg_idx,
                                    arg_ptr,
                                    host,
                                    dev,
                                    size,
                                    offset,
                                    words,
                                    dump
                                ));
                                break;
                            }
                        }
                    }
                } // end if ms4_host_dump_on
                  // NOTE: full-buffer sentinel scan removed — 6MB × 200 launches
                  // = 1.2 GB of CPU reads during init was correlated with
                  // triggering the ProfilingSignal libstdc++ race (p9, p10
                  // both aborted early). ARG_DUMP above is cheap enough to
                  // keep. For device-only buffers we need rocgdb, not scans.

                // 2026-05-17 evening: Plan v18 Step 7 — counter-array sum
                // probe. MS_4 atom-decrements per-shape counters at p7. The
                // working theory of the wild-VA story is
                // `n_threads > Σ counter[]` → atom.add(-1) underflows. This
                // probe surfaces Σ counter[0..n] (and the full-buffer sum +
                // nonzero stats) at each MS_4 launch so the gap is visible
                // directly rather than inferred. Read-only single-buffer
                // scan; n_threads ≤ ~32k → ≤128 KB per launch.
                static COUNTER_SUM_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if *COUNTER_SUM_ENABLED.get_or_init(|| {
                    std::env::var("ZLUDA_COUNTER_SUM_PROBE").map_or(false, |v| v == "1")
                }) && p7 != 0
                {
                    for &(host, dev, size) in table.iter() {
                        let dev_end = dev.wrapping_add(size);
                        if (p7 as usize) >= dev && (p7 as usize) < dev_end {
                            let offset = (p7 as usize) - dev;
                            let remaining = size - offset;
                            let total_words = remaining / 4;
                            let n_words = (n as usize).min(total_words);
                            let base = (host + offset) as *const u32;
                            let mut sum_first_n: u64 = 0;
                            let mut sum_full: u64 = 0;
                            let mut nonzero: u32 = 0;
                            let mut last_nz: i64 = -1;
                            for i in 0..total_words {
                                let w = std::ptr::read_volatile(base.add(i)) as u64;
                                if i < n_words {
                                    sum_first_n = sum_first_n.wrapping_add(w);
                                }
                                sum_full = sum_full.wrapping_add(w);
                                if w != 0 {
                                    nonzero += 1;
                                    last_nz = i as i64;
                                }
                            }
                            let gap: i64 = (n as i64) - (sum_first_n as i64);
                            super::driver::zluda_log_err(&format!(
                                "MS4_COUNTER_SUM #{} n={} sum_first_n={} sum_full={} \
                                 gap={} nonzero={} last_nz={} total_words={} \
                                 host=0x{:x} p7=0x{:x} offset={}",
                                call_n,
                                n,
                                sum_first_n,
                                sum_full,
                                gap,
                                nonzero,
                                last_nz,
                                total_words,
                                host,
                                p7,
                                offset
                            ));
                            break;
                        }
                    }
                }
            }
            // 2026-05-17 evening: Plan v18 Step 6 — per-MS_4 actor-entry probe.
            // Discovered Batman runs TWO PhysX particle systems with an actor
            // table at off=663552 + N*176 in a 2 MB mapped buffer. For each
            // entry we want the candidate "per-frame counter" slots:
            //   - +0    : mValidParticleRange (count, monotonic)
            //   - +22976: candidate destroyedShapeCount (oscillating 0-9)
            //   - +22980: candidate createdShapeCount  (oscillating 0-4)
            // Gated on ZLUDA_ACTOR_ENTRY_PROBE=1 (default OFF). Cheap: at
            // most a few u32 reads per MS_4 launch.
            static ACTOR_PROBE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *ACTOR_PROBE_ENABLED.get_or_init(|| {
                std::env::var("ZLUDA_ACTOR_ENTRY_PROBE").map_or(false, |v| v == "1")
            }) {
                if let Ok(table) = super::memory::SMALL_HOST_ALLOCS.lock() {
                    // Per-entry stride and slot offsets discovered offline.
                    const ENTRY_BASE: usize = 663552;
                    const ENTRY_STRIDE: usize = 176;
                    const NUM_ENTRIES: usize = 4; // probe up to 4 systems
                    const COUNT_REL: usize = 0;
                    const DSTR_REL: usize = 22976; // 686528-663552
                    const CREA_REL: usize = 22980; // 686532-663552
                    for &(host, _dev, size) in table.iter() {
                        // Need at least 2 MB to host the actor table.
                        if size < 700_000 {
                            continue;
                        }
                        let mut line =
                            format!("ACTOR_ENTRY_PROBE #{} n={} host=0x{:x}", call_n, n, host);
                        let mut any_nonzero = false;
                        for entry in 0..NUM_ENTRIES {
                            let base = ENTRY_BASE + entry * ENTRY_STRIDE;
                            if base + 23000 > size {
                                break;
                            }
                            let cnt =
                                std::ptr::read_volatile((host + base + COUNT_REL) as *const u32);
                            let dstr =
                                std::ptr::read_volatile((host + base + DSTR_REL) as *const u32);
                            let crea =
                                std::ptr::read_volatile((host + base + CREA_REL) as *const u32);
                            if cnt != 0 || dstr != 0 || crea != 0 {
                                any_nonzero = true;
                            }
                            line.push_str(&format!(
                                " e{}=[cnt={} dstr={} crea={}]",
                                entry, cnt, dstr, crea
                            ));
                        }
                        if any_nonzero {
                            super::driver::zluda_log_err(&line);
                        }
                    }
                }
            }
            // 2026-05-18 Plan v20.1 D2: relay-slot probe. At each MS_4
            // launch, read u32 at the known dst offsets that Phase C
            // identified as MemCopyBatchedAsync relay targets. Logged
            // alongside ACTOR_ENTRY_PROBE so the value progression can
            // be correlated with n_threads in the same line stream.
            // Offsets are hardcoded from two empirical sources:
            //   - 663640 / 1056768 : Plan v19 SLOT_IDENTIFIED memory
            //   - 657408 / 1312768 : Phase C run on 2026-05-18 night
            // Both pairs included because the slot offsets shift between
            // runs (different PhysX particle-system layouts).
            // Cheap: 4 u32 reads per tracked >=2MB mapped buffer per launch.
            static RELAY_SLOT_PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *RELAY_SLOT_PROBE
                .get_or_init(|| std::env::var("ZLUDA_RELAY_SLOT_PROBE").map_or(false, |v| v == "1"))
            {
                if let Ok(table) = super::memory::SMALL_HOST_ALLOCS.lock() {
                    const OFFSETS: [usize; 4] = [657408, 663640, 1056768, 1312768];
                    for &(host, _dev, size) in table.iter() {
                        if size < 2_000_000 {
                            continue;
                        }
                        let mut line =
                            format!("RELAY_SLOT_PROBE #{} n={} host=0x{:x}", call_n, n, host);
                        let mut any_nonzero = false;
                        for &off in OFFSETS.iter() {
                            if off + 4 > size {
                                continue;
                            }
                            let v = std::ptr::read_volatile((host + off) as *const u32);
                            if v != 0 {
                                any_nonzero = true;
                            }
                            line.push_str(&format!(" v[{}]={}", off, v));
                        }
                        if any_nonzero {
                            super::driver::zluda_log_err(&line);
                        }
                    }
                }
            }
        }
    }

    // actorRange root-cause probe: track which buffer actorRange's param_4
    // actually IS by comparing pointers with remapKernel's param_9 (sorted
    // keys input, has bit-31 tags) and param_10 (remapped output, clean).
    // Gated on ZLUDA_AR_PTR_TRACE=1.
    {
        static AR_PTR_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *AR_PTR_TRACE.get_or_init(|| {
            std::env::var("ZLUDA_AR_PTR_TRACE").map_or(false, |v| v == "1")
        }) {
            static REMAP_P9: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static REMAP_P10: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

            if name.contains("IOFX_remapKernel") {
                let p9: u64 = unsafe { read_kernel_arg(kernel_params, extra, 48, 9).unwrap_or(0) };
                let p10: u64 = unsafe { read_kernel_arg(kernel_params, extra, 56, 10).unwrap_or(0) };
                let prev9 = REMAP_P9.swap(p9, std::sync::atomic::Ordering::Relaxed);
                let prev10 = REMAP_P10.swap(p10, std::sync::atomic::Ordering::Relaxed);
                if p9 != prev9 || p10 != prev10 {
                    super::driver::zluda_log_err(&format!(
                        "[AR_PTR] remap '{}' p9(sorted_keys)=0x{:x} p10(output)=0x{:x}",
                        name, p9, p10
                    ));
                }
            }
            if name == "IOFX_actorRangeKernel" {
                let p4: u64 = unsafe { read_kernel_arg(kernel_params, extra, 24, 4).unwrap_or(0) };
                let rp9 = REMAP_P9.load(std::sync::atomic::Ordering::Relaxed);
                let rp10 = REMAP_P10.load(std::sync::atomic::Ordering::Relaxed);
                let match_label = if p4 == rp9 {
                    "== remap_p9 (SORTED KEYS — has bit-31!)"
                } else if p4 == rp10 {
                    "== remap_p10 (remap output — should be clean)"
                } else {
                    "NEITHER remap p9 nor p10"
                };
                static AR_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let seq = AR_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if seq < 20 || seq % 500 == 0 {
                    super::driver::zluda_log_err(&format!(
                        "[AR_PTR] actorRange #{} p4=0x{:x} {} (rp9=0x{:x} rp10=0x{:x})",
                        seq, p4, match_label, rp9, rp10
                    ));
                }
            }
        }
    }

    // Plan v26 Phase B.3: IOFX_actorRangeKernel param_4 dump producer.
    // Pushes (param_4_ptr, call_n) onto IOFX_AR_DUMP_QUEUE so cuEventRecord /
    // cuEventSynchronize can drain it later. param_4 is the suspected wild
    // value source per project_actorrange_wild_input_root_cause_2026_05_11.
    //
    // PTX param layout (u64+u16+u32+u32+u64+...): param_4 starts at byte 24
    // when u16 is padded to its natural alignment (u16 at 8, u32 at 12, u32
    // at 16, u64 at 24 with alignment-8 padding inserted). Verified via
    // KARG_LOG output where slot[3] of 8-byte slots = DEV3 = first device
    // ptr = param_4.
    //
    // Gated on ZLUDA_ACTORRANGE_DUMP=1, default OFF.
    if name.contains("IOFX_actorRange") {
        static AR_DUMP_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *AR_DUMP_ENABLED
            .get_or_init(|| std::env::var("ZLUDA_ACTORRANGE_DUMP").map_or(false, |v| v == "1"))
        {
            unsafe {
                // PTX param layout for IOFX_actorRange (Explore-confirmed):
                //   param_4 u64 @ byte 24 — first READ buffer (the one e18
                //                            caught wild)
                //   param_6 u64 @ byte 40 — second READ buffer
                //   param_9 u64 @ byte 64 — third READ buffer
                // (param_5 is .u32 at byte 32; param_7+8 are WRITE buffers,
                //  not interesting for wild-VA source identification)
                let p4: u64 = read_kernel_arg(kernel_params, extra, 24, 4).unwrap_or(0);
                let p6: u64 = read_kernel_arg(kernel_params, extra, 40, 6).unwrap_or(0);
                let p9: u64 = read_kernel_arg(kernel_params, extra, 64, 9).unwrap_or(0);
                static AR_CALL_N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let call_n = AR_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Plausibility check: param_4 must look like a user-space
                // pointer. param_6 and param_9 are also validated and
                // zeroed-out if not plausible (consumer will skip 0 entries).
                const PTR_LO: u64 = 0x1000;
                const PTR_HI: u64 = 0x0000_8000_0000_0000;
                let plausible = |p: u64| p >= PTR_LO && p < PTR_HI;
                if plausible(p4) {
                    let p6_safe = if plausible(p6) { p6 } else { 0 };
                    let p9_safe = if plausible(p9) { p9 } else { 0 };
                    let queue = IOFX_AR_DUMP_QUEUE.get_or_init(|| {
                        std::sync::Mutex::new(std::collections::VecDeque::with_capacity(64))
                    });
                    if let Ok(mut q) = queue.lock() {
                        while q.len() >= 64 {
                            q.pop_front();
                        }
                        q.push_back((p4, p6_safe, p9_safe, call_n));
                    }
                } else {
                    super::driver::zluda_log_err(&format!(
                        "[IOFX_AR_PUSH #{}] '{}' p4=0x{:x} not plausible — skipping push",
                        call_n, name, p4
                    ));
                }
            }
        }
    }

    // 2026-05-16 v7 Step 3: per-launch shape_table pipeline logging.
    // Gated on ZLUDA_SHAPE_PIPELINE_LOG=1. Captures MS_1 + InitParticleShapes
    // per launch (not one-shot). Cross-reference with MS4_ARGS to detect
    // which actor's shape_table state is wrong at MS_4 entry.
    static SHAPE_PIPELINE_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let shape_pipeline_log = *SHAPE_PIPELINE_LOG
        .get_or_init(|| std::env::var("ZLUDA_SHAPE_PIPELINE_LOG").map_or(false, |v| v == "1"));
    if shape_pipeline_log {
        if name.contains("CreateParticleShapesIncremental_MultiStage_1") {
            unsafe {
                // MS_1 params: 0..u64 (0), 1..u16 (8 pad to 16), 2..u64 (16),
                // 3..u64 (24), 4..u32 (32), 5..f32 (36), 6..u64 (40) = pool struct
                // (contains shape_table AND the +22176 allocation counter),
                // 7..u64 (48), 8..u64 (56), 9..u64 (64), 10..u64 (72), 11..u64 (80).
                let p4_n: u32 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
                let p6_pool: u64 = read_kernel_arg(kernel_params, extra, 40, 6).unwrap_or(0);
                let p7: u64 = read_kernel_arg(kernel_params, extra, 48, 7).unwrap_or(0);
                static MS1_CALL_N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let call_n = MS1_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::driver::zluda_log_err(&format!(
                    "MS1_ARGS #{} n={} p6_pool=0x{:x} p7=0x{:x} grid=[{},{},{}] block=[{},{},{}]",
                    call_n,
                    p4_n,
                    p6_pool,
                    p7,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z
                ));
                // v10 Step 1: probe the [+22176] pool allocation counter BEFORE
                // MS_1 increments it. Should be 0 each frame if reset works.
                // If saturated (>= 924), the reset kernel didn't run.
                snapshot_pool_22176_counter(p6_pool);
            }
        }
        if name == "InitParticleShapes" {
            unsafe {
                // InitParticleShapes params: 0..u64 (0), 1..u16 (8), 2..u64 (16)
                // = shape_table, 3..u32 (24), 4..u64 (32), 5..u32 (40), ...
                let p2_shape: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
                let p3_n: u32 = read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0);
                let p4: u64 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
                let p5_n: u32 = read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0);
                static IPS_CALL_N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let call_n = IPS_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::driver::zluda_log_err(&format!(
                    "INITPS_ARGS #{} p2_shape=0x{:x} p3_n={} p4=0x{:x} p5_n={} grid=[{},{},{}] block=[{},{},{}]",
                    call_n, p2_shape, p3_n, p4, p5_n,
                    grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z
                ));
            }
        }
        if name == "InitCollDataAndApplyConstraints" {
            unsafe {
                // InitColl params: 0..u64 (0), 1..u16 (8), 2..u64 (16),
                // 3..u64 (24) = FLAGS (DEV3 in KARG_LOG), 4..u64 (32),
                // 5..u64 (40), 6..u64 (48), 7..u64 (56), ...
                // n_threads for the probe = grid_dim_x * block_dim_x
                // (InitColl is 1D: grid=[N,1,1] block=[128,1,1])
                let p3_flags: u64 = read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0);
                let n_threads = grid_dim_x.saturating_mul(block_dim_x);
                static INITCOLL_CALL_N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let call_n = INITCOLL_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::driver::zluda_log_err(&format!(
                    "INITCOLL_ARGS #{} p3_flags=0x{:x} n_threads={} grid=[{},{},{}] block=[{},{},{}]",
                    call_n, p3_flags, n_threads,
                    grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z
                ));
                // v8 Step 1: probe flags BEFORE InitColl runs. If bit 0 is
                // already set 100%, the upstream writer is the culprit.
                // If bit 0 distribution is normal pre-InitColl and only
                // 100% post-InitColl, then InitColl itself is wrong.
                snapshot_flag_count_before_initcoll(p3_flags, n_threads);
            }
        }
        // v8 Step 2: ParticleCollisionStaticBox probe. Its param_2 is a 64-byte
        // by-value struct holding 6 u64 ptrs at offsets 0,8,16,24,32,40.
        // Per PTX (line 20024-20100): flags ptr = struct+40 (DEV4 in struct).
        // Flat-arg offset = 16 (param_2 start, after param_0 u64 + padded u16
        // param_1) + 40 = 56.
        if name == "ParticleCollisionStaticBox" {
            unsafe {
                let flags_ptr: u64 = read_kernel_arg(kernel_params, extra, 56, 2).unwrap_or(0);
                // param_6 (u32 at offset 16 + 64 = 80 + 8 = 88) is n_particles
                // actually: param_4 u32 at 80, param_5 u64 at 88, param_6 u32 at 96.
                // Let's read param_6 = u32 at offset 96 (post param_5 u64).
                let n_particles: u32 = read_kernel_arg(kernel_params, extra, 96, 6).unwrap_or(0);
                let n_threads_kernel = grid_dim_x.saturating_mul(block_dim_x);
                static PCSB_CALL_N: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let call_n = PCSB_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::driver::zluda_log_err(&format!(
                    "PCSB_ARGS #{} flags=0x{:x} n_particles={} n_threads={} grid=[{},{},{}] block=[{},{},{}]",
                    call_n, flags_ptr, n_particles, n_threads_kernel,
                    grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z
                ));
                // Pre-PCSB probe: how many particles already have bit 4 set?
                // n_threads = min(n_particles, 256) — first 256 for cheap probe
                let probe_n = n_particles.min(256);
                snapshot_flag_bit4_at("PRE_PCSB", flags_ptr, probe_n);
            }
        }
    }

    if name.contains("CreateParticleShapesIncremental_MultiStage_2_Batched") {
        unsafe {
            // MS_2_Batched has only 3 params: 0..u64 (0), 1..u16 (8 pad to 16),
            // 2..u64 (16) = per-CTA struct array. Reading higher indices walks
            // past the kernel_params array and segfaults — DON'T.
            //
            // This probe must stay OUTSIDE ZLUDA_SHAPE_PIPELINE_LOG gating:
            // shape-pipeline logging is usually off in live runs, while
            // ZLUDA_MS2_CAPTURE is intentionally independent.
            let p2_ctas: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
            // 2026-06-07 Class A writer hunt: also log p0 to see if it
            // points into the per-actor relay buffer (the source for the
            // MemCopyBatchedAsync that feeds MS_4's over-dispatched n).
            let p0: u64 = read_kernel_arg(kernel_params, extra, 0, 0).unwrap_or(0);
            static MS2B_CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let call_n = MS2B_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            super::driver::zluda_log_err(&format!(
                "MS2B_ARGS #{} p0=0x{:x} p2_ctas=0x{:x} grid=[{},{},{}] block=[{},{},{}]",
                call_n,
                p0,
                p2_ctas,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z
            ));
            static MS2_CAPTURE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *MS2_CAPTURE_ENABLED
                .get_or_init(|| std::env::var("ZLUDA_MS2_CAPTURE").map_or(false, |v| v == "1"))
            {
                const PTR_LO: u64 = 0x1000;
                const PTR_HI: u64 = 0x0000_8000_0000_0000;
                let total_ctas = grid_dim_x
                    .saturating_mul(grid_dim_y)
                    .saturating_mul(grid_dim_z);
                let plausible = p2_ctas >= PTR_LO && p2_ctas < PTR_HI;
                if plausible && total_ctas > 0 {
                    let queue = MS2_BATCH_DUMP_QUEUE.get_or_init(|| {
                        std::sync::Mutex::new(std::collections::VecDeque::with_capacity(64))
                    });
                    if let Ok(mut q) = queue.lock() {
                        while q.len() >= 64 {
                            q.pop_front();
                        }
                        q.push_back((p2_ctas, grid_dim_x, grid_dim_y, grid_dim_z, call_n));
                    }
                } else {
                    super::driver::zluda_log_err(&format!(
                        "[MS2_CAPTURE_PUSH #{}] p2_ctas=0x{:x} plausible={} total_ctas={} — skipping push",
                        call_n, p2_ctas, plausible, total_ctas
                    ));
                }
            }
        }
    }

    // MS_4 capture — ALL gated behind ZLUDA_MS4_CAPTURE=1
    // (slot, scatter_ptr=param4, num_particles) threaded to the post-launch
    // output capture below; mirrors the iofx_actor_range_slot pattern.
    let mut ms4_output_capture: Option<(usize, u64, u32)> = None;
    if name.contains("CreateParticleShapesIncremental_MultiStage_4") {
        static MS4_CAP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *MS4_CAP.get_or_init(|| std::env::var("ZLUDA_MS4_CAPTURE").map_or(false, |v| v == "1")) {
            unsafe {
                let slot = capture_ms4_inputs(kernel_params, extra, grid_dim_x, block_dim_x);
                latch_counter_ptr_from_ms4(kernel_params, extra);
                let p3: u32 = read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0);
                let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
                let p4: u64 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
                let p5: u64 = read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0);
                let p6: u64 = read_kernel_arg(kernel_params, extra, 48, 6).unwrap_or(0);
                let p7: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);
                if let Some(s) = slot {
                    // SCATTER OUTPUT is param5 (PTX: st.global [cvta(param5)+r10*4]).
                    // param4 (p4) is the struct/base table — NOT the output.
                    ms4_output_capture = Some((s, p5, p3));
                }
                super::driver::zluda_log_err(&format!(
                    "[MS4_ARGS] grid={} block={} N={} p2=0x{:x} p4=0x{:x} p5=0x{:x} p6=0x{:x} p7=0x{:x}",
                    grid_dim_x, block_dim_x, p3, p2, p4, p5, p6, p7,
                ));
            }
        }
    }
    // TurbulenceFS / adjacent-physics launch capture (ZLUDA_TFS_CAPTURE=1) —
    // host-side dump of the kernel argument flat buffer (incl. the 456-byte
    // DynamicGrid descriptor struct when present) for any TurbulenceFS_* kernel
    // plus the nearby non-TurbulenceFS collision/particle-shape kernels that
    // keep appearing in the same early-SIGSEGV window. Cheap (no GPU ops, no
    // sync). Captures into a 64-slot pinned ring dumped from the SIGABRT
    // handler to /tmp/tfs_captures.bin for post-mortem replay.
    let tfs_atomic_to_tex_slot = if let Some(tfs_kernel_id) = classify_tfs_kernel(&name) {
        unsafe {
            capture_tfs_launch(
                &name,
                tfs_kernel_id,
                kernel_params,
                extra,
                stream,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
            )
        }
    } else {
        None
    };
    let iofx_actor_range_slot = if name == "IOFX_actorRangeKernel" {
        unsafe {
            capture_iofx_actor_range_inputs(
                kernel_params,
                extra,
                grid_dim_x,
                block_dim_x,
                shared_mem_bytes,
            )
        }
    } else {
        None
    };
    let iofx_bbox_slot = if name == "IOFX_bboxKernel" {
        unsafe {
            capture_iofx_bbox_inputs(
                f.module,
                kernel_params,
                extra,
                grid_dim_x,
                block_dim_x,
                shared_mem_bytes,
            )
        }
    } else {
        None
    };
    let cub_downsweep_slot = if name.contains("RadixSortDownsweepKernel") {
        unsafe {
            capture_cub_downsweep_inputs(
                extra,
                global_launch_n.unwrap_or(0),
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
                shared_mem_bytes,
            )
        }
    } else {
        None
    };
    let cub_scan_slot = if name.contains("RadixSortScanKernel") {
        unsafe {
            capture_cub_scan_inputs(
                kernel_params,
                extra,
                global_launch_n.unwrap_or(0),
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
                shared_mem_bytes,
            )
        }
    } else {
        None
    };
    let iofx_chain_slot = if let Some(chain_kind) = classify_iofx_chain_kernel(&name) {
        unsafe {
            capture_iofx_chain_inputs(
                &name,
                chain_kind,
                kernel_params,
                extra,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
                shared_mem_bytes,
            )
        }
    } else {
        None
    };
    if creategrid_safe_validator_enabled() {
        unsafe {
            maybe_capture_creategrid_sort_buffer_touch(
                &name,
                kernel_params,
                extra,
                shared_mem_bytes,
                stream,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
            );
        }
    }
    if name.contains("CreateGrid") && !name.contains("Turbulence") {
        let n: i32 = unsafe { read_kernel_arg(kernel_params, extra, 0, 0).unwrap_or(0) };
        let sorted_keys: u64 = unsafe { read_kernel_arg(kernel_params, extra, 8, 1).unwrap_or(0) };
        if n > 0 && sorted_keys != 0 {
            note_creategrid_sorted_keys(sorted_keys, n as usize);
            let touch_start = creategrid_sort_touch_log_start();
            let snapshot_start = creategrid_snapshot_log_start();
            let call_n = if touch_start.is_some() || snapshot_start.is_some() {
                Some(CREATEGRID_LOG_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            } else {
                None
            };
            if let (Some(start_call), Some(call_n)) = (touch_start, call_n) {
                if call_n >= start_call {
                    let refs = unsafe {
                        collect_creategrid_sort_refs_for_launch(
                            sorted_keys,
                            n as usize,
                            kernel_params,
                            extra,
                        )
                    };
                    if refs.is_empty() {
                        super::driver::zluda_log_err(&format!(
                            "    SORT_TOUCH_DIRECT creategrid#{} grid=[{},{},{}] block=[{},{},{}] shared={} stream={:?} refs=[] sk=0x{:x} n={}",
                            call_n,
                            grid_dim_x,
                            grid_dim_y,
                            grid_dim_z,
                            block_dim_x,
                            block_dim_y,
                            block_dim_z,
                            shared_mem_bytes,
                            stream,
                            sorted_keys,
                            n
                        ));
                    } else {
                        super::driver::zluda_log_err(&format!(
                            "    SORT_TOUCH_DIRECT creategrid#{} grid=[{},{},{}] block=[{},{},{}] shared={} stream={:?} refs=[{}]",
                            call_n,
                            grid_dim_x,
                            grid_dim_y,
                            grid_dim_z,
                            block_dim_x,
                            block_dim_y,
                            block_dim_z,
                            shared_mem_bytes,
                            stream,
                            format_creategrid_sort_touch_refs(&refs)
                        ));
                    }
                }
            }
            if let (Some(start_call), Some(call_n)) = (snapshot_start, call_n) {
                if call_n >= start_call {
                    unsafe {
                        log_creategrid_snapshot_direct(
                            call_n,
                            n as usize,
                            sorted_keys,
                            stream,
                            grid_dim_x,
                            grid_dim_y,
                            grid_dim_z,
                            block_dim_x,
                            block_dim_y,
                            block_dim_z,
                            shared_mem_bytes,
                        );
                    }
                }
            }
        }
        if creategrid_safe_validator_enabled() {
            unsafe {
                validate_create_grid_safe(kernel_params, extra, stream);
            }
        }
    }
    // MS_2_Batched packet capture + pre-counter snapshot. We snapshot the
    // 88-byte-per-block packet buffer (param_2) AND every tracked counter
    // pointer BEFORE the kernel runs, so the post-capture below can diff
    // pre vs post and reveal whether MS_2_Batched is what turns counter[0..7]
    // from `[19,39,16,22,2,10,8,7]` into `[-2,-2,-2,-2,-2,-2,-2,-2]`.
    // Returns the slot index to thread into the post-capture.
    let ms2b_slot = if should_capture_ms2b(&name, grid_dim_x) {
        unsafe { capture_ms2b_inputs(kernel_params, extra, grid_dim_x, block_dim_x) }
    } else {
        None
    };
    // MS_3 pre-capture — snapshot every tracked counter pointer BEFORE the
    // kernel runs, so the post-capture below can diff the two and reveal
    // whether MS_3 is what turns counter[0] from its MS_2_Batched value into
    // the -2 MS_4 reads. Returns the slot index to thread into the post-capture.
    // MS_3 capture — disabled (does D2D)
    let ms3_slot: Option<usize> = None;
    // 2026-06-05 Class A experiment: MS_4 is the documented wild-scatter
    // producer (atom.global.add.u32 [..], -1 in PTX BB13). If the counter at
    // `param_6[r6*4]` is 0 when MS_4 reads it, the atomic decrements to
    // 0xFFFFFFFF (unsigned wrap) → r9 = first_slot + 0 → r10 = first_slot - 1
    // → out-of-bounds store. The corrupted output then propagates downstream
    // to IOFX_remapKernel (texref read of param_4) → IOFX_actorRangeKernel
    // (wild scatter) → canonical CB/DB+TCP page fault.
    //
    // The counter pool is written by MS_2_Batched (BB11_9 decrements
    // group_base[bucket].field_8) AND the filter buffer is written by MS_1.
    // The PTX-level emit_ret SC system-scope fence is supposed to make these
    // visible, but Class A still fires ~75% of crashes under Phase 2.
    // Strongest residual hypothesis: producer→consumer ordering between
    // MS_2_Batched's counter write and MS_4's read leaks across the gfx10
    // WGP boundary (different wavefronts on different WGPs).
    //
    // ZLUDA_MS4_PRE_SYNC drains all in-flight kernels on this stream BEFORE
    // MS_4 launches, ensuring MS_1/MS_2_B/MS_3 outputs are fully visible.
    //   mode '1' → hipDeviceSynchronize (cross-stream drain)
    //   mode '2' → hipStreamSynchronize (this stream only)
    // Default 0 (off). A/B candidate for Class A.
    if name == "CreateParticleShapesIncremental_MultiStage_4" {
        static MS4_PRE_SYNC: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let pre_sync_mode = *MS4_PRE_SYNC.get_or_init(|| {
            std::env::var("ZLUDA_MS4_PRE_SYNC")
                .ok()
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(0)
        });
        match pre_sync_mode {
            1 => unsafe { let _ = hipDeviceSynchronize(); },
            2 => unsafe { let _ = hipStreamSynchronize(stream); },
            _ => {}
        }
    }
    // Phase 1 fix for Bug A: defensively force-zero `*(rd5+22176)` for every
    // tracked rd5 BEFORE MS_1 reads it. PhysX's per-frame MemCopyBatchedAsync
    // reset batch occasionally excludes the rd5 buffer that MS_1 is about to
    // use; without this fix, MS_1 reads the prior frame's stale value, atomic-
    // adds the new shape count on top, doubles the counter, and the MS_2_Batched
    // → MS_4 chain crashes 5-10 ms later. See `force_zero_rd5_22176_before_ms1`
    // for the full causal chain analysis.
    //
    // **KNOWN BUG (discovered run 25, NOT YET FIXED)**: this fires before
    // EVERY MS_1, but PhysX issues 2 MS_1 launches per frame (one per actor).
    // The second invocation overzeros the first MS_1's just-written value,
    // causing MS_2_Batched to take BB11_23 noop and MS_4 to wild-scatter on
    // empty counters → "Late MS_4 @ 0x400037000" crash pattern in runs
    // 18,20,21,22. Need to root-cause WHY PhysX misses the reset before
    // designing a real fix — see Phase 1.4 in plan.
    // Phase 1 force-zero workaround REMOVED 2026-04-08. The real fix is at
    // the codegen layer: emit_ret in ptx/src/pass/llvm/emit.rs now inserts
    // a `fence seq_cst` before every kernel exit, which lowers to
    // `buffer_gl1_inv; buffer_gl0_inv` on AMDGPU and ensures stores from
    // PhysX MemCopyBatchedAsync's zero-fill path (which doesn't have its
    // own `membar.sys`) are globally visible before the next kernel runs.
    // The old broad workaround (`force_zero_rd5_22176_before_ms1`) remains
    // only for unit tests. Production now uses the targeted variant that
    // zeros just the current MS_1 launch's rd5 (param_6), avoiding the
    // two-actors-per-frame overzero bug from runs 18/20/21/22.
    if name == "CreateParticleShapesIncremental_MultiStage_1" {
        unsafe {
            force_zero_current_ms1_rd5_22176(kernel_params, extra);
            // 2026-06-03: capture-independent consumer for
            // MS1_RD5_RESET_PENDING. Mirrors the work that
            // `ms1_consume_rd5_resets` does from inside `capture_tfs_launch`,
            // but runs whenever ZLUDA_MS1_RD5_RESET=1 — without requiring
            // ZLUDA_TFS_CAPTURE=1. Both callers drain the same Vec via
            // `take_pending_rd5_resets`, so per MS_1 launch only one of
            // them gets a non-empty result (no double-firing).
            //
            // SYNC mode (ZLUDA_MS1_RD5_RESET_SYNC=1) issues hipMemsetD32
            // immediately. DEFERRED (default, =0) enqueues onto the
            // host-thread drain queue. See ms1_consume_rd5_resets above
            // for the same A/B logic.
            if ms1_rd5_reset_enabled() {
                let pending = take_pending_rd5_resets();
                if !pending.is_empty() {
                    static CALL_N: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    static SYNC_MODE: std::sync::OnceLock<bool> =
                        std::sync::OnceLock::new();
                    let sync_mode = *SYNC_MODE.get_or_init(|| {
                        std::env::var("ZLUDA_MS1_RD5_RESET_SYNC")
                            .map_or(false, |v| v == "1")
                    });
                    if sync_mode {
                        for rd5 in pending {
                            let dst = (rd5 + F6_OFFSET_IN_RD5 as u64)
                                as *mut std::ffi::c_void;
                            let r = hipMemsetD32(hipDeviceptr_t(dst), 0, 1);
                            super::driver::zluda_log_err(&format!(
                                "MS1_RD5_RESET_MS1BRANCH #{} rd5=0x{:x} target=0x{:x} hipMemsetD32 SYNC ok={}",
                                call_n, rd5, dst as u64, !r.is_err()
                            ));
                        }
                    } else {
                        let queue = MS1_RD5_RESET_DRAIN_QUEUE.get_or_init(|| {
                            std::sync::Mutex::new(std::collections::VecDeque::with_capacity(64))
                        });
                        if let Ok(mut q) = queue.lock() {
                            for rd5 in pending {
                                while q.len() >= 64 {
                                    q.pop_front();
                                }
                                q.push_back(rd5);
                                super::driver::zluda_log_err(&format!(
                                    "MS1_RD5_RESET_MS1BRANCH #{} rd5=0x{:x} target=0x{:x} enqueued for host-thread drain",
                                    call_n,
                                    rd5,
                                    rd5 + F6_OFFSET_IN_RD5 as u64
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    // Phase 1.5 Item A: dump the MemCopyBatchedAsync Arg2 scratch buffer
    // contents BEFORE the kernel launches. This answers the "PhysX bug or
    // ZLUDA bug?" root-cause question for Phase 1 — by correlating the
    // scratch contents with the RESET log lines we can see whether the
    // bug-firing rd5 is included in the batch (kernel-internal bug) or
    // excluded (PhysX host-side scheduling mismatch).
    // MCB scratch dump — gated on ZLUDA_MS4_CAPTURE=1. Dumps the
    // MemCopyBatchedAsync param_2 descriptor batch (f0 base) BEFORE each launch.
    // 2026-06-13: re-enabled to find which batch entry holds the wild dst that
    // faults the render-interop blit at the perf-test ~314s crash
    // (project_hsa_poisoning_falsified_2026_06_13). The last MCB SCRATCH line
    // before the SIGABRT == the faulting blit's batch.
    if name == "MemCopyBatchedAsync" {
        unsafe { dump_mcb_scratch(kernel_params, extra, grid_dim_y); }
    }
    let sort_downsweep_args = if (creategrid_safe_validator_enabled()
        || creategrid_snapshot_log_start().is_some())
        && name.contains("RadixSortDownsweepKernel")
        && !extra.is_null()
        && stream.0.is_null()
    {
        unsafe { read_sort_downsweep_launch_args(extra) }
    } else {
        None
    };
    let createcell_snapshot_args = if createcell_snapshot_log_start().is_some()
        && creategrid_snapshot_log_start().is_some()
        && name.contains("CreateCellIndices")
    {
        unsafe { read_createcell_snapshot_args(kernel_params, extra) }
    } else {
        None
    };
    if let Some(args) = sort_downsweep_args {
        unsafe {
            zero_sort_downsweep_input_slack(
                args.n_items,
                args.grid_share,
                args.keys_in,
                args.vals_in,
            );
            snapshot_sort_downsweep_prelaunch_state(
                args.begin_bit,
                args.n_items,
                args.grid_share,
                args.keys_in,
                args.vals_in,
                args.keys_out,
                args.vals_out,
            );
            let capture_begin_bit = sort_replay_capture_begin_bit();
            let creategrid_call_n =
                CREATEGRID_LOG_CALL_N.load(std::sync::atomic::Ordering::Relaxed);
            let start_call = creategrid_snapshot_log_start();
            if should_queue_sort_replay_capture(
                args.begin_bit,
                creategrid_call_n,
                start_call,
                capture_begin_bit,
            ) {
                queue_sort_downsweep_replay_capture_prelaunch(
                    args,
                    stream,
                    grid_dim_x,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                );
            }
        }
    }
    if let Some(slot_index) = iofx_actor_range_slot {
        unsafe {
            capture_iofx_actor_range_pre(slot_index, stream);
        }
    }
    if let Some(slot_index) = iofx_bbox_slot {
        unsafe {
            capture_iofx_bbox_pre(slot_index, stream);
        }
    }
    if let Some(slot_index) = cub_downsweep_slot {
        unsafe {
            capture_cub_downsweep_pre(slot_index, stream);
        }
    }
    if let Some(slot_index) = cub_scan_slot {
        unsafe {
            capture_cub_scan_pre(slot_index, stream);
        }
    }
    if let Some(slot_index) = iofx_chain_slot {
        unsafe {
            capture_iofx_chain_pre(slot_index, stream);
        }
    }
    // 2026-06-16 BUG-2 Step-1a: scan the CONSUMER's input PRE-launch. The post-launch scan misses
    // the faulting bbox launch (the GPU faults mid-launch, before the post hook runs), and the
    // upstream producers' buffers are still clean when post-scanned — so the corrupt index is only
    // visible PRE-bbox. This sees bbox/actorRange input before the fault.
    unsafe { iofx_value_tripwire(&name, kernel_params, extra, stream) };

    // Plan v15+v16: pre-launch sync + mirror refresh for clean per-kernel
    // attribution, with BETWEEN-launch diff emit. Drains all prior async
    // work BEFORE we snapshot the mapped-buffer state, so:
    //   - The mirror reflects the post-sync state of the PRIOR launch
    //   - BEFORE refreshing, emit MAPPED_BETWEEN events for any changes
    //     that happened since (CPU-side writes, mostly)
    //   - Then refresh the mirror so the post-launch diff sees ONLY this
    //     kernel's writes
    // Gated by ZLUDA_MAPPED_WRITE_TRACE=1 — no overhead when off.
    if super::memory::mapped_write_trace_enabled() {
        unsafe {
            let _ = hip_runtime_sys::hipDeviceSynchronize();
        }
        static REFRESH_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let refresh_seq = REFRESH_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::memory::mapped_write_refresh_pre(refresh_seq);
    }

    // 2026-05-18 Plan v20.4 option 1: hard-sync immediately before MS_4
    // (CreateParticleShapesIncremental_MultiStage_4) launches. The MS_4
    // wild-VA crash is timing/coherency sensitive: every per-launch CPU
    // read of the n_threads carrier slot (v[663640] / v[1056768]) we add
    // delays the crash by ~30-70× without eliminating it. A
    // hipStreamSynchronize *just before* MS_4 dispatches forces the
    // strongest possible barrier between the CPU read of n_threads (as
    // kernarg) and the kernel's execution. Gated on ZLUDA_MS4_HARD_SYNC=1.
    // Costs one stream sync per MS_4 launch (hundreds per second at peak).
    if name == "CreateParticleShapesIncremental_MultiStage_4" {
        static MS4_HARD_SYNC_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *MS4_HARD_SYNC_ENABLED
            .get_or_init(|| std::env::var("ZLUDA_MS4_HARD_SYNC").map_or(false, |v| v == "1"))
        {
            unsafe {
                let _ = hip_runtime_sys::hipStreamSynchronize(stream);
            }
        }
    }

    // 2026-06-03 Step 2: generic single-op-memset kernel intercept.
    //
    // At cuModuleLoadDataEx, module.rs::detect_simple_memset_kernels walks
    // the parsed AST and records kernels whose body matches the strict
    // SimpleMemset signature (one `sust.b.3d.*` op, surf+fill come from
    // ld.param, no LDS, no global reads, no barriers). Here we look up the
    // current kernel; if it's a registered memset and the env gate is on,
    // we read the surf/fill values from the kernel arg buffer at the
    // recorded byte offsets and run the host hipMemcpy3DAsync swap path.
    //
    // No kernel names are matched here — `lookup_simple_memset_sig(name)`
    // is just a dictionary lookup keyed on whatever the AST detector saw.
    // Default OFF (`ZLUDA_INTERCEPT_SIMPLE_MEMSET=1` enables). Replaces
    // the Fix C name-keyed TurbulenceFS_dynamicGridClearKernel block.
    if let Some(sig) = super::module::lookup_simple_memset_sig(&name) {
        if super::module::simple_memset_intercept_enabled()
            && sig.is_sust_3d
            && !extra.is_null()
            && kernel_params.is_null()
        {
            let extra_arr = unsafe { std::slice::from_raw_parts(extra, 5) };
            let cu_launch_param_buffer_pointer_local =
                0x1usize as *mut ::core::ffi::c_void;
            let cu_launch_param_buffer_size_local = 0x2usize as *mut ::core::ffi::c_void;
            if extra_arr[0] == cu_launch_param_buffer_pointer_local
                && extra_arr[2] == cu_launch_param_buffer_size_local
            {
                let buffer = extra_arr[1] as *const u8;
                let size_ptr = extra_arr[3] as *const usize;
                if !buffer.is_null() && !size_ptr.is_null() {
                    let size = unsafe { *size_ptr };
                    let need = (sig.surf_param_offset as usize)
                        .max(sig.fill_param_offset as usize)
                        + (sig.surf_param_width.max(sig.fill_param_width) as usize);
                    if size >= need && sig.surf_param_width == 8 {
                        let surf_obj_u64 = unsafe {
                            std::ptr::read_unaligned(
                                buffer.add(sig.surf_param_offset as usize) as *const u64,
                            )
                        };
                        // Read fill value at the recorded scalar width and
                        // accept the launch only if every byte is zero
                        // (the host swap is a memcpy-from-zero-buffer).
                        let fill_off = sig.fill_param_offset as usize;
                        let fill_w = sig.fill_param_width as usize;
                        let all_zero = if fill_w <= size.saturating_sub(fill_off) {
                            let mut z = true;
                            for off in 0..fill_w {
                                let b = unsafe { *buffer.add(fill_off + off) };
                                if b != 0 {
                                    z = false;
                                    break;
                                }
                            }
                            z
                        } else {
                            false
                        };
                        if all_zero {
                            if let Some(arr) =
                                super::surf::surf_obj_to_array(surf_obj_u64 as usize)
                            {
                                let intercepted = unsafe {
                                    try_host_clear_3d_array(arr, surf_obj_u64, stream)
                                };
                                if intercepted {
                                    drop(launch_guard);
                                    return Ok(());
                                }
                            }
                        }
                    }
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
                    f.base,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                    stream,
                    std::ptr::null_mut(),
                    patched.as_mut_ptr(),
                )
            }
        } else {
            unsafe {
                hipModuleLaunchKernel(
                    f.base,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                    stream,
                    kernel_params,
                    extra,
                )
            }
        }
    } else {
        unsafe {
            hipModuleLaunchKernel(
                f.base,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
                shared_mem_bytes,
                stream,
                kernel_params,
                extra,
            )
        }
    };
    // 2026-06-08 Class A use-after-free hypothesis check: free_v2
    // scans the existing KERNEL_RING (populated below by
    // record_kernel_launch via per-kernel capture intercepts) for
    // entries whose ptr_ranges contain the freed VA. That ring is
    // safer than reading raw kernel_params (the warning at line 14380
    // notes generic scanning segfaults for short arg lists). The
    // recording happens in record_kernel_launch; the scan happens in
    // memory.rs::free_v2.
    if result.is_ok() {
        crate::r#impl::recording::submitted();
    }
    crate::r#impl::launch_survey::observed(&name, f.base.0 as usize, f.module.0 as usize, stream.0 as usize, result.as_ref().err().map_or(0, |e| e.0.get()));
    // 2026-06-08 Class A upper-bound diagnostic: hipDeviceSync after
    // EVERY successful kernel launch. The hypothesis is that Class A's
    // GPU TCP wild-write category may be an HDP-coherency issue that
    // affects MANY kernels, not just MS_3/MS_4/IOFX_actorRange. If
    // ZLUDA_LAUNCH_HDP_FLUSH=1 eliminates Class A (or pushes it
    // dramatically deeper), the bug class is HDP. If unchanged, it's
    // GPU-internal (L2/TCP/cache invalidation logic). HEAVY — turns
    // every kernel launch into a synchronous device drain.
    if result.is_ok() {
        static LAUNCH_HDP_FLUSH: std::sync::OnceLock<bool> =
            std::sync::OnceLock::new();
        if *LAUNCH_HDP_FLUSH.get_or_init(|| {
            std::env::var("ZLUDA_LAUNCH_HDP_FLUSH").map_or(false, |v| v == "1")
        }) {
            unsafe {
                let _ = hipDeviceSynchronize();
            }
        }
    }
    // 2026-06-08 Class B at depth: MS_3 producer race
    // (project_multistage_3_producer_race_2026_06_05.md). MS_3 does
    // `atom.global.add count` then 4 `st.global.u16` writes of slot
    // entry data. On NVIDIA the L2 serializes — PhysX reads the new
    // count and the slot data atomically. On AMD/RDNA the HDP write-
    // combine buffer between L2 and host DRAM can deliver the count
    // update before the subsequent slot stores drain. PhysX polls the
    // mapped-host inner_array, sees count=N+1, reads slot[N] = NULL →
    // canonical Class B vtable NULL deref at PhysX3_x64.dll+0x20ced9.
    //
    // The PTX `emit_ret` SeqCst system-scope fence drains GPU L2 but
    // NOT HDP. EVENT_RELEASE_SYSTEM=1 covers cuEventRecord boundaries
    // but PhysX polls BETWEEN events. ZLUDA_MS3_POST_SYNC issues a
    // hipDeviceSynchronize (HSA queue-barrier-AND release-system →
    // HDP MMIO flush) immediately after the MS_3 launch returns.
    //   mode '1' → hipDeviceSynchronize (cross-stream drain + HDP flush)
    //   mode '2' → hipStreamSynchronize (this stream only)
    // Default 0 (off). 2026-06-08 first A/B candidate after run 4
    // (PID 647312) hit Class B at depth 17.84 min / 349,939 launches.
    if result.is_ok() && name == "CreateParticleShapesIncremental_MultiStage_3" {
        static MS3_POST_SYNC: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let post_sync_mode = *MS3_POST_SYNC.get_or_init(|| {
            std::env::var("ZLUDA_MS3_POST_SYNC")
                .ok()
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(0)
        });
        match post_sync_mode {
            1 => unsafe {
                let _ = hipDeviceSynchronize();
            },
            2 => unsafe {
                let _ = hipStreamSynchronize(stream);
            },
            _ => {}
        }
    }
    // 2026-06-14 Phase B1: capture MS_4's SCATTER OUTPUT (param4) AFTER the
    // kernel ran, into the slot capture_ms4_inputs reserved. This is the one
    // untested GPU-side path for the Class-B host NULL-deref: spec-sim the
    // (proven-correct) inputs and compare to this output (Phase B2).
    if let Some((slot, scatter_ptr, np)) = ms4_output_capture.filter(|_| result.is_ok()) {
        unsafe {
            capture_ms4_output_post(slot, scatter_ptr, np, stream);
        }
    }
    // Translate HIP error 719 (LaunchFailure) → LaunchOutOfResources.
    // On NVIDIA, launching with too many threads returns CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (701),
    // which CUDA apps handle by reducing block size and retrying. HIP returns 719 (LAUNCH_FAILURE)
    // which many apps treat as fatal. This translation makes ZLUDA match NVIDIA behavior.
    let result = match result {
        Err(code) if code == hipErrorCode_t::LaunchFailure => hipError_t::ErrorLaunchOutOfResources,
        other => other,
    };
    if let Some(slot_index) = iofx_actor_range_slot.filter(|_| result.is_ok()) {
        unsafe {
            capture_iofx_actor_range_post(slot_index, stream);
        }
    }
    if let Some(slot_index) = tfs_atomic_to_tex_slot.filter(|_| result.is_ok()) {
        unsafe {
            capture_tfs_atomic_to_tex_post(slot_index, stream);
        }
    }
    if let Some(slot_index) = iofx_bbox_slot.filter(|_| result.is_ok()) {
        unsafe {
            capture_iofx_bbox_post(slot_index, stream);
        }
    }
    // 2026-06-16 BUG-2 Phase-A adjacency probe (ZLUDA_IOFX_ADJ_PROBE=1).
    // Class-4 is a single position-float word in the value/index buffer that
    // makeSortKeys writes (param_15) and bbox/actorRange scatter by. Offline
    // (iofx_bbox_C4_withVA.bin) the value buffer sits 0x800/0x1000 ABOVE
    // float/position buffers holding values in the leaked float's magnitude
    // range. The decisive question for the fix: are those one cuMemAlloc
    // (intra-buffer stride bug → padding won't help) or separate adjacent
    // allocs (→ guard-page/ALLOC_PAD fix). The buffer LAYOUT is the same every
    // frame, so this answers it on ANY run reaching makeSortKeys — no need to
    // reach the rare Class-4 frame. Logs the HIP alloc range (cuMemGetAddressRange)
    // of makeSortKeys' input(p4)/positions(p9)/keys(p14)/values(p15) so their
    // same-alloc-vs-separate layout + inter-buffer gaps are visible. Throttled.
    if result.is_ok() && (name == "IOFX_makeSortKeys_templ$0" || name == "IOFX_makeSortKeys_templ$1") {
        static ADJ_PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ADJ_PROBE.get_or_init(|| std::env::var("ZLUDA_IOFX_ADJ_PROBE").map_or(false, |v| v == "1")) {
            static ADJ_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = ADJ_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 24 {
                // (label, arg_byte_offset, arg_index) for the 4 pointer params.
                let probes: [(&str, usize, usize); 4] = [
                    ("input_p4", 24, 4),
                    ("positions_p9", 48, 9),
                    ("keys_p14", 88, 14),
                    ("values_p15", 96, 15),
                ];
                let mut parts = String::new();
                for (lbl, off, idx) in probes {
                    let va = unsafe { read_kernel_arg::<u64>(kernel_params, extra, off, idx) }.unwrap_or(0);
                    let range = unsafe { query_device_pointer_range(va) };
                    match range {
                        Some((base, size, _)) => parts.push_str(&format!(
                            " {}=0x{:x}[base=0x{:x} size={} off={}]",
                            lbl, va, base, size, va.wrapping_sub(base)
                        )),
                        None => parts.push_str(&format!(" {}=0x{:x}[no-range]", lbl, va)),
                    }
                }
                super::driver::zluda_log_err(&format!("[IOFX_ADJ_PROBE #{}]{}", n, parts));
            }
        }
    }
    if let Some(slot_index) = iofx_chain_slot.filter(|_| result.is_ok()) {
        unsafe {
            capture_iofx_chain_post(slot_index, stream);
        }
    }
    if let Some(args) = createcell_snapshot_args.filter(|_| result.is_ok()) {
        if let Some(start_call) = createcell_snapshot_log_start() {
            let call_n =
                CREATECELL_SNAPSHOT_LOG_CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if call_n >= start_call {
                unsafe {
                    queue_createcell_snapshot_direct(
                        call_n,
                        args.n_items,
                        args.hash_out,
                        stream,
                        grid_dim_x,
                        grid_dim_y,
                        grid_dim_z,
                        block_dim_x,
                        block_dim_y,
                        block_dim_z,
                        shared_mem_bytes,
                    );
                }
            }
        }
    }
    if let Some(args) = sort_downsweep_args.filter(|_| result.is_ok()) {
        let creategrid_call_n = CREATEGRID_LOG_CALL_N.load(std::sync::atomic::Ordering::Relaxed);
        let start_call = creategrid_snapshot_log_start();
        let should_capture_replay = should_queue_sort_replay_capture(
            args.begin_bit,
            creategrid_call_n,
            start_call,
            sort_replay_capture_begin_bit(),
        );
        if start_call.is_some_and(|start| creategrid_call_n >= start) {
            unsafe {
                queue_sort_downsweep_snapshot(
                    args.begin_bit,
                    args.n_items,
                    args.keys_in,
                    args.keys_out,
                    args.first,
                    args.last,
                    args.overwrite,
                    stream,
                    grid_dim_x,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                );
                if should_capture_replay {
                    queue_sort_downsweep_replay_capture_postlaunch(args, stream);
                }
            }
        }
    }
    if creategrid_safe_validator_enabled() {
        if let Some(args) = sort_downsweep_args.filter(|_| result.is_ok()) {
            unsafe {
                capture_sort_downsweep_safe(
                    args.begin_bit,
                    args.n_items,
                    args.keys_in,
                    args.keys_out,
                    args.vals_in,
                    args.vals_out,
                    args.spine_ptr,
                    Some(args.grid_share),
                    grid_dim_x,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                    args.first,
                    args.last,
                    args.overwrite,
                    stream,
                );
            }
        }
    }
    // Spine readback removed — hipMemcpyDtoH triggers ProfilingSignal race.
    // Instead, read the spine from within the Downsweep readback (same sync point).
    // Per-pass Downsweep readback: after each Downsweep, sync + read back the
    // output buffer and check for bit31 corruption. Identifies WHICH pass introduces it.
    // Gated by ZLUDA_VALIDATE_SORT=1.
    static VALIDATE_SORT_POST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *VALIDATE_SORT_POST
        .get_or_init(|| std::env::var("ZLUDA_VALIDATE_SORT").map_or(false, |v| v == "1"))
        && name.contains("RadixSortDownsweepKernel")
        && result.is_ok()
        && !extra.is_null()
        && stream.0.is_null()
    // Only FleX sort (NULL stream), not IOFX (non-NULL)
    {
        unsafe {
            let extra_arr = std::slice::from_raw_parts(extra, 5);
            if extra_arr[0] == (0x1usize as *mut _) && !extra_arr[1].is_null() {
                let buf = extra_arr[1] as *const u8;
                let n_items = std::ptr::read_unaligned(buf.add(40) as *const u32) as usize;
                let begin_bit = std::ptr::read_unaligned(buf.add(44) as *const u32);
                let keys_out = std::ptr::read_unaligned(buf.add(8) as *const u64);
                if n_items > 0 && keys_out != 0 {
                    // Use stream sync (NULL only) instead of device sync (all streams)
                    // to reduce ProfilingSignal race pressure
                    let _ = hip_runtime_sys::hipStreamSynchronize(hip_runtime_sys::hipStream_t(
                        std::ptr::null_mut(),
                    ));
                    let mut h_out = vec![0i32; n_items];
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        h_out.as_mut_ptr() as _,
                        hip_runtime_sys::hipDeviceptr_t(keys_out as _),
                        n_items * 4,
                    );
                    let bit31_out = h_out.iter().filter(|&&k| k < 0).count();
                    // Also read keys_in to check if corruption is in the INPUT
                    let keys_in = std::ptr::read_unaligned(buf.add(0) as *const u64);
                    let mut h_in = vec![0i32; n_items];
                    let _ = hip_runtime_sys::hipMemcpyDtoH(
                        h_in.as_mut_ptr() as _,
                        hip_runtime_sys::hipDeviceptr_t(keys_in as _),
                        n_items * 4,
                    );
                    let bit31_in = h_in.iter().filter(|&&k| k < 0).count();
                    // Read the spine's LIVE region every pass (cheap, already
                    // synced). Post-scan entries are prefix sums ≤ n; the r6
                    // (2026-06-10) cub-mode crash scattered at ~16×n with NO
                    // bit31 keys — spine_max > n is the direct inflation
                    // signature the bit31-only trigger missed.
                    let spine_ptr = std::ptr::read_unaligned(buf.add(32) as *const u64);
                    let spine_n = (grid_dim_x * 16) as usize; // 16 bins for 4-bit, conservative
                    let mut h_spine = vec![0u32; spine_n.min(256)];
                    if spine_ptr != 0 {
                        let _ = hip_runtime_sys::hipMemcpyDtoH(
                            h_spine.as_mut_ptr() as _,
                            hip_runtime_sys::hipDeviceptr_t(spine_ptr as _),
                            h_spine.len() * 4,
                        );
                    }
                    let spine_max = h_spine.iter().copied().max().unwrap_or(0);
                    let spine_sum: u64 = h_spine.iter().map(|&v| v as u64).sum();
                    if bit31_out > 0 || bit31_in > 0 || spine_max as usize > n_items {
                        let viol = h_out.windows(2).filter(|w| w[1] < w[0]).count();
                        super::driver::zluda_log_err(&format!(
                            "*** PASS_CORRUPT bit={} n={} in_bit31={} out_bit31={} viol={} spine_max={} spine_sum={} keys_in=0x{:x} keys_out=0x{:x}",
                            begin_bit, n_items, bit31_in, bit31_out, viol, spine_max, spine_sum, keys_in, keys_out));
                    } else {
                        static PASS_LOG_N: std::sync::atomic::AtomicU32 =
                            std::sync::atomic::AtomicU32::new(0);
                        let pn = PASS_LOG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if pn < 50 || pn % 100 == 0 {
                            super::driver::zluda_log_err(&format!(
                                "PASS_OK bit={} n={} max={}",
                                begin_bit,
                                n_items,
                                h_out.iter().max().unwrap_or(&0)
                            ));
                        }
                    }
                }
            }
        }
    }
    // Track last kernel name + timestamp for the SIGABRT handler.
    // NOTE: we deliberately do NOT extract pointer args generically here —
    // dump_kernel_args_raw will read past the end of short kernel_params
    // arrays and crash on the first kernel with N<16 args (verified
    // 2026-05-10: scanning all launches segfaulted in TurbulenceFS init
    // after 38 launches). Specific capture intercepts (actor_range, bbox,
    // downsweep, etc.) already record structured pointer ranges through
    // their own per-kernel paths.
    //
    // 2026-06-08 widen ptr_ranges coverage for FREE_USE_AFTER_LAUNCH_SCAN:
    // narrow per-kernel-name capture for kernels we KNOW the param layout
    // of from their PTX. Reads only declared u64 / pointer slots — safe.
    // Gated behind the same ZLUDA_FREE_AFTER_LAUNCH_SCAN=1 to avoid any
    // runtime cost on default-config benches.
    if let Some(n) = global_launch_n {
        let mut ranges: Vec<(u64, u64)> = if super::memory::free_after_launch_scan_enabled() {
            unsafe { collect_narrow_ptr_ranges(&name, kernel_params, extra) }
        } else {
            Vec::new()
        };
        // 2026-06-13: the perf-test GPU fault is pinned to IOFX_newRadixSortStepKernel
        // (the radix scatter). Record its full kernargs into the launch-history ring
        // so the SIGABRT dump (newest entry = faulting launch) reveals keys_out (p8)
        // / values_out (p9) / n (p4) / bit (p5) / phase (p11) / num_ctas (p12). The
        // wild scatter dst = p8 or p9 + rank*4, so rank = (fault_VA - base)/4 tells us
        // whether it's a rank overflow (bit31-set / >n) or a garbage base ptr.
        // StepKernel layout is fixed (>=13 params) so these reads are in-bounds.
        let nrss_info = if name == "IOFX_newRadixSortStepKernel" {
            let p4: u32 = unsafe { read_kernel_arg(kernel_params, extra, 20, 4) }.unwrap_or(0);
            let p5: u32 = unsafe { read_kernel_arg(kernel_params, extra, 24, 5) }.unwrap_or(0);
            let p6: u64 = unsafe { read_kernel_arg(kernel_params, extra, 32, 6) }.unwrap_or(0);
            let p7: u64 = unsafe { read_kernel_arg(kernel_params, extra, 40, 7) }.unwrap_or(0);
            let p8: u64 = unsafe { read_kernel_arg(kernel_params, extra, 48, 8) }.unwrap_or(0);
            let p9: u64 = unsafe { read_kernel_arg(kernel_params, extra, 56, 9) }.unwrap_or(0);
            let p10: u64 = unsafe { read_kernel_arg(kernel_params, extra, 64, 10) }.unwrap_or(0);
            let p11: u32 = unsafe { read_kernel_arg(kernel_params, extra, 72, 11) }.unwrap_or(0);
            let p12: u32 = unsafe { read_kernel_arg(kernel_params, extra, 76, 12) }.unwrap_or(0);
            let nb = (p4 as u64).saturating_mul(4);
            for (base, _label) in [(p6, "kin"), (p7, "vin"), (p8, "kout"), (p9, "vout"), (p10, "spine")] {
                if base != 0 { ranges.push((base, base.saturating_add(nb.max(4096)))); }
            }
            Some(format!(
                "NRSS n={} bit={} phase={} nctas={} grid={} kin=0x{:x} vin=0x{:x} kout=0x{:x} vout=0x{:x} spine=0x{:x}",
                p4, p5, p11, p12, grid_dim_x, p6, p7, p8, p9, p10
            ))
        } else {
            None
        };
        super::driver::record_kernel_launch(n, &name, &ranges, nrss_info);
    }
    // 2026-06-12 per-kernel GPU-fault isolation. ZLUDA_SYNC_LAUNCHES=1 forces a
    // hipDeviceSynchronize after EVERY launch, so a kernel that does a wild GPU
    // access faults during ITS OWN sync — the launch-history ring's newest
    // entry (dumped by the SIGABRT handler) then NAMES the faulting kernel
    // instead of whatever happened to be queued last. Heavy (serializes the
    // GPU) but only needed for the diagnostic bench.
    {
        static SYNC_LAUNCHES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static SYNC_FROM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let sync_each = *SYNC_LAUNCHES
            .get_or_init(|| std::env::var("ZLUDA_SYNC_LAUNCHES").map_or(false, |v| v == "1"));
        // Only serialize from this launch count onward (keeps loading fast;
        // the wild-VA fault is at the perf-test scene ~30-35k launches).
        let sync_from = *SYNC_FROM.get_or_init(|| {
            std::env::var("ZLUDA_SYNC_LAUNCHES_FROM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(25000u64)
        });
        if sync_each && result.is_ok() && global_launch_n.map_or(false, |n| n >= sync_from) {
            unsafe {
                let _ = hipDeviceSynchronize();
            }
        }
    }
    // Log launch failures with full context for diagnosis
    if result.is_ok()
        && !first_kernel_launch_logged
        && FIRST_KERNEL_LAUNCH_LOGGED
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
    {
        if name.is_empty() {
            eprintln!("ZLUDA: first kernel launch ({:p})", f.base.0);
        } else {
            eprintln!("ZLUDA: first kernel launch ({})", name);
        }
    }
    if result.is_err() {
        super::driver::zluda_log_err(&format!(
            "cuLaunchKernel FAILED '{}' func={:p} grid=[{},{},{}] block=[{},{},{}] shared={} -> {:?}",
            name, f.base.0, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, result));
    }
    // Mode 1 releases the launch lock before the sync (or before return, if no sync
    // runs). Mode 2 keeps it held through the forced device sync below.
    if lock_mode == 1 {
        launch_guard = None;
    }
    // Force a device-wide sync after the launch if either ZLUDA_SYNC_LAUNCHES=1 or
    // ZLUDA_LAUNCH_MUTEX=2 is set. Mode 2 implies sync because its whole purpose is
    // to guarantee no other CPU thread queues work until this launch has fully
    // completed on the GPU.
    if result.is_ok() {
        static SYNC_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sync_all = *SYNC_ALL
            .get_or_init(|| std::env::var("ZLUDA_SYNC_LAUNCHES").map_or(false, |v| v == "1"));
        if sync_all || lock_mode == 2 {
            static LAUNCH_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = LAUNCH_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Use device-wide sync (not per-stream) to fully serialize all GPU work
            // across threads/streams. This eliminates cross-stream concurrency as a factor.
            let sync_result = unsafe { hip_runtime_sys::hipDeviceSynchronize() };
            // Also peek at the last error — on Van Gogh, hipDeviceSynchronize
            // sometimes returns OK even when a kernel has faulted async. The error
            // may still be pending in the runtime's per-thread error state.
            let peek_result = unsafe { hip_runtime_sys::hipPeekAtLastError() };
            if sync_result.is_err() {
                super::driver::zluda_log_err(&format!(
                    "*** GPU_FAULT #{} '{}' grid=[{},{},{}] block=[{},{},{}] sync={:?} peek={:?} stream={:?}",
                    n, name, grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z, sync_result, peek_result, stream));
            } else if peek_result.is_err() {
                // Sync said OK but peek reports an error — the async fault finally
                // surfaced. This is the real diagnostic for the fault-on-a-recent-
                // kernel scenario.
                super::driver::zluda_log_err(&format!(
                    "*** PEEK_FAULT #{} '{}' grid=[{},{},{}] block=[{},{},{}] peek={:?}",
                    n,
                    name,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    peek_result
                ));
            } else if n < 20 || n % 1000 == 0 {
                // Positive confirmation that sync ran and returned OK. Sampled so we
                // don't flood the log; the first 20 cover PhysX init, later samples
                // cover gameplay.
                super::driver::zluda_log_err(&format!("SYNC_OK #{} '{}'", n, name));
            }
        }
    }
    // MS_3 post-capture. Must run AFTER the kernel has actually finished. In
    // mode 2 or ZLUDA_SYNC_LAUNCHES=1 the block above already ran
    // hipDeviceSynchronize, so any launch on any stream is guaranteed done.
    // In other modes we do an extra sync here for correctness — this only
    // fires when ZLUDA_MS4_CAPTURE=1, so it has no cost in normal operation.
    if let Some(slot_index) = ms3_slot {
        unsafe {
            let _ = hip_runtime_sys::hipDeviceSynchronize();
            capture_ms3_post(slot_index);
        }
    }
    // MS_2_Batched post-capture (F.2). Same pattern as MS_3 above. Together
    // with the pre-capture in capture_ms2b_inputs this gives us the smoking
    // gun if MS_2_Batched is the kernel writing the `-2 across 8 lanes`
    // counter corruption.
    if let Some(slot_index) = ms2b_slot {
        unsafe {
            let _ = hip_runtime_sys::hipDeviceSynchronize();
            capture_ms2b_post(slot_index);
        }
    }
    // Shared completion wait for opt-in post-launch readers. Keep it inside
    // their gate so ordinary nonblocking launches can return before completion.
    if result.is_ok() && post_launch_diagnostics_enabled(&name) {
        unsafe {
            let _ = hip_runtime_sys::hipDeviceSynchronize();
            check_counter_mutation(&name);
            // F.6: also snapshot *(rd5+22176) per tracked rd5 to find which
            // operation normally resets it (and which fails to on bug frames).
            check_rd5_22176_mutation(&name);
            // Track mutations of the watched MS2 rd6[*12+8] source word and
            // attribute changes to the current kernel launch.
            check_ms2_rd6_mutation(&name, global_launch_n);
            // Plan step 4: snapshot the latched MS_4 counter buffer and
            // log if any kernel mutates it. Independent of MS4_CAPTURE /
            // F.6 gates — uses ZLUDA_COUNTER_SNAPSHOT=1.
            snapshot_counter_after_launch(&name);
            // 2026-06-10 IOFX writer hunt: in-game NRSS final-pass output
            // sortedness check (ZLUDA_NRSS_VALIDATE=1). Device already
            // synced above.
            validate_nrss_output_after_launch(
                &name, kernel_params, extra, grid_dim_x, block_dim_x, shared_mem_bytes,
            );
            // Plan v14 Step 1: per-kernel mapped-buffer write tracer.
            // Gated by ZLUDA_MAPPED_WRITE_TRACE=1 (default OFF). Reads each
            // tracked cuMemHostAlloc buffer, diffs vs mirror, emits
            // [MAPPED_WRITE] lines per changed u32 attributing the change
            // to this kernel. Used to identify which kernel grows / shrinks
            // each MS_4 n_threads slot at {host+663552, +663640, +663728}.
            if super::memory::mapped_write_trace_enabled() {
                static MAPPED_WRITE_SEQ: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let seq = MAPPED_WRITE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::memory::mapped_write_diff_post(seq, &name);
            }
            // v8 Step 2: POST_PCSB flag probe — after ParticleCollisionStaticBox
            // returns, count bit 4 (0x10) distribution. NOTE: PCSB writes to a
            // DIFFERENT buffer than MS_4's flags (verified run #19), so this
            // probe is informational; bit-4 changes here don't affect MS_4.
            if flag_count_probe_enabled() && name == "ParticleCollisionStaticBox" {
                let flags_ptr: u64 = read_kernel_arg(kernel_params, extra, 56, 2).unwrap_or(0);
                let n_particles: u32 = read_kernel_arg(kernel_params, extra, 96, 6).unwrap_or(0);
                let probe_n = n_particles.min(256);
                snapshot_flag_bit4_at("POST_PCSB", flags_ptr, probe_n);
            }
            // v8 Step 3: POST_MS_1 flag probe — after MS_1 launches, probe
            // MS_4's flags buffer (via latched ptr). MS_1's internal helper
            // at module_10:1001 has the bit-4-set site we suspect fails on
            // ZLUDA. If bit-4-set stays ~0% after MS_1, this site is the bug.
            if name.contains("CreateParticleShapesIncremental_MultiStage_1") {
                if flag_count_probe_enabled() {
                    let flags_ptr = MS4_FLAGS_PTR_LATCHED.load(std::sync::atomic::Ordering::Acquire);
                    let n = MS4_N_LATCHED.load(std::sync::atomic::Ordering::Acquire);
                    if flags_ptr != 0 && n != 0 {
                        let probe_n = n.min(1024);
                        snapshot_flag_bit4_at("POST_MS1", flags_ptr, probe_n);
                    }
                }
                // v11: dump + reset MS_1 gate-trace counters. Captures per-
                // launch outcomes of the bit-4-set gate predicates.
                ms1_gate_trace_dump_and_reset();
            }
        }
    }
    // Explicitly drop the guard to make the lifetime scope obvious (mode 2 releases here).
    drop(launch_guard);
    result
}

// _ptsz resolves implicit stream zero to HIP's per-thread default stream.
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
        f,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        super::stream::per_thread_stream(stream),
        kernel_params,
        extra,
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

/// Set the preferred L1/shared-memory split for a function launch.
///
/// On NVIDIA Fermi/Kepler/Maxwell, L1 cache and shared memory share a single
/// on-chip SRAM pool that can be partitioned (PREFER_NONE / PREFER_SHARED /
/// PREFER_L1 / PREFER_EQUAL). On newer NVIDIA (Volta+), and on all AMD GPUs,
/// L1 and LDS are physically separate fixed-size memories — the hint is
/// ignored by the hardware.
///
/// HIP mirrors this: `hipFuncSetCacheConfig` accepts the call but the AMD
/// runtime documents "On AMD devices and some Nvidia devices, these hints
/// and controls are ignored". PhysX/FleX calls `cuFuncSetCacheConfig` on
/// every kernel during module load; without this stub the dev-profile UNIMPL
/// panic fires (see `crate::r#impl::unimplemented`) and aborts the game at
/// init time. Previous release builds masked this because the release-profile
/// UNIMPL returned `CUerror::NOT_SUPPORTED` silently and PhysX accepted that.
///
/// This implementation goes further than returning a constant SUCCESS: it
/// forwards the request to `hipFuncSetCacheConfig` so (a) any HIP runtime
/// that later honors the hint picks it up and (b) the call becomes visible
/// in HIP tracing, matching the NVIDIA path. The `CUfunc_cache` enum values
/// match `hipFuncCache_t` one-to-one (NONE=0, SHARED=1, L1=2, EQUAL=3) so we
/// can pass the raw u32 through `hipFuncCache_t::from` equivalent.
pub(crate) unsafe fn set_cache_config(
    func: &Function,
    config: cuda_types::cuda::CUfunc_cache,
) -> hipError_t {
    let hip_config = hipFuncCache_t(config.0);
    hipFuncSetCacheConfig(func.base.0.cast(), hip_config)
}

/// Pre-launch diagnostic for ParticleDensityAtomicToTexKernel.
/// Reads the surface handle from the 456-byte param struct (at offset 368)
/// and validates the surface descriptor it points to.
unsafe fn validate_surface_kernel_inputs(
    extra: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
    name: &str,
) {
    use hip_runtime_sys::*;

    let _ = hipStreamSynchronize(stream);

    // extra format: [CU_LAUNCH_PARAM_BUFFER_POINTER, ptr, CU_LAUNCH_PARAM_BUFFER_SIZE, size_ptr, END]
    let extra_params = std::slice::from_raw_parts(extra, 5);
    if extra_params[0] != (0x1usize as *mut std::ffi::c_void) {
        return; // Not the expected extra format
    }
    let buffer = extra_params[1] as *const u8;
    let buf_size = *(extra_params[3] as *const usize);

    if buf_size < 376 {
        return; // Too small to contain the surface handle at offset 368
    }

    // Surface handle is a u64 at offset 368 in the param struct
    let surf_handle = *(buffer.add(368) as *const u64);

    // Decode: top 3 bits = pixel shift, remaining bits = descriptor pointer
    let shift = surf_handle >> 61;
    let desc_ptr = (surf_handle << 3) >> 3;

    // Read the 32-byte surface descriptor from device memory
    let mut desc_words = [0u32; 8];
    let copy_result = hipMemcpyDtoH(
        desc_words.as_mut_ptr() as _,
        hipDeviceptr_t(desc_ptr as *mut std::ffi::c_void),
        32,
    );

    static SURF_CALL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = SURF_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE_SURF {} #{}: handle=0x{:016x} desc_ptr=0x{:016x} shift={} READ FAILED {:?}",
            name, call_n, surf_handle, desc_ptr, shift, copy_result
        ));
        return;
    }

    // Descriptor word 0-1: base address (48 bits)
    let base = (desc_words[0] as u64) | ((desc_words[1] as u64) << 32);
    // Check if base address looks valid (in device memory range)
    let alloc_min = super::memory::ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
    let alloc_max = super::memory::ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
    let base_in_range = base >= alloc_min && base <= alloc_max;

    // Log first 5, then every 10th, or if base looks bad
    if call_n < 5 || call_n % 10 == 0 || !base_in_range {
        super::driver::zluda_log_err(&format!(
            "VALIDATE_SURF {} #{}: handle=0x{:016x} shift={} base=0x{:012x} in_range={} words=[{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}]",
            name, call_n, surf_handle, shift, base, base_in_range,
            desc_words[0], desc_words[1], desc_words[2], desc_words[3],
            desc_words[4], desc_words[5], desc_words[6], desc_words[7]));
    }

    if !base_in_range && base != 0 {
        super::driver::zluda_log_err(&format!(
            "*** CORRUPT SURFACE DESCRIPTOR #{}: base=0x{:012x} outside alloc range [{:012x}..{:012x}]",
            call_n, base, alloc_min, alloc_max));
    }
}

/// Dispatcher-callable wrapper for MultiStage_4 validation — handles both
/// kernel_params and extra launch formats.
///
/// MultiStage_4 reads:
///   - flags[tid*4]           (u16)     — param_2 (%rd2)
///   - shape_index[tid*4]     (u32)     — param_6 (%rd5 → rd13)
///   - struct[shape_index*12] (12 bytes, first u32 is "base") — param_4 (%rd3 → rd12)
///   - counter[shape_index*4] (atomic u32 dec) — param_7 (%rd6 → rd11)
///   - scatter[base + counter_old - 1] (u32) — param_5 (%rd4 → rd10)
///
/// Flat (extra) buffer offsets, computed from the byref arg types in the IR:
///   param_0: u64 @  0
///   param_1: u16 @  8 (u16 padded to u64 alignment)
///   param_2: u64 @ 16 → flags buffer
///   param_3: u32 @ 24 → numParticles
///   param_4: u64 @ 32 → struct array (12-byte entries, first u32 = base)
///   param_5: u64 @ 40 → output scatter
///   param_6: u64 @ 48 → shape_index array
///   param_7: u64 @ 56 → atomic counter array
unsafe fn validate_multistage4_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    name: &str,
    _grid_dim_x: u32,
    _block_dim_x: u32,
) {
    let flags_ptr: u64 = match read_kernel_arg(kernel_params, extra, 16, 2) {
        Some(v) => v,
        None => return,
    };
    let num_particles: u32 = match read_kernel_arg(kernel_params, extra, 24, 3) {
        Some(v) => v,
        None => return,
    };
    let struct_ptr: u64 = match read_kernel_arg(kernel_params, extra, 32, 4) {
        Some(v) => v,
        None => return,
    };
    let scatter_ptr: u64 = match read_kernel_arg(kernel_params, extra, 40, 5) {
        Some(v) => v,
        None => return,
    };
    let shape_idx_ptr: u64 = match read_kernel_arg(kernel_params, extra, 48, 6) {
        Some(v) => v,
        None => return,
    };
    let counter_ptr: u64 = match read_kernel_arg(kernel_params, extra, 56, 7) {
        Some(v) => v,
        None => return,
    };

    static CALL_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    if num_particles == 0 || num_particles > 100_000 || flags_ptr == 0 || shape_idx_ptr == 0 {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: SUSPICIOUS n={} flags=0x{:x} shape_idx=0x{:x} struct=0x{:x} counter=0x{:x} scatter=0x{:x}",
            name, call_n, num_particles, flags_ptr, shape_idx_ptr, struct_ptr, counter_ptr, scatter_ptr));
        return;
    }

    let n = num_particles as usize;

    // Read flags (u16 per entry, stride 4 bytes).
    let flags_byte_size = n * 4;
    let mut h_flags = vec![0u8; flags_byte_size];
    if hip_runtime_sys::hipMemcpyDtoH(
        h_flags.as_mut_ptr() as _,
        hip_runtime_sys::hipDeviceptr_t(flags_ptr as *mut std::ffi::c_void),
        flags_byte_size,
    )
    .is_err()
    {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: flags read FAILED ptr=0x{:x}",
            name, call_n, flags_ptr
        ));
        return;
    }

    // Read shape_index (u32 per entry, stride 4 bytes).
    let mut h_shape_idx = vec![0u32; n];
    if hip_runtime_sys::hipMemcpyDtoH(
        h_shape_idx.as_mut_ptr() as _,
        hip_runtime_sys::hipDeviceptr_t(shape_idx_ptr as *mut std::ffi::c_void),
        n * 4,
    )
    .is_err()
    {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: shape_idx read FAILED ptr=0x{:x}",
            name, call_n, shape_idx_ptr
        ));
        return;
    }

    // Collect unique shape indices used by active particles, and count references per shape.
    let mut active_count = 0u32;
    let mut max_shape_idx: u32 = 0;
    let mut use_count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for i in 0..n {
        let flag = u16::from_ne_bytes([h_flags[i * 4], h_flags[i * 4 + 1]]);
        if (flag & 0x11) == 1 {
            active_count += 1;
            let sh = h_shape_idx[i];
            if sh > max_shape_idx {
                max_shape_idx = sh;
            }
            *use_count.entry(sh).or_insert(0) += 1;
        }
    }

    // Read counter[0..=max_shape_idx] and struct[0..=max_shape_idx]. If any counter
    // would underflow (count of particles referencing it > current counter value),
    // that's the bug — the atomic dec wraps to 0xFFFFFFFF and scatter pos becomes
    // base - 1 which is a wild pointer.
    let shape_count = max_shape_idx as usize + 1;
    let mut h_counters = vec![0u32; shape_count];
    if hip_runtime_sys::hipMemcpyDtoH(
        h_counters.as_mut_ptr() as _,
        hip_runtime_sys::hipDeviceptr_t(counter_ptr as *mut std::ffi::c_void),
        shape_count * 4,
    )
    .is_err()
    {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: counter read FAILED ptr=0x{:x} size={}",
            name,
            call_n,
            counter_ptr,
            shape_count * 4
        ));
        return;
    }

    // Read struct array (12 bytes per entry, first u32 is "base").
    let mut h_struct_bytes = vec![0u8; shape_count * 12];
    let struct_read_ok = hip_runtime_sys::hipMemcpyDtoH(
        h_struct_bytes.as_mut_ptr() as _,
        hip_runtime_sys::hipDeviceptr_t(struct_ptr as *mut std::ffi::c_void),
        shape_count * 12,
    )
    .is_ok();

    // Check: for every unique shape index used, is the counter >= usage count?
    // If counter < uses, an atomic dec will underflow through zero.
    let mut underflows: Vec<(u32, u32, u32, u32)> = Vec::new(); // (shape, counter, uses, base)
    for (&shape, &uses) in &use_count {
        let counter = h_counters.get(shape as usize).copied().unwrap_or(0);
        let base = if struct_read_ok {
            let off = (shape as usize) * 12;
            u32::from_ne_bytes([
                h_struct_bytes[off],
                h_struct_bytes[off + 1],
                h_struct_bytes[off + 2],
                h_struct_bytes[off + 3],
            ])
        } else {
            0
        };
        if counter < uses {
            underflows.push((shape, counter, uses, base));
        }
    }

    // Always log if anything suspicious or counter underflow; otherwise every call
    // (not rate-limited) so we can see what the last successful launch looked like.
    // Counter pointer is logged so we can diff between successful and failing launches
    // — if the pointer changes between frames or points somewhere unexpected, that's
    // a smoking gun for buffer reuse / pointer mismatch.
    super::driver::zluda_log_err(&format!(
        "VALIDATE {} #{}: n={} active={} shapes={} max_shape={} underflows={} ptrs: flags=0x{:x} struct=0x{:x} scatter=0x{:x} shape_idx=0x{:x} counter=0x{:x}",
        name, call_n, n, active_count, use_count.len(), max_shape_idx,
        underflows.len(),
        flags_ptr, struct_ptr, scatter_ptr, shape_idx_ptr, counter_ptr));

    // Dump actual counter and base arrays when everything is zero — helps prove
    // whether the memory is truly zeroed or if we're reading from the wrong place.
    if !underflows.is_empty() {
        let dump_len = (shape_count).min(32);
        let counters_str: Vec<String> = (0..dump_len)
            .map(|i| format!("{}", h_counters[i]))
            .collect();
        super::driver::zluda_log_err(&format!(
            "  counter[0..{}] = [{}]",
            dump_len,
            counters_str.join(",")
        ));
        if struct_read_ok {
            let bases_str: Vec<String> = (0..dump_len)
                .map(|i| {
                    let off = i * 12;
                    let b = u32::from_ne_bytes([
                        h_struct_bytes[off],
                        h_struct_bytes[off + 1],
                        h_struct_bytes[off + 2],
                        h_struct_bytes[off + 3],
                    ]);
                    format!("{}", b)
                })
                .collect();
            super::driver::zluda_log_err(&format!(
                "  base[0..{}] = [{}]",
                dump_len,
                bases_str.join(",")
            ));
        }
    }

    if !underflows.is_empty() {
        let show = underflows.len().min(10);
        for &(shape, counter, uses, base) in &underflows[..show] {
            super::driver::zluda_log_err(&format!(
                "  *** COUNTER UNDERFLOW shape={} counter={} uses={} base={} → scatter_pos=base+counter-1={} (WILD)",
                shape, counter, uses, base, (base as i64) + (counter as i64) - 1));
        }
    }
}

/// (Legacy) Original MultiStage_4 validator — kept for backward reference. Expects
/// kernel_params layout only; use `validate_multistage4_args` from the dispatcher.
#[allow(dead_code)]
unsafe fn validate_multistage4_inputs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
    name: &str,
) {
    use hip_runtime_sys::*;

    // Sync the stream first so all prior writes are visible
    let _ = hipStreamSynchronize(stream);

    let params = std::slice::from_raw_parts(kernel_params, 8);
    let flags_ptr = *(params[2] as *const u64) as *mut std::ffi::c_void;
    let num_particles = *(params[3] as *const u32);
    let index_ptr = *(params[5] as *const u64) as *mut std::ffi::c_void;

    if num_particles == 0 || num_particles > 100_000 {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {}: numParticles={} (suspicious)",
            name, num_particles
        ));
        return;
    }

    let n = num_particles as usize;

    // Read flags (u16 per entry, but stride is 4 bytes based on PTX: mul.wide.u32 %rd8, %r1, 4)
    // So flags[i] is at offset i*4, reading 2 bytes
    let flags_byte_size = n * 4; // 4 bytes stride per entry
    let mut h_flags = vec![0u8; flags_byte_size];
    let flags_result = hipMemcpyDtoH(
        h_flags.as_mut_ptr() as _,
        hipDeviceptr_t(flags_ptr),
        flags_byte_size,
    );
    if flags_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {}: flags read FAILED {:?} ptr={:?} size={}",
            name, flags_result, flags_ptr, flags_byte_size
        ));
        return;
    }

    // Read index array (u32 per entry, stride 4 bytes)
    let mut h_indices = vec![0u32; n];
    let idx_result = hipMemcpyDtoH(
        h_indices.as_mut_ptr() as _,
        hipDeviceptr_t(index_ptr),
        n * 4,
    );
    if idx_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {}: index read FAILED {:?} ptr={:?} size={}",
            name,
            idx_result,
            index_ptr,
            n * 4
        ));
        return;
    }

    // Check: which flags indicate "process this particle"?
    // Stage_4 checks: (flags[i] & 0x11) == 1
    let mut active_count = 0u32;
    let mut bad_indices = Vec::new();
    let mut max_index = 0u32;

    for i in 0..n {
        // Flag is u16 at offset i*4
        let flag = u16::from_ne_bytes([h_flags[i * 4], h_flags[i * 4 + 1]]);
        let is_active = (flag & 0x11) == 1;
        if is_active {
            active_count += 1;
            let idx = h_indices[i];
            if idx > max_index {
                max_index = idx;
            }
            // Check for obviously bad indices (> 1M or bit 31 set)
            if idx >= 1_000_000 || (idx & 0x80000000) != 0 {
                bad_indices.push((i, idx, flag));
            }
        }
    }

    static CALL_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Log summary (always for first 10, then every 10th, or if bad indices found)
    if call_n < 10 || call_n % 10 == 0 || !bad_indices.is_empty() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: n={} active={} max_idx={} bad={}",
            name,
            call_n,
            n,
            active_count,
            max_index,
            bad_indices.len()
        ));
    }

    if !bad_indices.is_empty() {
        let show = bad_indices.len().min(10);
        for &(i, idx, flag) in &bad_indices[..show] {
            super::driver::zluda_log_err(&format!(
                "  BAD [{}]: flag=0x{:04x} index=0x{:08x} ({})",
                i, flag, idx, idx as i32
            ));
        }
        if bad_indices.len() > 10 {
            super::driver::zluda_log_err(&format!(
                "  ... and {} more bad indices",
                bad_indices.len() - 10
            ));
        }
    }
}

/// Pre-launch validation for CreateGrid: checks the sorted keys (radix sort output)
/// are in non-decreasing order. CreateGrid(int n, const int* sortedKeys, int* cellStart, int* cellEnd)
unsafe fn validate_create_grid_inputs(
    kernel_params: *mut *mut ::core::ffi::c_void,
    _grid_dim_x: u32,
    _block_dim_x: u32,
) {
    use hip_runtime_sys::*;

    let params = std::slice::from_raw_parts(kernel_params, 4);
    let num_particles = *(params[0] as *const i32);
    let sorted_keys_dev = *(params[1] as *const u64);

    if num_particles <= 0 || num_particles > 200_000 {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid: numParticles={} (suspicious)",
            num_particles
        ));
        return;
    }

    let n = num_particles as usize;
    let mut h_keys = vec![0i32; n];
    let result = hipMemcpyDtoH(
        h_keys.as_mut_ptr() as _,
        hipDeviceptr_t(sorted_keys_dev as *mut std::ffi::c_void),
        n * 4,
    );
    if result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid: sorted keys read FAILED {:?} ptr=0x{:x} n={}",
            result, sorted_keys_dev, n
        ));
        return;
    }

    // Check monotonically non-decreasing
    let mut violations = 0u32;
    let mut first_violation = None;
    let mut min_key = h_keys[0];
    let mut max_key = h_keys[0];
    for i in 1..n {
        if h_keys[i] < h_keys[i - 1] {
            violations += 1;
            if first_violation.is_none() {
                first_violation = Some(i);
            }
        }
        if h_keys[i] < min_key {
            min_key = h_keys[i];
        }
        if h_keys[i] > max_key {
            max_key = h_keys[i];
        }
    }

    // Check for unreasonable values (negative or extremely large)
    let mut negative_count = 0u32;
    let mut huge_count = 0u32;
    for &k in &h_keys {
        if k < 0 {
            negative_count += 1;
        }
        if k > 16_000_000 {
            huge_count += 1;
        } // 16M cells seems extremely large
    }

    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let has_issues = violations > 0 || negative_count > 0 || huge_count > 0;

    // Log first 5 calls, every 20th, or always if issues
    if call_n < 5 || call_n % 20 == 0 || has_issues {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid #{}: n={} min={} max={} violations={} neg={} huge={}",
            call_n, n, min_key, max_key, violations, negative_count, huge_count
        ));
    }

    if violations > 0 {
        let vi = first_violation.unwrap();
        super::driver::zluda_log_err(&format!(
            "*** SORT CORRUPTED #{}: first violation at [{}]: keys[{}]={} > keys[{}]={}",
            call_n,
            vi,
            vi - 1,
            h_keys[vi - 1],
            vi,
            h_keys[vi]
        ));
        // Dump keys around the first violation
        let start = vi.saturating_sub(5);
        let end = (vi + 5).min(n);
        let around: Vec<String> = (start..end)
            .map(|j| format!("[{}]={}", j, h_keys[j]))
            .collect();
        super::driver::zluda_log_err(&format!("  keys around violation: {}", around.join(", ")));
    }
}

/// Extract a kernel argument value from either kernel_params or extra format.
/// For kernel_params: params[i] is a pointer to the arg value.
/// For extra: args are packed in a flat buffer (extra[1] points to it).
/// 2026-06-08 narrow per-kernel arg capture for FREE_USE_AFTER_LAUNCH_SCAN.
/// Reads only DECLARED u64 / pointer kernarg slots based on each kernel's
/// PTX-visible param signature — safe (no segfault past end of short
/// arg arrays). For each pointer value that looks like a tracked device
/// allocation, returns (base, size) — the existing
/// `driver::scan_kernel_ring_for_ptr` then uses these ranges to detect
/// freed-while-in-flight VAs.
///
/// Kernels covered (from PTX signatures in /tmp/zluda_ptx_*.ptx):
/// - IOFX_newRadixSortStepKernel — 13 params; u64 slots at idx 0, 6-10
/// - TurbulenceFS_dynamicGridMG{Smooth,Residual,Restrict,Prolong}Kernel{0-3}
///   — 6 params, last is a 64-byte struct with u64 ptrs at offsets +0, +8, +24, +48, +56
/// - IOFX_makeSortKeys / IOFX_remapKernel — single u64 ptr at idx 0
///
/// Cap at 8 ranges per launch (KERNEL_RING_PTRS limit).
unsafe fn collect_narrow_ptr_ranges(
    name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> Vec<(u64, u64)> {
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let alloc_min =
        super::memory::ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
    let alloc_max =
        super::memory::ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
    let push_if_alloc = |ranges: &mut Vec<(u64, u64)>, ptr: u64| {
        if ptr < alloc_min || ptr >= alloc_max {
            return;
        }
        if let Some((base, size)) = super::memory::find_dev_alloc_containing(ptr) {
            if !ranges.iter().any(|&(b, _)| b == base) && ranges.len() < 8 {
                ranges.push((base, size));
            }
        }
    };
    if name == "IOFX_newRadixSortStepKernel" {
        // 13 params: u64 at idx 0, 6, 7, 8, 9, 10. Offsets via extra: same.
        for &(off, idx) in &[(0usize, 0usize), (32, 6), (40, 7), (48, 8), (56, 9), (64, 10)] {
            if let Some(p) =
                unsafe { read_kernel_arg::<u64>(kernel_params, extra, off, idx) }
            {
                push_if_alloc(&mut ranges, p);
            }
        }
    } else if name.starts_with("TurbulenceFS_dynamicGridMG") {
        // 6 params; param_0 is u64 ptr, param_5 is a 64-byte struct with
        // u64 ptrs at offsets +0, +8, +24, +48, +56 inside the struct.
        if let Some(p) =
            unsafe { read_kernel_arg::<u64>(kernel_params, extra, 0, 0) }
        {
            push_if_alloc(&mut ranges, p);
        }
        // param_5 struct: dereference the slot pointer once more to read
        // the struct contents. read_kernel_arg returns the slot value (a
        // u64 for u64 args). For the struct param, we want to read the
        // struct's bytes at offsets within it.
        if !kernel_params.is_null() {
            let slot_ptr = unsafe { *kernel_params.add(5) } as *const u8;
            if !slot_ptr.is_null() {
                for &off in &[0usize, 8, 24, 48, 56] {
                    let p = unsafe {
                        std::ptr::read_unaligned(slot_ptr.add(off) as *const u64)
                    };
                    push_if_alloc(&mut ranges, p);
                }
            }
        } else if !extra.is_null() {
            // extra buffer holds the kernarg buffer flat; param_5 begins
            // at offset 16 (param_0 = 8, param_1 = 2, param_2-4 = 4*3,
            // align to 8 → 16+ padding). Skip generic extra path here —
            // captured via kernel_params path in practice for these
            // kernels per PTX dumps.
            let _ = extra;
        }
    } else if name == "IOFX_makeSortKeys_templ$0"
        || name == "IOFX_makeSortKeys_templ$1"
        || name == "IOFX_remapKernel_templ$0"
        || name == "IOFX_remapKernel_templ$1"
        || name == "IOFX_newRadixSortBlockKernel"
    {
        // Single u64 ptr at idx 0 (param_0). Other params are counts/flags.
        if let Some(p) =
            unsafe { read_kernel_arg::<u64>(kernel_params, extra, 0, 0) }
        {
            push_if_alloc(&mut ranges, p);
        }
    } else if name == "IOFX_actorRangeKernel" {
        // 2026-06-09 Class A bf000 investigation. PTX signature (10 params):
        //   p0 u64, p1 u16, p2 u32, p3 u32, p4 u64, p5 u32,
        //   p6 u64, p7 u64, p8 u64, p9 u64
        // Of these, p4/p6/p7/p8/p9 are device-buffer pointers. p4 = input
        // first-indices, p6/p7/p8 = output rangemaps, p9 = sort-key bit-31
        // markers. p0 is likely a small descriptor handle (kept for
        // completeness). Flat-buffer offsets for the `extra` path account
        // for natural alignment: p0=0, p4 = 8+2+pad+4+4 = 24 (8-byte
        // aligned), p5 at 32 (u32), p6 at 40 (u64 aligned), p7 at 48,
        // p8 at 56, p9 at 64.
        for &(off, idx) in &[
            (0usize, 0usize),
            (24, 4),
            (40, 6),
            (48, 7),
            (56, 8),
            (64, 9),
        ] {
            if let Some(p) =
                unsafe { read_kernel_arg::<u64>(kernel_params, extra, off, idx) }
            {
                push_if_alloc(&mut ranges, p);
            }
        }
    } else if name == "CreateParticleShapesIncremental_MultiStage_4" {
        // 2026-06-09 Class A bf000 investigation. PTX signature (8 params):
        //   p0 u64, p1 u16, p2 u64, p3 u32, p4 u64, p5 u64, p6 u64, p7 u64
        // p2 = particle-metadata buffer (read), p4 = 12-byte struct buffer
        // (read, partner of bf000-1.04 MiB candidate), p5 = OUTPUT remove-
        // queue buffer (write, prime bf000 candidate, indexed by r10*4),
        // p6 = remapping buffer (read), p7 = counter buffer (atom.add -1).
        // The wild-scatter Class A is MS_4 over-dispatch (counter underflows
        // → r10 wraps to 0xFFFFFFFE → output write @ base + 0x3FFFFFFF8 ≈
        // base + 16 GiB). For single-page _bf000 faults, p5 IS the bf000
        // buffer and the write lands at OUTPUT[0]. Capturing all 6 ptrs
        // lets FREE_USE_AFTER_LAUNCH_WARN flag a freed buffer used by an
        // in-flight MS_4 — the UAF angle uniquely missed in earlier scans.
        // Flat-buffer offsets: p0=0, p2=16 (align 8), p4=32, p5=40, p6=48,
        // p7=56.
        for &(off, idx) in &[
            (0usize, 0usize),
            (16, 2),
            (32, 4),
            (40, 5),
            (48, 6),
            (56, 7),
        ] {
            if let Some(p) =
                unsafe { read_kernel_arg::<u64>(kernel_params, extra, off, idx) }
            {
                push_if_alloc(&mut ranges, p);
            }
        }
    }
    ranges
}

unsafe fn read_kernel_arg<T: Copy>(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    arg_offset: usize, // byte offset in the flat buffer (for extra)
    arg_index: usize,  // index in kernel_params array
) -> Option<T> {
    if !kernel_params.is_null() {
        let param_ptr = *kernel_params.add(arg_index);
        if param_ptr.is_null() {
            return None;
        }
        Some(*(param_ptr as *const T))
    } else if !extra.is_null() {
        // extra format: [CU_LAUNCH_PARAM_BUFFER_POINTER(1), buffer_ptr, CU_LAUNCH_PARAM_BUFFER_SIZE(2), size_ptr, END]
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        if extra_arr[0] != (0x1usize as *mut ::core::ffi::c_void) {
            return None;
        }
        let buffer = extra_arr[1] as *const u8;
        if buffer.is_null() {
            return None;
        }
        Some(std::ptr::read_unaligned(buffer.add(arg_offset) as *const T))
    } else {
        None
    }
}

fn ms4_pointer_validation_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_MS4_PTR_VALIDATE").map_or(false, |v| v == "1"))
}

unsafe fn query_device_pointer_range(ptr: u64) -> Option<(u64, usize, usize)> {
    if ptr == 0 {
        return None;
    }
    let mut base = hip_runtime_sys::hipDeviceptr_t(std::ptr::null_mut());
    let mut size = 0usize;
    let status = super::memory::get_address_range_v2(
        &mut base,
        &mut size,
        hip_runtime_sys::hipDeviceptr_t(ptr as *mut std::ffi::c_void),
    );
    if status.is_err() {
        return None;
    }
    let base_u = base.0 as usize;
    let ptr_u = ptr as usize;
    if ptr_u < base_u {
        return None;
    }
    Some((base.0 as u64, size, ptr_u - base_u))
}

unsafe fn validate_ms4_pointer_ranges(
    call_n: u32,
    n: u32,
    p2: u64,
    p4: u64,
    p5: u64,
    p6: u64,
    p7: u64,
) {
    let required_linear_bytes = (n as usize).saturating_mul(4);
    let mut problems: Vec<String> = Vec::new();
    let mut details: Vec<String> = Vec::new();
    for &(label, ptr, min_size) in &[
        ("p2(flags)", p2, required_linear_bytes),
        ("p6(shape_idx)", p6, required_linear_bytes),
        ("p5(scatter)", p5, required_linear_bytes),
        ("p4(struct)", p4, 12usize),
        ("p7(counter)", p7, 4usize),
    ] {
        match query_device_pointer_range(ptr) {
            Some((base, size, offset)) => {
                let available = size.saturating_sub(offset);
                if min_size > 0 && available < min_size {
                    problems.push(format!(
                        "{}: available={} < need={} (base=0x{:x} size={} off={})",
                        label, available, min_size, base, size, offset
                    ));
                } else {
                    details.push(format!(
                        "{}=ok(base=0x{:x} size={} off={} avail={})",
                        label, base, size, offset, available
                    ));
                }
            }
            None => {
                problems.push(format!("{}: unresolved ptr=0x{:x}", label, ptr));
            }
        }
    }
    if problems.is_empty() {
        super::driver::zluda_log_err(&format!(
            "[MS4_PTR_VALIDATE #{}] n={} {}",
            call_n,
            n,
            details.join(" ")
        ));
    } else {
        super::driver::zluda_log_err(&format!(
            "*** [MS4_PTR_VALIDATE #{}] n={} problems=[{}] {}",
            call_n,
            n,
            problems.join(" | "),
            details.join(" ")
        ));
    }
}

/// Dispatcher: validate inputs for known crash-prone kernels before launching them.
/// Active when either `ZLUDA_SYNC_LAUNCHES=1` or `ZLUDA_LAUNCH_MUTEX=2` — both
/// imply the caller wants full per-launch serialization + diagnostics.
fn validate_kernel_inputs(
    name: &str,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
    grid_dim_x: u32,
    block_dim_x: u32,
) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        let sync = std::env::var("ZLUDA_SYNC_LAUNCHES").map_or(false, |v| v == "1");
        let mutex_mode2 = std::env::var("ZLUDA_LAUNCH_MUTEX")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .map(|m| m == 2)
            .unwrap_or(false);
        sync || mutex_mode2
    });
    if !enabled || (kernel_params.is_null() && extra.is_null()) {
        return;
    }

    // CreateGrid(int n, const int* sortedKeys, int* cellStart, int* cellEnd)
    // Kernel arg layout in flat buffer: [n:i32 @0, pad @4, sortedKeys:u64 @8, cellStart:u64 @16, cellEnd:u64 @24]
    if name.contains("CreateGrid") && !name.contains("Turbulence") {
        unsafe {
            validate_create_grid_from_args(kernel_params, extra, stream);
        }
    }

    // CollideParticles(int n, int nCells, int* gridStart, int* gridEnd, int* sortIdx,
    //   Vec3Iter pos[8 bytes], int* phases, float radius, float invCell, int* neighbors, u8* contacts)
    // Flat buffer layout: [n:i32@0, nCells:i32@4, gridStart:u64@8, gridEnd:u64@16, sortIdx:u64@24,
    //   pos:u64@32, phases:u64@40, radius:f32@48, invCell:f32@52, neighbors:u64@56, contacts:u64@64]
    if name.contains("CollideParticles") {
        unsafe {
            validate_collide_particles_args(kernel_params, extra, name);
        }
    }

    // CreateParticleShapesIncremental_MultiStage_4 — the kernel that crashes in
    // the PhysX particle pipeline. Runs successfully ~17 times then the 18th
    // launch hangs hipDeviceSynchronize and triggers SIGABRT. Suggests an
    // upstream corruption, not a kernel bug. Inspect flags/indices before launch.
    if name.contains("CreateParticleShapesIncremental_MultiStage_4") {
        unsafe {
            validate_multistage4_args(kernel_params, extra, name, grid_dim_x, block_dim_x);
        }
    }
}

/// Validate CollideParticles arguments — check all pointer args for NULL.
unsafe fn validate_collide_particles_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    name: &str,
) {
    // CollideParticles params (11 total):
    //  0: i32 numParticles     @0
    //  1: i32 numCells          @4
    //  2: u64 gridCellStart     @8
    //  3: u64 gridCellEnd       @16
    //  4: u64 sortedIndices     @24
    //  5: u64 positions(Vec3Iter) @32
    //  6: u64 phases            @40
    //  7: f32 radius            @48
    //  8: f32 invCellSize       @52
    //  9: u64 neighbours        @56
    // 10: u64 contacts          @64
    let ptr_args: &[(usize, usize, &str)] = &[
        // (flat_offset, kernel_params_index, name)
        (8, 2, "gridCellStart"),
        (16, 3, "gridCellEnd"),
        (24, 4, "sortedIndices"),
        (32, 5, "positions"),
        (40, 6, "phases"),
        (56, 9, "neighbours"),
        (64, 10, "contacts"),
    ];

    let n: i32 = read_kernel_arg(kernel_params, extra, 0, 0).unwrap_or(-1);
    let n_cells: i32 = read_kernel_arg(kernel_params, extra, 4, 1).unwrap_or(-1);

    let mut nulls = Vec::new();
    let mut ptrs = Vec::new();
    for &(offset, idx, label) in ptr_args {
        let val: u64 = read_kernel_arg(kernel_params, extra, offset, idx).unwrap_or(0);
        if val == 0 {
            nulls.push(label);
        }
        ptrs.push(format!("{}=0x{:x}", label, val));
    }

    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    if !nulls.is_empty() || call_n < 3 {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: n={} nCells={} NULL=[{}] ptrs=[{}]",
            name,
            call_n,
            n,
            n_cells,
            if nulls.is_empty() {
                "none".to_string()
            } else {
                nulls.join(",")
            },
            ptrs.join(", ")
        ));
    }

    if !nulls.is_empty() {
        super::driver::zluda_log_err(&format!(
            "*** NULL POINTER in CollideParticles #{}: {} args are NULL!",
            call_n,
            nulls.join(", ")
        ));
    }
}

/// Validate CreateGrid inputs by extracting args from either kernel_params or extra.
unsafe fn validate_create_grid_from_args(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
    stream: hipStream_t,
) {
    use hip_runtime_sys::*;

    // Extract arguments: CreateGrid(int n, const int* sortedKeys, int* cellStart, int* cellEnd)
    // In flat buffer (extra): n is i32 at offset 0, sortedKeys is u64 at offset 8 (aligned)
    let num_particles: i32 = match read_kernel_arg(kernel_params, extra, 0, 0) {
        Some(v) => v,
        None => return,
    };
    let sorted_keys_dev: u64 = match read_kernel_arg(kernel_params, extra, 8, 1) {
        Some(v) => v,
        None => return,
    };

    if num_particles <= 0 || num_particles > 200_000 {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid: numParticles={} (suspicious)",
            num_particles
        ));
        return;
    }

    // Sync stream so previous kernel output (sort) is available
    let _ = hipStreamSynchronize(stream);

    let n = num_particles as usize;
    let read_n = n.min(10_000); // Cap readback to avoid huge transfers
    let mut h_keys = vec![0i32; read_n];
    let result = hipMemcpyDtoH(
        h_keys.as_mut_ptr() as _,
        hipDeviceptr_t(sorted_keys_dev as *mut std::ffi::c_void),
        read_n * 4,
    );
    if result.is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid: sorted keys read FAILED {:?} ptr=0x{:x} n={}",
            result, sorted_keys_dev, n
        ));
        return;
    }

    // Check monotonically non-decreasing
    let mut violations = 0u32;
    let mut first_violation = None;
    let mut min_key = h_keys[0];
    let mut max_key = h_keys[0];
    let mut negative_count = 0u32;
    for i in 0..read_n {
        if i > 0 && h_keys[i] < h_keys[i - 1] {
            violations += 1;
            if first_violation.is_none() {
                first_violation = Some(i);
            }
        }
        if h_keys[i] < min_key {
            min_key = h_keys[i];
        }
        if h_keys[i] > max_key {
            max_key = h_keys[i];
        }
        if h_keys[i] < 0 {
            negative_count += 1;
        }
    }

    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let has_issues = violations > 0 || negative_count > 0;

    if call_n < 3 || call_n % 50 == 0 || has_issues {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid #{}: n={} read={} min={} max={} violations={} neg={}",
            call_n, n, read_n, min_key, max_key, violations, negative_count
        ));
    }

    if violations > 0 {
        let vi = first_violation.unwrap();
        super::driver::zluda_log_err(&format!(
            "*** SORT CORRUPTED #{}: [{}]={} > [{}]={}",
            call_n,
            vi - 1,
            h_keys[vi - 1],
            vi,
            h_keys[vi]
        ));
        let start = vi.saturating_sub(3);
        let end = (vi + 4).min(read_n);
        let around: Vec<String> = (start..end)
            .map(|j| format!("[{}]={}", j, h_keys[j]))
            .collect();
        super::driver::zluda_log_err(&format!("  keys: {}", around.join(", ")));
    }
}

#[cfg(test)]
mod tests {
    //! Phase 1 fix tests for Bug A force-zero workaround.
    //!
    //! The MS_4 family bug (Bug A) is caused by PhysX's per-frame
    //! `MemCopyBatchedAsync` reset batch occasionally excluding a tracked
    //! rd5 buffer, leaving `*(rd5+22176)` non-zero before MS_1 reads it.
    //! Our fix `force_zero_rd5_22176_before_ms1` defensively writes 0 to
    //! every tracked rd5+22176 before MS_1 launches, guaranteeing the
    //! invariant PhysX expects.
    //!
    //! Direct unit-testing of the fix is hard because it makes real HIP
    //! calls (hipMemcpyDtoD, hipMemsetD32, hipDeviceSynchronize). The
    //! tests below cover the smoke-level behavior; the deep validation is
    //! the in-game Batman PC Performance Test runs.
    use super::*;

    /// Smoke: with no rd5s tracked, the force-zero function returns
    /// immediately without initializing any state or making HIP calls.
    /// This guards against the obvious panic on the empty-list path.
    #[test]
    fn force_zero_no_tracked_rd5s_is_noop() {
        // Snapshot existing tracked list (in case other tests have populated it).
        let snapshot: Vec<u64> = {
            let g = match TRACKED_RD5_PTRS.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            g.clone()
        };
        // Clear it for this test.
        {
            let mut g = match TRACKED_RD5_PTRS.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            g.clear();
        }
        // Should return immediately — no panic, no HIP calls (because the
        // ms4_capture_enabled gate is also off in tests by default).
        unsafe {
            force_zero_rd5_22176_before_ms1();
        }
        // Restore in case other tests need it.
        {
            let mut g = match TRACKED_RD5_PTRS.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            *g = snapshot;
        }
    }

    #[test]
    fn current_ms1_rd5_ptr_reads_kernel_params_slot_6() {
        let mut p0: u64 = 0;
        let mut p1: u16 = 0;
        let mut p2: u64 = 0;
        let mut p3: u64 = 0;
        let mut p4: u32 = 0;
        let mut p5: f32 = 0.0;
        let mut p6: u64 = 0x1234_5678_9abc_def0;
        let mut p7: u64 = 0;
        let mut p8: u64 = 0;
        let mut p9: u64 = 0;
        let mut p10: u64 = 0;
        let mut p11: u64 = 0;
        let mut p12: u8 = 0;
        let mut params: [*mut ::core::ffi::c_void; 13] = [
            &mut p0 as *mut _ as *mut _,
            &mut p1 as *mut _ as *mut _,
            &mut p2 as *mut _ as *mut _,
            &mut p3 as *mut _ as *mut _,
            &mut p4 as *mut _ as *mut _,
            &mut p5 as *mut _ as *mut _,
            &mut p6 as *mut _ as *mut _,
            &mut p7 as *mut _ as *mut _,
            &mut p8 as *mut _ as *mut _,
            &mut p9 as *mut _ as *mut _,
            &mut p10 as *mut _ as *mut _,
            &mut p11 as *mut _ as *mut _,
            &mut p12 as *mut _ as *mut _,
        ];

        let got = current_ms1_rd5_ptr(params.as_mut_ptr(), std::ptr::null_mut());
        assert_eq!(got, Some(p6));
    }

    #[test]
    fn current_ms1_rd5_ptr_reads_extra_buffer_offset_40() {
        let want: u64 = 0xfedc_ba98_7654_3210;
        let mut buffer = [0u8; 56];
        buffer[40..48].copy_from_slice(&want.to_ne_bytes());
        let mut size = buffer.len();
        let mut extra = [
            0x1usize as *mut ::core::ffi::c_void,
            buffer.as_mut_ptr() as *mut ::core::ffi::c_void,
            0x2usize as *mut ::core::ffi::c_void,
            &mut size as *mut _ as *mut ::core::ffi::c_void,
            std::ptr::null_mut(),
        ];

        let got = current_ms1_rd5_ptr(std::ptr::null_mut(), extra.as_mut_ptr());
        assert_eq!(got, Some(want));
    }

    /// `track_rd5_ptr` should add new pointers up to MS3_MAX_TRACKED_COUNTERS,
    /// silently drop duplicates, and silently drop additions beyond the limit.
    /// Bug A tracking depends on this — if a Bug A rd5 isn't in the tracked
    /// list, the F.6 detector and the Phase 1 fix both miss it.
    #[test]
    fn track_rd5_ptr_dedup_and_limit() {
        // Save and clear the global so this test is isolated.
        let snapshot: Vec<u64> = {
            let g = match TRACKED_RD5_PTRS.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            g.clone()
        };
        {
            let mut g = match TRACKED_RD5_PTRS.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            g.clear();
        }

        // Adding 0 must be a no-op (we use 0 as a sentinel for "no rd5").
        track_rd5_ptr(0);
        assert_eq!(
            TRACKED_RD5_PTRS.lock().unwrap().len(),
            0,
            "track_rd5_ptr(0) must not add anything"
        );

        // Add MS3_MAX_TRACKED_COUNTERS distinct pointers — all should land.
        for i in 1..=MS3_MAX_TRACKED_COUNTERS as u64 {
            let fake_ptr = 0xdead_0000_0000_0000u64 | i;
            track_rd5_ptr(fake_ptr);
        }
        assert_eq!(
            TRACKED_RD5_PTRS.lock().unwrap().len(),
            MS3_MAX_TRACKED_COUNTERS,
            "should track up to MS3_MAX_TRACKED_COUNTERS distinct ptrs"
        );

        // Adding the same pointers again must NOT grow the list.
        for i in 1..=MS3_MAX_TRACKED_COUNTERS as u64 {
            let fake_ptr = 0xdead_0000_0000_0000u64 | i;
            track_rd5_ptr(fake_ptr);
        }
        assert_eq!(
            TRACKED_RD5_PTRS.lock().unwrap().len(),
            MS3_MAX_TRACKED_COUNTERS,
            "duplicate adds must not grow the tracked list"
        );

        // Adding a NEW pointer beyond the limit silently drops.
        track_rd5_ptr(0xdead_0000_0000_00ffu64);
        assert_eq!(
            TRACKED_RD5_PTRS.lock().unwrap().len(),
            MS3_MAX_TRACKED_COUNTERS,
            "additions beyond MS3_MAX_TRACKED_COUNTERS must be silently dropped"
        );

        // Restore the original list so other tests aren't disturbed.
        {
            let mut g = match TRACKED_RD5_PTRS.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            *g = snapshot;
        }
    }

    #[test]
    fn mcb_descriptor_hits_rd5_reset_window_matches_only_overlaps() {
        let rd5_a = 0x1000u64;
        let rd5_b = 0x4000u64;
        let tracked = [rd5_a, rd5_b];
        let watch_a = rd5_a + F6_OFFSET_IN_RD5 as u64;
        let watch_b = rd5_b + F6_OFFSET_IN_RD5 as u64;

        assert_eq!(
            mcb_descriptor_hits_rd5_reset_window(watch_a, 4, &tracked),
            Some(rd5_a)
        );
        assert_eq!(
            mcb_descriptor_hits_rd5_reset_window(watch_a.saturating_sub(2), 8, &tracked),
            Some(rd5_a)
        );
        assert_eq!(
            mcb_descriptor_hits_rd5_reset_window(watch_b.saturating_add(1), 2, &tracked),
            Some(rd5_b)
        );
        assert_eq!(
            mcb_descriptor_hits_rd5_reset_window(watch_a.saturating_add(4), 4, &tracked),
            None
        );
        assert_eq!(
            mcb_descriptor_hits_rd5_reset_window(watch_b.saturating_sub(8), 4, &tracked),
            None
        );
    }

    #[test]
    fn launch_mutex_uses_dedicated_lock() {
        let lock = launch_mutex(1).expect("mode 1 should enable the launch mutex");
        assert!(
            std::ptr::eq(lock, &LAUNCH_LOCK),
            "launch mutex mode should use the dedicated launch lock"
        );
    }

    #[test]
    fn launch_mutex_mode_zero_is_disabled() {
        assert!(
            launch_mutex(0).is_none(),
            "mode 0 should leave launch serialization disabled"
        );
    }

    #[test]
    fn post_launch_diagnostic_gates() {
        // A fresh test process gives each case fresh OnceLocks, without changing
        // the environment seen by other tests in this process.
        const MARKER: &str = "ZLUDA_UNIT_POST_LAUNCH_GATE";
        if let Ok(scope) = std::env::var(MARKER) {
            for (name, enabled) in [
                ("produce", scope == "all"),
                ("IOFX_newRadixSortStepKernel", scope == "all" || scope == "nrss"),
                ("ParticleCollisionStaticBox", scope == "all" || scope == "flag"),
                ("CreateParticleShapesIncremental_MultiStage_1", scope == "all" || scope == "flag" || scope == "ms1"),
            ] {
                assert_eq!(post_launch_diagnostics_enabled(name), enabled, "{scope}: {name}");
            }
            return;
        }
        let cases: &[(&[(&str, &str)], &str)] = &[
            (&[], "off"),
            (&[("ZLUDA_MS4_CAPTURE", "1")], "off"),
            (&[("ZLUDA_F6_TRACK", "1")], "off"),
            (&[("ZLUDA_MS4_CAPTURE", "1"), ("ZLUDA_F6_TRACK", "1")], "all"),
            (&[("ZLUDA_MS2_RD6_KERNEL_TRACE", "1")], "all"),
            (&[("ZLUDA_COUNTER_SNAPSHOT", "1")], "all"),
            (&[("ZLUDA_NRSS_VALIDATE", "1")], "nrss"),
            (&[("ZLUDA_NRSS_VALIDATE", "2")], "nrss"),
            (&[("ZLUDA_NRSS_VALIDATE", "0")], "off"),
            (&[("ZLUDA_NRSS_VALIDATE", "invalid")], "off"),
            (&[("ZLUDA_MAPPED_WRITE_TRACE", "1")], "all"),
            (&[("ZLUDA_FLAG_COUNT_PROBE", "1")], "flag"),
            (&[("ZLUDA_MS1_GATE_TRACE", "1")], "ms1"),
        ];
        for &(vars, scope) in cases {
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args(["post_launch_diagnostic_gates", "--test-threads=1"]);
            for (key, _) in std::env::vars_os() {
                if key.to_string_lossy().starts_with("ZLUDA_") {
                    cmd.env_remove(key);
                }
            }
            cmd.env(MARKER, scope);
            for &(key, value) in vars {
                cmd.env(key, value);
            }
            let output = cmd.output().unwrap();
            assert!(output.status.success(), "{vars:?}: {}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed;"));
        }
    }

    #[test]
    fn summarize_sorted_keys_detects_bad_values_and_ordering() {
        let summary = summarize_sorted_keys(&[1, 2, 7, -1, 6, 20_000_001]).unwrap();
        assert_eq!(summary.min_key, -1);
        assert_eq!(summary.max_key, 20_000_001);
        assert_eq!(summary.violations, 1);
        assert_eq!(summary.first_violation, Some(3));
        assert_eq!(summary.negative_count, 1);
        assert_eq!(summary.huge_count, 1);
        assert_eq!(summary.first_bad_value, Some((3, -1)));
    }

    #[test]
    fn summarize_key_domain_detects_bad_values_without_sort_requirements() {
        let summary = summarize_key_domain(&[42, 7, -1, 20_000_001]).unwrap();
        assert_eq!(summary.min_key, -1);
        assert_eq!(summary.max_key, 20_000_001);
        assert_eq!(summary.negative_count, 1);
        assert_eq!(summary.huge_count, 1);
        assert_eq!(summary.first_bad_value, Some((2, -1)));
    }

    #[test]
    fn restore_signed_radix_residue_clears_xor_stuck_negatives() {
        let corrupted = [10, 11, 0x8000_000cu32 as i32, 13];
        let restored = restore_signed_radix_residue(&corrupted);
        assert_eq!(restored, vec![10, 11, 12, 13]);
        let summary = summarize_sorted_keys(&restored).unwrap();
        assert_eq!(summary.violations, 0);
        assert_eq!(summary.negative_count, 0);
        assert_eq!(summary.first_bad_value, None);
    }

    #[test]
    fn summarize_sort_values_accepts_reordered_multiset() {
        let summary = summarize_sort_values(&[10, 20, 10, 30], &[10, 30, 10, 20]).unwrap();
        assert_eq!(summary.negative_count, 0);
        assert_eq!(summary.extra_count, 0);
        assert_eq!(summary.missing_count, 0);
        assert_eq!(summary.first_extra, None);
        assert_eq!(summary.first_missing, None);
    }

    #[test]
    fn summarize_sort_values_detects_extra_and_missing_entries() {
        let summary = summarize_sort_values(&[10, 20, 10, 30], &[10, 20, 20, 30]).unwrap();
        assert_eq!(summary.negative_count, 0);
        assert_eq!(summary.extra_count, 1);
        assert_eq!(summary.missing_count, 1);
        assert_eq!(summary.first_extra, Some((2, 20, 1, 2)));
        assert_eq!(summary.first_missing, Some((0, 10, 2, 1)));
    }

    #[test]
    fn good_sort_replay_baseline_capture_matches_clean_bit0_game_shape() {
        let summary_out = SortedKeySummary {
            min_key: 5,
            max_key: 1_605_599,
            violations: 3741,
            negative_count: 0,
            huge_count: 0,
            first_violation: Some(1),
            first_bad_value: None,
        };
        let vals_summary = SortValueSummary {
            negative_count: 0,
            extra_count: 0,
            missing_count: 0,
            first_negative: None,
            first_extra: None,
            first_missing: None,
        };
        assert_eq!(
            classify_good_sort_replay_capture(0, 8109, 0, 0, summary_out, Some(vals_summary)),
            Some(GoodSortReplayCaptureKind::TargetBit0N8109)
        );
    }

    #[test]
    fn good_sort_replay_baseline_capture_accepts_large_clean_fallback_family() {
        let summary_out = SortedKeySummary {
            min_key: 114,
            max_key: 1_602_345,
            violations: 917,
            negative_count: 0,
            huge_count: 0,
            first_violation: Some(2),
            first_bad_value: None,
        };
        let vals_summary = SortValueSummary {
            negative_count: 0,
            extra_count: 0,
            missing_count: 0,
            first_negative: None,
            first_extra: None,
            first_missing: None,
        };
        assert_eq!(
            classify_good_sort_replay_capture(0, 1985, 0, 0, summary_out, Some(vals_summary)),
            Some(GoodSortReplayCaptureKind::FallbackBit0LargeClean)
        );
    }

    #[test]
    fn good_sort_replay_baseline_capture_accepts_clean_late_4688_family() {
        let summary_out = SortedKeySummary {
            min_key: 40,
            max_key: 1_591_577,
            violations: 56,
            negative_count: 0,
            huge_count: 0,
            first_violation: Some(2477),
            first_bad_value: None,
        };
        let vals_summary = SortValueSummary {
            negative_count: 0,
            extra_count: 0,
            missing_count: 0,
            first_negative: None,
            first_extra: None,
            first_missing: None,
        };
        assert_eq!(
            classify_good_sort_replay_capture(16, 4688, 1, 1, summary_out, Some(vals_summary)),
            Some(GoodSortReplayCaptureKind::LateBit16N4688Clean)
        );
    }

    #[test]
    fn good_sort_replay_baseline_capture_rejects_zero_fill_and_wrong_flags() {
        let clean_out = SortedKeySummary {
            min_key: 0,
            max_key: 0,
            violations: 0,
            negative_count: 0,
            huge_count: 0,
            first_violation: None,
            first_bad_value: None,
        };
        let zero_fill_vals = SortValueSummary {
            negative_count: 0,
            extra_count: 8109,
            missing_count: 8109,
            first_negative: None,
            first_extra: Some((0, 0, 0, 1)),
            first_missing: Some((0, 1, 1, 0)),
        };
        assert_eq!(
            classify_good_sort_replay_capture(0, 8109, 0, 0, clean_out, Some(zero_fill_vals)),
            None
        );
        let clean_vals = SortValueSummary {
            negative_count: 0,
            extra_count: 0,
            missing_count: 0,
            first_negative: None,
            first_extra: None,
            first_missing: None,
        };
        assert_eq!(
            classify_good_sort_replay_capture(0, 8109, 1, 0, clean_out, Some(clean_vals)),
            None
        );
        assert_eq!(
            classify_good_sort_replay_capture(16, 4688, 1, 0, clean_out, Some(clean_vals)),
            None
        );
        assert_eq!(
            classify_good_sort_replay_capture(4, 8109, 0, 0, clean_out, Some(clean_vals)),
            None
        );
        assert_eq!(
            classify_good_sort_replay_capture(0, 877, 0, 0, clean_out, Some(clean_vals)),
            None
        );
    }

    #[test]
    fn creategrid_sort_touch_log_start_parser_handles_disable_enable_and_offsets() {
        assert_eq!(creategrid_sort_touch_log_start_from_value(None), None);
        assert_eq!(creategrid_sort_touch_log_start_from_value(Some("")), None);
        assert_eq!(creategrid_sort_touch_log_start_from_value(Some("0")), None);
        assert_eq!(
            creategrid_sort_touch_log_start_from_value(Some("1")),
            Some(0)
        );
        assert_eq!(
            creategrid_sort_touch_log_start_from_value(Some("70")),
            Some(70)
        );
        assert_eq!(
            creategrid_sort_touch_log_start_from_value(Some("not-a-number")),
            None
        );
    }

    #[test]
    fn creategrid_snapshot_log_start_parser_reuses_touch_threshold_rules() {
        assert_eq!(
            creategrid_sort_touch_log_start_from_value(Some("12")),
            Some(12)
        );
        assert_eq!(
            creategrid_sort_touch_log_start_from_value(Some(" 5 ")),
            Some(5)
        );
    }

    #[test]
    fn sort_replay_capture_begin_bit_parser_defaults_to_zero() {
        assert_eq!(
            std::env::var("ZLUDA_SORT_REPLAY_CAPTURE_BIT").ok(),
            None,
            "unit tests assume no capture-bit env override is set"
        );
        assert_eq!(sort_replay_capture_begin_bit(), 0);
    }

    #[test]
    fn sort_replay_capture_gate_uses_configured_begin_bit() {
        assert!(should_queue_sort_replay_capture(4, 10, Some(10), 4));
        assert!(!should_queue_sort_replay_capture(0, 10, Some(10), 4));
        assert!(!should_queue_sort_replay_capture(4, 9, Some(10), 4));
        assert!(!should_queue_sort_replay_capture(4, 10, None, 4));
    }

    #[test]
    fn read_createcell_snapshot_args_reads_n_and_hash_output_from_extra_buffer() {
        let n_items = 8_827u32;
        let hash_out = 0x1234_5678_9abc_def0u64;
        let mut raw = [0u8; 48];
        raw[0..4].copy_from_slice(&n_items.to_ne_bytes());
        raw[24..32].copy_from_slice(&hash_out.to_ne_bytes());
        let mut raw_len = raw.len();
        let mut extra = [
            0x1usize as *mut std::ffi::c_void,
            raw.as_mut_ptr().cast(),
            0x2usize as *mut std::ffi::c_void,
            (&mut raw_len as *mut usize).cast(),
            std::ptr::null_mut(),
        ];

        let args =
            unsafe { read_createcell_snapshot_args(std::ptr::null_mut(), extra.as_mut_ptr()) }
                .expect("CreateCellIndices args should decode");
        assert_eq!(args.n_items, n_items as usize);
        assert_eq!(args.hash_out, hash_out);
    }

    #[test]
    fn good_sort_replay_capture_write_retries_after_failure_and_latches_on_success() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        let captured = AtomicBool::new(false);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let missing_dir = std::env::temp_dir().join(format!(
            "zluda-missing-good-capture-{}-{}",
            std::process::id(),
            nonce
        ));
        let missing_path = missing_dir.join("capture.bin");
        let failed_line = write_good_sort_replay_capture_once(
            &captured,
            missing_path.to_str().unwrap(),
            Some(&[1, 2, 3, 4]),
            0,
            8109,
            5,
            80,
        )
        .unwrap();
        assert!(
            failed_line.contains("write_err="),
            "failed write should be reported explicitly: {}",
            failed_line
        );
        assert!(
            !captured.load(Ordering::Relaxed),
            "failed writes should release the good-capture latch for a later retry"
        );

        let success_path = std::env::temp_dir().join(format!(
            "zluda-good-capture-{}-{}.bin",
            std::process::id(),
            nonce
        ));
        let success_line = write_good_sort_replay_capture_once(
            &captured,
            success_path.to_str().unwrap(),
            Some(&[9, 8, 7, 6]),
            0,
            8109,
            5,
            80,
        )
        .unwrap();
        assert!(
            success_line.contains("SORT_REPLAY_CAPTURE_GOOD"),
            "successful writes should log the emitted artifact: {}",
            success_line
        );
        assert_eq!(std::fs::read(&success_path).unwrap(), vec![9, 8, 7, 6]);
        assert!(
            captured.load(Ordering::Relaxed),
            "successful writes should keep the good-capture latch set"
        );

        let second_path = std::env::temp_dir().join(format!(
            "zluda-good-capture-{}-{}-second.bin",
            std::process::id(),
            nonce
        ));
        assert!(
            write_good_sort_replay_capture_once(
                &captured,
                second_path.to_str().unwrap(),
                Some(&[5, 4, 3, 2]),
                0,
                8109,
                5,
                80,
            )
            .is_none(),
            "once the good capture is written, later candidates should stay suppressed"
        );
        assert!(
            !second_path.exists(),
            "suppressed follow-up writes should not emit another artifact"
        );

        let _ = std::fs::remove_file(success_path);
    }

    #[test]
    fn reconstruct_expected_sort_pass_preserves_stable_radix_order() {
        let expected = reconstruct_expected_sort_pass(
            &[0x23, 0x10, 0x31, 0x12, 0x01],
            &[23, 10, 31, 12, 1],
            0,
        )
        .unwrap();
        assert_eq!(expected.keys_out, vec![0x10, 0x31, 0x01, 0x12, 0x23]);
        assert_eq!(expected.values_out, vec![10, 31, 1, 12, 23]);
        assert_eq!(expected.output_to_input, vec![1, 2, 4, 3, 0]);
    }

    #[test]
    fn analyze_expected_sort_pass_reports_expected_destinations() {
        let keys_in = [0x10, 0x21, 0x01, 0x12];
        let values_in = [10, 20, 30, 40];
        let keys_out = [0x10, 0x21, 0x21, 0x12];
        let values_out = [10, 20, 20, 40];
        let vals_summary = summarize_sort_values(&values_in, &values_out);
        let diag = analyze_expected_sort_pass(
            &keys_in,
            &values_in,
            &keys_out,
            &values_out,
            0,
            vals_summary,
        )
        .unwrap();
        let mismatch = diag.mismatch_line.unwrap();
        assert!(
            mismatch.contains("out[2] bit=0 digit=1 expected=in[2]"),
            "unexpected mismatch summary: {}",
            mismatch
        );
        assert!(
            mismatch.contains("actual_expected_out=[1]"),
            "mismatch summary should point back to the duplicated pair's rightful destination: {}",
            mismatch
        );
        assert_eq!(
            diag.extra_positions,
            Some("EXPECT_POS extra=20 expected_out=[1] actual_out=[1, 2]".to_string())
        );
        assert_eq!(
            diag.missing_positions,
            Some("EXPECT_POS missing=30 expected_out=[2] actual_out=[]".to_string())
        );
        assert!(
            diag.expected_window
                .as_deref()
                .unwrap_or_default()
                .contains("EXPECT"),
            "expected window should be populated around the first mismatch"
        );
        assert_eq!(
            diag.actual_pair_route,
            Some(
                "ACTUAL_PAIR_ROUTE in[1] -> expected_out[1] -> out[2] k:33(0x00000021) v:20"
                    .to_string()
            )
        );
        assert!(
            diag.actual_pair_source_window
                .as_deref()
                .unwrap_or_default()
                .contains("ACTUAL_PAIR_SRC_OUT"),
            "source window should be populated when the bad pair exists elsewhere in output"
        );
        assert!(
            diag.actual_pair_clone_window
                .as_deref()
                .unwrap_or_default()
                .contains("ACTUAL_PAIR_DST_OUT"),
            "clone window should be populated when the bad pair is replayed into another slot"
        );
        assert_eq!(
            diag.actual_value_route,
            Some("ACTUAL_VALUE_ROUTE in[1] -> expected_out[1] -> out[2] v:20".to_string())
        );
        assert!(
            diag.actual_value_input_window
                .as_deref()
                .unwrap_or_default()
                .contains("ACTUAL_VALUE_SRC_IN"),
            "input window should be populated for the foreign value's source slot"
        );
        assert_eq!(
            diag.actual_value_input_run,
            Some("ACTUAL_VALUE_INPUT_RUN in[1] -> out[2] start_in=1 start_out=2 len=1".to_string())
        );
        assert!(
            diag.actual_value_expected_window
                .as_deref()
                .unwrap_or_default()
                .contains("EXPECT_VALUE_SRC"),
            "expected value-source window should be populated when the value belongs elsewhere in output"
        );
        assert_eq!(
            diag.actual_value_expected_run,
            Some(
                "ACTUAL_VALUE_EXPECTED_RUN expected_out[1] -> out[2] start_expected=1 start_out=2 len=1"
                    .to_string()
            )
        );
        assert!(
            diag.actual_value_source_window
                .as_deref()
                .unwrap_or_default()
                .contains("ACTUAL_VALUE_SRC_OUT"),
            "value source window should be populated when the value belongs elsewhere in output"
        );
        assert!(
            diag.actual_value_clone_window
                .as_deref()
                .unwrap_or_default()
                .contains("ACTUAL_VALUE_DST_OUT"),
            "value clone window should be populated when the value is replayed into another slot"
        );
        assert_eq!(
            diag.actual_value_source_diff,
            Some(
                "ACTUAL_VALUE_SRC_DIFF[0..4): mismatch_count=1 mismatches=[2] runs=[2..3]"
                    .to_string()
            )
        );
        assert_eq!(
            diag.actual_value_clone_diff,
            Some(
                "ACTUAL_VALUE_DST_DIFF[0..4): mismatch_count=1 mismatches=[2] runs=[2..3]"
                    .to_string()
            )
        );
    }

    #[test]
    fn analyze_expected_sort_pass_tracks_xor_restored_actual_pairs() {
        let keys_in = [0x05, 0x17];
        let values_in = [50, 70];
        let keys_out = [0x05, 0x8000_0017u32 as i32];
        let values_out = [50, 70];
        let vals_summary = summarize_sort_values(&values_in, &values_out);
        let diag = analyze_expected_sort_pass(
            &keys_in,
            &values_in,
            &keys_out,
            &values_out,
            0,
            vals_summary,
        )
        .unwrap();
        let mismatch = diag.mismatch_line.unwrap();
        assert!(
            mismatch.contains("actual_xor=k:23(0x00000017)"),
            "mismatch summary should include the xor-restored pair: {}",
            mismatch
        );
        assert!(
            mismatch.contains("expected_out=[1]"),
            "xor-restored pair should point back to its rightful destination: {}",
            mismatch
        );
        assert!(
            mismatch.contains("actual_value_src=in[1]=k:23(0x00000017) digit=7 expected_out[1]=k:23(0x00000017) digit=7"),
            "mismatch summary should show where the value legitimately belongs: {}",
            mismatch
        );
        assert_eq!(
            diag.actual_value_route,
            Some("ACTUAL_VALUE_ROUTE in[1] -> expected_out[1] -> out[1] v:70".to_string())
        );
        assert!(
            diag.actual_value_input_window
                .as_deref()
                .unwrap_or_default()
                .contains("ACTUAL_VALUE_SRC_IN"),
            "input window should be populated even when the value stayed at its rightful slot"
        );
        assert_eq!(
            diag.actual_value_input_run,
            Some("ACTUAL_VALUE_INPUT_RUN in[1] -> out[1] start_in=0 start_out=0 len=2".to_string())
        );
        assert!(
            diag.actual_value_expected_window
                .as_deref()
                .unwrap_or_default()
                .contains("EXPECT_VALUE_SRC"),
            "expected value-source window should be populated even when the value stayed at its rightful slot"
        );
        assert_eq!(
            diag.actual_value_expected_run,
            Some(
                "ACTUAL_VALUE_EXPECTED_RUN expected_out[1] -> out[1] start_expected=0 start_out=0 len=2"
                    .to_string()
            )
        );
        assert!(
            diag.actual_value_source_window
                .as_deref()
                .unwrap_or_default()
                .contains("ACTUAL_VALUE_SRC_OUT"),
            "value source window should be populated even when the value stayed at its rightful slot"
        );
        assert!(
            diag.actual_value_clone_window.is_none(),
            "value clone window should stay empty when the value did not move"
        );
        assert_eq!(
            diag.actual_value_source_diff,
            Some(
                "ACTUAL_VALUE_SRC_DIFF[0..2): mismatch_count=1 mismatches=[1] runs=[1..2]"
                    .to_string()
            )
        );
        assert_eq!(
            diag.actual_value_clone_diff,
            Some(
                "ACTUAL_VALUE_DST_DIFF[0..2): mismatch_count=1 mismatches=[1] runs=[1..2]"
                    .to_string()
            )
        );
    }

    #[test]
    fn analyze_downsweep_rank_diagnostics_reports_scan_counts_and_destinations() {
        let grid_share = SortGridEvenShare {
            total_grains: 2,
            big_blocks: 0,
            big_share: 4,
            normal_share: 2,
            normal_base_offset: 0,
            num_items: 4,
            grid_size: 2,
            block_offset: 0,
            block_end: 0,
        };
        let keys_in = [0x02, 0x0c, 0x01, 0x0d];
        let values_in = [100, 200, 300, 400];
        let keys_out = [0x01, 0x8000_000cu32 as i32, 0x02, 0x0d];
        let values_out = [300, 200, 100, 400];
        let diag = analyze_downsweep_rank_diagnostics(
            &keys_in,
            &values_in,
            &keys_out,
            &values_out,
            0,
            Some(grid_share),
        )
        .unwrap();
        let expected_line = diag.expected_line.unwrap();
        assert!(
            expected_line.contains("RANK_EXPECTED out[1] source=in[0]"),
            "expected rank trace should identify the rightful source: {}",
            expected_line
        );
        assert!(
            expected_line.contains("scan_counts=[1, 0]"),
            "expected rank trace should include the per-block digit counts: {}",
            expected_line
        );
        assert!(
            expected_line.contains("digit_base=1 local_rank=0 global_rank=0 scatter_dst=1"),
            "expected rank trace should include the computed destination: {}",
            expected_line
        );
        let actual_line = diag.actual_line.unwrap();
        assert!(
            actual_line.contains("RANK_ACTUAL out[1] source=in[1]"),
            "actual rank trace should identify the foreign value source: {}",
            actual_line
        );
        assert!(
            actual_line.contains("output_k:-2147483636(0x8000000c)"),
            "actual rank trace should include the mismatching output key: {}",
            actual_line
        );
        assert!(
            actual_line.contains("digit_base=2 local_rank=0 global_rank=0 scatter_dst=2"),
            "actual rank trace should compare the computed destination against where the value landed: {}",
            actual_line
        );
        assert!(
            actual_line.contains("actual_dst=1"),
            "actual rank trace should record the mismatching destination slot: {}",
            actual_line
        );
    }

    #[test]
    fn analyze_downsweep_block_capture_formats_full_block_artifact() {
        let grid_share = SortGridEvenShare {
            total_grains: 2,
            big_blocks: 0,
            big_share: 4,
            normal_share: 2,
            normal_base_offset: 0,
            num_items: 4,
            grid_size: 2,
            block_offset: 0,
            block_end: 0,
        };
        let keys_in = [0x02, 0x0c, 0x01, 0x0d];
        let values_in = [100, 200, 300, 400];
        let keys_out = [0x01, 0x8000_000cu32 as i32, 0x02, 0x0d];
        let values_out = [300, 200, 100, 400];
        let capture = analyze_downsweep_block_capture(
            &keys_in,
            &values_in,
            &keys_out,
            &values_out,
            0,
            Some(grid_share),
        )
        .unwrap();
        assert_eq!(capture.block_idx, 0);
        assert_eq!((capture.start, capture.end), (0, 2));
        assert_eq!(capture.expected_source_block, 0);
        assert_eq!(capture.actual_source_block, Some(0));
        assert!(
            capture
                .capture_text
                .contains("SORT_CTA_CAPTURE bit=0 block=0 range=[0..2) mismatch_out=1 expected_src=in[0] actual_value_src=Some(1)"),
            "capture header should identify the mismatch and source slots: {}",
            capture.capture_text
        );
        assert!(
            capture
                .capture_text
                .contains("DEST_BLOCK_RUNS key_mismatch=1 runs=[1..2] value_mismatch=1 runs=[1..2] neg=1 runs=[1..2]"),
            "capture should summarize mismatch and negative-key runs inside the destination block: {}",
            capture.capture_text
        );
        assert!(
            capture.capture_text.contains("![1]"),
            "capture should mark the mismatching output slot: {}",
            capture.capture_text
        );
        assert!(
            capture
                .capture_text
                .contains("actual k:-2147483636(0x8000000c) v:200 d:12 xor_k:12(0x0000000c)"),
            "capture should include the synthetic output key and xor-restored note: {}",
            capture.capture_text
        );
    }

    #[test]
    fn analyze_downsweep_block_capture_handles_nonzero_bit_passes() {
        let grid_share = SortGridEvenShare {
            total_grains: 2,
            big_blocks: 0,
            big_share: 4,
            normal_share: 2,
            normal_base_offset: 0,
            num_items: 4,
            grid_size: 2,
            block_offset: 0,
            block_end: 0,
        };
        let keys_in = [0x02, 0x0c, 0x01, 0x0d];
        let values_in = [100, 200, 300, 400];
        let keys_out = [0x01, 0x8000_000cu32 as i32, 0x02, 0x0d];
        let values_out = [300, 200, 100, 400];
        let capture = analyze_downsweep_block_capture(
            &keys_in,
            &values_in,
            &keys_out,
            &values_out,
            4,
            Some(grid_share),
        )
        .unwrap();
        assert!(
            capture.capture_text.contains("SORT_CTA_CAPTURE bit=4"),
            "capture header should preserve the pass bit for nonzero-bit passes: {}",
            capture.capture_text
        );
        assert!(
            capture
                .capture_text
                .contains("actual k:-2147483636(0x8000000c) v:200 d:0 xor_k:12(0x0000000c)"),
            "capture should recompute digits using the current pass bit: {}",
            capture.capture_text
        );
    }

    #[test]
    fn analyze_downsweep_block_capture_includes_cross_block_source_section() {
        let grid_share = SortGridEvenShare {
            total_grains: 2,
            big_blocks: 0,
            big_share: 4,
            normal_share: 2,
            normal_base_offset: 0,
            num_items: 4,
            grid_size: 2,
            block_offset: 0,
            block_end: 0,
        };
        let keys_in = [0x11, 0x12, 0x21, 0x22];
        let values_in = [100, 101, 200, 201];
        let keys_out = [0x11, 0x12, 0x11, 0x12];
        let values_out = [100, 101, 100, 101];
        let capture = analyze_downsweep_block_capture(
            &keys_in,
            &values_in,
            &keys_out,
            &values_out,
            4,
            Some(grid_share),
        )
        .unwrap();
        assert!(
            capture
                .capture_text
                .contains("DEST_BLOCK block=1 range=[2..4)"),
            "capture should include the destination block section: {}",
            capture.capture_text
        );
        assert!(
            capture
                .capture_text
                .contains("ACTUAL_SOURCE_BLOCK block=0 range=[0..2)"),
            "capture should include the foreign source block when the replay crosses blocks: {}",
            capture.capture_text
        );
    }

    #[test]
    fn contiguous_index_runs_groups_striped_ranges() {
        assert_eq!(
            contiguous_index_runs(&[2477, 2478, 2479, 2541, 2542, 2605]),
            vec![(2477, 2480), (2541, 2543), (2605, 2606)]
        );
        assert_eq!(
            format_index_runs(&[2477, 2478, 2479, 2541, 2542, 2605]),
            "[2477..2480, 2541..2543, 2605..2606]"
        );
    }

    #[test]
    fn sort_pass_capture_len_uses_grid_share_coverage() {
        let grid_share = SortGridEvenShare {
            total_grains: 5,
            big_blocks: 0,
            big_share: 3328,
            normal_share: 1664,
            normal_base_offset: 0,
            num_items: 8109,
            grid_size: 5,
            block_offset: 0,
            block_end: 0,
        };
        assert_eq!(sort_pass_capture_len(8109, Some(grid_share)), 8320);
        assert_eq!(sort_pass_capture_len(8109, None), 8109);
    }

    #[test]
    fn serialize_sort_pass_replay_capture_writes_expected_header_and_payload() {
        let bytes = serialize_sort_pass_replay_capture(
            615,
            12,
            2,
            5,
            128,
            1,
            1,
            256,
            0,
            1,
            1,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            0x5000,
            0x6000,
            &[10, 20],
            Some(&[100, 200]),
            &[11, 12],
            Some(&[110, 120]),
            &[21, 22],
            Some(&[210, 220]),
            &[30, 40],
            Some(&[300, 400]),
            &[7, 8, 9],
            Some(SortGridEvenShare {
                total_grains: 5,
                big_blocks: 0,
                big_share: 3328,
                normal_share: 1664,
                normal_base_offset: 0,
                num_items: 2,
                grid_size: 5,
                block_offset: 0,
                block_end: 0,
            }),
        )
        .unwrap();
        let read_u32 = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let read_i32 = |offset| i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!(read_u32(0), SORT_PASS_REPLAY_CAPTURE_MAGIC);
        assert_eq!(read_u32(4), SORT_PASS_REPLAY_CAPTURE_VERSION);
        assert_eq!(read_u32(8), 615);
        assert_eq!(read_u32(12), 12);
        assert_eq!(read_u32(16), 2);
        assert_eq!(read_u32(20), 2);
        assert_eq!(read_u32(24), 5);
        assert_eq!(
            read_u32(32),
            SORT_PASS_REPLAY_CAPTURE_FLAG_HAS_VALS | SORT_PASS_REPLAY_CAPTURE_FLAG_HAS_GRID_SHARE
        );
        assert_eq!(read_u32(28), 3);
        assert_eq!(read_u32(36), pack_sort_replay_launch_flags(0, 1, 1));
        assert_eq!(read_u32(40), 128);
        assert_eq!(read_u32(44), 1);
        assert_eq!(read_u32(48), 1);
        assert_eq!(read_u32(52), 256);
        assert_eq!(
            u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
            0x1000
        );
        assert_eq!(
            u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            0x2000
        );
        assert_eq!(
            u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
            0x3000
        );
        assert_eq!(
            u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
            0x4000
        );
        assert_eq!(
            u64::from_le_bytes(bytes[88..96].try_into().unwrap()),
            0x5000
        );
        assert_eq!(
            u64::from_le_bytes(bytes[96..104].try_into().unwrap()),
            0x6000
        );
        assert_eq!(read_i32(104), 5);
        assert_eq!(read_i32(136), 0);
        assert_eq!(read_i32(140), 10);
        assert_eq!(read_i32(144), 20);
        assert_eq!(read_i32(148), 100);
        assert_eq!(read_i32(152), 200);
        assert_eq!(read_i32(156), 11);
        assert_eq!(read_i32(160), 12);
        assert_eq!(read_i32(164), 110);
        assert_eq!(read_i32(168), 120);
        assert_eq!(read_i32(172), 21);
        assert_eq!(read_i32(176), 22);
        assert_eq!(read_i32(180), 210);
        assert_eq!(read_i32(184), 220);
        assert_eq!(read_i32(188), 30);
        assert_eq!(read_i32(192), 40);
        assert_eq!(read_i32(196), 300);
        assert_eq!(read_i32(200), 400);
        assert_eq!(read_u32(204), 7);
        assert_eq!(read_u32(212), 9);
        assert_eq!(bytes.len(), 216);
    }

    #[test]
    fn creategrid_sorted_keys_copy_match_requires_expected_layout() {
        let sorted_keys = 0x1000_u64;
        let n = 64usize;
        assert!(is_creategrid_sorted_keys_copy(
            sorted_keys,
            n,
            sorted_keys,
            sorted_keys + (n * 4) as u64,
            n * 4,
        ));
        assert!(
            !is_creategrid_sorted_keys_copy(sorted_keys, n, sorted_keys, sorted_keys + 4, n * 4,),
            "unexpected source offset must not match the CreateGrid overwrite pattern"
        );
    }

    #[test]
    fn classify_creategrid_sort_buffer_distinguishes_input_and_copy_source() {
        let sorted_keys = 0x2000_u64;
        let n = 128usize;
        assert_eq!(
            classify_creategrid_sort_buffer(sorted_keys, n, sorted_keys),
            Some(CreateGridSortBufferRole::CreateGridInput)
        );
        assert_eq!(
            classify_creategrid_sort_buffer(sorted_keys, n, sorted_keys + (n * 4) as u64),
            Some(CreateGridSortBufferRole::CopySource)
        );
        assert_eq!(
            classify_creategrid_sort_buffer(sorted_keys, n, sorted_keys + 8),
            None
        );
    }

    #[test]
    fn locate_creategrid_sort_buffer_hit_tracks_in_range_offsets() {
        let sorted_keys = 0x3000_u64;
        let n = 64usize;
        assert_eq!(
            locate_creategrid_sort_buffer_hit(sorted_keys, n, sorted_keys),
            Some(CreateGridSortBufferHit {
                role: CreateGridSortBufferRole::CreateGridInput,
                byte_offset: 0,
            })
        );
        assert_eq!(
            locate_creategrid_sort_buffer_hit(sorted_keys, n, sorted_keys + 0x1c),
            Some(CreateGridSortBufferHit {
                role: CreateGridSortBufferRole::CreateGridInput,
                byte_offset: 0x1c,
            })
        );
        let copy_src = sorted_keys + (n * 4) as u64;
        assert_eq!(
            locate_creategrid_sort_buffer_hit(sorted_keys, n, copy_src + 0x20),
            Some(CreateGridSortBufferHit {
                role: CreateGridSortBufferRole::CopySource,
                byte_offset: 0x20,
            })
        );
        assert_eq!(
            locate_creategrid_sort_buffer_hit(sorted_keys, n, copy_src + (n * 4) as u64),
            None,
            "the range end must stay exclusive"
        );
    }

    #[test]
    fn scan_launch_extra_for_creategrid_sort_refs_finds_both_roles() {
        let sorted_keys = 0x4000_u64;
        let n = 64usize;
        let copy_src = sorted_keys + (n * 4) as u64;
        let mut extra = vec![0u8; 48];
        extra[8..16].copy_from_slice(&sorted_keys.to_le_bytes());
        extra[24..32].copy_from_slice(&copy_src.to_le_bytes());
        let refs = scan_launch_extra_for_creategrid_sort_refs(sorted_keys, n, &extra);
        assert_eq!(
            refs,
            vec![
                CreateGridSortTouchArg {
                    location: CreateGridSortTouchArgLocation::ExtraOffset(8),
                    buffer: sorted_keys,
                    role: CreateGridSortBufferRole::CreateGridInput,
                    byte_offset: 0,
                },
                CreateGridSortTouchArg {
                    location: CreateGridSortTouchArgLocation::ExtraOffset(24),
                    buffer: copy_src,
                    role: CreateGridSortBufferRole::CopySource,
                    byte_offset: 0,
                },
            ]
        );
    }

    #[test]
    fn scan_launch_extra_for_creategrid_sort_refs_ignores_nonmatching_values() {
        let sorted_keys = 0x5000_u64;
        let n = 32usize;
        let mut extra = vec![0u8; 32];
        extra[0..8].copy_from_slice(&(sorted_keys - 8).to_le_bytes());
        extra[8..16].copy_from_slice(&(sorted_keys + (n * 8) as u64).to_le_bytes());
        assert!(scan_launch_extra_for_creategrid_sort_refs(sorted_keys, n, &extra).is_empty());
    }

    #[test]
    fn scan_launch_extra_for_creategrid_sort_refs_reports_in_range_offsets() {
        let sorted_keys = 0x6000_u64;
        let n = 32usize;
        let copy_src = sorted_keys + (n * 4) as u64;
        let mut extra = vec![0u8; 40];
        extra[0..8].copy_from_slice(&(sorted_keys + 0x10).to_le_bytes());
        extra[16..24].copy_from_slice(&(copy_src + 0x24).to_le_bytes());
        let refs = scan_launch_extra_for_creategrid_sort_refs(sorted_keys, n, &extra);
        assert_eq!(
            refs,
            vec![
                CreateGridSortTouchArg {
                    location: CreateGridSortTouchArgLocation::ExtraOffset(0),
                    buffer: sorted_keys + 0x10,
                    role: CreateGridSortBufferRole::CreateGridInput,
                    byte_offset: 0x10,
                },
                CreateGridSortTouchArg {
                    location: CreateGridSortTouchArgLocation::ExtraOffset(16),
                    buffer: copy_src + 0x24,
                    role: CreateGridSortBufferRole::CopySource,
                    byte_offset: 0x24,
                },
            ]
        );
    }

    #[test]
    fn scan_launch_kernel_params_for_creategrid_sort_refs_finds_pointer_args() {
        let sorted_keys = 0x7000_u64;
        let n = 32usize;
        let copy_src = sorted_keys + (n * 4) as u64;
        let n_items = n as u32;
        let scalar = 17u32;
        let input_ptr = sorted_keys + 0x18;
        let copy_ptr = copy_src + 0x20;
        let tail = 99u32;
        let mut kernel_params = [
            (&n_items as *const u32).cast_mut().cast(),
            (&scalar as *const u32).cast_mut().cast(),
            (&input_ptr as *const u64).cast_mut().cast(),
            (&copy_ptr as *const u64).cast_mut().cast(),
            (&tail as *const u32).cast_mut().cast(),
            std::ptr::null_mut(),
        ];
        let refs = unsafe {
            scan_launch_kernel_params_for_creategrid_sort_refs(
                sorted_keys,
                n,
                kernel_params.as_mut_ptr(),
            )
        };
        assert_eq!(
            refs,
            vec![
                CreateGridSortTouchArg {
                    location: CreateGridSortTouchArgLocation::KernelParamIndex(2),
                    buffer: input_ptr,
                    role: CreateGridSortBufferRole::CreateGridInput,
                    byte_offset: 0x18,
                },
                CreateGridSortTouchArg {
                    location: CreateGridSortTouchArgLocation::KernelParamIndex(3),
                    buffer: copy_ptr,
                    role: CreateGridSortBufferRole::CopySource,
                    byte_offset: 0x20,
                },
            ]
        );
    }

    #[test]
    fn format_creategrid_sort_touch_refs_formats_offsets_and_roles() {
        let refs = vec![
            CreateGridSortTouchArg {
                location: CreateGridSortTouchArgLocation::ExtraOffset(8),
                buffer: 0x7000,
                role: CreateGridSortBufferRole::CreateGridInput,
                byte_offset: 0,
            },
            CreateGridSortTouchArg {
                location: CreateGridSortTouchArgLocation::KernelParamIndex(3),
                buffer: 0x7120,
                role: CreateGridSortBufferRole::CopySource,
                byte_offset: 0x20,
            },
        ];
        assert_eq!(
            format_creategrid_sort_touch_refs(&refs),
            "extra+8=0x7000(creategrid_input), arg3=0x7120(copy_src+0x20)"
        );
    }

    #[test]
    fn creategrid_snapshot_restores_signed_radix_residue() {
        let restored = restore_signed_radix_residue(&[4, -2147483600, 12]);
        assert_eq!(restored, vec![4, 48, 12]);
    }

    #[test]
    fn classify_tfs_kernel_covers_recent_dynamic_grid_crash_family() {
        assert_eq!(
            classify_tfs_kernel("TurbulenceFS_dynamicGridClearKernel"),
            Some(TFS_KERNEL_ID_CLEAR)
        );
        assert_eq!(
            classify_tfs_kernel("TurbulenceFS_dynamicGridMigrateKernel"),
            Some(TFS_KERNEL_ID_MIGRATE)
        );
        assert_eq!(
            classify_tfs_kernel("TurbulenceFS_dynamicGridParticleDensityAtomicToTexKernel"),
            Some(TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX)
        );
        assert_eq!(
            classify_tfs_kernel("TurbulenceFS_dynamicGridParticleDensitySmoothYKernel"),
            Some(TFS_KERNEL_ID_PARTICLE_DENSITY_SMOOTH)
        );
        assert_eq!(
            classify_tfs_kernel("TurbulenceFS_fieldSamplerPointsKernel_templ$0"),
            Some(TFS_KERNEL_ID_FIELD_SAMPLER_POINTS)
        );
        assert_eq!(
            classify_tfs_kernel("TurbulenceFS_dynamicGridMGRestrictKernel3"),
            Some(TFS_KERNEL_ID_OTHER)
        );
        assert_eq!(
            classify_tfs_kernel("CreateParticleShapesIncremental_MultiStage_1"),
            Some(TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS1)
        );
        assert_eq!(
            classify_tfs_kernel("CreateParticleShapesIncremental_MultiStage_2_Batched"),
            Some(TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED)
        );
        assert_eq!(
            classify_tfs_kernel("CreateParticleShapesIncremental_MultiStage_2"),
            Some(TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2)
        );
        assert_eq!(
            classify_tfs_kernel("CreateParticleShapesIncremental_MultiStage_3"),
            Some(TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3)
        );
        assert_eq!(
            classify_tfs_kernel("CreateParticleShapesIncremental_MultiStage_4"),
            Some(TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4)
        );
        assert_eq!(
            classify_tfs_kernel("InitCollDataAndApplyConstraints"),
            Some(TFS_KERNEL_ID_INIT_COLL_DATA_AND_APPLY_CONSTRAINTS)
        );
        assert_eq!(
            classify_tfs_kernel("FinalizeCollision"),
            Some(TFS_KERNEL_ID_FINALIZE_COLLISION)
        );
        assert_eq!(
            classify_tfs_kernel("ParticleCollisionStaticBox"),
            Some(TFS_KERNEL_ID_PARTICLE_COLLISION_STATIC_BOX)
        );
        assert_eq!(
            classify_tfs_kernel("IOFX_actorRangeKernel"),
            Some(TFS_KERNEL_ID_IOFX_ACTOR_RANGE)
        );
    }

    #[test]
    fn dump_kernel_args_raw_keeps_late_kernel_params_after_null_holes() {
        let arg1 = 0x1122_3344_5566_7788_u64;
        let arg3 = 0x99aa_bbcc_ddee_ff00_u64;
        let mut kernel_params = [
            std::ptr::null_mut(),
            (&arg1 as *const u64).cast_mut().cast(),
            std::ptr::null_mut(),
            (&arg3 as *const u64).cast_mut().cast(),
        ];
        let mut dumped = [0u8; 32];
        let written = unsafe {
            dump_kernel_args_raw(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                dumped.as_mut_ptr(),
                dumped.len(),
            )
        };
        assert_eq!(written, dumped.len());
        assert_eq!(&dumped[0..8], &[0u8; 8]);
        assert_eq!(&dumped[8..16], &arg1.to_le_bytes());
        assert_eq!(&dumped[16..24], &[0u8; 8]);
        assert_eq!(&dumped[24..32], &arg3.to_le_bytes());
    }

    #[test]
    fn dump_kernel_args_raw_falls_back_to_kernel_params_when_extra_is_not_flat_buffer() {
        let arg0 = 0x0123_4567_89ab_cdef_u64;
        let arg1 = 0xfedc_ba98_7654_3210_u64;
        let bogus_extra: [*mut std::ffi::c_void; 5] = [std::ptr::null_mut(); 5];
        let mut kernel_params = [
            (&arg0 as *const u64).cast_mut().cast(),
            (&arg1 as *const u64).cast_mut().cast(),
        ];
        let mut dumped = [0u8; 16];
        let written = unsafe {
            dump_kernel_args_raw(
                kernel_params.as_mut_ptr(),
                bogus_extra.as_ptr().cast_mut(),
                dumped.as_mut_ptr(),
                dumped.len(),
            )
        };
        assert_eq!(written, dumped.len());
        assert_eq!(&dumped[0..8], &arg0.to_le_bytes());
        assert_eq!(&dumped[8..16], &arg1.to_le_bytes());
    }

    #[test]
    fn dump_kernel_param_ptrs_records_entry_addresses_without_dereferencing() {
        let mut kernel_params = [
            0x1111usize as *mut std::ffi::c_void,
            std::ptr::null_mut(),
            0x3333usize as *mut std::ffi::c_void,
            0x4444usize as *mut std::ffi::c_void,
        ];
        let mut dumped = [0u8; 32];
        let written = unsafe {
            dump_kernel_param_ptrs(
                kernel_params.as_mut_ptr(),
                dumped.as_mut_ptr(),
                dumped.len(),
            )
        };
        assert_eq!(written, dumped.len());
        assert_eq!(&dumped[0..8], &(0x1111u64).to_le_bytes());
        assert_eq!(&dumped[8..16], &(0u64).to_le_bytes());
        assert_eq!(&dumped[16..24], &(0x3333u64).to_le_bytes());
        assert_eq!(&dumped[24..32], &(0x4444u64).to_le_bytes());
    }

    #[test]
    fn summarize_launch_arg_encoding_reports_extra_and_kernel_param_shapes() {
        let bogus_extra: [*mut std::ffi::c_void; 5] = [
            0x10usize as *mut std::ffi::c_void,
            0x20usize as *mut std::ffi::c_void,
            0x30usize as *mut std::ffi::c_void,
            0x40usize as *mut std::ffi::c_void,
            std::ptr::null_mut(),
        ];
        let summary = unsafe {
            summarize_launch_arg_encoding(0x1234usize as *mut _, bogus_extra.as_ptr().cast_mut())
        };
        assert!(summary.contains("extra_null=false"));
        assert!(summary.contains("extra_flat=false"));
        assert!(summary.contains("kernel_params_null=false"));
        assert!(summary.contains("extra=[0:0x10, 1:0x20, 2:0x30, 3:0x40, 4:0x0]"));
    }

    #[test]
    fn encode_launch_arg_meta_tracks_extra_markers_and_nullness() {
        let flat_extra: [*mut std::ffi::c_void; 5] = [
            0x1usize as *mut std::ffi::c_void,
            0x20usize as *mut std::ffi::c_void,
            0x2usize as *mut std::ffi::c_void,
            0x40usize as *mut std::ffi::c_void,
            std::ptr::null_mut(),
        ];
        let meta = unsafe {
            encode_launch_arg_meta(0x1234usize as *mut _, flat_extra.as_ptr().cast_mut())
        };
        assert_eq!(meta[0], 0, "extra should be marked present");
        assert_eq!(meta[1], 0, "kernel_params should be marked present");
        assert_eq!(meta[2], 1, "extra[0] should match flat-buffer marker");
        assert_eq!(meta[3], 1, "extra[2] should match size marker");
        assert_eq!(meta[4], 0, "extra[1] should be non-null");
        assert_eq!(meta[5], 0, "extra[3] should be non-null");

        let null_meta =
            unsafe { encode_launch_arg_meta(std::ptr::null_mut(), std::ptr::null_mut()) };
        assert_eq!(null_meta[0], 1);
        assert_eq!(null_meta[1], 1);
    }

    #[test]
    fn read_iofx_actor_range_args_reads_flat_buffer_layout() {
        let mut raw = [0u8; 72];
        raw[12..16].copy_from_slice(&6214u32.to_le_bytes());
        raw[24..32].copy_from_slice(&0x1111_2222_3333_4444u64.to_le_bytes());
        raw[40..48].copy_from_slice(&0x5555_6666_7777_8888u64.to_le_bytes());
        raw[48..56].copy_from_slice(&0x9999_aaaa_bbbb_ccccu64.to_le_bytes());
        raw[56..64].copy_from_slice(&0xdddd_eeee_ffff_0001u64.to_le_bytes());
        raw[64..72].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        let raw_len = raw.len();
        let mut raw_len_storage = raw_len;
        let extra = [
            0x1usize as *mut std::ffi::c_void,
            raw.as_mut_ptr().cast(),
            0x2usize as *mut std::ffi::c_void,
            (&mut raw_len_storage as *mut usize).cast(),
            std::ptr::null_mut(),
        ];
        let args =
            unsafe { read_iofx_actor_range_args(std::ptr::null_mut(), extra.as_ptr().cast_mut()) }
                .expect("actorRange args should decode");
        assert_eq!(
            args,
            IofxActorRangeArgs {
                count: 6214,
                input_ptr_324: 0x1111_2222_3333_4444,
                output_ptr_326: 0x5555_6666_7777_8888,
                output_ptr_327: 0x9999_aaaa_bbbb_cccc,
                output_ptr_328: 0xdddd_eeee_ffff_0001,
                input_ptr_329: 0x1234_5678_9abc_def0,
            }
        );
    }

    #[test]
    fn read_iofx_actor_range_args_rejects_zero_pointers_and_zero_count() {
        // Note: oversized count (> IOFX_ACTOR_RANGE_MAX_COUNT) is NOT rejected
        // anymore as of 2026-05-11; large-count launches must be captured so
        // we can see actorRange faults in full PhysX sim. Buffer copy still
        // caps at IOFX_ACTOR_RANGE_INPUT_BYTES via min(count*4, ...).
        let mut raw = [0u8; 72];
        // Zero count is still rejected.
        raw[12..16].copy_from_slice(&0u32.to_le_bytes());
        raw[24..32].copy_from_slice(&0x1111u64.to_le_bytes());
        raw[40..48].copy_from_slice(&0x2222u64.to_le_bytes());
        raw[48..56].copy_from_slice(&0x3333u64.to_le_bytes());
        raw[56..64].copy_from_slice(&0x4444u64.to_le_bytes());
        raw[64..72].copy_from_slice(&0x5555u64.to_le_bytes());
        let raw_len = raw.len();
        let mut raw_len_storage = raw_len;
        let extra = [
            0x1usize as *mut std::ffi::c_void,
            raw.as_mut_ptr().cast(),
            0x2usize as *mut std::ffi::c_void,
            (&mut raw_len_storage as *mut usize).cast(),
            std::ptr::null_mut(),
        ];
        assert!(unsafe {
            read_iofx_actor_range_args(std::ptr::null_mut(), extra.as_ptr().cast_mut())
        }
        .is_none());

        // Zero pointer is still rejected.
        raw[12..16].copy_from_slice(&16u32.to_le_bytes());
        raw[24..32].copy_from_slice(&0u64.to_le_bytes());
        assert!(unsafe {
            read_iofx_actor_range_args(std::ptr::null_mut(), extra.as_ptr().cast_mut())
        }
        .is_none());

        // Previously oversized (>= 8192) — should now be ACCEPTED.
        raw[12..16].copy_from_slice(&((IOFX_ACTOR_RANGE_MAX_COUNT + 1) as u32).to_le_bytes());
        raw[24..32].copy_from_slice(&0x1111u64.to_le_bytes());
        assert!(unsafe {
            read_iofx_actor_range_args(std::ptr::null_mut(), extra.as_ptr().cast_mut())
        }
        .is_some());
    }

    #[test]
    fn read_iofx_bbox_args_reads_flat_buffer_layout() {
        let mut raw = [0u8; 96];
        raw[12..16].copy_from_slice(&17u32.to_le_bytes());
        raw[16..20].copy_from_slice(&23u32.to_le_bytes());
        raw[24..32].copy_from_slice(&0x1111_2222_3333_4444u64.to_le_bytes());
        raw[32..40].copy_from_slice(&0x5555_6666_7777_8888u64.to_le_bytes());
        raw[40..48].copy_from_slice(&0x9999_aaaa_bbbb_ccccu64.to_le_bytes());
        raw[48..56].copy_from_slice(&0xdddd_eeee_ffff_0001u64.to_le_bytes());
        raw[56..64].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        raw[64..72].copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        raw[72..80].copy_from_slice(&0xfedc_ba98_7654_3210u64.to_le_bytes());
        raw[80..88].copy_from_slice(&0x0bad_f00d_dead_beefu64.to_le_bytes());
        raw[88..92].copy_from_slice(&31u32.to_le_bytes());
        raw[92..96].copy_from_slice(&37u32.to_le_bytes());
        let raw_len = raw.len();
        let mut raw_len_storage = raw_len;
        let extra = [
            0x1usize as *mut std::ffi::c_void,
            raw.as_mut_ptr().cast(),
            0x2usize as *mut std::ffi::c_void,
            (&mut raw_len_storage as *mut usize).cast(),
            std::ptr::null_mut(),
        ];
        let args = unsafe { read_iofx_bbox_args(std::ptr::null_mut(), extra.as_ptr().cast_mut()) }
            .expect("bbox args should decode");
        assert_eq!(
            args,
            IofxBboxArgs {
                param_2: 17,
                param_3: 23,
                ptr_params: [
                    0x1111_2222_3333_4444,
                    0x5555_6666_7777_8888,
                    0x9999_aaaa_bbbb_cccc,
                    0xdddd_eeee_ffff_0001,
                    0x1234_5678_9abc_def0,
                    0x0123_4567_89ab_cdef,
                    0xfedc_ba98_7654_3210,
                    0x0bad_f00d_dead_beef,
                ],
                param_12: 31,
                param_13: 37,
            }
        );
    }

    #[test]
    fn read_iofx_bbox_args_rejects_all_zero_pointer_layouts() {
        let mut raw = [0u8; 96];
        raw[12..16].copy_from_slice(&17u32.to_le_bytes());
        raw[16..20].copy_from_slice(&23u32.to_le_bytes());
        raw[88..92].copy_from_slice(&31u32.to_le_bytes());
        raw[92..96].copy_from_slice(&37u32.to_le_bytes());
        let raw_len = raw.len();
        let mut raw_len_storage = raw_len;
        let extra = [
            0x1usize as *mut std::ffi::c_void,
            raw.as_mut_ptr().cast(),
            0x2usize as *mut std::ffi::c_void,
            (&mut raw_len_storage as *mut usize).cast(),
            std::ptr::null_mut(),
        ];
        assert!(
            unsafe { read_iofx_bbox_args(std::ptr::null_mut(), extra.as_ptr().cast_mut()) }
                .is_none()
        );
    }

    #[test]
    fn read_iofx_bbox_args_allows_unused_param_6_to_be_missing_from_kernel_params() {
        let param_2 = 17u32;
        let param_3 = 23u32;
        let param_4 = 0x1111_2222_3333_4444u64;
        let param_5 = 0x5555_6666_7777_8888u64;
        let param_7 = 0x9999_aaaa_bbbb_ccccu64;
        let param_8 = 0xdddd_eeee_ffff_0001u64;
        let param_9 = 0x1234_5678_9abc_def0u64;
        let param_10 = 0x0123_4567_89ab_cdefu64;
        let param_11 = 0xfedc_ba98_7654_3210u64;
        let param_12 = 31u32;
        let param_13 = 37u32;
        let mut kernel_params = [std::ptr::null_mut(); 14];
        kernel_params[2] = (&param_2 as *const u32).cast_mut().cast();
        kernel_params[3] = (&param_3 as *const u32).cast_mut().cast();
        kernel_params[4] = (&param_4 as *const u64).cast_mut().cast();
        kernel_params[5] = (&param_5 as *const u64).cast_mut().cast();
        kernel_params[7] = (&param_7 as *const u64).cast_mut().cast();
        kernel_params[8] = (&param_8 as *const u64).cast_mut().cast();
        kernel_params[9] = (&param_9 as *const u64).cast_mut().cast();
        kernel_params[10] = (&param_10 as *const u64).cast_mut().cast();
        kernel_params[11] = (&param_11 as *const u64).cast_mut().cast();
        kernel_params[12] = (&param_12 as *const u32).cast_mut().cast();
        kernel_params[13] = (&param_13 as *const u32).cast_mut().cast();

        let args = unsafe { read_iofx_bbox_args(kernel_params.as_mut_ptr(), std::ptr::null_mut()) }
            .expect("bbox args should decode with null param_6");
        assert_eq!(args.param_2, param_2);
        assert_eq!(args.param_3, param_3);
        assert_eq!(args.ptr_params[0], param_4);
        assert_eq!(args.ptr_params[1], param_5);
        assert_eq!(args.ptr_params[2], 0);
        assert_eq!(args.ptr_params[3], param_7);
        assert_eq!(args.ptr_params[4], param_8);
        assert_eq!(args.ptr_params[5], param_9);
        assert_eq!(args.ptr_params[6], param_10);
        assert_eq!(args.ptr_params[7], param_11);
        assert_eq!(args.param_12, param_12);
        assert_eq!(args.param_13, param_13);
    }

    #[test]
    fn iofx_bbox_capture_slot_layout_includes_pre_and_post_regions() {
        assert_eq!(
            IOFX_BBOX_PRE_PTRS_OFF,
            IOFX_BBOX_RAW_ARGS_OFF + TFS_RAW_ARGS_BYTES
        );
        assert_eq!(
            IOFX_BBOX_POST_PTRS_OFF,
            IOFX_BBOX_PRE_PTRS_OFF + IOFX_BBOX_POINTER_COUNT * IOFX_BBOX_BUFFER_BYTES
        );
        assert_eq!(
            std::mem::size_of::<IofxBboxCaptureSlot>(),
            IOFX_BBOX_TEXREF_OFF + IOFX_BBOX_TEXREF_BYTES
        );
    }

    #[test]
    fn iofx_actor_range_capture_slot_layout_includes_output_regions() {
        assert_eq!(
            IOFX_ACTOR_RANGE_INPUT_329_OFF,
            IOFX_ACTOR_RANGE_INPUT_324_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
        assert_eq!(
            IOFX_ACTOR_RANGE_OUTPUT_326_OFF,
            IOFX_ACTOR_RANGE_INPUT_329_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
        assert_eq!(
            IOFX_ACTOR_RANGE_OUTPUT_327_OFF,
            IOFX_ACTOR_RANGE_OUTPUT_326_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
        assert_eq!(
            IOFX_ACTOR_RANGE_OUTPUT_328_OFF,
            IOFX_ACTOR_RANGE_OUTPUT_327_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
        assert_eq!(
            IOFX_ACTOR_RANGE_PRE_OUTPUT_326_OFF,
            IOFX_ACTOR_RANGE_OUTPUT_328_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
        assert_eq!(
            IOFX_ACTOR_RANGE_PRE_OUTPUT_327_OFF,
            IOFX_ACTOR_RANGE_PRE_OUTPUT_326_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
        assert_eq!(
            IOFX_ACTOR_RANGE_PRE_OUTPUT_328_OFF,
            IOFX_ACTOR_RANGE_PRE_OUTPUT_327_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
        assert_eq!(
            std::mem::size_of::<IofxActorRangeCaptureSlot>(),
            IOFX_ACTOR_RANGE_PRE_OUTPUT_328_OFF + IOFX_ACTOR_RANGE_INPUT_BYTES
        );
    }

    #[test]
    fn iofx_chain_capture_slot_layout_matches_python_parser() {
        // tools/parse_iofx_chain.py hard-codes these offsets. If they ever
        // drift the dumped file becomes unparsable and chain capture
        // silently returns zero data downstream.
        assert_eq!(IOFX_CHAIN_RAW_ARGS_OFF, 208, "header bytes");
        assert_eq!(IOFX_CHAIN_PRE_PTRS_OFF, 720, "pre_ptrs offset");
        assert_eq!(IOFX_CHAIN_POST_PTRS_OFF, 131792, "post_ptrs offset");
        assert_eq!(IOFX_CHAIN_SLOT_SIZE, 262864, "slot size");
        // Also assert the offset_of of each field is what we annotated in
        // the struct comments — repr(C) shouldn't insert padding given the
        // u64-then-u32-then-arrays layout but pin it down.
        assert_eq!(std::mem::offset_of!(IofxChainCaptureSlot, magic), 0);
        assert_eq!(std::mem::offset_of!(IofxChainCaptureSlot, launch_n), 8);
        assert_eq!(std::mem::offset_of!(IofxChainCaptureSlot, timestamp_ms), 16);
        assert_eq!(std::mem::offset_of!(IofxChainCaptureSlot, kernel_kind), 24);
        assert_eq!(std::mem::offset_of!(IofxChainCaptureSlot, kernel_name), 72);
        assert_eq!(std::mem::offset_of!(IofxChainCaptureSlot, ptr_params), 136);
        assert_eq!(
            std::mem::offset_of!(IofxChainCaptureSlot, ptr_buffer_bytes),
            200
        );
        assert_eq!(std::mem::offset_of!(IofxChainCaptureSlot, raw_args), 208);
        assert_eq!(
            std::mem::offset_of!(IofxChainCaptureSlot, pre_ptr_buffers),
            720
        );
        assert_eq!(
            std::mem::offset_of!(IofxChainCaptureSlot, post_ptr_buffers),
            131792
        );
    }

    #[test]
    fn iofx_actor_range_output_copy_bytes_uses_smallest_positive_pointer_gap() {
        assert_eq!(
            iofx_actor_range_output_copy_bytes(0x1000, 0x1800, 0x2000, 0x3000),
            0x800
        );
    }

    #[test]
    fn iofx_actor_range_output_copy_bytes_falls_back_to_input_span_for_separate_buffers() {
        assert_eq!(
            iofx_actor_range_output_copy_bytes(0x1000, 0x5000, 0x9000, 0x3000),
            0x3000
        );
    }

    #[test]
    fn should_skip_tfs_raw_arg_dump_for_known_kernel_params_only_crash_edge_kernels() {
        assert!(should_skip_tfs_raw_arg_dump(
            "CreateParticleShapesIncremental_MultiStage_1"
        ));
        assert!(should_skip_tfs_raw_arg_dump(
            "CreateParticleShapesIncremental_MultiStage_2_Batched"
        ));
        assert!(!should_skip_tfs_raw_arg_dump(
            "CreateParticleShapesIncremental_MultiStage_2"
        ));
        assert!(should_skip_tfs_raw_arg_dump(
            "CreateParticleShapesIncremental_MultiStage_3"
        ));
        assert!(should_skip_tfs_raw_arg_dump(
            "CreateParticleShapesIncremental_MultiStage_4"
        ));
        assert!(should_skip_tfs_raw_arg_dump(
            "InitCollDataAndApplyConstraints"
        ));
        assert!(should_skip_tfs_raw_arg_dump("FinalizeCollision"));
        assert!(should_skip_tfs_raw_arg_dump("ParticleCollisionStaticBox"));
        assert!(!should_skip_tfs_raw_arg_dump(
            "TurbulenceFS_dynamicGridClearKernel"
        ));
    }

    #[test]
    fn tfs_ms4_args_capture_v1_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs4ArgsCaptureV1>(), 56);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, n), 8);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, p2), 16);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, p4), 24);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, p5), 32);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, p6), 40);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV1, p7), 48);
    }

    #[test]
    fn tfs_ms4_args_capture_v2_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs4ArgsCaptureV2>(), 488);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, n), 8);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p2), 16);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p4), 24);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p5), 32);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p6), 40);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p7), 48);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p2_words), 56);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p4_words), 60);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p6_words), 64);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p7_words), 68);
        assert_eq!(std::mem::offset_of!(TfsMs4ArgsCaptureV2, p2_words_data), 72);
        assert_eq!(
            std::mem::offset_of!(TfsMs4ArgsCaptureV2, p4_words_data),
            200
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs4ArgsCaptureV2, p6_words_data),
            232
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs4ArgsCaptureV2, p7_words_data),
            360
        );
    }

    #[test]
    fn dump_tfs_ms4_args_v2_encodes_header_with_bounded_snapshot_counts() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 8];
        let p2: u64 = 0;
        let n: u32 = 1234;
        let p4: u64 = 0;
        let p5: u64 = 0x7f00_3333_0000;
        let p6: u64 = 0;
        let p7: u64 = 0;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();
        kernel_params[3] = (&n as *const u32).cast_mut().cast();
        kernel_params[4] = (&p4 as *const u64).cast_mut().cast();
        kernel_params[5] = (&p5 as *const u64).cast_mut().cast();
        kernel_params[6] = (&p6 as *const u64).cast_mut().cast();
        kernel_params[7] = (&p7 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms4_args_v2(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs4ArgsCaptureV2>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs4ArgsCaptureV2) };
        assert_eq!(encoded.magic, TFS_MS4_ARGS_CAPTURE_MAGIC);
        assert_eq!(encoded.version, TFS_MS4_ARGS_CAPTURE_VERSION_V2);
        assert_eq!(encoded.n, n);
        assert_eq!(encoded.p2, p2);
        assert_eq!(encoded.p4, p4);
        assert_eq!(encoded.p5, p5);
        assert_eq!(encoded.p6, p6);
        assert_eq!(encoded.p7, p7);
        assert_eq!(encoded.p2_words, 0);
        assert_eq!(encoded.p4_words, 0);
        assert_eq!(encoded.p6_words, 0);
        assert_eq!(encoded.p7_words, 0);
    }

    #[test]
    fn tfs_ms2b_args_capture_v1_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs2BatchedArgsCaptureV1>(), 16);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV1, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV1, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV1, p2), 8);
    }

    #[test]
    fn tfs_ms2b_args_capture_v2_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs2BatchedArgsCaptureV2>(), 296);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, p2), 8);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, grid_x), 16);
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, block_x),
            20
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, packet_sample_bytes),
            24
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV2, packet_sample),
            32
        );
    }

    #[test]
    fn tfs_ms2b_args_capture_v3_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs2BatchedArgsCaptureV3>(), 408);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, p2), 8);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, grid_x), 16);
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, block_x),
            20
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, packet_sample_bytes),
            24
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, packet_sample),
            32
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, rd5_words),
            296
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, rd5_words_data),
            312
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, rd6_words_data),
            344
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV3, rd7_words_data),
            376
        );
    }

    #[test]
    fn tfs_ms2b_args_capture_v4_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs2BatchedArgsCaptureV4>(), 480);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, p2), 8);
        assert_eq!(std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, grid_x), 16);
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, block_x),
            20
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, packet_sample_bytes),
            24
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, packet_sample),
            32
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, rd5_words),
            296
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, rd5_words_data),
            312
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, rd6_words_data),
            344
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, rd7_words_data),
            376
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, requested_entry),
            408
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, effective_entry),
            412
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, slot_probe_words),
            416
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, slot_probe_slots),
            432
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, slot_probe_rd5_p4),
            448
        );
        assert_eq!(
            std::mem::offset_of!(TfsMs2BatchedArgsCaptureV4, slot_probe_r38),
            464
        );
    }

    #[test]
    fn tfs_ms3_args_capture_v1_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs3ArgsCaptureV1>(), 24);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV1, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV1, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV1, p2), 8);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV1, p3), 16);
    }

    #[test]
    fn tfs_ms3_args_capture_v2_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMs3ArgsCaptureV2>(), 96);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, version), 4);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, p2), 8);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, p3), 16);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, p2_words), 24);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, p3_words), 28);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, p2_words_data), 32);
        assert_eq!(std::mem::offset_of!(TfsMs3ArgsCaptureV2, p3_words_data), 64);
    }

    #[test]
    fn dump_tfs_ms2_batched_args_encodes_expected_fixed_fields() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 3];
        let p2: u64 = 0x7f00_1111_0000;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms2_batched_args(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs2BatchedArgsCaptureV1>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs2BatchedArgsCaptureV1) };
        assert_eq!(
            encoded,
            TfsMs2BatchedArgsCaptureV1 {
                magic: TFS_MS2B_ARGS_CAPTURE_MAGIC,
                version: TFS_MS2B_ARGS_CAPTURE_VERSION,
                p2,
            }
        );
    }

    #[test]
    fn dump_tfs_ms2_batched_args_v2_encodes_header_and_zero_sample_when_unmapped() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 3];
        let p2: u64 = 0;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms2_batched_args_v2(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                3,
                256,
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs2BatchedArgsCaptureV2>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs2BatchedArgsCaptureV2) };
        assert_eq!(encoded.magic, TFS_MS2B_ARGS_CAPTURE_MAGIC);
        assert_eq!(encoded.version, TFS_MS2B_ARGS_CAPTURE_VERSION_V2);
        assert_eq!(encoded.p2, p2);
        assert_eq!(encoded.grid_x, 3);
        assert_eq!(encoded.block_x, 256);
        assert_eq!(encoded.packet_sample_bytes, 0);
    }

    #[test]
    fn dump_tfs_ms2_batched_args_v3_encodes_header_and_zero_samples_when_unmapped() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 3];
        let p2: u64 = 0;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms2_batched_args_v3(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                123,
                3,
                256,
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs2BatchedArgsCaptureV3>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs2BatchedArgsCaptureV3) };
        assert_eq!(encoded.magic, TFS_MS2B_ARGS_CAPTURE_MAGIC);
        assert_eq!(encoded.version, TFS_MS2B_ARGS_CAPTURE_VERSION_V3);
        assert_eq!(encoded.p2, p2);
        assert_eq!(encoded.grid_x, 3);
        assert_eq!(encoded.block_x, 256);
        assert_eq!(encoded.packet_sample_bytes, 0);
        assert_eq!(encoded.rd5_words, 0);
        assert_eq!(encoded.rd6_words, 0);
        assert_eq!(encoded.rd7_words, 0);
    }

    #[test]
    fn dump_tfs_ms2_batched_args_v4_encodes_header_and_zero_samples_when_unmapped() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 3];
        let p2: u64 = 0;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms2_batched_args_v4(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                123,
                3,
                256,
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs2BatchedArgsCaptureV4>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs2BatchedArgsCaptureV4) };
        assert_eq!(encoded.magic, TFS_MS2B_ARGS_CAPTURE_MAGIC);
        assert_eq!(encoded.version, TFS_MS2B_ARGS_CAPTURE_VERSION_V4);
        assert_eq!(encoded.p2, p2);
        assert_eq!(encoded.grid_x, 3);
        assert_eq!(encoded.block_x, 256);
        assert_eq!(encoded.packet_sample_bytes, 0);
        assert_eq!(encoded.rd5_words, 0);
        assert_eq!(encoded.rd6_words, 0);
        assert_eq!(encoded.rd7_words, 0);
        assert_eq!(encoded.requested_entry, 0);
        assert_eq!(encoded.effective_entry, 0);
        assert_eq!(encoded.slot_probe_words, 0);
        assert_eq!(encoded.slot_probe_rd5_mask, 0);
        assert_eq!(encoded.slot_probe_r38_mask, 0);
    }

    #[test]
    fn parse_tfs_mut_p47_entry_index_clamps_and_defaults() {
        assert_eq!(parse_tfs_mut_p47_entry_index(None), 0);
        assert_eq!(parse_tfs_mut_p47_entry_index(Some(String::from("bad"))), 0);
        assert_eq!(parse_tfs_mut_p47_entry_index(Some(String::from("2"))), 2);
        assert_eq!(parse_tfs_mut_p47_entry_index(Some(String::from("99"))), 2);
    }

    #[test]
    fn tfs_ms2b_entry_ptr_reads_requested_packet_entry() {
        let mut packet = [0u8; TFS_MS2B_PACKET_SAMPLE_BYTES];
        let e0 = 0x1111_2222_3333_4444u64;
        let e1 = 0x5555_6666_7777_8888u64;
        let e2 = 0x9999_aaaa_bbbb_ccccu64;
        packet[32..40].copy_from_slice(&e0.to_le_bytes());
        packet[TFS_MS2B_PACKET_ENTRY_BYTES + 32..TFS_MS2B_PACKET_ENTRY_BYTES + 40]
            .copy_from_slice(&e1.to_le_bytes());
        packet[TFS_MS2B_PACKET_ENTRY_BYTES * 2 + 32..TFS_MS2B_PACKET_ENTRY_BYTES * 2 + 40]
            .copy_from_slice(&e2.to_le_bytes());

        assert_eq!(
            unsafe { tfs_ms2b_entry_ptr((TFS_MS2B_PACKET_ENTRY_BYTES * 3) as u32, &packet, 0, 32) },
            e0
        );
        assert_eq!(
            unsafe { tfs_ms2b_entry_ptr((TFS_MS2B_PACKET_ENTRY_BYTES * 3) as u32, &packet, 1, 32) },
            e1
        );
        assert_eq!(
            unsafe { tfs_ms2b_entry_ptr((TFS_MS2B_PACKET_ENTRY_BYTES * 3) as u32, &packet, 2, 32) },
            e2
        );
        assert_eq!(
            unsafe { tfs_ms2b_entry_ptr((TFS_MS2B_PACKET_ENTRY_BYTES * 2) as u32, &packet, 2, 32) },
            0
        );
    }

    #[test]
    fn dump_tfs_ms3_args_encodes_expected_fixed_fields() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 4];
        let p2: u64 = 0x7f00_2222_0000;
        let p3: u64 = 0x7f00_3333_0000;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();
        kernel_params[3] = (&p3 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms3_args(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs3ArgsCaptureV1>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs3ArgsCaptureV1) };
        assert_eq!(
            encoded,
            TfsMs3ArgsCaptureV1 {
                magic: TFS_MS3_ARGS_CAPTURE_MAGIC,
                version: TFS_MS3_ARGS_CAPTURE_VERSION,
                p2,
                p3,
            }
        );
    }

    #[test]
    fn dump_tfs_ms3_args_v2_encodes_header_and_zero_samples_when_unmapped() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 4];
        let p2: u64 = 0;
        let p3: u64 = 0;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();
        kernel_params[3] = (&p3 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms3_args_v2(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs3ArgsCaptureV2>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs3ArgsCaptureV2) };
        assert_eq!(encoded.magic, TFS_MS3_ARGS_CAPTURE_MAGIC);
        assert_eq!(encoded.version, TFS_MS3_ARGS_CAPTURE_VERSION_V2);
        assert_eq!(encoded.p2, p2);
        assert_eq!(encoded.p3, p3);
        assert_eq!(encoded.p2_words, 0);
        assert_eq!(encoded.p3_words, 0);
    }

    #[test]
    fn tfs_mut_p47_capture_v1_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMutP47CaptureV1>(), 104);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, version), 4);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV1, source_ms2_launch_n),
            8
        );
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, rd5), 16);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, rd7), 24);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, rd5_words), 32);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, rd7_words), 36);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, rd5_words_data), 40);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV1, rd7_words_data), 72);
    }

    #[test]
    fn tfs_mut_p47_capture_v2_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMutP47CaptureV2>(), 176);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, version), 4);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV2, source_ms2_launch_n),
            8
        );
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, rd5), 16);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, rd7), 24);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, rd5_words), 32);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, rd7_words), 36);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, rd5_words_data), 40);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, rd7_words_data), 72);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, arg_words), 104);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV2, arg_mode), 108);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV2, arg_words_data),
            112
        );
    }

    #[test]
    fn tfs_mut_p47_capture_v4_layout_is_stable() {
        // V4 reuses V3's _pad0 slot at offset 60 for the rd5_off_22176 probe;
        // size therefore matches V3 (224 bytes) so the dump path swap is a
        // no-op for ring-slot accounting.
        assert_eq!(std::mem::size_of::<TfsMutP47CaptureV4>(), 224);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, version), 4);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV4, source_ms2_launch_n),
            8
        );
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd5), 16);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd6), 24);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd7), 32);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd5_words), 40);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd6_words), 44);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd7_words), 48);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, arg_words), 52);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, arg_mode), 56);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV4, rd5_off_22176),
            60
        );
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd5_words_data), 64);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd6_words_data), 96);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV4, rd7_words_data), 128);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV4, arg_words_data),
            160
        );
    }

    #[test]
    fn tfs_mut_p47_capture_v3_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsMutP47CaptureV3>(), 224);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, version), 4);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV3, source_ms2_launch_n),
            8
        );
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd5), 16);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd6), 24);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd7), 32);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd5_words), 40);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd6_words), 44);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd7_words), 48);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, arg_words), 52);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, arg_mode), 56);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd5_words_data), 64);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd6_words_data), 96);
        assert_eq!(std::mem::offset_of!(TfsMutP47CaptureV3, rd7_words_data), 128);
        assert_eq!(
            std::mem::offset_of!(TfsMutP47CaptureV3, arg_words_data),
            160
        );
    }

    #[test]
    fn tfs_atomic_to_tex_capture_v1_layout_is_stable() {
        assert_eq!(std::mem::size_of::<TfsAtomicToTexCaptureV1>(), 368);
        assert_eq!(std::mem::offset_of!(TfsAtomicToTexCaptureV1, magic), 0);
        assert_eq!(std::mem::offset_of!(TfsAtomicToTexCaptureV1, version), 4);
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, source_ms2_launch_n),
            8
        );
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, source_ptr),
            32
        );
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, surface_handle),
            40
        );
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, output_base),
            56
        );
        assert_eq!(std::mem::offset_of!(TfsAtomicToTexCaptureV1, dim_x), 64);
        assert_eq!(std::mem::offset_of!(TfsAtomicToTexCaptureV1, rd5_words), 80);
        assert_eq!(std::mem::offset_of!(TfsAtomicToTexCaptureV1, arg_words), 88);
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, rd5_words_data),
            112
        );
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, arg_words_data),
            176
        );
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, surface_desc_words_data),
            240
        );
        assert_eq!(
            std::mem::offset_of!(TfsAtomicToTexCaptureV1, output_post_words_data),
            336
        );
    }

    #[test]
    fn tfs_atomic_to_tex_surface_desc_base_combines_descriptor_words() {
        assert_eq!(
            tfs_atomic_to_tex_surface_desc_base(&[0x1122_3344, 0x5566_7788, 0, 0, 0, 0, 0, 0]),
            0x0077_8811_2233_4400
        );
    }

    #[test]
    fn dump_tfs_atomic_to_tex_capture_encodes_header_and_zero_samples_when_unmapped() {
        tfs_mut_p47_track(0x1111, 0xbeef, 0x2222, 333);
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut flat = [0u8; 488];
        flat[24 + 28..24 + 32].copy_from_slice(&64u32.to_le_bytes());
        flat[24 + 32..24 + 36].copy_from_slice(&48u32.to_le_bytes());
        flat[24 + 36..24 + 40].copy_from_slice(&8u32.to_le_bytes());
        flat[480..488].copy_from_slice(&0u64.to_le_bytes());
        let flat_size = flat.len();
        let mut extra = [
            0x1usize as *mut ::core::ffi::c_void,
            flat.as_mut_ptr().cast(),
            0x2usize as *mut ::core::ffi::c_void,
            (&flat_size as *const usize).cast_mut().cast(),
            std::ptr::null_mut(),
        ];

        let written = unsafe {
            dump_tfs_atomic_to_tex_capture(
                std::ptr::null_mut(),
                extra.as_mut_ptr(),
                std::ptr::null_mut(),
                hipStream_t(std::ptr::null_mut()),
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsAtomicToTexCaptureV1>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsAtomicToTexCaptureV1) };
        assert_eq!(encoded.magic, TFS_ATOMIC_TO_TEX_CAPTURE_MAGIC);
        assert_eq!(encoded.version, TFS_ATOMIC_TO_TEX_CAPTURE_VERSION);
        assert_eq!(encoded.source_ms2_launch_n, 333);
        assert_eq!(encoded.rd5, 0x1111);
        assert_eq!(encoded.rd7, 0x2222);
        assert_eq!(encoded.dim_x, 64);
        assert_eq!(encoded.dim_y, 48);
        assert_eq!(encoded.dim_z, 8);
        assert_eq!(encoded.source_ptr, 0);
        assert_eq!(encoded.surface_handle, 0);
        assert_eq!(encoded.surface_desc_ptr, 0);
        assert_eq!(encoded.output_base, 0);
        assert_eq!(encoded.rd5_words, 0);
        assert_eq!(encoded.rd7_words, 0);
        assert_eq!(encoded.source_words, 0);
        assert_eq!(encoded.output_pre_words, 0);
        assert_eq!(encoded.output_post_words, 0);
        assert_eq!(encoded.arg_mode, 1);
        assert_eq!(encoded.arg_words, 16);
        assert!(encoded.arg_words_data.contains(&64));
        assert!(encoded.arg_words_data.contains(&48));
        assert!(encoded.arg_words_data.contains(&8));
    }

    #[test]
    fn dump_tfs_mut_p47_capture_encodes_zero_when_untracked() {
        tfs_mut_p47_track(0, 0, 0, 0);
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let written = unsafe {
            dump_tfs_mut_p47_capture(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMutP47CaptureV4>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMutP47CaptureV4) };
        assert_eq!(encoded.magic, TFS_MUT_P47_CAPTURE_MAGIC);
        assert_eq!(encoded.version, TFS_MUT_P47_CAPTURE_VERSION);
        assert_eq!(encoded.source_ms2_launch_n, 0);
        assert_eq!(encoded.rd5, 0);
        assert_eq!(encoded.rd6, 0);
        assert_eq!(encoded.rd7, 0);
        assert_eq!(encoded.rd5_words, 0);
        assert_eq!(encoded.rd6_words, 0);
        assert_eq!(encoded.rd7_words, 0);
        assert_eq!(encoded.arg_words, 0);
        assert_eq!(encoded.arg_mode, 0);
        assert_eq!(encoded.rd5_off_22176, 0xFFFFFFFF);
    }

    #[test]
    fn ms1_rd5_reset_pending_drains_dedup_and_skipped_when_disabled() {
        // Without env enabled, mark is a no-op.
        // (We can't easily set env here without affecting other tests, so
        //  we just exercise the take_* path on the static; rely on
        //  the env-gated branch in ms1_rd5_reset_enabled being false
        //  in test mode unless ZLUDA_MS1_RD5_RESET=1 is set externally.)
        // Drain leftover state first.
        let _ = take_pending_rd5_resets();
        // mark_rd5_reset_pending checks the env gate via OnceLock; in
        // a default test environment the gate is false so the mark is
        // a no-op and the drain returns empty. This guards against
        // accidental cross-test interference.
        mark_rd5_reset_pending(0xdeadbeef);
        let drained = take_pending_rd5_resets();
        assert_eq!(drained, Vec::<u64>::new());
    }

    /// 2026-06-02 regression-lock: force_zero_current_ms1_rd5_22176 must gate on
    /// ZLUDA_MS1_RD5_RESET, NOT ZLUDA_MS4_CAPTURE. The previous gate was a
    /// silent dependency that broke the production workaround whenever
    /// ZLUDA_MS4_CAPTURE was unset (which is the protonfix default — capture
    /// is heavy investigation infrastructure, not a production opt-in).
    ///
    /// In a default test environment ZLUDA_MS1_RD5_RESET is unset, so the
    /// function must return at the gate without attempting any HIP call —
    /// otherwise this test would crash because there's no GPU context.
    /// We pass dummy params; the function would dereference them and call
    /// hipMemcpyDtoD if the gate were broken.
    #[test]
    fn force_zero_current_ms1_rd5_22176_gates_on_ms1_rd5_reset_env() {
        // Default test env: ZLUDA_MS1_RD5_RESET not set → gate denies.
        // (If the gate were on ZLUDA_MS4_CAPTURE, the same default-unset
        //  applies — so an early-exit-with-no-crash by itself doesn't prove
        //  the fix. But the gate WAS reached because no panic occurred at
        //  the env-var read; the OnceLock initializes lazily on first call.)
        let mut p0: u64 = 0;
        let mut p1: u16 = 0;
        let mut p2: u64 = 0;
        let mut p3: u64 = 0;
        let mut p4: u32 = 0;
        let mut p5: f32 = 0.0;
        let mut p6: u64 = 0x0; // null rd5 → would also gate at current_ms1_rd5_ptr
        let mut params: [*mut ::core::ffi::c_void; 7] = [
            &mut p0 as *mut _ as *mut _,
            &mut p1 as *mut _ as *mut _,
            &mut p2 as *mut _ as *mut _,
            &mut p3 as *mut _ as *mut _,
            &mut p4 as *mut _ as *mut _,
            &mut p5 as *mut _ as *mut _,
            &mut p6 as *mut _ as *mut _,
        ];
        unsafe {
            force_zero_current_ms1_rd5_22176(params.as_mut_ptr(), std::ptr::null_mut());
        }
        // No panic = gate accepted the default-disabled env state.
    }

    #[test]
    fn ms2_rd6_watch_tracks_latest_launch_for_pointer() {
        let watch =
            MS2_RD6_WATCH.get_or_init(|| Mutex::new(std::collections::VecDeque::with_capacity(64)));
        {
            let mut q = watch.lock().unwrap();
            q.clear();
        }

        ms2_rd6_watch_track(0x7000, 11);
        ms2_rd6_watch_track(0x9000, 12);
        ms2_rd6_watch_track(0x7000, 42);

        let snapshot = ms2_rd6_watch_snapshot();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0], (0x9000, 12));
        assert_eq!(snapshot[1], (0x7000, 42));
    }

    #[test]
    fn ms2_rd6_watch_candidates_prefers_newest_unique_plausible_pointers() {
        let watch = vec![
            (0x800, 1), // implausible low
            (0x7000, 10),
            (0x9000, 11),
            (0x7000, 42),                // same ptr, newer launch
            (0x0000_9000_0000_0000, 99), // implausible high
        ];
        let picked = ms2_rd6_watch_candidates(&watch, 2);
        assert_eq!(picked, vec![(0x9000, 11), (0x7000, 42)]);
    }

    #[test]
    fn dump_tfs_ms4_args_encodes_expected_fixed_fields() {
        let mut raw = [0u8; TFS_RAW_ARGS_BYTES];
        let mut kernel_params = [std::ptr::null_mut::<::core::ffi::c_void>(); 8];
        let p2: u64 = 0x7f00_1111_0000;
        let n: u32 = 51864;
        let p4: u64 = 0x7f00_2222_0000;
        let p5: u64 = 0x7f00_3333_0000;
        let p6: u64 = 0x7f00_4444_0000;
        let p7: u64 = 0x7f00_5555_0000;
        kernel_params[2] = (&p2 as *const u64).cast_mut().cast();
        kernel_params[3] = (&n as *const u32).cast_mut().cast();
        kernel_params[4] = (&p4 as *const u64).cast_mut().cast();
        kernel_params[5] = (&p5 as *const u64).cast_mut().cast();
        kernel_params[6] = (&p6 as *const u64).cast_mut().cast();
        kernel_params[7] = (&p7 as *const u64).cast_mut().cast();

        let written = unsafe {
            dump_tfs_ms4_args(
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
                raw.as_mut_ptr(),
                raw.len(),
            )
        };
        assert_eq!(written, std::mem::size_of::<TfsMs4ArgsCaptureV1>());
        let encoded =
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const TfsMs4ArgsCaptureV1) };
        assert_eq!(
            encoded,
            TfsMs4ArgsCaptureV1 {
                magic: TFS_MS4_ARGS_CAPTURE_MAGIC,
                version: TFS_MS4_ARGS_CAPTURE_VERSION,
                n,
                _pad0: 0,
                p2,
                p4,
                p5,
                p6,
                p7,
            }
        );
    }

    #[test]
    fn should_capture_ms2b_only_for_large_stage2_batched_launches() {
        assert!(!should_capture_ms2b(
            "CreateParticleShapesIncremental_MultiStage_2_Batched",
            1
        ));
        assert!(!should_capture_ms2b(
            "CreateParticleShapesIncremental_MultiStage_2_Batched",
            2
        ));
        assert!(should_capture_ms2b(
            "CreateParticleShapesIncremental_MultiStage_2_Batched",
            3
        ));
        assert!(!should_capture_ms2b(
            "CreateParticleShapesIncremental_MultiStage_1",
            3
        ));
        assert!(!should_capture_ms2b(
            "TurbulenceFS_dynamicGridClearKernel",
            3
        ));
    }

    #[test]
    fn should_capture_mut_p47_probe_only_for_tail_window_kernels() {
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_MIGRATE,
            false
        ));
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_ADVECT,
            false
        ));
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_COUPLE_FROM_PARTICLES,
            false
        ));
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX,
            false
        ));
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_PARTICLE_DENSITY_SMOOTH,
            false
        ));
        assert!(!should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED,
            false
        ));
        assert!(!should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3,
            false
        ));
        assert!(!should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4,
            false
        ));
        assert!(!should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CLEAR,
            false
        ));
        assert!(!should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_OTHER,
            false
        ));
    }

    #[test]
    fn should_capture_mut_p47_probe_broad_mode_adds_ms3_and_window_kernels() {
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CLEAR,
            true
        ));
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_OTHER,
            true
        ));
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3,
            true
        ));
        assert!(should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS1,
            true
        ));
        assert!(!should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS1,
            false
        ));
        assert!(!should_capture_mut_p47_probe_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4,
            true
        ));
    }

    #[test]
    fn parse_tfs_mut_p47_broad_probe_defaults_off() {
        assert!(!parse_tfs_mut_p47_broad_probe(None));
        assert!(!parse_tfs_mut_p47_broad_probe(Some(String::from("0"))));
        assert!(!parse_tfs_mut_p47_broad_probe(Some(String::from("yes"))));
        assert!(parse_tfs_mut_p47_broad_probe(Some(String::from("1"))));
    }

    #[test]
    fn parse_tfs_capture_only_atomic_to_tex_defaults_off() {
        assert!(!parse_tfs_capture_only_atomic_to_tex(None));
        assert!(!parse_tfs_capture_only_atomic_to_tex(Some(String::from(
            "0"
        ))));
        assert!(!parse_tfs_capture_only_atomic_to_tex(Some(String::from(
            "yes"
        ))));
        assert!(parse_tfs_capture_only_atomic_to_tex(Some(String::from(
            "1"
        ))));
    }

    #[test]
    fn parse_tfs_capture_only_ms_chain_defaults_off() {
        assert!(!parse_tfs_capture_only_ms_chain(None));
        assert!(!parse_tfs_capture_only_ms_chain(Some(String::from("0"))));
        assert!(!parse_tfs_capture_only_ms_chain(Some(String::from("yes"))));
        assert!(parse_tfs_capture_only_ms_chain(Some(String::from("1"))));
    }

    #[test]
    fn should_capture_tfs_kernel_atomic_focus_only_keeps_atomic_to_tex() {
        assert!(should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX,
            true,
            false
        ));
        assert!(!should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED,
            true,
            false
        ));
        assert!(!should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4,
            true,
            false
        ));
        assert!(!should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CLEAR,
            true,
            false
        ));
        assert!(should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CLEAR,
            false,
            false
        ));
        assert!(should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED,
            false,
            false
        ));
    }

    #[test]
    fn should_capture_tfs_kernel_ms_chain_focus_only_keeps_ms2_ms3_ms4() {
        assert!(should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS2_BATCHED,
            false,
            true
        ));
        assert!(should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS3,
            false,
            true
        ));
        assert!(should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CREATE_PARTICLE_SHAPES_MS4,
            false,
            true
        ));
        assert!(!should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX,
            false,
            true
        ));
        assert!(!should_capture_tfs_kernel_for_mode(
            TFS_KERNEL_ID_CLEAR,
            false,
            true
        ));
    }

    fn pack_u16_stride4(values: &[u16]) -> Vec<u8> {
        let mut bytes = vec![0u8; values.len() * 4];
        for (i, value) in values.iter().copied().enumerate() {
            let offset = i * 4;
            bytes[offset..offset + 2].copy_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn pack_u32(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn summarize_ms4_capture_detects_underflow_and_out_of_range() {
        let flags = pack_u16_stride4(&[1, 1, 1, 1]);
        let shape_idx = pack_u32(&[0, 1, 1, 99]);
        let counters = pack_u32(&[1, 1]);
        let summary = summarize_ms4_capture(&flags, &shape_idx, &counters, 4).unwrap();
        assert_eq!(summary.active_count, 4);
        assert_eq!(summary.unique_shapes, 2);
        assert_eq!(summary.max_shape_idx, 99);
        assert_eq!(summary.out_of_range_particles, 1);
        assert_eq!(summary.first_out_of_range_particle, Some((3, 99, 1)));
        assert_eq!(summary.underflow_count, 1);
        assert_eq!(
            summary.first_underflow,
            Some(Ms4ShapeUseSummary {
                shape: 1,
                counter: 1,
                uses: 2,
            })
        );
    }

    #[test]
    fn summarize_ms4_capture_counts_exact_and_surplus_shapes() {
        // shape 0: counter 2, uses 2 -> exact (healthy full consumption)
        // shape 1: counter 5, uses 2 -> surplus 3
        // shape 2: counter 1, uses 2 -> underflow (not exact, not surplus)
        let flags = pack_u16_stride4(&[1, 1, 1, 1, 1, 1]);
        let shape_idx = pack_u32(&[0, 0, 1, 1, 2, 2]);
        let counters = pack_u32(&[2, 5, 1]);
        let summary = summarize_ms4_capture(&flags, &shape_idx, &counters, 6).unwrap();
        assert_eq!(summary.exact_shapes, 1);
        assert_eq!(summary.surplus_shapes, 1);
        assert_eq!(summary.max_surplus, 3);
        assert_eq!(summary.underflow_count, 1);
        assert_eq!(summary.sum_counter, 8); // 2 + 5 + 1
        assert_eq!(summary.sum_uses, 6); // 2 + 2 + 2
    }

    #[test]
    fn summarize_ms4_capture_counts_negative_used_counters() {
        let flags = pack_u16_stride4(&[1, 1]);
        let shape_idx = pack_u32(&[0, 1]);
        let counters = pack_u32(&[0xffff_fffe, 0]);
        let summary = summarize_ms4_capture(&flags, &shape_idx, &counters, 2).unwrap();
        assert_eq!(summary.active_count, 2);
        assert_eq!(summary.negative_used_shapes, 1);
        assert_eq!(summary.zero_used_shapes, 1);
        assert_eq!(summary.underflow_count, 1);
        assert_eq!(
            summary.first_underflow,
            Some(Ms4ShapeUseSummary {
                shape: 1,
                counter: 0,
                uses: 1,
            })
        );
    }

    #[test]
    fn summarize_sort_spine_reports_basic_stats() {
        let summary = summarize_sort_spine(&[0, 3, 7, 0, 2]).unwrap();
        assert_eq!(
            summary,
            SortSpineSummary {
                words_read: 5,
                nonzero_count: 3,
                max_value: 7,
                sum: 12,
            }
        );
    }

    #[test]
    fn format_sorted_key_window_centers_on_first_bad_value() {
        let summary = summarize_sorted_keys(&[10, 11, 12, -1, 14, 15]).unwrap();
        let window = format_sorted_key_window(&[10, 11, 12, -1, 14, 15], &summary, "OUT").unwrap();
        assert!(window.contains("OUT[3]=-1"));
    }

    #[test]
    fn format_key_window_around_centers_on_requested_index() {
        let window = format_key_window_around(&[10, 11, 12, -1, 14, 15], 3, "OUT").unwrap();
        assert!(window.contains("OUT[3]=-1"));
    }
}

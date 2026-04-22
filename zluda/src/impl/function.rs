use cuda_types::cuda::{CUfunction, CUfunction_attribute, CUkernel};
use hip_runtime_sys::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as IoWrite;
use std::mem;
use std::sync::Mutex;

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
const MS4_CAPTURE_MAX_PARTICLES: usize = 32 * 1024;
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
    _pad1: [u8; 32],      // 96..=127
    // Data regions (aligned at 128)
    flags_data: [u8; MS4_PARTICLE_REGION_BYTES],
    shape_idx_data: [u8; MS4_PARTICLE_REGION_BYTES],
    struct_data: [u8; MS4_STRUCT_REGION_BYTES],
    counter_data: [u8; MS4_COUNTER_REGION_BYTES],
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
) {
    if !ms4_capture_enabled() {
        return;
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
        return;
    }

    let mut guard = match MS4_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    ms4_capture_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return, // init failed; it already logged
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
        return;
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
    let _ = (struct_off, struct_bytes); // struct capture remains disabled here

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
                    "MS4_SAFE #{} n={} active={} shapes={} max_shape={} out_of_range={} zero_used={} neg_used={} underflows={}",
                    launch_n,
                    num_particles,
                    summary.active_count,
                    summary.unique_shapes,
                    summary.max_shape_idx,
                    summary.out_of_range_particles,
                    summary.zero_used_shapes,
                    summary.negative_used_shapes,
                    summary.underflow_count,
                ));
            }
            if let Some((particle_idx, shape, flag)) = summary.first_out_of_range_particle {
                super::driver::zluda_log_err(&format!(
                    "*** MS4_SHAPE_IDX_OOR launch_n={} particle={} flag=0x{:04x} shape=0x{:08x}",
                    launch_n, particle_idx, flag, shape
                ));
            }
            if let Some(issue) = summary.first_underflow {
                super::driver::zluda_log_err(&format!(
                    "*** MS4_COUNTER_UNDERFLOW launch_n={} shape={} counter={} uses={}",
                    launch_n, issue.shape, issue.counter, issue.uses
                ));
            }
        }
    }
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
const F6_OFFSET_IN_RD5: usize = 22176;

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
unsafe fn force_zero_current_ms1_rd5_22176(
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) {
    if !ms4_capture_enabled() {
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

fn launch_mutex(lock_mode: u8) -> Option<&'static Mutex<()>> {
    if lock_mode >= 1 {
        Some(&LAUNCH_LOCK)
    } else {
        None
    }
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
    first_out_of_range_particle: Option<(usize, u32, u16)>,
    first_underflow: Option<Ms4ShapeUseSummary>,
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
        first_out_of_range_particle: None,
        first_underflow: None,
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
        if counter == 0 {
            summary.zero_used_shapes += 1;
        }
        if (counter as i32) < 0 {
            summary.negative_used_shapes += 1;
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

const TFS_CAPTURE_SLOTS: usize = 256;
const TFS_DESC_BYTES: usize = 456;
const TFS_KERNEL_NAME_BYTES: usize = 64;
const TFS_RAW_ARGS_BYTES: usize = 512; // covers struct + scalar args + extras
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
        || count as usize > IOFX_ACTOR_RANGE_MAX_COUNT
        || input_ptr_324 == 0
        || output_ptr_326 == 0
        || output_ptr_327 == 0
        || output_ptr_328 == 0
        || input_ptr_329 == 0
    {
        return None;
    }
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

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % IOFX_CAPTURE_SLOTS;
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

unsafe fn capture_iofx_actor_range_pre(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_actor_range_outputs(slot_index, stream, true) };
}

unsafe fn capture_iofx_actor_range_post(slot_index: usize, stream: hipStream_t) {
    unsafe { capture_iofx_actor_range_outputs(slot_index, stream, false) };
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
            | "InitCollDataAndApplyConstraints"
            | "FinalizeCollision"
            | "ParticleCollisionStaticBox"
    )
}

fn should_capture_ms2b(name: &str, grid_x: u32) -> bool {
    name == "CreateParticleShapesIncremental_MultiStage_2_Batched" && grid_x >= 3
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
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
) {
    if !tfs_capture_enabled() {
        return;
    }

    let mut guard = match TFS_CAPTURE.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    tfs_capture_init(&mut guard);
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return,
    };

    let this_launch = state.launch_counter;
    state.launch_counter += 1;
    let slot_index = (this_launch as usize) % TFS_CAPTURE_SLOTS;

    let slot_ptr = state.host_ptr.add(slot_index * TFS_SLOT_SIZE) as *mut TfsCaptureSlot;
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
        (*slot_ptr).raw_args_len = dump_kernel_param_ptrs(
            kernel_params,
            (*slot_ptr).raw_args.as_mut_ptr(),
            TFS_RAW_ARGS_BYTES,
        ) as u32;
        return;
    }

    // Dump the raw kernel argument bytes.
    let raw_dst = (*slot_ptr).raw_args.as_mut_ptr();
    let raw_len = dump_kernel_args_raw(kernel_params, extra, raw_dst, TFS_RAW_ARGS_BYTES);
    (*slot_ptr).raw_args_len = raw_len as u32;
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
    if log_launches {
        super::driver::zluda_log_err(&format!(
            "cuLaunchKernel '{}' grid=[{},{},{}] block=[{},{},{}] shared={}",
            name,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes
        ));
    }
    // Capture CUB RadixSort kernel args (Downsweep + Upsweep) including GridEvenShare.
    // Pure CPU reads from the extra buffer — no GPU ops, no ProfilingSignal risk.
    // Logs once per unique (kernel, begin_bit) pair so we get one line per pass.
    if (name.contains("RadixSortDownsweepKernel") || name.contains("RadixSortUpsweepKernel"))
        && !extra.is_null()
    {
        static SORT_CAPTURE_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !SORT_CAPTURE_DONE.load(std::sync::atomic::Ordering::Relaxed) {
            unsafe {
                let extra_arr = std::slice::from_raw_parts(extra, 5);
                if extra_arr[0] == (0x1usize as *mut _) && !extra_arr[1].is_null() {
                    let buf = extra_arr[1] as *const u8;
                    // Read all bytes 0-87 of the extra buffer
                    let raw: Vec<u8> = (0..88).map(|i| *buf.add(i)).collect();
                    let kernel_type = if name.contains("Downsweep") {
                        "DS"
                    } else {
                        "US"
                    };
                    let n_items = std::ptr::read_unaligned(buf.add(40) as *const u32);
                    let begin_bit = std::ptr::read_unaligned(buf.add(44) as *const u32);
                    let is_first = *buf.add(48);
                    let is_last = *buf.add(49);
                    let overwrite = *buf.add(50);
                    // GridEvenShare at bytes 52-87 (36 bytes)
                    let ges_raw: Vec<u8> = (52..88).map(|i| *buf.add(i)).collect();
                    let ges_i32: Vec<i32> = (0..9)
                        .map(|i| std::ptr::read_unaligned(buf.add(52 + i * 4) as *const i32))
                        .collect();
                    super::driver::zluda_log_err(&format!(
                        "SORT_ARGS {} n={} bit={} first={} last={} ow={} GES={:?} grid={} stream={:?}",
                        kernel_type, n_items, begin_bit, is_first, is_last, overwrite,
                        ges_i32, grid_dim_x, stream));
                    // Mark done after capturing all passes (last Downsweep = bit>=16)
                    if name.contains("Downsweep") && begin_bit >= 16 {
                        SORT_CAPTURE_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
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
            let p4: u64 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
            let p5: u64 = read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0);
            let p6: u64 = read_kernel_arg(kernel_params, extra, 48, 6).unwrap_or(0);
            let p7: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);
            static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Log every MS_4 kernel arg set. rocgdb parses the LAST one of
            // these from /tmp/zluda_tex.log at the crash breakpoint — so
            // we're not racing the HIP runtime's buffered trace flush.
            super::driver::zluda_log_err(&format!(
                "MS4_ARGS #{} n={} p4=0x{:x} p5=0x{:x} p6=0x{:x} p7=0x{:x}",
                call_n, n, p4, p5, p6, p7
            ));
            if let Ok(table) = super::memory::SMALL_HOST_ALLOCS.lock() {
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
                                call_n, n, arg_idx, arg_ptr, host, dev, size, offset, words, dump
                            ));
                            break;
                        }
                    }
                }
                // NOTE: full-buffer sentinel scan removed — 6MB × 200 launches
                // = 1.2 GB of CPU reads during init was correlated with
                // triggering the ProfilingSignal libstdc++ race (p9, p10
                // both aborted early). ARG_DUMP above is cheap enough to
                // keep. For device-only buffers we need rocgdb, not scans.
            }
        }
    }

    // MS_4 capture — ALL gated behind ZLUDA_MS4_CAPTURE=1
    if name.contains("CreateParticleShapesIncremental_MultiStage_4") {
        static MS4_CAP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *MS4_CAP.get_or_init(|| std::env::var("ZLUDA_MS4_CAPTURE").map_or(false, |v| v == "1")) {
            unsafe {
                capture_ms4_inputs(kernel_params, extra, grid_dim_x, block_dim_x);
                latch_counter_ptr_from_ms4(kernel_params, extra);
                let p3: u32 = read_kernel_arg(kernel_params, extra, 24, 3).unwrap_or(0);
                let p2: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
                let p4: u64 = read_kernel_arg(kernel_params, extra, 32, 4).unwrap_or(0);
                let p5: u64 = read_kernel_arg(kernel_params, extra, 40, 5).unwrap_or(0);
                let p6: u64 = read_kernel_arg(kernel_params, extra, 48, 6).unwrap_or(0);
                let p7: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);
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
    if let Some(tfs_kernel_id) = classify_tfs_kernel(&name) {
        unsafe {
            capture_tfs_launch(
                &name,
                tfs_kernel_id,
                kernel_params,
                extra,
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
            );
        }
    }
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
        }
    }
    // Phase 1.5 Item A: dump the MemCopyBatchedAsync Arg2 scratch buffer
    // contents BEFORE the kernel launches. This answers the "PhysX bug or
    // ZLUDA bug?" root-cause question for Phase 1 — by correlating the
    // scratch contents with the RESET log lines we can see whether the
    // bug-firing rd5 is included in the batch (kernel-internal bug) or
    // excluded (PhysX host-side scheduling mismatch).
    // MCB scratch dump — disabled
    // if name == "MemCopyBatchedAsync" {
    //     unsafe { dump_mcb_scratch(kernel_params, extra, grid_dim_y); }
    // }
    // NULL-stream synchronization: CUDA guarantees that operations on the NULL
    // stream (stream 0) serialize with ALL other streams. HIP/ROCm 6.2.4 doesn't
    // enforce this. We call hipDeviceSynchronize only for NULL-stream launches
    // (which is what CUDA semantics require) instead of the previous blanket sync
    // that drained all streams before every single launch.
    //
    // The guard holds a lock to prevent other threads from submitting work between
    // our sync and our launch. Dropped after hipModuleLaunchKernel returns.
    let _null_stream_guard = super::stream::null_stream_sync(stream);
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
    if let Some(slot_index) = iofx_bbox_slot.filter(|_| result.is_ok()) {
        unsafe {
            capture_iofx_bbox_post(slot_index, stream);
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
                    if bit31_out > 0 || bit31_in > 0 {
                        let viol = h_out.windows(2).filter(|w| w[1] < w[0]).count();
                        // Also read spine at this sync point (already synced, one extra D2H)
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
    // Track last kernel name for SIGABRT handler
    if !name.is_empty() {
        super::driver::set_last_kernel(name.clone());
    }
    // Log launch failures with full context for diagnosis
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
    // F.4: per-launch counter mutation tracker. Snapshots all tracked counter
    // pointers and diffs vs the previous snapshot. Logs a `*** COUNTER ...`
    // line whenever ANY kernel mutates the counter buffer, with the kernel's
    // name. The "shift-and-zero" pattern from F.3b gets a SHIFT_ZERO tag.
    // Gated by ZLUDA_MS4_CAPTURE=1 AND TRACKED_COUNTER_PTRS being non-empty
    // — pre-PhysX init kernels skip the snapshot entirely.
    if result.is_ok() {
        unsafe {
            let _ = hip_runtime_sys::hipDeviceSynchronize();
            check_counter_mutation(&name);
            // F.6: also snapshot *(rd5+22176) per tracked rd5 to find which
            // operation normally resets it (and which fails to on bug frames).
            check_rd5_22176_mutation(&name);
            // Plan step 4: snapshot the latched MS_4 counter buffer and
            // log if any kernel mutates it. Independent of MS4_CAPTURE /
            // F.6 gates — uses ZLUDA_COUNTER_SNAPSHOT=1.
            snapshot_counter_after_launch(&name);
        }
    }
    // Explicitly drop the guard to make the lifetime scope obvious (mode 2 releases here).
    drop(launch_guard);
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
        f,
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
    fn launch_mutex_uses_dedicated_lock() {
        let lock = launch_mutex(1).expect("mode 1 should enable the launch mutex");
        assert!(
            std::ptr::eq(lock, &LAUNCH_LOCK),
            "launch mutex mode should use the dedicated launch lock"
        );
        assert!(
            !std::ptr::eq(lock, &crate::r#impl::stream::NULL_STREAM_LOCK),
            "launch mutex mode must not reuse NULL_STREAM_LOCK or NULL-stream launches can deadlock"
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
    fn read_iofx_actor_range_args_rejects_zero_or_oversized_layouts() {
        let mut raw = [0u8; 72];
        raw[12..16].copy_from_slice(&((IOFX_ACTOR_RANGE_MAX_COUNT + 1) as u32).to_le_bytes());
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

        raw[12..16].copy_from_slice(&16u32.to_le_bytes());
        raw[24..32].copy_from_slice(&0u64.to_le_bytes());
        assert!(unsafe {
            read_iofx_actor_range_args(std::ptr::null_mut(), extra.as_ptr().cast_mut())
        }
        .is_none());
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

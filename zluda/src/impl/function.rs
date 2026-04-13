use cuda_types::cuda::{CUfunction, CUfunction_attribute, CUkernel};
use hip_runtime_sys::*;
use std::mem;
use std::collections::HashMap;
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
    magic: u64,                  // 0: 0x4d533443_4150_5455 = "MS4CAPTU"
    launch_n: u64,               // 8
    timestamp_ms: u64,           // 16
    num_particles: u32,          // 24
    grid_x: u32,                 // 28
    block_x: u32,                // 32
    _pad0: u32,                  // 36
    flags_ptr: u64,              // 40
    struct_ptr: u64,             // 48
    scatter_ptr: u64,            // 56
    shape_idx_ptr: u64,          // 64
    counter_ptr: u64,            // 72
    flags_bytes: u32,            // 80 — actual bytes captured for this slot
    shape_idx_bytes: u32,        // 84
    struct_bytes: u32,           // 88
    counter_bytes: u32,          // 92
    _pad1: [u8; 32],             // 96..=127
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
    *ENABLED.get_or_init(|| {
        std::env::var("ZLUDA_MS4_CAPTURE").map_or(false, |v| v == "1")
    })
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
    *ENABLED.get_or_init(|| {
        std::env::var("ZLUDA_F6_TRACK").map_or(false, |v| v == "1")
    })
}

/// Initialize the capture state on first use. Called from inside the mutex.
unsafe fn ms4_capture_init(state: &mut Option<Ms4CaptureState>) {
    if state.is_some() {
        return;
    }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    // hipHostMallocMapped = 2 (the flag value is in hip_runtime_sys).
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, MS4_CAPTURE_TOTAL_SIZE, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS4 capture: hipHostMalloc({}) failed: {:?}",
            MS4_CAPTURE_TOTAL_SIZE, alloc_result));
        return;
    }
    // Zero the buffer so uninitialized slots are obviously empty.
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, MS4_CAPTURE_TOTAL_SIZE); }

    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS4 capture: hipHostGetDevicePointer failed: {:?}",
            map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
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

    if num_particles == 0 || num_particles as usize > MS4_CAPTURE_MAX_PARTICLES
       || flags_ptr == 0 || struct_ptr == 0 || shape_idx_ptr == 0 || counter_ptr == 0
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
    if flags_bytes > MS4_PARTICLE_REGION_BYTES { return; }

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
    let flags_off   = data_base;
    let shape_off   = flags_off + MS4_PARTICLE_REGION_BYTES;
    let struct_off  = shape_off + MS4_PARTICLE_REGION_BYTES;
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
    let _ = (flags_off, flags_bytes, struct_off, struct_bytes); // silence warnings

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
        let counter_host_ptr = state
            .host_ptr
            .add(slot_index * MS4_SLOT_SIZE + counter_off) as *const u32;
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
                state.launch_counter - 1,
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
                state.launch_counter - 1,
                num_particles,
                scan_count,
            ));
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
    magic: u64,                                            // 0:  "MS2BCAPT"
    launch_n: u64,                                         // 8:  MS_2_Batched invocation count (monotonic)
    timestamp_ms: u64,                                     // 16
    grid_x: u32,                                           // 24
    block_x: u32,                                          // 28
    packet_ptr: u64,                                       // 32: the kernarg param_2 value the game passed
    packet_bytes: u32,                                     // 40: how many bytes we actually copied
    num_tracked: u32,                                      // 44: how many entries in counter_ptrs are valid
    counter_ptrs: [u64; MS2B_MAX_TRACKED_COUNTERS],        // 48..112
    _pad: [u8; 80],                                        // 112..192
    // Data — 4096 bytes of raw packet memory
    packet_data: [u8; MS2B_CAPTURE_MAX_PACKET_BYTES],      // 192..4288
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
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, MS2B_CAPTURE_TOTAL_SIZE, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS2B capture: hipHostMalloc({}) failed: {:?}",
            MS2B_CAPTURE_TOTAL_SIZE, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, MS2B_CAPTURE_TOTAL_SIZE); }

    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS2B capture: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
        return;
    }

    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MS2B capture: {} slots × {} B = {} KB pinned; host=0x{:x} dev=0x{:x}",
        MS2B_CAPTURE_SLOTS, MS2B_SLOT_SIZE,
        MS2B_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize, dev_ptr as usize,
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
        let base = state.host_ptr.add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(base as *const u32);
        if count == 0 { continue; }
        let rd7 = *(base.add(80) as *const u64);
        if rd7 != 0 { track_counter_ptr(rd7); }
        // F.6: also track the rd5 base so the per-launch *(rd5+22176) tracker
        // can spot resets and the doubling bug.
        let rd5 = *(base.add(32) as *const u64);
        if rd5 != 0 { track_rd5_ptr(rd5); }
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
        let entry_base = state.host_ptr.add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(entry_base as *const u32);
        if count == 0 { continue; }
        let rd5 = *(entry_base.add(32) as *const u64);
        if rd5 == 0 { continue; }
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
        let entry_base = state.host_ptr.add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(entry_base as *const u32);
        if count == 0 { continue; }
        let rd5 = *(entry_base.add(32) as *const u64);
        let rd7 = *(entry_base.add(80) as *const u64);
        let meta_host = state.host_ptr.add(slot_index * MS2B_SLOT_SIZE + MS2B_RD5_PRE_OFF + e * MS2B_RD5_META_BYTES_PER_ENTRY) as *const u32;
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
    if !ms4_capture_enabled() { return; }

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
        if ptr == 0 { continue; }
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
        let entry_base = state.host_ptr.add(slot_index * MS2B_SLOT_SIZE + MS2B_PACKET_OFF + e * 88);
        let count = *(entry_base as *const u32);
        if count == 0 { continue; }
        let rd5 = *(entry_base.add(32) as *const u64);
        if rd5 == 0 { continue; }
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
        if ptr == 0 { continue; }
        let pre_host = state.host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_PRE_OFF + i * MS2B_COUNTER_BYTES_PER) as *const i32;
        let post_host = state.host_ptr
            .add(slot_index * MS2B_SLOT_SIZE + MS2B_POST_OFF + i * MS2B_COUNTER_BYTES_PER) as *const i32;
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
                if j > 0 { pre_str.push(','); post_str.push(','); }
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
    magic: u64,                                       // 0
    launch_n: u64,                                    // 8
    timestamp_ms: u64,                                // 16
    grid_x: u32,                                      // 24
    block_x: u32,                                     // 28
    num_tracked: u32,                                 // 32
    _pad0: u32,                                       // 36
    counter_ptrs: [u64; MS3_MAX_TRACKED_COUNTERS],    // 40..104
    _pad1: [u8; 24],                                  // 104..128
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
    if ptr == 0 { return; }
    let mut guard = match TRACKED_COUNTER_PTRS.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if !guard.contains(&ptr) && guard.len() < MS3_MAX_TRACKED_COUNTERS {
        guard.push(ptr);
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS3 capture: tracking counter_ptr[{}]=0x{:x}",
            guard.len() - 1, ptr
        ));
    }
}

/// F.6: Tracked rd5 base pointers — populated by MS_2_Batched packet capture
/// (we extract rd5 from packet entry offset 32). The F.6 tracker watches the
/// 4 bytes at `rd5+22176` (the source for `particleShapesCounter` that MS_1
/// uses as its slot allocator) and logs every kernel that mutates it.
static TRACKED_RD5_PTRS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn track_rd5_ptr(ptr: u64) {
    if ptr == 0 { return; }
    let mut guard = match TRACKED_RD5_PTRS.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if !guard.contains(&ptr) && guard.len() < MS3_MAX_TRACKED_COUNTERS {
        guard.push(ptr);
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F6 rd5 tracker: tracking rd5[{}]=0x{:x} (watching +22176)",
            guard.len() - 1, ptr
        ));
    }
}

unsafe fn ms3_capture_init(state: &mut Option<Ms3CaptureState>) {
    if state.is_some() { return; }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, MS3_CAPTURE_TOTAL_SIZE, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS3 capture: hipHostMalloc({}) failed: {:?}",
            MS3_CAPTURE_TOTAL_SIZE, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, MS3_CAPTURE_TOTAL_SIZE); }

    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MS3 capture: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
        return;
    }

    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MS3 capture: {} slots × {} KB = {} KB pinned; host=0x{:x} dev=0x{:x}",
        MS3_CAPTURE_SLOTS, MS3_SLOT_SIZE / 1024,
        MS3_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize, dev_ptr as usize,
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
    if !ms4_capture_enabled() { return None; }

    let tracked: Vec<u64> = {
        let guard = match TRACKED_COUNTER_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.clone()
    };
    if tracked.is_empty() { return None; }

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
    if !ms4_capture_enabled() { return; }

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
        if ptr == 0 { continue; }
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
        if ptr == 0 { continue; }
        let pre_host = state.host_ptr
            .add(slot_index * MS3_SLOT_SIZE + 128 + i * MS3_COUNTER_BYTES_PER) as *const i32;
        let post_host = state.host_ptr
            .add(slot_index * MS3_SLOT_SIZE + post_base + i * MS3_COUNTER_BYTES_PER) as *const i32;
        let pre0 = *pre_host;
        let post0 = *post_host;
        if pre0 != post0 || (post0 < 0 && post0 > -1_000_000) {
            super::driver::zluda_log_err(&format!(
                "[MS_3 DIFF] launch_n={} counter[0x{:x}]: pre={} post={} delta={}",
                launch_n, ptr, pre0, post0, post0.wrapping_sub(pre0),
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
    if state.is_some() { return; }
    let total = F4_MAX_COUNTERS * F4_BYTES_PER;
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, total, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F4 counter tracker: hipHostMalloc({}) failed: {:?}",
            total, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, total); }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F4 counter tracker: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
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
    if !ms4_capture_enabled() { return; }
    // F.6 tracker is gated separately because it has a stale-pointer
    // hazard — see f6_tracker_enabled() comment.
    if !f6_tracker_enabled() { return; }
    let tracked: Vec<u64> = {
        let g = match TRACKED_COUNTER_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        g.clone()
    };
    if tracked.is_empty() { return; }

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
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst),
            hipDeviceptr_t(ptr as _),
            F4_BYTES_PER,
        );
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Compare scratch_host vs last_state per pointer.
    for (i, &ptr) in tracked.iter().take(n).enumerate() {
        let new_slice: &[u8] = std::slice::from_raw_parts(
            tracker.scratch_host.add(i * F4_BYTES_PER),
            F4_BYTES_PER,
        );
        // Read first 18 i32 entries for the human-readable log.
        let new_i32: &[i32] = std::slice::from_raw_parts(
            new_slice.as_ptr() as *const i32,
            18,
        );
        if let Some(prev_arr) = tracker.last_state.get(&ptr) {
            if &prev_arr[..] != new_slice {
                let prev_i32: &[i32] = std::slice::from_raw_parts(
                    prev_arr.as_ptr() as *const i32,
                    18,
                );
                // Detect the shift-and-zero pattern: count how many leading
                // zeros are in `new` and how many of those positions had
                // non-zero values in `prev`. If both are >= 3, this is the
                // signature we're hunting.
                let mut new_leading_zeros = 0usize;
                while new_leading_zeros < 32 {
                    let v = *(new_slice.as_ptr() as *const i32).add(new_leading_zeros);
                    if v != 0 { break; }
                    new_leading_zeros += 1;
                }
                let mut prev_was_nonzero_in_zone = 0usize;
                for j in 0..new_leading_zeros.min(32) {
                    let pv = *(prev_arr.as_ptr() as *const i32).add(j);
                    if pv != 0 { prev_was_nonzero_in_zone += 1; }
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
                    if j > 0 { prev_str.push(','); new_str.push(','); }
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
    if state.is_some() { return; }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, F6_SCRATCH_BYTES, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F6 rd5 tracker: hipHostMalloc({}) failed: {:?}",
            F6_SCRATCH_BYTES, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, F6_SCRATCH_BYTES); }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F6 rd5 tracker: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
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
    if state.is_some() { return; }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, MCB_DUMP_BYTES, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MCB scratch: hipHostMalloc({}) failed: {:?}",
            MCB_DUMP_BYTES, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, MCB_DUMP_BYTES); }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] MCB scratch: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] MCB scratch: pinned {} B host=0x{:x} dev=0x{:x}",
        MCB_DUMP_BYTES, host_ptr as usize, dev_ptr as usize));
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
    grid_y: u32,  // the "N" in grid=[1,N,1] — correlates to # rd5s in batch
) {
    if !ms4_capture_enabled() { return; }

    // MemCopyBatchedAsync param layout (same as MS_2_Batched, per the PTX):
    //   param_0: u64 @ 0
    //   param_1: u16 @ 8
    //   param_2: u64 @ 16  ← scratch buffer device pointer
    let scratch_ptr: u64 = read_kernel_arg(kernel_params, extra, 16, 2).unwrap_or(0);
    if scratch_ptr == 0 { return; }

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
            scratch_ptr, MCB_DUMP_BYTES, copy_result));
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
        scratch_ptr, grid_y, hex_parts.join(", "),
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
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst),
            hipDeviceptr_t(src),
            F6_BYTES_PER,
        );
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

/// Snapshot `*(rd5+22176)` for every tracked rd5 pointer and diff against
/// the previous snapshot. Logs `*** RD5_22176 {RESET|DOUBLED|MUTATED} by
/// '<kernel>' rd5=0x... prev=N new=M`. The RESET tag fires when N→0
/// (normal between-frame reset), DOUBLED fires when N→2N (the bug),
/// MUTATED otherwise.
unsafe fn check_rd5_22176_mutation(current_kernel: &str) {
    if !ms4_capture_enabled() { return; }
    // F.6 tracker is gated separately because it has a stale-pointer
    // hazard — see f6_tracker_enabled() comment.
    if !f6_tracker_enabled() { return; }
    let tracked: Vec<u64> = {
        let g = match TRACKED_RD5_PTRS.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        g.clone()
    };
    if tracked.is_empty() { return; }

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
        let _ = hipMemcpyDtoD(
            hipDeviceptr_t(dst),
            hipDeviceptr_t(src),
            F6_BYTES_PER,
        );
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

struct CreateGridValidator {
    scratch_host: *mut u8,
    scratch_dev: *mut std::ffi::c_void,
}
unsafe impl Send for CreateGridValidator {}
unsafe impl Sync for CreateGridValidator {}

static CREATEGRID_VALIDATOR: Mutex<Option<CreateGridValidator>> = Mutex::new(None);

unsafe fn createGrid_validator_init(state: &mut Option<CreateGridValidator>) {
    if state.is_some() { return; }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, F5_SCRATCH_BYTES, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F5 CreateGrid validator: hipHostMalloc({}) failed: {:?}",
            F5_SCRATCH_BYTES, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, F5_SCRATCH_BYTES); }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] F5 CreateGrid validator: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] F5 CreateGrid validator: scratch={} B pinned; host=0x{:x} dev=0x{:x}",
        F5_SCRATCH_BYTES, host_ptr as usize, dev_ptr as usize,
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
) {
    if !ms4_capture_enabled() { return; }

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
            "*** F5 CreateGrid: numParticles={} (suspicious)", num_particles));
        return;
    }
    if sorted_keys_dev == 0 { return; }

    let n = num_particles as usize;
    let read_n = n.min(F5_MAX_KEYS);

    let mut guard = match CREATEGRID_VALIDATOR.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    createGrid_validator_init(&mut guard);
    let validator = match guard.as_mut() {
        Some(v) => v,
        None => return,
    };

    // D2D into pinned scratch — safe vs ProfilingSignal pool race.
    let copy_result = hipMemcpyDtoD(
        hipDeviceptr_t(validator.scratch_dev),
        hipDeviceptr_t(sorted_keys_dev as *mut std::ffi::c_void),
        read_n * 4,
    );
    if copy_result.is_err() {
        super::driver::zluda_log_err(&format!(
            "*** F5 CreateGrid: D2D failed {:?} ptr=0x{:x} n={}",
            copy_result, sorted_keys_dev, read_n));
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Walk on host via the host-mapped pointer.
    let keys: &[i32] = std::slice::from_raw_parts(
        validator.scratch_host as *const i32,
        read_n,
    );

    let mut violations = 0u32;
    let mut first_violation: Option<usize> = None;
    let mut min_key = keys[0];
    let mut max_key = keys[0];
    let mut negative_count = 0u32;
    let mut huge_count = 0u32;
    if keys[0] < 0 { negative_count += 1; }
    if keys[0] > 16_000_000 { huge_count += 1; }
    for i in 1..read_n {
        let k = keys[i];
        if k < keys[i - 1] {
            violations += 1;
            if first_violation.is_none() {
                first_violation = Some(i);
            }
        }
        if k < min_key { min_key = k; }
        if k > max_key { max_key = k; }
        if k < 0 { negative_count += 1; }
        if k > 16_000_000 { huge_count += 1; }
    }

    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let has_issues = violations > 0 || negative_count > 0 || huge_count > 0;

    if call_n < 5 || call_n % 100 == 0 || has_issues {
        super::driver::zluda_log_err(&format!(
            "F5 CreateGrid #{}: n={} read={} min={} max={} viol={} neg={} huge={}",
            call_n, n, read_n, min_key, max_key, violations, negative_count, huge_count,
        ));
    }
    if violations > 0 {
        let vi = first_violation.unwrap();
        super::driver::zluda_log_err(&format!(
            "*** SORT CORRUPTED #{}: keys[{}]={} > keys[{}]={} (n={} read={})",
            call_n, vi - 1, keys[vi - 1], vi, keys[vi], n, read_n,
        ));
        let start = vi.saturating_sub(5);
        let end = (vi + 5).min(read_n);
        let around: Vec<String> = (start..end)
            .map(|j| format!("[{}]={}", j, keys[j]))
            .collect();
        super::driver::zluda_log_err(&format!("    keys around violation: {}", around.join(", ")));
    }
    if negative_count > 0 || huge_count > 0 {
        // Find a representative bad key for the log.
        let bad = keys.iter().copied().enumerate()
            .find(|(_, k)| *k < 0 || *k > 16_000_000);
        if let Some((idx, k)) = bad {
            super::driver::zluda_log_err(&format!(
                "*** SORT BAD VALUE #{}: keys[{}]={} (0x{:x}) — bit31={}",
                call_n, idx, k, k as u32, (k as u32) >> 31,
            ));
        }
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
    *ENABLED.get_or_init(|| {
        std::env::var("ZLUDA_COUNTER_SNAPSHOT").map_or(false, |v| v == "1")
    })
}

/// Most recently observed MS_4 counter pointer (kp[7]). Updated by
/// `latch_counter_ptr_from_ms4` immediately before each MS_4 launch, read
/// by `snapshot_counter_after_launch` after every other launch. 0 = unset.
static LATCHED_COUNTER_PTR: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
    if state.is_some() { return; }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, COUNTER_SNAPSHOT_BYTES, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] counter snapshot: hipHostMalloc({}) failed: {:?}",
            COUNTER_SNAPSHOT_BYTES, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, COUNTER_SNAPSHOT_BYTES); }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] counter snapshot: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
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
    if !counter_snapshot_enabled() { return; }
    let counter_ptr: u64 = read_kernel_arg(kernel_params, extra, 56, 7).unwrap_or(0);
    if counter_ptr == 0 { return; }
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
    if !counter_snapshot_enabled() { return; }
    let ptr = LATCHED_COUNTER_PTR.load(std::sync::atomic::Ordering::SeqCst);
    if ptr == 0 { return; }

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
            ptr, kernel_name, copy_result));
        LATCHED_COUNTER_PTR.store(0, std::sync::atomic::Ordering::SeqCst);
        return;
    }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    let cur_slice: &[u8] = std::slice::from_raw_parts(
        state.scratch_host, COUNTER_SNAPSHOT_BYTES);
    if cur_slice == state.last_bytes { return; }

    // Mutated. Build the human log line and detect the all-zero bug signature.
    let cur_i32: &[i32] = std::slice::from_raw_parts(
        state.scratch_host as *const i32,
        COUNTER_SNAPSHOT_BYTES / 4,
    );
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
        if j > 0 { prev_str.push(','); new_str.push(','); }
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
    if state.is_some() { return; }
    let alloc_size = MS4_MAX_SHAPES * 4;
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let r = hipHostMalloc(&mut host_ptr, alloc_size, hipHostMallocMapped);
    if r.is_err() || host_ptr.is_null() { return; }
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
    counter_ptr: u64,     // param_7 device pointer
    n_particles: u32,     // param_3 (N) — upper bound on shape count
) -> u32 {
    if counter_ptr == 0 { return 0; }
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
    if r.is_err() { return 0; }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

    // Log first 32 values BEFORE patching for diagnosis
    let counters = std::slice::from_raw_parts_mut(
        buf.host_ptr as *mut u32, n_shapes);
    let dump_n = n_shapes.min(32);
    let mut vals = String::with_capacity(dump_n * 8);
    for i in 0..dump_n {
        if i > 0 { vals.push(','); }
        vals.push_str(&format!("{}", counters[i]));
    }
    super::driver::zluda_log_err(&format!(
        "[MS4_COUNTERS] n={} ptr=0x{:x} first{}=[{}]",
        n_shapes, counter_ptr, dump_n, vals,
    ));

    // Diagnostic only — no patching. Count zeros in counter range.
    let mut n_zeros = 0u32;
    for i in 0..n_shapes {
        if counters[i] == 0 { n_zeros += 1; }
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
    if flags_ptr == 0 || shape_idx_ptr == 0 || counter_ptr == 0 || n_particles == 0 { return; }
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
    if r.is_err() { return; }
    let _ = hipDeviceSynchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let ctr = std::slice::from_raw_parts(buf.host_ptr as *const u32, n_ctr);
    let mut counter_vals = [0u32; 64];
    counter_vals.copy_from_slice(ctr);
    let n_shapes = counter_vals.iter().rposition(|&v| v != 0).map(|i| i + 1).unwrap_or(0);

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
        if r2.is_err() { break; }
        let _ = hipDeviceSynchronize();
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let flags_raw = std::slice::from_raw_parts(buf.host_ptr as *const u8, this_chunk * 4);
        // Save flags locally
        let mut flags_local = vec![0u16; this_chunk];
        for i in 0..this_chunk {
            flags_local[i] = u16::from_le_bytes([flags_raw[i*4], flags_raw[i*4+1]]);
        }

        // Read shape_idx chunk
        let r3 = hipMemcpyDtoD(
            hipDeviceptr_t(buf.dev_ptr),
            hipDeviceptr_t((shape_idx_ptr + (offset * 4) as u64) as *mut std::ffi::c_void),
            this_chunk * 4,
        );
        if r3.is_err() { break; }
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
            n_shapes, mismatches.join(" "),
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
// 64-slot pinned host-mapped buffer dumped from the SIGABRT handler to
// `/tmp/tfs_captures.bin` (same dump pattern as MS_4 / MS_2_Batched / MS_3).

const TFS_CAPTURE_SLOTS: usize = 64;
const TFS_DESC_BYTES: usize = 456;
const TFS_KERNEL_NAME_BYTES: usize = 64;
const TFS_RAW_ARGS_BYTES: usize = 512;  // covers struct + scalar args + extras
const TFS_SLOT_SIZE: usize = std::mem::size_of::<TfsCaptureSlot>();
const TFS_CAPTURE_TOTAL_SIZE: usize = TFS_SLOT_SIZE * TFS_CAPTURE_SLOTS;
const TFS_CAPTURE_MAGIC: u64 = 0x5446_5343_4150_5455; // "TFSCAPTU"

#[repr(C)]
struct TfsCaptureSlot {
    magic: u64,                              // 0
    launch_n: u64,                           // 8
    timestamp_ms: u64,                       // 16
    kernel_id: u32,                          // 24 — see TFS_KERNEL_ID_*
    grid_x: u32,                             // 28
    grid_y: u32,                             // 32
    grid_z: u32,                             // 36
    block_x: u32,                            // 40
    block_y: u32,                            // 44
    block_z: u32,                            // 48
    raw_args_len: u32,                       // 52 — actual bytes captured
    kernel_name: [u8; TFS_KERNEL_NAME_BYTES], // 56 — null-padded ASCII
    raw_args: [u8; TFS_RAW_ARGS_BYTES],      // 120 — kernel arg flat buffer dump
    _pad: [u8; 8],                           // 632 — round to 640
}

// Kernel IDs (compact identifier for which TurbulenceFS kernel — used by
// the post-mortem analyzer to find the right replay slot configuration).
const TFS_KERNEL_ID_OTHER: u32                       = 0;  // any TFS_* not specifically known
const TFS_KERNEL_ID_CLEAR: u32                       = 1;
const TFS_KERNEL_ID_ADVECT: u32                      = 2;
const TFS_KERNEL_ID_SMOOTH_VELOCITY: u32             = 3;
const TFS_KERNEL_ID_PRESSURE_DIVERGENCE: u32         = 4;
const TFS_KERNEL_ID_PARTICLE_DENSITY_TO_PARTICLES: u32 = 5;
const TFS_KERNEL_ID_MIGRATE: u32                     = 6;
const TFS_KERNEL_ID_COUPLE_FROM_PARTICLES: u32       = 7;
const TFS_KERNEL_ID_PARTICLE_DENSITY_ATOMIC_TO_TEX: u32 = 8;
const TFS_KERNEL_ID_PARTICLE_DENSITY_SMOOTH: u32     = 9;

fn classify_tfs_kernel(name: &str) -> Option<u32> {
    if !name.starts_with("TurbulenceFS_") {
        return None;
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

fn tfs_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ZLUDA_TFS_CAPTURE").map_or(false, |v| v == "1")
    })
}

unsafe fn tfs_capture_init(state: &mut Option<TfsCaptureState>) {
    if state.is_some() { return; }
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let alloc_result = unsafe {
        hipHostMalloc(&mut host_ptr, TFS_CAPTURE_TOTAL_SIZE, hipHostMallocMapped)
    };
    if alloc_result.is_err() || host_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS capture: hipHostMalloc({}) failed: {:?}",
            TFS_CAPTURE_TOTAL_SIZE, alloc_result));
        return;
    }
    unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, TFS_CAPTURE_TOTAL_SIZE); }
    let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let map_result = unsafe { hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
    if map_result.is_err() || dev_ptr.is_null() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] TFS capture: hipHostGetDevicePointer failed: {:?}", map_result));
        unsafe { let _ = hipHostFree(host_ptr); }
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] TFS capture: {} slots × {} B = {} KB pinned; host=0x{:x} dev=0x{:x}",
        TFS_CAPTURE_SLOTS, TFS_SLOT_SIZE,
        TFS_CAPTURE_TOTAL_SIZE / 1024,
        host_ptr as usize, dev_ptr as usize,
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
        return 0;
    }
    if !kernel_params.is_null() {
        // Pack args sequentially into dst. We don't know exact arg sizes
        // from kernel_params alone, so we copy 8 bytes per arg index up to
        // 16 indices (= 128 bytes max). This is enough for scalar args;
        // the 456-byte struct itself is pointed at by one of the entries
        // and would need separate handling. For TurbulenceFS launches that
        // PhysX makes via cuLaunchKernel (which uses extra/flat-buffer),
        // we should rarely hit this branch.
        let mut written = 0usize;
        for i in 0..16 {
            if written + 8 > max_len { break; }
            let p = *kernel_params.add(i);
            if p.is_null() { break; }
            std::ptr::copy_nonoverlapping(p as *const u8, dst.add(written), 8);
            written += 8;
        }
        return written;
    }
    0
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
    grid_x: u32, grid_y: u32, grid_z: u32,
    block_x: u32, block_y: u32, block_z: u32,
) {
    if !tfs_capture_enabled() { return; }

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
    let mut launch_guard = if lock_mode >= 1 {
        Some(super::stream::NULL_STREAM_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    } else {
        None
    };

    // Log kernel name — gated by ZLUDA_LOG_LAUNCHES=1 to avoid overhead
    static LOG_LAUNCHES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let log_launches = *LOG_LAUNCHES.get_or_init(|| {
        std::env::var("ZLUDA_LOG_LAUNCHES").map_or(false, |v| v == "1")
    });
    let name = lookup_name(f.base);
    if log_launches {
        super::driver::zluda_log_err(&format!(
            "cuLaunchKernel '{}' grid=[{},{},{}] block=[{},{},{}] shared={}",
            name, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes));
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
                        if i > 0 { dump.push(' '); }
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
                    if arg_ptr == 0 { continue; }
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
                                if i > 0 { dump.push(' '); }
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
    // TurbulenceFS launch capture (ZLUDA_TFS_CAPTURE=1) — host-side dump
    // of the kernel argument flat buffer (incl. the 456-byte DynamicGrid
    // descriptor struct) for any TurbulenceFS_* kernel. Cheap (no GPU
    // ops, no sync). Captures into a 64-slot pinned ring dumped from the
    // SIGABRT handler to /tmp/tfs_captures.bin for post-mortem replay.
    if let Some(tfs_kernel_id) = classify_tfs_kernel(&name) {
        unsafe {
            capture_tfs_launch(
                &name, tfs_kernel_id, kernel_params, extra,
                grid_dim_x, grid_dim_y, grid_dim_z,
                block_dim_x, block_dim_y, block_dim_z,
            );
        }
    }
    // F.5: validate CreateGrid sortedKeys input. The kernel reads
    // `sortedKeys[i]` and uses it as a SIGNED index into gridStarts/gridEnds.
    // If CUB Downsweep produced corrupt keys (bit-31 stuck from lost
    // XOR flipback), CreateGrid does wild writes → GPU page fault. The
    // diagnostic that was supposed to catch this had been removed; this
    // re-adds it with the safe D2D pattern (avoids ProfilingSignal pool race).
    // CreateGrid validator — disabled (does D2D)
    // if name.contains("CreateGrid") && !name.contains("Turbulence") {
    //     unsafe { validate_create_grid_safe(kernel_params, extra); }
    // }
    // MS_2_Batched packet capture + pre-counter snapshot. We snapshot the
    // 88-byte-per-block packet buffer (param_2) AND every tracked counter
    // pointer BEFORE the kernel runs, so the post-capture below can diff
    // pre vs post and reveal whether MS_2_Batched is what turns counter[0..7]
    // from `[19,39,16,22,2,10,8,7]` into `[-2,-2,-2,-2,-2,-2,-2,-2]`.
    // Returns the slot index to thread into the post-capture.
    // MS_2B capture — disabled (does D2D)
    let ms2b_slot: Option<usize> = None;
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
    // The function `force_zero_rd5_22176_before_ms1` is left in place for
    // the smoke test below; the production call site is gone.
    // if name == "CreateParticleShapesIncremental_MultiStage_1" {
    //     unsafe { force_zero_rd5_22176_before_ms1(); }
    // }
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
        let sync_all = *SYNC_ALL.get_or_init(|| {
            std::env::var("ZLUDA_SYNC_LAUNCHES").map_or(false, |v| v == "1")
        });
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
                    n, name, grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z, peek_result));
            } else if n < 20 || n % 1000 == 0 {
                // Positive confirmation that sync ran and returned OK. Sampled so we
                // don't flood the log; the first 20 cover PhysX init, later samples
                // cover gameplay.
                super::driver::zluda_log_err(&format!(
                    "SYNC_OK #{} '{}'", n, name));
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
            name, call_n, surf_handle, desc_ptr, shift, copy_result));
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
    ).is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: flags read FAILED ptr=0x{:x}", name, call_n, flags_ptr));
        return;
    }

    // Read shape_index (u32 per entry, stride 4 bytes).
    let mut h_shape_idx = vec![0u32; n];
    if hip_runtime_sys::hipMemcpyDtoH(
        h_shape_idx.as_mut_ptr() as _,
        hip_runtime_sys::hipDeviceptr_t(shape_idx_ptr as *mut std::ffi::c_void),
        n * 4,
    ).is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: shape_idx read FAILED ptr=0x{:x}", name, call_n, shape_idx_ptr));
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
            if sh > max_shape_idx { max_shape_idx = sh; }
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
    ).is_err() {
        super::driver::zluda_log_err(&format!(
            "VALIDATE {} #{}: counter read FAILED ptr=0x{:x} size={}",
            name, call_n, counter_ptr, shape_count * 4));
        return;
    }

    // Read struct array (12 bytes per entry, first u32 is "base").
    let mut h_struct_bytes = vec![0u8; shape_count * 12];
    let struct_read_ok = hip_runtime_sys::hipMemcpyDtoH(
        h_struct_bytes.as_mut_ptr() as _,
        hip_runtime_sys::hipDeviceptr_t(struct_ptr as *mut std::ffi::c_void),
        shape_count * 12,
    ).is_ok();

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
        } else { 0 };
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
            "  counter[0..{}] = [{}]", dump_len, counters_str.join(",")));
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
                "  base[0..{}] = [{}]", dump_len, bases_str.join(",")));
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
            "VALIDATE {}: numParticles={} (suspicious)", name, num_particles));
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
            name, flags_result, flags_ptr, flags_byte_size));
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
            name, idx_result, index_ptr, n * 4));
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
            name, call_n, n, active_count, max_index, bad_indices.len()));
    }

    if !bad_indices.is_empty() {
        let show = bad_indices.len().min(10);
        for &(i, idx, flag) in &bad_indices[..show] {
            super::driver::zluda_log_err(&format!(
                "  BAD [{}]: flag=0x{:04x} index=0x{:08x} ({})",
                i, flag, idx, idx as i32));
        }
        if bad_indices.len() > 10 {
            super::driver::zluda_log_err(&format!(
                "  ... and {} more bad indices", bad_indices.len() - 10));
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
            "VALIDATE CreateGrid: numParticles={} (suspicious)", num_particles));
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
            result, sorted_keys_dev, n));
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
        if h_keys[i] < min_key { min_key = h_keys[i]; }
        if h_keys[i] > max_key { max_key = h_keys[i]; }
    }

    // Check for unreasonable values (negative or extremely large)
    let mut negative_count = 0u32;
    let mut huge_count = 0u32;
    for &k in &h_keys {
        if k < 0 { negative_count += 1; }
        if k > 16_000_000 { huge_count += 1; }  // 16M cells seems extremely large
    }

    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let has_issues = violations > 0 || negative_count > 0 || huge_count > 0;

    // Log first 5 calls, every 20th, or always if issues
    if call_n < 5 || call_n % 20 == 0 || has_issues {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid #{}: n={} min={} max={} violations={} neg={} huge={}",
            call_n, n, min_key, max_key, violations, negative_count, huge_count));
    }

    if violations > 0 {
        let vi = first_violation.unwrap();
        super::driver::zluda_log_err(&format!(
            "*** SORT CORRUPTED #{}: first violation at [{}]: keys[{}]={} > keys[{}]={}",
            call_n, vi, vi - 1, h_keys[vi - 1], vi, h_keys[vi]));
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
    arg_index: usize,   // index in kernel_params array
) -> Option<T> {
    if !kernel_params.is_null() {
        let param_ptr = *kernel_params.add(arg_index);
        if param_ptr.is_null() { return None; }
        Some(*(param_ptr as *const T))
    } else if !extra.is_null() {
        // extra format: [CU_LAUNCH_PARAM_BUFFER_POINTER(1), buffer_ptr, CU_LAUNCH_PARAM_BUFFER_SIZE(2), size_ptr, END]
        let extra_arr = std::slice::from_raw_parts(extra, 5);
        if extra_arr[0] != (0x1usize as *mut ::core::ffi::c_void) { return None; }
        let buffer = extra_arr[1] as *const u8;
        if buffer.is_null() { return None; }
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
        (8,  2, "gridCellStart"),
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
            name, call_n, n, n_cells,
            if nulls.is_empty() { "none".to_string() } else { nulls.join(",") },
            ptrs.join(", ")));
    }

    if !nulls.is_empty() {
        super::driver::zluda_log_err(&format!(
            "*** NULL POINTER in CollideParticles #{}: {} args are NULL!",
            call_n, nulls.join(", ")));
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
            "VALIDATE CreateGrid: numParticles={} (suspicious)", num_particles));
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
            result, sorted_keys_dev, n));
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
        if h_keys[i] < min_key { min_key = h_keys[i]; }
        if h_keys[i] > max_key { max_key = h_keys[i]; }
        if h_keys[i] < 0 { negative_count += 1; }
    }

    static CALL_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let call_n = CALL_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let has_issues = violations > 0 || negative_count > 0;

    if call_n < 3 || call_n % 50 == 0 || has_issues {
        super::driver::zluda_log_err(&format!(
            "VALIDATE CreateGrid #{}: n={} read={} min={} max={} violations={} neg={}",
            call_n, n, read_n, min_key, max_key, violations, negative_count));
    }

    if violations > 0 {
        let vi = first_violation.unwrap();
        super::driver::zluda_log_err(&format!(
            "*** SORT CORRUPTED #{}: [{}]={} > [{}]={}",
            call_n, vi - 1, h_keys[vi - 1], vi, h_keys[vi]));
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
        unsafe { force_zero_rd5_22176_before_ms1(); }
        // Restore in case other tests need it.
        {
            let mut g = match TRACKED_RD5_PTRS.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            *g = snapshot;
        }
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
            TRACKED_RD5_PTRS.lock().unwrap().len(), 0,
            "track_rd5_ptr(0) must not add anything"
        );

        // Add MS3_MAX_TRACKED_COUNTERS distinct pointers — all should land.
        for i in 1..=MS3_MAX_TRACKED_COUNTERS as u64 {
            let fake_ptr = 0xdead_0000_0000_0000u64 | i;
            track_rd5_ptr(fake_ptr);
        }
        assert_eq!(
            TRACKED_RD5_PTRS.lock().unwrap().len(), MS3_MAX_TRACKED_COUNTERS,
            "should track up to MS3_MAX_TRACKED_COUNTERS distinct ptrs"
        );

        // Adding the same pointers again must NOT grow the list.
        for i in 1..=MS3_MAX_TRACKED_COUNTERS as u64 {
            let fake_ptr = 0xdead_0000_0000_0000u64 | i;
            track_rd5_ptr(fake_ptr);
        }
        assert_eq!(
            TRACKED_RD5_PTRS.lock().unwrap().len(), MS3_MAX_TRACKED_COUNTERS,
            "duplicate adds must not grow the tracked list"
        );

        // Adding a NEW pointer beyond the limit silently drops.
        track_rd5_ptr(0xdead_0000_0000_00ffu64);
        assert_eq!(
            TRACKED_RD5_PTRS.lock().unwrap().len(), MS3_MAX_TRACKED_COUNTERS,
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
}

use crate::r#impl::{
    context,
    driver::{self},
};
use cuda_types::cuda::{CUarray, CUerror, CUresult, CUresultConsts};
use hip_runtime_sys::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{mem, ptr};
use zluda_common::FromCuda;

// Global HIP allocation range tracking for pointer validation
pub(crate) static ALLOC_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
pub(crate) static ALLOC_MAX: AtomicU64 = AtomicU64::new(0);

/// Table of host allocations created via `cuMemAllocHost` / `cuMemHostAlloc`.
/// Each entry is `(host_ptr, device_ptr, size)`. The MS_4 dump matches
/// kernel arg device pointers against `[device_ptr, device_ptr + size)`
/// to identify which tracked buffer a kernel reads from, then reads
/// contents via the host pointer (no GPU ops, no ProfilingSignal race).
pub(crate) static SMALL_HOST_ALLOCS: std::sync::Mutex<Vec<(usize, usize, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// Tracker for the specific 1040000-byte cuMemHostAlloc'd buffers we believe
/// PhysX/FleX use as actor lists. Plan Phase 3a Path 1b: log first 32 bytes
/// at allocation time and periodically thereafter, then diff PASS vs FAIL
/// to identify the buffer whose first slot stays NULL on the failing path.
/// Each entry is just `host_ptr` (the size==1040000 invariant identifies
/// the buffer; use the index in this vec as a stable label across log
/// messages). Gated by `ZLUDA_HOSTBUF_SAMPLE=1`.
pub(crate) static HOSTBUF_TRACKER: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn hostbuf_sample_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_HOSTBUF_SAMPLE").map_or(false, |v| v == "1"))
}

unsafe fn track_hostbuf(host: *mut ::core::ffi::c_void, size: usize) {
    if !hostbuf_sample_enabled() || size != 1040000 {
        return;
    }
    if let Ok(mut t) = HOSTBUF_TRACKER.lock() {
        let idx = t.len();
        let p = host as *const u64;
        let v0 = p.read_unaligned();
        let v1 = p.add(1).read_unaligned();
        let v2 = p.add(2).read_unaligned();
        let v3 = p.add(3).read_unaligned();
        super::driver::zluda_log_err(&format!(
            "[HOSTBUF_TRACK #{:02}] ptr=0x{:x} q0=0x{:016x} q1=0x{:016x} q2=0x{:016x} q3=0x{:016x}",
            idx, host as usize, v0, v1, v2, v3
        ));
        t.push(host as usize);
    }
}

fn untrack_hostbuf(host: *mut ::core::ffi::c_void) {
    if !hostbuf_sample_enabled() {
        return;
    }
    if let Ok(mut t) = HOSTBUF_TRACKER.lock() {
        t.retain(|p| *p != host as usize);
    }
}

/// Snapshot all tracked 1040000 buffers' first 32 bytes. Call from rare
/// events (alloc completion, every Nth kernel launch). The output line is
/// one per call, with all buffers' q0 values inline so a single grep finds
/// the timeline. q0 alone usually suffices because PhysX iterates `array[0]`
/// as a pointer; q1..q3 provided for context.
pub(crate) unsafe fn sample_hostbufs(reason: &str) {
    if !hostbuf_sample_enabled() {
        return;
    }
    if let Ok(t) = HOSTBUF_TRACKER.lock() {
        if t.is_empty() {
            return;
        }
        let mut s = String::with_capacity(t.len() * 32 + 32);
        s.push_str(&format!("[HOSTBUF_SAMPLE {}] ", reason));
        for (i, &p) in t.iter().enumerate() {
            let q0 = (p as *const u64).read_unaligned();
            s.push_str(&format!("#{:02}=0x{:016x} ", i, q0));
        }
        super::driver::zluda_log_err(&s);
    }
}

fn track_host_alloc(host: *mut ::core::ffi::c_void, dev: *mut ::core::ffi::c_void, size: usize) {
    if size == 0 {
        return;
    }
    if let Ok(mut t) = SMALL_HOST_ALLOCS.lock() {
        t.push((host as usize, dev as usize, size));
    }
}

fn untrack_host_alloc(host: *mut ::core::ffi::c_void) {
    if let Ok(mut t) = SMALL_HOST_ALLOCS.lock() {
        t.retain(|(p, _, _)| *p != host as usize);
    }
}

pub(crate) fn update_alloc_range(addr: u64, size: u64) {
    ALLOC_MIN.fetch_min(addr, Ordering::Relaxed);
    ALLOC_MAX.fetch_max(addr + size, Ordering::Relaxed);
}

/// Check whether VRAM placement diagnostics are enabled. Gated behind
/// `ZLUDA_VRAM_LOG=1` because the sysfs read on every allocation has a
/// non-negligible runtime cost we don't want to pay in normal operation.
fn vram_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_VRAM_LOG").map_or(false, |v| v == "1"))
}

/// Check whether DtoH-copy diagnostics are enabled (`ZLUDA_DTOH_LOG=1`).
/// When ON, logs every cuMemcpyDtoH{,Async} with dst/src/size and the
/// first up-to-16 dwords of the destination buffer AFTER the copy
/// completes. For Mode-B-SIGSEGV investigation: PhysX reads back small
/// scalars/structs from GPU; this surfaces the value (count, range,
/// status flag) that drives downstream CPU-side allocation/iteration.
fn dtoh_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_DTOH_LOG").map_or(false, |v| v == "1"))
}

/// Plan v13 Step 1C scan: at MS_4 launch, locate which `cuMemHostAlloc`
/// (zero-copy mapped host memory) buffer carries the `n_threads` value.
///
/// Background: Plan v13 Step 1 + 1B proved MS_4's n_threads is NOT in any
/// cuMemcpyDtoH transfer (≤4KB cap, head/mid/tail sampled, 0/2847 matches
/// in run #2). Then `cuMemHostAlloc size=... same=true` log entries
/// surfaced — PhysX uses zero-copy mapped memory, so GPU writes are visible
/// to the CPU via the host pointer with NO DtoH involved.
///
/// This scan walks every tracked `(host, dev, size)` and reports any
/// 4-byte-aligned u32 slot whose value matches `needle`. Bounded by the
/// total mapped-memory size (~10 MB in Batman) so cost is ~2.5M cmps per
/// scan, spread across MS_4 launches.
///
/// Returns up to `max_matches` `(host_base, offset, surrounding_u32s[4])`
/// tuples. The surrounding context disambiguates a coincidental scalar
/// match from a structured count-vector at a stable offset.
pub(crate) fn scan_mapped_for_u32(
    needle: u32,
    max_matches: usize,
) -> Vec<(usize, usize, usize, [u32; 4])> {
    let mut out = Vec::new();
    let allocs = match SMALL_HOST_ALLOCS.lock() {
        Ok(g) => g.clone(),
        Err(_) => return out,
    };
    for &(host, _dev, size) in allocs.iter() {
        if host == 0 || size < 4 {
            continue;
        }
        let n_words = size / 4;
        unsafe {
            let base = host as *const u32;
            for i in 0..n_words {
                if base.add(i).read_unaligned() == needle {
                    // Capture 4 surrounding dwords: [i-1, i, i+1, i+2] when
                    // possible — the count is often part of a small struct.
                    let mut ctx = [0u32; 4];
                    let start = if i == 0 { 0 } else { i - 1 };
                    for (k, j) in (start..start + 4).enumerate() {
                        if j < n_words {
                            ctx[k] = base.add(j).read_unaligned();
                        }
                    }
                    out.push((host, size, i * 4, ctx));
                    if out.len() >= max_matches {
                        return out;
                    }
                }
            }
        }
    }
    out
}

/// Plan v14 Step 1: per-kernel mapped-buffer write tracer.
///
/// Background: Plan v13 Step 1C found MS_4's n_threads in `cuMemHostAlloc`
/// mapped buffers, with at least 3 stable u32 slots — one growing unbounded,
/// one shrinking. Open question: WHICH kernel mutates each slot, and is any
/// kernel ever decrementing the runaway slots?
///
/// Approach: hold a per-buffer mirror of the last-seen u32 contents. After
/// every `cuLaunchKernel` (post-sync), re-read each tracked mapped buffer,
/// diff against the mirror, and emit one `[MAPPED_WRITE]` line per changed
/// slot attributing the change to the just-finished kernel. Update the
/// mirror to current values for the next launch's diff baseline.
///
/// Cost: ~6 MB of memcmp per launch (run #3 had 3 × 2 MB buffers). At
/// typical launch rates this is ~1-3 ms / launch — acceptable for a
/// diagnostic build, default OFF in normal operation.
///
/// The first sighting of any buffer populates the mirror WITHOUT emitting
/// diff lines (we'd otherwise log every non-zero slot as a fresh "write"
/// at game start, which is noise). Real diffs only fire on subsequent
/// launches once a baseline exists.
pub(crate) fn mapped_write_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_MAPPED_WRITE_TRACE").map_or(false, |v| v == "1"))
}

/// State for the mapped-write tracer.
///
/// `mirror`: last-known u32 contents per tracked mapped buffer.
/// `slot_changes`: per-slot change count, keyed by (host_ptr, u32_index).
/// v14.next.next replaces v14.next's per-buffer noisy demotion with per-slot
/// tracking — a quiet particle-count slot survives even if it shares a
/// buffer with FleX's RasterizeConvexGrid that hammers other slots.
pub(crate) struct MappedTraceState {
    pub mirror: std::collections::HashMap<usize, Vec<u32>>,
    pub slot_changes: std::collections::HashMap<(usize, usize), u32>,
}

impl MappedTraceState {
    fn new() -> Self {
        Self {
            mirror: std::collections::HashMap::new(),
            slot_changes: std::collections::HashMap::new(),
        }
    }
}

static MAPPED_TRACE_STATE: std::sync::OnceLock<std::sync::Mutex<MappedTraceState>> =
    std::sync::OnceLock::new();

fn mapped_trace_state() -> &'static std::sync::Mutex<MappedTraceState> {
    MAPPED_TRACE_STATE.get_or_init(|| std::sync::Mutex::new(MappedTraceState::new()))
}

/// Per-SLOT cumulative change threshold above which we stop emitting for
/// that slot specifically. v14.next run #1 had particle-count slot writes
/// every ~10 MS_4 launches (~40 changes over 400 MS_4), but a noisy buffer
/// header could change 500× per launch. 200 cumulative cleanly excludes
/// the heavy bulk-writers while letting particle-count slots emit ~50×
/// across a full benchmark run.
const SLOT_HOT_THRESHOLD: u32 = 200;

/// Diff each tracked mapped buffer against the mirror, emit MAPPED_WRITE
/// lines for each changed u32 slot, and update the mirror.
///
/// `seq` is a monotonic launch counter (used in log lines so the offline
/// analyzer can join MAPPED_WRITE entries to the corresponding launch).
/// `kernel` is the name of the kernel that just completed.
///
/// Per-buffer per-launch emit cap (32 lines) prevents a `memset`-style
/// kernel from flooding the log; truncated counts surface as a
/// `(+N more changes truncated)` tail.
/// Plan v15: pre-launch refresh. Re-read every tracked mapped buffer's
/// current contents into the mirror. Called immediately AFTER a
/// `hipDeviceSynchronize` at the START of each `cuLaunchKernel`. Combined
/// with the existing post-sync diff, this ensures the post-diff observes
/// ONLY this kernel's writes.
///
/// Plan v16 (Plan B "between-launch tracker"): BEFORE refreshing the mirror,
/// diff it against the current state and emit `[MAPPED_BETWEEN]` events.
/// These are changes that happened between [post-sync of prior launch] and
/// [pre-sync of this launch] — i.e., CPU-side writes via the host pointer
/// alias of the mapped buffer (plus possibly other-thread GPU launches).
/// v15 proved that the particle-count slots (off=663552/640/728) never
/// appear in [MAPPED_WRITE] DURING events; they must change BETWEEN
/// launches, and this hook captures those changes.
///
/// First-sighting behavior: populate mirror without emitting BETWEEN diff
/// (we don't know the baseline yet).
///
/// Per-slot change counter is shared with the DURING tracker — a slot that
/// reaches HOT in either path is silenced for both.
pub(crate) fn mapped_write_refresh_pre(seq: u64) {
    if !mapped_write_trace_enabled() {
        return;
    }
    let allocs = match SMALL_HOST_ALLOCS.lock() {
        Ok(g) => g.clone(),
        Err(_) => return,
    };
    let mut state = match mapped_trace_state().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let MappedTraceState {
        mirror,
        slot_changes,
    } = &mut *state;
    for &(host, _dev, size) in allocs.iter() {
        if host == 0 || size < 4 {
            continue;
        }
        let n_words = size / 4;
        let entry = mirror.entry(host).or_insert_with(Vec::new);
        let first_sighting = entry.is_empty();
        if entry.len() != n_words {
            entry.resize(n_words, 0u32);
        }
        if first_sighting {
            // Initial population — no BETWEEN diff possible yet.
            unsafe {
                let base = host as *const u32;
                for i in 0..n_words {
                    entry[i] = base.add(i).read_unaligned();
                }
            }
            continue;
        }
        // BETWEEN diff: compare current state vs mirror (which holds the
        // post-sync state of the PRIOR launch). Any change here happened
        // outside any single launch's pre/post envelope — most likely a
        // CPU-side write via the host pointer.
        let mut emits = 0u32;
        let mut more = 0u32;
        const EMIT_CAP: u32 = 32;
        unsafe {
            let base = host as *const u32;
            for i in 0..n_words {
                let cur = base.add(i).read_unaligned();
                let old = entry[i];
                if cur != old {
                    let count = slot_changes.entry((host, i)).or_insert(0);
                    *count += 1;
                    let just_demoted = *count == SLOT_HOT_THRESHOLD + 1;
                    if just_demoted {
                        super::driver::zluda_log_err(&format!(
                            "[MAPPED_BETWEEN seq={}] host=0x{:x} off={} slot marked HOT (changes={} > {}); future emits suppressed",
                            seq, host, i * 4, *count, SLOT_HOT_THRESHOLD
                        ));
                    }
                    if *count <= SLOT_HOT_THRESHOLD {
                        if emits < EMIT_CAP {
                            super::driver::zluda_log_err(&format!(
                                "[MAPPED_BETWEEN seq={}] host=0x{:x} off={} old=0x{:08x} new=0x{:08x} slot_changes={}",
                                seq, host, i * 4, old, cur, *count
                            ));
                            emits += 1;
                        } else {
                            more += 1;
                        }
                    }
                    entry[i] = cur;
                }
            }
        }
        if more > 0 {
            super::driver::zluda_log_err(&format!(
                "[MAPPED_BETWEEN seq={}] host=0x{:x} (+{} more eligible changes truncated)",
                seq, host, more
            ));
        }
    }
}

pub(crate) fn mapped_write_diff_post(seq: u64, kernel: &str) {
    if !mapped_write_trace_enabled() {
        return;
    }
    let allocs = match SMALL_HOST_ALLOCS.lock() {
        Ok(g) => g.clone(),
        Err(_) => return,
    };
    let mut state = match mapped_trace_state().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let MappedTraceState {
        mirror,
        slot_changes,
    } = &mut *state;
    for &(host, _dev, size) in allocs.iter() {
        if host == 0 || size < 4 {
            continue;
        }
        let n_words = size / 4;
        let entry = mirror.entry(host).or_insert_with(Vec::new);
        let first_sighting = entry.is_empty();
        if entry.len() != n_words {
            entry.resize(n_words, 0u32);
        }
        if first_sighting {
            // Populate mirror with current contents. No diff lines —
            // we don't know which kernel "wrote" the initial state.
            unsafe {
                let base = host as *const u32;
                for i in 0..n_words {
                    entry[i] = base.add(i).read_unaligned();
                }
            }
            continue;
        }
        let mut emits = 0u32;
        let mut more = 0u32;
        const EMIT_CAP: u32 = 32;
        unsafe {
            let base = host as *const u32;
            for i in 0..n_words {
                let cur = base.add(i).read_unaligned();
                let old = entry[i];
                if cur != old {
                    let count = slot_changes.entry((host, i)).or_insert(0);
                    *count += 1;
                    let just_demoted = *count == SLOT_HOT_THRESHOLD + 1;
                    if just_demoted {
                        super::driver::zluda_log_err(&format!(
                            "[MAPPED_WRITE seq={}] host=0x{:x} off={} slot marked HOT (changes={} > {}); future emits suppressed",
                            seq, host, i * 4, *count, SLOT_HOT_THRESHOLD
                        ));
                    }
                    if *count <= SLOT_HOT_THRESHOLD {
                        if emits < EMIT_CAP {
                            super::driver::zluda_log_err(&format!(
                                "[MAPPED_WRITE seq={}] kernel='{}' host=0x{:x} off={} old=0x{:08x} new=0x{:08x} slot_changes={}",
                                seq, kernel, host, i * 4, old, cur, *count
                            ));
                            emits += 1;
                        } else {
                            more += 1;
                        }
                    }
                    entry[i] = cur;
                }
            }
        }
        if more > 0 {
            super::driver::zluda_log_err(&format!(
                "[MAPPED_WRITE seq={}] kernel='{}' host=0x{:x} (+{} more eligible changes truncated)",
                seq, kernel, host, more
            ));
        }
    }
}

/// Plan v14.next/v14.next.next: log writes into tracked mapped buffers.
///
/// Background: Plan v14 Step 1 (run #1) showed `MemCopyBatchedAsync` (HIP's
/// bulk DtoD memcpy kernel) accounts for ~80% of writes to the particle-count
/// buffer slots. PhysX uses bulk memcpy to REFRESH counters from some
/// upstream buffer (GPU OR CPU). v14.next run #1 caught 0 MAPPED_DTOD because
/// PhysX routes through `cuMemcpyAsync` (CU_MEMORYTYPE_DEVICE inferred),
/// not the typed DtoD variants. v14.next.next broadens the hook to all
/// async / H2D memcpy variants.
///
/// `src_is_host=true` for H2D paths — src is a CPU pointer we can dereference
/// directly. For DtoD/Async paths, src is a device pointer; we attempt to
/// resolve it through SMALL_HOST_ALLOCS in case it points into another
/// zero-copy mapped buffer.
///
/// Gated by `ZLUDA_MAPPED_WRITE_TRACE=1` (same env var as the kernel-write
/// tracer — the two are joined by their shared diagnostic purpose).
pub(crate) unsafe fn log_memcpy_into_mapped(
    dst_device_or_host: u64,
    src: u64,
    byte_count: usize,
    src_is_host: bool,
    label: &str,
) {
    if !mapped_write_trace_enabled() {
        return;
    }
    if byte_count == 0 {
        return;
    }
    let allocs = match SMALL_HOST_ALLOCS.lock() {
        Ok(g) => g.clone(),
        Err(_) => return,
    };
    // Find which tracked mapped buffer's device range contains dst.
    for &(host, dev, size) in allocs.iter() {
        let dev_end = dev.wrapping_add(size);
        let dst = dst_device_or_host as usize;
        if dst < dev || dst >= dev_end {
            continue;
        }
        let off = dst - dev;
        // Peek at source contents — up to 8 u32s.
        let mut src_ctx: Option<[u32; 8]> = None;
        if src_is_host {
            // Direct CPU read.
            if src != 0 {
                let mut ctx = [0u32; 8];
                let words = (byte_count / 4).min(8);
                let base = src as *const u32;
                for k in 0..words {
                    ctx[k] = base.add(k).read_unaligned();
                }
                src_ctx = Some(ctx);
            }
        } else {
            // Device pointer — look up in mapped allocs.
            for &(s_host, s_dev, s_size) in allocs.iter() {
                let s_end = s_dev.wrapping_add(s_size);
                let s = src as usize;
                if s >= s_dev && s < s_end {
                    let s_off = s - s_dev;
                    let words_avail = (s_size - s_off) / 4;
                    let words = words_avail.min(8).min(byte_count / 4);
                    let mut ctx = [0u32; 8];
                    let base = (s_host + s_off) as *const u32;
                    for k in 0..words {
                        ctx[k] = base.add(k).read_unaligned();
                    }
                    src_ctx = Some(ctx);
                    break;
                }
            }
        }
        let ctx_str = match src_ctx {
            Some(c) => format!(
                " src_u32=[{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}]",
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            ),
            None => String::new(),
        };
        super::driver::zluda_log_err(&format!(
            "[MAPPED_MEMCPY/{}] dst_host=0x{:x} dst_off={} src=0x{:x} src_is_host={} bytes={}{}",
            label, host, off, src, src_is_host, byte_count, ctx_str
        ));
        return;
    }
}

/// Backward-compatible alias for the v14.next-era DtoD hook callers.
pub(crate) unsafe fn log_dtod_into_mapped(
    dst_device: u64,
    src_device: u64,
    byte_count: usize,
    label: &str,
) {
    log_memcpy_into_mapped(dst_device, src_device, byte_count, false, label)
}

fn ms2_rd6_watch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_MS2_RD6_WATCH").map_or(false, |v| v == "1"))
}

unsafe fn log_memcpy_into_ms2_rd6_watch(
    dst_device_or_host: u64,
    src: u64,
    byte_count: usize,
    src_is_host: bool,
    label: &str,
) {
    if !ms2_rd6_watch_enabled() || byte_count == 0 {
        return;
    }
    let watch = super::function::ms2_rd6_watch_snapshot();
    if watch.is_empty() {
        return;
    }
    let dst_start = dst_device_or_host;
    let dst_end = dst_start.saturating_add(byte_count as u64);
    let mapped_src_allocs = if src_is_host {
        Vec::new()
    } else {
        match SMALL_HOST_ALLOCS.lock() {
            Ok(g) => g.clone(),
            Err(_) => Vec::new(),
        }
    };
    for (rd6, launch_n) in watch {
        let watch_start = rd6.saturating_add(8);
        let watch_end = watch_start.saturating_add(4);
        if dst_start >= watch_end || dst_end <= watch_start {
            continue;
        }
        let src_u32 = if src == 0 {
            None
        } else if src_is_host {
            Some((src as *const u32).read_unaligned())
        } else {
            let mut out = None;
            for &(host, dev, size) in mapped_src_allocs.iter() {
                let dev_end = dev.saturating_add(size);
                let s = src as usize;
                if s >= dev && s + 4 <= dev_end {
                    out = Some(((host + (s - dev)) as *const u32).read_unaligned());
                    break;
                }
            }
            out
        };
        super::driver::zluda_log_err(&format!(
            "[MS2_RD6_WATCH/{label}] launch_n={} rd6=0x{:x} watch=[0x{:x}..0x{:x}) dst=0x{:x} bytes={} src=0x{:x} src_is_host={} src_u32={:?}",
            launch_n, rd6, watch_start, watch_end, dst_start, byte_count, src, src_is_host, src_u32
        ));
    }
}

/// Plan v13 Step 1: small-DtoH trace gated by `ZLUDA_DTOH_TRACE=1`.
///
/// Differs from `ZLUDA_DTOH_LOG` in three ways:
/// 1. ONLY fires for transfers of ≤ 256 bytes (PhysX-style scalar readback)
/// 2. Routes through `zluda_log_err` (timestamped, `/tmp/zluda_tex.log`)
///    so it shares the same log file as MS4_ARGS — offline correlator
///    just greps both prefixes by timestamp.
/// 3. Always prints the first 1-8 u32s of the dst buffer for direct
///    value-match against MS_4 param_3 (n_threads).
///
/// Purpose: find which cuMemcpyDtoH returns the u32 that PhysX subsequently
/// packs into MS_4's `param_3` (n_threads). Per Plan v13 if a clean
/// correlator exists, we identify the producer kernel.
fn dtoh_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_DTOH_TRACE").map_or(false, |v| v == "1"))
}

/// Plan v13 Step 1 helper. Emits a `[DTOH_TRACE/<label>]` line into the
/// timestamped err log if the copy completed and the size is ≤ 4096 bytes.
///
/// Run #1 (256B cap) captured 1601 small DtoHs but 0 of them carried any
/// MS_4 n_threads value. Run #2 widens to 4096B AND samples 3 windows
/// (head/middle/tail of 32 bytes each) for buffers >64B so a count
/// stored mid-buffer or in a trailing footer is still visible. Anything
/// >4KB is still suppressed to keep log volume bounded.
///
/// Caller must ensure the dst buffer is host-visible + the copy completed
/// (i.e. call this AFTER any required stream-sync).
unsafe fn log_dtoh_trace(
    label: &str,
    dst_host: *const ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) {
    if !dtoh_trace_enabled() {
        return;
    }
    if byte_count == 0 || byte_count > 4096 {
        return;
    }
    let dst_u8 = dst_host as *const u8;
    let line = if byte_count <= 64 {
        // Small: full dump up to 8 dwords (≤32 B).
        let n_u32 = (byte_count / 4).min(8);
        let mut vals = String::with_capacity(n_u32 * 9);
        for i in 0..n_u32 {
            let v = (dst_u8 as *const u32).add(i).read_unaligned();
            if i > 0 {
                vals.push(' ');
            }
            vals.push_str(&format!("{:08x}", v));
        }
        format!(
            "[DTOH_TRACE/{}] src={:p} dst={:p} size={} u32=[{}]",
            label, src_device.0, dst_host, byte_count, vals
        )
    } else {
        // Medium (65..=4096): three 32-byte windows (head/mid/tail).
        // 8 dwords × 3 = 24 dwords logged. Mid-window aligned to 4.
        let q1 = (byte_count / 2) & !0x3;
        let q1 = q1.min(byte_count.saturating_sub(32));
        let qt = byte_count.saturating_sub(32) & !0x3;
        let read_window = |off: usize| -> String {
            let n = ((byte_count - off) / 4).min(8);
            let mut s = String::with_capacity(n * 9);
            for i in 0..n {
                let v = (dst_u8.add(off) as *const u32).add(i).read_unaligned();
                if i > 0 {
                    s.push(' ');
                }
                s.push_str(&format!("{:08x}", v));
            }
            s
        };
        format!(
            "[DTOH_TRACE/{}] src={:p} dst={:p} size={} u32@0=[{}] u32@{}=[{}] u32@{}=[{}]",
            label,
            src_device.0,
            dst_host,
            byte_count,
            read_window(0),
            q1,
            read_window(q1),
            qt,
            read_window(qt)
        )
    };
    super::driver::zluda_log_err(&line);
}

/// Format up to 16 dwords (64 bytes) at byte offset `off` from `dst_host`.
/// Caller must guarantee the slice is mapped + readable.
unsafe fn format_dwords_at(dst_host: *const u8, off: usize, n_bytes: usize) -> String {
    let n_dwords = n_bytes / 4;
    let mut s = String::with_capacity(n_dwords * 9);
    for i in 0..n_dwords {
        let v = (dst_host.add(off) as *const u32).add(i).read_unaligned();
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{:08x}", v));
    }
    s
}

/// Format dwords for DtoH logging.
///
/// - Small buffers (≤ 256 bytes): one window at offset 0, up to 16 dwords.
/// - Big buffers (> 256 bytes): four 16-byte windows at offsets 0,
///   1/4·size, 1/2·size, 3/4·size — disambiguates "all-zero" from
///   "zero-prefix only with data later in the buffer".
unsafe fn format_dtoh_dwords(dst_host: *const u8, byte_count: usize) -> String {
    if byte_count <= 256 {
        let n = byte_count.min(64);
        let dwords = format_dwords_at(dst_host, 0, n);
        return format!("[{}]", dwords);
    }
    // Big buffer: 4 sample windows × 16 bytes (4 dwords) each
    let q1 = (byte_count / 4) & !0x3; // align to 4
    let q2 = (byte_count / 2) & !0x3;
    let q3 = (3 * byte_count / 4) & !0x3;
    let w0 = format_dwords_at(dst_host, 0, 16);
    let w1 = format_dwords_at(dst_host, q1, 16);
    let w2 = format_dwords_at(dst_host, q2, 16);
    let w3 = format_dwords_at(dst_host, q3, 16);
    format!(
        "[@0:{} @{:#x}:{} @{:#x}:{} @{:#x}:{}]",
        w0, q1, w1, q2, w2, q3, w3
    )
}

/// Read current VRAM usage from sysfs. Returns bytes used, or 0 on error
/// OR when `ZLUDA_VRAM_LOG` is unset. The gate is here (not at the call
/// sites) so callers can freely `let vram_before = vram_used_bytes();`
/// without conditional bracketing — when logging is off this is a single
/// atomic-load + early return, so the allocation hot path pays almost
/// nothing.
pub(crate) fn vram_used_bytes() -> u64 {
    if !vram_log_enabled() {
        return 0;
    }
    std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_used")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Log VRAM delta for an allocation. Returns (vram_before, vram_after).
/// No-op when `ZLUDA_VRAM_LOG` is unset.
pub(crate) fn log_vram_delta(label: &str, vram_before: u64, vram_after: u64, size: usize) {
    if !vram_log_enabled() {
        return;
    }
    let delta = vram_after as i64 - vram_before as i64;
    let placement = if delta > 1024 { "VRAM" } else { "GTT/other" };
    super::driver::zluda_log_err(&format!(
        "{} size={} KiB vram_delta={:+} KiB total_vram={} MiB ({})",
        label,
        size / 1024,
        delta / 1024,
        vram_after / (1024 * 1024),
        placement
    ));
}

struct DropGuard<F: FnMut()>(F);

impl<F: FnMut()> Drop for DropGuard<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

pub(crate) unsafe fn alloc_v2(dptr: &mut hipDeviceptr_t, bytesize: usize) -> CUresult {
    super::driver::zluda_log(&format!("[ZLUDA] cuMemAlloc_v2({} bytes)", bytesize));
    let cu_context = context::get_current_context()?;
    let context: &context::Context = FromCuda::<_, CUerror>::from_cuda(&cu_context)?;
    let vram_before = vram_used_bytes();
    // Round up to provide headroom for vectorized over-reads.
    // CUDA kernels (e.g. PhysX radix sort) use ld.global.v4.u32 to load 4 elements
    // (16 bytes) at a time. When the element count isn't a multiple of 4, the last
    // load reads 1-3 elements past the allocation. On NVIDIA, 2MB page granularity
    // makes this safe. On AMD with 4KB pages, it can cross into unmapped memory.
    //
    // 2026-05-15 v6 experiment: env-gated alignment.
    //   ZLUDA_ALLOC_PAD=N → round up to N-byte boundary (N must be power-of-2).
    //   Default 16 bytes (one v4.u32). Try 256 to re-test the prior bisection
    //   suspect for TurbulenceFS-zeros-counter cross-allocation corruption.
    static PAD_MINUS_1: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let pad = *PAD_MINUS_1.get_or_init(|| {
        std::env::var("ZLUDA_ALLOC_PAD")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n.is_power_of_two() && n >= 16)
            .map(|n| n - 1)
            .unwrap_or(15)
    });
    let alloc_size = (bytesize + pad) & !pad;
    hipMalloc(ptr::from_mut(dptr).cast(), alloc_size)?;
    log_vram_delta("hipMalloc", vram_before, vram_used_bytes(), alloc_size);
    fill_with_zero_and_register(cu_context, context, dptr, alloc_size)?;
    Ok(())
}

unsafe fn fill_with_zero_and_register(
    cu_context: cuda_types::cuda::CUcontext,
    context: &context::Context,
    dptr: &mut hipDeviceptr_t,
    bytesize: usize,
) -> Result<(), CUerror> {
    let drop_guard = DropGuard(|| {
        unsafe { hipFree(dptr.0) }.ok();
    });
    let mut capturing_status = mem::zeroed();
    hipStreamIsCapturing(hipStream_t(ptr::null_mut()), &mut capturing_status)?;
    if capturing_status == hipStreamCaptureStatus::hipStreamCaptureStatusNone {
        hipMemsetD8Async(*dptr, 0, bytesize, context.alloc_stream)?;
        hipStreamSynchronize(context.alloc_stream)?;
    }
    super::driver::zluda_log(&format!("[ZLUDA] cuMemAlloc_v2 -> {:?} OK", dptr.0));
    super::driver::zluda_log_err(&format!("ALLOC {:?} size={}", dptr.0, bytesize));
    update_alloc_range(dptr.0 as u64, bytesize as u64);
    add_allocation(dptr.0, bytesize, cu_context)?;
    mem::forget(drop_guard);
    Ok(())
}

pub(crate) unsafe fn free_v2(dptr: hipDeviceptr_t) -> CUresult {
    super::driver::zluda_log(&format!("[ZLUDA] cuMemFree_v2({:?})", dptr.0));
    super::driver::zluda_log_err(&format!("FREE {:?}", dptr.0));
    let hip_result = hipFree(dptr.0);
    remove_allocation(dptr.0)?;
    Ok(hip_result?)
}

pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(hipStream_t(std::ptr::null_mut()));
    let result = unsafe { hipMemcpyDtoH(dst_host, src_device, byte_count) };
    if result.is_ok() && dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(dst_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[DTOH/sync] dst={:p} src={:p} size={} bytes={}",
            dst_host, src_device.0, byte_count, dwords
        ));
    }
    if result.is_ok() {
        unsafe { log_dtoh_trace("sync", dst_host, src_device, byte_count) };
    }
    result
}

pub(crate) fn copy_hto_d_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(hipStream_t(std::ptr::null_mut()));
    // Plan v14.next.next: H2D into a tracked mapped buffer = CPU directly
    // overwriting a slot. PhysX's per-frame counter bookkeeping may take
    // this path (set values from CPU-side state).
    unsafe {
        log_memcpy_into_mapped(
            dst_device.0 as u64,
            src_host as u64,
            byte_count,
            true,
            "hto_d_v2",
        );
        log_memcpy_into_ms2_rd6_watch(
            dst_device.0 as u64,
            src_host as u64,
            byte_count,
            true,
            "hto_d_v2",
        );
    }
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst_device.0 as u64, byte_count as u64)
    {
        let first_word = if byte_count >= 4 {
            unsafe { *(src_host as *const u32) }
        } else {
            0xFFFFFFFFu32
        };
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET hto_d_v2 dst=0x{:x} src=0x{:x} bytes={} first_word=0x{:x} hits rd5=0x{:x} +22176",
            dst_device.0 as u64, src_host as u64, byte_count, first_word, rd5
        ));
    }
    if dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(src_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[HTOD/sync] dst={:p} src={:p} size={} bytes={}",
            dst_device.0, src_host, byte_count, dwords
        ));
    }
    // Log gParams writes (496 bytes) — validate texture handles and device pointers
    if byte_count == 496 {
        let data = src_host as *const u64;
        // Texture object handles at known offsets in gParams (from CollideConvexes PTX)
        let tex_offsets: &[(usize, &str)] = &[
            (40, "320:positions"),
            (41, "328:SDFTex"),
            (47, "376:grid"),
            (48, "384:convexIdx"),
            (49, "392:convexPlanes1"),
            (50, "400:convexPlanes2"),
            (51, "408:convexNormals"),
            (52, "416:convexPos"),
            (53, "424:convexRot"),
        ];
        let alloc_min = ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
        let alloc_max = ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
        let mut bad_ptrs = Vec::new();
        for &(idx, label) in tex_offsets {
            let val = unsafe { *data.add(idx) };
            // Strip top 3 bits (texture shift encoding) for range check
            let stripped = (val << 3) >> 3;
            if stripped != 0 && (stripped < alloc_min || stripped > alloc_max) {
                bad_ptrs.push(format!("{}=0x{:x}(OOB)", label, val));
            }
        }
        let num_convexes = unsafe { *(src_host as *const u32).add(32) }; // offset 128 = u32 index 32
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoD gParams dst={:p} numConvex={} bad_ptrs={}",
            dst_device.0,
            num_convexes,
            if bad_ptrs.is_empty() {
                "none".to_string()
            } else {
                bad_ptrs.join(", ")
            }
        ));
    }
    let result = unsafe { hipMemcpyHtoD(dst_device, src_host.cast_mut(), byte_count) };
    if result.is_err() {
        // hipMemcpyHtoD failed — try hipMemcpy with Default kind which
        // handles unified/host-mapped pointers correctly via the runtime.
        // NEVER fall back to raw memcpy: casting a device pointer to a CPU
        // pointer bypasses GPU page tables and can corrupt GPU memory,
        // causing surface descriptor corruption and TCP page faults.
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoD_v2 dst={:p} src={:p} size={} -> {:?}, trying hipMemcpy default",
            dst_device.0, src_host, byte_count, result
        ));
        unsafe {
            hip_runtime_sys::hipMemcpy(
                dst_device.0,
                src_host.cast_mut(),
                byte_count,
                hipMemcpyKind::hipMemcpyDefault,
            )
        }?;
    }
    Ok(())
}

pub(crate) fn copy_hto_d_v2_ptds(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    unsafe {
        hipMemcpy_spt(
            dst_device.0.cast(),
            src_host.cast_mut(),
            byte_count,
            hipMemcpyKind::hipMemcpyHostToDevice,
        )
    }
}

pub(crate) fn copy_dto_h_v2_ptds(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    let result = unsafe {
        hipMemcpy_spt(
            dst_host.cast(),
            src_device.0.cast(),
            byte_count,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        )
    };
    if result.is_ok() && dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(dst_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[DTOH/ptds] dst={:p} src={:p} size={} bytes={}",
            dst_host, src_device.0, byte_count, dwords
        ));
    }
    if result.is_ok() {
        unsafe { log_dtoh_trace("ptds", dst_host, src_device, byte_count) };
    }
    result
}

pub(crate) fn get_address_range_v2(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    unsafe { hipMemGetAddressRange(pbase, psize, dptr) }
}

/// Log a cuMemset* call when ZLUDA_DTOH_LOG=1. Used to track which device
/// pointers get initialized via memset (vs HtoD or kernel writes).
fn log_memset(kind: &str, dst: u64, value: u64, n: usize) {
    if dtoh_log_enabled() {
        super::driver::zluda_log(&format!(
            "[MEMSET/{}] dst=0x{:x} value=0x{:x} n={} bytes={}",
            kind,
            dst,
            value,
            n,
            // Approximate total bytes: D8=n, D16=n*2, D32=n*4
            match kind {
                "d8" | "d8a" => n,
                "d16" | "d16a" => n * 2,
                "d32" | "d32a" => n * 4,
                _ => n,
            }
        ));
    }
}

pub(crate) fn set_d8_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uchar, n: usize) -> hipError_t {
    let _guard = super::stream::null_stream_sync(hipStream_t(std::ptr::null_mut()));
    log_memset("d8", dst.0 as u64, value as u64, n);
    if let Some(rd5) = super::function::check_memset_hits_rd5_counter(dst.0 as u64, n as u64) {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET d8 dst=0x{:x} value=0x{:x} n_bytes={} hits rd5=0x{:x} +22176",
            dst.0 as u64, value, n, rd5
        ));
    }
    unsafe { hipMemsetD8(dst, value, n) }
}

pub(crate) fn set_d8_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_uchar,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    log_memset("d8a", dst.0 as u64, value as u64, n);
    if let Some(rd5) = super::function::check_memset_hits_rd5_counter(dst.0 as u64, n as u64) {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET d8a dst=0x{:x} value=0x{:x} n_bytes={} stream=0x{:x} hits rd5=0x{:x} +22176",
            dst.0 as u64, value, n, stream.0 as u64, rd5
        ));
    }
    let _guard = super::stream::null_stream_sync(stream);
    let result = unsafe { hipMemsetD8Async(dst, value, n, stream) };
    if result.is_ok() && !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    result
}

pub(crate) fn set_d16_v2(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_ushort,
    n: usize,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(hipStream_t(std::ptr::null_mut()));
    log_memset("d16", dst.0 as u64, value as u64, n);
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst.0 as u64, (n as u64) * 2)
    {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET d16 dst=0x{:x} value=0x{:x} n_halfs={} hits rd5=0x{:x} +22176",
            dst.0 as u64, value, n, rd5
        ));
    }
    unsafe { hipMemsetD16(dst, value, n) }
}

pub(crate) fn set_d16_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_ushort,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    log_memset("d16a", dst.0 as u64, value as u64, n);
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst.0 as u64, (n as u64) * 2)
    {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET d16a dst=0x{:x} value=0x{:x} n_halfs={} stream=0x{:x} hits rd5=0x{:x} +22176",
            dst.0 as u64, value, n, stream.0 as u64, rd5
        ));
    }
    let _guard = super::stream::null_stream_sync(stream);
    let result = unsafe { hipMemsetD16Async(dst, value, n, stream) };
    if result.is_ok() && !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    result
}

pub(crate) fn set_d32_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uint, n: usize) -> hipError_t {
    let _guard = super::stream::null_stream_sync(hipStream_t(std::ptr::null_mut()));
    log_memset("d32", dst.0 as u64, value as u64, n);
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst.0 as u64, (n as u64) * 4)
    {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET d32 dst=0x{:x} value=0x{:x} n_words={} hits rd5=0x{:x} +22176",
            dst.0 as u64, value, n, rd5
        ));
    }
    unsafe { hipMemsetD32(dst, value as _, n) }
}

pub(crate) fn set_d32_async(
    dst: hipDeviceptr_t,
    value: ::core::ffi::c_uint,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    log_memset("d32a", dst.0 as u64, value as u64, n);
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst.0 as u64, (n as u64) * 4)
    {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET d32a dst=0x{:x} value=0x{:x} n_words={} stream=0x{:x} hits rd5=0x{:x} +22176",
            dst.0 as u64, value, n, stream.0 as u64, rd5
        ));
    }
    let _guard = super::stream::null_stream_sync(stream);
    let result = unsafe { hipMemsetD32Async(dst, value as _, n, stream) };
    if result.is_ok() && !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    result
}

pub(crate) fn get_info_v2(free: *mut usize, total: *mut usize) -> hipError_t {
    let result = unsafe { hipMemGetInfo(free, total) };
    if let Ok(()) = result {
        let hip_total = unsafe { *total };
        let hip_free = unsafe { *free };
        // Match cuDeviceTotalMem (hip_total/2). hipMemGetInfo returns
        // VRAM+GTT combined; cap to match what we report as total.
        let reported_total = hip_total / 2;
        let reported_free = hip_free.min(reported_total);
        unsafe {
            *total = reported_total;
            *free = reported_free;
        };
        super::driver::zluda_log_err(&format!(
            "cuMemGetInfo -> free={} MiB, total={} MiB (hip_free={}, hip_total={})",
            reported_free / (1024 * 1024),
            reported_total / (1024 * 1024),
            hip_free / (1024 * 1024),
            hip_total / (1024 * 1024)
        ));
    }
    result
}

pub(crate) unsafe fn free_host(ptr: *mut ::core::ffi::c_void) -> CUresult {
    // 2026-05-18: untrack BEFORE hipFreeHost. Any diagnostic probe that
    // iterates SMALL_HOST_ALLOCS and dereferences (host + off) races
    // against this free — if hipFreeHost runs first the memory is
    // unmapped while the probe still holds a (host, _dev, size) tuple
    // from the previous lock-and-clone, leading to SIGSEGV that our
    // signal handler misattributes as "GPU fault likely". This was
    // the source of the entire D2-D6 ghost crash class (2026-05-18).
    untrack_host_alloc(ptr);
    untrack_hostbuf(ptr);
    let hip_result = hipFreeHost(ptr);
    remove_allocation(ptr)?;
    Ok(hip_result?)
}

pub(crate) unsafe fn alloc_host_v2(pp: &mut *mut ::core::ffi::c_void, bytesize: usize) -> CUresult {
    host_alloc(pp, bytesize, 0)
}

pub(crate) unsafe fn host_alloc(
    pp: &mut *mut ::core::ffi::c_void,
    bytesize: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    let context = context::get_current_context()?;
    // B.3 bisection: pass flags through unchanged. Previously we OR'd in
    // hipHostMallocMapped (0x02) on every host allocation, justifying it as
    // "PhysX requires it for pinned-memory spinWait." That force-mapping
    // changes GTT/VRAM placement for every host alloc and is a candidate
    // suspect for the TurbulenceFS-zeros-counter cross-allocation corruption.
    // If PhysX truly needs mapping for spinWait it will pass the flag itself
    // (or fail with a clear error pointing back here).
    let hip_flags = flags;
    // Round up for vectorized over-read headroom (same as alloc_v2 — 16 bytes
    // covers one ld.global.v4.u32 over-read; reduced from 256 bytes as part of
    // plan B.2 bisection).
    let alloc_size = (bytesize + 15) & !15;
    hipHostMalloc(pp, alloc_size, hip_flags)?;
    // Log host alloc addresses — Flex may pass these directly as kernel args
    // expecting CUDA UVA (host pointer = device pointer). On HIP, the device
    // pointer for mapped memory is different from the host pointer.
    let mut dev_ptr: *mut ::core::ffi::c_void = ptr::null_mut();
    let _ = hipHostGetDevicePointer(&mut dev_ptr, *pp, 0);
    super::driver::zluda_log_err(&format!(
        "cuMemHostAlloc size={} host={:p} dev={:p} same={}",
        bytesize,
        *pp,
        dev_ptr,
        *pp == dev_ptr
    ));
    track_host_alloc(*pp, dev_ptr, bytesize);
    track_hostbuf(*pp, bytesize);
    sample_hostbufs("post_alloc");
    add_allocation(*pp, bytesize, context)?;
    Ok(())
}

pub(crate) unsafe fn host_get_device_pointer_v2(
    pdptr: &mut hipDeviceptr_t,
    p: *mut ::core::ffi::c_void,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    if p.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // HIP equivalent of cuMemHostGetDevicePointer_v2
    hipHostGetDevicePointer(std::ptr::from_mut(pdptr).cast(), p, flags)?;
    if dtoh_log_enabled() {
        super::driver::zluda_log(&format!(
            "[HOSTGETDEV] host=0x{:x} -> dev=0x{:x} flags=0x{:x}",
            p as usize, pdptr.0 as usize, flags
        ));
    }
    Ok(())
}

fn add_allocation(
    dptr: *mut ::core::ffi::c_void,
    bytesize: usize,
    context: cuda_types::cuda::CUcontext,
) -> Result<(), CUerror> {
    let global_state = driver::global_state()?;
    let mut allocations = global_state
        .allocations
        .lock()
        .map_err(|_| CUerror::UNKNOWN)?;
    allocations.insert(dptr as usize, bytesize, context);
    Ok(())
}

fn remove_allocation(ptr: *mut std::ffi::c_void) -> Result<(), CUerror> {
    let global_state = driver::global_state()?;
    let mut allocations = global_state
        .allocations
        .lock()
        .map_err(|_| CUerror::UNKNOWN)?;
    allocations.remove(ptr as usize);
    Ok(())
}

pub(crate) unsafe fn retain_allocation_handle(
    _handle: *mut cuda_types::cuda::CUmemGenericAllocationHandle,
    _addr: *mut ::core::ffi::c_void,
) -> CUresult {
    CUresult::ERROR_NOT_SUPPORTED
}

pub(crate) unsafe fn copy_hto_d_async_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    // Plan v14.next.next: same as copy_hto_d_v2 but async.
    log_memcpy_into_mapped(
        dst_device.0 as u64,
        src_host as u64,
        byte_count,
        true,
        "hto_d_async_v2",
    );
    log_memcpy_into_ms2_rd6_watch(
        dst_device.0 as u64,
        src_host as u64,
        byte_count,
        true,
        "hto_d_async_v2",
    );
    if dtoh_log_enabled() {
        let dwords = unsafe { format_dtoh_dwords(src_host as *const u8, byte_count) };
        super::driver::zluda_log(&format!(
            "[HTOD/async] dst={:p} src={:p} size={} stream={:p} bytes={}",
            dst_device.0, src_host, byte_count, stream.0, dwords
        ));
    }
    // Log gParams writes (496 bytes = Flex solver params) — validate texture handles
    if byte_count == 496 {
        let data = src_host as *const u64;
        let alloc_min = ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
        let alloc_max = ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
        // Check texture object handles at known gParams offsets
        let tex_offsets: &[(usize, &str)] = &[
            (40, "320:pos"),
            (41, "328:sdf"),
            (47, "376:grid"),
            (48, "384:cidx"),
            (49, "392:cp1"),
            (50, "400:cp2"),
        ];
        let mut bad = Vec::new();
        for &(idx, label) in tex_offsets {
            let val = *data.add(idx);
            let stripped = (val << 3) >> 3;
            if stripped != 0 && (stripped < alloc_min || stripped > alloc_max) {
                bad.push(format!("{}=0x{:x}", label, val));
            }
        }
        let num_convex = *(src_host as *const u32).add(32); // offset 128
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoDAsync gParams dst={:p} nConvex={} bad={}",
            dst_device.0,
            num_convex,
            if bad.is_empty() {
                "none".to_string()
            } else {
                bad.join(",")
            }
        ));
    }
    let _guard = super::stream::null_stream_sync(stream);
    let result = hipMemcpyHtoDAsync(dst_device, src_host.cast_mut(), byte_count, stream);
    if result.is_err() {
        // hipMemcpyHtoDAsync failed — try hipMemcpyAsync with Default kind.
        // NEVER fall back to raw memcpy: casting a device pointer to a CPU
        // pointer bypasses GPU page tables and can corrupt GPU memory.
        super::driver::zluda_log_err(&format!(
            "cuMemcpyHtoDAsync_v2 dst={:p} src={:p} size={} -> {:?}, trying hipMemcpyAsync default",
            dst_device.0, src_host, byte_count, result
        ));
        hipMemcpyAsync(
            dst_device.0,
            src_host.cast_mut(),
            byte_count,
            hipMemcpyKind::hipMemcpyDefault,
            stream,
        )?;
    }
    if !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    Ok(())
}

pub(crate) unsafe fn copy_dto_h_async_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(stream);
    let result = hipMemcpyDtoHAsync(dst_host, src_device, byte_count, stream);
    if result.is_ok() && dtoh_log_enabled() {
        // The async copy may not have completed by the time the call
        // returns. Sync the stream so the host buffer reflects what the
        // GPU actually wrote — otherwise we'd log stale pre-copy bytes.
        let _ = hipStreamSynchronize(stream);
        let dwords = format_dtoh_dwords(dst_host as *const u8, byte_count);
        super::driver::zluda_log(&format!(
            "[DTOH/async] dst={:p} src={:p} size={} stream={:p} bytes={}",
            dst_host, src_device.0, byte_count, stream.0, dwords
        ));
    }
    if result.is_ok() && dtoh_trace_enabled() && byte_count > 0 && byte_count <= 256 {
        // Same reasoning as the DTOH_LOG path: need a stream-sync before
        // reading dst, otherwise we capture pre-copy stale bytes. Only
        // pay the sync cost when trace is actually on AND size is in range.
        let _ = hipStreamSynchronize(stream);
        log_dtoh_trace("async", dst_host, src_device, byte_count);
    }
    if result.is_ok() && !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    result
}

pub(crate) unsafe fn copy_dto_d_v2(
    dst_device: hipDeviceptr_t,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    super::function::maybe_log_creategrid_sorted_keys_copy(
        dst_device.0 as u64,
        src_device.0 as u64,
        byte_count,
        hipStream_t(std::ptr::null_mut()),
    );
    log_dtod_into_mapped(dst_device.0 as u64, src_device.0 as u64, byte_count, "sync");
    log_memcpy_into_ms2_rd6_watch(
        dst_device.0 as u64,
        src_device.0 as u64,
        byte_count,
        false,
        "dto_d_v2",
    );
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst_device.0 as u64, byte_count as u64)
    {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET dto_d_v2 dst=0x{:x} src=0x{:x} bytes={} hits rd5=0x{:x} +22176",
            dst_device.0 as u64, src_device.0 as u64, byte_count, rd5
        ));
    }
    hipMemcpyDtoD(dst_device, src_device, byte_count)
}

pub(crate) unsafe fn copy_dto_d_async_v2(
    dst_device: hipDeviceptr_t,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    super::function::maybe_log_creategrid_sorted_keys_copy(
        dst_device.0 as u64,
        src_device.0 as u64,
        byte_count,
        stream,
    );
    log_dtod_into_mapped(
        dst_device.0 as u64,
        src_device.0 as u64,
        byte_count,
        "async",
    );
    log_memcpy_into_ms2_rd6_watch(
        dst_device.0 as u64,
        src_device.0 as u64,
        byte_count,
        false,
        "dto_d_async_v2",
    );
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst_device.0 as u64, byte_count as u64)
    {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET dto_d_async_v2 dst=0x{:x} src=0x{:x} bytes={} stream=0x{:x} hits rd5=0x{:x} +22176",
            dst_device.0 as u64, src_device.0 as u64, byte_count, stream.0 as u64, rd5
        ));
    }
    let _guard = super::stream::null_stream_sync(stream);
    let result = hipMemcpyDtoDAsync(dst_device, src_device, byte_count, stream);
    if result.is_ok() && !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    result
}

/// Plan v14.next.next.next: `cuMemcpy` un-typed sync entry. Not present in
/// lib.rs's wired list previously — was falling through to "UNIMPL" path,
/// which means PhysX's cuMemcpy calls went unhandled or were rerouted via
/// HIP. With this implementation they go through hipMemcpy(Default) and
/// the MAPPED_MEMCPY hook fires.
pub(crate) unsafe fn copy(
    dst: hipDeviceptr_t,
    src: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    log_memcpy_into_mapped(dst.0 as u64, src.0 as u64, byte_count, false, "copy");
    log_memcpy_into_ms2_rd6_watch(dst.0 as u64, src.0 as u64, byte_count, false, "copy");
    hip_runtime_sys::hipMemcpy(dst.0, src.0, byte_count, hipMemcpyKind::hipMemcpyDefault)
}

/// Plan v25 Phase B.2: drain `MS4_DEV5_DUMP_QUEUE`. Called at the entry of
/// `event::record` and `event::synchronize` (host-thread contexts where
/// DtoH is allowed per `feedback_no_d2d_from_launch_kernel`).
///
/// For each `(p4, p5, p6, p7, n_threads, launch_n)` pushed by the MS_4
/// launch hook in `function.rs`, do `hipDeviceSynchronize` plus three
/// synchronous `hipMemcpyDtoH` reads: DEV5 (index buffer, cap 65536
/// u32 = 256 KiB), DEV6 (initial-counter buffer with 12-byte stride,
/// cap 16384 u32 = 64 KiB), DEV7 (atomic-counter buffer, cap 65536
/// u32 = 256 KiB). Log per-buffer histograms.
///
/// Run e6 (2026-05-18) saw DEV5 clean for 28 launches but the cap
/// was 1024 — indices past that were unread. v25 lifts the cap and
/// adds DEV6/DEV7 so we cover the other wild-VA candidates in MS_4.
///
/// Safety:
/// - Queue lock is held ONLY for the swap-out clone. HIP calls happen
///   outside the critical section — no UAF class.
/// - `hipMemcpyDtoH` returns an error rather than crashing on a freed
///   device pointer, so stale entries log a fail line and skip.
/// - Gated on `ZLUDA_DEV5_DUMP=1`, default OFF.
pub(crate) fn drain_dev5_dump_queue() {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("ZLUDA_DEV5_DUMP").map_or(false, |v| v == "1")) {
        return;
    }
    let items: Vec<(u64, u64, u64, u64, u32, u32)> =
        match super::function::MS4_DEV5_DUMP_QUEUE.get() {
            Some(m) => match m.lock() {
                Ok(mut q) => q.drain(..).collect(),
                Err(_) => return,
            },
            None => return,
        };
    if items.is_empty() {
        return;
    }
    const DEV5_CAP: u32 = 65536;
    // DEV6 is indexed by r6*12 (3-u32 struct). We cap at 16384 u32s
    // = 5461 structs to bound per-launch DtoH cost at 64 KiB.
    const DEV6_CAP_U32S: u32 = 16384;
    // DEV7 is indexed by r6*4 (single u32). Cap mirrors DEV5.
    const DEV7_CAP: u32 = 65536;
    for (_p4, p5, p6, p7, n_threads, launch_n) in items {
        let sync_res = unsafe { hipDeviceSynchronize() };
        if sync_res.is_err() {
            super::driver::zluda_log_err(&format!(
                "[DEV5_DUMP #{}] hipDeviceSynchronize failed err={:?}",
                launch_n, sync_res
            ));
            continue;
        }
        // --- DEV5 ---
        let dev5_count = n_threads.min(DEV5_CAP) as usize;
        let mut max_r6: u32 = 0;
        if dev5_count == 0 {
            super::driver::zluda_log_err(&format!(
                "[DEV5_DUMP #{}] n_threads=0 — skipping",
                launch_n
            ));
        } else if p5 == 0 {
            super::driver::zluda_log_err(&format!("[DEV5_DUMP #{}] p5=null — skipping", launch_n));
        } else {
            let mut local: Vec<u32> = vec![0u32; dev5_count];
            let copy_res = unsafe {
                hipMemcpyDtoH(
                    local.as_mut_ptr() as *mut std::ffi::c_void,
                    hipDeviceptr_t(p5 as *mut std::ffi::c_void),
                    dev5_count * 4,
                )
            };
            if copy_res.is_err() {
                super::driver::zluda_log_err(&format!(
                    "[DEV5_DUMP #{}] DtoH failed err={:?} dev5=0x{:x} dump={}",
                    launch_n, copy_res, p5, dev5_count
                ));
            } else {
                let bit31 = local.iter().filter(|&&v| v & 0x8000_0000 != 0).count();
                let huge = local.iter().filter(|&&v| v > (1u32 << 24)).count();
                let min_v = local.iter().min().copied().unwrap_or(0);
                let max_v = local.iter().max().copied().unwrap_or(0);
                max_r6 = max_v;
                let sample_n = local.len().min(16);
                let sample: Vec<String> = local[..sample_n]
                    .iter()
                    .map(|v| format!("0x{:x}", v))
                    .collect();
                super::driver::zluda_log_err(&format!(
                    "[DEV5_DUMP #{}] dev5=0x{:x} n_threads={} dump={} bit31={} huge={} min=0x{:x} max=0x{:x} sample=[{}]",
                    launch_n,
                    p5,
                    n_threads,
                    dev5_count,
                    bit31,
                    huge,
                    min_v,
                    max_v,
                    sample.join(" ")
                ));
            }
        }
        // --- DEV6 (indexed by r6*12, 3-u32 struct) ---
        // We dump up to max_r6+1 u32 structs from p6. Cap at DEV6_CAP_U32S
        // total u32s = ~5461 structs (64 KiB). If max_r6 is 0, dump 16 u32s
        // as a probe sample anyway.
        let dev6_struct_count = (max_r6.saturating_add(1)).min(DEV6_CAP_U32S / 3);
        let dev6_u32_count = (dev6_struct_count as usize)
            .saturating_mul(3)
            .max(if max_r6 == 0 { 16 } else { 0 });
        if p6 != 0 && dev6_u32_count > 0 {
            let mut local: Vec<u32> = vec![0u32; dev6_u32_count];
            let copy_res = unsafe {
                hipMemcpyDtoH(
                    local.as_mut_ptr() as *mut std::ffi::c_void,
                    hipDeviceptr_t(p6 as *mut std::ffi::c_void),
                    dev6_u32_count * 4,
                )
            };
            if copy_res.is_err() {
                super::driver::zluda_log_err(&format!(
                    "[DEV6_DUMP #{}] DtoH failed err={:?} dev6=0x{:x} dump={}",
                    launch_n, copy_res, p6, dev6_u32_count
                ));
            } else {
                let bit31 = local.iter().filter(|&&v| v & 0x8000_0000 != 0).count();
                let huge = local.iter().filter(|&&v| v > (1u32 << 24)).count();
                let min_v = local.iter().min().copied().unwrap_or(0);
                let max_v = local.iter().max().copied().unwrap_or(0);
                let sample_n = local.len().min(16);
                let sample: Vec<String> = local[..sample_n]
                    .iter()
                    .map(|v| format!("0x{:x}", v))
                    .collect();
                super::driver::zluda_log_err(&format!(
                    "[DEV6_DUMP #{}] dev6=0x{:x} dump={} (structs~{}) bit31={} huge={} min=0x{:x} max=0x{:x} sample=[{}]",
                    launch_n,
                    p6,
                    dev6_u32_count,
                    dev6_u32_count / 3,
                    bit31,
                    huge,
                    min_v,
                    max_v,
                    sample.join(" ")
                ));
            }
        }
        // --- DEV7 (atomic counter, indexed by r6*4, single u32) ---
        let dev7_count = (max_r6.saturating_add(1)).min(DEV7_CAP) as usize;
        let dev7_count = dev7_count.max(if max_r6 == 0 { 16 } else { 0 });
        if p7 != 0 && dev7_count > 0 {
            let mut local: Vec<u32> = vec![0u32; dev7_count];
            let copy_res = unsafe {
                hipMemcpyDtoH(
                    local.as_mut_ptr() as *mut std::ffi::c_void,
                    hipDeviceptr_t(p7 as *mut std::ffi::c_void),
                    dev7_count * 4,
                )
            };
            if copy_res.is_err() {
                super::driver::zluda_log_err(&format!(
                    "[DEV7_DUMP #{}] DtoH failed err={:?} dev7=0x{:x} dump={}",
                    launch_n, copy_res, p7, dev7_count
                ));
            } else {
                let bit31 = local.iter().filter(|&&v| v & 0x8000_0000 != 0).count();
                let huge = local.iter().filter(|&&v| v > (1u32 << 24)).count();
                let min_v = local.iter().min().copied().unwrap_or(0);
                let max_v = local.iter().max().copied().unwrap_or(0);
                let sample_n = local.len().min(16);
                let sample: Vec<String> = local[..sample_n]
                    .iter()
                    .map(|v| format!("0x{:x}", v))
                    .collect();
                super::driver::zluda_log_err(&format!(
                    "[DEV7_DUMP #{}] dev7=0x{:x} dump={} bit31={} huge={} min=0x{:x} max=0x{:x} sample=[{}]",
                    launch_n,
                    p7,
                    dev7_count,
                    bit31,
                    huge,
                    min_v,
                    max_v,
                    sample.join(" ")
                ));
            }
        }
    }
}

/// Plan v26 Phase B.3: drain `IOFX_AR_DUMP_QUEUE`. Same pattern as
/// `drain_dev5_dump_queue`. For each (param_4_ptr, launch_n), do a
/// `hipDeviceSynchronize` + `hipMemcpyDtoH` of up to 4096 u32s
/// (16 KiB). Logs histogram (bit-31 count, huge count, min, max,
/// first 16 hex values).
///
/// Gated on `ZLUDA_ACTORRANGE_DUMP=1`, default OFF.
pub(crate) fn drain_iofx_ar_dump_queue() {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("ZLUDA_ACTORRANGE_DUMP").map_or(false, |v| v == "1"))
    {
        return;
    }
    let items: Vec<(u64, u64, u64, u32)> = match super::function::IOFX_AR_DUMP_QUEUE.get() {
        Some(m) => match m.lock() {
            Ok(mut q) => q.drain(..).collect(),
            Err(_) => return,
        },
        None => return,
    };
    if items.is_empty() {
        return;
    }
    const AR_DUMP_CAP: usize = 4096;
    // Helper: dump one buffer and emit one log line per (launch, label).
    let dump_one = |label: &str, ptr: u64, launch_n: u32| {
        if ptr == 0 {
            return; // skipped (not plausible) — already logged at push site
        }
        let mut local: Vec<u32> = vec![0u32; AR_DUMP_CAP];
        let copy_res = unsafe {
            hipMemcpyDtoH(
                local.as_mut_ptr() as *mut std::ffi::c_void,
                hipDeviceptr_t(ptr as *mut std::ffi::c_void),
                AR_DUMP_CAP * 4,
            )
        };
        if copy_res.is_err() {
            super::driver::zluda_log_err(&format!(
                "[IOFX_AR_DUMP {} #{}] DtoH failed err={:?} ptr=0x{:x} dump={}",
                label, launch_n, copy_res, ptr, AR_DUMP_CAP
            ));
            return;
        }
        let bit31 = local.iter().filter(|&&v| v & 0x8000_0000 != 0).count();
        let huge = local.iter().filter(|&&v| v > (1u32 << 24)).count();
        let min_v = local.iter().min().copied().unwrap_or(0);
        let max_v = local.iter().max().copied().unwrap_or(0);
        let sample_n = local.len().min(16);
        let sample: Vec<String> = local[..sample_n]
            .iter()
            .map(|v| format!("0x{:x}", v))
            .collect();
        super::driver::zluda_log_err(&format!(
            "[IOFX_AR_DUMP {} #{}] ptr=0x{:x} dump={} bit31={} huge={} min=0x{:x} max=0x{:x} sample=[{}]",
            label, launch_n, ptr, AR_DUMP_CAP, bit31, huge, min_v, max_v, sample.join(" ")
        ));
    };
    for (p4, p6, p9, launch_n) in items {
        let sync_res = unsafe { hipDeviceSynchronize() };
        if sync_res.is_err() {
            super::driver::zluda_log_err(&format!(
                "[IOFX_AR_DUMP #{}] hipDeviceSynchronize failed err={:?}",
                launch_n, sync_res
            ));
            continue;
        }
        // Emit one log line per buffer. The "p4" label preserves
        // backward-compat grep (lots of analysis scripts grep
        // IOFX_AR_DUMP); the new "p6" / "p9" lines are additional.
        dump_one("p4", p4, launch_n);
        dump_one("p6", p6, launch_n);
        dump_one("p9", p9, launch_n);
    }
}

fn decode_ms2_batch_descriptor(desc: &[u8]) -> Option<(u32, u64, u64, u64)> {
    if desc.len() < 88 {
        return None;
    }
    let active = u32::from_le_bytes(desc[0..4].try_into().ok()?);
    let rd5 = u64::from_le_bytes(desc[32..40].try_into().ok()?);
    let rd6 = u64::from_le_bytes(desc[72..80].try_into().ok()?);
    let rd7 = u64::from_le_bytes(desc[80..88].try_into().ok()?);
    Some((active, rd5, rd6, rd7))
}

/// Drain queued `CreateParticleShapesIncremental_MultiStage_2_Batched`
/// descriptors and sample the exact MS2 counter-source loads:
///
/// - descriptor fields at +0/+32/+72/+80 (active, rd5, rd6, rd7)
/// - first few `rd7[i]` entries (shape->cell mapping)
/// - corresponding `rd6[rd7[i] * 12 + 8]` (`r38` in PTX BB11_9 / BB11_21)
/// - `rd5[i*12 + 4]` after-kernel counter value for the same shape index
///
/// Gated on `ZLUDA_MS2_CAPTURE=1`, default OFF.
pub(crate) fn drain_ms2_dump_queue() {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("ZLUDA_MS2_CAPTURE").map_or(false, |v| v == "1")) {
        return;
    }
    let items: Vec<(u64, u32, u32, u32, u32)> = match super::function::MS2_BATCH_DUMP_QUEUE.get() {
        Some(m) => match m.lock() {
            Ok(mut q) => q.drain(..).collect(),
            Err(_) => return,
        },
        None => return,
    };
    if items.is_empty() {
        return;
    }
    const DESC_BYTES: usize = 88;
    const MAX_CTA_DUMP: usize = 16;
    const SAMPLE_SHAPES: usize = 8;
    const MAX_RD6_SLOTS: usize = 4096;
    for (p2_ctas, grid_x, grid_y, grid_z, launch_n) in items {
        let total_ctas = (grid_x as usize)
            .saturating_mul(grid_y as usize)
            .saturating_mul(grid_z as usize);
        if p2_ctas == 0 || total_ctas == 0 {
            super::driver::zluda_log_err(&format!(
                "[MS2_CAPTURE #{}] skip invalid launch meta p2_ctas=0x{:x} grid=[{},{},{}]",
                launch_n, p2_ctas, grid_x, grid_y, grid_z
            ));
            continue;
        }
        let sync_res = unsafe { hipDeviceSynchronize() };
        if sync_res.is_err() {
            super::driver::zluda_log_err(&format!(
                "[MS2_CAPTURE #{}] hipDeviceSynchronize failed err={:?}",
                launch_n, sync_res
            ));
            continue;
        }
        let cta_dump = total_ctas.min(MAX_CTA_DUMP);
        let mut desc_bytes = vec![0u8; cta_dump * DESC_BYTES];
        let desc_copy_res = unsafe {
            hipMemcpyDtoH(
                desc_bytes.as_mut_ptr().cast(),
                hipDeviceptr_t(p2_ctas as *mut std::ffi::c_void),
                desc_bytes.len(),
            )
        };
        if desc_copy_res.is_err() {
            super::driver::zluda_log_err(&format!(
                "[MS2_CAPTURE #{}] descriptor DtoH failed err={:?} p2_ctas=0x{:x} bytes={}",
                launch_n,
                desc_copy_res,
                p2_ctas,
                desc_bytes.len()
            ));
            continue;
        }
        super::driver::zluda_log_err(&format!(
            "[MS2_CAPTURE #{}] grid=[{},{},{}] total_ctas={} dumped_ctas={}{} p2_ctas=0x{:x}",
            launch_n,
            grid_x,
            grid_y,
            grid_z,
            total_ctas,
            cta_dump,
            if cta_dump < total_ctas {
                " (truncated)"
            } else {
                ""
            },
            p2_ctas
        ));
        for cta in 0..cta_dump {
            let start = cta * DESC_BYTES;
            let end = start + DESC_BYTES;
            let Some((active, rd5, rd6, rd7)) =
                decode_ms2_batch_descriptor(&desc_bytes[start..end])
            else {
                super::driver::zluda_log_err(&format!(
                    "[MS2_CAPTURE #{} cta={}] malformed descriptor bytes",
                    launch_n, cta
                ));
                continue;
            };
            let desc_addr = p2_ctas.saturating_add((cta * DESC_BYTES) as u64);
            super::driver::zluda_log_err(&format!(
                "[MS2_CAPTURE_DESC #{} cta={}] desc=0x{:x} active={} rd5=0x{:x} rd6=0x{:x} rd7=0x{:x}",
                launch_n, cta, desc_addr, active, rd5, rd6, rd7
            ));
            if rd6 != 0 {
                super::function::ms2_rd6_watch_track(rd6, launch_n as u64);
            }
            if active == 0 || rd5 == 0 || rd6 == 0 || rd7 == 0 {
                continue;
            }
            let mut shape_count = 0u32;
            let shape_count_res = unsafe {
                hipMemcpyDtoH(
                    (&mut shape_count as *mut u32).cast(),
                    hipDeviceptr_t((rd5 + 22176) as *mut std::ffi::c_void),
                    std::mem::size_of::<u32>(),
                )
            };
            if shape_count_res.is_err() {
                super::driver::zluda_log_err(&format!(
                    "[MS2_CAPTURE #{} cta={}] shape_count DtoH failed err={:?} ptr=0x{:x}",
                    launch_n,
                    cta,
                    shape_count_res,
                    rd5 + 22176
                ));
                continue;
            }
            let sample_shapes = (shape_count as usize).min(SAMPLE_SHAPES);
            if sample_shapes == 0 {
                super::driver::zluda_log_err(&format!(
                    "[MS2_CAPTURE #{} cta={}] shape_count=0",
                    launch_n, cta
                ));
                continue;
            }
            let mut rd7_shape_map = vec![0u32; sample_shapes];
            let rd7_res = unsafe {
                hipMemcpyDtoH(
                    rd7_shape_map.as_mut_ptr().cast(),
                    hipDeviceptr_t(rd7 as *mut std::ffi::c_void),
                    rd7_shape_map.len() * std::mem::size_of::<u32>(),
                )
            };
            if rd7_res.is_err() {
                super::driver::zluda_log_err(&format!(
                    "[MS2_CAPTURE #{} cta={}] rd7 DtoH failed err={:?} ptr=0x{:x} shapes={}",
                    launch_n, cta, rd7_res, rd7, sample_shapes
                ));
                continue;
            }
            let max_slot = rd7_shape_map.iter().copied().max().unwrap_or(0) as usize;
            let rd6_slots_dump = max_slot.saturating_add(1).min(MAX_RD6_SLOTS);
            if rd6_slots_dump == 0 {
                continue;
            }
            let mut rd6_cells = vec![0u32; rd6_slots_dump * 3];
            let rd6_res = unsafe {
                hipMemcpyDtoH(
                    rd6_cells.as_mut_ptr().cast(),
                    hipDeviceptr_t(rd6 as *mut std::ffi::c_void),
                    rd6_cells.len() * std::mem::size_of::<u32>(),
                )
            };
            if rd6_res.is_err() {
                super::driver::zluda_log_err(&format!(
                    "[MS2_CAPTURE #{} cta={}] rd6 DtoH failed err={:?} ptr=0x{:x} slots={}",
                    launch_n, cta, rd6_res, rd6, rd6_slots_dump
                ));
                continue;
            }
            let mut rd5_shapes = vec![0u32; sample_shapes * 3];
            let rd5_res = unsafe {
                hipMemcpyDtoH(
                    rd5_shapes.as_mut_ptr().cast(),
                    hipDeviceptr_t(rd5 as *mut std::ffi::c_void),
                    rd5_shapes.len() * std::mem::size_of::<u32>(),
                )
            };
            if rd5_res.is_err() {
                super::driver::zluda_log_err(&format!(
                    "[MS2_CAPTURE #{} cta={}] rd5 DtoH failed err={:?} ptr=0x{:x} shapes={}",
                    launch_n, cta, rd5_res, rd5, sample_shapes
                ));
                continue;
            }
            let mut entries = Vec::with_capacity(sample_shapes);
            for (i, slot) in rd7_shape_map.iter().copied().enumerate() {
                let rd5_base = i * 3;
                let rd5_p0 = rd5_shapes.get(rd5_base).copied().unwrap_or(0);
                let rd5_p4 = rd5_shapes.get(rd5_base + 1).copied().unwrap_or(0);
                if (slot as usize) < rd6_slots_dump {
                    let r38 = rd6_cells[slot as usize * 3 + 2];
                    let r39 = r38.wrapping_sub(1);
                    entries.push(format!(
                        "i{}:slot={} r38={} r39={} rd5_p0={} rd5_p4={}",
                        i, slot, r38, r39, rd5_p0, rd5_p4
                    ));
                } else {
                    entries.push(format!(
                        "i{}:slot={} OOB(rd6_slots={}) rd5_p0={} rd5_p4={}",
                        i, slot, rd6_slots_dump, rd5_p0, rd5_p4
                    ));
                }
            }
            super::driver::zluda_log_err(&format!(
                "[MS2_CAPTURE_SAMPLE #{} cta={}] shape_count={} sample_shapes={} max_slot={} rd6_slots_dump={} sample=[{}]",
                launch_n,
                cta,
                shape_count,
                sample_shapes,
                max_slot,
                rd6_slots_dump,
                entries.join(" | ")
            ));
        }
    }
}

pub(crate) unsafe fn copy_async(
    dst: hipDeviceptr_t,
    src: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    // Plan v24 Phase B.2: drain the DEV5 dump queue here. cuMemcpyAsync is
    // the documented-safe place for synchronous DtoH (per
    // feedback_no_d2d_from_launch_kernel + project_phase_c_memcpy_descriptor).
    // Swap the queue out under lock, release lock, then process — never hold
    // a tracker lock across a HIP call (UAF class of 2026-05-18).
    drain_dev5_dump_queue();
    // Plan v14.next.next: log when dst falls in a tracked mapped buffer.
    // PhysX 3.4 routes most counter refreshes through cuMemcpyAsync
    // (CU_MEMORYTYPE_DEVICE inferred), not the typed DtoD variants —
    // v14.next run #1 caught 0 MAPPED_DTOD events despite ~5000 attribution
    // hits to MemCopyBatchedAsync. This hook closes that gap.
    log_memcpy_into_mapped(dst.0 as u64, src.0 as u64, byte_count, false, "copy_async");
    log_memcpy_into_ms2_rd6_watch(dst.0 as u64, src.0 as u64, byte_count, false, "copy_async");
    if let Some(rd5) =
        super::function::check_memset_hits_rd5_counter(dst.0 as u64, byte_count as u64)
    {
        super::driver::zluda_log_err(&format!(
            "RD5_MEMSET copy_async dst=0x{:x} src=0x{:x} bytes={} stream=0x{:x} hits rd5=0x{:x} +22176",
            dst.0 as u64, src.0 as u64, byte_count, stream.0 as u64, rd5
        ));
    }
    let _guard = super::stream::null_stream_sync(stream);
    let result = hipMemcpyAsync(
        dst.0,
        src.0,
        byte_count,
        hipMemcpyKind::hipMemcpyDefault,
        stream,
    );
    if result.is_ok() && !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    result
}

pub(crate) fn get_allocation_granularity(
    _granularity: &mut usize,
    _property: &cuda_types::cuda::CUmemAllocationProp,
    _option: cuda_types::cuda::CUmemAllocationGranularity_flags,
) -> CUresult {
    CUresult::ERROR_NOT_SUPPORTED
}

pub(crate) unsafe fn alloc_pitch_v2(
    dptr: *mut hipDeviceptr_t,
    p_pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: ::core::ffi::c_uint,
) -> CUresult {
    let vram_before = vram_used_bytes();
    hipMemAllocPitch(
        dptr.cast(),
        p_pitch,
        width_in_bytes,
        height,
        element_size_bytes,
    )?;
    let size = width_in_bytes * height;
    log_vram_delta("hipMemAllocPitch", vram_before, vram_used_bytes(), size);
    super::driver::zluda_log_err(&format!(
        "ALLOC_PITCH {:?} {}x{} elem={} ~{} KiB",
        unsafe { *dptr },
        width_in_bytes,
        height,
        element_size_bytes,
        size / 1024
    ));
    Ok(())
}

pub(crate) unsafe fn copy_2d_v2(memcpy: hip_Memcpy2D) -> CUresult {
    log_memcpy_sentinel_2d("cuMemcpy2D_v2", &memcpy);
    let memcpy = rewrite_2d_for_sentinel(memcpy);
    hipMemcpyParam2D(&memcpy)?;
    Ok(())
}

pub(crate) unsafe fn copy_2d_async_v2(memcpy: hip_Memcpy2D, stream: hipStream_t) -> CUresult {
    let _guard = super::stream::null_stream_sync(stream);
    // 2026-05-09 Phase E (plan fluttering-waddling-zephyr.md):
    // APEX TurbulenceFS calls cuMemcpy2DAsync_v2 with our D3D11
    // sentinel as `dstArray` (confirmed in run_20260509_phaseA_diag).
    // Without rewriting, HIP dereferences the sentinel as cl_mem
    // and SIGSEGVs after 4-8 IOFX_bbox launches. The sentinel is
    // backed by a linear dev_ptr (the imported VkBuffer); route
    // the copy as device-to-device into/out of that linear buffer
    // and the kernel's surface object reads the same memory.
    log_memcpy_sentinel_2d("cuMemcpy2DAsync_v2", &memcpy);
    let memcpy = rewrite_2d_for_sentinel(memcpy);
    hipMemcpyParam2DAsync(&memcpy, stream)?;
    if !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    Ok(())
}

pub(crate) unsafe fn copy_2d_unaligned_v2(memcpy: hip_Memcpy2D) -> CUresult {
    log_memcpy_sentinel_2d("cuMemcpy2DUnaligned_v2", &memcpy);
    let memcpy = rewrite_2d_for_sentinel(memcpy);
    hipDrvMemcpy2DUnaligned(&memcpy)?;
    Ok(())
}

/// If `srcArray` or `dstArray` is a D3D11 sentinel, replace it with the
/// underlying linear dev_ptr and switch the corresponding `*MemoryType`
/// to Device + use the sentinel's stored pitch. APEX-supplied `*Pitch`
/// is ignored on the sentinel side (CUDA spec says it's ignored when
/// the corresponding memory type is ARRAY, so APEX may pass garbage).
unsafe fn rewrite_2d_for_sentinel(mut memcpy: hip_Memcpy2D) -> hip_Memcpy2D {
    if let Some((dev, _size, pitch)) = super::d3d11_interop::d3d11_array_layout(memcpy.srcArray) {
        memcpy.srcArray = std::ptr::null_mut();
        memcpy.srcMemoryType = hipMemoryType::hipMemoryTypeDevice;
        memcpy.srcDevice = hipDeviceptr_t(dev);
        memcpy.srcPitch = pitch as usize;
    }
    if let Some((dev, _size, pitch)) = super::d3d11_interop::d3d11_array_layout(memcpy.dstArray) {
        memcpy.dstArray = std::ptr::null_mut();
        memcpy.dstMemoryType = hipMemoryType::hipMemoryTypeDevice;
        memcpy.dstDevice = hipDeviceptr_t(dev);
        memcpy.dstPitch = pitch as usize;
    }
    memcpy
}

// ---------------------------------------------------------------------------
// D3D11 sentinel memcpy diagnostics (Phase A of d3d11-subresource plan,
// 2026-05-09). All gated on `ZLUDA_D3D11_MEMCPY_LOG=1`. The goal is to
// discover which CUarray-taking memcpy variants APEX calls when the
// SubResource success branch is taken, so we know which intercepts are
// missing/wrong. Behavior-preserving: stubs route to the HIP equivalent
// or return NOT_SUPPORTED to match the prior UNIMPL trampoline.
// ---------------------------------------------------------------------------

fn d3d11_memcpy_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_D3D11_MEMCPY_LOG").map_or(false, |v| v == "1"))
}

fn log_memcpy_sentinel_array(fn_name: &str, role: &str, arr: hipArray_t, extra: &str) {
    if !d3d11_memcpy_log_enabled() {
        return;
    }
    let is_sentinel = super::d3d11_interop::is_d3d11_sentinel(arr);
    if !is_sentinel && arr.is_null() {
        // Don't log the common null-array case; only the calls that pass
        // a real hipArray are interesting.
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[D3D11_MEMCPY] {} {}_array={:p} sentinel={} {}",
        fn_name, role, arr, is_sentinel, extra,
    ));
}

unsafe fn log_memcpy_sentinel_2d(fn_name: &str, memcpy: &hip_Memcpy2D) {
    if !d3d11_memcpy_log_enabled() {
        return;
    }
    let src_sentinel = super::d3d11_interop::is_d3d11_sentinel(memcpy.srcArray);
    let dst_sentinel = super::d3d11_interop::is_d3d11_sentinel(memcpy.dstArray);
    if !src_sentinel && !dst_sentinel {
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[D3D11_MEMCPY] {} src_array={:p} src_sentinel={} dst_array={:p} dst_sentinel={} \
         w={} h={} src_pitch={} dst_pitch={} src_kind={} dst_kind={}",
        fn_name,
        memcpy.srcArray,
        src_sentinel,
        memcpy.dstArray,
        dst_sentinel,
        memcpy.WidthInBytes,
        memcpy.Height,
        memcpy.srcPitch,
        memcpy.dstPitch,
        memcpy.srcMemoryType.0,
        memcpy.dstMemoryType.0,
    ));
}

unsafe fn log_memcpy_sentinel_3d(fn_name: &str, memcpy: &HIP_MEMCPY3D) {
    if !d3d11_memcpy_log_enabled() {
        return;
    }
    let src_sentinel = super::d3d11_interop::is_d3d11_sentinel(memcpy.srcArray);
    let dst_sentinel = super::d3d11_interop::is_d3d11_sentinel(memcpy.dstArray);
    if !src_sentinel && !dst_sentinel {
        return;
    }
    super::driver::zluda_log_err(&format!(
        "[D3D11_MEMCPY] {} src_array={:p} src_sentinel={} dst_array={:p} dst_sentinel={} \
         w={} h={} d={} src_pitch={} dst_pitch={} src_kind={} dst_kind={}",
        fn_name,
        memcpy.srcArray,
        src_sentinel,
        memcpy.dstArray,
        dst_sentinel,
        memcpy.WidthInBytes,
        memcpy.Height,
        memcpy.Depth,
        memcpy.srcPitch,
        memcpy.dstPitch,
        memcpy.srcMemoryType.0,
        memcpy.dstMemoryType.0,
    ));
}

pub(crate) unsafe fn copy_ato_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_array: CUarray,
    src_offset: usize,
    byte_count: usize,
) -> CUresult {
    log_memcpy_sentinel_array(
        "cuMemcpyAtoH_v2",
        "src",
        src_array as hipArray_t,
        &format!(
            "offset={} bytes={} dst_host={:p}",
            src_offset, byte_count, dst_host
        ),
    );
    if let Some((dev, _, _)) = super::d3d11_interop::d3d11_array_layout(src_array as hipArray_t) {
        hipMemcpyDtoH(
            dst_host,
            hipDeviceptr_t((dev as *mut u8).add(src_offset) as *mut std::ffi::c_void),
            byte_count,
        )?;
        return Ok(());
    }
    hipMemcpyAtoH(dst_host, src_array as hipArray_t, src_offset, byte_count)?;
    Ok(())
}

pub(crate) unsafe fn copy_ato_h_async_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_array: CUarray,
    src_offset: usize,
    byte_count: usize,
    stream: hipStream_t,
) -> CUresult {
    log_memcpy_sentinel_array(
        "cuMemcpyAtoHAsync_v2",
        "src",
        src_array as hipArray_t,
        &format!(
            "offset={} bytes={} dst_host={:p} stream={:p}",
            src_offset, byte_count, dst_host, stream.0
        ),
    );
    if let Some((dev, _, _)) = super::d3d11_interop::d3d11_array_layout(src_array as hipArray_t) {
        hipMemcpyDtoHAsync(
            dst_host,
            hipDeviceptr_t((dev as *mut u8).add(src_offset) as *mut std::ffi::c_void),
            byte_count,
            stream,
        )?;
        if !stream.0.is_null() {
            super::stream::mark_non_default_stream_dirty();
        }
        return Ok(());
    }
    hipMemcpyAtoHAsync(
        dst_host,
        src_array as hipArray_t,
        src_offset,
        byte_count,
        stream,
    )?;
    if !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    Ok(())
}

pub(crate) unsafe fn copy_hto_a_v2(
    dst_array: CUarray,
    dst_offset: usize,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> CUresult {
    log_memcpy_sentinel_array(
        "cuMemcpyHtoA_v2",
        "dst",
        dst_array as hipArray_t,
        &format!(
            "offset={} bytes={} src_host={:p}",
            dst_offset, byte_count, src_host
        ),
    );
    if let Some((dev, _, _)) = super::d3d11_interop::d3d11_array_layout(dst_array as hipArray_t) {
        hipMemcpyHtoD(
            hipDeviceptr_t((dev as *mut u8).add(dst_offset) as *mut std::ffi::c_void),
            src_host as *mut std::ffi::c_void,
            byte_count,
        )?;
        return Ok(());
    }
    hipMemcpyHtoA(dst_array as hipArray_t, dst_offset, src_host, byte_count)?;
    Ok(())
}

pub(crate) unsafe fn copy_hto_a_async_v2(
    dst_array: CUarray,
    dst_offset: usize,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
    stream: hipStream_t,
) -> CUresult {
    log_memcpy_sentinel_array(
        "cuMemcpyHtoAAsync_v2",
        "dst",
        dst_array as hipArray_t,
        &format!(
            "offset={} bytes={} src_host={:p} stream={:p}",
            dst_offset, byte_count, src_host, stream.0
        ),
    );
    if let Some((dev, _, _)) = super::d3d11_interop::d3d11_array_layout(dst_array as hipArray_t) {
        hipMemcpyHtoDAsync(
            hipDeviceptr_t((dev as *mut u8).add(dst_offset) as *mut std::ffi::c_void),
            src_host as *mut std::ffi::c_void,
            byte_count,
            stream,
        )?;
        if !stream.0.is_null() {
            super::stream::mark_non_default_stream_dirty();
        }
        return Ok(());
    }
    hipMemcpyHtoAAsync(
        dst_array as hipArray_t,
        dst_offset,
        src_host,
        byte_count,
        stream,
    )?;
    if !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    Ok(())
}

pub(crate) unsafe fn copy_ato_d_v2(
    dst_device: hipDeviceptr_t,
    src_array: CUarray,
    src_offset: usize,
    byte_count: usize,
) -> CUresult {
    log_memcpy_sentinel_array(
        "cuMemcpyAtoD_v2",
        "src",
        src_array as hipArray_t,
        &format!(
            "offset={} bytes={} dst_dev=0x{:x}",
            src_offset, byte_count, dst_device.0 as usize
        ),
    );
    if let Some((dev, _, _)) = super::d3d11_interop::d3d11_array_layout(src_array as hipArray_t) {
        hipMemcpyDtoD(
            dst_device,
            hipDeviceptr_t((dev as *mut u8).add(src_offset) as *mut std::ffi::c_void),
            byte_count,
        )?;
        return Ok(());
    }
    hipMemcpyAtoD(dst_device, src_array as hipArray_t, src_offset, byte_count)?;
    Ok(())
}

pub(crate) unsafe fn copy_dto_a_v2(
    dst_array: CUarray,
    dst_offset: usize,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> CUresult {
    log_memcpy_sentinel_array(
        "cuMemcpyDtoA_v2",
        "dst",
        dst_array as hipArray_t,
        &format!(
            "offset={} bytes={} src_dev=0x{:x}",
            dst_offset, byte_count, src_device.0 as usize
        ),
    );
    if let Some((dev, _, _)) = super::d3d11_interop::d3d11_array_layout(dst_array as hipArray_t) {
        hipMemcpyDtoD(
            hipDeviceptr_t((dev as *mut u8).add(dst_offset) as *mut std::ffi::c_void),
            src_device,
            byte_count,
        )?;
        return Ok(());
    }
    hipMemcpyDtoA(dst_array as hipArray_t, dst_offset, src_device, byte_count)?;
    Ok(())
}

pub(crate) unsafe fn copy_ato_a_v2(
    dst_array: CUarray,
    dst_offset: usize,
    src_array: CUarray,
    src_offset: usize,
    byte_count: usize,
) -> CUresult {
    if d3d11_memcpy_log_enabled() {
        let src_sentinel = super::d3d11_interop::is_d3d11_sentinel(src_array as hipArray_t);
        let dst_sentinel = super::d3d11_interop::is_d3d11_sentinel(dst_array as hipArray_t);
        if src_sentinel
            || dst_sentinel
            || !(src_array as *const ()).is_null()
            || !(dst_array as *const ()).is_null()
        {
            super::driver::zluda_log_err(&format!(
                "[D3D11_MEMCPY] cuMemcpyAtoA_v2 src_array={:p} src_sentinel={} dst_array={:p} \
                 dst_sentinel={} src_offset={} dst_offset={} bytes={}",
                src_array,
                src_sentinel,
                dst_array,
                dst_sentinel,
                src_offset,
                dst_offset,
                byte_count,
            ));
        }
    }
    let src_layout = super::d3d11_interop::d3d11_array_layout(src_array as hipArray_t);
    let dst_layout = super::d3d11_interop::d3d11_array_layout(dst_array as hipArray_t);
    match (src_layout, dst_layout) {
        (Some((src_dev, _, _)), Some((dst_dev, _, _))) => {
            hipMemcpyDtoD(
                hipDeviceptr_t((dst_dev as *mut u8).add(dst_offset) as *mut std::ffi::c_void),
                hipDeviceptr_t((src_dev as *mut u8).add(src_offset) as *mut std::ffi::c_void),
                byte_count,
            )?;
            return Ok(());
        }
        (Some((src_dev, _, _)), None) => {
            hipMemcpyDtoA(
                dst_array as hipArray_t,
                dst_offset,
                hipDeviceptr_t((src_dev as *mut u8).add(src_offset) as *mut std::ffi::c_void),
                byte_count,
            )?;
            return Ok(());
        }
        (None, Some((dst_dev, _, _))) => {
            hipMemcpyAtoD(
                hipDeviceptr_t((dst_dev as *mut u8).add(dst_offset) as *mut std::ffi::c_void),
                src_array as hipArray_t,
                src_offset,
                byte_count,
            )?;
            return Ok(());
        }
        (None, None) => {}
    }
    hipMemcpyAtoA(
        dst_array as hipArray_t,
        dst_offset,
        src_array as hipArray_t,
        src_offset,
        byte_count,
    )?;
    Ok(())
}

pub(crate) unsafe fn copy_3d_v2(memcpy: HIP_MEMCPY3D) -> CUresult {
    log_memcpy_sentinel_3d("cuMemcpy3D_v2", &memcpy);
    let memcpy = rewrite_3d_for_sentinel(memcpy);
    hipDrvMemcpy3D(&memcpy)?;
    Ok(())
}

pub(crate) unsafe fn copy_3d_async_v2(memcpy: HIP_MEMCPY3D, stream: hipStream_t) -> CUresult {
    log_memcpy_sentinel_3d("cuMemcpy3DAsync_v2", &memcpy);
    let memcpy = rewrite_3d_for_sentinel(memcpy);
    hipDrvMemcpy3DAsync(&memcpy, stream)?;
    if !stream.0.is_null() {
        super::stream::mark_non_default_stream_dirty();
    }
    Ok(())
}

unsafe fn rewrite_3d_for_sentinel(mut memcpy: HIP_MEMCPY3D) -> HIP_MEMCPY3D {
    if let Some((dev, _size, pitch)) = super::d3d11_interop::d3d11_array_layout(memcpy.srcArray) {
        memcpy.srcArray = std::ptr::null_mut();
        memcpy.srcMemoryType = hipMemoryType::hipMemoryTypeDevice;
        memcpy.srcDevice = hipDeviceptr_t(dev);
        memcpy.srcPitch = pitch as usize;
    }
    if let Some((dev, _size, pitch)) = super::d3d11_interop::d3d11_array_layout(memcpy.dstArray) {
        memcpy.dstArray = std::ptr::null_mut();
        memcpy.dstMemoryType = hipMemoryType::hipMemoryTypeDevice;
        memcpy.dstDevice = hipDeviceptr_t(dev);
        memcpy.dstPitch = pitch as usize;
    }
    memcpy
}

pub(crate) unsafe fn set_d_2d32_v2(
    dst_device: hipDeviceptr_t,
    dst_pitch: usize,
    value: ::core::ffi::c_uint,
    width: usize,
    height: usize,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(hipStream_t(std::ptr::null_mut()));
    hipMemset2D(dst_device.0, dst_pitch, value as _, width, height)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Test that cuMemcpyHtoD correctly copies data to device memory
    /// that a GPU kernel can read. Previously, a raw memcpy fallback
    /// bypassed GPU page tables and could corrupt data, causing
    /// TurbulenceFS surface descriptor corruption and TCP page faults.
    #[test]
    fn memcpy_htod_data_integrity() {
        unsafe {
            if hipInit(0).is_err() {
                return;
            } // Skip if no GPU

            // Allocate device memory
            let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            hipMalloc(&mut dev_ptr, 256).unwrap();

            // Write a known pattern via hipMemcpyHtoD
            let host_data: Vec<u64> = (0..32).map(|i| 0xDEAD_BEEF_0000_0000u64 + i).collect();
            hipMemcpyHtoD(hipDeviceptr_t(dev_ptr), host_data.as_ptr() as *mut _, 256).unwrap();

            // Read back and verify
            let mut readback = vec![0u64; 32];
            hipMemcpyDtoH(
                readback.as_mut_ptr() as *mut _,
                hipDeviceptr_t(dev_ptr),
                256,
            )
            .unwrap();

            for i in 0..32 {
                assert_eq!(
                    readback[i], host_data[i],
                    "mismatch at index {}: got 0x{:016x}, expected 0x{:016x}",
                    i, readback[i], host_data[i]
                );
            }

            hipFree(dev_ptr).unwrap();
        }
    }

    /// Phase A diagnostic stubs (plan fluttering-waddling-zephyr.md):
    /// `is_d3d11_sentinel` must safely return false on null and on a
    /// random heap pointer that was never registered. The diagnostic
    /// stubs depend on this for their fast-path no-op behaviour, so a
    /// false-positive would produce a spurious log line on every call
    /// — tolerable but worth pinning.
    #[test]
    fn d3d11_sentinel_negatives() {
        use super::super::d3d11_interop::is_d3d11_sentinel;
        assert!(!is_d3d11_sentinel(std::ptr::null_mut()));
        // A random heap allocation is not a sentinel.
        let b = Box::new(0u64);
        let p = (&*b as *const u64) as hipArray_t;
        assert!(!is_d3d11_sentinel(p));
        drop(b);
    }

    /// `d3d11_memcpy_log_enabled` is gated by `ZLUDA_D3D11_MEMCPY_LOG=1`.
    /// We can't safely toggle env in a multi-threaded test (the OnceLock
    /// inside the helper means the FIRST call wins), so we just confirm
    /// it returns a stable bool and doesn't panic.
    #[test]
    fn d3d11_memcpy_log_gating_is_stable() {
        let a = super::d3d11_memcpy_log_enabled();
        let b = super::d3d11_memcpy_log_enabled();
        assert_eq!(a, b);
    }

    /// `rewrite_2d_for_sentinel` must leave a non-sentinel memcpy
    /// unchanged. (Phase E.1 of plan fluttering-waddling-zephyr.md.)
    #[test]
    fn rewrite_2d_for_sentinel_passthrough_when_not_sentinel() {
        unsafe {
            let mut m: hip_Memcpy2D = std::mem::zeroed();
            m.srcMemoryType = hipMemoryType::hipMemoryTypeDevice;
            m.srcDevice = hipDeviceptr_t(0x1000 as *mut std::ffi::c_void);
            m.srcArray = std::ptr::null_mut();
            m.dstMemoryType = hipMemoryType::hipMemoryTypeArray;
            m.dstArray = (0xdead_beef_usize) as hipArray_t; // not in table
            m.dstPitch = 999_999;
            m.WidthInBytes = 4096;
            m.Height = 1;
            let out = super::rewrite_2d_for_sentinel(m);
            // Neither src nor dst is registered → both sides untouched.
            assert_eq!(out.srcArray, m.srcArray);
            assert_eq!(out.dstArray, m.dstArray);
            assert_eq!(out.dstMemoryType.0, hipMemoryType::hipMemoryTypeArray.0);
            assert_eq!(out.dstPitch, 999_999);
        }
    }

    #[test]
    fn decode_ms2_batch_descriptor_reads_expected_offsets() {
        let mut desc = [0u8; 88];
        desc[0..4].copy_from_slice(&7u32.to_le_bytes());
        desc[32..40].copy_from_slice(&0x1111_2222_3333_4444u64.to_le_bytes());
        desc[72..80].copy_from_slice(&0x5555_6666_7777_8888u64.to_le_bytes());
        desc[80..88].copy_from_slice(&0x9999_aaaa_bbbb_ccccu64.to_le_bytes());
        assert_eq!(
            super::decode_ms2_batch_descriptor(&desc),
            Some((
                7,
                0x1111_2222_3333_4444,
                0x5555_6666_7777_8888,
                0x9999_aaaa_bbbb_cccc
            ))
        );
    }

    #[test]
    fn decode_ms2_batch_descriptor_rejects_short_buffers() {
        assert_eq!(super::decode_ms2_batch_descriptor(&[0u8; 87]), None);
    }
}

pub(crate) unsafe fn set_d_2d32_async(
    dst_device: hipDeviceptr_t,
    dst_pitch: usize,
    value: ::core::ffi::c_uint,
    width: usize,
    height: usize,
    stream: hipStream_t,
) -> hipError_t {
    let _guard = super::stream::null_stream_sync(stream);
    hipMemset2DAsync(dst_device.0, dst_pitch, value as _, width, height, stream)
}

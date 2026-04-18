use crate::r#impl::{self, context, device, function, module};
use cuda_types::cuda::*;
use hip_runtime_sys::*;
use libloading::Library;
use std::{
    collections::BTreeMap,
    ffi::{c_void, CStr, CString},
    mem, ptr, slice,
    sync::{Mutex, OnceLock},
    usize,
};
use zluda_common::{constants, FromCuda, LiveCheck};

#[cfg_attr(windows, path = "os_win.rs")]
#[cfg_attr(not(windows), path = "os_unix.rs")]
mod os;

pub(crate) struct GlobalState {
    pub devices: Vec<Device>,
    pub cache_path: Option<String>,
    pub allocations: Mutex<Allocations>,
    pub compute_capability: (i32, i32),
}

pub(crate) struct Allocations {
    pub pointers: BTreeMap<usize, AllocationInfo>,
}

impl Allocations {
    pub fn new() -> Self {
        Allocations {
            pointers: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, ptr: usize, size: usize, context: CUcontext) {
        self.pointers.insert(ptr, AllocationInfo { size, context });
    }

    pub fn get_offset_and_info(&self, ptr: usize) -> Option<(usize, AllocationInfo)> {
        // Find last pair where `start <= ptr`
        let (start, alloc) = self.pointers.range(..=ptr).rev().next()?;
        // Check if allocation contains the pointer
        if start + alloc.size > ptr {
            Some((ptr - start, *alloc))
        } else {
            None
        }
    }

    pub fn remove(&mut self, ptr: usize) {
        self.pointers.remove(&ptr);
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct AllocationInfo {
    pub size: usize,
    pub context: CUcontext,
}

pub(crate) struct Device {
    pub(crate) _comgr_isa: CString,
    primary_context: LiveCheck<context::Context>,
}

impl Device {
    pub(crate) fn primary_context<'a>(&'a self) -> (&'a context::Context, CUcontext) {
        unsafe {
            (
                self.primary_context.data.assume_init_ref(),
                self.primary_context.as_handle(),
            )
        }
    }
}

pub(crate) fn device(dev: i32) -> Result<&'static Device, CUerror> {
    global_state()?
        .devices
        .get(dev as usize)
        .ok_or(CUerror::INVALID_DEVICE)
}

pub(crate) fn global_state() -> Result<&'static GlobalState, CUerror> {
    static GLOBAL_STATE: OnceLock<Result<GlobalState, CUerror>> = OnceLock::new();
    fn cast_slice<'a>(bytes: &'a [i8]) -> &'a [u8] {
        unsafe { slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) }
    }
    GLOBAL_STATE
        .get_or_init(|| {
            let mut device_count = 0;
            unsafe { hipGetDeviceCount(&mut device_count) }?;
            let allocations = Mutex::new(Allocations::new());
            Ok(GlobalState {
                allocations,
                devices: (0..device_count)
                    .map(|i| {
                        let mut props = unsafe { mem::zeroed() };
                        unsafe { hipGetDevicePropertiesR0600(&mut props, i) }?;
                        Ok::<_, CUerror>(Device {
                            _comgr_isa: CStr::from_bytes_until_nul(cast_slice(
                                &props.gcnArchName[..],
                            ))
                            .map_err(|_| CUerror::UNKNOWN)?
                            .to_owned(),
                            primary_context: LiveCheck::new(context::Context::new(i)?),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                cache_path: zluda_cache::ModuleCache::create_cache_dir_and_get_path(),
                compute_capability: constants::compute_capability(),
            })
        })
        .as_ref()
        .map_err(|e| *e)
}

// Global log file handles, opened once on first write and protected by a mutex so
// concurrent writers don't interleave at the byte level. Previously each call opened
// the file fresh and used `writeln!`, which issues multiple write() syscalls per
// format argument — concurrent threads produced output like
// `[[10721821072182ms #ms #531716531717] OK ] OK cuLaunchKernelcuLaunchKernel`,
// corrupting diagnostics during exactly the races we need to investigate.
//
// The SIGABRT handler in `install_fault_signal_handler` deliberately does NOT use
// these — Mutex is not async-signal-safe, so the crash path keeps its own per-call
// open-append pattern and accepts a small risk of one corrupted crash line.
static LOG_FILE: std::sync::LazyLock<Mutex<Option<std::fs::File>>> =
    std::sync::LazyLock::new(|| {
        Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/zluda_tex.log")
                .ok(),
        )
    });

static DEBUG_LOG_FILE: std::sync::LazyLock<Mutex<Option<std::fs::File>>> =
    std::sync::LazyLock::new(|| {
        Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/zluda_debug.log")
                .ok(),
        )
    });

pub(crate) fn zluda_log(msg: &str) {
    use std::io::Write;
    // Check ZLUDA_DEBUG env var (cached after first check)
    static ENABLED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let enabled = match ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        0 => {
            let e = std::env::var("ZLUDA_DEBUG").map_or(false, |v| v == "1");
            ENABLED.store(if e { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
            e
        }
        2 => true,
        _ => false,
    };
    if !enabled {
        return;
    }
    // Pre-format into a String so the whole line goes out in one write_all.
    let line = format!("{}\n", msg);
    if let Ok(mut guard) = DEBUG_LOG_FILE.lock() {
        if let Some(f) = guard.as_mut() {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

pub(crate) fn zluda_log_err(msg: &str) {
    use std::io::Write;
    let ms = crate::elapsed_ms();
    // Pre-format into a String so the whole line goes out in a single write_all —
    // avoids inter-thread interleaving of the timestamp prefix and message body.
    let line = format!("[{}ms] {}\n", ms, msg);
    if let Ok(mut guard) = LOG_FILE.lock() {
        if let Some(f) = guard.as_mut() {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Ring buffer of last N kernel names — the SIGABRT handler dumps all of them
/// so we see the crash context, not just the racy "last" slot. The fault is
/// asynchronous on Van Gogh APU, so the actually-faulting kernel is usually
/// NOT the single most recent one.
const KERNEL_RING_SIZE: usize = 32;
static KERNEL_RING: std::sync::Mutex<(
    [Option<String>; KERNEL_RING_SIZE],
    usize, // write index
    u64,   // total push count
)> = std::sync::Mutex::new(([const { None }; KERNEL_RING_SIZE], 0, 0));

pub(crate) fn set_last_kernel(info: String) {
    if let Ok(mut guard) = KERNEL_RING.lock() {
        let (ref mut buf, ref mut idx, ref mut total) = *guard;
        buf[*idx] = Some(info);
        *idx = (*idx + 1) % KERNEL_RING_SIZE;
        *total += 1;
    }
}

/// SIGABRT/SIGSEGV handler — fires when HSA runtime kills the process on GPU fault.
/// Dumps the last N kernel launches to the log so we can see the crash context.
#[cfg(not(windows))]
fn install_fault_signal_handler() {
    unsafe extern "C" fn sigabrt_handler(sig: i32) {
        // Minimal async-signal-safe operations only.
        // Write directly to the log file.
        use std::io::Write;
        let sig_name = match sig {
            6 => "SIGABRT",
            11 => "SIGSEGV",
            _ => "signal",
        };

        // Dump the MS_4 and MS_2_Batched capture ring buffers to
        // /tmp/ms4_captures.bin and /tmp/ms2b_captures.bin if capture was
        // enabled. Uses raw open/write/close syscalls via libc since they're
        // async-signal-safe (unlike std::fs::File which uses the allocator
        // and mutexes indirectly).
        unsafe fn dump_ring(host_ptr: usize, total_bytes: usize, path_cstr: *const libc::c_char) {
            if host_ptr == 0 || total_bytes == 0 {
                return;
            }
            let fd = libc::open(
                path_cstr,
                libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
                0o644,
            );
            if fd < 0 {
                return;
            }
            let mut written = 0usize;
            while written < total_bytes {
                let n = libc::write(
                    fd,
                    (host_ptr + written) as *const libc::c_void,
                    total_bytes - written,
                );
                if n <= 0 {
                    break;
                }
                written += n as usize;
            }
            libc::close(fd);
        }
        let ms4_host_ptr =
            super::function::MS4_CAPTURE_HOST_PTR.load(std::sync::atomic::Ordering::SeqCst);
        let ms4_total_bytes =
            super::function::MS4_CAPTURE_TOTAL_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        dump_ring(
            ms4_host_ptr,
            ms4_total_bytes,
            c"/tmp/ms4_captures.bin".as_ptr(),
        );

        let ms2b_host_ptr =
            super::function::MS2B_CAPTURE_HOST_PTR.load(std::sync::atomic::Ordering::SeqCst);
        let ms2b_total_bytes =
            super::function::MS2B_CAPTURE_TOTAL_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        dump_ring(
            ms2b_host_ptr,
            ms2b_total_bytes,
            c"/tmp/ms2b_captures.bin".as_ptr(),
        );

        let ms3_host_ptr =
            super::function::MS3_CAPTURE_HOST_PTR.load(std::sync::atomic::Ordering::SeqCst);
        let ms3_total_bytes =
            super::function::MS3_CAPTURE_TOTAL_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        dump_ring(
            ms3_host_ptr,
            ms3_total_bytes,
            c"/tmp/ms3_captures.bin".as_ptr(),
        );

        let tfs_host_ptr =
            super::function::TFS_CAPTURE_HOST_PTR.load(std::sync::atomic::Ordering::SeqCst);
        let tfs_total_bytes =
            super::function::TFS_CAPTURE_TOTAL_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        dump_ring(
            tfs_host_ptr,
            tfs_total_bytes,
            c"/tmp/tfs_captures.bin".as_ptr(),
        );

        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/zluda_tex.log")
        {
            let _ = write!(
                f,
                "\n*** {} (signal {}) — GPU fault likely ***\n",
                sig_name, sig
            );
            if ms4_host_ptr != 0 && ms4_total_bytes != 0 {
                let _ = write!(
                    f,
                    "*** MS4 capture dumped to /tmp/ms4_captures.bin ({} bytes)\n",
                    ms4_total_bytes
                );
            }
            if ms2b_host_ptr != 0 && ms2b_total_bytes != 0 {
                let _ = write!(
                    f,
                    "*** MS2B capture dumped to /tmp/ms2b_captures.bin ({} bytes)\n",
                    ms2b_total_bytes
                );
            }
            if ms3_host_ptr != 0 && ms3_total_bytes != 0 {
                let _ = write!(
                    f,
                    "*** MS3 capture dumped to /tmp/ms3_captures.bin ({} bytes)\n",
                    ms3_total_bytes
                );
            }
            if tfs_host_ptr != 0 && tfs_total_bytes != 0 {
                let _ = write!(
                    f,
                    "*** TFS capture dumped to /tmp/tfs_captures.bin ({} bytes)\n",
                    tfs_total_bytes
                );
            }
            if let Ok(guard) = KERNEL_RING.lock() {
                let (ref buf, idx, total) = *guard;
                let _ = write!(
                    f,
                    "*** Kernel launch history (total={}, newest last):\n",
                    total
                );
                // Iterate from oldest to newest entry in the ring.
                let start = if total >= KERNEL_RING_SIZE as u64 {
                    idx
                } else {
                    0
                };
                let count = if total >= KERNEL_RING_SIZE as u64 {
                    KERNEL_RING_SIZE
                } else {
                    total as usize
                };
                for i in 0..count {
                    let slot = (start + i) % KERNEL_RING_SIZE;
                    if let Some(ref name) = buf[slot] {
                        let launch_n = total.saturating_sub((count - i) as u64);
                        let _ = write!(f, "***   #{} {}\n", launch_n, name);
                    }
                }
            }
            // Read /proc/self/maps to find what's at common fault regions
            if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
                let _ = write!(f, "*** Process memory maps (0x7f region):\n");
                for line in maps.lines() {
                    if line.starts_with("7f") {
                        let _ = writeln!(f, "***   {}", line);
                    }
                }
            }
        }
        // Re-raise with default handler to actually die
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
    unsafe {
        libc::signal(libc::SIGABRT, sigabrt_handler as libc::sighandler_t);
        libc::signal(libc::SIGSEGV, sigabrt_handler as libc::sighandler_t);
    }
    zluda_log_err("[ZLUDA] Signal handlers installed (SIGABRT+SIGSEGV) for fault capture");
}

/// Load the ROCm debug agent library and call its OnLoad entry point.
///
/// The debug agent catches GPU faults (memory violations, illegal instructions,
/// assert traps) and dumps wavefront register state to stderr or a file
/// (configurable via ROCM_DEBUG_AGENT_OPTIONS env var). It uses amd_dbgapi to
/// attach to the process as a GPU debugger and receive hardware exceptions.
///
/// This bypasses HSA_TOOLS_LIB, which fails inside pressure-vessel Steam
/// Runtime containers because dlopen can't resolve transitive dependencies.
/// After hipInit(), the core ROCm libraries (libhsa-runtime64, libamd_comgr,
/// librocprofiler-register) are already loaded, so the debug agent's
/// dependencies resolve from the process and its co-located RUNPATH=$ORIGIN dir.
///
/// The agent also hooks hsa_executable_freeze and hsa_executable_destroy in the
/// API table to track loaded code objects for disassembly context. Since we pass
/// our own copy of the table (not the runtime's internal one), these hooks won't
/// intercept actual HSA calls -- code object tracking will be incomplete. The
/// core fault-catching and wavefront-dumping via amd_dbgapi works regardless.
///
/// Controlled by ZLUDA_DEBUG_AGENT=1. Options via ROCM_DEBUG_AGENT_OPTIONS
/// (e.g. "--all --save-code-objects --output=/tmp/rocm_fault.log").
#[cfg(not(windows))]
fn load_rocm_debug_agent() {
    // Path to the debug agent library. RUNPATH=$ORIGIN means its deps
    // (librocm-dbgapi.so.0, libamd_comgr.so.3) are found in the same dir.
    // Default path in the self-contained agent dir (all deps co-located).
    // Override with ZLUDA_DEBUG_AGENT_PATH env var if needed.
    let default_path = "/home/deck/.local/share/Steam/compatibilitytools.d/proton-cuda/files/lib64/rocm_debug_agent/librocm-debug-agent.so.2";
    let agent_path =
        std::env::var("ZLUDA_DEBUG_AGENT_PATH").unwrap_or_else(|_| default_path.to_string());

    zluda_log_err(&format!(
        "[ZLUDA] Loading ROCm debug agent from: {}",
        agent_path
    ));

    let c_path = match CString::new(agent_path.as_bytes()) {
        Ok(p) => p,
        Err(e) => {
            zluda_log_err(&format!(
                "[ZLUDA] Debug agent path contains null byte: {}",
                e
            ));
            return;
        }
    };

    // RTLD_NOW: resolve all symbols immediately (fail fast if deps missing).
    // RTLD_LOCAL: keep the agent's symbols (and its transitive deps like
    //             libamd_comgr.so.2) hidden from the main process, so HIP's
    //             own comgr.so.3 calls don't collide with the agent's comgr.so.2.
    // RTLD_DEEPBIND: force the agent and its deps to resolve symbols from their
    //                own namespace first, not from the global scope. This is the
    //                key bit: without DEEPBIND, librocm-dbgapi.so.0's comgr calls
    //                may still get routed to HIP's already-loaded comgr.so.3,
    //                which doesn't export the v2.0 symbols it needs → SIGSEGV.
    let handle = unsafe {
        libc::dlopen(
            c_path.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_DEEPBIND,
        )
    };
    if handle.is_null() {
        let err = unsafe { CStr::from_ptr(libc::dlerror()) };
        zluda_log_err(&format!(
            "[ZLUDA] Failed to dlopen debug agent: {}",
            err.to_string_lossy()
        ));
        return;
    }
    zluda_log_err("[ZLUDA] Debug agent library loaded");

    // Look up OnLoad. Signature from ROCr tool ABI (hsa_api_trace.h):
    //   bool OnLoad(void* api_table, uint64_t runtime_version,
    //               uint64_t failed_tool_count, const char* const* failed_tool_names)
    let onload_sym = unsafe { libc::dlsym(handle, c"OnLoad".as_ptr()) };
    if onload_sym.is_null() {
        zluda_log_err("[ZLUDA] Debug agent has no OnLoad symbol, skipping");
        return;
    }
    type OnLoadFn = unsafe extern "C" fn(*mut c_void, u64, u64, *const *const i8) -> bool;
    let onload: OnLoadFn = unsafe { mem::transmute(onload_sym) };

    // Build an HsaApiTable populated with real function pointers resolved from
    // the already-loaded libhsa-runtime64.so.1 via dlsym(RTLD_DEFAULT, ...).
    //
    // The agent modifies the table in-place (installs hooks) and keeps the
    // pointer for the process lifetime, so these must be static and mutable.
    //
    // Layout from ext/rocm/include/hsa/hsa_api_trace.h:
    //   Each table starts with ApiTableVersion { major_id, minor_id, step_id, reserved }
    //   (4 x u32 = 16 bytes = 2 usize slots), followed by function pointers.
    //
    //   HsaApiTable (root):
    //     [0..1] ApiTableVersion  (major=3, minor=sizeof, step=1)
    //     [2]    core_: *mut CoreApiTable       (offset 0x10)
    //     [3]    amd_ext_: *mut AmdExtTable     (offset 0x18)
    //     [4]    finalizer_ext_                 (offset 0x20)
    //     [5]    image_ext_                     (offset 0x28)
    //     [6]    tools_                         (offset 0x30)
    //     [7]    pc_sampling_ext_               (offset 0x38)

    // CoreApiTable: 127 entries (version + 125 function pointers) = ~1016 bytes.
    // Use 128 slots for safety against minor version differences.
    static mut CORE_TABLE: [usize; 128] = [0; 128];
    // AmdExtTable: ~74 entries in ROCm 6.4. Use 128 slots.
    static mut AMD_EXT_TABLE: [usize; 128] = [0; 128];
    // Root HsaApiTable: 8 entries.
    static mut HSA_TABLE: [usize; 8] = [0; 8];

    // Resolve an HSA function from the already-loaded runtime.
    // RTLD_DEFAULT searches all loaded shared objects.
    macro_rules! resolve {
        ($name:literal, $table:ident, $idx:expr) => {
            let ptr = libc::dlsym(
                libc::RTLD_DEFAULT,
                concat!($name, "\0").as_ptr() as *const _,
            );
            if !ptr.is_null() {
                *$table.add($idx) = ptr as usize;
            }
        };
    }

    unsafe {
        // Raw pointers to the static tables, avoiding mutable references to
        // mutable statics (which trigger static_mut_refs warnings in Rust 2024).
        let hsa_tbl = ptr::addr_of_mut!(HSA_TABLE) as *mut usize;
        let core_tbl = ptr::addr_of_mut!(CORE_TABLE) as *mut usize;
        let amd_tbl = ptr::addr_of_mut!(AMD_EXT_TABLE) as *mut usize;
        // --- Root table version ---
        let ver = hsa_tbl as *mut u32;
        *ver.add(0) = 0x03; // HSA_API_TABLE_MAJOR_VERSION
        *ver.add(1) = mem::size_of::<[usize; 8]>() as u32;
        *ver.add(2) = 0x01; // HSA_API_TABLE_STEP_VERSION
        *ver.add(3) = 0;
        *hsa_tbl.add(2) = core_tbl as usize; // core_
        *hsa_tbl.add(3) = amd_tbl as usize; // amd_ext_
                                            // finalizer_ext_, image_ext_, tools_, pc_sampling_ext_ left as null --
                                            // the agent doesn't access them.

        // --- CoreApiTable version ---
        let ver = core_tbl as *mut u32;
        *ver.add(0) = 0x02; // HSA_CORE_API_TABLE_MAJOR_VERSION
        *ver.add(1) = mem::size_of::<[usize; 128]>() as u32;
        *ver.add(2) = 0x00; // HSA_CORE_API_TABLE_STEP_VERSION
        *ver.add(3) = 0;

        // CoreApiTable function pointers (index = byte_offset / 8).
        // The agent hooks [86] hsa_executable_destroy and [88] hsa_executable_freeze.
        // We populate all entries so the hooks have valid originals to call through.
        resolve!("hsa_init", core_tbl, 2);
        resolve!("hsa_shut_down", core_tbl, 3);
        resolve!("hsa_system_get_info", core_tbl, 4);
        resolve!("hsa_system_extension_supported", core_tbl, 5);
        resolve!("hsa_system_get_extension_table", core_tbl, 6);
        resolve!("hsa_iterate_agents", core_tbl, 7);
        resolve!("hsa_agent_get_info", core_tbl, 8);
        resolve!("hsa_queue_create", core_tbl, 9);
        resolve!("hsa_soft_queue_create", core_tbl, 10);
        resolve!("hsa_queue_destroy", core_tbl, 11);
        resolve!("hsa_queue_inactivate", core_tbl, 12);
        resolve!("hsa_queue_load_read_index_scacquire", core_tbl, 13);
        resolve!("hsa_queue_load_read_index_relaxed", core_tbl, 14);
        resolve!("hsa_queue_load_write_index_scacquire", core_tbl, 15);
        resolve!("hsa_queue_load_write_index_relaxed", core_tbl, 16);
        resolve!("hsa_queue_store_write_index_relaxed", core_tbl, 17);
        resolve!("hsa_queue_store_write_index_screlease", core_tbl, 18);
        resolve!("hsa_queue_cas_write_index_scacq_screl", core_tbl, 19);
        resolve!("hsa_queue_cas_write_index_scacquire", core_tbl, 20);
        resolve!("hsa_queue_cas_write_index_relaxed", core_tbl, 21);
        resolve!("hsa_queue_cas_write_index_screlease", core_tbl, 22);
        resolve!("hsa_queue_add_write_index_scacq_screl", core_tbl, 23);
        resolve!("hsa_queue_add_write_index_scacquire", core_tbl, 24);
        resolve!("hsa_queue_add_write_index_relaxed", core_tbl, 25);
        resolve!("hsa_queue_add_write_index_screlease", core_tbl, 26);
        resolve!("hsa_queue_store_read_index_relaxed", core_tbl, 27);
        resolve!("hsa_queue_store_read_index_screlease", core_tbl, 28);
        resolve!("hsa_agent_iterate_regions", core_tbl, 29);
        resolve!("hsa_region_get_info", core_tbl, 30);
        resolve!("hsa_agent_get_exception_policies", core_tbl, 31);
        resolve!("hsa_agent_extension_supported", core_tbl, 32);
        resolve!("hsa_memory_register", core_tbl, 33);
        resolve!("hsa_memory_deregister", core_tbl, 34);
        resolve!("hsa_memory_allocate", core_tbl, 35);
        resolve!("hsa_memory_free", core_tbl, 36);
        resolve!("hsa_memory_copy", core_tbl, 37);
        resolve!("hsa_memory_assign_agent", core_tbl, 38);
        resolve!("hsa_signal_create", core_tbl, 39);
        resolve!("hsa_signal_destroy", core_tbl, 40);
        resolve!("hsa_signal_load_relaxed", core_tbl, 41);
        resolve!("hsa_signal_load_scacquire", core_tbl, 42);
        resolve!("hsa_signal_store_relaxed", core_tbl, 43);
        resolve!("hsa_signal_store_screlease", core_tbl, 44);
        resolve!("hsa_signal_wait_relaxed", core_tbl, 45);
        resolve!("hsa_signal_wait_scacquire", core_tbl, 46);
        resolve!("hsa_signal_and_relaxed", core_tbl, 47);
        resolve!("hsa_signal_and_scacquire", core_tbl, 48);
        resolve!("hsa_signal_and_screlease", core_tbl, 49);
        resolve!("hsa_signal_and_scacq_screl", core_tbl, 50);
        resolve!("hsa_signal_or_relaxed", core_tbl, 51);
        resolve!("hsa_signal_or_scacquire", core_tbl, 52);
        resolve!("hsa_signal_or_screlease", core_tbl, 53);
        resolve!("hsa_signal_or_scacq_screl", core_tbl, 54);
        resolve!("hsa_signal_xor_relaxed", core_tbl, 55);
        resolve!("hsa_signal_xor_scacquire", core_tbl, 56);
        resolve!("hsa_signal_xor_screlease", core_tbl, 57);
        resolve!("hsa_signal_xor_scacq_screl", core_tbl, 58);
        resolve!("hsa_signal_exchange_relaxed", core_tbl, 59);
        resolve!("hsa_signal_exchange_scacquire", core_tbl, 60);
        resolve!("hsa_signal_exchange_screlease", core_tbl, 61);
        resolve!("hsa_signal_exchange_scacq_screl", core_tbl, 62);
        resolve!("hsa_signal_add_relaxed", core_tbl, 63);
        resolve!("hsa_signal_add_scacquire", core_tbl, 64);
        resolve!("hsa_signal_add_screlease", core_tbl, 65);
        resolve!("hsa_signal_add_scacq_screl", core_tbl, 66);
        resolve!("hsa_signal_subtract_relaxed", core_tbl, 67);
        resolve!("hsa_signal_subtract_scacquire", core_tbl, 68);
        resolve!("hsa_signal_subtract_screlease", core_tbl, 69);
        resolve!("hsa_signal_subtract_scacq_screl", core_tbl, 70);
        resolve!("hsa_signal_cas_relaxed", core_tbl, 71);
        resolve!("hsa_signal_cas_scacquire", core_tbl, 72);
        resolve!("hsa_signal_cas_screlease", core_tbl, 73);
        resolve!("hsa_signal_cas_scacq_screl", core_tbl, 74);
        resolve!("hsa_isa_from_name", core_tbl, 75);
        resolve!("hsa_isa_get_info", core_tbl, 76);
        resolve!("hsa_isa_compatible", core_tbl, 77);
        resolve!("hsa_code_object_serialize", core_tbl, 78);
        resolve!("hsa_code_object_deserialize", core_tbl, 79);
        resolve!("hsa_code_object_destroy", core_tbl, 80);
        resolve!("hsa_code_object_get_info", core_tbl, 81);
        resolve!("hsa_code_object_get_symbol", core_tbl, 82);
        resolve!("hsa_code_symbol_get_info", core_tbl, 83);
        resolve!("hsa_code_object_iterate_symbols", core_tbl, 84);
        resolve!("hsa_executable_create", core_tbl, 85);
        resolve!("hsa_executable_destroy", core_tbl, 86); // agent hooks this
        resolve!("hsa_executable_load_code_object", core_tbl, 87);
        resolve!("hsa_executable_freeze", core_tbl, 88); // agent hooks this
        resolve!("hsa_executable_get_info", core_tbl, 89);
        resolve!("hsa_executable_global_variable_define", core_tbl, 90);
        resolve!("hsa_executable_agent_global_variable_define", core_tbl, 91);
        resolve!("hsa_executable_readonly_variable_define", core_tbl, 92);
        resolve!("hsa_executable_validate", core_tbl, 93);
        resolve!("hsa_executable_get_symbol", core_tbl, 94);
        resolve!("hsa_executable_symbol_get_info", core_tbl, 95);
        resolve!("hsa_executable_iterate_symbols", core_tbl, 96);
        resolve!("hsa_status_string", core_tbl, 97);
        // v1.1 additions
        resolve!("hsa_extension_get_name", core_tbl, 98);
        resolve!("hsa_system_major_extension_supported", core_tbl, 99);
        resolve!("hsa_system_get_major_extension_table", core_tbl, 100);
        resolve!("hsa_agent_major_extension_supported", core_tbl, 101);
        resolve!("hsa_cache_get_info", core_tbl, 102);
        resolve!("hsa_agent_iterate_caches", core_tbl, 103);
        resolve!("hsa_signal_silent_store_relaxed", core_tbl, 104);
        resolve!("hsa_signal_silent_store_screlease", core_tbl, 105);
        resolve!("hsa_signal_group_create", core_tbl, 106);
        resolve!("hsa_signal_group_destroy", core_tbl, 107);
        resolve!("hsa_signal_group_wait_any_scacquire", core_tbl, 108);
        resolve!("hsa_signal_group_wait_any_relaxed", core_tbl, 109);
        resolve!("hsa_agent_iterate_isas", core_tbl, 110);
        resolve!("hsa_isa_get_info_alt", core_tbl, 111);
        resolve!("hsa_isa_get_exception_policies", core_tbl, 112);
        resolve!("hsa_isa_get_round_method", core_tbl, 113);
        resolve!("hsa_wavefront_get_info", core_tbl, 114);
        resolve!("hsa_isa_iterate_wavefronts", core_tbl, 115);
        resolve!("hsa_code_object_get_symbol_from_name", core_tbl, 116);
        resolve!("hsa_code_object_reader_create_from_file", core_tbl, 117);
        resolve!("hsa_code_object_reader_create_from_memory", core_tbl, 118);
        resolve!("hsa_code_object_reader_destroy", core_tbl, 119);
        resolve!("hsa_executable_create_alt", core_tbl, 120);
        resolve!("hsa_executable_load_program_code_object", core_tbl, 121);
        resolve!("hsa_executable_load_agent_code_object", core_tbl, 122);
        resolve!("hsa_executable_validate_alt", core_tbl, 123);
        resolve!("hsa_executable_get_symbol_by_name", core_tbl, 124);
        resolve!("hsa_executable_iterate_agent_symbols", core_tbl, 125);
        resolve!("hsa_executable_iterate_program_symbols", core_tbl, 126);

        // --- AmdExtTable version ---
        let ver = amd_tbl as *mut u32;
        *ver.add(0) = 0x02; // HSA_AMD_EXT_API_TABLE_MAJOR_VERSION
        *ver.add(1) = mem::size_of::<[usize; 128]>() as u32;
        *ver.add(2) = 0x05; // HSA_AMD_EXT_API_TABLE_STEP_VERSION
        *ver.add(3) = 0;

        // AmdExtTable function pointers. Populate the critical ones the agent
        // may use during fault handling (e.g. hsa_amd_register_system_event_handler).
        resolve!("hsa_amd_coherency_get_type", amd_tbl, 2);
        resolve!("hsa_amd_coherency_set_type", amd_tbl, 3);
        resolve!("hsa_amd_profiling_set_profiler_enabled", amd_tbl, 4);
        resolve!("hsa_amd_profiling_async_copy_enable", amd_tbl, 5);
        resolve!("hsa_amd_profiling_get_dispatch_time", amd_tbl, 6);
        resolve!("hsa_amd_profiling_get_async_copy_time", amd_tbl, 7);
        resolve!(
            "hsa_amd_profiling_convert_tick_to_system_domain",
            amd_tbl,
            8
        );
        resolve!("hsa_amd_signal_async_handler", amd_tbl, 9);
        resolve!("hsa_amd_async_function", amd_tbl, 10);
        resolve!("hsa_amd_signal_wait_any", amd_tbl, 11);
        resolve!("hsa_amd_queue_cu_set_mask", amd_tbl, 12);
        resolve!("hsa_amd_memory_pool_get_info", amd_tbl, 13);
        resolve!("hsa_amd_agent_iterate_memory_pools", amd_tbl, 14);
        resolve!("hsa_amd_memory_pool_allocate", amd_tbl, 15);
        resolve!("hsa_amd_memory_pool_free", amd_tbl, 16);
        resolve!("hsa_amd_memory_async_copy", amd_tbl, 17);
        resolve!("hsa_amd_memory_async_copy_on_engine", amd_tbl, 18);
        resolve!("hsa_amd_memory_copy_engine_status", amd_tbl, 19);
        resolve!("hsa_amd_agent_memory_pool_get_info", amd_tbl, 20);
        resolve!("hsa_amd_agents_allow_access", amd_tbl, 21);
        resolve!("hsa_amd_memory_pool_can_migrate", amd_tbl, 22);
        resolve!("hsa_amd_memory_migrate", amd_tbl, 23);
        resolve!("hsa_amd_memory_lock", amd_tbl, 24);
        resolve!("hsa_amd_memory_unlock", amd_tbl, 25);
        resolve!("hsa_amd_memory_fill", amd_tbl, 26);
        resolve!("hsa_amd_interop_map_buffer", amd_tbl, 27);
        resolve!("hsa_amd_interop_unmap_buffer", amd_tbl, 28);
        resolve!("hsa_amd_image_create", amd_tbl, 29);
        resolve!("hsa_amd_pointer_info", amd_tbl, 30);
        resolve!("hsa_amd_pointer_info_set_userdata", amd_tbl, 31);
        resolve!("hsa_amd_ipc_memory_create", amd_tbl, 32);
        resolve!("hsa_amd_ipc_memory_attach", amd_tbl, 33);
        resolve!("hsa_amd_ipc_memory_detach", amd_tbl, 34);
        resolve!("hsa_amd_signal_create", amd_tbl, 35);
        resolve!("hsa_amd_ipc_signal_create", amd_tbl, 36);
        resolve!("hsa_amd_ipc_signal_attach", amd_tbl, 37);
        resolve!("hsa_amd_register_system_event_handler", amd_tbl, 38);
        resolve!("hsa_amd_queue_intercept_create", amd_tbl, 39);
        resolve!("hsa_amd_queue_intercept_register", amd_tbl, 40);
        resolve!("hsa_amd_queue_set_priority", amd_tbl, 41);
        resolve!("hsa_amd_memory_async_copy_rect", amd_tbl, 42);
        resolve!("hsa_amd_runtime_queue_create_register", amd_tbl, 43);
        resolve!("hsa_amd_memory_lock_to_pool", amd_tbl, 44);
        resolve!("hsa_amd_register_deallocation_callback", amd_tbl, 45);
        resolve!("hsa_amd_deregister_deallocation_callback", amd_tbl, 46);
        resolve!("hsa_amd_signal_value_pointer", amd_tbl, 47);

        // Log how many functions we resolved
        let mut core_count = 0usize;
        for i in 2..128 {
            if *core_tbl.add(i) != 0 {
                core_count += 1;
            }
        }
        let mut amd_count = 0usize;
        for i in 2..128 {
            if *amd_tbl.add(i) != 0 {
                amd_count += 1;
            }
        }
        zluda_log_err(&format!(
            "[ZLUDA] HSA API table: {} core + {} amd_ext functions resolved",
            core_count, amd_count
        ));

        // Call OnLoad(table, runtime_version=0, failed_count=0, failed_names=null)
        let table_ptr = hsa_tbl as *mut c_void;
        zluda_log_err("[ZLUDA] Calling debug agent OnLoad...");
        let result = onload(table_ptr, 0, 0, ptr::null());
        if result {
            zluda_log_err("[ZLUDA] Debug agent OnLoad succeeded - GPU fault handler active");
            zluda_log_err("[ZLUDA] Send SIGQUIT (kill -3 <pid>) to dump all wavefront state");
        } else {
            zluda_log_err("[ZLUDA] Debug agent OnLoad returned false (init failed)");
        }
    }

    // Intentionally leak the dlopen handle: the debug agent registers signal
    // handlers and amd_dbgapi callbacks that must persist for the process lifetime.
    // The static tables above must also remain valid.
}

/// Run hipInit on a native pthread to avoid Wine thread issues.
/// ROCm 6.4's hipInit crashes on Wine threads due to stack address assumptions.
#[cfg(not(windows))]
fn hip_init_on_native_thread(flags: ::core::ffi::c_uint) -> Result<(), cuda_types::cuda::CUerror> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let builder = std::thread::Builder::new().name("zluda-hip-init".into());
    let handle = builder
        .spawn(move || {
            let result = unsafe { hipInit(flags) };
            tx.send(result).ok();
        })
        .map_err(|e| {
            zluda_log(&format!("[ZLUDA] Failed to spawn hipInit thread: {}", e));
            cuda_types::cuda::CUerror::UNKNOWN
        })?;
    handle.join().map_err(|_| {
        zluda_log("[ZLUDA] hipInit thread panicked");
        cuda_types::cuda::CUerror::UNKNOWN
    })?;
    let hip_result = rx.recv().map_err(|_| {
        zluda_log("[ZLUDA] hipInit thread didn't send result");
        cuda_types::cuda::CUerror::UNKNOWN
    })?;
    hip_result.map_err(|e| {
        zluda_log(&format!("[ZLUDA] hipInit failed: {:?}", e));
        cuda_types::cuda::CUerror::UNKNOWN
    })?;
    Ok(())
}

#[cfg(windows)]
fn hip_init_on_native_thread(flags: ::core::ffi::c_uint) -> Result<(), cuda_types::cuda::CUerror> {
    unsafe { hipInit(flags) }.map_err(|_| cuda_types::cuda::CUerror::UNKNOWN)
}

pub(crate) fn init(flags: ::core::ffi::c_uint) -> CUresult {
    zluda_log(&format!("[ZLUDA] cuInit({})", flags));
    zluda_log_err(&format!(
        "=== ZLUDA init (flags={}) LOG_ALL={} ===",
        flags,
        crate::is_log_all_enabled()
    ));
    // Disable SDMA before HIP init. Wine's SIGUSR1 APC delivery interrupts the
    // KFD ioctl that waits for SDMA completion, causing hangs. On APUs (unified
    // memory), blit kernels are the correct path anyway — SDMA is for discrete GPUs.
    // Uses setenv with overwrite=0 so user can still override with HSA_ENABLE_SDMA=1.
    unsafe { libc::setenv(c"HSA_ENABLE_SDMA".as_ptr(), c"0".as_ptr(), 0) };
    // Enable HSA VM fault messages for GPU page fault diagnostics.
    // Prints "Memory access fault by GPU node on address %p. Reason: %s" to stderr.
    unsafe { libc::setenv(c"HSA_ENABLE_VM_FAULT_MESSAGE".as_ptr(), c"1".as_ptr(), 0) };
    unsafe { libc::setenv(c"HSA_ENABLE_QUEUE_FAULT_MESSAGE".as_ptr(), c"1".as_ptr(), 0) };
    // Run hipInit on a native pthread to avoid Wine thread incompatibility.
    // ROCm 6.4+ does stack validation and internal allocations during hipInit()
    // that crash with free(): invalid pointer on Wine threads (low NT stack
    // addresses confuse ROCm's thread initialization). A native pthread has a
    // normal 8MB stack at a high address, so ROCm's logic works correctly.
    // hipInit is process-global — initializing on any thread is fine.
    hip_init_on_native_thread(flags)?;
    // Install SIGABRT handler to dump last kernel info before dying.
    // The HSA runtime sends SIGABRT on GPU page faults — our normal
    // hipDeviceSynchronize error check doesn't run in time.
    #[cfg(not(windows))]
    install_fault_signal_handler();
    // Load ROCm debug agent if requested. Must happen after hipInit() so that
    // libhsa-runtime64.so.1 and other ROCm deps are already in the process,
    // allowing dlopen to resolve the agent's transitive dependencies.
    #[cfg(not(windows))]
    if std::env::var("ZLUDA_DEBUG_AGENT").map_or(false, |v| v == "1") {
        load_rocm_debug_agent();
    }
    // Verify HIP runtime version matches what ZLUDA was compiled against.
    // Mismatched versions cause silent GPU failures: hipModuleGetGlobal returns
    // success but doesn't write output parameters, leading to HOST addresses
    // and bytes=0 for .const globals (e.g. gParams/gBounds in PhysX FleX).
    {
        // ROCm version ZLUDA was compiled with (from hip_version.h via hip_runtime-sys build.rs)
        let compiled_major: i32 = env!("HIP_VERSION_MAJOR").parse().unwrap_or(0);
        let compiled_minor: i32 = env!("HIP_VERSION_MINOR").parse().unwrap_or(0);

        let mut runtime_ver = 0i32;
        if unsafe { hip_runtime_sys::hipRuntimeGetVersion(&mut runtime_ver) }.is_ok() {
            let rt_major = runtime_ver / 10_000_000;
            let rt_minor = (runtime_ver / 100_000) % 100;
            let rt_patch = runtime_ver % 100_000;
            zluda_log(&format!(
                "[ZLUDA] HIP runtime version: {}.{}.{} (compiled with {}.{})",
                rt_major, rt_minor, rt_patch, compiled_major, compiled_minor
            ));
            if rt_major != compiled_major {
                let msg = format!(
                    "[ZLUDA] FATAL: ROCm version mismatch! Compiled with {}.{}, runtime is {}.{}. \
                     GPU code compiled with one version cannot be loaded by another. \
                     Update Proton ROCm libs: cd proton-ge-custom/rocm && ./fetch-rocm.sh",
                    compiled_major, compiled_minor, rt_major, rt_minor
                );
                zluda_log(&msg);
                zluda_log_err(&msg);
                return Err(cuda_types::cuda::CUerror::UNKNOWN);
            }
        }
    }
    global_state()?;
    zluda_log("[ZLUDA] cuInit -> SUCCESS");
    zluda_log_err("=== ZLUDA cuInit -> SUCCESS ===");
    Ok(())
}

struct UnknownBuffer<const S: usize> {
    buffer: std::cell::UnsafeCell<[u32; S]>,
}

impl<const S: usize> UnknownBuffer<S> {
    const fn new() -> Self {
        UnknownBuffer {
            buffer: std::cell::UnsafeCell::new([0; S]),
        }
    }
    const fn len(&self) -> usize {
        S
    }
}

unsafe impl<const S: usize> Sync for UnknownBuffer<S> {}

static UNKNOWN_BUFFER1: UnknownBuffer<1024> = UnknownBuffer::new();
static UNKNOWN_BUFFER2: UnknownBuffer<14> = UnknownBuffer::new();

struct DarkApi {}

impl ::dark_api::cuda::CudaDarkApi for DarkApi {
    unsafe extern "system" fn get_module_from_cubin(
        result: *mut cuda_types::cuda::CUmodule,
        fatbinc_wrapper: *const cuda_types::dark_api::FatbincWrapper,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!(
            "dark_api::get_module_from_cubin(fatbinc={:p})",
            fatbinc_wrapper
        ));
        let result = match result.as_mut() {
            Some(p) => p,
            None => return CUresult::ERROR_INVALID_VALUE,
        };
        let data = fatbinc_wrapper
            .cast::<c_void>()
            .as_ref()
            .ok_or(CUerror::INVALID_VALUE)?;
        let r = module::load_data(result, data);
        zluda_log_err(&format!("dark_api::get_module_from_cubin -> {:?}", r));
        r
    }

    unsafe extern "system" fn cudart_interface_fn2(
        pctx: *mut cuda_types::cuda::CUcontext,
        hip_dev: hipDevice_t,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!("dark_api::cudart_interface_fn2(dev={})", hip_dev));
        let pctx = match pctx.as_mut() {
            Some(p) => p,
            None => return CUresult::ERROR_INVALID_VALUE,
        };

        let (_, cu_ctx) = device::get_primary_context(hip_dev)?;
        *pctx = cu_ctx;
        zluda_log_err(&format!(
            "dark_api::cudart_interface_fn2 -> OK (ctx={:p})",
            cu_ctx.0
        ));
        Ok(())
    }

    // PhysX uses these dark API functions - indices 4 and 5 in CUDART_INTERFACE
    // set_device_flags: configures scheduling behavior for CUDA on the specified device
    // set_device: makes the specified device active (like cudaSetDevice)
    unsafe extern "system" fn set_device_flags(
        dev: hipDevice_t,
        flags: std::ffi::c_uint,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!(
            "dark_api::set_device_flags(dev={}, flags={})",
            dev, flags
        ));
        let mut active_dev = 0i32;
        hipGetDevice(&mut active_dev)?;
        if dev != active_dev {
            hipSetDevice(dev)?;
        }
        let result = hipSetDeviceFlags(flags);
        if dev != active_dev {
            let _ = hipSetDevice(active_dev);
        }
        result?;
        Ok(())
    }

    unsafe extern "system" fn set_device(dev: hipDevice_t) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!("dark_api::set_device(dev={})", dev));
        hipSetDevice(dev)?;
        Ok(())
    }

    unsafe extern "system" fn get_module_from_cubin_ext1(
        result: *mut cuda_types::cuda::CUmodule,
        fatbinc_wrapper: *const cuda_types::dark_api::FatbincWrapper,
        arg3: *mut std::ffi::c_void,
        arg4: *mut std::ffi::c_void,
        arg5: u32,
    ) -> cuda_types::cuda::CUresult {
        zluda_log("[ZLUDA] dark_api::get_module_from_cubin_ext1");
        zluda_log_err(&format!(
            "dark_api::get_module_from_cubin_ext1(fatbinc={:p})",
            fatbinc_wrapper
        ));
        if arg3 != ptr::null_mut() || arg4 != ptr::null_mut() || arg5 != 0 {
            zluda_log(
                "[ZLUDA] dark_api::get_module_from_cubin_ext1 -> NOT_SUPPORTED (non-null args)",
            );
            zluda_log_err("dark_api::get_module_from_cubin_ext1 -> NOT_SUPPORTED (non-null args)");
            return CUresult::ERROR_NOT_SUPPORTED;
        }
        let result = match result.as_mut() {
            Some(p) => p,
            None => return CUresult::ERROR_INVALID_VALUE,
        };
        let data = fatbinc_wrapper
            .cast::<c_void>()
            .as_ref()
            .ok_or(CUerror::INVALID_VALUE)?;
        let r = module::load_data(result, data);
        zluda_log(&format!(
            "[ZLUDA] dark_api::get_module_from_cubin_ext1 -> {:?}",
            r
        ));
        zluda_log_err(&format!("dark_api::get_module_from_cubin_ext1 -> {:?}", r));
        r
    }

    unsafe extern "system" fn cudart_interface_fn7(_arg1: usize) -> () {
        zluda_log_err(&format!(
            "dark_api::cudart_interface_fn7(arg1=0x{:x})",
            _arg1
        ));
    }

    unsafe extern "system" fn get_module_from_cubin_ext2(
        fatbin_header: *const cuda_types::dark_api::FatbinHeader,
        result: *mut cuda_types::cuda::CUmodule,
        arg3: *mut std::ffi::c_void,
        arg4: *mut std::ffi::c_void,
        arg5: u32,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!(
            "dark_api::get_module_from_cubin_ext2(fatbin={:p})",
            fatbin_header
        ));
        if arg3 != ptr::null_mut() || arg4 != ptr::null_mut() || arg5 != 0 {
            zluda_log_err("dark_api::get_module_from_cubin_ext2 -> NOT_SUPPORTED (non-null args)");
            return CUresult::ERROR_NOT_SUPPORTED;
        }
        let result = match result.as_mut() {
            Some(p) => p,
            None => return CUresult::ERROR_INVALID_VALUE,
        };
        let data = fatbin_header
            .cast::<c_void>()
            .as_ref()
            .ok_or(CUerror::INVALID_VALUE)?;
        let r = module::load_data(result, data);
        zluda_log_err(&format!("dark_api::get_module_from_cubin_ext2 -> {:?}", r));
        r
    }

    unsafe extern "system" fn get_unknown_buffer1(
        ptr: *mut *mut std::ffi::c_void,
        size: *mut usize,
    ) -> () {
        *ptr = UNKNOWN_BUFFER1.buffer.get() as *mut std::ffi::c_void;
        *size = UNKNOWN_BUFFER1.len();
        zluda_log_err(&format!(
            "dark_api::get_unknown_buffer1 -> ptr={:p}, size={}",
            *ptr, *size
        ));
    }

    unsafe extern "system" fn get_unknown_buffer2(
        ptr: *mut *mut std::ffi::c_void,
        size: *mut usize,
    ) -> () {
        *ptr = UNKNOWN_BUFFER2.buffer.get() as *mut std::ffi::c_void;
        *size = UNKNOWN_BUFFER2.len();
        zluda_log_err(&format!(
            "dark_api::get_unknown_buffer2 -> ptr={:p}, size={}",
            *ptr, *size
        ));
    }

    unsafe extern "system" fn context_local_storage_put(
        cu_ctx: CUcontext,
        key: *mut c_void,
        value: *mut c_void,
        dtor_cb: Option<extern "system" fn(CUcontext, *mut c_void, *mut c_void)>,
    ) -> CUresult {
        zluda_log_err(&format!(
            "dark_api::context_local_storage_put(ctx={:p}, key={:p}, val={:p})",
            cu_ctx.0, key, value
        ));
        let _ctx = if cu_ctx.0 != ptr::null_mut() {
            cu_ctx
        } else {
            let mut current_ctx: CUcontext = CUcontext(ptr::null_mut());
            context::get_current(&mut current_ctx)?;
            current_ctx
        };
        let ctx_obj: &context::Context = FromCuda::<_, CUerror>::from_cuda(&_ctx)?;
        ctx_obj.with_state_mut(|state: &mut context::ContextState| {
            state.storage.insert(
                key as usize,
                context::StorageData {
                    value: value as usize,
                    reset_cb: dtor_cb,
                    handle: _ctx,
                },
            );
            Ok(())
        })?;
        Ok(())
    }

    unsafe extern "system" fn context_local_storage_delete(
        cu_ctx: CUcontext,
        key: *mut c_void,
    ) -> CUresult {
        zluda_log_err(&format!(
            "dark_api::context_local_storage_delete(ctx={:p}, key={:p})",
            cu_ctx.0, key
        ));
        let ctx_obj: &context::Context = FromCuda::<_, CUerror>::from_cuda(&cu_ctx)?;
        ctx_obj.with_state_mut(|state: &mut context::ContextState| {
            state.storage.remove(&(key as usize));
            Ok(())
        })?;
        Ok(())
    }

    unsafe extern "system" fn context_local_storage_get(
        value: *mut *mut c_void,
        cu_ctx: CUcontext,
        key: *mut c_void,
    ) -> CUresult {
        zluda_log_err(&format!(
            "dark_api::context_local_storage_get(ctx={:p}, key={:p})",
            cu_ctx.0, key
        ));
        let mut _ctx: CUcontext;
        if cu_ctx.0 == ptr::null_mut() {
            _ctx = context::get_current_context()?;
        } else {
            _ctx = cu_ctx
        };
        let ctx_obj: &context::Context = FromCuda::<_, CUerror>::from_cuda(&_ctx)?;
        ctx_obj.with_state(|state: &context::ContextState| {
            match state.storage.get(&(key as usize)) {
                Some(data) => *value = data.value as *mut c_void,
                None => return CUresult::ERROR_INVALID_HANDLE,
            }
            Ok(())
        })?;
        Ok(())
    }

    unsafe extern "system" fn ctx_create_v2_bypass(
        pctx: *mut cuda_types::cuda::CUcontext,
        flags: ::std::os::raw::c_uint,
        dev: cuda_types::cuda::CUdevice,
    ) -> cuda_types::cuda::CUresult {
        zluda_log(&format!(
            "[ZLUDA] dark_api::ctx_create_v2_bypass(dev={}, flags={})",
            dev, flags
        ));
        zluda_log_err(&format!(
            "dark_api::ctx_create_v2_bypass(dev={}, flags={})",
            dev, flags
        ));
        // PhysX uses this dark API to create a context without the normal
        // overhead. Implement it the same as cuCtxCreate_v2.
        let result = match pctx.as_mut() {
            Some(pctx) => r#impl::context::create_v2(pctx, flags, dev),
            None => Err(cuda_types::cuda::CUerror::INVALID_VALUE),
        };
        zluda_log(&format!(
            "[ZLUDA] dark_api::ctx_create_v2_bypass -> {:?}",
            result
        ));
        zluda_log_err(&format!("dark_api::ctx_create_v2_bypass -> {:?}", result));
        result
    }

    unsafe extern "system" fn heap_alloc(
        _heap_alloc_record_ptr: *mut *const std::ffi::c_void,
        _arg2: usize,
        _arg3: usize,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!(
            "dark_api::heap_alloc(arg2=0x{:x}, arg3=0x{:x})",
            _arg2, _arg3
        ));
        Err(r#impl::unimplemented())
    }

    unsafe extern "system" fn heap_free(
        _heap_alloc_record_ptr: *const std::ffi::c_void,
        _arg2: *mut usize,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!(
            "dark_api::heap_free(ptr={:p})",
            _heap_alloc_record_ptr
        ));
        Err(r#impl::unimplemented())
    }

    unsafe extern "system" fn device_get_attribute_ext(
        _dev: cuda_types::cuda::CUdevice,
        _attribute: std::ffi::c_uint,
        _unknown: std::ffi::c_int,
        _result: *mut [usize; 2],
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!(
            "dark_api::device_get_attribute_ext(dev={}, attr={}, unknown={})",
            _dev, _attribute, _unknown
        ));
        Err(r#impl::unimplemented())
    }

    unsafe extern "system" fn device_get_something(
        _result: *mut std::ffi::c_uchar,
        _dev: cuda_types::cuda::CUdevice,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!("dark_api::device_get_something(dev={})", _dev));
        Err(r#impl::unimplemented())
    }

    unsafe extern "system" fn integrity_check(
        version: u32,
        unix_seconds: u64,
        result: *mut [u64; 2],
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!(
            "dark_api::integrity_check(version={}, unix_seconds={})",
            version, unix_seconds
        ));
        let current_process = std::process::id();
        let current_thread = os::current_thread();

        let integrity_check_table = EXPORT_TABLE.INTEGRITY_CHECK.as_ptr().cast();
        let cudart_table = EXPORT_TABLE.CUDART_INTERFACE.as_ptr().cast();
        let fn_address = EXPORT_TABLE.INTEGRITY_CHECK[1];

        let devices = get_device_hash_info()?;
        let device_count = devices.len() as u32;
        let get_device = |dev| devices[dev as usize];

        let hash = ::dark_api::integrity_check(
            version,
            unix_seconds,
            cuda_types::cuda::CUDA_VERSION,
            current_process,
            current_thread,
            integrity_check_table,
            cudart_table,
            fn_address,
            device_count,
            get_device,
        );
        *result = hash;
        Ok(())
    }

    unsafe extern "system" fn context_check(
        _ctx_in: cuda_types::cuda::CUcontext,
        result1: *mut u32,
        _result2: *mut *const std::ffi::c_void,
    ) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!("dark_api::context_check(ctx={:p})", _ctx_in.0));
        *result1 = 0;
        CUresult::SUCCESS
    }

    unsafe extern "system" fn check_fn3() -> u32 {
        zluda_log_err("dark_api::check_fn3()");
        0
    }

    unsafe extern "system" fn hybrid_runtime_load_get_proc_address(
        name: *const std::ffi::c_char,
        fn_ptr: *mut *const std::ffi::c_void,
        token: *mut usize,
    ) -> cuda_types::cuda::CUresult {
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| CUerror::INVALID_VALUE)?;
        zluda_log_err(&format!(
            "dark_api::hybrid_runtime_load_get_proc_address(name='{}')",
            name
        ));
        if name != "nvcudart_hybrid64.dll" && name != "nvcudart_hybrid64a.dll" {
            return CUresult::ERROR_INVALID_VALUE;
        }
        let hybrid_runtime = &mut *HYBRID_RUNTIME_HANDLE.lock().map_err(|_| CUerror::UNKNOWN)?;
        let library = match hybrid_runtime {
            Some(lib) => lib,
            None => {
                let library =
                    crate::os::try_load_library(name).map_err(|_| CUerror::FILE_NOT_FOUND)?;
                *hybrid_runtime = Some(library);
                hybrid_runtime.as_ref().unwrap()
            }
        };
        let fn_ = library
            .get::<*const std::ffi::c_void>(b"__cudaGetProcAddress\0")
            .map_err(|_| CUerror::OPERATING_SYSTEM)?;

        *fn_ptr = *fn_;
        *token = 302100128;
        Ok(())
    }

    unsafe extern "system" fn hybrid_runtime_free(token: usize) -> cuda_types::cuda::CUresult {
        zluda_log_err(&format!("dark_api::hybrid_runtime_free(token={})", token));
        if token != 302100128 {
            return CUresult::ERROR_INVALID_HANDLE;
        }
        CUresult::SUCCESS
    }

    unsafe extern "system" fn load_compilers() -> cuda_types::cuda::CUresult {
        zluda_log_err("dark_api::load_compilers()");
        CUresult::SUCCESS
    }
}

static HYBRID_RUNTIME_HANDLE: Mutex<Option<Library>> = Mutex::new(None);

fn get_device_hash_info() -> Result<Vec<::dark_api::DeviceHashinfo>, CUerror> {
    let mut device_count = 0;
    device::get_count(&mut device_count)?;

    (0..device_count)
        .map(|dev| {
            let mut guid = unsafe { mem::zeroed() };
            device::get_uuid_v2(&mut guid, dev)?;

            let mut pci_domain = 0;
            device::get_attribute(
                &mut pci_domain,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                dev,
            )?;

            let mut pci_bus = 0;
            device::get_attribute(
                &mut pci_bus,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                dev,
            )?;

            let mut pci_device = 0;
            device::get_attribute(
                &mut pci_device,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                dev,
            )?;

            Ok(::dark_api::DeviceHashinfo {
                guid: unsafe { mem::transmute(guid) },
                pci_domain,
                pci_bus,
                pci_device,
            })
        })
        .collect()
}

static EXPORT_TABLE: ::dark_api::cuda::CudaDarkApiGlobalTable =
    ::dark_api::cuda::CudaDarkApiGlobalTable::new::<DarkApi>();

pub(crate) fn get_export_table(
    pp_export_table: &mut *const ::core::ffi::c_void,
    p_export_table_id: &CUuuid,
) -> CUresult {
    let guid_str = format!("{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        p_export_table_id.bytes[0], p_export_table_id.bytes[1], p_export_table_id.bytes[2], p_export_table_id.bytes[3],
        p_export_table_id.bytes[4], p_export_table_id.bytes[5],
        p_export_table_id.bytes[6], p_export_table_id.bytes[7],
        p_export_table_id.bytes[8], p_export_table_id.bytes[9],
        p_export_table_id.bytes[10], p_export_table_id.bytes[11],
        p_export_table_id.bytes[12], p_export_table_id.bytes[13],
        p_export_table_id.bytes[14], p_export_table_id.bytes[15]);
    if let Some(table) = EXPORT_TABLE.get(p_export_table_id) {
        *pp_export_table = table.start();
        zluda_log(&format!(
            "[ZLUDA] cuGetExportTable({}) -> SUCCESS",
            guid_str
        ));
        zluda_log_err(&format!("cuGetExportTable({}) -> SUCCESS", guid_str));
        cuda_types::cuda::CUresult::SUCCESS
    } else {
        zluda_log(&format!(
            "[ZLUDA] cuGetExportTable({}) -> ERROR_INVALID_VALUE (unknown GUID!)",
            guid_str
        ));
        zluda_log_err(&format!(
            "cuGetExportTable({}) -> ERROR_INVALID_VALUE (unknown GUID!)",
            guid_str
        ));
        cuda_types::cuda::CUresult::ERROR_INVALID_VALUE
    }
}

pub(crate) fn get_version(version: &mut ::core::ffi::c_int) -> CUresult {
    *version = cuda_types::cuda::CUDA_VERSION as i32;
    zluda_log(&format!("[ZLUDA] cuDriverGetVersion -> {}", *version));
    Ok(())
}

pub(crate) unsafe fn get_proc_address(
    symbol: &CStr,
    pfn: &mut *mut ::core::ffi::c_void,
    cuda_version: ::core::ffi::c_int,
    flags: cuda_types::cuda::cuuint64_t,
) -> CUresult {
    get_proc_address_v2(symbol, pfn, cuda_version, flags, None)
}

pub(crate) unsafe fn get_proc_address_v2(
    symbol: &CStr,
    pfn: &mut *mut ::core::ffi::c_void,
    cuda_version: ::core::ffi::c_int,
    flags: cuda_types::cuda::cuuint64_t,
    symbol_status: Option<&mut cuda_types::cuda::CUdriverProcAddressQueryResult>,
) -> CUresult {
    // This implementation is mostly the same as cuGetProcAddress_v2 in zluda_trace. We may want to factor out the duplication at some point.
    fn raw_match(name: &[u8], flag: u64, version: i32) -> *mut ::core::ffi::c_void {
        use crate::*;
        include!("../../../zluda_bindgen/src/process_table.rs")
    }
    let fn_ptr = raw_match(symbol.to_bytes(), flags, cuda_version);
    let status_str = if fn_ptr as usize == 0 {
        "NOT_FOUND"
    } else if fn_ptr as usize == usize::MAX {
        "VERSION_NOT_SUFFICIENT"
    } else {
        "FOUND"
    };
    zluda_log(&format!(
        "[ZLUDA] cuGetProcAddress('{}', ver={}, flags={}) -> {:?}",
        symbol.to_string_lossy(),
        cuda_version,
        flags,
        status_str
    ));
    zluda_log_err(&format!(
        "cuGetProcAddress('{}', ver={}, flags={}) -> {}",
        symbol.to_string_lossy(),
        cuda_version,
        flags,
        status_str
    ));
    match fn_ptr as usize {
        0 => {
            if let Some(symbol_status) = symbol_status {
                *symbol_status = cuda_types::cuda::CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            }
            *pfn = ptr::null_mut();
            CUresult::ERROR_NOT_FOUND
        }
        usize::MAX => {
            if let Some(symbol_status) = symbol_status {
                *symbol_status = cuda_types::cuda::CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;
            }
            *pfn = ptr::null_mut();
            CUresult::ERROR_NOT_FOUND
        }
        _ => {
            if let Some(symbol_status) = symbol_status {
                *symbol_status =
                    cuda_types::cuda::CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SUCCESS;
            }
            *pfn = fn_ptr;
            Ok(())
        }
    }
}

pub(crate) fn profiler_start() -> CUresult {
    Ok(())
}

pub(crate) fn profiler_stop() -> CUresult {
    Ok(())
}

pub(crate) unsafe fn thread_exchange_stream_capture_mode(
    mode: *mut hipStreamCaptureMode,
) -> hipError_t {
    hipThreadExchangeStreamCaptureMode(mode)
}

pub(crate) unsafe fn occupancy_max_active_blocks_per_multiprocessor_with_flags(
    num_blocks: &mut ::core::ffi::c_int,
    func: &function::Function,
    block_size: ::core::ffi::c_int,
    dynamic_smem_size: usize,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        num_blocks,
        func.base,
        block_size,
        dynamic_smem_size,
        flags,
    )?;
    *num_blocks = (*num_blocks).max(1);
    Ok(())
}

pub(crate) unsafe fn occupancy_max_potential_block_size(
    grid_size: &mut ::core::ffi::c_int,
    block_size: &mut ::core::ffi::c_int,
    f: &function::Function,
    _block_size_to_dynamic_smem_size: CUoccupancyB2DSize,
    dyn_shared_mem_per_blk: usize,
    block_size_limit: ::core::ffi::c_int,
) -> hipError_t {
    hipModuleOccupancyMaxPotentialBlockSize(
        grid_size,
        block_size,
        f.base,
        dyn_shared_mem_per_blk,
        block_size_limit,
    )?;
    Ok(())
}

pub(crate) unsafe fn launch_kernel_ex(
    config: &cuda_types::cuda::CUlaunchConfig,
    f: &function::Function,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> CUresult {
    let attrs = std::slice::from_raw_parts(config.attrs, config.numAttrs as usize);
    if attrs.iter().any(|&attr| {
        !(attr.id == CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            && attr.value.programmaticStreamSerializationAllowed == 0)
    }) {
        return CUresult::ERROR_NOT_SUPPORTED;
    }
    function::launch_kernel(
        f,
        config.gridDimX,
        config.gridDimY,
        config.gridDimZ,
        config.blockDimX,
        config.blockDimY,
        config.blockDimZ,
        config.sharedMemBytes,
        FromCuda::<_, CUerror>::from_cuda(&config.hStream)?,
        kernel_params,
        extra,
    )?;
    Ok(())
}

pub(crate) unsafe fn get_error_string(
    error: cuda_types::cuda::CUresult,
    error_string: &mut *const ::core::ffi::c_char,
) -> CUresult {
    *error_string = match error {
        CUresult::SUCCESS => c"no error".as_ptr(),
        CUresult::ERROR_INVALID_VALUE => c"invalid value".as_ptr(),
        CUresult::ERROR_OUT_OF_MEMORY => c"out of memory".as_ptr(),
        CUresult::ERROR_NOT_INITIALIZED => c"driver not initialized".as_ptr(),
        CUresult::ERROR_DEINITIALIZED => c"driver deinitialized".as_ptr(),
        CUresult::ERROR_NO_DEVICE => c"no CUDA-capable device is detected".as_ptr(),
        CUresult::ERROR_INVALID_DEVICE => c"invalid device".as_ptr(),
        CUresult::ERROR_INVALID_IMAGE => c"invalid kernel image".as_ptr(),
        CUresult::ERROR_INVALID_CONTEXT => c"invalid context".as_ptr(),
        CUresult::ERROR_CONTEXT_ALREADY_CURRENT => c"context already current".as_ptr(),
        CUresult::ERROR_MAP_FAILED => c"map failed".as_ptr(),
        CUresult::ERROR_UNMAP_FAILED => c"unmap failed".as_ptr(),
        CUresult::ERROR_ARRAY_IS_MAPPED => c"array is mapped".as_ptr(),
        CUresult::ERROR_ALREADY_MAPPED => c"already mapped".as_ptr(),
        CUresult::ERROR_NO_BINARY_FOR_GPU => c"no binary for GPU".as_ptr(),
        CUresult::ERROR_ALREADY_ACQUIRED => c"already acquired".as_ptr(),
        CUresult::ERROR_NOT_MAPPED => c"not mapped".as_ptr(),
        CUresult::ERROR_NOT_SUPPORTED => c"operation not supported".as_ptr(),
        CUresult::ERROR_INVALID_SOURCE => c"invalid source".as_ptr(),
        CUresult::ERROR_FILE_NOT_FOUND => c"file not found".as_ptr(),
        CUresult::ERROR_INVALID_HANDLE => c"invalid handle".as_ptr(),
        CUresult::ERROR_NOT_READY => c"not ready".as_ptr(),
        CUresult::ERROR_ILLEGAL_ADDRESS => c"illegal address".as_ptr(),
        CUresult::ERROR_LAUNCH_OUT_OF_RESOURCES => c"launch out of resources".as_ptr(),
        CUresult::ERROR_LAUNCH_TIMEOUT => c"launch timeout".as_ptr(),
        CUresult::ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => c"launch incompatible texturing".as_ptr(),
        CUresult::ERROR_PEER_ACCESS_ALREADY_ENABLED => c"peer access already enabled".as_ptr(),
        CUresult::ERROR_PEER_ACCESS_NOT_ENABLED => c"peer access not enabled".as_ptr(),
        CUresult::ERROR_PRIMARY_CONTEXT_ACTIVE => c"primary context active".as_ptr(),
        CUresult::ERROR_CONTEXT_IS_DESTROYED => c"context is destroyed".as_ptr(),
        CUresult::ERROR_ASSERT => c"device-side assert triggered".as_ptr(),
        CUresult::ERROR_TOO_MANY_PEERS => c"too many peers".as_ptr(),
        CUresult::ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
            c"host memory already registered".as_ptr()
        }
        CUresult::ERROR_HOST_MEMORY_NOT_REGISTERED => c"host memory not registered".as_ptr(),
        CUresult::ERROR_UNKNOWN => c"unknown error".as_ptr(),
        _ => c"error".as_ptr(),
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::i32;

    use crate::r#impl::driver::AllocationInfo;
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::CUcontext;

    #[test_cuda]
    fn init(api: impl CudaApi) {
        api.cuInit(0);
    }

    #[test]
    fn get_allocation() {
        let ctx1 = CUcontext(0x1234 as _);
        let ctx2 = CUcontext(0x5678 as _);
        let mut alloc_info = super::Allocations::new();
        alloc_info.insert(0x1000, 4, ctx1);
        alloc_info.insert(0x2000, 8, ctx2);
        for i in 0..4 {
            assert_eq!(
                alloc_info.get_offset_and_info(0x1000 + i),
                Some((
                    i,
                    AllocationInfo {
                        size: 4,
                        context: ctx1
                    }
                ))
            );
        }
        assert_eq!(alloc_info.get_offset_and_info(0x1000 + 4), None);
        for i in 0..8 {
            assert_eq!(
                alloc_info.get_offset_and_info(0x2000 + i),
                Some((
                    i,
                    AllocationInfo {
                        size: 8,
                        context: ctx2
                    }
                ))
            );
        }
        assert_eq!(alloc_info.get_offset_and_info(0x2000 + 8), None);
    }

    #[test_cuda]
    fn primary_context_is_inactive_on_init(api: impl CudaApi) {
        api.cuInit(0);
        let mut flags = u32::MAX;
        let mut active = i32::MAX;
        api.cuDevicePrimaryCtxGetState(0, &mut flags, &mut active);
        assert_eq!(flags, 0);
        assert_eq!(active, 0);
    }

    #[test_cuda]
    unsafe fn cudart_interface_fn2_creates_inactive_primary_ctx(api: impl CudaApi) {
        api.cuInit(0);
        let mut table_ptr = std::ptr::null();
        api.cuGetExportTable(&mut table_ptr, &dark_api::cuda::CudartInterface::GUID);
        let cuda_rt_iface = dark_api::cuda::CudartInterface::new(table_ptr);
        let mut dark_ctx = std::mem::zeroed();
        cuda_rt_iface
            .cudart_interface_fn2(&mut dark_ctx, 0)
            .unwrap();
        let mut flags = u32::MAX;
        let mut active = i32::MAX;
        api.cuDevicePrimaryCtxGetState(0, &mut flags, &mut active);
        assert_eq!(flags, 0);
        assert_eq!(active, 0);
        let mut primary_ctx = std::mem::zeroed();
        api.cuDevicePrimaryCtxRetain(&mut primary_ctx, 0);
        assert_eq!(dark_ctx.0, primary_ctx.0);
    }
}

// Forwarding functions for cuMipmappedArray* — the cuda_normalize_fn macro
// routes these to driver:: because "mipmapped" is not in the MODULES list.
pub(crate) unsafe fn mipmapped_array_create(
    p_handle: &mut cuda_types::cuda::CUmipmappedArray,
    p_mipmapped_array_desc: &cuda_types::cuda::CUDA_ARRAY3D_DESCRIPTOR,
    num_mipmap_levels: u32,
) -> cuda_types::cuda::CUresult {
    super::array::mipmapped_create(p_handle, p_mipmapped_array_desc, num_mipmap_levels)
}

pub(crate) unsafe fn mipmapped_array_destroy(
    h_mipmapped_array: cuda_types::cuda::CUmipmappedArray,
) -> cuda_types::cuda::CUresult {
    super::array::mipmapped_destroy(h_mipmapped_array)
}

pub(crate) unsafe fn mipmapped_array_get_level(
    p_level_array: &mut cuda_types::cuda::CUarray,
    h_mipmapped_array: cuda_types::cuda::CUmipmappedArray,
    level: u32,
) -> cuda_types::cuda::CUresult {
    super::array::mipmapped_get_level(p_level_array, h_mipmapped_array, level)
}

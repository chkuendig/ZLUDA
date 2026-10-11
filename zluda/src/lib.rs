use cuda_types::cuda::CUerror;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Global call counter for rate-limiting high-frequency API logging.
static CALL_COUNTER: AtomicU64 = AtomicU64::new(0);
const LOG_ALL_THRESHOLD: u64 = 5000;
const LOG_SAMPLE_INTERVAL: u64 = 1000;

/// Process-relative timestamp origin for log entries.
static PROCESS_START: OnceLock<Instant> = OnceLock::new();

/// 0 = not checked, 1 = disabled, 2 = enabled (ZLUDA_LOG_ALL env var)
static ZLUDA_LOG_ALL: AtomicU8 = AtomicU8::new(0);

/// Milliseconds since ZLUDA was first loaded.
pub(crate) fn elapsed_ms() -> u64 {
    PROCESS_START
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// When ZLUDA_LOG_ALL=1, log every CUDA API call without rate-limiting.
pub(crate) fn is_log_all_enabled() -> bool {
    match ZLUDA_LOG_ALL.load(Ordering::Relaxed) {
        0 => {
            let enabled = std::env::var("ZLUDA_LOG_ALL").map_or(false, |v| v == "1");
            ZLUDA_LOG_ALL.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
            enabled
        }
        2 => true,
        _ => false,
    }
}

pub(crate) mod r#impl;
#[cfg_attr(windows, path = "os_win.rs")]
#[cfg_attr(not(windows), path = "os_unix.rs")]
mod os;

static INITIALIZED: AtomicBool = AtomicBool::new(true);
// 0 = not checked, 1 = disabled, 2 = enabled
static ZLUDA_DEBUG: AtomicU8 = AtomicU8::new(0);
static DEBUG_LOG: Mutex<Option<std::fs::File>> = Mutex::new(None);

fn is_debug_enabled() -> bool {
    match ZLUDA_DEBUG.load(Ordering::Relaxed) {
        0 => {
            let enabled = std::env::var("ZLUDA_DEBUG").map_or(false, |v| v == "1");
            if enabled {
                if let Ok(f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/zluda_debug.log")
                {
                    if let Ok(mut guard) = DEBUG_LOG.lock() {
                        *guard = Some(f);
                    }
                }
            }
            ZLUDA_DEBUG.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
            enabled
        }
        2 => true,
        _ => false,
    }
}

#[allow(unused)]
fn debug_log(msg: &str) {
    eprintln!("{}", msg);
    if let Ok(mut guard) = DEBUG_LOG.lock() {
        if let Some(ref mut f) = *guard {
            let _ = writeln!(f, "{}", msg);
            let _ = f.flush();
        }
    }
}
pub(crate) fn initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

#[cfg_attr(not(windows), dtor::dtor)]
fn deinitialize() {
    INITIALIZED.store(false, Ordering::SeqCst);
    r#impl::recording::shutdown();
    r#impl::launch_survey::shutdown();
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            #[allow(unused_variables)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                // Always log unimplemented calls to catch missing APIs
                crate::r#impl::driver::zluda_log_err(
                    &format!("UNIMPL {}", stringify!($fn_name)));
                if is_debug_enabled() {
                    debug_log(&format!("[ZLUDA] UNIMPLEMENTED: {}", stringify!($fn_name)));
                }
                Err(r#impl::unimplemented())
            }
        )*
    };
}

macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                if !initialized() {
                    return Err(CUerror::DEINITIALIZED);
                }
                let result = cuda_macros::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(zluda_common::FromCuda::<_, CUerror>::from_cuda(&$arg_id)?),*);
                if is_debug_enabled() {
                    match &result {
                        Ok(()) => debug_log(&format!("[ZLUDA] {} -> OK", stringify!($fn_name))),
                        Err(e) => debug_log(&format!("[ZLUDA] {} -> {:?}", stringify!($fn_name), e)),
                    }
                }
                // Log calls to lightweight log for investigation
                // ZLUDA_LOG_ALL=1: log every call; otherwise rate-limit
                {
                    let fn_name = stringify!($fn_name);
                    let count = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let log_all = is_log_all_enabled();
                    let should_log = log_all
                        || count < LOG_ALL_THRESHOLD
                        || count % LOG_SAMPLE_INTERVAL == 0
                        || result.is_err();
                    if should_log {
                        match &result {
                            Ok(()) => crate::r#impl::driver::zluda_log_err(
                                &format!("#{} OK {}", count, fn_name)),
                            Err(ref e) => crate::r#impl::driver::zluda_log_err(
                                &format!("#{} ERR {} -> {:?}", count, fn_name, e)),
                        }
                    }
                }
                result?;
                Ok(())
            }
        )*
    };
}

macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                if !initialized() {
                    return Err(CUerror::DEINITIALIZED);
                }
                let result = cuda_macros::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(zluda_common::FromCuda::<_, CUerror>::from_cuda(&$arg_id)?),*);
                if is_debug_enabled() {
                    match &result {
                        Ok(()) => debug_log(&format!("[ZLUDA] {} -> OK", stringify!($fn_name))),
                        Err(e) => debug_log(&format!("[ZLUDA] {} -> {:?}", stringify!($fn_name), e)),
                    }
                }
                // Log calls to lightweight log for investigation
                {
                    let fn_name = stringify!($fn_name);
                    let count = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let log_all = is_log_all_enabled();
                    let should_log = log_all
                        || count < LOG_ALL_THRESHOLD
                        || count % LOG_SAMPLE_INTERVAL == 0
                        || result.is_err();
                    if should_log {
                        match &result {
                            Ok(()) => crate::r#impl::driver::zluda_log_err(
                                &format!("#{} OK {}", count, fn_name)),
                            Err(ref e) => crate::r#impl::driver::zluda_log_err(
                                &format!("#{} ERR {} -> {:?}", count, fn_name, e)),
                        }
                    }
                }
                result?;
                Ok(())
            }
        )*
    };
}

cuda_macros::cuda_function_declarations!(
    unimplemented,
    implemented
        <= [
            cuCtxCreate_v2,
            cuCtxCreate_v4,
            cuCtxDestroy_v2,
            cuCtxGetApiVersion,
            cuCtxGetCurrent,
            cuCtxGetDevice_v2,
            cuCtxGetDevice,
            cuCtxGetLimit,
            cuCtxGetSharedMemConfig,
            cuCtxGetStreamPriorityRange,
            cuCtxPopCurrent_v2,
            cuCtxPopCurrent,
            cuCtxPushCurrent_v2,
            cuCtxPushCurrent,
            cuCtxSetCurrent,
            cuCtxSetCacheConfig,
            cuCtxSetFlags,
            cuCtxSetLimit,
            cuCtxSetSharedMemConfig,
            cuCtxSynchronize_v2,
            cuCtxSynchronize,
            cuDeviceCanAccessPeer,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
            cuDeviceGetPCIBusId,
            cuDeviceGetProperties,
            cuDeviceGetUuid_v2,
            cuDeviceGetUuid,
            cuDevicePrimaryCtxGetState,
            cuDevicePrimaryCtxRelease_v2,
            cuDevicePrimaryCtxRelease,
            cuDevicePrimaryCtxReset,
            cuDevicePrimaryCtxRetain,
            cuDevicePrimaryCtxSetFlags_v2,
            cuDevicePrimaryCtxSetFlags,
            cuDeviceTotalMem,
            cuDeviceTotalMem_v2,
            cuDriverGetVersion,
            cuEventCreate,
            cuEventDestroy_v2,
            cuEventElapsedTime_v2,
            cuEventElapsedTime,
            cuEventQuery,
            cuEventRecord,
            cuEventRecordWithFlags,
            cuEventSynchronize,
            cuFuncGetAttribute,
            cuFuncSetAttribute,
            cuFuncSetCacheConfig,
            cuGetErrorString,
            cuGetExportTable,
            cuGetProcAddress_v2,
            cuGetProcAddress,
            cuGraphDestroy,
            cuGraphExecDestroy,
            cuGraphExecUpdate_v2,
            cuGraphGetNodes,
            cuGraphInstantiateWithFlags,
            cuGraphLaunch,
            cuInit,
            cuKernelGetAttribute,
            cuKernelGetFunction,
            cuKernelSetAttribute,
            cuLaunchKernel,
            cuLaunchKernelEx,
            cuLibraryGetGlobal,
            cuLibraryGetKernel,
            cuLibraryGetModule,
            cuLibraryLoadData,
            cuLibraryUnload,
            cuMemAlloc_v2,
            cuMemAllocHost_v2,
            cuMemAllocPitch_v2,
            cuMemcpy,
            cuMemcpy2D_v2,
            cuMemcpy2DAsync_v2,
            cuMemcpy2DUnaligned_v2,
            cuMemcpy3D_v2,
            cuMemcpy3DAsync_v2,
            cuMemcpyAsync,
            cuMemcpyAtoH_v2,
            cuMemcpyAtoHAsync_v2,
            cuMemcpyHtoA_v2,
            cuMemcpyHtoAAsync_v2,
            cuMemcpyAtoD_v2,
            cuMemcpyDtoA_v2,
            cuMemcpyAtoA_v2,
            cuMemcpyDtoD_v2,
            cuMemcpyDtoDAsync_v2,
            cuMemcpyDtoH_v2_ptds,
            cuMemcpyDtoH_v2,
            cuMemcpyDtoHAsync_v2,
            cuMemcpyHtoD_v2_ptds,
            cuMemcpyHtoD_v2,
            cuMemcpyHtoDAsync_v2,
            cuMemFree_v2,
            cuMemFreeHost,
            cuMemGetAddressRange_v2,
            cuMemGetAllocationGranularity,
            cuMemGetInfo_v2,
            cuMemHostAlloc,
            cuMemHostGetDevicePointer_v2,
            cuMemRetainAllocationHandle,
            cuMemsetD16_v2,
            cuMemsetD16Async,
            cuMemsetD2D32_v2,
            cuMemsetD2D32Async,
            cuMemsetD32_v2,
            cuMemsetD32Async,
            cuMemsetD8_v2,
            cuMemsetD8Async,
            cuModuleGetFunction,
            cuModuleGetGlobal_v2,
            cuModuleGetLoadingMode,
            cuModuleGetSurfRef,
            cuModuleGetTexRef,
            cuModuleLoad,
            cuModuleLoadData,
            cuModuleLoadDataEx,
            cuModuleLoadFatBinary,
            cuModuleUnload,
            cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
            cuOccupancyMaxPotentialBlockSize,
            cuPointerGetAttribute,
            cuPointerGetAttributes,
            cuProfilerStart,
            cuProfilerStop,
            cuStreamBeginCapture_v2,
            cuStreamCreate,
            cuStreamCreateWithPriority,
            cuStreamDestroy_v2,
            cuStreamGetFlags,
            cuStreamGetFlags_ptsz,
            cuStreamEndCapture,
            cuStreamGetCaptureInfo_v2,
            cuStreamGetCaptureInfo_v3,
            cuStreamIsCapturing,
            cuStreamQuery,
            cuStreamSynchronize,
            cuStreamWaitEvent,
            cuThreadExchangeStreamCaptureMode,
            cuArrayCreate_v2,
            cuArrayDestroy,
            cuArrayGetDescriptor_v2,
            cuArray3DCreate_v2,
            cuArray3DGetDescriptor_v2,
            cuMipmappedArrayCreate,
            cuMipmappedArrayDestroy,
            cuMipmappedArrayGetLevel,
            cuTexRefSetAddress_v2,
            cuTexRefSetAddress2D_v3,
            cuTexRefSetArray,
            cuTexRefSetMipmappedArray,
            cuTexRefSetFormat,
            cuTexRefSetAddressMode,
            cuTexRefSetFilterMode,
            cuTexRefSetMipmapFilterMode,
            cuTexRefSetMipmapLevelBias,
            cuTexRefSetMipmapLevelClamp,
            cuTexRefSetMaxAnisotropy,
            cuTexRefSetBorderColor,
            cuTexRefSetFlags,
            cuTexObjectCreate,
            cuTexObjectDestroy,
            cuSurfObjectCreate,
            cuSurfObjectDestroy,
            cuSurfRefSetArray,
            cuGraphicsGLRegisterBuffer,
            cuGraphicsGLRegisterImage,
            cuGraphicsMapResources,
            cuGraphicsMapResources_ptsz,
            cuGraphicsUnmapResources,
            cuGraphicsUnmapResources_ptsz,
            cuGraphicsSubResourceGetMappedArray,
            cuGraphicsResourceGetMappedPointer_v2,
            cuGraphicsResourceSetMapFlags_v2,
            cuGraphicsResourceSetMapFlags,
            cuGraphicsUnregisterResource,
            cuGraphicsResourceGetMappedMipmappedArray,
        ],
    // cuLaunchKernel_ptsz is the per-thread default stream variant.
    // cuGetProcAddress returns it for cuLaunchKernel with certain version/flags.
    // Without this, it falls through to the unimplemented stub and fails silently.
    implemented_in_function <= [cuLaunchKernel, cuLaunchKernel_ptsz,]
);

#[cfg(test)]
mod tests;

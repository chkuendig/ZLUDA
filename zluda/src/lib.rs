use cuda_types::cuda::CUerror;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;
use std::io::Write;

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
                    .create(true).append(true)
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
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            #[allow(unused_variables)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
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
                // Always log errors to lightweight error log
                {
                    use std::io::Write;
                    let fn_name = stringify!($fn_name);
                    // Log success for critical PhysX init functions
                    let log_success = matches!(fn_name,
                        "cuModuleLoadDataEx" | "cuModuleGetTexRef" | "cuModuleGetSurfRef" |
                        "cuModuleGetFunction" | "cuModuleGetGlobal_v2" |
                        "cuLaunchKernel" | "cuTexRefSetAddress_v2" |
                        "cuTexRefSetArray" | "cuTexRefSetFormat" | "cuTexRefSetFilterMode" |
                        "cuTexRefSetAddressMode" | "cuTexRefSetFlags" |
                        "cuTexObjectCreate" | "cuCtxCreate_v2" |
                        "cuMemAlloc_v2" | "cuMemHostAlloc" | "cuMemcpyHtoDAsync_v2"
                    );
                    match &result {
                        Ok(()) if log_success => {
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/zluda_tex.log") {
                                let _ = writeln!(f, "OK {}", fn_name);
                            }
                        }
                        Err(ref e) => {
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/zluda_tex.log") {
                                let _ = writeln!(f, "ERR {} -> {:?}", fn_name, e);
                            }
                        }
                        _ => {}
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
                // Always log errors to lightweight error log
                {
                    use std::io::Write;
                    let fn_name = stringify!($fn_name);
                    let log_success = matches!(fn_name,
                        "cuModuleLoadDataEx" | "cuModuleGetTexRef" | "cuModuleGetSurfRef" |
                        "cuModuleGetFunction" | "cuLaunchKernel"
                    );
                    match &result {
                        Ok(()) if log_success => {
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/zluda_tex.log") {
                                let _ = writeln!(f, "OK {}", fn_name);
                            }
                        }
                        Err(ref e) => {
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/zluda_tex.log") {
                                let _ = writeln!(f, "ERR {} -> {:?}", fn_name, e);
                            }
                        }
                        _ => {}
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
            cuCtxDestroy_v2,
            cuCtxGetApiVersion,
            cuCtxGetCurrent,
            cuCtxGetDevice_v2,
            cuCtxGetDevice,
            cuCtxGetLimit,
            cuCtxGetStreamPriorityRange,
            cuCtxPopCurrent_v2,
            cuCtxPopCurrent,
            cuCtxPushCurrent_v2,
            cuCtxPushCurrent,
            cuCtxSetCurrent,
            cuCtxSetFlags,
            cuCtxSetLimit,
            cuCtxSynchronize_v2,
            cuCtxSynchronize,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
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
            cuMemcpy2D_v2,
            cuMemcpy2DUnaligned_v2,
            cuMemcpyAsync,
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
    implemented_in_function <= [cuLaunchKernel,]
);

#[cfg(test)]
mod tests;

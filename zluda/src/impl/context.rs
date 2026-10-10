use super::module;
use cuda_types::cuda::*;
use hip_runtime_sys::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{cell::RefCell, ffi::c_void, ptr, sync::Mutex};
use zluda_common::{FromCuda, ZludaObject};

thread_local! {
    pub(crate) static STACK: RefCell<Vec<(CUcontext, hipDevice_t)>> = RefCell::new(Vec::new());
}

pub(crate) struct Context {
    pub(crate) device: hipDevice_t,
    pub(crate) alloc_stream: hipStream_t,
    pub(crate) state: Mutex<ContextState>,
}

pub(crate) struct ContextState {
    pub(crate) ref_count: u32,
    pub(crate) flags: u32,
    pub(crate) modules: FxHashSet<CUmodule>,
    pub(crate) storage: FxHashMap<usize, StorageData>,
}

pub(crate) struct StorageData {
    pub(crate) value: usize,
    pub(crate) reset_cb: Option<extern "system" fn(CUcontext, *mut c_void, *mut c_void)>,
    pub(crate) handle: CUcontext,
}

impl ContextState {
    pub(crate) fn new() -> Self {
        ContextState {
            ref_count: 0,
            flags: 0,
            modules: FxHashSet::default(),
            storage: FxHashMap::default(),
        }
    }

    pub(crate) fn reset(&mut self) -> CUresult {
        for (key, data) in self.storage.iter_mut() {
            if let Some(_cb) = data.reset_cb {
                // TODO: check that these callbacks do not call into the CUDA driver
                // since this could result in a recursive mutex lock.
                _cb(data.handle, *key as *mut c_void, data.value as *mut c_void);
            }
        }
        self.ref_count = 0;
        self.flags = 0;
        // drop all modules and return first error if any
        let result = self.modules.drain().fold(Ok(()), |res: CUresult, hmod| {
            match (res, zluda_common::drop_checked::<module::Module>(hmod)) {
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(e),
                _ => Ok(()),
            }
        });
        self.storage.clear();
        result
    }
}

impl Context {
    pub(crate) fn new(device: hipDevice_t) -> Result<Self, CUerror> {
        let mut alloc_stream = unsafe { std::mem::zeroed() };
        unsafe { hipStreamCreateWithFlags(&mut alloc_stream, hipStreamNonBlocking) }?;
        Ok(Self {
            device,
            alloc_stream,
            state: Mutex::new(ContextState::new()),
        })
    }

    pub(crate) fn with_state(&self, fn_: impl FnOnce(&ContextState) -> CUresult) -> CUresult {
        match self.state.lock() {
            Ok(guard) => fn_(&*guard),
            Err(_) => CUresult::ERROR_UNKNOWN,
        }
    }

    pub(crate) fn with_state_mut(
        &self,
        fn_: impl FnOnce(&mut ContextState) -> CUresult,
    ) -> CUresult {
        match self.state.lock() {
            Ok(mut guard) => fn_(&mut *guard),
            Err(_) => CUresult::ERROR_UNKNOWN,
        }
    }
}

impl ZludaObject for Context {
    const COOKIE: usize = 0x5f867c6d9cb73315;

    type Error = CUerror;
    type CudaHandle = CUcontext;

    fn drop_checked(&mut self) -> CUresult {
        Ok(())
    }
}

// This type is used in lieu of Option<&Context> because of trait coherence rules.
// We can't implement FromCuda for Option<&T> in this crate because Option<T> is defined in the standard library.
pub(crate) enum MaybeContextRef<'a> {
    None,
    Some(&'a Context),
}

impl<'a>
    zluda_common::FromCuda<
        'a,
        <Context as ZludaObject>::CudaHandle,
        <Context as ZludaObject>::Error,
    > for MaybeContextRef<'a>
{
    fn from_cuda(
        handle: &'a <Context as ZludaObject>::CudaHandle,
    ) -> Result<MaybeContextRef<'a>, <Context as ZludaObject>::Error> {
        match zluda_common::as_ref(handle) {
            Some(ctx) => Ok(MaybeContextRef::Some(ctx.as_result()?)),
            None => Ok(MaybeContextRef::None),
        }
    }
}

pub(crate) fn get_current_device() -> Result<hipDevice_t, CUerror> {
    STACK.with(|stack| {
        stack
            .try_borrow()
            .map_err(|_| CUerror::UNKNOWN)
            .and_then(|s| {
                s.last()
                    .ok_or(CUerror::INVALID_CONTEXT)
                    .map(|(_, dev)| *dev)
            })
    })
}

pub(crate) fn get_current_context() -> Result<CUcontext, CUerror> {
    if let Some(ctx) = STACK.with(|stack| stack.borrow().last().copied().map(|(ctx, _)| ctx)) {
        return Ok(ctx);
    }
    // Fall back to primary context for device 0 when stack is empty.
    // Real CUDA makes the primary context implicitly available after cuInit.
    // PhysX creates/destroys explicit contexts during probe, then expects
    // the primary context to still be accessible.
    match super::device::get_primary_context(0) {
        Ok((_, cu_ctx)) => Ok(cu_ctx),
        Err(_) => Err(CUerror::INVALID_CONTEXT),
    }
}

pub(crate) unsafe fn get_limit(pvalue: *mut usize, limit: hipLimit_t) -> hipError_t {
    unsafe { hipDeviceGetLimit(pvalue, limit) }
}

pub(crate) fn set_limit(limit: hipLimit_t, value: usize) -> hipError_t {
    unsafe { hipDeviceSetLimit(limit, value) }
}

pub(crate) fn set_cache_config(config: CUfunc_cache) -> CUresult {
    get_current_context()?;
    match config {
        CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE
        | CUfunc_cache::CU_FUNC_CACHE_PREFER_SHARED
        | CUfunc_cache::CU_FUNC_CACHE_PREFER_L1
        | CUfunc_cache::CU_FUNC_CACHE_PREFER_EQUAL => Ok(()),
        _ => Err(CUerror::INVALID_VALUE),
    }
}

pub(crate) fn synchronize() -> hipError_t {
    super::driver::sync_timeline_log("ctx", 0);
    unsafe { hipDeviceSynchronize() }
}

pub(crate) fn synchronize_v2(ctx: &Context) -> hipError_t {
    super::driver::sync_timeline_log("ctx_v2", ctx.device as usize);
    let mut current_device = 0;
    unsafe { hipGetDevice(&mut current_device)? };
    let _dev_drop = if current_device != ctx.device {
        unsafe { hipSetDevice(ctx.device) }?;
        Some(SetDeviceOnDrop(current_device))
    } else {
        None
    };
    unsafe { hipDeviceSynchronize() }
}

struct SetDeviceOnDrop(i32);

impl Drop for SetDeviceOnDrop {
    fn drop(&mut self) {
        unsafe { hipSetDevice(self.0) }.ok();
    }
}

pub(crate) fn set_current(raw_ctx: CUcontext) -> CUresult {
    let new_device = if raw_ctx.0 == ptr::null_mut() {
        STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((_, old_device)) = stack.pop() {
                if let Some((_, new_device)) = stack.last() {
                    if old_device != *new_device {
                        return Some(*new_device);
                    }
                }
            }
            None
        })
    } else {
        let ctx: &Context = FromCuda::<_, CUerror>::from_cuda(&raw_ctx)?;
        let device = ctx.device;
        STACK.with(move |stack| {
            let mut stack = stack.borrow_mut();
            let last_device = stack.last().map(|(_, dev)| *dev);
            stack.push((raw_ctx, device));
            match last_device {
                None => Some(device),
                Some(last_device) if last_device != device => Some(device),
                _ => None,
            }
        })
    };
    if let Some(dev) = new_device {
        unsafe { hipSetDevice(dev)? };
    }
    Ok(())
}

pub(crate) fn get_current(pctx: &mut CUcontext) -> CUresult {
    match get_current_context() {
        Ok(ctx) => *pctx = ctx,
        Err(_) => {
            // Return primary context instead of NULL to prevent APEX from
            // passing NULL to ContextStorage_Set (which causes NULL deref crash)
            super::driver::zluda_log(
                "[ZLUDA] cuCtxGetCurrent: stack empty, returning primary context",
            );
            super::driver::zluda_log_err("cuCtxGetCurrent: stack empty, returning primary context");
            match super::device::get_primary_context(0) {
                Ok((_, cu_ctx)) => *pctx = cu_ctx,
                Err(_) => *pctx = CUcontext(ptr::null_mut()),
            }
        }
    }
    CUresult::SUCCESS
}

pub(crate) fn get_device(dev: &mut hipDevice_t) -> CUresult {
    match get_device_v2(dev, MaybeContextRef::None) {
        Ok(()) => Ok(()),
        Err(_) => {
            // PhysX GPU dispatcher calls cuCtxGetDevice after destroying contexts.
            // Return device 0 instead of error to keep the dispatcher running.
            super::driver::zluda_log_err("cuCtxGetDevice failed, returning device 0");
            *dev = 0;
            Ok(())
        }
    }
}

pub(crate) fn get_device_v2(dev: &mut hipDevice_t, ctx: MaybeContextRef) -> CUresult {
    let device = match ctx {
        MaybeContextRef::Some(ctx) => ctx.device,
        MaybeContextRef::None => {
            match get_current_context() {
                Ok(cu_ctx) => {
                    let ctx: Result<&Context, CUerror> = FromCuda::from_cuda(&cu_ctx);
                    match ctx {
                        Ok(ctx) => ctx.device,
                        Err(_) => {
                            // Primary context fallback: return device 0
                            super::driver::zluda_log("[ZLUDA] cuCtxGetDevice: from_cuda failed, falling back to device 0");
                            super::driver::zluda_log_err(
                                "cuCtxGetDevice: from_cuda failed, falling back to device 0",
                            );
                            0
                        }
                    }
                }
                Err(_) => {
                    // No context at all: return device 0 as PhysX expects
                    super::driver::zluda_log(
                        "[ZLUDA] cuCtxGetDevice: no context, falling back to device 0",
                    );
                    super::driver::zluda_log_err(
                        "cuCtxGetDevice: no context, falling back to device 0",
                    );
                    0
                }
            }
        }
    };
    *dev = device;
    Ok(())
}

pub(crate) unsafe fn push_current(ctx: CUcontext) -> CUresult {
    if ctx == CUcontext(ptr::null_mut()) {
        return CUresult::ERROR_INVALID_VALUE;
    }
    set_current(ctx)
}

pub(crate) unsafe fn push_current_v2(ctx: CUcontext) -> CUresult {
    push_current(ctx)
}

pub(crate) unsafe fn pop_current(result: Option<&mut CUcontext>) -> CUresult {
    let old_ctx_and_new_device = STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack
            .pop()
            .map(|(ctx, _)| (ctx, stack.last().map(|(_, dev)| *dev)))
    });
    let ctx = match old_ctx_and_new_device {
        Some((old_ctx, new_device)) => {
            if let Some(new_device) = new_device {
                hipSetDevice(new_device)?;
            }
            old_ctx
        }
        None => return CUresult::ERROR_INVALID_CONTEXT,
    };
    if let Some(out) = result {
        *out = ctx;
    }
    Ok(())
}

pub(crate) unsafe fn create_v2(
    ctx: &mut CUcontext,
    _flags: ::core::ffi::c_uint,
    dev: cuda_types::cuda::CUdevice,
) -> CUresult {
    super::driver::zluda_log(&format!("[ZLUDA] cuCtxCreate_v2(dev={})", dev));
    super::driver::zluda_log_err(&format!("cuCtxCreate_v2(dev={})", dev));
    let handle = Context::wrap(Context::new(dev)?);
    // TODO: optimize
    set_current(handle)?;
    *ctx = handle;
    super::driver::zluda_log("[ZLUDA] cuCtxCreate_v2 -> SUCCESS");
    super::driver::zluda_log_err(&format!("cuCtxCreate_v2 -> SUCCESS (ctx={:p})", handle.0));
    Ok(())
}

pub(crate) unsafe fn create_v4(
    ctx: &mut CUcontext,
    _ctx_create_params: Option<&mut cuda_types::cuda::CUctxCreateParams>,
    flags: ::core::ffi::c_uint,
    dev: cuda_types::cuda::CUdevice,
) -> CUresult {
    create_v2(ctx, flags, dev)
}

pub(crate) unsafe fn destroy_v2(_ctx: CUcontext) -> CUresult {
    // Complete no-op: don't pop from stack, don't free memory.
    // Wine's nvcuda relay has a ContextStorage system that crashes (NULL deref)
    // when cuCtxGetCurrent returns NULL (empty stack). PhysX creates contexts
    // via dark API, "destroys" them, then later code (Relay1_func0) calls
    // ContextStorage_Get with whatever cuCtxGetCurrent returns. If we popped
    // the context, Wine gets NULL and crashes. By keeping the context on the
    // stack, Wine always has a valid context pointer.
    // On real NVIDIA, dark API contexts survive cuCtxDestroy anyway.
    super::driver::zluda_log("[ZLUDA] cuCtxDestroy_v2: no-op (context kept on stack)");
    super::driver::zluda_log_err(&format!(
        "cuCtxDestroy_v2(ctx={:p}): no-op (context kept on stack)",
        _ctx.0
    ));
    Ok(())
}

pub(crate) unsafe fn pop_current_v2(ctx: Option<&mut CUcontext>) -> CUresult {
    pop_current(ctx)
}

pub(crate) unsafe fn get_stream_priority_range(
    least_priority: *mut ::core::ffi::c_int,
    greatest_priority: *mut ::core::ffi::c_int,
) -> hipError_t {
    hipDeviceGetStreamPriorityRange(least_priority, greatest_priority)
}

pub(crate) unsafe fn set_flags(flags: ::core::ffi::c_uint) -> CUresult {
    let cu_ctx = get_current_context()?;
    let ctx: &Context = FromCuda::<_, CUerror>::from_cuda(&cu_ctx)?;
    ctx.with_state_mut(|state| {
        state.flags = flags;
        Ok(())
    })
}

/// cuCtxGetSharedMemConfig implementation.
/// cudart64_55.dll calls this during context init. Previously unimplemented,
/// which returned NOT_SUPPORTED, potentially causing the runtime to skip
/// texture object creation (gParams NULL texture bug).
pub(crate) unsafe fn get_shared_memory_config(p_config: *mut CUsharedconfig) -> hipError_t {
    let p_config = p_config as *mut hipSharedMemConfig;
    let result = hipCtxGetSharedMemConfig(p_config);
    super::driver::zluda_log_err(&format!("cuCtxGetSharedMemConfig -> result={:?}", result));
    match result {
        Ok(()) => Ok(()),
        Err(_) => {
            // Return default (4-byte banks) so cudart doesn't skip init
            if !p_config.is_null() {
                *p_config = hipSharedMemConfig(1);
            }
            Ok(())
        }
    }
}

/// cuCtxSetSharedMemConfig — set shared memory bank size.
/// No-op: hipCtxSetSharedMemConfig is unimplemented in ROCm (assertion failure).
/// AMDGPU LDS always uses 32-bit bank mode, so this is safe to ignore.
pub(crate) unsafe fn set_shared_memory_config(config: CUsharedconfig) -> hipError_t {
    super::driver::zluda_log_err(&format!(
        "cuCtxSetSharedMemConfig(config={}) -> no-op",
        config.0
    ));
    Ok(())
}

pub(crate) unsafe fn get_api_version(
    _ctx: CUcontext,
    version: &mut ::core::ffi::c_uint,
) -> CUresult {
    // That's what original CUDA driver does
    *version = 3020;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use std::mem;

    #[test_cuda]
    fn empty_pop_fails(api: impl CudaApi) {
        api.cuInit(0);
        assert_eq!(
            api.cuCtxPopCurrent_v2_unchecked(&mut unsafe { mem::zeroed() }),
            CUresult::ERROR_INVALID_CONTEXT
        );
    }

    #[test_cuda]
    fn pop_into_null_succeeds(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut unsafe { mem::zeroed() }, 0, 0);
        api.cuCtxPopCurrent_v2(ptr::null_mut());
    }

    #[test_cuda]
    fn set_cache_config_accepts_known_values(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut unsafe { mem::zeroed() }, 0, 0);
        for config in [
            CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE,
            CUfunc_cache::CU_FUNC_CACHE_PREFER_SHARED,
            CUfunc_cache::CU_FUNC_CACHE_PREFER_L1,
            CUfunc_cache::CU_FUNC_CACHE_PREFER_EQUAL,
        ] {
            api.cuCtxSetCacheConfig(config);
        }
    }

    #[test_cuda]
    fn set_cache_config_rejects_unknown_value(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut unsafe { mem::zeroed() }, 0, 0);
        assert_eq!(
            api.cuCtxSetCacheConfig_unchecked(CUfunc_cache(4)),
            CUresult::ERROR_INVALID_VALUE
        );
    }

    #[test_cuda]
    fn set_cache_config_requires_current_context(api: impl CudaApi) {
        api.cuInit(0);
        assert_eq!(
            api.cuCtxSetCacheConfig_unchecked(CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE),
            CUresult::ERROR_INVALID_CONTEXT
        );
    }
}

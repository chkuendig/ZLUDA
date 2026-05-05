// D3D11 interop — Linux side.
//
// All Windows-COM and Vulkan-FFI work happens in nvidia-libs/.../nvcuda.c
// (PE-side). nvcuda.c hands us a dma_buf fd plus the descriptor metadata via
// `zluda_d3d11_register_imported`; we hipImportExternalMemory the fd and hand
// back a CUgraphicsResource handle.
//
// This split means:
//   - Rust never calls Windows COM (no extern "win64" gymnastics)
//   - C never calls HIP (no Linux-only headers in nvcuda.c)
//   - The native ABI of each side stays clean
//
// The CUgraphicsResource pointer we return is `Box::into_raw` of an
// `Arc<Mutex<D3D11ResourceState>>`. We track every live resource in
// `D3D11_RESOURCE_TABLE` so the existing graphics.rs Map/Unmap/Unregister/
// GetMappedPointer entry points can dispatch GL vs D3D11 by looking up the
// pointer in the table.

use cuda_types::cuda::*;
use hip_runtime_sys::*;
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::{LazyLock, Mutex};

/// Descriptor handed to us by the PE-side wine_cuGraphicsD3D11RegisterResource
/// after it has set up the dma_buf via Vulkan. Layout MUST match the C struct
/// in nvidia-libs/.../nvcuda.c.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ZludaD3D11ImportedDesc {
    /// dma_buf fd handed out by vkGetMemoryFdKHR. Linux side takes ownership
    /// and is responsible for closing it on unregister. nvcuda.c MUST NOT
    /// close it after the call succeeds.
    pub fd: i32,
    /// In bytes; the size returned by vkGetImageMemoryRequirements / VkBuffer
    /// memory requirements. Used as the size argument to
    /// hipImportExternalMemory.
    pub size: u64,
    pub width: u32,
    pub height: u32,
    /// Row pitch in bytes (rounded up to whatever Vulkan / RADV alignment).
    pub pitch: u32,
    /// DXGI_FORMAT enum value (kept for logging + future format dispatch).
    /// We currently only support DXGI_FORMAT_R32G32B32A32_FLOAT (= 2).
    pub dxgi_format: u32,
    /// CUgraphicsRegisterFlags. Currently only NONE (0) is implemented.
    pub register_flags: u32,
    /// D3D11_BIND_FLAG bitfield (kept for diagnostics).
    pub bind_flags: u32,
    /// Original ID3D11Resource* — used as a stable identity key separately
    /// from the heap-allocated CUgraphicsResource so we can tell when DXVK
    /// recycles a resource without us seeing an unregister.
    pub d3d_resource: *mut c_void,
}

/// Per-resource state held on the Rust side. Pointed to by the
/// CUgraphicsResource handle that we hand back to the application.
#[derive(Debug)]
pub struct D3D11ResourceState {
    /// hipExternalMemory_t we got from hipImportExternalMemory. Owns the
    /// dma_buf fd internally for the lifetime of the import.
    pub ext_mem: hipExternalMemory_t,
    /// Device pointer from hipExternalMemoryGetMappedBuffer. This is what
    /// cuGraphicsResourceGetMappedPointer_v2 returns for D3D11 resources.
    pub dev_ptr: *mut c_void,
    /// Size in bytes — same as desc.size, kept for GetMappedPointer.
    pub size: u64,
    /// Original descriptor — kept for diagnostics + future Map/Unmap.
    pub desc: ZludaD3D11ImportedDesc,
    /// hipArray allocated for cuGraphicsSubResourceGetMappedArray. Batman's
    /// PhysX path uses the array variant (not GetMappedPointer_v2), so we
    /// allocate a real hipArray and copy our linear imported buffer's
    /// contents into it on each Map. ROCm 6.2.4 doesn't export
    /// hipExternalMemoryGetMappedMipmappedArray (verified via nm), so we
    /// can't share the array's memory directly with the import — we copy
    /// linear→tiled inside the GPU on each Map. Allocated lazily on first
    /// SubResource call so non-array consumers don't pay the alloc cost.
    pub hip_array: Mutex<hipArray_t>,
}

// SAFETY: hipExternalMemory_t and *mut c_void are raw HIP-owned pointers; we
// guard cross-thread access with the Mutex in D3D11_RESOURCE_TABLE.
unsafe impl Send for D3D11ResourceState {}
unsafe impl Sync for D3D11ResourceState {}

/// Identifies pointers we own. cuGraphicsMapResources / Unmap / Unregister /
/// GetMappedPointer can look here to dispatch D3D11 vs GL paths.
///
/// The CUgraphicsResource we hand back is `Box::into_raw(Box<D3D11ResourceState>)`,
/// so the pointer itself is the lookup key.
pub static D3D11_RESOURCE_TABLE: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// DXGI_FORMAT_R32G32B32A32_FLOAT — the only format the P0 probe observed Batman
/// register. See `project_d3d11_p0_resource_descriptor.md` in memory for the
/// full data set.
const DXGI_FORMAT_R32G32B32A32_FLOAT: u32 = 2;

/// Returns true if `cuda_res` is a D3D11 resource that we registered (vs a
/// GL hipGraphicsResource_t routed through hipGraphics*Register*).
pub fn is_d3d11_resource(cuda_res: CUgraphicsResource) -> bool {
    if cuda_res.is_null() {
        return false;
    }
    D3D11_RESOURCE_TABLE
        .lock()
        .map(|tbl| tbl.contains(&(cuda_res as usize)))
        .unwrap_or(false)
}

/// Borrow the state for a D3D11 resource. Returns None if not a D3D11
/// resource we registered. Caller MUST treat the result as borrowed for the
/// duration of the call — do NOT free.
pub unsafe fn state_for(cuda_res: CUgraphicsResource) -> Option<&'static D3D11ResourceState> {
    if !is_d3d11_resource(cuda_res) {
        return None;
    }
    Some(&*(cuda_res as *const D3D11ResourceState))
}

/// New ABI symbol exported to nvcuda.c. After nvcuda.c has done the COM
/// + Vulkan dance to allocate a dma_buf-backed VkBuffer, it hands us the fd
/// and we hipImportExternalMemory it.
///
/// `desc.fd` ownership transfers to libcuda.so on success (we will close it
/// in the Unregister path via hipDestroyExternalMemory). On failure we close
/// the fd here so nvcuda.c never has to.
#[no_mangle]
pub unsafe extern "system" fn zluda_d3d11_register_imported(
    p_cuda_resource: *mut CUgraphicsResource,
    desc: *const ZludaD3D11ImportedDesc,
) -> CUresult {
    if p_cuda_resource.is_null() || desc.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let desc = *desc;

    // ------------------------------------------------------------------
    // Validate descriptor against the P0-observed canonical shape.
    // We've only ever seen Tex2D R32G32B32A32_FLOAT mips=1 from Batman; in
    // dev profile we panic on anything else so unsupported workloads fail
    // loudly instead of silently producing garbage. In release we just log
    // and return NOT_SUPPORTED.
    // ------------------------------------------------------------------
    if desc.fd < 0 {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] zluda_d3d11_register_imported: invalid fd={} (caller bug)",
            desc.fd
        ));
        return Err(CUerror::INVALID_VALUE);
    }
    if desc.dxgi_format != DXGI_FORMAT_R32G32B32A32_FLOAT {
        let msg = format!(
            "[ZLUDA] zluda_d3d11_register_imported: unsupported DXGI_FORMAT {} \
             (only R32G32B32A32_FLOAT=2 implemented; see \
             project_d3d11_p0_resource_descriptor.md)",
            desc.dxgi_format
        );
        super::driver::zluda_log_err(&msg);
        debug_assert!(false, "{}", msg);
        libc::close(desc.fd);
        return Err(CUerror::NOT_SUPPORTED);
    }
    if desc.register_flags != 0 {
        let msg = format!(
            "[ZLUDA] zluda_d3d11_register_imported: register_flags={} not implemented \
             (only NONE=0 supported)",
            desc.register_flags
        );
        super::driver::zluda_log_err(&msg);
        debug_assert!(false, "{}", msg);
        libc::close(desc.fd);
        return Err(CUerror::NOT_SUPPORTED);
    }

    // ------------------------------------------------------------------
    // Import the dma_buf fd into HIP. After this, ext_mem owns the fd.
    // ------------------------------------------------------------------
    let mut handle_desc: hipExternalMemoryHandleDesc = std::mem::zeroed();
    handle_desc.type_ = hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueFd;
    handle_desc.handle.fd = desc.fd;
    handle_desc.size = desc.size;
    handle_desc.flags = 0;

    let mut ext_mem: hipExternalMemory_t = std::ptr::null_mut();
    if let Err(e) = hipImportExternalMemory(&mut ext_mem, &handle_desc) {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] hipImportExternalMemory failed: {:?} (fd={}, size={})",
            e, desc.fd, desc.size
        ));
        libc::close(desc.fd);
        return Err(CUerror::UNKNOWN);
    }

    // ------------------------------------------------------------------
    // Pin a buffer-backed device pointer over the imported memory. We use
    // GetMappedBuffer rather than GetMappedMipmappedArray because Batman is
    // observed to call cuGraphicsResourceGetMappedPointer_v2 (linear path)
    // and a LINEAR Vulkan allocation is what we asked for on the PE side.
    // ------------------------------------------------------------------
    let mut buffer_desc: hipExternalMemoryBufferDesc = std::mem::zeroed();
    buffer_desc.offset = 0;
    buffer_desc.size = desc.size;
    buffer_desc.flags = 0;

    let mut dev_ptr: *mut c_void = std::ptr::null_mut();
    if let Err(e) = hipExternalMemoryGetMappedBuffer(&mut dev_ptr, ext_mem, &buffer_desc) {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] hipExternalMemoryGetMappedBuffer failed: {:?}",
            e
        ));
        let _ = hipDestroyExternalMemory(ext_mem); // also closes the fd
        return Err(CUerror::UNKNOWN);
    }

    // ------------------------------------------------------------------
    // Allocate state, register in the lookup table, hand back the pointer.
    // ------------------------------------------------------------------
    let state = Box::new(D3D11ResourceState {
        ext_mem,
        dev_ptr,
        size: desc.size,
        desc,
        hip_array: Mutex::new(std::ptr::null_mut()),
    });
    let cuda_res = Box::into_raw(state) as CUgraphicsResource;

    if let Ok(mut tbl) = D3D11_RESOURCE_TABLE.lock() {
        tbl.insert(cuda_res as usize);
    }
    *p_cuda_resource = cuda_res;

    super::driver::zluda_log_err(&format!(
        "[ZLUDA] zluda_d3d11_register_imported OK ({}x{} fmt={} size={} -> dev_ptr={:p}, cuda_res={:p})",
        desc.width, desc.height, desc.dxgi_format, desc.size, dev_ptr, cuda_res
    ));
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers called by graphics.rs to dispatch GL vs D3D11.
// ---------------------------------------------------------------------------

/// Implementation of cuGraphicsResourceGetMappedPointer_v2 for D3D11 resources.
/// Caller in graphics.rs has already determined the resource IS a D3D11 one
/// via `is_d3d11_resource`.
pub unsafe fn get_mapped_pointer(
    cuda_res: CUgraphicsResource,
    p_dev_ptr: &mut hipDeviceptr_t,
    p_size: &mut usize,
) -> CUresult {
    let state = match state_for(cuda_res) {
        Some(s) => s,
        None => return Err(CUerror::INVALID_VALUE),
    };
    *p_dev_ptr = hipDeviceptr_t(state.dev_ptr);
    *p_size = state.size as usize;
    Ok(())
}

/// Allocate (lazily, once) a hipArray dimensioned to match the D3D11
/// texture descriptor and return its handle. Subsequent calls return the
/// cached handle. Format is fixed at R32G32B32A32_FLOAT (the only format
/// the P0 probe observed). Caller must hold no lock on `state.hip_array`
/// — this function takes its own mutex.
unsafe fn ensure_hip_array(state: &D3D11ResourceState) -> Result<hipArray_t, CUerror> {
    let mut guard = state.hip_array.lock().map_err(|_| CUerror::UNKNOWN)?;
    if !guard.is_null() {
        return Ok(*guard);
    }
    /* R32G32B32A32_FLOAT — 4×32-bit float channels. */
    let mut fmt: hipChannelFormatDesc = std::mem::zeroed();
    fmt.x = 32; fmt.y = 32; fmt.z = 32; fmt.w = 32;
    fmt.f = hipChannelFormatKind::hipChannelFormatKindFloat;
    let mut arr: hipArray_t = std::ptr::null_mut();
    if let Err(e) = hipMallocArray(
        &mut arr,
        &fmt,
        state.desc.width as usize,
        state.desc.height as usize,
        hipArrayDefault,
    ) {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] D3D11 ensure_hip_array hipMallocArray failed: {:?} ({}x{})",
            e, state.desc.width, state.desc.height
        ));
        return Err(CUerror::OUT_OF_MEMORY);
    }
    *guard = arr;
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] D3D11 ensure_hip_array allocated {}x{} R32G32B32A32_FLOAT array={:p}",
        state.desc.width, state.desc.height, arr
    ));
    Ok(arr)
}

/// Refresh the hipArray contents from the imported linear buffer. Called
/// from get_mapped_array (right before the application reads via texture
/// object) so the array contains the current frame's data. Uses 2D copy
/// with explicit pitch.
///
/// Env-gated `ZLUDA_D3D11_ARRAY_REFRESH=0` skips the copy entirely
/// (returns OK without touching the array). Used for an A/B test:
/// hipMemcpy2DToArray on every Map appears to interact badly with
/// DXVK's submit thread on iter46–iter51 (CUB-RadixSort dmesg page
/// faults, ~10/run). Skipping means PhysX gets an empty hipArray —
/// not correct, but lets us verify whether the CUB regression class
/// is caused by this specific copy.
unsafe fn refresh_array_from_buffer(state: &D3D11ResourceState) -> Result<(), CUerror> {
    let arr = ensure_hip_array(state)?;
    if std::env::var("ZLUDA_D3D11_ARRAY_REFRESH")
        .ok()
        .as_deref() == Some("0")
    {
        return Ok(());
    }
    let row_bytes = (state.desc.width as usize).saturating_mul(16); /* R32G32B32A32_FLOAT */
    if let Err(e) = hipMemcpy2DToArray(
        arr,
        0,
        0,
        state.dev_ptr,
        state.desc.pitch as usize,
        row_bytes,
        state.desc.height as usize,
        hipMemcpyKind::hipMemcpyDeviceToDevice,
    ) {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] D3D11 refresh_array_from_buffer hipMemcpy2DToArray failed: {:?}",
            e
        ));
        return Err(CUerror::UNKNOWN);
    }
    Ok(())
}

/// Implementation of cuGraphicsSubResourceGetMappedArray for D3D11 resources.
/// Returns a hipArray whose contents reflect the imported buffer at the time
/// of the call. ROCm 6.2.4 doesn't export hipExternalMemoryGetMappedMipmappedArray
/// so we can't share array memory with our import directly; we copy
/// buffer→array on each call instead. The cost is one GPU-side 2D memcpy
/// (~few hundred µs for the 256x254 textures Batman uses).
///
/// Env `ZLUDA_D3D11_SUBRESOURCE=0` reverts to NOT_SUPPORTED on this path,
/// causing PhysX to fall back to its non-D3D11 buffer allocations. Used for
/// A/B testing whether the SubResource hipArray path is itself a regression
/// against the historical TFS-late ~6% baseline. iter46–iter54 (with
/// SubResource active) showed CUB-RadixSort early crashes with 10+ dmesg
/// page faults; iter17b–iter40 (with SubResource returning NOT_SUPPORTED)
/// showed TFS-late ~6% rate. Toggling via env lets us isolate which
/// behaviour we want to ship.
pub unsafe fn get_mapped_array(
    cuda_res: CUgraphicsResource,
    array_index: u32,
    mip_level: u32,
) -> Result<hipArray_t, CUerror> {
    if std::env::var("ZLUDA_D3D11_SUBRESOURCE")
        .ok()
        .as_deref() == Some("0")
    {
        return Err(CUerror::NOT_SUPPORTED);
    }
    let state = match state_for(cuda_res) {
        Some(s) => s,
        None => return Err(CUerror::INVALID_VALUE),
    };
    if array_index != 0 || mip_level != 0 {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] D3D11 get_mapped_array: array_index={} mip_level={} unsupported (only 0/0)",
            array_index, mip_level
        ));
        return Err(CUerror::NOT_SUPPORTED);
    }
    refresh_array_from_buffer(state)?;
    let arr = state.hip_array.lock().map(|g| *g).unwrap_or(std::ptr::null_mut());
    if arr.is_null() {
        return Err(CUerror::UNKNOWN);
    }
    Ok(arr)
}

/// Implementation of cuGraphicsMapResources for D3D11 resources.
///
/// The actual DXVK→imported-memory blit happens on the PE side in
/// nvcuda.c::wine_cuGraphicsMapResources, BEFORE the call reaches us
/// (CopySurfaceToExternalBuffer through IDXGIVkInteropDevice2). By the
/// time we get here the imported VkBuffer holds the current D3D11 image
/// pixels, so we just need to hipDeviceSynchronize so any outstanding
/// HIP work touching that memory completes before the kernel runs.
pub unsafe fn map_resources(cuda_res: CUgraphicsResource) -> CUresult {
    let state = match state_for(cuda_res) {
        Some(s) => s,
        None => return Err(CUerror::INVALID_VALUE),
    };
    let _ = state; // borrow to avoid 'unused' warning when not logging
    let _ = hipDeviceSynchronize();
    Ok(())
}

/// Implementation of cuGraphicsUnmapResources for D3D11 resources.
/// One-way DXVK→CUDA flow (Batman's resources bind SRV|RTV; PhysX never
/// writes back), so no reverse blit is needed. Just sync to ensure any
/// HIP reads of the imported memory complete before the application
/// resumes Vulkan work on the source image.
pub unsafe fn unmap_resources(cuda_res: CUgraphicsResource) -> CUresult {
    if state_for(cuda_res).is_none() {
        return Err(CUerror::INVALID_VALUE);
    }
    let _ = hipDeviceSynchronize();
    Ok(())
}

/// Implementation of cuGraphicsUnregisterResource for D3D11 resources.
/// Tears down hipExternalMemory (which also closes the dma_buf fd) and frees
/// the boxed state. Removes the pointer from the lookup table FIRST so
/// concurrent callers cannot acquire stale references.
pub unsafe fn unregister(cuda_res: CUgraphicsResource) -> CUresult {
    if cuda_res.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let key = cuda_res as usize;
    let removed = D3D11_RESOURCE_TABLE
        .lock()
        .map(|mut tbl| tbl.remove(&key))
        .unwrap_or(false);
    if !removed {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] cuGraphicsUnregisterResource: D3D11 resource not in table: {:p}",
            cuda_res
        ));
        return Err(CUerror::INVALID_VALUE);
    }
    let state = Box::from_raw(cuda_res as *mut D3D11ResourceState);
    /* Free the hipArray if we allocated one (lazy SubResource path). */
    if let Ok(guard) = state.hip_array.lock() {
        let arr = *guard;
        if !arr.is_null() {
            let _ = hipFreeArray(arr);
        }
    }
    let _ = hipDestroyExternalMemory(state.ext_mem);
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnregisterResource: D3D11 freed (was {:p}, {}x{})",
        cuda_res, state.desc.width, state.desc.height
    ));
    Ok(())
}

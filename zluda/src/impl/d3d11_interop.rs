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
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// HSA bindings — minimal, just what we need for path-(b) interop bypass.
//
// Validated end-to-end by zluda/d3d11_interop_hsa_test crate (Phase
// A.HSA.0–5 on 2026-05-08): hsa_ext_image_create_with_layout on a
// HIP-imported dev_ptr returns a valid gfx10 T# at image.handle, which
// can be cast to hipSurfaceObject_t and used by kernel surf2Dwrite to
// write through to the original buffer.
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type hsa_status_t = u32;
const HSA_STATUS_SUCCESS: hsa_status_t = 0;
const HSA_STATUS_INFO_BREAK: hsa_status_t = 0x4001; // info codes are in 0x4xxx range; using 0x100 worked but be precise

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaAgent {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct HsaExtImage {
    handle: u64,
}

const HSA_EXT_IMAGE_GEOMETRY_2D: u32 = 1;
const HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT: u32 = 15;
const HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA: u32 = 8;
const HSA_ACCESS_PERMISSION_RW: u32 = 3;
const HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR: u32 = 0x1;
const HSA_AGENT_INFO_DEVICE: u32 = 17;
const HSA_DEVICE_TYPE_GPU: u32 = 1;

#[repr(C)]
struct HsaExtImageFormat {
    channel_type: u32,
    channel_order: u32,
}

#[repr(C)]
struct HsaExtImageDescriptor {
    geometry: u32,
    width: usize,
    height: usize,
    depth: usize,
    array_size: usize,
    format: HsaExtImageFormat,
}

#[link(name = "hsa-runtime64")]
extern "C" {
    fn hsa_init() -> hsa_status_t;
    fn hsa_iterate_agents(
        cb: extern "C" fn(HsaAgent, *mut c_void) -> hsa_status_t,
        data: *mut c_void,
    ) -> hsa_status_t;
    fn hsa_agent_get_info(
        agent: HsaAgent,
        attribute: u32,
        value: *mut c_void,
    ) -> hsa_status_t;
    fn hsa_ext_image_create_with_layout(
        agent: HsaAgent,
        descriptor: *const HsaExtImageDescriptor,
        image_data: *const c_void,
        access: u32,
        layout: u32,
        row_pitch: usize,
        slice_pitch: usize,
        image: *mut HsaExtImage,
    ) -> hsa_status_t;
    fn hsa_ext_image_destroy(agent: HsaAgent, image: HsaExtImage) -> hsa_status_t;
}

extern "C" fn hsa_find_gpu_cb(agent: HsaAgent, data: *mut c_void) -> hsa_status_t {
    unsafe {
        let mut device_type: u32 = 0;
        if hsa_agent_get_info(
            agent,
            HSA_AGENT_INFO_DEVICE,
            &mut device_type as *mut _ as *mut c_void,
        ) != HSA_STATUS_SUCCESS
        {
            return HSA_STATUS_SUCCESS;
        }
        if device_type == HSA_DEVICE_TYPE_GPU {
            *(data as *mut HsaAgent) = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        HSA_STATUS_SUCCESS
    }
}

/// Cached GPU agent (initialised on first ensure_hip_array call).
static HSA_GPU_AGENT: LazyLock<Mutex<Option<HsaAgent>>> =
    LazyLock::new(|| Mutex::new(None));

unsafe fn get_gpu_agent() -> Option<HsaAgent> {
    let mut g = HSA_GPU_AGENT.lock().ok()?;
    if let Some(a) = *g {
        return Some(a);
    }
    let _ = hsa_init();
    let mut a = HsaAgent { handle: 0 };
    let _ = hsa_iterate_agents(hsa_find_gpu_cb, &mut a as *mut _ as *mut c_void);
    if a.handle == 0 {
        super::driver::zluda_log_err("[ZLUDA] D3D11 HSA: no GPU agent found");
        return None;
    }
    *g = Some(a);
    Some(a)
}

/// Sentinel struct used as our hipArray_t for D3D11-imported memory.
/// Heap-allocated; its address is what we return from
/// `cuGraphicsSubResourceGetMappedArray`. surf::object_create looks
/// the address up in `D3D11_ARRAY_TABLE` to recognise our synthesis.
///
/// Layout doesn't have to match HIP's hipArray — we never let HIP
/// dereference it. Only our own intercepts touch it.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct D3D11FabricatedArray {
    /// Magic to make accidental bad casts more diagnosable.
    pub magic: u64,
    pub width: u32,
    pub height: u32,
    pub pixel_size: u32, // bytes per pixel
    /// HSA image whose .handle is the gfx10 T# pointing at the
    /// imported buffer's GPU virtual address.
    pub hsa_image_handle: u64,
}

const D3D11_FAB_MAGIC: u64 = 0xD3D11_5AB_0000_0001u64;

/// arr_ptr → image so surf::object_create can synthesize. Lifetime:
/// entries live as long as the cuda_res is registered; cleared in
/// `unregister`.
pub(crate) static D3D11_ARRAY_TABLE: LazyLock<Mutex<HashMap<usize, HsaImageRef>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Stored alongside the sentinel array pointer so unregister can free.
pub(crate) struct HsaImageRef {
    pub agent: HsaAgent,
    pub image: HsaExtImage,
    /// Strong owner of the heap-allocated D3D11FabricatedArray.
    /// Cleared on unregister; the Box is freed here.
    pub _sentinel_owner: Box<D3D11FabricatedArray>,
    /// Backing linear device pointer the sentinel was created over —
    /// `state.dev_ptr` at registration time. Sentinel-aware memcpy
    /// paths route through this so they hand HIP a real linear address
    /// instead of dereferencing the sentinel as `cl_mem`. Snapshot
    /// because state outlives the table entry.
    pub dev_ptr: *mut c_void,
    pub size: u64,
    pub pitch: u32,
}
// Manual Send/Sync: HSA agent + image handles are device handles
// safe to share; the boxed sentinel is also Send (raw u64s + u32s).
unsafe impl Send for HsaImageRef {}
unsafe impl Sync for HsaImageRef {}

/// Look up whether a hipArray_t was synthesized by the D3D11 path.
/// Returns the HSA image handle if so, None otherwise.
/// Returns None if the array is registered but the HSA image
/// creation was skipped (P1 probe — handle is 0) so callers
/// distinguish "not ours" from "ours but no SRD".
pub(crate) fn d3d11_array_to_hsa_image(arr: hipArray_t) -> Option<u64> {
    let key = arr as usize;
    let table = D3D11_ARRAY_TABLE.lock().ok()?;
    let h = table.get(&key)?.image.handle;
    if h == 0 { None } else { Some(h) }
}

/// Look up whether a hipArray_t is one of ours (for descriptor
/// synthesis in tex/surf paths).
pub(crate) fn d3d11_array_descriptor(arr: hipArray_t) -> Option<(u32, u32, u32)> {
    let key = arr as usize;
    let table = D3D11_ARRAY_TABLE.lock().ok()?;
    let entry = table.get(&key)?;
    // We currently only register R32G32B32A32_FLOAT 2D; pixel_size = 16.
    let sentinel = &*entry._sentinel_owner;
    Some((sentinel.width, sentinel.height, sentinel.pixel_size))
}

/// Yes/no probe for callers (memcpy diag stubs) that don't need the
/// descriptor — just whether `arr` is a D3D11 sentinel we returned from
/// `cuGraphicsSubResourceGetMappedArray`. Returns false if the table is
/// poisoned (treat as not-ours; the caller's pass-through path will deal).
pub(crate) fn is_d3d11_sentinel(arr: hipArray_t) -> bool {
    if arr.is_null() {
        return false;
    }
    let key = arr as usize;
    match D3D11_ARRAY_TABLE.lock() {
        Ok(table) => table.contains_key(&key),
        Err(_) => false,
    }
}

/// Look up the dev_ptr backing a D3D11 sentinel array, if any. The
/// sentinel-aware memcpy paths route through this so they hand HIP a
/// real linear device pointer instead of dereferencing the sentinel as
/// `cl_mem`.
pub(crate) fn d3d11_array_dev_ptr(arr: hipArray_t) -> Option<*mut c_void> {
    let key = arr as usize;
    let table = D3D11_ARRAY_TABLE.lock().ok()?;
    let entry = table.get(&key)?;
    Some(entry.dev_ptr)
}

/// Look up the (dev_ptr, size, pitch) backing a D3D11 sentinel array.
/// Used by the dump path and by sentinel-aware memcpy that needs the
/// row pitch to compute byte offsets.
pub(crate) fn d3d11_array_layout(arr: hipArray_t) -> Option<(*mut c_void, u64, u32)> {
    let key = arr as usize;
    let table = D3D11_ARRAY_TABLE.lock().ok()?;
    let entry = table.get(&key)?;
    Some((entry.dev_ptr, entry.size, entry.pitch))
}

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
/// Path-(b) implementation: synthesize a hipArray that aliases the
/// imported D3D11 memory via HSA. Validated end-to-end by the
/// `d3d11_interop_hsa_test` crate (Phase A.HSA.0–5, 2026-05-08):
/// kernel surf2Dwrite via the resulting surface object writes through
/// to the original buffer.
///
/// Returns a sentinel `hipArray_t` whose pointer is registered in
/// `D3D11_ARRAY_TABLE`. ZLUDA's surf::object_create / texref paths
/// look up the sentinel and bypass `hipCreateSurfaceObject` (which
/// SIGSEGVs on non-cl_mem `data` pointers) by returning the HSA
/// image's T# pointer directly.
unsafe fn ensure_hip_array(state: &D3D11ResourceState) -> Result<hipArray_t, CUerror> {
    let mut guard = state.hip_array.lock().map_err(|_| CUerror::UNKNOWN)?;
    if !guard.is_null() {
        return Ok(*guard);
    }

    let agent = match get_gpu_agent() {
        Some(a) => a,
        None => return Err(CUerror::UNKNOWN),
    };

    // P1 probe (2026-05-09): when ZLUDA_D3D11_HSA_SKIP_IMAGE=1, skip
    // the actual hsa_ext_image_create_with_layout call. We still
    // register a sentinel in D3D11_ARRAY_TABLE so descriptor queries
    // and cuMemcpy2DAsync_v2 succeed, but image.handle stays 0 and
    // surf::object_create will refuse to synthesize.
    //
    // This isolates: does just creating the HSA image break IOFX_bboxKernel?
    // Baseline (no HSA): 4014–5694 IOFX_bbox launches before crash.
    // v6 (HSA active):   4 IOFX_bbox launches before crash.
    let skip_image = std::env::var("ZLUDA_D3D11_HSA_SKIP_IMAGE")
        .ok()
        .as_deref()
        == Some("1");

    let mut image: HsaExtImage = HsaExtImage { handle: 0 };
    if !skip_image {
        // R32G32B32A32_FLOAT 2D image bound to the imported dev_ptr.
        let row_pitch = (state.desc.width as usize).saturating_mul(16);
        let descriptor = HsaExtImageDescriptor {
            geometry: HSA_EXT_IMAGE_GEOMETRY_2D,
            width: state.desc.width as usize,
            height: state.desc.height as usize,
            depth: 0,
            array_size: 0,
            format: HsaExtImageFormat {
                channel_type: HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT,
                channel_order: HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,
            },
        };
        let st = hsa_ext_image_create_with_layout(
            agent,
            &descriptor,
            state.dev_ptr as *const c_void,
            HSA_ACCESS_PERMISSION_RW,
            HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR,
            row_pitch,
            0,
            &mut image,
        );
        if st != HSA_STATUS_SUCCESS || image.handle == 0 {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] D3D11 ensure_hip_array hsa_ext_image_create_with_layout failed: 0x{:x} ({}x{})",
                st, state.desc.width, state.desc.height
            ));
            return Err(CUerror::UNKNOWN);
        }
    } else {
        super::driver::zluda_log_err(
            "[ZLUDA] D3D11 ensure_hip_array — HSA_SKIP_IMAGE=1, no HSA image created",
        );
    }

    // Allocate sentinel; its address is the hipArray_t we return.
    let sentinel = Box::new(D3D11FabricatedArray {
        magic: D3D11_FAB_MAGIC,
        width: state.desc.width,
        height: state.desc.height,
        pixel_size: 16, // R32G32B32A32_FLOAT
        hsa_image_handle: image.handle,
    });
    let arr_ptr = (&*sentinel as *const D3D11FabricatedArray) as hipArray_t;

    // Register so surf::object_create can find us.
    if let Ok(mut tbl) = D3D11_ARRAY_TABLE.lock() {
        tbl.insert(
            arr_ptr as usize,
            HsaImageRef {
                agent,
                image,
                _sentinel_owner: sentinel,
                dev_ptr: state.dev_ptr,
                size: state.size,
                pitch: state.desc.pitch,
            },
        );
    } else {
        // Couldn't register; tear down the HSA image.
        let _ = hsa_ext_image_destroy(agent, image);
        return Err(CUerror::UNKNOWN);
    }

    *guard = arr_ptr;
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] D3D11 ensure_hip_array (HSA path) {}x{} R32G32B32A32_FLOAT \
         arr={:p} hsa_image=0x{:x} dev_ptr={:p}",
        state.desc.width, state.desc.height, arr_ptr, image.handle, state.dev_ptr
    ));
    Ok(arr_ptr)
}

/// No-op stub. The HSA-backed hipArray sentinel from `ensure_hip_array`
/// already aliases the imported memory directly — there's nothing to
/// "refresh" because the kernel will read/write to the same physical
/// memory the imported buffer points at.
///
/// Kept as a function so the `get_mapped_array` call site below doesn't
/// need to change semantics.
unsafe fn refresh_array_from_buffer(state: &D3D11ResourceState) -> Result<(), CUerror> {
    let _ = ensure_hip_array(state)?;
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
    // Phase A.1 canary (2026-05-08): read back first 16 u32s; flag if
    // anything diverges from 0xDEADBEEF (a kernel wrote to our array).
    if std::env::var("ZLUDA_D3D11_CANARY").ok().as_deref() == Some("1") {
        let mut buf = [0u32; 16];
        if hipMemcpyAtoH(buf.as_mut_ptr() as *mut std::ffi::c_void, arr, 0, 64).is_ok() {
            let changed = buf.iter().any(|&w| w != 0xDEADBEEFu32);
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] D3D11 canary readback arr={:p} changed={} cuda_res={:p} buf=[{:08x} {:08x} {:08x} {:08x}]",
                arr, changed, cuda_res, buf[0], buf[1], buf[2], buf[3]
            ));
        }
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
    let _ = hipDeviceSynchronize();
    maybe_dump_post_blit_dev_ptr(cuda_res, state);
    Ok(())
}

/// Total dumps written across the run; cap at 8 to bound disk I/O.
static SUBRESOURCE_DUMP_TOTAL: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
/// Per-resource one-shot guard so each cuda_res gets at most one dump.
static SUBRESOURCE_DUMP_SEEN: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

const SUBRESOURCE_DUMP_BYTES: usize = 4096;
const SUBRESOURCE_DUMP_CAP: u32 = 8;

/// After the PE-side DXVK CopySurfaceToExternalBuffer + our
/// hipDeviceSynchronize, the imported `dev_ptr` should hold the
/// post-blit pixel data. Dump the first 4 KB to disk + a 4-pixel
/// preview to the log so we can verify the blit produced sane
/// R32G32B32A32_FLOAT pixels (vs zeros / garbage). Phase B of plan
/// fluttering-waddling-zephyr.md (2026-05-09).
unsafe fn maybe_dump_post_blit_dev_ptr(
    cuda_res: CUgraphicsResource,
    state: &D3D11ResourceState,
) {
    if std::env::var("ZLUDA_SUBRESOURCE_DUMP").ok().as_deref() != Some("1") {
        return;
    }
    let key = cuda_res as usize;
    {
        let mut seen = match SUBRESOURCE_DUMP_SEEN.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if !seen.insert(key) {
            return;
        }
    }
    let total = SUBRESOURCE_DUMP_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if total >= SUBRESOURCE_DUMP_CAP {
        return;
    }

    let bytes = SUBRESOURCE_DUMP_BYTES.min(state.size as usize);
    if bytes == 0 || state.dev_ptr.is_null() {
        return;
    }
    let mut buf = vec![0u8; bytes];
    if hipMemcpyDtoH(
        buf.as_mut_ptr() as *mut std::ffi::c_void,
        hipDeviceptr_t(state.dev_ptr),
        bytes,
    )
    .is_err()
    {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] SUBRESOURCE_DUMP cuda_res={:p} hipMemcpyDtoH failed (dev_ptr={:p} size={})",
            cuda_res, state.dev_ptr, state.size,
        ));
        return;
    }

    // Compute a 4-pixel preview as 4×R32G32B32A32_FLOAT (16 bytes/pixel).
    // Pixels at offsets 0, pitch, 2*pitch, 3*pitch (first column of first
    // 4 rows) — most informative if the blit got width vs pitch wrong.
    let pitch = state.desc.pitch as usize;
    let preview = |row: usize| -> Option<[f32; 4]> {
        let off = row * pitch;
        if off + 16 > bytes {
            return None;
        }
        let r = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let g = f32::from_le_bytes(buf[off + 4..off + 8].try_into().ok()?);
        let b = f32::from_le_bytes(buf[off + 8..off + 12].try_into().ok()?);
        let a = f32::from_le_bytes(buf[off + 12..off + 16].try_into().ok()?);
        Some([r, g, b, a])
    };
    let p0 = preview(0);
    let p1 = preview(1);
    let p2 = preview(2);
    let p3 = preview(3);

    // Detect "all zeros" — the most likely failure mode.
    let all_zero = buf.iter().all(|&b| b == 0);

    let path = format!("/tmp/d3d11_blit_dump_{:x}_{}.bin", key, total);
    match std::fs::File::create(&path) {
        Ok(mut f) => {
            use std::io::Write;
            let _ = f.write_all(&buf);
        }
        Err(e) => {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] SUBRESOURCE_DUMP fopen({}) failed: {}",
                path, e,
            ));
        }
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] SUBRESOURCE_DUMP cuda_res={:p} dev_ptr={:p} bytes={} pitch={} \
         all_zero={} path={} preview_rows=[{:?}, {:?}, {:?}, {:?}]",
        cuda_res, state.dev_ptr, bytes, pitch, all_zero, path, p0, p1, p2, p3,
    ));
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
    /* If we allocated an HSA-backed sentinel array, tear it down via
     * D3D11_ARRAY_TABLE (which owns the HSA image + sentinel struct).
     * Removing from the table drops the HsaImageRef which destroys
     * the image and frees the boxed sentinel. */
    if let Ok(guard) = state.hip_array.lock() {
        let arr = *guard;
        if !arr.is_null() {
            let key = arr as usize;
            if let Ok(mut tbl) = D3D11_ARRAY_TABLE.lock() {
                if let Some(entry) = tbl.remove(&key) {
                    if entry.image.handle != 0 {
                        let _ = hsa_ext_image_destroy(entry.agent, entry.image);
                    }
                    // _sentinel_owner Box drops here, freeing the heap
                    // allocation we used as the hipArray_t.
                }
            }
        }
    }
    let _ = hipDestroyExternalMemory(state.ext_mem);
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnregisterResource: D3D11 freed (was {:p}, {}x{})",
        cuda_res, state.desc.width, state.desc.height
    ));
    Ok(())
}

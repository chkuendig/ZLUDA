use cuda_types::cuda::*;
use hip_runtime_sys::*;
use std::ptr;

/// Internal representation of a CUDA graphics resource.
/// When APEX calls cuGraphicsD3D11RegisterResource, we create one of these
/// with a pre-allocated HIP array and device pointer so that subsequent
/// map/get-mapped-array/get-mapped-pointer calls return valid GPU memory.
#[repr(C)]
pub(crate) struct GraphicsResource {
    /// A small HIP array (64x64 FLOAT4) that cuGraphicsSubResourceGetMappedArray returns.
    pub array: hipArray_t,
    /// A device pointer allocation that cuGraphicsResourceGetMappedPointer returns.
    pub device_ptr: hipDeviceptr_t,
    /// Size of the device_ptr allocation in bytes.
    pub device_ptr_size: usize,
    /// Whether the resource is currently mapped.
    pub mapped: bool,
}

/// Default size for the dummy device pointer allocation (64 KB).
const DUMMY_DEVPTR_SIZE: usize = 64 * 1024;

/// Allocate a new GraphicsResource with a dummy HIP array and device pointer.
unsafe fn create_graphics_resource() -> Result<*mut GraphicsResource, CUerror> {
    // Create a 64x64 FLOAT x 4-channel array
    let desc = HIP_ARRAY_DESCRIPTOR {
        Width: 64,
        Height: 64,
        Format: hipArray_Format::HIP_AD_FORMAT_FLOAT,
        NumChannels: 4,
    };
    let mut hip_array: hipArray_t = ptr::null_mut();
    if let Err(e) = hipArrayCreate(&mut hip_array, &desc) {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] graphics: hipArrayCreate failed: {:?}", e
        ));
        return Err(CUerror::UNKNOWN);
    }

    // Allocate a small device buffer
    let mut dev_ptr: *mut ::core::ffi::c_void = ptr::null_mut();
    if let Err(e) = hipMalloc(&mut dev_ptr, DUMMY_DEVPTR_SIZE) {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] graphics: hipMalloc failed: {:?}", e
        ));
        let _ = hipArrayDestroy(hip_array);
        return Err(CUerror::UNKNOWN);
    }

    // Zero out the device memory so kernels read zeros
    if let Err(e) = hipMemsetD8(hipDeviceptr_t(dev_ptr), 0, DUMMY_DEVPTR_SIZE) {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] graphics: hipMemsetD8 failed: {:?}", e
        ));
        // Non-fatal, continue with uninitialized memory
    }

    let resource = Box::new(GraphicsResource {
        array: hip_array,
        device_ptr: hipDeviceptr_t(dev_ptr),
        device_ptr_size: DUMMY_DEVPTR_SIZE,
        mapped: false,
    });

    let ptr = Box::into_raw(resource);
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] graphics: created resource {:p} (array={:p}, devptr={:p})",
        ptr, hip_array, dev_ptr
    ));
    Ok(ptr)
}

/// Convert a CUgraphicsResource (opaque pointer) to our internal type.
unsafe fn as_resource(resource: CUgraphicsResource) -> Result<&'static mut GraphicsResource, CUerror> {
    if resource.is_null() {
        return Err(CUerror::INVALID_HANDLE);
    }
    Ok(&mut *(resource as *mut GraphicsResource))
}

// -- Stub implementations --
// Function names must match what cuda_normalize_fn! produces from the CUDA API names.
// E.g. cuGraphicsMapResources -> graphics::map_resources

/// cuGraphicsMapResources — mark resources as mapped (no-op for dummy resources).
/// The `resources` param is `*mut CUgraphicsResource` in CUDA, which arrives as
/// `&mut CUgraphicsResource` via `from_cuda_nop!`. We treat it as the first element
/// of an array of `count` resources.
pub(crate) unsafe fn map_resources(
    count: u32,
    resources: &mut CUgraphicsResource,
    _h_stream: hipStream_t,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsMapResources(count={})",
        count
    ));
    if count == 0 {
        return Ok(());
    }
    let resources_ptr = resources as *mut CUgraphicsResource;
    for i in 0..count as usize {
        let res = *resources_ptr.add(i);
        let gr = as_resource(res)?;
        gr.mapped = true;
    }
    Ok(())
}

/// cuGraphicsMapResources_ptsz — same as map_resources.
pub(crate) unsafe fn map_resources_ptsz(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    map_resources(count, resources, h_stream)
}

/// cuGraphicsUnmapResources — mark resources as unmapped.
pub(crate) unsafe fn unmap_resources(
    count: u32,
    resources: &mut CUgraphicsResource,
    _h_stream: hipStream_t,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnmapResources(count={})",
        count
    ));
    if count == 0 {
        return Ok(());
    }
    let resources_ptr = resources as *mut CUgraphicsResource;
    for i in 0..count as usize {
        let res = *resources_ptr.add(i);
        let gr = as_resource(res)?;
        gr.mapped = false;
    }
    Ok(())
}

/// cuGraphicsUnmapResources_ptsz — same as unmap_resources.
pub(crate) unsafe fn unmap_resources_ptsz(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    unmap_resources(count, resources, h_stream)
}

/// cuGraphicsSubResourceGetMappedArray — return the pre-allocated HIP array.
pub(crate) unsafe fn sub_resource_get_mapped_array(
    p_array: &mut CUarray,
    resource: CUgraphicsResource,
    _array_index: u32,
    _mip_level: u32,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsSubResourceGetMappedArray(resource={:p})",
        resource
    ));
    let gr = as_resource(resource)?;
    if !gr.mapped {
        return Err(CUerror::NOT_MAPPED);
    }
    // Return the pre-allocated array regardless of arrayIndex/mipLevel
    *p_array = gr.array as CUarray;
    Ok(())
}

/// cuGraphicsResourceGetMappedPointer_v2 — return the pre-allocated device pointer.
pub(crate) unsafe fn resource_get_mapped_pointer_v2(
    p_dev_ptr: &mut hipDeviceptr_t,
    p_size: &mut usize,
    resource: CUgraphicsResource,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsResourceGetMappedPointer_v2(resource={:p})",
        resource
    ));
    let gr = as_resource(resource)?;
    if !gr.mapped {
        return Err(CUerror::NOT_MAPPED);
    }
    *p_dev_ptr = gr.device_ptr;
    *p_size = gr.device_ptr_size;
    Ok(())
}

/// cuGraphicsResourceSetMapFlags_v2 — no-op stub.
pub(crate) unsafe fn resource_set_map_flags_v2(
    resource: CUgraphicsResource,
    _flags: u32,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsResourceSetMapFlags_v2(resource={:p})",
        resource
    ));
    if resource.is_null() {
        return Err(CUerror::INVALID_HANDLE);
    }
    Ok(())
}

/// cuGraphicsResourceSetMapFlags (v1) — no-op stub.
pub(crate) unsafe fn resource_set_map_flags(
    resource: CUgraphicsResource,
    flags: u32,
) -> CUresult {
    resource_set_map_flags_v2(resource, flags)
}

/// cuGraphicsUnregisterResource — free the dummy resource.
pub(crate) unsafe fn unregister_resource(
    resource: CUgraphicsResource,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnregisterResource(resource={:p})",
        resource
    ));
    let gr = as_resource(resource)?;

    // Free the HIP array
    if !gr.array.is_null() {
        if let Err(e) = hipArrayDestroy(gr.array) {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] graphics: hipArrayDestroy failed: {:?}", e
            ));
        }
    }

    // Free the device pointer
    if !gr.device_ptr.0.is_null() {
        if let Err(e) = hipFree(gr.device_ptr.0) {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] graphics: hipFree failed: {:?}", e
            ));
        }
    }

    // Free the Box
    let _ = Box::from_raw(resource as *mut GraphicsResource);
    Ok(())
}

/// cuGraphicsResourceGetMappedMipmappedArray — not needed by PhysX, returns NOT_SUPPORTED.
pub(crate) unsafe fn resource_get_mapped_mipmapped_array(
    _p_mipmapped_array: &mut CUmipmappedArray,
    _resource: CUgraphicsResource,
) -> CUresult {
    super::driver::zluda_log_err(
        "[ZLUDA] cuGraphicsResourceGetMappedMipmappedArray -> NOT_SUPPORTED",
    );
    Err(CUerror::NOT_SUPPORTED)
}

// -- D3D11 interop --
// These functions are NOT in the standard cuda.rs function table, so we export them
// manually with #[no_mangle].

/// cuGraphicsD3D11RegisterResource — creates a dummy graphics resource.
/// Called by APEX via ApexFramework to register D3D11 textures/buffers for CUDA interop.
/// We ignore the D3D resource pointer and create a dummy resource with pre-allocated GPU memory.
#[no_mangle]
pub unsafe extern "system" fn cuGraphicsD3D11RegisterResource(
    p_cuda_resource: *mut CUgraphicsResource,
    _d3d_resource: *mut ::core::ffi::c_void,
    flags: u32,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsD3D11RegisterResource(flags={})",
        flags
    ));
    if p_cuda_resource.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    match create_graphics_resource() {
        Ok(resource) => {
            *p_cuda_resource = resource as CUgraphicsResource;
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] cuGraphicsD3D11RegisterResource -> OK (resource={:p})",
                resource
            ));
            Ok(())
        }
        Err(e) => {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] cuGraphicsD3D11RegisterResource -> {:?}",
                e
            ));
            Err(e)
        }
    }
}

/// cuD3D11GetDevice — stub that returns device 0.
/// Called by Wine's nvcuda relay to check for D3D11-capable CUDA devices.
#[no_mangle]
pub unsafe extern "system" fn cuD3D11GetDevice(
    device: *mut CUdevice,
    _p_adapter: *mut ::core::ffi::c_void,
) -> CUresult {
    super::driver::zluda_log_err("[ZLUDA] cuD3D11GetDevice -> device 0");
    if device.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    *device = 0;
    Ok(())
}

use cuda_types::cuda::*;
use hip_runtime_sys::*;
use std::ptr;
use std::sync::Once;

/// Convert hipError_t to CUresult
fn hip_to_cuda(result: hipError_t) -> CUresult {
    match result {
        Ok(()) => Ok(()),
        Err(_) => Err(CUerror::UNKNOWN),
    }
}

/// One-time initialization of HIP-GL interop.
/// lshqqytiger pattern: call hipGLGetDevices with null args to trigger init.
static GL_INIT: Once = Once::new();
fn init_opengl() {
    GL_INIT.call_once(|| {
        unsafe { hipGLGetDevices(ptr::null_mut(), ptr::null_mut(), 0, 0) };
    });
}

// -- GL interop --
// These are #[no_mangle] because they're not in the standard CUDA function table;
// applications load them via cuGetProcAddress or dlsym.

/// cuGraphicsGLRegisterBuffer — register an OpenGL buffer for CUDA access.
pub(crate) unsafe fn g_l_register_buffer(
    p_cuda_resource: &mut CUgraphicsResource,
    buffer: cuda_types::cuda::GLuint,
    flags: ::core::ffi::c_uint,
) -> CUresult {
    init_opengl();
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsGLRegisterBuffer(buffer={}, flags={})",
        buffer, flags
    ));
    let result = hipGraphicsGLRegisterBuffer(
        p_cuda_resource as *mut CUgraphicsResource as *mut hipGraphicsResource_t,
        buffer,
        flags,
    );
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsGLRegisterBuffer -> {:?}",
        result
    ));
    hip_to_cuda(result)
}

/// cuGraphicsGLRegisterImage — register an OpenGL image/texture for CUDA access.
pub(crate) unsafe fn g_l_register_image(
    p_cuda_resource: &mut CUgraphicsResource,
    image: cuda_types::cuda::GLuint,
    target: cuda_types::cuda::GLenum,
    flags: ::core::ffi::c_uint,
) -> CUresult {
    init_opengl();
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsGLRegisterImage(image={}, target={}, flags={})",
        image, target, flags
    ));
    let result = hipGraphicsGLRegisterImage(
        p_cuda_resource as *mut CUgraphicsResource as *mut hipGraphicsResource_t,
        image,
        target,
        flags,
    );
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsGLRegisterImage -> {:?}",
        result
    ));
    hip_to_cuda(result)
}

// -- Generic graphics resource operations --
// These replace the dummy stubs with real HIP calls.

/// cuGraphicsMapResources
pub(crate) unsafe fn map_resources(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    super::driver::zluda_log_err(&format!("[ZLUDA] cuGraphicsMapResources(count={})", count));
    hip_to_cuda(hipGraphicsMapResources(
        count as i32,
        resources as *mut CUgraphicsResource as *mut hipGraphicsResource_t,
        h_stream,
    ))
}

/// cuGraphicsMapResources_ptsz
pub(crate) unsafe fn map_resources_ptsz(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    map_resources(count, resources, h_stream)
}

/// cuGraphicsUnmapResources
pub(crate) unsafe fn unmap_resources(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnmapResources(count={})",
        count
    ));
    hip_to_cuda(hipGraphicsUnmapResources(
        count as i32,
        resources as *mut CUgraphicsResource as *mut hipGraphicsResource_t,
        h_stream,
    ))
}

/// cuGraphicsUnmapResources_ptsz
pub(crate) unsafe fn unmap_resources_ptsz(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    unmap_resources(count, resources, h_stream)
}

/// cuGraphicsSubResourceGetMappedArray
pub(crate) unsafe fn sub_resource_get_mapped_array(
    p_array: &mut CUarray,
    resource: CUgraphicsResource,
    array_index: u32,
    mip_level: u32,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsSubResourceGetMappedArray(resource={:p})",
        resource
    ));
    hip_to_cuda(hipGraphicsSubResourceGetMappedArray(
        p_array as *mut CUarray as *mut hipArray_t,
        resource as hipGraphicsResource_t,
        array_index,
        mip_level,
    ))
}

/// cuGraphicsResourceGetMappedPointer_v2
pub(crate) unsafe fn resource_get_mapped_pointer_v2(
    p_dev_ptr: &mut hipDeviceptr_t,
    p_size: &mut usize,
    resource: CUgraphicsResource,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsResourceGetMappedPointer_v2(resource={:p})",
        resource
    ));
    hip_to_cuda(hipGraphicsResourceGetMappedPointer(
        p_dev_ptr as *mut hipDeviceptr_t as *mut *mut ::core::ffi::c_void,
        p_size,
        resource as hipGraphicsResource_t,
    ))
}

/// cuGraphicsResourceSetMapFlags_v2 — no-op (HIP doesn't have equivalent).
pub(crate) unsafe fn resource_set_map_flags_v2(
    _resource: CUgraphicsResource,
    _flags: u32,
) -> CUresult {
    Ok(())
}

/// cuGraphicsResourceSetMapFlags (v1)
pub(crate) unsafe fn resource_set_map_flags(resource: CUgraphicsResource, flags: u32) -> CUresult {
    resource_set_map_flags_v2(resource, flags)
}

/// cuGraphicsUnregisterResource
pub(crate) unsafe fn unregister_resource(resource: CUgraphicsResource) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnregisterResource(resource={:p})",
        resource
    ));
    hip_to_cuda(hipGraphicsUnregisterResource(
        resource as hipGraphicsResource_t,
    ))
}

/// cuGraphicsResourceGetMappedMipmappedArray — not implemented.
pub(crate) unsafe fn resource_get_mapped_mipmapped_array(
    _p_mipmapped_array: &mut CUmipmappedArray,
    _resource: CUgraphicsResource,
) -> CUresult {
    Err(CUerror::NOT_SUPPORTED)
}

// -- D3D11 interop --
// Under Wine/Proton, D3D11 resources are DXVK Vulkan resources.
// True interop requires Vulkan → HIP memory sharing which is complex.
// For now, create a dummy resource so PhysX doesn't crash at registration time.
// The actual data sharing won't work — this is a known limitation.

/// Internal dummy D3D11 resource.
#[repr(C)]
struct D3D11GraphicsResource {
    hip_resource: hipGraphicsResource_t,
    device_ptr: hipDeviceptr_t,
    device_ptr_size: usize,
}

const D3D11_DUMMY_SIZE: usize = 256 * 1024; // 256 KB

#[no_mangle]
pub unsafe extern "system" fn cuGraphicsD3D11RegisterResource(
    p_cuda_resource: *mut CUgraphicsResource,
    _d3d_resource: *mut ::core::ffi::c_void,
    _flags: u32,
) -> CUresult {
    super::driver::zluda_log_err("[ZLUDA] cuGraphicsD3D11RegisterResource (dummy stub)");
    if p_cuda_resource.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    // Allocate a dummy device buffer
    let mut dev_ptr: *mut ::core::ffi::c_void = ptr::null_mut();
    if let Err(e) = hipMalloc(&mut dev_ptr, D3D11_DUMMY_SIZE) {
        super::driver::zluda_log_err(&format!("[ZLUDA] D3D11 interop: hipMalloc failed: {:?}", e));
        return Err(CUerror::UNKNOWN);
    }
    let _ = hipMemsetD8(hipDeviceptr_t(dev_ptr), 0, D3D11_DUMMY_SIZE);
    let resource = Box::new(D3D11GraphicsResource {
        hip_resource: ptr::null_mut(),
        device_ptr: hipDeviceptr_t(dev_ptr),
        device_ptr_size: D3D11_DUMMY_SIZE,
    });
    *p_cuda_resource = Box::into_raw(resource) as CUgraphicsResource;
    Ok(())
}

#[no_mangle]
pub unsafe extern "system" fn cuD3D11GetDevice(
    device: *mut CUdevice,
    _p_adapter: *mut ::core::ffi::c_void,
) -> CUresult {
    if device.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    *device = 0;
    Ok(())
}

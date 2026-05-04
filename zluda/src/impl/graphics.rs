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
///
/// Dispatches by resource type:
/// - GL resources go through hipGraphicsMapResources (HIP's native impl)
/// - D3D11 resources (registered via zluda_d3d11_register_imported) go
///   through d3d11_interop::map_resources, which currently just synchronises;
///   a real DXVK→staging blit is still TODO (see plan step 5d).
///
/// We must dispatch per-element because a single Map call can mix types.
pub(crate) unsafe fn map_resources(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    super::driver::zluda_log_err(&format!("[ZLUDA] cuGraphicsMapResources(count={})", count));
    let resource_slice = std::slice::from_raw_parts_mut(
        resources as *mut CUgraphicsResource,
        count as usize,
    );
    let mut gl_resources: Vec<CUgraphicsResource> = Vec::with_capacity(count as usize);
    for &res in resource_slice.iter() {
        if super::d3d11_interop::is_d3d11_resource(res) {
            super::d3d11_interop::map_resources(res)?;
        } else {
            gl_resources.push(res);
        }
    }
    if !gl_resources.is_empty() {
        hip_to_cuda(hipGraphicsMapResources(
            gl_resources.len() as i32,
            gl_resources.as_mut_ptr() as *mut hipGraphicsResource_t,
            h_stream,
        ))?;
    }
    Ok(())
}

/// cuGraphicsMapResources_ptsz
pub(crate) unsafe fn map_resources_ptsz(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    map_resources(count, resources, h_stream)
}

/// cuGraphicsUnmapResources — symmetric to map_resources, dispatch on type.
pub(crate) unsafe fn unmap_resources(
    count: u32,
    resources: &mut CUgraphicsResource,
    h_stream: hipStream_t,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnmapResources(count={})",
        count
    ));
    let resource_slice = std::slice::from_raw_parts_mut(
        resources as *mut CUgraphicsResource,
        count as usize,
    );
    let mut gl_resources: Vec<CUgraphicsResource> = Vec::with_capacity(count as usize);
    for &res in resource_slice.iter() {
        if super::d3d11_interop::is_d3d11_resource(res) {
            super::d3d11_interop::unmap_resources(res)?;
        } else {
            gl_resources.push(res);
        }
    }
    if !gl_resources.is_empty() {
        hip_to_cuda(hipGraphicsUnmapResources(
            gl_resources.len() as i32,
            gl_resources.as_mut_ptr() as *mut hipGraphicsResource_t,
            h_stream,
        ))?;
    }
    Ok(())
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
///
/// D3D11 resources route to this function when an app prefers an array-style
/// hipArray_t over a raw device pointer. We don't yet have a D3D11
/// implementation — Batman is observed to use `cuGraphicsResourceGetMappedPointer_v2`
/// (the linear path), so this is unimplemented. Any other workload that hits
/// this path will land here with a D3D11 resource and we'd otherwise pass the
/// pointer to HIP as a `hipGraphicsResource_t`, which would dereference our
/// `Box<D3D11ResourceState>` as if it were a HIP-internal struct → crash.
/// Panic loudly in dev builds; return NOT_SUPPORTED otherwise.
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
    if super::d3d11_interop::is_d3d11_resource(resource) {
        let msg = format!(
            "[ZLUDA] cuGraphicsSubResourceGetMappedArray: D3D11 path not implemented \
             (resource={:p}, array_index={}, mip_level={}). Batman uses \
             GetMappedPointer_v2; if you hit this with another workload, the array \
             path needs hipExternalMemoryGetMappedMipmappedArray wiring.",
            resource, array_index, mip_level
        );
        super::driver::zluda_log_err(&msg);
        debug_assert!(false, "{}", msg);
        return Err(CUerror::NOT_SUPPORTED);
    }
    hip_to_cuda(hipGraphicsSubResourceGetMappedArray(
        p_array as *mut CUarray as *mut hipArray_t,
        resource as hipGraphicsResource_t,
        array_index,
        mip_level,
    ))
}

/// cuGraphicsResourceGetMappedPointer_v2 — dispatches GL vs D3D11.
pub(crate) unsafe fn resource_get_mapped_pointer_v2(
    p_dev_ptr: &mut hipDeviceptr_t,
    p_size: &mut usize,
    resource: CUgraphicsResource,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsResourceGetMappedPointer_v2(resource={:p})",
        resource
    ));
    if super::d3d11_interop::is_d3d11_resource(resource) {
        return super::d3d11_interop::get_mapped_pointer(resource, p_dev_ptr, p_size);
    }
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

/// cuGraphicsUnregisterResource — dispatches GL vs D3D11.
pub(crate) unsafe fn unregister_resource(resource: CUgraphicsResource) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsUnregisterResource(resource={:p})",
        resource
    ));
    if super::d3d11_interop::is_d3d11_resource(resource) {
        return super::d3d11_interop::unregister(resource);
    }
    hip_to_cuda(hipGraphicsUnregisterResource(
        resource as hipGraphicsResource_t,
    ))
}

/// cuGraphicsResourceGetMappedMipmappedArray — not implemented for any path.
/// Batman registers Tex2D mips=1 so this is never reached. Panic in dev if a
/// D3D11 resource hits this so we know to extend the impl when a new workload
/// surfaces.
pub(crate) unsafe fn resource_get_mapped_mipmapped_array(
    _p_mipmapped_array: &mut CUmipmappedArray,
    resource: CUgraphicsResource,
) -> CUresult {
    if super::d3d11_interop::is_d3d11_resource(resource) {
        let msg = format!(
            "[ZLUDA] cuGraphicsResourceGetMappedMipmappedArray: D3D11 path not \
             implemented (resource={:p}). P0 only ever observed mips=1 from \
             Batman; if you hit this another workload registered a mipmapped \
             D3D11 texture and we need a hipExternalMemoryGetMappedMipmappedArray \
             implementation.",
            resource
        );
        super::driver::zluda_log_err(&msg);
        debug_assert!(false, "{}", msg);
    }
    Err(CUerror::NOT_SUPPORTED)
}

// -- D3D11 interop --
// Under Wine/Proton, D3D11 resources are DXVK Vulkan resources. True interop
// requires Vulkan→HIP external-memory import (VK_KHR_external_memory_fd +
// hipImportExternalMemory), which is not yet implemented.
//
// Until the real implementation lands we return NOT_SUPPORTED. The previous
// version of this function silently allocated a 256 KB hipMalloc buffer and
// returned it as if registration had succeeded — that masked the real error
// and let the application read garbage. Returning NOT_SUPPORTED here lets the
// PE-side wrapper (nvidia-libs/.../nvcuda.c::wine_cuGraphicsD3D11RegisterResource)
// log P0 introspection of the resource type before this point so we can scope
// the eventual implementation correctly.

#[no_mangle]
pub unsafe extern "system" fn cuGraphicsD3D11RegisterResource(
    p_cuda_resource: *mut CUgraphicsResource,
    d3d_resource: *mut ::core::ffi::c_void,
    flags: u32,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] cuGraphicsD3D11RegisterResource(d3d={:p}, flags={}) — not yet implemented (returning NOT_SUPPORTED)",
        d3d_resource, flags
    ));
    if p_cuda_resource.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    Err(CUerror::NOT_SUPPORTED)
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

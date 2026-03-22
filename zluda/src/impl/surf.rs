use cuda_types::cuda::*;
use hip_runtime_sys::*;
use std::mem;

// Same as in zluda_ptx_impl.cpp
const IMAGE_RESERVED_TOP_BITS: u32 = 3;

pub(crate) unsafe fn object_create(
    p_surf_object: &mut CUsurfObject,
    p_res_desc: &CUDA_RESOURCE_DESC,
) -> CUresult {
    let desc = to_surface_desc(p_res_desc)?;
    if desc.resType != hipResourceType::hipResourceTypeArray {
        return Err(CUerror::NOT_SUPPORTED);
    }
    let mut surf_obj: hipSurfaceObject_t = mem::zeroed();
    hipCreateSurfaceObject(&mut surf_obj, &desc)?;
    let top_reserved_bits = surf_obj as usize >> (usize::BITS - IMAGE_RESERVED_TOP_BITS);
    if top_reserved_bits != 0 {
        super::driver::zluda_log(&format!(
            "[ZLUDA] cuSurfObjectCreate: top bits non-zero ({:#x}), rejecting",
            surf_obj as u64
        ));
        let _ = hipDestroySurfaceObject(surf_obj);
        return Err(CUerror::UNKNOWN);
    }
    // Get array descriptor to determine pixel size for encoding
    let array = desc.res.array.array;
    let mut array_desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
    hipArrayGetDescriptor(&mut array_desc, array)?;
    let format_size = format_size(array_desc.Format)?;
    let pixel_size = format_size * array_desc.NumChannels as usize;
    let shift_amount =
        (pixel_size.trailing_zeros() as usize) << (usize::BITS - IMAGE_RESERVED_TOP_BITS);
    surf_obj = (surf_obj as usize | shift_amount) as _;
    // Read back the 256-bit image descriptor to log its base address.
    // The descriptor is 8 x uint32 at the address pointed to by surf_obj (minus top bits).
    let desc_ptr = (surf_obj as usize & !((7usize) << (usize::BITS - IMAGE_RESERVED_TOP_BITS))) as *const u8;
    let mut desc_words = [0u32; 8];
    let copy_result = hipMemcpyDtoH(
        desc_words.as_mut_ptr() as *mut std::ffi::c_void,
        hipDeviceptr_t(desc_ptr as *mut std::ffi::c_void),
        32,
    );
    let base_addr = if copy_result.is_ok() {
        // Base address is in words 0-1 (48-bit address, bits [47:0] of word0 + word1)
        let base = (desc_words[0] as u64) | ((desc_words[1] as u64) << 32);
        format!("base={:#x} words=[{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x}]",
            base, desc_words[0], desc_words[1], desc_words[2], desc_words[3],
            desc_words[4], desc_words[5], desc_words[6], desc_words[7])
    } else {
        format!("descriptor read FAILED: {:?}", copy_result)
    };
    super::driver::zluda_log(&format!(
        "[ZLUDA] cuSurfObjectCreate: hip_obj={:#x} format={:?} channels={} pixel_size={} shift={} encoded={:#x} {}",
        (surf_obj as usize & !((7usize) << (usize::BITS - IMAGE_RESERVED_TOP_BITS))),
        array_desc.Format, array_desc.NumChannels, pixel_size,
        pixel_size.trailing_zeros(), surf_obj as u64, base_addr
    ));
    *p_surf_object = surf_obj as CUsurfObject;
    Ok(())
}


pub(crate) unsafe fn object_destroy(
    surf_object: CUsurfObject,
) -> CUresult {
    let surf_obj = surf_object as hipSurfaceObject_t;
    hipDestroySurfaceObject(
        (((surf_obj as usize) << IMAGE_RESERVED_TOP_BITS) >> IMAGE_RESERVED_TOP_BITS) as _,
    )?;
    Ok(())
}

pub(crate) unsafe fn ref_set_array(
    h_surf_ref: CUsurfref,
    h_array: CUarray,
    _flags: u32,
) -> CUresult {
    let surf_ref = h_surf_ref as *mut textureReference;
    if h_array.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let mut desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
    hipArrayGetDescriptor(&mut desc, h_array as hipArray_t)?;
    super::driver::zluda_log(&format!(
        "[ZLUDA] cuSurfRefSetArray: ref={:p} array={:p} format={:?} channels={}",
        surf_ref, h_array, desc.Format, desc.NumChannels
    ));
    hipTexRefSetFormat(surf_ref, desc.Format, desc.NumChannels as i32)?;
    hipTexRefSetArray(surf_ref, h_array as hipArray_t, CU_TRSA_OVERRIDE_FORMAT)?;
    crate::r#impl::tex::sync_synthetic_texref(surf_ref);
    Ok(())
}

fn format_size(f: hipArray_Format) -> Result<usize, CUerror> {
    Ok(match f {
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT8
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT8 => 1,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_HALF => 2,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_FLOAT => 4,
        _ => return Err(CUerror::NOT_SUPPORTED),
    })
}

unsafe fn to_surface_desc(res_desc: &CUDA_RESOURCE_DESC) -> Result<hipResourceDesc, CUerror> {
    let res_type = mem::transmute(res_desc.resType);
    let res: hipResourceDesc__bindgen_ty_1 = match res_desc.resType {
        CUresourcetype::CU_RESOURCE_TYPE_ARRAY => hipResourceDesc__bindgen_ty_1 {
            array: hipResourceDesc__bindgen_ty_1__bindgen_ty_1 {
                array: res_desc.res.array.hArray as hipArray_t,
            },
        },
        _ => return Err(CUerror::NOT_SUPPORTED),
    };
    Ok(hipResourceDesc {
        resType: res_type,
        res,
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::*;
    use std::mem;

    #[test_cuda]
    unsafe fn surf_object_create_destroy(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        // Create a 3D array with surface load/store capability
        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 32,
            Height: 32,
            Depth: 1,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 4,
            Flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &array_desc);
        assert!(!array.is_null());

        // Create surface object from the array
        let mut res_desc: CUDA_RESOURCE_DESC = mem::zeroed();
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;

        let mut surf_object: CUsurfObject = 0;
        api.cuSurfObjectCreate(&mut surf_object, &res_desc);

        api.cuSurfObjectDestroy(surf_object);
        api.cuArrayDestroy(array);
    }
}

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
    // Track surface object address in global alloc range
    super::memory::update_alloc_range(surf_obj as u64, 256);
    let top_reserved_bits = surf_obj as usize >> (usize::BITS - IMAGE_RESERVED_TOP_BITS);
    if top_reserved_bits != 0 {
        super::driver::zluda_log(&format!(
            "[ZLUDA] cuSurfObjectCreate: top bits non-zero ({:#x}), rejecting",
            surf_obj as u64
        ));
        let _ = hipDestroySurfaceObject(surf_obj);
        return Err(CUerror::UNKNOWN);
    }
    let array = desc.res.array.array;
    let mut array_desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
    hipArrayGetDescriptor(&mut array_desc, array)?;
    let fmt_size = format_size(array_desc.Format)?;
    let pixel_size = fmt_size * array_desc.NumChannels as usize;
    // Guard against pixel_size=0 (e.g., if NumChannels=0 for 3D arrays)
    let shift_bits = if pixel_size == 0 {
        super::driver::zluda_log_err(&format!(
            "WARN cuSurfObjectCreate: pixel_size=0! fmt={:?} ch={} array={:?} — defaulting shift to 0",
            array_desc.Format, array_desc.NumChannels, array));
        0usize
    } else {
        pixel_size.trailing_zeros() as usize
    };
    super::driver::zluda_log_err(&format!(
        "cuSurfObjectCreate: hip_obj=0x{:x} fmt={:?} ch={} pixel_size={} shift={} array={:?}",
        surf_obj as u64, array_desc.Format, array_desc.NumChannels, pixel_size, shift_bits, array));
    let shift_amount = shift_bits << (usize::BITS - IMAGE_RESERVED_TOP_BITS);
    surf_obj = (surf_obj as usize | shift_amount) as _;
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
    // For surface references, we need to encode the pixel shift in the top 3 bits
    // of the surface handle before writing it to the device global.
    // The PTX sust helper functions extract shift from top bits to compute
    // x-coordinate byte offsets. Without this, shift=0 → wrong offsets → GPU fault.
    let fmt_size = format_size(desc.Format).unwrap_or(4);
    let num_ch = if desc.NumChannels == 0 { 1 } else { desc.NumChannels as usize };
    let pixel_size = fmt_size * num_ch;
    let shift_bits = if pixel_size == 0 { 0usize } else { pixel_size.trailing_zeros() as usize };
    let tex_obj = (*surf_ref).textureObject;
    if !tex_obj.is_null() {
        let encoded = (tex_obj as usize) | (shift_bits << (usize::BITS - IMAGE_RESERVED_TOP_BITS));
        super::driver::zluda_log(&format!(
            "[ZLUDA] cuSurfRefSetArray: encoding shift={} into handle: {:#x} → {:#x}",
            shift_bits, tex_obj as u64, encoded as u64
        ));
        // Write the shift-encoded handle to the textureObject field
        (*surf_ref).textureObject = encoded as hipTextureObject_t;
    }
    crate::r#impl::tex::sync_synthetic_texref(surf_ref);
    Ok(())
}

pub(crate) fn format_size(f: hipArray_Format) -> Result<usize, CUerror> {
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

    /// Test that hipArrayGetDescriptor returns correct NumChannels for 3D arrays.
    /// Bug: hipArrayGetDescriptor returns NumChannels=0 for 3D arrays, causing
    /// pixel_size=0 → trailing_zeros overflow → shift=0 in surface handle encoding.
    #[test_cuda]
    unsafe fn array_3d_get_descriptor_channels(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 32,
            Height: 32,
            Depth: 32,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 1,
            Flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &array_desc);
        assert!(!array.is_null());

        // 2D descriptor query on a 3D array — check if NumChannels is correct
        let mut desc2d: CUDA_ARRAY_DESCRIPTOR = mem::zeroed();
        api.cuArrayGetDescriptor_v2(&mut desc2d, array);
        assert_ne!(desc2d.NumChannels, 0,
            "hipArrayGetDescriptor returns NumChannels=0 for 3D arrays — \
             this causes pixel_size=0 → shift overflow → corrupt surface handles");

        api.cuArrayDestroy(array);
    }

    /// Test that surface objects on 3D FLOAT arrays get the correct pixel shift encoding.
    /// This catches the bug where hipArrayGetDescriptor returns NumChannels=0 for 3D arrays,
    /// causing pixel_size=0 → shift=0 instead of shift=2 for 4-byte pixels.
    #[test_cuda]
    unsafe fn surf_3d_float_shift_encoding(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        // Create a 3D FLOAT array (same as TurbulenceFS uses)
        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 64,
            Height: 64,
            Depth: 64,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 1,
            Flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &array_desc);
        assert!(!array.is_null());

        // Create surface object
        let mut res_desc: CUDA_RESOURCE_DESC = mem::zeroed();
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;

        let mut surf_object: CUsurfObject = 0;
        api.cuSurfObjectCreate(&mut surf_object, &res_desc);
        assert_ne!(surf_object, 0, "surface object creation failed");

        // Check the shift encoding: top 3 bits should encode pixel shift
        // For FLOAT (4 bytes), 1 channel: pixel_size=4, trailing_zeros(4)=2
        // shift_amount = 2 << 61 = 0x4000000000000000
        let shift = (surf_object as u64) >> 61;
        assert_eq!(shift, 2,
            "FLOAT surface shift should be 2 (log2(4)), got {} — \
             hipArrayGetDescriptor may be returning wrong format for 3D arrays",
            shift);

        api.cuSurfObjectDestroy(surf_object);
        api.cuArrayDestroy(array);
    }

    /// Test that surface objects on 3D arrays with 4-channel FLOAT get shift=4.
    #[test_cuda]
    unsafe fn surf_3d_float4_shift_encoding(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 32,
            Height: 32,
            Depth: 32,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 4,
            Flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &array_desc);
        assert!(!array.is_null());

        let mut res_desc: CUDA_RESOURCE_DESC = mem::zeroed();
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;

        let mut surf_object: CUsurfObject = 0;
        api.cuSurfObjectCreate(&mut surf_object, &res_desc);
        assert_ne!(surf_object, 0);

        // FLOAT * 4 channels = 16 bytes, trailing_zeros(16) = 4
        let shift = (surf_object as u64) >> 61;
        assert_eq!(shift, 4,
            "FLOAT4 surface shift should be 4 (log2(16)), got {}", shift);

        api.cuSurfObjectDestroy(surf_object);
        api.cuArrayDestroy(array);
    }

    /// Test that hipSurfaceObject_t from hipCreateSurfaceObject is a raw pointer
    /// WITHOUT shift encoding in the top 3 bits.
    /// This confirms the root cause: surfref-bound surfaces pass this raw handle
    /// to kernels, but our sust helpers expect shift-encoded handles.
    #[test]
    fn surface_object_has_no_shift_bits() {
        use hip_runtime_sys::*;
        if unsafe { hipInit(0) }.is_err() {
            return; // skip if no GPU
        }

        unsafe {
            // Create a 3D FLOAT array (same as TurbulenceFS uses)
            let mut array: hipArray_t = std::ptr::null_mut();
            let desc = HIP_ARRAY3D_DESCRIPTOR {
                Width: 8,
                Height: 8,
                Depth: 8,
                Format: hipArray_Format::HIP_AD_FORMAT_FLOAT,
                NumChannels: 1,
                Flags: 0x02, // CUDA_ARRAY3D_SURFACE_LDST
            };
            hipArray3DCreate(&mut array, &desc).unwrap();

            // Create surface object (this is what hipTexRefSetArray does internally)
            let mut surf_obj: hipSurfaceObject_t = mem::zeroed();
            let res_desc = hipResourceDesc {
                resType: hipResourceType::hipResourceTypeArray,
                res: hipResourceDesc__bindgen_ty_1 {
                    array: hipResourceDesc__bindgen_ty_1__bindgen_ty_1 { array },
                },
            };
            hipCreateSurfaceObject(&mut surf_obj, &res_desc).unwrap();

            let handle = surf_obj as u64;
            let top_bits = handle >> 61;

            eprintln!("surface_object_has_no_shift_bits: raw handle = {:#x}, top_bits = {}", handle, top_bits);

            // The raw hipSurfaceObject_t should NOT have shift bits — it's just a pointer.
            // This is exactly what the surfref path passes to kernels.
            // Our sust helpers extract shift from top 3 bits and get 0 instead of 2.
            assert_eq!(top_bits, 0,
                "Raw hipSurfaceObject_t unexpectedly has non-zero top bits: {:#x}",
                handle);

            // For comparison: our cuSurfObjectCreate ADDS shift bits.
            // FLOAT (4 bytes) * 1 channel = 4 → trailing_zeros(4) = 2
            // encoded = handle | (2 << 61) → top bits = 2
            let encoded = handle | (2u64 << 61);
            let encoded_shift = encoded >> 61;
            assert_eq!(encoded_shift, 2);

            eprintln!("  raw handle {:#x} (shift=0) vs encoded {:#x} (shift=2)", handle, encoded);
            eprintln!("  BUG: surfref path passes raw handle → kernel gets shift=0 → wrong x-offset");

            hipDestroySurfaceObject(surf_obj).unwrap();
            hipFreeArray(array).unwrap();
        }
    }

    /// Test that hipArray3DGetDescriptor returns correct format for 3D arrays
    /// while hipArrayGetDescriptor may not.
    #[test_cuda]
    unsafe fn array_3d_descriptor_format(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 16,
            Height: 16,
            Depth: 16,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 1,
            Flags: 0,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &array_desc);
        assert!(!array.is_null());

        // Check 3D descriptor
        let mut out_desc3d: CUDA_ARRAY3D_DESCRIPTOR = mem::zeroed();
        api.cuArray3DGetDescriptor_v2(&mut out_desc3d, array);
        assert_eq!(out_desc3d.Width, 16);
        assert_eq!(out_desc3d.Height, 16);
        assert_eq!(out_desc3d.NumChannels, 1);
        // Format should match what we requested (or the non-broken substitute)
        let fmt = out_desc3d.Format;
        assert!(
            fmt == CUarray_format::CU_AD_FORMAT_FLOAT
                || fmt == CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32,
            "3D descriptor format should be FLOAT or UINT32, got {:?}", fmt);

        api.cuArrayDestroy(array);
    }

    /// Test that cuTexObjectCreate adds shift encoding to texture object handles.
    /// PhysX uses texture objects for surface writes (sust). Our sust helpers
    /// extract the shift from the top 3 bits. Without encoding, shift=0 → wrong x-offset.
    ///
    /// Before fix: FAILS (cuTexObjectCreate returns raw handle, shift=0)
    /// After fix: PASSES (cuTexObjectCreate adds shift=2 for FLOAT)
    #[test_cuda]
    unsafe fn texobj_for_sust_has_shift(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        // Create a 3D FLOAT array
        let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 64,
            Height: 64,
            Depth: 64,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 1,
            Flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &array_desc);
        assert!(!array.is_null());

        // Create a TEXTURE object via cuTexObjectCreate (not cuSurfObjectCreate)
        let mut res_desc: CUDA_RESOURCE_DESC = mem::zeroed();
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;

        let tex_desc: CUDA_TEXTURE_DESC = mem::zeroed();
        let mut tex_object: CUtexObject = 0;
        api.cuTexObjectCreate(&mut tex_object, &res_desc, &tex_desc, std::ptr::null());
        assert_ne!(tex_object, 0, "texture object creation failed");

        // The handle must have shift=2 in top 3 bits (FLOAT = 4 bytes, log2(4) = 2)
        let shift = (tex_object as u64) >> 61;
        assert_eq!(shift, 2,
            "cuTexObjectCreate handle has shift={} (expected 2 for FLOAT). \
             Texture objects used for sust need shift encoding. handle={:#x}",
            shift, tex_object as u64);

        api.cuTexObjectDestroy(tex_object);
        api.cuArrayDestroy(array);
    }

    /// Test that confirms the core issue: on AMDGPU, the image descriptor's format
    /// can be read from the descriptor words, enabling runtime shift computation.
    /// This is the mechanism the fix will use — read format from descriptor instead
    /// of relying on top-3-bit encoding in the handle.
    #[test]
    fn descriptor_format_readable_from_words() {
        use hip_runtime_sys::*;
        if unsafe { hipInit(0) }.is_err() {
            return;
        }

        unsafe {
            let mut array: hipArray_t = std::ptr::null_mut();
            let desc = HIP_ARRAY3D_DESCRIPTOR {
                Width: 8,
                Height: 8,
                Depth: 8,
                Format: hipArray_Format::HIP_AD_FORMAT_FLOAT,
                NumChannels: 1,
                Flags: 0x02,
            };
            hipArray3DCreate(&mut array, &desc).unwrap();

            let mut surf_obj: hipSurfaceObject_t = mem::zeroed();
            let res_desc = hipResourceDesc {
                resType: hipResourceType::hipResourceTypeArray,
                res: hipResourceDesc__bindgen_ty_1 {
                    array: hipResourceDesc__bindgen_ty_1__bindgen_ty_1 { array },
                },
            };
            hipCreateSurfaceObject(&mut surf_obj, &res_desc).unwrap();

            // Read the 32-byte image descriptor from device memory
            let mut words = [0u32; 8];
            hipMemcpyDtoH(
                words.as_mut_ptr() as _,
                hipDeviceptr_t(surf_obj as _),
                32,
            ).unwrap();

            eprintln!("descriptor_format_readable: words = {:08x?}", words);

            // The descriptor should have non-zero content (valid image descriptor)
            assert!(words[0] != 0 || words[1] != 0,
                "Descriptor is all zeros — surface object is not a valid descriptor pointer");

            // AMDGPU descriptor word 3 contains data_format and num_format fields
            // that encode the pixel format. If we can read these, the sust helper
            // can compute the shift at runtime without top-3-bit encoding.
            let word3 = words[3];
            eprintln!("  word3 = 0x{:08x} (contains data_format/num_format)", word3);
            assert_ne!(word3, 0, "Descriptor word 3 should contain format info");

            hipDestroySurfaceObject(surf_obj).unwrap();
            hipFreeArray(array).unwrap();
        }
    }
}

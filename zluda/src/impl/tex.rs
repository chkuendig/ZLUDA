use crate::r#impl::{hipfix, surf};
use cuda_types::cuda::CUresult;
use hip_runtime_sys::*;

pub(crate) unsafe fn object_create(
    p_tex_object: *mut hipTextureObject_t,
    p_res_desc: *const HIP_RESOURCE_DESC,
    p_tex_desc: *const HIP_TEXTURE_DESC,
    p_res_view_desc: *const HIP_RESOURCE_VIEW_DESC,
) -> CUresult {
    // HIP's sampler cannot program anisotropy or a mipmap level bias and
    // returns hipErrorNotSupported for either. Drop them; CUDA uses neither
    // for the non-mipmapped resources HIP supports.
    let mut tex_desc = *p_tex_desc.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    tex_desc.maxAnisotropy = 0;
    tex_desc.mipmapLevelBias = 0.0;
    hipTexObjectCreate(p_tex_object, p_res_desc, &tex_desc, p_res_view_desc)?;
    // PhysX also writes through texture objects with sust, so array-backed
    // ones carry the pixel shift of surface objects.
    if (*p_res_desc).resType == HIPresourcetype::HIP_RESOURCE_TYPE_ARRAY {
        let array = (*p_res_desc).res.array.hArray;
        match surf::encode_pixel_shift(*p_tex_object as usize, array) {
            Ok(handle) => *p_tex_object = handle as hipTextureObject_t,
            Err(error) => {
                let _ = hipDestroyTextureObject(*p_tex_object);
                return Err(error);
            }
        }
    }
    Ok(())
}

pub(crate) unsafe fn object_destroy(tex_object: hipTextureObject_t) -> hipError_t {
    hipDestroyTextureObject(surf::strip_pixel_shift(tex_object as usize) as hipTextureObject_t)
}

pub(crate) unsafe fn ref_set_array(
    texref: *mut textureReference,
    array: hipArray_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    // CUDA also accepts flags=0, HIP only HIP_TRSA_OVERRIDE_FORMAT. HIP's
    // texture references take the array's format either way.
    if flags & !hipfix::HIP_TRSA_OVERRIDE_FORMAT != 0 {
        return Err(hipErrorCode_t::InvalidValue);
    }
    hipTexRefSetArray(texref, array, hipfix::HIP_TRSA_OVERRIDE_FORMAT)
}

pub(crate) unsafe fn ref_set_flags(
    raw_texref: *mut textureReference,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    fn get_flags(texref: &textureReference) -> (u32, i32, i32) {
        (texref.readMode.0, texref.normalized, texref.sRGB)
    }
    // End the shared borrow before HIP mutates this reference.
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    let pre_flags = get_flags(&previous);
    hipTexRefSetFlags(raw_texref, flags)?;
    let post_flags = get_flags(raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?);
    if pre_flags != post_flags {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_filter_mode(
    raw_texref: *mut textureReference,
    filter_mode: hipTextureFilterMode,
) -> hipError_t {
    fn get_flags(texref: &textureReference) -> u32 {
        texref.filterMode.0
    }
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    let pre_flags = get_flags(&previous);
    hipTexRefSetFilterMode(raw_texref, filter_mode)?;
    let post_flags = get_flags(raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?);
    if pre_flags != post_flags {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_address_mode(
    raw_texref: *mut textureReference,
    dim: i32,
    address_mode: hipTextureAddressMode,
) -> hipError_t {
    fn get_flags(texref: &textureReference, dim: i32) -> u32 {
        texref.addressMode[dim as usize].0
    }
    if dim < 0 || dim > 2 {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    let pre_flags = get_flags(&previous, dim);
    hipTexRefSetAddressMode(raw_texref, dim, address_mode)?;
    let post_flags = get_flags(raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?, dim);
    if pre_flags != post_flags {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_mipmap_filter_mode(
    raw_texref: *mut textureReference,
    filter_mode: hipTextureFilterMode,
) -> hipError_t {
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    hipTexRefSetMipmapFilterMode(raw_texref, filter_mode)?;
    let current = raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?.mipmapFilterMode;
    if previous.mipmapFilterMode != current {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_mipmap_level_bias(
    raw_texref: *mut textureReference,
    bias: f32,
) -> hipError_t {
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    hipTexRefSetMipmapLevelBias(raw_texref, bias)?;
    let current = raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?.mipmapLevelBias;
    if previous.mipmapLevelBias.to_bits() != current.to_bits() {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_mipmap_level_clamp(
    raw_texref: *mut textureReference,
    min_clamp: f32,
    max_clamp: f32,
) -> hipError_t {
    fn clamps(texref: &textureReference) -> (u32, u32) {
        (texref.minMipmapLevelClamp.to_bits(), texref.maxMipmapLevelClamp.to_bits())
    }
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    hipTexRefSetMipmapLevelClamp(raw_texref, min_clamp, max_clamp)?;
    let current = clamps(raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?);
    if clamps(&previous) != current {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_max_anisotropy(
    raw_texref: *mut textureReference,
    max_anisotropy: u32,
) -> hipError_t {
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    hipTexRefSetMaxAnisotropy(raw_texref, max_anisotropy)?;
    let current = raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?.maxAnisotropy;
    if previous.maxAnisotropy != current {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_format(
    raw_texref: *mut textureReference,
    format: hipArray_Format,
    num_components: ::core::ffi::c_int,
) -> hipError_t {
    fn get_flags(texref: &textureReference) -> (u32, u32) {
        (texref.format.0, texref.numChannels as u32)
    }
    let previous = *raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?;
    let pre_flags = get_flags(&previous);
    hipTexRefSetFormat(raw_texref, format, num_components)?;
    let post_flags = get_flags(raw_texref.as_ref().ok_or(hipErrorCode_t::InvalidValue)?);
    if pre_flags != post_flags {
        hipfix::refresh_texref(raw_texref, previous)?;
    }
    Ok(())
}

pub(crate) unsafe fn ref_set_address_v2(
    byte_offset: *mut usize,
    texref: *mut textureReference,
    dev_ptr: hipDeviceptr_t,
    bytes: usize,
) -> hipError_t {
    if dev_ptr.0.is_null() {
        return hipUnbindTexture(texref);
    }
    // CUDA accepts an empty range, HIP rejects it. Nothing can be read
    // through an empty range, so keep the current binding.
    if bytes == 0 {
        if let Some(byte_offset) = byte_offset.as_mut() {
            *byte_offset = 0;
        }
        return Ok(());
    }
    hipTexRefSetAddress(byte_offset, texref, dev_ptr, bytes)
}

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::*;
    use std::collections::HashMap;
    use std::ffi::{c_void, CString};

    /*
     *   █████╗ ██╗   ██╗████████╗ ██████╗  ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ████████╗███████╗██████╗
     *  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██╔══██╗
     *  ███████║██║   ██║   ██║   ██║   ██║██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║   ██║   █████╗  ██║  ██║
     *  ██╔══██║██║   ██║   ██║   ██║   ██║██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██╔══╝  ██║  ██║
     *  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║   ██║   ███████╗██████╔╝
     *  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝
     *
     *   █████╗ ██╗    ███████╗██╗      ██████╗ ██████╗     ████████╗███████╗███████╗████████╗███████╗    ██████╗ ███████╗██╗      ██████╗ ██╗    ██╗
     *  ██╔══██╗██║    ██╔════╝██║     ██╔═══██╗██╔══██╗    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝    ██╔══██╗██╔════╝██║     ██╔═══██╗██║    ██║
     *  ███████║██║    ███████╗██║     ██║   ██║██████╔╝       ██║   █████╗  ███████╗   ██║   ███████╗    ██████╔╝█████╗  ██║     ██║   ██║██║ █╗ ██║
     *  ██╔══██║██║    ╚════██║██║     ██║   ██║██╔═══╝        ██║   ██╔══╝  ╚════██║   ██║   ╚════██║    ██╔══██╗██╔══╝  ██║     ██║   ██║██║███╗██║
     *  ██║  ██║██║    ███████║███████╗╚██████╔╝██║            ██║   ███████╗███████║   ██║   ███████║    ██████╔╝███████╗███████╗╚██████╔╝╚███╔███╔╝
     *  ╚═╝  ╚═╝╚═╝    ╚══════╝╚══════╝ ╚═════╝ ╚═╝            ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝    ╚═════╝ ╚══════╝╚══════╝ ╚═════╝  ╚══╝╚══╝
     */

    const TEX_READ_1D_INT_PTX: &str = concat!(include_str!("test_ptx/tex_read_1d_int.ptx"), "\0");
    const TEX_READ_1D_FLOAT_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_1d_float.ptx"), "\0");
    const TEX_READ_2D_INT_PTX: &str = concat!(include_str!("test_ptx/tex_read_2d_int.ptx"), "\0");
    const TEX_READ_2D_FLOAT_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_2d_float.ptx"), "\0");
    const TEX_READ_3D_INT_PTX: &str = concat!(include_str!("test_ptx/tex_read_3d_int.ptx"), "\0");
    const TEX_READ_3D_FLOAT_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_3d_float.ptx"), "\0");
    const TEX_READ_1D_INT_S32COORD_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_1d_int_s32coord.ptx"), "\0");
    const TEX_READ_1D_FLOAT_S32COORD_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_1d_float_s32coord.ptx"),
        "\0"
    );
    const TEX_READ_2D_INT_S32COORD_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_2d_int_s32coord.ptx"), "\0");
    const TEX_READ_2D_FLOAT_S32COORD_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_2d_float_s32coord.ptx"),
        "\0"
    );
    const TEX_READ_3D_INT_S32COORD_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_3d_int_s32coord.ptx"), "\0");
    const TEX_READ_3D_FLOAT_S32COORD_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_3d_float_s32coord.ptx"),
        "\0"
    );

    const TEX_READ_1D_INT_TEXOBJ_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_1d_int_texobj.ptx"), "\0");
    const TEX_READ_1D_FLOAT_TEXOBJ_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_1d_float_texobj.ptx"), "\0");
    const TEX_READ_2D_INT_TEXOBJ_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_2d_int_texobj.ptx"), "\0");
    const TEX_READ_2D_FLOAT_TEXOBJ_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_2d_float_texobj.ptx"), "\0");
    const TEX_READ_3D_INT_TEXOBJ_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_3d_int_texobj.ptx"), "\0");
    const TEX_READ_3D_FLOAT_TEXOBJ_PTX: &str =
        concat!(include_str!("test_ptx/tex_read_3d_float_texobj.ptx"), "\0");
    const TEX_READ_1D_INT_S32COORD_TEXOBJ_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_1d_int_s32coord_texobj.ptx"),
        "\0"
    );
    const TEX_READ_1D_FLOAT_S32COORD_TEXOBJ_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_1d_float_s32coord_texobj.ptx"),
        "\0"
    );
    const TEX_READ_2D_INT_S32COORD_TEXOBJ_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_2d_int_s32coord_texobj.ptx"),
        "\0"
    );
    const TEX_READ_2D_FLOAT_S32COORD_TEXOBJ_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_2d_float_s32coord_texobj.ptx"),
        "\0"
    );
    const TEX_READ_3D_INT_S32COORD_TEXOBJ_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_3d_int_s32coord_texobj.ptx"),
        "\0"
    );
    const TEX_READ_3D_FLOAT_S32COORD_TEXOBJ_PTX: &str = concat!(
        include_str!("test_ptx/tex_read_3d_float_s32coord_texobj.ptx"),
        "\0"
    );

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    enum TexDim {
        One,
        Two,
        Three,
    }

    /// Returns the byte size of a single element for the given format.
    fn format_byte_size(fmt: CUarray_format) -> usize {
        match fmt {
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8
            | CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8 => 1,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16
            | CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16
            | CUarray_format_enum::CU_AD_FORMAT_HALF => 2,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32
            | CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32
            | CUarray_format_enum::CU_AD_FORMAT_FLOAT => 4,
            _ => panic!("unsupported format"),
        }
    }

    /// Returns true if the format is a floating-point type (HALF or FLOAT).
    fn format_is_float(fmt: CUarray_format) -> bool {
        matches!(
            fmt,
            CUarray_format_enum::CU_AD_FORMAT_HALF | CUarray_format_enum::CU_AD_FORMAT_FLOAT
        )
    }

    /// Write a test value into a host buffer at the given byte offset.
    /// Returns the expected 4xi32 or 4xf32 result (as [u32; 4] bit pattern).
    fn write_test_pixel(
        host_buf: &mut [u8],
        offset: usize,
        fmt: CUarray_format,
        num_channels: u32,
    ) -> [u32; 4] {
        let elem_size = format_byte_size(fmt);

        // Channel test values - small enough for all integer formats
        let channel_values: [i32; 4] = [42, 17, 99, 7];
        let mut expected = [0u32; 4];

        for ch in 0..num_channels as usize {
            let val = channel_values[ch];
            let byte_off = offset + ch * elem_size;
            match fmt {
                CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8 => {
                    host_buf[byte_off] = val as u8;
                    expected[ch] = val as u32;
                }
                CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8 => {
                    host_buf[byte_off] = val as u8;
                    expected[ch] = val as u32;
                }
                CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16 => {
                    let bytes = (val as u16).to_ne_bytes();
                    host_buf[byte_off..byte_off + 2].copy_from_slice(&bytes);
                    expected[ch] = val as u32;
                }
                CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16 => {
                    let bytes = (val as i16).to_ne_bytes();
                    host_buf[byte_off..byte_off + 2].copy_from_slice(&bytes);
                    expected[ch] = val as u32;
                }
                CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32 => {
                    let bytes = (val as u32).to_ne_bytes();
                    host_buf[byte_off..byte_off + 4].copy_from_slice(&bytes);
                    expected[ch] = val as u32;
                }
                CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32 => {
                    let bytes = val.to_ne_bytes();
                    host_buf[byte_off..byte_off + 4].copy_from_slice(&bytes);
                    expected[ch] = val as u32;
                }
                CUarray_format_enum::CU_AD_FORMAT_HALF => {
                    let h = half::f16::from_f32(val as f32);
                    let bytes = h.to_ne_bytes();
                    host_buf[byte_off..byte_off + 2].copy_from_slice(&bytes);
                    expected[ch] = (val as f32).to_bits();
                }
                CUarray_format_enum::CU_AD_FORMAT_FLOAT => {
                    let bytes = (val as f32).to_ne_bytes();
                    host_buf[byte_off..byte_off + 4].copy_from_slice(&bytes);
                    expected[ch] = (val as f32).to_bits();
                }
                _ => panic!("unsupported format"),
            }
        }
        expected
    }

    /// Core test logic: create array, upload data, bind to texref, launch
    /// kernel, read back and verify.
    unsafe fn texref_read_test(
        api: &impl CudaApi,
        normalized: bool,
        dim: TexDim,
        fmt: CUarray_format,
        num_channels: u32,
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let elem_size = format_byte_size(fmt);
        let pixel_size = elem_size * num_channels as usize;
        let row_bytes = width * pixel_size;

        let (effective_height, effective_depth) = match dim {
            TexDim::One => (1, 1),
            TexDim::Two => (height, 1),
            TexDim::Three => (height, depth),
        };
        let total_bytes = row_bytes * effective_height * effective_depth;

        // Pick a deterministic "random" pixel based on dimensions
        let px = (width / 2).min(width - 1);
        let py = (effective_height / 2).min(effective_height - 1);
        let pz = (effective_depth / 2).min(effective_depth - 1);

        let pixel_offset = pz * (row_bytes * effective_height) + py * row_bytes + px * pixel_size;

        // Create host buffer filled with zeros, then set test pixel
        let mut host_buf = vec![0u8; total_bytes];
        let expected = write_test_pixel(&mut host_buf, pixel_offset, fmt, num_channels);

        // Create CUDA array and upload data
        let mut array: CUarray = std::mem::zeroed();
        match dim {
            TexDim::One => {
                let desc = CUDA_ARRAY_DESCRIPTOR {
                    Width: width,
                    Height: 0,
                    Format: fmt,
                    NumChannels: num_channels,
                };
                api.cuArrayCreate_v2(&mut array, &desc);
                api.cuMemcpyHtoA_v2(array, 0, host_buf.as_ptr() as *const c_void, total_bytes);
            }
            TexDim::Two => {
                let desc = CUDA_ARRAY_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Format: fmt,
                    NumChannels: num_channels,
                };
                api.cuArrayCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    dstPitch: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                };
                api.cuMemcpy2D_v2(&copy_params);
            }
            TexDim::Three => {
                let desc = CUDA_ARRAY3D_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Depth: effective_depth,
                    Format: fmt,
                    NumChannels: num_channels,
                    Flags: 0,
                };
                api.cuArray3DCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY3D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcZ: 0,
                    srcLOD: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    reserved0: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    srcHeight: effective_height,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstZ: 0,
                    dstLOD: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    reserved1: std::ptr::null_mut(),
                    dstPitch: 0,
                    dstHeight: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                    Depth: effective_depth,
                };
                api.cuMemcpy3D_v2(&copy_params);
            }
        }

        // Load PTX module and get texref + kernel function
        let is_float = format_is_float(fmt);
        let (ptx, kernel_name): (&str, &std::ffi::CStr) = match (dim, is_float) {
            (TexDim::One, false) => (TEX_READ_1D_INT_PTX, c"tex_read_1d_int"),
            (TexDim::One, true) => (TEX_READ_1D_FLOAT_PTX, c"tex_read_1d_float"),
            (TexDim::Two, false) => (TEX_READ_2D_INT_PTX, c"tex_read_2d_int"),
            (TexDim::Two, true) => (TEX_READ_2D_FLOAT_PTX, c"tex_read_2d_float"),
            (TexDim::Three, false) => (TEX_READ_3D_INT_PTX, c"tex_read_3d_int"),
            (TexDim::Three, true) => (TEX_READ_3D_FLOAT_PTX, c"tex_read_3d_float"),
        };

        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void);

        let mut func = std::mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, kernel_name.as_ptr());

        let mut texref = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texref, module, c"tex".as_ptr());

        // Bind array to texref
        api.cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);
        api.cuTexRefSetFormat(texref, fmt, num_channels as i32);
        api.cuTexRefSetFilterMode(texref, CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT);
        for d in 0..3 {
            api.cuTexRefSetAddressMode(texref, d, CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP);
        }
        let flags = match (!is_float, normalized) {
            (true, true) => CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES,
            (true, false) => CU_TRSF_READ_AS_INTEGER,
            (false, true) => CU_TRSF_NORMALIZED_COORDINATES,
            (false, false) => 0,
        };
        api.cuTexRefSetFlags(texref, flags);

        // Allocate output buffer on device (4 x f32 or 4 x s32 = 16 bytes)
        let mut d_output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_output, 16);

        // Texture coordinates for point sampling: pixel center = (p + 0.5)
        let mut coord_x: f32 = px as f32 + 0.5;
        if normalized {
            coord_x /= width as f32;
        }
        let mut coord_y: f32 = py as f32 + 0.5;
        if normalized {
            coord_y /= height as f32;
        }
        let mut coord_z: f32 = pz as f32 + 0.5;
        if normalized {
            coord_z /= depth as f32;
        }
        let mut params: [*mut c_void; 4] = [
            &d_output as *const _ as *mut _,
            &coord_x as *const _ as *mut _,
            &coord_y as *const _ as *mut _,
            &coord_z as *const _ as *mut _,
        ];
        api.cuLaunchKernel(
            func,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));

        // Read back result
        let mut result = [0u32; 4];
        api.cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, d_output, 16);

        // Verify
        let fmt_name = format!("{:?}", fmt);
        let dim_name = match dim {
            TexDim::One => "1d",
            TexDim::Two => "2d",
            TexDim::Three => "3d",
        };
        assert_eq!(
            result[..num_channels as usize],
            expected[..num_channels as usize],
            "texref mismatch for dim={dim_name}, format={fmt_name}, channels={num_channels}, \
             size={width}x{effective_height}x{effective_depth}, pixel=({px},{py},{pz}), normalized={normalized}, is_float={is_float}"
        );

        // Cleanup
        api.cuMemFree_v2(d_output);
        api.cuModuleUnload(module);
        api.cuArrayDestroy(array);
    }

    #[test_cuda]
    unsafe fn texref_formats_channels_dimensions(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);

        let formats = [
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_HALF,
            CUarray_format_enum::CU_AD_FORMAT_FLOAT,
        ];
        let channel_counts: [u32; 3] = [1, 2, 4];

        // 1D tests: vary width only
        let widths_1d: [usize; 4] = [1, 4, 16, 64];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &w in &widths_1d {
                    for normalized in [false, true] {
                        texref_read_test(&api, normalized, TexDim::One, fmt, num_ch, w, 1, 1);
                    }
                }
            }
        }

        // 2D tests: vary width x height
        let dimensions_2d: [(usize, usize); 4] = [(1, 1), (4, 4), (16, 16), (64, 37)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h) in &dimensions_2d {
                    for normalized in [false, true] {
                        texref_read_test(&api, normalized, TexDim::Two, fmt, num_ch, w, h, 1);
                    }
                }
            }
        }

        // 3D tests: vary width x height x depth
        let dimensions_3d: [(usize, usize, usize); 4] =
            [(1, 1, 1), (4, 4, 4), (8, 8, 8), (16, 13, 7)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h, d) in &dimensions_3d {
                    for normalized in [false, true] {
                        texref_read_test(&api, normalized, TexDim::Three, fmt, num_ch, w, h, d);
                    }
                }
            }
        }

        api.cuCtxDestroy_v2(ctx);
    }

    /// Core test logic for s32 coordinates: create array, upload data, bind to
    /// texref, launch kernel with integer coordinates, read back and verify.
    unsafe fn texref_read_s32coord_test(
        api: &impl CudaApi,
        dim: TexDim,
        fmt: CUarray_format,
        num_channels: u32,
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let elem_size = format_byte_size(fmt);
        let pixel_size = elem_size * num_channels as usize;
        let row_bytes = width * pixel_size;

        let (effective_height, effective_depth) = match dim {
            TexDim::One => (1, 1),
            TexDim::Two => (height, 1),
            TexDim::Three => (height, depth),
        };
        let total_bytes = row_bytes * effective_height * effective_depth;

        let px = (width / 2).min(width - 1);
        let py = (effective_height / 2).min(effective_height - 1);
        let pz = (effective_depth / 2).min(effective_depth - 1);

        let pixel_offset = pz * (row_bytes * effective_height) + py * row_bytes + px * pixel_size;

        let mut host_buf = vec![0u8; total_bytes];
        let expected = write_test_pixel(&mut host_buf, pixel_offset, fmt, num_channels);

        // Create CUDA array and upload data
        let mut array: CUarray = std::mem::zeroed();
        let mut dev_ptr: CUdeviceptr_v2 = std::mem::zeroed();
        match dim {
            TexDim::One => {
                api.cuMemAlloc_v2(&mut dev_ptr, total_bytes);
                api.cuMemcpyHtoD_v2(dev_ptr, host_buf.as_ptr() as *const c_void, total_bytes);
            }
            TexDim::Two => {
                let desc = CUDA_ARRAY_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Format: fmt,
                    NumChannels: num_channels,
                };
                api.cuArrayCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    dstPitch: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                };
                api.cuMemcpy2D_v2(&copy_params);
            }
            TexDim::Three => {
                let desc = CUDA_ARRAY3D_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Depth: effective_depth,
                    Format: fmt,
                    NumChannels: num_channels,
                    Flags: 0,
                };
                api.cuArray3DCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY3D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcZ: 0,
                    srcLOD: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    reserved0: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    srcHeight: effective_height,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstZ: 0,
                    dstLOD: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    reserved1: std::ptr::null_mut(),
                    dstPitch: 0,
                    dstHeight: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                    Depth: effective_depth,
                };
                api.cuMemcpy3D_v2(&copy_params);
            }
        }

        // Load PTX module and get texref + kernel function
        let is_float = format_is_float(fmt);
        let (ptx, kernel_name): (&str, &std::ffi::CStr) = match (dim, is_float) {
            (TexDim::One, false) => (TEX_READ_1D_INT_S32COORD_PTX, c"tex_read_1d_int_s32coord"),
            (TexDim::One, true) => (
                TEX_READ_1D_FLOAT_S32COORD_PTX,
                c"tex_read_1d_float_s32coord",
            ),
            (TexDim::Two, false) => (TEX_READ_2D_INT_S32COORD_PTX, c"tex_read_2d_int_s32coord"),
            (TexDim::Two, true) => (
                TEX_READ_2D_FLOAT_S32COORD_PTX,
                c"tex_read_2d_float_s32coord",
            ),
            (TexDim::Three, false) => (TEX_READ_3D_INT_S32COORD_PTX, c"tex_read_3d_int_s32coord"),
            (TexDim::Three, true) => (
                TEX_READ_3D_FLOAT_S32COORD_PTX,
                c"tex_read_3d_float_s32coord",
            ),
        };

        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void);

        let mut func = std::mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, kernel_name.as_ptr());

        let mut texref = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texref, module, c"tex".as_ptr());

        // Bind array to texref
        if dim == TexDim::One {
            api.cuTexRefSetAddress_v2(&mut 0, texref, dev_ptr, total_bytes);
        } else {
            api.cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);
        }
        api.cuTexRefSetFormat(texref, fmt, num_channels as i32);
        api.cuTexRefSetFilterMode(texref, CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT);
        for d in 0..3 {
            api.cuTexRefSetAddressMode(texref, d, CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP);
        }
        let flags = if !is_float {
            CU_TRSF_READ_AS_INTEGER
        } else {
            0
        };
        api.cuTexRefSetFlags(texref, flags);

        // Allocate output buffer on device (4 x f32 or 4 x s32 = 16 bytes)
        let mut d_output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_output, 16);

        // Integer coordinates: direct pixel index
        let coord_x: i32 = px as i32;
        let coord_y: i32 = py as i32;
        let coord_z: i32 = pz as i32;
        let mut params: [*mut c_void; 4] = [
            &d_output as *const _ as *mut _,
            &coord_x as *const _ as *mut _,
            &coord_y as *const _ as *mut _,
            &coord_z as *const _ as *mut _,
        ];
        api.cuLaunchKernel(
            func,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));

        // Read back result
        let mut result = [0u32; 4];
        api.cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, d_output, 16);

        // Verify
        let fmt_name = format!("{:?}", fmt);
        let dim_name = match dim {
            TexDim::One => "1d",
            TexDim::Two => "2d",
            TexDim::Three => "3d",
        };
        assert_eq!(
            result[..num_channels as usize],
            expected[..num_channels as usize],
            "texref s32coord mismatch for dim={dim_name}, format={fmt_name}, channels={num_channels}, \
             size={width}x{effective_height}x{effective_depth}, pixel=({px},{py},{pz}), is_float={is_float}"
        );

        // Cleanup
        api.cuMemFree_v2(d_output);
        api.cuModuleUnload(module);
        if !array.is_null() {
            api.cuArrayDestroy(array);
        }
        if !dev_ptr.0.is_null() {
            api.cuMemFree_v2(dev_ptr);
        }
    }

    #[test_cuda]
    unsafe fn texref_s32coord_formats_channels_dimensions(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);

        let formats = [
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_HALF,
            CUarray_format_enum::CU_AD_FORMAT_FLOAT,
        ];
        let channel_counts: [u32; 3] = [1, 2, 4];

        // 1D tests
        let widths_1d: [usize; 4] = [1, 4, 16, 64];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &w in &widths_1d {
                    texref_read_s32coord_test(&api, TexDim::One, fmt, num_ch, w, 1, 1);
                }
            }
        }

        // 2D tests
        let dimensions_2d: [(usize, usize); 4] = [(1, 1), (4, 4), (16, 16), (64, 37)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h) in &dimensions_2d {
                    texref_read_s32coord_test(&api, TexDim::Two, fmt, num_ch, w, h, 1);
                }
            }
        }

        // 3D tests
        let dimensions_3d: [(usize, usize, usize); 4] =
            [(1, 1, 1), (4, 4, 4), (8, 8, 8), (16, 13, 7)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h, d) in &dimensions_3d {
                    texref_read_s32coord_test(&api, TexDim::Three, fmt, num_ch, w, h, d);
                }
            }
        }

        api.cuCtxDestroy_v2(ctx);
    }

    /// Core test logic: create array, upload data, create texobj, launch
    /// kernel, read back and verify.
    unsafe fn texobj_read_test(
        api: &impl CudaApi,
        normalized: bool,
        dim: TexDim,
        fmt: CUarray_format,
        num_channels: u32,
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let elem_size = format_byte_size(fmt);
        let pixel_size = elem_size * num_channels as usize;
        let row_bytes = width * pixel_size;

        let (effective_height, effective_depth) = match dim {
            TexDim::One => (1, 1),
            TexDim::Two => (height, 1),
            TexDim::Three => (height, depth),
        };
        let total_bytes = row_bytes * effective_height * effective_depth;

        let px = (width / 2).min(width - 1);
        let py = (effective_height / 2).min(effective_height - 1);
        let pz = (effective_depth / 2).min(effective_depth - 1);

        let pixel_offset = pz * (row_bytes * effective_height) + py * row_bytes + px * pixel_size;

        let mut host_buf = vec![0u8; total_bytes];
        let expected = write_test_pixel(&mut host_buf, pixel_offset, fmt, num_channels);

        // Create CUDA array and upload data
        let mut array: CUarray = std::mem::zeroed();
        match dim {
            TexDim::One => {
                let desc = CUDA_ARRAY_DESCRIPTOR {
                    Width: width,
                    Height: 0,
                    Format: fmt,
                    NumChannels: num_channels,
                };
                api.cuArrayCreate_v2(&mut array, &desc);
                api.cuMemcpyHtoA_v2(array, 0, host_buf.as_ptr() as *const c_void, total_bytes);
            }
            TexDim::Two => {
                let desc = CUDA_ARRAY_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Format: fmt,
                    NumChannels: num_channels,
                };
                api.cuArrayCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    dstPitch: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                };
                api.cuMemcpy2D_v2(&copy_params);
            }
            TexDim::Three => {
                let desc = CUDA_ARRAY3D_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Depth: effective_depth,
                    Format: fmt,
                    NumChannels: num_channels,
                    Flags: 0,
                };
                api.cuArray3DCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY3D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcZ: 0,
                    srcLOD: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    reserved0: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    srcHeight: effective_height,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstZ: 0,
                    dstLOD: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    reserved1: std::ptr::null_mut(),
                    dstPitch: 0,
                    dstHeight: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                    Depth: effective_depth,
                };
                api.cuMemcpy3D_v2(&copy_params);
            }
        }

        // Create texture object
        let res_desc = CUDA_RESOURCE_DESC {
            resType: CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY,
            res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 { hArray: array },
            },
            flags: 0,
        };
        let is_float = format_is_float(fmt);
        let flags = match (!is_float, normalized) {
            (true, true) => CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES,
            (true, false) => CU_TRSF_READ_AS_INTEGER,
            (false, true) => CU_TRSF_NORMALIZED_COORDINATES,
            (false, false) => 0,
        };
        let tex_desc = CUDA_TEXTURE_DESC {
            addressMode: [CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP; 3],
            filterMode: CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
            flags,
            maxAnisotropy: 0,
            mipmapFilterMode: CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0; 12],
        };
        let mut texobj: CUtexObject = std::mem::zeroed();
        api.cuTexObjectCreate(&mut texobj, &res_desc, &tex_desc, std::ptr::null());

        // Load PTX module and get kernel function
        let (ptx, kernel_name): (&str, &std::ffi::CStr) = match (dim, is_float) {
            (TexDim::One, false) => (TEX_READ_1D_INT_TEXOBJ_PTX, c"tex_read_1d_int_texobj"),
            (TexDim::One, true) => (TEX_READ_1D_FLOAT_TEXOBJ_PTX, c"tex_read_1d_float_texobj"),
            (TexDim::Two, false) => (TEX_READ_2D_INT_TEXOBJ_PTX, c"tex_read_2d_int_texobj"),
            (TexDim::Two, true) => (TEX_READ_2D_FLOAT_TEXOBJ_PTX, c"tex_read_2d_float_texobj"),
            (TexDim::Three, false) => (TEX_READ_3D_INT_TEXOBJ_PTX, c"tex_read_3d_int_texobj"),
            (TexDim::Three, true) => (TEX_READ_3D_FLOAT_TEXOBJ_PTX, c"tex_read_3d_float_texobj"),
        };

        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void);

        let mut func = std::mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, kernel_name.as_ptr());

        // Allocate output buffer on device (4 x f32 or 4 x s32 = 16 bytes)
        let mut d_output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_output, 16);

        // Texture coordinates for point sampling: pixel center = (p + 0.5)
        let mut coord_x: f32 = px as f32 + 0.5;
        if normalized {
            coord_x /= width as f32;
        }
        let mut coord_y: f32 = py as f32 + 0.5;
        if normalized {
            coord_y /= height as f32;
        }
        let mut coord_z: f32 = pz as f32 + 0.5;
        if normalized {
            coord_z /= depth as f32;
        }
        let mut params: [*mut c_void; 5] = [
            &d_output as *const _ as *mut _,
            &texobj as *const _ as *mut _,
            &coord_x as *const _ as *mut _,
            &coord_y as *const _ as *mut _,
            &coord_z as *const _ as *mut _,
        ];
        api.cuLaunchKernel(
            func,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));

        // Read back result
        let mut result = [0u32; 4];
        api.cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, d_output, 16);

        // Verify
        let fmt_name = format!("{:?}", fmt);
        let dim_name = match dim {
            TexDim::One => "1d",
            TexDim::Two => "2d",
            TexDim::Three => "3d",
        };
        assert_eq!(
            result[..num_channels as usize],
            expected[..num_channels as usize],
            "texobj mismatch for dim={dim_name}, format={fmt_name}, channels={num_channels}, \
             size={width}x{effective_height}x{effective_depth}, pixel=({px},{py},{pz}), normalized={normalized}, is_float={is_float}"
        );

        // Cleanup
        api.cuMemFree_v2(d_output);
        api.cuTexObjectDestroy(texobj);
        api.cuModuleUnload(module);
        api.cuArrayDestroy(array);
    }

    #[test_cuda]
    unsafe fn texobj_formats_channels_dimensions(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);

        let formats = [
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_HALF,
            CUarray_format_enum::CU_AD_FORMAT_FLOAT,
        ];
        let channel_counts: [u32; 3] = [1, 2, 4];

        // 1D tests
        let widths_1d: [usize; 4] = [1, 4, 16, 64];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &w in &widths_1d {
                    for normalized in [false, true] {
                        texobj_read_test(&api, normalized, TexDim::One, fmt, num_ch, w, 1, 1);
                    }
                }
            }
        }

        // 2D tests
        let dimensions_2d: [(usize, usize); 4] = [(1, 1), (4, 4), (16, 16), (64, 37)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h) in &dimensions_2d {
                    for normalized in [false, true] {
                        texobj_read_test(&api, normalized, TexDim::Two, fmt, num_ch, w, h, 1);
                    }
                }
            }
        }

        // 3D tests
        let dimensions_3d: [(usize, usize, usize); 4] =
            [(1, 1, 1), (4, 4, 4), (8, 8, 8), (16, 13, 7)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h, d) in &dimensions_3d {
                    for normalized in [false, true] {
                        texobj_read_test(&api, normalized, TexDim::Three, fmt, num_ch, w, h, d);
                    }
                }
            }
        }

        api.cuCtxDestroy_v2(ctx);
    }

    /// Core test logic for s32 coordinates with texobj: create array, upload
    /// data, create texobj, launch kernel with integer coordinates, read back
    /// and verify.
    unsafe fn texobj_read_s32coord_test(
        api: &impl CudaApi,
        dim: TexDim,
        fmt: CUarray_format,
        num_channels: u32,
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let elem_size = format_byte_size(fmt);
        let pixel_size = elem_size * num_channels as usize;
        let row_bytes = width * pixel_size;

        let (effective_height, effective_depth) = match dim {
            TexDim::One => (1, 1),
            TexDim::Two => (height, 1),
            TexDim::Three => (height, depth),
        };
        let total_bytes = row_bytes * effective_height * effective_depth;

        let px = (width / 2).min(width - 1);
        let py = (effective_height / 2).min(effective_height - 1);
        let pz = (effective_depth / 2).min(effective_depth - 1);

        let pixel_offset = pz * (row_bytes * effective_height) + py * row_bytes + px * pixel_size;

        let mut host_buf = vec![0u8; total_bytes];
        let expected = write_test_pixel(&mut host_buf, pixel_offset, fmt, num_channels);

        // Create CUDA array and upload data
        let mut array: CUarray = std::mem::zeroed();
        let mut dev_ptr: CUdeviceptr_v2 = std::mem::zeroed();
        match dim {
            TexDim::One => {
                api.cuMemAlloc_v2(&mut dev_ptr, total_bytes);
                api.cuMemcpyHtoD_v2(dev_ptr, host_buf.as_ptr() as *const c_void, total_bytes);
            }
            TexDim::Two => {
                let desc = CUDA_ARRAY_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Format: fmt,
                    NumChannels: num_channels,
                };
                api.cuArrayCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    dstPitch: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                };
                api.cuMemcpy2D_v2(&copy_params);
            }
            TexDim::Three => {
                let desc = CUDA_ARRAY3D_DESCRIPTOR {
                    Width: width,
                    Height: effective_height,
                    Depth: effective_depth,
                    Format: fmt,
                    NumChannels: num_channels,
                    Flags: 0,
                };
                api.cuArray3DCreate_v2(&mut array, &desc);
                let copy_params = CUDA_MEMCPY3D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcZ: 0,
                    srcLOD: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host_buf.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    reserved0: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    srcHeight: effective_height,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstZ: 0,
                    dstLOD: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    reserved1: std::ptr::null_mut(),
                    dstPitch: 0,
                    dstHeight: 0,
                    WidthInBytes: row_bytes,
                    Height: effective_height,
                    Depth: effective_depth,
                };
                api.cuMemcpy3D_v2(&copy_params);
            }
        }

        // Create texture object
        let res_desc = if dim == TexDim::One {
            CUDA_RESOURCE_DESC {
                resType: CUresourcetype_enum::CU_RESOURCE_TYPE_LINEAR,
                res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                    linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
                        devPtr: dev_ptr,
                        format: fmt,
                        numChannels: num_channels,
                        sizeInBytes: total_bytes,
                    },
                },
                flags: 0,
            }
        } else {
            CUDA_RESOURCE_DESC {
                resType: CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY,
                res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                    array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 { hArray: array },
                },
                flags: 0,
            }
        };
        let is_float = format_is_float(fmt);
        let flags = if !is_float {
            CU_TRSF_READ_AS_INTEGER
        } else {
            0
        };
        let tex_desc = CUDA_TEXTURE_DESC {
            addressMode: [CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP; 3],
            filterMode: CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
            flags,
            maxAnisotropy: 0,
            mipmapFilterMode: CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0; 12],
        };
        let mut texobj: CUtexObject = std::mem::zeroed();
        api.cuTexObjectCreate(&mut texobj, &res_desc, &tex_desc, std::ptr::null());

        // Load PTX module and get kernel function
        let (ptx, kernel_name): (&str, &std::ffi::CStr) = match (dim, is_float) {
            (TexDim::One, false) => (
                TEX_READ_1D_INT_S32COORD_TEXOBJ_PTX,
                c"tex_read_1d_int_s32coord_texobj",
            ),
            (TexDim::One, true) => (
                TEX_READ_1D_FLOAT_S32COORD_TEXOBJ_PTX,
                c"tex_read_1d_float_s32coord_texobj",
            ),
            (TexDim::Two, false) => (
                TEX_READ_2D_INT_S32COORD_TEXOBJ_PTX,
                c"tex_read_2d_int_s32coord_texobj",
            ),
            (TexDim::Two, true) => (
                TEX_READ_2D_FLOAT_S32COORD_TEXOBJ_PTX,
                c"tex_read_2d_float_s32coord_texobj",
            ),
            (TexDim::Three, false) => (
                TEX_READ_3D_INT_S32COORD_TEXOBJ_PTX,
                c"tex_read_3d_int_s32coord_texobj",
            ),
            (TexDim::Three, true) => (
                TEX_READ_3D_FLOAT_S32COORD_TEXOBJ_PTX,
                c"tex_read_3d_float_s32coord_texobj",
            ),
        };

        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void);

        let mut func = std::mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, kernel_name.as_ptr());

        // Allocate output buffer on device (4 x f32 or 4 x s32 = 16 bytes)
        let mut d_output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_output, 16);

        // Integer coordinates: direct pixel index
        let coord_x: i32 = px as i32;
        let coord_y: i32 = py as i32;
        let coord_z: i32 = pz as i32;
        let mut params: [*mut c_void; 5] = [
            &d_output as *const _ as *mut _,
            &texobj as *const _ as *mut _,
            &coord_x as *const _ as *mut _,
            &coord_y as *const _ as *mut _,
            &coord_z as *const _ as *mut _,
        ];
        api.cuLaunchKernel(
            func,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));

        // Read back result
        let mut result = [0u32; 4];
        api.cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, d_output, 16);

        // Verify
        let fmt_name = format!("{:?}", fmt);
        let dim_name = match dim {
            TexDim::One => "1d",
            TexDim::Two => "2d",
            TexDim::Three => "3d",
        };
        assert_eq!(
            result[..num_channels as usize],
            expected[..num_channels as usize],
            "texobj s32coord mismatch for dim={dim_name}, format={fmt_name}, channels={num_channels}, \
             size={width}x{effective_height}x{effective_depth}, pixel=({px},{py},{pz}), is_float={is_float}"
        );

        // Cleanup
        api.cuMemFree_v2(d_output);
        api.cuTexObjectDestroy(texobj);
        api.cuModuleUnload(module);
        if !array.is_null() {
            api.cuArrayDestroy(array);
        }
        if !dev_ptr.0.is_null() {
            api.cuMemFree_v2(dev_ptr);
        }
    }

    #[test_cuda]
    unsafe fn texobj_s32coord_formats_channels_dimensions(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);

        let formats = [
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_HALF,
            CUarray_format_enum::CU_AD_FORMAT_FLOAT,
        ];
        let channel_counts: [u32; 3] = [1, 2, 4];

        // 1D tests
        let widths_1d: [usize; 4] = [1, 4, 16, 64];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &w in &widths_1d {
                    texobj_read_s32coord_test(&api, TexDim::One, fmt, num_ch, w, 1, 1);
                }
            }
        }

        // 2D tests
        let dimensions_2d: [(usize, usize); 4] = [(1, 1), (4, 4), (16, 16), (64, 37)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h) in &dimensions_2d {
                    texobj_read_s32coord_test(&api, TexDim::Two, fmt, num_ch, w, h, 1);
                }
            }
        }

        // 3D tests
        let dimensions_3d: [(usize, usize, usize); 4] =
            [(1, 1, 1), (4, 4, 4), (8, 8, 8), (16, 13, 7)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h, d) in &dimensions_3d {
                    texobj_read_s32coord_test(&api, TexDim::Three, fmt, num_ch, w, h, d);
                }
            }
        }

        api.cuCtxDestroy_v2(ctx);
    }

    #[test_cuda]
    unsafe fn texref_set_address_accepts_zero_bytes(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(
            &mut module,
            TEX_READ_1D_INT_S32COORD_PTX.as_ptr() as *const c_void,
        );
        let mut texref = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texref, module, c"tex".as_ptr());
        api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32, 1);
        let mut buffer = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut buffer, 16);
        let mut offset = usize::MAX;
        api.cuTexRefSetAddress_v2(&mut offset, texref, buffer, 0);
        assert_eq!(offset, 0);
        api.cuMemFree_v2(buffer);
        api.cuModuleUnload(module);
        api.cuCtxDestroy_v2(ctx);
    }

    #[test_cuda]
    unsafe fn texref_set_array_accepts_zero_flags(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let desc = CUDA_ARRAY_DESCRIPTOR {
            Width: 4,
            Height: 0,
            Format: CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            NumChannels: 1,
        };
        let mut array = std::mem::zeroed();
        api.cuArrayCreate_v2(&mut array, &desc);
        let texels = [3i32, 5, 7, 11];
        api.cuMemcpyHtoA_v2(array, 0, texels.as_ptr() as *const c_void, 16);
        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, TEX_READ_1D_INT_PTX.as_ptr() as *const c_void);
        let mut func = std::mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, c"tex_read_1d_int".as_ptr());
        let mut texref = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texref, module, c"tex".as_ptr());
        api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32, 1);
        api.cuTexRefSetFlags(texref, CU_TRSF_READ_AS_INTEGER);
        api.cuTexRefSetArray(texref, array, 0);
        let mut d_output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_output, 16);
        let coord_x = 2.5f32;
        let mut params: [*mut c_void; 2] = [
            &d_output as *const _ as *mut _,
            &coord_x as *const _ as *mut _,
        ];
        api.cuLaunchKernel(
            func,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));
        let mut result = [0i32; 4];
        api.cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, d_output, 16);
        assert_eq!(result[0], 7);
        api.cuMemFree_v2(d_output);
        api.cuModuleUnload(module);
        api.cuArrayDestroy(array);
        api.cuCtxDestroy_v2(ctx);
    }

    #[test_cuda]
    unsafe fn texobj_create_ignores_anisotropy_and_mip_bias(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let desc = CUDA_ARRAY_DESCRIPTOR {
            Width: 4,
            Height: 0,
            Format: CUarray_format_enum::CU_AD_FORMAT_FLOAT,
            NumChannels: 1,
        };
        let mut array = std::mem::zeroed();
        api.cuArrayCreate_v2(&mut array, &desc);
        let texels = [0.25f32, 0.5, 0.75, 1.0];
        api.cuMemcpyHtoA_v2(array, 0, texels.as_ptr() as *const c_void, 16);
        let mut res_desc: CUDA_RESOURCE_DESC = std::mem::zeroed();
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;
        let mut tex_desc: CUDA_TEXTURE_DESC = std::mem::zeroed();
        tex_desc.maxAnisotropy = 4;
        tex_desc.mipmapLevelBias = 0.5;
        let mut texobj = std::mem::zeroed();
        api.cuTexObjectCreate(&mut texobj, &res_desc, &tex_desc, std::ptr::null());
        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(
            &mut module,
            TEX_READ_1D_FLOAT_TEXOBJ_PTX.as_ptr() as *const c_void,
        );
        let mut func = std::mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, c"tex_read_1d_float_texobj".as_ptr());
        let mut d_output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_output, 16);
        let coord_x = 2.5f32;
        let mut params: [*mut c_void; 3] = [
            &d_output as *const _ as *mut _,
            &texobj as *const _ as *mut _,
            &coord_x as *const _ as *mut _,
        ];
        api.cuLaunchKernel(
            func,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));
        let mut result = [0f32; 4];
        api.cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, d_output, 16);
        assert_eq!(result[0], 0.75);
        api.cuMemFree_v2(d_output);
        api.cuModuleUnload(module);
        api.cuTexObjectDestroy(texobj);
        api.cuArrayDestroy(array);
        api.cuCtxDestroy_v2(ctx);
    }

    // HALF arrays keep their format in HIP, so legacy binds and the rebuilds
    // after each setter must keep reading signed half values.
    #[test_cuda]
    unsafe fn texref_half4_array_reads_across_rebinds(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let ptx = concat!(include_str!("test_ptx/texref_half4_rebind.ptx"), "\0");
        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void);
        let mut texref = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texref, module, c"legacy_half".as_ptr());
        let mut func = std::mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, c"sample_half".as_ptr());
        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 4,
            Height: 4,
            Depth: 2,
            Format: CUarray_format_enum::CU_AD_FORMAT_HALF,
            NumChannels: 4,
            Flags: 0,
        };
        let mut array = std::mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &desc);
        // HALF encodings of 0.5, 1, 2 and -3 in every texel.
        let pixels = [0x3800u16, 0x3c00, 0x4000, 0xc200].repeat(32);
        let mut copy: CUDA_MEMCPY3D = std::mem::zeroed();
        copy.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
        copy.srcHost = pixels.as_ptr() as *const c_void;
        copy.srcPitch = 32;
        copy.srcHeight = 4;
        copy.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        copy.WidthInBytes = 32;
        copy.Height = 4;
        copy.Depth = 2;
        api.cuMemcpy3D_v2(&copy);
        api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_HALF, 4);
        api.cuTexRefSetFlags(texref, 0);
        api.cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);
        let mut d_output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_output, 16);
        for state in 0..6 {
            match state {
                0 => {}
                1 => api.cuTexRefSetFilterMode(texref, CUfilter_mode_enum::CU_TR_FILTER_MODE_LINEAR),
                2 => api.cuTexRefSetAddressMode(texref, 0, CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP),
                3 => api.cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES),
                4 => api.cuTexRefSetFlags(texref, 0),
                5 => api.cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT),
                _ => unreachable!(),
            }
            let mut params: [*mut c_void; 1] = [&d_output as *const _ as *mut _];
            api.cuLaunchKernel(
                func,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                CUstream(std::ptr::null_mut()),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));
            let mut result = [0f32; 4];
            api.cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, d_output, 16);
            assert_eq!(result, [0.5, 1.0, 2.0, -3.0], "after setter state {state}");
        }
        api.cuMemFree_v2(d_output);
        api.cuModuleUnload(module);
        api.cuArrayDestroy(array);
        api.cuCtxDestroy_v2(ctx);
    }

    unsafe fn read_linear_texel(api: &impl CudaApi, function: CUfunction, output: CUdeviceptr) -> [u32; 4] {
        let mut output = output;
        let mut params = [&mut output as *mut CUdeviceptr as *mut c_void];
        api.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
            CUstream(std::ptr::null_mut()), params.as_mut_ptr(), std::ptr::null_mut());
        api.cuCtxSynchronize();
        let mut actual = [0u32; 4];
        api.cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, 16);
        actual
    }

    // Validate device reads across repeated setters, a real format change,
    // and a new resource. Descriptor equality alone cannot catch a stale
    // device-global texture handle or a skipped format rebuild.
    #[test_cuda]
    unsafe fn legacy_linear_repeated_and_changed_state(api: impl CudaApi) {
        api.cuInit(0);
        let mut context = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut context, 0, 0);
        let ptx = concat!(include_str!("test_ptx/texref_linear_reconfigure.ptx"), "\0");
        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, ptx.as_ptr().cast());
        let mut texref = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texref, module, c"legacy_linear".as_ptr());
        // Setters must also work before binding a resource. Restore the
        // non-mipmapped settings before testing reads through a linear binding.
        api.cuTexRefSetMipmapLevelBias(texref, 0.5);
        api.cuTexRefSetMipmapLevelBias(texref, 0.0);
        api.cuTexRefSetMipmapLevelClamp(texref, 1.0, 2.0);
        api.cuTexRefSetMipmapLevelClamp(texref, 0.0, 0.0);
        api.cuTexRefSetMaxAnisotropy(texref, 4);
        api.cuTexRefSetMaxAnisotropy(texref, 0);
        let mut function = std::mem::zeroed();
        api.cuModuleGetFunction(&mut function, module, c"sample_linear".as_ptr());
        let mut source = std::mem::zeroed();
        let mut output = std::mem::zeroed();
        let mut alternate = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut source, 16);
        api.cuMemAlloc_v2(&mut output, 16);
        api.cuMemAlloc_v2(&mut alternate, 16);
        let first = [7u32, 11, 13, 17];
        api.cuMemcpyHtoD_v2(source, first.as_ptr().cast(), 16);
        api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32, 4);
        api.cuTexRefSetFlags(texref, CU_TRSF_READ_AS_INTEGER);
        api.cuTexRefSetAddress_v2(&mut 0, texref, source, 16);
        for state in 0..17 {
            match state {
                0 => {},
                1 => api.cuTexRefSetFilterMode(texref, CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT),
                2 => api.cuTexRefSetAddressMode(texref, 0, CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP),
                3 => api.cuTexRefSetFlags(texref, CU_TRSF_READ_AS_INTEGER),
                4 => api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16, 4),
                5 => api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32, 4),
                6 => {
                    let second = [19u32, 23, 29, 31];
                    api.cuMemcpyHtoD_v2(alternate, second.as_ptr().cast(), 16);
                    api.cuTexRefSetAddress_v2(&mut 0, texref, alternate, 16);
                },
                7 => api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32, 4),
                8 | 10 => api.cuTexRefSetMipmapFilterMode(texref, CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT),
                9 => api.cuTexRefSetMipmapFilterMode(texref, CUfilter_mode_enum::CU_TR_FILTER_MODE_LINEAR),
                11 | 14 => api.cuTexRefSetMipmapLevelBias(texref, 0.0),
                12 | 15 => api.cuTexRefSetMipmapLevelClamp(texref, 0.0, 0.0),
                13 | 16 => api.cuTexRefSetMaxAnisotropy(texref, 0),
                _ => unreachable!(),
            }
            let actual = read_linear_texel(&api, function, output);
            let expected = match state {
                4 => [7, 0, 11, 0],
                6..=16 => [19, 23, 29, 31],
                _ => first,
            };
            assert_eq!(actual, expected, "linear texref state {state}");
        }
        api.cuModuleUnload(module);
        api.cuMemFree_v2(source);
        api.cuMemFree_v2(output);
        api.cuMemFree_v2(alternate);
        api.cuCtxDestroy_v2(context);
    }

    // HIP destroys the bound object before rebuilding it and rejects the
    // rebuild for three channels. The failed setter must leave the previous
    // state bound, and later setters must still read the bound buffer.
    #[test_cuda]
    unsafe fn legacy_linear_recovers_after_failed_rebuild(api: impl CudaApi) {
        api.cuInit(0);
        let mut context = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut context, 0, 0);
        let ptx = concat!(include_str!("test_ptx/texref_linear_reconfigure.ptx"), "\0");
        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, ptx.as_ptr().cast());
        let mut texref = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texref, module, c"legacy_linear".as_ptr());
        let mut function = std::mem::zeroed();
        api.cuModuleGetFunction(&mut function, module, c"sample_linear".as_ptr());
        let mut source = std::mem::zeroed();
        let mut output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut source, 16);
        api.cuMemAlloc_v2(&mut output, 16);
        let texel = [7u32, 11, 13, 17];
        api.cuMemcpyHtoD_v2(source, texel.as_ptr().cast(), 16);
        api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32, 4);
        api.cuTexRefSetFlags(texref, CU_TRSF_READ_AS_INTEGER);
        api.cuTexRefSetAddress_v2(&mut 0, texref, source, 16);
        assert_eq!(read_linear_texel(&api, function, output), texel, "before");
        let failed = api.cuTexRefSetFormat_unchecked(
            texref,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            3,
        );
        eprintln!("three-channel cuTexRefSetFormat: {failed:?}");
        // A rejected setter must leave the previous format and buffer bound.
        if failed.is_err() {
            assert_eq!(read_linear_texel(&api, function, output), texel, "after failed rebuild");
        }
        api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32, 4);
        assert_eq!(read_linear_texel(&api, function, output), texel, "restored format");
        api.cuTexRefSetFormat(texref, CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32, 4);
        assert_eq!(read_linear_texel(&api, function, output), texel, "repeated format");
        api.cuModuleUnload(module);
        api.cuMemFree_v2(source);
        api.cuMemFree_v2(output);
        api.cuCtxDestroy_v2(context);
    }

    /*
     *  ███████╗██╗   ██╗██████╗ ███████╗ █████╗  ██████╗███████╗███████╗
     *  ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝
     *  ███████╗██║   ██║██████╔╝█████╗  ███████║██║     █████╗  ███████╗
     *  ╚════██║██║   ██║██╔══██╗██╔══╝  ██╔══██║██║     ██╔══╝  ╚════██║
     *  ███████║╚██████╔╝██║  ██║██║     ██║  ██║╚██████╗███████╗███████║
     *  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝
     */

    /// How a kernel refers to its surface.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum SurfBinding {
        /// `.global .surfref surf;` bound with `cuSurfRefSetArray`
        Reference,
        /// `.param .u64 surfobj` created with `cuSurfObjectCreate`
        Object,
    }

    impl SurfBinding {
        fn name(self) -> &'static str {
            match self {
                SurfBinding::Reference => "surfref",
                SurfBinding::Object => "surfobj",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum SurfOp {
        /// `suld.b`
        Load,
        /// `sust.b`
        Store,
    }

    impl SurfOp {
        fn name(self) -> &'static str {
            match self {
                SurfOp::Load => "load",
                SurfOp::Store => "store",
            }
        }
    }

    /// Data type and vector width of an unformatted `suld.b`/`sust.b` access.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct SurfAccess {
        /// Vector width: 1, 2 or 4
        vec: u32,
        /// Element width in bits: 8, 16, 32 or 64
        bits: u32,
    }

    impl SurfAccess {
        /// Every `.vec.dtype` combination accepted by ptxas. `.v4.b64` is
        /// not on the list: 256-bit surface accesses are rejected.
        const ALL: [SurfAccess; 11] = [
            SurfAccess { vec: 1, bits: 8 },
            SurfAccess { vec: 1, bits: 16 },
            SurfAccess { vec: 1, bits: 32 },
            SurfAccess { vec: 1, bits: 64 },
            SurfAccess { vec: 2, bits: 8 },
            SurfAccess { vec: 2, bits: 16 },
            SurfAccess { vec: 2, bits: 32 },
            SurfAccess { vec: 2, bits: 64 },
            SurfAccess { vec: 4, bits: 8 },
            SurfAccess { vec: 4, bits: 16 },
            SurfAccess { vec: 4, bits: 32 },
        ];

        /// Number of bytes moved by a single access
        fn byte_size(self) -> usize {
            (self.vec * self.bits / 8) as usize
        }

        /// `.b32`, `.v4.b8`, ...
        fn ptx_suffix(self) -> String {
            if self.vec == 1 {
                format!(".b{}", self.bits)
            } else {
                format!(".v{}.b{}", self.vec, self.bits)
            }
        }

        /// `b32`, `v4_b8`, ...
        fn name(self) -> String {
            if self.vec == 1 {
                format!("b{}", self.bits)
            } else {
                format!("v{}_b{}", self.vec, self.bits)
            }
        }

        /// Data operand of the surface instruction, e.g. `{%rs0, %rs1}`.
        /// 8- and 16-bit values live in 16-bit registers like nvcc does it.
        fn data_regs(self) -> String {
            let (prefix, base) = match self.bits {
                8 | 16 => ("rs", 0),
                32 => ("r", 10),
                64 => ("rd", 4),
                _ => unreachable!(),
            };
            let regs = (0..self.vec)
                .map(|i| format!("%{prefix}{}", base + i))
                .collect::<Vec<_>>();
            if self.vec == 1 {
                regs.into_iter().next().unwrap()
            } else {
                format!("{{{}}}", regs.join(", "))
            }
        }
    }

    fn surf_kernel_name(op: SurfOp, access: SurfAccess) -> String {
        format!("surf_{}_{}", op.name(), access.name())
    }

    /// Generates a kernel that copies a whole surface to or from a linear
    /// buffer with one `suld.b`/`sust.b` per thread. Thread `tid.x` of block
    /// (y, z) handles the chunk at byte `x = tid.x * access_size` of row `y`
    /// (`ctaid.y`) in slice `z` (`ctaid.z`); the matching location in `ptr`
    /// is `((z * nctaid.y) + y) * row_bytes + x`. Launch with a block of
    /// `row_bytes / access_size` threads and a grid of (1, height, depth).
    /// Loads copy surface -> `ptr`, stores copy `ptr` -> surface.
    fn surf_kernel_ptx(
        op: SurfOp,
        dim: TexDim,
        access: SurfAccess,
        binding: SurfBinding,
        pixel_size: usize,
    ) -> String {
        let kernel_name = surf_kernel_name(op, access);
        let suffix = access.ptx_suffix();
        let regs = access.data_regs();
        let (surfref_decl, surfobj_param, surfobj_load, surf_operand) = match binding {
            SurfBinding::Reference => (".global .surfref surf;\n", "", "", "surf"),
            SurfBinding::Object => (
                "",
                "    .param .u64 surfobj,\n",
                "    ld.param.u64 %rd3, [surfobj];\n",
                "%rd3",
            ),
        };
        let geom = match dim {
            TexDim::One => "1d",
            TexDim::Two => "2d",
            TexDim::Three => "3d",
        };
        // 3D coordinates are a 4-element vector whose last element is ignored
        let coords = match dim {
            TexDim::One => "{%r1}",
            TexDim::Two => "{%r1, %r2}",
            TexDim::Three => "{%r1, %r2, %r3, %r3}",
        };
        let body = match op {
            SurfOp::Load => format!(
                "    suld.b.{geom}{suffix}.trap {regs}, [{surf_operand}, {coords}];\n    \
                 st.global{suffix} [%rd1], {regs};\n"
            ),
            SurfOp::Store => format!(
                "    ld.global{suffix} {regs}, [%rd1];\n    \
                 sust.b.{geom}{suffix}.trap [{surf_operand}, {coords}], {regs};\n"
            ),
        };
        format!(
            ".version 6.5\n\
             .target sm_30\n\
             .address_size 64\n\
             {surfref_decl}\
             .visible .entry {kernel_name}(\n    \
                 .param .u64 ptr,\n\
             {surfobj_param}    \
                 .param .u32 row_bytes\n\
             ) {{\n    \
                 .reg .b16 %rs<4>;\n    \
                 .reg .b32 %r<16>;\n    \
                 .reg .b64 %rd<8>;\n    \
                 ld.param.u64 %rd1, [ptr];\n\
             {surfobj_load}    \
                 ld.param.u32 %r6, [row_bytes];\n    \
                 mov.u32 %r1, %tid.x;\n    \
                 mul.lo.u32 %r1, %r1, {pixel_size};\n    \
                 mov.u32 %r2, %ctaid.y;\n    \
                 mov.u32 %r3, %ctaid.z;\n    \
                 mov.u32 %r4, %nctaid.y;\n    \
                 mad.lo.u32 %r5, %r3, %r4, %r2;\n    \
                 mad.lo.u32 %r5, %r5, %r6, %r1;\n    \
                 cvt.u64.u32 %rd2, %r5;\n    \
                 add.s64 %rd1, %rd1, %rd2;\n\
             {body}    \
                 ret;\n\
             }}\n\0"
        )
    }

    /// Surface kernels loaded on first use, one module per (op, dim, access)
    struct SurfKernels {
        binding: SurfBinding,
        modules: HashMap<(SurfOp, TexDim, SurfAccess, usize), (CUmodule, CUfunction)>,
    }

    impl SurfKernels {
        fn new(binding: SurfBinding) -> Self {
            Self {
                binding,
                modules: HashMap::new(),
            }
        }

        unsafe fn get(
            &mut self,
            api: &impl CudaApi,
            op: SurfOp,
            dim: TexDim,
            access: SurfAccess,
            pixel_size: usize,
        ) -> (CUmodule, CUfunction) {
            let binding = self.binding;
            *self
                .modules
                .entry((op, dim, access, pixel_size))
                .or_insert_with(|| {
                    let ptx = surf_kernel_ptx(op, dim, access, binding, pixel_size);
                    let mut module = std::mem::zeroed();
                    api.cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void);
                    let kernel_name = CString::new(surf_kernel_name(op, access)).unwrap();
                    let mut func = std::mem::zeroed();
                    api.cuModuleGetFunction(&mut func, module, kernel_name.as_ptr());
                    (module, func)
                })
        }

        unsafe fn unload(self, api: &impl CudaApi) {
            for (module, _) in self.modules.into_values() {
                api.cuModuleUnload(module);
            }
        }
    }

    /// Deterministic fill byte for byte `offset` of a surface. `salt` makes
    /// the fill differ between rounds so that stale contents from a previous
    /// round can't pass. Odd bytes are kept in 0x40..=0x7B so that any 16- or
    /// 32-bit float formed from the pattern is a positive normal number: it
    /// can't be changed by NaN quieting or denormal flushing should an
    /// implementation convert on the way.
    fn pattern_byte(offset: usize, salt: u32) -> u8 {
        let hash = ((offset as u32) ^ salt.wrapping_mul(0x0101_0101)).wrapping_mul(0x9E37_79B1);
        let byte = (hash >> 24) as u8;
        if offset % 2 == 1 {
            0x40 + byte % 0x3C
        } else {
            byte
        }
    }

    // Filler for bytes not touched by a surface access: 00, 01, 02, ...
    fn filler_byte(offset: usize) -> u8 {
        offset as u8
    }

    unsafe fn surf_upload(
        api: &impl CudaApi,
        dim: TexDim,
        array: CUarray,
        host: &[u8],
        row_bytes: usize,
        height: usize,
        depth: usize,
    ) {
        match dim {
            TexDim::One => {
                api.cuMemcpyHtoA_v2(array, 0, host.as_ptr() as *const c_void, host.len());
            }
            TexDim::Two => {
                let copy_params = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    dstPitch: 0,
                    WidthInBytes: row_bytes,
                    Height: height,
                };
                api.cuMemcpy2D_v2(&copy_params);
            }
            TexDim::Three => {
                let copy_params = CUDA_MEMCPY3D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcZ: 0,
                    srcLOD: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: host.as_ptr() as *const c_void,
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: std::ptr::null_mut(),
                    reserved0: std::ptr::null_mut(),
                    srcPitch: row_bytes,
                    srcHeight: height,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstZ: 0,
                    dstLOD: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: array,
                    reserved1: std::ptr::null_mut(),
                    dstPitch: 0,
                    dstHeight: height,
                    WidthInBytes: row_bytes,
                    Height: height,
                    Depth: depth,
                };
                api.cuMemcpy3D_v2(&copy_params);
            }
        }
    }

    unsafe fn surf_download(
        api: &impl CudaApi,
        dim: TexDim,
        array: CUarray,
        host: &mut [u8],
        row_bytes: usize,
        height: usize,
        depth: usize,
    ) {
        match dim {
            TexDim::One => {
                api.cuMemcpyAtoH_v2(host.as_mut_ptr() as *mut c_void, array, 0, host.len());
            }
            TexDim::Two => {
                let copy_params = CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    srcHost: std::ptr::null(),
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: array,
                    srcPitch: 0,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    dstHost: host.as_mut_ptr() as *mut c_void,
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: std::ptr::null_mut(),
                    dstPitch: row_bytes,
                    WidthInBytes: row_bytes,
                    Height: height,
                };
                api.cuMemcpy2D_v2(&copy_params);
            }
            TexDim::Three => {
                let copy_params = CUDA_MEMCPY3D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcZ: 0,
                    srcLOD: 0,
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    srcHost: std::ptr::null(),
                    srcDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    srcArray: array,
                    reserved0: std::ptr::null_mut(),
                    srcPitch: 0,
                    srcHeight: 0,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstZ: 0,
                    dstLOD: 0,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    dstHost: host.as_mut_ptr() as *mut c_void,
                    dstDevice: CUdeviceptr_v2(std::ptr::null_mut()),
                    dstArray: std::ptr::null_mut(),
                    reserved1: std::ptr::null_mut(),
                    dstPitch: row_bytes,
                    dstHeight: height,
                    WidthInBytes: row_bytes,
                    Height: height,
                    Depth: depth,
                };
                api.cuMemcpy3D_v2(&copy_params);
            }
        }
    }

    /// Points the module's `surf` surfref at `array`. No-op for surface
    /// objects, which are passed as a kernel parameter instead.
    unsafe fn surf_bind(
        api: &impl CudaApi,
        binding: SurfBinding,
        module: CUmodule,
        array: CUarray,
    ) {
        if binding == SurfBinding::Reference {
            let mut surfref = std::mem::zeroed();
            api.cuModuleGetSurfRef(&mut surfref, module, c"surf".as_ptr());
            api.cuSurfRefSetArray(surfref, array, 0);
        }
    }

    /// Runs a kernel from `surf_kernel_ptx` over the whole surface: one
    /// block per row, one thread per `access_size` bytes of the row
    unsafe fn surf_launch(
        api: &impl CudaApi,
        func: CUfunction,
        binding: SurfBinding,
        surfobj: CUsurfObject,
        d_buf: CUdeviceptr,
        row_bytes: usize,
        pixel_size: usize,
        height: usize,
        depth: usize,
    ) {
        let row_bytes_param = row_bytes as u32;
        let mut params: Vec<*mut c_void> = vec![&d_buf as *const _ as *mut _];
        if binding == SurfBinding::Object {
            params.push(&surfobj as *const _ as *mut _);
        }
        params.push(&row_bytes_param as *const _ as *mut _);
        api.cuLaunchKernel(
            func,
            1,
            height as u32,
            depth as u32,
            (row_bytes / pixel_size) as u32,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));
    }

    /// Compares a surface-sized byte buffer with its expected contents and
    /// panics on the first difference, located as (x in bytes, y, z)
    fn check_surface_bytes(
        actual: &[u8],
        expected: &[u8],
        row_bytes: usize,
        slice_bytes: usize,
        pixel_size: usize,
        context: &str,
    ) {
        if let Some(first) = actual.iter().zip(expected).position(|(a, e)| a != e) {
            let z = first / slice_bytes;
            let y = first % slice_bytes / row_bytes;
            let x = first % row_bytes;
            let pixel_start = (first / pixel_size) * pixel_size;
            let pixel = pixel_start..(pixel_start + pixel_size).min(expected.len());
            let pixel_end = pixel.end;
            panic!(
                "{context}: first mismatch at byte {first} (x={x}, y={y}, z={z}), \
                 pixel bytes {pixel_start}..{pixel_end}: expected {:02x?}, got {:02x?}",
                &expected[pixel.clone()],
                &actual[pixel]
            );
        }
    }

    /// Core surface test: create an array and a surface over it, then for
    /// every write access type fill the whole surface with `sust.b` (checked
    /// against a memcpy readback) and read all of it back with `suld.b` using
    /// every read access type.
    unsafe fn surf_read_write_test(
        api: &impl CudaApi,
        kernels: &mut SurfKernels,
        dim: TexDim,
        fmt: CUarray_format,
        num_channels: u32,
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let binding = kernels.binding;
        let elem_size = format_byte_size(fmt);
        let pixel_size = elem_size * num_channels as usize;
        let row_bytes = width * pixel_size;

        let (effective_height, effective_depth) = match dim {
            TexDim::One => (1, 1),
            TexDim::Two => (height, 1),
            TexDim::Three => (height, depth),
        };
        let slice_bytes = row_bytes * effective_height;
        let total_bytes = slice_bytes * effective_depth;

        // Surface load/store must be requested at array creation. 1D and 2D
        // arrays are created through the 3D entry point with zero height/depth
        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: width,
            Height: if dim == TexDim::One {
                0
            } else {
                effective_height
            },
            Depth: if dim == TexDim::Three {
                effective_depth
            } else {
                0
            },
            Format: fmt,
            NumChannels: num_channels,
            Flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mut array: CUarray = std::mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &desc);

        let mut surfobj: CUsurfObject = std::mem::zeroed();
        if binding == SurfBinding::Object {
            let res_desc = CUDA_RESOURCE_DESC {
                resType: CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY,
                res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                    array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 { hArray: array },
                },
                flags: 0,
            };
            api.cuSurfObjectCreate(&mut surfobj, &res_desc);
        }

        // Linear copy of the surface: source of sust data, destination of
        // suld data
        let mut d_buf: CUdeviceptr = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut d_buf, total_bytes);

        let dim_name = match dim {
            TexDim::One => "1d",
            TexDim::Two => "2d",
            TexDim::Three => "3d",
        };
        let surface_name = format!(
            "{} dim={dim_name}, format={:?}, channels={num_channels}, \
             size={width}x{effective_height}x{effective_depth}",
            binding.name(),
            fmt
        );

        let accesses = SurfAccess::ALL
            .into_iter()
            .filter(|access| access.byte_size() <= pixel_size)
            .collect::<Vec<_>>();

        let filler = (0..total_bytes).map(filler_byte).collect::<Vec<_>>();
        let mut actual = vec![0u8; total_bytes];
        for (round, &write) in accesses.iter().enumerate() {
            fn surface_pattern(total_bytes: usize, pixel_size: usize, salt: u32) -> Vec<u8> {
                (0..total_bytes)
                    .map(|offset| {
                        if offset % pixel_size == 0 {
                            pattern_byte(offset, salt)
                        } else {
                            filler_byte(offset)
                        }
                    })
                    .collect()
            }

            fn load_pattern(
                surface: &[u8],
                total_bytes: usize,
                pixel_size: usize,
                access_size: usize,
            ) -> Vec<u8> {
                (0..total_bytes)
                    .map(|offset| {
                        let base = (offset / pixel_size) * pixel_size;
                        if offset >= base && offset < base + access_size {
                            surface[offset]
                        } else {
                            filler_byte(offset)
                        }
                    })
                    .collect()
            }

            // Surface accesses are valid only at pixel starts: for a 2-channel
            // u8 pixel, byte 1 is not a valid read/write location.
            let store_pattern = surface_pattern(total_bytes, pixel_size, round as u32);
            surf_upload(
                api,
                dim,
                array,
                &filler,
                row_bytes,
                effective_height,
                effective_depth,
            );
            api.cuMemcpyHtoD_v2(d_buf, store_pattern.as_ptr() as *const c_void, total_bytes);
            let (module, func) = kernels.get(api, SurfOp::Store, dim, write, pixel_size);
            surf_bind(api, binding, module, array);
            surf_launch(
                api,
                func,
                binding,
                surfobj,
                d_buf,
                row_bytes,
                pixel_size,
                effective_height,
                effective_depth,
            );
            surf_download(
                api,
                dim,
                array,
                &mut actual,
                row_bytes,
                effective_height,
                effective_depth,
            );
            check_surface_bytes(
                &actual,
                &store_pattern,
                row_bytes,
                slice_bytes,
                pixel_size,
                &format!("sust.b{} mismatch for {surface_name}", write.ptx_suffix()),
            );

            // Read the surface back through suld with every access type
            for &read in &accesses {
                api.cuMemcpyHtoD_v2(d_buf, filler.as_ptr() as *const c_void, total_bytes);
                let (module, func) = kernels.get(api, SurfOp::Load, dim, read, pixel_size);
                surf_bind(api, binding, module, array);
                surf_launch(
                    api,
                    func,
                    binding,
                    surfobj,
                    d_buf,
                    row_bytes,
                    pixel_size,
                    effective_height,
                    effective_depth,
                );
                api.cuMemcpyDtoH_v2(actual.as_mut_ptr() as *mut c_void, d_buf, total_bytes);
                let load_pattern =
                    load_pattern(&store_pattern, total_bytes, pixel_size, read.byte_size());
                check_surface_bytes(
                    &actual,
                    &load_pattern,
                    row_bytes,
                    slice_bytes,
                    pixel_size,
                    &format!(
                        "suld.b{} after sust.b{} mismatch for {surface_name}",
                        read.ptx_suffix(),
                        write.ptx_suffix()
                    ),
                );
            }
        }

        // Cleanup
        api.cuMemFree_v2(d_buf);
        if binding == SurfBinding::Object {
            api.cuSurfObjectDestroy(surfobj);
        }
        api.cuArrayDestroy(array);
    }

    unsafe fn surf_formats_channels_dimensions(api: &impl CudaApi, binding: SurfBinding) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let mut kernels = SurfKernels::new(binding);

        let formats = [
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
            CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
            CUarray_format_enum::CU_AD_FORMAT_HALF,
            CUarray_format_enum::CU_AD_FORMAT_FLOAT,
        ];
        let channel_counts: [u32; 3] = [1, 2, 4];

        // 1D tests
        let widths_1d: [usize; 4] = [1, 4, 16, 64];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &w in &widths_1d {
                    surf_read_write_test(api, &mut kernels, TexDim::One, fmt, num_ch, w, 1, 1);
                }
            }
        }

        // 2D tests
        let dimensions_2d: [(usize, usize); 4] = [(1, 1), (4, 4), (16, 16), (64, 37)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h) in &dimensions_2d {
                    surf_read_write_test(api, &mut kernels, TexDim::Two, fmt, num_ch, w, h, 1);
                }
            }
        }

        // 3D tests
        let dimensions_3d: [(usize, usize, usize); 4] =
            [(1, 1, 1), (4, 4, 4), (8, 8, 8), (16, 13, 7)];
        for &fmt in &formats {
            for &num_ch in &channel_counts {
                for &(w, h, d) in &dimensions_3d {
                    surf_read_write_test(api, &mut kernels, TexDim::Three, fmt, num_ch, w, h, d);
                }
            }
        }

        kernels.unload(api);
        api.cuCtxDestroy_v2(ctx);
    }

    #[test_cuda]
    unsafe fn surfref_formats_channels_dimensions(api: impl CudaApi) {
        surf_formats_channels_dimensions(&api, SurfBinding::Reference);
    }

    #[test_cuda]
    unsafe fn surfobj_formats_channels_dimensions(api: impl CudaApi) {
        surf_formats_channels_dimensions(&api, SurfBinding::Object);
    }
}

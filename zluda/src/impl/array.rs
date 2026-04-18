use std::{mem, ptr};

use super::hipfix;
use cuda_types::cuda::*;
use hip_runtime_sys::*;

fn format_byte_size(fmt: hipArray_Format) -> usize {
    match fmt {
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT8
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT8 => 1,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_HALF => 2,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_FLOAT => 4,
        _ => 4,
    }
}

pub(crate) unsafe fn create_v2(
    p_handle: &mut CUarray,
    p_allocate_array: &CUDA_ARRAY_DESCRIPTOR,
) -> CUresult {
    let mut desc =
        *(p_allocate_array as *const CUDA_ARRAY_DESCRIPTOR as *const HIP_ARRAY_DESCRIPTOR);
    let (hack_flag, new_format) = hipfix::get_non_broken_format(desc.Format);
    desc.Format = new_format;
    let vram_before = super::memory::vram_used_bytes();
    let mut hip_array = ptr::null_mut();
    hipArrayCreate(&mut hip_array, &desc)?;
    let elem_size = format_byte_size(desc.Format) * desc.NumChannels as usize;
    let size = desc.Width as usize * desc.Height.max(1) as usize * elem_size;
    super::memory::log_vram_delta(
        "hipArrayCreate",
        vram_before,
        super::memory::vram_used_bytes(),
        size,
    );
    // Record the format-transformation hack so cuTexObjectCreate can build a
    // HIP_RESOURCE_VIEW_DESC that re-interprets the bytes as the original format.
    if hack_flag != 0 {
        hipfix::set_array_hack(
            hip_array,
            hipfix::ArrayHack {
                hack_flag,
                num_channels: desc.NumChannels,
                width: desc.Width as usize,
                height: desc.Height as usize,
                depth: 0,
            },
        );
    }
    super::driver::zluda_log_err(&format!(
        "ARRAY_CREATE {}x{} fmt={} ch={} ~{} KiB ptr={:?} hack={}",
        desc.Width,
        desc.Height,
        desc.Format.0,
        desc.NumChannels,
        size / 1024,
        hip_array,
        hack_flag
    ));
    *p_handle = hip_array.cast();
    Ok(())
}

pub(crate) unsafe fn destroy(h_array: CUarray) -> CUresult {
    hipfix::forget_array_hack(h_array as hipArray_t);
    hipArrayDestroy(h_array as hipArray_t)?;
    Ok(())
}

pub(crate) unsafe fn get_descriptor_v2(
    p_array_descriptor: &mut CUDA_ARRAY_DESCRIPTOR,
    h_array: CUarray,
) -> CUresult {
    let mut desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
    hipArrayGetDescriptor(&mut desc, h_array as hipArray_t)?;
    *p_array_descriptor = *((&desc) as *const HIP_ARRAY_DESCRIPTOR as *const CUDA_ARRAY_DESCRIPTOR);
    Ok(())
}

pub(crate) unsafe fn _3d_create_v2(
    p_handle: &mut CUarray,
    p_allocate_array: &CUDA_ARRAY3D_DESCRIPTOR,
) -> CUresult {
    let mut desc =
        *(p_allocate_array as *const CUDA_ARRAY3D_DESCRIPTOR as *const HIP_ARRAY3D_DESCRIPTOR);
    let (hack_flag, new_format) = hipfix::get_non_broken_format(desc.Format);
    desc.Format = new_format;
    hipfix::array_3d_create(&mut desc);
    let vram_before = super::memory::vram_used_bytes();
    let mut hip_array = mem::zeroed();
    hipArray3DCreate(&mut hip_array, &mut desc as _)?;
    let elem_size = format_byte_size(desc.Format) * desc.NumChannels as usize;
    let size =
        desc.Width as usize * desc.Height.max(1) as usize * desc.Depth.max(1) as usize * elem_size;
    super::memory::log_vram_delta(
        "hipArray3DCreate",
        vram_before,
        super::memory::vram_used_bytes(),
        size,
    );
    super::memory::update_alloc_range(hip_array as u64, size.max(0x10000) as u64);
    if hack_flag != 0 {
        hipfix::set_array_hack(
            hip_array,
            hipfix::ArrayHack {
                hack_flag,
                num_channels: desc.NumChannels,
                width: desc.Width as usize,
                height: desc.Height as usize,
                depth: desc.Depth as usize,
            },
        );
    }
    super::driver::zluda_log_err(&format!(
        "ARRAY3D_CREATE {}x{}x{} fmt={} ch={} flags={} ~{} KiB ptr={:?} hack={}",
        desc.Width,
        desc.Height,
        desc.Depth,
        desc.Format.0,
        desc.NumChannels,
        desc.Flags,
        size / 1024,
        hip_array,
        hack_flag
    ));
    *p_handle = hip_array as CUarray;
    Ok(())
}

pub(crate) unsafe fn _3d_get_descriptor_v2(
    p_array_descriptor: &mut CUDA_ARRAY3D_DESCRIPTOR,
    h_array: CUarray,
) -> CUresult {
    let mut desc: HIP_ARRAY3D_DESCRIPTOR = mem::zeroed();
    hipArray3DGetDescriptor(&mut desc, h_array as hipArray_t)?;
    *p_array_descriptor =
        *((&desc) as *const HIP_ARRAY3D_DESCRIPTOR as *const CUDA_ARRAY3D_DESCRIPTOR);
    Ok(())
}

pub(crate) unsafe fn mipmapped_create(
    p_handle: &mut CUmipmappedArray,
    p_mipmapped_array_desc: &CUDA_ARRAY3D_DESCRIPTOR,
    num_mipmap_levels: u32,
) -> CUresult {
    let mut desc = *(p_mipmapped_array_desc as *const CUDA_ARRAY3D_DESCRIPTOR
        as *const HIP_ARRAY3D_DESCRIPTOR);
    let (hack_flag, new_format) = hipfix::get_non_broken_format(desc.Format);
    desc.Format = new_format;
    let vram_before = super::memory::vram_used_bytes();
    let mut hip_array = ptr::null_mut();
    hipMipmappedArrayCreate(&mut hip_array, &mut desc, num_mipmap_levels)?;
    super::memory::log_vram_delta(
        "hipMipmappedArrayCreate",
        vram_before,
        super::memory::vram_used_bytes(),
        0,
    );
    if hack_flag != 0 {
        hipfix::set_mipmapped_hack(
            hip_array,
            hipfix::ArrayHack {
                hack_flag,
                num_channels: desc.NumChannels,
                width: desc.Width as usize,
                height: desc.Height as usize,
                depth: desc.Depth as usize,
            },
        );
    }
    *p_handle = hip_array.cast();
    Ok(())
}

pub(crate) unsafe fn mipmapped_destroy(h_mipmapped_array: CUmipmappedArray) -> CUresult {
    hipfix::forget_mipmapped_hack(h_mipmapped_array as hipMipmappedArray_t);
    hipMipmappedArrayDestroy(h_mipmapped_array as hipMipmappedArray_t)?;
    Ok(())
}

pub(crate) unsafe fn mipmapped_get_level(
    p_level_array: &mut CUarray,
    h_mipmapped_array: CUmipmappedArray,
    level: u32,
) -> CUresult {
    let mut hip_array: hipArray_t = mem::zeroed();
    hipMipmappedArrayGetLevel(
        &mut hip_array,
        h_mipmapped_array as hipMipmappedArray_t,
        level,
    )?;
    *p_level_array = hip_array as CUarray;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::*;
    use std::mem;

    #[test_cuda]
    unsafe fn array_create_destroy(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        let desc = CUDA_ARRAY_DESCRIPTOR {
            Width: 64,
            Height: 64,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 4,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArrayCreate_v2(&mut array, &desc);
        assert!(!array.is_null());
        api.cuArrayDestroy(array);
    }

    #[test_cuda]
    unsafe fn array_create_get_descriptor(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        let desc = CUDA_ARRAY_DESCRIPTOR {
            Width: 128,
            Height: 32,
            Format: CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8,
            NumChannels: 1,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArrayCreate_v2(&mut array, &desc);

        let mut out_desc: CUDA_ARRAY_DESCRIPTOR = mem::zeroed();
        api.cuArrayGetDescriptor_v2(&mut out_desc, array);
        assert_eq!(out_desc.Width, 128);
        assert_eq!(out_desc.Height, 32);
        assert_eq!(out_desc.NumChannels, 1);

        api.cuArrayDestroy(array);
    }

    #[test_cuda]
    unsafe fn array_3d_create_destroy(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 32,
            Height: 32,
            Depth: 4,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 4,
            Flags: 0,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &desc);
        assert!(!array.is_null());

        let mut out_desc: CUDA_ARRAY3D_DESCRIPTOR = mem::zeroed();
        api.cuArray3DGetDescriptor_v2(&mut out_desc, array);
        assert_eq!(out_desc.Width, 32);
        assert_eq!(out_desc.Height, 32);
        assert_eq!(out_desc.Depth, 4);

        api.cuArrayDestroy(array);
    }

    #[test_cuda]
    unsafe fn array_null_returns_error(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);

        let result = api.cuArrayDestroy_unchecked(std::ptr::null_mut());
        assert!(result.is_err());
    }
}

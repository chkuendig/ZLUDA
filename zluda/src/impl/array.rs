use std::{mem, ptr};

use super::hipfix;
use cuda_types::cuda::*;
use hip_runtime_sys::*;

pub(crate) unsafe fn create_v2(
    p_handle: &mut CUarray,
    p_allocate_array: &CUDA_ARRAY_DESCRIPTOR,
) -> CUresult {
    let mut desc = *(p_allocate_array as *const CUDA_ARRAY_DESCRIPTOR as *const HIP_ARRAY_DESCRIPTOR);
    desc.Format = hipfix::get_non_broken_format(desc.Format);
    let mut hip_array = ptr::null_mut();
    hipArrayCreate(&mut hip_array, &desc)?;
    *p_handle = hip_array.cast();
    Ok(())
}

pub(crate) unsafe fn destroy(h_array: CUarray) -> CUresult {
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
    let mut desc = *(p_allocate_array as *const CUDA_ARRAY3D_DESCRIPTOR as *const HIP_ARRAY3D_DESCRIPTOR);
    desc.Format = hipfix::get_non_broken_format(desc.Format);
    hipfix::array_3d_create(&mut desc);
    let mut hip_array = mem::zeroed();
    hipArray3DCreate(&mut hip_array, &mut desc as _)?;
    *p_handle = hip_array as CUarray;
    Ok(())
}

pub(crate) unsafe fn _3d_get_descriptor_v2(
    p_array_descriptor: &mut CUDA_ARRAY3D_DESCRIPTOR,
    h_array: CUarray,
) -> CUresult {
    let mut desc: HIP_ARRAY3D_DESCRIPTOR = mem::zeroed();
    hipArray3DGetDescriptor(&mut desc, h_array as hipArray_t)?;
    *p_array_descriptor = *((&desc) as *const HIP_ARRAY3D_DESCRIPTOR as *const CUDA_ARRAY3D_DESCRIPTOR);
    Ok(())
}

pub(crate) unsafe fn mipmapped_create(
    p_handle: &mut CUmipmappedArray,
    p_mipmapped_array_desc: &CUDA_ARRAY3D_DESCRIPTOR,
    num_mipmap_levels: u32,
) -> CUresult {
    let mut desc = *(p_mipmapped_array_desc as *const CUDA_ARRAY3D_DESCRIPTOR as *const HIP_ARRAY3D_DESCRIPTOR);
    desc.Format = hipfix::get_non_broken_format(desc.Format);
    let mut hip_array = ptr::null_mut();
    hipMipmappedArrayCreate(&mut hip_array, &mut desc, num_mipmap_levels)?;
    *p_handle = hip_array.cast();
    Ok(())
}

pub(crate) unsafe fn mipmapped_destroy(h_mipmapped_array: CUmipmappedArray) -> CUresult {
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

use hip_runtime_sys::*;

pub(crate) unsafe fn create_v2(
    handle: *mut hipArray_t,
    desc: *const HIP_ARRAY_DESCRIPTOR,
) -> hipError_t {
    hipArrayCreate(handle, desc)
}

pub(crate) unsafe fn destroy(array: hipArray_t) -> hipError_t {
    hipArrayDestroy(array)
}

pub(crate) unsafe fn _3d_create_v2(
    handle: *mut hipArray_t,
    desc: *const HIP_ARRAY3D_DESCRIPTOR,
) -> hipError_t {
    hipArray3DCreate(handle, desc)
}

pub(crate) unsafe fn _3d_get_descriptor_v2(
    array_descriptor: *mut HIP_ARRAY3D_DESCRIPTOR,
    array: hipArray_t,
) -> hipError_t {
    hipArray3DGetDescriptor(array_descriptor, array)
}

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::*;

    #[test_cuda]
    unsafe fn array_3d_get_descriptor(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: 16,
            Height: 8,
            Depth: 4,
            Format: CUarray_format::CU_AD_FORMAT_HALF,
            NumChannels: 4,
            Flags: 0,
        };
        let mut array = std::mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &desc);
        let mut actual: CUDA_ARRAY3D_DESCRIPTOR = std::mem::zeroed();
        api.cuArray3DGetDescriptor_v2(&mut actual, array);
        assert_eq!(
            (actual.Width, actual.Height, actual.Depth),
            (desc.Width, desc.Height, desc.Depth)
        );
        assert_eq!(actual.Format, desc.Format);
        assert_eq!(actual.NumChannels, desc.NumChannels);
        api.cuArrayDestroy(array);
        api.cuCtxDestroy_v2(ctx);
    }
}

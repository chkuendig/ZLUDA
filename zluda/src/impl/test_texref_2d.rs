use crate::tests::CudaApi;
use cuda_macros::test_cuda;
use cuda_types::cuda::*;
use std::ffi::c_void;

#[test]
fn pitched_texture_shape_checks_rows_channels_and_overflow() {
    use hip_runtime_sys::*;
    let mut desc = HIP_ARRAY_DESCRIPTOR { Width: 3, Height: 2, Format: hipArray_Format(0x20), NumChannels: 4 };
    assert_eq!(super::pitched_row_bytes(&desc, 256).unwrap(), 48);
    assert!(super::pitched_row_bytes(&desc, 47).is_err());
    desc.NumChannels = 3; assert!(super::pitched_row_bytes(&desc, 256).is_err());
    desc.NumChannels = 4; desc.Width = usize::MAX;
    assert!(super::pitched_row_bytes(&desc, 256).is_err());
    desc.Width = 3; desc.Height = usize::MAX;
    assert!(super::pitched_row_bytes(&desc, 256).is_err());
    desc.Height = 0; assert!(super::pitched_row_bytes(&desc, 256).is_err());
}

unsafe fn upload(api: &impl CudaApi, destination: CUdeviceptr, pitch: usize, add: f32) {
    let source: Vec<f32> = (1..=6).map(|n| n as f32+add).collect();
    let mut copy: CUDA_MEMCPY2D = std::mem::zeroed();
    copy.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
    copy.srcHost = source.as_ptr().cast(); copy.srcPitch = 12;
    copy.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = destination; copy.dstPitch = pitch;
    copy.WidthInBytes = 12; copy.Height = 2;
    api.cuMemcpy2D_v2(&copy);
}

unsafe fn read(api: &impl CudaApi, function: CUfunction, mut output: CUdeviceptr) -> f32 {
    let (mut x, mut y) = (1.5f32, 1.5f32);
    let mut args = [&mut output as *mut _ as *mut c_void,
        &mut x as *mut _ as *mut c_void, &mut y as *mut _ as *mut c_void];
    api.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
        CUstream(std::ptr::null_mut()), args.as_mut_ptr(), std::ptr::null_mut());
    api.cuCtxSynchronize();
    let mut result = [0f32; 4];
    api.cuMemcpyDtoH_v2(result.as_mut_ptr().cast(), output, 16);
    result[0]
}

#[test_cuda]
unsafe fn pitched_texref_reads_rebinds_and_preserves_binding_on_failure(api: impl CudaApi) {
    api.cuInit(0);
    let mut context = std::mem::zeroed();
    api.cuCtxCreate_v2(&mut context, 0, 0);
    let source = concat!(include_str!("test_ptx/tex_read_2d_float.ptx"), "\0");
    let mut module = std::mem::zeroed();
    api.cuModuleLoadData(&mut module, source.as_ptr().cast());
    let mut function = std::mem::zeroed();
    api.cuModuleGetFunction(&mut function, module, c"tex_read_2d_float".as_ptr());
    let mut texref = std::mem::zeroed();
    api.cuModuleGetTexRef(&mut texref, module, c"tex".as_ptr());
    api.cuTexRefSetFormat(texref, CUarray_format::CU_AD_FORMAT_FLOAT, 1);
    api.cuTexRefSetFlags(texref, 0);
    api.cuTexRefSetFilterMode(texref, CUfilter_mode::CU_TR_FILTER_MODE_POINT);
    let (mut a, mut b, mut output) = (std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed());
    let (mut pitch_a, mut pitch_b) = (0, 0);
    api.cuMemAllocPitch_v2(&mut a, &mut pitch_a, 12, 2, 4);
    api.cuMemAllocPitch_v2(&mut b, &mut pitch_b, 12, 2, 4);
    api.cuMemAlloc_v2(&mut output, 16);
    upload(&api, a, pitch_a, 0.0); upload(&api, b, pitch_b, 10.0);
    let desc = CUDA_ARRAY_DESCRIPTOR { Width: 3, Height: 2, Format: CUarray_format::CU_AD_FORMAT_FLOAT, NumChannels: 1 };
    api.cuTexRefSetAddress2D_v3(texref, &desc, a, pitch_a);
    assert_eq!(read(&api, function, output), 5.0);
    api.cuTexRefSetAddress2D_v3(texref, &desc, b, pitch_b);
    assert_eq!(read(&api, function, output), 15.0);
    upload(&api, a, pitch_a, 20.0);
    api.cuTexRefSetAddress2D_v3(texref, &desc, a, pitch_a);
    assert_eq!(read(&api, function, output), 25.0);
    assert!(api.cuTexRefSetAddress2D_v3_unchecked(texref, &desc, a, 1).is_err());
    assert_eq!(read(&api, function, output), 25.0);
    api.cuTexRefSetAddress2D_v3(texref, &desc, b, pitch_b);
    api.cuMemFree_v2(a);
    assert_eq!(read(&api, function, output), 15.0);
    api.cuModuleUnload(module);
    api.cuMemFree_v2(b); api.cuMemFree_v2(output);
    api.cuCtxDestroy_v2(context);
}

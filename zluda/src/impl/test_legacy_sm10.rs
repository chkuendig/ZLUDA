use crate::tests::CudaApi;
use cuda_macros::test_cuda;
use cuda_types::cuda::*;
use std::ffi::c_void;

#[test_cuda]
unsafe fn legacy_sm10_maps_float_instructions_and_keeps_storage(api: impl CudaApi) {
    api.cuInit(0);
    let mut context = std::mem::zeroed();
    api.cuCtxCreate_v2(&mut context, 0, 0);
    let source = concat!(include_str!("test_ptx/legacy_sm10.ptx"), "\0");
    let mut module = std::mem::zeroed();
    api.cuModuleLoadData(&mut module, source.as_ptr().cast());
    let mut function = std::mem::zeroed();
    api.cuModuleGetFunction(&mut function, module, c"legacy_sm10".as_ptr());
    let mut storage = std::mem::zeroed();
    let mut size = 0;
    api.cuModuleGetGlobal_v2(&mut storage, &mut size, module, c"storage".as_ptr());
    assert_eq!(size, 16, "two .f64 storage elements retain their eight-byte stride");
    let doubles = [16777217.0f64, -3.5];
    api.cuMemcpyHtoD_v2(storage, doubles.as_ptr().cast(), 16);
    let mut output = std::mem::zeroed();
    api.cuMemAlloc_v2(&mut output, 32);
    let mut actual = [0x11223344u32; 8];
    api.cuMemcpyHtoD_v2(output, actual.as_ptr().cast(), 32);
    let mut parameters = [&mut output as *mut _ as *mut c_void];
    api.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
        CUstream(std::ptr::null_mut()), parameters.as_mut_ptr(), std::ptr::null_mut());
    api.cuCtxSynchronize();
    api.cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, 32);
    assert_eq!(actual, [16777216f32.to_bits(), 0x11223344, 0, 0x41700000,
        (2f32/3f32).to_bits(), 1, (-3.5f32).to_bits(), 0x11223344]);
    api.cuMemFree_v2(output);
    api.cuModuleUnload(module);
    api.cuCtxDestroy_v2(context);
}

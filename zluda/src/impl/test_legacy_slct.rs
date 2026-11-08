use crate::tests::CudaApi;
use cuda_macros::test_cuda;
use cuda_types::cuda::*;
use std::ffi::c_void;

#[test_cuda]
unsafe fn legacy_signed_selection_keeps_selected_bits(api: impl CudaApi) {
    api.cuInit(0);
    let mut context = std::mem::zeroed();
    api.cuCtxCreate_v2(&mut context, 0, 0);
    let source = concat!(include_str!("test_ptx/legacy_slct.ptx"), "\0");
    let mut module = std::mem::zeroed();
    api.cuModuleLoadData(&mut module, source.as_ptr().cast());
    let mut function = std::mem::zeroed();
    api.cuModuleGetFunction(&mut function, module, c"legacy_slct".as_ptr());
    let mut output = std::mem::zeroed();
    api.cuMemAlloc_v2(&mut output, 20);
    for mut sign in [i32::MIN, -1, 0, 1, i32::MAX] {
        let mut args = [
            &mut output as *mut _ as *mut c_void,
            &mut sign as *mut _ as *mut c_void,
        ];
        api.cuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            CUstream(std::ptr::null_mut()),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuCtxSynchronize();
        let mut actual = [0u32; 5];
        api.cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, 20);
        assert_eq!(
            actual,
            if sign >= 0 {
                [0x1234, 0x01234567, 0x89abcdef, 0x01234567, 0x80000000]
            } else {
                [0xfedc, 0x89abcdef, 0x76543210, 0xfedcba98, 0x7fc01234]
            }
        );
    }
    api.cuMemFree_v2(output);
    api.cuModuleUnload(module);
    api.cuCtxDestroy_v2(context);
}

use crate::tests::CudaApi;
use cuda_macros::test_cuda;
use cuda_types::cuda::*;
use std::ffi::c_void;

#[test_cuda]
unsafe fn legacy_typed_globals_keep_initializers(api: impl CudaApi) {
    let source = concat!(include_str!("test_ptx/legacy_typed_globals.ptx"), "\0");
    run(
        &api,
        source,
        c"legacy_typed_globals",
        1,
        &[0x3fc00000, 0, 0x40200000, 0xc0600000],
    );
}

#[test_cuda]
unsafe fn legacy_sreg16_reads_low_bits(api: impl CudaApi) {
    let source = concat!(include_str!("test_ptx/legacy_sreg16.ptx"), "\0");
    for grid in [1u32, 65535] {
        let block = (grid - 1) & 0xffff;
        let count = grid & 0xffff;
        run(
            &api,
            source,
            c"legacy_sreg16",
            grid,
            &[0, 1, block, count, 0, 1, block, count],
        );
    }
}

unsafe fn run(
    api: &impl CudaApi,
    source: &str,
    name: &std::ffi::CStr,
    grid: u32,
    expected: &[u32],
) {
    api.cuInit(0);
    let mut context = std::mem::zeroed();
    api.cuCtxCreate_v2(&mut context, 0, 0);
    let mut module = std::mem::zeroed();
    api.cuModuleLoadData(&mut module, source.as_ptr().cast());
    let mut function = std::mem::zeroed();
    api.cuModuleGetFunction(&mut function, module, name.as_ptr());
    let bytes = std::mem::size_of_val(expected);
    let mut output = std::mem::zeroed();
    api.cuMemAlloc_v2(&mut output, bytes);
    let mut actual = vec![0x11223344u32; expected.len()];
    api.cuMemcpyHtoD_v2(output, actual.as_ptr().cast(), bytes);
    let mut unused_input = 0u64;
    let mut parameters = [
        &mut unused_input as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
    ];
    api.cuLaunchKernel(
        function,
        grid,
        1,
        1,
        1,
        1,
        1,
        0,
        CUstream(std::ptr::null_mut()),
        parameters.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    api.cuCtxSynchronize();
    api.cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, bytes);
    assert_eq!(actual, expected);
    api.cuMemFree_v2(output);
    api.cuModuleUnload(module);
    api.cuCtxDestroy_v2(context);
}

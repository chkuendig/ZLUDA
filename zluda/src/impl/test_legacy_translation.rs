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

#[test_cuda]
unsafe fn legacy_tex2d_ignores_coordinate_padding(api: impl CudaApi) {
    api.cuInit(0);
    let mut context = std::mem::zeroed();
    api.cuCtxCreate_v2(&mut context, 0, 0);
    let source = concat!(include_str!("test_ptx/legacy_tex2d_coords_v4.ptx"), "\0");
    let mut module = std::mem::zeroed();
    api.cuModuleLoadData(&mut module, source.as_ptr().cast());
    let mut function = std::mem::zeroed();
    api.cuModuleGetFunction(&mut function, module, c"legacy_tex2d_coords_v4".as_ptr());
    let mut texture = std::mem::zeroed();
    api.cuModuleGetTexRef(&mut texture, module, c"legacy_texture".as_ptr());
    let descriptor = CUDA_ARRAY_DESCRIPTOR {
        Width: 2,
        Height: 2,
        Format: CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32,
        NumChannels: 4,
    };
    let mut array = std::mem::zeroed();
    api.cuArrayCreate_v2(&mut array, &descriptor);
    let pixels: [u32; 16] = std::array::from_fn(|index| index as u32 + 1);
    let mut copy: CUDA_MEMCPY2D = std::mem::zeroed();
    copy.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
    copy.srcHost = pixels.as_ptr().cast();
    copy.srcPitch = 32;
    copy.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = 32;
    copy.Height = 2;
    api.cuMemcpy2D_v2(&copy);
    api.cuTexRefSetArray(texture, array, CU_TRSA_OVERRIDE_FORMAT);
    for dimension in [0, 1] {
        api.cuTexRefSetAddressMode(texture, dimension, CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP);
    }
    api.cuTexRefSetFilterMode(texture, CUfilter_mode::CU_TR_FILTER_MODE_POINT);
    api.cuTexRefSetFlags(texture, CU_TRSF_READ_AS_INTEGER);
    let mut output = std::mem::zeroed();
    api.cuMemAlloc_v2(&mut output, 16);
    let mut actual = [0x11223344u32; 4];
    api.cuMemcpyHtoD_v2(output, actual.as_ptr().cast(), 16);
    let mut parameters = [&mut output as *mut _ as *mut c_void];
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
        parameters.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    api.cuCtxSynchronize();
    api.cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, 16);
    api.cuMemFree_v2(output);
    api.cuModuleUnload(module);
    api.cuArrayDestroy(array);
    api.cuCtxDestroy_v2(context);
    assert_eq!(actual, [1, 2, 3, 4]);
}

#[test_cuda]
unsafe fn legacy_tex1d_ignores_coordinate_padding(api: impl CudaApi) {
    for source in [
        concat!(
            include_str!("test_ptx/legacy_tex1d_coords_v1_control.ptx"),
            "\0"
        ),
        concat!(include_str!("test_ptx/legacy_tex1d_coords_v4.ptx"), "\0"),
    ] {
        api.cuInit(0);
        let mut context = std::mem::zeroed();
        api.cuCtxCreate_v2(&mut context, 0, 0);
        let mut module = std::mem::zeroed();
        api.cuModuleLoadData(&mut module, source.as_ptr().cast());
        let mut function = std::mem::zeroed();
        api.cuModuleGetFunction(&mut function, module, c"fetch_1d".as_ptr());
        let mut texture = std::mem::zeroed();
        api.cuModuleGetTexRef(&mut texture, module, c"sample_tex".as_ptr());
        let pixels = [1.25f32, -2.5, 3.75, 8.0, 0.5, 4.0, 16.0, -0.75];
        let bytes = std::mem::size_of_val(&pixels);
        let mut input = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut input, bytes);
        api.cuMemcpyHtoD_v2(input, pixels.as_ptr().cast(), bytes);
        api.cuTexRefSetFormat(texture, CUarray_format::CU_AD_FORMAT_FLOAT, 4);
        api.cuTexRefSetFlags(texture, 0);
        let mut offset = usize::MAX;
        api.cuTexRefSetAddress_v2(&mut offset, texture, input, bytes);
        let mut output = std::mem::zeroed();
        api.cuMemAlloc_v2(&mut output, 16);
        let mut actual = [0x11223344u32; 4];
        api.cuMemcpyHtoD_v2(output, actual.as_ptr().cast(), 16);
        let mut parameters = [&mut output as *mut _ as *mut c_void];
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
            parameters.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        api.cuCtxSynchronize();
        api.cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, 16);
        api.cuMemFree_v2(output);
        api.cuModuleUnload(module);
        api.cuMemFree_v2(input);
        api.cuCtxDestroy_v2(context);
        assert_eq!(offset, 0);
        assert_eq!(actual, [0.5f32, 4.0, 16.0, -0.75].map(f32::to_bits));
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

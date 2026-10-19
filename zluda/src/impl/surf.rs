use cuda_types::cuda::*;
use hip_runtime_sys::*;
use std::mem;

// Same as in zluda_ptx_impl.cpp: the top bits of a surface handle hold log2
// of the pixel size, which sust/suld use to turn CUDA's byte x-coordinate
// into a pixel index.
const IMAGE_RESERVED_TOP_BITS: u32 = 3;

pub(crate) unsafe fn encode_pixel_shift(handle: usize, array: hipArray_t) -> Result<usize, CUerror> {
    if handle >> (usize::BITS - IMAGE_RESERVED_TOP_BITS) != 0 {
        return Err(CUerror::UNKNOWN);
    }
    let mut desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
    hipArrayGetDescriptor(&mut desc, array)?;
    let pixel_size = format_size(desc.Format)? * desc.NumChannels.max(1) as usize;
    let shift = pixel_size.trailing_zeros() as usize;
    Ok(handle | (shift << (usize::BITS - IMAGE_RESERVED_TOP_BITS)))
}

pub(crate) fn strip_pixel_shift(handle: usize) -> usize {
    (handle << IMAGE_RESERVED_TOP_BITS) >> IMAGE_RESERVED_TOP_BITS
}

pub(crate) unsafe fn object_create(
    p_surf_object: &mut CUsurfObject,
    p_res_desc: &CUDA_RESOURCE_DESC,
) -> CUresult {
    if p_res_desc.resType != CUresourcetype::CU_RESOURCE_TYPE_ARRAY {
        return Err(CUerror::NOT_SUPPORTED);
    }
    let array = p_res_desc.res.array.hArray as hipArray_t;
    let desc = hipResourceDesc {
        resType: hipResourceType::hipResourceTypeArray,
        res: hipResourceDesc__bindgen_ty_1 {
            array: hipResourceDesc__bindgen_ty_1__bindgen_ty_1 { array },
        },
    };
    let mut surf_obj: hipSurfaceObject_t = mem::zeroed();
    hipCreateSurfaceObject(&mut surf_obj, &desc)?;
    match encode_pixel_shift(surf_obj as usize, array) {
        Ok(handle) => {
            *p_surf_object = handle as CUsurfObject;
            Ok(())
        }
        Err(error) => {
            let _ = hipDestroySurfaceObject(surf_obj);
            Err(error)
        }
    }
}

pub(crate) unsafe fn object_destroy(surf_object: CUsurfObject) -> CUresult {
    hipDestroySurfaceObject(strip_pixel_shift(surf_object as usize) as hipSurfaceObject_t)?;
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

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::*;
    use std::ffi::c_void;
    use std::mem;

    const SURF_STORE_3D_B64_PTX: &str =
        concat!(include_str!("test_ptx/surf_store_3d_b64.ptx"), "\0");
    // HALF encodings of 0.5, 1, 2 and -3.
    const HALF4_TEXEL: u64 = 0xc200_4000_3c00_3800;
    const WIDTH: usize = 4;
    const HEIGHT: usize = 2;
    const DEPTH: usize = 2;

    unsafe fn copy_array(api: &impl CudaApi, array: CUarray, host: *mut c_void, to_array: bool) {
        let mut copy: CUDA_MEMCPY3D = mem::zeroed();
        if to_array {
            copy.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
            copy.srcHost = host;
            copy.srcPitch = WIDTH * 8;
            copy.srcHeight = HEIGHT;
            copy.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
            copy.dstArray = array;
        } else {
            copy.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
            copy.srcArray = array;
            copy.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
            copy.dstHost = host;
            copy.dstPitch = WIDTH * 8;
            copy.dstHeight = HEIGHT;
        }
        copy.WidthInBytes = WIDTH * 8;
        copy.Height = HEIGHT;
        copy.Depth = DEPTH;
        api.cuMemcpy3D_v2(&copy);
    }

    // Stores one HALF4 texel through `create_handle` with sust.b and checks
    // the raw bytes of every texel, so a wrong x-offset or a converting store
    // is caught.
    unsafe fn store_half4_texel(
        api: &impl CudaApi,
        create_handle: impl FnOnce(&CUDA_RESOURCE_DESC) -> u64,
        destroy_handle: impl FnOnce(u64),
    ) {
        api.cuInit(0);
        let mut ctx = mem::zeroed();
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            Width: WIDTH,
            Height: HEIGHT,
            Depth: DEPTH,
            Format: CUarray_format::CU_AD_FORMAT_HALF,
            NumChannels: 4,
            Flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mut array = mem::zeroed();
        api.cuArray3DCreate_v2(&mut array, &desc);
        let mut texels = [0u64; WIDTH * HEIGHT * DEPTH];
        copy_array(api, array, texels.as_mut_ptr().cast(), true);
        let mut res_desc: CUDA_RESOURCE_DESC = mem::zeroed();
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;
        let handle = create_handle(&res_desc);
        let mut module = mem::zeroed();
        api.cuModuleLoadData(&mut module, SURF_STORE_3D_B64_PTX.as_ptr().cast());
        let mut func = mem::zeroed();
        api.cuModuleGetFunction(&mut func, module, c"surf_store_3d_b64".as_ptr());
        let (x, y, z) = (1usize, 1usize, 1usize);
        let x_bytes = (x * 8) as i32;
        let (y_arg, z_arg) = (y as i32, z as i32);
        let mut params: [*mut c_void; 5] = [
            &handle as *const _ as *mut _,
            &x_bytes as *const _ as *mut _,
            &y_arg as *const _ as *mut _,
            &z_arg as *const _ as *mut _,
            &HALF4_TEXEL as *const _ as *mut _,
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
        copy_array(api, array, texels.as_mut_ptr().cast(), false);
        let mut expected = [0u64; WIDTH * HEIGHT * DEPTH];
        expected[(z * HEIGHT + y) * WIDTH + x] = HALF4_TEXEL;
        assert_eq!(texels, expected);
        api.cuModuleUnload(module);
        destroy_handle(handle);
        api.cuArrayDestroy(array);
        api.cuCtxDestroy_v2(ctx);
    }

    #[test_cuda]
    unsafe fn surface_object_stores_half4_texel(api: impl CudaApi) {
        store_half4_texel(
            &api,
            |res_desc| {
                let mut surface = 0;
                api.cuSurfObjectCreate(&mut surface, res_desc);
                surface
            },
            |surface| api.cuSurfObjectDestroy(surface),
        );
    }

    #[test_cuda]
    unsafe fn texture_object_stores_half4_texel(api: impl CudaApi) {
        store_half4_texel(
            &api,
            |res_desc| {
                let tex_desc: CUDA_TEXTURE_DESC = mem::zeroed();
                let mut texture = 0;
                api.cuTexObjectCreate(&mut texture, res_desc, &tex_desc, std::ptr::null());
                texture
            },
            |texture| api.cuTexObjectDestroy(texture),
        );
    }
}

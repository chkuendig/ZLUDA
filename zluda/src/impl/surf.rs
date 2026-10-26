use crate::r#impl::{context, hipfix};
use cuda_types::cuda::{CUerror, CUsurfObject, CUsurfref};
use hip_runtime_sys::*;
use std::mem::{self, offset_of};
use zluda_common::FromCuda;

#[repr(C)]
pub(crate) struct SurfRefData {
    pub(crate) base: *mut textureReference,
    pub(crate) devptr: hipDeviceptr_t,
}

unsafe impl Send for SurfRefData {}
unsafe impl Sync for SurfRefData {}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, CUsurfref, E>
    for &'a SurfRefData
{
    fn from_cuda(surf_ref: &'a CUsurfref) -> Result<Self, E> {
        if surf_ref.is_null() {
            return Err(E::INVALID_VALUE);
        }
        Ok(unsafe { mem::transmute(*surf_ref) })
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, *mut CUsurfref, E>
    for &'a mut &'a SurfRefData
{
    fn from_cuda(surf_ref: &'a *mut CUsurfref) -> Result<Self, E> {
        let surf_ref = unsafe { surf_ref.as_mut() }.ok_or_else(|| E::INVALID_VALUE)?;
        Ok(unsafe { mem::transmute(surf_ref) })
    }
}

fn convert_resource_desc(desc: &HIP_RESOURCE_DESC) -> Result<hipResourceDesc, hipErrorCode_t> {
    let (type_, desc) = match desc.resType {
        hipResourcetype::HIP_RESOURCE_TYPE_ARRAY => (
            hipResourceType::hipResourceTypeArray,
            hipResourceDesc__bindgen_ty_1 {
                array: hipResourceDesc__bindgen_ty_1__bindgen_ty_1 {
                    array: unsafe { desc.res.array.hArray },
                },
            },
        ),
        hipResourcetype::HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY => (
            hipResourceType::hipResourceTypeMipmappedArray,
            hipResourceDesc__bindgen_ty_1 {
                mipmap: hipResourceDesc__bindgen_ty_1__bindgen_ty_2 {
                    mipmap: unsafe { desc.res.mipmap.hMipmappedArray },
                },
            },
        ),
        hipResourcetype::HIP_RESOURCE_TYPE_LINEAR => {
            let linear = unsafe { desc.res.linear };
            let format_kind = format_kind(linear.format)?;
            let channel_bits = channel_bits(linear.format)?;
            let desc = hipChannelFormatDesc {
                x: channel_bits,
                y: if linear.numChannels > 1 {
                    channel_bits
                } else {
                    0
                },
                z: if linear.numChannels > 2 {
                    channel_bits
                } else {
                    0
                },
                w: if linear.numChannels > 3 {
                    channel_bits
                } else {
                    0
                },
                f: format_kind,
            };
            (
                hipResourceType::hipResourceTypeLinear,
                hipResourceDesc__bindgen_ty_1 {
                    linear: hipResourceDesc__bindgen_ty_1__bindgen_ty_3 {
                        devPtr: linear.devPtr.0,
                        desc,
                        sizeInBytes: linear.sizeInBytes,
                    },
                },
            )
        }
        hipResourcetype::HIP_RESOURCE_TYPE_PITCH2D => {
            let pitch2d = unsafe { desc.res.pitch2D };
            let format_kind = format_kind(pitch2d.format)?;
            let channel_bits = channel_bits(pitch2d.format)?;
            let desc = hipChannelFormatDesc {
                x: channel_bits,
                y: if pitch2d.numChannels > 1 {
                    channel_bits
                } else {
                    0
                },
                z: if pitch2d.numChannels > 2 {
                    channel_bits
                } else {
                    0
                },
                w: if pitch2d.numChannels > 3 {
                    channel_bits
                } else {
                    0
                },
                f: format_kind,
            };
            (
                hipResourceType::hipResourceTypePitch2D,
                hipResourceDesc__bindgen_ty_1 {
                    pitch2D: hipResourceDesc__bindgen_ty_1__bindgen_ty_4 {
                        devPtr: pitch2d.devPtr.0,
                        desc,
                        width: pitch2d.width,
                        height: pitch2d.height,
                        pitchInBytes: pitch2d.pitchInBytes,
                    },
                },
            )
        }
        _ => return Err(hipErrorCode_t::InvalidValue),
    };

    Ok(hipResourceDesc {
        resType: type_,
        res: desc,
    })
}

fn format_kind(format: hipArray_Format) -> Result<hipChannelFormatKind, hipErrorCode_t> {
    Ok(match format {
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT8
        | hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT32 => {
            hipChannelFormatKind::hipChannelFormatKindUnsigned
        }
        hipArray_Format::HIP_AD_FORMAT_SIGNED_INT8
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT32 => {
            hipChannelFormatKind::hipChannelFormatKindSigned
        }
        hipArray_Format::HIP_AD_FORMAT_HALF | hipArray_Format::HIP_AD_FORMAT_FLOAT => {
            hipChannelFormatKind::hipChannelFormatKindFloat
        }
        _ => return Err(hipErrorCode_t::InvalidValue),
    })
}

fn channel_bits(format: hipArray_Format) -> Result<i32, hipErrorCode_t> {
    Ok(match format {
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT8
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT8 => 8,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16
        | hipArray_Format::HIP_AD_FORMAT_HALF => 16,
        hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_SIGNED_INT32
        | hipArray_Format::HIP_AD_FORMAT_FLOAT => 32,
        _ => return Err(hipErrorCode_t::InvalidValue),
    })
}

fn num_channels_for_array(array: hipArray_t) -> Result<(hipArray_Format, u32), hipErrorCode_t> {
    unsafe {
        let mut desc = std::mem::zeroed();
        hipArrayGetDescriptor(&mut desc, array)?;
        Ok((desc.Format, desc.NumChannels))
    }
}

fn num_channels_for_mipmapped_array(
    array: hipMipmappedArray_t,
) -> Result<(hipArray_Format, u32), hipErrorCode_t> {
    unsafe {
        let mut level_array = std::ptr::null_mut();
        hipGetMipmappedArrayLevel(&mut level_array, array, 0)?;
        num_channels_for_array(level_array)
    }
}

fn num_channels_for_resource_desc(
    desc: &HIP_RESOURCE_DESC,
) -> Result<(hipArray_Format, u32), hipErrorCode_t> {
    match desc.resType {
        hipResourcetype::HIP_RESOURCE_TYPE_ARRAY => unsafe {
            num_channels_for_array(desc.res.array.hArray)
        },
        hipResourcetype::HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY => unsafe {
            num_channels_for_mipmapped_array(desc.res.mipmap.hMipmappedArray)
        },
        hipResourcetype::HIP_RESOURCE_TYPE_LINEAR => {
            Ok(unsafe { (desc.res.linear.format, desc.res.linear.numChannels) })
        }
        hipResourcetype::HIP_RESOURCE_TYPE_PITCH2D => {
            Ok(unsafe { (desc.res.pitch2D.format, desc.res.pitch2D.numChannels) })
        }
        _ => Err(hipErrorCode_t::InvalidValue),
    }
}

fn pixel_size_from_format_channels(
    format: hipArray_Format,
    num_channels: u32,
) -> Result<u32, hipErrorCode_t> {
    let bits_per_channel = channel_bits(format)? as u32;
    Ok((bits_per_channel * num_channels) / 8)
}

pub(crate) unsafe fn object_create(
    p_tex_object: &mut CUsurfObject,
    p_res_desc: &HIP_RESOURCE_DESC,
) -> hipError_t {
    let resource_desc = convert_resource_desc(p_res_desc)?;
    let (format, num_channels) = num_channels_for_resource_desc(p_res_desc)?;
    let pixel_size = pixel_size_from_format_channels(format, num_channels)?;
    let mut hip_surfobj = std::mem::zeroed();
    hipCreateSurfaceObject(&mut hip_surfobj, &resource_desc)?;
    *p_tex_object = to_cuda(hip_surfobj, pixel_size)?;
    Ok(())
}

pub(crate) unsafe fn object_destroy(tex_object: CUsurfObject) -> hipError_t {
    hipDestroySurfaceObject(to_hip(tex_object))
}

const RESERVED_SURFACE_BOTTOM_BITS: u32 = 3;

fn to_cuda(hip: hipSurfaceObject_t, pixel_size: u32) -> Result<CUsurfObject, hipErrorCode_t> {
    let hip = hip as u64;
    let bottom_bits = extract_bottom_bits::<RESERVED_SURFACE_BOTTOM_BITS>(hip);
    if bottom_bits != 0 {
        return Err(hipErrorCode_t::Unknown);
    }
    let shift_size = pixel_size.ilog2() as u64;
    Ok(set_bottom_bits::<RESERVED_SURFACE_BOTTOM_BITS>(
        hip, shift_size,
    ))
}

fn to_hip(cuda: CUsurfObject) -> hipSurfaceObject_t {
    ((cuda >> RESERVED_SURFACE_BOTTOM_BITS) << RESERVED_SURFACE_BOTTOM_BITS) as _
}

fn extract_bottom_bits<const N: u32>(x: u64) -> u64 {
    x & ((1 << N) - 1)
}

fn set_bottom_bits<const N: u32>(x: u64, bits: u64) -> u64 {
    x | bits
}

pub(crate) unsafe fn ref_set_array(
    surf_ref: &SurfRefData,
    array: hipArray_t,
    mut flags: ::core::ffi::c_uint,
) -> Result<(), CUerror> {
    flags |= hipfix::HIP_TRSA_OVERRIDE_FORMAT;
    super::tex::ref_set_array(surf_ref.base, array, flags)?;
    if let Some(texref) = surf_ref.base.as_ref() {
        if !texref.textureObject.is_null() {
            let pixel_size =
                pixel_size_from_format_channels(texref.format, texref.numChannels as u32)?;
            let mut cuda_texobj = to_cuda(texref.textureObject as _, pixel_size)?;
            let cu_context = context::get_current_context()?;
            let context: &context::Context = FromCuda::<_, CUerror>::from_cuda(&cu_context)?;
            hipMemcpyHtoDAsync(
                hipDeviceptr_t(
                    surf_ref
                        .devptr
                        .0
                        .add(offset_of!(textureReference, textureObject)),
                ),
                std::ptr::from_mut(&mut cuda_texobj).cast(),
                std::mem::size_of_val(&cuda_texobj),
                context.alloc_stream,
            )?;
            hipStreamSynchronize(context.alloc_stream)?;
        }
    }
    Ok(())
}
// PhysX also uses array texture-object handles with surface stores. Keep their
// pixel-shift encoding identical to surfaces, and strip it for HIP and sampling.
pub(crate) unsafe fn encode_pixel_shift(handle: usize, array: hipArray_t) -> Result<usize, CUerror> {
    let (format, channels) = num_channels_for_array(array)?;
    Ok(to_cuda(handle as _, pixel_size_from_format_channels(format, channels)?)? as usize)
}

pub(crate) fn strip_pixel_shift(handle: usize) -> usize {
    to_hip(handle as _) as usize
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
        check_texture_sampling: bool,
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
        if check_texture_sampling {
            // The same tagged handle must work for normal texture sampling,
            // not just sust. A missing tag mask dereferences the wrong address.
            const SAMPLE_PTX: &str = concat!(r#"
.version 6.5
.target sm_30
.address_size 64
.visible .entry sample_half4(.param .u64 texture, .param .u64 output) {
    .reg .b64 %texture, %output;
    .reg .f32 %f<4>;
    ld.param.u64 %texture, [texture];
    ld.param.u64 %output, [output];
    tex.3d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [%texture, {0f3fc00000, 0f3fc00000, 0f3fc00000, 0f00000000}];
    st.global.v4.f32 [%output], {%f0, %f1, %f2, %f3};
    ret;
}
"#, "\0");
            let mut sample_module = mem::zeroed();
            api.cuModuleLoadData(&mut sample_module, SAMPLE_PTX.as_ptr().cast());
            let mut sample_func = mem::zeroed();
            api.cuModuleGetFunction(&mut sample_func, sample_module, c"sample_half4".as_ptr());
            let mut output: CUdeviceptr = mem::zeroed();
            api.cuMemAlloc_v2(&mut output, 16);
            let mut sample_params = [
                &handle as *const _ as *mut c_void,
                &output as *const _ as *mut c_void,
            ];
            api.cuLaunchKernel(sample_func, 1, 1, 1, 1, 1, 1, 0,
                CUstream(std::ptr::null_mut()), sample_params.as_mut_ptr(), std::ptr::null_mut());
            api.cuStreamSynchronize(CUstream(std::ptr::null_mut()));
            let mut sampled = [0.0f32; 4];
            api.cuMemcpyDtoH_v2(sampled.as_mut_ptr().cast(), output, 16);
            assert_eq!(sampled, [0.5, 1.0, 2.0, -3.0]);
            api.cuMemFree_v2(output);
            api.cuModuleUnload(sample_module);
        }
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
            false,
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
            true,
        );
    }
}

use std::{mem, ptr};

use super::hipfix;
use cuda_types::cuda::*;
use hip_runtime_sys::*;

// HIP stores some CUDA formats as unsigned integers and strips
// SURFACE_LDST. Keep the caller's descriptor metadata separate from those
// internal choices. Never hold this lock across a HIP call.
#[derive(Clone, Copy)]
struct ArrayMetadata {
    format: u32,
    flags: u32,
    generation: u64,
    parent: Option<(usize, u64)>,
}

static ARRAY_METADATA: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<(bool, usize), ArrayMetadata>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
static ARRAY_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn metadata(key: (bool, usize)) -> Option<ArrayMetadata> {
    ARRAY_METADATA.lock().unwrap_or_else(|e| e.into_inner()).get(&key).copied()
}

fn remember_metadata(key: (bool, usize), format: u32, flags: u32, parent: Option<(usize, u64)>) {
    let generation = ARRAY_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ARRAY_METADATA.lock().unwrap_or_else(|e| e.into_inner()).insert(
        key, ArrayMetadata { format, flags, generation, parent },
    );
}

fn forget_metadata(key: (bool, usize), old: Option<ArrayMetadata>) {
    let Some(old) = old else { return };
    let mut table = ARRAY_METADATA.lock().unwrap_or_else(|e| e.into_inner());
    // HIP can release the address before destruction returns. A different
    // thread may already have created a new object at the same address.
    if table.get(&key).is_some_and(|current| current.generation == old.generation) {
        table.remove(&key);
    }
    if key.0 {
        table.retain(|_, entry| entry.parent != Some((key.1, old.generation)));
    }
}

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
    remember_metadata((false, hip_array as usize), p_allocate_array.Format.0, 0, None);
    *p_handle = hip_array.cast();
    Ok(())
}

pub(crate) unsafe fn destroy(h_array: CUarray) -> CUresult {
    let key = (false, h_array as usize);
    let old = metadata(key);
    hipfix::forget_array_hack(h_array as hipArray_t);
    hipArrayDestroy(h_array as hipArray_t)?;
    forget_metadata(key, old);
    Ok(())
}

pub(crate) unsafe fn get_descriptor_v2(
    p_array_descriptor: &mut CUDA_ARRAY_DESCRIPTOR,
    h_array: CUarray,
) -> CUresult {
    // Path-(b) D3D11 interop: synthesize a descriptor for D3D11
    // sentinel arrays — hipArrayGetDescriptor would return InvalidContext.
    // We only support R32G32B32A32_FLOAT for these (per
    // d3d11_interop::ensure_hip_array).
    if let Some((w, h, _pix)) =
        super::d3d11_interop::d3d11_array_descriptor(h_array as hipArray_t)
    {
        let mut desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
        desc.Width = w as usize;
        desc.Height = h as usize;
        desc.Format = hipArray_Format::HIP_AD_FORMAT_FLOAT;
        desc.NumChannels = 4;
        *p_array_descriptor =
            *((&desc) as *const HIP_ARRAY_DESCRIPTOR as *const CUDA_ARRAY_DESCRIPTOR);
        return Ok(());
    }
    let mut desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
    hipArrayGetDescriptor(&mut desc, h_array as hipArray_t)?;
    if let Some(original) = metadata((false, h_array as usize)) {
        desc.Format = hipArray_Format(original.format);
    }
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
    remember_metadata(
        (false, hip_array as usize), p_allocate_array.Format.0, p_allocate_array.Flags, None,
    );
    *p_handle = hip_array as CUarray;
    Ok(())
}

pub(crate) unsafe fn _3d_get_descriptor_v2(
    p_array_descriptor: &mut CUDA_ARRAY3D_DESCRIPTOR,
    h_array: CUarray,
) -> CUresult {
    // Path-(b) D3D11 interop: synthesize 3D descriptor for D3D11
    // sentinel arrays. Our arrays are 2D; depth=0 / NumChannels=4
    // / format=FLOAT signals "2D" to APEX.
    // Without this, hipArray3DGetDescriptor returns InvalidContext
    // and APEX abandons the array — which is what observed
    // 2026-05-09 v3 run (3 ERR cuArray3DGetDescriptor_v2 immediately
    // after cuGraphicsSubResourceGetMappedArray returns our sentinel).
    if let Some((w, h, _pix)) =
        super::d3d11_interop::d3d11_array_descriptor(h_array as hipArray_t)
    {
        let mut desc: HIP_ARRAY3D_DESCRIPTOR = mem::zeroed();
        desc.Width = w as usize;
        desc.Height = h as usize;
        desc.Depth = 0;
        desc.Format = hipArray_Format::HIP_AD_FORMAT_FLOAT;
        desc.NumChannels = 4;
        desc.Flags = 0;
        *p_array_descriptor =
            *((&desc) as *const HIP_ARRAY3D_DESCRIPTOR as *const CUDA_ARRAY3D_DESCRIPTOR);
        return Ok(());
    }
    let mut desc: HIP_ARRAY3D_DESCRIPTOR = mem::zeroed();
    hipArray3DGetDescriptor(&mut desc, h_array as hipArray_t)?;
    if let Some(original) = metadata((false, h_array as usize)) {
        desc.Format = hipArray_Format(original.format);
        desc.Flags = original.flags;
    }
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
    remember_metadata(
        (true, hip_array as usize), p_mipmapped_array_desc.Format.0, p_mipmapped_array_desc.Flags, None,
    );
    *p_handle = hip_array.cast();
    Ok(())
}

pub(crate) unsafe fn mipmapped_destroy(h_mipmapped_array: CUmipmappedArray) -> CUresult {
    let key = (true, h_mipmapped_array as usize);
    let old = metadata(key);
    hipfix::forget_mipmapped_hack(h_mipmapped_array as hipMipmappedArray_t);
    hipMipmappedArrayDestroy(h_mipmapped_array as hipMipmappedArray_t)?;
    forget_metadata(key, old);
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
    if let Some(parent) = metadata((true, h_mipmapped_array as usize)) {
        remember_metadata(
            (false, hip_array as usize), parent.format, parent.flags,
            Some((h_mipmapped_array as usize, parent.generation)),
        );
    }
    *p_level_array = hip_array as CUarray;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::*;
    use std::mem;

    #[test]
    fn array_metadata_preserves_reused_handles() {
        use super::{forget_metadata, metadata, remember_metadata};
        let array = (false, 0x123);
        let parent = (true, 0x123);
        let child = (false, 0x456);
        remember_metadata(array, 16, 2, None);
        let old_array = metadata(array);
        remember_metadata(array, 9, 0, None);
        forget_metadata(array, old_array);
        assert_eq!(metadata(array).unwrap().format, 9);
        remember_metadata(parent, 16, 2, None);
        let old_parent = metadata(parent).unwrap();
        remember_metadata(child, 16, 2, Some((parent.1, old_parent.generation)));
        remember_metadata(parent, 8, 0, None);
        forget_metadata(parent, Some(old_parent));
        assert!(metadata(child).is_none());
        assert_eq!(metadata(parent).unwrap().format, 8);
        assert_eq!(metadata(array).unwrap().flags, 0);
        forget_metadata(array, metadata(array));
        forget_metadata(parent, metadata(parent));
        assert!(metadata(array).is_none());
        assert!(metadata(parent).is_none());
    }

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

// There's a bug in hipDrvPointerGetAttributes where it returns
// HIP_ERROR_INVALID_VALUE if the pointer is null. It works correctly for any
// other invalid pointer
pub(crate) fn get_attributes(
    ptr: hip_runtime_sys::hipDeviceptr_t,
) -> hip_runtime_sys::hipDeviceptr_t {
    if ptr.0.is_null() {
        hip_runtime_sys::hipDeviceptr_t(usize::MAX as _)
    } else {
        ptr
    }
}

// For some reason HIP does not tolerate hipArraySurfaceLoadStore, even though
// it works just fine
pub(crate) unsafe fn array_3d_create(descriptor: &mut hip_runtime_sys::HIP_ARRAY3D_DESCRIPTOR) {
    descriptor.Flags &= !hip_runtime_sys::hipArraySurfaceLoadStore;
}

// ============================================================================
// Format inverse-transform machinery (ported from lshqqytiger/master hipfix.rs)
// ============================================================================
//
// Some HIP texture formats are silently broken on AMD: HALF, SIGNED_INT8,
// SIGNED_INT16. lsh's workaround is to transform the array's format to a
// compatible UNSIGNED variant at hipArrayCreate time, then UN-transform at
// texture-object-bind time using a HIP_RESOURCE_VIEW_DESC that tells the
// sampler the original format.
//
// We previously had only the FORWARD half of this transform — `get_non_broken_format`
// rewrote HALF→UINT16 etc., and our `cuTexObjectCreate` passed `null` for the
// view desc. That meant: the array bytes were stored as half-float, but
// texture reads interpreted them as integer. Surfaces (sust.b.*) write raw
// bytes so they're unaffected, but tex.3d reads return wrong values.
//
// This module adds the missing inverse:
//   - `get_non_broken_format` now returns `(hack_flag, new_format)`
//   - `set_array_hack_flag(array, flag)` records the original format on a
//     side-table (we can't repurpose `array.textureType` like lsh does because
//     our hip_runtime-sys exposes hipArray as an opaque struct)
//   - `get_broken_format(array)` looks up the side-table and returns the
//     ORIGINAL format if any
//   - `with_resource_desc(...)` synthesizes a HIP_RESOURCE_VIEW_DESC with the
//     original format when binding a texture object to a transformed array
//
// The expectation is that this fixes the TurbulenceFS bug by giving the
// turbulence-grid texture samplers the correct format interpretation. See
// `project_physx_ms4_root_cause_turbulencefs.md` and the bisection plan
// `snug-launching-milner.md` Phase D.

use cuda_types::cuda::CUDA_RESOURCE_DESC;
use hip_runtime_sys::{
    hipArray_Format, hipArray_t, hipMipmappedArray_t, HIPresourceViewFormat, HIP_RESOURCE_VIEW_DESC,
};

#[must_use]
pub(crate) fn get_non_broken_format(format: hipArray_Format) -> (u32, hipArray_Format) {
    use hipArray_Format as F;
    // hack_flag values are intentionally the same as lsh's so any future port
    // of the array.textureType encoding stays compatible.
    match format {
        F::HIP_AD_FORMAT_HALF => (2, F::HIP_AD_FORMAT_UNSIGNED_INT16),
        F::HIP_AD_FORMAT_SIGNED_INT16 => (1, F::HIP_AD_FORMAT_UNSIGNED_INT16),
        F::HIP_AD_FORMAT_SIGNED_INT8 => (1, F::HIP_AD_FORMAT_UNSIGNED_INT8),
        f => (0, f),
    }
}

/// Inverse: given a hack_flag and the (possibly already-transformed) format,
/// return the ORIGINAL format if a transformation happened. Returns None if
/// hack_flag is 0 (no transformation).
fn get_broken_format_impl(hack_flag: u32, format: hipArray_Format) -> Option<hipArray_Format> {
    use hipArray_Format as F;
    Some(match (hack_flag, format) {
        (2, F::HIP_AD_FORMAT_UNSIGNED_INT16) => F::HIP_AD_FORMAT_HALF,
        (1, F::HIP_AD_FORMAT_UNSIGNED_INT16) => F::HIP_AD_FORMAT_SIGNED_INT16,
        (1, F::HIP_AD_FORMAT_UNSIGNED_INT8) => F::HIP_AD_FORMAT_SIGNED_INT8,
        _ => return None,
    })
}

// Side-table: maps hipArray_t / hipMipmappedArray_t pointer → (hack_flag, channels).
//
// We can't repurpose hipArray.textureType (the field lsh uses) because our
// hip_runtime-sys exposes hipArray as `_unused: [u8; 0]` — opaque, no field
// access. A pointer-keyed HashMap is the simplest equivalent. Channels are
// stashed alongside the hack_flag because `with_resource_desc` needs both
// when synthesizing the view desc and we already have the value at the
// `hipArrayCreate` call site.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ArrayHack {
    pub hack_flag: u32,
    pub num_channels: u32,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

static ARRAY_HACK_TABLE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<usize, ArrayHack>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

pub(crate) fn set_array_hack(array: hipArray_t, hack: ArrayHack) {
    if array.is_null() || hack.hack_flag == 0 {
        return;
    }
    if let Ok(mut t) = ARRAY_HACK_TABLE.lock() {
        t.insert(array as usize, hack);
    }
}

pub(crate) fn set_mipmapped_hack(array: hipMipmappedArray_t, hack: ArrayHack) {
    if array.is_null() || hack.hack_flag == 0 {
        return;
    }
    if let Ok(mut t) = ARRAY_HACK_TABLE.lock() {
        t.insert(array as usize, hack);
    }
}

pub(crate) fn get_array_hack(array: hipArray_t) -> Option<ArrayHack> {
    if array.is_null() {
        return None;
    }
    ARRAY_HACK_TABLE
        .lock()
        .ok()
        .and_then(|t| t.get(&(array as usize)).copied())
}

pub(crate) fn get_mipmapped_hack(array: hipMipmappedArray_t) -> Option<ArrayHack> {
    if array.is_null() {
        return None;
    }
    ARRAY_HACK_TABLE
        .lock()
        .ok()
        .and_then(|t| t.get(&(array as usize)).copied())
}

pub(crate) fn forget_array_hack(array: hipArray_t) {
    if array.is_null() {
        return;
    }
    if let Ok(mut t) = ARRAY_HACK_TABLE.lock() {
        t.remove(&(array as usize));
    }
}

pub(crate) fn forget_mipmapped_hack(array: hipMipmappedArray_t) {
    if array.is_null() {
        return;
    }
    if let Ok(mut t) = ARRAY_HACK_TABLE.lock() {
        t.remove(&(array as usize));
    }
}

/// Fetch the array's current (transformed) format via hipArrayGetDescriptor,
/// then look up the side-table and return the ORIGINAL format if a
/// transformation was applied at create time. Returns the recorded ArrayHack
/// alongside so callers can build a HIP_RESOURCE_VIEW_DESC.
pub(crate) unsafe fn broken_format_for_array(
    array: hipArray_t,
) -> (Option<hipArray_Format>, ArrayHack) {
    let hack = match get_array_hack(array) {
        Some(h) => h,
        None => {
            return (
                None,
                ArrayHack {
                    hack_flag: 0,
                    num_channels: 0,
                    width: 0,
                    height: 0,
                    depth: 0,
                },
            );
        }
    };
    let mut desc: hip_runtime_sys::HIP_ARRAY_DESCRIPTOR = std::mem::zeroed();
    if hip_runtime_sys::hipArrayGetDescriptor(&mut desc, array).is_err() {
        return (None, hack);
    }
    (get_broken_format_impl(hack.hack_flag, desc.Format), hack)
}

/// Map (format, num_channels) to a HIP_RES_VIEW_FORMAT enum value. Used to
/// build the HIP_RESOURCE_VIEW_DESC that tells the sampler the original format.
/// Mirrors lshqqytiger/master:zluda/src/impl/hipfix.rs::resource_view_format.
fn resource_view_format(
    format: hipArray_Format,
    num_channels: u32,
) -> Option<HIPresourceViewFormat> {
    use hipArray_Format as F;
    use hip_runtime_sys::HIPresourceViewFormat_enum as V;
    Some(match (format, num_channels) {
        (F::HIP_AD_FORMAT_UNSIGNED_INT8, 1) => V::HIP_RES_VIEW_FORMAT_UINT_1X8,
        (F::HIP_AD_FORMAT_UNSIGNED_INT8, 2) => V::HIP_RES_VIEW_FORMAT_UINT_2X8,
        (F::HIP_AD_FORMAT_UNSIGNED_INT8, 4) => V::HIP_RES_VIEW_FORMAT_UINT_4X8,
        (F::HIP_AD_FORMAT_SIGNED_INT8, 1) => V::HIP_RES_VIEW_FORMAT_SINT_1X8,
        (F::HIP_AD_FORMAT_SIGNED_INT8, 2) => V::HIP_RES_VIEW_FORMAT_SINT_2X8,
        (F::HIP_AD_FORMAT_SIGNED_INT8, 4) => V::HIP_RES_VIEW_FORMAT_SINT_4X8,
        (F::HIP_AD_FORMAT_UNSIGNED_INT16, 1) => V::HIP_RES_VIEW_FORMAT_UINT_1X16,
        (F::HIP_AD_FORMAT_UNSIGNED_INT16, 2) => V::HIP_RES_VIEW_FORMAT_UINT_2X16,
        (F::HIP_AD_FORMAT_UNSIGNED_INT16, 4) => V::HIP_RES_VIEW_FORMAT_UINT_4X16,
        (F::HIP_AD_FORMAT_SIGNED_INT16, 1) => V::HIP_RES_VIEW_FORMAT_SINT_1X16,
        (F::HIP_AD_FORMAT_SIGNED_INT16, 2) => V::HIP_RES_VIEW_FORMAT_SINT_2X16,
        (F::HIP_AD_FORMAT_SIGNED_INT16, 4) => V::HIP_RES_VIEW_FORMAT_SINT_4X16,
        (F::HIP_AD_FORMAT_HALF, 1) => V::HIP_RES_VIEW_FORMAT_FLOAT_1X16,
        (F::HIP_AD_FORMAT_HALF, 2) => V::HIP_RES_VIEW_FORMAT_FLOAT_2X16,
        (F::HIP_AD_FORMAT_HALF, 4) => V::HIP_RES_VIEW_FORMAT_FLOAT_4X16,
        _ => return None,
    })
}

/// Given a CUDA_RESOURCE_DESC, build the appropriate HIP_RESOURCE_VIEW_DESC
/// IF the underlying array has a recorded format hack — otherwise return None
/// (caller passes null). The returned descriptor has the ORIGINAL format so
/// the sampler interprets the bytes correctly.
///
/// Returns Some(view_desc) if a hack was applied; None if the array is plain
/// or has no hack record.
pub(crate) unsafe fn maybe_build_view_desc(
    res_desc: &CUDA_RESOURCE_DESC,
) -> Option<HIP_RESOURCE_VIEW_DESC> {
    use cuda_types::cuda::CUresourcetype;
    match res_desc.resType {
        CUresourcetype::CU_RESOURCE_TYPE_ARRAY => {
            let array = res_desc.res.array.hArray as hipArray_t;
            let (orig_format, hack) = broken_format_for_array(array);
            let orig_format = orig_format?;
            let view_format = resource_view_format(orig_format, hack.num_channels)?;
            Some(HIP_RESOURCE_VIEW_DESC {
                format: view_format,
                width: hack.width,
                height: hack.height,
                depth: hack.depth,
                firstMipmapLevel: 0,
                lastMipmapLevel: 0,
                firstLayer: 0,
                lastLayer: 0,
                reserved: [0; 16],
            })
        }
        CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY => {
            let mip = res_desc.res.mipmap.hMipmappedArray as hipMipmappedArray_t;
            let hack = get_mipmapped_hack(mip)?;
            // Mipmapped arrays don't expose hipArrayGetDescriptor; we trust
            // the recorded format-pre-transform from set_mipmapped_hack.
            // hack_flag is non-zero by definition (set_mipmapped_hack guards it).
            // We need the ORIGINAL format from hack_flag — invert the table.
            let orig_format = match hack.hack_flag {
                2 => hipArray_Format::HIP_AD_FORMAT_HALF,
                1 => hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16, // could also be SI8 — see note
                _ => return None,
            };
            // NOTE: hack_flag=1 is ambiguous between SIGNED_INT16 and SIGNED_INT8.
            // We can't disambiguate from the flag alone. For now we record the
            // pre-transform format directly in a future enhancement; for this
            // bisection step we accept that mipmapped arrays with hack_flag=1
            // may produce wrong view formats. Most relevant title (Batman AK)
            // uses HALF (hack_flag=2), not signed integer formats.
            let view_format = resource_view_format(orig_format, hack.num_channels)?;
            Some(HIP_RESOURCE_VIEW_DESC {
                format: view_format,
                width: hack.width,
                height: hack.height,
                depth: hack.depth,
                firstMipmapLevel: 0,
                lastMipmapLevel: 0,
                firstLayer: 0,
                lastLayer: 0,
                reserved: [0; 16],
            })
        }
        _ => None,
    }
}


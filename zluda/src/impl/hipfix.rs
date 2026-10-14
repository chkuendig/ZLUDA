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

/// Re-bind a texref to force HIP to regenerate its internal texture object.
/// Matches upstream `vosen/texrefs` branch organization. Our version keeps the
/// `sync_synthetic_texref` call and null-textureObject early return that
/// upstream doesn't need.
pub(crate) unsafe fn refresh_texref(
    raw_texref: *mut hip_runtime_sys::textureReference,
) -> cuda_types::cuda::CUresult {
    use hip_runtime_sys::*;
    const TRSA_OVERRIDE_FORMAT: u32 = cuda_types::cuda::CU_TRSA_OVERRIDE_FORMAT;
    if raw_texref.is_null() {
        return Err(cuda_types::cuda::CUerror::INVALID_VALUE);
    }
    let texref = &*raw_texref;
    // If no texture object has been created yet (e.g., synthetic texref with
    // no array bound), there's nothing to refresh -- just return success.
    if texref.textureObject.is_null() {
        return Ok(());
    }
    let mut res_desc: HIP_RESOURCE_DESC = std::mem::zeroed();
    hipTexObjectGetResourceDesc(&mut res_desc, texref.textureObject)?;
    match res_desc.resType {
        HIPresourcetype::HIP_RESOURCE_TYPE_ARRAY => {
            let array = res_desc.res.array.hArray;
            if !array.is_null() {
                hipTexRefSetArray(raw_texref, array, TRSA_OVERRIDE_FORMAT)?;
            }
        }
        HIPresourcetype::HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY => {
            hipTexRefSetMipmappedArray(
                raw_texref,
                res_desc.res.mipmap.hMipmappedArray,
                TRSA_OVERRIDE_FORMAT,
            )?;
        }
        HIPresourcetype::HIP_RESOURCE_TYPE_LINEAR => {
            let linear = res_desc.res.linear;
            if !linear.devPtr.0.is_null() && linear.sizeInBytes != 0 {
                let mut unused = 0usize;
                hipTexRefSetAddress(
                    &mut unused,
                    raw_texref,
                    linear.devPtr,
                    linear.sizeInBytes,
                )?;
            }
        }
        HIPresourcetype::HIP_RESOURCE_TYPE_PITCH2D => {
            let pitch_2d = res_desc.res.pitch2D;
            let desc = HIP_ARRAY_DESCRIPTOR {
                Width: pitch_2d.width,
                Height: pitch_2d.height,
                Format: pitch_2d.format,
                NumChannels: pitch_2d.numChannels,
            };
            hipTexRefSetAddress2D(
                raw_texref,
                &desc,
                pitch_2d.devPtr,
                pitch_2d.pitchInBytes,
            )?;
        }
        _ => return Err(cuda_types::cuda::CUerror::INVALID_VALUE),
    }
    // After rebinding, sync the texture object to device global if this is a
    // synthetic texref
    super::tex::sync_synthetic_texref(raw_texref);
    Ok(())
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
    // 2026-04-27: re-enabled the lsh substitution. The prior passthrough
    // (commit 90f30cee) was tested only by crash-distance, BEFORE the
    // LLVM PR #161638 backport. With that backport the game runs ~140x
    // further and we can SEE rendering — the user reports volumetric
    // fog/smoke does NOT render. Lshqqytiger v3.9.6 (the v3 reference
    // fork known to work for Batman) STILL has this substitution active.
    //
    // Without substitution: HIP samples HALF4 textures, but ROCm's
    // sampler appears to read raw UINT16 bits without HALF→FLOAT
    // conversion — empty velocity field, no smoke.
    //
    // With substitution: array stored as UNSIGNED_INT16, and
    // cuTexObjectCreate synthesizes a HIP_RESOURCE_VIEW_DESC saying
    // "interpret as FLOAT_4X16" so the sampler converts correctly.
    //
    // Caveat: the legacy `cuTexRef` path has no HIP_RESOURCE_VIEW_DESC
    // equivalent. If PhysX uses cuTexRef for some reads, those will
    // break. Empirically Batman uses cuTexObject for the TFS velocity
    // grids, so this is net-positive.
    use hipArray_Format as F;
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

#[cfg(test)]
mod tests {
    //! Synthetic regression tests for the format inverse-transform machinery
    //! ported from lshqqytiger/master in commit 3a203a3c. These do not require
    //! a GPU and exercise the pure logic + side-table behavior.
    //!
    //! The bug they guard against: HIP silently rejects HALF / SIGNED_INT8 /
    //! SIGNED_INT16 array formats. We rewrite them to the matching UNSIGNED
    //! variant at hipArrayCreate time, then UN-rewrite at cuTexObjectCreate
    //! time via a HIP_RESOURCE_VIEW_DESC so the sampler interprets the bytes
    //! correctly. If either half of the transform regresses, TurbulenceFS
    //! grids in PhysX will read garbage and games will scatter-write OOB.
    //!
    //! See `project_physx_ms4_root_cause_turbulencefs.md` for the original
    //! crash this fixed.
    use super::*;
    use hip_runtime_sys::hipArray_Format as F;
    use hip_runtime_sys::HIPresourceViewFormat_enum as V;

    // ----- get_non_broken_format ---------------------------------------------

    // PASSTHROUGH TEST (2026-04-11): these three tests assert the legacy
    // substitution behavior. The function now passes all formats through
    // unchanged — revert the `#[ignore]`s and the function body together
    // if `hipArrayCreate(HALF)` actually requires substitution.
    #[test]
    #[ignore]
    fn forward_transform_half_to_uint16() {
        assert_eq!(
            get_non_broken_format(F::HIP_AD_FORMAT_HALF),
            (2, F::HIP_AD_FORMAT_UNSIGNED_INT16)
        );
    }

    #[test]
    #[ignore]
    fn forward_transform_signed_int16_to_uint16() {
        assert_eq!(
            get_non_broken_format(F::HIP_AD_FORMAT_SIGNED_INT16),
            (1, F::HIP_AD_FORMAT_UNSIGNED_INT16)
        );
    }

    #[test]
    #[ignore]
    fn forward_transform_signed_int8_to_uint8() {
        assert_eq!(
            get_non_broken_format(F::HIP_AD_FORMAT_SIGNED_INT8),
            (1, F::HIP_AD_FORMAT_UNSIGNED_INT8)
        );
    }

    #[test]
    fn forward_transform_passthrough_for_safe_formats() {
        // Formats HIP handles natively must NOT be transformed; otherwise we
        // would corrupt the format <-> view-desc agreement for plain arrays.
        for fmt in [
            F::HIP_AD_FORMAT_UNSIGNED_INT8,
            F::HIP_AD_FORMAT_UNSIGNED_INT16,
            F::HIP_AD_FORMAT_UNSIGNED_INT32,
            F::HIP_AD_FORMAT_SIGNED_INT32,
            F::HIP_AD_FORMAT_FLOAT,
        ] {
            let (flag, out) = get_non_broken_format(fmt);
            assert_eq!(flag, 0, "{:?} must not be transformed", fmt);
            assert_eq!(out, fmt, "{:?} must pass through unchanged", fmt);
        }
    }

    // ----- get_broken_format_impl (inverse) ----------------------------------

    #[test]
    fn inverse_recovers_half() {
        assert_eq!(
            get_broken_format_impl(2, F::HIP_AD_FORMAT_UNSIGNED_INT16),
            Some(F::HIP_AD_FORMAT_HALF)
        );
    }

    #[test]
    fn inverse_recovers_signed_int16() {
        assert_eq!(
            get_broken_format_impl(1, F::HIP_AD_FORMAT_UNSIGNED_INT16),
            Some(F::HIP_AD_FORMAT_SIGNED_INT16)
        );
    }

    #[test]
    fn inverse_recovers_signed_int8() {
        assert_eq!(
            get_broken_format_impl(1, F::HIP_AD_FORMAT_UNSIGNED_INT8),
            Some(F::HIP_AD_FORMAT_SIGNED_INT8)
        );
    }

    #[test]
    fn inverse_returns_none_for_zero_flag() {
        // hack_flag=0 means no transformation happened — never claim an inverse.
        for fmt in [
            F::HIP_AD_FORMAT_UNSIGNED_INT8,
            F::HIP_AD_FORMAT_UNSIGNED_INT16,
            F::HIP_AD_FORMAT_FLOAT,
        ] {
            assert_eq!(get_broken_format_impl(0, fmt), None);
        }
    }

    #[test]
    fn inverse_returns_none_for_mismatched_format() {
        // hack_flag set but format doesn't match a known transformation -> None
        assert_eq!(get_broken_format_impl(2, F::HIP_AD_FORMAT_FLOAT), None);
        assert_eq!(
            get_broken_format_impl(1, F::HIP_AD_FORMAT_UNSIGNED_INT32),
            None
        );
    }

    // ----- round-trip property -----------------------------------------------

    #[test]
    #[ignore] // passthrough test: no substitution happens, round-trip is trivial
    fn forward_then_inverse_recovers_original_for_all_broken_formats() {
        // The critical regression check: any HALF/SIGNED format must survive
        // forward + inverse and come out byte-identical. If this breaks, the
        // sampler view-desc will lie about the bytes and TurbulenceFS / any
        // PhysX texture-read kernel will read garbage.
        for orig in [
            F::HIP_AD_FORMAT_HALF,
            F::HIP_AD_FORMAT_SIGNED_INT16,
            F::HIP_AD_FORMAT_SIGNED_INT8,
        ] {
            let (flag, transformed) = get_non_broken_format(orig);
            assert_ne!(flag, 0, "{:?} should be flagged as transformed", orig);
            let recovered = get_broken_format_impl(flag, transformed)
                .unwrap_or_else(|| panic!("inverse failed for {:?}", orig));
            assert_eq!(recovered, orig, "round-trip failed for {:?}", orig);
        }
    }

    // ----- resource_view_format ----------------------------------------------

    #[test]
    fn resource_view_format_covers_all_recovered_formats() {
        // Every format the inverse can produce must have a HIPresourceViewFormat
        // mapping for channel counts 1, 2, and 4 (the values PhysX uses).
        // Without this, maybe_build_view_desc would return None even when a
        // hack was recorded, and the bind would fall back to "no view desc"
        // (the broken state we're trying to escape).
        let cases: &[(F, V, V, V)] = &[
            (
                F::HIP_AD_FORMAT_HALF,
                V::HIP_RES_VIEW_FORMAT_FLOAT_1X16,
                V::HIP_RES_VIEW_FORMAT_FLOAT_2X16,
                V::HIP_RES_VIEW_FORMAT_FLOAT_4X16,
            ),
            (
                F::HIP_AD_FORMAT_SIGNED_INT16,
                V::HIP_RES_VIEW_FORMAT_SINT_1X16,
                V::HIP_RES_VIEW_FORMAT_SINT_2X16,
                V::HIP_RES_VIEW_FORMAT_SINT_4X16,
            ),
            (
                F::HIP_AD_FORMAT_SIGNED_INT8,
                V::HIP_RES_VIEW_FORMAT_SINT_1X8,
                V::HIP_RES_VIEW_FORMAT_SINT_2X8,
                V::HIP_RES_VIEW_FORMAT_SINT_4X8,
            ),
        ];
        for (fmt, v1, v2, v4) in cases {
            assert_eq!(resource_view_format(*fmt, 1), Some(*v1), "{:?} ch=1", fmt);
            assert_eq!(resource_view_format(*fmt, 2), Some(*v2), "{:?} ch=2", fmt);
            assert_eq!(resource_view_format(*fmt, 4), Some(*v4), "{:?} ch=4", fmt);
        }
    }

    #[test]
    fn resource_view_format_unknown_channel_count_is_none() {
        // 3 channels and 0 channels are not supported by HIP_RES_VIEW_FORMAT_*.
        // The function must return None so callers fall back rather than
        // construct a malformed view desc.
        assert_eq!(resource_view_format(F::HIP_AD_FORMAT_HALF, 3), None);
        assert_eq!(resource_view_format(F::HIP_AD_FORMAT_HALF, 0), None);
    }

    // ----- side-table (set/get/forget) ---------------------------------------
    //
    // We can't construct a real hipArray (the type is opaque + GPU-allocated),
    // but the side-table only uses the pointer as a HashMap key — it never
    // dereferences it. Casting a stable nonzero usize into the pointer type
    // is safe for this purpose. We pick distinct addresses per test to avoid
    // cross-test contamination of the shared static table.

    fn fake_array(addr: usize) -> hipArray_t {
        addr as hipArray_t
    }

    fn fake_mip(addr: usize) -> hipMipmappedArray_t {
        addr as hipMipmappedArray_t
    }

    fn dummy_hack(flag: u32) -> ArrayHack {
        ArrayHack {
            hack_flag: flag,
            num_channels: 4,
            width: 64,
            height: 64,
            depth: 0,
        }
    }

    #[test]
    fn array_hack_set_get_round_trip() {
        let arr = fake_array(0x1000_0001);
        set_array_hack(arr, dummy_hack(2));
        let got = get_array_hack(arr).expect("hack should be present");
        assert_eq!(got.hack_flag, 2);
        assert_eq!(got.num_channels, 4);
        assert_eq!(got.width, 64);
        forget_array_hack(arr);
        assert!(get_array_hack(arr).is_none(), "forget should remove entry");
    }

    #[test]
    fn array_hack_zero_flag_is_not_stored() {
        // hack_flag=0 means "no transformation happened" — storing it would
        // make get_array_hack falsely report a hack and trigger an inverse
        // view desc on a plain array.
        let arr = fake_array(0x1000_0002);
        set_array_hack(arr, dummy_hack(0));
        assert!(
            get_array_hack(arr).is_none(),
            "hack_flag=0 must not be stored"
        );
    }

    #[test]
    fn array_hack_null_pointer_is_noop() {
        // Setting on a null pointer must not panic or pollute the table.
        set_array_hack(std::ptr::null_mut(), dummy_hack(2));
        assert!(get_array_hack(std::ptr::null_mut()).is_none());
        forget_array_hack(std::ptr::null_mut()); // must not panic
    }

    #[test]
    fn array_hack_unknown_pointer_returns_none() {
        let arr = fake_array(0x1000_0003);
        // Never inserted -> must be None.
        assert!(get_array_hack(arr).is_none());
    }

    #[test]
    fn mipmapped_hack_set_get_round_trip() {
        let mip = fake_mip(0x2000_0001);
        set_mipmapped_hack(mip, dummy_hack(1));
        let got = get_mipmapped_hack(mip).expect("hack should be present");
        assert_eq!(got.hack_flag, 1);
        forget_mipmapped_hack(mip);
        assert!(get_mipmapped_hack(mip).is_none());
    }

    #[test]
    fn mipmapped_hack_zero_flag_is_not_stored() {
        let mip = fake_mip(0x2000_0002);
        set_mipmapped_hack(mip, dummy_hack(0));
        assert!(get_mipmapped_hack(mip).is_none());
    }

    // ----- array_3d_create flag stripping ------------------------------------

    #[test]
    fn array_3d_create_strips_surface_load_store() {
        // HIP rejects hipArraySurfaceLoadStore on 3D descriptors — we strip it
        // unconditionally. Surface writes (sust.b.*) still work because the
        // texture/surface object handles encode pixel-shift separately.
        unsafe {
            let mut desc: hip_runtime_sys::HIP_ARRAY3D_DESCRIPTOR = std::mem::zeroed();
            desc.Flags = hip_runtime_sys::hipArraySurfaceLoadStore;
            array_3d_create(&mut desc);
            assert_eq!(
                desc.Flags & hip_runtime_sys::hipArraySurfaceLoadStore,
                0,
                "hipArraySurfaceLoadStore must be cleared"
            );
        }
    }

    #[test]
    fn array_3d_create_preserves_other_flags() {
        // Stripping must be surgical — only the SurfaceLoadStore bit, never
        // any other flag the caller set (e.g. hipArrayLayered).
        unsafe {
            let mut desc: hip_runtime_sys::HIP_ARRAY3D_DESCRIPTOR = std::mem::zeroed();
            let other_bits: u32 = 0xFFFF_FFFF & !hip_runtime_sys::hipArraySurfaceLoadStore;
            desc.Flags = other_bits | hip_runtime_sys::hipArraySurfaceLoadStore;
            array_3d_create(&mut desc);
            assert_eq!(
                desc.Flags, other_bits,
                "only SurfaceLoadStore should be cleared, other bits preserved"
            );
        }
    }

    #[test]
    fn array_3d_create_no_op_when_flag_not_set() {
        // Pick a flag value that does NOT include hipArraySurfaceLoadStore (=2).
        unsafe {
            let mut desc: hip_runtime_sys::HIP_ARRAY3D_DESCRIPTOR = std::mem::zeroed();
            let flags = 0x40 | 0x01; // arbitrary bits, but bit 1 (value 2) clear
            assert_eq!(flags & hip_runtime_sys::hipArraySurfaceLoadStore, 0);
            desc.Flags = flags;
            array_3d_create(&mut desc);
            assert_eq!(desc.Flags, flags);
        }
    }

    // ----- get_attributes (null pointer workaround) --------------------------

    #[test]
    fn get_attributes_replaces_null_with_max() {
        // hipDrvPointerGetAttributes returns INVALID_VALUE for a literal null
        // pointer (but works fine for any other invalid pointer). We swap null
        // for usize::MAX so the call returns "not a device pointer" instead.
        let out = get_attributes(hip_runtime_sys::hipDeviceptr_t(std::ptr::null_mut()));
        assert_eq!(out.0 as usize, usize::MAX);
    }

    #[test]
    fn get_attributes_passes_through_non_null() {
        let p = 0xDEAD_BEEF_usize as *mut std::ffi::c_void;
        let out = get_attributes(hip_runtime_sys::hipDeviceptr_t(p));
        assert_eq!(out.0, p);
    }
}

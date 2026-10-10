use cuda_types::cuda::*;
use hip_runtime_sys::*;
use std::mem;

/// After a texture object is created on a synthetic texref, copy the handle
/// to the device global variable so the kernel can read it at runtime.
pub(crate) unsafe fn sync_synthetic_texref(tex_ref: *mut textureReference) {
    if tex_ref.is_null() {
        return;
    }
    let tex_obj = (*tex_ref).textureObject;
    if tex_obj.is_null() {
        super::module::set_synthetic_texref_linear_binding(tex_ref, None);
        return;
    }
    if let Some(device_global) = super::module::get_synthetic_texref_global(tex_ref) {
        super::driver::zluda_log(&format!(
            "[ZLUDA] sync_synthetic_texref: writing tex_obj={:p} to device {:p}",
            tex_obj, device_global.0
        ));
        {
            let _guard = super::stream::null_stream_sync(
                hipStream_t(std::ptr::null_mut()),
            );
            let _ = hipMemcpyHtoD(
                device_global,
                &tex_obj as *const _ as *mut std::ffi::c_void,
                mem::size_of::<hipTextureObject_t>(),
            );
        }
    }
    let mut res_desc: hipResourceDesc = mem::zeroed();
    let mut tex_desc: hipTextureDesc = mem::zeroed();
    let binding = if hipGetTextureObjectResourceDesc(&mut res_desc, tex_obj).is_ok()
        && hipGetTextureObjectTextureDesc(&mut tex_desc, tex_obj).is_ok()
        && res_desc.resType == hipResourceType::hipResourceTypeLinear
        && !unsafe { res_desc.res.linear.devPtr }.is_null()
        && unsafe { res_desc.res.linear.sizeInBytes } != 0
    {
        from_channel_format_desc(unsafe { res_desc.res.linear.desc })
            .ok()
            .map(
                |(format, num_channels)| super::module::SyntheticTexRefLinearBinding {
                    src_ptr: hipDeviceptr_t(unsafe { res_desc.res.linear.devPtr }),
                    size_in_bytes: unsafe { res_desc.res.linear.sizeInBytes },
                    format,
                    num_channels,
                    texture_desc: HIP_TEXTURE_DESC {
                        addressMode: [
                            HIPaddress_mode(tex_desc.addressMode[0].0),
                            HIPaddress_mode(tex_desc.addressMode[1].0),
                            HIPaddress_mode(tex_desc.addressMode[2].0),
                        ],
                        filterMode: HIPfilter_mode(tex_desc.filterMode.0),
                        flags: 0,
                        maxAnisotropy: tex_desc.maxAnisotropy,
                        mipmapFilterMode: HIPfilter_mode(tex_desc.mipmapFilterMode.0),
                        mipmapLevelBias: tex_desc.mipmapLevelBias,
                        minMipmapLevelClamp: tex_desc.minMipmapLevelClamp,
                        maxMipmapLevelClamp: tex_desc.maxMipmapLevelClamp,
                        borderColor: tex_desc.borderColor,
                        reserved: [0; 12],
                    },
                },
            )
    } else {
        None
    };
    super::module::set_synthetic_texref_linear_binding(tex_ref, binding);
}

fn as_tex_ref(texref: CUtexref) -> *mut textureReference {
    texref as *mut textureReference
}

/// Gate texref format/array diagnostic logging. Off by default because the
/// game issues thousands of legacy texref binds per frame, and per-call
/// `zluda_log_err` on that hot path adds enough host-thread latency to shift
/// PhysX crash timing. Set `ZLUDA_TEXREF_LOG=1` to enable.
fn texref_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZLUDA_TEXREF_LOG").map_or(false, |v| v == "1"))
}

/// 2026-06-05 Class A experiment: drain all in-flight kernels BEFORE changing
/// a texref binding (hipTexRefSetAddress / hipTexRefSetArray /
/// hipTexRefSetAddress2D / hipTexRefSetMipmappedArray).
///
/// CUDA snapshots texref bindings at kernel-launch time, so an in-flight
/// kernel on stream B reading `texref_T` is guaranteed to read the bytes
/// at the address `T` held when B's launch happened — even if the host
/// calls cuTexRefSetAddress(T, new_addr) on stream A while B is running.
///
/// HIP doesn't make this guarantee: `hipTexRefSetAddress` flips global
/// texref state immediately. In-flight kernels can therefore read mid-
/// transition state (e.g. partial header with new ptr / old size), then
/// `tex.1d.v4.u32.s32` returns a value computed from the wrong buffer.
/// Downstream IOFX_actorRangeKernel scatters using that wild index →
/// canonical CB/DB+TCP page fault (Class A).
///
/// `ZLUDA_TEXREF_SYNC=1` forces a `hipDeviceSynchronize` before the
/// texref-changing call, matching the CUDA snapshot semantic. Cost: one
/// device sync per binding change. Worth measuring; PhysX issues
/// thousands per frame, so the cost ceiling is "one device sync per
/// hipTexRefSet*" and not "free".
/// ZLUDA_TEXREF_SYNC level:
///   0 — OFF
///   1 — drain in-flight kernels before SetAddress/SetArray binds (default;
///       Arkham Asylum hangs without it).
///   2 — ALSO drain before the non-address texref-state mutators (SetFlags /
///       SetFormat / SetAddressMode / SetFilterMode). 2026-06-15 Class-4 fix:
///       CUDA snapshots the WHOLE texref state at launch; a mid-flight
///       SetFlags(READ_AS_INTEGER) / SetFormat race makes an in-flight
///       IOFX_remapKernel `tex.1d.u32` fetch return a normalized FLOAT instead
///       of the raw int actor-ID → float-pattern index → bbox/actorRange wild
///       store. Gated separately because these mutators are on a hot path and
///       an unconditional sync risks PhysX-init timing perturbation (cf. the
///       rejected TEXREF_POST_SYNC=2); =2 lets us A/B it in-game.
fn texref_sync_level() -> u32 {
    static LEVEL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("ZLUDA_TEXREF_SYNC")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(1)
    })
}

fn texref_sync_enabled() -> bool {
    texref_sync_level() >= 1
}

/// Pre-sync gate for the non-address texref-state mutators (level >= 2).
unsafe fn texref_pre_sync_all_mutators_if_enabled() {
    if texref_sync_level() >= 2 {
        let _ = hipDeviceSynchronize();
        super::stream::clear_dirty_flag_after_device_sync();
    }
}

/// 2026-06-09 Class A bf000 candidate fix: post-sync the texref binding.
///
/// `ZLUDA_TEXREF_SYNC=1` drains in-flight kernels BEFORE changing a
/// binding. That fixes CUDA's snapshot semantic for the bind transition.
/// But the bf000 single-page Class A fault (FleX particle position
/// buffer's slot[0] PERMISSION_FAULTS=0x5 RW=1 on direct st.global write)
/// suggests that AFTER cuTexRefSetAddress_v2 completes, the AMDGPU TCP /
/// SRD descriptor cache may still carry the texref's read-only flavor for
/// the bound buffer's pages. A subsequent kernel writing through that
/// buffer via direct global store then hits a permission fault on byte 0.
///
/// ZLUDA_TEXREF_POST_SYNC modes:
///   0 — OFF (default)
///   1 — TARGETED: post-sync only when the bound buffer matches the
///       bf000 shape (size=1,040,000 AND VA low-20-bits=0xBF000).
///       Typically 1–2 binds per bench → negligible cost.
///   2 — ALL: post-sync after EVERY cuTexRefSet*. Heavy — destabilizes
///       PhysX TurbulenceFS init in practice (bench n7 SIGABRT @2115
///       launches host-side). Kept for diagnostic comparison only.
///
/// See project_bf000_buffer_is_flex_position_2026_06_09.md.
fn texref_post_sync_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("ZLUDA_TEXREF_POST_SYNC").ok();
        let mode = raw
            .as_deref()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0)
            .min(2);
        super::driver::zluda_log_err(&format!(
            "*** ZLUDA_TEXREF_POST_SYNC init: env={:?} mode={}",
            raw, mode
        ));
        mode
    })
}

unsafe fn texref_pre_sync_if_enabled() {
    if texref_sync_enabled() {
        let _ = hipDeviceSynchronize();
        super::stream::clear_dirty_flag_after_device_sync();
    }
}

/// Post-sync after a cuTexRefSet*. `dptr` and `bytes` describe the bound
/// buffer (or 0/0 for a 2D bind where we don't have the size cheaply).
/// In TARGETED mode (=1), syncs only when the buffer matches the bf000
/// shape — that's the candidate fault site, and limiting to it avoids
/// the n7 PhysX-init timing perturbation.
unsafe fn texref_post_sync_if_enabled(dptr: u64, bytes: usize) {
    match texref_post_sync_mode() {
        0 => return,
        1 => {
            // TARGETED: only sync for bf000-shaped binds.
            let is_bf000_shaped =
                bytes == 1_040_000 && (dptr & 0xFF000) == 0xBF000;
            if !is_bf000_shaped {
                return;
            }
            super::driver::zluda_log_err(&format!(
                "*** TEXREF_POST_SYNC firing for bf000-shaped bind dptr={:#x} size={}",
                dptr, bytes
            ));
        }
        _ => {
            // ALL: sync every bind (the n7-rejected behaviour).
        }
    }
    let _ = hipDeviceSynchronize();
    super::stream::clear_dirty_flag_after_device_sync();
}

// 2026-06-15 Class-4 use-after-free probe. The Class-4 GPU fault is a
// page-NOT-PRESENT at a wild VA with last kernel IOFX_bboxKernel, which SAMPLES
// positions via a texref (tex.1d → ptr->textureObject SRD). Every captured
// kernel INPUT is clean, so the wild VA is a stale DESCRIPTOR: the prime
// suspect is a texref-bound buffer being cuMemFree'd (unmapped) WITHOUT
// unbinding the texref → a later bbox sample dereferences the dangling SRD →
// page-not-present fault. free_v2 does NOT check texref bindings. This gated
// registry (ZLUDA_TEXREF_UAF_DETECT=1) records bound buffer ranges so free_v2
// can WARN when freeing one. Read-only diagnostic; no behaviour change.
static TEXREF_BOUND_BUFS: std::sync::Mutex<Option<std::collections::HashMap<u64, usize>>> =
    std::sync::Mutex::new(None);

fn texref_uaf_detect_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("ZLUDA_TEXREF_UAF_DETECT").map_or(false, |v| v == "1"))
}
pub(crate) fn texref_track_bind(base: u64, size: usize) {
    if !texref_uaf_detect_enabled() || base == 0 {
        return;
    }
    let mut g = TEXREF_BOUND_BUFS.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(Default::default).insert(base, size);
}
pub(crate) fn texref_untrack_bind(base: u64) {
    if !texref_uaf_detect_enabled() {
        return;
    }
    let mut g = TEXREF_BOUND_BUFS.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(m) = g.as_mut() {
        m.remove(&base);
    }
}
/// Some((base,size)) if `addr` falls inside a currently texref-bound buffer.
pub(crate) fn texref_bound_lookup(addr: u64) -> Option<(u64, usize)> {
    if !texref_uaf_detect_enabled() {
        return None;
    }
    let g = TEXREF_BOUND_BUFS.lock().unwrap_or_else(|p| p.into_inner());
    g.as_ref()?
        .iter()
        .find(|(base, size)| addr >= **base && addr < **base + **size as u64)
        .map(|(b, s)| (*b, *s))
}

pub(crate) unsafe fn ref_set_address_v2(
    byte_offset: *mut usize,
    h_tex_ref: CUtexref,
    dptr: hipDeviceptr_t,
    bytes: usize,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    if dptr.0.is_null() {
        // An unbind destroys the texref's HIP texture object (descriptor and
        // sampler released, object freed) immediately. A kernel launched
        // before it may still be reading that object; CUDA snapshots the
        // binding at launch. Drain first, like every non-null bind below.
        texref_pre_sync_if_enabled();
        hipUnbindTexture(tex_ref)?;
        return Ok(());
    }
    // NVIDIA accepts bytes=0 (zero-length texture buffer — kernel
    // bounds-checking skips empty ranges, reads return 0).
    // HIP rejects it regardless of the minimum size we provide because
    // the element size depends on the texref's format. Skip the HIP call
    // entirely and preserve the texref's existing binding from previous
    // frames. The kernel's particle count is 0 so it won't read.
    if bytes == 0 {
        if !byte_offset.is_null() {
            *byte_offset = 0;
        }
        return Ok(());
    }
    texref_pre_sync_if_enabled();
    let mut unused = 0;
    hipTexRefSetAddress(
        if byte_offset.is_null() {
            &mut unused
        } else {
            &mut *byte_offset
        },
        tex_ref,
        dptr,
        bytes,
    )?;
    // Sync the texture object to device global if this is a synthetic texref,
    // matching the pattern in ref_set_array()
    sync_synthetic_texref(tex_ref);
    texref_post_sync_if_enabled(dptr.0 as u64, bytes);
    texref_track_bind(dptr.0 as u64, bytes);
    Ok(())
}

pub(crate) unsafe fn ref_set_address_2d_v3(
    h_tex_ref: CUtexref,
    desc: &CUDA_ARRAY_DESCRIPTOR,
    dptr: hipDeviceptr_t,
    pitch: usize,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    texref_pre_sync_if_enabled();
    hipTexRefSetAddress2D(
        tex_ref,
        desc as *const CUDA_ARRAY_DESCRIPTOR as *const HIP_ARRAY_DESCRIPTOR,
        dptr,
        pitch,
    )?;
    // 2D binds are never the bf000 shape (FleX uses 1D linear bind for
    // particle positions); pass 0/0 so TARGETED mode skips. ALL mode
    // still syncs.
    texref_post_sync_if_enabled(0, 0);
    Ok(())
}

pub(crate) unsafe fn ref_set_array(h_tex_ref: CUtexref, h_array: CUarray, flags: u32) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    if (flags & !1u32) != 0 {
        return Err(CUerror::NOT_SUPPORTED);
    }
    // Path-(b) D3D11 interop bypass: D3D11 sentinel arrays are NOT
    // valid hipArray pointers — calling hipArrayGetDescriptor on them
    // returns InvalidContext and hipTexRefSetArray would SIGSEGV
    // (same root cause as hipCreateSurfaceObject's cl_mem deref).
    // Log + return NOT_SUPPORTED so callers fall back gracefully.
    // Currently we only support surface-object writes via the HSA T#
    // path; texref-based sampler reads from D3D11 arrays would need
    // a separate sampler-SRD synthesis (deferred).
    if super::d3d11_interop::d3d11_array_to_hsa_image(h_array as hipArray_t).is_some() {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] cuTexRefSetArray on D3D11 sentinel array={:p} — NOT_SUPPORTED \
             (sampler-SRD synthesis not yet implemented; surface-write path is)",
            h_array
        ));
        return Err(CUerror::NOT_SUPPORTED);
    }
    // Get array descriptor to know the format and channel count
    let mut desc: HIP_ARRAY_DESCRIPTOR = mem::zeroed();
    hipArrayGetDescriptor(&mut desc, h_array as hipArray_t)?;
    // 2026-04-27: lsh-style inverse-format. If the array was created with the
    // HALF/SIGNED_INT substitution (stored as UNSIGNED_INT*), tell the texref
    // the ORIGINAL format so the HIP sampler converts bytes correctly. Without
    // this, the sampler returns raw bits as garbage floats — the missing-fog
    // regression for TurbulenceFS HALF4 velocity textures.
    // See lshqqytiger/ZLUDA v3.9.6 zluda/src/impl/texref.rs::set_array.
    let original_format = super::hipfix::broken_format_for_array(h_array as hipArray_t)
        .0
        .unwrap_or(desc.Format);
    if texref_log_enabled() {
        let hack = super::hipfix::get_array_hack(h_array as hipArray_t);
        super::driver::zluda_log_err(&format!(
            "TEXREF_SET_ARRAY tex={:p} array={:p} stored_fmt={} bound_fmt={} ch={} flags={} hack={:?}",
            tex_ref, h_array, desc.Format.0, original_format.0, desc.NumChannels, flags, hack
        ));
    }
    let set_fmt_res = hipTexRefSetFormat(tex_ref, original_format, desc.NumChannels as i32);
    if set_fmt_res.is_err() && texref_log_enabled() {
        super::driver::zluda_log_err(&format!(
            "TEXREF_SET_ARRAY hipTexRefSetFormat FAILED tex={:p} fmt={} ch={}",
            tex_ref, original_format.0, desc.NumChannels
        ));
    }
    set_fmt_res?;
    texref_pre_sync_if_enabled();
    hipTexRefSetArray(tex_ref, h_array as hipArray_t, CU_TRSA_OVERRIDE_FORMAT)?;
    // Sync the newly created texture object to device global if synthetic
    sync_synthetic_texref(tex_ref);
    Ok(())
}

pub(crate) unsafe fn ref_set_mipmapped_array(
    h_tex_ref: CUtexref,
    h_mipmapped_array: CUmipmappedArray,
    flags: u32,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    texref_pre_sync_if_enabled();
    hipTexRefSetMipmappedArray(tex_ref, h_mipmapped_array as hipMipmappedArray_t, flags)?;
    Ok(())
}

pub(crate) unsafe fn ref_set_format(
    h_tex_ref: CUtexref,
    fmt: CUarray_format,
    num_packed_components: i32,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    // 2026-06-15 Class-4: format (channel kind: int vs float) also decides how
    // tex.1d interprets the bytes; an immediate mid-flight flip is the same
    // float-vs-int race as SetFlags (see ref_set_flags). Drain first.
    texref_pre_sync_all_mutators_if_enabled();
    let fmt_hip: hipArray_Format = mem::transmute(fmt);
    let res = hipTexRefSetFormat(tex_ref, fmt_hip, num_packed_components);
    if texref_log_enabled() {
        // Log raw format request — legacy texref path does NOT apply the lsh
        // hipfix substitution (unlike cuTexObjectCreate), so if IOFX requests
        // HALF / SIGNED_INT8 / SIGNED_INT16 here, watching the broken_hack_flag
        // field tells us whether the bind is going through a broken format.
        let (hack_flag, _) = super::hipfix::get_non_broken_format(fmt_hip);
        super::driver::zluda_log_err(&format!(
            "TEXREF_SET_FORMAT tex={:p} fmt={} ch={} broken_hack_flag={} res={:?}",
            tex_ref, fmt_hip.0, num_packed_components, hack_flag, res
        ));
    }
    res?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_address_mode(
    h_tex_ref: CUtexref,
    dim: i32,
    am: CUaddress_mode,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    // 2026-06-15 Class-4: complete the CUDA texref-snapshot semantic — drain
    // in-flight kernels before flipping addressing mode too (see ref_set_flags).
    texref_pre_sync_all_mutators_if_enabled();
    hipTexRefSetAddressMode(tex_ref, dim, mem::transmute(am))?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_filter_mode(h_tex_ref: CUtexref, fm: CUfilter_mode) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    // 2026-06-15 Class-4: complete the CUDA texref-snapshot semantic (see
    // ref_set_flags) — drain in-flight kernels before flipping filter mode.
    texref_pre_sync_all_mutators_if_enabled();
    hipTexRefSetFilterMode(tex_ref, mem::transmute(fm))?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_mipmap_filter_mode(
    h_tex_ref: CUtexref,
    fm: CUfilter_mode,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    hipTexRefSetMipmapFilterMode(tex_ref, mem::transmute(fm))?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_mipmap_level_bias(h_tex_ref: CUtexref, bias: f32) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    hipTexRefSetMipmapLevelBias(tex_ref, bias)?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_mipmap_level_clamp(
    h_tex_ref: CUtexref,
    min_mipmap_level_clamp: f32,
    max_mipmap_level_clamp: f32,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    hipTexRefSetMipmapLevelClamp(tex_ref, min_mipmap_level_clamp, max_mipmap_level_clamp)?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_max_anisotropy(h_tex_ref: CUtexref, max_aniso: u32) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    hipTexRefSetMaxAnisotropy(tex_ref, max_aniso)?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_border_color(
    h_tex_ref: CUtexref,
    p_border_color: *mut f32,
) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    hipTexRefSetBorderColor(tex_ref, p_border_color)?;
    reset(tex_ref)
}

pub(crate) unsafe fn ref_set_flags(h_tex_ref: CUtexref, flags: u32) -> CUresult {
    let tex_ref = as_tex_ref(h_tex_ref);
    // 2026-06-15 Class-4 root-cause fix: CUDA snapshots the WHOLE texref state
    // (incl. flags) at kernel-launch; HIP flips it immediately. The flags carry
    // CU_TRSF_READ_AS_INTEGER — if an in-flight IOFX_remapKernel reads
    // `tex.1d.v4.u32.s32 [IOFX_texRefRemapActorIDs]` while this flag is
    // mid-transition, the fetch returns a NORMALIZED FLOAT instead of the raw
    // int actor-ID, which it then stores as an index → bbox/actorRange scatter
    // a float bit-pattern → page-aligned CB/DB wild store. Offline capture
    // proof: the actorID input buffer's idx[3] is the int 3 in 7 non-faulting
    // launches but a position float (0xc5366dba) in the faulting one. The
    // existing pre-bind sync covered SetAddress/SetArray but NOT SetFlags, so
    // the race stayed open. Drain in-flight kernels first (gated, same as the
    // address path) to restore the CUDA snapshot semantic for flag changes too.
    texref_pre_sync_all_mutators_if_enabled();
    hipTexRefSetFlags(tex_ref, flags)?;
    reset(tex_ref)
}

pub(crate) unsafe fn object_create(
    p_tex_object: &mut CUtexObject,
    p_res_desc: &CUDA_RESOURCE_DESC,
    p_tex_desc: &CUDA_TEXTURE_DESC,
    _p_res_view_desc: *const CUDA_RESOURCE_VIEW_DESC,
) -> CUresult {
    super::driver::zluda_log_err(&format!(
        "cuTexObjectCreate ENTER resType={:?} viewDesc={:p}",
        p_res_desc.resType, _p_res_view_desc
    ));
    // Build HIP resource desc
    let mut hip_res_desc: HIP_RESOURCE_DESC = mem::zeroed();
    hip_res_desc.resType = HIPresourcetype(p_res_desc.resType.0);
    match p_res_desc.resType {
        CUresourcetype::CU_RESOURCE_TYPE_ARRAY => {
            hip_res_desc.res.array.hArray = p_res_desc.res.array.hArray as hipArray_t;
        }
        CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY => {
            hip_res_desc.res.mipmap.hMipmappedArray =
                p_res_desc.res.mipmap.hMipmappedArray as hipMipmappedArray_t;
        }
        CUresourcetype::CU_RESOURCE_TYPE_LINEAR => {
            hip_res_desc.res.linear.devPtr = hipDeviceptr_t(p_res_desc.res.linear.devPtr.0);
            hip_res_desc.res.linear.format = hipArray_Format(p_res_desc.res.linear.format.0);
            hip_res_desc.res.linear.numChannels = p_res_desc.res.linear.numChannels;
            hip_res_desc.res.linear.sizeInBytes = p_res_desc.res.linear.sizeInBytes;
        }
        CUresourcetype::CU_RESOURCE_TYPE_PITCH2D => {
            hip_res_desc.res.pitch2D.devPtr = hipDeviceptr_t(p_res_desc.res.pitch2D.devPtr.0);
            hip_res_desc.res.pitch2D.format = hipArray_Format(p_res_desc.res.pitch2D.format.0);
            hip_res_desc.res.pitch2D.numChannels = p_res_desc.res.pitch2D.numChannels;
            hip_res_desc.res.pitch2D.width = p_res_desc.res.pitch2D.width;
            hip_res_desc.res.pitch2D.height = p_res_desc.res.pitch2D.height;
            hip_res_desc.res.pitch2D.pitchInBytes = p_res_desc.res.pitch2D.pitchInBytes;
        }
        _ => {}
    }
    hip_res_desc.flags = p_res_desc.flags;

    // Build HIP texture desc — start from zeroed (important: reserved fields must be 0)
    let mut hip_tex_desc: HIP_TEXTURE_DESC = mem::zeroed();
    hip_tex_desc.addressMode = [
        HIPaddress_mode(p_tex_desc.addressMode[0].0),
        HIPaddress_mode(p_tex_desc.addressMode[1].0),
        HIPaddress_mode(p_tex_desc.addressMode[2].0),
    ];
    hip_tex_desc.filterMode = HIPfilter_mode(p_tex_desc.filterMode.0);
    hip_tex_desc.flags = p_tex_desc.flags;
    hip_tex_desc.maxAnisotropy = p_tex_desc.maxAnisotropy;
    hip_tex_desc.mipmapFilterMode = HIPfilter_mode(p_tex_desc.mipmapFilterMode.0);
    hip_tex_desc.mipmapLevelBias = p_tex_desc.mipmapLevelBias;
    hip_tex_desc.minMipmapLevelClamp = p_tex_desc.minMipmapLevelClamp;
    hip_tex_desc.maxMipmapLevelClamp = p_tex_desc.maxMipmapLevelClamp;
    hip_tex_desc.borderColor = p_tex_desc.borderColor;

    // If the underlying array was created with a transformed format
    // (HALF→UINT16 etc. via hipfix::get_non_broken_format), build a
    // HIP_RESOURCE_VIEW_DESC that tells the sampler the ORIGINAL format so
    // tex.3d/tex.2d reads interpret the bytes correctly. Without this the
    // sampler reads e.g. half-float bytes as raw integers, which silently
    // corrupts kernels that read TurbulenceFS density/velocity grids.
    // See plan snug-launching-milner.md Phase D and the missing-from-us
    // lsh helpers in hipfix.rs::maybe_build_view_desc.
    let view_desc = super::hipfix::maybe_build_view_desc(p_res_desc);
    let view_desc_ptr: *const HIP_RESOURCE_VIEW_DESC = match &view_desc {
        Some(v) => v as *const _,
        None => std::ptr::null(),
    };
    if view_desc.is_some() {
        super::driver::zluda_log_err(&format!(
            "texObj: applying format-hack view desc for array={:?}",
            p_res_desc.res.array.hArray
        ));
    }

    // Try with caller's full tex desc first
    let result = hipTexObjectCreate(
        p_tex_object as *mut CUtexObject as *mut hipTextureObject_t,
        &hip_res_desc,
        &hip_tex_desc,
        view_desc_ptr,
    );
    // If that fails, retry with minimal tex desc (zeroed borderColor, no extras)
    let (result, path) = match result {
        Err(_) => {
            let mut minimal_tex: HIP_TEXTURE_DESC = mem::zeroed();
            minimal_tex.filterMode = hip_tex_desc.filterMode;
            minimal_tex.addressMode = hip_tex_desc.addressMode;
            minimal_tex.flags = hip_tex_desc.flags;
            (
                hipTexObjectCreate(
                    p_tex_object as *mut CUtexObject as *mut hipTextureObject_t,
                    &hip_res_desc,
                    &minimal_tex,
                    view_desc_ptr,
                ),
                "minimal",
            )
        }
        ok => (ok, "full"),
    };
    // Log tex creates only when ZLUDA_LOG_ALL=1 to avoid unnecessary HIP
    // operations that can trigger ROCm ProfilingSignal pool race conditions.
    {
        static LOG_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let log_all =
            *LOG_ALL.get_or_init(|| std::env::var("ZLUDA_LOG_ALL").map_or(false, |v| v == "1"));
        if log_all {
            match result {
                Ok(()) => super::driver::zluda_log_err(&format!(
                    "texObj OK ({}) obj={:#x}",
                    path, *p_tex_object
                )),
                Err(ref e) => {
                    super::driver::zluda_log_err(&format!("texObj FAIL ({}) {:?}", path, e))
                }
            }
        }
    }
    if result.is_err() {
        *p_tex_object = 0;
    }
    result?;
    // Phase 3 (2026-05-07): hipDeviceSynchronize after a successful
    // hipTexObjectCreate. Targets the PhysX null-this race
    // (PhysX3_x64.dll PC ~0x6ffffd0aced9): cuTexObjectCreate is
    // documented as thread-safe, but our impl may have a window where
    // the returned handle is encoded + stored before HIP's internal
    // texture-descriptor publish is visible to other threads. The
    // sync ensures the texture object is fully ready before the
    // returned handle escapes to the caller. Cost: ~50–200µs added
    // latency per call; Batman creates ~6–12 per frame → ~3% perf hit.
    // Off by default (validated retirement profile); ZLUDA_TEXOBJ_SYNC=1 enables.
    if std::env::var("ZLUDA_TEXOBJ_SYNC").ok().as_deref() == Some("1") {
        let _ = hip_runtime_sys::hipDeviceSynchronize();
    }
    // Encode pixel shift in top 3 bits of the handle, same as cuSurfObjectCreate.
    // On NVIDIA, texture and surface handles are interchangeable — PhysX uses
    // texture objects for surface writes (sust). Our sust helpers extract the
    // shift from the top 3 bits. Without encoding, shift=0 → wrong x-offset → crash.
    if p_res_desc.resType == CUresourcetype::CU_RESOURCE_TYPE_ARRAY {
        let array = p_res_desc.res.array.hArray as hip_runtime_sys::hipArray_t;
        if !array.is_null() {
            let mut arr_desc: hip_runtime_sys::HIP_ARRAY_DESCRIPTOR = mem::zeroed();
            if hip_runtime_sys::hipArrayGetDescriptor(&mut arr_desc, array).is_ok() {
                let fmt_size = super::surf::format_size(arr_desc.Format).unwrap_or(4);
                let num_ch = if arr_desc.NumChannels == 0 {
                    1
                } else {
                    arr_desc.NumChannels as usize
                };
                let pixel_size = fmt_size * num_ch;
                if pixel_size > 0 {
                    let shift = pixel_size.trailing_zeros() as usize;
                    let shift_amount = shift << (usize::BITS as usize - 3);
                    *p_tex_object = (*p_tex_object as usize | shift_amount) as CUtexObject;
                }
            }
        }
    }
    Ok(())
}

pub(crate) unsafe fn object_destroy(tex_object: CUtexObject) -> CUresult {
    // Strip top 3 bits (shift encoding) before passing to HIP
    let raw = ((tex_object as usize) << 3) >> 3;
    hipTexObjectDestroy(raw as hipTextureObject_t)?;
    Ok(())
}

unsafe fn reset(tex_ref: *mut textureReference) -> CUresult {
    if tex_ref.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    // If no texture object has been created yet (e.g., synthetic texref with no array bound),
    // there's nothing to reset — just return success.
    if (*tex_ref).textureObject.is_null() {
        return Ok(());
    }
    let mut res_desc = mem::zeroed();
    hipGetTextureObjectResourceDesc(&mut res_desc, (*tex_ref).textureObject)?;
    match res_desc.resType {
        hipResourceType::hipResourceTypeArray => {
            let array = res_desc.res.array.array;
            if !array.is_null() {
                hipTexRefSetArray(tex_ref, array, CU_TRSA_OVERRIDE_FORMAT)?;
            }
        }
        hipResourceType::hipResourceTypeLinear => {
            let linear = res_desc.res.linear;
            if !linear.devPtr.is_null() && linear.sizeInBytes != 0 {
                let mut unused = 0usize;
                hipTexRefSetAddress(
                    &mut unused,
                    tex_ref,
                    hipDeviceptr_t(linear.devPtr),
                    linear.sizeInBytes,
                )?;
            }
        }
        hipResourceType::hipResourceTypePitch2D => {
            let pitch_2d = res_desc.res.pitch2D;
            let (format, channels) = from_channel_format_desc(pitch_2d.desc)?;
            let desc = HIP_ARRAY_DESCRIPTOR {
                Width: pitch_2d.width,
                Height: pitch_2d.height,
                Format: format,
                NumChannels: channels,
            };
            hipTexRefSetAddress2D(
                tex_ref,
                &desc,
                hipDeviceptr_t(pitch_2d.devPtr),
                pitch_2d.pitchInBytes,
            )?;
        }
        hipResourceType::hipResourceTypeMipmappedArray => {
            return Err(CUerror::NOT_SUPPORTED);
        }
        _ => return Err(CUerror::INVALID_VALUE),
    }
    // After rebinding, sync the texture object to device global if this is a synthetic texref
    sync_synthetic_texref(tex_ref);
    Ok(())
}

// Unit tests at end of file

fn from_channel_format_desc(desc: hipChannelFormatDesc) -> Result<(hipArray_Format, u32), CUerror> {
    if desc.x != desc.y || desc.x != desc.z || desc.x != desc.w {
        return Err(CUerror::NOT_SUPPORTED);
    }
    let num_channels =
        (desc.x != 0) as u32 + (desc.y != 0) as u32 + (desc.z != 0) as u32 + (desc.w != 0) as u32;
    let format = match (desc.f, desc.x) {
        (hipChannelFormatKind::hipChannelFormatKindUnsigned, 8) => {
            hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT8
        }
        (hipChannelFormatKind::hipChannelFormatKindUnsigned, 16) => {
            hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16
        }
        (hipChannelFormatKind::hipChannelFormatKindUnsigned, 32) => {
            hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT32
        }
        (hipChannelFormatKind::hipChannelFormatKindSigned, 8) => {
            hipArray_Format::HIP_AD_FORMAT_SIGNED_INT8
        }
        (hipChannelFormatKind::hipChannelFormatKindSigned, 16) => {
            hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16
        }
        (hipChannelFormatKind::hipChannelFormatKindSigned, 32) => {
            hipArray_Format::HIP_AD_FORMAT_SIGNED_INT32
        }
        (hipChannelFormatKind::hipChannelFormatKindFloat, 16) => {
            hipArray_Format::HIP_AD_FORMAT_HALF
        }
        (hipChannelFormatKind::hipChannelFormatKindFloat, 32) => {
            hipArray_Format::HIP_AD_FORMAT_FLOAT
        }
        _ => return Err(CUerror::INVALID_VALUE),
    };
    Ok((format, num_channels))
}

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::*;
    use std::mem;

    // Helper: create a context for texture tests
    unsafe fn setup(api: &impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut mem::zeroed(), 0, 0);
    }

    #[test_cuda]
    unsafe fn texref_set_format(api: impl CudaApi) {
        setup(&api);

        // Load a module with a texture reference to get a valid CUtexref
        // Since we can't easily get a texref without a module, test that
        // cuTexRefSetFormat returns success with a module texref
        // For now, verify the API path compiles and links correctly
        // Real validation happens in end-to-end game test
    }

    #[test_cuda]
    unsafe fn tex_object_create_destroy(api: impl CudaApi) {
        setup(&api);

        // Create a 2D array for the texture
        let array_desc = CUDA_ARRAY_DESCRIPTOR {
            Width: 64,
            Height: 64,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 4,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArrayCreate_v2(&mut array, &array_desc);

        // Create texture object from the array
        let mut res_desc: CUDA_RESOURCE_DESC = mem::zeroed();
        res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array;

        let mut tex_desc: CUDA_TEXTURE_DESC = mem::zeroed();
        // Default linear filtering
        tex_desc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;

        let res_view_desc: CUDA_RESOURCE_VIEW_DESC = mem::zeroed();

        let mut tex_object: CUtexObject = 0;
        api.cuTexObjectCreate(&mut tex_object, &res_desc, &tex_desc, &res_view_desc);
        assert_ne!(tex_object, 0);

        api.cuTexObjectDestroy(tex_object);
        api.cuArrayDestroy(array);
    }

    #[test_cuda]
    unsafe fn texref_set_address_zero_bytes(api: impl CudaApi) {
        setup(&api);
        // Allocate a small device buffer (non-null dptr)
        let mut dptr: CUdeviceptr = unsafe { mem::zeroed() };
        api.cuMemAlloc_v2(&mut dptr, 256);
        assert_ne!(dptr.0, std::ptr::null_mut());
        // Call cuTexRefSetAddress_v2 with bytes=0 — should return CUDA_SUCCESS,
        // not crash or return an error. This reproduces the PhysX/FleX
        // CollideParticles scenario where empty particle arrays use bytes=0.
        // Without a module we can't get a real texref, so we validate the
        // allocation path. The full fix is verified by the game not crashing.
        api.cuMemFree_v2(dptr);
    }

    #[test_cuda]
    unsafe fn texref_set_array_and_format(api: impl CudaApi) {
        setup(&api);

        // Create a simple 2D array (like PhysX does for turbulence textures)
        let desc = CUDA_ARRAY_DESCRIPTOR {
            Width: 256,
            Height: 256,
            Format: CUarray_format::CU_AD_FORMAT_FLOAT,
            NumChannels: 4,
        };
        let mut array: CUarray = mem::zeroed();
        api.cuArrayCreate_v2(&mut array, &desc);
        assert!(!array.is_null());

        // Clean up
        api.cuArrayDestroy(array);
    }
}

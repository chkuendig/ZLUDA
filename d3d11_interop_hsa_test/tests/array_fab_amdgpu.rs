//! Phase A.HSA.2 — fabricate a hipArray pointing at imported memory.
//!
//! The crucial unsolved question: can we hand-construct a hipArray
//! struct (with its `data` field set to a HIP-imported dev_ptr from
//! a dma_buf) and pass it to `hipCreateSurfaceObject`? If yes, we
//! don't need OpenCL/HSA bypass — just struct fabrication is enough.
//! If no, the failure mode tells us which downstream API is rejecting
//! us, which informs the bypass design.
//!
//! Per ROCm clr source, hipArray is:
//!   typedef struct hipArray {
//!       void* data;                            // ← cl_mem cast inside CLR
//!       struct hipChannelFormatDesc desc;
//!       unsigned int type;
//!       unsigned int width, height, depth;
//!       enum hipArray_Format Format;
//!       unsigned int NumChannels;
//!       bool isDrv;
//!       unsigned int textureType;
//!       unsigned int flags;
//!   } hipArray;
//!
//! `data` is treated as a `cl_mem` by `as_amd(memObj)->asImage()` in
//! ihipCreateTextureObject. If our dev_ptr (raw GPU VA) isn't a
//! valid cl_mem, hipCreateSurfaceObject will fail. That failure mode
//! is the actual signal we want.

use hip_runtime_sys::*;
use std::os::raw::{c_int, c_void};
use std::ptr;

type HsaStatus = u32;
const HSA_STATUS_SUCCESS: HsaStatus = 0;

#[link(name = "hsa-runtime64")]
extern "C" {
    fn hsa_amd_portable_export_dmabuf(
        ptr: *const c_void,
        size: usize,
        dmabuf: *mut c_int,
        offset: *mut u64,
    ) -> HsaStatus;
}

/// Mirror of clr's `struct hipArray` on rocm-6.2.x. Layout MUST match
/// what `libamdhip64.so` expects when callers dereference a
/// `hipArray_t`.
///
/// SAFETY-CRITICAL: this struct's layout is derived from inspection of
/// ROCm/clr source at the rocm-6.2.1 tag. If ROCm changes the layout
/// in a future release, this fabrication breaks silently. Tests in
/// this file are the canary.
#[repr(C)]
#[derive(Debug)]
#[allow(non_snake_case)]
struct FabricatedHipArray {
    data: *mut c_void,
    desc: hipChannelFormatDesc,
    type_: u32,
    width: u32,
    height: u32,
    depth: u32,
    Format: u32, // hipArray_Format enum value
    NumChannels: u32,
    isDrv: bool,
    textureType: u32,
    flags: u32,
}

fn ok(res: Result<(), hipErrorCode_t>) {
    if let Err(e) = res {
        panic!("HIP call failed: {:?}", e);
    }
}

fn init_hip() {
    unsafe {
        ok(hipSetDevice(0));
    }
}

/// Build a hipArray struct backed by the imported dev_ptr. Returns the
/// boxed struct; caller passes &raw mut as hipArray_t to HIP APIs.
fn fabricate_array(
    dev_ptr: *mut c_void,
    width: u32,
    height: u32,
) -> Box<FabricatedHipArray> {
    Box::new(FabricatedHipArray {
        data: dev_ptr,
        desc: hipChannelFormatDesc {
            x: 32,
            y: 32,
            z: 32,
            w: 32,
            f: hipChannelFormatKind::hipChannelFormatKindFloat,
        },
        // hipResourceType: hipResourceTypeArray = 0x0
        type_: 0,
        width,
        height,
        depth: 0,
        // hipArray_Format::HIP_AD_FORMAT_FLOAT = 0x20
        Format: 0x20,
        NumChannels: 4,
        isDrv: true,
        textureType: 0,
        flags: 0,
    })
}

#[test]
fn fabricate_struct_alone_amdgpu() {
    // Lower bar: just construct the struct, pass to
    // hipArrayGetDescriptor. If HIP reads back our metadata correctly,
    // the struct layout is right. If it crashes or returns garbage,
    // layout is wrong.
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip();

    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 254;
    const N: usize = (WIDTH as usize) * (HEIGHT as usize) * 16; // float4

    unsafe {
        let mut dev_a: *mut c_void = ptr::null_mut();
        ok(hipMalloc(&mut dev_a, N));

        let mut fd: c_int = -1;
        let mut offset: u64 = 0;
        let st = hsa_amd_portable_export_dmabuf(dev_a, N, &mut fd, &mut offset);
        assert_eq!(st, HSA_STATUS_SUCCESS);

        let mut hd: hipExternalMemoryHandleDesc = std::mem::zeroed();
        hd.type_ = hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueFd;
        hd.handle.fd = fd;
        hd.size = N as u64;
        let mut ext_mem: hipExternalMemory_t = ptr::null_mut();
        ok(hipImportExternalMemory(&mut ext_mem, &hd));

        let mut bd: hipExternalMemoryBufferDesc = std::mem::zeroed();
        bd.offset = offset;
        bd.size = N as u64;
        let mut dev_b: *mut c_void = ptr::null_mut();
        ok(hipExternalMemoryGetMappedBuffer(&mut dev_b, ext_mem, &bd));

        // FABRICATE: build a hipArray struct ourselves.
        let fab = fabricate_array(dev_b, WIDTH, HEIGHT);
        let arr_ptr: hipArray_t = &*fab as *const FabricatedHipArray as hipArray_t;

        // PROBE: query HIP for the descriptor. If layout is right,
        // HIP reads our width/height/format and reports them back.
        let mut got: HIP_ARRAY_DESCRIPTOR = std::mem::zeroed();
        let res = hipArrayGetDescriptor(&mut got, arr_ptr);
        eprintln!("[test] hipArrayGetDescriptor on fabricated -> {:?}", res);

        match res {
            Ok(()) => {
                eprintln!(
                    "[test] descriptor: width={} height={} fmt={:?} ch={}",
                    got.Width, got.Height, got.Format, got.NumChannels
                );
                assert_eq!(got.Width, WIDTH as usize, "width mismatch");
                assert_eq!(got.Height, HEIGHT as usize, "height mismatch");
                assert_eq!(got.NumChannels, 4, "NumChannels mismatch");
                eprintln!("[test] STRUCT FAB ACCEPTED — proceed to surface-object test");
            }
            Err(e) => {
                // Failure is informative — likely means HIP looks up
                // the array in an internal registry and rejects ours.
                eprintln!(
                    "[test] STRUCT FAB REJECTED by hipArrayGetDescriptor: {:?} \
                     — must use real cl_mem or HSA bypass",
                    e
                );
            }
        }

        // Cleanup. We don't call hipFreeArray on `fab` because we never
        // allocated it via HIP (would crash).
        drop(fab);
        ok(hipDestroyExternalMemory(ext_mem));
        ok(hipFree(dev_a));
    }
}

#[test]
fn real_hip_array_works_amdgpu() {
    // POSITIVE CONTROL: a real hipMallocArray-allocated array MUST
    // accept hipArrayGetDescriptor + hipCreateSurfaceObject without
    // SIGSEGV or InvalidContext. If this fails, the test setup itself
    // is broken (not the fabrication being wrong).
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip();

    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 254;

    unsafe {
        let mut fmt: hipChannelFormatDesc = std::mem::zeroed();
        fmt.x = 32; fmt.y = 32; fmt.z = 32; fmt.w = 32;
        fmt.f = hipChannelFormatKind::hipChannelFormatKindFloat;

        let mut arr: hipArray_t = ptr::null_mut();
        ok(hipMallocArray(&mut arr, &fmt, WIDTH as usize, HEIGHT as usize, hipArrayDefault));
        assert!(!arr.is_null());

        // Descriptor query.
        let mut got: HIP_ARRAY_DESCRIPTOR = std::mem::zeroed();
        ok(hipArrayGetDescriptor(&mut got, arr));
        assert_eq!(got.Width, WIDTH as usize);
        assert_eq!(got.Height, HEIGHT as usize);

        // Surface object create.
        let mut rd: hipResourceDesc = std::mem::zeroed();
        rd.resType = hipResourceType::hipResourceTypeArray;
        rd.res.array.array = arr;
        let mut surf: hipSurfaceObject_t = std::mem::zeroed();
        ok(hipCreateSurfaceObject(&mut surf, &rd));
        ok(hipDestroySurfaceObject(surf));
        ok(hipFreeArray(arr));
    }
}

/// Documented to SIGSEGV — `hipCreateSurfaceObject` casts the array's
/// `data` field to `cl_mem` and derefs the C++ vtable. Our raw GPU VA
/// from `hipExternalMemoryGetMappedBuffer` is not a valid cl_mem
/// pointer. Marked `#[ignore]` so the rest of the suite isn't killed
/// by the SIGSEGV (it propagates from the kernel driver as an
/// unrecoverable signal at process level). Run explicitly with
/// `cargo test -- --ignored fabricate_struct_into_surface_object`.
#[test]
#[ignore = "SIGSEGV is the documented A.HSA.2 finding; run --ignored to confirm"]
fn fabricate_struct_into_surface_object_amdgpu() {
    // Higher bar: try to build a surface object from our fabricated
    // array. This is the actual API APEX uses (cuSurfObjectCreate ->
    // hipCreateSurfaceObject).
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip();

    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 254;
    const N: usize = (WIDTH as usize) * (HEIGHT as usize) * 16;

    unsafe {
        let mut dev_a: *mut c_void = ptr::null_mut();
        ok(hipMalloc(&mut dev_a, N));

        let mut fd: c_int = -1;
        let mut offset: u64 = 0;
        assert_eq!(
            hsa_amd_portable_export_dmabuf(dev_a, N, &mut fd, &mut offset),
            HSA_STATUS_SUCCESS
        );

        let mut hd: hipExternalMemoryHandleDesc = std::mem::zeroed();
        hd.type_ = hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueFd;
        hd.handle.fd = fd;
        hd.size = N as u64;
        let mut ext_mem: hipExternalMemory_t = ptr::null_mut();
        ok(hipImportExternalMemory(&mut ext_mem, &hd));

        let mut bd: hipExternalMemoryBufferDesc = std::mem::zeroed();
        bd.offset = offset;
        bd.size = N as u64;
        let mut dev_b: *mut c_void = ptr::null_mut();
        ok(hipExternalMemoryGetMappedBuffer(&mut dev_b, ext_mem, &bd));

        let fab = fabricate_array(dev_b, WIDTH, HEIGHT);
        let arr_ptr: hipArray_t = &*fab as *const FabricatedHipArray as hipArray_t;

        // Build a hipResourceDesc pointing at our array.
        let mut rd: hipResourceDesc = std::mem::zeroed();
        rd.resType = hipResourceType::hipResourceTypeArray;
        rd.res.array.array = arr_ptr;

        let mut surf_obj: hipSurfaceObject_t = std::mem::zeroed();
        let res = hipCreateSurfaceObject(&mut surf_obj, &rd);
        eprintln!("[test] hipCreateSurfaceObject on fabricated array -> {:?}", res);

        if let Ok(()) = res {
            eprintln!(
                "[test] SURFACE OBJ CREATED — surf_obj=0x{:x}",
                surf_obj as u64
            );
            // Cleanup: destroy. If hipCreateSurfaceObject succeeded
            // with our fabrication, hipDestroySurfaceObject should
            // also work or fail gracefully.
            let _ = hipDestroySurfaceObject(surf_obj);
        } else {
            eprintln!("[test] surface obj creation rejected — see error code above");
        }

        drop(fab);
        ok(hipDestroyExternalMemory(ext_mem));
        ok(hipFree(dev_a));
    }
}

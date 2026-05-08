//! Phase A.HSA.3 — HSA image creation on imported memory.
//!
//! Foundation for path (b) of the surface-bypass interop fix:
//! can `hsa_ext_image_create_with_layout` accept a HIP-imported
//! dev_ptr (round-tripped via dma_buf) and bind a 2D image at it?
//!
//! If yes → we have an HSA image whose memory aliases the original
//! (Vulkan/D3D11) buffer; subsequent steps extract the SRD and
//! synthesize a hipSurfaceObject_t.
//!
//! If no → we'd need `hsa_amd_interop_map_buffer` to import the
//! dma_buf into HSA directly, bypassing HIP, before image creation.

use hip_runtime_sys::*;
use std::os::raw::{c_int, c_void};
use std::ptr;

// ---- HSA bindings (just what we need) ----

type HsaStatus = u32;
const HSA_STATUS_SUCCESS: HsaStatus = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaAgent {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct HsaExtImage {
    handle: u64,
}

// hsa_ext_image_geometry_t
const HSA_EXT_IMAGE_GEOMETRY_2D: u32 = 1;

// hsa_ext_image_channel_type_t
const HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT: u32 = 15;

// hsa_ext_image_channel_order_t
const HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA: u32 = 8;

// hsa_access_permission_t
const HSA_ACCESS_PERMISSION_RW: u32 = 3;

// hsa_ext_image_data_layout_t
const HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR: u32 = 0x1;

// hsa_agent_info_t
const HSA_AGENT_INFO_DEVICE: u32 = 17;

// hsa_device_type_t
const HSA_DEVICE_TYPE_GPU: u32 = 1;

#[repr(C)]
struct HsaExtImageFormat {
    channel_type: u32,   // hsa_ext_image_channel_type32_t
    channel_order: u32,  // hsa_ext_image_channel_order32_t
}

#[repr(C)]
struct HsaExtImageDescriptor {
    geometry: u32,        // hsa_ext_image_geometry_t
    width: usize,
    height: usize,
    depth: usize,
    array_size: usize,
    format: HsaExtImageFormat,
}

#[link(name = "hsa-runtime64")]
extern "C" {
    fn hsa_init() -> HsaStatus;
    fn hsa_iterate_agents(
        cb: extern "C" fn(HsaAgent, *mut c_void) -> HsaStatus,
        data: *mut c_void,
    ) -> HsaStatus;
    fn hsa_agent_get_info(
        agent: HsaAgent,
        attribute: u32,
        value: *mut c_void,
    ) -> HsaStatus;
    fn hsa_amd_portable_export_dmabuf(
        ptr: *const c_void,
        size: usize,
        dmabuf: *mut c_int,
        offset: *mut u64,
    ) -> HsaStatus;
    fn hsa_ext_image_create_with_layout(
        agent: HsaAgent,
        descriptor: *const HsaExtImageDescriptor,
        image_data: *const c_void,
        access: u32,
        layout: u32,
        row_pitch: usize,
        slice_pitch: usize,
        image: *mut HsaExtImage,
    ) -> HsaStatus;
    fn hsa_ext_image_destroy(agent: HsaAgent, image: HsaExtImage) -> HsaStatus;
}

// Find a GPU agent via hsa_iterate_agents callback.
extern "C" fn find_gpu_cb(agent: HsaAgent, data: *mut c_void) -> HsaStatus {
    unsafe {
        let mut device_type: u32 = 0;
        let st = hsa_agent_get_info(
            agent,
            HSA_AGENT_INFO_DEVICE,
            &mut device_type as *mut _ as *mut c_void,
        );
        if st != HSA_STATUS_SUCCESS {
            return HSA_STATUS_SUCCESS; // continue iteration
        }
        if device_type == HSA_DEVICE_TYPE_GPU {
            let out = data as *mut HsaAgent;
            *out = agent;
            // Returning non-zero status stops iteration; we use a sentinel
            // value to mean "found, stop". HSA will surface this status
            // back to us. We return STATUS_INFO_BREAK = 0x100 (defined as
            // a non-zero non-error to signal early termination).
            return 0x100; // HSA_STATUS_INFO_BREAK
        }
        HSA_STATUS_SUCCESS
    }
}

fn ok_hip(res: Result<(), hipErrorCode_t>) {
    if let Err(e) = res {
        panic!("HIP call failed: {:?}", e);
    }
}

fn init_hip() {
    unsafe {
        ok_hip(hipSetDevice(0));
    }
}

fn find_gpu() -> HsaAgent {
    let mut agent = HsaAgent { handle: 0 };
    unsafe {
        // hsa_init is idempotent (refcounted); HIP has likely already
        // called it but we ensure.
        let _ = hsa_init();
        let _ = hsa_iterate_agents(
            find_gpu_cb,
            &mut agent as *mut _ as *mut c_void,
        );
    }
    assert!(agent.handle != 0, "no HSA GPU agent found");
    agent
}

#[test]
fn hsa_image_create_on_hip_imported_amdgpu() {
    let _g = d3d11_interop_hsa_test::serialize();
    init_hip();
    let agent = find_gpu();

    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 254;
    const ROW_PITCH: usize = (WIDTH as usize) * 16;
    const N: usize = ROW_PITCH * (HEIGHT as usize);

    unsafe {
        // Allocate HIP memory and round-trip via dma_buf import so we
        // have an "imported" dev_ptr (mirrors what we'd have in
        // d3d11_interop.rs).
        let mut dev_a: *mut c_void = ptr::null_mut();
        ok_hip(hipMalloc(&mut dev_a, N));

        let mut fd: c_int = -1;
        let mut offset: u64 = 0;
        let st = hsa_amd_portable_export_dmabuf(dev_a, N, &mut fd, &mut offset);
        assert_eq!(st, HSA_STATUS_SUCCESS);

        let mut hd: hipExternalMemoryHandleDesc = std::mem::zeroed();
        hd.type_ = hipExternalMemoryHandleType::hipExternalMemoryHandleTypeOpaqueFd;
        hd.handle.fd = fd;
        hd.size = N as u64;
        let mut ext_mem: hipExternalMemory_t = ptr::null_mut();
        ok_hip(hipImportExternalMemory(&mut ext_mem, &hd));

        let mut bd: hipExternalMemoryBufferDesc = std::mem::zeroed();
        bd.offset = offset;
        bd.size = N as u64;
        let mut dev_b: *mut c_void = ptr::null_mut();
        ok_hip(hipExternalMemoryGetMappedBuffer(&mut dev_b, ext_mem, &bd));
        assert!(!dev_b.is_null(), "imported dev_ptr null");

        // Now the actual experiment: create an HSA image at dev_b.
        let descriptor = HsaExtImageDescriptor {
            geometry: HSA_EXT_IMAGE_GEOMETRY_2D,
            width: WIDTH as usize,
            height: HEIGHT as usize,
            depth: 0,
            array_size: 0,
            format: HsaExtImageFormat {
                channel_type: HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT,
                channel_order: HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,
            },
        };

        let mut image: HsaExtImage = HsaExtImage { handle: 0 };
        let img_status = hsa_ext_image_create_with_layout(
            agent,
            &descriptor,
            dev_b as *const c_void,
            HSA_ACCESS_PERMISSION_RW,
            HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR,
            ROW_PITCH,
            0, // slice_pitch — 0 for 2D
            &mut image,
        );

        eprintln!(
            "[test] hsa_ext_image_create_with_layout -> status=0x{:x} handle=0x{:x}",
            img_status, image.handle
        );

        if img_status == HSA_STATUS_SUCCESS {
            assert!(image.handle != 0, "success but handle is zero");
            eprintln!("[test] HSA IMAGE CREATED — image.handle = 0x{:x}", image.handle);
            // Cleanup.
            let _ = hsa_ext_image_destroy(agent, image);
        } else {
            eprintln!(
                "[test] HSA IMAGE CREATION FAILED — status 0x{:x}. \
                 Likely the imported dev_ptr isn't an HSA-aware allocation. \
                 Next test: hsa_amd_interop_map_buffer the dma_buf directly.",
                img_status
            );
        }

        // Cleanup HIP/dma_buf side regardless.
        let _ = hipDestroyExternalMemory(ext_mem);
        ok_hip(hipFree(dev_a));

        assert_eq!(
            img_status, HSA_STATUS_SUCCESS,
            "hsa_ext_image_create_with_layout failed; see stderr for diagnostic"
        );
    }
}

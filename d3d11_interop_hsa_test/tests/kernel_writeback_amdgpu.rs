//! Phase A.HSA.5 — kernel surf2Dwrite round-trip.
//!
//! THE definitive test: compile a tiny HIP kernel that does
//! surf2Dwrite using a synthesized hipSurfaceObject_t (pointer to
//! the gfx10 T# returned by hsa_ext_image_create_with_layout).
//! Run it. Read the imported dev_ptr back to host. Verify the bytes
//! the kernel wrote appear in the original buffer.
//!
//! If this passes, the path-(b) HSA bypass is fully validated and
//! we can wire it into surf::object_create.

use hip_runtime_sys::*;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

// ---- HSA + HIPRTC bindings ----

type HsaStatus = u32;
const HSA_STATUS_SUCCESS: HsaStatus = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaAgent { handle: u64 }

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaExtImage { handle: u64 }

const HSA_EXT_IMAGE_GEOMETRY_2D: u32 = 1;
const HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT: u32 = 15;
const HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA: u32 = 8;
const HSA_ACCESS_PERMISSION_RW: u32 = 3;
const HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR: u32 = 0x1;
const HSA_AGENT_INFO_DEVICE: u32 = 17;
const HSA_DEVICE_TYPE_GPU: u32 = 1;

#[repr(C)]
struct HsaExtImageFormat { channel_type: u32, channel_order: u32 }

#[repr(C)]
struct HsaExtImageDescriptor {
    geometry: u32,
    width: usize,
    height: usize,
    depth: usize,
    array_size: usize,
    format: HsaExtImageFormat,
}

type HiprtcProgram = *mut c_void;
type HiprtcResult = u32;
const HIPRTC_SUCCESS: HiprtcResult = 0;

#[link(name = "hsa-runtime64")]
extern "C" {
    fn hsa_init() -> HsaStatus;
    fn hsa_iterate_agents(
        cb: extern "C" fn(HsaAgent, *mut c_void) -> HsaStatus,
        data: *mut c_void,
    ) -> HsaStatus;
    fn hsa_agent_get_info(agent: HsaAgent, attribute: u32, value: *mut c_void) -> HsaStatus;
    fn hsa_amd_portable_export_dmabuf(
        ptr: *const c_void, size: usize,
        dmabuf: *mut c_int, offset: *mut u64,
    ) -> HsaStatus;
    fn hsa_ext_image_create_with_layout(
        agent: HsaAgent,
        descriptor: *const HsaExtImageDescriptor,
        image_data: *const c_void,
        access: u32, layout: u32,
        row_pitch: usize, slice_pitch: usize,
        image: *mut HsaExtImage,
    ) -> HsaStatus;
    fn hsa_ext_image_destroy(agent: HsaAgent, image: HsaExtImage) -> HsaStatus;
}

#[link(name = "hiprtc")]
extern "C" {
    fn hiprtcCreateProgram(
        prog: *mut HiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> HiprtcResult;
    fn hiprtcCompileProgram(
        prog: HiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> HiprtcResult;
    fn hiprtcGetProgramLogSize(prog: HiprtcProgram, size: *mut usize) -> HiprtcResult;
    fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut c_char) -> HiprtcResult;
    fn hiprtcGetCodeSize(prog: HiprtcProgram, size: *mut usize) -> HiprtcResult;
    fn hiprtcGetCode(prog: HiprtcProgram, code: *mut c_char) -> HiprtcResult;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> HiprtcResult;
}

extern "C" fn find_gpu_cb(agent: HsaAgent, data: *mut c_void) -> HsaStatus {
    unsafe {
        let mut device_type: u32 = 0;
        if hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE,
                              &mut device_type as *mut _ as *mut c_void) != HSA_STATUS_SUCCESS {
            return HSA_STATUS_SUCCESS;
        }
        if device_type == HSA_DEVICE_TYPE_GPU {
            *(data as *mut HsaAgent) = agent;
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

fn find_gpu() -> HsaAgent {
    let mut a = HsaAgent { handle: 0 };
    unsafe {
        let _ = hsa_init();
        let _ = hsa_iterate_agents(find_gpu_cb, &mut a as *mut _ as *mut c_void);
    }
    assert!(a.handle != 0, "no HSA GPU agent");
    a
}

/// Get the gfx arch name for the current device (e.g. "gfx1033") so
/// hiprtc knows what to compile for.
fn current_gcn_arch() -> String {
    unsafe {
        let mut props: hipDeviceProp_tR0600 = std::mem::zeroed();
        ok_hip(hipGetDevicePropertiesR0600(&mut props, 0));
        // gcnArchName is a fixed-size [c_char; 256]
        let cstr = std::ffi::CStr::from_ptr(props.gcnArchName.as_ptr());
        // Strip any ":xnack-" / ":sramecc-" suffixes that hiprtc doesn't accept
        // as part of the arch flag — keep only the bare gfx#### prefix.
        let raw = cstr.to_string_lossy().to_string();
        raw.split(':').next().unwrap_or("gfx1033").to_string()
    }
}

const KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Writes a 16-byte uint4 pixel at (0,0) of a 2D R32G32B32A32_FLOAT
// surface. The synthesized hipSurfaceObject_t (pointer to gfx10 T#
// from HSA) is passed as `surf`.
extern "C" __global__ void surf_write_u4(
    hipSurfaceObject_t surf,
    unsigned int v0, unsigned int v1,
    unsigned int v2, unsigned int v3)
{
    uint4 val = make_uint4(v0, v1, v2, v3);
    surf2Dwrite(val, surf, 0, 0);
}
"#;

fn compile_kernel(arch: &str) -> Vec<u8> {
    unsafe {
        let src = CString::new(KERNEL_SRC).unwrap();
        let name = CString::new("kern.hip").unwrap();
        let mut prog: HiprtcProgram = ptr::null_mut();
        let r = hiprtcCreateProgram(
            &mut prog,
            src.as_ptr(),
            name.as_ptr(),
            0, ptr::null(), ptr::null(),
        );
        assert_eq!(r, HIPRTC_SUCCESS, "hiprtcCreateProgram failed");

        let arch_opt = CString::new(format!("--offload-arch={}", arch)).unwrap();
        let opts: [*const c_char; 1] = [arch_opt.as_ptr()];
        let r = hiprtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr());
        if r != HIPRTC_SUCCESS {
            let mut log_size: usize = 0;
            let _ = hiprtcGetProgramLogSize(prog, &mut log_size);
            let mut log = vec![0u8; log_size + 1];
            let _ = hiprtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
            let msg = std::ffi::CStr::from_ptr(log.as_ptr() as *const c_char)
                .to_string_lossy().to_string();
            panic!("hiprtcCompileProgram failed: {} — log: {}", r, msg);
        }

        let mut size: usize = 0;
        assert_eq!(hiprtcGetCodeSize(prog, &mut size), HIPRTC_SUCCESS);
        let mut code = vec![0u8; size];
        assert_eq!(hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char), HIPRTC_SUCCESS);

        let _ = hiprtcDestroyProgram(&mut prog);
        code
    }
}

#[test]
fn kernel_surf2dwrite_visible_in_original_amdgpu() {
    let _g = d3d11_interop_hsa_test::serialize();
    unsafe { ok_hip(hipSetDevice(0)); }
    let agent = find_gpu();

    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 254;
    const ROW_PITCH: usize = (WIDTH as usize) * 16;
    const N: usize = ROW_PITCH * (HEIGHT as usize);

    unsafe {
        // 1. Allocate + dma_buf round-trip → dev_b is the imported alias.
        let mut dev_a: *mut c_void = ptr::null_mut();
        ok_hip(hipMalloc(&mut dev_a, N));
        ok_hip(hipMemsetD8(hipDeviceptr_t(dev_a), 0, N));
        ok_hip(hipDeviceSynchronize());

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
        ok_hip(hipImportExternalMemory(&mut ext_mem, &hd));

        let mut bd: hipExternalMemoryBufferDesc = std::mem::zeroed();
        bd.offset = offset;
        bd.size = N as u64;
        let mut dev_b: *mut c_void = ptr::null_mut();
        ok_hip(hipExternalMemoryGetMappedBuffer(&mut dev_b, ext_mem, &bd));

        // 2. Build HSA image → image.handle is our T#.
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
        assert_eq!(
            hsa_ext_image_create_with_layout(
                agent, &descriptor, dev_b as *const c_void,
                HSA_ACCESS_PERMISSION_RW,
                HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR,
                ROW_PITCH, 0, &mut image,
            ),
            HSA_STATUS_SUCCESS
        );

        // 3. Compile kernel.
        let arch = current_gcn_arch();
        eprintln!("[test] target arch = {}", arch);
        let code = compile_kernel(&arch);
        eprintln!("[test] kernel code size = {} bytes", code.len());

        let mut module: hipModule_t = std::mem::zeroed();
        ok_hip(hipModuleLoadData(&mut module, code.as_ptr() as *const c_void));
        let fn_name = CString::new("surf_write_u4").unwrap();
        let mut func: hipFunction_t = std::mem::zeroed();
        ok_hip(hipModuleGetFunction(&mut func, module, fn_name.as_ptr()));

        // 4. Launch with our synthesized surface object.
        // hipSurfaceObject_t IS the image.handle pointer.
        let surf_obj: hipSurfaceObject_t = image.handle as hipSurfaceObject_t;
        let v0: u32 = 0xDEADBEEF;
        let v1: u32 = 0x12345678;
        let v2: u32 = 0xCAFEBABE;
        let v3: u32 = 0x87654321;

        // Kernel args: address-of each arg, in declaration order.
        let mut surf_obj_box = surf_obj;
        let mut v0_box = v0;
        let mut v1_box = v1;
        let mut v2_box = v2;
        let mut v3_box = v3;
        let mut args: [*mut c_void; 5] = [
            &mut surf_obj_box as *mut _ as *mut c_void,
            &mut v0_box as *mut _ as *mut c_void,
            &mut v1_box as *mut _ as *mut c_void,
            &mut v2_box as *mut _ as *mut c_void,
            &mut v3_box as *mut _ as *mut c_void,
        ];

        ok_hip(hipModuleLaunchKernel(
            func,
            1, 1, 1,   // grid 1x1x1
            1, 1, 1,   // block 1x1x1
            0,         // shared mem
            hipStream_t(ptr::null_mut()), // stream
            args.as_mut_ptr(),
            ptr::null_mut(), // extra
        ));
        ok_hip(hipDeviceSynchronize());

        // 5. Read first 16 bytes through original dev_a.
        let mut host = [0u32; 4];
        ok_hip(hipMemcpyDtoH(
            host.as_mut_ptr() as *mut c_void,
            hipDeviceptr_t(dev_a),
            16,
        ));

        eprintln!(
            "[test] read pixel(0,0) = [{:08x} {:08x} {:08x} {:08x}]",
            host[0], host[1], host[2], host[3]
        );

        let want = [v0, v1, v2, v3];
        let got_match = host == want;

        // Cleanup.
        let _ = hipModuleUnload(module);
        let _ = hsa_ext_image_destroy(agent, image);
        let _ = hipDestroyExternalMemory(ext_mem);
        ok_hip(hipFree(dev_a));

        assert!(
            got_match,
            "kernel surf2Dwrite did NOT propagate to original buffer.\n\
             want = {:08x?}\n\
             got  = {:08x?}\n\
             — path (b) round-trip not yet working; investigate.",
            want, host,
        );

        eprintln!("[test] *** PATH (b) ROUND-TRIP CONFIRMED ***");
    }
}

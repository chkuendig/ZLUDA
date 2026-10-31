// SPDX-License-Identifier: MIT OR Apache-2.0
//! PE32 frontend ported from fa81f481 wow64/pe.c; native addresses stay Unix.
#![cfg_attr(not(windows), allow(dead_code))]
use crate::wow64_protocol::*;
use std::{ffi::{c_char, c_void}, sync::atomic::{AtomicBool, Ordering}};
use crate::pe_sync::{Mutex, Once as OnceLock};
use crate::wow64_local::storage_destroy;
pub use crate::wow64_local::cuGetExportTable;

static ATTACHED: AtomicBool = AtomicBool::new(false);
static HANDSHAKE: OnceLock<bool> = OnceLock::new();
extern "C" {
    fn zluda_wine_attach(module: *mut c_void) -> i32;
    fn zluda_wine_call(args: *mut c_void) -> i32;
}
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn DllMain(module: *mut c_void, reason: u32, _: *mut c_void) -> i32 {
    if reason == 1 {
        if std::env::var("PROTON_ENABLE_ZLUDA").as_deref() != Ok("1") { return 0; }
        let attached = zluda_wine_attach(module) != 0;
        ATTACHED.store(attached, Ordering::Release);
        return i32::from(attached);
    }
    1
}
pub(super) unsafe fn call(op: u32, args: &[u32]) -> i32 {
    if !ATTACHED.load(Ordering::Acquire) { return 3; }
    if !*HANDSHAKE.get_or_init(|| {
        let mut hello = Call32::new(HELLO, &[]);
        zluda_wine_call((&mut hello as *mut Call32).cast()) == 0 && hello.result == 0
    }) { return 3; }
    let mut packet = Call32::new(op, args);
    if zluda_wine_call((&mut packet as *mut Call32).cast()) != 0 { return 3; }
    packet.result
}
macro_rules! forward {
    ($name:ident($($arg:ident: $ty:ty),*) => $op:ident [$($value:expr),*]) => {
        #[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
        pub unsafe extern "system" fn $name($($arg: $ty),*) -> i32 { call($op, &[$($value),*]) }
    };
}
forward!(cuInit(flags: u32) => C32_INIT [flags]);
forward!(cuDriverGetVersion(out: *mut i32) => C32_VERSION [pointer(out)]);
forward!(cuDeviceGetCount(out: *mut i32) => C32_DEVICE_COUNT [pointer(out)]);

#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuDeviceGetLuid(out: *mut c_char, node: *mut u32, device: i32) -> i32 {
    if out.is_null() || node.is_null() { return INVALID; }
    let mut uuid = crate::cuda::CUuuid { bytes: [0; 16] };
    let mut fake = 0u32;
    let status = call(C32_DEVICE_UUID, &[pointer(&mut uuid), device as u32, pointer(&mut fake)]);
    if status != 0 { return status; }
    match crate::pe_luid::translate(out, node, &uuid, fake != 0) {
        Ok(()) => 0, Err(error) => error.0.get() as i32,
    }
}
forward!(cuDeviceGet(out: *mut i32, ordinal: i32) => C32_DEVICE_GET [pointer(out), ordinal as u32]);
forward!(cuDeviceGetName(out: *mut c_char, bytes: i32, dev: i32) => C32_DEVICE_NAME [pointer(out), bytes as u32, dev as u32]);
forward!(cuDeviceGetProperties(out: *mut c_void, dev: i32) => C32_DEVICE_PROPERTIES [pointer(out), dev as u32]);
forward!(cuDeviceComputeCapability(major: *mut i32, minor: *mut i32, dev: i32) => C32_COMPUTE_CAPABILITY [pointer(major), pointer(minor), dev as u32]);
forward!(cuDeviceTotalMem(out: *mut u32, dev: i32) => C32_TOTAL_MEM [pointer(out), dev as u32]);
forward!(cuDeviceTotalMem_v2(out: *mut usize, dev: i32) => C32_TOTAL_MEM [pointer(out), dev as u32]);
forward!(cuCtxPushCurrent(ctx: *mut c_void) => C32_CTX_PUSH [pointer(ctx)]);
forward!(cuCtxPushCurrent_v2(ctx: *mut c_void) => C32_CTX_PUSH [pointer(ctx)]);
forward!(cuCtxPopCurrent(ctx: *mut *mut c_void) => C32_CTX_POP [pointer(ctx)]);
forward!(cuCtxPopCurrent_v2(ctx: *mut *mut c_void) => C32_CTX_POP [pointer(ctx)]);
forward!(cuCtxCreate(ctx: *mut *mut c_void, flags: u32, dev: i32) => C32_CTX_CREATE [pointer(ctx), flags, dev as u32]);
forward!(cuCtxCreate_v2(ctx: *mut *mut c_void, flags: u32, dev: i32) => C32_CTX_CREATE [pointer(ctx), flags, dev as u32]);
forward!(cuCtxAttach(ctx: *mut *mut c_void, flags: u32) => C32_CTX_ATTACH [pointer(ctx), flags]);
forward!(cuCtxSynchronize() => C32_CTX_SYNC [0]);
forward!(cuCtxGetCurrent(ctx: *mut *mut c_void) => C32_CTX_CURRENT [pointer(ctx)]);
forward!(cuCtxSetCurrent(ctx: *mut c_void) => C32_CTX_SET [pointer(ctx)]);
forward!(cuCtxGetDevice(out: *mut i32) => C32_CTX_DEVICE [pointer(out)]);
forward!(cuCtxGetApiVersion(ctx: *mut c_void, version: *mut u32) => C32_CTX_API_VERSION [pointer(ctx), pointer(version)]);
forward!(cuMemAlloc(out: *mut u32, bytes: u32) => C32_ALLOC [pointer(out), bytes, 4]);
forward!(cuMemAlloc_v2(out: *mut u32, bytes: usize) => C32_ALLOC [pointer(out), bytes as u32, 4]);
forward!(cuMemFree(device: u32) => C32_FREE [device]);
forward!(cuMemFree_v2(device: u32) => C32_FREE [device]);

#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuCtxDestroy_v2(ctx: *mut c_void) -> i32 {
    let status = call(C32_CTX_DESTROY, &[pointer(ctx)]);
    if status == 0 { storage_destroy(ctx); }
    status
}
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuCtxDestroy(ctx: *mut c_void) -> i32 { cuCtxDestroy_v2(ctx) }
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuCtxDetach(ctx: *mut c_void) -> i32 {
    let mut destroyed = 0u32;
    let status = call(C32_CTX_DETACH, &[pointer(ctx), pointer(&mut destroyed)]);
    if status == 0 && destroyed != 0 { storage_destroy(ctx); }
    status
}

#[cfg_attr(windows, link(name = "kernel32"))]
extern "system" {
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn VirtualAlloc(address: *mut c_void, bytes: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, bytes: usize, kind: u32) -> i32;
}
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuDeviceGetAttribute(out: *mut i32, attr: i32, dev: i32) -> i32 {
    if attr != 0x20080402 { return call(C32_DEVICE_ATTR, &[pointer(out), attr as u32, dev as u32]); }
    if out.is_null() { return INVALID; }
    let mut count = 0;
    let result = cuDeviceGetCount(&mut count);
    if result != 0 { return result; }
    if dev < 0 || dev >= count { return 101; }
    if count != 1 || dev != 0 { return UNSUPPORTED; }
    let nvapi = GetModuleHandleA(b"nvapi.dll\0".as_ptr().cast());
    if nvapi.is_null() { return UNSUPPORTED; }
    let query = GetProcAddress(nvapi, b"nvapi_QueryInterface\0".as_ptr().cast());
    if query.is_null() { return UNSUPPORTED; }
    let query: unsafe extern "C" fn(u32) -> *mut c_void = std::mem::transmute(query);
    // Keep the actual PE32 wire layout even when native tests type-check this
    // file on x64: eight (u32 handle, u32 flags) records, 72 bytes in total.
    #[repr(C)] struct Topology { version: u32, count: u32, gpu: [[u32; 2]; 8] }
    const _: () = assert!(std::mem::size_of::<Topology>() == 72);
    let mut topo = Topology { version: 72 | (1 << 16), count: 0, gpu: [[0; 2]; 8] };
    let enumerate = query(0x5786cc6e);
    let get_id = query(0x6533ea3e);
    if enumerate.is_null() || get_id.is_null() { return UNSUPPORTED; }
    let enumerate: unsafe extern "C" fn(*mut Topology) -> i32 = std::mem::transmute(enumerate);
    let get_id: unsafe extern "C" fn(*mut c_void, *mut u32) -> i32 = std::mem::transmute(get_id);
    if enumerate(&mut topo) != 0 || topo.count != 1 { return UNSUPPORTED; }
    let mut id = 0u32;
    if get_id(ptr(topo.gpu[0][0]), &mut id) != 0 { return UNSUPPORTED; }
    std::ptr::write_unaligned(out.cast::<u32>(), id);
    0
}

static HOST: Mutex<[usize; 128]> = Mutex::new([0; 128]);
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuMemHostAlloc(out: *mut *mut c_void, bytes: usize, flags: u32) -> i32 {
    if out.is_null() || bytes == 0 { return INVALID; }
    if flags & !1 != 0 { return UNSUPPORTED; }
    let mut allocations = HOST.lock();
    let Some(slot) = allocations.iter_mut().find(|slot| **slot == 0) else { return OOM; };
    let memory = VirtualAlloc(std::ptr::null_mut(), bytes, 0x3000, 4);
    if memory.is_null() { return OOM; }
    let result = call(C32_HOST_REGISTER, &[pointer(memory), bytes as u32, flags]);
    if result != 0 { VirtualFree(memory, 0, 0x8000); }
    else { *slot = memory as usize; *out = memory; }
    result
}
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuMemAllocHost(out: *mut *mut c_void, bytes: u32) -> i32 { cuMemHostAlloc(out, bytes as usize, 0) }
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuMemAllocHost_v2(out: *mut *mut c_void, bytes: usize) -> i32 { cuMemHostAlloc(out, bytes, 0) }
#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuMemFreeHost(memory: *mut c_void) -> i32 {
    let mut allocations = HOST.lock();
    let Some(slot) = allocations.iter_mut().find(|slot| !memory.is_null() && **slot == memory as usize) else { return INVALID; };
    let result = call(C32_HOST_UNREGISTER, &[pointer(memory)]);
    if result == 0 { VirtualFree(memory, 0, 0x8000); *slot = 0; }
    result
}

macro_rules! copies {
    ($htod:ident, $dtoh:ident, $dtod:ident, $ha:ident, $da:ident, $dd:ident, $width:ty) => {
        forward!($htod(dst: u32, src: *const c_void, bytes: $width) => C32_HTOD [dst, pointer(src), bytes as u32]);
        forward!($dtoh(dst: *mut c_void, src: u32, bytes: $width) => C32_DTOH [pointer(dst), src, bytes as u32]);
        forward!($dtod(dst: u32, src: u32, bytes: $width) => C32_DTOD [dst, src, bytes as u32]);
        forward!($ha(dst: u32, src: *const c_void, bytes: $width, stream: *mut c_void) => C32_HTOD_ASYNC [dst, pointer(src), bytes as u32, pointer(stream)]);
        forward!($da(dst: *mut c_void, src: u32, bytes: $width, stream: *mut c_void) => C32_DTOH_ASYNC [pointer(dst), src, bytes as u32, pointer(stream)]);
        forward!($dd(dst: u32, src: u32, bytes: $width, stream: *mut c_void) => C32_DTOD_ASYNC [dst, src, bytes as u32, pointer(stream)]);
    };
}
copies!(cuMemcpyHtoD, cuMemcpyDtoH, cuMemcpyDtoD, cuMemcpyHtoDAsync, cuMemcpyDtoHAsync, cuMemcpyDtoDAsync, u32);
copies!(cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuMemcpyDtoD_v2, cuMemcpyHtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyDtoDAsync_v2, usize);
forward!(cuMemGetAddressRange(base: *mut u32, bytes: *mut u32, device: u32) => C32_ADDRESS_RANGE [pointer(base), pointer(bytes), device]);
forward!(cuMemGetAddressRange_v2(base: *mut u32, bytes: *mut usize, device: u32) => C32_ADDRESS_RANGE [pointer(base), pointer(bytes), device]);
forward!(cuEventCreate(out: *mut *mut c_void, flags: u32) => C32_EVENT_CREATE [pointer(out), flags]);
forward!(cuEventDestroy(event: *mut c_void) => C32_EVENT_DESTROY [pointer(event)]);
forward!(cuEventDestroy_v2(event: *mut c_void) => C32_EVENT_DESTROY [pointer(event)]);
forward!(cuEventRecord(event: *mut c_void, stream: *mut c_void) => C32_EVENT_RECORD [pointer(event), pointer(stream)]);
forward!(cuEventSynchronize(event: *mut c_void) => C32_EVENT_SYNC [pointer(event)]);
forward!(cuEventQuery(event: *mut c_void) => C32_EVENT_QUERY [pointer(event)]);
forward!(cuEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) => C32_EVENT_ELAPSED [pointer(ms), pointer(start), pointer(end)]);
forward!(cuStreamCreate(out: *mut *mut c_void, flags: u32) => C32_STREAM_CREATE [pointer(out), flags]);
forward!(cuStreamDestroy(stream: *mut c_void) => C32_STREAM_DESTROY [pointer(stream)]);
forward!(cuStreamDestroy_v2(stream: *mut c_void) => C32_STREAM_DESTROY [pointer(stream)]);
forward!(cuStreamSynchronize(stream: *mut c_void) => C32_STREAM_SYNC [pointer(stream)]);
forward!(cuStreamQuery(stream: *mut c_void) => C32_STREAM_QUERY [pointer(stream)]);
forward!(cuStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) => C32_STREAM_WAIT [pointer(event), pointer(stream), flags]);
macro_rules! memsets {
    ($d8:ident, $d16:ident, $d32:ident, $width:ty) => {
        forward!($d8(dst: u32, value: u8, count: $width) => C32_MEMSET8 [dst, value as u32, count as u32]);
        forward!($d16(dst: u32, value: u16, count: $width) => C32_MEMSET16 [dst, value as u32, count as u32]);
        forward!($d32(dst: u32, value: u32, count: $width) => C32_MEMSET [dst, value, count as u32]);
    };
}
memsets!(cuMemsetD8, cuMemsetD16, cuMemsetD32, u32);
memsets!(cuMemsetD8_v2, cuMemsetD16_v2, cuMemsetD32_v2, usize);
forward!(cuMemsetD8Async(dst: u32, value: u8, count: usize, stream: *mut c_void) => C32_MEMSET_ASYNC [dst, value as u32, count as u32, pointer(stream), 1]);
forward!(cuMemsetD16Async(dst: u32, value: u16, count: usize, stream: *mut c_void) => C32_MEMSET_ASYNC [dst, value as u32, count as u32, pointer(stream), 2]);
forward!(cuMemsetD32Async(dst: u32, value: u32, count: usize, stream: *mut c_void) => C32_MEMSET_ASYNC [dst, value, count as u32, pointer(stream), 4]);
forward!(cuModuleLoadData(out: *mut *mut c_void, image: *const c_void) => C32_MODULE_LOAD [pointer(out), pointer(image)]);
forward!(cuModuleUnload(module: *mut c_void) => C32_MODULE_UNLOAD [pointer(module)]);
forward!(cuModuleGetGlobal(device: *mut u32, bytes: *mut u32, module: *mut c_void, name: *const c_char) => C32_MODULE_GLOBAL [pointer(device), pointer(bytes), pointer(module), pointer(name), 4]);
forward!(cuModuleGetGlobal_v2(device: *mut u32, bytes: *mut usize, module: *mut c_void, name: *const c_char) => C32_MODULE_GLOBAL [pointer(device), pointer(bytes), pointer(module), pointer(name), 4]);
forward!(cuModuleGetTexRef(out: *mut *mut c_void, module: *mut c_void, name: *const c_char) => C32_TEX_GET [pointer(out), pointer(module), pointer(name)]);
forward!(cuTexRefSetFormat(tex: *mut c_void, format: i32, channels: i32) => C32_TEX_FORMAT [pointer(tex), format as u32, channels as u32]);
forward!(cuTexRefSetFlags(tex: *mut c_void, flags: u32) => C32_TEX_FLAGS [pointer(tex), flags]);
forward!(cuTexRefSetFilterMode(tex: *mut c_void, mode: i32) => C32_TEX_FILTER [pointer(tex), mode as u32]);
forward!(cuTexRefSetAddressMode(tex: *mut c_void, dim: i32, mode: i32) => C32_TEX_ADDRESS_MODE [pointer(tex), dim as u32, mode as u32]);
forward!(cuTexRefSetMipmapFilterMode(tex: *mut c_void, mode: i32) => C32_TEX_MIPMAP [pointer(tex), 0, mode as u32]);
forward!(cuTexRefSetMipmapLevelBias(tex: *mut c_void, bias: f32) => C32_TEX_MIPMAP [pointer(tex), 1, bias.to_bits()]);
forward!(cuTexRefSetMipmapLevelClamp(tex: *mut c_void, lo: f32, hi: f32) => C32_TEX_MIPMAP [pointer(tex), 2, lo.to_bits(), hi.to_bits()]);
forward!(cuTexRefSetMaxAnisotropy(tex: *mut c_void, value: u32) => C32_TEX_MIPMAP [pointer(tex), 3, value]);
forward!(cuTexRefSetAddress(out: *mut usize, tex: *mut c_void, device: u32, bytes: u32) => C32_TEX_ADDRESS [pointer(tex), device, bytes, pointer(out)]);
forward!(cuTexRefSetAddress_v2(out: *mut usize, tex: *mut c_void, device: u32, bytes: usize) => C32_TEX_ADDRESS [pointer(tex), device, bytes as u32, pointer(out)]);
forward!(cuModuleGetFunction(out: *mut *mut c_void, module: *mut c_void, name: *const c_char) => C32_FUNCTION [pointer(out), pointer(module), pointer(name)]);
forward!(cuLaunchKernel(function: *mut c_void, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, shared: u32,
    stream: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void) => C32_LAUNCH
    [pointer(function), gx, gy, gz, bx, by, bz, shared, pointer(stream), pointer(params), pointer(extra)]);

// SPDX-License-Identifier: MIT OR Apache-2.0
//! PE-owned WoW64 dark tables, context storage and heap records from fa81f481.
//! The off root does not contain this code. No native code/data address is
//! returned to PE32, and destructors execute after releasing the storage lock.
#![cfg_attr(not(windows), allow(dead_code))]
use crate::{wow64_protocol::*, wow64_pe::*};
use std::ffi::c_void;
use crate::pe_sync::Mutex;

struct Table<const N: usize>([*const c_void; N]);
// Immutable PE tables contain only size words and process-lifetime PE thunks.
unsafe impl<const N: usize> Sync for Table<N> {}
const fn size_word(words: usize) -> *const c_void {
    (words * std::mem::size_of::<usize>()) as *const c_void
}
const NULL: *const c_void = std::ptr::null();

unsafe extern "system" fn rt_load(out: *mut *mut c_void, image: *const c_void) -> i32 {
    cuModuleLoadData(out, image)
}
unsafe extern "system" fn rt_context(ctx: *mut *mut c_void, dev: i32) -> i32 {
    call(C32_DARK_RUNTIME, &[2, pointer(ctx), dev as u32])
}
unsafe extern "system" fn rt_flags(dev: i32, flags: u32) -> i32 { call(C32_DARK_RUNTIME, &[4, dev as u32, flags]) }
unsafe extern "system" fn rt_device(dev: i32) -> i32 { call(C32_DARK_RUNTIME, &[5, dev as u32]) }
unsafe extern "system" fn rt_load_ext(out: *mut *mut c_void, image: *const c_void,
    a: *mut c_void, b: *mut c_void, c: u32) -> i32 {
    if !a.is_null() || !b.is_null() || c != 0 { UNSUPPORTED } else { cuModuleLoadData(out, image) }
}
unsafe extern "system" fn rt_done(value: u32) { let _ = call(C32_DARK_RUNTIME, &[7, value]); }
unsafe extern "system" fn rt_load_header(image: *const c_void, out: *mut *mut c_void,
    a: *mut c_void, b: *mut c_void, c: u32) -> i32 { rt_load_ext(out, image, a, b, c) }
unsafe extern "system" fn rt_compilers() -> i32 { call(C32_DARK_RUNTIME, &[12]) }
static RUNTIME: Table<13> = Table([size_word(13), rt_load as *const c_void,
    rt_context as *const c_void, NULL, rt_flags as *const c_void, rt_device as *const c_void,
    rt_load_ext as *const c_void, rt_done as *const c_void, rt_load_header as *const c_void,
    NULL, NULL, NULL, rt_compilers as *const c_void]);
static CREATE: Table<2> = Table([size_word(2), cuCtxCreate_v2 as *const c_void]);
static TLS: Table<4> = Table([size_word(4), NULL, NULL, NULL]);

// These buffers deliberately remain writable for the CUDA runtime, with the
// same lifetime and word counts as the original PE32 frontend.
static mut HOOK_BUFFER1: [u32; 1024] = [0; 1024];
static mut HOOK_BUFFER2: [u32; 14] = [0; 14];
unsafe extern "system" fn hooks_buffer1(out: *mut *mut c_void, count: *mut usize) {
    if !out.is_null() { *out = std::ptr::addr_of_mut!(HOOK_BUFFER1).cast(); }
    if !count.is_null() { *count = 1024; }
}
unsafe extern "system" fn hooks_buffer2(out: *mut *mut c_void, count: *mut usize) {
    if !out.is_null() { *out = std::ptr::addr_of_mut!(HOOK_BUFFER2).cast(); }
    if !count.is_null() { *count = 14; }
}
static HOOKS: Table<7> = Table([size_word(7), NULL, hooks_buffer1 as *const c_void,
    NULL, NULL, NULL, hooks_buffer2 as *const c_void]);

#[derive(Clone, Copy, Default)]
struct Entry { ctx: usize, key: usize, value: usize, destructor: usize, used: bool }
static STORAGE: Mutex<[Entry; 128]> = Mutex::new([Entry { ctx: 0, key: 0, value: 0, destructor: 0, used: false }; 128]);
type Destructor = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);

unsafe fn storage_context(ctx: &mut *mut c_void) -> i32 {
    if ctx.is_null() {
        let result = cuCtxGetCurrent(ctx);
        if result != 0 { return result; }
    }
    let mut version = 0;
    cuCtxGetApiVersion(*ctx, &mut version)
}
unsafe extern "system" fn storage_put(mut ctx: *mut c_void, key: *mut c_void,
    value: *mut c_void, destructor: Option<Destructor>) -> i32 {
    let result = storage_context(&mut ctx);
    if result != 0 { return result; }
    let mut storage = STORAGE.lock();
    let found = storage.iter().position(|entry| entry.used && entry.ctx == ctx as usize && entry.key == key as usize)
        .or_else(|| storage.iter().position(|entry| !entry.used));
    let Some(slot) = found else { return OOM; };
    storage[slot] = Entry { ctx: ctx as usize, key: key as usize, value: value as usize,
        destructor: destructor.map(|fn_ptr| fn_ptr as usize).unwrap_or(0), used: true };
    0
}
unsafe extern "system" fn storage_delete(mut ctx: *mut c_void, key: *mut c_void) -> i32 {
    let result = storage_context(&mut ctx);
    if result != 0 { return result; }
    let mut storage = STORAGE.lock();
    let Some(entry) = storage.iter_mut().find(|entry| entry.used && entry.ctx == ctx as usize && entry.key == key as usize)
        else { return INVALID; };
    *entry = Entry::default();
    0
}
unsafe extern "system" fn storage_get(out: *mut *mut c_void, mut ctx: *mut c_void, key: *mut c_void) -> i32 {
    if out.is_null() { return INVALID; }
    let result = storage_context(&mut ctx);
    if result != 0 { return result; }
    let storage = STORAGE.lock();
    let Some(entry) = storage.iter().find(|entry| entry.used && entry.ctx == ctx as usize && entry.key == key as usize)
        else { return INVALID; };
    *out = entry.value as *mut c_void;
    0
}
pub(super) unsafe fn storage_destroy(ctx: *mut c_void) {
    let mut callbacks = [Entry::default(); 128];
    let mut count = 0;
    {
        let mut storage = STORAGE.lock();
        for entry in storage.iter_mut().filter(|entry| entry.used && entry.ctx == ctx as usize) {
            callbacks[count] = *entry;
            count += 1;
            *entry = Entry::default();
        }
    }
    for entry in &callbacks[..count] {
        if entry.destructor != 0 {
            let destructor: Destructor = std::mem::transmute(entry.destructor);
            destructor(ctx, entry.key as *mut c_void, entry.value as *mut c_void);
        }
    }
}
// Original PE32's context storage includes a trailing null word. This differs
// from the x64 C table's three words and is intentionally architecture-local.
static CONTEXT_STORAGE: Table<4> = Table([storage_put as *const c_void,
    storage_delete as *const c_void, storage_get as *const c_void, NULL]);

#[cfg_attr(windows, link(name = "kernel32"))]
extern "system" {
    fn GetProcessHeap() -> *mut c_void;
    fn HeapAlloc(heap: *mut c_void, flags: u32, bytes: usize) -> *mut c_void;
    fn HeapFree(heap: *mut c_void, flags: u32, memory: *mut c_void) -> i32;
}
#[repr(C)]
struct HeapRecord { x: usize, y: usize }
static HEAP: Mutex<[usize; 128]> = Mutex::new([0; 128]);
unsafe extern "system" fn heap_alloc(out: *mut *const c_void, x: usize, y: usize) -> i32 {
    if out.is_null() { return INVALID; }
    let mut heap = HEAP.lock();
    let Some(slot) = heap.iter_mut().find(|slot| **slot == 0) else { return OOM; };
    let memory = HeapAlloc(GetProcessHeap(), 0, std::mem::size_of::<HeapRecord>()).cast::<HeapRecord>();
    if memory.is_null() { return OOM; }
    memory.write(HeapRecord { x, y });
    *slot = memory as usize;
    *out = memory.cast();
    0
}
unsafe extern "system" fn heap_free(record: *const c_void, out_y: *mut usize) -> i32 {
    let mut heap = HEAP.lock();
    let Some(slot) = heap.iter_mut().find(|slot| !record.is_null() && **slot == record as usize)
        else { return INVALID; };
    if !out_y.is_null() { *out_y = (*record.cast::<HeapRecord>()).y; }
    HeapFree(GetProcessHeap(), 0, record.cast_mut());
    *slot = 0;
    0
}
static HEAP_TABLE: Table<3> = Table([size_word(3), heap_alloc as *const c_void, heap_free as *const c_void]);

#[cfg_attr(all(windows, target_arch = "x86"), no_mangle)]
pub unsafe extern "system" fn cuGetExportTable(out: *mut *const c_void, uuid: *const u8) -> i32 {
    if out.is_null() || uuid.is_null() { return INVALID; }
    *out = std::ptr::null();
    let uuid = std::ptr::read_unaligned(uuid.cast::<[u8; 16]>());
    let table = match uuid {
        [0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9] => RUNTIME.0.as_ptr(),
        [0xa0,0x94,0x79,0x8c,0x2e,0x74,0x2e,0x74,0x93,0xf2,0x08,0x00,0x20,0x0c,0x0a,0x66] => HOOKS.0.as_ptr(),
        [0x42,0xd8,0x5a,0x81,0x23,0xf6,0xcb,0x47,0x82,0x98,0xf6,0xe7,0x8a,0x3a,0xec,0xdc] => TLS.0.as_ptr(),
        [0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93] => CONTEXT_STORAGE.0.as_ptr(),
        [0x19,0x5b,0xcb,0xf4,0xd6,0x7d,0x02,0x4a,0xac,0xc5,0x1d,0x29,0xce,0xa6,0x31,0xae] => HEAP_TABLE.0.as_ptr(),
        [0x0c,0xa5,0x0b,0x8c,0x10,0x04,0x92,0x9a,0x89,0xa7,0xd0,0xdf,0x10,0xe7,0x72,0x86] => CREATE.0.as_ptr(),
        _ => return UNSUPPORTED,
    };
    *out = table.cast();
    0
}

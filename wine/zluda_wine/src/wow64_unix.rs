// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rust port of fa81f481's bounded WoW64 dispatcher. See LICENSES/PROVENANCE.md.
//! The mutex and all limits are inherited C behaviour. No Windows callback is
//! invoked here, and native function pointers remain on the Unix side.
use crate::{unixlib, wow64_protocol::*};
use std::{ffi::{c_char, c_void, CStr, CString}, sync::{Mutex, OnceLock}};

#[path = "wow64_modules.rs"] mod modules;
#[path = "wow64_ops.rs"] mod ops;
#[cfg(all(test,zluda_wine_oracles))] #[path = "wow64_fixtures.rs"] mod fixtures;
#[cfg(test)] #[path = "../tests/pci_parity.rs"] mod pci_tests;
const POOL_BYTES: u32 = 512 * 1024 * 1024;
const MAX_HANDLES: usize = 2048;
const MAX_ALLOCS: usize = 256;
const MAX_CONTEXTS: usize = 16;
const MAX_ARGS: usize = 32;
const MAX_GLOBALS: usize = 32;
type Answer = Result<(), i32>;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct ArgInfo { size: u32, align: u32 }
#[derive(Default)]
struct Global { name: CString, offset: u32, bytes: u32 }
#[derive(Default)]
struct Handle {
    native: usize, name: Option<CString>, id: u32, owner: u32, ctx_id: u32,
    kind: u8, nargs: u32, args: [ArgInfo; MAX_ARGS], globals: Vec<Global>,
}
#[derive(Clone, Copy, Default)]
struct Allocation { offset: u32, bytes: u32, reserved: u32, owner: u32 }
struct Context { native: usize, pool: u64, id: u32, refs: u32, ranges: [Allocation; MAX_ALLOCS] }
impl Default for Context {
    fn default() -> Self { Self { native: 0, pool: 0, id: 0, refs: 0,
        ranges: [Allocation::default(); MAX_ALLOCS] } }
}
struct State {
    ready: bool, generation: u32, handles: Vec<Handle>, contexts: Vec<Context>,
    active: Option<usize>, dark: Option<[usize; 2]>,
}
impl Default for State {
    fn default() -> Self { Self { ready: false, generation: 0,
        handles: (0..MAX_HANDLES).map(|_| Handle::default()).collect(),
        contexts: (0..MAX_CONTEXTS).map(|_| Context::default()).collect(), active: None, dark: None } }
}
static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn address<T>(value: &T) -> u64 { value as *const T as usize as u64 }
fn mutable<T>(value: &mut T) -> u64 { value as *mut T as usize as u64 }
unsafe fn write32(target: u32, value: u32) {
    std::ptr::write_unaligned(ptr::<u32>(target), value);
}
unsafe fn read32(target: u32) -> u32 { std::ptr::read_unaligned(ptr::<u32>(target)) }

unsafe fn native(name: &str, args: &[u64]) -> Answer {
    let mut call = unixlib::Call::new(unixlib::function_id(name));
    call.args[..args.len()].copy_from_slice(args);
    crate::unix_thunks::dispatch(&mut call).map_err(|error| error.0.get() as i32)?;
    if call.result == 0 { Ok(()) } else { Err(call.result as i32) }
}
unsafe fn get_table(uuid: &[u8; 16]) -> Result<*const usize, i32> {
    let function: unsafe extern "C" fn(*mut *const usize, *const u8) -> i32 =
        std::mem::transmute(crate::unix_wrappers::symbol(b"cuGetExportTable\0")
                            .map_err(|error| error.0.get() as i32)?);
    let mut table = std::ptr::null();
    let status = function(&mut table, uuid.as_ptr());
    if status != 0 { return Err(status); }
    if table.is_null() || table as usize % 8 != 0 { return Err(UNSUPPORTED); }
    Ok(table)
}
impl State {
    unsafe fn prepare(&mut self) -> Answer {
        if !self.ready {
            for name in REQUIRED {
                crate::unix_wrappers::symbol(name).map_err(|_| UNSUPPORTED)?;
            }
            self.ready = true;
        }
        Ok(())
    }
    fn handle(&self, id: u32, kind: u8) -> Option<usize> {
        self.handles.iter().position(|handle| handle.id == id && handle.kind == kind &&
            (kind == 1 || kind == 4 || self.active.is_some_and(|i| handle.ctx_id == self.contexts[i].id)))
    }
    fn context_handle(&self, id: u32) -> Option<usize> {
        self.handle(id, 1).or_else(|| self.handle(id, 4))
    }
    fn new_handle(&mut self, value: usize, kind: u8) -> Result<usize, i32> {
        let slot = self.handles.iter().position(|handle| handle.kind == 0).ok_or(OOM)?;
        if self.generation == 0x00ffffff { return Err(OOM); }
        self.generation += 1;
        let id = 0x40000000 | (self.generation << 4);
        let ctx_id = if kind == 1 || kind == 4 { id }
            else { self.contexts[self.active.ok_or(INVALID_CONTEXT)?].id };
        self.handles[slot] = Handle { native: value, id, kind, ctx_id, ..Handle::default() };
        Ok(slot)
    }
    fn context_slot(&self, value: usize) -> Option<usize> {
        self.contexts.iter().position(|ctx| ctx.native == value)
            .or_else(|| self.contexts.iter().position(|ctx| ctx.native == 0))
    }
    unsafe fn current_context(&mut self) -> Answer {
        self.active = None;
        let mut actual = 0usize;
        native("cuCtxGetCurrent", &[mutable(&mut actual)])?;
        self.active = self.contexts.iter().position(|ctx| actual != 0 && ctx.native == actual);
        self.active.map(|_| ()).ok_or(INVALID_CONTEXT)
    }
    unsafe fn destroy_context(&mut self, slot: usize) -> Answer {
        let value = self.handles[slot].native;
        let ctx = self.context_slot(value).filter(|&i| self.contexts[i].native == value)
            .ok_or(INVALID_CONTEXT)?;
        let mut previous = 0usize;
        native("cuCtxGetCurrent", &[mutable(&mut previous)])?;
        if previous != value { native("cuCtxPushCurrent_v2", &[value as u64])?; }
        self.active = Some(ctx);
        let mut result = Ok(());
        if self.contexts[ctx].pool != 0 {
            result = native("cuMemFree_v2", &[self.contexts[ctx].pool]);
            if result.is_ok() { self.contexts[ctx].pool = 0; }
        }
        if previous != value {
            let mut popped = 0usize;
            let restore = native("cuCtxPopCurrent_v2", &[mutable(&mut popped)]);
            if result.is_ok() { result = restore; }
        }
        result?;
        native("cuCtxDestroy_v2", &[value as u64])?;
        let id = self.contexts[ctx].id;
        for handle in &mut self.handles {
            if handle.kind != 0 && handle.ctx_id == id { *handle = Handle::default(); }
        }
        self.contexts[ctx] = Context::default();
        self.active = None;
        Ok(())
    }
    fn context(&self) -> &Context { &self.contexts[self.active.expect("resolved current context")] }
    fn context_mut(&mut self) -> &mut Context { &mut self.contexts[self.active.expect("resolved current context")] }
    fn translate(&self, offset: u32, bytes: u32) -> Result<u64, i32> {
        if offset == 0 || bytes == 0 { return Err(INVALID); }
        for range in &self.context().ranges {
            if range.bytes != 0 && offset >= range.offset && offset - range.offset < range.bytes &&
                bytes <= range.bytes - (offset - range.offset) { return Ok(self.context().pool + offset as u64); }
        }
        Err(INVALID)
    }
    unsafe fn alloc_range(&mut self, requested: u32, owner: u32) -> Result<u32, i32> {
        if requested == 0 { return Err(INVALID); }
        if requested > POOL_BYTES - 256 { return Err(OOM); }
        let slot = self.context().ranges.iter().position(|range| range.bytes == 0).ok_or(OOM)?;
        let bytes = (requested + 255) & !255;
        let mut offset = 256;
        loop {
            if offset > POOL_BYTES - bytes { return Err(OOM); }
            let mut next = offset;
            for range in &self.context().ranges {
                if range.bytes != 0 && offset < range.offset + range.reserved && offset + bytes > range.offset {
                    next = next.max(range.offset + range.reserved);
                }
            }
            if next == offset { break; }
            offset = next;
        }
        if self.context().pool == 0 {
            let mut pool = 0u64;
            native("cuMemAlloc_v2", &[mutable(&mut pool), POOL_BYTES as u64])?;
            self.context_mut().pool = pool;
        }
        self.context_mut().ranges[slot] = Allocation { offset, bytes: requested, reserved: bytes, owner };
        Ok(offset)
    }
    unsafe fn dark_runtime(&mut self, a: &[u32; 16]) -> Answer {
        let table = get_table(&[0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9])?;
        let slot = a[0] as usize;
        if *table < 13 * 8 || slot >= 13 || *table.add(slot) == 0 { return Err(UNSUPPORTED); }
        let function = *table.add(slot);
        let status = match slot {
            2 => {
                if a[1] == 0 { return Err(INVALID); }
                let mut value = 0usize;
                let function: unsafe extern "C" fn(*mut usize, i32) -> i32 = std::mem::transmute(function);
                let status = function(&mut value, a[2] as i32);
                if status != 0 { return Err(status); }
                let h = match self.handles.iter().position(|h| (h.kind == 1 || h.kind == 4) && h.native == value) {
                    Some(h) => h, None => self.new_handle(value, 4)?,
                };
                let ctx = self.context_slot(value).ok_or(OOM)?;
                if self.contexts[ctx].native == 0 {
                    self.contexts[ctx].native = value;
                    self.contexts[ctx].id = self.handles[h].id;
                }
                write32(a[1], self.handles[h].id);
                0
            }
            4 => {
                let function: unsafe extern "C" fn(i32, u32) -> i32 = std::mem::transmute(function);
                function(a[1] as i32, a[2])
            }
            5 => {
                let function: unsafe extern "C" fn(i32) -> i32 = std::mem::transmute(function);
                function(a[1] as i32)
            }
            7 => {
                let function: unsafe extern "C" fn(usize) = std::mem::transmute(function);
                function(a[1] as usize); 0
            }
            12 => {
                let function: unsafe extern "C" fn() -> i32 = std::mem::transmute(function);
                function()
            }
            _ => return Err(UNSUPPORTED),
        };
        if status == 0 { Ok(()) } else { Err(status) }
    }
    unsafe fn dispatch(&mut self, op: u32, a: &[u32; 16]) -> Answer {
        self.prepare()?;
        let x = a.map(u64::from);
        match op {
            C32_INIT => return native("cuInit", &x[..1]),
            C32_VERSION | C32_DEVICE_COUNT | C32_DEVICE_GET | C32_DEVICE_NAME | C32_DEVICE_ATTR |
            C32_DEVICE_PROPERTIES | C32_COMPUTE_CAPABILITY => {
                if a[0] == 0 || (op == C32_COMPUTE_CAPABILITY && a[1] == 0) { return Err(INVALID); }
                let (name, count) = match op {
                    C32_VERSION => ("cuDriverGetVersion", 1), C32_DEVICE_COUNT => ("cuDeviceGetCount", 1),
                    C32_DEVICE_GET => ("cuDeviceGet", 2), C32_DEVICE_NAME => ("cuDeviceGetName", 3),
                    C32_DEVICE_ATTR => ("cuDeviceGetAttribute", 3),
                    C32_DEVICE_PROPERTIES => ("cuDeviceGetProperties", 2),
                    _ => ("cuDeviceComputeCapability", 3),
                };
                return native(name, &x[..count]);
            }
            C32_TOTAL_MEM => {
                if a[0] == 0 { return Err(INVALID); }
                let mut bytes = 0usize;
                native("cuDeviceTotalMem_v2", &[mutable(&mut bytes), x[1]])?;
                write32(a[0], bytes.min((POOL_BYTES - 256) as usize) as u32);
                return Ok(());
            }
            C32_CTX_PUSH => {
                let h = self.context_handle(a[0]).ok_or(INVALID_HANDLE)?;
                return native("cuCtxPushCurrent_v2", &[self.handles[h].native as u64]);
            }
            C32_CTX_POP | C32_CTX_CURRENT => {
                if a[0] == 0 { return Err(INVALID); }
                let mut value = 0usize;
                native(if op == C32_CTX_POP { "cuCtxPopCurrent_v2" } else { "cuCtxGetCurrent" }, &[mutable(&mut value)])?;
                let found = self.handles.iter().filter(|h| (h.kind == 1 || h.kind == 4) && h.native == value);
                let id = if op == C32_CTX_POP { found.map(|h| h.id).next().ok_or(INVALID_HANDLE)? }
                    else { found.map(|h| h.id).last().unwrap_or(0) };
                write32(a[0], id);
                return Ok(());
            }
            C32_CTX_SET => {
                let value = if a[0] == 0 { 0 } else {
                    self.handles[self.context_handle(a[0]).ok_or(INVALID_CONTEXT)?].native };
                return native("cuCtxSetCurrent", &[value as u64]);
            }
            C32_CTX_API_VERSION => {
                let h = self.context_handle(a[0]).ok_or(INVALID_CONTEXT)?;
                if a[1] == 0 { return Err(INVALID); }
                return native("cuCtxGetApiVersion", &[self.handles[h].native as u64, x[1]]);
            }
            C32_DARK_RUNTIME => return self.dark_runtime(a),
            C32_CTX_CREATE => {
                if a[0] == 0 { return Err(INVALID); }
                let ctx = self.context_slot(0).ok_or(OOM)?;
                let mut value = 0usize;
                native("cuCtxCreate_v2", &[mutable(&mut value), x[1], x[2]])?;
                let h = match self.new_handle(value, 1) { Ok(h) => h, Err(error) => {
                    let _ = native("cuCtxDestroy_v2", &[value as u64]); return Err(error);
                }};
                self.contexts[ctx].native = value;
                self.contexts[ctx].id = self.handles[h].id;
                self.contexts[ctx].refs = 1;
                self.active = Some(ctx);
                write32(a[0], self.handles[h].id);
                return Ok(());
            }
            C32_CTX_DESTROY => {
                let h = self.handle(a[0], 1).ok_or(INVALID_HANDLE)?;
                return self.destroy_context(h);
            }
            _ => {}
        }
        self.current_context()?;
        self.dispatch_current(op, a)
    }
}

// PCI identifiers use the same native Call dispatch as PE64, without the
// unrelated bounded-symbol preflight, state mutex or current-context lookup.
// In particular, let CUDA decide null-output/device/initialization errors.
unsafe fn pci_attribute(a: &[u32; 16]) -> i32 {
    let mut request = unixlib::Call::new(unixlib::function_id("cuDeviceGetAttribute"));
    request.args[0] = u64::from(a[0]);
    request.args[1] = u64::from(a[1]);
    request.args[2] = u64::from(a[2]);
    if crate::unix_wrappers::dispatch((&mut request as *mut unixlib::Call).cast()) != 0 {
        return 3; // Same transport failure mapping as PE64.
    }
    request.result as i32
}

unsafe extern "C" fn dispatch32(pointer: *mut c_void) -> i32 {
    const BAD: i32 = 0xc000000du32 as i32;
    if pointer.is_null() || pointer as usize % std::mem::align_of::<Call32>() != 0 { return BAD; }
    let bytes = pointer.cast::<u8>();
    if std::ptr::read(bytes.cast::<u64>()) != MAGIC ||
        std::ptr::read(bytes.add(8).cast::<u32>()) != VERSION ||
        std::ptr::read(bytes.add(12).cast::<u32>()) as usize != std::mem::size_of::<Call32>() { return BAD; }
    let call = &mut *pointer.cast::<Call32>();
    if !call.valid() { return BAD; }
    if call.opcode == HELLO { call.result = 0; return 0; } // No backend or HIP load.
    // The retired operation is a tombstone, never a UUID or native request.
    if call.opcode == C32_RESERVED_59 { call.result = UNSUPPORTED; return 0; }
    if call.opcode == C32_DEVICE_ATTR && matches!(call.args[1], 50 | 33 | 34) {
        call.result = pci_attribute(&call.args);
        return 0;
    }
    if !crate::unix_wrappers::enabled() { call.result = UNSUPPORTED; return 0; }
    call.result = match STATE.get_or_init(|| Mutex::new(State::default())).lock() {
        Ok(mut state) => state.dispatch(call.opcode, &call.args).err().unwrap_or(0),
        Err(_) => 999,
    };
    0
}

#[no_mangle]
pub static __wine_unix_call_wow64_funcs: [unixlib::UnixEntry; 1] = [dispatch32];

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Module globals, PTX selection and the bounded 32-bit kernel argument ABI.
use super::*;
type GlobalsFn = unsafe extern "C" fn(*mut *const c_char, *mut *const c_void,
    *mut u32, *mut u32, *mut u32, usize) -> i32;
type ArgsFn = unsafe extern "C" fn(*mut ArgInfo, *mut u32, usize) -> i32;

unsafe fn status(value: i32) -> Answer { if value == 0 { Ok(()) } else { Err(value) } }
unsafe fn fatbin_has_ptx(data: *const u8) -> i32 {
    let magic = std::ptr::read_unaligned(data.cast::<u32>());
    let version = std::ptr::read_unaligned(data.add(4).cast::<u16>());
    let header = std::ptr::read_unaligned(data.add(6).cast::<u16>());
    let mut bytes = std::ptr::read_unaligned(data.add(8).cast::<u64>());
    if magic != 0xba55ed50 || version != 1 || header < 16 || bytes > 64 * 1024 * 1024 { return -1; }
    let mut data = data.add(header as usize);
    let mut ptx = 0;
    while bytes != 0 {
        if bytes < 16 { return -1; }
        let kind = std::ptr::read_unaligned(data.cast::<u16>());
        let version = std::ptr::read_unaligned(data.add(2).cast::<u16>());
        let header = std::ptr::read_unaligned(data.add(4).cast::<u32>()) as u64;
        let payload = std::ptr::read_unaligned(data.add(8).cast::<u64>());
        if ![0x100, 0x101].contains(&version) || ![1, 2].contains(&kind) ||
            header < 16 || header > bytes || payload > bytes - header { return -1; }
        if kind == 1 { ptx = 1; }
        let step = header + payload;
        data = data.add(step as usize);
        bytes -= step;
    }
    ptx
}

impl State {
    fn release_globals(&mut self, slot: usize) {
        let id = self.handles[slot].id;
        self.handles[slot].globals.clear();
        for range in &mut self.context_mut().ranges {
            if range.owner == id { *range = Allocation::default(); }
        }
    }
    unsafe fn init_globals(&mut self, slot: usize, count: u32) -> Answer {
        if count as usize > MAX_GLOBALS { return Err(UNSUPPORTED); }
        if count == 0 { return Ok(()); }
        let function: GlobalsFn = std::mem::transmute(self.dark.ok_or(UNSUPPORTED)?[0]);
        let mut names = [std::ptr::null::<c_char>(); MAX_GLOBALS];
        let mut initializers = [std::ptr::null::<c_void>(); MAX_GLOBALS];
        let mut sizes = [0u32; MAX_GLOBALS];
        let mut alignments = [0u32; MAX_GLOBALS];
        let mut returned = count;
        status(function(names.as_mut_ptr(), initializers.as_mut_ptr(), sizes.as_mut_ptr(),
            alignments.as_mut_ptr(), &mut returned, self.handles[slot].native))?;
        if returned != count { return Err(UNSUPPORTED); }
        for i in 0..count as usize {
            if names[i].is_null() || initializers[i].is_null() || sizes[i] == 0 || alignments[i] == 0 ||
                alignments[i] > 256 || !alignments[i].is_power_of_two() { return Err(UNSUPPORTED); }
            let name = CStr::from_ptr(names[i]).to_owned();
            let offset = self.alloc_range(sizes[i], self.handles[slot].id)?;
            self.handles[slot].globals.push(Global { name, offset, bytes: sizes[i] });
            native("cuMemcpyHtoD_v2", &[self.context().pool + offset as u64,
                initializers[i] as usize as u64, sizes[i] as u64])?;
        }
        Ok(())
    }
    unsafe fn load_module(&mut self, a: &[u32; 16]) -> Answer {
        if a[0] == 0 || a[1] == 0 { return Err(INVALID); }
        let mut module = 0usize;
        let mut image = ptr::<u8>(a[1]).cast_const();
        #[repr(C)]
        struct Wrapper { magic: u32, version: u32, data: *const u8, extra: *const u8 }
        let mut wrapper = Wrapper { magic: 0, version: 0, data: std::ptr::null(), extra: std::ptr::null() };
        if read32(a[1]) == 0x466243b1 {
            if read32(a[1] + 4) != 1 { return Err(UNSUPPORTED); }
            wrapper.magic = 0x466243b1;
            wrapper.version = 1;
            wrapper.data = ptr(read32(a[1] + 8));
            wrapper.extra = ptr(read32(a[1] + 12));
            if wrapper.data.is_null() { return Err(INVALID); }
            if fatbin_has_ptx(wrapper.data) == 0 { return Err(209); }
            image = (&wrapper as *const Wrapper).cast();
        }
        native("cuModuleLoadData", &[mutable(&mut module), image as usize as u64])?;
        let metadata = (|| {
            let table = get_table(&[0xee,0x37,0x6a,0x1b,0xb8,0x58,0x42,0xeb,0xa5,0x2d,0x9a,0x10,0x6d,0x50,0xd9,0x14])?;
            if *table == 0 || *table.add(1) == 0 { return Err(UNSUPPORTED); }
            self.dark = Some([*table, *table.add(1)]);
            let function: GlobalsFn = std::mem::transmute(*table);
            let mut count = 0;
            status(function(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), &mut count, module))?;
            Ok(count)
        })();
        let count = match metadata { Ok(count) => count, Err(error) => {
            let _ = native("cuModuleUnload", &[module as u64]); return Err(error);
        }};
        let slot = match self.new_handle(module, 2) { Ok(slot) => slot, Err(error) => {
            let _ = native("cuModuleUnload", &[module as u64]); return Err(error);
        }};
        if let Err(error) = self.init_globals(slot, count) {
            self.release_globals(slot);
            self.handles[slot] = Handle::default();
            let _ = native("cuModuleUnload", &[module as u64]);
            return Err(error);
        }
        write32(a[0], self.handles[slot].id);
        Ok(())
    }
    unsafe fn launch(&mut self, a: &[u32; 16]) -> Answer {
        let h = self.handle(a[0], 3).ok_or(INVALID_HANDLE)?;
        if a[9] != 0 && a[10] != 0 { return Err(INVALID); }
        let stream = self.stream(a[8])?;
        let nargs = self.handles[h].nargs as usize;
        if self.context().pool == 0 || (nargs != 0 && a[9] == 0 && a[10] == 0) { return Err(INVALID); }
        let module = self.handle(self.handles[h].owner, 2).ok_or(INVALID_HANDLE)?;
        let mut params = [0usize; MAX_ARGS + 1 + MAX_GLOBALS];
        if a[10] != 0 {
            let (mut buffer, mut bytes) = (0u32, 0u32);
            let (mut have_buffer, mut have_size, mut ended) = (false, false, false);
            for i in (0..16).step_by(2) {
                let tag = std::ptr::read_unaligned(ptr::<u32>(a[10]).add(i));
                if tag == 0 { ended = true; break; }
                let value = std::ptr::read_unaligned(ptr::<u32>(a[10]).add(i + 1));
                match tag {
                    1 if !have_buffer => { buffer = value; have_buffer = true; }
                    2 if !have_size && value != 0 => { bytes = read32(value); have_size = true; }
                    _ => return Err(INVALID),
                }
            }
            if !ended || !have_buffer || !have_size || (buffer == 0 && nargs != 0) || bytes > u32::MAX - buffer {
                return Err(INVALID);
            }
            let mut offset = 0u32;
            for (i, param) in params[..nargs].iter_mut().enumerate() {
                let ArgInfo { size, align } = self.handles[h].args[i];
                if align == 0 || !align.is_power_of_two() || offset > u32::MAX - (align - 1) { return Err(INVALID); }
                offset = (offset + align - 1) & !(align - 1);
                if offset > bytes || size > bytes - offset { return Err(INVALID); }
                *param = (buffer + offset) as usize;
                offset += size;
            }
        } else {
            for (i, param) in params[..nargs].iter_mut().enumerate() {
                let value = std::ptr::read_unaligned(ptr::<u32>(a[9]).add(i));
                if value == 0 { return Err(INVALID); }
                *param = value as usize;
            }
        }
        params[nargs] = address(&self.context().pool) as usize;
        for (i, global) in self.handles[module].globals.iter().enumerate() {
            params[nargs + 1 + i] = address(&global.offset) as usize;
        }
        native("cuLaunchKernel", &[self.handles[h].native as u64,
            a[1] as u64, a[2] as u64, a[3] as u64, a[4] as u64, a[5] as u64, a[6] as u64,
            a[7] as u64, stream as u64, params.as_mut_ptr() as usize as u64, 0])
    }
    pub(super) unsafe fn dispatch_module(&mut self, op: u32, a: &[u32; 16]) -> Answer {
        match op {
            C32_MODULE_LOAD => self.load_module(a),
            C32_MODULE_UNLOAD => {
                let slot = self.handle(a[0], 2).ok_or(INVALID_HANDLE)?;
                native("cuCtxSynchronize", &[])?;
                native("cuModuleUnload", &[self.handles[slot].native as u64])?;
                let owner = self.handles[slot].id;
                for h in &mut self.handles {
                    if (h.kind == 3 || h.kind == 5) && h.owner == owner { *h = Handle::default(); }
                }
                self.release_globals(slot);
                self.handles[slot] = Handle::default();
                Ok(())
            }
            C32_MODULE_GLOBAL => {
                let m = self.handle(a[2], 2).ok_or(INVALID_HANDLE)?;
                if a[3] == 0 || (a[0] == 0 && a[1] == 0) || a[4] != 4 { return Err(INVALID); }
                let name = CStr::from_ptr(ptr(a[3]));
                let global = self.handles[m].globals.iter().find(|g| g.name.as_c_str() == name).ok_or(500)?;
                if a[0] != 0 { write32(a[0], global.offset); }
                if a[1] != 0 { write32(a[1], global.bytes); }
                Ok(())
            }
            C32_TEX_GET => {
                if a[0] == 0 || a[2] == 0 { return Err(INVALID); }
                let m = self.handle(a[1], 2).ok_or(INVALID_HANDLE)?;
                let owner = self.handles[m].id;
                let name = CStr::from_ptr(ptr(a[2]));
                if let Some(h) = self.handles.iter().find(|h| h.kind == 5 && h.owner == owner &&
                    h.name.as_deref() == Some(name)) { write32(a[0], h.id); return Ok(()); }
                let name = name.to_owned();
                let mut value = 0usize;
                native("cuModuleGetTexRef", &[mutable(&mut value), self.handles[m].native as u64,
                    name.as_ptr() as usize as u64])?;
                let h = self.new_handle(value, 5)?;
                self.handles[h].owner = owner;
                self.handles[h].name = Some(name);
                write32(a[0], self.handles[h].id);
                Ok(())
            }
            C32_FUNCTION => {
                if a[0] == 0 || a[2] == 0 { return Err(INVALID); }
                let m = self.handle(a[1], 2).ok_or(INVALID_HANDLE)?;
                let mut value = 0usize;
                native("cuModuleGetFunction", &[mutable(&mut value), self.handles[m].native as u64, a[2] as u64])?;
                let function: ArgsFn = std::mem::transmute(self.dark.ok_or(UNSUPPORTED)?[1]);
                let mut count = 0u32;
                status(function(std::ptr::null_mut(), &mut count, value))?;
                if count as usize > MAX_ARGS { return Err(UNSUPPORTED); }
                let h = self.new_handle(value, 3)?;
                self.handles[h].owner = self.handles[m].id;
                self.handles[h].nargs = count;
                let result = status(function(self.handles[h].args.as_mut_ptr(), &mut count, value));
                // As in C, the first query fixes nargs. The fill call's count
                // is not substituted into the published handle's arity.
                if result.is_err() {
                    self.handles[h] = Handle::default();
                    return result;
                }
                write32(a[0], self.handles[h].id);
                Ok(())
            }
            C32_LAUNCH => self.launch(a),
            _ => Err(UNSUPPORTED),
        }
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Checked bounded memory, event, stream and texture operations from fa81f481.
use super::*;

impl State {
    pub(super) fn stream(&self, id: u32) -> Result<usize, i32> {
        if id == 0 { Ok(0) }
        else { self.handle(id, 6).map(|i| self.handles[i].native).ok_or(INVALID_HANDLE) }
    }
    unsafe fn host_memory(&self, op: u32, a: &[u32; 16]) -> Answer {
        // Both functions must come from the selected backend's dependency
        // closure. Never open a second HIP runtime with a separate dlopen.
        let register: unsafe extern "C" fn(*mut c_void, usize, u32) -> i32 =
            std::mem::transmute(crate::unix_wrappers::symbol(b"hipHostRegister\0").map_err(|_| UNSUPPORTED)?);
        let unregister: unsafe extern "C" fn(*mut c_void) -> i32 =
            std::mem::transmute(crate::unix_wrappers::symbol(b"hipHostUnregister\0").map_err(|_| UNSUPPORTED)?);
        if a[0] == 0 { return Err(INVALID); }
        let status = if op == C32_HOST_REGISTER {
            if a[1] == 0 { return Err(INVALID); }
            if a[2] & !1 != 0 { return Err(UNSUPPORTED); }
            register(ptr(a[0]), a[1] as usize, a[2])
        } else {
            let mut active = 0usize;
            native("cuCtxGetCurrent", &[mutable(&mut active)])?;
            if active != 0 { native("cuCtxSynchronize", &[])?; }
            unregister(ptr(a[0]))
        };
        match status {
            0 => Ok(()), 1 | 2 | 100 | 101 | 201 | 712 | 713 | 801 => Err(status), _ => Err(999),
        }
    }
    pub(super) unsafe fn dispatch_current(&mut self, op: u32, a: &[u32; 16]) -> Answer {
        let x = a.map(u64::from);
        match op {
            C32_CTX_DEVICE => {
                if a[0] == 0 { return Err(INVALID); }
                native("cuCtxGetDevice", &x[..1])
            }
            C32_CTX_ATTACH => {
                if a[0] == 0 || a[1] != 0 || self.context().refs == u32::MAX { return Err(INVALID); }
                let h = self.handles.iter().find(|h| h.kind == 1 && h.native == self.context().native)
                    .ok_or(INVALID_CONTEXT)?;
                let id = h.id;
                self.context_mut().refs += 1;
                write32(a[0], id);
                Ok(())
            }
            C32_CTX_DETACH => {
                if a[1] != 0 { write32(a[1], 0); }
                let h = self.handle(a[0], 1).ok_or(INVALID_CONTEXT)?;
                if self.handles[h].native != self.context().native || self.context().refs == 0 {
                    return Err(INVALID_CONTEXT);
                }
                if self.context().refs > 1 { self.context_mut().refs -= 1; return Ok(()); }
                self.destroy_context(h)?;
                if a[1] != 0 { write32(a[1], 1); }
                Ok(())
            }
            C32_CTX_SYNC => native("cuCtxSynchronize", &[]),
            C32_ALLOC => {
                if a[0] == 0 || a[2] != 4 { return Err(INVALID); }
                let offset = self.alloc_range(a[1], 0)?;
                write32(a[0], offset);
                Ok(())
            }
            C32_FREE => {
                let slot = self.context().ranges.iter().position(|r| r.bytes != 0 && r.offset == a[0])
                    .ok_or(INVALID)?;
                if self.context().ranges[slot].owner != 0 { return Err(INVALID); }
                native("cuCtxSynchronize", &[])?;
                self.context_mut().ranges[slot] = Allocation::default();
                Ok(())
            }
            C32_MEMSET8 | C32_MEMSET16 | C32_MEMSET_ASYNC => {
                let width = match op { C32_MEMSET8 => 1, C32_MEMSET16 => 2, _ => a[4] };
                if ![1, 2, 4].contains(&width) || a[2] > u32::MAX / width || a[0] & (width - 1) != 0 {
                    return Err(INVALID);
                }
                let stream = if op == C32_MEMSET_ASYNC { self.stream(a[3])? } else { 0 };
                if a[2] == 0 { return Ok(()); }
                let address = self.translate(a[0], a[2] * width)?;
                let name = match op {
                    C32_MEMSET8 => "cuMemsetD8_v2", C32_MEMSET16 => "cuMemsetD16_v2",
                    _ => match width { 1 => "cuMemsetD8Async", 2 => "cuMemsetD16Async", _ => "cuMemsetD32Async" },
                };
                native(name, &[address, x[1], x[2], stream as u64])
            }
            C32_HTOD | C32_DTOH | C32_MEMSET => {
                let offset = if op == C32_DTOH { a[1] } else { a[0] };
                let mut bytes = a[2];
                if op == C32_MEMSET {
                    if bytes > u32::MAX / 4 || offset & 3 != 0 { return Err(INVALID); }
                    bytes *= 4;
                }
                if bytes == 0 { return Ok(()); }
                let address = self.translate(offset, bytes)?;
                if op == C32_MEMSET { return native("cuMemsetD32_v2", &[address, x[1], x[2]]); }
                if op == C32_HTOD {
                    if a[1] == 0 { return Err(INVALID); }
                    native("cuMemcpyHtoD_v2", &[address, x[1], bytes as u64])
                } else {
                    if a[0] == 0 { return Err(INVALID); }
                    native("cuMemcpyDtoH_v2", &[x[0], address, bytes as u64])
                }
            }
            C32_HOST_REGISTER | C32_HOST_UNREGISTER => self.host_memory(op, a),
            C32_EVENT_CREATE | C32_STREAM_CREATE => {
                if a[0] == 0 { return Err(INVALID); }
                let mut value = 0usize;
                let event = op == C32_EVENT_CREATE;
                native(if event { "cuEventCreate" } else { "cuStreamCreate" }, &[mutable(&mut value), x[1]])?;
                let h = match self.new_handle(value, if event { 7 } else { 6 }) {
                    Ok(h) => h, Err(error) => {
                        let _ = native(if event { "cuEventDestroy_v2" } else { "cuStreamDestroy_v2" }, &[value as u64]);
                        return Err(error);
                    }
                };
                write32(a[0], self.handles[h].id);
                Ok(())
            }
            C32_EVENT_DESTROY | C32_EVENT_SYNC | C32_EVENT_QUERY => {
                let h = self.handle(a[0], 7).ok_or(INVALID_HANDLE)?;
                let name = match op { C32_EVENT_SYNC => "cuEventSynchronize", C32_EVENT_QUERY => "cuEventQuery", _ => "cuEventDestroy_v2" };
                native(name, &[self.handles[h].native as u64])?;
                if op == C32_EVENT_DESTROY { self.handles[h] = Handle::default(); }
                Ok(())
            }
            C32_STREAM_DESTROY | C32_STREAM_SYNC | C32_STREAM_QUERY => {
                let value = self.stream(a[0])?;
                let name = match op { C32_STREAM_SYNC => "cuStreamSynchronize", C32_STREAM_QUERY => "cuStreamQuery", _ => "cuStreamDestroy_v2" };
                if op == C32_STREAM_DESTROY && a[0] == 0 { return Err(INVALID_HANDLE); }
                native(name, &[value as u64])?;
                if op == C32_STREAM_DESTROY {
                    let h = self.handle(a[0], 6).ok_or(INVALID_HANDLE)?;
                    self.handles[h] = Handle::default();
                }
                Ok(())
            }
            C32_EVENT_RECORD | C32_STREAM_WAIT => {
                let event = self.handle(a[0], 7).or_else(|| {
                    if op == C32_STREAM_WAIT { self.handles.iter().position(|h| h.id == a[0] && h.kind == 7) }
                    else { None }
                }).ok_or(INVALID_HANDLE)?;
                let stream = self.stream(a[1])? as u64;
                let value = self.handles[event].native as u64;
                if op == C32_EVENT_RECORD { native("cuEventRecord", &[value, stream]) }
                else { native("cuStreamWaitEvent", &[stream, value, x[2]]) }
            }
            C32_EVENT_ELAPSED => {
                let start = self.handle(a[1], 7).ok_or(INVALID_HANDLE)?;
                let end = self.handle(a[2], 7).ok_or(INVALID_HANDLE)?;
                if a[0] == 0 { return Err(INVALID); }
                native("cuEventElapsedTime", &[x[0], self.handles[start].native as u64, self.handles[end].native as u64])
            }
            C32_ADDRESS_RANGE => {
                if a[0] == 0 && a[1] == 0 { return Err(INVALID); }
                let range = self.context().ranges.iter().find(|r| r.bytes != 0 && a[2] >= r.offset && a[2] - r.offset < r.bytes)
                    .ok_or(INVALID)?;
                if a[0] != 0 { write32(a[0], range.offset); }
                if a[1] != 0 { write32(a[1], range.bytes); }
                Ok(())
            }
            C32_DTOD | C32_DTOD_ASYNC => {
                let stream = if op == C32_DTOD_ASYNC { self.stream(a[3])? as u64 } else { 0 };
                if a[2] == 0 { return Ok(()); }
                let dst = self.translate(a[0], a[2])?;
                let src = self.translate(a[1], a[2])?;
                native(if op == C32_DTOD { "cuMemcpyDtoD_v2" } else { "cuMemcpyDtoDAsync_v2" }, &[dst, src, x[2], stream])
            }
            C32_HTOD_ASYNC | C32_DTOH_ASYNC => {
                let stream = self.stream(a[3])? as u64;
                if a[2] == 0 { return Ok(()); }
                let (offset, host) = if op == C32_HTOD_ASYNC { (a[0], a[1]) } else { (a[1], a[0]) };
                if host == 0 { return Err(INVALID); }
                let address = self.translate(offset, a[2])?;
                if op == C32_HTOD_ASYNC { native("cuMemcpyHtoDAsync_v2", &[address, host as u64, x[2], stream]) }
                else { native("cuMemcpyDtoHAsync_v2", &[host as u64, address, x[2], stream]) }
            }
            C32_TEX_FORMAT | C32_TEX_FLAGS | C32_TEX_ADDRESS | C32_TEX_FILTER | C32_TEX_ADDRESS_MODE | C32_TEX_MIPMAP => {
                let h = self.handle(a[0], 5).ok_or(INVALID_HANDLE)?;
                let value = self.handles[h].native as u64;
                match op {
                    C32_TEX_FILTER => {
                        if a[1] > 1 { return Err(INVALID); }
                        native("cuTexRefSetFilterMode", &[value, x[1]])
                    }
                    C32_TEX_ADDRESS_MODE => {
                        if a[1] > 2 || a[2] > 3 { return Err(INVALID); }
                        native("cuTexRefSetAddressMode", &[value, x[1], x[2]])
                    }
                    C32_TEX_FORMAT => native("cuTexRefSetFormat", &[value, x[1], x[2]]),
                    C32_TEX_FLAGS => native("cuTexRefSetFlags", &[value, x[1]]),
                    C32_TEX_MIPMAP => match a[1] {
                        0 if a[2] <= 1 => native("cuTexRefSetMipmapFilterMode", &[value, x[2]]),
                        1 => native("cuTexRefSetMipmapLevelBias", &[value, x[2]]),
                        2 => native("cuTexRefSetMipmapLevelClamp", &[value, x[2], x[3]]),
                        3 => native("cuTexRefSetMaxAnisotropy", &[value, x[2]]),
                        _ => Err(INVALID),
                    },
                    _ => {
                        let address = if a[1] == 0 { 0 } else { self.translate(a[1], a[2].max(1))? };
                        let mut offset = 0usize;
                        native("cuTexRefSetAddress_v2", &[mutable(&mut offset), value, address, x[2]])?;
                        if offset > u32::MAX as usize { return Err(UNSUPPORTED); }
                        if a[3] != 0 { write32(a[3], offset as u32); }
                        Ok(())
                    }
                }
            }
            _ => self.dispatch_module(op, a),
        }
    }
}

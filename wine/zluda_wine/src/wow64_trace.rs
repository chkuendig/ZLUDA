// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure PE32 helpers shared by the frontend and native tests: host allocation
//! flags, cache/shared-memory hints and the optional per-call trace line.
#![allow(dead_code)]
use crate::wow64_protocol::INVALID;

pub const HOST_PORTABLE: u32 = 1;
pub const HOST_DEVICEMAP: u32 = 2;
pub const HOST_WRITECOMBINED: u32 = 4;

/// cuMemHostAlloc flags to hipHostRegister flags. PORTABLE and DEVICEMAP
/// match hipHostRegisterPortable/Mapped. WRITECOMBINED is a caching hint;
/// ordinary cacheable memory is a valid implementation, so it is dropped.
pub fn host_register_flags(flags: u32) -> Result<u32, i32> {
    if flags & !(HOST_PORTABLE | HOST_DEVICEMAP | HOST_WRITECOMBINED) != 0 { return Err(INVALID); }
    Ok(flags & (HOST_PORTABLE | HOST_DEVICEMAP))
}

/// CU_FUNC_CACHE_PREFER_NONE..PREFER_EQUAL and CU_SHARED_MEM_CONFIG_*.
/// Both are scheduling hints that HIP on AMD does not act on.
pub fn cache_config(config: i32) -> Result<(), i32> {
    if (0..=3).contains(&config) { Ok(()) } else { Err(INVALID) }
}
pub fn shared_config(config: i32) -> Result<(), i32> {
    if (0..=2).contains(&config) { Ok(()) } else { Err(INVALID) }
}

/// One trace record: `<us> <tid> <name> r=<status> dt=<us>[ a=<hex>,...]\n`.
/// Writes at most `out.len()` bytes; an overlong record is cut and still
/// ends in a newline, so concurrent appends stay line-separated.
pub fn trace_line(out: &mut [u8], time_us: u64, tid: u32, name: &str, status: i32, duration_us: u64, args: &[u32]) -> usize {
    use std::fmt::Write;
    struct Cursor<'a> { out: &'a mut [u8], len: usize }
    impl Write for Cursor<'_> {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            let room = self.out.len().saturating_sub(self.len + 1);
            let take = text.len().min(room);
            self.out[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
            self.len += take;
            if take == text.len() { Ok(()) } else { Err(std::fmt::Error) }
        }
    }
    if out.is_empty() { return 0; }
    let mut cursor = Cursor { out, len: 0 };
    let _ = (|| -> std::fmt::Result {
        write!(cursor, "{time_us} {tid:x} {name} r={status} dt={duration_us}")?;
        for (i, arg) in args.iter().enumerate() {
            write!(cursor, "{}{arg:x}", if i == 0 { " a=" } else { "," })?;
        }
        Ok(())
    })();
    let len = cursor.len;
    cursor.out[len] = b'\n';
    len + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_flags_accept_cuda_set_and_map_to_hip() {
        assert_eq!(host_register_flags(0), Ok(0));
        assert_eq!(host_register_flags(HOST_PORTABLE), Ok(1));
        assert_eq!(host_register_flags(HOST_DEVICEMAP), Ok(2));
        assert_eq!(host_register_flags(HOST_WRITECOMBINED), Ok(0));
        assert_eq!(host_register_flags(7), Ok(3));
        assert_eq!(host_register_flags(8), Err(INVALID));
        assert_eq!(host_register_flags(0x8000_0001), Err(INVALID));
    }

    #[test]
    fn config_hints_validate_cuda_ranges() {
        for value in 0..=3 { assert_eq!(cache_config(value), Ok(())); }
        for value in 0..=2 { assert_eq!(shared_config(value), Ok(())); }
        assert_eq!(cache_config(4), Err(INVALID));
        assert_eq!(cache_config(-1), Err(INVALID));
        assert_eq!(shared_config(3), Err(INVALID));
    }

    #[test]
    fn trace_line_is_exact_and_newline_terminated() {
        let mut buffer = [0u8; 256];
        let len = trace_line(&mut buffer, 12, 0x1bc, "cuMemHostAlloc", 0, 7, &[0x4000_0010, 0x40, 2]);
        assert_eq!(&buffer[..len], b"12 1bc cuMemHostAlloc r=0 dt=7 a=40000010,40,2\n");
        let len = trace_line(&mut buffer, 1, 2, "cuCtxSetCacheConfig", 801, 0, &[]);
        assert_eq!(&buffer[..len], b"1 2 cuCtxSetCacheConfig r=801 dt=0\n");
    }

    #[test]
    fn trace_line_truncates_inside_the_buffer() {
        let mut buffer = [0xffu8; 16];
        let len = trace_line(&mut buffer, 123456, 0xabc, "cuModuleLoadData", -1, 99, &[1]);
        assert_eq!(len, 16);
        assert_eq!(&buffer[..len], b"123456 abc cuMo\n");
        let mut tiny = [0u8; 1];
        assert_eq!(trace_line(&mut tiny, 0, 0, "x", 0, 0, &[]), 1);
        assert_eq!(tiny, *b"\n");
        assert_eq!(trace_line(&mut [], 0, 0, "x", 0, 0, &[]), 0);
    }
}

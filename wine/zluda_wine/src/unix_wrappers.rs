// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hand-written Unix library lifetime, backend lookup and protocol validation.
use crate::{cuda::*, unixlib::{self, Call, UnixEntry}};
use std::{ffi::{c_char, c_void, CStr, CString, OsStr}, io::Read, path::PathBuf, sync::OnceLock};
#[link(name = "dl")]
extern "C" {
    fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn getenv(name:*const c_char)->*const c_char;
}
// Retain the library for the process lifetime: cached addresses and handles
// cannot outlive it. No global lock is held while a CUDA function runs.
static BACKEND: OnceLock<Result<usize, CUerror>> = OnceLock::new();
static ENABLED:OnceLock<bool>=OnceLock::new();
include!(concat!(env!("OUT_DIR"),"/native_symbols.rs"));
#[cfg(test)]
thread_local! {
    static TEST_LOADER:std::cell::Cell<Option<unsafe fn(*const c_char)->*mut c_void>>=const {std::cell::Cell::new(None)};
    static LOADER_COUNTS:std::cell::Cell<[u32;3]>=const {std::cell::Cell::new([0;3])};
}
#[cfg(test)] pub fn loader_counts()->[u32;3] {LOADER_COUNTS.with(|v|v.get())}
#[cfg(test)] fn loader_count(index:usize) {LOADER_COUNTS.with(|v|{let mut c=v.get();c[index]+=1;v.set(c);});}
#[cfg(test)] pub fn with_fixture_loader<T>(resolve:unsafe fn(*const c_char)->*mut c_void,action:impl FnOnce()->T)->T {
    assert!(BACKEND.get().is_none() && ENABLED.get().is_none(),"production loader fixture requires a fresh child process");
    TEST_LOADER.with(|slot| {
        assert!(slot.replace(Some(resolve)).is_none());
        struct Restore<'a>(&'a std::cell::Cell<Option<unsafe fn(*const c_char)->*mut c_void>>);
        impl Drop for Restore<'_> {fn drop(&mut self) {self.0.set(None);}}
        let _restore=Restore(slot);action()
    })
}
unsafe fn loader_getenv(name:*const c_char)->*const c_char {
    #[cfg(test)] if TEST_LOADER.with(|slot|slot.get().is_some()) {loader_count(0);}
    getenv(name)
}
unsafe fn loader_open(name:*const c_char,flags:i32)->*mut c_void {
    #[cfg(test)] if TEST_LOADER.with(|slot|slot.get().is_some()) {loader_count(2);return 1usize as *mut c_void;}
    dlopen(name,flags)
}
unsafe fn loader_symbol(handle:*mut c_void,name:*const c_char)->*mut c_void {
    #[cfg(test)] if let Some(resolve)=TEST_LOADER.with(|slot|slot.get()) {assert_eq!(handle as usize,1);loader_count(1);return resolve(name);}
    dlsym(handle,name)
}
unsafe fn environment_is_one(name:&CStr)->bool {
    let value=loader_getenv(name.as_ptr());!value.is_null() && CStr::from_ptr(value).to_bytes()==b"1"
}
pub fn enabled()->bool {
    // Root choice is immutable per process. Off never loads this module, and
    // direct unexpected Unix calls cannot enable it after an initial refusal.
    // Neither an env lookup nor an allocation belongs on a CUDA frame path.
    *ENABLED.get_or_init(||unsafe {environment_is_one(CStr::from_bytes_with_nul_unchecked(b"PROTON_ENABLE_ZLUDA\0"))})
}

fn validate_backend_path(value: Option<&OsStr>) -> Result<PathBuf, CUerror> {
    let path = std::path::Path::new(value.ok_or(CUerror::NOT_INITIALIZED)?);
    if !path.is_absolute() { return Err(CUerror::NOT_INITIALIZED); }
    let path = path.canonicalize().map_err(|_| CUerror::NOT_INITIALIZED)?;
    let mut file = std::fs::File::open(&path).map_err(|_| CUerror::NOT_INITIALIZED)?;
    if !file.metadata().map_err(|_| CUerror::NOT_INITIALIZED)?.is_file() {
        return Err(CUerror::NOT_INITIALIZED);
    }
    let mut header = [0u8; 20];
    file.read_exact(&mut header).map_err(|_| CUerror::NOT_INITIALIZED)?;
    // Linux ELF64, little endian, shared library, native x86_64 machine.
    if &header[..6] != b"\x7fELF\x02\x01" || header[16..20] != [3, 0, 62, 0] {
        return Err(CUerror::NOT_INITIALIZED);
    }
    Ok(path)
}

pub unsafe fn symbol(name: &[u8]) -> Result<*mut c_void, CUerror> {
    // Root selection is the normal off-mode policy. Also reject unexpected
    // direct Unix calls before caching a loader result or touching CUDA/HIP.
    if !enabled() {
        return Err(CUerror::NOT_INITIALIZED);
    }
    let handle = BACKEND.get_or_init(|| {
        use std::os::unix::ffi::OsStrExt;
        let value = std::env::var_os("ZLUDA_WINE_BACKEND_PATH");
        let path = validate_backend_path(value.as_deref())?;
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| CUerror::NOT_INITIALIZED)?;
        let handle = loader_open(path.as_ptr(), 2); // RTLD_NOW | RTLD_LOCAL
        if handle.is_null() { Err(CUerror::NOT_INITIALIZED) } else { Ok(handle as usize) }
    });
    let handle = match handle { Ok(handle) => *handle, Err(error) => return Err(*error) };
    let name = CStr::from_bytes_with_nul(name).map_err(|_| CUerror::INVALID_VALUE)?;
    let resolve=|| {
        let address=loader_symbol(handle as *mut c_void,name.as_ptr());
        if address.is_null() {Err(CUerror::NOT_SUPPORTED)}else{Ok(address as usize)}
    };
    let result=match NATIVE_SYMBOLS.binary_search_by(|(candidate,_)|candidate.cmp(&name.to_bytes_with_nul())) {
        Ok(index)=>*NATIVE_SYMBOLS[index].1.get_or_init(resolve),
        // Unknown HAS_SYMBOL probes are not retained: no unbounded cache of
        // caller-controlled names. Never return this raw address to Windows.
        Err(_)=>resolve(),
    };
    result.map(|address|address as *mut c_void)
}

thread_local! {static CALL_DEPTH:std::cell::Cell<u32>=const {std::cell::Cell::new(0)};}
pub fn inside_call()->bool {CALL_DEPTH.with(|v|v.get()!=0)}
#[cfg(test)] thread_local! {static TRANSITIONS:std::cell::Cell<u32>=const {std::cell::Cell::new(0)};}
#[cfg(test)] pub fn transition_count()->u32 {TRANSITIONS.with(|v|v.get())}
struct CallScope;
impl CallScope {fn enter()->Self {CALL_DEPTH.with(|v|v.set(v.get()+1));Self}}
impl Drop for CallScope {fn drop(&mut self) {CALL_DEPTH.with(|v|v.set(v.get()-1));}}
pub(crate) unsafe extern "C" fn dispatch(pointer: *mut c_void) -> i32 {
    #[cfg(test)] TRANSITIONS.with(|v|v.set(v.get()+1));
    let _scope=CallScope::enter();
    if pointer.is_null() { return 0xc000000du32 as i32; } // STATUS_INVALID_PARAMETER
    // Check the fixed header before constructing a reference to the full
    // packet. A valid caller with an older, shorter packet is rejected first.
    if (pointer as usize) % std::mem::align_of::<Call>() != 0 {
        return 0xc000000du32 as i32;
    }
    let bytes = pointer.cast::<u8>();
    if std::ptr::read(bytes.cast::<u64>()) != unixlib::MAGIC
        || std::ptr::read(bytes.add(8).cast::<u32>()) != unixlib::VERSION
        || std::ptr::read(bytes.add(12).cast::<u32>()) as usize != std::mem::size_of::<Call>() {
        return 0xc000000du32 as i32;
    }
    let call = &mut *pointer.cast::<Call>();
    if !call.valid() { return 0xc000000du32 as i32; }
    let result = if call.function == 0 {
        call.result = 0; // Handshake never loads CUDA/HIP.
        Ok(())
    } else if call.function == unixlib::HAS_SYMBOL {
        let name = unixlib::unpack::<*const c_char>(call.args[0]);
        if name.is_null() { Err(CUerror::INVALID_VALUE) }
        else { symbol(CStr::from_ptr(name).to_bytes_with_nul()).map(|_| { call.result = 0; }) }
    } else if (crate::callback_protocol::ALLOC..=crate::callback_protocol::REENTER).contains(&call.function) {
        if !enabled() { Err(CUerror::NOT_INITIALIZED) }
        else { crate::unix_callbacks::service(call) }
    } else if call.function==crate::context_protocol::INITIALIZE {
        crate::unix_context_storage::initialize(call)
    } else if call.function == crate::interop_protocol::ENVIRONMENT {
        if !enabled() {Err(CUerror::NOT_INITIALIZED)}
        else {
            call.args[0]=u64::from(std::env::var_os("ZLUDA_D3D11_CPU_COPY").is_some_and(|v|v=="1"))*crate::interop_protocol::COPY
                |u64::from(std::env::var_os("ZLUDA_D3D11_CPU_COPY_ENABLE_MT").is_some_and(|v|v=="1"))*crate::interop_protocol::ENABLE_MT;
            call.result=0;Ok(())
        }
    } else if call.function == crate::luid_protocol::UUID {
        use std::os::unix::ffi::OsStrExt;
        (|| {
            let function: unsafe extern "C" fn(*mut CUuuid,CUdevice) -> CUresult =
                std::mem::transmute(symbol(b"cuDeviceGetUuid\0")?);
            let result = function(call.get(0)?,call.get(1)?);
            call.result = unixlib::code(result);
            if result.is_ok() {
                call.args[2] = u64::from(std::env::var_os("CUDA_FAKE_LUID")
                    .is_some_and(|value| value.as_bytes().first()==Some(&b'1')));
            }
            Ok(())
        })()
    } else if call.function == crate::dark_protocol::GET {
        crate::dark_unix::get(call)
    } else if call.function == crate::dark_protocol::CALL {
        match crate::unix_context_storage::dispatch(call) { Some(result)=>result,None=>crate::dark_unix::typed_call(call) }
    } else if call.function == crate::dark_protocol::LEGACY_QUERY {
        crate::dark_unix::legacy_query(call)
    } else if let Some(result) = crate::unix_callback_api::dispatch(call) {
        result
    } else {
        crate::unix_thunks::dispatch(call)
    };
    if let Err(error) = result { call.result = error.0.get(); }
    0
}

#[no_mangle]
pub static __wine_unix_call_funcs: [UnixEntry; 2] = [dispatch,transition_noop];
// Private diagnostic index1 measures the Wine transition alone. No CUDA,
// protocol parsing, loader, TLS scope, allocation or public PE export.
unsafe extern "C" fn transition_noop(_: *mut c_void)->i32 {0}
// The separate fixed-width dispatcher is exported from wow64_unix.rs.

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_location_has_no_implicit_or_relative_fallback() {
        assert_eq!(validate_backend_path(None), Err(CUerror::NOT_INITIALIZED));
        assert_eq!(validate_backend_path(Some(OsStr::new(""))), Err(CUerror::NOT_INITIALIZED));
        assert_eq!(validate_backend_path(Some(OsStr::new("libcuda.so.1"))), Err(CUerror::NOT_INITIALIZED));
        assert_eq!(validate_backend_path(Some(OsStr::new("../libcuda.so.1"))), Err(CUerror::NOT_INITIALIZED));
        assert_eq!(validate_backend_path(Some(OsStr::new("/a/nonexistent/libcuda.so.1"))), Err(CUerror::NOT_INITIALIZED));
    }
    #[test]
    fn selected_backend_must_be_a_native_elf_shared_library() {
        let file = std::env::temp_dir().join(format!("zluda-wine-backend-header-{}", std::process::id()));
        std::fs::write(&file, b"not an ELF shared object").unwrap();
        assert_eq!(validate_backend_path(Some(file.as_os_str())), Err(CUerror::NOT_INITIALIZED));
        let mut header = [0u8; 20];
        header[..6].copy_from_slice(b"\x7fELF\x02\x01");
        header[16..20].copy_from_slice(&[3, 0, 62, 0]);
        std::fs::write(&file, header).unwrap();
        assert_eq!(validate_backend_path(Some(file.as_os_str())), Ok(file.canonicalize().unwrap()));
        header[18] = 3; // i386 is not a native Unix backend in this tool.
        std::fs::write(&file, header).unwrap();
        assert_eq!(validate_backend_path(Some(file.as_os_str())), Err(CUerror::NOT_INITIALIZED));
        std::fs::remove_file(file).unwrap();
    }
}

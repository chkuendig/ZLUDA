// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generated native CUDA calls. Linux symbols are always called as SysV C.
use crate::unixlib::{self, Call};
use std::ffi::c_void;
#[cfg(test)]
thread_local! { static TEST_SYMBOL: std::cell::Cell<Option<unsafe fn(u64)->usize>> = const {std::cell::Cell::new(None)}; }
#[cfg(test)]
pub fn with_symbols<T>(resolve: unsafe fn(u64)->usize, action: impl FnOnce()->T)->T {
    TEST_SYMBOL.with(|slot| {
        assert!(slot.replace(Some(resolve)).is_none());
        struct Restore<'a>(&'a std::cell::Cell<Option<unsafe fn(u64)->usize>>);
        impl Drop for Restore<'_> { fn drop(&mut self) {self.0.set(None);} }
        let _restore=Restore(slot);action()
    })
}
unsafe fn symbol(name: &[u8])->Result<*mut c_void,crate::cuda::CUerror> {
    #[cfg(test)]
    if let Some(resolve)=TEST_SYMBOL.with(|slot| slot.get()) {
        let address=resolve(unixlib::function_id(std::str::from_utf8(&name[..name.len()-1]).unwrap()));
        return if address==0 {Err(crate::cuda::CUerror::NOT_SUPPORTED)} else {Ok(address as *mut c_void)};
    }
    crate::unix_wrappers::symbol(name)
}
type SymbolCache=std::sync::OnceLock<Result<usize,crate::cuda::CUerror>>;
unsafe fn cached_symbol(cache:&SymbolCache,name:&[u8])->Result<*mut c_void,crate::cuda::CUerror> {
    #[cfg(test)]
    if TEST_SYMBOL.with(|slot|slot.get().is_some()) {return symbol(name);}
    // A cached symbol must not bypass unexpected off-mode Unix calls.
    if !crate::unix_wrappers::enabled() {return Err(crate::cuda::CUerror::NOT_INITIALIZED);}
    cache.get_or_init(||symbol(name).map(|value|value as usize)).map(|value|value as *mut c_void)
}

macro_rules! forwarded {
    ($($abi:literal fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        unsafe fn forward(call: &mut Call) -> Result<(), crate::cuda::CUerror> {
            $(if call.function == unixlib::function_id(stringify!($name)) {
                static SYMBOL:SymbolCache=SymbolCache::new();
                let address = cached_symbol(&SYMBOL,concat!(stringify!($name), "\0").as_bytes())?;
                let native: unsafe extern "C" fn($($ty),*) -> $ret = std::mem::transmute(address);
                #[allow(unused_mut)]
                let mut index = 0;
                $(let $arg: $ty = call.get(index)?; index += 1;)*
                let _ = index;
                call.result = unixlib::code(native($($arg),*));
                return Ok(());
            })*
            Err(crate::cuda::CUerror::NOT_SUPPORTED)
        }
    };
}
macro_rules! backend_status {
    ($($abi:literal fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        pub unsafe fn dispatch(call: &mut Call) -> Result<(), crate::cuda::CUerror> {
            $(if call.function==unixlib::function_id(stringify!($name)) {
                if call.args != [0;16] || call.aggregate != [0;64] {return Err(crate::cuda::CUerror::INVALID_VALUE);}
                static SYMBOL:SymbolCache=SymbolCache::new();
                let address=cached_symbol(&SYMBOL,concat!(stringify!($name),"\0").as_bytes())?;
                let native: unsafe extern "C" fn($($ty),*) -> $ret=std::mem::transmute(address);
                call.result=unixlib::code(native($(std::mem::zeroed::<$ty>()),*));
                return Ok(());
            })*
            forward(call)
        }
    };
}
include!(concat!(env!("OUT_DIR"), "/unix_declarations.rs"));

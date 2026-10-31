// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generated PE exports; Windows ABI is explicit and never inferred on Unix.
#![cfg_attr(not(windows), allow(dead_code))]
use crate::unixlib::{self, Call};
use std::ffi::c_void;

#[allow(unused_macros)]
macro_rules! pending {
    ($($abi:literal fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => { $(
        #[cfg_attr(windows, no_mangle)]
        pub unsafe extern "system" fn $name($($arg: $ty),*) -> $ret {
            $(let _ = $arg;)*
            Err(crate::cuda::CUerror::NOT_SUPPORTED)
        }
    )* };
}
macro_rules! pe_export {
    ("system", $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty, $body:block) => {
        #[cfg_attr(windows, no_mangle)] pub unsafe extern "system" fn $name($($arg: $ty),*) -> $ret $body
    };
    ("private", $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty, $body:block) => {
        pub unsafe extern "system" fn $name($($arg: $ty),*) -> $ret $body
    };
}
macro_rules! forwarded {
    ($($abi:tt fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        $(pe_export! { $abi, $name($($arg: $ty),*) -> $ret, {
            let mut call = Call::new(unixlib::function_id(stringify!($name)));
            #[allow(unused_mut)]
            let mut index = 0;
            $(call.put(index, $arg); index += 1;)*
            let _ = index;
            crate::pe_wrappers::submit(&mut call)
        } })*
    };
}
macro_rules! manual {
    ($($abi:literal fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => { $(
        const _: unsafe extern "system" fn($($ty),*) -> $ret = crate::pe_wrappers::$name;
    )* };
}
macro_rules! backend_status {
    ($($abi:tt fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => { $(
        pe_export! { $abi, $name($($arg: $ty),*) -> $ret, {
            $(let _ = $arg;)*
            // build.rs attests that the packaged release backend ignores all
            // inputs. No Windows callback/union pointer, token or worker is
            // installed for an API it cannot implement.
            let mut call = Call::new(unixlib::function_id(stringify!($name)));
            crate::pe_wrappers::submit(&mut call)
        } }
    )* };
}
macro_rules! adapted { ($($tokens:tt)*) => { manual! { $($tokens)* } }; }
include!(concat!(env!("OUT_DIR"), "/pe_declarations.rs"));
include!(concat!(env!("OUT_DIR"), "/pe_pointers.rs"));
include!(concat!(env!("OUT_DIR"), "/pe_stubs.rs"));

use cuda_types::cuda::*;
use zluda_common::from_cuda_object;

pub(super) mod array;
pub(super) mod context;
pub(super) mod device;
pub(super) mod driver;
pub(super) mod event;
pub(super) mod function;
pub(super) mod graph;
mod hardware;
pub(super) mod hipfix;
mod iofx_bbox;
pub(super) mod kernel;
pub(super) mod library;
pub(super) mod memory;
pub(super) mod module;
pub(super) mod pointer;
mod signal;
pub(super) mod stream;
pub(super) mod surf;
pub(super) mod tex;

#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> CUerror {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> CUerror {
    CUerror::NOT_SUPPORTED
}

from_cuda_object!(module::Module, context::Context, library::Library);

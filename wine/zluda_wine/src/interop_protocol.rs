// SPDX-License-Identifier: MIT OR Apache-2.0
//! Private native environment query; D3D11 COM remains entirely in the PE.
//! No inputs; args[0] returns COPY=1 and ENABLE_MT=2, both exact getenv "1".
//! This request itself never resolves/loads CUDA. Include in the schema hash.
pub const ENVIRONMENT:u64=11;
pub const COPY:u64=1;
pub const ENABLE_MT:u64=2;

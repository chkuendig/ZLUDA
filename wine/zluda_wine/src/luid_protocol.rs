// SPDX-License-Identifier: MIT OR Apache-2.0
//! Private x64 LUID input request. Keep Windows registry access in the PE.
//! args[0]: writable CUuuid pointer; args[1]: CUdevice; args[2]: returned
//! native CUDA_FAKE_LUID first-byte-'1' flag. Native CUDA status is unchanged.
pub const UUID: u64 = 10;

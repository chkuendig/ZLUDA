// SPDX-License-Identifier: MIT OR Apache-2.0
//! Opaque compatibility identities; facts: specification §16.
use std::ffi::c_void;
pub enum CUcoredumpCallbackIdentity {}
pub type CUcoredumpCallbackHandle = *mut CUcoredumpCallbackIdentity;
pub type CUcoredumpStatusCallback = Option<unsafe extern "system" fn(*mut c_void, i32, crate::cuda::CUdevice)>;
#[allow(non_camel_case_types)]
pub enum CU_DEV_SM_RESOURCE_GROUP_PARAMS {}
pub enum CUstreamCigCaptureParams {}

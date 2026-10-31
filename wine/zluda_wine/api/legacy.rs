// SPDX-License-Identifier: MIT OR Apache-2.0
// Fresh declarations from CLEAN-ROOM-SPEC revision 2 sections 10–16 and ABI-FACTS.
use crate::{cuda::*, legacy::*};
extern "system" {
    fn cuCoredumpDeregisterCompleteCallback(arg0: CUcoredumpCallbackHandle) -> CUresult;
    fn cuCoredumpDeregisterStartCallback(arg0: CUcoredumpCallbackHandle) -> CUresult;
    fn cuCoredumpRegisterCompleteCallback(arg0: CUcoredumpStatusCallback, arg1: *mut ::core::ffi::c_void, arg2: *mut CUcoredumpCallbackHandle) -> CUresult;
    fn cuCoredumpRegisterStartCallback(arg0: CUcoredumpStatusCallback, arg1: *mut ::core::ffi::c_void, arg2: *mut CUcoredumpCallbackHandle) -> CUresult;
    fn cuDevSmResourceSplit(arg0: *mut CUdevResource, arg1: u32, arg2: *const CUdevResource, arg3: *mut CUdevResource, arg4: u32, arg5: *mut CU_DEV_SM_RESOURCE_GROUP_PARAMS) -> CUresult;
    fn cuDriverGetGpuCodeIsaVersion(arg0: *mut ::core::ffi::c_void, arg1: *mut ::core::ffi::c_void) -> CUresult;
    fn cuFuncGetParamCount(arg0: CUfunction, arg1: *mut usize) -> CUresult;
    fn cuGraphExecGetId(arg0: CUgraphExec, arg1: *mut u32) -> CUresult;
    fn cuGraphGetId(arg0: CUgraph, arg1: *mut u32) -> CUresult;
    fn cuGraphNodeGetContainingGraph(arg0: CUgraphNode, arg1: *mut CUgraph) -> CUresult;
    fn cuGraphNodeGetLocalId(arg0: CUgraphNode, arg1: *mut u32) -> CUresult;
    fn cuGraphNodeGetParams(arg0: CUgraphNode, arg1: *mut CUgraphNodeParams) -> CUresult;
    fn cuGraphNodeGetToolsId(arg0: CUgraphNode, arg1: *mut u64) -> CUresult;
    fn cuKernelGetParamCount(arg0: CUkernel, arg1: *mut usize) -> CUresult;
    fn cuLaunchHostFunc_v2(arg0: CUstream, arg1: CUhostFn, arg2: *mut ::core::ffi::c_void, arg3: u32) -> CUresult;
    fn cuLaunchHostFunc_v2_ptsz(arg0: CUstream, arg1: CUhostFn, arg2: *mut ::core::ffi::c_void, arg3: u32) -> CUresult;
    fn cuMemcpy3DWithAttributesAsync(arg0: *mut CUDA_MEMCPY3D_BATCH_OP_v1, arg1: u64, arg2: CUstream) -> CUresult;
    fn cuMemcpy3DWithAttributesAsync_ptsz(arg0: *mut CUDA_MEMCPY3D_BATCH_OP_v1, arg1: u64, arg2: CUstream) -> CUresult;
    fn cuMemcpyWithAttributesAsync(arg0: CUdeviceptr, arg1: CUdeviceptr, arg2: usize, arg3: *mut CUmemcpyAttributes_v1, arg4: CUstream) -> CUresult;
    fn cuMemcpyWithAttributesAsync_ptsz(arg0: CUdeviceptr, arg1: CUdeviceptr, arg2: usize, arg3: *mut CUmemcpyAttributes_v1, arg4: CUstream) -> CUresult;
    fn cuMulticastBindAddr_v2(arg0: CUmemGenericAllocationHandle, arg1: CUdevice, arg2: usize, arg3: CUdeviceptr, arg4: usize, arg5: u64) -> CUresult;
    fn cuMulticastBindMem_v2(arg0: CUmemGenericAllocationHandle, arg1: CUdevice, arg2: usize, arg3: CUmemGenericAllocationHandle, arg4: usize, arg5: usize, arg6: u64) -> CUresult;
    fn cuStreamBeginCaptureToCig(arg0: CUstream, arg1: *mut CUstreamCigCaptureParams) -> CUresult;
    fn cuStreamBeginCaptureToCig_ptsz(arg0: CUstream, arg1: *mut CUstreamCigCaptureParams) -> CUresult;
    fn cuStreamEndCaptureToCig(arg0: CUstream) -> CUresult;
    fn cuStreamEndCaptureToCig_ptsz(arg0: CUstream) -> CUresult;
    fn cuStreamGetDevResource(arg0: CUstream, arg1: *mut CUdevResource, arg2: CUdevResourceType) -> CUresult;
    fn cuStreamGetDevResource_ptsz(arg0: CUstream, arg1: *mut CUdevResource, arg2: CUdevResourceType) -> CUresult;
    fn cuWGLGetDevice(arg0: *mut CUdevice, arg1: *mut ::core::ffi::c_void) -> CUresult;
}

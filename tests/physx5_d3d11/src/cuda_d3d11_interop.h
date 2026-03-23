#pragma once

// CUDA-D3D11 interop layer
// Dynamically loads nvcuda.dll (Wine relay or ZLUDA) and provides
// cuGraphicsD3D11RegisterResource and related functions.
//
// No CUDA SDK dependency — we declare the minimal types ourselves.

#include <cstdint>
#include <cstddef>

// --- Minimal CUDA type declarations ---
// These match the real CUDA driver API types without requiring the CUDA SDK.

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUgraphicsResource;
typedef void* CUarray;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_UNKNOWN 999

#define CU_GRAPHICS_REGISTER_FLAGS_NONE 0

// --- CUDA function pointer types ---
typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*PFN_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*PFN_cuCtxDestroy)(CUcontext);
typedef CUresult (*PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (*PFN_cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*PFN_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int,
                                        unsigned int, CUstream, void**, void**);
typedef CUresult (*PFN_cuStreamSynchronize)(CUstream);

// D3D11 interop
typedef CUresult (*PFN_cuGraphicsD3D11RegisterResource)(CUgraphicsResource*, void*, unsigned int);
typedef CUresult (*PFN_cuGraphicsMapResources)(unsigned int, CUgraphicsResource*, CUstream);
typedef CUresult (*PFN_cuGraphicsUnmapResources)(unsigned int, CUgraphicsResource*, CUstream);
typedef CUresult (*PFN_cuGraphicsSubResourceGetMappedArray)(CUarray*, CUgraphicsResource,
                                                             unsigned int, unsigned int);
typedef CUresult (*PFN_cuGraphicsResourceGetMappedPointer)(CUdeviceptr*, size_t*,
                                                            CUgraphicsResource);
typedef CUresult (*PFN_cuGraphicsUnregisterResource)(CUgraphicsResource);
typedef CUresult (*PFN_cuD3D11GetDevice)(CUdevice*, void*);

// --- Interop state ---
struct CudaInterop {
    // Loaded library handle
    void* cudaLib = nullptr;

    // Core CUDA functions
    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuCtxCreate cuCtxCreate = nullptr;
    PFN_cuCtxDestroy cuCtxDestroy = nullptr;
    PFN_cuMemAlloc cuMemAlloc = nullptr;
    PFN_cuMemFree cuMemFree = nullptr;
    PFN_cuMemcpyDtoD cuMemcpyDtoD = nullptr;
    PFN_cuMemcpyHtoD cuMemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH cuMemcpyDtoH = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;
    PFN_cuStreamSynchronize cuStreamSynchronize = nullptr;

    // D3D11 interop functions
    PFN_cuGraphicsD3D11RegisterResource cuGraphicsD3D11RegisterResource = nullptr;
    PFN_cuGraphicsMapResources cuGraphicsMapResources = nullptr;
    PFN_cuGraphicsUnmapResources cuGraphicsUnmapResources = nullptr;
    PFN_cuGraphicsSubResourceGetMappedArray cuGraphicsSubResourceGetMappedArray = nullptr;
    PFN_cuGraphicsResourceGetMappedPointer cuGraphicsResourceGetMappedPointer = nullptr;
    PFN_cuGraphicsUnregisterResource cuGraphicsUnregisterResource = nullptr;
    PFN_cuD3D11GetDevice cuD3D11GetDevice = nullptr;

    CUcontext context = nullptr;
    bool initialized = false;
};

// Load nvcuda.dll and resolve all function pointers.
// On Windows/Wine: loads nvcuda.dll (Wine relay or ZLUDA override)
// On Linux: loads libcuda.so (ZLUDA or real CUDA)
bool cudaInteropInit(CudaInterop& interop);

// Create CUDA context
bool cudaInteropCreateContext(CudaInterop& interop);

// Register a D3D11 resource with CUDA for interop.
// d3d11Resource should be ID3D11Buffer* or ID3D11Texture2D*.
bool cudaInteropRegisterD3D11(CudaInterop& interop, void* d3d11Resource,
                               CUgraphicsResource* outResource);

// Map a registered resource — makes it accessible from CUDA.
bool cudaInteropMap(CudaInterop& interop, CUgraphicsResource resource);

// Get the device pointer of a mapped buffer resource.
bool cudaInteropGetMappedPointer(CudaInterop& interop, CUgraphicsResource resource,
                                  CUdeviceptr* outPtr, size_t* outSize);

// Get the array of a mapped texture resource.
bool cudaInteropGetMappedArray(CudaInterop& interop, CUgraphicsResource resource,
                                CUarray* outArray);

// Unmap a resource — D3D11 can use it again.
bool cudaInteropUnmap(CudaInterop& interop, CUgraphicsResource resource);

// Unregister a resource.
bool cudaInteropUnregister(CudaInterop& interop, CUgraphicsResource resource);

// Clean up
void cudaInteropCleanup(CudaInterop& interop);

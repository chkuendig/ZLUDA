// CUDA-D3D11 interop implementation
// Dynamically loads CUDA functions from nvcuda.dll (Wine/ZLUDA) or libcuda.so

#include "cuda_d3d11_interop.h"
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define LOAD_LIBRARY(name)      LoadLibraryA(name)
#define GET_PROC(lib, name)     GetProcAddress((HMODULE)(lib), name)
#define FREE_LIBRARY(lib)       FreeLibrary((HMODULE)(lib))
#else
#include <dlfcn.h>
#define LOAD_LIBRARY(name)      dlopen(name, RTLD_LAZY)
#define GET_PROC(lib, name)     dlsym(lib, name)
#define FREE_LIBRARY(lib)       dlclose(lib)
#endif

// Helper macro to load a function pointer
#define LOAD_FUNC(interop, name) do { \
    interop.name = (PFN_##name)GET_PROC(interop.cudaLib, #name); \
    if (!interop.name) { \
        /* Try versioned name */ \
        interop.name = (PFN_##name)GET_PROC(interop.cudaLib, #name "_v2"); \
    } \
} while(0)

#define LOAD_FUNC_REQUIRED(interop, name) do { \
    LOAD_FUNC(interop, name); \
    if (!interop.name) { \
        fprintf(stderr, "[CUDA] ERROR: Failed to load " #name "\n"); \
        return false; \
    } \
} while(0)

bool cudaInteropInit(CudaInterop& interop) {
    fprintf(stderr, "[CUDA] Loading CUDA library...\n");

#ifdef _WIN32
    // On Windows/Wine: nvcuda.dll
    // - Default: Wine's nvcuda relay (may intercept D3D11 calls)
    // - With WINEDLLOVERRIDES="nvcuda=n": ZLUDA's nvcuda.dll
    interop.cudaLib = LOAD_LIBRARY("nvcuda.dll");
    if (!interop.cudaLib) {
        fprintf(stderr, "[CUDA] ERROR: Failed to load nvcuda.dll\n");
        fprintf(stderr, "[CUDA] Under Wine: nvcuda.dll should be provided by relay or ZLUDA\n");
        return false;
    }
#else
    // On Linux: libcuda.so (ZLUDA via LD_PRELOAD or real CUDA)
    interop.cudaLib = LOAD_LIBRARY("libcuda.so.1");
    if (!interop.cudaLib) {
        interop.cudaLib = LOAD_LIBRARY("libcuda.so");
    }
    if (!interop.cudaLib) {
        fprintf(stderr, "[CUDA] ERROR: Failed to load libcuda.so\n");
        return false;
    }
#endif
    fprintf(stderr, "[CUDA] Library loaded\n");

    // --- Load core functions ---
    LOAD_FUNC_REQUIRED(interop, cuInit);
    LOAD_FUNC_REQUIRED(interop, cuDeviceGet);
    LOAD_FUNC_REQUIRED(interop, cuCtxCreate);
    LOAD_FUNC(interop, cuCtxDestroy);
    LOAD_FUNC(interop, cuMemAlloc);
    LOAD_FUNC(interop, cuMemFree);
    LOAD_FUNC(interop, cuMemcpyDtoD);
    LOAD_FUNC(interop, cuMemcpyHtoD);
    LOAD_FUNC(interop, cuMemcpyDtoH);
    LOAD_FUNC(interop, cuModuleLoadData);
    LOAD_FUNC(interop, cuModuleGetFunction);
    LOAD_FUNC(interop, cuLaunchKernel);
    LOAD_FUNC(interop, cuStreamSynchronize);

    // --- Load D3D11 interop functions ---
    // These are the critical ones for PhysX interop testing
    LOAD_FUNC(interop, cuGraphicsD3D11RegisterResource);
    LOAD_FUNC(interop, cuGraphicsMapResources);
    LOAD_FUNC(interop, cuGraphicsUnmapResources);
    LOAD_FUNC(interop, cuGraphicsSubResourceGetMappedArray);
    LOAD_FUNC(interop, cuGraphicsResourceGetMappedPointer);
    LOAD_FUNC(interop, cuGraphicsUnregisterResource);
    LOAD_FUNC(interop, cuD3D11GetDevice);

    // Report which functions are available
    fprintf(stderr, "[CUDA] Function availability:\n");
    fprintf(stderr, "  cuInit:                         %s\n", interop.cuInit ? "YES" : "NO");
    fprintf(stderr, "  cuGraphicsD3D11RegisterResource: %s\n",
            interop.cuGraphicsD3D11RegisterResource ? "YES" : "NO");
    fprintf(stderr, "  cuGraphicsMapResources:          %s\n",
            interop.cuGraphicsMapResources ? "YES" : "NO");
    fprintf(stderr, "  cuGraphicsResourceGetMappedPointer: %s\n",
            interop.cuGraphicsResourceGetMappedPointer ? "YES" : "NO");
    fprintf(stderr, "  cuD3D11GetDevice:                %s\n",
            interop.cuD3D11GetDevice ? "YES" : "NO");

    return true;
}

bool cudaInteropCreateContext(CudaInterop& interop) {
    if (!interop.cuInit) return false;

    CUresult res = interop.cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuInit failed (result=%d)\n", res);
        return false;
    }
    fprintf(stderr, "[CUDA] cuInit OK\n");

    CUdevice device;
    res = interop.cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuDeviceGet failed (result=%d)\n", res);
        return false;
    }

    res = interop.cuCtxCreate(&interop.context, 0, device);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuCtxCreate failed (result=%d)\n", res);
        return false;
    }

    fprintf(stderr, "[CUDA] Context created (device=%d)\n", device);
    interop.initialized = true;
    return true;
}

bool cudaInteropRegisterD3D11(CudaInterop& interop, void* d3d11Resource,
                               CUgraphicsResource* outResource) {
    if (!interop.cuGraphicsD3D11RegisterResource) {
        fprintf(stderr, "[CUDA] ERROR: cuGraphicsD3D11RegisterResource not available!\n");
        fprintf(stderr, "[CUDA] This function may not be exported by Wine's nvcuda relay.\n");
        fprintf(stderr, "[CUDA] Try: WINEDLLOVERRIDES=\"nvcuda=n\" to use ZLUDA directly.\n");
        return false;
    }

    fprintf(stderr, "[CUDA] Calling cuGraphicsD3D11RegisterResource(resource=%p)...\n",
            d3d11Resource);
    CUresult res = interop.cuGraphicsD3D11RegisterResource(
        outResource, d3d11Resource, CU_GRAPHICS_REGISTER_FLAGS_NONE);

    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuGraphicsD3D11RegisterResource failed (result=%d)\n", res);
        return false;
    }

    fprintf(stderr, "[CUDA] Registration OK (cuResource=%p)\n", *outResource);
    return true;
}

bool cudaInteropMap(CudaInterop& interop, CUgraphicsResource resource) {
    if (!interop.cuGraphicsMapResources) return false;

    CUresult res = interop.cuGraphicsMapResources(1, &resource, nullptr);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuGraphicsMapResources failed (result=%d)\n", res);
        return false;
    }
    fprintf(stderr, "[CUDA] Map OK\n");
    return true;
}

bool cudaInteropGetMappedPointer(CudaInterop& interop, CUgraphicsResource resource,
                                  CUdeviceptr* outPtr, size_t* outSize) {
    if (!interop.cuGraphicsResourceGetMappedPointer) return false;

    CUresult res = interop.cuGraphicsResourceGetMappedPointer(outPtr, outSize, resource);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuGraphicsResourceGetMappedPointer failed (result=%d)\n", res);
        return false;
    }
    fprintf(stderr, "[CUDA] Mapped pointer: 0x%llx (size=%zu)\n",
            (unsigned long long)*outPtr, *outSize);
    return true;
}

bool cudaInteropGetMappedArray(CudaInterop& interop, CUgraphicsResource resource,
                                CUarray* outArray) {
    if (!interop.cuGraphicsSubResourceGetMappedArray) return false;

    CUresult res = interop.cuGraphicsSubResourceGetMappedArray(outArray, resource, 0, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuGraphicsSubResourceGetMappedArray failed (result=%d)\n", res);
        return false;
    }
    fprintf(stderr, "[CUDA] Mapped array: %p\n", *outArray);
    return true;
}

bool cudaInteropUnmap(CudaInterop& interop, CUgraphicsResource resource) {
    if (!interop.cuGraphicsUnmapResources) return false;

    CUresult res = interop.cuGraphicsUnmapResources(1, &resource, nullptr);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuGraphicsUnmapResources failed (result=%d)\n", res);
        return false;
    }
    fprintf(stderr, "[CUDA] Unmap OK\n");
    return true;
}

bool cudaInteropUnregister(CudaInterop& interop, CUgraphicsResource resource) {
    if (!interop.cuGraphicsUnregisterResource) return false;

    CUresult res = interop.cuGraphicsUnregisterResource(resource);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[CUDA] ERROR: cuGraphicsUnregisterResource failed (result=%d)\n", res);
        return false;
    }
    fprintf(stderr, "[CUDA] Unregister OK\n");
    return true;
}

void cudaInteropCleanup(CudaInterop& interop) {
    fprintf(stderr, "[CUDA] Cleaning up...\n");
    if (interop.context && interop.cuCtxDestroy) {
        interop.cuCtxDestroy(interop.context);
        interop.context = nullptr;
    }
    if (interop.cudaLib) {
        FREE_LIBRARY(interop.cudaLib);
        interop.cudaLib = nullptr;
    }
    interop.initialized = false;
    fprintf(stderr, "[CUDA] Cleanup complete\n");
}

/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Shared helpers for the W1 bridge fixtures. Black-box only: every CUDA entry
 * point is resolved from whatever nvcuda.dll the Wine loader selects, through
 * LoadLibrary/GetProcAddress, against the public CUDA Driver API. The types
 * below are declared from the public cuda.h ABI; nothing here comes from a
 * bridge implementation.
 */
#ifndef W1_H
#define W1_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN64
#define W1_ARCH "x64"
#else
#define W1_ARCH "pe32"
#endif

#define CUDAAPI __stdcall

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUgraphicsResource_st *CUgraphicsResource;
typedef struct CUarray_st *CUarray;
typedef struct CUgraph_st *CUgraph;
typedef struct CUgraphExec_st *CUgraphExec;
typedef struct CUmemPoolHandle_st *CUmemoryPool;
/* cuda.h: CUdeviceptr is pointer-sized on the caller's architecture. */
typedef uintptr_t CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef struct { unsigned char bytes[16]; } CUuuid;
typedef struct { char reserved[64]; } CUipcEventHandle;
typedef struct { char reserved[64]; } CUipcMemHandle;
typedef struct { int type; int id; } CUmemLocation;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_ALREADY_MAPPED 208
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_NOT_READY 600
#define CUDA_ERROR_NOT_PERMITTED 800
#define CUDA_ERROR_NOT_SUPPORTED 801

#define CU_STREAM_NON_BLOCKING 0x1
#define CU_CTX_SCHED_AUTO 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_STREAM_PER_THREAD ((CUstream)0x2)

/* ---- output: one JSON object per line on stdout, flushed every line ---- */

static CRITICAL_SECTION w1_out_lock;
static LARGE_INTEGER w1_qpf, w1_qpc0;
static int w1_failures;

static double w1_ms(void)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - w1_qpc0.QuadPart) * 1000.0 / (double)w1_qpf.QuadPart;
}

/* Fields whose names start with "v_" are volatile (addresses, timings,
 * thread ids) and are ignored by tools/compare.py. Everything else is a
 * deterministic outcome that must match between control and candidate. */
static void w1_emit(const char *fmt, ...)
{
    va_list ap;
    EnterCriticalSection(&w1_out_lock);
    printf("{\"arch\":\"" W1_ARCH "\",\"v_ms\":%.3f,", w1_ms());
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("}\n");
    fflush(stdout);
    LeaveCriticalSection(&w1_out_lock);
}

static void w1_check(const char *name, int ok, const char *fmt, ...)
{
    char detail[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(detail, sizeof detail, fmt, ap);
    va_end(ap);
    if (!ok)
        InterlockedIncrement((LONG *)&w1_failures);
    w1_emit("\"check\":\"%s\",\"pass\":%s,%s", name, ok ? "true" : "false", detail);
}

/* ---- deadline: a timeout is never success (exit 124) ---- */

static DWORD w1_deadline_ms;

static DWORD WINAPI w1_watchdog(void *unused)
{
    (void)unused;
    Sleep(w1_deadline_ms);
    w1_emit("\"event\":\"deadline\",\"deadline_ms\":%lu", (unsigned long)w1_deadline_ms);
    TerminateProcess(GetCurrentProcess(), 124);
    return 0;
}

static void w1_start(const char *fixture, DWORD deadline_ms)
{
    InitializeCriticalSection(&w1_out_lock);
    QueryPerformanceFrequency(&w1_qpf);
    QueryPerformanceCounter(&w1_qpc0);
    const char *env = getenv("W1_DEADLINE_MS");
    w1_deadline_ms = env ? (DWORD)strtoul(env, NULL, 10) : deadline_ms;
    HANDLE t = CreateThread(NULL, 0, w1_watchdog, NULL, 0, NULL);
    if (t)
        CloseHandle(t);
    w1_emit("\"event\":\"start\",\"fixture\":\"%s\",\"deadline_ms\":%lu,\"v_pid\":%lu", fixture,
            (unsigned long)w1_deadline_ms, (unsigned long)GetCurrentProcessId());
}

/* Exit 0 = every check passed; 1 = at least one check failed; 2 = setup could
 * not run (no nvcuda, missing export); 124 = deadline. */
static int w1_finish(const char *fixture)
{
    w1_emit("\"event\":\"finish\",\"fixture\":\"%s\",\"failures\":%d", fixture, w1_failures);
    return w1_failures ? 1 : 0;
}

/* ---- nvcuda resolution ---- */

static HMODULE w1_nvcuda;

static void w1_module_identity(HMODULE module, const char *label)
{
    WCHAR path[MAX_PATH] = {0};
    char narrow[MAX_PATH * 3] = {0};
    GetModuleFileNameW(module, path, MAX_PATH);
    WideCharToMultiByte(CP_UTF8, 0, path, -1, narrow, sizeof narrow, NULL, NULL);
    for (char *c = narrow; *c; c++)
        if (*c == '\\')
            *c = '/';
    const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)module;
    const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const char *)module + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY *dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    unsigned long names = 0, functions = 0;
    if (dir->VirtualAddress) {
        const IMAGE_EXPORT_DIRECTORY *exp =
            (const IMAGE_EXPORT_DIRECTORY *)((const char *)module + dir->VirtualAddress);
        names = exp->NumberOfNames;
        functions = exp->NumberOfFunctions;
    }
    w1_emit("\"event\":\"module\",\"label\":\"%s\",\"export_names\":%lu,\"export_functions\":%lu,"
            "\"machine\":%u,\"v_path\":\"%s\",\"v_base\":\"%p\"",
            label, names, functions, nt->FileHeader.Machine, narrow, (void *)module);
}

static HMODULE w1_load_nvcuda(void)
{
    const char *name = getenv("W1_NVCUDA");
    if (!name)
        name = "nvcuda.dll";
    w1_nvcuda = LoadLibraryA(name);
    if (!w1_nvcuda) {
        w1_emit("\"event\":\"nvcuda-load\",\"ok\":false,\"v_error\":%lu", (unsigned long)GetLastError());
        return NULL;
    }
    w1_module_identity(w1_nvcuda, "nvcuda");
    return w1_nvcuda;
}

static void *w1_sym(const char *name)
{
    return w1_nvcuda ? (void *)GetProcAddress(w1_nvcuda, name) : NULL;
}

/* Declare a function pointer resolved from nvcuda; missing exports are
 * reported and leave the pointer NULL. */
#define W1_FN(ret, name, args) typedef ret(CUDAAPI *PFN_##name) args; static PFN_##name p_##name
#define W1_RESOLVE(name, exported)                                                           \
    do {                                                                                     \
        p_##name = (PFN_##name)w1_sym(exported);                                             \
        if (!p_##name)                                                                       \
            w1_emit("\"event\":\"missing-export\",\"name\":\"%s\"", exported);               \
    } while (0)

/* The public driver API subset used across fixtures. */
W1_FN(CUresult, cuInit, (unsigned int));
W1_FN(CUresult, cuDriverGetVersion, (int *));
W1_FN(CUresult, cuDeviceGetCount, (int *));
W1_FN(CUresult, cuDeviceGet, (CUdevice *, int));
W1_FN(CUresult, cuDeviceGetName, (char *, int, CUdevice));
W1_FN(CUresult, cuDeviceTotalMem, (size_t *, CUdevice));
W1_FN(CUresult, cuDeviceGetAttribute, (int *, int, CUdevice));
W1_FN(CUresult, cuCtxCreate, (CUcontext *, unsigned int, CUdevice));
W1_FN(CUresult, cuCtxDestroy, (CUcontext));
W1_FN(CUresult, cuCtxSetCurrent, (CUcontext));
W1_FN(CUresult, cuCtxSynchronize, (void));
W1_FN(CUresult, cuDevicePrimaryCtxRetain, (CUcontext *, CUdevice));
W1_FN(CUresult, cuDevicePrimaryCtxRelease, (CUdevice));
W1_FN(CUresult, cuDevicePrimaryCtxReset, (CUdevice));
W1_FN(CUresult, cuMemAlloc, (CUdeviceptr *, size_t));
W1_FN(CUresult, cuMemFree, (CUdeviceptr));
W1_FN(CUresult, cuMemGetInfo, (size_t *, size_t *));
W1_FN(CUresult, cuMemAllocHost, (void **, size_t));
W1_FN(CUresult, cuMemFreeHost, (void *));
W1_FN(CUresult, cuMemcpyHtoD, (CUdeviceptr, const void *, size_t));
W1_FN(CUresult, cuMemcpyDtoH, (void *, CUdeviceptr, size_t));
W1_FN(CUresult, cuMemcpyDtoD, (CUdeviceptr, CUdeviceptr, size_t));
W1_FN(CUresult, cuMemcpyHtoDAsync, (CUdeviceptr, const void *, size_t, CUstream));
W1_FN(CUresult, cuMemcpyDtoHAsync, (void *, CUdeviceptr, size_t, CUstream));
W1_FN(CUresult, cuMemsetD8, (CUdeviceptr, unsigned char, size_t));
W1_FN(CUresult, cuMemsetD32, (CUdeviceptr, unsigned int, size_t));
W1_FN(CUresult, cuModuleLoadData, (CUmodule *, const void *));
W1_FN(CUresult, cuModuleUnload, (CUmodule));
W1_FN(CUresult, cuModuleGetFunction, (CUfunction *, CUmodule, const char *));
W1_FN(CUresult, cuLaunchKernel, (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                 unsigned, CUstream, void **, void **));
W1_FN(CUresult, cuStreamCreate, (CUstream *, unsigned int));
W1_FN(CUresult, cuStreamDestroy, (CUstream));
W1_FN(CUresult, cuStreamSynchronize, (CUstream));
W1_FN(CUresult, cuStreamQuery, (CUstream));
W1_FN(CUresult, cuEventCreate, (CUevent *, unsigned int));
W1_FN(CUresult, cuEventDestroy, (CUevent));
W1_FN(CUresult, cuEventRecord, (CUevent, CUstream));
W1_FN(CUresult, cuEventSynchronize, (CUevent));
W1_FN(CUresult, cuEventElapsedTime, (float *, CUevent, CUevent));
W1_FN(CUresult, cuStreamWaitEvent, (CUstream, CUevent, unsigned int));
W1_FN(CUresult, cuGetExportTable, (const void **, const CUuuid *));
W1_FN(CUresult, cuGetErrorName, (CUresult, const char **));

/* Resolve the shared subset. cuda.h maps the unversioned names onto _v2 for
 * the size_t/CUdeviceptr entries; resolve exactly those exported names. */
static void w1_resolve_common(void)
{
    W1_RESOLVE(cuInit, "cuInit");
    W1_RESOLVE(cuDriverGetVersion, "cuDriverGetVersion");
    W1_RESOLVE(cuDeviceGetCount, "cuDeviceGetCount");
    W1_RESOLVE(cuDeviceGet, "cuDeviceGet");
    W1_RESOLVE(cuDeviceGetName, "cuDeviceGetName");
    W1_RESOLVE(cuDeviceTotalMem, "cuDeviceTotalMem_v2");
    W1_RESOLVE(cuDeviceGetAttribute, "cuDeviceGetAttribute");
    W1_RESOLVE(cuCtxCreate, "cuCtxCreate_v2");
    W1_RESOLVE(cuCtxDestroy, "cuCtxDestroy_v2");
    W1_RESOLVE(cuCtxSetCurrent, "cuCtxSetCurrent");
    W1_RESOLVE(cuCtxSynchronize, "cuCtxSynchronize");
    W1_RESOLVE(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    W1_RESOLVE(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    W1_RESOLVE(cuDevicePrimaryCtxReset, "cuDevicePrimaryCtxReset_v2");
    W1_RESOLVE(cuMemAlloc, "cuMemAlloc_v2");
    W1_RESOLVE(cuMemFree, "cuMemFree_v2");
    W1_RESOLVE(cuMemGetInfo, "cuMemGetInfo_v2");
    W1_RESOLVE(cuMemAllocHost, "cuMemAllocHost_v2");
    W1_RESOLVE(cuMemFreeHost, "cuMemFreeHost");
    W1_RESOLVE(cuMemcpyHtoD, "cuMemcpyHtoD_v2");
    W1_RESOLVE(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
    W1_RESOLVE(cuMemcpyDtoD, "cuMemcpyDtoD_v2");
    W1_RESOLVE(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    W1_RESOLVE(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    W1_RESOLVE(cuMemsetD8, "cuMemsetD8_v2");
    W1_RESOLVE(cuMemsetD32, "cuMemsetD32_v2");
    W1_RESOLVE(cuModuleLoadData, "cuModuleLoadData");
    W1_RESOLVE(cuModuleUnload, "cuModuleUnload");
    W1_RESOLVE(cuModuleGetFunction, "cuModuleGetFunction");
    W1_RESOLVE(cuLaunchKernel, "cuLaunchKernel");
    W1_RESOLVE(cuStreamCreate, "cuStreamCreate");
    W1_RESOLVE(cuStreamDestroy, "cuStreamDestroy_v2");
    W1_RESOLVE(cuStreamSynchronize, "cuStreamSynchronize");
    W1_RESOLVE(cuStreamQuery, "cuStreamQuery");
    W1_RESOLVE(cuEventCreate, "cuEventCreate");
    W1_RESOLVE(cuEventDestroy, "cuEventDestroy_v2");
    W1_RESOLVE(cuEventRecord, "cuEventRecord");
    W1_RESOLVE(cuEventSynchronize, "cuEventSynchronize");
    W1_RESOLVE(cuEventElapsedTime, "cuEventElapsedTime");
    W1_RESOLVE(cuStreamWaitEvent, "cuStreamWaitEvent");
    W1_RESOLVE(cuGetExportTable, "cuGetExportTable");
    W1_RESOLVE(cuGetErrorName, "cuGetErrorName");
}

/* Call through a resolved pointer; a missing export yields a sentinel that
 * no driver returns, so the check fails instead of crashing. */
#define W1_MISSING (-1)
#define CU(name, ...) (p_##name ? p_##name(__VA_ARGS__) : W1_MISSING)

/* Bring up device 0 and a fresh context. Returns 0 on success. */
static int w1_bring_up(CUdevice *dev, CUcontext *ctx)
{
    CUresult r = CU(cuInit, 0);
    w1_check("cuInit", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (r)
        return -1;
    int count = 0;
    r = CU(cuDeviceGetCount, &count);
    w1_check("cuDeviceGetCount", r == CUDA_SUCCESS && count > 0, "\"status\":%d,\"count\":%d", r, count);
    if (r || count <= 0)
        return -1;
    r = CU(cuDeviceGet, dev, 0);
    if (r) {
        w1_check("cuDeviceGet", 0, "\"status\":%d", r);
        return -1;
    }
    r = CU(cuCtxCreate, ctx, CU_CTX_SCHED_AUTO, *dev);
    w1_check("cuCtxCreate", r == CUDA_SUCCESS, "\"status\":%d", r);
    return r ? -1 : 0;
}

#endif

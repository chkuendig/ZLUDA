/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Implemented part of the fake libcuda.so.1 (SysV ABI, no GPU). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef struct { unsigned char bytes[16]; } CUuuid;

#define FAKE_CTX ((void *)0x10000)
#define FAKE_MODULE ((void *)0x20000)
#define FAKE_FUNCTION ((void *)0x30000)

static int verbose(void)
{
    static int v = -1;
    if (v < 0)
        v = getenv("STUB_QUIET") == NULL;
    return v;
}

#define LOG(...) do { if (verbose()) { fprintf(stderr, "STUB: " __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void stub_unimplemented(const char *name) { LOG("unimplemented %s", name); }

CUresult cuInit(unsigned flags) { LOG("cuInit(%u)", flags); return 0; }
CUresult cuDriverGetVersion(int *v) { *v = 12020; return 0; }
CUresult cuDeviceGetCount(int *c) { LOG("cuDeviceGetCount"); *c = 1; return 0; }
CUresult cuDeviceGet(int *d, int ordinal) { LOG("cuDeviceGet(%d)", ordinal); *d = ordinal; return ordinal ? 101 : 0; }
CUresult cuDeviceGetName(char *name, int len, int dev)
{
    LOG("cuDeviceGetName(%d, %d)", len, dev);
    snprintf(name, len, "zluda32 route stub");
    return 0;
}
CUresult cuDeviceTotalMem_v2(size_t *bytes, int dev) { *bytes = (size_t)1 << 30; return 0; }
CUresult cuDeviceComputeCapability(int *major, int *minor, int dev) { *major = 8; *minor = 6; return 0; }
CUresult cuDeviceGetUuid(CUuuid *uuid, int dev)
{
    static const unsigned char id[16] = {0x5a, 0x4c, 0x55, 0x44, 0x41, 0x33, 0x32, 0x2d,
                                         0x72, 0x6f, 0x75, 0x74, 0x65, 0x2d, 0x30, 0x31};
    LOG("cuDeviceGetUuid(%d)", dev);
    memcpy(uuid->bytes, id, 16);
    return 0;
}
CUresult cuDeviceGetAttribute(int *pi, int attrib, int dev) { *pi = 1; return 0; }
CUresult cuDevicePrimaryCtxRetain(void **ctx, int dev) { LOG("cuDevicePrimaryCtxRetain(%d)", dev); *ctx = FAKE_CTX; return 0; }
CUresult cuCtxCreate_v2(void **ctx, unsigned flags, int dev) { *ctx = FAKE_CTX; return 0; }
CUresult cuCtxSetCurrent(void *ctx) { LOG("cuCtxSetCurrent(%p)", ctx); return ctx == FAKE_CTX ? 0 : 201; }
CUresult cuCtxGetDevice(int *dev) { *dev = 0; return 0; }
CUresult cuCtxGetApiVersion(void *ctx, unsigned *v) { *v = 3020; return 0; }
CUresult cuCtxSynchronize(void) { return 0; }
CUresult cuStreamSynchronize(void *stream) { return 0; }
CUresult cuModuleLoadData(void **module, const void *image)
{
    LOG("cuModuleLoadData(%.40s...)", (const char *)image);
    *module = FAKE_MODULE;
    return 0;
}
CUresult cuModuleGetFunction(void **fn, void *module, const char *name)
{
    LOG("cuModuleGetFunction(%p, %s)", module, name);
    if (module != FAKE_MODULE)
        return 400;
    *fn = FAKE_FUNCTION;
    return 0;
}
/* CUdeviceptr is pointer-sized: 64-bit on x86_64, 32-bit for the i386 build. */
typedef uintptr_t CUdeviceptr;
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) { LOG("cuMemAlloc_v2(%zu)", size); *dptr = (CUdeviceptr)0x7f000000u << (sizeof(CUdeviceptr) > 4 ? 8 : 0); return 0; }
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n) { return 0; }
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t n, void *stream) { return 0; }
CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t n, void *stream) { memset(dst, 0, n); return 0; }
CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shmem, void *stream, void **params, void **extra)
{
    return f == FAKE_FUNCTION ? 0 : 400;
}

/* ZLUDA32_INTERNAL {EE376A1B-B858-42EB-A52D-9A106D50D914}: two entries, no size field. */
static const unsigned char zluda32_guid[16] = {0xEE, 0x37, 0x6A, 0x1B, 0xB8, 0x58, 0x42, 0xEB,
                                               0xA5, 0x2D, 0x9A, 0x10, 0x6D, 0x50, 0xD9, 0x14};
static char global_name[] = "g_stub";
static const unsigned char global_init[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

/* Six arguments: under the Windows x64 ABI the last two are on the stack, so a
 * missing ABI conversion shows up as a wrong `count` or `module`. */
static CUresult get_module_globals(char **names, const unsigned char **initializers, unsigned *sizes,
                                   unsigned *alignments, unsigned *count, void *module)
{
    LOG("ZLUDA32_INTERNAL.get_module_globals(%p, %p, %p, %p, %p, %p) module_ok=%d", (void *)names,
        (void *)initializers, (void *)sizes, (void *)alignments, (void *)count, module, module == FAKE_MODULE);
    if (module != FAKE_MODULE || !count)
        return 400;
    *count = 1;
    if (names) names[0] = global_name;
    if (initializers) initializers[0] = global_init;
    if (sizes) sizes[0] = sizeof(global_init);
    if (alignments) alignments[0] = 4;
    return 0;
}

struct arg_info { uint32_t size, align; };

static CUresult get_function_info(struct arg_info *args, unsigned *count, void *fn)
{
    LOG("ZLUDA32_INTERNAL.get_function_info(%p, %p, %p) function_ok=%d", (void *)args, (void *)count, fn,
        fn == FAKE_FUNCTION);
    if (fn != FAKE_FUNCTION)
        return 400;
    if (args) {
        args[0] = (struct arg_info){4, 4};
        args[1] = (struct arg_info){8, 8};
    }
    if (count)
        *count = 2;
    return 0;
}

static const void *zluda32_table[2] = {(void *)get_module_globals, (void *)get_function_info};

CUresult cuGetExportTable(const void **table, const CUuuid *id)
{
    if (!memcmp(id->bytes, zluda32_guid, 16)) {
        LOG("cuGetExportTable(ZLUDA32_INTERNAL)");
        *table = zluda32_table;
        return 0;
    }
    return 1;
}

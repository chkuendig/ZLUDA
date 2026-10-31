/* SPDX-License-Identifier: MIT
 * GPU-free, independently typed SysV backend for C/Rust WoW64 comparisons.
 * A 512MiB pool is reserved virtually; fixtures touch only a few pages.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "wow64-native-contract.h"

struct ctx { void *id; uint64_t pool; size_t bytes; };
static struct ctx contexts[16];
static _Thread_local void *current, *stack[32];
static _Thread_local unsigned depth;
static uintptr_t next_ctx = 0x1000, next_object = 0x8000;
static uint32_t snapshot[16];
static int fail(const char *name)
{
    const char *wanted = getenv("ZLUDA_FIXTURE_FAIL");
    ++snapshot[0];
    return wanted && !strcmp(wanted, name) ? 999 : 0;
}
#define ENTER(name) do { int rc = fail(#name); if (rc) return rc; } while (0)
static struct ctx *lookup(void *id)
{
    for (unsigned i = 0; i < 16; ++i) if (id && contexts[i].id == id) return &contexts[i];
    return NULL;
}
static int context(void) { return lookup(current) ? 0 : 201; }
void zluda_fixture_snapshot(uint32_t *out) { memcpy(out, snapshot, sizeof(snapshot)); }

int cuInit(unsigned flags) { ENTER(cuInit); return flags ? 1 : 0; }
int cuDriverGetVersion(int *out) { ENTER(cuDriverGetVersion); *out = 12020; return 0; }
int cuDeviceGetCount(int *out) { ENTER(cuDeviceGetCount); *out = 1; return 0; }
int cuDeviceGet(int *out, int ordinal) { ENTER(cuDeviceGet); if (ordinal) return 101; *out = 0; return 0; }
int cuDeviceGetName(char *out, int bytes, int device)
{ ENTER(cuDeviceGetName); if (device) return 101; if (bytes > 0) { size_t n = (size_t)bytes < 9 ? (size_t)bytes : 9; memcpy(out, "fake GPU", n); } return 0; }
int cuDeviceGetAttribute(int *out, int attribute, int device)
{ ENTER(cuDeviceGetAttribute); if (device) return 101; *out = attribute + 7; return 0; }
int cuDeviceGetProperties(void *out, int device)
{ ENTER(cuDeviceGetProperties); if (device) return 101; memset(out, 0x37, 56); return 0; }
int cuDeviceComputeCapability(int *major, int *minor, int device)
{ ENTER(cuDeviceComputeCapability); if (device) return 101; *major = 9; *minor = 0; return 0; }
int cuDeviceTotalMem_v2(size_t *bytes, int device)
{ ENTER(cuDeviceTotalMem_v2); if (device) return 101; *bytes = (size_t)8 << 30; return 0; }
int cuCtxCreate_v2(void **out, unsigned flags, int device)
{
    ENTER(cuCtxCreate_v2); if (flags || device) return 1;
    for (unsigned i = 0; i < 16; ++i) if (!contexts[i].id) {
        contexts[i].id = (void *)(next_ctx += 16); *out = contexts[i].id;
        stack[depth++] = current; current = *out; return 0;
    }
    return 2;
}
int cuCtxDestroy_v2(void *ctx)
{
    ENTER(cuCtxDestroy_v2); struct ctx *c = lookup(ctx); if (!c) return 201;
    if (c->pool) munmap((void *)(uintptr_t)c->pool, c->bytes);
    memset(c, 0, sizeof(*c));
    if (current == ctx) current = depth ? stack[--depth] : NULL;
    return 0;
}
int cuCtxGetCurrent(void **out) { ENTER(cuCtxGetCurrent); *out = current; return 0; }
int cuCtxSetCurrent(void *ctx) { ENTER(cuCtxSetCurrent); if (ctx && !lookup(ctx)) return 201; current = ctx; return 0; }
int cuCtxPushCurrent_v2(void *ctx)
{ ENTER(cuCtxPushCurrent_v2); if (!lookup(ctx)) return 201; stack[depth++] = current; current = ctx; return 0; }
int cuCtxPopCurrent_v2(void **out)
{ ENTER(cuCtxPopCurrent_v2); if (!current) return 201; *out = current; current = depth ? stack[--depth] : NULL; return 0; }
int cuCtxGetDevice(int *out) { ENTER(cuCtxGetDevice); if (context()) return 201; *out = 0; return 0; }
int cuCtxGetApiVersion(void *ctx, unsigned *out)
{ ENTER(cuCtxGetApiVersion); if (!lookup(ctx)) return 201; *out = 3020; return 0; }
int cuCtxSynchronize(void) { ENTER(cuCtxSynchronize); return context(); }
int cuMemAlloc_v2(uint64_t *out, size_t bytes)
{
    ENTER(cuMemAlloc_v2); struct ctx *ctx = lookup(current); if (!ctx) return 201;
    void *memory = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) return 2;
    *out = ctx->pool = (uintptr_t)memory; ctx->bytes = bytes; return 0;
}
int cuMemFree_v2(uint64_t address)
{
    ENTER(cuMemFree_v2); struct ctx *ctx = lookup(current); if (!ctx || ctx->pool != address) return 1;
    munmap((void *)(uintptr_t)address, ctx->bytes); ctx->pool = 0; ctx->bytes = 0; return 0;
}
int cuMemcpyHtoD_v2(uint64_t dst, const void *src, size_t bytes)
{ ENTER(cuMemcpyHtoD_v2); memcpy((void *)(uintptr_t)dst, src, bytes); return 0; }
int cuMemcpyDtoH_v2(void *dst, uint64_t src, size_t bytes)
{ ENTER(cuMemcpyDtoH_v2); memcpy(dst, (void *)(uintptr_t)src, bytes); return 0; }
int cuMemcpyDtoD_v2(uint64_t dst, uint64_t src, size_t bytes)
{ ENTER(cuMemcpyDtoD_v2); memmove((void *)(uintptr_t)dst, (void *)(uintptr_t)src, bytes); return 0; }
int cuMemcpyHtoDAsync_v2(uint64_t dst, const void *src, size_t bytes, void *stream)
{ ENTER(cuMemcpyHtoDAsync_v2); snapshot[1] = (uintptr_t)stream; memcpy((void *)(uintptr_t)dst, src, bytes); return 0; }
int cuMemcpyDtoHAsync_v2(void *dst, uint64_t src, size_t bytes, void *stream)
{ ENTER(cuMemcpyDtoHAsync_v2); snapshot[1] = (uintptr_t)stream; memcpy(dst, (void *)(uintptr_t)src, bytes); return 0; }
int cuMemcpyDtoDAsync_v2(uint64_t dst, uint64_t src, size_t bytes, void *stream)
{ ENTER(cuMemcpyDtoDAsync_v2); snapshot[1] = (uintptr_t)stream; memmove((void *)(uintptr_t)dst, (void *)(uintptr_t)src, bytes); return 0; }
int cuMemsetD8_v2(uint64_t dst, unsigned char value, size_t count)
{ ENTER(cuMemsetD8_v2); memset((void *)(uintptr_t)dst, value, count); return 0; }
int cuMemsetD16_v2(uint64_t dst, unsigned short value, size_t count)
{ ENTER(cuMemsetD16_v2); for (size_t i = 0; i < count; ++i) ((uint16_t *)(uintptr_t)dst)[i] = value; return 0; }
int cuMemsetD32_v2(uint64_t dst, unsigned value, size_t count)
{ ENTER(cuMemsetD32_v2); for (size_t i = 0; i < count; ++i) ((uint32_t *)(uintptr_t)dst)[i] = value; return 0; }
int cuMemsetD8Async(uint64_t dst, unsigned char value, size_t count, void *stream)
{ ENTER(cuMemsetD8Async); snapshot[1] = (uintptr_t)stream; memset((void *)(uintptr_t)dst, value, count); return 0; }
int cuMemsetD16Async(uint64_t dst, unsigned short value, size_t count, void *stream)
{ ENTER(cuMemsetD16Async); snapshot[1] = (uintptr_t)stream; for (size_t i = 0; i < count; ++i) ((uint16_t *)(uintptr_t)dst)[i] = value; return 0; }
int cuMemsetD32Async(uint64_t dst, unsigned value, size_t count, void *stream)
{ ENTER(cuMemsetD32Async); snapshot[1] = (uintptr_t)stream; for (size_t i = 0; i < count; ++i) ((uint32_t *)(uintptr_t)dst)[i] = value; return 0; }
int cuModuleLoadData(void **out, const void *image)
{ ENTER(cuModuleLoadData); if (!image) return 1; *out = (void *)0x2000; return 0; }
int cuModuleUnload(void *module) { ENTER(cuModuleUnload); return module == (void *)0x2000 ? 0 : 400; }
int cuModuleGetFunction(void **out, void *module, const char *name)
{ ENTER(cuModuleGetFunction); if (module != (void *)0x2000 || strcmp(name, "kernel")) return 500; *out = (void *)0x3000; return 0; }
int cuModuleGetTexRef(void **out, void *module, const char *name)
{ ENTER(cuModuleGetTexRef); if (module != (void *)0x2000 || strcmp(name, "texture")) return 500; *out = (void *)0x5000; return 0; }
int cuTexRefSetFormat(void *tex, int format, int channels)
{ ENTER(cuTexRefSetFormat); if (tex != (void *)0x5000) return 400; snapshot[2] = format; snapshot[3] = channels; return 0; }
int cuTexRefSetFlags(void *tex, unsigned flags)
{ ENTER(cuTexRefSetFlags); if (tex != (void *)0x5000) return 400; snapshot[4] = flags; return 0; }
int cuTexRefSetFilterMode(void *tex, int mode)
{ ENTER(cuTexRefSetFilterMode); if (tex != (void *)0x5000) return 400; snapshot[5] = mode; return 0; }
int cuTexRefSetAddressMode(void *tex, int dim, int mode)
{ ENTER(cuTexRefSetAddressMode); if (tex != (void *)0x5000) return 400; snapshot[6] = dim; snapshot[7] = mode; return 0; }
int cuTexRefSetMipmapFilterMode(void *tex, int mode)
{ ENTER(cuTexRefSetMipmapFilterMode); if (tex != (void *)0x5000) return 400; snapshot[8] = mode; return 0; }
int cuTexRefSetMipmapLevelBias(void *tex, float value)
{ ENTER(cuTexRefSetMipmapLevelBias); if (tex != (void *)0x5000) return 400; memcpy(&snapshot[9], &value, 4); return 0; }
int cuTexRefSetMipmapLevelClamp(void *tex, float lo, float hi)
{ ENTER(cuTexRefSetMipmapLevelClamp); if (tex != (void *)0x5000) return 400; memcpy(&snapshot[10], &lo, 4); memcpy(&snapshot[11], &hi, 4); return 0; }
int cuTexRefSetMaxAnisotropy(void *tex, unsigned value)
{ ENTER(cuTexRefSetMaxAnisotropy); if (tex != (void *)0x5000) return 400; snapshot[12] = value; return 0; }
int cuTexRefSetAddress_v2(size_t *offset, void *tex, uint64_t address, size_t bytes)
{ ENTER(cuTexRefSetAddress_v2); if (tex != (void *)0x5000 || (!address && bytes)) return 1; *offset = 0; return 0; }

int cuEventCreate(void **out, unsigned flags) { ENTER(cuEventCreate); snapshot[13] = flags; *out = (void *)(next_object += 16); return 0; }
int cuEventDestroy_v2(void *event) { ENTER(cuEventDestroy_v2); return event ? 0 : 400; }
int cuEventRecord(void *event, void *stream) { ENTER(cuEventRecord); snapshot[1] = (uintptr_t)stream; return event ? 0 : 400; }
int cuEventSynchronize(void *event) { ENTER(cuEventSynchronize); return event ? 0 : 400; }
int cuEventQuery(void *event) { ENTER(cuEventQuery); return event ? 0 : 400; }
int cuEventElapsedTime(float *out, void *start, void *end)
{ ENTER(cuEventElapsedTime); if (!start || !end) return 400; *out = 2.5f; return 0; }
int cuStreamCreate(void **out, unsigned flags) { ENTER(cuStreamCreate); snapshot[13] = flags; *out = (void *)(next_object += 16); return 0; }
int cuStreamDestroy_v2(void *stream) { ENTER(cuStreamDestroy_v2); return stream ? 0 : 400; }
int cuStreamSynchronize(void *stream) { ENTER(cuStreamSynchronize); snapshot[1] = (uintptr_t)stream; return 0; }
int cuStreamQuery(void *stream) { ENTER(cuStreamQuery); snapshot[1] = (uintptr_t)stream; return 0; }
int cuStreamWaitEvent(void *stream, void *event, unsigned flags)
{ ENTER(cuStreamWaitEvent); snapshot[1] = (uintptr_t)stream; return event && flags == 0 ? 0 : 1; }
int hipHostRegister(void *memory, size_t bytes, unsigned flags)
{ ENTER(hipHostRegister); return memory && bytes && !(flags & ~1u) ? 0 : 1; }
int hipHostUnregister(void *memory) { ENTER(hipHostUnregister); return memory ? 0 : 1; }

struct arg_info { uint32_t size, align; };
static const char *global_name = "global";
static uint32_t global_initial = 7;
static int globals(char **names, const void **initializers, uint32_t *sizes, uint32_t *alignments, uint32_t *count, void *module)
{
    ENTER(globals); if (module != (void *)0x2000) return 400;
    if (names) { if (*count < 1) return 1; names[0] = (char *)global_name; initializers[0] = &global_initial; sizes[0] = 4; alignments[0] = 4; }
    *count = 1; return 0;
}
static int args(struct arg_info *info, uint32_t *count, void *function)
{
    ENTER(args); if (function != (void *)0x3000) return 400;
    if (info) { if (*count < 2) return 1; info[0] = (struct arg_info){4,4}; info[1] = (struct arg_info){4,4}; }
    *count = 2; return 0;
}
int cuLaunchKernel(void *function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
    unsigned shared, void *stream, void **parameters, void **extra)
{
    ENTER(cuLaunchKernel); snapshot[1] = (uintptr_t)stream;
    if (function != (void *)0x3000 || gx != 2 || gy != 3 || gz != 4 || bx != 5 || by != 6 || bz != 7 || shared != 8 || extra) return 1;
    uint32_t offset = *(uint32_t *)parameters[0], value = *(uint32_t *)parameters[1], global = *(uint32_t *)parameters[3];
    uint64_t pool = *(uint64_t *)parameters[2];
    *(uint32_t *)(uintptr_t)(pool + offset) = value + *(uint32_t *)(uintptr_t)(pool + global);
    ++snapshot[14]; return 0;
}
static int rt_context(void **out, int device)
{
    ENTER(rt_context); if (device) return 101;
    if (!lookup((void *)0x7000)) for (unsigned i = 0; i < 16; ++i) if (!contexts[i].id) { contexts[i].id = (void *)0x7000; break; }
    *out = (void *)0x7000; return 0;
}
static int rt_flags(int device, unsigned flags) { ENTER(rt_flags); return device || flags > 7 ? 1 : 0; }
static int rt_reset(int device) { ENTER(rt_reset); return device ? 101 : 0; }
static void rt_done(uintptr_t value) { ++snapshot[0]; snapshot[15] = value; }
static int rt_compilers(void) { ENTER(rt_compilers); return 0; }
int cuGetExportTable(const void **out, const void *uuid)
{
    ENTER(cuGetExportTable);
    static const unsigned char internal_guid[16] = {0xee,0x37,0x6a,0x1b,0xb8,0x58,0x42,0xeb,0xa5,0x2d,0x9a,0x10,0x6d,0x50,0xd9,0x14};
    static const unsigned char runtime_guid[16] = {0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9};
    static const void *internal[] = {globals, args};
    static const void *runtime[13] = {(void *)(13 * sizeof(void *)), NULL, rt_context, NULL, rt_flags, rt_reset, NULL, rt_done, NULL, NULL, NULL, NULL, rt_compilers};
    if (!memcmp(uuid, internal_guid, 16)) { *out = internal; return 0; }
    if (!memcmp(uuid, runtime_guid, 16)) { *out = runtime; return 0; }
    *out = NULL; return 801;
}

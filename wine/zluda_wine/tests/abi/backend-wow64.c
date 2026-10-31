/* SPDX-License-Identifier: MIT
 * ELF64 fixture for real PE32/WoW64 execution. Reuse the independently typed
 * CPU backend, instrumenting only calls used by the narrow Wine probe.
 * No GPU/HIP runtime is loaded. The pool is sparse mmap, not a 512MiB memset.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define cuInit fixture_inner_cuInit
#define cuMemAlloc_v2 fixture_inner_cuMemAlloc_v2
#define cuMemFree_v2 fixture_inner_cuMemFree_v2
#define cuMemcpyHtoD_v2 fixture_inner_cuMemcpyHtoD_v2
#define cuMemcpyDtoH_v2 fixture_inner_cuMemcpyDtoH_v2
#define cuCtxDestroy_v2 fixture_inner_cuCtxDestroy_v2
#include "../wow64/backend.c"
#undef cuInit
#undef cuMemAlloc_v2
#undef cuMemFree_v2
#undef cuMemcpyHtoD_v2
#undef cuMemcpyDtoH_v2
#undef cuCtxDestroy_v2
/* Re-check the exported wrappers against the independent pinned prototypes. */
#include "wow64-native-contract.h"
_Static_assert(sizeof(void *)==8 && sizeof(size_t)==8,"fixture backend must be native ELF64");
static uint64_t reserved_base;
static size_t reserved_bytes,htod_offset,dtoh_offset;
static int pool_released,context_released;
static void audit(const char *event)
{
    const char *path=getenv("ZLUDA_WINE_FIXTURE_AUDIT");
    if (!path) abort();
    FILE *file=fopen(path,"a");
    if (!file || fprintf(file,"%s\n",event)<0 || fclose(file)) abort();
}
static void json_string(FILE *file,const char *value)
{
    fputc('"',file);
    for (;*value;++value) {
        unsigned char byte=(unsigned char)*value;
        if (byte=='"' || byte=='\\') {fputc('\\',file);fputc(byte,file);}
        else if (byte<32) fprintf(file,"\\u%04x",byte);
        else fputc(byte,file);
    }
    fputc('"',file);
}
static void witness(void)
{
    const char *output=getenv("ZLUDA_WINE_FIXTURE_WITNESS");
    const char *expected=getenv("ZLUDA_WINE_FIXTURE_UNIXLIB");
    if (!output || !expected) abort();
    int mapped=0,i386=0,unexpected=0;
    FILE *maps=fopen("/proc/self/maps","r");
    if (!maps) abort();
    char line[8192];
    while (fgets(line,sizeof(line),maps)) {
        char *path=strchr(line,'/');
        if (!path) continue;
        path[strcspn(path,"\n")]=0;
        const char *base=strrchr(path,'/');
        if (!base || (strcmp(base+1,"nvcuda.so") && strcmp(base+1,"nvcuda.dll.so"))) continue;
        if (strstr(path,"/i386-unix/")) i386=1;
        if (!strcmp(path,expected)) mapped=1; else unexpected=1;
    }
    if (ferror(maps) || fclose(maps)) abort();
    FILE *file=fopen(output,"w");
    if (!file) abort();
    fprintf(file,"{\"native_bits\":%zu,\"native_pid\":%ld,\"unixlib_path\":",sizeof(void *)*8,(long)getpid());
    json_string(file,expected);
    fprintf(file,",\"unixlib_mapped\":%s,\"i386_unix_cuda_mapped\":%s,\"unexpected_cuda_mapping\":%s,"
        "\"pool_reserved_bytes\":%zu,\"pool_native_base\":%llu,\"htod_offset\":%zu,\"dtoh_offset\":%zu,"
        "\"pool_released\":%s,\"context_released\":%s}\n",
        mapped?"true":"false",i386?"true":"false",unexpected?"true":"false",reserved_bytes,
        (unsigned long long)reserved_base,htod_offset,dtoh_offset,pool_released?"true":"false",context_released?"true":"false");
    if (ferror(file) || fclose(file)) abort();
}
__attribute__((constructor)) static void loaded(void)
{
    /* Native context values intentionally cannot be copied into PE32 handles. */
    next_ctx=(uintptr_t)UINT64_C(0x100000000);
    audit("backend-loaded");witness();
}
int cuInit(unsigned flags)
{ audit("cuInit");return fixture_inner_cuInit(flags); }
int cuMemAlloc_v2(uint64_t *out,size_t bytes)
{
    if (!out || bytes!=512u*1024u*1024u || reserved_base) return 1;
    int rc=fixture_inner_cuMemAlloc_v2(out,bytes);
    if (rc) return rc;
    if (*out<=UINT32_MAX) abort();
    reserved_base=*out;reserved_bytes=bytes;
    audit("sparse-512MiB-pool");witness();return 0;
}
static int valid_range(uint64_t pointer,size_t bytes)
{
    struct ctx *ctx=lookup(current);
    return ctx && ctx->pool && pointer>=ctx->pool && bytes &&
        pointer-ctx->pool<ctx->bytes && bytes<=ctx->bytes-(size_t)(pointer-ctx->pool);
}
int cuMemcpyHtoD_v2(uint64_t dst,const void *src,size_t bytes)
{
    if (!src || !valid_range(dst,bytes) || bytes!=32) {audit("BAD-native-htod-input");return 1;}
    htod_offset=(size_t)(dst-reserved_base);
    if (htod_offset!=260) {audit("BAD-native-htod-offset");return 1;}
    int rc=fixture_inner_cuMemcpyHtoD_v2(dst,src,bytes);
    if (!rc) {audit("bounded-interior-htod");witness();}return rc;
}
int cuMemcpyDtoH_v2(void *dst,uint64_t src,size_t bytes)
{
    if (!dst || !valid_range(src,bytes) || bytes!=32) {audit("BAD-native-dtoh-input");return 1;}
    dtoh_offset=(size_t)(src-reserved_base);
    if (dtoh_offset!=260) {audit("BAD-native-dtoh-offset");return 1;}
    int rc=fixture_inner_cuMemcpyDtoH_v2(dst,src,bytes);
    if (!rc) {audit("bounded-interior-dtoh");witness();}return rc;
}
int cuMemFree_v2(uint64_t address)
{
    int rc=fixture_inner_cuMemFree_v2(address);
    if (!rc) {pool_released=1;audit("pool-released");witness();}return rc;
}
int cuCtxDestroy_v2(void *ctx)
{
    int rc=fixture_inner_cuCtxDestroy_v2(ctx);
    if (!rc) {context_released=1;audit("context-released");witness();}return rc;
}
/* Neither Windows get-proc API may leak/call a native procedure selector. */
int cuGetProcAddress_v2(const char *name,void **out,int version,uint64_t flags,unsigned *status)
{
    (void)name;(void)out;(void)version;(void)flags;(void)status;
    audit("BAD-native-getproc-called");return 999;
}
int cuGetProcAddress(const char *name,void **out,int version,uint64_t flags)
{return cuGetProcAddress_v2(name,out,version,flags,NULL);}

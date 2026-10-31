/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * W1.6 / plan §4.6 PE32 stack-balance fixture for the 24 corrected widths.
 *
 *   pe32_stack.exe [--expect correct|old]
 *
 * For each entry, two arms:
 *   raw    pushes exactly the correct argument bytes from assembly and reads
 *          ESP after return (pe32_probe.S); immune to compiler epilogues.
 *   typed  calls through the public cuda.h prototype compiled as CUDAAPI
 *          (__stdcall) with frame pointers, reading ESP around the call and
 *          checking a local canary, i.e. what a real 32-bit caller does.
 * Required per entry: return 801 (CUDA_ERROR_NOT_SUPPORTED), surviving
 * canaries, and ESP displacement 0 (--expect correct, the default) or the
 * historical old-minus-correct displacement (--expect old, the negative
 * control for 42b557a2/A1 and the self-test stub).
 * nvcuda is whatever LoadLibrary("nvcuda.dll") selects, or W1_NVCUDA.
 * cuInit is deliberately not called: these are pure unsupported stubs and the
 * fixture must work with a fake or absent backend.
 */
#ifdef _WIN64
#error "pe32_stack is an i386 fixture"
#endif
#include "w1.h"
#include "pe32_widths.h"

int32_t w1_raw_stdcall(void *fn, const uint32_t *words, uint32_t nwords, int32_t *delta, uint32_t *pad_out);

static uint32_t scratch[1024];
#define SP ((void *)scratch)
static CUipcEventHandle ipc_event;
static CUipcMemHandle ipc_mem;
static CUmemLocation location = {1, 0};

/* typed callers: one per entry, public prototypes */
typedef CUresult(CUDAAPI *T_PPQ)(CUgraphExec *, CUgraph, unsigned long long);
typedef CUresult(CUDAAPI *T_IpcEvent)(CUevent *, CUipcEventHandle);
typedef CUresult(CUDAAPI *T_IpcMem)(CUdeviceptr *, CUipcMemHandle, unsigned int);
typedef CUresult(CUDAAPI *T_Reserve)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
typedef CUresult(CUDAAPI *T_Advise)(CUdeviceptr, size_t, int, CUmemLocation);
typedef CUresult(CUDAAPI *T_Create)(CUmemGenericAllocationHandle *, size_t, const void *, unsigned long long);
typedef CUresult(CUDAAPI *T_Batch8)(CUdeviceptr *, size_t *, size_t, CUmemLocation *, size_t *, size_t,
                                    unsigned long long, CUstream);
typedef CUresult(CUDAAPI *T_Discard)(CUdeviceptr *, size_t *, size_t, unsigned long long, CUstream);
typedef CUresult(CUDAAPI *T_Export)(void *, CUmemGenericAllocationHandle, int, unsigned long long);
typedef CUresult(CUDAAPI *T_Range)(void *, CUdeviceptr, size_t, int, unsigned long long);
typedef CUresult(CUDAAPI *T_PoolExport)(void *, CUmemoryPool, int, unsigned long long);
typedef CUresult(CUDAAPI *T_PoolImport)(CUmemoryPool *, void *, int, unsigned long long);
typedef CUresult(CUDAAPI *T_Prefetch)(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream);
typedef CUresult(CUDAAPI *T_Batch3D)(size_t, void *, unsigned long long, CUstream);
typedef CUresult(CUDAAPI *T_Attr3D)(void *, unsigned long long, CUstream);
typedef CUresult(CUDAAPI *T_Multicast)(CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t,
                                       unsigned long long);

#define ESP(v) __asm__ volatile("movl %%esp, %0" : "=r"(v)::"memory")

static __attribute__((noinline)) CUresult typed_call(const char *name, void *fn, int32_t *delta, int *canary_ok)
{
    volatile uint32_t canary = 0xC4A7C4A7u;
    uint32_t before, after;
    CUresult r = W1_MISSING;
#define T(match, type, ...)                                                                                        \
    if (!strcmp(name, match)) {                                                                                    \
        ESP(before);                                                                                               \
        r = ((type)fn)(__VA_ARGS__);                                                                               \
        ESP(after);                                                                                                \
        goto done;                                                                                                 \
    }
    T("cuGraphInstantiateWithFlags", T_PPQ, SP, NULL, 0ULL)
    T("cuIpcOpenEventHandle", T_IpcEvent, SP, ipc_event)
    T("cuIpcOpenMemHandle", T_IpcMem, SP, ipc_mem, 0)
    T("cuIpcOpenMemHandle_v2", T_IpcMem, SP, ipc_mem, 0)
    T("cuMemAddressReserve", T_Reserve, SP, 0, 0, 0, 0ULL)
    T("cuMemAdvise_v2", T_Advise, 0, 0, 0, location)
    T("cuMemCreate", T_Create, SP, 0, SP, 0ULL)
    T("cuMemDiscardAndPrefetchBatchAsync", T_Batch8, SP, SP, 0, SP, SP, 0, 0ULL, NULL)
    T("cuMemDiscardAndPrefetchBatchAsync_ptsz", T_Batch8, SP, SP, 0, SP, SP, 0, 0ULL, NULL)
    T("cuMemDiscardBatchAsync", T_Discard, SP, SP, 0, 0ULL, NULL)
    T("cuMemDiscardBatchAsync_ptsz", T_Discard, SP, SP, 0, 0ULL, NULL)
    T("cuMemExportToShareableHandle", T_Export, SP, 0ULL, 0, 0ULL)
    T("cuMemGetHandleForAddressRange", T_Range, SP, 0, 0, 0, 0ULL)
    T("cuMemPoolExportToShareableHandle", T_PoolExport, SP, NULL, 0, 0ULL)
    T("cuMemPoolImportFromShareableHandle", T_PoolImport, SP, SP, 0, 0ULL)
    T("cuMemPrefetchAsync_v2", T_Prefetch, 0, 0, location, 0, NULL)
    T("cuMemPrefetchAsync_v2_ptsz", T_Prefetch, 0, 0, location, 0, NULL)
    T("cuMemPrefetchBatchAsync", T_Batch8, SP, SP, 0, SP, SP, 0, 0ULL, NULL)
    T("cuMemPrefetchBatchAsync_ptsz", T_Batch8, SP, SP, 0, SP, SP, 0, 0ULL, NULL)
    T("cuMemcpy3DBatchAsync_v2", T_Batch3D, 0, SP, 0ULL, NULL)
    T("cuMemcpy3DBatchAsync_v2_ptsz", T_Batch3D, 0, SP, 0ULL, NULL)
    T("cuMemcpy3DWithAttributesAsync", T_Attr3D, SP, 0ULL, NULL)
    T("cuMemcpy3DWithAttributesAsync_ptsz", T_Attr3D, SP, 0ULL, NULL)
    T("cuMulticastBindAddr_v2", T_Multicast, 0ULL, 0, 0, 0, 0, 0ULL)
#undef T
    *delta = 0x7fffffff;
    *canary_ok = canary == 0xC4A7C4A7u;
    return r;
done:
    *delta = (int32_t)(after - before);
    *canary_ok = canary == 0xC4A7C4A7u;
    return r;
}

static unsigned shape_words(const char *shape, uint32_t *words)
{
    unsigned n = 0;
    for (const char *c = shape; *c; c++) {
        switch (*c) {
        case 'P': words[n++] = (uint32_t)(uintptr_t)SP; break;
        case 'S': words[n++] = 0; break;
        case 'Q': words[n++] = 0; words[n++] = 0; break;
        case 'L': words[n++] = 1; words[n++] = 0; break;
        case 'A': for (int i = 0; i < 16; i++) words[n++] = 0; break;
        }
    }
    return n;
}

int main(int argc, char **argv)
{
    int expect_old = argc > 2 && !strcmp(argv[1], "--expect") && !strcmp(argv[2], "old");
    w1_start("pe32_stack", 60000);
    w1_emit("\"event\":\"expect\",\"mode\":\"%s\"", expect_old ? "old" : "correct");
    if (!w1_load_nvcuda()) {
        w1_check("nvcuda-load", 0, "\"x\":0");
        return 2;
    }
    struct entry { const char *name; int correct, old; const char *shape; } table[] = {
#define X(name, correct, old, shape) {#name, correct, old, shape},
        W1_PE32_WIDTHS(X)
#undef X
    };
    unsigned passed = 0, total = sizeof table / sizeof table[0];
    for (unsigned i = 0; i < total; i++) {
        const struct entry *e = &table[i];
        uint32_t words[32], pad[16];
        unsigned n = shape_words(e->shape, words);
        int shape_ok = n * 4 == (unsigned)e->correct;
        void *fn = w1_sym(e->name);
        if (!fn || !shape_ok) {
            w1_check(e->name, 0, "\"exported\":%s,\"shape_ok\":%s", fn ? "true" : "false", shape_ok ? "true" : "false");
            continue;
        }
        int want = expect_old ? e->old - e->correct : 0;
        int32_t raw_delta = 0x7fffffff, typed_delta = 0x7fffffff;
        int canary_ok = 0;
        int32_t raw_ret = w1_raw_stdcall(fn, words, n, &raw_delta, pad);
        int pad_ok = 1;
        for (unsigned k = 0; k < 16; k++)
            pad_ok &= pad[k] == ((k + 1) ^ 0x5A5A0000u);
        CUresult typed_ret = typed_call(e->name, fn, &typed_delta, &canary_ok);
        int ok = raw_ret == CUDA_ERROR_NOT_SUPPORTED && typed_ret == CUDA_ERROR_NOT_SUPPORTED && pad_ok && canary_ok &&
                 raw_delta == want && typed_delta == want;
        passed += ok;
        w1_check(e->name, ok,
                 "\"correct_bytes\":%d,\"old_bytes\":%d,\"expected_delta\":%d,\"raw_delta\":%ld,\"typed_delta\":%ld,"
                 "\"raw_status\":%ld,\"typed_status\":%d,\"pad_ok\":%s,\"canary_ok\":%s",
                 e->correct, e->old, want, (long)raw_delta, (long)typed_delta, (long)raw_ret, typed_ret,
                 pad_ok ? "true" : "false", canary_ok ? "true" : "false");
    }
    w1_emit("\"event\":\"summary\",\"expect\":\"%s\",\"matched\":%u,\"total\":%u", expect_old ? "old" : "correct",
            passed, total);
    return w1_finish("pe32_stack");
}

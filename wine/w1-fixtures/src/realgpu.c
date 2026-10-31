/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * W1.1 real-GPU fixture (x64 and PE32). Deterministic CPU-reference outputs,
 * device-side guard bands (canaries), stream/event ordering, get-proc and
 * per-thread-stream variants, and teardown with a free-memory leak check.
 *
 * Every status is printed. Run the same binary on the control and the
 * candidate and diff with tools/compare.py: non-v_ fields must be equal.
 * An unsupported entry stays an error and is reported, never forced to pass.
 */
#include "w1.h"
#include "w1_ptx.h"

W1_FN(CUresult, cuMemcpyHtoDAsync_ptsz, (CUdeviceptr, const void *, size_t, CUstream));
W1_FN(CUresult, cuLaunchKernel_ptsz, (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                      unsigned, CUstream, void **, void **));
W1_FN(CUresult, cuStreamSynchronize_ptsz, (CUstream));
W1_FN(CUresult, cuGetProcAddress, (const char *, void **, int, unsigned long long));
W1_FN(CUresult, cuGetProcAddress_v2, (const char *, void **, int, unsigned long long, int *));

#define GUARD_WORDS 1024u
#define GUARD 0xDEADBEEFu
#define N (1u << 16)

static CUfunction f_fill, f_saxpy, f_sum, f_atomic;
static uint32_t host_a[N], host_b[N], host_c[N + 2 * GUARD_WORDS];

static unsigned launch1d(CUfunction f, unsigned n, void **params, CUstream s)
{
    return (unsigned)CU(cuLaunchKernel, f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, s, params, NULL);
}

/* buffer layout: [GUARD_WORDS guard][N payload][GUARD_WORDS guard] */
static int guards_intact(const uint32_t *all)
{
    for (unsigned i = 0; i < GUARD_WORDS; i++)
        if (all[i] != GUARD || all[GUARD_WORDS + N + i] != GUARD)
            return 0;
    return 1;
}

static void kernels(CUmodule mod)
{
    CUresult r = CU(cuModuleGetFunction, &f_fill, mod, "w1_fill");
    r |= CU(cuModuleGetFunction, &f_saxpy, mod, "w1_saxpy");
    r |= CU(cuModuleGetFunction, &f_sum, mod, "w1_block_sum");
    r |= CU(cuModuleGetFunction, &f_atomic, mod, "w1_atomic_count");
    w1_check("get-functions", r == CUDA_SUCCESS, "\"status\":%d", r);
}

static void sync_paths(CUdeviceptr whole)
{
    CUdeviceptr payload = whole + GUARD_WORDS * 4;
    unsigned n = N, seed = 0xC0FFEEu, a = 7;
    CUresult r = CU(cuMemsetD32, whole, GUARD, N + 2 * GUARD_WORDS);
    w1_check("memsetD32", r == CUDA_SUCCESS, "\"status\":%d", r);

    void *fill_params[] = {&payload, &n, &seed};
    unsigned l = launch1d(f_fill, N, fill_params, NULL);
    r = CU(cuCtxSynchronize);
    CUresult c = CU(cuMemcpyDtoH, host_c, whole, sizeof host_c);
    unsigned bad = 0;
    for (unsigned i = 0; i < N; i++)
        bad += host_c[GUARD_WORDS + i] != w1_ref_fill(i, seed);
    w1_check("fill", !l && !r && !c && !bad, "\"launch\":%u,\"sync\":%d,\"copy\":%d,\"mismatches\":%u", l, r, c,
             bad);
    w1_check("fill-guards", !c && guards_intact(host_c), "\"x\":0");

    /* saxpy against host-uploaded x, y = fill output */
    CUdeviceptr x = 0;
    r = CU(cuMemAlloc, &x, N * 4);
    for (unsigned i = 0; i < N; i++)
        host_a[i] = i * 3u + 1u;
    r |= CU(cuMemcpyHtoD, x, host_a, N * 4);
    void *saxpy_params[] = {&payload, &x, &a, &n};
    l = launch1d(f_saxpy, N, saxpy_params, NULL);
    r |= CU(cuCtxSynchronize);
    c = CU(cuMemcpyDtoH, host_c, whole, sizeof host_c);
    bad = 0;
    for (unsigned i = 0; i < N; i++)
        bad += host_c[GUARD_WORDS + i] != a * host_a[i] + w1_ref_fill(i, seed);
    w1_check("saxpy", !l && !r && !c && !bad, "\"launch\":%u,\"status\":%d,\"mismatches\":%u", l, r, bad);
    w1_check("saxpy-guards", !c && guards_intact(host_c), "\"x\":0");

    /* DtoD copy of payload into x, then block sums of x */
    r = CU(cuMemcpyDtoD, x, payload, N * 4);
    CUdeviceptr sums = 0;
    r |= CU(cuMemAlloc, &sums, (N / 256) * 4);
    void *sum_params[] = {&x, &sums};
    l = (unsigned)CU(cuLaunchKernel, f_sum, N / 256, 1, 1, 256, 1, 1, 0, NULL, sum_params, NULL);
    r |= CU(cuCtxSynchronize);
    c = CU(cuMemcpyDtoH, host_b, sums, (N / 256) * 4);
    bad = 0;
    for (unsigned b = 0; b < N / 256; b++) {
        uint32_t s = 0;
        for (unsigned i = 0; i < 256; i++)
            s += host_c[GUARD_WORDS + b * 256 + i];
        bad += host_b[b] != s;
    }
    w1_check("dtod-block-sum", !l && !r && !c && !bad, "\"launch\":%u,\"status\":%d,\"mismatches\":%u", l, r, bad);

    /* atomics */
    CUdeviceptr counter = 0;
    r = CU(cuMemAlloc, &counter, 4);
    r |= CU(cuMemsetD8, counter, 0, 4);
    void *atomic_params[] = {&counter, &n};
    l = launch1d(f_atomic, N, atomic_params, NULL);
    r |= CU(cuCtxSynchronize);
    uint32_t count = 0;
    c = CU(cuMemcpyDtoH, &count, counter, 4);
    w1_check("atomic-count", !l && !r && !c && count == N, "\"launch\":%u,\"status\":%d,\"count\":%u", l, r,
             count);
    CU(cuMemFree, counter);
    CU(cuMemFree, sums);
    CU(cuMemFree, x);
}

static void stream_paths(CUdeviceptr whole)
{
    CUdeviceptr payload = whole + GUARD_WORDS * 4;
    CUstream s = NULL;
    CUevent e0 = NULL, e1 = NULL;
    void *pinned = NULL;
    CUresult r = CU(cuStreamCreate, &s, CU_STREAM_NON_BLOCKING);
    r |= CU(cuEventCreate, &e0, 0);
    r |= CU(cuEventCreate, &e1, 0);
    r |= CU(cuMemAllocHost, &pinned, N * 4);
    w1_check("stream-setup", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (r)
        return;
    uint32_t *p = pinned;
    for (unsigned i = 0; i < N; i++)
        p[i] = 0x01010101u * (i & 0xff);
    unsigned n = N, a = 3;
    CUdeviceptr x = 0;
    r = CU(cuMemAlloc, &x, N * 4);
    r |= CU(cuMemsetD32, payload, 5, N);
    r |= CU(cuCtxSynchronize);
    r |= CU(cuEventRecord, e0, s);
    r |= CU(cuMemcpyHtoDAsync, x, p, N * 4, s);
    void *params[] = {&payload, &x, &a, &n};
    r |= launch1d(f_saxpy, N, params, s);
    memset(host_b, 0, sizeof host_b);
    r |= CU(cuMemcpyDtoHAsync, p, payload, N * 4, s);
    r |= CU(cuEventRecord, e1, s);
    CUresult es = CU(cuEventSynchronize, e1);
    float ms = -1;
    CUresult et = CU(cuEventElapsedTime, &ms, e0, e1);
    unsigned bad = 0;
    for (unsigned i = 0; i < N; i++)
        bad += p[i] != a * (0x01010101u * (i & 0xff)) + 5u;
    w1_check("stream-order", !r && !es && !bad, "\"status\":%d,\"event_sync\":%d,\"mismatches\":%u", r, es, bad);
    w1_check("event-elapsed", !et && ms >= 0, "\"status\":%d,\"v_ms\":%.4f", et, ms);
    w1_check("stream-query-idle", CU(cuStreamQuery, s) == CUDA_SUCCESS, "\"x\":0");

    /* cross-stream ordering through an event */
    CUstream s2 = NULL;
    r = CU(cuStreamCreate, &s2, CU_STREAM_NON_BLOCKING);
    r |= CU(cuMemsetD32, payload, 0, N);
    r |= CU(cuCtxSynchronize);
    unsigned seed = 77;
    void *fill_params[] = {&payload, &n, &seed};
    r |= launch1d(f_fill, N, fill_params, s);
    r |= CU(cuEventRecord, e1, s);
    r |= CU(cuStreamWaitEvent, s2, e1, 0);
    r |= CU(cuMemcpyDtoHAsync, p, payload, N * 4, s2);
    r |= CU(cuStreamSynchronize, s2);
    bad = 0;
    for (unsigned i = 0; i < N; i++)
        bad += p[i] != w1_ref_fill(i, seed);
    w1_check("stream-wait-event", !r && !bad, "\"status\":%d,\"mismatches\":%u", r, bad);

    CU(cuStreamDestroy, s2);
    CU(cuMemFree, x);
    w1_check("stream-teardown",
             !CU(cuMemFreeHost, pinned) && !CU(cuEventDestroy, e0) && !CU(cuEventDestroy, e1) &&
                 !CU(cuStreamDestroy, s),
             "\"x\":0");
}

static void variant_paths(CUdeviceptr whole)
{
    CUdeviceptr payload = whole + GUARD_WORDS * 4;
    W1_RESOLVE(cuMemcpyHtoDAsync_ptsz, "cuMemcpyHtoDAsync_v2_ptsz");
    W1_RESOLVE(cuLaunchKernel_ptsz, "cuLaunchKernel_ptsz");
    W1_RESOLVE(cuStreamSynchronize_ptsz, "cuStreamSynchronize_ptsz");
    W1_RESOLVE(cuGetProcAddress, "cuGetProcAddress");
    W1_RESOLVE(cuGetProcAddress_v2, "cuGetProcAddress_v2");

    /* per-thread default stream variants */
    unsigned n = N, seed = 99;
    void *params[] = {&payload, &n, &seed};
    CUresult l = CU(cuLaunchKernel_ptsz, f_fill, N / 256, 1, 1, 256, 1, 1, 0, NULL, params, NULL);
    CUresult s = CU(cuStreamSynchronize_ptsz, NULL);
    CUresult c = CU(cuMemcpyDtoH, host_b, payload, N * 4);
    unsigned bad = 0;
    for (unsigned i = 0; i < N; i++)
        bad += host_b[i] != w1_ref_fill(i, seed);
    w1_check("ptsz-launch", !l && !s && !c && !bad, "\"launch\":%d,\"sync\":%d,\"mismatches\":%u", l, s, bad);
    for (unsigned i = 0; i < N; i++)
        host_a[i] = ~i;
    CUresult h = CU(cuMemcpyHtoDAsync_ptsz, payload, host_a, N * 4, CU_STREAM_PER_THREAD);
    s = CU(cuStreamSynchronize, CU_STREAM_PER_THREAD);
    c = CU(cuMemcpyDtoH, host_b, payload, N * 4);
    w1_check("ptsz-copy", !h && !s && !c && !memcmp(host_a, host_b, N * 4), "\"copy\":%d,\"sync\":%d", h, s);

    /* get-proc: the returned pointer must behave like the export */
    void *fn = NULL;
    int found = -1;
    CUresult g1 = CU(cuGetProcAddress, "cuDriverGetVersion", &fn, 12000, 0);
    int v_export = 0, v_proc = 0;
    CU(cuDriverGetVersion, &v_export);
    if (!g1 && fn)
        ((PFN_cuDriverGetVersion)fn)(&v_proc);
    w1_emit("\"event\":\"get-proc\",\"entry\":\"cuGetProcAddress\",\"status\":%d,\"same_result\":%s", g1,
            (!g1 && fn && v_proc == v_export) ? "true" : "false");
    fn = NULL;
    CUresult g2 = CU(cuGetProcAddress_v2, "cuMemAlloc", &fn, 12000, 0, &found);
    CUdeviceptr probe = 0;
    CUresult pr = (!g2 && fn) ? ((PFN_cuMemAlloc)fn)(&probe, 4096) : W1_MISSING;
    if (!pr)
        CU(cuMemFree, probe);
    w1_emit("\"event\":\"get-proc\",\"entry\":\"cuGetProcAddress_v2\",\"status\":%d,\"found\":%d,\"call\":%d", g2,
            found, pr);
    fn = NULL;
    CUresult g3 = CU(cuGetProcAddress_v2, "cuNoSuchEntryW1", &fn, 12000, 0, &found);
    w1_check("get-proc-unknown", g3 != CUDA_SUCCESS || fn == NULL, "\"status\":%d,\"found\":%d", g3, found);
}

int main(void)
{
    w1_start("realgpu", 300000);
    if (!w1_load_nvcuda()) {
        w1_check("nvcuda-load", 0, "\"x\":0");
        return 2;
    }
    w1_resolve_common();
    CUdevice dev;
    CUcontext ctx;
    if (w1_bring_up(&dev, &ctx))
        return w1_finish("realgpu");
    size_t free0 = 0, total = 0;
    CU(cuMemGetInfo, &free0, &total);
    CUmodule mod = NULL;
    CUresult r = CU(cuModuleLoadData, &mod, w1_ptx);
    w1_check("cuModuleLoadData", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (!r) {
        kernels(mod);
        CUdeviceptr whole = 0;
        r = CU(cuMemAlloc, &whole, (N + 2 * GUARD_WORDS) * 4);
        w1_check("cuMemAlloc", r == CUDA_SUCCESS, "\"status\":%d", r);
        if (!r) {
            sync_paths(whole);
            stream_paths(whole);
            variant_paths(whole);
            CU(cuMemFree, whole);
        }
        CUfunction none = NULL;
        w1_emit("\"event\":\"error-contract\",\"entry\":\"cuModuleGetFunction(missing)\",\"status\":%d",
                CU(cuModuleGetFunction, &none, mod, "w1_missing"));
        w1_check("cuModuleUnload", CU(cuModuleUnload, mod) == CUDA_SUCCESS, "\"x\":0");
    }
    /* error contracts: these must fail, with the status the control returns */
    CUdeviceptr nothing = 0;
    w1_emit("\"event\":\"error-contract\",\"entry\":\"cuMemFree(0)\",\"status\":%d", CU(cuMemFree, nothing));
    CU(cuCtxSynchronize);
    size_t free1 = 0;
    CU(cuMemGetInfo, &free1, &total);
    /* Allocator granularity may keep a little; 4 MiB tolerance, printed exactly. */
    w1_check("no-leak", free1 + (4u << 20) >= free0, "\"v_free_before\":%llu,\"v_free_after\":%llu",
             (unsigned long long)free0, (unsigned long long)free1);
    w1_check("cuCtxDestroy", CU(cuCtxDestroy, ctx) == CUDA_SUCCESS, "\"x\":0");
    return w1_finish("realgpu");
}

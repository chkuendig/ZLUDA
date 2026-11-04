/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Plan §4.8 retired-record churn workload (x64 and PE32).
 *
 *   churn.exe [--cycles N] [--threads T] [--keys K] [--ctx-every M]
 *
 * T threads run N cycles in total of context-storage put / overwrite / get /
 * remove across K keys and three long-lived contexts (primary + two created).
 * Every M cycles a thread also creates a short-lived context, stores and
 * overwrites into it, and destroys it. A sampler emits process private bytes
 * once per second ("sample" lines) and the fixture prints the private-byte
 * growth per million cycles over the steady part (after 10 % of the cycles).
 * The bridge's live/retired record counters are read by the diagnostic build's
 * collector from outside; this process only generates the load and the
 * private-bytes series. Defaults: 1,000,000 cycles, 4 threads, 16 keys, 1000.
 */
#include "w1_dark.h"
#include <psapi.h>

static struct w1_cls cls;
static CUcontext ctxs[3];
static unsigned cycles = 1000000, threads = 4, keys = 16, ctx_every = 1000;
static volatile LONG done_cycles, failures_put, failures_get, failures_del, failures_ctx;
static volatile LONG stop_sampler, last_get_status;
static char key_storage[64];
static CUdevice dev;

static unsigned long long private_bytes(void)
{
    PROCESS_MEMORY_COUNTERS_EX pmc = {.cb = sizeof pmc};
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof pmc);
    return pmc.PrivateUsage;
}

static DWORD WINAPI sampler(void *unused)
{
    (void)unused;
    while (!stop_sampler) {
        w1_emit("\"event\":\"sample\",\"v_cycles\":%ld,\"v_private_bytes\":%llu", done_cycles, private_bytes());
        Sleep(1000);
    }
    return 0;
}

static DWORD WINAPI worker(void *arg)
{
    unsigned id = (unsigned)(uintptr_t)arg, share = cycles / threads;
    CU(cuCtxSetCurrent, ctxs[0]);
    for (unsigned i = 0; i < share; i++) {
        CUcontext ctx = ctxs[(i + id) % 3];
        void *key = &key_storage[(i * 7 + id) % keys];
        void *v1 = (void *)(uintptr_t)(0x1000 + i), *v2 = (void *)(uintptr_t)(0x2000 + i), *got = NULL;
        if (cls.put(ctx, key, v1, NULL))
            InterlockedIncrement(&failures_put);
        if (i & 1 && cls.put(ctx, key, v2, NULL))
            InterlockedIncrement(&failures_put);
        /* another thread may have removed this key meanwhile: a miss is
         * legitimate here, so only the miss count and status are reported */
        CUresult g = cls.get(&got, ctx, key);
        if (g) {
            InterlockedIncrement(&failures_get);
            last_get_status = g;
        }
        if (i % 3 == 0)
            cls.del(ctx, key);
        if (ctx_every && i % ctx_every == ctx_every - 1) {
            CUcontext tmp = NULL;
            CUresult r = CU(cuCtxCreate, &tmp, 0, dev);
            r = r ? r : cls.put(tmp, key, v1, NULL);
            r = r ? r : cls.put(tmp, key, v2, NULL);
            r = r ? r : cls.del(tmp, key);
            if (tmp && CU(cuCtxDestroy, tmp))
                r = r ? r : -2;
            if (r)
                InterlockedIncrement(&failures_ctx);
            CU(cuCtxSetCurrent, ctxs[0]);
        }
        InterlockedIncrement(&done_cycles);
    }
    return 0;
}

int main(int argc, char **argv)
{
    for (int i = 1; i + 1 < argc; i += 2) {
        unsigned v = (unsigned)strtoul(argv[i + 1], NULL, 10);
        if (!strcmp(argv[i], "--cycles"))
            cycles = v;
        else if (!strcmp(argv[i], "--threads"))
            threads = v ? v : 1;
        else if (!strcmp(argv[i], "--keys"))
            keys = v ? (v > 64 ? 64 : v) : 1;
        else if (!strcmp(argv[i], "--ctx-every"))
            ctx_every = v;
    }
    w1_start("churn", 3600000);
    w1_emit("\"event\":\"config\",\"cycles\":%u,\"threads\":%u,\"keys\":%u,\"ctx_every\":%u", cycles, threads, keys,
            ctx_every);
    if (!w1_load_nvcuda()) {
        w1_check("nvcuda-load", 0, "\"x\":0");
        return 2;
    }
    w1_resolve_common();
    CUresult r = CU(cuInit, 0);
    r = r ? r : CU(cuDeviceGet, &dev, 0);
    r = r ? r : CU(cuDevicePrimaryCtxRetain, &ctxs[0], dev);
    r = r ? r : CU(cuCtxCreate, &ctxs[1], 0, dev);
    r = r ? r : CU(cuCtxCreate, &ctxs[2], 0, dev);
    r = r ? r : CU(cuCtxSetCurrent, ctxs[0]);
    w1_check("contexts", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (r || w1_get_cls(&cls))
        return w1_finish("churn");

    unsigned long long p0 = private_bytes();
    HANDLE s = CreateThread(NULL, 0, sampler, NULL, 0, NULL);
    HANDLE *t = calloc(threads, sizeof *t);
    for (unsigned i = 0; i < threads; i++)
        t[i] = CreateThread(NULL, 0, worker, (void *)(uintptr_t)i, 0, NULL);
    /* steady baseline at 10 % */
    unsigned long long p10 = 0;
    LONG c10 = 0;
    while (done_cycles < (LONG)(cycles / threads * threads / 10))
        Sleep(10);
    p10 = private_bytes();
    c10 = done_cycles;
    WaitForMultipleObjects(threads, t, TRUE, INFINITE);
    unsigned long long p1 = private_bytes();
    LONG c1 = done_cycles;
    stop_sampler = 1;
    WaitForSingleObject(s, 5000);
    /* all keys removed from every long-lived context before teardown */
    for (unsigned c = 0; c < 3; c++)
        for (unsigned k = 0; k < keys; k++)
            cls.del(ctxs[c], &key_storage[k]);
    unsigned long long p2 = private_bytes();
    double per_million = c1 > c10 ? ((double)p1 - (double)p10) * 1e6 / (double)(c1 - c10) : 0;
    w1_emit("\"event\":\"growth\",\"v_private_start\":%llu,\"v_private_10pct\":%llu,\"v_private_end\":%llu,"
            "\"v_private_after_cleanup\":%llu,\"v_bytes_per_million_cycles\":%.1f",
            p0, p10, p1, p2, per_million);
    w1_check("cycles-complete", c1 == (LONG)(cycles / threads * threads), "\"cycles\":%ld", c1);
    w1_check("no-put-failures", failures_put == 0, "\"failures\":%ld", failures_put);
    w1_emit("\"event\":\"racing-get-misses\",\"v_count\":%ld,\"v_last_status\":%ld", failures_get,
            last_get_status);
    w1_check("no-context-churn-failures", failures_ctx == 0, "\"failures\":%ld", failures_ctx);
    CU(cuCtxDestroy, ctxs[2]);
    CU(cuCtxDestroy, ctxs[1]);
    CU(cuDevicePrimaryCtxRelease, dev);
    return w1_finish("churn");
}

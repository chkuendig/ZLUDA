/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * W1.4 / W1.5 D3D11 <-> CUDA interop fixture (x64; PE32 build exists but the
 * PE32 interop route is 801 by contract and is reported, not required).
 *
 *   interop.exe data         Knight D' data conditions with deterministic readbacks
 *   interop.exe contention   barrier-controlled threads, bounded watchdog
 *   interop.exe reentry      host-callback and same-thread re-entry statuses
 *
 * data: RGBA32F 256x256, one mip/array/sample, MT-protected immediate
 * context. Checks D3D->CUDA (first map after registration, READ_ONLY),
 * CUDA->D3D (NONE), WRITE_DISCARD persistence of unwritten rows over primed
 * repeated discard frames, the same CUarray across maps, ptsz mapping, two
 * resources in one map call, and the error/cleanup contracts (double map,
 * unmap unmapped, unregister mapped, foreign context). Every word is compared.
 *
 * contention: T threads (default 4) register concurrently, then run F frames
 * (default 2000) of map / copy / verify / unmap on their own resource; thread
 * 0 additionally registers/unregisters a scratch resource every 50 frames
 * (registration vs frame). Two threads also race on one shared resource:
 * exactly one map may win at a time (others: CUDA_ERROR_ALREADY_MAPPED).
 * Per thread: map wait distribution (QPC), process/thread CPU times.
 * The watchdog bound (default 120 s) is the progress gate.
 *
 * Lease-internal same-thread re-entry (bridge returns 800 instead of C's
 * possible deadlock) cannot be reached from a black-box caller except through
 * a callback; "reentry" covers the callback path and the ordinary same-thread
 * double map. A deeper lease re-entry needs the diagnostic build's hook.
 */
#define COBJMACROS
#define CINTERFACE
#include "w1.h"
#include <initguid.h> /* define IID_ID3D11Multithread here: not in mingw libdxguid */
#include <d3d11_4.h>
#include <dxgi.h>

typedef struct {
    size_t srcXInBytes, srcY;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
} CUDA_MEMCPY2D;
#define CU_MEMORYTYPE_HOST 1
#define CU_MEMORYTYPE_ARRAY 3
#define MAP_NONE 0
#define MAP_READ_ONLY 1
#define MAP_WRITE_DISCARD 2
#define CUDA_ERROR_NOT_MAPPED 211

W1_FN(CUresult, cuD3D11GetDevice, (CUdevice *, IDXGIAdapter *));
W1_FN(CUresult, cuGraphicsD3D11RegisterResource, (CUgraphicsResource *, ID3D11Resource *, unsigned int));
W1_FN(CUresult, cuGraphicsUnregisterResource, (CUgraphicsResource));
W1_FN(CUresult, cuGraphicsResourceSetMapFlags, (CUgraphicsResource, unsigned int));
W1_FN(CUresult, cuGraphicsMapResources, (unsigned int, CUgraphicsResource *, CUstream));
W1_FN(CUresult, cuGraphicsUnmapResources, (unsigned int, CUgraphicsResource *, CUstream));
W1_FN(CUresult, cuGraphicsMapResources_ptsz, (unsigned int, CUgraphicsResource *, CUstream));
W1_FN(CUresult, cuGraphicsUnmapResources_ptsz, (unsigned int, CUgraphicsResource *, CUstream));
W1_FN(CUresult, cuGraphicsSubResourceGetMappedArray, (CUarray *, CUgraphicsResource, unsigned int, unsigned int));
W1_FN(CUresult, cuMemcpy2D, (const CUDA_MEMCPY2D *));
W1_FN(CUresult, cuLaunchHostFunc, (CUstream, void(CUDAAPI *)(void *), void *));

#define W 256u
#define H 256u
#define TEXEL_WORDS 4u
#define ROW_BYTES (W * TEXEL_WORDS * 4u)
#define WORDS (W * H * TEXEL_WORDS)

static ID3D11Device *dev3d;
static ID3D11DeviceContext *imm;
static CUcontext ctx;
static CUdevice cudev;

static void pattern(uint32_t *p, uint32_t seed, unsigned row0, unsigned rows)
{
    for (unsigned y = row0; y < row0 + rows; y++)
        for (unsigned x = 0; x < W * TEXEL_WORDS; x++) {
            /* finite float bit patterns only: exponent bits forced to 0x3f8 range */
            uint32_t v = (y * 2654435761u) ^ (x * 40503u) ^ seed;
            p[y * W * TEXEL_WORDS + x] = 0x3F800000u | (v & 0x007FFFFFu);
        }
}

static unsigned mismatches(const uint32_t *a, const uint32_t *b, unsigned row0, unsigned rows)
{
    unsigned bad = 0;
    for (unsigned i = row0 * W * TEXEL_WORDS; i < (row0 + rows) * W * TEXEL_WORDS; i++)
        bad += a[i] != b[i];
    return bad;
}

static ID3D11Texture2D *texture(D3D11_USAGE usage, UINT bind, UINT cpu)
{
    D3D11_TEXTURE2D_DESC d = {W, H, 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, {1, 0}, usage, bind, cpu, 0};
    ID3D11Texture2D *t = NULL;
    HRESULT hr = ID3D11Device_CreateTexture2D(dev3d, &d, NULL, &t);
    return SUCCEEDED(hr) ? t : NULL;
}

static ID3D11Texture2D *staging;

/* D3D readback of the whole texture through a staging copy */
static int d3d_read(ID3D11Texture2D *tex, uint32_t *out)
{
    ID3D11DeviceContext_CopyResource(imm, (ID3D11Resource *)staging, (ID3D11Resource *)tex);
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ID3D11DeviceContext_Map(imm, (ID3D11Resource *)staging, 0, D3D11_MAP_READ, 0, &m)))
        return -1;
    for (unsigned y = 0; y < H; y++)
        memcpy(out + y * W * TEXEL_WORDS, (char *)m.pData + y * m.RowPitch, ROW_BYTES);
    ID3D11DeviceContext_Unmap(imm, (ID3D11Resource *)staging, 0);
    return 0;
}

static CUresult array_copy(CUarray a, void *host, int to_array, unsigned row0, unsigned rows)
{
    CUDA_MEMCPY2D c = {0};
    c.WidthInBytes = ROW_BYTES;
    c.Height = rows;
    if (to_array) {
        c.srcMemoryType = CU_MEMORYTYPE_HOST, c.srcHost = (char *)host + row0 * ROW_BYTES, c.srcPitch = ROW_BYTES;
        c.dstMemoryType = CU_MEMORYTYPE_ARRAY, c.dstArray = a, c.dstY = row0;
    } else {
        c.srcMemoryType = CU_MEMORYTYPE_ARRAY, c.srcArray = a, c.srcY = row0;
        c.dstMemoryType = CU_MEMORYTYPE_HOST, c.dstHost = (char *)host + row0 * ROW_BYTES, c.dstPitch = ROW_BYTES;
    }
    return CU(cuMemcpy2D, &c);
}

static int setup(void)
{
    W1_RESOLVE(cuD3D11GetDevice, "cuD3D11GetDevice");
    W1_RESOLVE(cuGraphicsD3D11RegisterResource, "cuGraphicsD3D11RegisterResource");
    W1_RESOLVE(cuGraphicsUnregisterResource, "cuGraphicsUnregisterResource");
    W1_RESOLVE(cuGraphicsResourceSetMapFlags, "cuGraphicsResourceSetMapFlags_v2");
    W1_RESOLVE(cuGraphicsMapResources, "cuGraphicsMapResources");
    W1_RESOLVE(cuGraphicsUnmapResources, "cuGraphicsUnmapResources");
    W1_RESOLVE(cuGraphicsMapResources_ptsz, "cuGraphicsMapResources_ptsz");
    W1_RESOLVE(cuGraphicsUnmapResources_ptsz, "cuGraphicsUnmapResources_ptsz");
    W1_RESOLVE(cuGraphicsSubResourceGetMappedArray, "cuGraphicsSubResourceGetMappedArray");
    W1_RESOLVE(cuMemcpy2D, "cuMemcpy2D_v2");
    W1_RESOLVE(cuLaunchHostFunc, "cuLaunchHostFunc");
    /* runtime-loaded: without DXVK/wined3d d3d11 the fixture reports
     * "d3d11-unavailable" (exit 2) instead of failing to start */
    HMODULE d3d11 = LoadLibraryA("d3d11.dll");
    PFN_D3D11_CREATE_DEVICE create = d3d11 ? (PFN_D3D11_CREATE_DEVICE)(void *)GetProcAddress(d3d11, "D3D11CreateDevice") : NULL;
    if (!create) {
        w1_emit("\"event\":\"d3d11-unavailable\",\"v_error\":%lu", (unsigned long)GetLastError());
        return 2;
    }
    D3D_FEATURE_LEVEL fl;
    HRESULT hr = create(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &dev3d, &fl,
                                   &imm);
    if (FAILED(hr)) {
        w1_emit("\"event\":\"d3d11-unavailable\",\"v_hr\":\"0x%08lx\"", (unsigned long)hr);
        return 2;
    }
    ID3D11Multithread *mt = NULL;
    if (SUCCEEDED(ID3D11DeviceContext_QueryInterface(imm, &IID_ID3D11Multithread, (void **)&mt))) {
        ID3D11Multithread_SetMultithreadProtected(mt, TRUE);
        ID3D11Multithread_Release(mt);
    }
    w1_check("d3d11-mt-protected", mt != NULL, "\"x\":0");
    IDXGIDevice *dxgi = NULL;
    IDXGIAdapter *adapter = NULL;
    ID3D11Device_QueryInterface(dev3d, &IID_IDXGIDevice, (void **)&dxgi);
    if (dxgi)
        IDXGIDevice_GetAdapter(dxgi, &adapter);
    CUresult r = CU(cuInit, 0);
    r = r ? r : CU(cuD3D11GetDevice, &cudev, adapter);
    w1_check("cuD3D11GetDevice", r == CUDA_SUCCESS, "\"status\":%d", r);
    r = r ? r : CU(cuCtxCreate, &ctx, 0, cudev);
    w1_check("context", r == CUDA_SUCCESS, "\"status\":%d", r);
    staging = texture(D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);
    return r || !staging ? 1 : 0;
}

static uint32_t want[WORDS], got[WORDS];

static void case_data(void)
{
    ID3D11Texture2D *tex = texture(D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 0);
    ID3D11Texture2D *tex2 = texture(D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 0);
    pattern(want, 0x1111, 0, H);
    ID3D11DeviceContext_UpdateSubresource(imm, (ID3D11Resource *)tex, 0, NULL, want, ROW_BYTES, 0);
    CUgraphicsResource res = NULL, res2 = NULL;
    CUresult r = CU(cuGraphicsD3D11RegisterResource, &res, (ID3D11Resource *)tex, 0);
    w1_check("register", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (r)
        return;
    /* 1. D3D -> CUDA, first map after registration, READ_ONLY */
    CUarray a = NULL, first = NULL;
    r = CU(cuGraphicsResourceSetMapFlags, res, MAP_READ_ONLY);
    r |= CU(cuGraphicsMapResources, 1, &res, NULL);
    r |= CU(cuGraphicsSubResourceGetMappedArray, &a, res, 0, 0);
    first = a;
    r |= array_copy(a, got, 0, 0, H);
    r |= CU(cuGraphicsUnmapResources, 1, &res, NULL);
    w1_check("d3d-to-cuda-first-map", !r && !mismatches(want, got, 0, H), "\"status\":%d,\"mismatches\":%u", r,
             mismatches(want, got, 0, H));
    /* 2. CUDA -> D3D, NONE */
    pattern(want, 0x2222, 0, H);
    r = CU(cuGraphicsResourceSetMapFlags, res, MAP_NONE);
    r |= CU(cuGraphicsMapResources, 1, &res, NULL);
    r |= CU(cuGraphicsSubResourceGetMappedArray, &a, res, 0, 0);
    r |= array_copy(a, want, 1, 0, H);
    r |= CU(cuGraphicsUnmapResources, 1, &res, NULL);
    int rd = d3d_read(tex, got);
    w1_check("cuda-to-d3d", !r && !rd && !mismatches(want, got, 0, H), "\"status\":%d,\"mismatches\":%u", r,
             mismatches(want, got, 0, H));
    w1_check("same-array-across-maps", a == first, "\"x\":0");
    /* 3. WRITE_DISCARD: write the top half only, over primed repeated frames;
     * the unwritten bottom half must survive (D' finding) */
    r = CU(cuGraphicsResourceSetMapFlags, res, MAP_WRITE_DISCARD);
    for (unsigned frame = 0; frame < 4; frame++) {
        pattern(want, 0x3000 + frame, 0, H / 2);
        CUresult fr = CU(cuGraphicsMapResources, 1, &res, NULL);
        fr |= CU(cuGraphicsSubResourceGetMappedArray, &a, res, 0, 0);
        fr |= array_copy(a, want, 1, 0, H / 2);
        fr |= CU(cuGraphicsUnmapResources, 1, &res, NULL);
        rd = d3d_read(tex, got);
        unsigned top = mismatches(want, got, 0, H / 2), bottom = mismatches(want, got, H / 2, H / 2);
        char name[48];
        snprintf(name, sizeof name, "write-discard-frame-%u", frame);
        w1_check(name, !r && !fr && !rd && !top && !bottom,
                 "\"status\":%d,\"top_mismatches\":%u,\"unwritten_mismatches\":%u,\"same_array\":%s", fr, top, bottom,
                 a == first ? "true" : "false");
    }
    /* 4. ptsz map/unmap on the per-thread stream */
    r = CU(cuGraphicsResourceSetMapFlags, res, MAP_NONE);
    r |= CU(cuGraphicsMapResources_ptsz, 1, &res, NULL);
    r |= CU(cuGraphicsSubResourceGetMappedArray, &a, res, 0, 0);
    r |= array_copy(a, got, 0, 0, H);
    r |= CU(cuGraphicsUnmapResources_ptsz, 1, &res, NULL);
    w1_check("ptsz-map", !r && !mismatches(want, got, 0, H), "\"status\":%d", r);
    /* 5. two resources in one call */
    pattern(want, 0x5555, 0, H);
    ID3D11DeviceContext_UpdateSubresource(imm, (ID3D11Resource *)tex2, 0, NULL, want, ROW_BYTES, 0);
    r = CU(cuGraphicsD3D11RegisterResource, &res2, (ID3D11Resource *)tex2, 0);
    CUgraphicsResource both[2] = {res, res2};
    CUarray a2 = NULL;
    r |= CU(cuGraphicsMapResources, 2, both, NULL);
    r |= CU(cuGraphicsSubResourceGetMappedArray, &a2, res2, 0, 0);
    r |= array_copy(a2, got, 0, 0, H);
    r |= CU(cuGraphicsUnmapResources, 2, both, NULL);
    w1_check("two-resources-one-map", !r && !mismatches(want, got, 0, H), "\"status\":%d", r);
    /* 6. error and cleanup contracts */
    CUresult m1 = CU(cuGraphicsMapResources, 1, &res, NULL);
    CUresult m2 = CU(cuGraphicsMapResources, 1, &res, NULL);
    CUresult u1 = CU(cuGraphicsUnmapResources, 1, &res, NULL);
    CUresult u2 = CU(cuGraphicsUnmapResources, 1, &res, NULL);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"double-map\",\"first\":%d,\"second\":%d", m1, m2);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"double-unmap\",\"first\":%d,\"second\":%d", u1, u2);
    w1_check("double-map-refused", !m1 && m2 == CUDA_ERROR_ALREADY_MAPPED, "\"second\":%d", m2);
    w1_check("unmap-unmapped-refused", !u1 && u2 == CUDA_ERROR_NOT_MAPPED, "\"second\":%d", u2);
    CUcontext other = NULL;
    CU(cuCtxCreate, &other, 0, cudev);
    CUresult foreign = CU(cuGraphicsMapResources, 1, &res2, NULL);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"map-from-other-context\",\"status\":%d", foreign);
    if (!foreign)
        CU(cuGraphicsUnmapResources, 1, &res2, NULL);
    CU(cuCtxDestroy, other);
    CU(cuCtxSetCurrent, ctx);
    CUresult mm = CU(cuGraphicsMapResources, 1, &res2, NULL);
    CUresult unreg_mapped = CU(cuGraphicsUnregisterResource, res2);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"unregister-mapped\",\"map\":%d,\"status\":%d", mm,
            unreg_mapped);
    if (unreg_mapped) {
        CU(cuGraphicsUnmapResources, 1, &res2, NULL);
        unreg_mapped = CU(cuGraphicsUnregisterResource, res2);
    }
    w1_check("unregister-all", !unreg_mapped && !CU(cuGraphicsUnregisterResource, res), "\"x\":0");
    ID3D11Texture2D_Release(tex2);
    ID3D11Texture2D_Release(tex);
}

/* ---- contention ---- */

#define MAX_THREADS 16
static unsigned n_threads = 4, frames = 2000;
static volatile LONG arrived, go;
static CUgraphicsResource shared_res;
struct stats {
    unsigned ok, bad_data, failures, shared_wins, shared_busy, reg_cycles, reg_failures;
    double wait_max, wait_sum;
    unsigned hist[8]; /* map wait: <10us <100us <1ms <10ms <100ms <1s <10s >=10s */
};
static struct stats st[MAX_THREADS];

static void barrier(void)
{
    InterlockedIncrement(&arrived);
    while (!go)
        YieldProcessor();
}

static DWORD WINAPI contender(void *arg)
{
    unsigned id = (unsigned)(uintptr_t)arg;
    struct stats *s = &st[id];
    uint32_t *mine = malloc(WORDS * 4), *back = malloc(WORDS * 4);
    CU(cuCtxSetCurrent, ctx);
    ID3D11Texture2D *tex = texture(D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
    barrier(); /* concurrent registration */
    CUgraphicsResource res = NULL;
    if (!tex || CU(cuGraphicsD3D11RegisterResource, &res, (ID3D11Resource *)tex, 0)) {
        s->failures++;
        return 1;
    }
    for (unsigned f = 0; f < frames; f++) {
        pattern(mine, id * 0x10000u + f, 0, 4);
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        CUresult r = CU(cuGraphicsMapResources, 1, &res, NULL);
        QueryPerformanceCounter(&t1);
        double us = (double)(t1.QuadPart - t0.QuadPart) * 1e6 / (double)w1_qpf.QuadPart;
        s->wait_sum += us;
        if (us > s->wait_max)
            s->wait_max = us;
        unsigned b = 0;
        for (double lim = 10; us >= lim && b < 7; lim *= 10)
            b++;
        s->hist[b]++;
        CUarray a = NULL;
        r = r ? r : CU(cuGraphicsSubResourceGetMappedArray, &a, res, 0, 0);
        r = r ? r : array_copy(a, mine, 1, 0, 4);
        r = r ? r : array_copy(a, back, 0, 0, 4);
        CUresult u = CU(cuGraphicsUnmapResources, 1, &res, NULL);
        if (r || u)
            s->failures++;
        else if (mismatches(mine, back, 0, 4))
            s->bad_data++;
        else
            s->ok++;
        if (id < 2) {
            CUresult sm = CU(cuGraphicsMapResources, 1, &shared_res, NULL);
            if (!sm) {
                s->shared_wins++;
                CU(cuGraphicsUnmapResources, 1, &shared_res, NULL);
            } else if (sm == CUDA_ERROR_ALREADY_MAPPED) {
                s->shared_busy++;
            } else {
                s->failures++;
            }
        }
        if (id == 0 && f % 50 == 0) {
            ID3D11Texture2D *scratch = texture(D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
            CUgraphicsResource sr = NULL;
            CUresult rr = CU(cuGraphicsD3D11RegisterResource, &sr, (ID3D11Resource *)scratch, 0);
            rr = rr ? rr : CU(cuGraphicsUnregisterResource, sr);
            s->reg_cycles++;
            s->reg_failures += rr != 0;
            if (scratch)
                ID3D11Texture2D_Release(scratch);
        }
    }
    CU(cuGraphicsUnregisterResource, res);
    ID3D11Texture2D_Release(tex);
    free(mine);
    free(back);
    return 0;
}

static double filetime_ms(FILETIME f)
{
    return (double)(((unsigned long long)f.dwHighDateTime << 32) | f.dwLowDateTime) / 10000.0;
}

static void case_contention(void)
{
    ID3D11Texture2D *shared = texture(D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
    CUresult r = CU(cuGraphicsD3D11RegisterResource, &shared_res, (ID3D11Resource *)shared, 0);
    w1_check("register-shared", r == CUDA_SUCCESS, "\"status\":%d", r);
    HANDLE t[MAX_THREADS];
    for (unsigned i = 0; i < n_threads; i++)
        t[i] = CreateThread(NULL, 0, contender, (void *)(uintptr_t)i, 0, NULL);
    while (arrived < (LONG)n_threads)
        Sleep(1);
    double start = w1_ms();
    go = 1;
    WaitForMultipleObjects(n_threads, t, TRUE, INFINITE);
    double wall = w1_ms() - start;
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    w1_emit("\"event\":\"contention-process\",\"threads\":%u,\"frames\":%u,\"v_wall_ms\":%.1f,\"v_cpu_user_ms\":%.1f,"
            "\"v_cpu_kernel_ms\":%.1f",
            n_threads, frames, wall, filetime_ms(u), filetime_ms(k));
    unsigned total_ok = 0, wins = 0;
    for (unsigned i = 0; i < n_threads; i++) {
        struct stats *s = &st[i];
        GetThreadTimes(t[i], &c, &e, &k, &u);
        w1_emit("\"event\":\"contention-thread\",\"thread\":%u,\"ok\":%u,\"bad_data\":%u,\"failures\":%u,"
                "\"reg_failures\":%u,\"v_reg_cycles\":%u,\"v_shared_wins\":%u,\"v_shared_busy\":%u,"
                "\"v_wait_mean_us\":%.2f,\"v_wait_max_us\":%.1f,\"v_hist\":[%u,%u,%u,%u,%u,%u,%u,%u],"
                "\"v_cpu_user_ms\":%.1f,\"v_cpu_kernel_ms\":%.1f",
                i, s->ok, s->bad_data, s->failures, s->reg_failures, s->reg_cycles, s->shared_wins, s->shared_busy,
                frames ? s->wait_sum / frames : 0, s->wait_max, s->hist[0], s->hist[1], s->hist[2], s->hist[3],
                s->hist[4], s->hist[5], s->hist[6], s->hist[7], filetime_ms(u), filetime_ms(k));
        total_ok += s->ok;
        wins += s->shared_wins;
    }
    w1_check("contention-all-frames-correct", total_ok == n_threads * frames, "\"ok\":%u,\"want\":%u", total_ok,
             n_threads * frames);
    w1_check("shared-resource-progress", wins > 0, "\"x\":0");
    CU(cuGraphicsUnregisterResource, shared_res);
    ID3D11Texture2D_Release(shared);
}

/* ---- reentry ---- */

static CUgraphicsResource reentry_res;
static volatile LONG cb_status = -2, cb_done;

static void CUDAAPI host_cb(void *unused)
{
    (void)unused;
    cb_status = CU(cuGraphicsMapResources, 1, &reentry_res, NULL);
    if (!cb_status)
        CU(cuGraphicsUnmapResources, 1, &reentry_res, NULL);
    cb_done = 1;
}

static void case_reentry(void)
{
    ID3D11Texture2D *tex = texture(D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
    CUresult r = CU(cuGraphicsD3D11RegisterResource, &reentry_res, (ID3D11Resource *)tex, 0);
    w1_check("register", r == CUDA_SUCCESS, "\"status\":%d", r);
    CUstream s = NULL;
    CU(cuStreamCreate, &s, 0);
    /* the callback runs while this thread holds the mapping on stream s */
    r = CU(cuGraphicsMapResources, 1, &reentry_res, s);
    w1_emit("\"event\":\"marker\",\"marker\":\"mapped\",\"status\":%d", r);
    CUresult l = CU(cuLaunchHostFunc, s, host_cb, NULL);
    CUresult sy = CU(cuStreamSynchronize, s);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"host-callback-map\",\"launch\":%d,\"sync\":%d,\"done\":%ld,"
            "\"status\":%ld",
            l, sy, cb_done, cb_status);
    CUresult same = CU(cuGraphicsMapResources, 1, &reentry_res, s);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"same-thread-map-while-mapped\",\"status\":%d", same);
    CU(cuGraphicsUnmapResources, 1, &reentry_res, s);
    CU(cuStreamSynchronize, s);
    w1_check("reentry-no-hang", cb_done || l, "\"x\":0");
    CU(cuGraphicsUnregisterResource, reentry_res);
    CU(cuStreamDestroy, s);
    ID3D11Texture2D_Release(tex);
}

int main(int argc, char **argv)
{
    const char *which = argc > 1 ? argv[1] : "data";
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--threads"))
            n_threads = (unsigned)strtoul(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--frames"))
            frames = (unsigned)strtoul(argv[i + 1], NULL, 10);
    }
    if (n_threads < 2)
        n_threads = 2;
    if (n_threads > MAX_THREADS)
        n_threads = MAX_THREADS;
    w1_start("interop", 120000);
    w1_emit("\"event\":\"case\",\"case\":\"%s\"", which);
    if (!w1_load_nvcuda()) {
        w1_check("nvcuda-load", 0, "\"x\":0");
        return 2;
    }
    w1_resolve_common();
    int s = setup();
    if (s) {
        w1_finish("interop");
        return s == 2 ? 2 : 1;
    }
    if (!strcmp(which, "data"))
        case_data();
    else if (!strcmp(which, "contention"))
        case_contention();
    else if (!strcmp(which, "reentry"))
        case_reentry();
    else
        w1_check("known-case", 0, "\"case\":\"%s\"", which);
    return w1_finish("interop");
}

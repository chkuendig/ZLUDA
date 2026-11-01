/* SPDX-License-Identifier: MIT OR Apache-2.0
 * CUDA driver probe for the zluda32 -> zluda64_server -> nvcuda chain.
 * Built for i686 (through zluda32) and x86_64 (direct to nvcuda) Windows.
 * Prints one JSON record per line on stdout.
 *
 *   probe.exe [iterations]
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
#if defined(_WIN64)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

#define API __stdcall
static CUresult (API *p_cuInit)(unsigned);
static CUresult (API *p_cuDriverGetVersion)(int *);
static CUresult (API *p_cuDeviceGetCount)(int *);
static CUresult (API *p_cuDeviceGet)(CUdevice *, int);
static CUresult (API *p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (API *p_cuDeviceTotalMem_v2)(size_t *, CUdevice);
static CUresult (API *p_cuDeviceComputeCapability)(int *, int *, CUdevice);
static CUresult (API *p_cuDeviceGetLuid)(char *, unsigned *, CUdevice);
static CUresult (API *p_cuCtxPushCurrent_v2)(CUcontext);
static CUresult (API *p_cuCtxPopCurrent_v2)(CUcontext *);
static CUresult (API *p_cuCtxGetCurrent)(CUcontext *);
static CUresult (API *p_cuCtxGetApiVersion)(CUcontext, unsigned *);
static CUresult (API *p_cuModuleGetTexRef)(void **, CUmodule, const char *);
static CUresult (API *p_cuTexRefSetAddressMode)(void *, int, int);
static CUresult (API *p_cuTexRefSetFilterMode)(void *, int);
static CUresult (API *p_cuTexRefSetMaxAnisotropy)(void *, unsigned);
static CUresult (API *p_cuTexRefSetMipmapFilterMode)(void *, int);
static CUresult (API *p_cuTexRefSetMipmapLevelBias)(void *, float);
static CUresult (API *p_cuTexRefSetMipmapLevelClamp)(void *, float, float);
static CUresult (API *p_cuCtxCreate_v2)(CUcontext *, unsigned, CUdevice);
static CUresult (API *p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (API *p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (API *p_cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (API *p_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
static CUresult (API *p_cuMemcpyDtoHAsync_v2)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (API *p_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                        unsigned, CUstream, void **, void **);

static const char ptx[] =
    ".version 6.0\n.target sm_60\n.address_size 32\n"
    ".global .align 4 .b8 g_stub[16];\n"
    ".visible .entry k(.param .u32 a, .param .u64 b) { ret; }\n";

static LARGE_INTEGER freq;
static double now_us(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e6 / (double)freq.QuadPart;
}

static void step(const char *name, CUresult r, double us)
{
    printf("{\"step\":\"%s\",\"result\":%d,\"us\":%.1f}\n", name, r, us);
    fflush(stdout);
}

#define LOAD(f) (*(FARPROC *)&p_##f = GetProcAddress(lib, #f))

static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* Times `iters` calls of one operation; reports mean, median and p99 per call. */
#define TIME(label, call)                                                                    \
    do {                                                                                     \
        double *s = malloc(sizeof(double) * iters), total = 0;                               \
        CUresult r = 0;                                                                      \
        for (int i = 0; i < iters && r == 0; i++) {                                          \
            double t0 = now_us();                                                            \
            r = (call);                                                                      \
            s[i] = now_us() - t0;                                                            \
            total += s[i];                                                                   \
        }                                                                                    \
        qsort(s, iters, sizeof(double), compare_double);                                     \
        printf("{\"timing\":\"%s\",\"result\":%d,\"iterations\":%d,\"mean_us\":%.2f,"        \
               "\"median_us\":%.2f,\"p99_us\":%.2f}\n",                                      \
               label, r, iters, total / iters, s[iters / 2], s[iters * 99 / 100]);           \
        fflush(stdout);                                                                      \
        free(s);                                                                             \
    } while (0)

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    QueryPerformanceFrequency(&freq);
    printf("{\"probe\":\"start\",\"bits\":%d}\n", (int)sizeof(void *) * 8);
    fflush(stdout);

    double t0 = now_us();
    HMODULE lib = LoadLibraryA("nvcuda.dll");
    step("LoadLibrary", lib ? 0 : (int)GetLastError(), now_us() - t0);
    if (!lib)
        return 2;
    LOAD(cuInit); LOAD(cuDriverGetVersion); LOAD(cuDeviceGetCount); LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName); LOAD(cuDeviceTotalMem_v2); LOAD(cuDeviceComputeCapability); LOAD(cuDeviceGetLuid);
    LOAD(cuCtxPushCurrent_v2); LOAD(cuCtxPopCurrent_v2); LOAD(cuCtxGetCurrent); LOAD(cuCtxGetApiVersion);
    LOAD(cuModuleGetTexRef); LOAD(cuTexRefSetAddressMode); LOAD(cuTexRefSetFilterMode);
    LOAD(cuTexRefSetMaxAnisotropy); LOAD(cuTexRefSetMipmapFilterMode); LOAD(cuTexRefSetMipmapLevelBias);
    LOAD(cuTexRefSetMipmapLevelClamp);
    const char *mode = argc > 2 ? argv[2] : "";
    LOAD(cuCtxCreate_v2); LOAD(cuModuleLoadData); LOAD(cuModuleGetFunction); LOAD(cuMemAlloc_v2);
    LOAD(cuMemcpyHtoD_v2); LOAD(cuMemcpyDtoHAsync_v2); LOAD(cuLaunchKernel);
    /* A MinGW PE32 build can export stdcall names decorated (cuInit@4). */
    if (!p_cuInit || !p_cuDriverGetVersion || !p_cuDeviceGetCount || !p_cuDeviceGet || !p_cuDeviceGetName ||
        !p_cuDeviceTotalMem_v2 || !p_cuDeviceComputeCapability || !p_cuCtxCreate_v2 || !p_cuModuleLoadData ||
        !p_cuModuleGetFunction || !p_cuMemAlloc_v2 || !p_cuMemcpyHtoD_v2 || !p_cuMemcpyDtoHAsync_v2 ||
        !p_cuLaunchKernel) {
        step("GetProcAddress", -1, 0);
        return 4;
    }

    /* The first call starts zluda64_server under zluda32. */
    t0 = now_us();
    CUresult r = p_cuInit(0);
    step("cuInit", r, now_us() - t0);
    if (r)
        return 3;

    int version = 0, count = 0, major = 0, minor = 0;
    CUdevice dev = -1;
    char name[64] = "";
    size_t mem = 0;
    t0 = now_us(); r = p_cuDriverGetVersion(&version); step("cuDriverGetVersion", r, now_us() - t0);
    t0 = now_us(); r = p_cuDeviceGetCount(&count); step("cuDeviceGetCount", r, now_us() - t0);
    t0 = now_us(); r = p_cuDeviceGet(&dev, 0); step("cuDeviceGet", r, now_us() - t0);
    t0 = now_us(); r = p_cuDeviceGetName(name, sizeof(name), dev); step("cuDeviceGetName", r, now_us() - t0);
    t0 = now_us(); r = p_cuDeviceTotalMem_v2(&mem, dev); step("cuDeviceTotalMem_v2", r, now_us() - t0);
    t0 = now_us(); r = p_cuDeviceComputeCapability(&major, &minor, dev); step("cuDeviceComputeCapability", r, now_us() - t0);
    {
        unsigned char luid[8] = {0};
        unsigned mask = 0;
        r = p_cuDeviceGetLuid ? p_cuDeviceGetLuid((char *)luid, &mask, dev) : -1;
        printf("{\"luid\":{\"result\":%d,\"bytes\":\"%02x%02x%02x%02x%02x%02x%02x%02x\",\"node_mask\":%u}}\n", r,
               luid[0], luid[1], luid[2], luid[3], luid[4], luid[5], luid[6], luid[7], mask);
    }
    printf("{\"device\":{\"version\":%d,\"count\":%d,\"ordinal\":%d,\"name\":\"%s\",\"mem\":%llu,\"cc\":\"%d.%d\"}}\n",
           version, count, dev, name, (unsigned long long)mem, major, minor);
    fflush(stdout);

    CUcontext ctx = NULL;
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    CUdeviceptr dptr = 0;
    t0 = now_us(); r = p_cuCtxCreate_v2(&ctx, 0, dev); step("cuCtxCreate_v2", r, now_us() - t0);
    t0 = now_us(); r = p_cuModuleLoadData(&mod, ptx); step("cuModuleLoadData", r, now_us() - t0);
    t0 = now_us(); r = p_cuModuleGetFunction(&fn, mod, "k"); step("cuModuleGetFunction", r, now_us() - t0);
    t0 = now_us(); r = p_cuMemAlloc_v2(&dptr, 1 << 20); step("cuMemAlloc_v2", r, now_us() - t0);

    /* Mode "death": the driver dies inside cuLaunchKernel; later calls must fail fast. */
    if (!strcmp(mode, "death")) {
        unsigned char a[16] = {0};
        size_t asz = sizeof(a);
        void *ex[5] = {(void *)1, a, (void *)2, &asz, (void *)0};
        int v = 0;
        t0 = now_us(); r = p_cuLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, NULL, NULL, ex); step("death_cuLaunchKernel", r, now_us() - t0);
        t0 = now_us(); r = p_cuDriverGetVersion(&v); step("death_cuDriverGetVersion", r, now_us() - t0);
        printf("{\"probe\":\"finish\"}\n");
        return 0;
    }

    if (p_cuCtxPushCurrent_v2 && p_cuCtxPopCurrent_v2 && sizeof(void *) == 4) {
        /* cuCtxCreate_v2 made ctx current: pop it, pop an empty stack, push it back. */
        CUcontext popped = NULL, cur = NULL;
        r = p_cuCtxPopCurrent_v2(&popped);
        printf("{\"check\":\"pop_created\",\"result\":%d,\"same\":%d}\n", r, popped == ctx);
        r = p_cuCtxPopCurrent_v2(&popped);
        printf("{\"check\":\"pop_empty\",\"result\":%d}\n", r);
        r = p_cuCtxPushCurrent_v2(ctx);
        p_cuCtxGetCurrent(&cur);
        printf("{\"check\":\"push_current\",\"result\":%d,\"same\":%d}\n", r, cur == ctx);
        r = p_cuCtxPushCurrent_v2(NULL);
        printf("{\"check\":\"push_null\",\"result\":%d}\n", r);
        r = p_cuCtxPushCurrent_v2((CUcontext)0x7777);
        printf("{\"check\":\"push_unknown\",\"result\":%d}\n", r);
        fflush(stdout);
    }
    if (p_cuModuleGetTexRef && sizeof(void *) == 4) {
        /* Each setter: value A, A again (must not reach the driver), then B. */
        void *tex = NULL;
        r = p_cuModuleGetTexRef(&tex, mod, "tex");
        int rs[6][3];
        rs[0][0] = p_cuTexRefSetAddressMode(tex, 1, 2); rs[0][1] = p_cuTexRefSetAddressMode(tex, 1, 2); rs[0][2] = p_cuTexRefSetAddressMode(tex, 0, 3);
        rs[1][0] = p_cuTexRefSetFilterMode(tex, 1); rs[1][1] = p_cuTexRefSetFilterMode(tex, 1); rs[1][2] = p_cuTexRefSetFilterMode(tex, 0);
        rs[2][0] = p_cuTexRefSetMaxAnisotropy(tex, 4); rs[2][1] = p_cuTexRefSetMaxAnisotropy(tex, 4); rs[2][2] = p_cuTexRefSetMaxAnisotropy(tex, 8);
        rs[3][0] = p_cuTexRefSetMipmapFilterMode(tex, 1); rs[3][1] = p_cuTexRefSetMipmapFilterMode(tex, 1); rs[3][2] = p_cuTexRefSetMipmapFilterMode(tex, 0);
        rs[4][0] = p_cuTexRefSetMipmapLevelBias(tex, 0.5f); rs[4][1] = p_cuTexRefSetMipmapLevelBias(tex, 0.5f); rs[4][2] = p_cuTexRefSetMipmapLevelBias(tex, -1.0f);
        rs[5][0] = p_cuTexRefSetMipmapLevelClamp(tex, 0.0f, 8.0f); rs[5][1] = p_cuTexRefSetMipmapLevelClamp(tex, 0.0f, 8.0f); rs[5][2] = p_cuTexRefSetMipmapLevelClamp(tex, 1.0f, 4.0f);
        int bad = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 3; j++)
                bad |= rs[i][j];
        printf("{\"check\":\"texref_setters\",\"gettexref\":%d,\"any_error\":%d}\n", r, bad);
        fflush(stdout);
    }
    if (iters <= 0)
        return 0;

    static char host[256 * 1024];
    unsigned char args[16] = {0};
    size_t args_size = sizeof(args);
    void *extra[5] = {(void *)1, args, (void *)2, &args_size, (void *)0};
    TIME("cuDriverGetVersion", p_cuDriverGetVersion(&version));
    if (p_cuCtxGetApiVersion) {
        unsigned api = 0;
        TIME("cuCtxGetApiVersion", p_cuCtxGetApiVersion(ctx, &api));
        printf("{\"check\":\"api_version\",\"value\":%u}\n", api);
    }
    TIME("cuMemcpyHtoD_v2_4KiB", p_cuMemcpyHtoD_v2(dptr, host, 4096));
    TIME("cuMemcpyHtoD_v2_256KiB", p_cuMemcpyHtoD_v2(dptr, host, sizeof(host)));
    TIME("cuMemcpyDtoHAsync_v2_4KiB", p_cuMemcpyDtoHAsync_v2(host, dptr, 4096, NULL));
    if (fn)
        TIME("cuLaunchKernel_2args", p_cuLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, NULL, NULL, extra));
    printf("{\"probe\":\"finish\"}\n");
    return 0;
}

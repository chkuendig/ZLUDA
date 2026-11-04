/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * W1.0 / W1.6 identity fixture (x64 and PE32), real backend.
 *
 *   identity.exe --expect on  [--exports N]   ZLUDA root selected: nvcuda exports
 *                                              N names, CUDA init/device/context/
 *                                              alloc/module/kernel all succeed
 *   identity.exe --expect off                 GE empty stub selected: nvcuda loads
 *                                              but exports no CUDA entry points
 *
 * The process loads nothing else itself. The external harness pairs this
 * output with /proc/<pid>/maps of the Wine process (ZLUDA/HIP/ROCr present
 * on, absent off), prefix hashes and WINEDLLPATH. The fixture prints its own
 * pid and waits for W1_HOLD_MS (default 0) before teardown so the harness can
 * snapshot maps while the backend is live.
 */
#include "w1.h"
#include "w1_ptx.h"
#include <psapi.h>

static void list_modules(void)
{
    HMODULE modules[512];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), modules, sizeof modules, &needed))
        return;
    unsigned count = needed / sizeof(HMODULE);
    for (unsigned i = 0; i < count && i < 512; i++) {
        char name[MAX_PATH] = {0};
        GetModuleBaseNameA(GetCurrentProcess(), modules[i], name, sizeof name);
        w1_emit("\"event\":\"pe-module\",\"v_name\":\"%s\"", name);
    }
}

static void real_path(unsigned expected_exports)
{
    w1_resolve_common();
    if (expected_exports) {
        const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)w1_nvcuda;
        const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const char *)w1_nvcuda + dos->e_lfanew);
        const IMAGE_EXPORT_DIRECTORY *exp = (const IMAGE_EXPORT_DIRECTORY *)(
            (const char *)w1_nvcuda +
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress);
        w1_check("export-count", exp->NumberOfNames == expected_exports, "\"expected\":%u,\"actual\":%lu",
                 expected_exports, (unsigned long)exp->NumberOfNames);
    }
    int version = 0;
    CUresult r = CU(cuDriverGetVersion, &version);
    w1_check("cuDriverGetVersion", r == CUDA_SUCCESS && version > 0, "\"status\":%d,\"version\":%d", r, version);

    CUdevice dev;
    CUcontext ctx;
    if (w1_bring_up(&dev, &ctx))
        return;
    char name[256] = {0};
    size_t total = 0;
    int major = 0, minor = 0, sms = 0;
    r = CU(cuDeviceGetName, name, sizeof name, dev);
    w1_check("cuDeviceGetName", r == CUDA_SUCCESS && name[0], "\"status\":%d,\"v_name\":\"%s\"", r, name);
    r = CU(cuDeviceTotalMem, &total, dev);
    w1_check("cuDeviceTotalMem", r == CUDA_SUCCESS && total > 0, "\"status\":%d,\"v_bytes\":%llu", r,
             (unsigned long long)total);
    CU(cuDeviceGetAttribute, &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    CU(cuDeviceGetAttribute, &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    CU(cuDeviceGetAttribute, &sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    w1_emit("\"event\":\"device\",\"v_cc\":\"%d.%d\",\"v_sms\":%d", major, minor, sms);

    enum { N = 4096 };
    CUdeviceptr buf = 0;
    r = CU(cuMemAlloc, &buf, N * sizeof(uint32_t));
    w1_check("cuMemAlloc", r == CUDA_SUCCESS && buf, "\"status\":%d,\"v_ptr\":\"0x%llx\"", r,
             (unsigned long long)buf);
    CUmodule mod = NULL;
    CUfunction fill = NULL;
    r = CU(cuModuleLoadData, &mod, w1_ptx);
    w1_check("cuModuleLoadData", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (!r) {
        r = CU(cuModuleGetFunction, &fill, mod, "w1_fill");
        w1_check("cuModuleGetFunction", r == CUDA_SUCCESS, "\"status\":%d", r);
    }
    if (fill && buf) {
        unsigned n = N, seed = 0x1234567u;
        void *params[] = {&buf, &n, &seed};
        r = CU(cuLaunchKernel, fill, N / 256, 1, 1, 256, 1, 1, 0, NULL, params, NULL);
        CUresult s = CU(cuCtxSynchronize);
        static uint32_t host[N];
        CUresult c = CU(cuMemcpyDtoH, host, buf, sizeof host);
        unsigned bad = 0;
        for (unsigned i = 0; i < N; i++)
            bad += host[i] != w1_ref_fill(i, seed);
        w1_check("kernel-fill", !r && !s && !c && !bad, "\"launch\":%d,\"sync\":%d,\"copy\":%d,\"mismatches\":%u",
                 r, s, c, bad);
    }
    list_modules();
    const char *hold = getenv("W1_HOLD_MS");
    if (hold) {
        w1_emit("\"event\":\"hold\",\"v_ms\":%s", hold);
        Sleep((DWORD)strtoul(hold, NULL, 10));
    }
    if (mod)
        w1_check("cuModuleUnload", CU(cuModuleUnload, mod) == CUDA_SUCCESS, "\"x\":0");
    if (buf)
        w1_check("cuMemFree", CU(cuMemFree, buf) == CUDA_SUCCESS, "\"x\":0");
    w1_check("cuCtxDestroy", CU(cuCtxDestroy, ctx) == CUDA_SUCCESS, "\"x\":0");
}

int main(int argc, char **argv)
{
    const char *expect = "on";
    unsigned exports = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--expect"))
            expect = argv[i + 1];
        else if (!strcmp(argv[i], "--exports"))
            exports = (unsigned)strtoul(argv[i + 1], NULL, 10);
    }
    w1_start("identity", 120000);
    w1_emit("\"event\":\"expect\",\"mode\":\"%s\"", expect);
    if (!w1_load_nvcuda()) {
        w1_check("nvcuda-load", 0, "\"mode\":\"%s\"", expect);
        return w1_finish("identity");
    }
    if (!strcmp(expect, "off")) {
        /* GE's stub is an empty PE: no CUDA entry point may resolve. */
        int any = w1_sym("cuInit") || w1_sym("cuGetExportTable") || w1_sym("cuDriverGetVersion");
        w1_check("off-no-cuda-exports", !any, "\"cuInit\":%s", w1_sym("cuInit") ? "true" : "false");
        list_modules();
    } else {
        real_path(exports);
    }
    return w1_finish("identity");
}

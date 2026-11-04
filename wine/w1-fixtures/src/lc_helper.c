/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * lc_helper.dll for lifecycle.exe freelibrary|exitprocess. lc_arm()
 * registers a context-storage destructor on the primary context; the helper's
 * DLL_PROCESS_DETACH (FreeLibrary: reserved == NULL; ExitProcess: != NULL)
 * resets the primary context under the loader lock and requires that the
 * destructor ran exactly once, synchronously, on this Windows thread with this
 * thread's TLS value. In FreeLibrary mode a worker created at arm time then
 * resets a second registration while the loader lock is still held; its
 * status is the loader-lock worker contract (reported, compared with the
 * control; the bridge contract expects 801). ExitProcess terminates other
 * threads first, so that probe only exists for FreeLibrary.
 * An external deadline covers detach hangs: the exe's watchdog thread is
 * already gone during ExitProcess detach.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include "w1_dark.h"

static w1_cls_put cls_put;
static PFN_cuDevicePrimaryCtxReset reset;
static CUdevice device;
static DWORD tls_slot;
static int armed;
static volatile LONG calls;
static volatile DWORD dtor_thread;
static void *volatile dtor_tls;
static char key, value, worker_key;
static HANDLE worker, worker_go, worker_done;
static volatile LONG worker_status = -2, worker_dtor_calls;
static volatile DWORD worker_dtor_thread, worker_thread_id;
static PFN_cuDevicePrimaryCtxRetain retain;
static PFN_cuCtxSetCurrent set_current;

static void CUDAAPI worker_dtor(CUcontext ctx, void *k, void *v)
{
    (void)ctx;
    (void)v;
    if (k == &worker_key) {
        InterlockedIncrement(&worker_dtor_calls);
        worker_dtor_thread = GetCurrentThreadId();
    }
}

static void say(const char *fmt, ...)
{
    va_list ap;
    printf("{\"arch\":\"" W1_ARCH "\",\"helper\":true,");
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("}\n");
    fflush(stdout);
}

static void CUDAAPI dtor(CUcontext ctx, void *k, void *v)
{
    (void)ctx;
    if (k == &key && v == &value) {
        InterlockedIncrement(&calls);
        dtor_thread = GetCurrentThreadId();
        dtor_tls = TlsGetValue(tls_slot);
    }
}

static DWORD WINAPI loader_worker(void *unused)
{
    (void)unused;
    worker_thread_id = GetCurrentThreadId();
    WaitForSingleObject(worker_go, INFINITE);
    worker_status = reset(device);
    /* the thread itself cannot finish exiting while the loader lock is held */
    SetEvent(worker_done);
    return 0;
}

__declspec(dllexport) int __cdecl lc_arm(int exit_process)
{
    HMODULE nv = GetModuleHandleA(getenv("W1_NVCUDA") ? getenv("W1_NVCUDA") : "nvcuda.dll");
    if (!nv)
        return -10;
    PFN_cuInit init = (PFN_cuInit)GetProcAddress(nv, "cuInit");
    PFN_cuDeviceGet get = (PFN_cuDeviceGet)GetProcAddress(nv, "cuDeviceGet");
    retain = (PFN_cuDevicePrimaryCtxRetain)GetProcAddress(nv, "cuDevicePrimaryCtxRetain");
    PFN_cuCtxSetCurrent set = set_current = (PFN_cuCtxSetCurrent)GetProcAddress(nv, "cuCtxSetCurrent");
    PFN_cuGetExportTable table = (PFN_cuGetExportTable)GetProcAddress(nv, "cuGetExportTable");
    reset = (PFN_cuDevicePrimaryCtxReset)GetProcAddress(nv, "cuDevicePrimaryCtxReset_v2");
    if (!init || !get || !retain || !set || !table || !reset)
        return -11;
    CUcontext ctx;
    const void *cls = NULL;
    CUresult r = init(0);
    r = r ? r : get(&device, 0);
    r = r ? r : retain(&ctx, device);
    r = r ? r : set(ctx);
    r = r ? r : table(&cls, &W1_CLS_UUID);
    if (r)
        return r;
    cls_put = (w1_cls_put)((void *const *)cls)[0];
    r = cls_put(ctx, &key, &value, dtor);
    if (r)
        return r;
    if (!exit_process) {
        worker_go = CreateEventA(NULL, TRUE, FALSE, NULL);
        worker_done = CreateEventA(NULL, TRUE, FALSE, NULL);
        worker = CreateThread(NULL, 0, loader_worker, NULL, 0, NULL);
    }
    tls_slot = TlsAlloc();
    armed = r == CUDA_SUCCESS;
    return r;
}

BOOL WINAPI DllMain(HINSTANCE self, DWORD reason, void *reserved)
{
    (void)self;
    if (reason != DLL_PROCESS_DETACH || !armed)
        return TRUE;
    const char *how = reserved ? "exitprocess" : "freelibrary";
    say("\"event\":\"marker\",\"marker\":\"detach-before-reset\",\"how\":\"%s\"", how);
    TlsSetValue(tls_slot, (void *)(uintptr_t)0x5678);
    CUresult r = reset(device);
    int ok = r == CUDA_SUCCESS && calls == 1 && dtor_thread == GetCurrentThreadId() &&
             dtor_tls == (void *)(uintptr_t)0x5678;
    say("\"check\":\"detach-result\",\"how\":\"%s\",\"pass\":%s,\"reset_status\":%d,\"calls\":%ld,"
        "\"same_thread\":%s,\"tls_ok\":%s",
        how, ok ? "true" : "false", r, calls, dtor_thread == GetCurrentThreadId() ? "true" : "false",
        dtor_tls == (void *)(uintptr_t)0x5678 ? "true" : "false");
    if (!reserved && worker) {
        /* a fresh registration, then a reset from the worker while this
         * thread still holds the loader lock */
        CUcontext ctx;
        CUresult p = retain(&ctx, device);
        p = p ? p : set_current(ctx);
        p = p ? p : cls_put(ctx, &worker_key, &value, worker_dtor);
        say("\"event\":\"loader-lock-worker-armed\",\"status\":%d", p);
        SetEvent(worker_go);
        DWORD w = WaitForSingleObject(worker_done, 10000);
        say("\"event\":\"loader-lock-worker\",\"finished\":%s,\"status\":%ld,\"dtor_calls\":%ld,"
            "\"dtor_on_worker\":%s",
            w == WAIT_OBJECT_0 ? "true" : "false", worker_status, worker_dtor_calls,
            worker_dtor_calls && worker_dtor_thread == worker_thread_id ? "true" : "false");
    }
    return TRUE;
}

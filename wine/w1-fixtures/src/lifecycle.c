/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * W1.7 context-storage / destructor / TLS / reentry fixture (x64 and PE32).
 * One case per disposable process:
 *
 *   lifecycle.exe cls-basic     put/get/remove, failed removal, no-CUDA dtor on reset
 *   lifecycle.exe cls-overwrite overwrite retention: one dtor, newest value
 *   lifecycle.exe cls-two       two keys x two contexts (primary + created)
 *   lifecycle.exe cls-drained   reset after drained native kernel work, then reuse
 *   lifecycle.exe tls-notify    HEAP_ACCESS thread-exit callbacks: order,
 *                               self-removal, removal of a later entry, ignored arg2
 *   lifecycle.exe freelibrary   dtor from lc_helper.dll FreeLibrary detach
 *   lifecycle.exe exitprocess   dtor from lc_helper.dll ExitProcess detach
 *   lifecycle.exe no-user32     registration without a user32 dispatcher
 *   lifecycle.exe reentry       backend context-mutex reentry gate (§4.7.5):
 *                               dtor re-enters same-context get during reset.
 *                               Exit 124 (deadline) = BLOCKED = unresolved.
 *
 * Every destructor records its thread id and a TLS value set by the thread
 * that triggered destruction, so "same Windows thread/TLS" is checked
 * directly. Destructor counts are exact: each registration fires once or the
 * case fails.
 */
#include "w1_dark.h"

static struct w1_cls cls;
static DWORD tls_slot;
#define TLS_MARK ((void *)(uintptr_t)0x5678)

struct dtor_log {
    volatile LONG calls;
    CUcontext ctx[8];
    void *key[8], *value[8], *tls[8];
    DWORD thread[8];
};
static struct dtor_log dlog;

static void CUDAAPI quiet_dtor(CUcontext ctx, void *key, void *value)
{
    LONG i = InterlockedIncrement(&dlog.calls) - 1;
    if (i < 8) {
        dlog.ctx[i] = ctx;
        dlog.key[i] = key;
        dlog.value[i] = value;
        dlog.tls[i] = TlsGetValue(tls_slot);
        dlog.thread[i] = GetCurrentThreadId();
    }
}

static void expect_dtors(const char *name, LONG want, CUcontext ctx, void *key, void *value)
{
    int ok = dlog.calls == want;
    for (LONG i = 0; i < want && i < 8; i++)
        ok &= dlog.thread[i] == GetCurrentThreadId() && dlog.tls[i] == TLS_MARK;
    if (want == 1 && ok)
        ok = dlog.ctx[0] == ctx && dlog.key[0] == key && dlog.value[0] == value;
    w1_check(name, ok, "\"calls\":%ld,\"want\":%ld,\"same_thread\":%s,\"tls_ok\":%s,\"args_ok\":%s", dlog.calls,
             want, dlog.calls && dlog.thread[0] == GetCurrentThreadId() ? "true" : "false",
             dlog.calls && dlog.tls[0] == TLS_MARK ? "true" : "false",
             dlog.calls && dlog.ctx[0] == ctx && dlog.key[0] == key && dlog.value[0] == value ? "true" : "false");
}

static int primary(CUdevice *dev, CUcontext *pctx)
{
    CUresult r = CU(cuInit, 0);
    r = r ? r : CU(cuDeviceGet, dev, 0);
    r = r ? r : CU(cuDevicePrimaryCtxRetain, pctx, *dev);
    r = r ? r : CU(cuCtxSetCurrent, *pctx);
    w1_check("primary-context", r == CUDA_SUCCESS, "\"status\":%d", r);
    return r ? -1 : w1_get_cls(&cls);
}

static void case_basic(void)
{
    CUdevice dev;
    CUcontext ctx;
    if (primary(&dev, &ctx))
        return;
    static char k1, k2, v1, v2;
    void *got = (void *)1;
    CUresult r = cls.put(ctx, &k1, &v1, quiet_dtor);
    w1_check("put", r == CUDA_SUCCESS, "\"status\":%d", r);
    r = cls.get(&got, ctx, &k1);
    w1_check("get", r == CUDA_SUCCESS && got == &v1, "\"status\":%d,\"value_ok\":%s", r, got == &v1 ? "true" : "false");
    r = cls.del(ctx, &k1);
    w1_check("remove", r == CUDA_SUCCESS, "\"status\":%d", r);
    got = (void *)1;
    r = cls.get(&got, ctx, &k1);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"get-after-remove\",\"status\":%d,\"output_unchanged\":%s", r,
            got == (void *)1 ? "true" : "false");
    w1_check("get-after-remove-fails", r != CUDA_SUCCESS, "\"status\":%d", r);
    r = cls.del(ctx, &k2);
    w1_emit("\"event\":\"error-contract\",\"entry\":\"remove-unknown\",\"status\":%d", r);
    w1_check("remove-unknown-fails", r != CUDA_SUCCESS, "\"status\":%d", r);
    /* a removed entry must not fire; a live one fires once on reset */
    r = cls.put(ctx, &k2, &v2, quiet_dtor);
    w1_check("put-second", r == CUDA_SUCCESS, "\"status\":%d", r);
    TlsSetValue(tls_slot, TLS_MARK);
    r = CU(cuDevicePrimaryCtxReset, dev);
    w1_check("reset", r == CUDA_SUCCESS, "\"status\":%d", r);
    expect_dtors("dtor-once-on-reset", 1, ctx, &k2, &v2);
    CUcontext again = NULL;
    r = CU(cuDevicePrimaryCtxRetain, &again, dev);
    w1_check("retain-after-reset", r == CUDA_SUCCESS, "\"status\":%d", r);
    CU(cuDevicePrimaryCtxRelease, dev);
}

static void case_overwrite(void)
{
    CUdevice dev;
    CUcontext ctx;
    if (primary(&dev, &ctx))
        return;
    static char k, v1, v2;
    CUresult r = cls.put(ctx, &k, &v1, quiet_dtor);
    r |= cls.put(ctx, &k, &v2, quiet_dtor);
    void *got = NULL;
    r |= cls.get(&got, ctx, &k);
    w1_check("overwrite-get-newest", r == CUDA_SUCCESS && got == &v2, "\"status\":%d", r);
    TlsSetValue(tls_slot, TLS_MARK);
    r = CU(cuDevicePrimaryCtxReset, dev);
    w1_check("reset", r == CUDA_SUCCESS, "\"status\":%d", r);
    /* canonical overwrite drops the old mapping without its dtor */
    expect_dtors("overwrite-one-dtor-newest", 1, ctx, &k, &v2);
}

static void case_two(void)
{
    CUdevice dev;
    CUcontext pctx, cctx = NULL;
    if (primary(&dev, &pctx))
        return;
    static char ka, kb, va, vb, vc, vd;
    CUresult r = CU(cuCtxCreate, &cctx, 0, dev);
    w1_check("create-second-context", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (r)
        return;
    r = cls.put(pctx, &ka, &va, quiet_dtor);
    r |= cls.put(pctx, &kb, &vb, quiet_dtor);
    r |= cls.put(cctx, &ka, &vc, quiet_dtor);
    r |= cls.put(cctx, &kb, &vd, quiet_dtor);
    void *g1 = NULL, *g2 = NULL, *g3 = NULL, *g4 = NULL;
    r |= cls.get(&g1, pctx, &ka);
    r |= cls.get(&g2, pctx, &kb);
    r |= cls.get(&g3, cctx, &ka);
    r |= cls.get(&g4, cctx, &kb);
    w1_check("two-by-two-isolated", r == CUDA_SUCCESS && g1 == &va && g2 == &vb && g3 == &vc && g4 == &vd,
             "\"status\":%d", r);
    TlsSetValue(tls_slot, TLS_MARK);
    r = CU(cuCtxDestroy, cctx);
    w1_check("destroy-created", r == CUDA_SUCCESS, "\"status\":%d", r);
    LONG after_destroy = dlog.calls;
    int created_only = 1;
    for (LONG i = 0; i < after_destroy && i < 8; i++)
        created_only &= dlog.ctx[i] == cctx;
    w1_emit("\"event\":\"observed\",\"entry\":\"created-context-destroy-dtors\",\"calls\":%ld,\"created_only\":%s",
            after_destroy, created_only ? "true" : "false");
    CU(cuCtxSetCurrent, pctx);
    r = CU(cuDevicePrimaryCtxReset, dev);
    w1_check("reset-primary", r == CUDA_SUCCESS, "\"status\":%d", r);
    unsigned primary_calls = 0;
    for (LONG i = after_destroy; i < dlog.calls && i < 8; i++)
        primary_calls += dlog.ctx[i] == pctx;
    w1_check("primary-two-dtors", primary_calls == 2 && dlog.calls - after_destroy == 2, "\"primary_calls\":%u",
             primary_calls);
    int same = 1;
    for (LONG i = 0; i < dlog.calls && i < 8; i++)
        same &= dlog.thread[i] == GetCurrentThreadId() && dlog.tls[i] == TLS_MARK;
    w1_check("all-dtors-same-thread-tls", same, "\"calls\":%ld", dlog.calls);
}

static void case_drained(void)
{
    CUdevice dev;
    CUcontext ctx;
    if (primary(&dev, &ctx))
        return;
    CUdeviceptr buf = 0;
    CUresult r = CU(cuMemAlloc, &buf, 1 << 20);
    r |= CU(cuMemsetD32, buf, 0x11223344u, (1 << 20) / 4);
    r |= CU(cuCtxSynchronize);
    w1_check("native-work-drained", r == CUDA_SUCCESS, "\"status\":%d", r);
    static char k, v;
    r = cls.put(ctx, &k, &v, quiet_dtor);
    TlsSetValue(tls_slot, TLS_MARK);
    r |= CU(cuDevicePrimaryCtxReset, dev);
    w1_check("reset-after-work", r == CUDA_SUCCESS, "\"status\":%d", r);
    expect_dtors("dtor-once", 1, ctx, &k, &v);
    CUcontext again;
    r = CU(cuDevicePrimaryCtxRetain, &again, dev);
    r |= CU(cuCtxSetCurrent, again);
    r |= CU(cuMemAlloc, &buf, 4096);
    r |= CU(cuMemFree, buf);
    w1_check("usable-after-reset", r == CUDA_SUCCESS, "\"status\":%d", r);
}

/* ---- HEAP_ACCESS thread-exit notifications ---- */

static struct w1_tls tls;
static void *h_a, *h_b, *h_c;
static volatile LONG seq_len;
static char seq[64];
static DWORD seq_thread[64];
static volatile int b_removes_self, a_removes_c;

static void record(char who, void *zero)
{
    LONG i = InterlockedIncrement(&seq_len) - 1;
    if (i < 63) {
        seq[i] = zero ? (char)(who - 'A' + 'a') : who; /* lowercase = nonzero first argument */
        seq_thread[i] = GetCurrentThreadId();
    }
}

static void CUDAAPI cb_a(void *zero, void *userdata)
{
    record(userdata == &h_a ? 'A' : '?', zero);
    if (a_removes_c) {
        a_removes_c = 0;
        CUresult r = tls.remove(h_c, NULL);
        w1_emit("\"event\":\"remove-later-from-callback\",\"status\":%d", r);
    }
}

static void CUDAAPI cb_b(void *zero, void *userdata)
{
    record(userdata == &h_b ? 'B' : '?', zero);
    if (b_removes_self) {
        b_removes_self = 0;
        CUresult r = tls.remove(h_b, (void *)(uintptr_t)0xBADC0DE);
        w1_emit("\"event\":\"self-remove-from-callback\",\"status\":%d", r);
    }
}

static void CUDAAPI cb_c(void *zero, void *userdata)
{
    record(userdata == &h_c ? 'C' : '?', zero);
}

static DWORD WINAPI just_exit(void *unused)
{
    (void)unused;
    return 0;
}

static void exit_one_thread(const char *name, const char *want)
{
    seq_len = 0;
    memset(seq, 0, sizeof seq);
    DWORD tid = 0;
    HANDLE t = CreateThread(NULL, 0, just_exit, NULL, 0, &tid);
    WaitForSingleObject(t, 10000);
    CloseHandle(t);
    int thread_ok = 1;
    for (LONG i = 0; i < seq_len && i < 63; i++)
        thread_ok &= seq_thread[i] == tid;
    w1_check(name, !strcmp(seq, want) && thread_ok, "\"observed\":\"%s\",\"want\":\"%s\",\"on_exiting_thread\":%s",
             seq, want, thread_ok ? "true" : "false");
}

static void case_tls(void)
{
    CUresult r = CU(cuInit, 0);
    w1_check("cuInit", r == CUDA_SUCCESS, "\"status\":%d", r);
    if (r || w1_get_tls(&tls))
        return;
    r = tls.set(&h_a, cb_a, &h_a);
    r |= tls.set(&h_b, cb_b, &h_b);
    w1_check("register-a-b", r == CUDA_SUCCESS && h_a && h_b, "\"status\":%d", r);
    exit_one_thread("exit-order", "AB");
    b_removes_self = 1;
    exit_one_thread("exit-self-removal-runs-once", "AB");
    exit_one_thread("exit-after-self-removal", "A");
    r = tls.set(&h_c, cb_c, &h_c);
    w1_check("register-c", r == CUDA_SUCCESS && h_c, "\"status\":%d", r);
    exit_one_thread("exit-with-c", "AC");
    a_removes_c = 1;
    exit_one_thread("exit-a-removes-later-c", "A");
    exit_one_thread("exit-after-removing-c", "A");
    r = tls.remove(h_a, (void *)(uintptr_t)0x1234);
    w1_check("remove-ignores-arg2", r == CUDA_SUCCESS, "\"status\":%d", r);
    exit_one_thread("exit-none-registered", "");
}

/* ---- FreeLibrary / ExitProcess detach through lc_helper.dll ---- */

typedef int(__cdecl *lc_arm_fn)(int mode);

static void case_detach(int exit_process)
{
    /* nvcuda first, so it is detached after the helper at process exit */
    HMODULE helper = LoadLibraryA("lc_helper.dll");
    w1_check("load-helper", helper != NULL, "\"v_error\":%lu", (unsigned long)GetLastError());
    if (!helper)
        return;
    lc_arm_fn arm = (lc_arm_fn)GetProcAddress(helper, "lc_arm");
    int armed = arm ? arm(exit_process) : -1;
    w1_check("helper-armed", armed == 0, "\"status\":%d", armed);
    if (armed)
        return;
    if (exit_process) {
        w1_emit("\"event\":\"exit-process\"");
        /* The helper reports from DLL_PROCESS_DETACH and sets the exit code
         * there; the harness requires its "detach-result" line. */
        ExitProcess(0);
    }
    BOOL freed = FreeLibrary(helper);
    w1_check("freelibrary", freed, "\"x\":0");
}

/* ---- registration without a user32 dispatcher ---- */

static unsigned thread_count(void);

static void case_no_user32(void)
{
    int before = GetModuleHandleA("user32.dll") != NULL;
    CUdevice dev;
    CUcontext ctx;
    int pre = primary(&dev, &ctx);
    int after_load = GetModuleHandleA("user32.dll") != NULL;
    w1_emit("\"event\":\"user32\",\"before_nvcuda\":%s,\"after_init\":%s", before ? "true" : "false",
            after_load ? "true" : "false");
    if (pre)
        return;
    if (after_load) {
        w1_emit("\"event\":\"not-applicable\",\"reason\":\"user32 loaded in process\"");
        return;
    }
    static char k, v;
    unsigned t0 = thread_count();
    CUresult r = cls.put(ctx, &k, &v, quiet_dtor);
    unsigned t1 = thread_count();
    w1_emit("\"event\":\"error-contract\",\"entry\":\"put-without-dispatcher\",\"status\":%d", r);
    w1_check("refused-without-dispatcher", r != CUDA_SUCCESS, "\"status\":%d", r);
    w1_check("no-worker-created", t1 <= t0, "\"v_before\":%u,\"v_after\":%u", t0, t1);
    if (!r)
        cls.del(ctx, &k);
}

/* ---- backend context-mutex reentry gate ---- */

static CUcontext reentry_ctx;
static char reentry_key, reentry_value, other_key;

static void CUDAAPI reentering_dtor(CUcontext ctx, void *key, void *value)
{
    (void)key;
    (void)value;
    w1_emit("\"event\":\"marker\",\"marker\":\"dtor-entered\"");
    void *got = NULL;
    CUresult r = cls.get(&got, ctx, &other_key);
    w1_emit("\"event\":\"marker\",\"marker\":\"dtor-get-returned\",\"status\":%d", r);
}

static void case_reentry(void)
{
    CUdevice dev;
    if (primary(&dev, &reentry_ctx))
        return;
    CUresult r = cls.put(reentry_ctx, &other_key, &reentry_value, NULL);
    r |= cls.put(reentry_ctx, &reentry_key, &reentry_value, reentering_dtor);
    w1_check("reentry-armed", r == CUDA_SUCCESS, "\"status\":%d", r);
    w1_emit("\"event\":\"marker\",\"marker\":\"before-reset\"");
    r = CU(cuDevicePrimaryCtxReset, dev);
    w1_emit("\"event\":\"marker\",\"marker\":\"reset-returned\",\"status\":%d", r);
    /* returning at all = not blocked; the dtor-get status is the contract */
}

#include <tlhelp32.h>
static unsigned thread_count(void)
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te = {sizeof te};
    unsigned n = 0;
    if (snap == INVALID_HANDLE_VALUE)
        return 0;
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te))
        n += te.th32OwnerProcessID == GetCurrentProcessId();
    CloseHandle(snap);
    return n;
}

int main(int argc, char **argv)
{
    const char *which = argc > 1 ? argv[1] : "cls-basic";
    w1_start("lifecycle", !strcmp(which, "reentry") ? 30000 : 60000);
    w1_emit("\"event\":\"case\",\"case\":\"%s\"", which);
    tls_slot = TlsAlloc();
    if (!w1_load_nvcuda()) {
        w1_check("nvcuda-load", 0, "\"x\":0");
        return 2;
    }
    w1_resolve_common();
    if (!strcmp(which, "cls-basic"))
        case_basic();
    else if (!strcmp(which, "cls-overwrite"))
        case_overwrite();
    else if (!strcmp(which, "cls-two"))
        case_two();
    else if (!strcmp(which, "cls-drained"))
        case_drained();
    else if (!strcmp(which, "tls-notify"))
        case_tls();
    else if (!strcmp(which, "freelibrary"))
        case_detach(0);
    else if (!strcmp(which, "exitprocess"))
        case_detach(1);
    else if (!strcmp(which, "no-user32"))
        case_no_user32();
    else if (!strcmp(which, "reentry"))
        case_reentry();
    else
        w1_check("known-case", 0, "\"case\":\"%s\"", which);
    return w1_finish("lifecycle");
}

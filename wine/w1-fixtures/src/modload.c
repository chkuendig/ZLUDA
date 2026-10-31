/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Plan §4.10 cuModuleLoad stack-headroom driver (x64 and PE32).
 *
 *   modload.exe [--stack-reserve BYTES] [--self] FILE...
 *
 * Loads each module image (PTX text, cubin or fatbin, as staged on the Deck
 * lane; never shipped in this repository) with cuModuleLoadData, twice: the
 * first load must be cold (the harness points the ZLUDA cache at an empty
 * private directory), the second shows the warm/cache-hit time. For each load
 * the fixture prints begin/end markers with the module's size and SHA-256 so
 * the diagnostic build's Unix-side kernel-stack paint (the gating
 * measurement) can be joined to it, plus its own PE-side measurement:
 *
 *   - the calling thread's stack extent from the TEB (base, limit,
 *     reservation) = the game-thread-like stack the call ran on;
 *   - a paint of the unused PE stack below SP before the call and a scan for
 *     the deepest touched byte after it. Under Wine the Unix side runs on the
 *     thread's separate kernel stack, so this sees only PE-side frames; it is
 *     context, not the headroom gate.
 *
 * The PE paint is calibrated first: a 64 KiB stack touch must be measured
 * as 64 KiB - 256 B .. 68 KiB (+ slack). --stack-reserve runs the loads on a new thread with that
 * reservation (STACK_SIZE_PARAM_IS_A_RESERVATION); default is the main
 * thread; use at least 512 KiB (the paint commits 264 KiB). --self loads the built-in w1 PTX (a smoke module, no corpus).
 */
#include "w1.h"
#include "w1_ptx.h"

/* ---- SHA-256 (FIPS 180-4) ---- */
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_block(uint32_t h[8], const unsigned char *p)
{
    uint32_t w[64], a, b, c, d, e, f, g, k;
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++)
        w[i] = w[i - 16] + (ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10));
    a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
}

static void sha256_hex(const unsigned char *data, size_t len, char out[65])
{
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char tail[128] = {0};
    size_t full = len & ~(size_t)63, rest = len - full;
    for (size_t i = 0; i < full; i += 64)
        sha256_block(h, data + i);
    memcpy(tail, data + full, rest);
    tail[rest] = 0x80;
    size_t tl = rest < 56 ? 64 : 128;
    unsigned long long bits = (unsigned long long)len * 8;
    for (int i = 0; i < 8; i++)
        tail[tl - 1 - i] = (unsigned char)(bits >> (8 * i));
    sha256_block(h, tail);
    if (tl == 128)
        sha256_block(h, tail + 64);
    for (int i = 0; i < 8; i++)
        sprintf(out + 8 * i, "%08x", h[i]);
}

/* ---- PE-side stack paint ---- */
#define PAINT 0xA5u
#define PAINT_BYTES (256u << 10)
#define REDZONE 128u

struct stack_extent { uintptr_t base, limit, reservation; };

static struct stack_extent extent(void)
{
    NT_TIB *tib = (NT_TIB *)NtCurrentTeb();
    struct stack_extent e = {(uintptr_t)tib->StackBase, (uintptr_t)tib->StackLimit, 0};
#ifdef _WIN64
    e.reservation = *(uintptr_t *)((char *)tib + 0x1478); /* TEB.DeallocationStack */
#else
    e.reservation = *(uintptr_t *)((char *)tib + 0xE0C);
#endif
    return e;
}

/* Commit PAINT_BYTES of stack below the caller (chkstk probes each page). */
static __attribute__((noinline)) void commit_stack(void)
{
    volatile unsigned char *grow = __builtin_alloca(PAINT_BYTES + 8192);
    grow[0] = 0;
}

/* The paint target is derived from the real SP (asm), not from a local's
 * address: a store below a dead local is otherwise removed as a dead store. */
static __attribute__((noinline)) unsigned char *paint_below(void)
{
    uintptr_t sp;
#ifdef _WIN64
    __asm__ volatile("movq %%rsp, %0" : "=r"(sp));
#else
    __asm__ volatile("movl %%esp, %0" : "=r"(sp));
#endif
    unsigned char *top = (unsigned char *)(sp - REDZONE - 4096); /* leave room for memset's own frame */
    __asm__ volatile("" : "+r"(top));
    memset(top - PAINT_BYTES, PAINT, PAINT_BYTES);
    __asm__ volatile("" ::"r"(top) : "memory");
    return top;
}

static size_t scan_touched(const unsigned char *top);

/* calibration diagnostics: the lowest few touched offsets above the paint floor */
static void describe_touched(const unsigned char *top)
{
    const unsigned char *low = top - PAINT_BYTES;
    unsigned long long first[4] = {0};
    unsigned found = 0, total = 0;
    for (const unsigned char *p = low; p < top; p++)
        if (*p != PAINT) {
            if (found < 4 && (!found || (size_t)(p - low) > first[found - 1] + 64))
                first[found++] = (unsigned long long)(p - low);
            total++;
        }
    w1_emit("\"event\":\"paint-diagnostic\",\"v_touched_bytes\":%u,\"v_first_offsets\":[%llu,%llu,%llu,%llu]",
            total, first[0], first[1], first[2], first[3]);
}

static size_t scan_touched(const unsigned char *top)
{
    __asm__ volatile("" ::: "memory");
    const unsigned char *low = top - PAINT_BYTES;
    for (const unsigned char *p = low; p < top; p++)
        if (*p != PAINT)
            return (size_t)(top - p) + 4096 + REDZONE;
    return 0;
}

static __attribute__((noinline)) void touch(unsigned bytes)
{
    volatile unsigned char *buf = __builtin_alloca(bytes);
    for (unsigned i = 0; i < bytes; i += 64)
        buf[i] = (unsigned char)i;
    buf[0] = 1;
}

static int calibrate(void)
{
    commit_stack();
    unsigned char *top = paint_below();
    touch(64u << 10);
    size_t got = scan_touched(top);
    describe_touched(top);
    /* resolution is about one frame: the reference SP is paint_below's,
     * not the callee's, so allow 256 bytes below the nominal 64 KiB */
    int ok = got >= (64u << 10) - 256 && got <= (68u << 10) + 4096 + REDZONE;
    w1_check("pe-paint-calibration", ok, "\"touched\":%llu,\"expected_min\":%u", (unsigned long long)got, 64u << 10);
    return ok;
}

/* ---- module loads ---- */
struct job { int argc; char **argv; int self; CUcontext ctx; };

static unsigned char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)n + 1);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    if (buf)
        buf[n] = 0; /* PTX text must be NUL terminated */
    *len = (size_t)n;
    return buf;
}

static void load_one(const char *label, const unsigned char *image, size_t len)
{
    char sha[65];
    sha256_hex(image, len, sha);
    struct stack_extent e = extent();
    for (int pass = 0; pass < 2; pass++) {
        const char *kind = pass ? "warm" : "cold";
        w1_emit("\"event\":\"marker\",\"marker\":\"load-begin\",\"pass\":\"%s\",\"sha256\":\"%s\",\"bytes\":%llu", kind,
                sha, (unsigned long long)len);
        double t0 = w1_ms();
        /* paint last: nothing but the load may run below this frame */
        commit_stack();
        unsigned char *top = paint_below();
        CUmodule mod = NULL;
        CUresult r = CU(cuModuleLoadData, &mod, image);
        size_t touched = scan_touched(top);
        double t1 = w1_ms();
        w1_emit("\"event\":\"marker\",\"marker\":\"load-end\",\"pass\":\"%s\",\"sha256\":\"%s\"", kind, sha);
        w1_check("module-load", r == CUDA_SUCCESS,
                 "\"label\":\"%s\",\"pass\":\"%s\",\"sha256\":\"%s\",\"bytes\":%llu,\"status\":%d,\"v_ms\":%.1f,"
                 "\"v_pe_stack_touched\":%llu,\"v_stack_base\":\"0x%llx\",\"v_stack_limit\":\"0x%llx\","
                 "\"v_stack_reserved\":%llu",
                 label, kind, sha, (unsigned long long)len, r, t1 - t0, (unsigned long long)touched,
                 (unsigned long long)e.base, (unsigned long long)e.limit,
                 (unsigned long long)(e.base - e.reservation));
        if (!r)
            CU(cuModuleUnload, mod);
    }
}

static DWORD WINAPI run_jobs(void *arg)
{
    struct job *job = arg;
    CU(cuCtxSetCurrent, job->ctx);
    calibrate();
    if (job->self)
        load_one("w1-self", (const unsigned char *)w1_ptx, sizeof w1_ptx);
    for (int i = 0; i < job->argc; i++) {
        size_t len = 0;
        unsigned char *image = read_file(job->argv[i], &len);
        if (!image) {
            w1_check("module-read", 0, "\"index\":%d", i);
            continue;
        }
        char label[16];
        snprintf(label, sizeof label, "file-%d", i);
        load_one(label, image, len);
        free(image);
    }
    return 0;
}

int main(int argc, char **argv)
{
    unsigned long reserve = 0;
    struct job job = {0};
    char **files = calloc((size_t)argc, sizeof *files);
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--stack-reserve") && i + 1 < argc)
            reserve = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--self"))
            job.self = 1;
        else
            files[job.argc++] = argv[i];
    }
    job.argv = files;
    w1_start("modload", 1800000);
    const char *cache = getenv("ZLUDA_CACHE_DIR"), *xdg = getenv("XDG_CACHE_HOME");
    w1_emit("\"event\":\"config\",\"files\":%d,\"self\":%d,\"stack_reserve\":%lu,\"v_ZLUDA_CACHE_DIR\":\"%s\","
            "\"v_XDG_CACHE_HOME\":\"%s\"",
            job.argc, job.self, reserve, cache ? cache : "", xdg ? xdg : "");
    if (!w1_load_nvcuda()) {
        w1_check("nvcuda-load", 0, "\"x\":0");
        return 2;
    }
    w1_resolve_common();
    CUdevice dev;
    CUcontext ctx;
    if (w1_bring_up(&dev, &ctx)) {
        /* still calibrate, so CPU validation covers the paint method */
        calibrate();
        return w1_finish("modload");
    }
    job.ctx = ctx;
    if (reserve) {
        HANDLE t = CreateThread(NULL, reserve, run_jobs, &job, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
        WaitForSingleObject(t, INFINITE);
    } else {
        run_jobs(&job);
    }
    CU(cuCtxDestroy, ctx);
    return w1_finish("modload");
}

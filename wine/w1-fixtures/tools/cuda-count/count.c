/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * libw1-cuda-count.so: an external, black-box call counter for the
 * nvcuda.so -> backend libcuda boundary (plan §4.5 Knight per-frame counts).
 * Diagnostic arms only, never W2.
 *
 * LD_PRELOAD it into the Wine process. It interposes dlopen/dlsym: when a
 * library whose path contains one of W1C_MATCH (default
 * "libcuda.so,libnvcuda,libzluda") is opened, every later dlsym of a "cu*"
 * name on that handle returns a counting trampoline (tramp.S: one locked
 * increment and a tail jump; no allocation, syscall or lock in the call path).
 * cuGetExportTable is wrapped with its public signature to also record which
 * interface UUIDs were requested.
 *
 * Snapshots: set W1C_CTL to a FIFO path. Once a matching library is opened, a
 * reader thread blocks on the FIFO; every line written to it (a pass marker
 * such as "pass1-begin") dumps one JSON line with all counts to
 * W1C_OUT.<pid> (default /tmp/w1c). A final snapshot "exit" is written from a
 * destructor. Per-frame counts = (end - begin) / frames from the title's own
 * frame counter for the same pass.
 *
 * Sees: every backend entry point the bridge resolved by dlsym, per name, for
 * startup and per pass. Does not see: calls the bridge serves without the
 * backend (cached context-get, unsupported 801 stubs, PE-only helpers),
 * dark-table slots (raw function pointers returned by cuGetExportTable),
 * symbols bound through DT_NEEDED/PLT rather than dlsym, or PE->Unix
 * transitions as such. Those need the bridge's diagnostic counters.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define W1C_SLOTS 1024
#define MAX_HANDLES 16
#define MAX_UUIDS 64

extern uint64_t w1c_counts[W1C_SLOTS];
extern void *w1c_real[W1C_SLOTS];
extern void *const w1c_tramps[W1C_SLOTS];

static char names[W1C_SLOTS][64];
static unsigned used;
static void *handles[MAX_HANDLES];
static unsigned n_handles;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t reader_once = PTHREAD_ONCE_INIT;
static void *(*real_dlopen)(const char *, int);
static void *(*real_dlsym)(void *, const char *);
static int (*real_get_export_table)(const void **, const void *);
static unsigned char uuids[MAX_UUIDS][16];
static uint64_t uuid_counts[MAX_UUIDS];
static unsigned n_uuids;

static void resolve_real(void)
{
    if (!real_dlsym) {
        real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (!real_dlsym)
            real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    if (!real_dlopen)
        real_dlopen = real_dlsym(RTLD_NEXT, "dlopen");
}

static int matches(const char *path)
{
    const char *list = getenv("W1C_MATCH");
    char buf[512];
    snprintf(buf, sizeof buf, "%s", list ? list : "libcuda.so,libnvcuda,libzluda");
    for (char *save = NULL, *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save))
        if (*tok && strstr(path, tok))
            return 1;
    return 0;
}

static void snapshot(const char *label)
{
    char path[512];
    const char *out = getenv("W1C_OUT");
    snprintf(path, sizeof path, "%s.%d", out ? out : "/tmp/w1c", (int)getpid());
    FILE *f = fopen(path, "a");
    if (!f)
        return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    fprintf(f, "{\"label\":\"%s\",\"t_ns\":%lld,\"counts\":{", label, (long long)ts.tv_sec * 1000000000ll + ts.tv_nsec);
    unsigned n = __atomic_load_n(&used, __ATOMIC_ACQUIRE);
    for (unsigned i = 0; i < n; i++)
        fprintf(f, "%s\"%s\":%llu", i ? "," : "", names[i],
                (unsigned long long)__atomic_load_n(&w1c_counts[i], __ATOMIC_RELAXED));
    fprintf(f, "},\"export_tables\":{");
    for (unsigned i = 0; i < n_uuids; i++) {
        fprintf(f, "%s\"", i ? "," : "");
        for (int b = 0; b < 16; b++)
            fprintf(f, "%02x", uuids[i][b]);
        fprintf(f, "\":%llu", (unsigned long long)__atomic_load_n(&uuid_counts[i], __ATOMIC_RELAXED));
    }
    fprintf(f, "}}\n");
    fclose(f);
}

static void *reader(void *unused)
{
    (void)unused;
    const char *ctl = getenv("W1C_CTL");
    char line[128];
    for (;;) {
        FILE *f = fopen(ctl, "r");
        if (!f)
            return NULL;
        while (fgets(line, sizeof line, f)) {
            line[strcspn(line, "\r\n\"\\")] = 0;
            snapshot(line);
        }
        fclose(f);
    }
}

static void start_reader(void)
{
    if (!getenv("W1C_CTL"))
        return;
    pthread_t t;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    pthread_create(&t, &a, reader, NULL);
    pthread_attr_destroy(&a);
}

static int counted_get_export_table(const void **table, const void *uuid)
{
    if (uuid) {
        unsigned n = __atomic_load_n(&n_uuids, __ATOMIC_ACQUIRE), i;
        for (i = 0; i < n && memcmp(uuids[i], uuid, 16); i++)
            ;
        if (i == n) {
            pthread_mutex_lock(&lock);
            n = n_uuids;
            for (i = 0; i < n && memcmp(uuids[i], uuid, 16); i++)
                ;
            if (i == n && n < MAX_UUIDS) {
                memcpy(uuids[n], uuid, 16);
                __atomic_store_n(&n_uuids, n + 1, __ATOMIC_RELEASE);
            }
            pthread_mutex_unlock(&lock);
        }
        if (i < MAX_UUIDS)
            __atomic_fetch_add(&uuid_counts[i], 1, __ATOMIC_RELAXED);
    }
    return real_get_export_table(table, uuid);
}

void *dlopen(const char *path, int flags)
{
    resolve_real();
    void *h = real_dlopen(path, flags);
    if (h && path && matches(path)) {
        pthread_mutex_lock(&lock);
        unsigned i;
        for (i = 0; i < n_handles && handles[i] != h; i++)
            ;
        if (i == n_handles && n_handles < MAX_HANDLES)
            handles[n_handles++] = h;
        pthread_mutex_unlock(&lock);
        pthread_once(&reader_once, start_reader);
    }
    return h;
}

void *dlsym(void *handle, const char *name)
{
    resolve_real();
    void *real = real_dlsym(handle, name);
    if (!real || !name || name[0] != 'c' || name[1] != 'u')
        return real;
    int tracked = 0;
    pthread_mutex_lock(&lock);
    for (unsigned i = 0; i < n_handles; i++)
        tracked |= handles[i] == handle;
    if (!tracked) {
        pthread_mutex_unlock(&lock);
        return real;
    }
    if (!strcmp(name, "cuGetExportTable")) {
        real_get_export_table = real;
        real = (void *)counted_get_export_table;
    }
    unsigned i;
    for (i = 0; i < used && strcmp(names[i], name); i++)
        ;
    if (i == used) {
        if (used == W1C_SLOTS || strlen(name) >= sizeof names[0]) {
            pthread_mutex_unlock(&lock);
            return real;
        }
        strcpy(names[i], name);
        w1c_real[i] = real;
        __atomic_store_n(&used, used + 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&lock);
    return w1c_tramps[i];
}

__attribute__((destructor)) static void final_snapshot(void)
{
    if (n_handles)
        snapshot("exit");
}

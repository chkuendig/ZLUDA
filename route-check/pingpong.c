/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Cross-process round trip over shared memory + futex, as an L32 client
 * (i386) <-> server (x86_64) IPC would do it. No Wine involved.
 *
 *   pingpong server NAME          (waits for requests, answers each one)
 *   pingpong client NAME [N]      (sends N requests, prints mean/median/p99 us)
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/futex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

struct chan { volatile uint32_t req, resp; uint32_t payload[1024]; };

static void wait_for(volatile uint32_t *word, uint32_t old)
{
    while (__atomic_load_n(word, __ATOMIC_ACQUIRE) == old)
        syscall(SYS_futex, word, FUTEX_WAIT, old, NULL, NULL, 0);
}

static void post(volatile uint32_t *word)
{
    __atomic_add_fetch(word, 1, __ATOMIC_RELEASE);
    syscall(SYS_futex, word, FUTEX_WAKE, 1, NULL, NULL, 0);
}

static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static int cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    if (argc < 3)
        return 2;
    int server = !strcmp(argv[1], "server");
    int fd = shm_open(argv[2], O_CREAT | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, sizeof(struct chan)))
        return 3;
    struct chan *c = mmap(NULL, sizeof(*c), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (server) {
        for (uint32_t seen = 0;; seen++) {
            wait_for(&c->req, seen);
            if (c->payload[0] == 0xdead)
                return 0;
            c->payload[1] = c->payload[0] + 1;
            post(&c->resp);
        }
    }
    int n = argc > 3 ? atoi(argv[3]) : 20000;
    double *s = malloc(sizeof(double) * n), total = 0;
    for (int i = 0; i < n; i++) {
        uint32_t resp = c->resp;
        double t0 = now_us();
        c->payload[0] = i;
        post(&c->req);
        wait_for(&c->resp, resp);
        s[i] = now_us() - t0;
        total += s[i];
    }
    qsort(s, n, sizeof(double), cmp);
    printf("{\"pingpong\":\"futex\",\"client_bits\":%d,\"iterations\":%d,\"mean_us\":%.2f,\"median_us\":%.2f,\"p99_us\":%.2f}\n",
           (int)sizeof(void *) * 8, n, total / n, s[n / 2], s[n * 99 / 100]);
    c->payload[0] = 0xdead;
    post(&c->req);
    shm_unlink(argv[2]);
    return 0;
}

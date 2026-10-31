/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Counter self-test driver: dlopen the fake backend, call through dlsym'd
 * pointers a known number of times between FIFO markers. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void mark(const char *ctl, const char *label)
{
    FILE *f = fopen(ctl, "w");
    fprintf(f, "%s\n", label);
    fclose(f);
    usleep(300000); /* snapshots are asynchronous to the marker */
}

int main(int argc, char **argv)
{
    void *h = dlopen(argv[1], RTLD_NOW);
    int (*init)(unsigned) = dlsym(h, "cuInit");
    int (*ver)(int *) = dlsym(h, "cuDriverGetVersion");
    int (*table)(const void **, const void *) = dlsym(h, "cuGetExportTable");
    int (*other)(void) = dlsym(h, "notCuda");
    const char *ctl = getenv("W1C_CTL");
    unsigned char uuid[16] = {0xC6, 0x93, 0x33, 0x6E};
    const void *t;
    int v = 0, bad = 0;
    mark(ctl, "begin");
    for (int i = 0; i < 1000; i++)
        bad |= init(0) != 0;
    for (int i = 0; i < 250; i++)
        bad |= ver(&v) != 0 || v != 12000;
    for (int i = 0; i < 3; i++)
        bad |= table(&t, uuid);
    bad |= other() != 7;
    mark(ctl, "end");
    (void)argc;
    return bad;
}

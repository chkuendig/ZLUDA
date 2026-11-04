/* SPDX-License-Identifier: MIT
 * Same-OS-thread guard-page/stack-paint probe for the actual zoc pipeline.
 * Test-only preload: never link this into ZLUDA, Wine or the shipped bridge.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

typedef int (*main_fn)(int,char **,char **);
typedef int (*start_fn)(main_fn,int,char **,void (*)(void),void (*)(void),void (*)(void),void *);
static main_fn original_main;
static int saved_argc,main_status;
static char **saved_argv,**saved_env;
static unsigned char *mapping,*stack_low,*stack_high;
static size_t page,reserve;
static unsigned seed;
static long initial_tid,compile_tid;
static ucontext_t caller,callee;
static volatile unsigned calibrated;

static void fault(int signal,siginfo_t *info,void *context)
{
    (void)signal;(void)context;
    uintptr_t address=(uintptr_t)info->si_addr;
    if ((address>=(uintptr_t)mapping && address<(uintptr_t)stack_low) ||
        (address>=(uintptr_t)stack_high && address<(uintptr_t)stack_high+page)) {
        static const char text[]="STACK_PROBE_GUARD_HIT\n";
        (void)write(STDERR_FILENO,text,sizeof(text)-1);_exit(125);
    }
    _exit(126);
}
static __attribute__((noinline)) unsigned recurse(unsigned depth)
{
    volatile unsigned char block[8192];
    block[0]=(unsigned char)depth;block[sizeof(block)-1]=(unsigned char)(depth^0x5a);
    unsigned value=depth ? recurse(depth-1):0;
    return value+block[0]+block[sizeof(block)-1];
}
static void compile_on_stack(void)
{
    compile_tid=syscall(SYS_gettid);
    const char *test=getenv("ZLUDA_STACK_PROBE_SELFTEST");
    if (test) {calibrated=recurse(!strcmp(test,"overflow") ? 1024:8);main_status=0;}
    else main_status=original_main(saved_argc,saved_argv,saved_env);
}
static unsigned char paint(size_t offset) {return (unsigned char)(offset*131u+seed);}
static int measured_main(int argc,char **argv,char **env)
{
    const char *report=getenv("ZLUDA_STACK_PROBE_REPORT");
    const char *seed_text=getenv("ZLUDA_STACK_PROBE_SEED");
    if (!report || report[0]!='/') return 127;
    seed=seed_text ? (unsigned)strtoul(seed_text,NULL,10):91;
    initial_tid=syscall(SYS_gettid);saved_argc=argc;saved_argv=argv;saved_env=env;
    page=(size_t)sysconf(_SC_PAGESIZE);reserve=1024*1024;
    if (!page || reserve%page) return 127;
    // Wine's reserve includes its low guard. Keep exactly that1MiB extent,
    // plus an extra upper guard to catch accidental overflow in either direction.
    mapping=mmap(NULL,reserve+page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (mapping==MAP_FAILED) return 127;
    stack_low=mapping+page;stack_high=mapping+reserve;
    if (mprotect(stack_low,reserve-page,PROT_READ|PROT_WRITE)) return 127;
    for (size_t i=0;i<reserve-page;++i) stack_low[i]=paint(i);
    stack_t alternate={0};alternate.ss_size=(size_t)SIGSTKSZ*4;alternate.ss_sp=malloc(alternate.ss_size);
    if (!alternate.ss_sp || sigaltstack(&alternate,NULL)) return 127;
    struct sigaction action={0};action.sa_sigaction=fault;action.sa_flags=SA_SIGINFO|SA_ONSTACK;
    if (sigemptyset(&action.sa_mask) || sigaction(SIGSEGV,&action,NULL)) return 127;
    if (getcontext(&callee)) return 127;
    callee.uc_stack.ss_sp=stack_low;callee.uc_stack.ss_size=reserve-page;callee.uc_link=&caller;
    makecontext(&callee,compile_on_stack,0);
    if (swapcontext(&caller,&callee)) return 127;
    size_t first=0;
    while (first<reserve-page && stack_low[first]==paint(first)) ++first;
    size_t touched=reserve-page-first;
    // Touched bytes are a high-water observation, not a proof about untouched
    // SP reservations. Add x64 red-zone allowance and round upwards one page.
    size_t bound=((touched+128+page-1)/page)*page;
    int descriptor=open(report,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if (descriptor<0) return 127;
    int bytes=dprintf(descriptor,
        "{\"reserve_bytes\":%zu,\"low_guard_bytes\":%zu,\"upper_guard_bytes\":%zu,"
        "\"usable_bytes\":%zu,\"touched_high_water_bytes\":%zu,\"rounded_observed_bound_bytes\":%zu,"
        "\"initial_tid\":%ld,\"compile_tid\":%ld,\"same_thread\":%s,\"main_status\":%d,\"seed\":%u}\n",
        reserve,page,page,reserve-page,touched,bound,initial_tid,compile_tid,
        initial_tid==compile_tid ? "true":"false",main_status,seed);
    if (close(descriptor) || bytes<=0 || initial_tid!=compile_tid) return 127;
    return main_status;
}
int __libc_start_main(main_fn main,int argc,char **argv,void (*init)(void),
    void (*fini)(void),void (*rtld_fini)(void),void *stack_end)
{
    start_fn next=(start_fn)dlsym(RTLD_NEXT,"__libc_start_main");
    if (!next) _exit(127);
    original_main=main;
    return next(measured_main,argc,argv,init,fini,rtld_fini,stack_end);
}

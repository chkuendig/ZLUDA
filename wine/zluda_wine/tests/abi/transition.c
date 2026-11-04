/* SPDX-License-Identifier: MIT
 * Compile with the exact Wine SDK. Measure private no-op index1 to separate
 * Wine's x64 PE->Unix transition from CUDA packet/backend work. No GPU call.
 */
#include <stdint.h>
#include <stdio.h>
#include <windef.h>
#include <winbase.h>
#include <winternl.h>
#include <wine/unixlib.h>
typedef NTSTATUS (WINAPI *query_fn)(HANDLE,const void *,MEMORY_INFORMATION_CLASS,void *,SIZE_T,SIZE_T *);
typedef NTSTATUS (WINAPI *dispatch_fn)(unixlib_handle_t,unsigned,void *);
static __attribute__((noinline)) NTSTATUS WINAPI local_noop(unixlib_handle_t handle,unsigned index,void *data)
{(void)handle;(void)index;(void)data;return 0;}
static dispatch_fn volatile local=local_noop;
static uint64_t ticks(dispatch_fn function,unixlib_handle_t handle,unsigned iterations)
{
    LARGE_INTEGER a,b;
    if (!QueryPerformanceCounter(&a)) ExitProcess(2);
    for (unsigned i=0;i<iterations;++i) if (function(handle,1,NULL)) ExitProcess(3);
    if (!QueryPerformanceCounter(&b) || b.QuadPart<=a.QuadPart) ExitProcess(4);
    return (uint64_t)(b.QuadPart-a.QuadPart);
}
int main(void)
{
    HMODULE module=LoadLibraryW(L"nvcuda.dll"),ntdll=GetModuleHandleW(L"ntdll.dll");
    query_fn query=(void *)GetProcAddress(ntdll,"NtQueryVirtualMemory");
    dispatch_fn *slot=(void *)GetProcAddress(ntdll,"__wine_unix_call_dispatcher");
    unixlib_handle_t handle=0;LARGE_INTEGER frequency;
    if (!module || !query || !slot || !*slot ||
        query((HANDLE)(LONG_PTR)-1,module,MemoryWineUnixFuncs,&handle,sizeof(handle),NULL) ||
        !handle || !QueryPerformanceFrequency(&frequency) || frequency.QuadPart<=0) return 1;
    for (unsigned i=0;i<1000;++i) if ((*slot)(handle,1,NULL)) return 1;
    uint64_t baseline[11],samples[11];const unsigned iterations=200000;
    for (unsigned i=0;i<11;++i) {baseline[i]=ticks(local,handle,iterations);samples[i]=ticks(*slot,handle,iterations);}
    printf("TRANSITIONS: {\"iterations\":%u,\"frequency_hz\":%llu,\"samples_ticks\":[",iterations,(unsigned long long)frequency.QuadPart);
    for (unsigned i=0;i<11;++i) printf("%s%llu",i ? ",":"",(unsigned long long)samples[i]);
    printf("],\"baseline_ticks\":[");
    for (unsigned i=0;i<11;++i) printf("%s%llu",i ? ",":"",(unsigned long long)baseline[i]);
    puts("]}");puts("PASS: pinned Wine PE64 Unix transition timing");return 0;
}

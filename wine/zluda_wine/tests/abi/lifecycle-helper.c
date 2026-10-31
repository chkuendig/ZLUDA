/* SPDX-License-Identifier: MIT
 * This DLL deliberately triggers CUDA reset while Wine owns the loader lock.
 * No CRT, heap allocation, worker, event wait or stderr is used in detach.
 */
#include "lifecycle.h"
static struct lifecycle_setup state;
static volatile LONG seen;
static volatile LONG armed;
static void WINAPI destructor(void *context,void *key,void *value)
{
    void *current=NULL;
    HMODULE module=GetModuleHandleA("nvcuda.dll");
    int (WINAPI *get_current)(void **)=(void *)GetProcAddress(module,"cuCtxGetCurrent");
    lifecycle_status("destructor-thread",GetCurrentThreadId());
    lifecycle_status("destructor-tls",(uintptr_t)TlsGetValue(state.tls));
    if (context!=state.context || key!=state.key || value!=state.value ||
        GetCurrentThreadId()!=state.caller || TlsGetValue(state.tls)!=state.value ||
        !get_current || get_current(&current) || current!=context ||
        InterlockedIncrement(&seen)!=1 || !lifecycle_event(state.events,"same-thread-destructor\n"))
        TerminateProcess(GetCurrentProcess(),91);
}
__declspec(dllexport) int WINAPI fixture_setup(const struct lifecycle_setup *setup)
{
    static const unsigned char uuid[16]={0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93};
    HMODULE module=GetModuleHandleA("nvcuda.dll");
    int (WINAPI *get_table)(const void **,const void *)=(void *)GetProcAddress(module,"cuGetExportTable");
    const uintptr_t *table=NULL;void *out=NULL;
    lifecycle_status("setup-entered",GetCurrentThreadId());
    lifecycle_status("user32-module",(uintptr_t)GetModuleHandleA("user32.dll"));
    if (!setup || !get_table) return 999;
    int status=get_table((const void **)&table,uuid);
    lifecycle_status("get-context-table",(uintptr_t)(unsigned)status);
    if (status || !table) return status ? status:999;
    state=*setup;
    int (WINAPI *set)(void *,void *,void *,void *)=(void *)table[0];
    int (WINAPI *get)(void **,void *,void *)=(void *)table[2];
    status=set(state.context,state.key,state.value,(void *)destructor);
    lifecycle_status("storage-set",(uintptr_t)(unsigned)status);
    if (status) return status;
    status=get(&out,NULL,state.key);
    lifecycle_status("storage-get",(uintptr_t)(unsigned)status);
    lifecycle_status("storage-value",(uintptr_t)out);
    if (status || out!=state.value) return status ? status:999;
    InterlockedExchange(&armed,1);
    lifecycle_status("helper-armed",1);
    return 0;
}
static size_t WINAPI forbidden_worker_callback(int block)
{ (void)block;TerminateProcess(GetCurrentProcess(),92);return 0; }
BOOL WINAPI DllMain(HINSTANCE module,DWORD reason,void *reserved)
{
    (void)module;(void)reserved;
    if (reason==DLL_PROCESS_ATTACH) lifecycle_status("helper-process-attach",GetCurrentThreadId());
    if (reason==DLL_PROCESS_DETACH && InterlockedCompareExchange(&armed,0,1)==1) {
        int minimum=0,block=0;
        HMODULE cuda=GetModuleHandleA("nvcuda.dll");
        int (WINAPI *occupancy)(int *,int *,void *,size_t (WINAPI *)(int),size_t,int)=
            (void *)GetProcAddress(cuda,"cuOccupancyMaxPotentialBlockSize");
        if (!lifecycle_event(state.events,"detach-begin\n") || !occupancy) TerminateProcess(GetCurrentProcess(),93);
        int status=occupancy(&minimum,&block,(void *)(uintptr_t)UINT64_C(0x778899aabbccdd01),forbidden_worker_callback,0,0);
        lifecycle_status("loader-occupancy",(uintptr_t)(unsigned)status);
        if (status!=801 || !lifecycle_event(state.events,"loader-worker-refused\n")) TerminateProcess(GetCurrentProcess(),93);
        status=state.destroy(state.context);
        lifecycle_status("context-destroy",(uintptr_t)(unsigned)status);
        lifecycle_status("destructors-seen",(uintptr_t)seen);
        if (status || seen!=1 || !lifecycle_event(state.events,"detach-returned\n")) TerminateProcess(GetCurrentProcess(),93);
        static const char message[]="PASS: loader-lock destructor returned on the calling PE thread\n";
        DWORD written;
        if (!WriteFile(GetStdHandle(STD_OUTPUT_HANDLE),message,sizeof(message)-1,&written,NULL) || written!=sizeof(message)-1)
            TerminateProcess(GetCurrentProcess(),95);
    }
    return TRUE;
}

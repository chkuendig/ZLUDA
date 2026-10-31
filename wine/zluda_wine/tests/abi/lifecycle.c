/* SPDX-License-Identifier: MIT
 * Actual Wine lifecycle and private Unix transition probe. A Python watchdog
 * owns this process; a hang in DLL detach cannot become an apparent pass.
 */
#include "lifecycle.h"
#include <stdio.h>
#include <string.h>
#define REQUIRE(value) do {if (!(value)) {fprintf(stderr,"lifecycle failure %d: %s (%lu)\n",__LINE__,#value,GetLastError());return 1;}} while (0)
static HANDLE events;
static DWORD exiting_thread;
static const void *notify_handles[3];
static int (WINAPI *remove_notify)(const void *,uintptr_t *);
static volatile LONG sequence;
static void WINAPI notify(unsigned reason,void *data)
{
    uintptr_t id=(uintptr_t)data;LONG index=InterlockedIncrement(&sequence);
    uintptr_t ignored=UINT64_C(0xabcddcba12344321);
    if (reason || GetCurrentThreadId()!=exiting_thread || id!=(uintptr_t)index || index>2 ||
        remove_notify(notify_handles[index-1],&ignored) || ignored!=UINT64_C(0xabcddcba12344321) ||
        !lifecycle_event(events,index==1 ? "tls-first-self-remove\n":"tls-second-self-remove\n"))
        TerminateProcess(GetCurrentProcess(),94);
}
static DWORD WINAPI worker(void *unused)
{(void)unused;exiting_thread=GetCurrentThreadId();return 0;}
int main(int argc,char **argv)
{
    REQUIRE(argc==4);
    events=CreateFileA(argv[3],GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,NULL);
    REQUIRE(events!=INVALID_HANDLE_VALUE);
    HMODULE cuda=LoadLibraryA("nvcuda.dll");REQUIRE(cuda);
    int (WINAPI *init)(unsigned)=(void *)GetProcAddress(cuda,"cuInit");REQUIRE(init && !init(0));
    if (!strcmp(argv[1],"thread-exit")) {
        static const unsigned char uuid[16]={0x19,0x5b,0xcb,0xf4,0xd6,0x7d,0x02,0x4a,0xac,0xc5,0x1d,0x29,0xce,0xa6,0x31,0xae};
        int (WINAPI *get_table)(const void **,const void *)=(void *)GetProcAddress(cuda,"cuGetExportTable");
        const uintptr_t *table=NULL;
        REQUIRE(get_table && !get_table((const void **)&table,uuid) && table && table[0]==24);
        int (WINAPI *add)(const void **,uintptr_t,uintptr_t)=(void *)table[1];remove_notify=(void *)table[2];
        REQUIRE(!add(&notify_handles[0],(uintptr_t)notify,1) && !add(&notify_handles[1],(uintptr_t)notify,2));
        HANDLE thread=CreateThread(NULL,0,worker,NULL,0,NULL);REQUIRE(thread);
        REQUIRE(WaitForSingleObject(thread,10000)==WAIT_OBJECT_0 && sequence==2);CloseHandle(thread);
        // Both registrations removed themselves while refcounted. A second
        // actual thread exit must invoke nothing and must not touch freed data.
        thread=CreateThread(NULL,0,worker,NULL,0,NULL);REQUIRE(thread);
        REQUIRE(WaitForSingleObject(thread,10000)==WAIT_OBJECT_0 && sequence==2);CloseHandle(thread);
        REQUIRE(remove_notify(notify_handles[0],NULL)==1 && remove_notify(notify_handles[1],NULL)==1);
        REQUIRE(lifecycle_event(events,"tls-thread-exit-returned\n"));
    } else {
        REQUIRE(!strcmp(argv[1],"free-library") || !strcmp(argv[1],"exit-process"));
        HMODULE helper=LoadLibraryA(argv[2]);REQUIRE(helper);
        lifecycle_status("helper-load-returned",GetCurrentThreadId());
        lifecycle_setup_fn setup=(void *)GetProcAddress(helper,"fixture_setup");REQUIRE(setup);
        int (WINAPI *create)(void **,unsigned,int)=(void *)GetProcAddress(cuda,"cuCtxCreate_v2");
        int (WINAPI *destroy)(void *)=(void *)GetProcAddress(cuda,"cuCtxDestroy_v2");
        struct lifecycle_setup state={events,GetCurrentThreadId(),TlsAlloc(),NULL,(void *)0x1234,(void *)0x5678,destroy};
        REQUIRE(create && destroy && state.tls!=TLS_OUT_OF_INDEXES && !create(&state.context,0,0));
        REQUIRE(TlsSetValue(state.tls,state.value));
        lifecycle_status("tls-set",(uintptr_t)TlsGetValue(state.tls));
        /* This exported call is after LoadLibrary returned, never DllMain. */
        int status=setup(&state);
        lifecycle_status("setup-returned",(uintptr_t)(unsigned)status);
        REQUIRE(!status);
        if (!strcmp(argv[1],"exit-process")) {
            REQUIRE(lifecycle_event(events,"calling-exit-process\n"));
            // No manual reset or FreeLibrary: the loader must run the callback
            // after ExitProcess has terminated all other Windows threads.
            ExitProcess(0);
        }
        REQUIRE(FreeLibrary(helper));
        REQUIRE(lifecycle_event(events,"free-library-returned\n"));TlsFree(state.tls);
    }
    CloseHandle(events);puts("PASS: synchronous context/TLS lifecycle");return 0;
}

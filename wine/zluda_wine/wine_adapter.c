/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Independent projection of reviewed ABI facts, specification §§10 and 12.
 * No Wine headers or header-inline implementation enter this adapter.
 */
#include <stdint.h>
#include <stddef.h>
#define WINAPI __stdcall
#define IMPORT __declspec(dllimport)
typedef void *HANDLE;
typedef int32_t NTSTATUS;
typedef NTSTATUS (WINAPI *query_fn)(HANDLE,const void *,uint32_t,void *,size_t,size_t *);
typedef NTSTATUS (WINAPI *dispatch_fn)(uint64_t,uint32_t,void *);
IMPORT HANDLE WINAPI GetModuleHandleA(const char *);
IMPORT void * WINAPI GetProcAddress(HANDLE,const char *);
static uint64_t unix_handle;
static dispatch_fn dispatcher;
int32_t zluda_wine_attach(void *module)
{
    HANDLE ntdll=GetModuleHandleA("ntdll.dll");
    query_fn query=(void *)GetProcAddress(ntdll,"NtQueryVirtualMemory");
    dispatch_fn *slot=(void *)GetProcAddress(ntdll,"__wine_unix_call_dispatcher");
    if (!query || !slot || !*slot) return 0;
    if (query((HANDLE)(intptr_t)-1,module,1000,&unix_handle,8,NULL) || !unix_handle) return 0;
    dispatcher=*slot;
    return 1;
}
int32_t zluda_wine_call(void *packet)
{
    return dispatcher ? dispatcher(unix_handle,0,packet) : (int32_t)0xc000000d;
}
#ifdef _WIN64
IMPORT HANDLE WINAPI GetProcessWindowStation(void);
IMPORT int32_t WINAPI GetModuleHandleExW(uint32_t,const uint16_t *,HANDLE *);
IMPORT uint32_t WINAPI GetCurrentThreadId(void);
IMPORT uint8_t WINAPI RtlDllShutdownInProgress(void);
static unsigned char *peb(void)
{
    unsigned char *teb;
    __asm__("movq %%gs:0x30,%0" : "=r"(teb));
    return *(unsigned char **)(teb+0x60);
}
int32_t zluda_wine_callback_ready(void)
{
    HANDLE module;
    void **table;
    (void)GetProcessWindowStation();
    table=*(void ***)(peb()+0x58);
    if (!table || !table[0]) return 0;
    return GetModuleHandleExW(5,(const uint16_t *)(uintptr_t)zluda_wine_callback_ready,&module);
}
int32_t zluda_wine_worker_allowed(void)
{
    unsigned char *lock=*(unsigned char **)(peb()+0x110);
    return !RtlDllShutdownInProgress() && lock &&
        *(HANDLE *)(lock+16)!=(HANDLE)(uintptr_t)GetCurrentThreadId();
}
#endif

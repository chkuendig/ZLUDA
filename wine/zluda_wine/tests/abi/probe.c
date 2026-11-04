/* SPDX-License-Identifier: MIT
 * Real Wine PE64 probe. Every callback calls Windows APIs; a direct invocation
 * on the fixture backend's native pthread is therefore not a valid substitute.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "fixture failure at line %d: %s (last error %lu)\n", __LINE__, #condition, GetLastError()); return 1; \
} } while (0)
#define LOAD(type, name) type name = (void *)GetProcAddress(module, #name); REQUIRE(name != NULL)
typedef int (WINAPI *init_fn)(unsigned);
typedef int (WINAPI *out_int_fn)(int *);
typedef int (WINAPI *ctx_create_fn)(void **, unsigned, int);
typedef int (WINAPI *ctx_current_fn)(void **);
typedef int (WINAPI *ctx_destroy_fn)(void *);
typedef int (WINAPI *alloc_fn)(uint64_t *, size_t);
typedef int (WINAPI *free_fn)(uint64_t);
typedef int (WINAPI *htod_fn)(uint64_t, const void *, size_t);
typedef int (WINAPI *dtoh_fn)(void *, uint64_t, size_t);
typedef int (WINAPI *getproc_fn)(const char *, void **, int, uint64_t, unsigned *);
typedef int (WINAPI *table_fn)(const void **, const void *);
typedef int (WINAPI *pci_fn)(char *,int,int);
typedef void (WINAPI *stream_callback)(void *,int,void *);
typedef void (WINAPI *host_callback)(void *);
typedef int (WINAPI *stream_add_fn)(void *,stream_callback,void *,unsigned);
typedef int (WINAPI *host_add_fn)(void *,host_callback,void *);
typedef int (WINAPI *sync_fn)(void *);
typedef size_t (WINAPI *occupancy_callback)(int);
typedef int (WINAPI *occupancy_fn)(int *,int *,void *,occupancy_callback,size_t,int);
typedef void (WINAPI *log_callback)(void *,unsigned,char *,size_t);
typedef int (WINAPI *log_register_fn)(log_callback,void *,void **);
typedef int (WINAPI *log_unregister_fn)(void *);

static DWORD caller_id, tls;
struct callback_state { uint64_t before; volatile LONG seen,failed; uint64_t after; };
static void observe(struct callback_state *state,int valid)
{
    if (!valid || GetCurrentThreadId()==caller_id || !GetCurrentThreadId() ||
        !GetCurrentProcessId() || TlsGetValue(tls) != NULL ||
        state->before!=UINT64_C(0x1122334455667788) || state->after!=UINT64_C(0x8877665544332211))
        InterlockedIncrement(&state->failed);
    InterlockedIncrement(&state->seen);
}
static void WINAPI on_stream(void *stream,int status,void *data)
{ observe(data,stream==NULL && status==0); }
static void WINAPI on_host(void *data) { observe(data,1); }
static struct callback_state occupancy_state;
static size_t WINAPI on_occupancy(int block)
{ observe(&occupancy_state,block==17); return UINT64_C(0x123456789abc0000)+(unsigned)block; }
static void WINAPI on_log(void *data,unsigned level,char *text,size_t bytes)
{ observe(data,level==4 && bytes==sizeof("borrowed native log buffer") && !memcmp(text,"borrowed native log buffer",bytes)); }
static int pe_pointer(HMODULE module,const void *pointer)
{
    MEMORY_BASIC_INFORMATION info;
    return VirtualQuery(pointer,&info,sizeof(info))==sizeof(info) &&
        info.AllocationBase==module && info.Type==MEM_IMAGE;
}
static unsigned exports(HMODULE module)
{
    IMAGE_DOS_HEADER *dos=(void *)module;
    IMAGE_NT_HEADERS *nt=(void *)((unsigned char *)module+dos->e_lfanew);
    if (!nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress) return 0;
    IMAGE_EXPORT_DIRECTORY *exp=(void *)((unsigned char *)module+
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress);
    return exp->NumberOfNames;
}

int main(int argc,char **argv)
{
    HMODULE module=LoadLibraryA("nvcuda.dll");
    REQUIRE(module!=NULL);
    REQUIRE(!memcmp((unsigned char *)module+0x40,"Wine builtin DLL",16));
    if (argc==2 && !strcmp(argv[1],"--expect-stub")) {
        REQUIRE(exports(module)==0);
        REQUIRE(GetProcAddress(module,"cuInit")==NULL);
        puts("PASS: exact GE empty builtin surface, cuInit unavailable"); return 0;
    }
    REQUIRE(exports(module)==788);
    LOAD(init_fn,cuInit);
    REQUIRE(pe_pointer(module,(void *)cuInit));
    if (argc==2 && !strcmp(argv[1],"--expect-backend-refusal")) {
        REQUIRE(cuInit(0)==3);
        puts("PASS: on-root refuses an absent/relative backend without fallback"); return 0;
    }
    REQUIRE(argc==1);
    REQUIRE(GetProcAddress(module,"DllMain")==NULL);
    REQUIRE(GetProcAddress(module,"zluda_wine_call")==NULL);
    LOAD(out_int_fn,cuDriverGetVersion);
    LOAD(out_int_fn,cuDeviceGetCount);
    LOAD(ctx_create_fn,cuCtxCreate_v2);
    LOAD(ctx_current_fn,cuCtxGetCurrent);
    LOAD(ctx_destroy_fn,cuCtxDestroy_v2);
    LOAD(alloc_fn,cuMemAlloc_v2);
    LOAD(free_fn,cuMemFree_v2);
    LOAD(htod_fn,cuMemcpyHtoD_v2);
    LOAD(dtoh_fn,cuMemcpyDtoH_v2);
    LOAD(getproc_fn,cuGetProcAddress_v2);
    LOAD(table_fn,cuGetExportTable);
    LOAD(pci_fn,cuDeviceGetPCIBusId);
    REQUIRE(cuInit(0)==0 && cuInit(1)==1);
    int version=0;
    REQUIRE(cuDriverGetVersion(&version)==0 && version==11009977);
    struct { uint32_t before; int value; uint32_t after; } count={0x12345678,-1,0x87654321};
    REQUIRE(cuDeviceGetCount(&count.value)==0 && count.value==1);
    REQUIRE(count.before==0x12345678 && count.after==0x87654321);
    REQUIRE(cuDeviceGetPCIBusId(NULL,0,0)==801);
    void *context=NULL,*current=NULL;
    REQUIRE(cuCtxCreate_v2(&context,0,0)==0 && (uint64_t)(uintptr_t)context==UINT64_C(0x100000001));
    REQUIRE(cuCtxGetCurrent(&current)==0 && current==context);
    struct { uint64_t before,pointer,after; } allocation={0x1234,0,0x5678};
    REQUIRE(cuMemAlloc_v2(&allocation.pointer,32)==0);
    REQUIRE(allocation.before==0x1234 && allocation.after==0x5678 && allocation.pointer>UINT32_MAX);
    unsigned char input[32],output[32];
    for (unsigned i=0;i<sizeof(input);++i) input[i]=(unsigned char)(i*7+3);
    memset(output,0,sizeof(output));
    REQUIRE(cuMemcpyHtoD_v2(allocation.pointer,input,sizeof(input))==0);
    REQUIRE(cuMemcpyDtoH_v2(output,allocation.pointer,sizeof(output))==0);
    REQUIRE(!memcmp(input,output,sizeof(input)));
    REQUIRE(cuMemFree_v2(allocation.pointer)==0);
    void *function=NULL; unsigned status=99;
    REQUIRE(cuGetProcAddress_v2("cuDeviceGetCount",&function,2000,0,&status)==0);
    REQUIRE(status==0 && function==(void *)cuDeviceGetCount && pe_pointer(module,function));
    count.value=0;
    REQUIRE(((out_int_fn)function)(&count.value)==0 && count.value==1);
    REQUIRE(cuGetProcAddress_v2("cuMemAlloc",&function,3020,0,&status)==0);
    REQUIRE(status==0 && function==(void *)cuMemAlloc_v2);
    REQUIRE(cuGetProcAddress_v2("absent",&function,12000,0,&status)==0 && status==1 && !function);
    REQUIRE(cuGetProcAddress_v2("cuMemAlloc",&function,1000,0,&status)==0 && status==1 && !function);
    REQUIRE(cuGetProcAddress_v2("cuDeviceGetCount",&function,2000,2,&status)==0 && status==1 && !function);
    const void *table=(void *)(uintptr_t)UINT64_C(0x100000001);
    unsigned char unknown[16]={0};
    REQUIRE(cuGetExportTable(&table,unknown)==999 && table==NULL);
    static const unsigned char cudart[16]={0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9};
    REQUIRE(cuGetExportTable(&table,cudart)==0 && table!=NULL);
    const uintptr_t *words=table;
    REQUIRE(words[0]==13*sizeof(uintptr_t) && !words[3] && !words[9] && !words[10] && !words[11]);
    REQUIRE(pe_pointer(module,(void *)words[6]) && pe_pointer(module,(void *)words[7]));
    typedef int (WINAPI *ext1_fn)(void **,const void *,void *,void *,uint32_t);
    void *out=NULL;
    REQUIRE(((ext1_fn)words[6])(&out,(void *)(uintptr_t)UINT64_C(0x1122334455667788),
        (void *)(uintptr_t)UINT64_C(0x2233445566778899),(void *)(uintptr_t)UINT64_C(0x33445566778899aa),0xabcde123)==201);
    REQUIRE((uintptr_t)out==UINT64_C(0x445566778899aabb));
    ((void (WINAPI *)(uintptr_t))words[7])(UINT64_C(0xabcdef1234567890));
    static const unsigned char heap[16]={0x19,0x5b,0xcb,0xf4,0xd6,0x7d,0x02,0x4a,0xac,0xc5,0x1d,0x29,0xce,0xa6,0x31,0xae};
    REQUIRE(cuGetExportTable(&table,heap)==0 && table!=NULL);
    words=table; REQUIRE(words[0]==3*sizeof(uintptr_t) && pe_pointer(module,(void *)words[1]));
    typedef int (WINAPI *heap_alloc_fn)(const void **,uintptr_t,uintptr_t);
    typedef int (WINAPI *heap_free_fn)(const void *,uintptr_t *);
    const void *record=NULL; uintptr_t result=0;
    REQUIRE(((heap_alloc_fn)words[1])(&record,0x1234,UINT64_C(0x123456789abcdef0))==0);
    REQUIRE(HeapValidate(GetProcessHeap(),0,record));
    REQUIRE(((heap_free_fn)words[2])(record,&result)==0 && result==0);
    REQUIRE(((heap_free_fn)words[2])(record,NULL)==1);

    LOAD(stream_add_fn,cuStreamAddCallback);
    LOAD(host_add_fn,cuLaunchHostFunc);
    LOAD(sync_fn,cuStreamSynchronize);
    LOAD(occupancy_fn,cuOccupancyMaxPotentialBlockSize);
    LOAD(log_register_fn,cuLogsRegisterCallback);
    LOAD(log_unregister_fn,cuLogsUnregisterCallback);
    tls=TlsAlloc(); caller_id=GetCurrentThreadId();
    REQUIRE(tls!=TLS_OUT_OF_INDEXES && TlsSetValue(tls,(void *)(uintptr_t)0x12345));
    struct callback_state callbacks={UINT64_C(0x1122334455667788),0,0,UINT64_C(0x8877665544332211)};
    REQUIRE(cuStreamAddCallback(NULL,on_stream,&callbacks,1)==1 && callbacks.seen==0);
    REQUIRE(cuStreamAddCallback(NULL,on_stream,&callbacks,0)==0);
    REQUIRE(cuStreamSynchronize(NULL)==0 && callbacks.seen==1 && callbacks.failed==0);
    REQUIRE(cuLaunchHostFunc(NULL,on_host,&callbacks)==801);
    REQUIRE(cuStreamSynchronize(NULL)==0 && callbacks.seen==1 && callbacks.failed==0);
    occupancy_state=callbacks; occupancy_state.seen=0;
    int minimum=0,block=0;
    REQUIRE(cuOccupancyMaxPotentialBlockSize(&minimum,&block,(void *)(uintptr_t)UINT64_C(0x778899aabbccdd01),on_occupancy,0,0)==0);
    REQUIRE(minimum==0x12345 && block==17 && occupancy_state.seen==1 && occupancy_state.failed==0);
    void *handle=NULL;
    REQUIRE(cuLogsRegisterCallback(on_log,&callbacks,&handle)==0);
    REQUIRE((uintptr_t)handle==UINT64_C(0x12340000abcd4321) && callbacks.seen==2 && callbacks.failed==0);
    REQUIRE(cuLogsUnregisterCallback(handle)==0);
    REQUIRE(TlsGetValue(tls)==(void *)(uintptr_t)0x12345 && TlsFree(tls));
    REQUIRE(GetModuleHandleA("libgcc_s_seh-1.dll")==NULL && GetModuleHandleA("libwinpthread-1.dll")==NULL);
    REQUIRE(cuCtxDestroy_v2(context)==0);
    printf("PASS: real PE64 ABI/proc/dark/heap/callbacks; Windows caller thread %lu\n",caller_id);
    return 0;
}

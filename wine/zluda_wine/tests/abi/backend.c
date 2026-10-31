/* SPDX-License-Identifier: MIT
 * CPU-only independently typed SysV CUDA fixture, never a production backend.
 * Native pthreads deliberately have no Wine/Windows thread environment.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static _Thread_local void *current_context;
static unsigned char *allocation;
static size_t allocation_size;
typedef void (*storage_destructor)(void *,void *,void *);
static struct storage_record {void *context,*key,*value;storage_destructor destructor;} storage[8];
static void audit(const char *event)
{
    const char *path = getenv("ZLUDA_WINE_FIXTURE_AUDIT");
    if (!path) return;
    FILE *file = fopen(path, "a");
    if (file) { fprintf(file, "%s\n", event); fclose(file); }
}
__attribute__((constructor)) static void loaded(void) { audit("backend-loaded"); }
int cuInit(unsigned flags) { audit("cuInit"); return flags ? 1 : 0; }
int cuDriverGetVersion(int *version) { if (!version) return 1; *version = 11009977; return 0; }
int cuDeviceGetCount(int *count) { if (!count) return 1; *count = 1; return 0; }
int cuDeviceGet(int *device, int ordinal)
{ if (!device || ordinal != 0) return 101; *device = 0; return 0; }
int cuCtxCreate_v2(void **context, unsigned flags, int device)
{
    if (!context || flags || device) return 1;
    current_context = (void *)(uintptr_t)UINT64_C(0x100000001);
    *context = current_context; return 0;
}
int cuCtxGetCurrent(void **context)
{ if (!context) return 1; *context = current_context; return 0; }
int cuCtxDestroy_v2(void *context)
{
    if (context != current_context) return 201;
    unsigned count=0;
    for (unsigned i=0;i<8;++i) if (storage[i].context==context) ++count;
    if (count) audit("context-reset-begin");
    for (unsigned i=0;i<8;++i) if (storage[i].context==context) {
        struct storage_record saved=storage[i];storage[i]=(struct storage_record){0};
        audit("context-destructor-native");saved.destructor(context,saved.key,saved.value);
    }
    if (count) audit("context-reset-end");
    current_context=NULL;return 0;
}
static int storage_set(void *context,void *key,void *value,storage_destructor destructor)
{
    if (!context) context=current_context;
    if (!context || !value || !destructor) return 1;
    for (unsigned i=0;i<8;++i) if (!storage[i].context) {
        storage[i]=(struct storage_record){context,key,value,destructor};audit("context-set");return 0;
    }
    return 2;
}
static int storage_remove(void *context,void *key)
{
    if (!context) context=current_context;
    for (unsigned i=0;i<8;++i) if (storage[i].context==context && storage[i].key==key) {
        storage[i]=(struct storage_record){0};return 0;
    }
    return 0;
}
static int storage_get(void **out,void *context,void *key)
{
    if (!context) context=current_context;
    for (unsigned i=0;i<8;++i) if (storage[i].context==context && storage[i].key==key) {
        *out=storage[i].value;audit("context-get");return 0;
    }
    return 400;
}
static const struct {
    int (*set)(void *,void *,void *,storage_destructor);
    int (*remove)(void *,void *);
    int (*get)(void **,void *,void *);
    void *reserved;
} context_storage={storage_set,storage_remove,storage_get,NULL};
int cuMemAlloc_v2(uint64_t *address, size_t bytes)
{
    if (!address || !bytes || allocation) return 1;
    allocation = malloc(bytes);
    if (!allocation) return 2;
    allocation_size = bytes; *address = (uint64_t)(uintptr_t)allocation; return 0;
}
int cuMemFree_v2(uint64_t address)
{
    if (!allocation || address != (uint64_t)(uintptr_t)allocation) return 1;
    free(allocation); allocation = NULL; allocation_size = 0; return 0;
}
int cuMemcpyHtoD_v2(uint64_t address, const void *host, size_t bytes)
{
    if (!host || address != (uint64_t)(uintptr_t)allocation || bytes > allocation_size) return 1;
    memcpy(allocation, host, bytes); return 0;
}
int cuMemcpyDtoH_v2(void *host, uint64_t address, size_t bytes)
{
    if (!host || address != (uint64_t)(uintptr_t)allocation || bytes > allocation_size) return 1;
    memcpy(host, allocation, bytes); return 0;
}
/* The PE bridge must select from its own exact-version table, not call either
 * backend get-proc or expose a native function pointer to a Windows caller. */
int cuGetProcAddress_v2(const char *name, void **out, int version, uint64_t flags, unsigned *status)
{
    (void)name; (void)out; (void)version; (void)flags; (void)status;
    audit("BAD-native-getproc-called"); return 999;
}
int cuGetProcAddress(const char *name, void **out, int version, uint64_t flags)
{ return cuGetProcAddress_v2(name,out,version,flags,NULL); }

/* Explicit canonical CUDART table. Slots 6/8 have five arguments and slot 7
 * returns void; treating these as the old C relay's signatures is a failure. */
static int cubin(void **out, const void *wrapper) { (void)out; (void)wrapper; return 1; }
static int context(void **out, int device) { (void)out; (void)device; return 1; }
static int retain(int device, unsigned flags) { (void)device; (void)flags; return 1; }
static int reset(int device) { (void)device; return 1; }
static int ext1(void **out, const void *wrapper, void *arg3, void *arg4, uint32_t arg5)
{
    audit("dark-five-arguments");
    if (!out || (uintptr_t)wrapper != UINT64_C(0x1122334455667788) ||
        (uintptr_t)arg3 != UINT64_C(0x2233445566778899) ||
        (uintptr_t)arg4 != UINT64_C(0x33445566778899aa) || arg5 != UINT32_C(0xabcde123)) return 1;
    *out = (void *)(uintptr_t)UINT64_C(0x445566778899aabb); return 201;
}
static void void_slot(uintptr_t value)
{ audit(value == UINT64_C(0xabcdef1234567890) ? "dark-void-argument" : "BAD-dark-void-argument"); }
static int ext2(const void *wrapper, void **out, void *a, void *b, uint32_t flags)
{ return ext1(out,wrapper,a,b,flags); }
static int compilers(void) { return 1; }
static const struct {
    uintptr_t bytes;
    int (*cubin)(void **,const void *);
    int (*context)(void **,int);
    void *reserved3;
    int (*retain)(int,unsigned);
    int (*reset)(int);
    int (*ext1)(void **,const void *,void *,void *,uint32_t);
    void (*void_slot)(uintptr_t);
    int (*ext2)(const void *,void **,void *,void *,uint32_t);
    void *reserved9, *reserved10, *reserved11;
    int (*compilers)(void);
} cudart = {13*sizeof(uintptr_t),cubin,context,NULL,retain,reset,ext1,void_slot,ext2,NULL,NULL,NULL,compilers};
_Static_assert(sizeof(cudart)==13*sizeof(uintptr_t),"independent CUDART layout");
int cuGetExportTable(const void **out, const void *uuid)
{
    static const unsigned char id[16] = {0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9};
    static const unsigned char storage_id[16]={0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93};
    audit("native-dark-query");
    if (!out || !uuid) return 1;
    *out = NULL;
    if (!memcmp(storage_id,uuid,sizeof(storage_id))) {*out=&context_storage;return 0;}
    if (memcmp(id,uuid,sizeof(id))) return 1;
    *out = &cudart; return 0;
}

typedef void (*stream_callback)(void *,int,void *);
typedef void (*host_callback)(void *);
static pthread_t pending;
static int has_pending;
struct work { void *stream, *data; stream_callback stream_fn; host_callback host_fn; };
static void *run(void *opaque)
{
    struct work *work = opaque;
    audit("native-pthread-callback");
    if (work->stream_fn) work->stream_fn(work->stream,0,work->data);
    else work->host_fn(work->data);
    free(work); return NULL;
}
static int schedule(void *stream,stream_callback callback,host_callback host,void *data)
{
    if (has_pending) return 1;
    struct work *work = malloc(sizeof(*work));
    if (!work) return 2;
    *work = (struct work){stream,data,callback,host};
    if (pthread_create(&pending,NULL,run,work)) { free(work); return 2; }
    has_pending = 1; return 0;
}
int cuStreamAddCallback(void *stream,stream_callback callback,void *data,unsigned flags)
{ if (!callback || flags) return 1; return schedule(stream,callback,NULL,data); }
int cuStreamAddCallback_ptsz(void *s,stream_callback f,void *d,unsigned flags)
{ return cuStreamAddCallback(s,f,d,flags); }
int cuLaunchHostFunc(void *stream,host_callback callback,void *data)
{
    /* The production pinned backend is an input-independent unsupported
     * function. The bridge's attested status route must redact all inputs. */
    if (stream || callback || data) return 999;
    audit("unsupported-host-status");return 801;
}
int cuLaunchHostFunc_ptsz(void *s,host_callback f,void *d) { return cuLaunchHostFunc(s,f,d); }
int cuStreamIsCapturing(void *stream,unsigned *status)
{ (void)stream; if (!status) return 1; *status=0; return 0; }
int cuStreamIsCapturing_ptsz(void *stream,unsigned *status) { return cuStreamIsCapturing(stream,status); }
int cuStreamSynchronize(void *stream)
{
    (void)stream;
    if (!has_pending) return 0;
    int result = pthread_join(pending,NULL); has_pending=0; return result ? 999 : 0;
}
struct occupancy { size_t (*callback)(int); size_t result; };
static void *occupancy_thread(void *opaque)
{ struct occupancy *call=opaque; audit("native-pthread-occupancy"); call->result=call->callback(17); return NULL; }
int cuOccupancyMaxPotentialBlockSize(int *minimum,int *block,void *function,
    size_t (*callback)(int),size_t bytes,int limit)
{
    if (!minimum || !block || function!=(void *)(uintptr_t)UINT64_C(0x778899aabbccdd01) || !callback || bytes || limit) return 1;
    struct occupancy call = {callback,0}; pthread_t thread;
    if (pthread_create(&thread,NULL,occupancy_thread,&call) || pthread_join(thread,NULL)) return 999;
    if (call.result != UINT64_C(0x123456789abc0011)) return 1;
    *minimum=0x12345; *block=17; return 0;
}
typedef void (*log_callback)(void *,unsigned,char *,size_t);
static void *log_handle;
int cuLogsRegisterCallback(log_callback callback,void *data,void **out)
{
    if (!callback || !out || log_handle) return 1;
    char text[]="borrowed native log buffer";
    audit("inline-native-log-callback"); callback(data,4,text,sizeof(text));
    *out=log_handle=(void *)(uintptr_t)UINT64_C(0x12340000abcd4321); return 0;
}
int cuLogsUnregisterCallback(void *handle)
{ if (!log_handle || handle!=log_handle) return 1; log_handle=NULL; return 0; }

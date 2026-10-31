/* SPDX-License-Identifier: MIT
 * CPU-only real PE32/WoW64 probe. All pointers and CUdeviceptr are four bytes;
 * the backend must observe ELF64 addresses plus the bounded pool offset.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define REQUIRE(condition) do {if (!(condition)) { \
 fprintf(stderr,"PE32 fixture failure at line %d: %s (last error %lu)\n",__LINE__,#condition,GetLastError());return 1; \
}} while (0)
typedef uint32_t CUdeviceptr;
_Static_assert(sizeof(void *)==4 && sizeof(size_t)==4 && sizeof(CUdeviceptr)==4,"PE32 CUDA widths");
typedef int (WINAPI *init_fn)(unsigned);
typedef int (WINAPI *out_int_fn)(int *);
typedef int (WINAPI *device_fn)(int *,int);
typedef int (WINAPI *total_fn)(size_t *,int);
typedef int (WINAPI *ctx_create_fn)(void **,unsigned,int);
typedef int (WINAPI *ctx_current_fn)(void **);
typedef int (WINAPI *ctx_destroy_fn)(void *);
typedef int (WINAPI *alloc_fn)(CUdeviceptr *,size_t);
typedef int (WINAPI *free_fn)(CUdeviceptr);
typedef int (WINAPI *htod_fn)(CUdeviceptr,const void *,size_t);
typedef int (WINAPI *dtoh_fn)(void *,CUdeviceptr,size_t);
typedef int (WINAPI *range_fn)(CUdeviceptr *,size_t *,CUdeviceptr);
typedef int (WINAPI *getproc_fn)(const char *,void **,int,uint64_t);
typedef int (WINAPI *getproc2_fn)(const char *,void **,int,uint64_t,unsigned *);
typedef int (WINAPI *luid_fn)(char *,unsigned *,int);
typedef int (WINAPI *interop_fn)(void **,void *,unsigned);
typedef void (WINAPI *host_fn)(void *);
typedef int (WINAPI *launch_host_fn)(void *,host_fn,void *);
typedef int (WINAPI *exec_host_fn)(void *,void *,const void *);
typedef BOOL (WINAPI *wow_fn)(HANDLE,USHORT *,USHORT *);
static int pe_pointer(HMODULE module,const void *pointer)
{
 MEMORY_BASIC_INFORMATION info;
 return VirtualQuery(pointer,&info,sizeof(info))==sizeof(info) && info.AllocationBase==module && info.Type==MEM_IMAGE;
}
#define LOAD(type,name) type name=(void *)GetProcAddress(module,#name);REQUIRE(name && pe_pointer(module,(void *)name))
static IMAGE_NT_HEADERS32 *headers(HMODULE module)
{return (void *)((unsigned char *)module+((IMAGE_DOS_HEADER *)module)->e_lfanew);}
static unsigned exports(HMODULE module)
{
 DWORD rva=headers(module)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
 return rva?((IMAGE_EXPORT_DIRECTORY *)((unsigned char *)module+rva))->NumberOfNames:0;
}
static void receipt(unsigned names)
{
 printf("ARCHITECTURE: {\"process_machine\":\"0x014c\",\"native_machine\":\"0x8664\","
   "\"pointer_bits\":32,\"module_machine\":\"0x014c\",\"module_bits\":32,"
   "\"cu_deviceptr_bits\":32,\"exports\":%u}\n",names);
}
int main(int argc,char **argv)
{
 USHORT process=0,native=0;
 wow_fn wow=(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"),"IsWow64Process2");
 REQUIRE(wow && wow(GetCurrentProcess(),&process,&native));
 REQUIRE(process==IMAGE_FILE_MACHINE_I386 && native==IMAGE_FILE_MACHINE_AMD64);
 HMODULE module=LoadLibraryA("nvcuda.dll");REQUIRE(module);
 REQUIRE(!memcmp((unsigned char *)module+0x40,"Wine builtin DLL",16));
 REQUIRE(headers(module)->FileHeader.Machine==IMAGE_FILE_MACHINE_I386);
 REQUIRE(headers(module)->OptionalHeader.Magic==IMAGE_NT_OPTIONAL_HDR32_MAGIC);
 unsigned names=exports(module);receipt(names);
 if (argc==2 && !strcmp(argv[1],"--expect-stub")) {
  REQUIRE(names==0 && !GetProcAddress(module,"cuInit"));
  puts("PASS: exact GE empty PE32 builtin surface under WoW64, cuInit unavailable");return 0;
 }
 REQUIRE(names==690);
 REQUIRE(!GetProcAddress(module,"DllMain") && !GetProcAddress(module,"zluda_wine_call"));
 LOAD(init_fn,cuInit);
 if (argc==2 && !strcmp(argv[1],"--expect-backend-refusal")) {
  REQUIRE(cuInit(0)==801);
  puts("PASS: WoW64 on-root refuses absent/relative backend without fallback");return 0;
 }
 REQUIRE(argc==1);
 /* Every exported callable address belongs to this PE, never an ELF pointer. */
 IMAGE_EXPORT_DIRECTORY *exp=(void *)((unsigned char *)module+
  headers(module)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress);
 DWORD *name_rvas=(void *)((unsigned char *)module+exp->AddressOfNames);
 for (unsigned i=0;i<names;++i) {
  const char *name=(void *)((unsigned char *)module+name_rvas[i]);
  REQUIRE(pe_pointer(module,(void *)GetProcAddress(module,name)));
 }
 LOAD(out_int_fn,cuDriverGetVersion);LOAD(out_int_fn,cuDeviceGetCount);LOAD(device_fn,cuDeviceGet);
 LOAD(total_fn,cuDeviceTotalMem_v2);LOAD(ctx_create_fn,cuCtxCreate_v2);
 LOAD(ctx_current_fn,cuCtxGetCurrent);LOAD(ctx_destroy_fn,cuCtxDestroy_v2);
 LOAD(alloc_fn,cuMemAlloc_v2);LOAD(free_fn,cuMemFree_v2);
 LOAD(htod_fn,cuMemcpyHtoD_v2);LOAD(dtoh_fn,cuMemcpyDtoH_v2);LOAD(range_fn,cuMemGetAddressRange_v2);
 LOAD(getproc_fn,cuGetProcAddress);LOAD(getproc2_fn,cuGetProcAddress_v2);
 LOAD(luid_fn,cuDeviceGetLuid);LOAD(interop_fn,cuGraphicsD3D11RegisterResource);
 LOAD(launch_host_fn,cuLaunchHostFunc);LOAD(exec_host_fn,cuGraphExecHostNodeSetParams);
 REQUIRE(cuInit(0)==0 && cuInit(1)==1);
 struct {uint32_t before;int value;uint32_t after;} number={0x12345678,-1,0x87654321};
 REQUIRE(cuDriverGetVersion(&number.value)==0 && number.value==12020);
 REQUIRE(cuDeviceGetCount(&number.value)==0 && number.value==1);
 REQUIRE(cuDeviceGet(&number.value,0)==0 && number.value==0);
 REQUIRE(number.before==0x12345678 && number.after==0x87654321);
 struct {uint32_t before;size_t bytes;uint32_t after;} total={0x11223344,0,0x44332211};
 REQUIRE(cuDeviceTotalMem_v2(&total.bytes,0)==0 && total.bytes==512u*1024u*1024u-256);
 REQUIRE(total.before==0x11223344 && total.after==0x44332211);
 struct {uint32_t before;void *ctx;uint32_t after;} context={0x1234,NULL,0x5678};
 REQUIRE(cuCtxCreate_v2(&context.ctx,0,0)==0);
 REQUIRE(((uintptr_t)context.ctx&0xff000000u)==0x40000000u);
 REQUIRE(context.before==0x1234 && context.after==0x5678);
 void *current=NULL;REQUIRE(cuCtxGetCurrent(&current)==0 && current==context.ctx);
 struct {uint32_t before;CUdeviceptr pointer;uint32_t after;} allocation={0x1234,0,0x5678};
 REQUIRE(cuMemAlloc_v2(&allocation.pointer,40)==0 && allocation.pointer==256);
 REQUIRE(allocation.before==0x1234 && allocation.after==0x5678);
 unsigned char input[32],output[32];
 for (unsigned i=0;i<sizeof(input);++i) input[i]=(unsigned char)(i*7+3);
 memset(output,0,sizeof(output));
 REQUIRE(cuMemcpyHtoD_v2(allocation.pointer+4,input,sizeof(input))==0);
 REQUIRE(cuMemcpyDtoH_v2(output,allocation.pointer+4,sizeof(output))==0 && !memcmp(input,output,sizeof(input)));
 memset(output,0x77,sizeof(output));
 REQUIRE(cuMemcpyDtoH_v2(output,allocation.pointer+39,2)==1);
 for (unsigned i=0;i<sizeof(output);++i) REQUIRE(output[i]==0x77);
 CUdeviceptr base=0;size_t bytes=0;
 REQUIRE(cuMemGetAddressRange_v2(&base,&bytes,allocation.pointer+4)==0 && base==allocation.pointer && bytes==40);
 /* These belong to the original 603 unsupported PE32 names. */
 void *function=(void *)(uintptr_t)0x12345678;unsigned status=0x87654321;
 REQUIRE(cuGetProcAddress("cuInit",&function,2000,0)==801);
 REQUIRE(cuGetProcAddress_v2("cuInit",&function,12020,UINT64_C(2),&status)==801);
 REQUIRE(function==(void *)(uintptr_t)0x12345678 && status==0x87654321);
 char luid[8];memset(luid,0x44,sizeof(luid));unsigned mask=0x55555555;
 REQUIRE(cuDeviceGetLuid(luid,&mask,0)==801 && mask==0x55555555);
 for (unsigned i=0;i<sizeof(luid);++i) REQUIRE(luid[i]==0x44);
 REQUIRE(cuGraphicsD3D11RegisterResource(&function,NULL,0)==801 && function==(void *)(uintptr_t)0x12345678);
 REQUIRE(cuLaunchHostFunc(NULL,NULL,NULL)==801);
 REQUIRE(cuGraphExecHostNodeSetParams(NULL,NULL,NULL)==801);
 REQUIRE(cuMemFree_v2(allocation.pointer)==0);
 REQUIRE(cuCtxDestroy_v2(context.ctx)==0);
 puts("PASS: real PE32 WoW64 bounded CPU ABI/690 exports/32-bit pointers/ELF64 translation/refusals");
 return 0;
}

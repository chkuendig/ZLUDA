/* SPDX-License-Identifier: MIT
 * Test-only local ELF provider with Wine's canonical native callback ABI.
 */
#include <stdint.h>
#ifdef OMIT_CALLBACK
int fixture_without_callback = 1;
#else
int32_t KeUserModeCallback(uint32_t id,const void *args,uint32_t len,void **out,uint32_t *bytes)
{
    if (id || !args || len!=48 || !out || !bytes) return (int32_t)0xc000000d;
    typedef int32_t (__attribute__((ms_abi)) *callback)(void *,uint32_t);
    callback invoke=(callback)(uintptr_t)*(const uint64_t *)args;
    if (!invoke) return (int32_t)0xc000000d;
    *out=0;*bytes=0;
    return invoke((void *)args,len);
}
#endif

/* SPDX-License-Identifier: MIT
 * Strong DT_NEEDED import, as Wine's own Unix libraries use for ntdll.so.
 */
#include <stdint.h>
extern int32_t KeUserModeCallback(uint32_t,const void *,uint32_t,void **,uint32_t *);
void *fixture_callback_address(void) {return (void *)KeUserModeCallback;}
int32_t fixture_callback_invoke(uint32_t id,const void *args,uint32_t len,void **out,uint32_t *bytes)
{return KeUserModeCallback(id,args,len,out,bytes);}

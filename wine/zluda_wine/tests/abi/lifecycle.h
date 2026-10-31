/* SPDX-License-Identifier: MIT */
#ifndef ZLUDA_FIXTURE_LIFECYCLE_H
#define ZLUDA_FIXTURE_LIFECYCLE_H
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
struct lifecycle_setup {HANDLE events;DWORD caller,tls;void *context,*key,*value;int (WINAPI *destroy)(void *);};
typedef int (WINAPI *lifecycle_setup_fn)(const struct lifecycle_setup *);
static inline int lifecycle_event(HANDLE output,const char *text)
{
    DWORD bytes=(DWORD)lstrlenA(text),written=0;
    return WriteFile(output,text,bytes,&written,NULL) && bytes==written && FlushFileBuffers(output);
}
/* Fixed stack buffer + WriteFile only: also safe for failure diagnostics from
 * loader callouts. Diagnostics are separate from the strict event receipts. */
static inline void lifecycle_status(const char *name,uintptr_t value)
{
    static const char hex[]="0123456789abcdef";
    char text[160];unsigned n=0;DWORD written;
    const char *prefix="LIFECYCLE ";
    while (*prefix) text[n++]=*prefix++;
    while (*name && n<120) text[n++]=*name++;
    text[n++]='=';text[n++]='0';text[n++]='x';
    for (unsigned i=0;i<sizeof(value)*2;++i)
        text[n++]=hex[(value>>((sizeof(value)*2-1-i)*4))&15];
    text[n++]='\n';
    if (!WriteFile(GetStdHandle(STD_OUTPUT_HANDLE),text,n,&written,NULL) || written!=n)
        TerminateProcess(GetCurrentProcess(),96);
}
#endif

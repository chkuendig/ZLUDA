/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The two cuGetExportTable interfaces the lifecycle and churn fixtures use,
 * as the CUDA runtime sees them through the public cuGetExportTable entry.
 * UUIDs and slot shapes come from ZLUDA's own dark_api crate
 * (dark_api/src/lib.rs, upstream ZLUDA, MIT OR Apache-2.0); the Windows
 * prefixes follow the reviewed bridge contract (EQUIVALENCE.md):
 *   CONTEXT_LOCAL_STORAGE_INTERFACE_V0301: no size header, slots 0..2
 *   HEAP_ACCESS (TlsNotify on x64): [0] size, [1] set, [2] remove
 */
#ifndef W1_DARK_H
#define W1_DARK_H
#include "w1.h"

/* uuid string byte order, as dark_api compares them */
static const CUuuid W1_CLS_UUID = {{0xC6, 0x93, 0x33, 0x6E, 0x11, 0x21, 0xDF, 0x11, 0xA8, 0xC3, 0x68, 0xF3, 0x55,
                                    0xD8, 0x95, 0x93}};
static const CUuuid W1_HEAP_UUID = {{0x19, 0x5B, 0xCB, 0xF4, 0xD6, 0x7D, 0x02, 0x4A, 0xAC, 0xC5, 0x1D, 0x29, 0xCE,
                                     0xA6, 0x31, 0xAE}};

typedef void(CUDAAPI *w1_cls_dtor)(CUcontext, void *key, void *value);
typedef CUresult(CUDAAPI *w1_cls_put)(CUcontext, void *key, void *value, w1_cls_dtor);
typedef CUresult(CUDAAPI *w1_cls_delete)(CUcontext, void *key);
typedef CUresult(CUDAAPI *w1_cls_get)(void **value, CUcontext, void *key);

typedef void(CUDAAPI *w1_tls_cb)(void *zero, void *userdata);
typedef CUresult(CUDAAPI *w1_tls_set)(void **handle, w1_tls_cb, void *userdata);
typedef CUresult(CUDAAPI *w1_tls_remove)(void *handle, void *ignored);

struct w1_cls {
    w1_cls_put put;
    w1_cls_delete del;
    w1_cls_get get;
};

struct w1_tls {
    w1_tls_set set;
    w1_tls_remove remove;
};

static int w1_get_cls(struct w1_cls *out)
{
    const void *table = NULL;
    CUresult r = CU(cuGetExportTable, &table, &W1_CLS_UUID);
    w1_check("cls-export-table", r == CUDA_SUCCESS && table, "\"status\":%d", r);
    if (r || !table)
        return -1;
    void *const *slots = table;
    out->put = (w1_cls_put)slots[0];
    out->del = (w1_cls_delete)slots[1];
    out->get = (w1_cls_get)slots[2];
    return 0;
}

static int w1_get_tls(struct w1_tls *out)
{
    const void *table = NULL;
    CUresult r = CU(cuGetExportTable, &table, &W1_HEAP_UUID);
    w1_check("heap-export-table", r == CUDA_SUCCESS && table, "\"status\":%d", r);
    if (r || !table)
        return -1;
    void *const *slots = table;
    w1_emit("\"event\":\"heap-table\",\"size_word\":%llu", (unsigned long long)(uintptr_t)slots[0]);
    out->set = (w1_tls_set)slots[1];
    out->remove = (w1_tls_remove)slots[2];
    return 0;
}

#endif

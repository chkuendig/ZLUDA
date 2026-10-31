/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The 24 PE32 CUDAAPI entries whose stdcall cleanup width was corrected
 * (evidence/zluda-wine-license-20261030/PE32-WIDTH-FINDING.md). Columns:
 * export name, correct argument bytes (i386 __stdcall of the public cuda.h
 * prototype), historical wrong bytes, and the argument shape:
 *   P pointer/handle (4)  S 32-bit scalar, size_t or CUdeviceptr (4)
 *   Q 64-bit scalar (8)   L CUmemLocation by value (8)
 *   A 64-byte IPC handle by value (64)
 * pe32_stack.c checks that each shape sums to the correct byte count.
 */
#define W1_PE32_WIDTHS(X)                                              \
    X(cuGraphInstantiateWithFlags, 16, 12, "PPQ")                      \
    X(cuIpcOpenEventHandle, 68, 8, "PA")                               \
    X(cuIpcOpenMemHandle, 72, 12, "PAS")                               \
    X(cuIpcOpenMemHandle_v2, 72, 12, "PAS")                            \
    X(cuMemAddressReserve, 24, 20, "PSSSQ")                            \
    X(cuMemAdvise_v2, 20, 16, "SSSL")                                  \
    X(cuMemCreate, 20, 16, "PSPQ")                                     \
    X(cuMemDiscardAndPrefetchBatchAsync, 36, 32, "PPSPPSQP")           \
    X(cuMemDiscardAndPrefetchBatchAsync_ptsz, 36, 32, "PPSPPSQP")      \
    X(cuMemDiscardBatchAsync, 24, 20, "PPSQP")                         \
    X(cuMemDiscardBatchAsync_ptsz, 24, 20, "PPSQP")                    \
    X(cuMemExportToShareableHandle, 24, 20, "PQSQ")                    \
    X(cuMemGetHandleForAddressRange, 24, 20, "PSSSQ")                  \
    X(cuMemPoolExportToShareableHandle, 20, 16, "PPSQ")                \
    X(cuMemPoolImportFromShareableHandle, 20, 16, "PPSQ")              \
    X(cuMemPrefetchAsync_v2, 24, 20, "SSLSP")                          \
    X(cuMemPrefetchAsync_v2_ptsz, 24, 20, "SSLSP")                     \
    X(cuMemPrefetchBatchAsync, 36, 32, "PPSPPSQP")                     \
    X(cuMemPrefetchBatchAsync_ptsz, 36, 32, "PPSPPSQP")                \
    X(cuMemcpy3DBatchAsync_v2, 20, 16, "SPQP")                         \
    X(cuMemcpy3DBatchAsync_v2_ptsz, 20, 16, "SPQP")                    \
    X(cuMemcpy3DWithAttributesAsync, 16, 12, "PQP")                    \
    X(cuMemcpy3DWithAttributesAsync_ptsz, 16, 12, "PQP")               \
    X(cuMulticastBindAddr_v2, 32, 36, "QSSSSQ")

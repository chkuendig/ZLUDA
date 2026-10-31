/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Fake backend for the counter self-test only (named libcuda.so.1 in a temp dir). */
int cuInit(unsigned flags) { return (int)flags; }
int cuDriverGetVersion(int *v) { *v = 12000; return 0; }
int cuGetExportTable(const void **table, const void *uuid) { (void)uuid; *table = 0; return 0; }
int notCuda(void) { return 7; }

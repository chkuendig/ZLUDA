/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Self-test oracle for pe32_stack.exe, not a CUDA implementation: exports the
 * 24 names as __stdcall functions returning 801, popping either the correct
 * byte count (-DW1_STUB_CORRECT -> w1stub_correct.dll) or the historical
 * wrong count (-DW1_STUB_OLD -> w1stub_old.dll). Link with --kill-at so the
 * exports are undecorated like nvcuda's.
 */
#ifdef _WIN64
#error "i386 only"
#endif
#include <stdint.h>
#include "pe32_widths.h"

#define W1A1 uint32_t a1
#define W1A2 W1A1, uint32_t a2
#define W1A3 W1A2, uint32_t a3
#define W1A4 W1A3, uint32_t a4
#define W1A5 W1A4, uint32_t a5
#define W1A6 W1A5, uint32_t a6
#define W1A7 W1A6, uint32_t a7
#define W1A8 W1A7, uint32_t a8
#define W1A9 W1A8, uint32_t a9
#define W1A10 W1A9, uint32_t a10
#define W1A11 W1A10, uint32_t a11
#define W1A12 W1A11, uint32_t a12
#define W1A13 W1A12, uint32_t a13
#define W1A14 W1A13, uint32_t a14
#define W1A15 W1A14, uint32_t a15
#define W1A16 W1A15, uint32_t a16
#define W1A17 W1A16, uint32_t a17
#define W1A18 W1A17, uint32_t a18
#define W1ARGS(words) W1A##words
#define W1EXPAND(words) W1ARGS(words)

/* byte counts are multiples of 4 by construction; map bytes -> words */
#define W1W_8 2
#define W1W_12 3
#define W1W_16 4
#define W1W_20 5
#define W1W_24 6
#define W1W_32 8
#define W1W_36 9
#define W1W_68 17
#define W1W_72 18
#define W1WORDS(bytes) W1W_##bytes

#ifdef W1_STUB_CORRECT
#define X(name, correct, old, shape) \
    __declspec(dllexport) int __stdcall name(W1EXPAND(W1WORDS(correct))) { return 801; }
#elif defined(W1_STUB_OLD)
#define X(name, correct, old, shape) \
    __declspec(dllexport) int __stdcall name(W1EXPAND(W1WORDS(old))) { return 801; }
#else
#error "define W1_STUB_CORRECT or W1_STUB_OLD"
#endif

W1_PE32_WIDTHS(X)

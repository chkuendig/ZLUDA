/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * W1C_SLOTS argument-agnostic counting trampolines (x86_64 SysV):
 *   slot i:  lock incq w1c_counts[i]; jmp *w1c_real[i]
 * No register, stack or flags state beyond EFLAGS is touched, so any
 * signature (including varargs and struct returns) passes through unchanged.
 */
#define W1C_SLOTS 1024
    .text
    .altmacro
    .macro w1c_tramp i
    .globl w1c_tramp_\i
    .hidden w1c_tramp_\i
    .type w1c_tramp_\i, @function
w1c_tramp_\i:
    lock incq w1c_counts+8*\i(%rip)
    jmp *w1c_real+8*\i(%rip)
    .endm

    .set i, 0
    .rept W1C_SLOTS
    w1c_tramp %i
    .set i, i+1
    .endr

    .section .data.rel.ro,"aw"
    .globl w1c_tramps
    .hidden w1c_tramps
    .p2align 3
w1c_tramps:
    .macro w1c_addr i
    .quad w1c_tramp_\i
    .endm
    .set i, 0
    .rept W1C_SLOTS
    w1c_addr %i
    .set i, i+1
    .endr

    .bss
    .globl w1c_counts, w1c_real
    .hidden w1c_counts
    .hidden w1c_real
    .p2align 6
w1c_counts:
    .zero 8*W1C_SLOTS
w1c_real:
    .zero 8*W1C_SLOTS

    .section .note.GNU-stack,"",@progbits

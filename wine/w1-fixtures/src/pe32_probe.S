/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * int32_t w1_raw_stdcall(void *fn, const uint32_t *words, uint32_t nwords,
 *                        int32_t *delta, uint32_t pad_out[16])   (cdecl, i386)
 *
 * Reserves a 64-byte canary pad, records ESP (E0), pushes nwords argument
 * words, calls fn and stores ESP_after - E0 in *delta: 0 for a correct
 * __stdcall callee, negative when the callee pops too little, positive when
 * it pops too much. ESP is then restored from E0 (ESI, callee-saved), so the
 * probe survives a defective callee. The pad words are copied to pad_out so
 * the caller can verify that an over-popping callee did not write into them.
 */
    .text
    .globl _w1_raw_stdcall
_w1_raw_stdcall:
    pushl %ebp
    movl %esp, %ebp
    pushl %ebx
    pushl %esi
    pushl %edi
    subl $64, %esp
    movl $16, %ecx
1:  movl %ecx, %eax
    xorl $0x5A5A0000, %eax
    movl %eax, -4(%esp,%ecx,4)
    loop 1b
    movl %esp, %esi
    movl 16(%ebp), %ecx
    movl 12(%ebp), %edx
2:  testl %ecx, %ecx
    jz 3f
    pushl -4(%edx,%ecx,4)
    decl %ecx
    jmp 2b
3:  movl 8(%ebp), %edx
    call *%edx
    movl %esp, %ecx
    subl %esi, %ecx
    movl %esi, %esp
    movl 20(%ebp), %edx
    movl %ecx, (%edx)
    movl 24(%ebp), %edx
    movl $16, %ecx
4:  movl -4(%esp,%ecx,4), %ebx
    movl %ebx, -4(%edx,%ecx,4)
    loop 4b
    addl $64, %esp
    popl %edi
    popl %esi
    popl %ebx
    popl %ebp
    ret

/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Hand-written PTX for the W1 real-GPU fixtures. x64 callers use 64-bit
 * addressing; PE32 callers use .address_size 32 with 4-byte pointer
 * parameters, as a 32-bit CUDA application would. The CPU references are in
 * w1_ref_* below and must stay in sync with these kernels.
 */
#ifndef W1_PTX_H
#define W1_PTX_H

#ifdef _WIN64
#define W1P_AS "64"
#define W1P_PTR "u64"
#define W1P_REGS ".reg .b64 %a<8>;\n"
#define W1P_IDX(dst, base, idx) "mul.wide.u32 %a7, " idx ", 4;\nadd.u64 " dst ", " base ", %a7;\n"
#else
#define W1P_AS "32"
#define W1P_PTR "u32"
#define W1P_REGS ".reg .b32 %a<8>;\n"
#define W1P_IDX(dst, base, idx) "mul.lo.u32 %a7, " idx ", 4;\nadd.u32 " dst ", " base ", %a7;\n"
#endif

static const char w1_ptx[] =
    ".version 6.4\n"
    ".target sm_52\n"
    ".address_size " W1P_AS "\n"
    "\n"
    /* out[i] = (i * 0x9E3779B1) ^ seed for i < n */
    ".visible .entry w1_fill(.param ." W1P_PTR " out, .param .u32 n, .param .u32 seed)\n"
    "{\n"
    ".reg .pred %p<2>;\n.reg .b32 %r<8>;\n" W1P_REGS
    "ld.param." W1P_PTR " %a1, [out];\n"
    "ld.param.u32 %r1, [n];\n"
    "ld.param.u32 %r2, [seed];\n"
    "mov.u32 %r3, %ctaid.x;\nmov.u32 %r4, %ntid.x;\nmov.u32 %r5, %tid.x;\n"
    "mad.lo.u32 %r6, %r3, %r4, %r5;\n"
    "setp.ge.u32 %p1, %r6, %r1;\n"
    "@%p1 bra FILL_DONE;\n"
    "mul.lo.u32 %r7, %r6, 0x9E3779B1;\n"
    "xor.b32 %r7, %r7, %r2;\n"
    "cvta.to.global." W1P_PTR " %a2, %a1;\n"
    W1P_IDX("%a3", "%a2", "%r6")
    "st.global.u32 [%a3], %r7;\n"
    "FILL_DONE:\nret;\n"
    "}\n"
    "\n"
    /* y[i] = a * x[i] + y[i] (mod 2^32) for i < n */
    ".visible .entry w1_saxpy(.param ." W1P_PTR " y, .param ." W1P_PTR " x, .param .u32 a, .param .u32 n)\n"
    "{\n"
    ".reg .pred %p<2>;\n.reg .b32 %r<10>;\n" W1P_REGS
    "ld.param." W1P_PTR " %a1, [y];\n"
    "ld.param." W1P_PTR " %a2, [x];\n"
    "ld.param.u32 %r1, [a];\n"
    "ld.param.u32 %r2, [n];\n"
    "mov.u32 %r3, %ctaid.x;\nmov.u32 %r4, %ntid.x;\nmov.u32 %r5, %tid.x;\n"
    "mad.lo.u32 %r6, %r3, %r4, %r5;\n"
    "setp.ge.u32 %p1, %r6, %r2;\n"
    "@%p1 bra SAXPY_DONE;\n"
    "cvta.to.global." W1P_PTR " %a1, %a1;\n"
    "cvta.to.global." W1P_PTR " %a2, %a2;\n"
    W1P_IDX("%a3", "%a1", "%r6")
    W1P_IDX("%a4", "%a2", "%r6")
    "ld.global.u32 %r7, [%a4];\n"
    "ld.global.u32 %r8, [%a3];\n"
    "mad.lo.u32 %r9, %r1, %r7, %r8;\n"
    "st.global.u32 [%a3], %r9;\n"
    "SAXPY_DONE:\nret;\n"
    "}\n"
    "\n"
    /* 256-thread blocks: out[block] = sum(in[block*256 .. block*256+255]) */
    ".visible .entry w1_block_sum(.param ." W1P_PTR " in, .param ." W1P_PTR " out)\n"
    "{\n"
    ".reg .pred %p<2>;\n.reg .b32 %r<12>;\n" W1P_REGS
    ".shared .align 4 .b32 w1_sdata[256];\n"
    "ld.param." W1P_PTR " %a1, [in];\n"
    "ld.param." W1P_PTR " %a2, [out];\n"
    "cvta.to.global." W1P_PTR " %a1, %a1;\n"
    "cvta.to.global." W1P_PTR " %a2, %a2;\n"
    "mov.u32 %r1, %tid.x;\nmov.u32 %r2, %ctaid.x;\n"
    "shl.b32 %r3, %r2, 8;\nadd.u32 %r3, %r3, %r1;\n"
    W1P_IDX("%a3", "%a1", "%r3")
    "ld.global.u32 %r4, [%a3];\n"
    "mov.u32 %r5, w1_sdata;\n"
    "shl.b32 %r6, %r1, 2;\nadd.u32 %r6, %r5, %r6;\n"
    "st.shared.u32 [%r6], %r4;\n"
    "bar.sync 0;\n"
    "mov.u32 %r7, 128;\n"
    "SUM_LOOP:\n"
    "setp.ge.u32 %p1, %r1, %r7;\n"
    "@%p1 bra SUM_SKIP;\n"
    "shl.b32 %r8, %r7, 2;\nadd.u32 %r8, %r6, %r8;\n"
    "ld.shared.u32 %r9, [%r8];\nld.shared.u32 %r10, [%r6];\n"
    "add.u32 %r10, %r10, %r9;\nst.shared.u32 [%r6], %r10;\n"
    "SUM_SKIP:\n"
    "bar.sync 0;\n"
    "shr.u32 %r7, %r7, 1;\n"
    "setp.ne.u32 %p1, %r7, 0;\n"
    "@%p1 bra SUM_LOOP;\n"
    "setp.ne.u32 %p1, %r1, 0;\n"
    "@%p1 bra SUM_DONE;\n"
    "ld.shared.u32 %r11, [w1_sdata];\n"
    W1P_IDX("%a4", "%a2", "%r2")
    "st.global.u32 [%a4], %r11;\n"
    "SUM_DONE:\nret;\n"
    "}\n"
    "\n"
    /* every thread i < n adds 1 to *counter */
    ".visible .entry w1_atomic_count(.param ." W1P_PTR " counter, .param .u32 n)\n"
    "{\n"
    ".reg .pred %p<2>;\n.reg .b32 %r<8>;\n" W1P_REGS
    "ld.param." W1P_PTR " %a1, [counter];\n"
    "ld.param.u32 %r1, [n];\n"
    "mov.u32 %r3, %ctaid.x;\nmov.u32 %r4, %ntid.x;\nmov.u32 %r5, %tid.x;\n"
    "mad.lo.u32 %r6, %r3, %r4, %r5;\n"
    "setp.ge.u32 %p1, %r6, %r1;\n"
    "@%p1 bra ATOM_DONE;\n"
    "cvta.to.global." W1P_PTR " %a1, %a1;\n"
    "atom.global.add.u32 %r7, [%a1], 1;\n"
    "ATOM_DONE:\nret;\n"
    "}\n";

static uint32_t w1_ref_fill(uint32_t i, uint32_t seed)
{
    return (i * 0x9E3779B1u) ^ seed;
}

#endif

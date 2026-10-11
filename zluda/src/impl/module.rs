use super::driver;
use crate::r#impl::function;
use crate::r#impl::{driver::GlobalState, function::Function};
use cuda_types::{cuda::*, dark_api::FatbinFileHeader};
use dark_api::FunctionArgInfo;
use hip_runtime_sys::*;
use rustc_hash::FxHashMap;
use std::collections::hash_map;
use std::collections::HashMap;
use std::sync::Mutex;
use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    fs, mem,
    ops::ControlFlow,
};
use zluda_common::{CodeLibraryRef, CodeModuleRef, ZludaObject};

/// Work around an AMDGPU divergence bug in PhysX IOFX_bbox{,Sync}Kernel.
///
/// ============================================================
/// WORKAROUND for an AMDGPU LLVM backend miscompilation.
/// A value conditionally written only by tid=0 is wrongly classified as
/// uniform, gets SGPR-allocated, and the tid=0 store becomes visible to
/// every lane in the wave.
///
/// CANDIDATE upstream fixes (merged in llvm/llvm-project main, NOT yet
/// in our LLVM tree — cherry-pick stalled 2026-04-28 due to substantial
/// conflicts in rocm-rel-6.4 base):
///   - PR #180509 (f8416c8643534) "Replace DivergentValues with
///     UniformValues for conservative divergence queries" — inverts the
///     analysis to default to divergent when not proven uniform. Highest
///     leverage candidate.
///   - PR #174938 (da94edf535e7) + #174117 — "Jump over reducible cycles
///     when locating join blocks" — companion to already-cherry-picked
///     #139667.
/// Already cherry-picked but byte-identical IR for our case (refuted):
///   - PR #139667 (6421248c) "Fixed control-div early stop"
///
/// REMOVAL PLAN: when LLVM rebases past PR #180509 + #174938, retest
/// with patch DELETED + multi-run protocol (≥3 runs). If smoke renders
/// + game stability holds, delete the patch entirely.
///
/// Related tree artifacts:
///   - ptx/experiments/bbox/README.md (bisection record)
///   - ext/llvm-project branch zluda-physx-pr145902 (different bug)
///   - ~/.claude/plans/can-tou-peppy-milner.md (investigation plan)
/// ============================================================
///
/// The kernel's tail-phase "per-warp cross-warp merge" section contains a
/// `setp.eq.s64 %pN, %rdN, 0; @%pN bra BB_EXIT;` guard that is semantically
/// supposed to let only tid=0 fall through (other lanes have
/// rd=0 set at BB10_175 / equivalent and should branch to BB_EXIT).
///
/// On gfx1030 under LLVM's AMDGPU divergence analysis, the guard misfires
/// and some non-tid=0 lanes pass through. Those lanes then execute
/// `st.global.v4.f32 [%rdX]` with computed `rdX` pointers that are bogus
/// for those lanes, triggering a page-fault-worth-27GB-past-param_7 crash.
/// Three different PTX-level edits all fix it (explicit tid==0 guard,
/// re-zero rd before guard, swap guard to tid==0); the narrowest correct
/// fix is to leave the existing rd==0 guard in place and ADD a tid==0
/// guard right after it, so only a lane that is BOTH tid==0 AND has
/// rd != 0 falls through.
///
/// Detection: find every `setp.eq.s64 %pN, %rdM, 0;\n@%pN bra TARGET;`
/// that immediately follows a `BBLABEL:` line, within a `.visible .entry
/// IOFX_bbox*` function body. Insert the additional tid==0 guard after
/// the existing branch.
///
/// Narrow by entry-name prefix `IOFX_bbox` to avoid touching unrelated
/// kernels. Tracked and verified: applies cleanly to IOFX_bboxKernel
/// (BB10_184, rd305 guard, target BB10_186) and IOFX_bboxSyncKernel
/// (BB9_146, rd540 guard, target BB9_148).
pub fn patch_iofx_bbox_tail_merge(ptx: &str) -> Option<String> {
    // 2026-04-28: env-gated for retirement testing on zluda-physx-clean.
    // Set ZLUDA_IOFX_BBOX_GUARD=0 to disable. If smoke + game stability
    // unaffected, the patch can be deleted entirely.
    let env_off = std::env::var("ZLUDA_IOFX_BBOX_GUARD").map_or(false, |v| v == "0");
    if env_off {
        return None;
    }
    if !ptx.contains("IOFX_bbox") {
        return None;
    }
    // The regex would be ideal here but ptx_parser is the only regex-y dep
    // in the module chain. Do it by line scan instead.
    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = String::with_capacity(ptx.len() + 1024);
    let mut i = 0;
    let mut in_bbox_kernel = false;
    let mut brace_depth: i32 = 0;
    let mut patch_count = 0usize;
    // Counter for synthesized predicate/register names so multiple guards in
    // the same kernel don't collide on the same identifier.
    let mut guard_uid: usize = 0;
    while i < lines.len() {
        let line = lines[i];
        out.push_str(line);
        out.push('\n');
        let trimmed = line.trim_start();
        // Enter/leave IOFX_bbox* kernel. Kernel bodies are delimited by `{ ... }`
        // at file scope.
        if trimmed.starts_with(".visible .entry IOFX_bbox")
            || trimmed.starts_with(".entry IOFX_bbox")
        {
            in_bbox_kernel = true;
            brace_depth = 0;
        }
        if in_bbox_kernel {
            for c in line.chars() {
                if c == '{' {
                    brace_depth += 1;
                } else if c == '}' {
                    brace_depth -= 1;
                    if brace_depth <= 0 {
                        in_bbox_kernel = false;
                        break;
                    }
                }
            }
        }
        // Pattern A — BB10_184 style (guard a tid=0 store via rd305==0 check):
        //   ^BB<N>_<M>:\n
        //   ^setp.eq.s64\s+%p<X>, %rd<Y>, 0;\n
        //   ^@%p<X>\s+bra\s+BB<A>_<B>;\n
        if in_bbox_kernel && brace_depth >= 1 && is_bb_label(trimmed) && i + 2 < lines.len() {
            let setp_line = lines[i + 1].trim_start();
            let bra_line = lines[i + 2].trim_start();
            if let Some((pred_a, target_label)) = match_setp_eq_s64_zero_pair(setp_line, bra_line) {
                // Emit setp + bra as-is, then insert the additional tid==0 guard.
                out.push_str(lines[i + 1]);
                out.push('\n');
                out.push_str(lines[i + 2]);
                out.push('\n');
                emit_lane0_guard(&mut out, &target_label, guard_uid);
                guard_uid += 1;
                let _ = pred_a;
                i += 3;
                patch_count += 1;
                continue;
            }
        }
        // Pattern B — BB10_188 / BB9_202 style: `setp.eq.s32 %pX, %rY, -1; @%pX bra TARGET;`
        // where %rY is loaded from uninitialized sWarpFirstActorID LDS and used
        // shortly after as `mul.wide.u32 %rdZ, %rY, 16;` to compute a per-lane
        // global store pointer. Non-tid=0 lanes on AMDGPU have garbage rY; NV
        // happens to have 0/-1 which is either benign-race or filtered by the
        // -1 guard. Adding a tid==0 guard after the -1 check prevents the wild
        // stores.
        if in_bbox_kernel && brace_depth >= 1 && i + 1 < lines.len() {
            let setp_line = lines[i].trim_start();
            let bra_line = lines[i + 1].trim_start();
            if let Some((reg_name, target_label)) =
                match_setp_eq_s32_minus_one_pair(setp_line, bra_line)
            {
                // Confirmation: within the next 20 lines, the same %rY must be
                // used as `mul.wide.u32 %rd?, %rY, 16;`. This anchors the patch
                // to the wild-pointer computation and avoids touching unrelated
                // `-1` checks (e.g., r91/r797 initialization reads).
                let mut uses_in_mul_wide = false;
                let look_end = (i + 22).min(lines.len());
                for j in (i + 2)..look_end {
                    let l = lines[j].trim_start();
                    if is_bb_label(l) {
                        break;
                    }
                    if is_mul_wide_u32_by_reg_times_16(l, &reg_name) {
                        uses_in_mul_wide = true;
                        break;
                    }
                }
                if uses_in_mul_wide {
                    // NOTE: we've already pushed lines[i] (the setp) via the
                    // loop-top push. Now push lines[i+1] (the bra), then
                    // insert the tid==0 guard that also branches to
                    // `target_label`.
                    out.push_str(lines[i + 1]);
                    out.push('\n');
                    emit_lane0_guard(&mut out, &target_label, guard_uid);
                    guard_uid += 1;
                    i += 2;
                    patch_count += 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    if patch_count == 0 {
        None
    } else {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_iofx_bbox_tail_merge: applied {} guard(s)",
            patch_count
        ));
        Some(out)
    }
}

/// 2026-06-15 Class-4 bit31 SAFETY GUARD on the IOFX_bbox wild store.
///
/// Measured root cause (fault-VA correlation): IOFX_bboxKernel does
/// `mul.wide.u32 %rdX, %rY, 16; add.s64 %rdA,<out>,%rdX; st.global.v4.f32 [%rdA],…`
/// where %rY (= sWarpLastActorID[tid], unguarded `setp.ne -1` site) holds a
/// GARBAGE bit31-set value (a float bit-pattern ~0xC0000000 read as a u32 index)
/// for some lane → index*16 ≈ 48GB → page-not-present wild store = the Class-4
/// SIGABRT (last kernel IOFX_bboxKernel; fault VA ~48GB above every allocation).
///
/// A store index with bit31 set is UNAMBIGUOUSLY ILLEGAL (valid actor indices are
/// small), so skipping that store is a SAFETY fix, not value-mutation (sanctioned
/// by CLAUDE.md for the radix bit31 case). For each `mul.wide.u32 %rdX,%rY,16`
/// that feeds a `st.global` (a WRITE), inject before it a bit31 check that, when
/// set, branches PAST the store to the next basic-block boundary. The check lives
/// in a nested `{ .reg … }` block so no function-scope register renumbering is
/// needed. Gate: ZLUDA_BBOX_BIT31_GUARD=0.
pub fn patch_iofx_bbox_bit31_guard(ptx: &str) -> Option<String> {
    if std::env::var("ZLUDA_BBOX_BIT31_GUARD").map_or(false, |v| v == "0") {
        return None;
    }
    if !ptx.contains("IOFX_bbox") {
        return None;
    }
    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = String::with_capacity(ptx.len() + 2048);
    let mut in_bbox = false;
    let mut depth: i32 = 0;
    let mut uid = 0usize;
    let mut count = 0usize;
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim_start();
        if t.starts_with(".visible .entry IOFX_bbox") || t.starts_with(".entry IOFX_bbox") {
            in_bbox = true;
            depth = 0;
        }
        // Detect `mul.wide.u32 %rdX, %rY, 16;` that feeds a st.global within the BB.
        let guard_target: Option<(String, String)> = if in_bbox && depth >= 1 {
            parse_mul_wide_by_16(t).and_then(|(_rd, ry)| {
                // Look ahead (until a label / unconditional bra) for a st.global
                // (a WRITE), and capture the skip target (first label, or the
                // target of the first unconditional bra).
                let mut feeds_store = false;
                let mut skip: Option<String> = None;
                let mut k = i + 1;
                while k < lines.len() && k <= i + 30 {
                    let l = lines[k].trim_start();
                    if let Some(lbl) = l.strip_suffix(':') {
                        if !lbl.contains(char::is_whitespace) {
                            skip = Some(lbl.to_string());
                            break;
                        }
                    }
                    if l.starts_with("st.global") {
                        feeds_store = true;
                    }
                    if l.starts_with("bra.uni ") {
                        skip = Some(l["bra.uni ".len()..].trim_end_matches(';').to_string());
                        break;
                    }
                    k += 1;
                }
                if feeds_store {
                    skip.map(|s| (ry, s))
                } else {
                    None
                }
            })
        } else {
            None
        };
        if let Some((ry, skip)) = guard_target {
            out.push_str(&format!(
                "{{\n.reg .b32 %rbg{u};\n.reg .pred %pbg{u};\nand.b32 %rbg{u}, %{ry}, -2147483648;\nsetp.ne.s32 %pbg{u}, %rbg{u}, 0;\n@%pbg{u} bra {skip};\n}}\n",
                u = uid, ry = ry, skip = skip
            ));
            uid += 1;
            count += 1;
        }
        out.push_str(lines[i]);
        out.push('\n');
        if in_bbox {
            for c in lines[i].chars() {
                if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                    if depth <= 0 {
                        in_bbox = false;
                        break;
                    }
                }
            }
        }
        i += 1;
    }
    if count == 0 {
        return None;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] patch_iofx_bbox_bit31_guard: guarded {} bbox wild-store site(s)",
        count
    ));
    Some(out)
}

/// Parse `mul.wide.u32 %rdX, %rY, 16;` → (rdX-without-%, rY-without-%).
fn parse_mul_wide_by_16(t: &str) -> Option<(String, String)> {
    let rest = t.strip_prefix("mul.wide.u32 ")?.trim_end_matches(';');
    let mut p = rest.split(',').map(|s| s.trim());
    let rd = p.next()?.strip_prefix('%')?;
    let ry = p.next()?.strip_prefix('%')?;
    if p.next()? != "16" || p.next().is_some() {
        return None;
    }
    Some((rd.to_string(), ry.to_string()))
}

/// 2026-06-09 MS_4 underflow guard — surgical fix for the Class A wild-VA
/// scatter Bug at the root.
///
/// Background (see `project_class_a_is_genuine_physx_bug_2026_06_09.md` and the
/// large comment block at `function.rs:12780`): Batman: Arkham Knight's PhysX
/// 3.3 (Rocksteady-customized) ships a `CreateParticleShapesIncremental_MultiStage_4`
/// kernel with this scatter pattern:
///
/// ```ptx
/// atom.global.add.u32 %r8, [%rd19], -1;   // decrement counter, r8 = OLD value
/// add.s32 %r9, %r7, %r8;                  // r9 = first_slot + old_counter
/// add.s32 %r10, %r9, -1;                  // r10 = first_slot + old_counter - 1
/// mul.wide.u32 %rd20, %r10, 4;            // ZERO-EXTEND to 64-bit
/// add.s64 %rd21, %rd10, %rd20;
/// st.global.u32 [%rd21], %r1;             // OUTPUT[r10 * 4] = thread_id
/// ```
///
/// When the host launches MS_4 with `n_threads` greater than the actual number of
/// active particles (which Batman:AK does because APEX 1.4 bypasses PhysX's
/// `RemoveParticleShapes` API → `removed_count` never decrements → PhysX C++
/// computes `n_threads = active_count - removed_count = active_count` = full
/// 65000-particle ceiling), the over-dispatched threads underflow the counter:
///
///   - First over-dispatch: atom returns 0, decrements counter to 0xFFFFFFFF.
///     r10 = first_slot - 1. If first_slot=0, r10 = -1 = 0xFFFFFFFF as u32.
///     mul.wide.u32 zero-extends to 0x3FFFFFFFC ≈ +16 GiB.
///     OUTPUT + 16 GiB = wild scatter into unmapped pages → SIGABRT.
///   - Subsequent over-dispatches: atom returns 0xFFFFFFFF and decrements
///     further. r10 wraps further negative; same wild scatter.
///
/// The bug also fires on NVIDIA driver ≥ 461.09; pre-460 drivers placed the +16 GiB
/// target in benign unmapped memory (graceful kernel-side fault). Steam Community
/// Guide 2433732129 documents the NVIDIA workaround (downgrade to 456.55).
///
/// **Fix**: insert a `setp.lt.s32; @%p_new bra <ret_label>` immediately before the
/// `mul.wide.u32` to short-circuit the wild store when r10 has gone negative.
/// Cost: 1 setp + 1 conditional bra per MS_4 thread.
///
/// Pattern match:
/// ```text
///   add.s32 %r<X>, %r<Y>, -1;
///   mul.wide.u32 %rd<Z>, %r<X>, 4;
///   add.s64 %rd<W>, %rd<V>, %rd<Z>;
///   st.global.u32 [%rd<W>], %r<U>;
/// ```
/// Where the same `%r<X>` is used in the add and mul. Followed by a `BB<n>_<m>:`
/// label and `ret;` (the kernel's natural exit). The bra target is that ret label.
///
/// Env-gate: `ZLUDA_MS4_UNDERFLOW_GUARD=0` to disable.
///
/// Only matches inside `CreateParticleShapesIncremental_MultiStage_4` (or in tests,
/// any kernel containing that name substring) to avoid touching unrelated kernels.
pub fn patch_ms4_underflow_guard(ptx: &str) -> Option<String> {
    let env_off = std::env::var("ZLUDA_MS4_UNDERFLOW_GUARD").map_or(false, |v| v == "0");
    if env_off {
        return None;
    }
    if !ptx.contains("CreateParticleShapesIncremental_MultiStage_4") {
        return None;
    }
    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = String::with_capacity(ptx.len() + 1024);
    let mut in_ms4_kernel = false;
    let mut brace_depth: i32 = 0;
    let mut patch_count = 0usize;
    let mut guard_uid: usize = 0;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        // Enter MS_4 kernel
        if trimmed.starts_with(".visible .entry CreateParticleShapesIncremental_MultiStage_4")
            || trimmed.starts_with(".entry CreateParticleShapesIncremental_MultiStage_4")
        {
            in_ms4_kernel = true;
            brace_depth = 0;
        }
        if in_ms4_kernel {
            for c in line.chars() {
                if c == '{' {
                    brace_depth += 1;
                } else if c == '}' {
                    brace_depth -= 1;
                    if brace_depth <= 0 {
                        in_ms4_kernel = false;
                        break;
                    }
                }
            }
        }
        // Match the 4-line scatter pattern. We're at the `add.s32 r<X>, r<Y>, -1` line.
        if in_ms4_kernel && brace_depth >= 1 && i + 3 < lines.len() {
            let l0 = lines[i].trim_start();
            let l1 = lines[i + 1].trim_start();
            let l2 = lines[i + 2].trim_start();
            let l3 = lines[i + 3].trim_start();
            if let Some(r_underflow) = match_ms4_scatter_pattern(l0, l1, l2, l3) {
                // Find the bra target: the next `BB<n>_<m>:` label inside this
                // kernel that is immediately followed by `ret;`. The canonical
                // PTX from /tmp/zluda_ptx_dump/module_3.ptx places this right
                // after the store.
                let bra_target = find_kernel_ret_label(&lines, i + 4).unwrap_or_default();
                if !bra_target.is_empty() {
                    // Emit the original `add.s32` line as-is.
                    out.push_str(lines[i]);
                    out.push('\n');
                    // Emit our guard BEFORE the mul.wide.u32.
                    emit_ms4_underflow_guard(&mut out, &r_underflow, &bra_target, guard_uid);
                    guard_uid += 1;
                    // Emit the remaining 3 lines (mul, add, st) as-is.
                    out.push_str(lines[i + 1]);
                    out.push('\n');
                    out.push_str(lines[i + 2]);
                    out.push('\n');
                    out.push_str(lines[i + 3]);
                    out.push('\n');
                    i += 4;
                    patch_count += 1;
                    continue;
                }
            }
        }
        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    if patch_count == 0 {
        None
    } else {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_ms4_underflow_guard: applied {} guard(s)",
            patch_count
        ));
        Some(out)
    }
}

/// Match the 4-line MS_4 wild-scatter pattern. Returns the underflow-prone
/// register name (e.g. `%r10`) on match.
///
/// Lines must be:
///   l0: `add.s32 %r<X>, %r<Y>, -1;`
///   l1: `mul.wide.u32 %rd<Z>, %r<X>, 4;`
///   l2: `add.s64 %rd<W>, %rd<V>, %rd<Z>;`
///   l3: `st.global.u32 [%rd<W>], %r<U>;`
///
/// The same `%r<X>` must appear in l0 and l1. l3's `[%rd<W>]` must match l2's
/// destination `%rd<W>`.
fn match_ms4_scatter_pattern(l0: &str, l1: &str, l2: &str, l3: &str) -> Option<String> {
    let l0 = l0.trim_end_matches(';').trim();
    let l1 = l1.trim_end_matches(';').trim();
    let l2 = l2.trim_end_matches(';').trim();
    let l3 = l3.trim_end_matches(';').trim();
    // l0: add.s32 %rX, %rY, -1
    let l0_rest = l0.strip_prefix("add.s32")?.trim();
    let mut parts = l0_rest.split(',').map(str::trim);
    let r_x = parts.next()?;
    let _r_y = parts.next()?;
    let neg_one = parts.next()?;
    if neg_one != "-1" {
        return None;
    }
    if !r_x.starts_with('%') {
        return None;
    }
    // l1: mul.wide.u32 %rdZ, %rX, 4  (must use same %rX as l0)
    let l1_rest = l1.strip_prefix("mul.wide.u32")?.trim();
    let mut parts = l1_rest.split(',').map(str::trim);
    let rd_z = parts.next()?;
    let r_x2 = parts.next()?;
    let four = parts.next()?;
    if r_x2 != r_x || four != "4" {
        return None;
    }
    if !rd_z.starts_with('%') {
        return None;
    }
    // l2: add.s64 %rdW, %rdV, %rdZ  (must use same %rdZ as l1)
    let l2_rest = l2.strip_prefix("add.s64")?.trim();
    let mut parts = l2_rest.split(',').map(str::trim);
    let rd_w = parts.next()?;
    let _rd_v = parts.next()?;
    let rd_z2 = parts.next()?;
    if rd_z2 != rd_z {
        return None;
    }
    if !rd_w.starts_with('%') {
        return None;
    }
    // l3: st.global.u32 [%rdW], %rU  (must reference %rdW from l2)
    if !l3.starts_with("st.global.u32") {
        return None;
    }
    let bracket_form = format!("[{}]", rd_w);
    if !l3.contains(&bracket_form) {
        return None;
    }
    Some(r_x.to_string())
}

/// Find the kernel's natural ret label by scanning forward from `start_idx`
/// within the current kernel body. We're looking for:
///   `BB<n>_<m>:` followed by `ret;` (possibly with `.loc` directives between).
fn find_kernel_ret_label(lines: &[&str], start_idx: usize) -> Option<String> {
    let mut i = start_idx;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        // Match BB<n>_<m>: label
        if is_bb_label(trimmed) {
            // Scan ahead a few lines to verify `ret;` is reachable
            let label = trimmed.trim_end_matches(':').to_string();
            for j in (i + 1)..(i + 6).min(lines.len()) {
                let t = lines[j].trim();
                if t == "ret;" {
                    return Some(label);
                }
                // Skip .loc directives and empty lines
                if t.is_empty() || t.starts_with(".loc") {
                    continue;
                }
                break;
            }
        }
        // Stop if we hit end-of-kernel
        if trimmed.starts_with('}') {
            return None;
        }
        i += 1;
    }
    None
}

fn emit_ms4_underflow_guard(out: &mut String, r_underflow: &str, ret_label: &str, uid: usize) {
    out.push_str("// ZLUDA MS_4 underflow guard\n");
    out.push_str("{\n");
    out.push_str(&format!(".reg .pred %p_ms4_uf_{};\n", uid));
    out.push_str(&format!(
        "setp.lt.s32 %p_ms4_uf_{}, {}, 0;\n",
        uid, r_underflow
    ));
    out.push_str(&format!(
        "@%p_ms4_uf_{} bra {};\n",
        uid, ret_label
    ));
    out.push_str("}\n");
}

fn emit_lane0_guard(out: &mut String, target_label: &str, uid: usize) {
    out.push_str("// ZLUDA IOFX_bbox fix: AMDGPU divergence guard\n");
    out.push_str("{\n");
    out.push_str(&format!(".reg .pred %p_zluda_bbox_lane0_{};\n", uid));
    out.push_str(&format!(".reg .s32 %r_zluda_bbox_tid_{};\n", uid));
    out.push_str(&format!("mov.u32 %r_zluda_bbox_tid_{}, %tid.x;\n", uid));
    out.push_str(&format!(
        "setp.ne.s32 %p_zluda_bbox_lane0_{}, %r_zluda_bbox_tid_{}, 0;\n",
        uid, uid
    ));
    out.push_str(&format!(
        "@%p_zluda_bbox_lane0_{} bra {};\n",
        uid, target_label
    ));
    out.push_str("}\n");
}

/// Parses:
///   `setp.eq.s32<ws>%p<digits>, %r<digits>, -1;`
///   `@%p<same_digits> bra BB<digits>_<digits>;`
/// Returns (register_name_with_%, target_label) on match.
fn match_setp_eq_s32_minus_one_pair(setp_line: &str, bra_line: &str) -> Option<(String, String)> {
    let setp = setp_line.trim_end();
    let rest = setp.strip_prefix("setp.eq.s32")?;
    let rest = rest.trim_start();
    let mut parts = rest.splitn(3, ',');
    let p_tok = parts.next()?.trim();
    let r_tok = parts.next()?.trim();
    let minus_tok = parts.next()?.trim();
    if !p_tok.starts_with("%p") {
        return None;
    }
    let pred_name = &p_tok[1..];
    if !r_tok.starts_with("%r") || r_tok.starts_with("%rd") {
        return None;
    }
    if !(minus_tok.starts_with("-1;") || minus_tok == "-1") {
        return None;
    }
    let bra = bra_line.trim_end();
    let bra_rest = bra.strip_prefix("@%")?;
    let bra_rest = bra_rest.strip_prefix(pred_name)?;
    let bra_rest = bra_rest.trim_start();
    let bra_rest = bra_rest.strip_prefix("bra")?;
    let target = bra_rest.trim().trim_end_matches(';').trim();
    if !target.starts_with("BB") {
        return None;
    }
    Some((r_tok.to_string(), target.to_string()))
}

/// Matches `mul.wide.u32 %rd<N>, <reg>, 16;` where <reg> is the given register
/// identifier (e.g. "%r94"). The destination is irrelevant — we only care that
/// <reg> is the left-hand multiplicand being scaled by 16.
fn is_mul_wide_u32_by_reg_times_16(line: &str, reg: &str) -> bool {
    let line = line.trim_end();
    let rest = match line.strip_prefix("mul.wide.u32") {
        Some(r) => r.trim_start(),
        None => return false,
    };
    let mut parts = rest.splitn(3, ',');
    let _dest = match parts.next() {
        Some(s) => s.trim(),
        None => return false,
    };
    let src1 = match parts.next() {
        Some(s) => s.trim(),
        None => return false,
    };
    let src2 = match parts.next() {
        Some(s) => s.trim().trim_end_matches(';').trim(),
        None => return false,
    };
    src1 == reg && src2 == "16"
}

fn is_bb_label(s: &str) -> bool {
    // Matches "BB<digits>_<digits>:" with optional leading `$`, followed by nothing else.
    let s = s.trim_end();
    if !s.ends_with(':') {
        return false;
    }
    let name = &s[..s.len() - 1];
    if !name.starts_with("BB") {
        return false;
    }
    let rest = &name[2..];
    let mut parts = rest.split('_');
    let first = parts.next().unwrap_or("");
    let second = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return false;
    }
    !first.is_empty()
        && !second.is_empty()
        && first.chars().all(|c| c.is_ascii_digit())
        && second.chars().all(|c| c.is_ascii_digit())
}

/// Parses:
///   `setp.eq.s64<ws>%p<digits>, %rd<digits>, 0;`
///   `@%p<same_digits> bra BB<digits>_<digits>;`
/// Returns (predicate_name_without_%, target_label) on match.
fn match_setp_eq_s64_zero_pair(setp_line: &str, bra_line: &str) -> Option<(String, String)> {
    let setp = setp_line.trim_end();
    // Expect prefix "setp.eq.s64" (possibly with tab/space separator).
    let rest = setp.strip_prefix("setp.eq.s64")?;
    let rest = rest.trim_start();
    // "%pNN, %rdMM, 0;"
    let mut parts = rest.splitn(3, ',');
    let p_tok = parts.next()?.trim();
    let rd_tok = parts.next()?.trim();
    let zero_tok = parts.next()?.trim();
    if !p_tok.starts_with("%p") {
        return None;
    }
    let pred_name = &p_tok[1..]; // "pNN" (without the leading %)
    if !rd_tok.starts_with("%rd") {
        return None;
    }
    if !zero_tok.starts_with("0;") && zero_tok != "0" {
        return None;
    }
    // bra line must be `@%pNN bra BBX_Y;`.
    let bra = bra_line.trim_end();
    let bra_rest = bra.strip_prefix("@%")?;
    let bra_rest = bra_rest.strip_prefix(pred_name)?;
    let bra_rest = bra_rest.trim_start();
    let bra_rest = bra_rest.strip_prefix("bra")?;
    let target = bra_rest.trim().trim_end_matches(';').trim();
    if !target.starts_with("BB") {
        return None;
    }
    Some((pred_name.to_string(), target.to_string()))
}

/// 2026-04-29: convert `ld.shared.<T>` → `ld.volatile.shared.<T>` across
/// PhysX/FleX/TFS/IOFX/ParticleIOS kernels affected by the cross-wave LDS
/// visibility class. The membar.cta + s_waitcnt fixes target HW-level
/// visibility but inconsistent results show silicon-level intermittence —
/// adding the volatile qualifier defeats any compiler-level CSE / dead-load
/// elimination that might otherwise hoist a stale read past a barrier.
///
/// First applied IOFX-only (5/5 runs zero IOFX wild-scatter, but Mode B
/// SIGSEGVs continued from TFS/ParticleIOS family kernels with the same
/// LDS visibility class). Broadened to match the same kernel prefixes
/// as patch_broad_membar_before_bar_sync.
///
/// Skip CUB / cuda::* — those are not our suspect class.
///
/// Env-gate: `ZLUDA_BROAD_VOLATILE_LDS=1` (off by default since the retirement profile).
///
/// Originally kernel-name-targeted (PhysX/TFS/IOFX/FleX prefixes, skip
/// `_ZN3cub`). After 2026-05-05's SubResource interop landing made PhysX
/// exercise CUB Radix Sort with larger inputs, CUB kernels started
/// crashing — the same memory-model bug class the prefix-list workaround
/// was masking elsewhere. Broadened to apply to ALL kernels: per user
/// direction, prefer one generic memory-model fix over many per-kernel
/// patches. The opt-out for `_ZN3cub` and `cuda::*` is dropped.
///
/// Risk: makes legitimate CUB scan/sort code slower on workloads that
/// don't trigger the AMDGPU race. For our hardware (gfx1033) and goal
/// (gameplay correctness over benchmark perf), correctness wins.
pub fn patch_broad_volatile_lds(ptx: &str) -> Option<String> {
    // Off by default since the validated retirement profile; =1 enables.
    let env_off = std::env::var("ZLUDA_BROAD_VOLATILE_LDS").map_or(true, |v| v != "1");
    if env_off {
        return None;
    }
    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = String::with_capacity(ptx.len() + 256);
    let mut in_kernel = false;
    let mut brace_depth: i32 = 0;
    let mut total_patches = 0usize;
    let mut patched_kernels: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut current_name = String::new();
    for line in &lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with(".visible .entry ") || trimmed.starts_with(".entry ") {
            let after = if let Some(rest) = trimmed.strip_prefix(".visible .entry ") {
                rest
            } else if let Some(rest) = trimmed.strip_prefix(".entry ") {
                rest
            } else {
                ""
            };
            let cn = after
                .trim_start()
                .split(|c: char| c == '(' || c.is_whitespace())
                .next()
                .unwrap_or("")
                .to_string();
            current_name = cn.clone();
            /* Apply to ALL kernels — we want one generic fix, not per-kernel
             * exclusions. See doc comment above for the rationale. */
            in_kernel = true;
            brace_depth = 0;
        }
        if in_kernel {
            for c in line.chars() {
                if c == '{' {
                    brace_depth += 1;
                } else if c == '}' {
                    brace_depth -= 1;
                    if brace_depth <= 0 {
                        in_kernel = false;
                        break;
                    }
                }
            }
        }
        if in_kernel && brace_depth >= 1 {
            // Match a leading `ld.shared.<type>` (no existing volatile/acquire
            // qualifier). Convert to `ld.volatile.shared.<type>`.
            //
            // 2026-04-29: tried adding `st.shared` → `st.volatile.shared` too,
            // but that REGRESSED IOFX_actorRangeKernel back to Mode A SIGABRT
            // wild scatter. Atomic store semantics on AMDGPU appear to break
            // something at this layer. Keeping load-only as the working state.
            let leading_ws_len = line.len() - line.trim_start().len();
            let body = &line[leading_ws_len..];
            if body.starts_with("ld.shared.") {
                let prefix = &line[..leading_ws_len];
                let rest = &body["ld.shared.".len()..];
                out.push_str(&format!("{}ld.volatile.shared.{}\n", prefix, rest));
                patched_kernels.insert(current_name.clone());
                total_patches += 1;
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    if total_patches == 0 {
        None
    } else {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_broad_volatile_lds: converted {} ld.shared → ld.volatile.shared across {} kernels",
            total_patches,
            patched_kernels.len()
        ));
        Some(out)
    }
}

/// 2026-06-11: scoped producer-store volatility for the cross-warp LDS
/// handoff that feeds the radix-sort SCATTER output offset.
///
/// Root cause (r19/r20b in-game NRSS_VALIDATE=2, per-pass localization):
/// `IOFX_newRadixSortStepKernel` phase-3 scatter intermittently writes
/// wrong per-thread output offsets under co-residency — with a VALID
/// global histogram and NO foreign values (internal, not an external
/// clobber). The per-digit base `sGlobalOffsets[digit]` is written by a
/// subset of threads via a NON-volatile `st.shared` and read cross-warp
/// by `ld.volatile.shared` after only a `bar.sync 0`. On gfx10
/// (FeatureBackOffBarrier) the barrier does not reliably flush a
/// non-volatile LDS store before the cross-warp read → stale base →
/// wrong scatter offset → collisions / mis-order. `patch_broad_volatile_lds`
/// deliberately left STORES non-volatile (a BLANKET st.shared→volatile
/// regressed actorRange, 2026-04-29). This patch is the SCOPED fix:
/// convert ONLY `st.shared` producers that sit immediately before a
/// `bar.sync` (the producer→barrier→cross-warp-read handoff) — which
/// catches the sGlobalOffsets base stores feeding the scatter address
/// without touching the far-apart warp-total stores or unrelated kernels.
///
/// Generic (pattern-based, ALL kernels — no name gate). Env-gate
/// `ZLUDA_LDS_PRODUCER_VOLATILE`: "1" = address-matched (weak — PTX
/// recomputes LDS addrs into fresh regs, so it misses most pairs),
/// "2" = ALL st.shared → volatile (full coverage).
///
/// VALIDATED 2026-06-11 (r22, mode=2): cuts the NRSS scatter corruption
/// 100-1000× (break-pass violations thousands → 3-318, no key loss) with
/// NO actorRange regression (reached 91k launches) — the April blanket-
/// volatile→actorRange regression does NOT recur in the current codebase.
/// Default OFF pending n≥3 game validation before flipping default-ON.
///
/// RESIDUAL (does NOT close with full LDS volatile, so it is NOT a store-
/// visibility bug): the scatter's local rank comes from a `shfl.up`
/// warp-scan that lowers to `__builtin_amdgcn_ds_bpermute`
/// (zluda_ptx_impl.cpp). ds_bpermute reads source-lane VGPRs without
/// guarding against INACTIVE (EXEC-off) lanes; at tile boundaries where
/// the wave partially diverges, an in-bounds-but-inactive source lane
/// returns 0 → wrong local rank → wrong scatter offset (the 3-318
/// residual). The generic fix is in the shfl translation (mask the
/// permute index / result by the active-lane set), NOT here.
pub fn patch_lds_producer_volatile_store(ptx: &str) -> Option<String> {
    let env_on =
        std::env::var("ZLUDA_LDS_PRODUCER_VOLATILE").map_or(false, |v| v == "1" || v == "2");
    if !env_on {
        return None;
    }
    // Address-matched producer↔consumer pairing. `patch_broad_volatile_lds`
    // already made the cross-warp CONSUMER loads `ld.volatile.shared`; this
    // converts a `st.shared.<T> [%addr], ...` to volatile exactly when the
    // SAME `[%addr]` operand is read by an `ld.volatile.shared` later in the
    // same kernel. That precisely covers BOTH the sGlobalOffsets base stores
    // AND the warp-total stores feeding the local rank, without a fragile
    // line-distance heuristic, and without touching producer stores whose
    // value is never read cross-warp (which is what regressed actorRange
    // when ALL st.shared were blindly volatilized).
    // Mode select: "1" = address-matched (producer addr also read by a
    // volatile load — precise but misses pairs where PTX recomputes the LDS
    // address into a fresh register between store and load, which is most of
    // them). "2" = ALL st.shared → volatile (full coverage; the decisive
    // test of whether store-volatility fixes the scatter race — the April
    // actorRange regression is re-measurable now via NRSS_VALIDATE +
    // actorRange fault watch).
    let full = std::env::var("ZLUDA_LDS_PRODUCER_VOLATILE")
        .map_or(false, |v| v == "2");
    fn addr_operand(body: &str) -> Option<&str> {
        // `<op> [<addr>], <val>;` — extract the bracketed address token.
        let lb = body.find('[')?;
        let rb = body[lb..].find(']')? + lb;
        Some(body[lb + 1..rb].trim())
    }
    if full {
        let mut out = String::with_capacity(ptx.len() + 256);
        let mut total = 0usize;
        for line in ptx.lines() {
            let lead = line.len() - line.trim_start().len();
            let body = &line[lead..];
            if body.starts_with("st.shared.") {
                out.push_str(&line[..lead]);
                out.push_str("st.volatile.shared.");
                out.push_str(&body["st.shared.".len()..]);
                out.push('\n');
                total += 1;
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        if total == 0 {
            return None;
        }
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_lds_producer_volatile_store: FULL mode — {} st.shared → st.volatile.shared",
            total
        ));
        return Some(out);
    }
    let lines: Vec<&str> = ptx.lines().collect();
    // Per-kernel set of addresses read by a volatile shared load. We scope by
    // kernel (brace depth) so an address reused in another kernel doesn't leak.
    let mut out = String::with_capacity(ptx.len() + 256);
    let mut total = 0usize;
    let mut i = 0usize;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        let is_entry =
            trimmed.starts_with(".visible .entry ") || trimmed.starts_with(".entry ");
        if !is_entry {
            out.push_str(lines[i]);
            out.push('\n');
            i += 1;
            continue;
        }
        // Find this kernel's body span [i .. end] by brace matching.
        let start = i;
        let mut depth: i32 = 0;
        let mut end = i;
        let mut opened = false;
        for (k, l) in lines.iter().enumerate().skip(i) {
            for c in l.chars() {
                if c == '{' {
                    depth += 1;
                    opened = true;
                } else if c == '}' {
                    depth -= 1;
                }
            }
            if opened && depth <= 0 {
                end = k;
                break;
            }
        }
        if end <= start {
            end = lines.len() - 1;
        }
        // Collect volatile-load addresses in this kernel body.
        let mut volatile_load_addrs: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        for l in &lines[start..=end] {
            let b = l.trim_start();
            if b.starts_with("ld.volatile.shared.") {
                if let Some(a) = addr_operand(b) {
                    volatile_load_addrs.insert(a);
                }
            }
        }
        // Emit body, volatilizing matching producer stores.
        for l in &lines[start..=end] {
            let lead = l.len() - l.trim_start().len();
            let body = &l[lead..];
            if body.starts_with("st.shared.") {
                if let Some(a) = addr_operand(body) {
                    if volatile_load_addrs.contains(a) {
                        out.push_str(&l[..lead]);
                        out.push_str("st.volatile.shared.");
                        out.push_str(&body["st.shared.".len()..]);
                        out.push('\n');
                        total += 1;
                        continue;
                    }
                }
            }
            out.push_str(l);
            out.push('\n');
        }
        i = end + 1;
    }
    if total == 0 {
        None
    } else {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_lds_producer_volatile_store: {} pre-barrier st.shared → st.volatile.shared",
            total
        ));
        Some(out)
    }
}

/// 2026-04-29: insert `membar.cta;` immediately before EVERY `bar.sync 0;`
/// in EVERY kernel. PTX `membar.cta` lowers to a CTA-scope fence on AMDGPU.
/// Supersedes the previous IOFX_actorRange-only and family-prefix variants
/// (removed 2026-06-03) — applies broadly to address thread-0-store +
/// bar.sync LDS visibility patterns wherever they appear.
///
/// Skip CUB / cub-prefixed kernels (`_ZN3cub`) — those have their own
/// micro-optimized barrier patterns we don't want to perturb. PR #161638
/// fixed the relevant CUB issue.
///
/// Env-gate `ZLUDA_BROAD_MEMBAR=1` (default OFF — opt-in workaround).
pub fn patch_broad_membar_before_bar_sync(ptx: &str) -> Option<String> {
    let env_on = std::env::var("ZLUDA_BROAD_MEMBAR").map_or(false, |v| v == "1");
    if !env_on {
        return None;
    }
    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = String::with_capacity(ptx.len() + 1024);
    let mut in_kernel = false;
    let mut current_name = String::new();
    let mut brace_depth: i32 = 0;
    let mut patched_kernels: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total_patches = 0usize;
    for line in &lines {
        let trimmed = line.trim_start();
        // Detect kernel entry start
        if let Some(name) = trimmed
            .strip_prefix(".visible .entry ")
            .or_else(|| trimmed.strip_prefix(".entry "))
        {
            // Extract name up to '('
            let nm = name.split('(').next().unwrap_or(name).trim().to_string();
            current_name = nm;
            /* Apply to ALL kernels, including CUB. Per user direction
             * 2026-05-05: prefer generic memory-model fixes over per-
             * kernel exclusions. The historical CUB-skip was based on
             * "CUB doesn't crash" which only held when PhysX wasn't
             * exercising it via D3D11 textures; the SubResource interop
             * landing exposed CUB to the same memory-model bug class. */
            let _ = current_name; // borrow to silence dead-code lint
            in_kernel = true;
            brace_depth = 0;
        }
        if in_kernel {
            for c in line.chars() {
                if c == '{' {
                    brace_depth += 1;
                } else if c == '}' {
                    brace_depth -= 1;
                    if brace_depth <= 0 {
                        in_kernel = false;
                        break;
                    }
                }
            }
        }
        if in_kernel && brace_depth >= 1 && trimmed == "bar.sync 0;" {
            out.push_str("// ZLUDA broad membar workaround\n");
            out.push_str("membar.cta;\n");
            patched_kernels.insert(current_name.clone());
            total_patches += 1;
        }
        out.push_str(line);
        out.push('\n');
    }
    if total_patches == 0 {
        None
    } else {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_broad_membar_before_bar_sync: inserted {} membar.cta across {} kernels",
            total_patches,
            patched_kernels.len()
        ));
        Some(out)
    }
}

/// Reader-side acquire fence for the cross-warp LDS handshake in
/// `RadixSortScanKernel`. Symmetric companion to
/// `patch_broad_membar_before_bar_sync` (which inserts membar.cta BEFORE
/// bar.sync = release semantics): this inserts membar.cta AFTER each
/// bar.sync 0 in RadixSortScanKernel = acquire semantics. Without it,
/// post-barrier reads can use stale cached LDS values from before the
/// barrier.
///
/// Phase 8 (2026-04-27) in-game evidence: warp 0 lane 2 stores correct
/// exclusive-prefix to LDS, thread 32 in warp 1 reads back its own
/// pre-barrier `thread_partial` — `bar.sync 0` fails to publish writes
/// cross-wave. Originally probed in commit 8daf62f8, gated default off
/// when CUB Scan was no longer the primary blocker. Re-enabled
/// 2026-05-05 because CUB Downsweep (which consumes Scan's output) is
/// now the dominant residual failure (9/22 = 41%).
///
/// PTX `membar.cta` lowers to LLVM `fence(SEQ_CST, workgroup)`, which on
/// AMDGPU CU-mode (with PR #161638 backport) emits at minimum
/// `s_waitcnt lgkmcnt(0)` after `s_barrier`, giving reader threads a
/// fresh acquire of LDS state.
///
/// Off by default (retirement profile). Set `ZLUDA_SCAN_READER_ACQUIRE=1` to enable.
pub fn patch_scan_kernel_reader_acquire(ptx: &str) -> Option<String> {
    // Off by default since the validated retirement profile; =1 enables.
    let env_off = std::env::var("ZLUDA_SCAN_READER_ACQUIRE").map_or(true, |v| v != "1");
    if env_off {
        return None;
    }
    if !ptx.contains("RadixSortScanKernel") {
        return None;
    }

    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = String::with_capacity(ptx.len() + 256);
    let mut in_scan = false;
    let mut brace_depth: i32 = 0;
    let mut applied = 0usize;

    for line in &lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with(".visible .entry ") || trimmed.starts_with(".entry ") {
            in_scan = trimmed.contains("RadixSortScanKernel");
            brace_depth = 0;
        }
        out.push_str(line);
        out.push('\n');
        if in_scan && brace_depth >= 1 {
            let stripped = trimmed.trim_end_matches(|c: char| c.is_whitespace());
            if stripped == "bar.sync 0;" {
                out.push_str("// ZLUDA scan reader-acquire fence\n");
                out.push_str("membar.cta;\n");
                applied += 1;
            }
        }
        if in_scan {
            for c in line.chars() {
                if c == '{' {
                    brace_depth += 1;
                } else if c == '}' {
                    brace_depth -= 1;
                    if brace_depth <= 0 {
                        in_scan = false;
                        break;
                    }
                }
            }
        }
    }

    if applied == 0 {
        return None;
    }
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] patch_scan_kernel_reader_acquire: injected membar.cta after {} bar.sync 0 site(s) in RadixSortScanKernel",
        applied
    ));
    Some(out)
}

pub fn patch_divergent_lds_merge_sync(ptx: &str) -> Option<String> {
    // Off by default since the validated retirement profile; =1 enables.
    let env_off = std::env::var("ZLUDA_DIVERGENT_LDS_MERGE_SYNC").map_or(true, |v| v != "1");
    if env_off {
        return None;
    }

    let lines: Vec<&str> = ptx.lines().collect();
    let mut out = String::with_capacity(ptx.len() + 1024);
    let mut in_kernel = false;
    let mut brace_depth: i32 = 0;
    let mut kernel_start_idx: usize = 0;
    let mut patched = 0usize;
    let mut patched_kernels: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut current_kernel_name = String::new();

    for i in 0..lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();

        // Track kernel boundaries
        if let Some(name) = trimmed
            .strip_prefix(".visible .entry ")
            .or_else(|| trimmed.strip_prefix(".entry "))
        {
            current_kernel_name = name.split('(').next().unwrap_or(name).trim().to_string();
            in_kernel = true;
            brace_depth = 0;
            kernel_start_idx = i;
        }
        if in_kernel {
            for c in line.chars() {
                if c == '{' {
                    brace_depth += 1;
                } else if c == '}' {
                    brace_depth -= 1;
                    if brace_depth <= 0 {
                        in_kernel = false;
                        break;
                    }
                }
            }
        }

        out.push_str(line);
        out.push('\n');

        if !in_kernel || brace_depth < 1 {
            continue;
        }

        // Detect label line: trimmed ends with ':' and no whitespace/colons inside
        let label_name = match extract_ptx_label(trimmed) {
            Some(name) => name,
            None => continue,
        };

        // Skip if already followed by bar.sync
        if i + 1 < lines.len() {
            let next = lines[i + 1].trim_start();
            if next.starts_with("bar.sync") {
                continue;
            }
        }

        // Check: is there a risky conditional bra targeting this label
        // earlier in the same kernel?
        if !has_risky_bra_to_label(&lines, kernel_start_idx, i, &label_name) {
            continue;
        }

        // Check: does the post-label code have ld.shared.* before the
        // next bar.sync (within the same kernel)?
        if !has_ld_shared_before_next_barrier(&lines, i) {
            continue;
        }

        // 2026-05-13 v3 experiment (REVERTED): tried emitting `membar.cta;`
        // before `bar.sync 0;` to match what patch_broad_membar would have
        // inserted. v3 build caused IOFX_bboxKernel to fault on first frame
        // of bench (single GPU fault, status 0x..50, before bench progressed).
        // Reverted to bare bar.sync 0 (the bar.sync wrapper in
        // zluda_ptx_impl.cpp already emits an explicit s_waitcnt(0) before
        // s_barrier, providing the release semantics we need).
        out.push_str("// ZLUDA divergent-LDS-write merge sync\n");
        out.push_str("bar.sync 0;\n");
        patched += 1;
        patched_kernels.insert(current_kernel_name.clone());
    }

    if patched == 0 {
        None
    } else {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_divergent_lds_merge_sync: inserted {} bar.sync at merge points across {} kernel(s)",
            patched,
            patched_kernels.len()
        ));
        Some(out)
    }
}

/// v11 (2026-05-16): MS_1 gate-trace instrumentation. Per-kernel-name
/// PTX patch (allowed for INSTRUMENTATION; forbidden for fixes/workarounds).
///
/// MS_1 entry has a bit-4-set gate at line 2122-2128 that determines whether
/// flag bit 4 is set for a particle. Two predicates gate it:
///   setp.eq.s32 %p54, %r178, 0;       // %r178 = state code; if 0, skip
///   @%p54 bra BB9_91;
///   ld.local.u32 %r162, [%rd36];       // [SPL+16] = secondary state
///   setp.eq.s32 %p55, %r162, 0;        // if [SPL+16] == 0, skip
///   @%p55 bra BB9_91;
///
/// Run #20 verified bit-4-set NEVER fires on ZLUDA (2845/2845 launches).
/// One of these predicates is always failing. This patch inserts an atomic
/// counter increment at 4 outcomes:
///   [trace+0]:  r178 == 0 (skip via first predicate)
///   [trace+4]:  r178 != 0 AND state == 0 (skip via second predicate)
///   [trace+8]:  r178 != 0 AND state != 0 (bit-4 would fire)
///   [trace+12]: total threads reaching the gate
///
/// Gated on ZLUDA_MS1_GATE_TRACE=1. Trace buffer is allocated host-side
/// (in function.rs) and dev_ptr passed as literal into the inserted PTX.
pub fn patch_ms1_gate_trace(ptx: &str, trace_dev_ptr: u64) -> Option<String> {
    // DIAGNOSTIC: log entry conditions to debug patch application failures
    let has_ms1_entry =
        ptx.contains(".visible .entry CreateParticleShapesIncremental_MultiStage_1(");
    if has_ms1_entry {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_ms1_gate_trace ENTRY: dev_ptr=0x{:x} ptx_len={}",
            trace_dev_ptr,
            ptx.len()
        ));
    }
    if trace_dev_ptr == 0 {
        if has_ms1_entry {
            super::driver::zluda_log_err("[ZLUDA] patch_ms1_gate_trace: SKIP (trace_dev_ptr=0)");
        }
        return None;
    }
    // Only patch MS_1 entry kernel (per-kernel-name allowed for trace).
    // CRITICAL: there's a templated HELPER kernel (.visible .func) that
    // also contains the same bit-4-set pattern at line ~1001, but that
    // helper is NEVER LAUNCHED in practice. We must patch only the
    // .visible .entry (MS_1 entry) site at line ~2137.
    let entry_marker = ".visible .entry CreateParticleShapesIncremental_MultiStage_1(";
    let ms1_entry_start = ptx.find(entry_marker)?;
    let ms2_entry_marker = ".visible .entry CreateParticleShapesIncremental_MultiStage_2";
    let ms1_entry_end = ptx[ms1_entry_start..]
        .find(ms2_entry_marker)
        .map(|n| ms1_entry_start + n)
        .unwrap_or(ptx.len());
    let entry_block = &ptx[ms1_entry_start..ms1_entry_end];

    // Find the gate signature: `setp.eq.s32 %pXX, %r178, 0;\n@%pXX bra BBn_NN;`
    // followed by `ld.local.u32 %rNN, [%rdNN];\nsetp.eq.s32 %pNN, %rNN, 0;\n@%pNN bra BBn_NN;`
    // We detect via the unique register %r178 and the local-load + bra pattern.
    // The bit-4-set `or.b32 %r165, %r164, 16;` is the immediately-following
    // instruction. Search ONLY within MS_1 entry block.
    let bit4_marker = "or.b32 %r165, %r164, 16;";
    let bit4_idx_in_entry = match entry_block.find(bit4_marker) {
        Some(i) => i,
        None => {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] patch_ms1_gate_trace: SKIP (no bit4_marker in entry; entry_block_len={})",
                entry_block.len()
            ));
            return None;
        }
    };
    let bit4_idx = ms1_entry_start + bit4_idx_in_entry;
    // Search backwards from bit4_marker for the first setp on %r178.
    // The gate sequence is:
    //   setp.eq.s32 %pXX, %r178, 0;
    //   @%pXX bra BBn_NN;
    //   ld.local.u32 %rYY, [%rd36];
    //   setp.eq.s32 %pYY, %rYY, 0;
    //   @%pYY bra BBn_NN;
    //   (bit-4 set follows immediately)
    // Look for `%r178, 0;` (more specific) then walk back to start of its line.
    let window_start = bit4_idx.saturating_sub(800);
    let window = &ptx[window_start..bit4_idx];
    let r178_substr = "%r178, 0;";
    let r178_offset = match window.rfind(r178_substr) {
        Some(o) => o,
        None => {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA] patch_ms1_gate_trace: SKIP (no '%r178, 0;' in 800-char window before bit4_marker)"
            ));
            return None;
        }
    };
    // Find the start of the line containing this %r178 reference.
    let line_start_in_window = window[..r178_offset]
        .rfind('\n')
        .map(|n| n + 1)
        .unwrap_or(0);
    let absolute_insert_idx = window_start + line_start_in_window;

    // Confirm: at absolute_insert_idx, we should see "setp.eq.s32" leading
    // up to "%r178, 0;", and within the next few lines we should see
    // "ld.local.u32" loading [%rd36].
    let confirm_window_end = (absolute_insert_idx + 400).min(ptx.len());
    let confirm_window = &ptx[absolute_insert_idx..confirm_window_end];
    if !confirm_window.starts_with("setp.eq.s32") || !confirm_window.contains("[%rd36]") {
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] patch_ms1_gate_trace: SKIP (confirm_window mismatch; first_200={:?})",
            &confirm_window[..confirm_window.len().min(200)]
        ));
        return None;
    }

    // Generate trace PTX. Use high-numbered registers that won't conflict.
    // MS_1 entry declares .reg .s32 %r<187> and .reg .pred %p<67>; we'll
    // patch those declarations too. Use unique register IDs 187,188,189 for
    // s32, and 67,68 for pred.
    let trace_code = format!(
        "\t// === ZLUDA MS1_GATE_TRACE (v11) ===\n\t\
         mov.u64 %rd172, 0x{:x};\n\t\
         ld.local.u32 %r187, [%rd36];\n\t\
         atom.global.add.u32 %r188, [%rd172+12], 1;\n\t\
         setp.eq.s32 %p67, %r178, 0;\n\t\
         @%p67 atom.global.add.u32 %r189, [%rd172], 1;\n\t\
         @%p67 bra ZLUDA_TRACE_DONE_MS1;\n\t\
         setp.eq.s32 %p68, %r187, 0;\n\t\
         @%p68 atom.global.add.u32 %r189, [%rd172+4], 1;\n\t\
         @!%p68 atom.global.add.u32 %r189, [%rd172+8], 1;\n\
         ZLUDA_TRACE_DONE_MS1:\n\t\
         // === END TRACE ===\n",
        trace_dev_ptr
    );

    // Build result: ptx[..absolute_insert_idx] + trace_code + ptx[absolute_insert_idx..]
    let mut out = String::with_capacity(ptx.len() + trace_code.len() + 200);
    out.push_str(&ptx[..absolute_insert_idx]);
    out.push_str(&trace_code);
    out.push_str(&ptx[absolute_insert_idx..]);

    // Bump register declarations in MS_1 entry to make room for our temps.
    // Replace `.reg .s32 %r<187>;` -> `.reg .s32 %r<200>;`
    //         `.reg .s64 %rd<173>;` -> `.reg .s64 %rd<175>;`
    //         `.reg .pred %p<67>;` -> `.reg .pred %p<72>;`
    // We patch only within the MS_1 entry block. Find the entry block.
    let ms1_start = out.find(".visible .entry CreateParticleShapesIncremental_MultiStage_1(")?;
    let ms1_end_next = ".visible .entry CreateParticleShapesIncremental_MultiStage_2";
    let ms1_end = out[ms1_start..]
        .find(ms1_end_next)
        .map(|n| ms1_start + n)
        .unwrap_or(out.len());
    let ms1_block = &out[ms1_start..ms1_end];
    let mut patched_ms1 = ms1_block.to_string();
    for (from, to) in &[
        (".reg .s32 %r<187>;", ".reg .s32 %r<200>;"),
        (".reg .s64 %rd<173>;", ".reg .s64 %rd<175>;"),
        (".reg .pred %p<67>;", ".reg .pred %p<72>;"),
    ] {
        patched_ms1 = patched_ms1.replacen(from, to, 1);
    }
    out.replace_range(ms1_start..ms1_end, &patched_ms1);

    super::driver::zluda_log_err(&format!(
        "[ZLUDA] patch_ms1_gate_trace: inserted trace at offset {} (trace_buf=0x{:x})",
        absolute_insert_idx, trace_dev_ptr
    ));
    Some(out)
}

// ============================================================================
// 2026-06-03 Step 2: Generic single-op memset kernel detection
// ============================================================================
//
// Replaces the kernel-name-specific Fix C (TurbulenceFS_dynamicGridClearKernel
// intercept) with an AST-based pattern detector that runs once per module
// load. A kernel is recognized as a "simple memset" iff its body matches
// the strict signature:
//
//   - body has exactly one global side-effect: `sust.b.<geom>.<width>.<mode>`
//     OR `st.global.<scalar>`
//   - body has zero `bar.sync`, zero `ld.shared`, zero `st.shared`,
//     zero `ld.global`
//   - the surface/address operand of the sust/st.global comes from
//     a single ld.param.<u64|s64|b64> reading a kernel param
//   - the value operand of the sust/st.global comes from a single
//     ld.param.<scalar> reading a kernel param
//
// If matched, the detector records `(kernel_name → SimpleMemsetSig)` keyed
// on the kernel's name. `launch_kernel` looks up by name; on hit it reads
// the surface and fill values from the kernel arg buffer at the recorded
// byte offsets (NVCC ABI: natural alignment, tight packing) and runs the
// existing `try_host_clear_3d_array` host-side path.
//
// Strict matching is intentional. False positives (a non-memset kernel
// swapped out by mistake) would silently produce wrong results, so the
// signature rejects anything with LDS traffic, branches more than one
// early-out, or fills sourced from non-param registers. The TurbulenceFS_*
// family's grid-clear kernels match cleanly (see
// `/tmp/zluda_ptx_042.ptx:8637-8676` for a reference body).

/// Recorded signature for a kernel that can be intercepted with a host-side
/// memset. Populated by `detect_simple_memset_kernels` at module load,
/// consumed by `function.rs::launch_kernel`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SimpleMemsetSig {
    /// Index of the surface/address kernel param (0-based in input_arguments).
    pub surf_param_idx: u16,
    /// Byte offset of the surf param in the launch arg buffer.
    pub surf_param_offset: u32,
    /// Width of the surf param in bytes (must be 8 — u64 surface handle or pointer).
    pub surf_param_width: u8,
    /// Index of the fill-value kernel param.
    pub fill_param_idx: u16,
    /// Byte offset of the fill param in the launch arg buffer.
    pub fill_param_offset: u32,
    /// Width of the fill scalar in bytes (1/2/4/8).
    pub fill_param_width: u8,
    /// `true` if the side-effect is `sust.b.3d.*` (host swap is hipMemcpy3DAsync
    /// into a hipArray). `false` if `st.global.*` (host swap would be linear
    /// hipMemset — not yet implemented; intercept skips this case).
    pub is_sust_3d: bool,
}

/// Reverse-table from kernel name → memset signature. Populated by
/// `detect_simple_memset_kernels` during `cuModuleLoadDataEx` processing.
pub(crate) static SIMPLE_MEMSET_TABLE: std::sync::LazyLock<Mutex<HashMap<String, SimpleMemsetSig>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Env-gated master switch. Default OFF — the intercept replaces an actual
/// kernel dispatch with a host-side memcpy3D, which has slightly different
/// memory access patterns than the kernel's `sust.b` store. Enable per
/// game via `ZLUDA_INTERCEPT_SIMPLE_MEMSET=1` in protonfix.
pub(crate) fn simple_memset_intercept_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = std::env::var("ZLUDA_INTERCEPT_SIMPLE_MEMSET").map_or(false, |v| v == "1");
        super::driver::zluda_log_err(&format!(
            "[ZLUDA] ZLUDA_INTERCEPT_SIMPLE_MEMSET={} (generic simple-memset kernel intercept)",
            if on { "1" } else { "0" }
        ));
        on
    })
}

/// Lookup helper for `function.rs::launch_kernel`. Returns the recorded
/// signature if the kernel name was registered as a simple memset at module
/// load time.
pub(crate) fn lookup_simple_memset_sig(name: &str) -> Option<SimpleMemsetSig> {
    SIMPLE_MEMSET_TABLE
        .lock()
        .ok()
        .and_then(|tbl| tbl.get(name).copied())
}

/// Walk a parsed PTX module, identify kernels whose body matches the
/// SimpleMemset signature, and insert them into `SIMPLE_MEMSET_TABLE`.
/// Idempotent — re-registering the same name overwrites with the latest
/// signature (matches the ptx_parser AST's view of the same module).
pub(crate) fn detect_simple_memset_kernels(module: &ptx_parser::Module<'_>) {
    let mut newly_detected: Vec<(String, SimpleMemsetSig)> = Vec::new();
    for directive in &module.directives {
        if let ptx_parser::Directive::Method(_, function) = directive {
            let name = match function.func_directive.name {
                ptx_parser::MethodName::Kernel(n) => n,
                _ => continue,
            };
            if let Some(sig) = analyze_kernel_for_simple_memset(function) {
                newly_detected.push((name.to_string(), sig));
            }
        }
    }
    if newly_detected.is_empty() {
        return;
    }
    if let Ok(mut tbl) = SIMPLE_MEMSET_TABLE.lock() {
        for (name, sig) in &newly_detected {
            tbl.insert(name.clone(), *sig);
        }
    }
    let names: Vec<&str> = newly_detected.iter().map(|(n, _)| n.as_str()).collect();
    super::driver::zluda_log_err(&format!(
        "[ZLUDA] SimpleMemset detector: registered {} kernel(s) for potential intercept: {:?}",
        newly_detected.len(),
        names
    ));
}

/// Compute byte offsets (per NVCC kernel-arg ABI: natural alignment, tight
/// packing) for each entry in `params`. Returns Vec of (offset, size) tuples
/// indexed parallel to `params`.
fn compute_param_offsets<ID>(
    params: &[ptx_parser::Variable<ID>],
) -> Vec<(u32, u8)> {
    let mut offsets = Vec::with_capacity(params.len());
    let mut cursor: u32 = 0;
    for var in params {
        // Type::layout() is None only for unsized types, which kernel params never are.
        let Some(layout) = var.info.v_type.layout() else {
            offsets.push((cursor, 0));
            continue;
        };
        let size = layout.size() as u32;
        let align = layout.align() as u32;
        if align > 0 {
            let misalignment = cursor % align;
            if misalignment != 0 {
                cursor += align - misalignment;
            }
        }
        offsets.push((cursor, size.min(u8::MAX as u32) as u8));
        cursor = cursor.saturating_add(size);
    }
    offsets
}

/// Inspect a kernel function's body for the SimpleMemset signature.
/// Returns `Some(sig)` if matched, `None` otherwise.
///
/// Single-pass scan: for each statement, classify, reject on any
/// disqualifying instruction, track ld.param-to-param-index mapping.
/// At the end, if exactly one sust or st.global was seen and its operands
/// map back to known kernel params, build the signature.
fn analyze_kernel_for_simple_memset<'a>(
    function: &ptx_parser::Function<
        'a,
        &'a str,
        ptx_parser::Statement<ptx_parser::ParsedOperand<&'a str>>,
    >,
) -> Option<SimpleMemsetSig> {
    let body = function.body.as_ref()?;

    // Map register-name → kernel-param-index, populated by ld.param.* ops.
    let mut param_reg_map: HashMap<&str, u16> = HashMap::new();
    // Build the kernel's param-name → param-index reverse map for fast
    // lookup when we encounter `ld.param ... [name_param_N]`.
    let mut param_name_to_idx: HashMap<&str, u16> = HashMap::new();
    for (idx, var) in function.func_directive.input_arguments.iter().enumerate() {
        param_name_to_idx.insert(var.name, idx as u16);
    }

    let mut sust_count: u32 = 0;
    let mut st_global_count: u32 = 0;
    let mut sust_surf_reg: Option<&str> = None;
    let mut sust_fill_reg: Option<&str> = None;
    let mut sust_geom_is_3d = false;
    let mut sust_fill_width_bytes: u8 = 0;
    let mut st_global_addr_reg: Option<&str> = None;
    let mut st_global_val_reg: Option<&str> = None;
    let mut st_global_val_width: u8 = 0;
    let mut early_exit_brs: u32 = 0;

    for stmt in body {
        match stmt {
            ptx_parser::Statement::Instruction(_, instr) => match instr {
                // Disqualifiers — any of these means NOT a simple memset.
                ptx_parser::Instruction::Bar { .. }
                | ptx_parser::Instruction::BarRed { .. }
                | ptx_parser::Instruction::BarWarp { .. } => return None,

                // ld.* — record param mappings; reject ld.global/ld.shared.
                ptx_parser::Instruction::Ld { data, arguments } => {
                    match data.state_space {
                        ptx_parser::StateSpace::Param => {
                            // dst is a register; src is the param name.
                            let dst_reg = parsed_operand_reg(&arguments.dst)?;
                            let src_param = parsed_operand_reg(&arguments.src)?;
                            if let Some(&idx) = param_name_to_idx.get(src_param) {
                                param_reg_map.insert(dst_reg, idx);
                            }
                        }
                        ptx_parser::StateSpace::Global
                        | ptx_parser::StateSpace::Shared => return None,
                        // Const/Local/Reg loads are unusual in a strict
                        // memset kernel; reject to keep the signature
                        // conservative.
                        _ => return None,
                    }
                }

                // st.* — disqualify shared; count st.global; reject local etc.
                ptx_parser::Instruction::St { data, arguments } => {
                    match data.state_space {
                        ptx_parser::StateSpace::Shared => return None,
                        ptx_parser::StateSpace::Global => {
                            st_global_count += 1;
                            st_global_addr_reg = parsed_operand_reg(&arguments.src1);
                            st_global_val_reg = parsed_operand_reg(&arguments.src2);
                            // Type width = scalar size in bytes.
                            if let ptx_parser::Type::Scalar(scalar) = &data.typ {
                                st_global_val_width = scalar.layout().size() as u8;
                            }
                        }
                        _ => return None,
                    }
                }

                // sust — count + capture operands; require 3D geometry
                // for the host hipMemcpy3DAsync swap path.
                ptx_parser::Instruction::Sust { data, arguments } => {
                    sust_count += 1;
                    sust_surf_reg = parsed_operand_reg(&arguments.src_surface);
                    sust_fill_reg = parsed_operand_reg(&arguments.src_val);
                    sust_geom_is_3d = matches!(
                        data.geometry,
                        ptx_parser::TextureGeometry::Dim3d
                    );
                    sust_fill_width_bytes = match data.data_type {
                        ptx_parser::ScalarType::B8 | ptx_parser::ScalarType::U8
                        | ptx_parser::ScalarType::S8 => 1,
                        ptx_parser::ScalarType::B16 | ptx_parser::ScalarType::U16
                        | ptx_parser::ScalarType::S16 | ptx_parser::ScalarType::F16 => 2,
                        ptx_parser::ScalarType::B32 | ptx_parser::ScalarType::U32
                        | ptx_parser::ScalarType::S32 | ptx_parser::ScalarType::F32 => 4,
                        ptx_parser::ScalarType::B64 | ptx_parser::ScalarType::U64
                        | ptx_parser::ScalarType::S64 | ptx_parser::ScalarType::F64 => 8,
                        _ => 0,
                    };
                }

                // Address-compute ops are fine (mov, mad, mul, shl, add,
                // cvt, and.b32 masks, setp for the optional early-exit
                // predicate). We don't track their dataflow; the rule is
                // "no LDS / no global reads / no barriers / one sust",
                // so anything else that doesn't touch memory is harmless.

                // Branches: count predicate-guarded `bra` (early-exit
                // pattern). Unconditional bra to label is fine; backward
                // bra would be a loop, but we don't detect that explicitly.
                ptx_parser::Instruction::Bra { .. } => {
                    early_exit_brs += 1;
                    if early_exit_brs > 2 {
                        // Two-bra cap: one optional bounds-check + one
                        // optional skip to ret. More than that suggests
                        // a real control-flow graph; reject.
                        return None;
                    }
                }

                // Surface-load is the opposite direction — definitely
                // not a memset.
                ptx_parser::Instruction::Suld { .. } => return None,

                // Atomic ops, texture reads, etc. — disqualify.
                ptx_parser::Instruction::Atom { .. }
                | ptx_parser::Instruction::AtomCas { .. }
                | ptx_parser::Instruction::Tex { .. } => return None,

                // Everything else (mov, cvt, setp, mad, shl, shr, add,
                // sub, and, or, xor, ret, etc.) — silently allowed.
                _ => {}
            },
            // Labels, variable declarations, nested blocks — allowed.
            ptx_parser::Statement::Label(_)
            | ptx_parser::Statement::Variable(_) => {}
            ptx_parser::Statement::Block(_) => {
                // We don't traverse nested blocks (very rare in entry
                // kernels). Conservatively reject to keep correctness.
                return None;
            }
        }
    }

    // Need exactly one global side-effect.
    if sust_count + st_global_count != 1 {
        return None;
    }

    let param_offsets =
        compute_param_offsets(&function.func_directive.input_arguments);

    if sust_count == 1 {
        let surf_reg = sust_surf_reg?;
        let fill_reg = sust_fill_reg?;
        let surf_idx = *param_reg_map.get(surf_reg)?;
        let fill_idx = *param_reg_map.get(fill_reg)?;
        let (surf_off, surf_sz) = *param_offsets.get(surf_idx as usize)?;
        let (fill_off, _fill_sz) = *param_offsets.get(fill_idx as usize)?;
        if surf_sz != 8 || sust_fill_width_bytes == 0 {
            return None;
        }
        return Some(SimpleMemsetSig {
            surf_param_idx: surf_idx,
            surf_param_offset: surf_off,
            surf_param_width: surf_sz,
            fill_param_idx: fill_idx,
            fill_param_offset: fill_off,
            fill_param_width: sust_fill_width_bytes,
            is_sust_3d: sust_geom_is_3d,
        });
    }

    // st.global single-op path — not currently used by the host swap
    // (needs hipMemset on a linear range; the current host path expects a
    // hipArray-backed surface). Keep detection but skip registering for
    // now by returning None — extends cleanly when we add linear support.
    if st_global_count == 1 {
        let _ = (st_global_addr_reg, st_global_val_reg, st_global_val_width);
        return None;
    }

    None
}

/// Helper: extract the register identifier from `ParsedOperand::Reg(_)` or
/// a single-element `VecPack([Reg(_)])` (PTX renders single-channel sust
/// fill values as `{%r4}`). Returns None for empty/multi/imm operands —
/// the SimpleMemset signature requires single-register address and value.
fn parsed_operand_reg<'a>(
    op: &ptx_parser::ParsedOperand<&'a str>,
) -> Option<&'a str> {
    match op {
        ptx_parser::ParsedOperand::Reg(id) => Some(*id),
        ptx_parser::ParsedOperand::VecPack(elems) => {
            if elems.len() == 1 {
                match &elems[0] {
                    Some(ptx_parser::RegOrImmediate::Reg(id)) => Some(*id),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract the label name from a line that's just `LABEL:` or `LABEL: comment`.
/// Returns None if the line is not a bare label.
fn extract_ptx_label(trimmed: &str) -> Option<&str> {
    let head = trimmed.split("//").next()?.trim();
    if !head.ends_with(':') {
        return None;
    }
    let name = &head[..head.len() - 1];
    if name.is_empty() || name.contains(char::is_whitespace) {
        return None;
    }
    // Skip PTX directives like ".pragma:" — labels are identifier-like
    let first = name.chars().next()?;
    if !(first.is_alphabetic() || first == '_' || first == '$') {
        return None;
    }
    Some(name)
}

/// Scan backward from label_idx-1 to kernel_start_idx, accumulating
/// `st.shared.*` sightings. On any `@<pred> bra <label_name>;` line:
///   - If accumulator is set → this bra is risky, return true
///   - Else → continue (this bra has no st.shared between it and the label)
/// On `ret;` → bail out (avoid deadlock inserting bar.sync); return false
/// On reaching kernel_start_idx → no risky bra found.
fn has_risky_bra_to_label(
    lines: &[&str],
    kernel_start_idx: usize,
    label_idx: usize,
    label_name: &str,
) -> bool {
    if label_idx == 0 {
        return false;
    }
    let mut accum_st_shared = false;
    let mut j = label_idx - 1;
    while j > kernel_start_idx {
        let prev = lines[j].trim_start();
        // Deadlock guard: a `ret;` or `exit;` between bra and label means
        // some threads exited before the merge — inserting bar.sync at
        // the label would deadlock.
        if prev == "ret;" || prev == "exit;" {
            return false;
        }
        // Match both plain and volatile stores. `patch_broad_volatile_lds`
        // runs BEFORE this pass in the chain and converts ld.shared → ld.
        // volatile.shared; defensively the store side could be similarly
        // converted in a future patch, so we accept either qualifier here.
        if prev.starts_with("st.shared.") || prev.starts_with("st.volatile.shared.") {
            accum_st_shared = true;
        }
        if is_predicated_bra_to(prev, label_name) {
            if accum_st_shared {
                // Warp-divergence discriminator (2026-05-14): only consider
                // a bra "risky" if its predicate is computed from a
                // warp-id derivation (`shr.u32 _, %tid.x, 5/6`). NRSS's
                // BB20_201 bra is `setp.lt.u32 %p, %warpid, %count` →
                // genuine cross-warp divergence → LDS merge sync needed.
                // IOFX_bboxKernel's BB10_184 bra is `setp.eq.s32 %p, %tid.x, 0`
                // → uniform per-warp (all threads in a warp take it
                // together) → no race → inserting bar.sync was a false
                // positive that caused the 2026-05-14 in-game regression.
                // See plan validated-inventing-babbage.md Track B.
                if is_bra_predicate_warp_divergent(lines, j, kernel_start_idx) {
                    return true;
                }
                // Predicate is uniform-per-warp (lane-check, shared-mem-fed,
                // composite `and.pred`, etc.) → SAFE to skip. Don't return
                // true; continue scanning backward in case an EARLIER bra
                // to the same label is warp-divergent.
            }
        }
        if j == 0 {
            break;
        }
        j -= 1;
    }
    false
}

/// Trace back from a conditional `bra` to determine whether its predicate
/// is *warp-divergent* (derived from `%tid.x >> N` where N is 5 or 6) or
/// uniform-per-warp (derived from `%tid.x` directly, shared-memory loads,
/// or composite `and.pred`/`or.pred` chains).
///
/// Returns `true` ONLY for warp-divergent predicates that match the NRSS
/// BB20_201 pattern. Returns `false` for the IOFX_bboxKernel pattern and
/// for any predicate we can't definitively classify (conservative — better
/// to miss a real race than to over-insert bar.sync at a uniform site
/// where it broke the bench).
///
/// Search bounded to ~200 lines backward to avoid quadratic cost on huge
/// kernels (NRSS body is ~1850 lines; the predicate def is always close
/// to the bra).
fn is_bra_predicate_warp_divergent(
    lines: &[&str],
    bra_idx: usize,
    kernel_start_idx: usize,
) -> bool {
    // Extract predicate name: "@!%p6 bra LABEL;" → "%p6"
    let bra_line = lines[bra_idx].trim();
    let body = bra_line.trim_start_matches('@').trim_start_matches('!');
    let pred: String = body.chars().take_while(|c| !c.is_whitespace()).collect();
    if !pred.starts_with('%') {
        return false;
    }

    // 2026-06-02: scan all the way back to kernel_start_idx for the setp
    // (was bounded to 200 lines, but predicate def can be far from the bra
    // in CUB-style kernels — see is_register_bound_to_tid_x comment).
    let lower = kernel_start_idx;
    let mut i = bra_idx;
    while i > lower {
        i -= 1;
        let line = lines[i].trim().trim_end_matches(';');
        // Composite predicate (and.pred/or.pred/not.pred) → indeterminate,
        // be conservative and skip. NRSS's bra predicate is a direct setp;
        // IOFX_bboxKernel's chains through and.pred. Treating composites
        // as "unknown" matches our intent: never patch a composite.
        if line.starts_with("and.pred ")
            || line.starts_with("or.pred ")
            || line.starts_with("not.pred ")
            || line.starts_with("xor.pred ")
        {
            // Check that this pred line is defining OUR predicate (first operand).
            let after_op = line.splitn(2, char::is_whitespace).nth(1).unwrap_or("");
            let first_operand = after_op.split(',').next().unwrap_or("").trim();
            if first_operand == pred {
                return false;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("setp.") {
            // setp.<cmp>.<type> %dst, %srcA, %srcB
            // rest = "lt.u32 %p6, %r983, %r984"
            let after_type = match rest.split_once(char::is_whitespace) {
                Some((_, r)) => r.trim(),
                None => continue,
            };
            let operands: Vec<&str> = after_type.split(',').map(|s| s.trim()).collect();
            if operands.len() < 3 {
                continue;
            }
            if operands[0] != pred {
                continue;
            }
            let src_a = operands[1];
            let src_b = operands[2];
            return is_operand_warp_id_derived(lines, i, kernel_start_idx, src_a)
                || is_operand_warp_id_derived(lines, i, kernel_start_idx, src_b);
        }
    }
    false
}

/// Check if a setp operand register traces back (one level) to
/// `shr.u32 _, %tid.x, 5` or `..., 6` — i.e., a warp-id computation
/// for wave32 / wave64 sizing respectively.
fn is_operand_warp_id_derived(
    lines: &[&str],
    setp_idx: usize,
    kernel_start_idx: usize,
    operand: &str,
) -> bool {
    // Immediates / non-register operands cannot be warp-id-derived.
    if !operand.starts_with('%') {
        return false;
    }
    // 2026-06-02: scan all the way back to kernel_start_idx — the 200-line
    // cap was missing the NRSS BB20_201 shr.u32 chain whose %tid.x mov
    // lives 1500+ lines before its use. See is_register_bound_to_tid_x
    // for the full mechanism.
    let lower = kernel_start_idx;
    let mut i = setp_idx;
    while i > lower {
        i -= 1;
        let line = lines[i].trim().trim_end_matches(';');
        // Match `shr.u32 <operand>, <src>, <shift>` or `shr.s32` variant.
        // The shift source can be `%tid.x` directly OR a register that
        // was mov'd from `%tid.x` (NRSS's actual pattern: `mov.u32 %r1,
        // %tid.x; ... shr.u32 %r983, %r1, 5;`). Chase mov-chains via
        // `is_register_bound_to_tid_x`.
        for prefix in &["shr.u32 ", "shr.s32 "] {
            if let Some(rest) = line.strip_prefix(prefix) {
                let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
                if parts.len() == 3 && parts[0] == operand {
                    if let Ok(shift) = parts[2].parse::<u32>() {
                        if shift == 5 || shift == 6 {
                            return is_register_bound_to_tid_x(
                                lines,
                                i,
                                kernel_start_idx,
                                parts[1],
                            );
                        }
                    }
                }
            }
        }
        // Match `mov.u32 <operand>, <src>` — chase one level.
        for prefix in &["mov.u32 ", "mov.s32 ", "mov.b32 "] {
            if let Some(rest) = line.strip_prefix(prefix) {
                let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
                if parts.len() == 2 && parts[0] == operand {
                    // Continue chasing this src by recursing on the
                    // upstream line. One-level recursion bound to avoid
                    // deep stacks; the typical NRSS chain is shr→setp
                    // directly without intermediate movs anyway.
                    return is_operand_warp_id_derived(lines, i, kernel_start_idx, parts[1]);
                }
            }
        }
    }
    false
}

/// Returns true if `op` is `%tid.x` directly, or is a register chain that
/// transitively `mov`s from `%tid.x` (bounded depth ~10 to keep cost flat).
/// NRSS uses `mov.u32 %r1, %tid.x;` once near kernel entry then uses `%r1`
/// in `shr.u32 _, %r1, 5;` for warp-id, so we must chase the mov chain.
///
/// 2026-06-02: scan all the way back to kernel_start_idx (no 200-line cap).
/// `%tid.x` is captured into a virtual register exactly once per kernel,
/// typically right after the entry's `.reg` directives — that's hundreds
/// to thousands of lines before the actual use in CUB-style code. NRSS in
/// particular has `mov.u32 %r1, %tid.x;` at relative line 36 of the kernel
/// but `shr.u32 %r983, %r1, 5;` at relative line ~1683 (1500+ lines later),
/// so a 200-line cap caused this discriminator to silently misclassify the
/// BB20_201 bra as non-divergent, blocking patch_divergent_lds_merge_sync
/// from firing at the canonical CUB BlockScan underflow site. NRSS kernels
/// run ~1850 lines; scanning to start is O(lines) per call (a few ms per
/// module at compile time, not per launch).
fn is_register_bound_to_tid_x(
    lines: &[&str],
    from_idx: usize,
    kernel_start_idx: usize,
    op: &str,
) -> bool {
    if op == "%tid.x" {
        return true;
    }
    if !op.starts_with('%') {
        return false;
    }
    let mut current = op.to_string();
    let mut depth = 0;
    let lower = kernel_start_idx;
    let mut i = from_idx;
    while depth < 10 && i > lower {
        i -= 1;
        let line = lines[i].trim().trim_end_matches(';');
        for prefix in &["mov.u32 ", "mov.s32 ", "mov.b32 "] {
            if let Some(rest) = line.strip_prefix(prefix) {
                let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
                if parts.len() == 2 && parts[0] == current {
                    if parts[1] == "%tid.x" {
                        return true;
                    }
                    if parts[1].starts_with('%') {
                        current = parts[1].to_string();
                        depth += 1;
                        // Reset scan from this line and keep going back
                        // for the new register's source.
                        continue;
                    }
                    return false; // mov from immediate
                }
            }
        }
    }
    false
}

/// Check `@<pred> bra <label>;` or `@!<pred> bra <label>;` patterns.
fn is_predicated_bra_to(line: &str, label: &str) -> bool {
    let line = line.trim();
    if !line.starts_with('@') {
        return false;
    }
    // Strip predicate token, then expect "bra <label>;" or "bra.uni <label>;"
    let after_pred = match line.split_once(char::is_whitespace) {
        Some((_, rest)) => rest.trim(),
        None => return false,
    };
    let target = if let Some(rest) = after_pred.strip_prefix("bra.uni") {
        rest.trim()
    } else if let Some(rest) = after_pred.strip_prefix("bra") {
        rest.trim()
    } else {
        return false;
    };
    let target = target.trim_end_matches(';').trim();
    target == label
}

/// Scan forward from label_idx+1 until next `bar.sync` or end of kernel.
/// If we encounter `ld.shared.<T>` before the next barrier → return true.
fn has_ld_shared_before_next_barrier(lines: &[&str], label_idx: usize) -> bool {
    let mut k = label_idx + 1;
    while k < lines.len() {
        let next = lines[k].trim_start();
        // End of kernel
        if next.starts_with('}') {
            return false;
        }
        // Next barrier — no read before it, safe
        if next.starts_with("bar.sync") {
            return false;
        }
        // Bra.uni out → leaves the region, safe to ignore further
        if next.starts_with("bra.uni") {
            return false;
        }
        // Match both plain and volatile loads. `patch_broad_volatile_lds`
        // converts `ld.shared.<T>` → `ld.volatile.shared.<T>` before this
        // pass runs; without this we silently miss every merge site whose
        // reader-only branch reads from LDS (the production blocker found
        // 2026-05-13 — my patch fired 0 times on NRSS for this reason).
        if next.starts_with("ld.shared.") || next.starts_with("ld.volatile.shared.") {
            return true;
        }
        k += 1;
    }
    false
}

/// Global map from synthetic texture reference pointers to their device global variable addresses.
/// When .texref variables are compiled as .global .u64 in PTX, hipModuleGetTexRef can't find them.
/// We create synthetic textureReference structs and track which device global they correspond to.
/// After a texture object is created (via hipTexRefSetArray), we copy the handle to the device global
/// so the kernel can read it at runtime.
static SYNTHETIC_TEXREFS: std::sync::LazyLock<Mutex<HashMap<usize, SyntheticTexRefState>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));
static TEXREFS_BY_MODULE_AND_NAME: std::sync::LazyLock<Mutex<HashMap<(usize, String), usize>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Copy)]
pub(crate) struct SyntheticTexRefLinearBinding {
    pub(crate) src_ptr: hipDeviceptr_t,
    pub(crate) size_in_bytes: usize,
    pub(crate) format: hipArray_Format,
    pub(crate) num_channels: u32,
    pub(crate) texture_desc: HIP_TEXTURE_DESC,
}

#[derive(Clone, Copy)]
struct SyntheticTexRefState {
    device_global: hipDeviceptr_t,
    linear_binding: Option<SyntheticTexRefLinearBinding>,
}

/// Check if a texture reference is synthetic and get its device global address
pub(crate) fn get_synthetic_texref_global(
    tex_ref: *const textureReference,
) -> Option<hipDeviceptr_t> {
    SYNTHETIC_TEXREFS.lock().ok().and_then(|map| {
        map.get(&(tex_ref as usize))
            .and_then(|state| (!state.device_global.0.is_null()).then_some(state.device_global))
    })
}

pub(crate) fn set_synthetic_texref_linear_binding(
    tex_ref: *const textureReference,
    binding: Option<SyntheticTexRefLinearBinding>,
) {
    let Ok(mut map) = SYNTHETIC_TEXREFS.lock() else {
        return;
    };
    let Some(state) = map.get_mut(&(tex_ref as usize)) else {
        return;
    };
    state.linear_binding = binding;
}

pub(crate) fn get_synthetic_texref_linear_binding_for_global(
    device_global: hipDeviceptr_t,
) -> Option<SyntheticTexRefLinearBinding> {
    SYNTHETIC_TEXREFS.lock().ok().and_then(|map| {
        map.values()
            .find(|state| state.device_global.0 == device_global.0)
            .and_then(|state| state.linear_binding)
    })
}

fn register_texref(
    module: hipModule_t,
    name: &str,
    tex_ref: *mut textureReference,
    device_global: Option<hipDeviceptr_t>,
) {
    if let Ok(mut map) = SYNTHETIC_TEXREFS.lock() {
        map.entry(tex_ref as usize).or_insert(SyntheticTexRefState {
            device_global: device_global.unwrap_or(hipDeviceptr_t(std::ptr::null_mut())),
            linear_binding: None,
        });
    }
    if let Ok(mut map) = TEXREFS_BY_MODULE_AND_NAME.lock() {
        map.insert((module.0 as usize, name.to_string()), tex_ref as usize);
    }
}

pub(crate) fn get_texref_linear_binding(
    module: hipModule_t,
    name: &str,
) -> Option<SyntheticTexRefLinearBinding> {
    let tex_ref = TEXREFS_BY_MODULE_AND_NAME
        .lock()
        .ok()
        .and_then(|map| map.get(&(module.0 as usize, name.to_string())).copied())?;
    SYNTHETIC_TEXREFS
        .lock()
        .ok()
        .and_then(|map| map.get(&tex_ref).and_then(|state| state.linear_binding))
}

pub(crate) struct Module {
    pub(crate) base: hipModule_t,
    pub(crate) sm_version: u32,
    pub(crate) bit32: Option<Metadata32Bit>,
    mutable: Mutex<ModuleMutable>,
}

struct ModuleMutable {
    functions: FxHashMap<CString, Box<Function>>,
}

impl ModuleMutable {
    pub(crate) fn get_function(
        &mut self,
        module: hipModule_t,
        sm_version: u32,
        name: CString,
        explicit_args_size_align: Option<Vec<FunctionArgInfo>>,
    ) -> Result<&Function, CUerror> {
        Ok(match self.functions.entry(name) {
            hash_map::Entry::Occupied(entry) => &*entry.into_mut(),
            hash_map::Entry::Vacant(entry) => {
                let mut func_handle = unsafe { std::mem::zeroed() };
                unsafe { hipModuleGetFunction(&mut func_handle, module, entry.key().as_ptr()) }?;
                let func = Box::new(Function {
                    base: func_handle,
                    module,
                    sm_version,
                    explicit_args_size_align,
                });
                &*entry.insert(func)
            }
        })
    }
}

impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type Error = CUerror;
    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        unsafe { hipModuleUnload(self.base) }?;
        Ok(())
    }
}

pub(crate) struct Metadata32Bit {
    pub globals: Vec<Global32Bit>,
    pub explicit_args_size_align: FxHashMap<String, Vec<FunctionArgInfo>>,
}

impl Metadata32Bit {
    fn new(meta: &kernel_metadata::ModuleMetadata32Bit) -> Self {
        let globals = meta
            .globals
            .iter()
            .map(|g| Global32Bit {
                name: CString::new(&*g.name).unwrap(),
                initializer: g.initializer.to_vec(),
                align: g.align,
            })
            .collect();
        let explicit_args_size_align = meta
            .explicit_args_size_align
            .iter()
            .map(|(key, value)| {
                let key = key.to_string();
                let value = value
                    .iter()
                    .map(|(size, align)| FunctionArgInfo {
                        size: *size,
                        align: *align,
                    })
                    .collect();
                (key, value)
            })
            .collect();
        Self {
            globals,
            explicit_args_size_align,
        }
    }

    fn from_archived(archived: &kernel_metadata::ArchivedModuleMetadata32Bit) -> Self {
        let globals = archived
            .globals
            .iter()
            .map(|g| Global32Bit {
                name: CString::new(g.name.as_str()).unwrap(),
                initializer: g.initializer.to_vec(),
                align: g.align.to_native(),
            })
            .collect();
        let explicit_args_size_align = archived
            .explicit_args_size_align
            .iter()
            .map(|kv| {
                (
                    kv.0.to_string(),
                    kv.1.iter()
                        .map(|x| FunctionArgInfo {
                            size: x.0.to_native(),
                            align: x.1.to_native(),
                        })
                        .collect(),
                )
            })
            .collect();
        Self {
            globals,
            explicit_args_size_align,
        }
    }
}

pub(crate) struct Global32Bit {
    pub name: CString,
    pub initializer: Vec<u8>,
    pub align: u32,
}

impl Module {
    pub(crate) fn new(base: hipModule_t, sm_version: u32, bit32: Option<Metadata32Bit>) -> Self {
        Self {
            base,
            sm_version,
            mutable: Mutex::new(ModuleMutable {
                functions: FxHashMap::default(),
            }),
            bit32,
        }
    }

    pub(crate) fn get_function<'a>(&'a self, name: &CStr) -> Result<&'static Function, CUerror> {
        let mut mutable = self.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        let explicit_args = self.bit32.as_ref().and_then(|meta| {
            meta.explicit_args_size_align
                .get(name.to_str().ok()?)
                .map(Vec::clone)
        });
        mutable
            .get_function(self.base, self.sm_version, name.to_owned(), explicit_args)
            .map(|f| unsafe { (f as *const Function).as_ref().unwrap() })
    }
}

fn get_best_ptx_and_compile(
    global_state: &GlobalState,
    image: CodeLibraryRef<'_>,
) -> Result<(hipModule_t, u32, Option<Metadata32Bit>), CUerror> {
    super::driver::zluda_log("[ZLUDA]   get_best_ptx_and_compile: iterating modules");
    let mut ptx_modules = Vec::new();
    let mut module_count = 0u32;
    unsafe {
        CodeLibraryRef::iterate_modules(image, |_idx, module| {
            module_count += 1;
            match module {
                Ok(CodeModuleRef::Text(ptx)) => {
                    super::driver::zluda_log(&format!(
                        "[ZLUDA]   module[{}]: Text PTX ({} bytes)",
                        module_count,
                        ptx.len()
                    ));
                    let after_bbox: Cow<str> = match patch_iofx_bbox_tail_merge(ptx) {
                        Some(patched) => Cow::Owned(patched),
                        None => Cow::Borrowed(ptx),
                    };
                    // 2026-06-15 Class-4 bit31 safety guard on the bbox wild store.
                    let after_bbox_b31: Cow<str> =
                        match patch_iofx_bbox_bit31_guard(after_bbox.as_ref()) {
                            Some(p) => Cow::Owned(p),
                            None => after_bbox,
                        };
                    let after_volatile: Cow<str> =
                        match patch_broad_volatile_lds(after_bbox_b31.as_ref()) {
                            Some(p) => Cow::Owned(p),
                            None => after_bbox_b31,
                        };
                    let after_membar: Cow<str> =
                        match patch_broad_membar_before_bar_sync(after_volatile.as_ref()) {
                            Some(p) => Cow::Owned(p),
                            None => after_volatile,
                        };
                    let after_scan: Cow<str> =
                        match patch_scan_kernel_reader_acquire(after_membar.as_ref()) {
                            Some(p) => Cow::Owned(p),
                            None => after_membar,
                        };
                    let after_divergent: Cow<str> =
                        match patch_divergent_lds_merge_sync(after_scan.as_ref()) {
                            Some(p) => Cow::Owned(p),
                            None => after_scan,
                        };
                    // 2026-06-15: the radix-sort branchless OOB-load PTX patches
                    // (count/scatter GLOBAL + cross-warp-scan SHARED) are RETIRED.
                    // Root cause was an LLVM AMDGPU SIWholeQuadMode miscompile that
                    // hoisted the divergent OOB-guarded load into whole-wave EXEC,
                    // reading wild addresses for the inactive lanes. Fixed directly
                    // in the LLVM fork (cl::opt -amdgpu-wwm-disable-strict-on-loads,
                    // default on); the original divergent PTX now compiles correctly
                    // (lab nrss_replay 0/20, in-game radix clean ~100k launches).
                    let after_scan_branchless: Cow<str> = after_divergent;
                    // 2026-06-11 scoped scatter-offset producer-store volatility
                    // (env-gated, default OFF — A/B pending).
                    let after_prod: Cow<str> =
                        match patch_lds_producer_volatile_store(after_scan_branchless.as_ref()) {
                            Some(p) => Cow::Owned(p),
                            None => after_scan_branchless,
                        };
                    // 2026-06-09 MS_4 underflow guard (default ON, env-gated).
                    let after_ms4: Cow<str> =
                        match patch_ms4_underflow_guard(after_prod.as_ref()) {
                            Some(p) => Cow::Owned(p),
                            None => after_prod,
                        };
                    // v11: MS_1 gate-trace instrumentation (env-gated).
                    let trace_dev_ptr = super::function::ms1_gate_trace_get_or_init_dev_ptr();
                    let final_ptx: Cow<str> =
                        match patch_ms1_gate_trace(after_ms4.as_ref(), trace_dev_ptr) {
                            Some(p) => Cow::Owned(p),
                            None => after_ms4,
                        };
                    ptx_modules.push(final_ptx);
                }
                Ok(CodeModuleRef::File(file)) => {
                    let kind = if file.header.kind == FatbinFileHeader::HEADER_KIND_PTX {
                        "PTX"
                    } else {
                        "non-PTX"
                    };
                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: File kind={} sm={} payload_size={} compressed_size={} uncompressed_payload={} flags={:?}", module_count, kind, file.header.sm_version, file.header.payload_size, file.header.compressed_size, file.header.uncompressed_payload, file.header.flags));
                    if file.header.kind != FatbinFileHeader::HEADER_KIND_PTX {
                        return;
                    }
                    match file.get_or_decompress_content(true) {
                        Ok(text) => {
                            super::driver::zluda_log(&format!(
                                "[ZLUDA]   module[{}]: decompressed OK, {} bytes",
                                module_count,
                                text.len()
                            ));
                            match cow_bytes_to_str(text) {
                                Some(text) => {
                                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: PTX text extracted ({} bytes), first 100 chars: {}", module_count, text.len(), &text[..text.len().min(100)]));
                                    // Dump PTX to file for analysis
                                    {
                                        use std::io::Write;
                                        let _ = std::fs::create_dir_all("/tmp/zluda_ptx_dump");
                                        static DUMP_COUNT: std::sync::atomic::AtomicU32 =
                                            std::sync::atomic::AtomicU32::new(0);
                                        let n = DUMP_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        if let Ok(mut f) = std::fs::File::create(format!(
                                            "/tmp/zluda_ptx_dump/module_{}.ptx",
                                            n
                                        )) {
                                            let _ = f.write_all(text.as_bytes());
                                        }
                                    }
                                    let after_bbox: Cow<str> =
                                        match patch_iofx_bbox_tail_merge(&text) {
                                            Some(p) => Cow::Owned(p),
                                            None => text,
                                        };
                                    // 2026-06-15 Class-4 bit31 safety guard.
                                    let after_bbox_b31: Cow<str> =
                                        match patch_iofx_bbox_bit31_guard(after_bbox.as_ref()) {
                                            Some(p) => Cow::Owned(p),
                                            None => after_bbox,
                                        };
                                    let after_volatile: Cow<str> =
                                        match patch_broad_volatile_lds(after_bbox_b31.as_ref()) {
                                            Some(p) => Cow::Owned(p),
                                            None => after_bbox_b31,
                                        };
                                    let after_membar: Cow<str> =
                                        match patch_broad_membar_before_bar_sync(
                                            after_volatile.as_ref(),
                                        ) {
                                            Some(p) => Cow::Owned(p),
                                            None => after_volatile,
                                        };
                                    let after_scan: Cow<str> =
                                        match patch_scan_kernel_reader_acquire(
                                            after_membar.as_ref(),
                                        ) {
                                            Some(p) => Cow::Owned(p),
                                            None => after_membar,
                                        };
                                    let after_divergent: Cow<str> =
                                        match patch_divergent_lds_merge_sync(after_scan.as_ref()) {
                                            Some(p) => Cow::Owned(p),
                                            None => after_scan,
                                        };
                                    // 2026-06-15: radix branchless OOB-load patches
                                    // RETIRED — superseded by the LLVM AMDGPU
                                    // SIWholeQuadMode load-hoist fix (cl::opt
                                    // -amdgpu-wwm-disable-strict-on-loads). The
                                    // original divergent PTX now compiles correctly.
                                    let after_scan_branchless: Cow<str> = after_divergent;
                                    // 2026-06-11 scoped scatter-offset producer
                                    // store volatility (env-gated, default OFF).
                                    let after_prod: Cow<str> =
                                        match patch_lds_producer_volatile_store(
                                            after_scan_branchless.as_ref(),
                                        ) {
                                            Some(p) => Cow::Owned(p),
                                            None => after_scan_branchless,
                                        };
                                    // 2026-06-09 MS_4 underflow guard.
                                    let after_ms4: Cow<str> =
                                        match patch_ms4_underflow_guard(after_prod.as_ref()) {
                                            Some(p) => Cow::Owned(p),
                                            None => after_prod,
                                        };
                                    // v11: MS_1 gate-trace instrumentation.
                                    let trace_dev_ptr =
                                        super::function::ms1_gate_trace_get_or_init_dev_ptr();
                                    let final_ptx: Cow<str> = match patch_ms1_gate_trace(
                                        after_ms4.as_ref(),
                                        trace_dev_ptr,
                                    ) {
                                        Some(p) => Cow::Owned(p),
                                        None => after_ms4,
                                    };
                                    ptx_modules.push(final_ptx);
                                }
                                None => {
                                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: cow_bytes_to_str FAILED (not valid UTF-8)", module_count));
                                }
                            }
                        }
                        Err(_) => {
                            super::driver::zluda_log(&format!(
                                "[ZLUDA]   module[{}]: get_or_decompress_content FAILED",
                                module_count
                            ));
                        }
                    }
                }
                _ => {
                    super::driver::zluda_log(&format!(
                        "[ZLUDA]   module[{}]: other format (ELF/Archive/Error)",
                        module_count
                    ));
                }
            }
        })
    };
    super::driver::zluda_log(&format!(
        "[ZLUDA]   found {} total modules, {} PTX candidates",
        module_count,
        ptx_modules.len()
    ));
    if ptx_modules.is_empty() {
        super::driver::zluda_log("[ZLUDA]   NO PTX modules found!");
    }
    let maybe_module = ptx_modules
        .iter()
        .rev() // TODO: actually sort by SM
        .try_fold(
            None,
            |acc: Option<(&Cow<'_, str>, ptx_parser::Module<'_>)>, src| {
                let maybe_ast = if cfg!(debug_assertions) {
                    ptx_parser::parse_module_checked(src)
                } else {
                    Ok(ptx_parser::parse_module_unchecked(src))
                };
                match maybe_ast {
                    Err(_) => {
                        super::driver::zluda_log("[ZLUDA]   PTX parse FAILED");
                        ControlFlow::Continue(acc)
                    }
                    Ok(ast) => {
                        super::driver::zluda_log(&format!(
                            "[ZLUDA]   PTX parsed OK, invalid_directives={}, directives={}",
                            ast.invalid_directives,
                            ast.directives.len()
                        ));
                        // 2026-06-03 Step 2: scan parsed AST for kernels
                        // matching the SimpleMemset signature and register
                        // them in SIMPLE_MEMSET_TABLE. Cheap traversal,
                        // amortized to module load. Runs on the first
                        // accepted AST in this fold pass — the same AST
                        // that becomes the load winner.
                        detect_simple_memset_kernels(&ast);
                        if ast.invalid_directives == 0 {
                            return ControlFlow::Break((src, ast));
                        } else {
                            ControlFlow::Continue(Some(match acc {
                                Some(best_known) => {
                                    if ast.invalid_directives < best_known.1.invalid_directives {
                                        (src, ast)
                                    } else {
                                        best_known
                                    }
                                }
                                None => (src, ast),
                            }))
                        }
                    }
                }
            },
        );
    let (text, module) = match maybe_module {
        ControlFlow::Break((ast, module)) | ControlFlow::Continue(Some((ast, module))) => {
            super::driver::zluda_log("[ZLUDA]   selected PTX module for compilation");
            // Explicit debug output only; normal module loads do not write PTX.
            if std::env::var("ZLUDA_DUMP_PTX").map_or(false, |value| value == "1") {
                static PTX_DUMP_CTR: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let n = PTX_DUMP_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let dump_path = format!("/tmp/zluda_ptx_{:03}.ptx", n);
                let _ = std::fs::write(&dump_path, ast.as_bytes());
                super::driver::zluda_log(&format!("[ZLUDA]   dumped PTX to {}", dump_path));
            }
            (Some(ast), module)
        }
        ControlFlow::Continue(None) => {
            super::driver::zluda_log("[ZLUDA]   no valid PTX module found, using empty module");
            if cfg!(debug_assertions) {
                return Err(CUerror::NO_BINARY_FOR_GPU);
            }
            (
                None,
                ptx_parser::Module {
                    ptx_version: (1, 0),
                    sm_version: 0,
                    directives: Vec::new(),
                    invalid_directives: usize::MAX,
                    address_size: 64,
                },
            )
        }
    };
    // TODO: get this information on initialization
    super::driver::zluda_log("[ZLUDA]   getting HIP properties");
    let hip_properties = get_hip_properties()?;
    let gcn_arch = get_gcn_arch(&hip_properties)?;
    super::driver::zluda_log(&format!("[ZLUDA]   gcn_arch={}", gcn_arch));
    let attributes = ExtraCacheAttributes {
        clock_rate: hip_properties.clockRate as u32,
        is_debug: cfg!(debug_assertions),
    };
    let mut cache_with_key = match (text, global_state.cache_path.as_ref()) {
        (Some(text), Some(p)) => (|| {
            let cache = zluda_cache::ModuleCache::open(p)?;
            let key = get_cache_key(gcn_arch, &text, &attributes)?;
            Some((cache, key))
        })(),
        _ => None,
    };
    let cached_binary = load_cached_binary(&mut cache_with_key);
    let from_cache = cached_binary.is_some();
    let (elf_module, sm_version, zluda32) = cached_binary.ok_or(CUerror::UNKNOWN).or_else(|_| {
        super::driver::zluda_log("[ZLUDA]   no cache hit, compiling PTX -> AMDGPU");
        compile_and_cache(gcn_arch, attributes, module, &mut cache_with_key)
    })?;
    super::driver::zluda_log(&format!(
        "[ZLUDA]   ELF module ready ({} bytes, from_cache={})",
        elf_module.len(),
        from_cache
    ));
    maybe_dump_radixsort_downsweep_module(
        text.as_ref().map(|text| text.as_ref()),
        &elf_module,
        from_cache,
    );
    let mut hip_module = unsafe { mem::zeroed() };
    super::driver::zluda_log("[ZLUDA]   calling hipModuleLoadData");
    unsafe { hipModuleLoadData(&mut hip_module, elf_module.as_ptr().cast()) }?;
    super::driver::zluda_log("[ZLUDA]   hipModuleLoadData OK");
    Ok((hip_module, sm_version, zluda32))
}

fn cow_bytes_to_str<'a>(data: Cow<'a, [u8]>) -> Option<Cow<'a, str>> {
    match data {
        Cow::Borrowed(bytes) => std::str::from_utf8(bytes).map(Cow::Borrowed).ok(),
        Cow::Owned(bytes) => String::from_utf8(bytes).map(Cow::Owned).ok(),
    }
}

pub(crate) fn load_hip_module(
    library: CodeLibraryRef,
) -> Result<(hipModule_t, u32, Option<Metadata32Bit>), CUerror> {
    let global_state = driver::global_state()?;
    get_best_ptx_and_compile(global_state, library)
}

#[derive(serde::Serialize)]
struct ExtraCacheAttributes {
    is_debug: bool,
    clock_rate: u32,
}

fn get_hip_properties<'a>() -> Result<hipDeviceProp_tR0600, CUerror> {
    let hip_dev = super::context::get_current_device()?;
    let mut props = unsafe { mem::zeroed() };
    unsafe { hipGetDevicePropertiesR0600(&mut props, hip_dev) }?;
    Ok(props)
}

fn get_gcn_arch<'a>(props: &'a hipDeviceProp_tR0600) -> Result<&'a str, CUerror> {
    let gcn_arch = unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) };
    gcn_arch.to_str().map_err(|_| CUerror::UNKNOWN)
}

fn get_cache_key<'a, 'b>(
    isa: &'a str,
    text: &str,
    attributes: &impl serde::Serialize,
) -> Option<zluda_cache::ModuleKey<'a>> {
    // Serialization here is deterministic. When marking a type with
    // #[derive(serde::Serialize)] the derived implementation will just write
    // fields in the order of their declaration. It's not explictly guaranteed
    // by serde, but it is the only sensible thing to do, so I feel safe
    // to rely on it
    // PTX text does not encode the optional emitter/backend mitigations.
    // Include their settings so retirement comparisons cannot reuse legacy code.
    let compiler_options = [
        "ZLUDA_KERNEL_FENCES",
        "ZLUDA_STRONG_ATOMICS",
        "ZLUDA_VOLATILE_SHARED",
        "ZLUDA_ZERO_UNDEF",
        "ZLUDA_SCALAR_CACHE_INV",
        "ZLUDA_LLVM_ARGS",
        "ZLUDA_CODEGEN_OPT_LEVEL",
    ]
    .map(|name| (name, std::env::var(name).ok()));
    let serialized_attributes = serde_json::to_string(&(attributes, compiler_options)).ok()?;
    Some(zluda_cache::ModuleKey {
        hash: blake3::hash(text.as_bytes()).to_hex(),
        compiler_version: "builtin",
        zluda_version: env!("VERGEN_GIT_SHA"),
        device: isa,
        backend_key: serialized_attributes,
        last_access: zluda_cache::ModuleCache::time_now(),
    })
}

fn is_radixsort_downsweep_module(text: &str) -> bool {
    text.contains("RadixSortDownsweepKernel") && text.contains("GridEvenShare")
}

fn should_dump_radixsort_downsweep_module(text: &str) -> bool {
    std::env::var("ZLUDA_DUMP_IR").map_or(false, |value| value == "1")
        && is_radixsort_downsweep_module(text)
}

fn maybe_dump_radixsort_downsweep_module(text: Option<&str>, elf_module: &[u8], from_cache: bool) {
    let Some(text) = text else {
        return;
    };
    if !should_dump_radixsort_downsweep_module(text) {
        return;
    }
    static DID_DUMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if DID_DUMP.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let dump_dir = "/tmp/zluda_live_sort_module";
    let _ = fs::create_dir_all(dump_dir);
    let ptx_path = format!("{dump_dir}/radixsort_downsweep.ptx");
    let elf_path = format!("{dump_dir}/radixsort_downsweep.elf");
    let meta_path = format!("{dump_dir}/meta.txt");
    let ptx_hash = blake3::hash(text.as_bytes());
    let elf_hash = blake3::hash(elf_module);
    let _ = fs::write(&ptx_path, text);
    let _ = fs::write(&elf_path, elf_module);
    let _ = fs::write(
        &meta_path,
        format!(
            "from_cache={from_cache}\nptx_bytes={}\nelf_bytes={}\nptx_hash={}\nelf_hash={}\n",
            text.len(),
            elf_module.len(),
            ptx_hash,
            elf_hash
        ),
    );
    super::driver::zluda_log_err(&format!(
        "LIVE SORT MODULE DUMPED: ptx={} elf={} meta={}",
        ptx_path, elf_path, meta_path
    ));
}

fn load_cached_binary(
    cache_with_key: &mut Option<(zluda_cache::ModuleCache, zluda_cache::ModuleKey)>,
) -> Option<(Vec<u8>, u32, Option<Metadata32Bit>)> {
    let binary = cache_with_key
        .as_mut()
        .and_then(|(c, key)| c.get_module_binary(key))?;
    let sm_version = kernel_metadata::ModuleMetadataV1::read_object(&binary)?
        .sm_version
        .to_native();
    let zluda32 = kernel_metadata::ModuleMetadata32Bit::read_object(&binary)
        .map(Metadata32Bit::from_archived);
    Some((binary, sm_version, zluda32))
}

fn compile_and_cache(
    gcn_arch: &str,
    attributes: ExtraCacheAttributes,
    ast: ptx_parser::Module,
    cache_with_key: &mut Option<(zluda_cache::ModuleCache, zluda_cache::ModuleKey)>,
) -> Result<(Vec<u8>, u32, Option<Metadata32Bit>), CUerror> {
    let compile_start = std::time::Instant::now();
    super::driver::zluda_log("[ZLUDA]   ptx::to_llvm_module starting");
    let llvm_module = ptx::to_llvm_module(
        ast,
        ptx::Attributes {
            clock_rate: attributes.clock_rate,
        },
        |pass_name| {
            super::driver::zluda_log(&format!("[ZLUDA]     pass complete: {}", pass_name));
        },
    )
    .map_err(|e| {
        super::driver::zluda_log(&format!("[ZLUDA]   ptx::to_llvm_module FAILED: {}", e));
        CUerror::UNKNOWN
    })?;
    super::driver::zluda_log(&format!(
        "[ZLUDA]   ptx::to_llvm_module OK in {:?}, starting llvm_zluda::compile",
        compile_start.elapsed()
    ));
    let ptx_impl = llvm_module.linked_bitcode();
    let sm_version = llvm_module.metadata.sm_version;
    // Dump LLVM IR and assembly for debugging PTX compilation issues
    static COMPILE_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COMPILE_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Direct test: write a marker file to confirm this code path runs
    if std::env::var("ZLUDA_DUMP_IR").map_or(false, |value| value == "1") {
        let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
        let _ = std::fs::write(
            format!("/tmp/zluda_ir_dump/MARKER_{}", n),
            format!("compile #{}", n),
        );
    }
    let metadata32 = llvm_module.metadata32.as_ref().map(Metadata32Bit::new);
    // Compile in a forked subprocess to survive LLVM abort() crashes.
    // The AMDGPU backend can crash in loadRegFromStackSlot for very large
    // kernels (e.g. BasicIOS_simulateKernel_templ). A subprocess isolates
    // the crash so the game can continue without the affected module.
    let elf_module = compile_in_subprocess(gcn_arch, llvm_module, n).map_err(|e| {
        super::driver::zluda_log(&format!("[ZLUDA]   compile_in_subprocess FAILED: {:?}", e));
        e
    })?;
    super::driver::zluda_log(&format!(
        "[ZLUDA]   llvm_zluda::compile OK in {:?}, {} bytes ELF",
        compile_start.elapsed(),
        elf_module.len()
    ));
    if let Some((cache, key)) = cache_with_key {
        key.last_access = zluda_cache::ModuleCache::time_now();
        cache.insert_module(key, &elf_module);
    }
    Ok((elf_module, sm_version, metadata32))
}

#[allow(dead_code)]
fn compile_in_subprocess(
    gcn_arch: &str,
    llvm_module: ptx::Module,
    compile_n: u32,
) -> Result<Vec<u8>, CUerror> {
    use std::io::Write;
    // Create a temp file for the output
    let output_path = format!("/tmp/zluda_compile_{}.elf", std::process::id());
    let ptx_impl = llvm_module.linked_bitcode();

    // Backend codegen opt levels to try, in order. Normally a single
    // Aggressive pass suffices. A handful of pathologically large kernels
    // (Batman:AK _Z14simulateClothsv — 80k PTX lines, hundreds of
    // strict.wwm shfl regions) trip an LLVM AMDGPU greedy-RA/WWM
    // MIR-corruption SIGSEGV in SIPreAllocateWWMRegs at -O1+; the same
    // module codegens correctly at -O0 (fast RA). On a child crash we retry
    // the SAME module with ZLUDA_CODEGEN_OPT_LEVEL=0 (correct, just
    // less-optimized codegen). fork() copies the address space, so the
    // parent's llvm_module stays intact for the retry — the child mutates
    // only its own copy. This is a generic robustness fallback (no
    // per-kernel name matching); it fires for ANY module the optimized
    // backend can't compile.
    // Ladder tried in order on a crash. None = Aggressive (best perf, the
    // normal case). "0" = fast RA — avoids the greedy-RA/WWM
    // SIPreAllocateWWMRegs crash (simulateCloths needs this). "1" = greedy RA
    // at low opt — safety net for a module that the fast allocator can't
    // satisfy ("ran out of registers" under heavy WWM pressure) but whose
    // Aggressive crash was incidental. A module that fails every rung is
    // genuinely uncompilable and surfaces as a hard error.
    let codegen_levels: &[Option<&str>] = &[None, Some("0"), Some("1")];

    for (attempt, level) in codegen_levels.iter().enumerate() {
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            super::driver::zluda_log("[ZLUDA]   fork() failed");
            return Err(CUerror::UNKNOWN);
        }
        if pid == 0 {
            // Child process — do the compilation. ptr::read duplicates the
            // module fields behind the borrow checker's back; sound because
            // the child _exit()s without ever dropping them, and the
            // parent's copy lives in a separate address space.
            //
            // Reset SIGSEGV/SIGABRT to their default action FIRST. The
            // process-wide ZLUDA fault-capture handler (installed at init)
            // would otherwise fire when the LLVM AMDGPU backend crashes
            // compiling a WWM-heavy module, printing the misleading
            // "*** SIGSEGV — GPU fault likely ***" banner into the trace log
            // — a benign, expected, retried event that nonetheless fools
            // log-scraping harnesses into reporting a game crash. We handle
            // the child's death via waitpid in the parent, so the child
            // should just die quietly with the default action.
            unsafe {
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                libc::signal(libc::SIGABRT, libc::SIG_DFL);
                libc::signal(libc::SIGBUS, libc::SIG_DFL);
                libc::signal(libc::SIGILL, libc::SIG_DFL);
                libc::signal(libc::SIGFPE, libc::SIG_DFL);
            }
            if let Some(lvl) = level {
                std::env::set_var("ZLUDA_CODEGEN_OPT_LEVEL", lvl);
            }
            let m: ptx::Module = unsafe { std::ptr::read(&llvm_module) };
            let hook =
                move |data: &Vec<u8>, name: String| dump_compile_artifact(compile_n, data, name);
            let result = llvm_zluda::compile(
                &m.context,
                gcn_arch,
                m.llvm_ir,
                ptx_impl,
                m.attributes_ir,
                m.metadata,
                m.metadata32,
                if std::env::var("ZLUDA_DUMP_IR").map_or(false, |value| value == "1") {
                    Some(&hook)
                } else {
                    None
                },
            );
            match result {
                Ok(elf_module) => {
                    if let Ok(mut f) = std::fs::File::create(&output_path) {
                        let _ = f.write_all(&elf_module);
                    }
                    unsafe { libc::_exit(0) };
                }
                Err(e) => {
                    let _ = std::fs::write(&output_path, format!("ERROR: {:?}", e));
                    unsafe { libc::_exit(1) };
                }
            }
        }
        // Parent process — wait for child
        super::driver::zluda_log(&format!(
            "[ZLUDA]   waiting for compile subprocess pid={} (codegen_opt={:?})",
            pid, level
        ));
        let mut status: i32 = 0;
        let wait_result = unsafe { libc::waitpid(pid, &mut status, 0) };
        if wait_result < 0 {
            super::driver::zluda_log("[ZLUDA]   waitpid failed");
            let _ = std::fs::remove_file(&output_path);
            return Err(CUerror::UNKNOWN);
        }
        let crashed = libc::WIFSIGNALED(status);
        let clean_exit = libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0;
        if clean_exit {
            // Read the ELF output
            let elf_module = std::fs::read(&output_path).map_err(|e| {
                super::driver::zluda_log(&format!(
                    "[ZLUDA]   failed to read compiled ELF: {:?}",
                    e
                ));
                CUerror::UNKNOWN
            })?;
            let _ = std::fs::remove_file(&output_path);
            if elf_module.starts_with(b"ERROR:") {
                super::driver::zluda_log(&format!(
                    "[ZLUDA]   compile error: {}",
                    String::from_utf8_lossy(&elf_module)
                ));
                return Err(CUerror::UNKNOWN);
            }
            if attempt > 0 {
                super::driver::zluda_log_err(&format!(
                    "[ZLUDA]   compile #{} SUCCEEDED on fallback codegen_opt={:?} after a crash",
                    compile_n, level
                ));
            }
            return Ok(elf_module);
        }
        let exit_code = if libc::WIFEXITED(status) {
            libc::WEXITSTATUS(status)
        } else {
            -1
        };
        let signal = if crashed { libc::WTERMSIG(status) } else { 0 };
        super::driver::zluda_log(&format!(
            "[ZLUDA]   compile subprocess failed: exit={}, signal={}, codegen_opt={:?}",
            exit_code, signal, level
        ));
        if let Ok(msg) = std::fs::read_to_string(&output_path) {
            super::driver::zluda_log(&format!("[ZLUDA]   compile error: {}", msg));
        }
        let _ = std::fs::remove_file(&output_path);
        // Retry at the next codegen opt level on ANY failure (signal crash
        // OR clean error-exit) — different opt levels genuinely produce
        // different outcomes: the greedy-RA/WWM crash happens only at -O1+,
        // while "ran out of registers" happens only at the fast (-O0)
        // allocator. Trying the whole ladder maximizes the chance some level
        // compiles the module. Only after every rung fails do we give up.
        if attempt + 1 >= codegen_levels.len() {
            super::driver::zluda_log_err(&format!(
                "[ZLUDA]   compile #{} FAILED at every codegen opt level (last: exit={} signal={})",
                compile_n, exit_code, signal
            ));
            return Err(CUerror::UNKNOWN);
        }
        super::driver::zluda_log_err(&format!(
            "[ZLUDA]   compile #{} failed (exit={} signal={}); retrying at lower codegen opt level",
            compile_n, exit_code, signal
        ));
    }
    Err(CUerror::UNKNOWN)
}

fn dump_compile_artifact(n: u32, data: &Vec<u8>, name: String) {
    let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
    let path = format!("/tmp/zluda_ir_dump/module_{}_{}", n, name);
    super::driver::zluda_log_err(&format!(
        "IR DUMP: {} ({} bytes) -> {}",
        name,
        data.len(),
        path
    ));
    let _ = std::fs::write(&path, data);
}

pub(crate) fn load(module: &mut CUmodule, fname: &CStr) -> CUresult {
    let mut image = fs::read(fname.to_str().map_err(|_| CUerror::INVALID_VALUE)?)
        .map_err(|_| CUerror::INVALID_VALUE)?;
    // Null-terminate the image, in case it's text
    image.push(0);
    let library = unsafe { CodeLibraryRef::try_load(image.as_ptr() as *const std::ffi::c_void) }
        .map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
    let (hip_module, sm_version, meta32) = load_hip_module(library)?;
    *module = Module::new(hip_module, sm_version, meta32).wrap();
    Ok(())
}

pub(crate) fn load_data(module: &mut CUmodule, image: &std::ffi::c_void) -> CUresult {
    super::driver::zluda_log("[ZLUDA] cuModuleLoadData: entering");
    let library = unsafe { CodeLibraryRef::try_load(image) }.map_err(|e| {
        super::driver::zluda_log(&format!(
            "[ZLUDA] cuModuleLoadData -> NO_BINARY_FOR_GPU (try_load failed: {:?})",
            e
        ));
        CUerror::NO_BINARY_FOR_GPU
    })?;
    let format = match library {
        CodeLibraryRef::FatbincWrapper(_) => "FatbincWrapper",
        CodeLibraryRef::FatbinHeader(_) => "FatbinHeader",
        CodeLibraryRef::Text(_) => "Text",
        CodeLibraryRef::Elf(_) => "Elf",
        CodeLibraryRef::Archive(_) => "Archive",
    };
    super::driver::zluda_log(&format!(
        "[ZLUDA] cuModuleLoadData: try_load OK, format={}",
        format
    ));
    let (hip_module, sm_version, meta32) = load_hip_module(library)?;
    *module = Module::new(hip_module, sm_version, meta32).wrap();
    super::driver::zluda_log("[ZLUDA] cuModuleLoadData -> SUCCESS");
    Ok(())
}

pub(crate) fn load_data_ex(
    module: &mut CUmodule,
    image: &std::ffi::c_void,
    _num_options: ::std::os::raw::c_uint,
    _options: Option<&mut CUjit_option_enum>,
    _option_values: Option<&mut *mut ::core::ffi::c_void>,
) -> CUresult {
    load_data(module, image)
}

pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    zluda_common::drop_checked::<Module>(hmod)
}

pub(crate) fn get_function(
    hfunc: &mut &function::Function,
    module: &Module,
    name: &CStr,
) -> CUresult {
    let name_str = name.to_string_lossy();
    super::driver::zluda_log(&format!("[ZLUDA] cuModuleGetFunction('{}')", name_str));
    let func = module.get_function(name)?;
    super::function::register_name(func.base, &name_str);
    super::driver::zluda_log_err(&format!(
        "cuModuleGetFunction('{}') -> func={:p}",
        name_str, func.base.0
    ));
    *hfunc = func;
    Ok(())
}

pub(crate) fn get_global_v2(
    dptr: *mut hipDeviceptr_t,
    bytes: *mut usize,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> hipError_t {
    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
    // lsh hipfix item 7: hipModuleGetGlobal called with bytes==NULL leaves the
    // global "unusable" — subsequent operations against it return
    // CUDA_ERROR_INVALID_VALUE or silently corrupt. Always pass a real local
    // out-pointer and copy back only if the caller asked for it.
    let mut local_bytes: usize = 0;
    let result = unsafe { hipModuleGetGlobal(dptr, &mut local_bytes, hmod.base, name) };
    if !bytes.is_null() {
        unsafe { *bytes = local_bytes };
    }
    let dptr_val = unsafe { (*dptr).0 };
    let addr_space = if (dptr_val as u64) < 0x7f0000000000 {
        "HOST"
    } else {
        "DEV"
    };
    // Track module global address in alloc range
    if result.is_ok() {
        super::memory::update_alloc_range(dptr_val as u64, local_bytes.max(256) as u64);
    }
    super::driver::zluda_log(&format!(
        "[ZLUDA] cuModuleGetGlobal('{}') -> {:?} dptr={:p} bytes={} [{}]",
        name_str, result, dptr_val, local_bytes, addr_space
    ));
    result
}

pub(crate) fn get_tex_ref(
    p_tex_ref: &mut CUtexref,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
    let mut tex_ref: *mut textureReference = std::ptr::null_mut();
    let result = unsafe { hipModuleGetTexRef(&mut tex_ref, hmod.base, name) };
    match result {
        Ok(()) => {
            super::driver::zluda_log(&format!(
                "[ZLUDA] cuModuleGetTexRef('{}') -> {:p}",
                name_str, tex_ref
            ));
            register_texref(hmod.base, &name_str, tex_ref, None);
            *p_tex_ref = tex_ref as CUtexref;
            Ok(())
        }
        Err(e) => {
            // .texref in PTX is compiled as a .global .u64 variable, not a HIP texture reference.
            // Try to find it as a global variable and create a synthetic texture reference.
            let mut dptr = hipDeviceptr_t(std::ptr::null_mut());
            let mut bytes = 0usize;
            let global_result =
                unsafe { hipModuleGetGlobal(&mut dptr, &mut bytes, hmod.base, name) };
            if global_result.is_ok() {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetTexRef('{}') -> found as global variable at {:p} ({} bytes), creating synthetic texref",
                    name_str, dptr.0, bytes
                ));
                // Allocate a textureReference on the heap and return it.
                // The device-side global variable holds the texture object handle (u64).
                // When cuTexRefSetArray/cuTexObjectCreate is called, we'll write the
                // texture object handle to this device global.
                let synth_ref = Box::new(SyntheticTexRef {
                    tex_ref: unsafe { mem::zeroed::<textureReference>() },
                    device_global: dptr,
                });
                let tex_ref_ptr =
                    &synth_ref.tex_ref as *const textureReference as *mut textureReference;
                // Register in global map so tex.rs can find the device global
                register_texref(hmod.base, &name_str, tex_ref_ptr, Some(dptr));
                // Leak the Box — the texture reference lives as long as the module
                Box::leak(synth_ref);
                *p_tex_ref = tex_ref_ptr as CUtexref;
                Ok(())
            } else {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetTexRef('{}') -> NOT FOUND (hip={:?}, global={:?})",
                    name_str, e, global_result
                ));
                Err(e.into())
            }
        }
    }
}

/// A synthetic texture reference for .texref variables compiled as .global .u64
#[repr(C)]
struct SyntheticTexRef {
    tex_ref: textureReference,
    device_global: hipDeviceptr_t,
}

pub(crate) fn get_surf_ref(
    p_surf_ref: &mut cuda_types::cuda::CUsurfref,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
    // HIP doesn't have a separate hipModuleGetSurfRef - use hipModuleGetTexRef
    // since surface and texture references share the same underlying textureReference struct
    let mut tex_ref: *mut textureReference = std::ptr::null_mut();
    let result = unsafe { hipModuleGetTexRef(&mut tex_ref, hmod.base, name) };
    match result {
        Ok(()) => {
            super::driver::zluda_log(&format!(
                "[ZLUDA] cuModuleGetSurfRef('{}') -> {:p}",
                name_str, tex_ref
            ));
            *p_surf_ref = tex_ref as cuda_types::cuda::CUsurfref;
            Ok(())
        }
        Err(_) => {
            // .surfref compiled as .global .u64 — same approach as .texref
            let mut dptr = hipDeviceptr_t(std::ptr::null_mut());
            let mut bytes = 0usize;
            let global_result =
                unsafe { hipModuleGetGlobal(&mut dptr, &mut bytes, hmod.base, name) };
            if global_result.is_ok() {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetSurfRef('{}') -> found as global at {:p}, creating synthetic surfref",
                    name_str, dptr.0
                ));
                let synth_ref = Box::new(SyntheticTexRef {
                    tex_ref: unsafe { mem::zeroed::<textureReference>() },
                    device_global: dptr,
                });
                let tex_ref_ptr =
                    &synth_ref.tex_ref as *const textureReference as *mut textureReference;
                if let Ok(mut map) = SYNTHETIC_TEXREFS.lock() {
                    map.insert(
                        tex_ref_ptr as usize,
                        SyntheticTexRefState {
                            device_global: dptr,
                            linear_binding: None,
                        },
                    );
                }
                Box::leak(synth_ref);
                *p_surf_ref = tex_ref_ptr as cuda_types::cuda::CUsurfref;
                Ok(())
            } else {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetSurfRef('{}') -> NOT FOUND",
                    name_str
                ));
                Err(result.unwrap_err().into())
            }
        }
    }
}

pub(crate) fn get_loading_mode(mode: &mut cuda_types::cuda::CUmoduleLoadingMode) -> CUresult {
    *mode = cuda_types::cuda::CUmoduleLoadingMode::CU_MODULE_LAZY_LOADING;
    Ok(())
}

pub(crate) fn load_fat_binary(module: &mut CUmodule, image: &std::ffi::c_void) -> CUresult {
    load_data(module, image)
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    #[test]
    fn patch_skips_non_bbox_ptx() {
        let ptx = "// not a bbox kernel\n.visible .entry foo() { ret; }\n";
        assert!(patch_iofx_bbox_tail_merge(ptx).is_none());
    }

    #[test]
    fn bbox_bit31_guard_injects_and_parses() {
        std::env::remove_var("ZLUDA_BBOX_BIT31_GUARD");
        let ptx = "\
.version 7.0
.target sm_50
.address_size 64
.visible .entry IOFX_bboxKernel()
{
.reg .b32 %r<8>;
.reg .f32 %f<8>;
.reg .b64 %rd<8>;
mov.u32 %r1, %tid.x;
BB_W:
mul.wide.u32 %rd1, %r22, 16;
add.s64 %rd2, %rd5, %rd1;
st.global.v4.f32 [%rd2], {%f1, %f2, %f3, %f4};
BB_NEXT:
ret;
}
";
        let out = patch_iofx_bbox_bit31_guard(ptx).expect("bbox write should be guarded");
        assert!(out.contains("and.b32 %rbg0, %r22, -2147483648;"), "bit31 mask missing:\n{}", out);
        assert!(out.contains("@%pbg0 bra BB_NEXT;"), "guard branch to next BB missing:\n{}", out);
        // guard must be BEFORE the mul.wide store-index computation.
        assert!(out.find("@%pbg0 bra BB_NEXT;").unwrap() < out.find("mul.wide.u32 %rd1, %r22, 16;").unwrap());
        ptx_parser::parse_module_checked(&out).expect("guarded bbox PTX must parse");
        std::env::set_var("ZLUDA_BBOX_BIT31_GUARD", "0");
        assert!(patch_iofx_bbox_bit31_guard(ptx).is_none(), "gate ignored");
        std::env::remove_var("ZLUDA_BBOX_BIT31_GUARD");
    }

    #[test]
    #[ignore = "reads run_archives/classb_2026_06_14/bbox_module.ptx — verifies the real bbox kernel guards + parses"]
    fn bbox_bit31_guard_on_real_module_parses() {
        std::env::remove_var("ZLUDA_BBOX_BIT31_GUARD");
        let src = std::fs::read_to_string("../../run_archives/classb_2026_06_14/bbox_module.ptx")
            .expect("bbox_module.ptx present");
        let out = patch_iofx_bbox_bit31_guard(&src).expect("real bbox module should be guarded");
        ptx_parser::parse_module_checked(&out).expect("guarded real bbox module must parse");
        let n = out.matches("and.b32 %rbg").count();
        eprintln!("real bbox module guarded {} store sites + parsed OK", n);
    }


    #[test]
    fn patch_inserts_guard_for_bboxkernel_pattern() {
        let ptx = r#"
.visible .entry IOFX_bboxKernel() {
.reg .pred %p<200>;
.reg .s32 %r<800>;
.reg .s64 %rd<400>;
BB10_184:
setp.eq.s64	%p122, %rd305, 0;
@%p122 bra BB10_186;

ld.global.v4.f32 %f1, [%rd304];
BB10_186:
ret;
}
"#;
        let patched = patch_iofx_bbox_tail_merge(ptx).expect("should patch");
        assert!(
            patched.contains("ZLUDA IOFX_bbox fix"),
            "patched PTX missing fix marker:\n{}",
            patched
        );
        // Confirm original guard preserved AND new tid-0 guard added.
        assert!(patched.contains("setp.eq.s64\t%p122, %rd305, 0;"));
        assert!(patched.contains("@%p122 bra BB10_186;"));
        assert!(patched.contains("setp.ne.s32 %p_zluda_bbox_lane0_0, %r_zluda_bbox_tid_0, 0;"));
        assert!(patched.contains("@%p_zluda_bbox_lane0_0 bra BB10_186;"));
    }

    #[test]
    fn patch_handles_bboxsync_different_register_numbers() {
        // Same structural pattern, different register numbers (matching the
        // actual IOFX_bboxSyncKernel from Batman AK).
        let ptx = r#"
.visible .entry IOFX_bboxSyncKernel() {
.reg .pred %p<200>;
.reg .s32 %r<800>;
.reg .s64 %rd<600>;
BB9_146:
setp.eq.s64	%p103, %rd540, 0;
@%p103 bra BB9_148;

ld.global.v4.f32 %f1, [%rd539];
BB9_148:
ret;
}
"#;
        let patched = patch_iofx_bbox_tail_merge(ptx).expect("should patch");
        assert!(patched.contains("@%p_zluda_bbox_lane0_0 bra BB9_148;"));
        // Original predicate/register preserved.
        assert!(patched.contains("setp.eq.s64\t%p103, %rd540, 0;"));
    }

    #[test]
    fn patch_inserts_guard_for_bb10_188_r94_pattern() {
        // BB10_188-style pattern: a `setp.eq.s32 %pX, %rY, -1; @%pX bra TARGET;`
        // followed shortly by `mul.wide.u32 %rd?, %rY, 16;` — the per-lane
        // wild-pointer compute. A tid==0 guard must be inserted after @%pX bra.
        let ptx = r#"
.visible .entry IOFX_bboxKernel() {
.reg .pred %p<200>;
.reg .s32 %r<800>;
.reg .s64 %rd<400>;
BB10_188:
setp.eq.s32	%p124, %r94, -1;
@%p124 bra BB10_195;

@%p117 bra BB10_191;

mul.wide.u32 %rd290, %r94, 16;
add.s64 %rd307, %rd3, %rd290;
BB10_195:
ret;
}
"#;
        let patched = patch_iofx_bbox_tail_merge(ptx).expect("should patch");
        assert!(
            patched.contains("@%p_zluda_bbox_lane0_0 bra BB10_195;"),
            "missing tid==0 guard branch to BB10_195:\n{}",
            patched
        );
        // Original -1 guard preserved.
        assert!(patched.contains("setp.eq.s32\t%p124, %r94, -1;"));
        assert!(patched.contains("@%p124 bra BB10_195;"));
    }

    #[test]
    fn patch_inserts_guard_for_bb9_202_sync_pattern() {
        // BB9_202-style pattern in IOFX_bboxSyncKernel: load -> setp -1 -> bra
        // -> mul.wide.u32 %r, 16.
        let ptx = r#"
.visible .entry IOFX_bboxSyncKernel() {
.reg .pred %p<200>;
.reg .s32 %r<800>;
.reg .s64 %rd<600>;
BB9_202:
ld.volatile.shared.u32 %r89, [%rd9];
setp.eq.s32	%p132, %r89, -1;
@%p132 bra BB9_204;

mul.wide.u32 %rd532, %r89, 16;
add.s64 %rd533, %rd530, %rd532;
BB9_204:
ret;
}
"#;
        let patched = patch_iofx_bbox_tail_merge(ptx).expect("should patch");
        assert!(
            patched.contains("@%p_zluda_bbox_lane0_0 bra BB9_204;"),
            "missing tid==0 guard branch to BB9_204:\n{}",
            patched
        );
    }

    #[test]
    fn patch_skips_minus_one_pattern_without_mul_wide() {
        // A setp.eq.s32 ..., -1 pattern that is NOT followed by
        // `mul.wide.u32 %rY, 16` within the BB must not be patched
        // (common case: a check on a uniformly-loaded shared value that
        // does not produce a per-lane store pointer).
        let ptx = r#"
.visible .entry IOFX_bboxKernel() {
.reg .pred %p<200>;
.reg .s32 %r<800>;
BB10_10:
setp.eq.s32	%p1, %r1, -1;
@%p1 bra BB10_11;

mov.u32 %r2, 42;
BB10_11:
ret;
}
"#;
        // No mul.wide.u32 ..., 16 in sight → no patch for this pattern.
        assert!(patch_iofx_bbox_tail_merge(ptx).is_none());
    }

    #[test]
    fn patch_both_bb184_and_bb188_patterns_together() {
        // Both patterns in the same kernel should produce two guards.
        let ptx = r#"
.visible .entry IOFX_bboxKernel() {
.reg .pred %p<200>;
.reg .s32 %r<800>;
.reg .s64 %rd<400>;
BB10_184:
setp.eq.s64	%p122, %rd305, 0;
@%p122 bra BB10_186;

BB10_188:
setp.eq.s32	%p124, %r94, -1;
@%p124 bra BB10_195;

mul.wide.u32 %rd290, %r94, 16;
BB10_195:
ret;
}
"#;
        let patched = patch_iofx_bbox_tail_merge(ptx).expect("should patch");
        assert_eq!(
            patched.matches("ZLUDA IOFX_bbox fix").count(),
            2,
            "expected 2 guards, got:\n{}",
            patched
        );
        assert!(patched.contains("@%p_zluda_bbox_lane0_0 bra BB10_186;"));
        assert!(patched.contains("@%p_zluda_bbox_lane0_1 bra BB10_195;"));
    }

    #[test]
    fn patch_skips_same_pattern_outside_bbox_kernels() {
        // Same structural pattern but inside a kernel that doesn't start with
        // IOFX_bbox — must NOT patch. (This is the pattern we use for null
        // pointer checks; over-patching would break unrelated kernels.)
        let ptx = r#"
.visible .entry something_else() {
.reg .pred %p<200>;
.reg .s64 %rd<400>;
BB1_1:
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB1_2;
BB1_2:
ret;
}
"#;
        // Must return None because kernel name doesn't contain "IOFX_bbox".
        assert!(patch_iofx_bbox_tail_merge(ptx).is_none());
    }

    #[test]
    fn patch_handles_multiple_occurrences_in_same_kernel() {
        // If the compiler emitted more than one of these guards, patch each.
        let ptx = r#"
.visible .entry IOFX_bboxKernel() {
BB1_1:
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB1_2;
BB1_2:
setp.eq.s64	%p2, %rd2, 0;
@%p2 bra BB1_3;
BB1_3:
ret;
}
"#;
        let patched = patch_iofx_bbox_tail_merge(ptx).expect("should patch");
        let insertions = patched.matches("ZLUDA IOFX_bbox fix").count();
        assert_eq!(insertions, 2, "expected 2 patches, got:\n{}", patched);
        assert!(patched.contains("@%p_zluda_bbox_lane0_0 bra BB1_2;"));
        assert!(patched.contains("@%p_zluda_bbox_lane0_1 bra BB1_3;"));
    }

    // ----------------------------------------------------------------
    // 2026-06-09 MS_4 underflow guard tests
    // ----------------------------------------------------------------

    #[test]
    fn ms4_patch_skips_non_ms4_ptx() {
        let ptx = "// not MS_4\n.visible .entry foo() { ret; }\n";
        assert!(patch_ms4_underflow_guard(ptx).is_none());
    }

    #[test]
    fn ms4_patch_skips_ms4_without_scatter_pattern() {
        // Has the kernel name but lacks the 4-line scatter pattern.
        let ptx = r#"
.visible .entry CreateParticleShapesIncremental_MultiStage_4() {
.reg .pred %p<3>;
.reg .s32 %r<10>;
BB13_3:
ret;
}
"#;
        assert!(patch_ms4_underflow_guard(ptx).is_none());
    }

    #[test]
    fn ms4_patch_inserts_underflow_guard_canonical_pattern() {
        // Verbatim shape of the PTX dump (BB13_3 = ret label).
        let ptx = r#"
.visible .entry CreateParticleShapesIncremental_MultiStage_4() {
.reg .pred %p<3>;
.reg .s32 %r<11>;
.reg .s64 %rd<22>;
atom.global.add.u32 %r8, [%rd19], -1;
add.s32 %r9, %r7, %r8;
add.s32 %r10, %r9, -1;
mul.wide.u32 %rd20, %r10, 4;
add.s64 %rd21, %rd10, %rd20;
st.global.u32 [%rd21], %r1;

BB13_3:
.loc 4 1071 2
ret;
}
"#;
        let patched =
            patch_ms4_underflow_guard(ptx).expect("MS_4 underflow guard should patch canonical");
        // setp.lt.s32 should appear, guarding %r10 against negative
        assert!(
            patched.contains("setp.lt.s32 %p_ms4_uf_0, %r10, 0"),
            "missing setp.lt:\n{}",
            patched
        );
        // bra should target BB13_3 (the kernel's ret label)
        assert!(
            patched.contains("@%p_ms4_uf_0 bra BB13_3"),
            "missing bra to BB13_3:\n{}",
            patched
        );
        // Original mul.wide.u32 must still be present (not stripped)
        assert!(
            patched.contains("mul.wide.u32 %rd20, %r10, 4"),
            "mul.wide.u32 was stripped:\n{}",
            patched
        );
        // Guard must be BEFORE the mul.wide.u32 line
        let guard_pos = patched.find("setp.lt.s32 %p_ms4_uf_0").unwrap();
        let mul_pos = patched.find("mul.wide.u32 %rd20").unwrap();
        assert!(guard_pos < mul_pos, "guard not before mul:\n{}", patched);
    }

    #[test]
    fn ms4_patch_skips_when_register_mismatch() {
        // %r10 in add, but mul.wide uses %r11 — must NOT patch.
        let ptx = r#"
.visible .entry CreateParticleShapesIncremental_MultiStage_4() {
add.s32 %r10, %r9, -1;
mul.wide.u32 %rd20, %r11, 4;
add.s64 %rd21, %rd10, %rd20;
st.global.u32 [%rd21], %r1;
BB13_3:
ret;
}
"#;
        assert!(patch_ms4_underflow_guard(ptx).is_none());
    }

    #[test]
    fn ms4_patch_skips_when_addend_not_minus_one() {
        // add.s32 with +1 instead of -1 — not the underflow pattern.
        let ptx = r#"
.visible .entry CreateParticleShapesIncremental_MultiStage_4() {
add.s32 %r10, %r9, 1;
mul.wide.u32 %rd20, %r10, 4;
add.s64 %rd21, %rd10, %rd20;
st.global.u32 [%rd21], %r1;
BB13_3:
ret;
}
"#;
        assert!(patch_ms4_underflow_guard(ptx).is_none());
    }

    #[test]
    fn ms4_patch_respects_env_off_switch() {
        std::env::set_var("ZLUDA_MS4_UNDERFLOW_GUARD", "0");
        let ptx = r#"
.visible .entry CreateParticleShapesIncremental_MultiStage_4() {
add.s32 %r10, %r9, -1;
mul.wide.u32 %rd20, %r10, 4;
add.s64 %rd21, %rd10, %rd20;
st.global.u32 [%rd21], %r1;
BB13_3:
ret;
}
"#;
        let result = patch_ms4_underflow_guard(ptx);
        std::env::remove_var("ZLUDA_MS4_UNDERFLOW_GUARD");
        assert!(result.is_none(), "should skip when env=0");
    }

    // ----------------------------------------------------------------
    // 2026-06-03 Step 2: SimpleMemset AST detector tests
    // ----------------------------------------------------------------

    fn simple_memset_clear_clean_table() {
        if let Ok(mut tbl) = super::SIMPLE_MEMSET_TABLE.lock() {
            tbl.clear();
        }
    }

    fn parse_and_detect(ptx: &str) {
        let module = ptx_parser::parse_module_checked(ptx)
            .expect("test PTX must parse");
        super::detect_simple_memset_kernels(&module);
    }

    /// Real PhysX TFS_dynamicGridClearKernel body (taken from
    /// /tmp/zluda_ptx_042.ptx:8637-8676) must be recognized as a
    /// SimpleMemset and registered with the correct (surf=5, fill=6)
    /// param indices.
    #[test]
    fn simple_memset_detects_real_tfs_dynamic_grid_clear() {
        simple_memset_clear_clean_table();
        let ptx = r#".version 6.0
.target sm_30
.address_size 64

.visible .entry TurbulenceFS_dynamicGridClearKernel(
    .param .u64 TurbulenceFS_dynamicGridClearKernel_param_0,
    .param .u16 TurbulenceFS_dynamicGridClearKernel_param_1,
    .param .u32 TurbulenceFS_dynamicGridClearKernel_param_2,
    .param .u32 TurbulenceFS_dynamicGridClearKernel_param_3,
    .param .u32 TurbulenceFS_dynamicGridClearKernel_param_4,
    .param .u64 TurbulenceFS_dynamicGridClearKernel_param_5,
    .param .u32 TurbulenceFS_dynamicGridClearKernel_param_6
)
{
    .reg .s32 %r<15>;
    .reg .s64 %rd<2>;

    ld.param.u64 %rd1, [TurbulenceFS_dynamicGridClearKernel_param_5];
    ld.param.u32 %r4, [TurbulenceFS_dynamicGridClearKernel_param_6];
    mov.u32 %r5, %ctaid.z;
    mov.u32 %r6, %ctaid.y;
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mov.u32 %r9, %tid.x;
    mad.lo.s32 %r10, %r8, %r7, %r9;
    mov.u32 %r11, %ntid.y;
    mov.u32 %r12, %tid.y;
    mad.lo.s32 %r2, %r11, %r6, %r12;
    mov.u32 %r13, %ntid.z;
    mov.u32 %r14, %tid.z;
    mad.lo.s32 %r3, %r13, %r5, %r14;
    shl.b32 %r1, %r10, 2;
    sust.b.3d.b32.zero [%rd1, {%r1, %r2, %r3, %r3}], {%r4};
    ret;
}
"#;
        parse_and_detect(ptx);
        let sig = super::lookup_simple_memset_sig("TurbulenceFS_dynamicGridClearKernel")
            .expect("clear kernel should be registered");
        assert_eq!(sig.surf_param_idx, 5, "expected surf=param_5");
        assert_eq!(sig.fill_param_idx, 6, "expected fill=param_6");
        assert!(sig.is_sust_3d, "expected sust.b.3d");
        assert_eq!(sig.surf_param_width, 8, "u64 surface handle");
        assert_eq!(sig.fill_param_width, 4, "b32 fill value");
        // NVCC ABI offsets:
        //   _0 u64 @0..8;  _1 u16 @8..10;  _2 u32 @12..16 (align 4);
        //   _3 u32 @16..20; _4 u32 @20..24; _5 u64 @24..32 (align 8);
        //   _6 u32 @32..36
        assert_eq!(sig.surf_param_offset, 24);
        assert_eq!(sig.fill_param_offset, 32);
    }

    /// A kernel with a `bar.sync` in its body must NOT be flagged as
    /// a SimpleMemset — barriers indicate the kernel relies on lane
    /// coordination and is not a trivial fill.
    #[test]
    fn simple_memset_rejects_kernels_with_bar_sync() {
        simple_memset_clear_clean_table();
        let ptx = r#".version 6.0
.target sm_30
.address_size 64

.visible .entry KernelWithBar(
    .param .u64 KernelWithBar_param_0,
    .param .u32 KernelWithBar_param_1
)
{
    .reg .s32 %r<5>;
    .reg .s64 %rd<2>;
    ld.param.u64 %rd1, [KernelWithBar_param_0];
    ld.param.u32 %r1, [KernelWithBar_param_1];
    mov.u32 %r2, %tid.x;
    bar.sync 0;
    sust.b.3d.b32.zero [%rd1, {%r2, %r2, %r2, %r2}], {%r1};
    ret;
}
"#;
        parse_and_detect(ptx);
        assert!(
            super::lookup_simple_memset_sig("KernelWithBar").is_none(),
            "kernel with bar.sync must not be registered"
        );
    }

    /// A kernel that reads from `ld.global` (i.e., its address or fill
    /// depends on values fetched from device memory) is NOT a simple
    /// memset — its semantic depends on memory state. Reject.
    #[test]
    fn simple_memset_rejects_kernels_with_ld_global() {
        simple_memset_clear_clean_table();
        let ptx = r#".version 6.0
.target sm_30
.address_size 64

.visible .entry KernelWithGlobalLoad(
    .param .u64 KernelWithGlobalLoad_param_0,
    .param .u64 KernelWithGlobalLoad_param_1
)
{
    .reg .s32 %r<5>;
    .reg .s64 %rd<3>;
    ld.param.u64 %rd1, [KernelWithGlobalLoad_param_0];
    ld.param.u64 %rd2, [KernelWithGlobalLoad_param_1];
    ld.global.u32 %r1, [%rd2];
    mov.u32 %r2, %tid.x;
    sust.b.3d.b32.zero [%rd1, {%r2, %r2, %r2, %r2}], {%r1};
    ret;
}
"#;
        parse_and_detect(ptx);
        assert!(
            super::lookup_simple_memset_sig("KernelWithGlobalLoad").is_none(),
            "kernel with ld.global must not be registered"
        );
    }

    /// A kernel with TWO `sust` ops (or a `sust` plus an `st.global`)
    /// is more than a trivial fill — reject.
    #[test]
    fn simple_memset_rejects_kernels_with_two_sust_ops() {
        simple_memset_clear_clean_table();
        let ptx = r#".version 6.0
.target sm_30
.address_size 64

.visible .entry TwoSustKernel(
    .param .u64 TwoSustKernel_param_0,
    .param .u64 TwoSustKernel_param_1,
    .param .u32 TwoSustKernel_param_2
)
{
    .reg .s32 %r<3>;
    .reg .s64 %rd<3>;
    ld.param.u64 %rd1, [TwoSustKernel_param_0];
    ld.param.u64 %rd2, [TwoSustKernel_param_1];
    ld.param.u32 %r1, [TwoSustKernel_param_2];
    mov.u32 %r2, %tid.x;
    sust.b.3d.b32.zero [%rd1, {%r2, %r2, %r2, %r2}], {%r1};
    sust.b.3d.b32.zero [%rd2, {%r2, %r2, %r2, %r2}], {%r1};
    ret;
}
"#;
        parse_and_detect(ptx);
        assert!(
            super::lookup_simple_memset_sig("TwoSustKernel").is_none(),
            "kernel with two sust ops must not be registered"
        );
    }



    #[test]
    #[ignore = "writes /tmp/iofx_radixsort_patched.ptx — the fixture run through the production default-ON patch pipeline, for the NRSS harness (NRSS_PTX_PATH)"]
    fn patch_produce_iofx_radixsort_fixture() {
        // Mirror get_best_ptx_and_compile's default-ON pipeline order so the
        // solo NRSS harness sees the SAME PTX the game compiles. Honors the
        // same env gates (set ZLUDA_* to flip a stage). Stages that don't
        // match (bbox/scan/ms4) no-op on this fixture.
        let src =
            std::fs::read_to_string("../ptx/src/test/spirv_run/iofx_radixsort.ptx").unwrap();
        let mut text = std::borrow::Cow::Borrowed(src.as_str());
        let mut fired = Vec::new();
        macro_rules! stage {
            ($name:literal, $f:expr) => {
                if let Some(p) = $f(text.as_ref()) {
                    text = std::borrow::Cow::Owned(p);
                    fired.push($name);
                }
            };
        }
        stage!("iofx_bbox_tail_merge", patch_iofx_bbox_tail_merge);
        stage!("broad_volatile_lds", patch_broad_volatile_lds);
        stage!("broad_membar_before_bar_sync", patch_broad_membar_before_bar_sync);
        stage!("scan_kernel_reader_acquire", patch_scan_kernel_reader_acquire);
        stage!("divergent_lds_merge_sync", patch_divergent_lds_merge_sync);
        stage!("lds_producer_volatile_store", patch_lds_producer_volatile_store);
        stage!("ms4_underflow_guard", patch_ms4_underflow_guard);
        std::fs::write("/tmp/iofx_radixsort_patched.ptx", text.as_ref()).unwrap();
        eprintln!(
            "wrote {} bytes (was {} bytes) to /tmp/iofx_radixsort_patched.ptx; stages fired: {:?}",
            text.len(),
            src.len(),
            fired
        );
    }

    #[test]
    #[ignore = "writes /tmp/module_56_zluda_patched.ptx from /tmp/zluda_ptx_dump/module_56.ptx"]
    fn patch_produce_patched_module_56() {
        let src = std::fs::read_to_string("/tmp/zluda_ptx_dump/module_56.ptx").unwrap();
        let patched = patch_iofx_bbox_tail_merge(&src)
            .expect("expected module_56.ptx to match the bbox pattern");
        std::fs::write("/tmp/module_56_zluda_patched.ptx", &patched).unwrap();
        eprintln!(
            "wrote {} bytes (was {} bytes) to /tmp/module_56_zluda_patched.ptx",
            patched.len(),
            src.len()
        );
    }

    #[test]
    #[ignore = "writes /tmp/module_23_divergent_patched.ptx from /tmp/zluda_ptx_dump/module_23.ptx"]
    fn patch_produce_divergent_patched_module_23() {
        let src = std::fs::read_to_string("/tmp/zluda_ptx_dump/module_23.ptx").unwrap();
        let patched = patch_divergent_lds_merge_sync(&src);
        match patched {
            Some(p) => {
                std::fs::write("/tmp/module_23_divergent_patched.ptx", &p).unwrap();
                eprintln!(
                    "PATCHED: wrote {} bytes (was {} bytes) to /tmp/module_23_divergent_patched.ptx",
                    p.len(),
                    src.len()
                );
            }
            None => {
                eprintln!("NOT PATCHED (patch_divergent_lds_merge_sync returned None)");
                panic!("expected module_23.ptx to match divergent_lds_merge_sync pattern");
            }
        }
    }

    #[test]
    fn patch_leaves_unrelated_bb_labels_alone() {
        // A BBLabel with no matching setp/bra pair following must not be touched.
        let ptx = r#"
.visible .entry IOFX_bboxKernel() {
BB1_1:
setp.ne.s64	%p1, %rd1, 0;
@%p1 bra BB1_2;
BB1_2:
ret;
}
"#;
        // setp.ne not setp.eq → not the pattern → no patch → None.
        assert!(patch_iofx_bbox_tail_merge(ptx).is_none());
    }

    // ===== patch_divergent_lds_merge_sync tests =====

    fn ensure_default_on() {
        // The patch is off by default since the retirement profile; these
        // tests exercise it, so enable it explicitly.
        std::env::set_var("ZLUDA_DIVERGENT_LDS_MERGE_SYNC", "1");
    }

    #[test]
    fn divergent_sync_inserts_at_merge_when_st_then_ld_shared() {
        ensure_default_on();
        // Canonical NRSS-style pattern: bar.sync 0, conditional bra to
        // merge label, st.shared in the fall-through, then merge with
        // ld.shared. Must insert bar.sync 0 at the merge.
        let ptx = r#".visible .entry test_k() {
bar.sync 0;
shr.u32 %r5, %tid.x, 5;
setp.lt.u32 %p1, %r5, %r6;
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx).expect("should patch");
        assert!(
            patched.contains("// ZLUDA divergent-LDS-write merge sync\nbar.sync 0;\nld.shared"),
            "expected bar.sync 0 inserted between L_MERGE: and the ld.shared, got:\n{}",
            patched
        );
    }

    #[test]
    fn divergent_sync_handles_at_bang_predicate() {
        ensure_default_on();
        let ptx = r#".visible .entry test_k() {
bar.sync 0;
shr.u32 %r10, %tid.x, 5;
setp.lt.u32 %p1, %r10, %r20;
@%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx).expect("should patch (@p variant)");
        assert!(patched.matches("bar.sync 0;").count() >= 2);
    }

    #[test]
    fn divergent_sync_skips_if_no_st_shared_in_fall_through() {
        ensure_default_on();
        // No st.shared between bra and merge → no race possible → no patch.
        let ptx = r#".visible .entry test_k() {
@!%p1 bra L_MERGE;
add.s32 %r5, %r6, %r7;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        assert!(
            patch_divergent_lds_merge_sync(ptx).is_none(),
            "should not patch when fall-through has no st.shared"
        );
    }

    #[test]
    fn divergent_sync_skips_if_no_ld_shared_after_merge() {
        ensure_default_on();
        // Merge label has no ld.shared before next bar.sync → no race.
        let ptx = r#".visible .entry test_k() {
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
add.s32 %r3, %r4, %r5;
bar.sync 0;
ret;
}
"#;
        assert!(
            patch_divergent_lds_merge_sync(ptx).is_none(),
            "should not patch when post-merge has no ld.shared before next bar.sync"
        );
    }

    #[test]
    fn divergent_sync_skips_if_already_followed_by_bar_sync() {
        ensure_default_on();
        let ptx = r#".visible .entry test_k() {
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
bar.sync 0;
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        assert!(
            patch_divergent_lds_merge_sync(ptx).is_none(),
            "should not patch when merge label is already followed by bar.sync"
        );
    }

    #[test]
    fn divergent_sync_skips_if_ret_between_bra_and_merge() {
        ensure_default_on();
        // Early `ret;` in writer block → inserting bar.sync at merge
        // would deadlock since the exited threads can't reach it.
        let ptx = r#".visible .entry test_k() {
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
ret;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        assert!(
            patch_divergent_lds_merge_sync(ptx).is_none(),
            "should not patch when there's an early ret in the writer block"
        );
    }

    #[test]
    fn divergent_sync_env_off_disables() {
        std::env::set_var("ZLUDA_DIVERGENT_LDS_MERGE_SYNC", "0");
        let ptx = r#".visible .entry test_k() {
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let result = patch_divergent_lds_merge_sync(ptx);
        std::env::remove_var("ZLUDA_DIVERGENT_LDS_MERGE_SYNC");
        assert!(result.is_none(), "env=0 should disable the patch");
    }

    #[test]
    fn divergent_sync_does_not_double_patch_same_label() {
        ensure_default_on();
        // Two conditional bras to the same merge label, each with st.shared
        // in its fall-through. Should produce exactly one inserted barrier.
        let ptx = r#".visible .entry test_k() {
shr.u32 %r10, %tid.x, 5;
setp.lt.u32 %p1, %r10, %r20;
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
setp.lt.u32 %p2, %r10, %r21;
@%p2 bra L_MERGE;
st.shared.u32 [%r3], %r4;
L_MERGE:
ld.shared.u32 %r5, [%r6];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx).expect("should patch");
        let injected = patched
            .matches("// ZLUDA divergent-LDS-write merge sync")
            .count();
        assert_eq!(
            injected, 1,
            "should insert exactly one bar.sync at the shared merge, got {}:\n{}",
            injected, patched
        );
    }

    #[test]
    fn divergent_sync_matches_volatile_shared_load() {
        // Production-critical: patch_broad_volatile_lds runs BEFORE this
        // pass and converts ld.shared.* → ld.volatile.shared.*. Without
        // matching the volatile form, this pass would silently miss every
        // merge site where reader-only warps read from LDS — exactly the
        // production bug found 2026-05-13 (run 3 SIGABRT after run 1+2
        // looked clean).
        ensure_default_on();
        let ptx = r#".visible .entry test_k() {
shr.u32 %r10, %tid.x, 5;
setp.lt.u32 %p1, %r10, %r20;
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.volatile.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx).expect("should patch (volatile ld)");
        assert!(
            patched.contains("// ZLUDA divergent-LDS-write merge sync\nbar.sync 0;"),
            "expected insertion; got:\n{}",
            patched
        );
    }

    #[test]
    fn divergent_sync_matches_volatile_shared_store() {
        // Defensive: although `patch_broad_volatile_lds` currently only
        // touches loads, a future patch (or upstream code) may emit
        // `st.volatile.shared.*` directly. We must catch that as a writer
        // store too.
        ensure_default_on();
        let ptx = r#".visible .entry test_k() {
shr.u32 %r10, %tid.x, 5;
setp.lt.u32 %p1, %r10, %r20;
@!%p1 bra L_MERGE;
st.volatile.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx).expect("should patch (volatile st)");
        assert!(patched.contains("bar.sync 0;\nld.shared"));
    }

    // ===== Warp-divergence discriminator tests (2026-05-14) =====
    // The synthetic NRSS race is real, but the v1 generic pattern detector
    // over-matched uniform-per-warp predicates in IOFX_bbox/sprite/modifier
    // kernels, causing in-game regression. These tests enforce that the
    // tightened pattern detector ONLY fires on warp-id-derived predicates.

    #[test]
    fn divergent_sync_fires_on_warp_id_predicate_nrss_style() {
        ensure_default_on();
        // NRSS BB20_201 pattern: `setp.lt.u32 %p, %warpid, %active_warps`
        // where %warpid is `shr.u32 %_, %tid.x, 5`. Cross-warp divergent.
        let ptx = r#".visible .entry test_k() {
shr.u32 %r983, %tid.x, 5;
setp.lt.u32 %p6, %r983, %r984;
@!%p6 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx).expect("warp-id-derived bra must patch");
        assert!(patched.contains("// ZLUDA divergent-LDS-write merge sync\nbar.sync 0;\nld.shared"));
    }

    #[test]
    fn divergent_sync_fires_on_wave64_warp_id_shift_6() {
        ensure_default_on();
        // Wave64 variant: warpid = tid.x >> 6.
        let ptx = r#".visible .entry test_k() {
shr.u32 %r10, %tid.x, 6;
setp.lt.u32 %p1, %r10, %r20;
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx).expect("wave64 (>>6) must also match");
        assert!(patched.contains("bar.sync 0;\nld.shared"));
    }

    #[test]
    fn divergent_sync_skips_on_tid_zero_predicate_bbox_style() {
        ensure_default_on();
        // IOFX_bboxKernel BB10_184 pattern: tid.x == 0 check — uniform per
        // warp (all threads in a warp take the bra together).
        let ptx = r#".visible .entry test_k() {
setp.eq.s32 %p117, %tid.x, 0;
@!%p117 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        assert!(
            patch_divergent_lds_merge_sync(ptx).is_none(),
            "uniform tid==0 predicate must NOT patch (was the in-game regression cause)"
        );
    }

    #[test]
    fn divergent_sync_skips_on_composite_and_pred() {
        ensure_default_on();
        // IOFX_bboxKernel's actual chain: bra → @!%p121, where %p121 is
        // and.pred composite of multiple inner predicates. Composite
        // predicates are indeterminate — be conservative and skip.
        let ptx = r#".visible .entry test_k() {
setp.eq.s32 %p117, %tid.x, 0;
setp.eq.s32 %p118, %r722, -1;
and.pred %p119, %p117, %p118;
ld.volatile.shared.u32 %r723, [%r_addr];
setp.ne.s32 %p120, %r723, -1;
and.pred %p121, %p119, %p120;
@!%p121 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        assert!(
            patch_divergent_lds_merge_sync(ptx).is_none(),
            "composite and.pred predicate must NOT patch"
        );
    }

    #[test]
    fn divergent_sync_skips_on_shared_mem_fed_predicate() {
        ensure_default_on();
        // A setp fed by ld.shared/ld.const/ld.global is uniform per warp
        // (assuming the loaded value is wave-uniform, which it typically
        // is for the bbox-class pattern). Conservative: skip.
        let ptx = r#".visible .entry test_k() {
ld.shared.u32 %r5, [%r_addr];
setp.eq.s32 %p1, %r5, -1;
@!%p1 bra L_MERGE;
st.shared.u32 [%r10], %r11;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        assert!(
            patch_divergent_lds_merge_sync(ptx).is_none(),
            "shared-mem-fed predicate must NOT patch"
        );
    }

    #[test]
    fn divergent_sync_fires_when_shr_source_is_mov_of_tid_x() {
        ensure_default_on();
        // This is the ACTUAL NRSS pattern: mov.u32 %r1, %tid.x;
        // ... shr.u32 %r983, %r1, 5; — the shr source is NOT %tid.x
        // directly but a register that mov'd from %tid.x. Discriminator
        // must chase the mov chain.
        let ptx = r#".visible .entry test_k() {
mov.u32 %r1, %tid.x;
shr.u32 %r983, %r1, 5;
setp.lt.u32 %p6, %r983, %r984;
@!%p6 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx)
            .expect("shr source via mov-chain from %tid.x must still patch (NRSS pattern)");
        assert!(patched.contains("bar.sync 0;\nld.shared"));
    }

    #[test]
    fn divergent_sync_fires_when_tid_x_mov_is_far_from_use() {
        ensure_default_on();
        // Regression for 2026-06-02 bug found via iter15d NRSS crash analysis.
        // Real NRSS has `mov.u32 %r1, %tid.x;` right after kernel-entry
        // `.reg` directives, then USES `%r1` in `shr.u32 _, %r1, 5;` >1500
        // lines later inside the per-pass BlockScan code at BB20_201. The
        // old 200-line back-scan cap in is_register_bound_to_tid_x silently
        // failed to find the mov, the discriminator returned false, and
        // patch_divergent_lds_merge_sync skipped the BB20_201 merge site —
        // letting the BlockScan underflow → bit-31 → +8-10 GB wild store
        // ship to production. Fix removes the cap; this test ensures the
        // fix stays in place.
        let mut ptx = String::from(".visible .entry test_k() {\n");
        ptx.push_str("mov.u32 %r1, %tid.x;\n");
        // Pad with 500 unrelated instructions to defeat any cap below 500.
        for k in 0..500 {
            ptx.push_str(&format!("mov.u32 %r{}, 0;\n", k + 100));
        }
        ptx.push_str("shr.u32 %r983, %r1, 5;\n");
        ptx.push_str("setp.lt.u32 %p6, %r983, %r984;\n");
        ptx.push_str("@!%p6 bra L_MERGE;\n");
        ptx.push_str("st.shared.u32 [%r1], %r2;\n");
        ptx.push_str("L_MERGE:\n");
        ptx.push_str("ld.shared.u32 %r3, [%r4];\n");
        ptx.push_str("ret;\n}\n");
        let patched = patch_divergent_lds_merge_sync(&ptx)
            .expect("NRSS far-mov pattern must still patch (>500 lines between mov and use)");
        assert!(patched.contains("bar.sync 0;\nld.shared"));
    }

    #[test]
    fn divergent_sync_chases_one_mov_to_find_warp_id() {
        ensure_default_on();
        // A `mov` between `shr.u32 ..., %tid.x, 5` and the setp should
        // still resolve to warp-id-derived.
        let ptx = r#".visible .entry test_k() {
shr.u32 %r99, %tid.x, 5;
mov.u32 %r100, %r99;
setp.lt.u32 %p1, %r100, %r200;
@!%p1 bra L_MERGE;
st.shared.u32 [%r1], %r2;
L_MERGE:
ld.shared.u32 %r3, [%r4];
ret;
}
"#;
        let patched = patch_divergent_lds_merge_sync(ptx)
            .expect("warp-id via single mov chain must still patch");
        assert!(patched.contains("bar.sync 0;\nld.shared"));
    }

    #[test]
    fn divergent_sync_extract_label_handles_comments_and_dollars() {
        // Internal extraction helper covers ID forms used in real PTX.
        assert_eq!(extract_ptx_label("BB20_201:"), Some("BB20_201"));
        assert_eq!(extract_ptx_label("L_MERGE:"), Some("L_MERGE"));
        assert_eq!(extract_ptx_label("_label$0:"), Some("_label$0"));
        assert_eq!(extract_ptx_label("LABEL: // comment"), Some("LABEL"));
        assert_eq!(extract_ptx_label("not_a_label"), None);
        assert_eq!(extract_ptx_label("st.shared.u32 [%r1], %r2;"), None);
        assert_eq!(extract_ptx_label(".visible .entry foo:"), None); // has space
    }
}

#[cfg(test)]
mod tests {
    //! Regression tests for `cuModuleGetGlobal_v2`.
    //!
    //! lsh hipfix item 7: hipModuleGetGlobal called with `bytes == NULL`
    //! leaves the global "unusable" — subsequent operations against the
    //! returned device pointer return CUDA_ERROR_INVALID_VALUE or silently
    //! corrupt. Our wrapper always passes a real local out-pointer to HIP
    //! and copies back only if the caller asked for it.
    //!
    //! These tests go through the full Zluda extern "C" entry points (via
    //! `crate::cuXxx`) so they exercise the same code path the game hits.
    //! Skipped cleanly when no AMD GPU is available.
    use cuda_types::cuda::*;
    use std::ffi::CStr;
    use std::mem;
    use std::ptr;

    /// Tiny PTX with a single 4-byte global. We initialize it to a known
    /// non-zero value so a follow-up DtoH copy can verify the global is
    /// actually usable after the NULL-bytes get_global call.
    const PTX_WITH_GLOBAL: &CStr = c"
.version 7.0
.target sm_60
.address_size 64

.visible .global .align 4 .u32 my_global = 305419896;
";

    /// Helper: minimal "init + ctx + module load" boilerplate. Returns the
    /// loaded module handle, or `None` if the GPU/runtime is unavailable
    /// (test should skip cleanly in that case).
    unsafe fn try_load_test_module() -> Option<CUmodule> {
        if crate::cuInit(0).is_err() {
            return None;
        }
        let mut ctx: CUcontext = mem::zeroed();
        if crate::cuCtxCreate_v2(&mut ctx, 0, 0 as CUdevice).is_err() {
            return None;
        }
        let mut module: CUmodule = mem::zeroed();
        if crate::cuModuleLoadData(&mut module, PTX_WITH_GLOBAL.as_ptr().cast()).is_err() {
            return None;
        }
        Some(module)
    }

    /// REGRESSION: calling cuModuleGetGlobal_v2 with bytes=NULL must leave
    /// the global usable for subsequent device-memory operations. Without
    /// the lsh hipfix item 7 workaround in `get_global_v2`, hipModuleGetGlobal
    /// internally marks the global as broken and the follow-up DtoH copy
    /// fails or returns garbage.
    #[test]
    fn get_global_with_null_bytes_keeps_global_usable() {
        unsafe {
            let module = match try_load_test_module() {
                Some(m) => m,
                None => return, // No GPU / runtime — skip
            };

            // Call cuModuleGetGlobal_v2 with bytes = NULL — the bug case.
            let mut dptr: CUdeviceptr = mem::zeroed();
            let result = crate::cuModuleGetGlobal_v2(
                &mut dptr,
                ptr::null_mut(),
                module,
                c"my_global".as_ptr(),
            );
            assert!(
                result.is_ok(),
                "cuModuleGetGlobal_v2 with bytes=NULL must succeed, got {:?}",
                result
            );
            assert!(
                !dptr.0.is_null(),
                "returned device pointer must not be null"
            );

            // The actual regression check: read the global back via DtoH and
            // verify the value matches what the PTX initialized it to. If
            // hipModuleGetGlobal corrupted the global (the lsh-documented bug),
            // this readback returns 0 or fails outright.
            let mut value: u32 = 0;
            let copy_result =
                crate::cuMemcpyDtoH_v2(&mut value as *mut u32 as *mut std::ffi::c_void, dptr, 4);
            assert!(
                copy_result.is_ok(),
                "cuMemcpyDtoH_v2 from a NULL-bytes-fetched global failed: {:?} \
                 (lsh hipfix item 7 regression)",
                copy_result
            );
            assert_eq!(
                value, 305419896,
                "global value mismatch after NULL-bytes fetch \
                 (lsh hipfix item 7 regression — global was silently corrupted)"
            );
        }
    }

    /// Sanity: when the caller DOES pass a real `bytes` out-pointer, the
    /// wrapper must still write the size into it. This guards against the
    /// obvious mistake of dropping the write-back when adding the local
    /// out-pointer plumbing.
    #[test]
    fn get_global_with_real_bytes_pointer_writes_size() {
        unsafe {
            let module = match try_load_test_module() {
                Some(m) => m,
                None => return,
            };

            let mut dptr: CUdeviceptr = mem::zeroed();
            let mut bytes: usize = 0;
            let result =
                crate::cuModuleGetGlobal_v2(&mut dptr, &mut bytes, module, c"my_global".as_ptr());
            assert!(result.is_ok(), "cuModuleGetGlobal_v2 failed: {:?}", result);
            assert!(!dptr.0.is_null());
            assert_eq!(
                bytes, 4,
                "bytes out-pointer must be written with the global's size (4 for u32)"
            );
        }
    }

    #[test]
    fn detects_radixsort_downsweep_modules() {
        assert!(super::is_radixsort_downsweep_module(
            ".visible .entry _ZN3cub24RadixSortDownsweepKernel(...)\nGridEvenShare"
        ));
        assert!(!super::is_radixsort_downsweep_module(
            ".visible .entry SomeOtherKernel(...)\nGridEvenShare"
        ));
        assert!(!super::is_radixsort_downsweep_module(
            ".visible .entry _ZN3cub24RadixSortDownsweepKernel(...)"
        ));
    }

    #[test]
    fn lds_producer_volatile_address_matched() {
        std::env::set_var("ZLUDA_LDS_PRODUCER_VOLATILE", "1");
        // [%rd2] is read by a volatile load -> its producer store must become
        // volatile (even though it's far from the load). [%rd1] is never read
        // by a volatile load -> stays non-volatile (this is what avoids the
        // actorRange regression from blanket volatilization).
        let ptx = "\
.visible .entry K()
{
st.shared.u32 [%rd2], %r2;
add.s32 %r3, %r3, 1;
add.s32 %r4, %r4, 1;
add.s32 %r5, %r5, 1;
st.shared.u32 [%rd1], %r9;
bar.sync 0;
ld.volatile.shared.u32 %r6, [%rd2];
ld.shared.u32 %r7, [%rd1];
ret;
}
";
        let out = super::patch_lds_producer_volatile_store(ptx).expect("should patch");
        assert!(
            out.contains("st.volatile.shared.u32 [%rd2], %r2;"),
            "producer of volatile-read addr must be volatilized"
        );
        assert!(
            out.contains("st.shared.u32 [%rd1], %r9;"),
            "producer of non-volatile-read addr must stay plain"
        );
        std::env::remove_var("ZLUDA_LDS_PRODUCER_VOLATILE");
    }

    #[test]
    fn lds_producer_volatile_default_off() {
        std::env::remove_var("ZLUDA_LDS_PRODUCER_VOLATILE");
        let ptx = ".visible .entry K()\n{\nst.shared.u32 [%rd1], %r1;\nbar.sync 0;\nret;\n}\n";
        assert!(
            super::patch_lds_producer_volatile_store(ptx).is_none(),
            "must be a no-op when env unset"
        );
    }
}

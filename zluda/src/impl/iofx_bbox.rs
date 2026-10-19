// Fork-only workaround for the APEX IOFX bounding-box kernels (IOFX_bbox*) that
// Batman: Arkham Knight compiles from PTX.
//
// The kernels' cross-warp merge tail is guarded by `setp.eq.s64 %p, %rd, 0;
// @%p bra EXIT;` or `setp.eq.s32 %p, %r, -1; @%p bra EXIT;`, where only thread 0
// is meant to fall through. On gfx1030 other lanes pass the guard with values
// read from uninitialized shared memory and store through wild pointers, which
// faults. Adding a `tid.x != 0` branch to the same target after each guard
// limits the tail to thread 0. Without it Arkham Knight faults before its
// benchmark warmup completes.
use std::borrow::Cow;

pub(crate) fn patch(ptx: Cow<'_, str>) -> Cow<'_, str> {
    match patch_tail_merge(&ptx) {
        Some(patched) => Cow::Owned(patched),
        None => ptx,
    }
}

fn patch_tail_merge(ptx: &str) -> Option<String> {
    if !ptx.contains("IOFX_bbox") {
        return None;
    }
    let lines = ptx.lines().collect::<Vec<_>>();
    let mut out = String::with_capacity(ptx.len() + 1024);
    let mut in_bbox_kernel = false;
    let mut brace_depth = 0i32;
    let mut guards = 0usize;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        out.push_str(line);
        out.push('\n');
        let trimmed = line.trim_start();
        if trimmed.starts_with(".visible .entry IOFX_bbox")
            || trimmed.starts_with(".entry IOFX_bbox")
        {
            in_bbox_kernel = true;
            brace_depth = 0;
        }
        if in_bbox_kernel {
            for c in line.chars() {
                match c {
                    '{' => brace_depth += 1,
                    '}' => {
                        brace_depth -= 1;
                        if brace_depth <= 0 {
                            in_bbox_kernel = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
        if !in_bbox_kernel || brace_depth < 1 {
            i += 1;
            continue;
        }
        // BBx_y: / setp.eq.s64 %p, %rd, 0; / @%p bra TARGET;
        if is_bb_label(trimmed) && i + 2 < lines.len() {
            if let Some(target) = match_guard(lines[i + 1], lines[i + 2], "setp.eq.s64", "%rd", "0")
            {
                out.push_str(lines[i + 1]);
                out.push('\n');
                out.push_str(lines[i + 2]);
                out.push('\n');
                emit_thread0_guard(&mut out, &target.1, guards);
                guards += 1;
                i += 3;
                continue;
            }
        }
        // setp.eq.s32 %p, %r, -1; / @%p bra TARGET; with %r scaled by
        // `mul.wide.u32 %rd, %r, 16` into a store address before the next label
        if i + 1 < lines.len() {
            if let Some((reg, target)) = match_guard(lines[i], lines[i + 1], "setp.eq.s32", "%r", "-1")
            {
                let feeds_address = lines[i + 2..(i + 22).min(lines.len())]
                    .iter()
                    .map(|l| l.trim())
                    .take_while(|l| !is_bb_label(l))
                    .any(|l| is_mul_wide_by_16(l, &reg));
                if !reg.starts_with("%rd") && feeds_address {
                    out.push_str(lines[i + 1]);
                    out.push('\n');
                    emit_thread0_guard(&mut out, &target, guards);
                    guards += 1;
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    (guards > 0).then_some(out)
}

fn emit_thread0_guard(out: &mut String, target: &str, uid: usize) {
    out.push_str(&format!(
        "{{\n.reg .pred %p_zluda_bbox_{uid};\n.reg .s32 %r_zluda_bbox_{uid};\n\
         mov.u32 %r_zluda_bbox_{uid}, %tid.x;\n\
         setp.ne.s32 %p_zluda_bbox_{uid}, %r_zluda_bbox_{uid}, 0;\n\
         @%p_zluda_bbox_{uid} bra {target};\n}}\n"
    ));
}

// Matches `<setp> %pN, <reg>, <value>;` followed by `@%pN bra BBx_y;` and
// returns the compared register and the branch target
fn match_guard(
    setp_line: &str,
    bra_line: &str,
    setp: &str,
    reg_prefix: &str,
    value: &str,
) -> Option<(String, String)> {
    let operands = setp_line.trim().strip_prefix(setp)?.trim_end_matches(';');
    let mut operands = operands.split(',').map(str::trim);
    let pred = operands.next()?.strip_prefix('%')?;
    let reg = operands.next()?;
    if !pred.starts_with('p')
        || !reg.starts_with(reg_prefix)
        || operands.next()? != value
        || operands.next().is_some()
    {
        return None;
    }
    let target = bra_line
        .trim()
        .strip_prefix("@%")?
        .strip_prefix(pred)?
        .trim_start()
        .strip_prefix("bra")?
        .trim()
        .trim_end_matches(';')
        .trim();
    target
        .starts_with("BB")
        .then(|| (reg.to_string(), target.to_string()))
}

fn is_mul_wide_by_16(line: &str, reg: &str) -> bool {
    let Some(operands) = line.strip_prefix("mul.wide.u32") else {
        return false;
    };
    let operands = operands
        .trim_end_matches(';')
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    operands.len() == 3 && operands[1] == reg && operands[2] == "16"
}

fn is_bb_label(line: &str) -> bool {
    let Some(name) = line.trim_end().strip_suffix(':') else {
        return false;
    };
    let Some((block, index)) = name.strip_prefix("BB").and_then(|n| n.split_once('_')) else {
        return false;
    };
    [block, index]
        .iter()
        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::patch_tail_merge;

    const ZERO_GUARD: &str = "\
.visible .entry IOFX_bboxTest(
.param .u64 out
)
{
.reg .pred %p<4>;
.reg .b64 %rd<8>;
BB0_1:
setp.eq.s64 %p1, %rd3, 0;
@%p1 bra BB0_3;
st.global.u64 [%rd3], %rd4;
BB0_3:
ret;
}
";

    const MINUS_ONE_GUARD: &str = "\
.visible .entry IOFX_bboxTest(
.param .u64 out
)
{
.reg .pred %p<4>;
.reg .b32 %r<8>;
.reg .b64 %rd<8>;
setp.eq.s32 %p2, %r5, -1;
@%p2 bra BB0_4;
mul.wide.u32 %rd2, %r5, 16;
st.global.u64 [%rd2], %rd4;
BB0_4:
ret;
}
";

    #[test]
    fn guards_zero_check_after_label() {
        let patched = patch_tail_merge(ZERO_GUARD).unwrap();
        let guard = "@%p1 bra BB0_3;\n{\n.reg .pred %p_zluda_bbox_0;";
        assert!(patched.contains(guard), "{patched}");
        assert!(patched.contains("@%p_zluda_bbox_0 bra BB0_3;"));
    }

    #[test]
    fn guards_minus_one_check_feeding_address() {
        let patched = patch_tail_merge(MINUS_ONE_GUARD).unwrap();
        assert!(patched.contains("@%p2 bra BB0_4;\n{\n"), "{patched}");
        assert!(patched.contains("@%p_zluda_bbox_0 bra BB0_4;"));
    }

    #[test]
    fn skips_minus_one_check_not_feeding_address() {
        let ptx = MINUS_ONE_GUARD.replace("mul.wide.u32 %rd2, %r5, 16;", "mov.u64 %rd2, 0;");
        assert!(patch_tail_merge(&ptx).is_none());
    }

    #[test]
    fn skips_other_kernels() {
        let ptx = ZERO_GUARD.replace("IOFX_bboxTest", "IOFX_otherTest");
        assert!(patch_tail_merge(&ptx).is_none());
        let ptx = format!("// IOFX_bbox\n{}", ZERO_GUARD.replace("IOFX_bboxTest", "other"));
        assert!(patch_tail_merge(&ptx).is_none());
    }
}

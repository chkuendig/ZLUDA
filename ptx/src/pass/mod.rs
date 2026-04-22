use ptx_parser as ast;
use quick_error::quick_error;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::{
    borrow::Cow,
    collections::{hash_map, HashMap},
    iter,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

mod deparamize_functions;
mod expand_operands;
mod fix_special_registers;
mod hoist_globals;
mod insert_explicit_load_store;
mod insert_implicit_conversions;
mod insert_post_saturation;
mod instruction_mode_to_global_mode;
pub mod llvm;
mod normalize_basic_blocks;
mod normalize_identifiers;
mod normalize_predicates;
mod remove_unreachable_basic_blocks;
mod replace_instructions_with_functions;
mod replace_instructions_with_functions_fp_required;
mod replace_known_functions;
mod resolve_function_pointers;

#[cfg(test)]
mod test;

static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
const ZLUDA_PTX_PREFIX: &'static str = "__zluda_ptx_impl_";

quick_error! {
    #[derive(Debug, strum_macros::AsRefStr)]
    pub enum TranslateError {
        UnknownSymbol(symbol: String) {
            display("Unknown symbol: \"{}\"", symbol)
        }
        UntypedSymbol {}
        MismatchedType {}
        Unreachable {}
        Todo(msg: String) {
            display("TODO: {}", msg)
        }
    }
}

/// GPU attributes needed at compile time.
#[derive(Copy, Clone)]
pub struct Attributes {
    /// Clock frequency in kHz.
    pub clock_rate: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SharedWarBarrierOp {
    SharedRead,
    GlobalStore,
    SharedStoreWord,
    SharedStoreOther,
    Barrier,
    Other,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SharedWaveBarrierOp {
    SharedStoreWord,
    SharedRead,
    Barrier,
    Other,
}

fn classify_shared_war_barrier_op(
    stmt: &Statement<ast::Instruction<SpirvWord>, SpirvWord>,
) -> SharedWarBarrierOp {
    match stmt {
        Statement::Instruction(ast::Instruction::Ld { data, .. })
            if matches!(
                data.state_space,
                ast::StateSpace::Shared
                    | ast::StateSpace::SharedCta
                    | ast::StateSpace::SharedCluster
            ) =>
        {
            SharedWarBarrierOp::SharedRead
        }
        Statement::Instruction(ast::Instruction::St { data, .. })
            if matches!(
                data.state_space,
                ast::StateSpace::Global | ast::StateSpace::Generic
            ) =>
        {
            SharedWarBarrierOp::GlobalStore
        }
        Statement::Instruction(ast::Instruction::St { data, .. })
            if matches!(
                data.state_space,
                ast::StateSpace::Shared
                    | ast::StateSpace::SharedCta
                    | ast::StateSpace::SharedCluster
            ) && matches!(
                data.typ,
                ast::Type::Scalar(scalar) if scalar.size_of() == 4
            ) =>
        {
            SharedWarBarrierOp::SharedStoreWord
        }
        Statement::Instruction(ast::Instruction::St { data, .. })
            if matches!(
                data.state_space,
                ast::StateSpace::Shared
                    | ast::StateSpace::SharedCta
                    | ast::StateSpace::SharedCluster
            ) =>
        {
            SharedWarBarrierOp::SharedStoreOther
        }
        Statement::Instruction(ast::Instruction::Bar { .. }) => SharedWarBarrierOp::Barrier,
        _ => SharedWarBarrierOp::Other,
    }
}

fn classify_shared_wave_barrier_op(
    stmt: &Statement<ast::Instruction<SpirvWord>, SpirvWord>,
) -> SharedWaveBarrierOp {
    match stmt {
        Statement::Instruction(ast::Instruction::St { data, .. })
            if matches!(
                data.state_space,
                ast::StateSpace::Shared
                    | ast::StateSpace::SharedCta
                    | ast::StateSpace::SharedCluster
            ) && matches!(
                data.typ,
                ast::Type::Scalar(scalar) if scalar.size_of() == 4
            ) =>
        {
            SharedWaveBarrierOp::SharedStoreWord
        }
        Statement::Instruction(ast::Instruction::Ld { data, .. })
            if matches!(
                data.state_space,
                ast::StateSpace::Shared
                    | ast::StateSpace::SharedCta
                    | ast::StateSpace::SharedCluster
            ) =>
        {
            SharedWaveBarrierOp::SharedRead
        }
        Statement::Instruction(ast::Instruction::Bar { .. } | ast::Instruction::BarWarp { .. }) => {
            SharedWaveBarrierOp::Barrier
        }
        _ => SharedWaveBarrierOp::Other,
    }
}

fn shared_war_barrier_insert_positions(
    ops: impl IntoIterator<Item = SharedWarBarrierOp>,
) -> Vec<usize> {
    let mut insert_positions = Vec::new();
    let mut shared_reads_since_barrier: u32 = 0;
    let mut global_stores_since_shared_read: u32 = 0;
    // To avoid inserting barriers in conditional/divergent tile-loop
    // tails (which can deadlock s_barrier), we require a convincing
    // "bulk" pattern: at least 4 shared reads followed by at least 4
    // global stores before triggering. CUB's Downsweep reads 13 keys
    // from LDS then stores 13 keys to global; casual ld.shared/st.global
    // interleaving in counter-update or tail-check code won't reach this.
    const MIN_SHARED_READS: u32 = 4;
    const MIN_GLOBAL_STORES: u32 = 4;
    for (idx, op) in ops.into_iter().enumerate() {
        match op {
            SharedWarBarrierOp::SharedRead => {
                shared_reads_since_barrier += 1;
            }
            SharedWarBarrierOp::GlobalStore => {
                if shared_reads_since_barrier >= MIN_SHARED_READS {
                    global_stores_since_shared_read += 1;
                }
            }
            SharedWarBarrierOp::SharedStoreWord => {
                if shared_reads_since_barrier >= MIN_SHARED_READS
                    && global_stores_since_shared_read >= MIN_GLOBAL_STORES
                {
                    insert_positions.push(idx);
                }
                shared_reads_since_barrier = 0;
                global_stores_since_shared_read = 0;
            }
            SharedWarBarrierOp::SharedStoreOther | SharedWarBarrierOp::Barrier => {
                shared_reads_since_barrier = 0;
                global_stores_since_shared_read = 0;
            }
            SharedWarBarrierOp::Other => {}
        }
    }
    insert_positions
}

/// Detect "bulk LDS reshuffle" WAR patterns where many ld.shared reads are
/// followed by at least one st.shared write, without an intervening global
/// store — the shape used by PhysX's own radix sort kernels
/// (`IOFX_newRadixSortStepKernel`, `IOFX_radixSortStepKernel`,
/// `radixSortStep_New_MultiStage_3`), which reshuffle histogram/key data
/// entirely within LDS without a global-store phase.
///
/// The existing `shared_war_barrier_insert_positions` classifier requires
/// ≥4 ld.shared followed by ≥4 st.global before the st.shared — that matches
/// CUB's Downsweep (which writes sorted keys to global between LDS phases)
/// but misses kernels whose sort is entirely LDS-local.
///
/// Divergent-branch deadlock risk: same as the existing classifier. We keep
/// the 4-ld.shared threshold; any store (of any size) resets the counter.
/// The pass is opt-in by default (gated by `ZLUDA_PHYSX_WAR_BARRIERS=1`)
/// until we validate it in a game run.
///
/// ============================================================
/// WORKAROUND for an AMDGPU LLVM backend miscompilation. See the big
/// comment block on `insert_shared_war_barriers` below for the LLVM
/// upstream fix location.
/// ============================================================
fn shared_bulk_reshuffle_barrier_insert_positions(
    ops: impl IntoIterator<Item = SharedWarBarrierOp>,
) -> Vec<usize> {
    let mut insert_positions = Vec::new();
    let mut shared_reads_since_barrier: u32 = 0;
    const MIN_SHARED_READS: u32 = 4;
    for (idx, op) in ops.into_iter().enumerate() {
        match op {
            SharedWarBarrierOp::SharedRead => {
                shared_reads_since_barrier += 1;
            }
            SharedWarBarrierOp::SharedStoreWord | SharedWarBarrierOp::SharedStoreOther => {
                if shared_reads_since_barrier >= MIN_SHARED_READS {
                    insert_positions.push(idx);
                }
                shared_reads_since_barrier = 0;
            }
            SharedWarBarrierOp::Barrier => {
                shared_reads_since_barrier = 0;
            }
            SharedWarBarrierOp::GlobalStore | SharedWarBarrierOp::Other => {}
        }
    }
    insert_positions
}

fn shared_wave_barrier_insert_positions(ops: &[SharedWaveBarrierOp]) -> Vec<usize> {
    let mut insert_positions = Vec::new();
    let mut idx = 0;
    while idx < ops.len() {
        if ops[idx] != SharedWaveBarrierOp::SharedStoreWord {
            idx += 1;
            continue;
        }
        let mut cursor = idx + 1;
        let mut shared_reads = 0u32;
        let mut first_read_idx = None;
        while cursor < ops.len() {
            match ops[cursor] {
                SharedWaveBarrierOp::SharedRead => {
                    shared_reads += 1;
                    first_read_idx.get_or_insert(cursor);
                    cursor += 1;
                }
                SharedWaveBarrierOp::Other => {
                    cursor += 1;
                }
                SharedWaveBarrierOp::SharedStoreWord | SharedWaveBarrierOp::Barrier => {
                    break;
                }
            }
        }
        if shared_reads >= 2 {
            insert_positions.push(first_read_idx.unwrap());
            idx = cursor;
        } else {
            idx += 1;
        }
    }
    insert_positions
}

pub fn to_llvm_module<'input>(
    ast: ast::Module<'input>,
    attributes: Attributes,
    mut on_pass_end: impl FnMut(&str),
) -> Result<Module, TranslateError> {
    let sm_version = ast.sm_version;
    let mut flat_resolver = GlobalStringIdentResolver2::<'input>::new(SpirvWord(1));
    let mut scoped_resolver = ScopedResolver::new(&mut flat_resolver);
    let sreg_map = SpecialRegistersMap::new(&mut scoped_resolver)?;
    let directives = normalize_identifiers::run(&mut scoped_resolver, ast.directives)?;
    on_pass_end("normalize_identifiers");
    let directives = replace_known_functions::run(&mut flat_resolver, directives);
    on_pass_end("replace_known_functions");
    let directives = normalize_predicates::run(&mut flat_resolver, directives)?;
    on_pass_end("normalize_predicates");
    let directives = resolve_function_pointers::run(directives)?;
    on_pass_end("resolve_function_pointers");
    let directives = fix_special_registers::run(&mut flat_resolver, &sreg_map, directives)?;
    on_pass_end("fix_special_registers");
    let directives = expand_operands::run(&mut flat_resolver, directives)?;
    on_pass_end("expand_operands");
    let directives = insert_post_saturation::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_post_saturation");
    let directives = deparamize_functions::run(&mut flat_resolver, directives)?;
    on_pass_end("deparamize_functions");
    let directives =
        replace_instructions_with_functions_fp_required::run(&mut flat_resolver, directives)?;
    on_pass_end("replace_instructions_with_functions_fp_required");
    let directives = normalize_basic_blocks::run(&mut flat_resolver, directives)?;
    on_pass_end("normalize_basic_blocks");
    let directives = remove_unreachable_basic_blocks::run(directives)?;
    on_pass_end("remove_unreachable_basic_blocks");
    let directives = instruction_mode_to_global_mode::run(&mut flat_resolver, directives)?;
    on_pass_end("instruction_mode_to_global_mode");
    let directives = insert_explicit_load_store::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_explicit_load_store");
    // Mark Tex instructions that use .texref globals (vs object-based access).
    // Must run before insert_implicit_conversions, which replaces the texref
    // operand with an AddressOf conversion (losing the original type info).
    let mut directives = directives;
    for directive in &mut directives {
        if let Directive2::Method(ref mut method) = directive {
            if let Some(ref mut stmts) = method.body {
                for stmt in stmts.iter_mut() {
                    if let Statement::Instruction(ast::Instruction::Tex {
                        ref mut data,
                        ref arguments,
                    }) = stmt
                    {
                        if let Ok((ty, _)) = flat_resolver.get_typed(arguments.src_texture) {
                            data.is_texref = matches!(ty, ast::Type::Texref);
                        }
                    }
                }
            }
        }
    }
    let directives = insert_implicit_conversions::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_implicit_conversions");
    // Insert workgroup barriers at shared-memory WAR hazard points.
    //
    // ============================================================
    // WORKAROUND for an AMDGPU LLVM backend miscompilation.
    // Upstream fix would live in: llvm/lib/Target/AMDGPU/
    //   (most likely SIInsertWaitcnts / SIMemoryLegalizer, or the
    //   LDS alias-analysis that feeds scheduling under +precise-memory).
    // Revisit this pass if we land a proper LLVM fix — at that point
    // the inserted bar.sync becomes redundant and should be removed
    // to avoid penalizing well-formed kernels.
    // Related tree artifacts:
    //   - memory/project_war_barrier_fix_verified.md
    //   - ext/llvm-project branch zluda-physx-pr145902
    // ============================================================
    //
    // CUDA code (CUB RadixSort Downsweep, PhysX FleX) often reuses LDS across
    // key/value phases without an explicit bar.sync, relying on NVIDIA's
    // implicit warp-synchronous execution to keep all warps roughly in step.
    // On AMDGPU, waves within a workgroup can diverge under concurrent GPU
    // load: a fast wave may reach the value-write-to-LDS phase while a slow
    // wave is still reading keys from overlapping LDS addresses. This WAR
    // hazard corrupts the slow wave's key reads with value data.
    //
    // The pattern we detect:
    //   ld.shared → ... → st.global → ... → st.shared  (no bar.sync between)
    // We insert bar.sync 0 before the st.shared to synchronize all waves.
    let mut directives = directives;
    for directive in &mut directives {
        if let Directive2::Method(ref mut method) = directive {
            if let Some(ref mut stmts) = method.body {
                let insert_positions = shared_war_barrier_insert_positions(
                    stmts.iter().map(classify_shared_war_barrier_op),
                );
                // Insert bar.sync 0 at marked positions (iterate in reverse so
                // indices stay valid).
                for &pos in insert_positions.iter().rev() {
                    let barrier_id = flat_resolver.register_unnamed(Some((
                        ast::Type::Scalar(ast::ScalarType::U32),
                        ast::StateSpace::Reg,
                    )));
                    stmts.insert(
                        pos,
                        Statement::Instruction(ast::Instruction::Bar {
                            data: ast::BarData { aligned: false },
                            arguments: ast::BarArgs {
                                src1: barrier_id,
                                src2: None,
                            },
                        }),
                    );
                    stmts.insert(
                        pos,
                        Statement::Constant(ConstantDefinition {
                            dst: barrier_id,
                            typ: ast::ScalarType::U32,
                            value: ast::ImmediateValue::U64(0),
                        }),
                    );
                }
            }
        }
    }
    on_pass_end("insert_shared_war_barriers");
    // Optional second WAR classifier for PhysX sort kernels that reshuffle
    // histogram/key data entirely within LDS (no global-store phase between
    // ld.shared and st.shared). Opt-in because we haven't yet live-tested
    // that the added barriers don't trigger divergent-path deadlocks on
    // every affected kernel.
    //
    // See also: project_war_barrier_fix_verified.md memory note, and the
    // 2026-04-22 post-bbox-fix crash investigation (prime suspect:
    // IOFX_newRadixSortStepKernel).
    let enable_physx_war = std::env::var("ZLUDA_PHYSX_WAR_BARRIERS")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if enable_physx_war {
        for directive in &mut directives {
            if let Directive2::Method(ref mut method) = directive {
                if let Some(ref mut stmts) = method.body {
                    let insert_positions = shared_bulk_reshuffle_barrier_insert_positions(
                        stmts.iter().map(classify_shared_war_barrier_op),
                    );
                    for &pos in insert_positions.iter().rev() {
                        let barrier_id = flat_resolver.register_unnamed(Some((
                            ast::Type::Scalar(ast::ScalarType::U32),
                            ast::StateSpace::Reg,
                        )));
                        stmts.insert(
                            pos,
                            Statement::Instruction(ast::Instruction::Bar {
                                data: ast::BarData { aligned: false },
                                arguments: ast::BarArgs {
                                    src1: barrier_id,
                                    src2: None,
                                },
                            }),
                        );
                        stmts.insert(
                            pos,
                            Statement::Constant(ConstantDefinition {
                                dst: barrier_id,
                                typ: ast::ScalarType::U32,
                                value: ast::ImmediateValue::U64(0),
                            }),
                        );
                    }
                }
            }
        }
        on_pass_end("insert_shared_bulk_reshuffle_barriers");
    }
    for directive in &mut directives {
        if let Directive2::Method(ref mut method) = directive {
            if let Some(ref mut stmts) = method.body {
                let ops: Vec<_> = stmts
                    .iter()
                    .map(classify_shared_wave_barrier_op)
                    .collect();
                let insert_positions = shared_wave_barrier_insert_positions(&ops);
                for &pos in insert_positions.iter().rev() {
                    let barrier_mask_id = flat_resolver.register_unnamed(Some((
                        ast::Type::Scalar(ast::ScalarType::U32),
                        ast::StateSpace::Reg,
                    )));
                    stmts.insert(
                        pos,
                        Statement::Instruction(ast::Instruction::BarWarp {
                            data: (),
                            arguments: ast::BarWarpArgs {
                                src: barrier_mask_id,
                            },
                        }),
                    );
                    stmts.insert(
                        pos,
                        Statement::Constant(ConstantDefinition {
                            dst: barrier_mask_id,
                            typ: ast::ScalarType::U32,
                            value: ast::ImmediateValue::U64(u32::MAX as u64),
                        }),
                    );
                }
            }
        }
    }
    on_pass_end("insert_shared_wave_barriers");
    let directives = replace_instructions_with_functions::run(&mut flat_resolver, directives)?;
    on_pass_end("replace_instructions_with_functions");
    let directives = hoist_globals::run(directives)?;
    on_pass_end("hoist_globals");
    let context = llvm_zluda::utils::Context::new();
    let llvm_ir = llvm::emit::run(&context, flat_resolver, directives)?;
    let attributes_ir = llvm::attributes::run(&context, attributes)?;
    on_pass_end("emit_llvm");
    Ok(Module {
        llvm_ir,
        attributes_ir,
        kernel_info: HashMap::new(),
        context,
        metadata: kernel_metadata::KernelMetadataV1::new(sm_version),
    })
}

#[cfg(test)]
mod shared_war_barrier_tests {
    use super::{
        shared_bulk_reshuffle_barrier_insert_positions, shared_war_barrier_insert_positions,
        shared_wave_barrier_insert_positions, SharedWarBarrierOp, SharedWaveBarrierOp,
    };

    #[test]
    fn inserts_multiple_barriers_for_bulk_word_shared_writes() {
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::SharedStoreWord,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::SharedStoreWord,
        ];
        assert_eq!(shared_war_barrier_insert_positions(ops), vec![8, 17]);
    }

    #[test]
    fn skips_non_word_shared_stores_and_resets_at_barriers() {
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::SharedStoreOther,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::Barrier,
            SharedWarBarrierOp::SharedStoreWord,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::SharedStoreWord,
        ];
        assert_eq!(shared_war_barrier_insert_positions(ops), vec![27]);
    }

    #[test]
    fn inserts_wave_barrier_before_cross_lane_shared_reads() {
        let ops = vec![
            SharedWaveBarrierOp::Other,
            SharedWaveBarrierOp::SharedStoreWord,
            SharedWaveBarrierOp::Other,
            SharedWaveBarrierOp::SharedRead,
            SharedWaveBarrierOp::Other,
            SharedWaveBarrierOp::SharedRead,
            SharedWaveBarrierOp::Other,
        ];
        assert_eq!(shared_wave_barrier_insert_positions(&ops), vec![3]);
    }

    #[test]
    fn skips_wave_barrier_without_multiple_shared_reads() {
        let ops = vec![
            SharedWaveBarrierOp::SharedStoreWord,
            SharedWaveBarrierOp::Other,
            SharedWaveBarrierOp::SharedRead,
            SharedWaveBarrierOp::Other,
            SharedWaveBarrierOp::Barrier,
            SharedWaveBarrierOp::SharedStoreWord,
            SharedWaveBarrierOp::SharedRead,
        ];
        assert!(shared_wave_barrier_insert_positions(&ops).is_empty());
    }

    // --- shared_bulk_reshuffle_barrier_insert_positions tests --------------
    // The new "PhysX sort" classifier fires on ≥4 ld.shared followed by any
    // st.shared, with no intervening bar.sync. No st.global required. Same
    // divergence-risk profile as the existing WAR classifier (4-read threshold).

    #[test]
    fn bulk_reshuffle_inserts_barrier_before_shared_store_after_four_reads() {
        // 4 reads, then a word store → barrier at position 4.
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedStoreWord,
        ];
        assert_eq!(
            shared_bulk_reshuffle_barrier_insert_positions(ops),
            vec![4]
        );
    }

    #[test]
    fn bulk_reshuffle_skips_below_threshold() {
        // Only 3 reads → no barrier.
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedStoreWord,
        ];
        assert!(shared_bulk_reshuffle_barrier_insert_positions(ops).is_empty());
    }

    #[test]
    fn bulk_reshuffle_resets_after_barrier() {
        // 3 reads then bar.sync then word store → no barrier (reset).
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::Barrier,
            SharedWarBarrierOp::SharedStoreWord,
        ];
        assert!(shared_bulk_reshuffle_barrier_insert_positions(ops).is_empty());
    }

    #[test]
    fn bulk_reshuffle_ignores_global_stores_between_reads_and_shared_store() {
        // The new classifier doesn't care about global stores in the middle —
        // what matters is that ≥4 shared reads precede the shared store.
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::GlobalStore,
            SharedWarBarrierOp::SharedStoreWord,
        ];
        assert_eq!(
            shared_bulk_reshuffle_barrier_insert_positions(ops),
            vec![5]
        );
    }

    #[test]
    fn bulk_reshuffle_inserts_before_shared_store_other_too() {
        // Vector / non-word shared stores (SharedStoreOther) also trigger the
        // barrier, because the WAR hazard is about LDS addresses, not width.
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedStoreOther,
        ];
        assert_eq!(
            shared_bulk_reshuffle_barrier_insert_positions(ops),
            vec![4]
        );
    }

    #[test]
    fn bulk_reshuffle_inserts_multiple_barriers_across_a_sort_pass() {
        // Two separate ≥4-read → store phases in one function body, no
        // bar.sync between them. Both get a barrier. The second phase starts
        // fresh because the previous store resets the counter.
        let ops = vec![
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedStoreWord, // → barrier at 4
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedRead,
            SharedWarBarrierOp::SharedStoreWord, // → barrier at 9
        ];
        assert_eq!(
            shared_bulk_reshuffle_barrier_insert_positions(ops),
            vec![4, 9]
        );
    }
}

pub struct Module {
    pub llvm_ir: llvm_zluda::utils::Module,
    pub attributes_ir: llvm_zluda::utils::Module,
    pub kernel_info: HashMap<String, KernelInfo>,
    pub context: llvm_zluda::utils::Context,
    pub metadata: kernel_metadata::KernelMetadataV1,
}

impl Module {
    pub fn linked_bitcode(&self) -> &'static [u8] {
        ZLUDA_PTX_IMPL
    }
}

pub struct KernelInfo {
    pub arguments_sizes: Vec<(usize, bool)>,
    pub uses_shared_mem: bool,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone, EnumIter)]
enum PtxSpecialRegister {
    Tid,
    Ntid,
    Ctaid,
    Nctaid,
    Clock,
    LanemaskLe,
    LanemaskLt,
    LanemaskGe,
    Laneid,
}

impl PtxSpecialRegister {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tid => "%tid",
            Self::Ntid => "%ntid",
            Self::Ctaid => "%ctaid",
            Self::Nctaid => "%nctaid",
            Self::Clock => "%clock",
            Self::LanemaskLe => "%lanemask_le",
            Self::LanemaskLt => "%lanemask_lt",
            Self::LanemaskGe => "%lanemask_ge",
            Self::Laneid => "%laneid",
        }
    }

    fn get_type(self) -> ast::Type {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid => ast::Type::Vector(4, self.get_function_return_type()),
            _ => ast::Type::Scalar(self.get_function_return_type()),
        }
    }

    fn get_function_return_type(self) -> ast::ScalarType {
        match self {
            PtxSpecialRegister::Tid => ast::ScalarType::U32,
            PtxSpecialRegister::Ntid => ast::ScalarType::U32,
            PtxSpecialRegister::Ctaid => ast::ScalarType::U32,
            PtxSpecialRegister::Nctaid => ast::ScalarType::U32,
            PtxSpecialRegister::Clock => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskLe => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskLt => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskGe => ast::ScalarType::U32,
            PtxSpecialRegister::Laneid => ast::ScalarType::U32,
        }
    }

    fn get_function_input_type(self) -> Option<ast::ScalarType> {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid => Some(ast::ScalarType::U8),
            PtxSpecialRegister::Clock
            | PtxSpecialRegister::LanemaskLe
            | PtxSpecialRegister::LanemaskLt
            | PtxSpecialRegister::LanemaskGe
            | PtxSpecialRegister::Laneid => None,
        }
    }

    fn get_unprefixed_function_name(self) -> &'static str {
        match self {
            PtxSpecialRegister::Tid => "sreg_tid",
            PtxSpecialRegister::Ntid => "sreg_ntid",
            PtxSpecialRegister::Ctaid => "sreg_ctaid",
            PtxSpecialRegister::Nctaid => "sreg_nctaid",
            PtxSpecialRegister::Clock => "sreg_clock",
            PtxSpecialRegister::LanemaskLt => "sreg_lanemask_lt",
            PtxSpecialRegister::LanemaskLe => "sreg_lanemask_le",
            PtxSpecialRegister::LanemaskGe => "sreg_lanemask_ge",
            PtxSpecialRegister::Laneid => "sreg_laneid",
        }
    }
}

#[cfg(debug_assertions)]
fn error_unreachable() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_unreachable() -> TranslateError {
    TranslateError::Unreachable
}

#[cfg(debug_assertions)]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    unreachable!("{}", msg.into())
}

#[cfg(not(debug_assertions))]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    TranslateError::Todo(msg.into())
}

#[cfg(debug_assertions)]
fn error_todo() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_todo() -> TranslateError {
    TranslateError::Todo("".to_string())
}

#[cfg(debug_assertions)]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    panic!("Unknown symbol: \"{}\"", symbol.into())
}

#[cfg(not(debug_assertions))]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    TranslateError::UnknownSymbol(symbol.into())
}

#[cfg(debug_assertions)]
fn error_mismatched_type() -> TranslateError {
    panic!("Mismatched type")
}

#[cfg(not(debug_assertions))]
fn error_mismatched_type() -> TranslateError {
    TranslateError::MismatchedType
}

enum Statement<I, P: ast::Operand> {
    Label(SpirvWord),
    Variable(ast::Variable<P::Ident>),
    Instruction(I),
    // SPIR-V compatible replacement for PTX predicates
    Conditional(BranchCondition),
    Conversion(ImplicitConversion),
    Constant(ConstantDefinition),
    RetValue(ast::RetData, Vec<(SpirvWord, ast::Type)>),
    PtrAccess(PtrAccess<P>),
    RepackVector(RepackVectorDetails),
    FunctionPointer(FunctionPointerDetails),
    VectorRead(VectorRead),
    VectorWrite(VectorWrite),
    SetMode(ModeRegister),
    // This instruction is a nop, it serves as a marker to indicate that the
    // next instruction requires certain floating-point modes to be set.
    // Some transcendentals compile to a sequence of instructions that
    // require certain modes to be set _mid-function_.
    // See replace_instructions_with_functions_fp_required pass for details
    FpModeRequired {
        ftz_f32: Option<bool>,
        rnd_f32: Option<ast::RoundingMode>,
    },
    FpSaturate {
        dst: SpirvWord,
        src: SpirvWord,
        type_: ast::ScalarType,
    },
}

#[derive(Eq, PartialEq, Clone, Copy)]
#[cfg_attr(test, derive(Debug))]
enum ModeRegister {
    Denormal {
        f32: bool,
        f16f64: bool,
    },
    Rounding {
        f32: ast::RoundingMode,
        f16f64: ast::RoundingMode,
    },
}

impl std::fmt::Display for ModeRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModeRegister::Denormal { f32, f16f64 } => write!(
                f,
                "zluda.set_mode.denormal.f32::{}.f16f64::{}",
                if *f32 { "on" } else { "off" },
                if *f16f64 { "on" } else { "off" },
            ),
            ModeRegister::Rounding { f32, f16f64 } => {
                write!(f, "zluda.set_mode.rounding.f32{}.f16f64{}", f32, f16f64)
            }
        }
    }
}

impl<T: ast::Operand<Ident = SpirvWord>> Statement<ast::Instruction<T>, T> {
    fn visit_map<To: ast::Operand<Ident = SpirvWord>, Err>(
        self,
        visitor: &mut impl ast::VisitorMap<T, To, Err>,
    ) -> std::result::Result<Statement<ast::Instruction<To>, To>, Err> {
        Ok(match self {
            Statement::Instruction(i) => {
                return ast::visit_map(i, visitor).map(Statement::Instruction)
            }
            Statement::Label(label) => {
                Statement::Label(visitor.visit_ident(label, None, false, false)?)
            }
            Statement::Variable(var) => {
                let name = visitor.visit_ident(
                    var.name,
                    Some((&var.info.v_type, var.info.state_space)),
                    true,
                    false,
                )?;
                Statement::Variable(ast::Variable {
                    info: ast::VariableInfo {
                        align: var.info.align,
                        v_type: var.info.v_type,
                        state_space: var.info.state_space,
                        array_init: var.info.array_init,
                    },
                    name,
                })
            }
            Statement::Conditional(conditional) => {
                let predicate = visitor.visit_ident(
                    conditional.predicate,
                    Some((&ast::ScalarType::Pred.into(), ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                let if_true = visitor.visit_ident(conditional.if_true, None, false, false)?;
                let if_false = visitor.visit_ident(conditional.if_false, None, false, false)?;
                Statement::Conditional(BranchCondition {
                    predicate,
                    if_true,
                    if_false,
                })
            }
            Statement::Conversion(ImplicitConversion {
                src,
                dst,
                from_type,
                to_type,
                from_space,
                to_space,
                kind,
            }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&to_type, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    src,
                    Some((&from_type, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::Conversion(ImplicitConversion {
                    src,
                    dst,
                    from_type,
                    to_type,
                    from_space,
                    to_space,
                    kind,
                })
            }
            Statement::Constant(ConstantDefinition { dst, typ, value }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&typ.into(), ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                Statement::Constant(ConstantDefinition { dst, typ, value })
            }
            Statement::RetValue(data, value) => {
                let value = value
                    .into_iter()
                    .map(|(ident, type_)| {
                        Ok((
                            visitor.visit_ident(
                                ident,
                                Some((&type_, ast::StateSpace::Local)),
                                false,
                                false,
                            )?,
                            type_,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Statement::RetValue(data, value)
            }
            Statement::PtrAccess(PtrAccess {
                underlying_type,
                state_space,
                dst,
                ptr_src,
                offset_src,
            }) => {
                let dst =
                    visitor.visit_ident(dst, Some((&underlying_type, state_space)), true, false)?;
                let ptr_src = visitor.visit_ident(
                    ptr_src,
                    Some((&underlying_type, state_space)),
                    false,
                    false,
                )?;
                let offset_src = visitor.visit(
                    offset_src,
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::S64),
                        ast::StateSpace::Reg,
                    )),
                    false,
                    false,
                )?;
                Statement::PtrAccess(PtrAccess {
                    underlying_type,
                    state_space,
                    dst,
                    ptr_src,
                    offset_src,
                })
            }
            Statement::VectorRead(VectorRead {
                scalar_type,
                vector_width,
                scalar_dst: dst,
                vector_src,
                member,
            }) => {
                let scalar_t = scalar_type.into();
                let vector_t = ast::Type::Vector(vector_width, scalar_type);
                let dst: SpirvWord = visitor.visit_ident(
                    dst,
                    Some((&scalar_t, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    vector_src,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::VectorRead(VectorRead {
                    scalar_type,
                    vector_width,
                    scalar_dst: dst,
                    vector_src: src,
                    member,
                })
            }
            Statement::VectorWrite(VectorWrite {
                scalar_type,
                vector_width,
                vector_dst,
                vector_src,
                scalar_src,
                member,
            }) => {
                let scalar_t = scalar_type.into();
                let vector_t = ast::Type::Vector(vector_width, scalar_type);
                let vector_dst = visitor.visit_ident(
                    vector_dst,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let vector_src = visitor.visit_ident(
                    vector_src,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                let scalar_src = visitor.visit_ident(
                    scalar_src,
                    Some((&scalar_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::VectorWrite(VectorWrite {
                    vector_dst,
                    vector_src,
                    scalar_src,
                    scalar_type,
                    vector_width,
                    member,
                })
            }
            Statement::RepackVector(RepackVectorDetails {
                is_extract,
                typ,
                packed,
                unpacked,
                relaxed_type_check,
            }) => {
                let (packed, unpacked) = if is_extract {
                    let unpacked = unpacked
                        .into_iter()
                        .map(|ident| {
                            visitor.visit_ident(
                                ident,
                                Some((&typ.into(), ast::StateSpace::Reg)),
                                true,
                                relaxed_type_check,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let packed = visitor.visit_ident(
                        packed,
                        Some((
                            &ast::Type::Vector(unpacked.len() as u8, typ),
                            ast::StateSpace::Reg,
                        )),
                        false,
                        false,
                    )?;
                    (packed, unpacked)
                } else {
                    let packed = visitor.visit_ident(
                        packed,
                        Some((
                            &ast::Type::Vector(unpacked.len() as u8, typ),
                            ast::StateSpace::Reg,
                        )),
                        true,
                        false,
                    )?;
                    let unpacked = unpacked
                        .into_iter()
                        .map(|ident| {
                            visitor.visit_ident(
                                ident,
                                Some((&typ.into(), ast::StateSpace::Reg)),
                                false,
                                relaxed_type_check,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (packed, unpacked)
                };
                Statement::RepackVector(RepackVectorDetails {
                    is_extract,
                    typ,
                    packed,
                    unpacked,
                    relaxed_type_check,
                })
            }
            Statement::FunctionPointer(FunctionPointerDetails { dst, src }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::U64),
                        ast::StateSpace::Reg,
                    )),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(src, None, false, false)?;
                Statement::FunctionPointer(FunctionPointerDetails { dst, src })
            }
            Statement::SetMode(mode_register) => Statement::SetMode(mode_register),
            Statement::FpSaturate { dst, src, type_ } => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&type_.into(), ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    src,
                    Some((&type_.into(), ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::FpSaturate { dst, src, type_ }
            }
            Statement::FpModeRequired { ftz_f32, rnd_f32 } => {
                Statement::FpModeRequired { ftz_f32, rnd_f32 }
            }
        })
    }
}

struct BranchCondition {
    predicate: SpirvWord,
    if_true: SpirvWord,
    if_false: SpirvWord,
}

impl std::fmt::Display for BranchCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zluda.branch")
    }
}

#[derive(Clone)]
struct ImplicitConversion {
    src: SpirvWord,
    dst: SpirvWord,
    from_type: ast::Type,
    to_type: ast::Type,
    from_space: ast::StateSpace,
    to_space: ast::StateSpace,
    kind: ConversionKind,
}

impl std::fmt::Display for ImplicitConversion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "zluda.convert_implicit{}{}{}{}{}",
            self.kind, self.to_space, self.to_type, self.from_space, self.from_type
        )
    }
}

#[derive(PartialEq, Clone, strum_macros::Display)]
#[strum(serialize_all = "snake_case", prefix = ".")]
enum ConversionKind {
    Default,
    // zero-extend/chop/bitcast depending on types
    SignExtend,
    BitToPtr,
    PtrToPtr,
    AddressOf,
}

struct ConstantDefinition {
    pub dst: SpirvWord,
    pub typ: ast::ScalarType,
    pub value: ast::ImmediateValue,
}

impl std::fmt::Display for ConstantDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zluda.constant{} {}", self.typ, self.value)
    }
}

pub struct PtrAccess<T> {
    underlying_type: ast::Type,
    state_space: ast::StateSpace,
    dst: SpirvWord,
    ptr_src: SpirvWord,
    offset_src: T,
}

struct RepackVectorDetails {
    is_extract: bool,
    typ: ast::ScalarType,
    packed: SpirvWord,
    unpacked: Vec<SpirvWord>,
    relaxed_type_check: bool,
}

impl std::fmt::Display for RepackVectorDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let extract = if self.is_extract {
            ".extract"
        } else {
            ".composite"
        };
        let relaxed = if self.relaxed_type_check {
            ".relaxed"
        } else {
            ""
        };
        write!(f, "zluda.repack_vector{}{}{}", extract, relaxed, self.typ)
    }
}

struct FunctionPointerDetails {
    dst: SpirvWord,
    src: SpirvWord,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SpirvWord(u32);

impl std::fmt::Display for SpirvWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl From<u32> for SpirvWord {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
impl From<SpirvWord> for u32 {
    fn from(value: SpirvWord) -> Self {
        value.0
    }
}

impl ast::Operand for SpirvWord {
    type Ident = Self;

    fn from_ident(ident: Self::Ident) -> Self {
        ident
    }
}

type ExpandedStatement = Statement<ast::Instruction<SpirvWord>, SpirvWord>;

type NormalizedStatement = Statement<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

enum Directive2<Instruction, Operand: ast::Operand> {
    Variable(ast::LinkingDirective, ast::Variable<SpirvWord>),
    Method(Function2<Instruction, Operand>),
}

struct KernelAttributes {
    flush_to_zero_f32: bool,
    flush_to_zero_f16f64: bool,
    rounding_mode_f32: ast::RoundingMode,
    rounding_mode_f16f64: ast::RoundingMode,
}

struct Function2<Instruction, Operand: ast::Operand> {
    pub return_arguments: Vec<ast::Variable<Operand::Ident>>,
    pub name: Operand::Ident,
    pub input_arguments: Vec<ast::Variable<Operand::Ident>>,
    pub body: Option<Vec<Statement<Instruction, Operand>>>,
    kernel_attributes: Option<KernelAttributes>,
    import_as: Option<String>,
    tuning: Vec<ast::TuningDirective>,
    linkage: ast::LinkingDirective,
}

impl<I, O: ast::Operand> Function2<I, O> {
    fn is_kernel(&self) -> bool {
        self.kernel_attributes.is_some()
    }
}

type NormalizedDirective2 = Directive2<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type NormalizedFunction2 = Function2<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type UnconditionalDirective =
    Directive2<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

type UnconditionalFunction =
    Function2<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

struct GlobalStringIdentResolver2<'input> {
    pub(crate) current_id: SpirvWord,
    pub(crate) ident_map: FxHashMap<SpirvWord, IdentEntry<'input>>,
}

impl<'input> GlobalStringIdentResolver2<'input> {
    fn new(spirv_word: SpirvWord) -> Self {
        Self {
            current_id: spirv_word,
            ident_map: FxHashMap::default(),
        }
    }

    fn register_named(
        &mut self,
        name: Cow<'input, str>,
        type_space: Option<(ast::Type, ast::StateSpace)>,
    ) -> SpirvWord {
        let new_id = self.current_id;
        self.ident_map.insert(
            new_id,
            IdentEntry {
                name: Some(name),
                type_space,
            },
        );
        self.current_id.0 += 1;
        new_id
    }

    fn register_unnamed(&mut self, type_space: Option<(ast::Type, ast::StateSpace)>) -> SpirvWord {
        let new_id = self.current_id;
        self.ident_map.insert(
            new_id,
            IdentEntry {
                name: None,
                type_space,
            },
        );
        self.current_id.0 += 1;
        new_id
    }

    fn get_typed(&self, id: SpirvWord) -> Result<&(ast::Type, ast::StateSpace), TranslateError> {
        match self.ident_map.get(&id) {
            Some(IdentEntry {
                type_space: Some(type_space),
                ..
            }) => Ok(type_space),
            _ => Err(error_unknown_symbol(format!("{:?}", id))),
        }
    }
}

struct IdentEntry<'input> {
    name: Option<Cow<'input, str>>,
    type_space: Option<(ast::Type, ast::StateSpace)>,
}

struct ScopedResolver<'input, 'b> {
    flat_resolver: &'b mut GlobalStringIdentResolver2<'input>,
    scopes: Vec<ScopeMarker<'input>>,
}

impl<'input, 'b> ScopedResolver<'input, 'b> {
    fn new(flat_resolver: &'b mut GlobalStringIdentResolver2<'input>) -> Self {
        Self {
            flat_resolver,
            scopes: vec![ScopeMarker::new()],
        }
    }

    fn start_scope(&mut self) {
        self.scopes.push(ScopeMarker::new());
    }

    fn end_scope(&mut self) {
        let scope = self.scopes.pop().unwrap();
        scope.flush(self.flat_resolver);
    }

    fn add_or_get_in_current_scope_untyped(
        &mut self,
        name: &'input str,
    ) -> Result<SpirvWord, TranslateError> {
        let current_scope = self.scopes.last_mut().unwrap();
        Ok(
            match current_scope.name_to_ident.entry(Cow::Borrowed(name)) {
                hash_map::Entry::Occupied(occupied_entry) => {
                    let ident = *occupied_entry.get();
                    let entry = current_scope
                        .ident_map
                        .get(&ident)
                        .ok_or_else(|| error_unreachable())?;
                    if entry.type_space.is_some() {
                        return Err(error_unknown_symbol(name));
                    }
                    ident
                }
                hash_map::Entry::Vacant(vacant_entry) => {
                    let new_id = self.flat_resolver.current_id;
                    self.flat_resolver.current_id.0 += 1;
                    vacant_entry.insert(new_id);
                    current_scope.ident_map.insert(
                        new_id,
                        IdentEntry {
                            name: Some(Cow::Borrowed(name)),
                            type_space: None,
                        },
                    );
                    new_id
                }
            },
        )
    }

    fn add(
        &mut self,
        name: Cow<'input, str>,
        type_space: Option<(ast::Type, ast::StateSpace)>,
    ) -> Result<SpirvWord, TranslateError> {
        let result = self.flat_resolver.current_id;
        self.flat_resolver.current_id.0 += 1;
        let current_scope = self.scopes.last_mut().unwrap();
        if current_scope
            .name_to_ident
            .insert(name.clone(), result)
            .is_some()
        {
            return Err(error_unknown_symbol(name));
        }
        current_scope.ident_map.insert(
            result,
            IdentEntry {
                name: Some(name),
                type_space,
            },
        );
        Ok(result)
    }

    fn get(&mut self, name: &str) -> Result<SpirvWord, TranslateError> {
        self.scopes
            .iter()
            .rev()
            .find_map(|resolver| resolver.name_to_ident.get(name).copied())
            .ok_or_else(|| error_unknown_symbol(name))
    }

    fn get_in_current_scope(&self, label: &'input str) -> Result<SpirvWord, TranslateError> {
        let current_scope = self.scopes.last().unwrap();
        current_scope
            .name_to_ident
            .get(label)
            .copied()
            .ok_or_else(|| error_unreachable())
    }
}

struct ScopeMarker<'input> {
    ident_map: FxHashMap<SpirvWord, IdentEntry<'input>>,
    name_to_ident: FxHashMap<Cow<'input, str>, SpirvWord>,
}

impl<'input> ScopeMarker<'input> {
    fn new() -> Self {
        Self {
            ident_map: FxHashMap::default(),
            name_to_ident: FxHashMap::default(),
        }
    }

    fn flush(self, resolver: &mut GlobalStringIdentResolver2<'input>) {
        resolver.ident_map.extend(self.ident_map);
    }
}

struct SpecialRegistersMap {
    reg_to_id: FxHashMap<PtxSpecialRegister, SpirvWord>,
    id_to_reg: FxHashMap<SpirvWord, PtxSpecialRegister>,
}

impl SpecialRegistersMap {
    fn new(resolver: &mut ScopedResolver) -> Result<Self, TranslateError> {
        let mut result = SpecialRegistersMap {
            reg_to_id: FxHashMap::default(),
            id_to_reg: FxHashMap::default(),
        };
        for sreg in PtxSpecialRegister::iter() {
            let text = sreg.as_str();
            let id = resolver.add(
                Cow::Borrowed(text),
                Some((sreg.get_type(), ast::StateSpace::Reg)),
            )?;
            result.reg_to_id.insert(sreg, id);
            result.id_to_reg.insert(id, sreg);
        }
        Ok(result)
    }

    fn get(&self, id: SpirvWord) -> Option<PtxSpecialRegister> {
        self.id_to_reg.get(&id).copied()
    }

    fn len() -> usize {
        PtxSpecialRegister::iter().len()
    }

    fn foreach_declaration<'a, 'input>(
        resolver: &'a mut GlobalStringIdentResolver2<'input>,
        mut fn_: impl FnMut(
            PtxSpecialRegister,
            (
                Vec<ast::Variable<SpirvWord>>,
                SpirvWord,
                Vec<ast::Variable<SpirvWord>>,
            ),
        ),
    ) {
        for sreg in PtxSpecialRegister::iter() {
            let external_fn_name = [ZLUDA_PTX_PREFIX, sreg.get_unprefixed_function_name()].concat();
            let name = resolver.register_named(Cow::Owned(external_fn_name), None);
            let return_type = sreg.get_function_return_type();
            let input_type = sreg.get_function_input_type();
            let return_arguments = vec![ast::Variable {
                info: ast::VariableInfo {
                    align: None,
                    v_type: return_type.into(),
                    state_space: ast::StateSpace::Reg,
                    array_init: Vec::new(),
                },
                name: resolver.register_unnamed(Some((return_type.into(), ast::StateSpace::Reg))),
            }];
            let input_arguments = input_type
                .into_iter()
                .map(|type_| ast::Variable {
                    info: ast::VariableInfo {
                        align: None,
                        v_type: type_.into(),
                        state_space: ast::StateSpace::Reg,
                        array_init: Vec::new(),
                    },
                    name: resolver.register_unnamed(Some((type_.into(), ast::StateSpace::Reg))),
                })
                .collect::<Vec<_>>();
            fn_(sreg, (return_arguments, name, input_arguments));
        }
    }
}

pub struct VectorRead {
    scalar_type: ast::ScalarType,
    vector_width: u8,
    scalar_dst: SpirvWord,
    vector_src: SpirvWord,
    member: u8,
}

pub struct VectorWrite {
    scalar_type: ast::ScalarType,
    vector_width: u8,
    vector_dst: SpirvWord,
    vector_src: SpirvWord,
    scalar_src: SpirvWord,
    member: u8,
}

fn scalar_to_ptx_name(this: ast::ScalarType) -> &'static str {
    match this {
        ast::ScalarType::B8 => "b8",
        ast::ScalarType::B16 => "b16",
        ast::ScalarType::B32 => "b32",
        ast::ScalarType::B64 => "b64",
        ast::ScalarType::B128 => "b128",
        ast::ScalarType::U8 => "u8",
        ast::ScalarType::U16 => "u16",
        ast::ScalarType::U16x2 => "u16x2",
        ast::ScalarType::U32 => "u32",
        ast::ScalarType::U64 => "u64",
        ast::ScalarType::S8 => "s8",
        ast::ScalarType::S16 => "s16",
        ast::ScalarType::S16x2 => "s16x2",
        ast::ScalarType::S32 => "s32",
        ast::ScalarType::S64 => "s64",
        ast::ScalarType::F16 => "f16",
        ast::ScalarType::F16x2 => "f16x2",
        ast::ScalarType::F32 => "f32",
        ast::ScalarType::F64 => "f64",
        ast::ScalarType::BF16 => "bf16",
        ast::ScalarType::BF16x2 => "bf16x2",
        ast::ScalarType::Pred => "pred",
        ast::ScalarType::E4m3x2 => "e4m3x2",
        ast::ScalarType::E5m2x2 => "e5m2x2",
    }
}

type UnconditionalStatement =
    Statement<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

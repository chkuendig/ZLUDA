// This pass modifies 32 bit module to run on 64 bit implementation
// It extracts each global variable (except `.shared`) and then goes through
// every global and:
// * If it's a global variable it gets erased
// * If it's a kernel:
//   * We injects new arguments
//      * 64 bit pointer to the global memory
//      * For each global variable, a 32 bit pseudo-pointer
// A later pass will transform all loads from 32 bit pseudo-pointers into loads
// from the 64 bit pointer with an offset

use crate::{
    pass::{
        error_todo, error_unreachable, Directive2, Function, Function32,
        GlobalStringIdentResolver2, SpirvWord, Statement,
    },
    TranslateError,
};
use kernel_metadata::{Global32Bit, ModuleMetadata32Bit};
use ptx_parser::{self as ast, LdDetails, RegOrImmediate, Variable};
use rustc_hash::FxHashMap;

pub(super) fn run<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<
    (
        Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
        ModuleMetadata32Bit,
    ),
    TranslateError,
> {
    let (global_ids, globals) = collect_globals(resolver, &directives)?;
    let mut explicit_args_size_align = Vec::new();
    let directives = directives
        .into_iter()
        .filter_map(|directive| {
            run_directive(
                resolver,
                directive,
                &global_ids,
                &mut explicit_args_size_align,
            )
            .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        directives,
        ModuleMetadata32Bit {
            globals,
            explicit_args_size_align,
        },
    ))
}

pub(super) fn collect_globals<'a, 'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    directives: &[Directive2<ast::Instruction<SpirvWord>, SpirvWord>],
) -> Result<(Vec<SpirvWord>, Vec<Global32Bit>), TranslateError> {
    let (ids, globals): (Vec<SpirvWord>, Vec<Global32Bit>) = directives
        .iter()
        .filter_map(|directive| match directive {
            Directive2::Variable(_, var) if pass_through_variable(var) => None,
            Directive2::Variable(_, var)
                if matches!(
                    var.info.state_space,
                    ast::StateSpace::Const | ast::StateSpace::Global
                ) =>
            {
                Some(get_global_details(resolver, var))
            }
            Directive2::Method(..) => None,
            _ => Some(Err(error_todo())),
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    Ok((ids, globals))
}

fn pass_through_variable(variable: &Variable<SpirvWord>) -> bool {
    matches!(variable.info.state_space, ast::StateSpace::Shared)
        || matches!(variable.info.v_type, ast::Type::Texref | ast::Type::Surfref)
}

fn get_global_details<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    variable: &Variable<SpirvWord>,
) -> Result<(SpirvWord, Global32Bit), TranslateError> {
    let (scalar, count) = match &variable.info.v_type {
        ast::Type::Scalar(scalar) => (*scalar, 1usize),
        ast::Type::Array(None, scalar, dimensions) => {
            let count = dimensions.iter().try_fold(1usize, |n, &dim| {
                n.checked_mul(dim as usize).ok_or_else(error_todo)
            })?;
            (*scalar, count)
        }
        _ => return Err(error_todo()),
    };
    let size = scalar.size_of() as usize;
    let bytes = count.checked_mul(size).ok_or_else(error_todo)?;
    if bytes == 0 || variable.info.array_init.len() > count {
        return Err(error_todo());
    }
    let entry = resolver
        .ident_map
        .get(&variable.name)
        .ok_or_else(error_unreachable)?;
    let name = entry
        .name
        .as_ref()
        .ok_or_else(error_unreachable)?
        .to_string();
    let align = variable
        .info
        .align
        .unwrap_or(scalar.layout().align() as u32);
    let mut initializer = vec![0; bytes];
    for (chunk, value) in initializer
        .chunks_exact_mut(size)
        .zip(&variable.info.array_init)
    {
        let value = match value {
            RegOrImmediate::Imm(value) => value,
            RegOrImmediate::Reg(_) => return Err(error_todo()),
        };
        let bits = match value {
            ast::ImmediateValue::U64(n) => {
                if scalar.kind() == ast::ScalarKind::Float && *n != 0 {
                    return Err(error_todo());
                }
                *n as u128
            }
            ast::ImmediateValue::S64(n) => {
                if scalar.kind() == ast::ScalarKind::Float && *n != 0 {
                    return Err(error_todo());
                }
                *n as i128 as u128
            }
            ast::ImmediateValue::F32(n) if scalar == ast::ScalarType::F32 => n.to_bits() as u128,
            ast::ImmediateValue::F32(n) => float_initializer_bits(scalar, *n as f64)?,
            ast::ImmediateValue::F64(n) => float_initializer_bits(scalar, *n)?,
        };
        chunk.copy_from_slice(&bits.to_le_bytes()[..size]);
    }
    Ok((
        variable.name,
        Global32Bit {
            name,
            align,
            initializer,
        },
    ))
}

fn float_initializer_bits(scalar: ast::ScalarType, value: f64) -> Result<u128, TranslateError> {
    Ok(match scalar {
        ast::ScalarType::F16 => half::f16::from_f64(value).to_bits() as u128,
        ast::ScalarType::BF16 => half::bf16::from_f64(value).to_bits() as u128,
        ast::ScalarType::F32 => (value as f32).to_bits() as u128,
        ast::ScalarType::F64 => value.to_bits() as u128,
        _ if value == 0.0 => 0,
        _ => return Err(error_todo()),
    })
}

fn run_directive<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    directive: Directive2<ast::Instruction<SpirvWord>, SpirvWord>,
    globals_ids: &[SpirvWord],
    explicit_args_size_align: &mut Vec<(String, Vec<(u32, u32)>)>,
) -> Result<Option<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>, TranslateError> {
    Ok(match directive {
        Directive2::Variable(linking, varinfo) if pass_through_variable(&varinfo) => {
            Some(Directive2::Variable(linking, varinfo))
        }
        Directive2::Method(method) => Some(Directive2::Method(run_method(
            resolver,
            method,
            globals_ids,
            explicit_args_size_align,
        )?)),
        _ => None,
    })
}

fn run_method<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    mut method: Function<ast::Instruction<SpirvWord>, SpirvWord>,
    globals_ids: &[SpirvWord],
    explicit_args_size_align: &mut Vec<(String, Vec<(u32, u32)>)>,
) -> Result<Function<ast::Instruction<SpirvWord>, SpirvWord>, TranslateError> {
    let is_kernel = method.is_kernel();
    if let Some(ref mut body) = method.body {
        if !is_kernel {
            return Err(error_todo());
        }
        let mut result = Vec::with_capacity(body.len());
        let mut old_body = std::mem::take(body).into_iter().peekable();
        while let Some(Statement::Label(_)) = old_body.peek() {
            result.push(old_body.next().unwrap());
        }
        let text_name = method
            .import_as
            .as_deref()
            .or_else(|| {
                resolver
                    .ident_map
                    .get(&method.name)?
                    .name
                    .as_ref()
                    .map(|x| x.as_ref())
            })
            .ok_or_else(error_unreachable)?
            .to_string();
        let argument_sizes = method
            .input_arguments
            .iter()
            .map(|arg| {
                let layout = arg.info.v_type.layout().unwrap();
                let size = layout.size() as u32;
                let align = layout.align() as u32;
                (size, align)
            })
            .collect();
        explicit_args_size_align.push((text_name, argument_sizes));
        let implicit_memory_ptr = add_hidden_argument(
            resolver,
            ast::ScalarType::B64,
            &mut method.input_arguments,
            &mut result,
        );
        let globals_remapping = globals_ids
            .iter()
            .map(|&id| {
                (
                    id,
                    add_hidden_argument(
                        resolver,
                        ast::ScalarType::B32,
                        &mut method.input_arguments,
                        &mut result,
                    ),
                )
            })
            .collect::<FxHashMap<_, _>>();
        for statement in old_body {
            run_statement(&mut result, &globals_remapping, statement)?;
        }
        *body = result;
        method.kernel_meta32 = Some(Function32 {
            implicit_memory_ptr,
        });
    }
    Ok(method)
}

fn run_statement<'input>(
    result: &mut Vec<Statement<ptx_parser::Instruction<SpirvWord>, SpirvWord>>,
    globals_remapping: &FxHashMap<SpirvWord, SpirvWord>,
    statement: Statement<ptx_parser::Instruction<SpirvWord>, SpirvWord>,
) -> Result<(), TranslateError> {
    let statement =
        statement.visit_map(&mut |mut arg: SpirvWord,
                                   _type_space: Option<(&ast::Type, ast::StateSpace)>,
                                   _is_dst: bool,
                                   _relaxed_type_check: bool| {
            if let Some(new_arg) = globals_remapping.get(&arg) {
                arg = *new_arg;
            }
            Ok(arg)
        })?;
    result.push(statement);
    Ok(())
}

fn add_hidden_argument<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    type_: ast::ScalarType,
    input_arguments: &mut Vec<ast::Variable<SpirvWord>>,
    body: &mut Vec<Statement<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> SpirvWord {
    let implicit_argument_param = resolver.register_unnamed(Some((
        ast::Type::Scalar(type_),
        ast::StateSpace::ParamEntry,
    )));
    input_arguments.push(ast::Variable {
        name: implicit_argument_param,
        info: ast::VariableInfo {
            align: None,
            v_type: ast::Type::Scalar(type_),
            state_space: ast::StateSpace::ParamEntry,
            array_init: Vec::new(),
        },
    });
    let implicit_argument =
        resolver.register_unnamed(Some((ast::Type::Scalar(type_), ast::StateSpace::Reg)));
    body.push(Statement::Instruction(ast::Instruction::Ld {
        data: LdDetails {
            qualifier: ast::LdStQualifier::Weak,
            state_space: ast::StateSpace::ParamEntry,
            caching: ast::LdCacheOperator::Cached,
            typ: ast::Type::Scalar(type_),
            non_coherent: false,
        },
        arguments: ast::LdArgs {
            dst: implicit_argument,
            src: implicit_argument_param,
        },
    }));
    implicit_argument
}

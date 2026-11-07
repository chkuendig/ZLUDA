use super::*;

pub(super) fn run<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    mut directives: Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>, TranslateError> {
    for directive in &mut directives {
        let Directive2::Method(method) = directive else {
            continue;
        };
        let Some(body) = method.body.take() else {
            continue;
        };
        let mut constants = FxHashMap::default();
        let mut output = Vec::with_capacity(body.len());
        for statement in body {
            match statement {
                Statement::Constant(constant) => {
                    if constant.typ == ast::ScalarType::F64 {
                        let ast::ImmediateValue::F64(value) = constant.value else {
                            return Err(TranslateError::MismatchedType);
                        };
                        constants.insert(constant.dst, value);
                    }
                    output.push(Statement::Constant(constant));
                }
                Statement::Instruction(mut instruction) => {
                    instruction = match instruction {
                        ast::Instruction::Ld {
                            data,
                            mut arguments,
                        } if contains_f64(&data.typ) => {
                            if data.typ != ast::Type::Scalar(ast::ScalarType::F64) {
                                return Err(TranslateError::Todo(
                                    "unsupported map_f64_to_f32 instruction".into(),
                                ));
                            }
                            let wide = temporary(resolver, ast::ScalarType::F64);
                            let single = temporary(resolver, ast::ScalarType::F32);
                            let dst = arguments.dst;
                            arguments.dst = wide;
                            output.push(Statement::Instruction(ast::Instruction::Ld {
                                data,
                                arguments,
                            }));
                            output.push(convert(
                                single,
                                wide,
                                ast::ScalarType::F64,
                                ast::ScalarType::F32,
                            ));
                            output.push(storage_conversion(resolver, dst, single, true)?);
                            continue;
                        }
                        ast::Instruction::St {
                            data,
                            mut arguments,
                        } if contains_f64(&data.typ) => {
                            if data.typ != ast::Type::Scalar(ast::ScalarType::F64) {
                                return Err(TranslateError::Todo(
                                    "unsupported map_f64_to_f32 instruction".into(),
                                ));
                            }
                            let single = temporary(resolver, ast::ScalarType::F32);
                            let wide = temporary(resolver, ast::ScalarType::F64);
                            if let Some(value) = constants.get(&arguments.src2) {
                                output.push(Statement::Constant(ConstantDefinition {
                                    dst: single,
                                    typ: ast::ScalarType::F32,
                                    value: ast::ImmediateValue::F32(*value as f32),
                                }));
                            } else {
                                output.push(storage_conversion(
                                    resolver,
                                    arguments.src2,
                                    single,
                                    false,
                                )?);
                            }
                            output.push(convert(
                                wide,
                                single,
                                ast::ScalarType::F32,
                                ast::ScalarType::F64,
                            ));
                            arguments.src2 = wide;
                            output.push(Statement::Instruction(ast::Instruction::St {
                                data,
                                arguments,
                            }));
                            continue;
                        }
                        // An f64 atomic must not become a 32-bit access to 64-bit storage.
                        ast::Instruction::Atom { ref data, .. } if contains_f64(&data.type_) => {
                            return Err(TranslateError::Todo(
                                "unsupported map_f64_to_f32 instruction".into(),
                            ))
                        }
                        instruction => instruction,
                    };
                    if matches!(&instruction, ast::Instruction::Mov { data, .. }
                        if contains_f64(&data.typ) && data.typ != ast::Type::Scalar(ast::ScalarType::F64))
                    {
                        return Err(TranslateError::Todo(
                            "unsupported map_f64_to_f32 instruction".into(),
                        ));
                    }
                    let mapped = map_instruction(&mut instruction);
                    let instruction = ast::visit_map(
                        instruction,
                        &mut |operand: SpirvWord,
                              type_space: Option<(&ast::Type, ast::StateSpace)>,
                              _: bool,
                              _: bool| {
                            if type_space.is_some_and(|(typ, _)| contains_f64(typ)) {
                                return Err(TranslateError::Todo(
                                    "unsupported map_f64_to_f32 instruction".into(),
                                ));
                            }
                            Ok(operand)
                        },
                    )?;
                    if !mapped {
                        output.push(Statement::Instruction(instruction));
                        continue;
                    }
                    let mut after = Vec::new();
                    let instruction = ast::visit_map(
                        instruction,
                        &mut |operand: SpirvWord,
                              type_space: Option<(&ast::Type, ast::StateSpace)>,
                              is_dst: bool,
                              _: bool| {
                            let Some((
                                ast::Type::Scalar(ast::ScalarType::F32),
                                ast::StateSpace::Reg,
                            )) = type_space
                            else {
                                return Ok::<_, TranslateError>(operand);
                            };
                            let (storage, space) = resolver.get_typed(operand)?;
                            let ast::Type::Scalar(type_) = storage else {
                                return Ok(operand);
                            };
                            if type_.size_of() != 8 || *space != ast::StateSpace::Reg {
                                return Ok(operand);
                            }
                            let storage = storage.clone();
                            let single = ast::Type::Scalar(ast::ScalarType::F32);
                            let temporary = resolver
                                .register_unnamed(Some((single.clone(), ast::StateSpace::Reg)));
                            if !is_dst {
                                if let Some(value) = constants.get(&operand) {
                                    output.push(Statement::Constant(ConstantDefinition {
                                        dst: temporary,
                                        typ: ast::ScalarType::F32,
                                        value: ast::ImmediateValue::F32(*value as f32),
                                    }));
                                    return Ok(temporary);
                                }
                            }
                            // Keep the low float word in its original 64-bit storage.
                            // Default conversions bitcast/chop/extend; they do not fptrunc/fpext.
                            let conversion = Statement::Conversion(ImplicitConversion {
                                src: if is_dst { temporary } else { operand },
                                dst: if is_dst { operand } else { temporary },
                                from_type: if is_dst {
                                    single.clone()
                                } else {
                                    storage.clone()
                                },
                                to_type: if is_dst { storage } else { single },
                                from_space: ast::StateSpace::Reg,
                                to_space: ast::StateSpace::Reg,
                                kind: ConversionKind::Default,
                            });
                            if is_dst {
                                after.push(conversion);
                            } else {
                                output.push(conversion);
                            }
                            Ok(temporary)
                        },
                    )?;
                    output.push(Statement::Instruction(instruction));
                    output.append(&mut after);
                }
                statement => output.push(statement),
            }
        }
        method.body = Some(output);
    }
    Ok(directives)
}

fn temporary(resolver: &mut GlobalStringIdentResolver2, scalar: ast::ScalarType) -> SpirvWord {
    resolver.register_unnamed(Some((ast::Type::Scalar(scalar), ast::StateSpace::Reg)))
}

fn contains_f64(typ: &ast::Type) -> bool {
    matches!(
        typ,
        ast::Type::Scalar(ast::ScalarType::F64)
            | ast::Type::Vector(_, ast::ScalarType::F64)
            | ast::Type::Array(_, ast::ScalarType::F64, _)
    )
}

fn storage_conversion(
    resolver: &GlobalStringIdentResolver2,
    storage: SpirvWord,
    single: SpirvWord,
    is_dst: bool,
) -> Result<Statement<ast::Instruction<SpirvWord>, SpirvWord>, TranslateError> {
    let (typ, space) = resolver.get_typed(storage)?;
    if *space != ast::StateSpace::Reg {
        return Err(error_mismatched_type());
    }
    let f32 = ast::Type::Scalar(ast::ScalarType::F32);
    Ok(Statement::Conversion(ImplicitConversion {
        src: if is_dst { single } else { storage },
        dst: if is_dst { storage } else { single },
        from_type: if is_dst { f32.clone() } else { typ.clone() },
        to_type: if is_dst { typ.clone() } else { f32 },
        from_space: ast::StateSpace::Reg,
        to_space: ast::StateSpace::Reg,
        kind: ConversionKind::Default,
    }))
}

fn convert(
    dst: SpirvWord,
    src: SpirvWord,
    from: ast::ScalarType,
    to: ast::ScalarType,
) -> Statement<ast::Instruction<SpirvWord>, SpirvWord> {
    let mode = if to == ast::ScalarType::F64 {
        ast::CvtMode::FPExtend {
            flush_to_zero: None,
            saturate: false,
        }
    } else {
        ast::CvtMode::FPTruncate {
            rounding: ast::RoundingMode::NearestEven,
            is_integer_rounding: false,
            flush_to_zero: None,
            saturate: false,
            relu: false,
        }
    };
    Statement::Instruction(ast::Instruction::Cvt {
        data: ast::CvtDetails { from, to, mode },
        arguments: ast::CvtArgs {
            dst,
            src,
            src2: None,
        },
    })
}

use ast::{ScalarType, Type};

fn scalar(type_: &mut ScalarType) -> bool {
    if *type_ != ScalarType::F64 {
        return false;
    }
    *type_ = ScalarType::F32;
    true
}

fn ty(type_: &mut Type) -> bool {
    match type_ {
        Type::Scalar(type_) | Type::Vector(_, type_) | Type::Array(_, type_, _) => scalar(type_),
        _ => false,
    }
}

// PTX .target map_f64_to_f32 changes instructions, not storage layouts.
fn map_instruction<T: ast::Operand>(instruction: &mut ast::Instruction<T>) -> bool {
    use ast::*;
    match instruction {
        Instruction::Abs { data, .. }
        | Instruction::Neg { data, .. }
        | Instruction::Rsqrt { data, .. } => scalar(&mut data.type_),
        Instruction::Add {
            data: ArithDetails::Float(data),
            ..
        }
        | Instruction::Sub {
            data: ArithDetails::Float(data),
            ..
        }
        | Instruction::Mul {
            data: MulDetails::Float(data),
            ..
        }
        | Instruction::Mad {
            data: MadDetails::Float(data),
            ..
        }
        | Instruction::Fma { data, .. } => scalar(&mut data.type_),
        Instruction::Min {
            data: MinMaxDetails::Float(data),
            ..
        }
        | Instruction::Max {
            data: MinMaxDetails::Float(data),
            ..
        } => scalar(&mut data.type_),
        Instruction::Div {
            data: DivDetails::Float(data),
            ..
        } => scalar(&mut data.type_),
        Instruction::Rcp { data, .. } | Instruction::Sqrt { data, .. } => scalar(&mut data.type_),
        Instruction::Mov { data, .. } => ty(&mut data.typ),
        Instruction::Ld { data, .. } => ty(&mut data.typ),
        Instruction::St { data, .. } => ty(&mut data.typ),
        Instruction::Atom { data, .. } => ty(&mut data.type_),
        Instruction::Selp { data, .. } | Instruction::Slct { data, .. } => scalar(data),
        Instruction::Set { data, .. } => scalar(&mut data.dtype) | scalar(&mut data.base.type_),
        Instruction::SetBool { data, .. } => {
            scalar(&mut data.dtype) | scalar(&mut data.base.base.type_)
        }
        Instruction::Setp { data, .. } => scalar(&mut data.type_),
        Instruction::SetpBool { data, .. } => scalar(&mut data.base.type_),
        Instruction::Cvt { data, .. } => {
            let changed = scalar(&mut data.from) | scalar(&mut data.to);
            if changed && data.from == data.to {
                data.mode = match data.mode {
                    CvtMode::FPExtend {
                        flush_to_zero,
                        saturate,
                    } => CvtMode::FPRound {
                        integer_rounding: None,
                        flush_to_zero,
                        saturate,
                    },
                    CvtMode::FPTruncate {
                        rounding,
                        is_integer_rounding,
                        flush_to_zero,
                        saturate,
                        ..
                    } => CvtMode::FPRound {
                        integer_rounding: is_integer_rounding.then_some(rounding),
                        flush_to_zero,
                        saturate,
                    },
                    mode => mode,
                };
            }
            changed
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_cvt_uses_single_precision_rounding_modes() {
        let text = ".version 1.4\n.target sm_10, map_f64_to_f32\n.entry k() { .reg .f64 a; .reg .f32 b; .reg .s32 i; cvt.f64.f32 a,b; cvt.rn.f32.f64 b,a; cvt.rzi.s32.f64 i,a; ret; }";
        let mut module = ast::parse_module_checked(text).unwrap();
        let ast::Directive::Method(_, method) = &mut module.directives[0] else {
            panic!()
        };
        let mut changed = 0;
        for statement in method.body.as_mut().unwrap() {
            if let ast::Statement::Instruction(_, instruction) = statement {
                if !map_instruction(instruction) {
                    continue;
                }
                let ast::Instruction::Cvt { data, .. } = instruction else {
                    panic!()
                };
                assert!(data.from == ast::ScalarType::F32);
                if data.to == ast::ScalarType::F32 {
                    assert!(matches!(data.mode, ast::CvtMode::FPRound { .. }));
                } else {
                    assert!(matches!(data.mode, ast::CvtMode::SignedFromFP { .. }));
                }
                changed += 1;
            }
        }
        assert_eq!(changed, 3);
    }

    #[test]
    fn unsupported_f64_instructions_are_rejected() {
        for instruction in [
            "copysign.f64 a,a,b;",
            "mov.v2.f64 {a,b},{a,b};",
            "mov.f64 a,1;",
        ] {
            let text = format!(".version 2.3\n.target sm_10, map_f64_to_f32\n.entry k() {{ .reg .f64 a,b; .reg .pred p; {instruction} ret; }}");
            let module = ast::parse_module_checked(&text).unwrap();
            assert!(
                super::super::to_llvm_module(
                    module,
                    super::super::Attributes {
                        clock_rate: 1000000
                    },
                    |_| {}
                )
                .is_err(),
                "{instruction}"
            );
        }
    }
}

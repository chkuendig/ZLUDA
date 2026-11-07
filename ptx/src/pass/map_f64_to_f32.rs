use super::*;

pub(super) fn run<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    mut directives: Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>, TranslateError> {
    for directive in &mut directives {
        let Directive2::Method(method) = directive else { continue; };
        let Some(body) = method.body.take() else { continue; };
        let mut constants = FxHashMap::default();
        let mut output = Vec::with_capacity(body.len());
        for statement in body {
            match statement {
                Statement::Constant(constant) => {
                    if let (ast::ScalarType::F64, ast::ImmediateValue::F64(value)) = (&constant.typ, &constant.value) {
                        constants.insert(constant.dst, *value);
                    }
                    output.push(Statement::Constant(constant));
                }
                Statement::Instruction(mut instruction) => {
                    instruction = match instruction {
                        ast::Instruction::Ld { data, mut arguments } if contains_f64(&data.typ) => {
                            if data.typ != ast::Type::Scalar(ast::ScalarType::F64) { return Err(error_todo()); }
                            let wide = temporary(resolver, ast::ScalarType::F64);
                            let single = temporary(resolver, ast::ScalarType::F32);
                            let dst = arguments.dst;
                            arguments.dst = wide;
                            output.push(Statement::Instruction(ast::Instruction::Ld { data, arguments }));
                            output.push(convert(single, wide, ast::ScalarType::F64, ast::ScalarType::F32));
                            output.push(storage_conversion(resolver, dst, single, true)?);
                            continue;
                        }
                        ast::Instruction::St { data, mut arguments } if contains_f64(&data.typ) => {
                            if data.typ != ast::Type::Scalar(ast::ScalarType::F64) { return Err(error_todo()); }
                            let single = temporary(resolver, ast::ScalarType::F32);
                            let wide = temporary(resolver, ast::ScalarType::F64);
                            if let Some(value) = constants.get(&arguments.src2) {
                                output.push(Statement::Constant(ConstantDefinition {
                                    dst: single, typ: ast::ScalarType::F32,
                                    value: ast::ImmediateValue::F32(*value as f32),
                                }));
                            } else {
                                output.push(storage_conversion(resolver, arguments.src2, single, false)?);
                            }
                            output.push(convert(wide, single, ast::ScalarType::F32, ast::ScalarType::F64));
                            arguments.src2 = wide;
                            output.push(Statement::Instruction(ast::Instruction::St { data, arguments }));
                            continue;
                        }
                        // An f64 atomic must not become a 32-bit access to 64-bit storage.
                        ast::Instruction::Atom { ref data, .. } if contains_f64(&data.type_) => return Err(error_todo()),
                        instruction => instruction,
                    };
                    if !instruction.map_f64_to_f32() {
                        output.push(Statement::Instruction(instruction));
                        continue;
                    }
                    let mut after = Vec::new();
                    let instruction = ast::visit_map(instruction, &mut |operand: SpirvWord,
                        type_space: Option<(&ast::Type, ast::StateSpace)>, is_dst: bool, _: bool| {
                        let Some((ast::Type::Scalar(ast::ScalarType::F32), ast::StateSpace::Reg)) = type_space else {
                            return Ok::<_, TranslateError>(operand);
                        };
                        let (storage, space) = resolver.get_typed(operand)?;
                        let ast::Type::Scalar(type_) = storage else { return Ok(operand); };
                        if type_.size_of() != 8 || *space != ast::StateSpace::Reg { return Ok(operand); }
                        let storage = storage.clone();
                        let single = ast::Type::Scalar(ast::ScalarType::F32);
                        let temporary = resolver.register_unnamed(Some((single.clone(), ast::StateSpace::Reg)));
                        if !is_dst {
                            if let Some(value) = constants.get(&operand) {
                                output.push(Statement::Constant(ConstantDefinition {
                                    dst: temporary, typ: ast::ScalarType::F32,
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
                            from_type: if is_dst { single.clone() } else { storage.clone() },
                            to_type: if is_dst { storage } else { single },
                            from_space: ast::StateSpace::Reg, to_space: ast::StateSpace::Reg,
                            kind: ConversionKind::Default,
                        });
                        if is_dst { after.push(conversion); } else { output.push(conversion); }
                        Ok(temporary)
                    })?;
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
    matches!(typ, ast::Type::Scalar(ast::ScalarType::F64) |
        ast::Type::Vector(_, ast::ScalarType::F64) | ast::Type::Array(_, ast::ScalarType::F64, _))
}

fn storage_conversion(
    resolver: &GlobalStringIdentResolver2,
    storage: SpirvWord, single: SpirvWord, is_dst: bool,
) -> Result<Statement<ast::Instruction<SpirvWord>, SpirvWord>, TranslateError> {
    let (typ, space) = resolver.get_typed(storage)?;
    if *space != ast::StateSpace::Reg { return Err(error_mismatched_type()); }
    let f32 = ast::Type::Scalar(ast::ScalarType::F32);
    Ok(Statement::Conversion(ImplicitConversion {
        src: if is_dst { single } else { storage }, dst: if is_dst { storage } else { single },
        from_type: if is_dst { f32.clone() } else { typ.clone() },
        to_type: if is_dst { typ.clone() } else { f32 },
        from_space: ast::StateSpace::Reg, to_space: ast::StateSpace::Reg,
        kind: ConversionKind::Default,
    }))
}

fn convert(dst: SpirvWord, src: SpirvWord, from: ast::ScalarType, to: ast::ScalarType)
    -> Statement<ast::Instruction<SpirvWord>, SpirvWord> {
    let mode = if to == ast::ScalarType::F64 {
        ast::CvtMode::FPExtend { flush_to_zero: None, saturate: false }
    } else {
        ast::CvtMode::FPTruncate { rounding: ast::RoundingMode::NearestEven,
            is_integer_rounding: false, flush_to_zero: None, saturate: false, relu: false }
    };
    Statement::Instruction(ast::Instruction::Cvt { data: ast::CvtDetails { from, to, mode },
        arguments: ast::CvtArgs { dst, src, src2: None } })
}

#[cfg(test)]
mod tests {
    #[test]
    fn map_f64_to_f32_keeps_storage_and_emits_float_arithmetic() {
        let source = include_str!("../../../zluda/src/impl/test_ptx/legacy_sm10.ptx");
        let ast = ptx_parser::parse_module_checked(source).unwrap();
        let module = super::super::to_llvm_module(ast, super::super::Attributes { clock_rate: 1000000 }, |_| {}).unwrap();
        let text = module.llvm_ir.print_module_to_string().to_str().into_owned();
        assert!(text.contains("fadd float"), "{text}");
        assert!(!text.contains("fadd double"), "{text}");
        assert!(text.contains("fptrunc double") && text.contains("fpext float"), "{text}");
        assert!(text.contains("[2 x double]"), "{text}");
        assert!(text.contains("trunc i64") && text.contains("zext i32"), "{text}");
    }

    #[test]
    fn map_f64_to_f32_slct_keeps_signed_selector_and_selected_bits() {
        for typ in ["b16","b32","b64","u16","u32","u64","s16","s32","s64","f32","f64"] {
            let source = format!(".version 1.4\n.target sm_10, map_f64_to_f32\n.entry k(.param .{typ} x, .param .{typ} y, .param .s32 z, .param .u64 out) {{ .reg .{typ} a,b,d; .reg .s32 c; .reg .u64 p; ld.param.{typ} a,[x]; ld.param.{typ} b,[y]; ld.param.s32 c,[z]; ld.param.u64 p,[out]; slct.{typ}.s32 d,a,b,c; st.global.{typ} [p],d; ret; }}");
            let ast = ptx_parser::parse_module_checked(&source).unwrap();
            let module = super::super::to_llvm_module(ast, super::super::Attributes { clock_rate: 1000000 }, |_| {}).unwrap();
            let ir = module.llvm_ir.print_module_to_string().to_str().into_owned();
            assert!(ir.contains("icmp sge i32"), "{typ}: {ir}");
            assert!(ir.contains("select i1"), "{typ}: {ir}");
        }
    }

}

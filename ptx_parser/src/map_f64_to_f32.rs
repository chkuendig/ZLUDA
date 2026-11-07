use crate::*;

fn scalar(type_: &mut ScalarType) -> bool {
    if *type_ != ScalarType::F64 { return false; }
    *type_ = ScalarType::F32;
    true
}

fn ty(type_: &mut Type) -> bool {
    match type_ {
        Type::Scalar(type_) | Type::Vector(_, type_) | Type::Array(_, type_, _) => scalar(type_),
        _ => false,
    }
}

impl<T: Operand> Instruction<T> {
    // PTX .target map_f64_to_f32 changes instructions, not storage layouts.
    pub fn map_f64_to_f32(&mut self) -> bool {
        match self {
            Instruction::Abs { data, .. } | Instruction::Neg { data, .. } |
            Instruction::Rsqrt { data, .. } => scalar(&mut data.type_),
            Instruction::Add { data: ArithDetails::Float(data), .. } |
            Instruction::Sub { data: ArithDetails::Float(data), .. } |
            Instruction::Mul { data: MulDetails::Float(data), .. } |
            Instruction::Mad { data: MadDetails::Float(data), .. } |
            Instruction::Fma { data, .. } => scalar(&mut data.type_),
            Instruction::Min { data: MinMaxDetails::Float(data), .. } |
            Instruction::Max { data: MinMaxDetails::Float(data), .. } => scalar(&mut data.type_),
            Instruction::Div { data: DivDetails::Float(data), .. } => scalar(&mut data.type_),
            Instruction::Rcp { data, .. } | Instruction::Sqrt { data, .. } => scalar(&mut data.type_),
            Instruction::Mov { data, .. } => ty(&mut data.typ),
            Instruction::Ld { data, .. } => ty(&mut data.typ),
            Instruction::St { data, .. } => ty(&mut data.typ),
            Instruction::Atom { data, .. } => ty(&mut data.type_),
            Instruction::Selp { data, .. } => scalar(data),
            Instruction::Set { data, .. } => scalar(&mut data.dtype) | scalar(&mut data.base.type_),
            Instruction::SetBool { data, .. } => scalar(&mut data.dtype) | scalar(&mut data.base.base.type_),
            Instruction::Setp { data, .. } => scalar(&mut data.type_),
            Instruction::SetpBool { data, .. } => scalar(&mut data.base.type_),
            Instruction::Cvt { data, .. } => {
                let changed = scalar(&mut data.from) | scalar(&mut data.to);
                if changed && data.from == data.to {
                    data.mode = match data.mode {
                        CvtMode::FPExtend { flush_to_zero, saturate } => CvtMode::FPRound {
                            integer_rounding: None, flush_to_zero, saturate,
                        },
                        CvtMode::FPTruncate { rounding, is_integer_rounding, flush_to_zero, saturate, .. } => CvtMode::FPRound {
                            integer_rounding: is_integer_rounding.then_some(rounding), flush_to_zero, saturate,
                        },
                        mode => mode,
                    };
                }
                changed
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_maps_instructions_and_preserves_eight_byte_storage() {
        let text = ".version 1.4\n.target sm_10, map_f64_to_f32\n.global .f64 g[2];\n.entry k() { .reg .f64 a; mov.f64 a, 0d3ff0000000000000; add.f64 a, a, a; ret; }";
        let mut module = parse_module_checked(text).unwrap();
        assert!(module.map_f64_to_f32);
        assert_eq!(module.address_size, 32);
        let Directive::Variable(_, global) = &module.directives[0] else { panic!() };
        assert_eq!(global.info.v_type.layout().unwrap().size(), 16);
        let Directive::Method(_, method) = &mut module.directives[1] else { panic!() };
        let body = method.body.as_mut().unwrap();
        let Statement::Variable(MultiVariable::Names { info, .. }) = &body[0] else { panic!() };
        assert_eq!(info.v_type.layout().unwrap().size(), 8);
        let mut changed = 0;
        for statement in body {
            if let Statement::Instruction(_, instruction) = statement {
                changed += usize::from(instruction.map_f64_to_f32());
                assert!(!instruction.map_f64_to_f32());
            }
        }
        assert_eq!(changed, 2);
        assert!(!parse_module_checked(".version 1.4\n.target sm_10\n").unwrap().map_f64_to_f32);
        assert!(parse_module_checked(".version 1.4\n.target sm_10, unknown\n").is_err());
    }

    #[test]
    fn mapped_cvt_uses_single_precision_rounding_modes() {
        let text = ".version 1.4\n.target sm_10, map_f64_to_f32\n.entry k() { .reg .f64 a; .reg .f32 b; .reg .s32 i; cvt.f64.f32 a,b; cvt.rn.f32.f64 b,a; cvt.rzi.s32.f64 i,a; ret; }";
        let mut module = parse_module_checked(text).unwrap();
        let Directive::Method(_, method) = &mut module.directives[0] else { panic!() };
        let mut changed = 0;
        for statement in method.body.as_mut().unwrap() {
            if let Statement::Instruction(_, instruction) = statement {
                if !instruction.map_f64_to_f32() { continue; }
                let Instruction::Cvt { data, .. } = instruction else { panic!() };
                assert!(data.from == ScalarType::F32);
                if data.to == ScalarType::F32 { assert!(matches!(data.mode, CvtMode::FPRound { .. })); }
                else { assert!(matches!(data.mode, CvtMode::SignedFromFP { .. })); }
                changed += 1;
            }
        }
        assert_eq!(changed, 3);
    }
}

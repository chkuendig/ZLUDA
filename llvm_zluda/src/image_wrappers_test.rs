use crate::utils::{Context, Module};
use llvm_sys::{core::*, ir_reader::LLVMParseIRInContext, prelude::*, LLVMAttributeFunctionIndex};
use std::{ffi::CStr, ptr};

fn fixture(context: &Context) -> Module {
    let ir = include_bytes!("image_wrappers_test.ll");
    unsafe {
        // Release compilation discards names; LLVM's textual parser needs them.
        LLVMContextSetDiscardValueNames(context.get(), 0);
        let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            ir.as_ptr().cast(),
            ir.len(),
            c"image-wrapper-test".as_ptr(),
        );
        let mut module = ptr::null_mut();
        let mut error = ptr::null_mut();
        let rc = LLVMParseIRInContext(context.get(), buffer, &mut module, &mut error);
        if rc != 0 {
            let message = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            panic!("{message}");
        }
        let module = Module::from_raw(module);
        module.verify().unwrap();
        module
    }
}

unsafe fn attributes(function: LLVMValueRef) -> Vec<LLVMAttributeRef> {
    let count = LLVMGetAttributeCountAtIndex(function, LLVMAttributeFunctionIndex);
    let mut result = vec![ptr::null_mut(); count as usize];
    LLVMGetAttributesAtIndex(function, LLVMAttributeFunctionIndex, result.as_mut_ptr());
    result
}

#[test]
fn image_wrapper_preserves_fp_modes_and_descriptor_operands() {
    let context = Context::new();
    let module = fixture(&context);
    unsafe {
        let callers = [c"strict_caller", c"ieee_caller"];
        let before: Vec<_> = callers
            .iter()
            .map(|name| attributes(LLVMGetNamedFunction(module.get(), name.as_ptr())))
            .collect();
        assert_eq!(crate::LLVMZludaInlineImageWrappers(module.get()), 2);
        module.verify().unwrap();
        let intrinsic = LLVMGetNamedFunction(
            module.get(),
            c"llvm.amdgcn.image.sample.lz.3d.v4f32.f32.v8i32.v4i32".as_ptr(),
        );
        for (name, expected) in callers.iter().zip(before) {
            let function = LLVMGetNamedFunction(module.get(), name.as_ptr());
            assert_eq!(attributes(function), expected);
            let call = LLVMGetFirstInstruction(LLVMGetFirstBasicBlock(function));
            assert_eq!(LLVMGetCalledValue(call), intrinsic);
            assert_eq!(LLVMConstIntGetZExtValue(LLVMGetArgOperand(call, 0)), 15);
            assert_eq!(LLVMGetArgOperand(call, 1), LLVMGetParam(function, 0));
            assert_eq!(LLVMGetArgOperand(call, 4), LLVMGetParam(function, 1));
            assert_eq!(LLVMGetArgOperand(call, 5), LLVMGetParam(function, 2));
        }
    }
}

#[test]
fn image_wrapper_rejects_arithmetic_control_flow_and_non_image_calls() {
    let context = Context::new();
    let module = fixture(&context);
    unsafe {
        assert_eq!(crate::LLVMZludaInlineImageWrappers(module.get()), 2);
        for name in [
            c"arithmetic_caller",
            c"branch_caller",
            c"non_image_caller",
            c"noinline_caller",
        ] {
            let function = LLVMGetNamedFunction(module.get(), name.as_ptr());
            let call = LLVMGetFirstInstruction(LLVMGetFirstBasicBlock(function));
            let callee = LLVMGetCalledValue(call);
            assert_eq!(LLVMGetIntrinsicID(callee), 0, "{name:?}");
        }
        assert_eq!(crate::LLVMZludaInlineImageWrappers(module.get()), 0);
    }
    module.verify().unwrap();
}

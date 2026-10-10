use crate::utils::{Context, Message, PassBuilderOptions, TargetMachine};
use crate::LLVMZludaParseCommandLineOptions;
use crate::{ffi::LLVMZludaLinkWithLLD, utils::Module};
use llvm_sys::{
    core::*,
    target::{
        LLVMInitializeAMDGPUAsmParser, LLVMInitializeAMDGPUAsmPrinter, LLVMInitializeAMDGPUTarget,
        LLVMInitializeAMDGPUTargetInfo, LLVMInitializeAMDGPUTargetMC,
    },
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMGetTargetFromTriple,
        LLVMRelocMode,
    },
    transforms::pass_builder::LLVMRunPasses,
};
use llvm_sys::{LLVMLinkage, LLVMVisibility};
use std::ffi::{CStr, CString};
use std::sync::OnceLock;
use std::{fs, ptr};
use tempfile::NamedTempFile;

const OCKL_MODULE: &[u8] = include_bytes!("device-libs/ockl.bc");

// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
const CONSTANT_ADDRESS_SPACE: u32 = 4;

fn load_module(ctx: &Context, bc: &[u8], name: &std::ffi::CStr) -> Result<Module, String> {
    let module = Module::try_from_bitcode(ctx, bc, Some(name))
        .ok_or(("Failed to parse bitcode").to_string())?;
    module.verify()?;
    Ok(module)
}

// TODO: see if there's a way to reduce duplication with attributes.rs
fn add_constant(context: &Context, module: &Module, name: &CStr, attribute: u32) {
    let attribute_type = unsafe { LLVMInt32TypeInContext(context.get()) };
    let global = unsafe {
        LLVMAddGlobalInAddressSpace(
            module.get(),
            attribute_type,
            name.as_ptr(),
            CONSTANT_ADDRESS_SPACE,
        )
    };
    unsafe { LLVMSetLinkage(global, LLVMLinkage::LLVMExternalLinkage) };
    unsafe { LLVMSetVisibility(global, LLVMVisibility::LLVMHiddenVisibility) };
    unsafe { LLVMSetInitializer(global, LLVMConstInt(attribute_type, attribute as u64, 0)) };
    unsafe { LLVMSetGlobalConstant(global, 1) };
}

fn path_to_cstring(path: &std::path::Path) -> Result<CString, String> {
    let path_str = path
        .to_str()
        .ok_or(("path is not valid as str").to_string())?;
    CString::new(path_str).map_err(|_| ("path includes invalid null byte").to_string())
}

fn get_isa_version_from_gcn_arch(gcn_arch: &str) -> Result<u32, String> {
    let base: u32 = gcn_arch
        .replace("gfx", "")
        .parse()
        .map_err(|_| ("could not get ISA version from gcn_arch").to_string())?;
    let stepping = base % 10;
    let minor = (base / 10) % 10;
    let major = base / 100;
    Ok(major * 1000 + minor * 100 + stepping)
}

fn create_oclc_constants(ctx: &Context, gcn_arch: &str) -> Result<Module, String> {
    let module = Module::new(ctx, c"oclc_constants");

    // used by ockl
    add_constant(ctx, &module, c"__oclc_wavefrontsize64", 0);
    add_constant(ctx, &module, c"__oclc_wavefrontsize_log2", 5);
    add_constant(ctx, &module, c"__oclc_ABI_version", 500);
    add_constant(
        ctx,
        &module,
        c"__oclc_ISA_version",
        get_isa_version_from_gcn_arch(gcn_arch)?,
    );

    // used by ocml
    add_constant(ctx, &module, c"__oclc_unsafe_math_opt", 0);
    add_constant(ctx, &module, c"__oclc_correctly_rounded_sqrt32", 1);
    add_constant(ctx, &module, c"__oclc_finite_only_opt", 0);
    Ok(module)
}

pub fn compile(
    ctx: &Context,
    gcn_arch: &str,
    main: Module,
    ptx_impl: &[u8],
    attributes: Module,
    metadata: kernel_metadata::ModuleMetadataV1,
    metadata32: Option<kernel_metadata::ModuleMetadata32Bit>,
    compiler_hook: Option<&dyn Fn(&Vec<u8>, String)>,
) -> Result<Vec<u8>, String> {
    init_globals()?;

    let linked = Module::new(ctx, c"llvm-link");

    let ptx_impl = load_module(ctx, ptx_impl, c"ptx_impl.bc")?;
    let ockl = load_module(ctx, OCKL_MODULE, c"ockl.bc")?;

    let oclc_constants = create_oclc_constants(ctx, gcn_arch)?;

    linked.link(main)?;
    linked.link(attributes)?;
    linked.link(oclc_constants)?;
    linked.link(ptx_impl)?;
    linked.link(ockl)?;

    linked.verify()?;

    if let Some(hook) = compiler_hook {
        // Run compiler hook on human-readable LLVM IR
        let message = linked.print_module_to_string();
        let data = message.to_bytes().to_vec();
        hook(&data, String::from("linked.ll"));
    }

    let triple = c"amdgcn-amd-amdhsa";
    let cpu = std::ffi::CString::new(gcn_arch).map_err(|_| ("invalid gcn_arch").to_string())?;
    /* Disable FeatureBackOffBarrier when ZLUDA_DISABLE_BACK_OFF_BARRIER=1.
     * The feature (added by upstream D120544) lets gfx10.x hardware omit
     * the implicit s_waitcnt before s_barrier — fine for graphics
     * workloads that emit explicit fences but breaks CUDA's implicit
     * `bar.sync 0` LDS-commit semantic. We compensate via an explicit
     * `s_waitcnt(0)` in `__zluda_ptx_impl_bar_sync` (commit 579bcc05),
     * but that's only invoked from PTX-translated `bar.sync`, not from
     * any other LDS-fence path the LLVM codegen might emit. Disabling
     * the optimisation entirely makes the codegen always emit the wait,
     * which should make residual TFS/IOFX races go away if they're
     * caused by some non-bar.sync LDS visibility gap. */
    let features_static: &CStr = if std::env::var("ZLUDA_DISABLE_BACK_OFF_BARRIER")
        .ok()
        .as_deref() == Some("1")
    {
        c"-wavefrontsize64,+cumode,-back-off-barrier"
    } else {
        c"-wavefrontsize64,+cumode"
    };
    let features = features_static;

    let mut target = unsafe { std::mem::zeroed() };
    let mut err = std::ptr::null_mut();
    let status = unsafe { LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut err) };
    if status != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        return Err(message.to_str().to_string());
    }

    // Backend (register-allocation / scheduling) opt level. Normally
    // Aggressive. A few pathologically large kernels (e.g. Batman:AK's
    // _Z14simulateClothsv, 80k PTX lines with hundreds of strict.wwm shfl
    // regions) hit an LLVM AMDGPU greedy-RA/WWM MIR-corruption SIGSEGV in
    // `SIPreAllocateWWMRegs` at -O1+; the same module codegens correctly at
    // -O0 (fast RA). `module.rs::compile_in_subprocess` retries a crashed
    // module with ZLUDA_CODEGEN_OPT_LEVEL=0 so it still compiles (correct,
    // just less-optimized codegen) instead of failing the whole module.
    // The middle-end default<O3> pipeline above is unaffected.
    let codegen_level = match std::env::var("ZLUDA_CODEGEN_OPT_LEVEL").ok().as_deref() {
        Some("0") | Some("none") => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
        Some("1") => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
        Some("2") => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
        _ => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
    };
    let target_machine = TargetMachine::new(
        target,
        triple,
        &cpu,
        features,
        codegen_level,
        LLVMRelocMode::LLVMRelocDefault,
        LLVMCodeModel::LLVMCodeModelDefault,
    );

    let pb_options = PassBuilderOptions::new();
    let error = unsafe {
        LLVMRunPasses(
            linked.get(),
            c"default<O3>".as_ptr(),
            target_machine.get(),
            pb_options.get(),
        )
    };

    if !error.is_null() {
        let err_msg = unsafe { llvm_sys::error::LLVMGetErrorMessage(error) };
        let message = Message::new(unsafe { CStr::from_ptr(err_msg) });
        return Err(message.to_str().to_string());
    }

    // Legacy experiment: choose zero for undef phi inputs after optimization.
    // This is not a PTX guarantee that uninitialized registers are zero, nor
    // proof of an optimizer bug. The original experiment did not fix MS_4.
    // Keep the old control available; retirement comparisons disable it.
    if crate::legacy_workaround_enabled("ZLUDA_ZERO_UNDEF") {
        unsafe {
            let mut func = LLVMGetFirstFunction(linked.get());
            while !func.is_null() {
                let mut bb = LLVMGetFirstBasicBlock(func);
                while !bb.is_null() {
                    let mut inst = LLVMGetFirstInstruction(bb);
                    while !inst.is_null() {
                        if !LLVMIsAPHINode(inst).is_null() {
                            let count = LLVMCountIncoming(inst);
                            for i in 0..count {
                                let val = LLVMGetIncomingValue(inst, i);
                                if LLVMIsUndef(val) != 0 {
                                    let ty = LLVMTypeOf(val);
                                    let zero = LLVMConstNull(ty);
                                    LLVMSetOperand(inst, i, zero);
                                }
                            }
                        }
                        inst = LLVMGetNextInstruction(inst);
                    }
                    bb = LLVMGetNextBasicBlock(bb);
                }
                func = LLVMGetNextFunction(func);
            }
        }
    }

    if let Some(hook) = compiler_hook {
        // Run compiler hook on optimized human-readable LLVM IR
        let message = linked.print_module_to_string();
        let data = message.to_bytes().to_vec();
        hook(&data, String::from("opt.ll"));

        // Running a disassembler would be a bit of a pain, so run codegen as assembly
        let assembly = target_machine
            .emit_to_memory_buffer(&linked.clone(), LLVMCodeGenFileType::LLVMAssemblyFile)?
            .to_vec();
        hook(&assembly, String::from("asm"))
    }

    let object_file = target_machine
        .emit_to_memory_buffer(&linked, LLVMCodeGenFileType::LLVMObjectFile)?
        .to_vec();

    if let Some(hook) = compiler_hook {
        // Run compiler hook for object file
        hook(&object_file, String::from("o"));
    }

    let object_path = NamedTempFile::with_prefix("zluda.o")
        .map_err(|e| format!("Failed to create temporary file: {}", e))?
        .into_temp_path();
    let mut section_path = NamedTempFile::with_prefix("zluda_section")
        .map_err(|e| format!("Failed to create temporary file: {}", e))?;
    let executable_path = NamedTempFile::with_prefix("zluda.elf")
        .map_err(|e| format!("Failed to create temporary file: {}", e))?
        .into_temp_path();

    fs::write(&object_path, &object_file)
        .map_err(|e| format!("Failed to write object file: {}", e))?;
    metadata
        .write_object(&object_file, section_path.as_file_mut())
        .map_err(|e| format!("Failed to write metadata section: {}", e))?;
    let section32_path = if let Some(metadata32) = metadata32 {
        let mut section32_path = NamedTempFile::with_prefix("zluda32_section")
            .map_err(|e| format!("Failed to create temporary file: {}", e))?;
        metadata32
            .write_object(&object_file, section32_path.as_file_mut())
            .map_err(|e| format!("Failed to write 32-bit metadata section: {}", e))?;
        Some(section32_path)
    } else {
        None
    };

    let object_path_cstr = path_to_cstring(&object_path)?;
    let section_path_cstr = path_to_cstring(section_path.as_ref())?;
    let section32_path_cstr = section32_path
        .as_ref()
        .map(|p| path_to_cstring(p.as_ref()))
        .transpose()?;
    let executable_path_cstr = path_to_cstring(&executable_path)?;

    let (inputs, inputs_len) = if let Some(ref section32_path_cstr) = section32_path_cstr {
        (
            [
                object_path_cstr.as_ptr(),
                section_path_cstr.as_ptr(),
                section32_path_cstr.as_ptr(),
            ],
            3,
        )
    } else {
        (
            [
                object_path_cstr.as_ptr(),
                section_path_cstr.as_ptr(),
                ptr::null(),
            ],
            2,
        )
    };

    let mut err = std::ptr::null_mut();
    let result = unsafe {
        LLVMZludaLinkWithLLD(
            inputs_len,
            inputs.as_ptr(),
            executable_path_cstr.as_ptr(),
            &mut err,
        )
    };
    if result != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        return Err(message.to_str().to_string());
    }

    let executable =
        fs::read(&executable_path).map_err(|_| ("Failed to read executable file").to_string())?;

    if let Some(hook) = compiler_hook {
        // Run compiler hook for final executable
        hook(&executable, String::from("elf"));
    }

    Ok(executable)
}

fn init_globals() -> Result<(), String> {
    static INIT_AMDGPU: OnceLock<Result<(), Message>> = OnceLock::new();
    INIT_AMDGPU
        .get_or_init(|| {
            let common_options = vec![
                // Uncomment for LLVM debug
                //c"-debug",
                // Uncomment to save passes
                // c"-print-before-all",
                c"llvm_zluda",
                //c"-debug-only=isel",
                c"-ignore-tti-inline-compatible",
                // c"-amdgpu-early-inline-all=true",
                c"-amdgpu-internalize-symbols",
                c"-amdhsa-code-object-version=5",
                //c"--pass-remarks-missed=.*inlin.*",
            ]
            .into_iter();
            /* Does not provide as much performance improvement on ROCm 7.2
            let opt_options = if cfg!(debug_assertions) {
                vec![]
            } else {
                vec![
                    // default inlining threshold times 10
                    c"-inline-threshold=2250",
                    c"-inlinehint-threshold=3250",
                ]
            };
            */
            // ZLUDA_LLVM_ARGS: whitespace-separated extra LLVM cl::opt flags,
            // injected at first-compile init. Diagnostic-only — used to bisect
            // AMDGPU codegen passes (e.g. "-amdgpu-enable-whole-quad-mode=0")
            // against the deterministic nrss_replay GPU A/B without rebuilding
            // LLVM. The CStrings must outlive ParseCommandLineOptions, so leak
            // them (one-shot init, negligible).
            let extra_args: Vec<&'static CStr> = match std::env::var("ZLUDA_LLVM_ARGS") {
                Ok(s) if !s.trim().is_empty() => s
                    .split_whitespace()
                    .map(|tok| {
                        let cs = std::ffi::CString::new(tok).unwrap();
                        &*Box::leak(cs.into_boxed_c_str())
                    })
                    .collect(),
                _ => vec![],
            };
            let llvm_args_ptrs: Vec<*const i8> = common_options
                //.chain(opt_options)
                .chain(extra_args.into_iter())
                .map(|s| s.as_ptr())
                .collect();
            let mut err_msg = std::ptr::null_mut();
            let success = unsafe {
                LLVMZludaParseCommandLineOptions(
                    llvm_args_ptrs.len() as i32,
                    llvm_args_ptrs.as_ptr(),
                    &mut err_msg,
                )
            };
            if !success {
                return Err(Message::new(unsafe { CStr::from_ptr(err_msg) }));
            }
            unsafe { LLVMInitializeAMDGPUTargetInfo() };
            unsafe { LLVMInitializeAMDGPUTarget() };
            unsafe { LLVMInitializeAMDGPUTargetMC() };
            unsafe { LLVMInitializeAMDGPUAsmPrinter() };
            unsafe { LLVMInitializeAMDGPUAsmParser() };
            Ok(())
        })
        .as_ref()
        .map(|()| ())
        .map_err(|e| e.to_str().to_string())
}

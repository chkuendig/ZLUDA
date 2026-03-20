use crate::utils::{Context, Message, PassBuilderOptions, TargetMachine};
use crate::LLVMZludaParseCommandLineOptions;
use crate::{ffi::LLVMZludaLinkWithLLD, utils::Module};
use llvm_sys::{
    core::*,
    target::{
        LLVMInitializeAMDGPUAsmPrinter, LLVMInitializeAMDGPUTarget, LLVMInitializeAMDGPUTargetInfo,
        LLVMInitializeAMDGPUTargetMC,
    },
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMGetTargetFromTriple,
        LLVMRelocMode,
    },
    transforms::pass_builder::LLVMRunPasses,
};
use llvm_sys::{LLVMLinkage, LLVMVisibility};
use std::ffi::{CStr, CString};
use std::fs;
use std::sync::OnceLock;
use std::time::Duration;
use tempfile::NamedTempFile;

const OCKL_MODULE: &[u8] = include_bytes!("device-libs/ockl.bc");
const OCML_MODULE: &[u8] = include_bytes!("device-libs/ocml.bc");

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

fn compile_log(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/zluda_compile.log") {
        let _ = writeln!(f, "{}", msg);
    }
}

pub fn compile(
    ctx: &Context,
    gcn_arch: &str,
    main: Module,
    ptx_impl: &[u8],
    attributes: Module,
    compiler_hook: Option<&dyn Fn(&Vec<u8>, String)>,
) -> Result<Vec<u8>, String> {
    // Wine threads typically have 1MB stacks, but LLVM's bitcode parser
    // and optimization passes need much more. Spawn a native thread with
    // a 64MB stack to avoid SIGSEGV during compilation.
    compile_log("compile: spawning large-stack thread (64MB) for LLVM");
    let stack_size = 64 * 1024 * 1024; // 64MB

    // Pack raw pointers to move across thread boundary.
    // SAFETY: LLVM contexts/modules are not thread-safe, but we block
    // the calling thread (join) so there's no concurrent access.
    struct SendPtrs {
        ctx: usize,
        main: usize,
        attr: usize,
        ptx_impl: Vec<u8>,
        gcn_arch: String,
    }
    unsafe impl Send for SendPtrs {}

    let ptrs = SendPtrs {
        ctx: ctx as *const Context as usize,
        main: main.get() as usize,
        attr: attributes.get() as usize,
        ptx_impl: ptx_impl.to_vec(),
        gcn_arch: gcn_arch.to_string(),
    };

    // Prevent the caller's Modules from calling LLVMDisposeModule —
    // ownership transfers to the thread.
    std::mem::forget(main);
    std::mem::forget(attributes);

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("zluda-llvm-compile".into())
        .stack_size(stack_size)
        .spawn(move || {
            let ctx = unsafe { &*(ptrs.ctx as *const Context) };
            let main = unsafe { Module::from_raw(ptrs.main as *mut _) };
            let attributes = unsafe { Module::from_raw(ptrs.attr as *mut _) };
            let result = compile_inner(ctx, &ptrs.gcn_arch, main, &ptrs.ptx_impl, attributes, None);
            let _ = tx.send(result);
        })
        .map_err(|e| format!("Failed to spawn compile thread: {}", e))?;

    match rx.recv_timeout(Duration::from_secs(120)) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            compile_log("compile: TIMED OUT after 120s — abandoning module");
            Err("LLVM compilation timed out after 120s".to_string())
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            compile_log("compile: thread panicked (channel disconnected)");
            Err("Compile thread panicked".to_string())
        }
    }
}

fn compile_inner(
    ctx: &Context,
    gcn_arch: &str,
    main: Module,
    ptx_impl: &[u8],
    attributes: Module,
    compiler_hook: Option<&dyn Fn(&Vec<u8>, String)>,
) -> Result<Vec<u8>, String> {
    compile_log("compile: init_globals");
    init_globals()?;

    compile_log("compile: Module::new linked");
    let linked = Module::new(ctx, c"llvm-link");

    compile_log(&format!("compile: load ptx_impl.bc ({} bytes)", ptx_impl.len()));
    let ptx_impl_mod = Module::try_from_bitcode(ctx, ptx_impl, Some(c"ptx_impl.bc"))
        .ok_or("Failed to parse ptx_impl bitcode".to_string())?;
    compile_log("compile: ptx_impl parsed, verifying");
    ptx_impl_mod.verify()?;
    compile_log("compile: ptx_impl verified");
    let ptx_impl = ptx_impl_mod;
    compile_log("compile: load ockl.bc");
    let ockl = load_module(ctx, OCKL_MODULE, c"ockl.bc")?;
    compile_log("compile: load ocml.bc");
    let ocml = load_module(ctx, OCML_MODULE, c"ocml.bc")?;

    compile_log("compile: create oclc_constants");
    let oclc_constants = create_oclc_constants(ctx, gcn_arch)?;

    compile_log("compile: linking main");
    linked.link(main)?;
    compile_log("compile: linking attributes");
    linked.link(attributes)?;
    compile_log("compile: linking oclc_constants");
    linked.link(oclc_constants)?;
    compile_log("compile: linking ptx_impl");
    linked.link(ptx_impl)?;
    compile_log("compile: linking ockl");
    linked.link(ockl)?;
    compile_log("compile: linking ocml");
    linked.link(ocml)?;

    compile_log("compile: verify");
    linked.verify()?;
    compile_log("compile: verify OK");

    // Always dump linked (pre-optimization) IR for debugging
    let linked_ir_size;
    {
        static LINKED_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = LINKED_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
        let message = linked.print_module_to_string();
        let data = message.to_bytes();
        linked_ir_size = data.len();
        let _ = std::fs::write(format!("/tmp/zluda_ir_dump/module_{}_linked.ll", n), data);
        compile_log(&format!("compile: dumped linked IR to /tmp/zluda_ir_dump/module_{}_linked.ll ({} bytes)", n, linked_ir_size));
    }

    if let Some(hook) = compiler_hook {
        // Run compiler hook on human-readable LLVM IR
        let message = linked.print_module_to_string();
        let data = message.to_bytes().to_vec();
        hook(&data, String::from("linked.ll"));
    }

    compile_log("compile: setting up target machine");
    let triple = c"amdgcn-amd-amdhsa";
    let cpu = std::ffi::CString::new(gcn_arch).map_err(|_| ("invalid gcn_arch").to_string())?;
    let features = c"-wavefrontsize64,+cumode";

    let mut target = unsafe { std::mem::zeroed() };
    let mut err = std::ptr::null_mut();
    let status = unsafe { LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut err) };
    if status != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        return Err(message.to_str().to_string());
    }
    compile_log("compile: target found, creating target machine");

    let target_machine = TargetMachine::new(
        target,
        triple,
        &cpu,
        features,
        LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        LLVMRelocMode::LLVMRelocDefault,
        LLVMCodeModel::LLVMCodeModelDefault,
    );

    // Use O2 for large modules to avoid pathological O3 compile times
    // (e.g. TurbulenceFS at 2.8MB linked IR hangs indefinitely with O3)
    let opt_passes = if linked_ir_size > 2_000_000 {
        compile_log(&format!("compile: large module ({} bytes), using default<O2>", linked_ir_size));
        c"default<O2>"
    } else {
        compile_log(&format!("compile: running LLVMRunPasses (default<O3>, {} bytes)", linked_ir_size));
        c"default<O3>"
    };
    let pb_options = PassBuilderOptions::new();
    let error = unsafe {
        LLVMRunPasses(
            linked.get(),
            opt_passes.as_ptr(),
            target_machine.get(),
            pb_options.get(),
        )
    };

    if !error.is_null() {
        let err_msg = unsafe { llvm_sys::error::LLVMGetErrorMessage(error) };
        let message = Message::new(unsafe { CStr::from_ptr(err_msg) });
        compile_log(&format!("compile: LLVMRunPasses FAILED: {}", message.to_str()));
        return Err(message.to_str().to_string());
    }
    compile_log("compile: LLVMRunPasses OK");

    // Always dump LLVM IR for debugging
    {
        static DUMP_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = DUMP_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
        let message = linked.print_module_to_string();
        let data = message.to_bytes();
        let _ = std::fs::write(format!("/tmp/zluda_ir_dump/module_{}_opt.ll", n), data);
        compile_log(&format!("compile: dumped IR to /tmp/zluda_ir_dump/module_{}_opt.ll ({} bytes)", n, data.len()));
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

    compile_log("compile: emitting object file");
    let object_file = target_machine
        .emit_to_memory_buffer(&linked, LLVMCodeGenFileType::LLVMObjectFile)?
        .to_vec();
    compile_log(&format!("compile: object file {} bytes", object_file.len()));

    if let Some(hook) = compiler_hook {
        // Run compiler hook for object file
        hook(&object_file, String::from("o"));
    }

    let object_path = NamedTempFile::with_prefix("zluda.o")
        .map_err(|_| ("Failed to create temporary file").to_string())?
        .into_temp_path();
    let executable_path = NamedTempFile::with_prefix("zluda.elf")
        .map_err(|_| ("Failed to create temporary file").to_string())?
        .into_temp_path();

    fs::write(&object_path, &object_file)
        .map_err(|_| ("Failed to write object file").to_string())?;

    let object_path_cstr = path_to_cstring(&object_path)?;
    let executable_path_cstr = path_to_cstring(&executable_path)?;

    compile_log("compile: linking with LLD");
    let mut err = std::ptr::null_mut();
    let result = unsafe {
        LLVMZludaLinkWithLLD(
            object_path_cstr.as_ptr(),
            executable_path_cstr.as_ptr(),
            &mut err,
        )
    };
    if result != 0 {
        let message = Message::new(unsafe { CStr::from_ptr(err) });
        compile_log(&format!("compile: LLD FAILED: {}", message.to_str()));
        return Err(message.to_str().to_string());
    }

    compile_log("compile: LLD link OK, reading executable");
    let executable =
        fs::read(&executable_path).map_err(|_| ("Failed to read executable file").to_string())?;
    compile_log(&format!("compile: DONE, executable {} bytes", executable.len()));

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
                c"-ignore-tti-inline-compatible",
                // c"-amdgpu-early-inline-all=true",
                c"-amdgpu-internalize-symbols",
                c"-amdhsa-code-object-version=5",
            ]
            .into_iter();
            let opt_options = if cfg!(debug_assertions) {
                vec![]
            } else {
                vec![
                    // default inlining threshold times 10
                    c"-inline-threshold=2250",
                    c"-inlinehint-threshold=3250",
                ]
            };
            let llvm_args_ptrs: Vec<*const i8> = common_options
                .chain(opt_options)
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
            Ok(())
        })
        .as_ref()
        .map(|()| ())
        .map_err(|e| e.to_str().to_owned())
}

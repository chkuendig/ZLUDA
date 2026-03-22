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
use llvm_sys::prelude::LLVMValueRef;
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

    // Count internal helpers to decide emission strategy.
    // Modules with helpers need per-kernel emission to avoid the AMDGPU
    // backend's SCC-ordered codegen hanging on large cross-kernel call graphs
    // (e.g. TurbulenceFS: 31 kernels × 7 helpers).
    let helper_count = count_internal_helpers(&linked);

    let emit_start = std::time::Instant::now();
    let object_files = if helper_count > 0 {
        match emit_per_kernel(&target_machine, &linked) {
            Ok(objects) => objects,
            Err(e) => {
                // Per-kernel emission failed (e.g. sust_2d SGPR constraint).
                // Fall back to full-module alwaysinline — this produces an
                // object with diagnostic warnings but continues (same as
                // committed behavior before per-kernel emission).
                compile_log(&format!(
                    "compile: per-kernel emission failed: {}, falling back to full-module alwaysinline",
                    e
                ));
                mark_helpers_alwaysinline(&linked);
                let obj = target_machine
                    .emit_to_memory_buffer(&linked, LLVMCodeGenFileType::LLVMObjectFile)?
                    .to_vec();
                vec![obj]
            }
        }
    } else {
        compile_log("compile: emitting single object file (no helpers)");
        let obj = target_machine
            .emit_to_memory_buffer(&linked, LLVMCodeGenFileType::LLVMObjectFile)?
            .to_vec();
        vec![obj]
    };
    compile_log(&format!(
        "compile: emitted {} object(s) in {:.1}s",
        object_files.len(),
        emit_start.elapsed().as_secs_f64()
    ));

    if let Some(hook) = compiler_hook {
        hook(&object_files[0], String::from("o"));
    }

    // Write objects to temp files for LLD
    let object_paths: Vec<_> = object_files
        .iter()
        .enumerate()
        .map(|(i, obj)| -> Result<_, String> {
            let path = NamedTempFile::with_prefix(&format!("zluda_{}.o", i))
                .map_err(|_| "Failed to create temporary file".to_string())?
                .into_temp_path();
            fs::write(&path, obj)
                .map_err(|_| "Failed to write object file".to_string())?;
            Ok(path)
        })
        .collect::<Result<Vec<_>, String>>()?;

    let executable_path = NamedTempFile::with_prefix("zluda.elf")
        .map_err(|_| "Failed to create temporary file".to_string())?
        .into_temp_path();

    let path_cstrs: Vec<CString> = object_paths
        .iter()
        .map(|p| path_to_cstring(p))
        .collect::<Result<Vec<_>, String>>()?;
    let path_ptrs: Vec<*const i8> = path_cstrs.iter().map(|cs| cs.as_ptr()).collect();
    let executable_path_cstr = path_to_cstring(&executable_path)?;

    compile_log(&format!("compile: linking with LLD ({} objects)", path_ptrs.len()));
    let mut err = std::ptr::null_mut();
    let result = unsafe {
        LLVMZludaLinkWithLLD(
            path_ptrs.as_ptr(),
            path_ptrs.len() as i32,
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
    let mut executable =
        fs::read(&executable_path).map_err(|_| ("Failed to read executable file").to_string())?;
    if object_files.len() > 1 {
        merge_amdgpu_metadata_notes(&mut executable)?;
    }
    compile_log(&format!("compile: DONE, executable {} bytes", executable.len()));

    // Dump final ELF for readelf inspection
    {
        static ELF_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = ELF_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
        let _ = std::fs::write(format!("/tmp/zluda_ir_dump/module_{}.elf", n), &executable);
    }

    if let Some(hook) = compiler_hook {
        // Run compiler hook for final executable
        hook(&executable, String::from("elf"));
    }

    Ok(executable)
}

/// Count internal non-kernel function definitions (helpers).
fn count_internal_helpers(module: &Module) -> u32 {
    unsafe {
        let mut count = 0u32;
        let mut func = LLVMGetFirstFunction(module.get());
        while !func.is_null() {
            let is_definition = LLVMCountBasicBlocks(func) > 0;
            let is_kernel = LLVMGetFunctionCallConv(func) == 91;
            if is_definition && !is_kernel {
                count += 1;
            }
            func = LLVMGetNextFunction(func);
        }
        count
    }
}

/// Collect names of kernel functions (callconv 91 with body).
fn collect_kernel_names(module: &Module) -> Vec<String> {
    unsafe {
        let mut names = Vec::new();
        let mut func = LLVMGetFirstFunction(module.get());
        while !func.is_null() {
            let is_definition = LLVMCountBasicBlocks(func) > 0;
            let is_kernel = LLVMGetFunctionCallConv(func) == 91;
            if is_definition && is_kernel {
                let mut name_len = 0;
                let name_ptr = LLVMGetValueName2(func, &mut name_len);
                if !name_ptr.is_null() && name_len > 0 {
                    let name = std::str::from_utf8_unchecked(
                        std::slice::from_raw_parts(name_ptr as *const u8, name_len),
                    );
                    names.push(name.to_string());
                }
            }
            func = LLVMGetNextFunction(func);
        }
        names
    }
}

/// Emit one object file per kernel for modules with internal helpers.
///
/// Clones the module for each kernel, makes all other kernels internal,
/// then runs globaldce to strip unreachable code. Each resulting sub-module
/// has exactly 1 kernel + only its reachable helpers, eliminating the
/// cross-kernel SCC that hangs the AMDGPU backend.
fn emit_per_kernel(
    tm: &TargetMachine,
    module: &Module,
) -> Result<Vec<Vec<u8>>, String> {
    let kernel_names = collect_kernel_names(module);
    compile_log(&format!(
        "compile: per-kernel emission for {} kernels",
        kernel_names.len()
    ));

    // Check if per-kernel IR/ASM dumps are requested
    static DUMP_PER_KERNEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let dump = *DUMP_PER_KERNEL.get_or_init(|| {
        std::env::var("ZLUDA_DUMP_PER_KERNEL").map_or(false, |v| v == "1")
    });

    let mut objects = Vec::new();
    for (idx, name) in kernel_names.iter().enumerate() {
        compile_log(&format!(
            "compile:   [{}/{}] {} cloning",
            idx + 1, kernel_names.len(), name
        ));
        let sub = module.clone();
        internalize_other_kernels(&sub, name);
        run_dce_pass(tm, &sub);
        delete_unused_globals(&sub);
        mark_helpers_alwaysinline(&sub);

        // Dump per-kernel IR and assembly for diagnosis
        if dump {
            let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
            let ir = sub.print_module_to_string();
            let _ = std::fs::write(
                format!("/tmp/zluda_ir_dump/kernel_{}.ll", name),
                ir.to_bytes(),
            );
            // Also dump assembly to see VGPR usage in kernel metadata
            if let Ok(asm_buf) = tm.emit_to_memory_buffer(&sub.clone(), LLVMCodeGenFileType::LLVMAssemblyFile) {
                let asm = asm_buf.to_vec();
                let _ = std::fs::write(
                    format!("/tmp/zluda_ir_dump/kernel_{}.s", name),
                    &asm,
                );
                // Extract and log VGPR info from assembly metadata
                if let Ok(asm_str) = std::str::from_utf8(&asm) {
                    for line in asm_str.lines() {
                        if line.contains(".vgpr_count:") || line.contains(".sgpr_count:")
                            || line.contains(".max_flat_workgroup_size:")
                            || line.contains(".vgpr_spill_count:")
                        {
                            compile_log(&format!("compile:   {} {}", name, line.trim()));
                        }
                    }
                }
            }
        }

        compile_log(&format!(
            "compile:   [{}/{}] {} emitting",
            idx + 1, kernel_names.len(), name
        ));
        let obj = tm
            .emit_to_memory_buffer(&sub, LLVMCodeGenFileType::LLVMObjectFile)?
            .to_vec();
        compile_log(&format!(
            "compile:   [{}/{}] {} -> {} bytes",
            idx + 1, kernel_names.len(), name, obj.len()
        ));
        objects.push(obj);
    }
    Ok(objects)
}

/// Make all kernel definitions except `keep` internal so globaldce can remove them.
fn internalize_other_kernels(module: &Module, keep: &str) {
    unsafe {
        let mut func = LLVMGetFirstFunction(module.get());
        while !func.is_null() {
            let is_definition = LLVMCountBasicBlocks(func) > 0;
            let is_kernel = LLVMGetFunctionCallConv(func) == 91;
            if is_definition && is_kernel {
                let mut name_len = 0;
                let name_ptr = LLVMGetValueName2(func, &mut name_len);
                let name = if !name_ptr.is_null() && name_len > 0 {
                    std::str::from_utf8_unchecked(
                        std::slice::from_raw_parts(name_ptr as *const u8, name_len),
                    )
                } else {
                    ""
                };
                if name != keep {
                    LLVMSetLinkage(func, LLVMLinkage::LLVMInternalLinkage);
                }
            }
            func = LLVMGetNextFunction(func);
        }
    }
}

/// Delete global variables that have no code uses and aren't in @llvm.used.
///
/// Removes dead globals (e.g. `$str` / `__CUDA_FTZ`) that create .data sections
/// in per-kernel objects. When per-kernel objects are linked by LLD, these dead
/// globals produce an ELF with both .data (SHT_PROGBITS) and .bss (SHT_NOBITS)
/// sections. Empirically, hipModuleGetGlobal then returns HOST addresses and
/// bytes=0 for .bss globals instead of valid device pointers.
///
/// Safety: @llvm.used protects all host-accessible globals (set during PTX emission
/// in emit.rs). Globals not in @llvm.used and with no code uses are truly dead.
fn delete_unused_globals(module: &Module) {
    unsafe {
        // Collect @llvm.used entries (protected globals)
        let mut used_set = std::collections::HashSet::new();
        let llvm_used = LLVMGetNamedGlobal(module.get(), c"llvm.used".as_ptr());
        if !llvm_used.is_null() {
            let init = LLVMGetInitializer(llvm_used);
            if !init.is_null() {
                let num = LLVMGetNumOperands(init);
                for i in 0..num {
                    let op = LLVMGetOperand(init, i as u32);
                    // @llvm.used entries may be addrspacecast(ptr addrspace(N) @X to ptr)
                    // Unwrap the cast to get the underlying global
                    let underlying = if LLVMIsAConstantExpr(op).is_null() {
                        op
                    } else {
                        LLVMGetOperand(op, 0)
                    };
                    used_set.insert(underlying);
                }
            }
        }

        // Collect globals to delete (can't delete while iterating)
        let mut to_delete = Vec::new();
        let mut global = LLVMGetFirstGlobal(module.get());
        while !global.is_null() {
            let next = LLVMGetNextGlobal(global);
            let is_llvm_meta = global == llvm_used;
            let is_protected = used_set.contains(&(global as LLVMValueRef));
            let has_uses = !LLVMGetFirstUse(global as LLVMValueRef).is_null();

            // Log each global's status for diagnosis
            let mut name_len = 0;
            let name_ptr = LLVMGetValueName2(global as LLVMValueRef, &mut name_len);
            let name = if !name_ptr.is_null() && name_len > 0 {
                std::str::from_utf8_unchecked(
                    std::slice::from_raw_parts(name_ptr as *const u8, name_len),
                )
            } else {
                "<unnamed>"
            };
            compile_log(&format!(
                "  global '{}': meta={} protected={} has_uses={} -> {}",
                name, is_llvm_meta, is_protected, has_uses,
                if !is_llvm_meta && !is_protected && !has_uses { "DELETE" } else { "keep" }
            ));

            if !is_llvm_meta && !is_protected && !has_uses {
                to_delete.push(global);
            }
            global = next;
        }

        // Delete collected globals
        for g in &to_delete {
            LLVMDeleteGlobal(*g);
        }
        if !to_delete.is_empty() {
            compile_log(&format!("  deleted {} unused globals", to_delete.len()));
        }
    }
}

/// Strip noinline/convergent and add alwaysinline to all non-kernel definitions.
/// This lets the AMDGPU codegen pipeline inline helpers, avoiding the SCC hang.
fn mark_helpers_alwaysinline(module: &Module) {
    unsafe {
        let noinline_kind = LLVMGetEnumAttributeKindForName(
            b"noinline".as_ptr().cast(),
            b"noinline".len(),
        );
        let convergent_kind = LLVMGetEnumAttributeKindForName(
            b"convergent".as_ptr().cast(),
            b"convergent".len(),
        );
        let alwaysinline_kind = LLVMGetEnumAttributeKindForName(
            b"alwaysinline".as_ptr().cast(),
            b"alwaysinline".len(),
        );
        let func_index = llvm_sys::LLVMAttributeFunctionIndex;
        // Create attribute using the module's context
        let ctx = LLVMGetModuleContext(module.get());
        let alwaysinline_attr = LLVMCreateEnumAttribute(ctx, alwaysinline_kind, 0);

        let mut func = LLVMGetFirstFunction(module.get());
        while !func.is_null() {
            let is_definition = LLVMCountBasicBlocks(func) > 0;
            let is_kernel = LLVMGetFunctionCallConv(func) == 91;
            if is_definition && !is_kernel {
                LLVMRemoveEnumAttributeAtIndex(func, func_index, noinline_kind);
                LLVMRemoveEnumAttributeAtIndex(func, func_index, convergent_kind);
                LLVMAddAttributeAtIndex(func, func_index, alwaysinline_attr);
            }
            func = LLVMGetNextFunction(func);
        }
    }
}

/// Run GlobalDCE to remove unreachable internal functions.
fn run_dce_pass(tm: &TargetMachine, module: &Module) {
    let pb = PassBuilderOptions::new();
    unsafe {
        LLVMRunPasses(module.get(), c"globaldce".as_ptr(), tm.get(), pb.get());
    }
}

/// Merge fragmented NT_AMDGPU_METADATA notes in an ELF binary.
///
/// When per-kernel emission produces multiple .o files, LLD concatenates
/// their NT_AMDGPU_METADATA notes instead of merging them. The HIP runtime
/// only reads the first note, so kernels from later objects have missing
/// metadata. This function merges all metadata notes into one.
fn merge_amdgpu_metadata_notes(elf: &mut Vec<u8>) -> Result<(), String> {
    const SHT_NOTE: u32 = 7;
    const PT_NOTE: u32 = 4;
    const NT_AMDGPU_METADATA: u32 = 0x20;
    const AMDGPU_NAME: &[u8] = b"AMDGPU\0";

    fn r_u16(data: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([data[off], data[off + 1]])
    }
    fn r_u32(data: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
    }
    fn r_u64(data: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
    }
    fn w_u64(data: &mut [u8], off: usize, val: u64) {
        data[off..off + 8].copy_from_slice(&val.to_le_bytes());
    }
    fn pad4(n: usize) -> usize {
        (n + 3) & !3
    }

    if elf.len() < 64 {
        return Ok(());
    }

    // Parse ELF64 header
    let e_shoff = r_u64(elf, 0x28) as usize;
    let e_shentsize = r_u16(elf, 0x3A) as usize;
    let e_shnum = r_u16(elf, 0x3C) as usize;
    let e_phoff = r_u64(elf, 0x20) as usize;
    let e_phentsize = r_u16(elf, 0x36) as usize;
    let e_phnum = r_u16(elf, 0x38) as usize;

    // Find .note section (type SHT_NOTE)
    let mut note_sh_idx = None;
    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        if sh + 0x28 > elf.len() {
            break;
        }
        let sh_type = r_u32(elf, sh + 4);
        if sh_type == SHT_NOTE {
            note_sh_idx = Some(i);
            break;
        }
    }
    let note_sh_idx = match note_sh_idx {
        Some(i) => i,
        None => return Ok(()),
    };

    let sh = e_shoff + note_sh_idx * e_shentsize;
    let note_offset = r_u64(elf, sh + 0x18) as usize;
    let note_size = r_u64(elf, sh + 0x20) as usize;

    if note_offset + note_size > elf.len() {
        return Ok(());
    }

    // Walk note entries, identify metadata notes
    struct NoteEntry {
        offset: usize,
        total_size: usize,
        is_metadata: bool,
        desc_offset: usize,
        desc_size: usize,
    }

    let mut entries = Vec::new();
    let mut pos = note_offset;
    let note_end = note_offset + note_size;
    while pos + 12 <= note_end {
        let namesz = r_u32(elf, pos) as usize;
        let descsz = r_u32(elf, pos + 4) as usize;
        let note_type = r_u32(elf, pos + 8);
        let name_start = pos + 12;
        let desc_start = name_start + pad4(namesz);
        let entry_end = desc_start + pad4(descsz);
        if entry_end > note_end {
            break;
        }
        let is_metadata = note_type == NT_AMDGPU_METADATA
            && namesz == AMDGPU_NAME.len()
            && elf[name_start..name_start + namesz] == *AMDGPU_NAME;
        entries.push(NoteEntry {
            offset: pos,
            total_size: entry_end - pos,
            is_metadata,
            desc_offset: desc_start,
            desc_size: descsz,
        });
        pos = entry_end;
    }

    let metadata_count = entries.iter().filter(|e| e.is_metadata).count();
    if metadata_count <= 1 {
        return Ok(());
    }

    compile_log(&format!(
        "merge_amdgpu_metadata_notes: merging {} metadata notes",
        metadata_count
    ));

    // Decode all metadata notes as msgpack, collect kernels
    let mut all_kernels: Vec<rmpv::Value> = Vec::new();
    let mut base_map: Option<Vec<(rmpv::Value, rmpv::Value)>> = None;

    for entry in &entries {
        if !entry.is_metadata {
            continue;
        }
        let desc = &elf[entry.desc_offset..entry.desc_offset + entry.desc_size];
        let val = rmpv::decode::read_value(&mut &desc[..])
            .map_err(|e| format!("Failed to decode AMDGPU metadata msgpack: {}", e))?;
        if let rmpv::Value::Map(pairs) = val {
            let mut kernels = Vec::new();
            let mut other_pairs = Vec::new();
            for (k, v) in pairs {
                if k.as_str() == Some("amdhsa.kernels") {
                    if let rmpv::Value::Array(arr) = v {
                        kernels = arr;
                    }
                } else {
                    other_pairs.push((k, v));
                }
            }
            all_kernels.extend(kernels);
            if base_map.is_none() {
                base_map = Some(other_pairs);
            }
        }
    }

    // Build merged map: base metadata + all kernels combined
    let mut merged_pairs = base_map.unwrap_or_default();
    merged_pairs.push((
        rmpv::Value::String("amdhsa.kernels".into()),
        rmpv::Value::Array(all_kernels),
    ));
    let merged_value = rmpv::Value::Map(merged_pairs);

    // Encode merged msgpack
    let mut merged_desc = Vec::new();
    rmpv::encode::write_value(&mut merged_desc, &merged_value)
        .map_err(|e| format!("Failed to encode merged AMDGPU metadata: {}", e))?;

    // Build new note section: non-metadata notes + single merged metadata note
    let mut new_notes = Vec::new();
    for entry in &entries {
        if !entry.is_metadata {
            new_notes.extend_from_slice(&elf[entry.offset..entry.offset + entry.total_size]);
        }
    }

    // Append merged metadata note
    let namesz = AMDGPU_NAME.len() as u32;
    let descsz = merged_desc.len() as u32;
    new_notes.extend_from_slice(&namesz.to_le_bytes());
    new_notes.extend_from_slice(&descsz.to_le_bytes());
    new_notes.extend_from_slice(&NT_AMDGPU_METADATA.to_le_bytes());
    new_notes.extend_from_slice(AMDGPU_NAME);
    let name_pad = pad4(AMDGPU_NAME.len()) - AMDGPU_NAME.len();
    for _ in 0..name_pad {
        new_notes.push(0);
    }
    new_notes.extend_from_slice(&merged_desc);
    let desc_pad = pad4(merged_desc.len()) - merged_desc.len();
    for _ in 0..desc_pad {
        new_notes.push(0);
    }

    if new_notes.len() > note_size {
        return Err(format!(
            "Merged notes ({} bytes) exceed original note section ({} bytes)",
            new_notes.len(),
            note_size
        ));
    }

    // Write merged notes into .note region, zero-pad remainder
    elf[note_offset..note_offset + new_notes.len()].copy_from_slice(&new_notes);
    for b in &mut elf[note_offset + new_notes.len()..note_offset + note_size] {
        *b = 0;
    }

    // Update .note section header sh_size
    let new_note_size = new_notes.len() as u64;
    w_u64(elf, sh + 0x20, new_note_size);

    // Update PT_NOTE program header p_filesz and p_memsz
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        if ph + 0x30 > elf.len() {
            break;
        }
        let p_type = r_u32(elf, ph);
        if p_type == PT_NOTE {
            w_u64(elf, ph + 0x20, new_note_size);
            w_u64(elf, ph + 0x28, new_note_size);
            break;
        }
    }

    compile_log(&format!(
        "merge_amdgpu_metadata_notes: merged {} notes -> {} bytes (was {} bytes)",
        metadata_count,
        new_notes.len(),
        note_size
    ));

    Ok(())
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

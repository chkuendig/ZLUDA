use super::driver;
use crate::r#impl::function;
use crate::r#impl::{driver::GlobalState, function::Function};
use cuda_types::{cuda::*, dark_api::FatbinFileHeader};
use hip_runtime_sys::*;
use rustc_hash::FxHashMap;
use std::collections::hash_map;
use std::collections::HashMap;
use std::sync::Mutex;
use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    fs, mem,
    ops::ControlFlow,
};
use zluda_common::{CodeLibraryRef, CodeModuleRef, ZludaObject};

/// Global map from synthetic texture reference pointers to their device global variable addresses.
/// When .texref variables are compiled as .global .u64 in PTX, hipModuleGetTexRef can't find them.
/// We create synthetic textureReference structs and track which device global they correspond to.
/// After a texture object is created (via hipTexRefSetArray), we copy the handle to the device global
/// so the kernel can read it at runtime.
static SYNTHETIC_TEXREFS: std::sync::LazyLock<Mutex<HashMap<usize, hipDeviceptr_t>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Check if a texture reference is synthetic and get its device global address
pub(crate) fn get_synthetic_texref_global(tex_ref: *const textureReference) -> Option<hipDeviceptr_t> {
    SYNTHETIC_TEXREFS
        .lock()
        .ok()
        .and_then(|map| map.get(&(tex_ref as usize)).copied())
}

pub(crate) struct Module {
    pub(crate) base: hipModule_t,
    pub(crate) sm_version: u32,
    mutable: Mutex<ModuleMutable>,
}

struct ModuleMutable {
    functions: FxHashMap<CString, Box<Function>>,
}

impl ModuleMutable {
    pub(crate) fn get_function(
        &mut self,
        module: hipModule_t,
        sm_version: u32,
        name: CString,
    ) -> Result<&Function, CUerror> {
        Ok(match self.functions.entry(name) {
            hash_map::Entry::Occupied(entry) => &*entry.into_mut(),
            hash_map::Entry::Vacant(entry) => {
                let mut func_handle = unsafe { std::mem::zeroed() };
                unsafe { hipModuleGetFunction(&mut func_handle, module, entry.key().as_ptr()) }?;
                let func = Box::new(Function {
                    base: func_handle,
                    sm_version,
                });
                &*entry.insert(func)
            }
        })
    }
}

impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type Error = CUerror;
    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        unsafe { hipModuleUnload(self.base) }?;
        Ok(())
    }
}

impl Module {
    pub(crate) fn new(base: hipModule_t, sm_version: u32) -> Self {
        Self {
            base,
            sm_version,
            mutable: Mutex::new(ModuleMutable {
                functions: FxHashMap::default(),
            }),
        }
    }

    pub(crate) fn get_function<'a>(&'a self, name: &CStr) -> Result<&'static Function, CUerror> {
        let mut mutable = self.mutable.lock().map_err(|_| CUerror::UNKNOWN)?;
        mutable
            .get_function(self.base, self.sm_version, name.to_owned())
            .map(|f| unsafe { (f as *const Function).as_ref().unwrap() })
    }
}

fn get_best_ptx_and_compile(
    global_state: &GlobalState,
    image: CodeLibraryRef<'_>,
) -> Result<(hipModule_t, u32), CUerror> {
    super::driver::zluda_log("[ZLUDA]   get_best_ptx_and_compile: iterating modules");
    let mut ptx_modules = Vec::new();
    let mut module_count = 0u32;
    unsafe {
        CodeLibraryRef::iterate_modules(image, |_idx, module| {
            module_count += 1;
            match module {
                Ok(CodeModuleRef::Text(ptx)) => {
                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: Text PTX ({} bytes)", module_count, ptx.len()));
                    ptx_modules.push(Cow::Borrowed(ptx));
                }
                Ok(CodeModuleRef::File(file)) => {
                    let kind = if file.header.kind == FatbinFileHeader::HEADER_KIND_PTX { "PTX" } else { "non-PTX" };
                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: File kind={} sm={} payload_size={} compressed_size={} uncompressed_payload={} flags={:?}", module_count, kind, file.header.sm_version, file.header.payload_size, file.header.compressed_size, file.header.uncompressed_payload, file.header.flags));
                    if file.header.kind != FatbinFileHeader::HEADER_KIND_PTX {
                        return;
                    }
                    match file.get_or_decompress_content() {
                        Ok(text) => {
                            super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: decompressed OK, {} bytes", module_count, text.len()));
                            match cow_bytes_to_str(text) {
                                Some(text) => {
                                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: PTX text extracted ({} bytes), first 100 chars: {}", module_count, text.len(), &text[..text.len().min(100)]));
                                    // Dump PTX to file for analysis
                                    {
                                        use std::io::Write;
                                        let _ = std::fs::create_dir_all("/tmp/zluda_ptx_dump");
                                        static DUMP_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                                        let n = DUMP_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        if let Ok(mut f) = std::fs::File::create(format!("/tmp/zluda_ptx_dump/module_{}.ptx", n)) {
                                            let _ = f.write_all(text.as_bytes());
                                        }
                                    }
                                    ptx_modules.push(text);
                                }
                                None => {
                                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: cow_bytes_to_str FAILED (not valid UTF-8)", module_count));
                                }
                            }
                        }
                        Err(_) => {
                            super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: get_or_decompress_content FAILED", module_count));
                        }
                    }
                }
                _ => {
                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: other format (ELF/Archive/Error)", module_count));
                }
            }
        })
    };
    super::driver::zluda_log(&format!("[ZLUDA]   found {} total modules, {} PTX candidates", module_count, ptx_modules.len()));
    if ptx_modules.is_empty() {
        super::driver::zluda_log("[ZLUDA]   NO PTX modules found!");
    }
    let maybe_module = ptx_modules
        .iter()
        .rev() // TODO: actually sort by SM
        .try_fold(
            None,
            |acc: Option<(&Cow<'_, str>, ptx_parser::Module<'_>)>, src| {
                let maybe_ast = if cfg!(debug_assertions) {
                    ptx_parser::parse_module_checked(src)
                } else {
                    Ok(ptx_parser::parse_module_unchecked(src))
                };
                match maybe_ast {
                    Err(_) => {
                        super::driver::zluda_log("[ZLUDA]   PTX parse FAILED");
                        ControlFlow::Continue(acc)
                    }
                    Ok(ast) => {
                        super::driver::zluda_log(&format!("[ZLUDA]   PTX parsed OK, invalid_directives={}, directives={}", ast.invalid_directives, ast.directives.len()));
                        if ast.invalid_directives == 0 {
                            return ControlFlow::Break((src, ast));
                        } else {
                            ControlFlow::Continue(Some(match acc {
                                Some(best_known) => {
                                    if ast.invalid_directives < best_known.1.invalid_directives {
                                        (src, ast)
                                    } else {
                                        best_known
                                    }
                                }
                                None => (src, ast),
                            }))
                        }
                    }
                }
            },
        );
    let (text, module) = match maybe_module {
        ControlFlow::Break((ast, module)) | ControlFlow::Continue(Some((ast, module))) => {
            super::driver::zluda_log("[ZLUDA]   selected PTX module for compilation");
            // Dump each PTX module with a unique counter for offline debugging
            static PTX_DUMP_CTR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = PTX_DUMP_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dump_path = format!("/tmp/zluda_ptx_{:03}.ptx", n);
            let _ = std::fs::write(&dump_path, ast.as_bytes());
            super::driver::zluda_log(&format!("[ZLUDA]   dumped PTX to {}", dump_path));
            (Some(ast), module)
        }
        ControlFlow::Continue(None) => {
            super::driver::zluda_log("[ZLUDA]   no valid PTX module found, using empty module");
            if cfg!(debug_assertions) {
                return Err(CUerror::NO_BINARY_FOR_GPU);
            }
            (
                None,
                ptx_parser::Module {
                    ptx_version: (1, 0),
                    sm_version: 0,
                    directives: Vec::new(),
                    invalid_directives: usize::MAX,
                },
            )
        }
    };
    // TODO: get this information on initialization
    super::driver::zluda_log("[ZLUDA]   getting HIP properties");
    let hip_properties = get_hip_properties()?;
    let gcn_arch = get_gcn_arch(&hip_properties)?;
    super::driver::zluda_log(&format!("[ZLUDA]   gcn_arch={}", gcn_arch));
    let attributes = ExtraCacheAttributes {
        clock_rate: hip_properties.clockRate as u32,
        is_debug: cfg!(debug_assertions),
    };
    let mut cache_with_key = match (text, global_state.cache_path.as_ref()) {
        (Some(text), Some(p)) => (|| {
            let cache = zluda_cache::ModuleCache::open(p)?;
            let key = get_cache_key(gcn_arch, &text, &attributes)?;
            Some((cache, key))
        })(),
        _ => None,
    };
    let cached_binary = load_cached_binary(&mut cache_with_key);
    let from_cache = cached_binary.is_some();
    let (elf_module, sm_version) = cached_binary
        .ok_or(CUerror::UNKNOWN)
        .or_else(|_| {
            super::driver::zluda_log("[ZLUDA]   no cache hit, compiling PTX -> AMDGPU");
            compile_and_cache(gcn_arch, attributes, module, &mut cache_with_key)
        })?;
    super::driver::zluda_log(&format!("[ZLUDA]   ELF module ready ({} bytes, from_cache={})", elf_module.len(), from_cache));
    let mut hip_module = unsafe { mem::zeroed() };
    super::driver::zluda_log("[ZLUDA]   calling hipModuleLoadData");
    unsafe { hipModuleLoadData(&mut hip_module, elf_module.as_ptr().cast()) }?;
    super::driver::zluda_log("[ZLUDA]   hipModuleLoadData OK");
    // Write allocation bounds to module globals for scatter-write bounds checking.
    // The emit pass inserts branchless checks: if addr is outside [min, max],
    // redirect to __zluda_scatter_sink. We write current bounds here; the defaults
    // (min=0, max=u64::MAX) accept all addresses until the first allocation.
    unsafe {
        let alloc_min = super::memory::ALLOC_MIN.load(std::sync::atomic::Ordering::Relaxed);
        let alloc_max = super::memory::ALLOC_MAX.load(std::sync::atomic::Ordering::Relaxed);
        // Only write if we have valid bounds (at least one allocation has happened)
        if alloc_min < alloc_max {
            let mut min_dptr: hipDeviceptr_t = hipDeviceptr_t(std::ptr::null_mut());
            let mut min_size = 0usize;
            let mut max_dptr: hipDeviceptr_t = hipDeviceptr_t(std::ptr::null_mut());
            let mut max_size = 0usize;
            let r1 = hipModuleGetGlobal(
                &mut min_dptr, &mut min_size, hip_module,
                c"__zluda_alloc_min".as_ptr());
            let r2 = hipModuleGetGlobal(
                &mut max_dptr, &mut max_size, hip_module,
                c"__zluda_alloc_max".as_ptr());
            if r1.is_ok() && r2.is_ok() && min_size >= 8 && max_size >= 8 {
                let _ = hipMemcpyHtoD(min_dptr, &alloc_min as *const _ as *mut _, 8);
                let _ = hipMemcpyHtoD(max_dptr, &alloc_max as *const _ as *mut _, 8);
            }
        }
    }
    Ok((hip_module, sm_version))
}

fn cow_bytes_to_str<'a>(data: Cow<'a, [u8]>) -> Option<Cow<'a, str>> {
    match data {
        Cow::Borrowed(bytes) => std::str::from_utf8(bytes).map(Cow::Borrowed).ok(),
        Cow::Owned(bytes) => String::from_utf8(bytes).map(Cow::Owned).ok(),
    }
}

pub(crate) fn load_hip_module(library: CodeLibraryRef) -> Result<(hipModule_t, u32), CUerror> {
    let global_state = driver::global_state()?;
    get_best_ptx_and_compile(global_state, library)
}

#[derive(serde::Serialize)]
struct ExtraCacheAttributes {
    is_debug: bool,
    clock_rate: u32,
}

fn get_hip_properties<'a>() -> Result<hipDeviceProp_tR0600, CUerror> {
    let hip_dev = super::context::get_current_device()?;
    let mut props = unsafe { mem::zeroed() };
    unsafe { hipGetDevicePropertiesR0600(&mut props, hip_dev) }?;
    Ok(props)
}

fn get_gcn_arch<'a>(props: &'a hipDeviceProp_tR0600) -> Result<&'a str, CUerror> {
    let gcn_arch = unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) };
    gcn_arch.to_str().map_err(|_| CUerror::UNKNOWN)
}

fn get_cache_key<'a, 'b>(
    isa: &'a str,
    text: &str,
    attributes: &impl serde::Serialize,
) -> Option<zluda_cache::ModuleKey<'a>> {
    // Serialization here is deterministic. When marking a type with
    // #[derive(serde::Serialize)] the derived implementation will just write
    // fields in the order of their declaration. It's not explictly guaranteed
    // by serde, but it is the only sensible thing to do, so I feel safe
    // to rely on it
    let serialized_attributes = serde_json::to_string(attributes).ok()?;
    Some(zluda_cache::ModuleKey {
        hash: blake3::hash(text.as_bytes()).to_hex(),
        compiler_version: "builtin",
        zluda_version: env!("VERGEN_GIT_SHA"),
        device: isa,
        backend_key: serialized_attributes,
        last_access: zluda_cache::ModuleCache::time_now(),
    })
}

fn load_cached_binary(
    cache_with_key: &mut Option<(zluda_cache::ModuleCache, zluda_cache::ModuleKey)>,
) -> Option<(Vec<u8>, u32)> {
    let binary = cache_with_key
        .as_mut()
        .and_then(|(c, key)| c.get_module_binary(key))?;
    let sm_version = kernel_metadata::KernelMetadataV1::read_object(&binary)?.sm_version;
    Some((binary, sm_version))
}

fn compile_and_cache(
    gcn_arch: &str,
    attributes: ExtraCacheAttributes,
    ast: ptx_parser::Module,
    cache_with_key: &mut Option<(zluda_cache::ModuleCache, zluda_cache::ModuleKey)>,
) -> Result<(Vec<u8>, u32), CUerror> {
    let compile_start = std::time::Instant::now();
    super::driver::zluda_log("[ZLUDA]   ptx::to_llvm_module starting");
    let llvm_module = ptx::to_llvm_module(
        ast,
        ptx::Attributes {
            clock_rate: attributes.clock_rate,
        },
        |pass_name| {
            super::driver::zluda_log(&format!("[ZLUDA]     pass complete: {}", pass_name));
        },
    )
    .map_err(|e| {
        super::driver::zluda_log(&format!("[ZLUDA]   ptx::to_llvm_module FAILED: {}", e));
        CUerror::UNKNOWN
    })?;
    super::driver::zluda_log(&format!("[ZLUDA]   ptx::to_llvm_module OK in {:?}, starting llvm_zluda::compile", compile_start.elapsed()));
    let ptx_impl = llvm_module.linked_bitcode();
    let sm_version = llvm_module.metadata.sm_version;
    // Dump LLVM IR and assembly for debugging PTX compilation issues
    static COMPILE_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COMPILE_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Direct test: write a marker file to confirm this code path runs
    {
        let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
        let _ = std::fs::write(format!("/tmp/zluda_ir_dump/MARKER_{}", n), format!("compile #{}", n));
    }
    let hook = move |data: &Vec<u8>, name: String| {
        let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
        let path = format!("/tmp/zluda_ir_dump/module_{}_{}", n, name);
        super::driver::zluda_log_err(&format!("IR DUMP: {} ({} bytes) -> {}", name, data.len(), path));
        let _ = std::fs::write(&path, data);
    };
    let elf_module = llvm_zluda::compile(
        &llvm_module.context,
        gcn_arch,
        llvm_module.llvm_ir,
        ptx_impl,
        llvm_module.attributes_ir,
        llvm_module.metadata,
        Some(&hook),
    )
    .map_err(|e| {
        super::driver::zluda_log(&format!("[ZLUDA]   llvm_zluda::compile FAILED: {}", e));
        CUerror::UNKNOWN
    })?;
    super::driver::zluda_log(&format!("[ZLUDA]   llvm_zluda::compile OK in {:?}, {} bytes ELF", compile_start.elapsed(), elf_module.len()));
    if let Some((cache, key)) = cache_with_key {
        key.last_access = zluda_cache::ModuleCache::time_now();
        cache.insert_module(key, &elf_module);
    }
    Ok((elf_module, sm_version))
}

#[allow(dead_code)]
fn compile_in_subprocess(
    gcn_arch: &str,
    llvm_module: ptx::Module,
) -> Result<Vec<u8>, CUerror> {
    use std::io::Write;
    // Create a temp file for the output
    let output_path = format!("/tmp/zluda_compile_{}.elf", std::process::id());
    let ptx_impl = llvm_module.linked_bitcode();

    // Use fork to compile in a child process
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        super::driver::zluda_log("[ZLUDA]   fork() failed");
        return Err(CUerror::UNKNOWN);
    }
    if pid == 0 {
        // Child process — do the compilation
        let result = llvm_zluda::compile(
            &llvm_module.context,
            gcn_arch,
            llvm_module.llvm_ir,
            ptx_impl,
            llvm_module.attributes_ir,
            llvm_module.metadata,
            None,
        );
        match result {
            Ok(elf_module) => {
                if let Ok(mut f) = std::fs::File::create(&output_path) {
                    let _ = f.write_all(&elf_module);
                }
                unsafe { libc::_exit(0) };
            }
            Err(e) => {
                let _ = std::fs::write(&output_path, format!("ERROR: {:?}", e));
                unsafe { libc::_exit(1) };
            }
        }
    }
    // Parent process — wait for child
    super::driver::zluda_log(&format!("[ZLUDA]   waiting for compile subprocess pid={}", pid));
    let mut status: i32 = 0;
    let wait_result = unsafe { libc::waitpid(pid, &mut status, 0) };
    if wait_result < 0 {
        super::driver::zluda_log("[ZLUDA]   waitpid failed");
        let _ = std::fs::remove_file(&output_path);
        return Err(CUerror::UNKNOWN);
    }
    if !libc::WIFEXITED(status) || libc::WEXITSTATUS(status) != 0 {
        let exit_code = if libc::WIFEXITED(status) { libc::WEXITSTATUS(status) } else { -1 };
        let signal = if libc::WIFSIGNALED(status) { libc::WTERMSIG(status) } else { 0 };
        super::driver::zluda_log(&format!("[ZLUDA]   compile subprocess failed: exit={}, signal={}", exit_code, signal));
        // Try to read error message
        if let Ok(msg) = std::fs::read_to_string(&output_path) {
            super::driver::zluda_log(&format!("[ZLUDA]   compile error: {}", msg));
        }
        let _ = std::fs::remove_file(&output_path);
        return Err(CUerror::UNKNOWN);
    }
    // Read the ELF output
    let elf_module = std::fs::read(&output_path).map_err(|e| {
        super::driver::zluda_log(&format!("[ZLUDA]   failed to read compiled ELF: {:?}", e));
        CUerror::UNKNOWN
    })?;
    let _ = std::fs::remove_file(&output_path);
    if elf_module.starts_with(b"ERROR:") {
        super::driver::zluda_log(&format!("[ZLUDA]   compile error: {}", String::from_utf8_lossy(&elf_module)));
        return Err(CUerror::UNKNOWN);
    }
    Ok(elf_module)
}

pub(crate) fn load(module: &mut CUmodule, fname: &CStr) -> CUresult {
    let mut image = fs::read(fname.to_str().map_err(|_| CUerror::INVALID_VALUE)?)
        .map_err(|_| CUerror::INVALID_VALUE)?;
    // Null-terminate the image, in case it's text
    image.push(0);
    let library = unsafe { CodeLibraryRef::try_load(image.as_ptr() as *const std::ffi::c_void) }
        .map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
    let (hip_module, sm_version) = load_hip_module(library)?;
    *module = Module::new(hip_module, sm_version).wrap();
    Ok(())
}

pub(crate) fn load_data(module: &mut CUmodule, image: &std::ffi::c_void) -> CUresult {
    super::driver::zluda_log("[ZLUDA] cuModuleLoadData: entering");
    let library =
        unsafe { CodeLibraryRef::try_load(image) }.map_err(|e| {
            super::driver::zluda_log(&format!("[ZLUDA] cuModuleLoadData -> NO_BINARY_FOR_GPU (try_load failed: {:?})", e));
            CUerror::NO_BINARY_FOR_GPU
        })?;
    let format = match library {
        CodeLibraryRef::FatbincWrapper(_) => "FatbincWrapper",
        CodeLibraryRef::FatbinHeader(_) => "FatbinHeader",
        CodeLibraryRef::Text(_) => "Text",
        CodeLibraryRef::Elf(_) => "Elf",
        CodeLibraryRef::Archive(_) => "Archive",
    };
    super::driver::zluda_log(&format!("[ZLUDA] cuModuleLoadData: try_load OK, format={}", format));
    let (hip_module, sm_version) = load_hip_module(library)?;
    *module = Module::new(hip_module, sm_version).wrap();
    super::driver::zluda_log("[ZLUDA] cuModuleLoadData -> SUCCESS");
    Ok(())
}

pub(crate) fn load_data_ex(
    module: &mut CUmodule,
    image: &std::ffi::c_void,
    _num_options: ::std::os::raw::c_uint,
    _options: Option<&mut CUjit_option_enum>,
    _option_values: Option<&mut *mut ::core::ffi::c_void>,
) -> CUresult {
    load_data(module, image)
}

pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    zluda_common::drop_checked::<Module>(hmod)
}

pub(crate) fn get_function(
    hfunc: &mut &function::Function,
    module: &Module,
    name: &CStr,
) -> CUresult {
    let name_str = name.to_string_lossy();
    super::driver::zluda_log(&format!("[ZLUDA] cuModuleGetFunction('{}')", name_str));
    let func = module.get_function(name)?;
    super::function::register_name(func.base, &name_str);
    super::driver::zluda_log_err(&format!(
        "cuModuleGetFunction('{}') -> func={:p}", name_str, func.base.0));
    *hfunc = func;
    Ok(())
}

pub(crate) fn get_global_v2(
    dptr: *mut hipDeviceptr_t,
    bytes: *mut usize,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> hipError_t {
    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
    let result = unsafe { hipModuleGetGlobal(dptr, bytes, hmod.base, name) };
    let dptr_val = unsafe { (*dptr).0 };
    let bytes_val = if bytes.is_null() { 0 } else { unsafe { *bytes } };
    let addr_space = if (dptr_val as u64) < 0x7f0000000000 { "HOST" } else { "DEV" };
    // Track module global address in alloc range
    if result.is_ok() {
        super::memory::update_alloc_range(dptr_val as u64, bytes_val.max(256) as u64);
    }
    super::driver::zluda_log(&format!(
        "[ZLUDA] cuModuleGetGlobal('{}') -> {:?} dptr={:p} bytes={} [{}]",
        name_str, result, dptr_val, bytes_val, addr_space));
    result
}

pub(crate) fn get_tex_ref(
    p_tex_ref: &mut CUtexref,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
    let mut tex_ref: *mut textureReference = std::ptr::null_mut();
    let result = unsafe { hipModuleGetTexRef(&mut tex_ref, hmod.base, name) };
    match result {
        Ok(()) => {
            super::driver::zluda_log(&format!("[ZLUDA] cuModuleGetTexRef('{}') -> {:p}", name_str, tex_ref));
            *p_tex_ref = tex_ref as CUtexref;
            Ok(())
        }
        Err(e) => {
            // .texref in PTX is compiled as a .global .u64 variable, not a HIP texture reference.
            // Try to find it as a global variable and create a synthetic texture reference.
            let mut dptr = hipDeviceptr_t(std::ptr::null_mut());
            let mut bytes = 0usize;
            let global_result = unsafe { hipModuleGetGlobal(&mut dptr, &mut bytes, hmod.base, name) };
            if global_result.is_ok() {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetTexRef('{}') -> found as global variable at {:p} ({} bytes), creating synthetic texref",
                    name_str, dptr.0, bytes
                ));
                // Allocate a textureReference on the heap and return it.
                // The device-side global variable holds the texture object handle (u64).
                // When cuTexRefSetArray/cuTexObjectCreate is called, we'll write the
                // texture object handle to this device global.
                let synth_ref = Box::new(SyntheticTexRef {
                    tex_ref: unsafe { mem::zeroed::<textureReference>() },
                    device_global: dptr,
                });
                let tex_ref_ptr = &synth_ref.tex_ref as *const textureReference as *mut textureReference;
                // Register in global map so tex.rs can find the device global
                if let Ok(mut map) = SYNTHETIC_TEXREFS.lock() {
                    map.insert(tex_ref_ptr as usize, dptr);
                }
                // Leak the Box — the texture reference lives as long as the module
                Box::leak(synth_ref);
                *p_tex_ref = tex_ref_ptr as CUtexref;
                Ok(())
            } else {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetTexRef('{}') -> NOT FOUND (hip={:?}, global={:?})",
                    name_str, e, global_result
                ));
                Err(e.into())
            }
        }
    }
}

/// A synthetic texture reference for .texref variables compiled as .global .u64
#[repr(C)]
struct SyntheticTexRef {
    tex_ref: textureReference,
    device_global: hipDeviceptr_t,
}

pub(crate) fn get_surf_ref(
    p_surf_ref: &mut cuda_types::cuda::CUsurfref,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    let name_str = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
    // HIP doesn't have a separate hipModuleGetSurfRef - use hipModuleGetTexRef
    // since surface and texture references share the same underlying textureReference struct
    let mut tex_ref: *mut textureReference = std::ptr::null_mut();
    let result = unsafe { hipModuleGetTexRef(&mut tex_ref, hmod.base, name) };
    match result {
        Ok(()) => {
            super::driver::zluda_log(&format!("[ZLUDA] cuModuleGetSurfRef('{}') -> {:p}", name_str, tex_ref));
            *p_surf_ref = tex_ref as cuda_types::cuda::CUsurfref;
            Ok(())
        }
        Err(_) => {
            // .surfref compiled as .global .u64 — same approach as .texref
            let mut dptr = hipDeviceptr_t(std::ptr::null_mut());
            let mut bytes = 0usize;
            let global_result = unsafe { hipModuleGetGlobal(&mut dptr, &mut bytes, hmod.base, name) };
            if global_result.is_ok() {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetSurfRef('{}') -> found as global at {:p}, creating synthetic surfref",
                    name_str, dptr.0
                ));
                let synth_ref = Box::new(SyntheticTexRef {
                    tex_ref: unsafe { mem::zeroed::<textureReference>() },
                    device_global: dptr,
                });
                let tex_ref_ptr = &synth_ref.tex_ref as *const textureReference as *mut textureReference;
                if let Ok(mut map) = SYNTHETIC_TEXREFS.lock() {
                    map.insert(tex_ref_ptr as usize, dptr);
                }
                Box::leak(synth_ref);
                *p_surf_ref = tex_ref_ptr as cuda_types::cuda::CUsurfref;
                Ok(())
            } else {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetSurfRef('{}') -> NOT FOUND",
                    name_str
                ));
                Err(result.unwrap_err().into())
            }
        }
    }
}

pub(crate) fn get_loading_mode(mode: &mut cuda_types::cuda::CUmoduleLoadingMode) -> CUresult {
    *mode = cuda_types::cuda::CUmoduleLoadingMode::CU_MODULE_LAZY_LOADING;
    Ok(())
}

pub(crate) fn load_fat_binary(module: &mut CUmodule, image: &std::ffi::c_void) -> CUresult {
    load_data(module, image)
}

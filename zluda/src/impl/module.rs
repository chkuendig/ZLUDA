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
pub(crate) fn get_synthetic_texref_global(
    tex_ref: *const textureReference,
) -> Option<hipDeviceptr_t> {
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
                    super::driver::zluda_log(&format!(
                        "[ZLUDA]   module[{}]: Text PTX ({} bytes)",
                        module_count,
                        ptx.len()
                    ));
                    ptx_modules.push(Cow::Borrowed(ptx));
                }
                Ok(CodeModuleRef::File(file)) => {
                    let kind = if file.header.kind == FatbinFileHeader::HEADER_KIND_PTX {
                        "PTX"
                    } else {
                        "non-PTX"
                    };
                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: File kind={} sm={} payload_size={} compressed_size={} uncompressed_payload={} flags={:?}", module_count, kind, file.header.sm_version, file.header.payload_size, file.header.compressed_size, file.header.uncompressed_payload, file.header.flags));
                    if file.header.kind != FatbinFileHeader::HEADER_KIND_PTX {
                        return;
                    }
                    match file.get_or_decompress_content() {
                        Ok(text) => {
                            super::driver::zluda_log(&format!(
                                "[ZLUDA]   module[{}]: decompressed OK, {} bytes",
                                module_count,
                                text.len()
                            ));
                            match cow_bytes_to_str(text) {
                                Some(text) => {
                                    super::driver::zluda_log(&format!("[ZLUDA]   module[{}]: PTX text extracted ({} bytes), first 100 chars: {}", module_count, text.len(), &text[..text.len().min(100)]));
                                    // Dump PTX to file for analysis
                                    {
                                        use std::io::Write;
                                        let _ = std::fs::create_dir_all("/tmp/zluda_ptx_dump");
                                        static DUMP_COUNT: std::sync::atomic::AtomicU32 =
                                            std::sync::atomic::AtomicU32::new(0);
                                        let n = DUMP_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        if let Ok(mut f) = std::fs::File::create(format!(
                                            "/tmp/zluda_ptx_dump/module_{}.ptx",
                                            n
                                        )) {
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
                            super::driver::zluda_log(&format!(
                                "[ZLUDA]   module[{}]: get_or_decompress_content FAILED",
                                module_count
                            ));
                        }
                    }
                }
                _ => {
                    super::driver::zluda_log(&format!(
                        "[ZLUDA]   module[{}]: other format (ELF/Archive/Error)",
                        module_count
                    ));
                }
            }
        })
    };
    super::driver::zluda_log(&format!(
        "[ZLUDA]   found {} total modules, {} PTX candidates",
        module_count,
        ptx_modules.len()
    ));
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
                        super::driver::zluda_log(&format!(
                            "[ZLUDA]   PTX parsed OK, invalid_directives={}, directives={}",
                            ast.invalid_directives,
                            ast.directives.len()
                        ));
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
            static PTX_DUMP_CTR: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
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
    let (elf_module, sm_version) = cached_binary.ok_or(CUerror::UNKNOWN).or_else(|_| {
        super::driver::zluda_log("[ZLUDA]   no cache hit, compiling PTX -> AMDGPU");
        compile_and_cache(gcn_arch, attributes, module, &mut cache_with_key)
    })?;
    super::driver::zluda_log(&format!(
        "[ZLUDA]   ELF module ready ({} bytes, from_cache={})",
        elf_module.len(),
        from_cache
    ));
    maybe_dump_radixsort_downsweep_module(
        text.as_ref().map(|text| text.as_ref()),
        &elf_module,
        from_cache,
    );
    let mut hip_module = unsafe { mem::zeroed() };
    super::driver::zluda_log("[ZLUDA]   calling hipModuleLoadData");
    unsafe { hipModuleLoadData(&mut hip_module, elf_module.as_ptr().cast()) }?;
    super::driver::zluda_log("[ZLUDA]   hipModuleLoadData OK");
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

fn should_dump_radixsort_downsweep_module(text: &str) -> bool {
    text.contains("RadixSortDownsweepKernel") && text.contains("GridEvenShare")
}

fn maybe_dump_radixsort_downsweep_module(text: Option<&str>, elf_module: &[u8], from_cache: bool) {
    let Some(text) = text else {
        return;
    };
    if !should_dump_radixsort_downsweep_module(text) {
        return;
    }
    static DID_DUMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if DID_DUMP.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let dump_dir = "/tmp/zluda_live_sort_module";
    let _ = fs::create_dir_all(dump_dir);
    let ptx_path = format!("{dump_dir}/radixsort_downsweep.ptx");
    let elf_path = format!("{dump_dir}/radixsort_downsweep.elf");
    let meta_path = format!("{dump_dir}/meta.txt");
    let ptx_hash = blake3::hash(text.as_bytes());
    let elf_hash = blake3::hash(elf_module);
    let _ = fs::write(&ptx_path, text);
    let _ = fs::write(&elf_path, elf_module);
    let _ = fs::write(
        &meta_path,
        format!(
            "from_cache={from_cache}\nptx_bytes={}\nelf_bytes={}\nptx_hash={}\nelf_hash={}\n",
            text.len(),
            elf_module.len(),
            ptx_hash,
            elf_hash
        ),
    );
    super::driver::zluda_log_err(&format!(
        "LIVE SORT MODULE DUMPED: ptx={} elf={} meta={}",
        ptx_path, elf_path, meta_path
    ));
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
    super::driver::zluda_log(&format!(
        "[ZLUDA]   ptx::to_llvm_module OK in {:?}, starting llvm_zluda::compile",
        compile_start.elapsed()
    ));
    let ptx_impl = llvm_module.linked_bitcode();
    let sm_version = llvm_module.metadata.sm_version;
    // Dump LLVM IR and assembly for debugging PTX compilation issues
    static COMPILE_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COMPILE_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Direct test: write a marker file to confirm this code path runs
    {
        let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
        let _ = std::fs::write(
            format!("/tmp/zluda_ir_dump/MARKER_{}", n),
            format!("compile #{}", n),
        );
    }
    // Compile in a forked subprocess to survive LLVM abort() crashes.
    // The AMDGPU backend can crash in loadRegFromStackSlot for very large
    // kernels (e.g. BasicIOS_simulateKernel_templ). A subprocess isolates
    // the crash so the game can continue without the affected module.
    let elf_module = compile_in_subprocess(gcn_arch, llvm_module, n).map_err(|e| {
        super::driver::zluda_log(&format!("[ZLUDA]   compile_in_subprocess FAILED: {:?}", e));
        e
    })?;
    super::driver::zluda_log(&format!(
        "[ZLUDA]   llvm_zluda::compile OK in {:?}, {} bytes ELF",
        compile_start.elapsed(),
        elf_module.len()
    ));
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
    compile_n: u32,
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
        let hook = move |data: &Vec<u8>, name: String| dump_compile_artifact(compile_n, data, name);
        let result = llvm_zluda::compile(
            &llvm_module.context,
            gcn_arch,
            llvm_module.llvm_ir,
            ptx_impl,
            llvm_module.attributes_ir,
            llvm_module.metadata,
            Some(&hook),
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
    super::driver::zluda_log(&format!(
        "[ZLUDA]   waiting for compile subprocess pid={}",
        pid
    ));
    let mut status: i32 = 0;
    let wait_result = unsafe { libc::waitpid(pid, &mut status, 0) };
    if wait_result < 0 {
        super::driver::zluda_log("[ZLUDA]   waitpid failed");
        let _ = std::fs::remove_file(&output_path);
        return Err(CUerror::UNKNOWN);
    }
    if !libc::WIFEXITED(status) || libc::WEXITSTATUS(status) != 0 {
        let exit_code = if libc::WIFEXITED(status) {
            libc::WEXITSTATUS(status)
        } else {
            -1
        };
        let signal = if libc::WIFSIGNALED(status) {
            libc::WTERMSIG(status)
        } else {
            0
        };
        super::driver::zluda_log(&format!(
            "[ZLUDA]   compile subprocess failed: exit={}, signal={}",
            exit_code, signal
        ));
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
        super::driver::zluda_log(&format!(
            "[ZLUDA]   compile error: {}",
            String::from_utf8_lossy(&elf_module)
        ));
        return Err(CUerror::UNKNOWN);
    }
    Ok(elf_module)
}

fn dump_compile_artifact(n: u32, data: &Vec<u8>, name: String) {
    let _ = std::fs::create_dir_all("/tmp/zluda_ir_dump");
    let path = format!("/tmp/zluda_ir_dump/module_{}_{}", n, name);
    super::driver::zluda_log_err(&format!(
        "IR DUMP: {} ({} bytes) -> {}",
        name,
        data.len(),
        path
    ));
    let _ = std::fs::write(&path, data);
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
    let library = unsafe { CodeLibraryRef::try_load(image) }.map_err(|e| {
        super::driver::zluda_log(&format!(
            "[ZLUDA] cuModuleLoadData -> NO_BINARY_FOR_GPU (try_load failed: {:?})",
            e
        ));
        CUerror::NO_BINARY_FOR_GPU
    })?;
    let format = match library {
        CodeLibraryRef::FatbincWrapper(_) => "FatbincWrapper",
        CodeLibraryRef::FatbinHeader(_) => "FatbinHeader",
        CodeLibraryRef::Text(_) => "Text",
        CodeLibraryRef::Elf(_) => "Elf",
        CodeLibraryRef::Archive(_) => "Archive",
    };
    super::driver::zluda_log(&format!(
        "[ZLUDA] cuModuleLoadData: try_load OK, format={}",
        format
    ));
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
        "cuModuleGetFunction('{}') -> func={:p}",
        name_str, func.base.0
    ));
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
    // lsh hipfix item 7: hipModuleGetGlobal called with bytes==NULL leaves the
    // global "unusable" — subsequent operations against it return
    // CUDA_ERROR_INVALID_VALUE or silently corrupt. Always pass a real local
    // out-pointer and copy back only if the caller asked for it.
    let mut local_bytes: usize = 0;
    let result = unsafe { hipModuleGetGlobal(dptr, &mut local_bytes, hmod.base, name) };
    if !bytes.is_null() {
        unsafe { *bytes = local_bytes };
    }
    let dptr_val = unsafe { (*dptr).0 };
    let addr_space = if (dptr_val as u64) < 0x7f0000000000 {
        "HOST"
    } else {
        "DEV"
    };
    // Track module global address in alloc range
    if result.is_ok() {
        super::memory::update_alloc_range(dptr_val as u64, local_bytes.max(256) as u64);
    }
    super::driver::zluda_log(&format!(
        "[ZLUDA] cuModuleGetGlobal('{}') -> {:?} dptr={:p} bytes={} [{}]",
        name_str, result, dptr_val, local_bytes, addr_space
    ));
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
            super::driver::zluda_log(&format!(
                "[ZLUDA] cuModuleGetTexRef('{}') -> {:p}",
                name_str, tex_ref
            ));
            *p_tex_ref = tex_ref as CUtexref;
            Ok(())
        }
        Err(e) => {
            // .texref in PTX is compiled as a .global .u64 variable, not a HIP texture reference.
            // Try to find it as a global variable and create a synthetic texture reference.
            let mut dptr = hipDeviceptr_t(std::ptr::null_mut());
            let mut bytes = 0usize;
            let global_result =
                unsafe { hipModuleGetGlobal(&mut dptr, &mut bytes, hmod.base, name) };
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
                let tex_ref_ptr =
                    &synth_ref.tex_ref as *const textureReference as *mut textureReference;
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
            super::driver::zluda_log(&format!(
                "[ZLUDA] cuModuleGetSurfRef('{}') -> {:p}",
                name_str, tex_ref
            ));
            *p_surf_ref = tex_ref as cuda_types::cuda::CUsurfref;
            Ok(())
        }
        Err(_) => {
            // .surfref compiled as .global .u64 — same approach as .texref
            let mut dptr = hipDeviceptr_t(std::ptr::null_mut());
            let mut bytes = 0usize;
            let global_result =
                unsafe { hipModuleGetGlobal(&mut dptr, &mut bytes, hmod.base, name) };
            if global_result.is_ok() {
                super::driver::zluda_log(&format!(
                    "[ZLUDA] cuModuleGetSurfRef('{}') -> found as global at {:p}, creating synthetic surfref",
                    name_str, dptr.0
                ));
                let synth_ref = Box::new(SyntheticTexRef {
                    tex_ref: unsafe { mem::zeroed::<textureReference>() },
                    device_global: dptr,
                });
                let tex_ref_ptr =
                    &synth_ref.tex_ref as *const textureReference as *mut textureReference;
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

#[cfg(test)]
mod tests {
    //! Regression tests for `cuModuleGetGlobal_v2`.
    //!
    //! lsh hipfix item 7: hipModuleGetGlobal called with `bytes == NULL`
    //! leaves the global "unusable" — subsequent operations against the
    //! returned device pointer return CUDA_ERROR_INVALID_VALUE or silently
    //! corrupt. Our wrapper always passes a real local out-pointer to HIP
    //! and copies back only if the caller asked for it.
    //!
    //! These tests go through the full Zluda extern "C" entry points (via
    //! `crate::cuXxx`) so they exercise the same code path the game hits.
    //! Skipped cleanly when no AMD GPU is available.
    use cuda_types::cuda::*;
    use std::ffi::CStr;
    use std::mem;
    use std::ptr;

    /// Tiny PTX with a single 4-byte global. We initialize it to a known
    /// non-zero value so a follow-up DtoH copy can verify the global is
    /// actually usable after the NULL-bytes get_global call.
    const PTX_WITH_GLOBAL: &CStr = c"
.version 7.0
.target sm_60
.address_size 64

.visible .global .align 4 .u32 my_global = 305419896;
";

    /// Helper: minimal "init + ctx + module load" boilerplate. Returns the
    /// loaded module handle, or `None` if the GPU/runtime is unavailable
    /// (test should skip cleanly in that case).
    unsafe fn try_load_test_module() -> Option<CUmodule> {
        if crate::cuInit(0).is_err() {
            return None;
        }
        let mut ctx: CUcontext = mem::zeroed();
        if crate::cuCtxCreate_v2(&mut ctx, 0, 0 as CUdevice).is_err() {
            return None;
        }
        let mut module: CUmodule = mem::zeroed();
        if crate::cuModuleLoadData(&mut module, PTX_WITH_GLOBAL.as_ptr().cast()).is_err() {
            return None;
        }
        Some(module)
    }

    /// REGRESSION: calling cuModuleGetGlobal_v2 with bytes=NULL must leave
    /// the global usable for subsequent device-memory operations. Without
    /// the lsh hipfix item 7 workaround in `get_global_v2`, hipModuleGetGlobal
    /// internally marks the global as broken and the follow-up DtoH copy
    /// fails or returns garbage.
    #[test]
    fn get_global_with_null_bytes_keeps_global_usable() {
        unsafe {
            let module = match try_load_test_module() {
                Some(m) => m,
                None => return, // No GPU / runtime — skip
            };

            // Call cuModuleGetGlobal_v2 with bytes = NULL — the bug case.
            let mut dptr: CUdeviceptr = mem::zeroed();
            let result = crate::cuModuleGetGlobal_v2(
                &mut dptr,
                ptr::null_mut(),
                module,
                c"my_global".as_ptr(),
            );
            assert!(
                result.is_ok(),
                "cuModuleGetGlobal_v2 with bytes=NULL must succeed, got {:?}",
                result
            );
            assert!(
                !dptr.0.is_null(),
                "returned device pointer must not be null"
            );

            // The actual regression check: read the global back via DtoH and
            // verify the value matches what the PTX initialized it to. If
            // hipModuleGetGlobal corrupted the global (the lsh-documented bug),
            // this readback returns 0 or fails outright.
            let mut value: u32 = 0;
            let copy_result =
                crate::cuMemcpyDtoH_v2(&mut value as *mut u32 as *mut std::ffi::c_void, dptr, 4);
            assert!(
                copy_result.is_ok(),
                "cuMemcpyDtoH_v2 from a NULL-bytes-fetched global failed: {:?} \
                 (lsh hipfix item 7 regression)",
                copy_result
            );
            assert_eq!(
                value, 305419896,
                "global value mismatch after NULL-bytes fetch \
                 (lsh hipfix item 7 regression — global was silently corrupted)"
            );
        }
    }

    /// Sanity: when the caller DOES pass a real `bytes` out-pointer, the
    /// wrapper must still write the size into it. This guards against the
    /// obvious mistake of dropping the write-back when adding the local
    /// out-pointer plumbing.
    #[test]
    fn get_global_with_real_bytes_pointer_writes_size() {
        unsafe {
            let module = match try_load_test_module() {
                Some(m) => m,
                None => return,
            };

            let mut dptr: CUdeviceptr = mem::zeroed();
            let mut bytes: usize = 0;
            let result =
                crate::cuModuleGetGlobal_v2(&mut dptr, &mut bytes, module, c"my_global".as_ptr());
            assert!(result.is_ok(), "cuModuleGetGlobal_v2 failed: {:?}", result);
            assert!(!dptr.0.is_null());
            assert_eq!(
                bytes, 4,
                "bytes out-pointer must be written with the global's size (4 for u32)"
            );
        }
    }

    #[test]
    fn detects_radixsort_downsweep_modules() {
        assert!(super::should_dump_radixsort_downsweep_module(
            ".visible .entry _ZN3cub24RadixSortDownsweepKernel(...)\nGridEvenShare"
        ));
        assert!(!super::should_dump_radixsort_downsweep_module(
            ".visible .entry SomeOtherKernel(...)\nGridEvenShare"
        ));
        assert!(!super::should_dump_radixsort_downsweep_module(
            ".visible .entry _ZN3cub24RadixSortDownsweepKernel(...)"
        ));
    }
}

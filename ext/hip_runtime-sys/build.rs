fn main() {
    if cfg!(windows) {
        println!("cargo:rustc-link-arg=/DELAYLOAD:amdhip64_7.dll");
    } else {
        // Check LIBRARY_PATH (set during build), /opt/rocm, and ext/rocm/lib
        let mut found = false;
        let library_path = std::env::var("LIBRARY_PATH").unwrap_or_default();
        let ext_rocm = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../rocm/lib");
        let candidates: Vec<String> = library_path
            .split(':')
            .map(String::from)
            .chain(std::iter::once("/opt/rocm/lib".to_string()))
            .chain(std::iter::once(ext_rocm.to_string_lossy().to_string()))
            .collect();
        for dir in &candidates {
            if !dir.is_empty() && std::path::Path::new(dir).join("libamdhip64.so").exists() {
                println!("cargo:rustc-link-lib=dylib=amdhip64");
                println!("cargo:rustc-link-search=native={}", dir);
                found = true;
                break;
            }
        }
        if !found {
            eprintln!("cargo:warning=ROCm not found, skipping HIP linking");
        }
    }

    // Parse HIP version from hip_version.h and expose as DEP_HIP_RUNTIME_SYS_*
    // env vars to downstream crates (standard Rust -sys crate pattern).
    // Any crate with `hip_runtime-sys` in its dep tree gets these automatically.
    let header = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rocm/include/hip/hip_version.h");
    println!("cargo:rerun-if-changed={}", header.display());
    if let Ok(contents) = std::fs::read_to_string(&header) {
        for line in contents.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("#define HIP_VERSION_MAJOR ") {
                println!("cargo:version_major={}", val.trim());
                println!("cargo:rustc-env=HIP_VERSION_MAJOR={}", val.trim());
            }
            if let Some(val) = line.strip_prefix("#define HIP_VERSION_MINOR ") {
                println!("cargo:version_minor={}", val.trim());
                println!("cargo:rustc-env=HIP_VERSION_MINOR={}", val.trim());
            }
            if let Some(val) = line.strip_prefix("#define HIP_VERSION_PATCH ") {
                println!("cargo:version_patch={}", val.trim());
                println!("cargo:rustc-env=HIP_VERSION_PATCH={}", val.trim());
            }
        }
    }
}

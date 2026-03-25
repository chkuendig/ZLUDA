fn main() {
    if cfg!(windows) {
        println!("cargo:rustc-link-arg=/DELAYLOAD:amdhip64_7.dll");
    } else if std::path::Path::new("/opt/rocm/lib/libamdhip64.so").exists() {
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    } else {
        // No ROCm installed — skip linking (e.g. NVIDIA-only machine for reference tests)
        eprintln!("cargo:warning=ROCm not found, skipping HIP linking");
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

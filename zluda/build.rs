use vergen_gix::{Emitter, Gix};

fn main() {
    if cfg!(windows) {
        println!("cargo:rustc-link-arg=delayimp.lib");
        println!("cargo:rustc-link-arg=/DELAYLOAD:amdhip64_7.dll");
        let dll_path = "bin/nvcudart_hybrid64.dll";
        println!("cargo:rerun-if-changed={}", dll_path);
        check_lfs_file(dll_path);
    }
    let git = Gix::builder().sha(false).build();
    Emitter::default()
        .add_instructions(&git)
        .unwrap()
        .emit()
        .unwrap();

    // Forward HIP compile-time version to our crate for the runtime mismatch check.
    // Try DEP_ metadata from hip_runtime-sys first, fall back to parsing the header directly.
    let hip_major = std::env::var("DEP_AMDHIP64_VERSION_MAJOR")
        .or_else(|_| std::env::var("DEP_HIP_RUNTIME_SYS_VERSION_MAJOR"))
        .ok();
    let hip_minor = std::env::var("DEP_AMDHIP64_VERSION_MINOR")
        .or_else(|_| std::env::var("DEP_HIP_RUNTIME_SYS_VERSION_MINOR"))
        .ok();

    // Fallback: parse hip_version.h directly
    let (hip_major, hip_minor) = match (hip_major, hip_minor) {
        (Some(maj), Some(min)) => (maj, min),
        _ => {
            let header = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../ext/rocm/include/hip/hip_version.h");
            println!("cargo:rerun-if-changed={}", header.display());
            let mut maj = String::from("0");
            let mut min = String::from("0");
            if let Ok(contents) = std::fs::read_to_string(&header) {
                for line in contents.lines() {
                    let line = line.trim();
                    if let Some(v) = line.strip_prefix("#define HIP_VERSION_MAJOR ") {
                        maj = v.trim().to_string();
                    }
                    if let Some(v) = line.strip_prefix("#define HIP_VERSION_MINOR ") {
                        min = v.trim().to_string();
                    }
                }
            }
            (maj, min)
        }
    };
    println!("cargo:rustc-env=HIP_VERSION_MAJOR={}", hip_major);
    println!("cargo:rustc-env=HIP_VERSION_MINOR={}", hip_minor);
}

fn check_lfs_file(bc_path: &str) {
    use std::fs::File;
    use std::io::Read;

    let mut magic = [0u8; 2];
    File::open(bc_path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", bc_path, e))
        .read_exact(&mut magic)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", bc_path, e));
    assert!(
        &magic == b"MZ",
        "{} is a git lfs stub and not the actual file. Run `git lfs pull` to fetch it",
        bc_path
    );
}

use vergen_gix::{Emitter, Gix};

fn main() {
    // SteamOS links an older glibc than the build box; an explicitly supplied
    // object provides versioned math forwarders for the release binary.
    println!("cargo:rerun-if-env-changed=ZLUDA_MATHCOMPAT_OBJECT");
    if let Some(object) = std::env::var_os("ZLUDA_MATHCOMPAT_OBJECT") {
        let object = std::path::PathBuf::from(object);
        assert!(object.is_file(), "math compatibility object is missing");
        println!("cargo:rerun-if-changed={}", object.display());
        println!("cargo:rustc-link-arg={}", object.display());
    }
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

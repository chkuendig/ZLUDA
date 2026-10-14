use std::env::VarError;

fn main() -> Result<(), VarError> {
    if std::path::Path::new("/opt/rocm/lib").exists() {
        println!("cargo:rustc-link-lib=dylib=rocm_smi64");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    }
    Ok(())
}

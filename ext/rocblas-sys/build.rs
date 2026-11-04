use std::env::VarError;

fn main() -> Result<(), VarError> {
    if !cfg!(windows) && std::path::Path::new("/opt/rocm/lib").exists() {
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    }
    Ok(())
}

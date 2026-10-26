use std::env::VarError;

fn main() -> Result<(), VarError> {
    // CUDA type conversions also use this crate without calling hipFFT.
    // Keep native linking for actual FFT consumers through the default feature.
    if !cfg!(windows) && cfg!(feature = "link") {
        println!("cargo:rustc-link-lib=dylib=hipfft");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    }
    Ok(())
}

fn main() {
    // Build scripts run on the host, so check the target, not cfg!(windows):
    // a Linux-hosted MinGW cross build must not link the Linux HIP library.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "windows" {
        if target_env == "msvc" {
            println!("cargo:rustc-link-arg=/DELAYLOAD:amdhip64_7.dll");
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    }
}

//! Real ELF scope/strong-import reproduction, isolated behind a watchdog.
use std::{ffi::{c_char,c_void,CStr,CString},process::{Command,Stdio},time::{Duration,Instant}};
extern "C" {
    fn dlopen(name:*const c_char,flags:i32)->*mut c_void;
    fn dlsym(handle:*mut c_void,name:*const c_char)->*mut c_void;
    fn dlerror()->*const c_char;
}
const NOW:i32=2;
unsafe extern "win64" fn callback(data:*mut c_void,len:u32)->i32 {
    let words=std::slice::from_raw_parts(data.cast::<u64>(),6);
    assert_eq!(len,48);assert_eq!(words[1..],[11,22,33,44,55]);
    0x12345678
}

#[test]
fn local_provider_strong_import_and_early_refusals() {
    for mode in ["missing-provider","missing-symbol","local-provider"] {
        let path=std::path::Path::new(env!("OUT_DIR")).join(format!("callback-loader-{mode}.log"));
        let output=std::fs::File::create(&path).unwrap();
        let mut child=Command::new(std::env::current_exe().unwrap())
            .args(["--ignored","--exact","unix_context_storage::loader_tests::loader_child","--nocapture"])
            .env("ZLUDA_CALLBACK_LOADER_MODE",mode).stdin(Stdio::null())
            .stdout(Stdio::from(output.try_clone().unwrap())).stderr(Stdio::from(output)).spawn().unwrap();
        let start=Instant::now();
        loop {
            if let Some(status)=child.try_wait().unwrap() {
                let log=std::fs::read_to_string(&path).unwrap();
                assert!(status.success(),"{mode}: {log}");assert!(log.contains("SCOPE PASS:"));break;
            }
            if start.elapsed()>Duration::from_secs(10) {let _=child.kill();let _=child.wait();panic!("ELF callback scope child timed out: {mode}");}
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
#[ignore]
fn loader_child() {unsafe {
    let mode=std::env::var("ZLUDA_CALLBACK_LOADER_MODE").expect("invoke the bounded parent");
    let symbol=b"KeUserModeCallback\0".as_ptr().cast();
    assert!(dlsym(std::ptr::null_mut(),symbol).is_null());
    let provider=match mode.as_str() {
        "missing-provider"=>None,
        "missing-symbol"=>Some("callback-empty-provider.so"),
        "local-provider"=>Some("callback-provider.so"),
        _=>panic!("unknown callback loader fixture mode"),
    };
    let mut handle=std::ptr::null_mut();
    if let Some(name)=provider {
        let path=CString::new(std::path::Path::new(env!("OUT_DIR")).join(name).to_str().unwrap()).unwrap();
        handle=dlopen(path.as_ptr(),NOW);assert!(!handle.is_null());
    }
    // Reproduce the original bridge bug even after loading Wine's analogue:
    // RTLD_DEFAULT does not see its private symbol in the local namespace.
    assert!(dlsym(std::ptr::null_mut(),symbol).is_null());
    let path=CString::new(concat!(env!("OUT_DIR"),"/callback-consumer.so")).unwrap();
    let consumer=dlopen(path.as_ptr(),NOW);
    if mode!="local-provider" {
        assert!(consumer.is_null(),"a missing provider/symbol must fail at eager module load");
        let error=dlerror();assert!(!error.is_null());let error=CStr::from_ptr(error).to_str().unwrap();
        assert!(error.contains(if mode=="missing-provider" {"zluda-callback-fixture-ntdll.so"} else {"KeUserModeCallback"}),"{error}");
        println!("SCOPE PASS: {mode} rejected before registration: {error}");return;
    }
    assert!(!consumer.is_null());
    let address:unsafe extern "C" fn()->*mut c_void=std::mem::transmute(dlsym(consumer,b"fixture_callback_address\0".as_ptr().cast()));
    assert_eq!(address(),dlsym(handle,symbol));
    let invoke:unsafe extern "C" fn(u32,*const c_void,u32,*mut *mut c_void,*mut u32)->i32=
        std::mem::transmute(dlsym(consumer,b"fixture_callback_invoke\0".as_ptr().cast()));
    let words=[callback as *const () as usize as u64,11,22,33,44,55];let mut out=1usize as *mut c_void;let mut bytes=99;
    assert_eq!(invoke(0,words.as_ptr().cast(),48,&mut out,&mut bytes),0x12345678);
    assert!(out.is_null());assert_eq!(bytes,0);
    assert!(dlsym(std::ptr::null_mut(),symbol).is_null(),"the provider must remain local");
    println!("SCOPE PASS: strong import reuses local provider; global lookup absent; exact Win64 callback args/return");
}}

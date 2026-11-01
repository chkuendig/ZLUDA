// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare the actual PE32 and x64 native dispatches with one fake CUDA backend.
use super::*;
use crate::cuda::*;
use std::{process::{Command,Stdio},sync::atomic::{AtomicUsize,Ordering},time::{Duration,Instant}};
static CALLS:AtomicUsize=AtomicUsize::new(0);
static LOOKUPS:AtomicUsize=AtomicUsize::new(0);
unsafe extern "C" fn attribute(out:*mut i32,attr:CUdevice_attribute,dev:CUdevice)->CUresult {
    CALLS.fetch_add(1,Ordering::Relaxed);
    assert!(matches!(attr.0,50|33|34));
    if dev<0{return Err(CUerror::INVALID_DEVICE);}
    if out.is_null(){return Err(CUerror::INVALID_VALUE);}
    *out=match attr.0{50=>0x12,33=>0x65,34=>2,_=>unreachable!()}+100*dev;Ok(())
}
unsafe fn symbol(name:*const c_char)->*mut c_void {
    assert_eq!(CStr::from_ptr(name).to_bytes(),b"cuDeviceGetAttribute","PCI must not preflight unrelated bounded-backend symbols");
    LOOKUPS.fetch_add(1,Ordering::Relaxed);attribute as *const () as *mut c_void
}
extern "C" {
    fn mmap(p:*mut c_void,n:usize,prot:i32,flags:i32,fd:i32,offset:i64)->*mut c_void;
    fn munmap(p:*mut c_void,n:usize)->i32;
}
#[test]
fn pci_attributes_match_x64_and_reserved_op_and_old_version_fail_closed(){
    let file=std::env::temp_dir().join(format!("zluda-pci-child-{}.log",std::process::id()));
    let log=std::fs::File::create(&file).unwrap();
    let mut child=Command::new(std::env::current_exe().unwrap())
        .args(["--exact","wow64_unix::pci_tests::pci_parity_child","--ignored","--nocapture","--test-threads=1"])
        .stdin(Stdio::null()).stdout(Stdio::from(log.try_clone().unwrap())).stderr(Stdio::from(log)).spawn().unwrap();
    let start=Instant::now();
    loop {
        if let Some(status)=child.try_wait().unwrap(){
            let text=std::fs::read_to_string(&file).unwrap();std::fs::remove_file(&file).unwrap();
            assert!(status.success(),"PCI parity child: {text}");assert!(text.contains("PCI PARITY PASS"));break;
        }
        if start.elapsed()>Duration::from_secs(15){let _=child.kill();let _=child.wait();panic!("PCI parity child timed out");}
        std::thread::sleep(Duration::from_millis(5));
    }
}
#[test]
#[ignore]
fn pci_parity_child(){unsafe{
    let memory=mmap(std::ptr::null_mut(),4096,3,2|0x20|0x40,-1,0);
    assert!(!memory.is_null() && (memory as usize)<0x80000000);
    let words=memory.cast::<u32>();let output=words.add(1).cast::<i32>();
    let path=std::env::temp_dir().join(format!("zluda-pci-backend-{}.so",std::process::id()));
    let mut header=[0u8;20];header[..6].copy_from_slice(b"\x7fELF\x02\x01");header[16..20].copy_from_slice(&[3,0,62,0]);std::fs::write(&path,header).unwrap();
    std::env::set_var("PROTON_ENABLE_ZLUDA","1");std::env::set_var("ZLUDA_WINE_BACKEND_PATH",&path);
    crate::unix_wrappers::with_fixture_loader(symbol,||{
        for attr in [50i32,33,34] {for dev in [0i32,1,-1] {for null in [false,true] {
            let before=CALLS.load(Ordering::Relaxed);
            words.write(0x11223344);words.add(1).write(0xcccccccc);words.add(2).write(0xaabbccdd);
            let address=if null{0}else{output as usize as u64};
            let mut x64=unixlib::Call::new(unixlib::function_id("cuDeviceGetAttribute"));
            x64.args[..3].copy_from_slice(&[address,unixlib::pack(CUdevice_attribute(attr as u32)),unixlib::pack(dev)]);
            assert_eq!(crate::unix_wrappers::dispatch((&mut x64 as *mut unixlib::Call).cast()),0);
            let expected=output.read();assert_eq!(CALLS.load(Ordering::Relaxed),before+1);
            output.write(0xccccccccu32 as i32);
            let mut pe32=Call32::new(C32_DEVICE_ATTR,&[address as u32,attr as u32,dev as u32]);
            assert_eq!(dispatch32((&mut pe32 as *mut Call32).cast()),0);
            assert_eq!(pe32.result as u32,x64.result,"attribute {attr}, device {dev}, null {null}");
            assert_eq!(output.read(),expected);assert_eq!(CALLS.load(Ordering::Relaxed),before+2);
            assert_eq!(words.read(),0x11223344);assert_eq!(words.add(2).read(),0xaabbccdd);
        }}}
        let count=CALLS.load(Ordering::Relaxed);
        let mut reserved=Call32::new(C32_RESERVED_59,&[0xdeadbeef;16]);
        assert_eq!(dispatch32((&mut reserved as *mut Call32).cast()),0);assert_eq!(reserved.result,801);
        let mut old=Call32::new(C32_DEVICE_ATTR,&[0xdeadbeef,50,0]);old.version=3;
        assert_eq!(dispatch32((&mut old as *mut Call32).cast()),0xc000000du32 as i32);assert_eq!(old.result,801);
        assert_eq!(CALLS.load(Ordering::Relaxed),count);assert_eq!(LOOKUPS.load(Ordering::Relaxed),1);
        assert!(STATE.get().is_none(),"PCI and reserved operations must not initialize bounded context state");
    });
    assert_eq!(munmap(memory,4096),0);std::fs::remove_file(path).unwrap();
    println!("PCI PARITY PASS: attributes=50/33/34 combinations=18 backend_calls=36; status/value/canaries equal; op59 and v3 refused without backend/state work");
}}

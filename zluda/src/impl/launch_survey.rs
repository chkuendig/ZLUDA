//! Bounded opt-in cumulative launch survey, sampled by a native external reader.
//! HIP return outcomes only: not execution, completion or correctness.
//! No GPU work, per-launch I/O, argument inspection or background thread.
//! Linux/x86-64 only; the reader must use native atomic loads of the shared ABI.

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod linux {
    use std::{
        cell::Cell,
        ffi::CString,
        fs::{File, OpenOptions},
        io,
        mem::{offset_of, size_of},
        os::fd::{AsRawFd, FromRawFd, IntoRawFd},
        os::unix::fs::{MetadataExt, OpenOptionsExt},
        ptr,
        sync::{atomic::{AtomicBool, AtomicU64, Ordering}, OnceLock, Mutex},
    };

    const ROWS: usize = 256;
    const ROW_BYTES: usize = 512;
    const BYTES: usize = 4096 + ROWS * ROW_BYTES;
    #[repr(C, align(64))]
    struct Row {
        name: [u8; 384],
        function: u64, module: u64, tls_context: u64, stream: u64, native_tid: u64,
        padding: [u8; 24],
        published: AtomicU64, success: AtomicU64, failure: AtomicU64, last_error: AtomicU64,
    }
    const _: () = assert!(offset_of!(Row, published) == 448);
    const _: () = assert!(size_of::<Row>() == ROW_BYTES);
    #[derive(PartialEq, Eq)]
    struct Key { name: String, function: usize, module: usize, context: usize, stream: usize, native_tid: u64 }

    #[repr(C, align(64))]
    struct Header {
        magic: [u8; 8],
        schema: u32,
        bytes: u32,
        pid: u32,
        reserved: u32,
        start_ticks: u64,
        boot: [u8; 36],
        app: [u8; 32],
        build: [u8; 64],
        instance: [u8; 32],
        padding: [u8; 60],
        active: AtomicU64,
        total: AtomicU64,
        failed: AtomicU64, dropped: AtomicU64, unknown: AtomicU64,
    }
    const _: () = assert!(offset_of!(Header, active) == 256);
    const _: () = assert!(offset_of!(Header, total) == 264);
    const _: () = assert!(size_of::<Header>() == 320);

    struct Recording {
        header: *mut Header,
        owner: libc::pid_t,
        directory: libc::c_int,
        name: CString,
        device: libc::dev_t,
        inode: libc::ino_t,
        keys: Mutex<Vec<Key>>,
    }
    // Immutable ownership information, and aligned atomic fields in the mapping.
    // Unload requires all callers to have stopped, like the rest of the library.
    unsafe impl Send for Recording {}
    unsafe impl Sync for Recording {}
    static RECORDING: OnceLock<Option<Recording>> = OnceLock::new();
    static STOPPED: AtomicBool = AtomicBool::new(false);
    thread_local! { static NATIVE_TID: Cell<u64> = const { Cell::new(0) }; }
    fn stream_thread(stream: usize) -> u64 {
        // PTDS token 2 denotes a distinct stream per native thread.
        // Other raw handles remain observed tokens, not lifetime-unique identities.
        if stream != 2 { return 0; }
        NATIVE_TID.with(|cached| {
            let previous = cached.get();
            if previous != 0 { return previous; }
            let tid = unsafe { libc::syscall(libc::SYS_gettid) } as u64;
            cached.set(tid); tid
        })
    }

    fn invalid() -> io::Error { io::Error::other("invalid recording metadata or directory") }
    fn fixed<const N: usize>(value: &str) -> io::Result<[u8; N]> {
        if value.len() >= N || !value.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err(invalid());
        }
        let mut out = [0; N];
        out[..value.len()].copy_from_slice(value.as_bytes());
        Ok(out)
    }
    fn start_ticks() -> io::Result<u64> {
        // /proc/self denotes the process leader even when cuInit runs on a worker.
        let stat = std::fs::read_to_string("/proc/self/stat")?;
        stat.rsplit_once(')').ok_or_else(invalid)?.1.split_whitespace()
            .nth(19).ok_or_else(invalid)?.parse().map_err(|_| invalid())
    }
    fn private_directory(fd: libc::c_int) -> io::Result<()> {
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut stat) } != 0 || unsafe { libc::fstatfs(fd, &mut fs) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if stat.st_uid != unsafe { libc::getuid() } || stat.st_mode & 0o7777 != 0o700
            || stat.st_mode & libc::S_IFMT != libc::S_IFDIR || fs.f_type != libc::TMPFS_MAGIC {
            return Err(invalid());
        }
        Ok(())
    }

    fn setup() -> io::Result<Recording> {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR").ok_or_else(invalid)?;
        if !std::path::Path::new(&runtime).is_absolute() { return Err(invalid()); }
        let root = OpenOptions::new().read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC).open(runtime)?;
        private_directory(root.as_raw_fd())?;
        let directory_name = c"zluda-launch-survey";
        if unsafe { libc::mkdirat(root.as_raw_fd(), directory_name.as_ptr(), 0o700) } != 0
            && io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST) {
            return Err(io::Error::last_os_error());
        }
        let fd = unsafe { libc::openat(root.as_raw_fd(), directory_name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC) };
        if fd < 0 { return Err(io::Error::last_os_error()); }
        let directory = unsafe { File::from_raw_fd(fd) };
        private_directory(directory.as_raw_fd())?;

        let owner = unsafe { libc::getpid() };
        let mut random = [0u8; 16];
        if unsafe { libc::getrandom(random.as_mut_ptr().cast(), random.len(), 0) } != random.len() as isize {
            return Err(io::Error::last_os_error());
        }
        let instance: String = random.iter().map(|b| format!("{b:02x}")).collect();
        let name = CString::new(format!("{owner}-{instance}.survey")).map_err(|_| invalid())?;
        let pending = CString::new(format!(".pending-{owner}-{instance}")).map_err(|_| invalid())?;
        let start_ticks = start_ticks()?;
        let boot_text = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
        let boot: [u8; 36] = boot_text.trim().as_bytes().try_into().map_err(|_| invalid())?;
        let app_text = std::env::var("SteamAppId").unwrap_or_else(|_| "0".into());
        if app_text.is_empty() || !app_text.bytes().all(|b| b.is_ascii_digit()) { return Err(invalid()); }
        let app = fixed(&app_text)?;
        let build = fixed(env!("VERGEN_GIT_SHA"))?;
        let instance_bytes: [u8; 32] = instance.as_bytes().try_into().map_err(|_| invalid())?;

        let fd = unsafe { libc::openat(directory.as_raw_fd(), pending.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC, 0o600) };
        if fd < 0 { return Err(io::Error::last_os_error()); }
        let file = unsafe { File::from_raw_fd(fd) };
        // The inode is private and unpublished while its length and contents change.
        let initialized = (|| {
            let identity = file.metadata()?;
            if unsafe { libc::fchmod(file.as_raw_fd(), 0o600) } != 0 { return Err(io::Error::last_os_error()); }
            file.set_len(BYTES as u64)?;
            // ftruncate alone does not reserve tmpfs backing. Without this,
            // pre-touch can raise SIGBUS when the runtime filesystem is full.
            let allocated = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, BYTES as i64) };
            if allocated != 0 { return Err(io::Error::from_raw_os_error(allocated)); }
            let mapping = unsafe { libc::mmap(ptr::null_mut(), BYTES, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED, file.as_raw_fd(), 0) };
            if mapping == libc::MAP_FAILED { return Err(io::Error::last_os_error()); }
            let header = mapping.cast::<Header>();
            unsafe {
                // Pre-touch the entire mapping before exposing it to launches/readers.
                ptr::write_bytes(mapping.cast::<u8>(), 0, BYTES);
                ptr::write(header, Header { magic: *b"ZLUDAL01", schema: 1, bytes: BYTES as u32,
                    pid: owner as u32, reserved: 0, start_ticks, boot, app, build,
                    instance: instance_bytes, padding: [0; 60], active: AtomicU64::new(0), total: AtomicU64::new(0), failed: AtomicU64::new(0), dropped: AtomicU64::new(0), unknown: AtomicU64::new(0) });
                for index in 0..ROWS {
                    ptr::write((mapping.cast::<u8>().add(4096 + index * ROW_BYTES)).cast::<Row>(), Row {
                        name: [0; 384], function: 0, module: 0, tls_context: 0, stream: 0, native_tid: 0, padding: [0; 24],
                        published: AtomicU64::new(0), success: AtomicU64::new(0), failure: AtomicU64::new(0), last_error: AtomicU64::new(0),
                    });
                }
                (*header).active.store(1, Ordering::Release);
            }
            // linkat publishes without replacing any existing inode. Readers ignore .pending files.
            if unsafe { libc::linkat(directory.as_raw_fd(), pending.as_ptr(), directory.as_raw_fd(), name.as_ptr(), 0) } != 0 {
                let error = io::Error::last_os_error();
                unsafe { libc::munmap(mapping, BYTES); }
                return Err(error);
            }
            Ok((header, identity))
        })();
        unsafe { libc::unlinkat(directory.as_raw_fd(), pending.as_ptr(), 0); }
        let (header, identity) = initialized?;
        Ok(Recording { header, owner, directory: directory.into_raw_fd(), name,
            device: identity.dev(), inode: identity.ino(), keys: Mutex::new(Vec::with_capacity(ROWS)) })
    }

    pub(super) fn initialize() {
        // Called only after successful initialization. Failure never changes CUDA results.
        RECORDING.get_or_init(|| {
            if std::env::var("ZLUDA_LAUNCH_SURVEY").as_deref() != Ok("1") { return None; }
            setup().ok()
        });
    }
    pub(super) fn observed(name: &str, function: usize, module: usize, stream: usize, error: u32) {
        let Some(Some(recording)) = RECORDING.get() else { return; };
        if unsafe { libc::getpid() } != recording.owner { return; }
        let header = unsafe { &*recording.header };
        if error == 0 { header.total.fetch_add(1, Ordering::Relaxed); }
        else { header.failed.fetch_add(1, Ordering::Relaxed); }
        if name.is_empty() { header.unknown.fetch_add(1, Ordering::Relaxed); }
        // Context is the observed ZLUDA TLS stack, not a HIP query. Zero is unknown.
        let context = super::super::context::STACK.try_with(|stack| {
            stack.try_borrow().ok().and_then(|s| s.last().map(|v| v.0.0 as usize)).unwrap_or(0)
        }).unwrap_or(0);
        let native_tid = stream_thread(stream);
        let Ok(mut keys) = recording.keys.lock() else {
            header.dropped.fetch_add(1, Ordering::Relaxed); return;
        };
        let found = keys.iter().position(|key| key.name == name && key.function == function
            && key.module == module && key.context == context && key.stream == stream && key.native_tid == native_tid);
        let index = match found {
            Some(index) => index,
            None => {
                if keys.len() == ROWS || name.len() >= 384 || name.as_bytes().contains(&0) {
                    header.dropped.fetch_add(1, Ordering::Relaxed); return;
                }
                let index = keys.len();
                let row = unsafe { (recording.header.cast::<u8>().add(4096 + index * ROW_BYTES)).cast::<Row>() };
                unsafe {
                    ptr::copy_nonoverlapping(name.as_ptr(), ptr::addr_of_mut!((*row).name).cast(), name.len());
                    (*row).function = function as u64; (*row).module = module as u64;
                    (*row).tls_context = context as u64; (*row).stream = stream as u64; (*row).native_tid = native_tid;
                    (*row).published.store(1, Ordering::Release);
                }
                keys.push(Key { name: name.to_owned(), function, module, context, stream, native_tid });
                index
            }
        };
        let row = unsafe { &*(recording.header.cast::<u8>().add(4096 + index * ROW_BYTES)).cast::<Row>() };
        if error == 0 { row.success.fetch_add(1, Ordering::Relaxed); }
        else { row.last_error.store(error as u64, Ordering::Relaxed); row.failure.fetch_add(1, Ordering::Relaxed); }
    }
    pub(super) fn shutdown() {
        if let Some(Some(recording)) = RECORDING.get() {
            if unsafe { libc::getpid() } != recording.owner || STOPPED.swap(true, Ordering::Relaxed) { return; }
            // No allocation, locks, logging, HIP calls or joins in the unload path.
            unsafe {
                (*recording.header).active.store(0, Ordering::Release);
                let mut identity: libc::stat = std::mem::zeroed();
                if libc::fstatat(recording.directory, recording.name.as_ptr(), &mut identity, libc::AT_SYMLINK_NOFOLLOW) == 0
                    && identity.st_dev == recording.device && identity.st_ino == recording.inode {
                    libc::unlinkat(recording.directory, recording.name.as_ptr(), 0);
                }
                libc::munmap(recording.header.cast(), BYTES);
                libc::close(recording.directory);
            }
        }
    }


    #[cfg(test)]
    mod tests {
        use super::*;
        use std::{os::unix::fs::{DirBuilderExt,PermissionsExt}, process::Command};
        #[test]
        fn launch_survey_records() {
            const CASE: &str="ZLUDA_SURVEY_TEST_CASE";
            if let Ok(case)=std::env::var(CASE) {
                let runtime=std::path::PathBuf::from(format!("/dev/shm/zluda-survey-test-{}",std::process::id()));
                std::fs::DirBuilder::new().mode(0o700).create(&runtime).unwrap();
                std::env::set_var("XDG_RUNTIME_DIR",&runtime);std::env::set_var("SteamAppId","123");
                std::env::remove_var("ZLUDA_LAUNCH_SURVEY");
                if case=="disabled" {
                    initialize();observed("TurbulenceFS_test",1,2,3,0);shutdown();
                    assert!(!runtime.join("zluda-launch-survey").exists());
                } else if case=="setup-failure" {
                    std::env::set_var("ZLUDA_LAUNCH_SURVEY","1");
                    std::fs::set_permissions(&runtime,std::fs::Permissions::from_mode(0o755)).unwrap();
                    initialize();observed("test",1,2,3,0);shutdown();
                    assert!(RECORDING.get().unwrap().is_none());
                    assert!(!runtime.join("zluda-launch-survey").exists());
                } else {
                    assert_eq!(case,"counts");std::env::set_var("ZLUDA_LAUNCH_SURVEY","1");initialize();
                    let recording=RECORDING.get().unwrap().as_ref().unwrap();
                    let path=runtime.join("zluda-launch-survey").join(recording.name.to_str().unwrap());
                    assert!(path.is_file());
                    let barrier=std::sync::Barrier::new(4);
                    std::thread::scope(|scope| {for _ in 0..4 {let barrier=&barrier;scope.spawn(move|| {
                        barrier.wait();for _ in 0..1000 {observed("TurbulenceFS_dynamicGridApplyForcesKernel",1,2,3,0);}
                    });}});
                    for _ in 0..3 {observed("TurbulenceFS_dynamicGridApplyForcesKernel",1,2,3,17);}
                    // Two PTDS callers share context/function/token but must occupy separate rows.
                    let barrier=std::sync::Barrier::new(2);
                    std::thread::scope(|scope| {for _ in 0..2 {let barrier=&barrier;scope.spawn(move|| {
                        barrier.wait();for _ in 0..13 {observed("ptds",1,2,2,0);}
                    });}});
                    observed("stream-separated",1,2,3,0);observed("stream-separated",1,2,4,0);
                    super::super::super::context::STACK.with(|stack| {
                        stack.borrow_mut().push((cuda_types::cuda::CUcontext(1usize as *mut _),0));
                    });
                    observed("context-separated",1,2,3,0);
                    super::super::super::context::STACK.with(|stack| {stack.borrow_mut().pop();});
                    observed("context-separated",1,2,3,0);
                    {
                        let keys=recording.keys.lock().unwrap();
                        let ptds:Vec<_>=keys.iter().filter(|k|k.name=="ptds").collect();
                        assert_eq!(ptds.len(),2);assert_ne!(ptds[0].native_tid,ptds[1].native_tid);
                        assert!(ptds.iter().all(|k|k.native_tid!=0 && k.context==0 && k.stream==2));
                        assert_eq!(keys.iter().filter(|k|k.name=="stream-separated").count(),2);
                        let contexts:Vec<_>=keys.iter().filter(|k|k.name=="context-separated").map(|k|k.context).collect();
                        assert_eq!(contexts,vec![1,0]);
                    }
                    observed("",4,2,3,0);observed(&"x".repeat(384),5,2,3,0);
                    for i in 0..ROWS {observed(&format!("kernel{i}"),i+10,2,3,0);}
                    let header=unsafe {&*recording.header};
                    assert_eq!(header.total.load(Ordering::Relaxed),4000+26+4+2+ROWS as u64);
                    assert_eq!(header.failed.load(Ordering::Relaxed),3);
                    assert_eq!(header.unknown.load(Ordering::Relaxed),1);
                    assert_eq!(header.dropped.load(Ordering::Relaxed),9);
                    let row=unsafe {&*(recording.header.cast::<u8>().add(4096)).cast::<Row>()};
                    assert_eq!(row.success.load(Ordering::Relaxed),4000);assert_eq!(row.failure.load(Ordering::Relaxed),3);
                    assert_eq!(row.last_error.load(Ordering::Relaxed),17);assert_eq!(recording.keys.lock().unwrap().len(),ROWS);
                    let fd=OpenOptions::new().read(true).write(true).open(&path).unwrap();
                    let read=unsafe {libc::mmap(ptr::null_mut(),BYTES,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_SHARED,fd.as_raw_fd(),0)};
                    assert_ne!(read,libc::MAP_FAILED);shutdown();shutdown();assert!(!path.exists());
                    assert_eq!(unsafe {&*read.cast::<Header>()}.active.load(Ordering::Acquire),0);
                    unsafe {libc::munmap(read,BYTES);}
                }
                std::fs::remove_dir_all(runtime).unwrap();return;
            }
            for case in ["disabled","setup-failure","counts"] {
                let result=Command::new(std::env::current_exe().unwrap()).arg("launch_survey_records").arg("--nocapture")
                    .env(CASE,case).env_remove("LD_PRELOAD").output().unwrap();
                assert!(result.status.success(),"{case}: {} {}",String::from_utf8_lossy(&result.stdout),String::from_utf8_lossy(&result.stderr));
                assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
            }
        }
    }

}

pub(crate) fn initialize() {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    linux::initialize();
}
pub(crate) fn observed(name: &str, function: usize, module: usize, stream: usize, error: u32) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    linux::observed(name, function, module, stream, error);
}
pub(crate) fn shutdown() {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    linux::shutdown();
}

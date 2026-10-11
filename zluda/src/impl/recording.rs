//! Opt-in recording telemetry. Counts successful common-path HIP submissions,
//! not execution, completion, utilization, graph launches or copy interceptions.
//! Linux/x86-64 only; the reader must use native atomic loads of the shared ABI.

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod linux {
    use std::{
        ffi::CString,
        fs::{File, OpenOptions},
        io,
        mem::{offset_of, size_of},
        os::fd::{AsRawFd, FromRawFd, IntoRawFd},
        os::unix::fs::{MetadataExt, OpenOptionsExt},
        ptr,
        sync::{atomic::{AtomicBool, AtomicU64, Ordering}, OnceLock},
    };

    const BYTES: usize = 4096;
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
    }
    // Immutable ownership information, and aligned atomic fields in the mapping.
    // Unload requires all callers to have stopped, like the rest of the library.
    unsafe impl Send for Recording {}
    unsafe impl Sync for Recording {}
    static RECORDING: OnceLock<Option<Recording>> = OnceLock::new();
    static STOPPED: AtomicBool = AtomicBool::new(false);

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
        let directory_name = c"zluda-recording";
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
        let name = CString::new(format!("{owner}-{instance}.rec")).map_err(|_| invalid())?;
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
            #[cfg(test)]
            if std::env::var("ZLUDA_RECORDING_TEST_CASE").as_deref() == Ok("allocation-failure") {
                return Err(io::Error::from_raw_os_error(libc::ENOSPC));
            }
            let allocated = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, BYTES as i64) };
            if allocated != 0 { return Err(io::Error::from_raw_os_error(allocated)); }
            let mapping = unsafe { libc::mmap(ptr::null_mut(), BYTES, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED, file.as_raw_fd(), 0) };
            if mapping == libc::MAP_FAILED { return Err(io::Error::last_os_error()); }
            let header = mapping.cast::<Header>();
            unsafe {
                // Pre-touch the entire mapping before exposing it to launches/readers.
                ptr::write_bytes(mapping.cast::<u8>(), 0, BYTES);
                ptr::write(header, Header { magic: *b"ZLUDAS01", schema: 1, bytes: BYTES as u32,
                    pid: owner as u32, reserved: 0, start_ticks, boot, app, build,
                    instance: instance_bytes, padding: [0; 60], active: AtomicU64::new(0), total: AtomicU64::new(0) });
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
            device: identity.dev(), inode: identity.ino() })
    }

    pub(super) fn initialize() {
        // Called only after successful initialization. Failure never changes CUDA results.
        RECORDING.get_or_init(|| {
            if std::env::var("ZLUDA_RECORDING").as_deref() != Ok("1") { return None; }
            setup().ok()
        });
    }
    pub(super) fn submitted() {
        if let Some(Some(recording)) = RECORDING.get() {
            // A fork child must not contribute to its parent's shared counter.
            if unsafe { libc::getpid() } == recording.owner {
                unsafe { (*recording.header).total.fetch_add(1, Ordering::Relaxed); }
            }
        }
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
        use std::{os::unix::fs::{DirBuilderExt, PermissionsExt}, process::Command};

        #[test]
        fn recording_lifecycle() {
            const CASE: &str = "ZLUDA_RECORDING_TEST_CASE";
            if let Ok(case) = std::env::var(CASE) {
                let runtime = std::path::PathBuf::from(format!("/dev/shm/zluda-recording-test-{}", std::process::id()));
                std::fs::DirBuilder::new().mode(0o700).create(&runtime).unwrap();
                std::env::set_var("XDG_RUNTIME_DIR", &runtime);
                std::env::set_var("SteamAppId", "123");
                std::env::remove_var("ZLUDA_RECORDING");
                if case == "disabled" {
                    initialize(); submitted(); shutdown();
                    assert!(!runtime.join("zluda-recording").exists());
                } else if case == "setup-failure" || case == "allocation-failure" {
                    std::env::set_var("ZLUDA_RECORDING", "1");
                    if case == "setup-failure" { std::fs::set_permissions(&runtime, std::fs::Permissions::from_mode(0o755)).unwrap(); }
                    initialize(); submitted(); shutdown();
                    assert!(RECORDING.get().unwrap().is_none());
                    if case == "setup-failure" { assert!(!runtime.join("zluda-recording").exists()); }
                    else { assert_eq!(std::fs::read_dir(runtime.join("zluda-recording")).unwrap().count(), 0); }
                } else {
                    std::env::set_var("ZLUDA_RECORDING", "1"); initialize();
                    let recording = RECORDING.get().unwrap().as_ref().unwrap();
                    let entries: Vec<_> = std::fs::read_dir(runtime.join("zluda-recording")).unwrap().collect();
                    assert_eq!(entries.len(), 1); // No partial-publication remnants.
                    let path = entries[0].as_ref().unwrap().path();
                    let file = OpenOptions::new().read(true).write(true).open(&path).unwrap();
                    assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
                    let read_mapping = unsafe { libc::mmap(ptr::null_mut(), BYTES, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, file.as_raw_fd(), 0) };
                    assert_ne!(read_mapping, libc::MAP_FAILED);
                    let observed = unsafe { &*read_mapping.cast::<Header>() };
                    assert_eq!(observed.active.load(Ordering::Acquire), 1);
                    assert_eq!(observed.start_ticks, start_ticks().unwrap());
                    assert_eq!(observed.pid, std::process::id());
                    if case == "concurrent" {
                        std::thread::scope(|s| { for _ in 0..4 { s.spawn(|| { for _ in 0..10000 { submitted(); } }); } });
                        assert_eq!(observed.total.load(Ordering::Relaxed), 40000);
                    } else if case == "fork" {
                        submitted();
                        let child = unsafe { libc::fork() }; assert!(child >= 0);
                        if child == 0 {
                            for _ in 0..100 { submitted(); }
                            shutdown();
                            unsafe { libc::_exit(0); }
                        }
                        let mut status = -1;
                        assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
                        assert_eq!(status, 0);
                        assert_eq!(observed.total.load(Ordering::Relaxed), 1);
                        assert_eq!(observed.active.load(Ordering::Acquire), 1);
                        assert!(path.exists());
                    } else { panic!("unknown child case"); }
                    assert_eq!(recording.owner, unsafe { libc::getpid() });
                    shutdown(); shutdown();
                    assert!(!path.exists());
                    // Existing reader maps survive unlink + producer unmap.
                    assert_eq!(observed.active.load(Ordering::Acquire), 0);
                    unsafe { libc::munmap(read_mapping, BYTES); }
                }
                std::fs::remove_dir_all(runtime).unwrap();
                return;
            }
            for case in ["disabled", "setup-failure", "allocation-failure", "concurrent", "fork"] {
                let result = Command::new(std::env::current_exe().unwrap())
                    .arg("recording_lifecycle").arg("--nocapture").env(CASE, case)
                    .env_remove("LD_PRELOAD").output().unwrap();
                assert!(result.status.success(), "{case}: {} {}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
                assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
            }
        }
    }
}

pub(crate) fn initialize() {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    linux::initialize();
}
pub(crate) fn submitted() {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    linux::submitted();
}
pub(crate) fn shutdown() {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    linux::shutdown();
}

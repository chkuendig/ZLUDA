//! Preserve the embedding application's signal state across LLVM/comgr calls.
//!
//! Wine requires its own alternate stack to recover the TEB in signal handlers.
//! LLVM's RemoveFileOnSignal can replace that stack, even though comgr restores
//! the signal dispositions. An alternate stack belongs to the calling thread;
//! dispositions belong to the process. The mutex prevents overlapping ZLUDA
//! snapshots, but cannot protect other threads from LLVM's temporary handlers,
//! or from a host changing its dispositions during compilation.

use cuda_types::cuda::CUerror;

#[cfg(not(target_os = "linux"))]
pub(super) fn with_preserved_state<T>(call: impl FnOnce() -> Result<T, CUerror>) -> Result<T, CUerror> {
    call()
}

#[cfg(target_os = "linux")]
pub(super) fn with_preserved_state<T>(call: impl FnOnce() -> Result<T, CUerror>) -> Result<T, CUerror> {
    let mut state = linux::SignalState::save()?;
    let result = call();
    state.restore()?;
    result
}

#[cfg(target_os = "linux")]
mod linux {
    use super::CUerror;
    use std::sync::{Mutex, MutexGuard};

    // LLVM Support/Unix/Signals.inc: IntSigs, KillSigs, InfoSigs, and optional
    // SIGPIPE. Linux has neither the conditional SIGEMT nor SIGINFO entries.
    const SIGNALS: &[libc::c_int] = &[
        libc::SIGHUP,
        libc::SIGINT,
        libc::SIGTERM,
        libc::SIGUSR2,
        libc::SIGILL,
        libc::SIGTRAP,
        libc::SIGABRT,
        libc::SIGFPE,
        libc::SIGBUS,
        libc::SIGSEGV,
        libc::SIGQUIT,
        libc::SIGSYS,
        libc::SIGXCPU,
        libc::SIGXFSZ,
        libc::SIGUSR1,
        libc::SIGPIPE,
    ];
    static COMPILER_SIGNAL_STATE: Mutex<()> = Mutex::new(());

    pub(super) struct SignalState {
        // Also makes the snapshot !Send: restore must run on the same thread.
        _lock: MutexGuard<'static, ()>,
        stack: libc::stack_t,
        actions: [libc::sigaction; SIGNALS.len()],
        armed: bool,
    }

    fn failed(operation: &str) -> CUerror {
        eprintln!("ZLUDA: {operation}: {}", std::io::Error::last_os_error());
        CUerror::UNKNOWN
    }

    impl SignalState {
        pub(super) fn save() -> Result<Self, CUerror> {
            let lock = COMPILER_SIGNAL_STATE
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut stack = unsafe { std::mem::zeroed() };
            let mut actions = std::array::from_fn(|_| unsafe { std::mem::zeroed() });
            if unsafe { libc::sigaltstack(std::ptr::null(), &mut stack) } != 0 {
                return Err(failed("query compiler caller's sigaltstack"));
            }
            for (&signal, action) in SIGNALS.iter().zip(&mut actions) {
                if unsafe { libc::sigaction(signal, std::ptr::null(), action) } != 0 {
                    return Err(failed("query compiler caller's sigaction"));
                }
            }
            Ok(Self {
                _lock: lock,
                stack,
                actions,
                armed: true,
            })
        }

        pub(super) fn restore(&mut self) -> Result<(), CUerror> {
            if !self.armed {
                return Ok(());
            }
            self.armed = false;
            let mut error = None;
            // Restore the stack before handing signals back to the host. Skip
            // an unchanged stack, including calls already on an alternate stack
            // (sigaltstack would otherwise fail with EPERM there).
            let mut current: libc::stack_t = unsafe { std::mem::zeroed() };
            if unsafe { libc::sigaltstack(std::ptr::null(), &mut current) } != 0 {
                error = Some(failed("query compiler return sigaltstack"));
            } else if (current.ss_sp != self.stack.ss_sp
                || current.ss_size != self.stack.ss_size
                || current.ss_flags != self.stack.ss_flags)
                && unsafe { libc::sigaltstack(&self.stack, std::ptr::null_mut()) } != 0
            {
                error = Some(failed("restore compiler caller's sigaltstack"));
            }
            // Attempt every restore even if an earlier one failed.
            for (&signal, action) in SIGNALS.iter().zip(&self.actions) {
                if unsafe { libc::sigaction(signal, action, std::ptr::null_mut()) } != 0 {
                    error = Some(failed("restore compiler caller's sigaction"));
                }
            }
            error.map_or(Ok(()), Err)
        }
    }

    impl Drop for SignalState {
        fn drop(&mut self) {
            // Also restore on an early return or Rust unwind. Errors are logged
            // here; the normal path propagates restoration failures to CUDA.
            let _ = self.restore();
        }
    }
}

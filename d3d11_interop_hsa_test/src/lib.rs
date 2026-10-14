//! D3D11 → CUDA texture interop test harness.
//!
//! Test logic lives in the integration tests under `tests/`. This crate's
//! library exists only for the serialization helper: HIP / HSA do not
//! tolerate two threads in the same process running dma_buf import
//! concurrently (`hipImportExternalMemory` returns INVALID_VALUE), so
//! every test acquires a global mutex on entry.
//!
//! See Cargo.toml comment + plan resilient-sprouting-pie.md (Phase A.HSA).

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Global lock guarding HIP/HSA calls across parallel test threads. Holds
/// the lock for the lifetime of the returned guard — drop it at end of test.
pub fn serialize() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        // Recover from poisoning — a previous test panicking shouldn't
        // poison subsequent unrelated runs.
        .unwrap_or_else(|p| p.into_inner())
}

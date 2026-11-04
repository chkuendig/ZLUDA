// SPDX-License-Identifier: MIT OR Apache-2.0
//! Short process-owned registry locks, usable after Windows std TLS teardown.
use std::{cell::UnsafeCell, marker::PhantomData, mem::MaybeUninit, ops::{Deref, DerefMut}, sync::atomic::{AtomicBool, AtomicU8, Ordering}};

#[cfg(windows)]
#[link(name="kernel32")]
extern "system" {fn Sleep(milliseconds:u32);}

fn yield_wait() {
    #[cfg(windows)] unsafe {Sleep(1);}
    // Native CPU projections use their own scheduler, not the mocked PE
    // Sleep service that records the public D3D11 retry sequence.
    #[cfg(not(windows))] std::thread::yield_now();
}

fn backoff(retries:&mut u32) {
    if *retries<64 { *retries+=1;std::hint::spin_loop(); }
    else { yield_wait(); }
}

/// Immutable process-lifetime values. Initializers run once on success;
/// an unwind releases the initialization claim without consulting std TLS.
pub struct Once<T> {
    state: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T: Send + Sync> Sync for Once<T> {}

impl<T> Once<T> {
    pub const fn new() -> Self {
        Self { state: AtomicU8::new(0), value: UnsafeCell::new(MaybeUninit::uninit()) }
    }

    pub fn get(&self) -> Option<&T> {
        if self.state.load(Ordering::Acquire) == 2 {
            // State 2 is immutable and published only after the value write.
            Some(unsafe { (&*self.value.get()).assume_init_ref() })
        } else { None }
    }

    pub fn get_or_init(&self, initialize: impl FnOnce() -> T) -> &T {
        struct Claim<'a> { state: &'a AtomicU8, complete: bool }
        impl Drop for Claim<'_> {
            fn drop(&mut self) {
                if !self.complete { self.state.store(0, Ordering::Release); }
            }
        }
        let mut initialize=Some(initialize);
        let mut retries=0;
        loop {
            if let Some(value)=self.get() { return value; }
            if self.state.compare_exchange(0,1,Ordering::Acquire,Ordering::Relaxed).is_ok() {
                let mut claim=Claim { state: &self.state, complete: false };
                let value=initialize.take().unwrap()();
                unsafe { (*self.value.get()).write(value); }
                self.state.store(2,Ordering::Release);
                claim.complete=true;
                return unsafe { (&*self.value.get()).assume_init_ref() };
            }
            backoff(&mut retries);
        }
    }

    pub fn set(&self, value: T) -> Result<(),T> {
        let mut value=Some(value);
        self.get_or_init(||value.take().unwrap());
        match value { Some(value)=>Err(value),None=>Ok(()) }
    }
}

pub struct Mutex<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// The acquire/release lock is the only route to the protected value.
unsafe impl<T: Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self { locked: AtomicBool::new(false), value: UnsafeCell::new(value) }
    }

    pub fn lock(&self) -> Guard<'_, T> {
        let mut retries=0;
        while self.locked.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            while self.locked.load(Ordering::Relaxed) { backoff(&mut retries); }
        }
        Guard { mutex: self, _thread_bound: PhantomData }
    }
}

pub struct Guard<'a, T> {
    mutex: &'a Mutex<T>,
    _thread_bound: PhantomData<*mut T>,
}

impl<T> Deref for Guard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { unsafe { &*self.mutex.value.get() } }
}

impl<T> DerefMut for Guard<'_, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.mutex.value.get() } }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) { self.mutex.locked.store(false, Ordering::Release); }
}

// Logical operation owners wait outside their short registry lock. No std
// thread identity, parking data, poisoning state or TLS destructor is involved.
#[cfg(any(all(windows,target_arch="x86_64"),test))]
pub fn wait() { yield_wait(); }

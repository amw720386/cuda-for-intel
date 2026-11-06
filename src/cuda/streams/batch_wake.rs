use super::CuResult;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

pub(super) enum Wake {
    Condvar(Mutex<bool>, Condvar),
    #[cfg(windows)]
    HighResolution(windows::Timer),
}

impl Wake {
    pub(super) fn new(high_resolution: bool) -> Result<Self, CuResult> {
        if high_resolution {
            #[cfg(windows)]
            return windows::Timer::new().map(Self::HighResolution);
            #[cfg(not(windows))]
            return Err(CuResult::NotSupported);
        }
        Ok(Self::Condvar(Mutex::new(false), Condvar::new()))
    }

    pub(super) fn notify(&self) {
        match self {
            Self::Condvar(pending, wake) => {
                *pending.lock().unwrap() = true;
                wake.notify_one();
            }
            #[cfg(windows)]
            Self::HighResolution(timer) => timer.notify(),
        }
    }

    // A notification is latched even if the runtime lock was just released
    // and the worker has not yet entered the wait.
    pub(super) fn wait(&self, deadline: Option<Instant>) -> Result<(), CuResult> {
        match self {
            Self::Condvar(pending, wake) => {
                let mut pending = pending.lock().map_err(|_| CuResult::Unknown)?;
                while !*pending {
                    pending = if let Some(deadline) = deadline {
                        let left = deadline.saturating_duration_since(Instant::now());
                        if left == Duration::ZERO {
                            break;
                        }
                        wake.wait_timeout(pending, left)
                            .map_err(|_| CuResult::Unknown)?
                            .0
                    } else {
                        wake.wait(pending).map_err(|_| CuResult::Unknown)?
                    };
                }
                *pending = false;
                Ok(())
            }
            #[cfg(windows)]
            Self::HighResolution(timer) => timer.wait(deadline),
        }
    }
}

#[cfg(windows)]
mod windows {
    use super::*;
    use std::ffi::c_void;
    use std::ptr::null;

    type Raw = *mut c_void;
    const INFINITE: u32 = u32::MAX;
    const SYNCHRONIZE: u32 = 0x0010_0000;
    const TIMER_MODIFY_STATE: u32 = 2;
    const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 2;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateEventW(attrs: *const c_void, manual: i32, initial: i32, name: *const u16) -> Raw;
        fn CreateWaitableTimerExW(
            attrs: *const c_void,
            name: *const u16,
            flags: u32,
            access: u32,
        ) -> Raw;
        fn SetWaitableTimerEx(
            timer: Raw,
            due: *const i64,
            period: i32,
            completion: Option<unsafe extern "system" fn(Raw, u32, u32)>,
            arg: Raw,
            wake_context: *const c_void,
            tolerable_delay: u32,
        ) -> i32;
        fn CancelWaitableTimer(timer: Raw) -> i32;
        fn WaitForMultipleObjects(count: u32, handles: *const Raw, all: i32, timeout: u32) -> u32;
        fn SetEvent(event: Raw) -> i32;
        fn CloseHandle(handle: Raw) -> i32;
    }

    struct Handle(Raw);

    // Both handles are private and live until the last reference to Wake drops.
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    impl Handle {
        fn new(raw: Raw) -> Result<Self, CuResult> {
            if raw.is_null() {
                Err(os_error())
            } else {
                Ok(Self(raw))
            }
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            if unsafe { CloseHandle(self.0) } == 0 {
                os_error();
            }
        }
    }

    fn os_error() -> CuResult {
        crate::cuda::diagnostic(&std::io::Error::last_os_error());
        CuResult::OperatingSystem
    }

    pub(in super::super) struct Timer {
        event: Handle,
        timer: Handle,
    }

    impl Timer {
        pub(super) fn new() -> Result<Self, CuResult> {
            let event = Handle::new(unsafe { CreateEventW(null(), 0, 0, null()) })?;
            let timer = Handle::new(unsafe {
                CreateWaitableTimerExW(
                    null(),
                    null(),
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    SYNCHRONIZE | TIMER_MODIFY_STATE,
                )
            })?;
            Ok(Self { event, timer })
        }

        pub(super) fn notify(&self) {
            if unsafe { SetEvent(self.event.0) } == 0 {
                os_error();
            }
        }

        pub(super) fn wait(&self, deadline: Option<Instant>) -> Result<(), CuResult> {
            let handles = [self.event.0, self.timer.0];
            let count = if let Some(deadline) = deadline {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Ok(());
                }
                // Negative due times are relative, in units of 100 ns.
                let due = -(remaining.as_nanos().div_ceil(100) as i64);
                if unsafe {
                    SetWaitableTimerEx(self.timer.0, &due, 0, None, std::ptr::null_mut(), null(), 0)
                } == 0
                {
                    return Err(os_error());
                }
                2
            } else {
                1
            };
            let result = unsafe { WaitForMultipleObjects(count, handles.as_ptr(), 0, INFINITE) };
            if count == 2 && result == 0 {
                // An event (new work or shutdown) superseded the deadline.
                if unsafe { CancelWaitableTimer(self.timer.0) } == 0 {
                    return Err(os_error());
                }
            }
            if result < count {
                Ok(())
            } else {
                Err(os_error())
            }
        }
    }
}

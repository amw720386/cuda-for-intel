//! CUDA Driver ABI. Public pointers follow CUDA's caller validity requirements.
use crate::level_zero::{DeviceInfo, Error, Runtime};
use std::ffi::{c_char, c_int, c_uint};
use std::sync::Mutex;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CuResult {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    NoDevice = 100,
    InvalidDevice = 101,
    NotSupported = 801,
    Unknown = 999,
}

use CuResult::*;

static RUNTIME: Mutex<Option<Runtime>> = Mutex::new(None);

fn backend_error(error: Error) -> CuResult {
    match error {
        Error::Loader(_) | Error::Driver("zeInit", 0x78000001) => NoDevice,
        Error::Driver(_, 0x70000002 | 0x70000003) => OutOfMemory,
        Error::Driver(_, 0x78000002 | 0x78000003) => NotSupported,
        _ => Unknown,
    }
}

fn runtime<T>(action: impl FnOnce(&Runtime) -> Result<T, CuResult>) -> Result<T, CuResult> {
    let guard = RUNTIME.lock().map_err(|_| Unknown)?;
    action(guard.as_ref().ok_or(NotInitialized)?)
}

fn device(ordinal: c_int, action: impl FnOnce(&DeviceInfo) -> CuResult) -> CuResult {
    runtime(|rt| {
        let device = rt.devices.get(ordinal as usize).ok_or(InvalidDevice)?;
        Ok(action(&device.info))
    })
    .unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
pub extern "system" fn cuInit(flags: c_uint) -> CuResult {
    if flags != 0 {
        return InvalidValue;
    }
    let Ok(mut guard) = RUNTIME.lock() else {
        return Unknown;
    };
    if guard.is_some() {
        return Success;
    }
    match Runtime::load() {
        Ok(rt) if rt.devices.is_empty() => NoDevice,
        Ok(rt) => {
            *guard = Some(rt);
            Success
        }
        Err(error) => backend_error(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetCount(count: *mut c_int) -> CuResult {
    runtime(|rt| {
        if count.is_null() {
            return Err(InvalidValue);
        }
        let value = c_int::try_from(rt.devices.len()).map_err(|_| Unknown)?;
        unsafe { count.write(value) };
        Ok(Success)
    })
    .unwrap_or_else(|error| error)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGet(out: *mut c_int, ordinal: c_int) -> CuResult {
    device(ordinal, |_| {
        if out.is_null() {
            return InvalidValue;
        }
        unsafe { out.write(ordinal) };
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetName(
    out: *mut c_char,
    len: c_int,
    ordinal: c_int,
) -> CuResult {
    device(ordinal, |info| {
        if out.is_null() || len <= 0 {
            return InvalidValue;
        }
        let bytes = info.name.as_bytes();
        let copied = bytes.len().min(len as usize - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), copied);
            out.add(copied).write(0);
        }
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceTotalMem_v2(out: *mut usize, ordinal: c_int) -> CuResult {
    device(ordinal, |info| {
        if out.is_null() {
            return InvalidValue;
        }
        unsafe { out.write(info.total_memory as usize) };
        Success
    })
}

fn attribute(info: &DeviceInfo, attr: c_int) -> Result<c_int, CuResult> {
    let value = match attr {
        1 => info.max_workgroup_size,
        2..=4 => info.max_workgroup_dimensions[(attr - 2) as usize],
        5..=7 => info.max_grid_dimensions[(attr - 5) as usize],
        8 => info.max_shared_local_memory,
        _ => return Err(NotSupported),
    };
    // CUDA uses a signed int even where the backend can represent larger limits.
    Ok(value.min(c_int::MAX as u32) as c_int)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetAttribute(
    out: *mut c_int,
    attr: c_int,
    ordinal: c_int,
) -> CuResult {
    device(ordinal, |info| {
        if out.is_null() {
            return InvalidValue;
        }
        match attribute(info, attr) {
            Ok(value) => {
                unsafe { out.write(value) };
                Success
            }
            Err(error) => error,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_reserved_init_flags_without_a_driver() {
        assert_eq!(cuInit(1), InvalidValue);
    }

    #[test]
    fn device_queries_require_initialization() {
        let mut out = -1;
        assert_eq!(unsafe { cuDeviceGetCount(&mut out) }, NotInitialized);
        assert_eq!(out, -1);
    }
}

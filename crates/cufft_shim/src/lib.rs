//! cuFFT surface. There is no FFT backend on the Intel path yet: every plan, exec and
//! query entry point returns `CUFFT_NOT_SUPPORTED` (16) (see `stubs.rs`), so an
//! application that imports cuFFT loads and then gets a clean error when it uses it.
//! Version queries are answered.
#![allow(non_snake_case)]

mod stubs;

pub const CUFFT_SUCCESS: i32 = 0;
pub const CUFFT_INVALID_VALUE: i32 = 4;
pub const CUFFT_NOT_SUPPORTED: i32 = 16;

/// Reported cuFFT version (11.4.0 -> 11400).
pub const VERSION: i32 = 11400;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetVersion(version: *mut i32) -> i32 {
    if version.is_null() {
        return CUFFT_INVALID_VALUE;
    }
    unsafe { version.write(VERSION) };
    CUFFT_SUCCESS
}

/// `libraryPropertyType`: MAJOR_VERSION 0, MINOR_VERSION 1, PATCH_LEVEL 2.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetProperty(kind: i32, value: *mut i32) -> i32 {
    if value.is_null() {
        return CUFFT_INVALID_VALUE;
    }
    let field = match kind {
        0 => VERSION / 1000,
        1 => (VERSION / 100) % 10,
        2 => VERSION % 100,
        _ => return CUFFT_INVALID_VALUE,
    };
    unsafe { value.write(field) };
    CUFFT_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;

    #[test]
    fn version_queries() {
        let mut v = 0;
        assert_eq!(unsafe { cufftGetVersion(&mut v) }, 0);
        assert_eq!(v, 11400);
        assert_eq!(unsafe { cufftGetVersion(null_mut()) }, CUFFT_INVALID_VALUE);
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        assert_eq!(unsafe { cufftGetProperty(0, &mut major) }, 0);
        assert_eq!(unsafe { cufftGetProperty(1, &mut minor) }, 0);
        assert_eq!(unsafe { cufftGetProperty(2, &mut patch) }, 0);
        assert_eq!((major, minor, patch), (11, 4, 0));
        assert_eq!(unsafe { cufftGetProperty(9, &mut patch) }, CUFFT_INVALID_VALUE);
    }

    #[test]
    fn plans_and_transforms_report_not_supported() {
        assert_eq!(stubs::cufftCreate(), CUFFT_NOT_SUPPORTED);
        assert_eq!(stubs::cufftPlan1d(), CUFFT_NOT_SUPPORTED);
        assert_eq!(stubs::cufftPlanMany(), CUFFT_NOT_SUPPORTED);
        assert_eq!(stubs::cufftExecC2C(), CUFFT_NOT_SUPPORTED);
        assert_eq!(stubs::cufftDestroy(), CUFFT_NOT_SUPPORTED);
    }
}

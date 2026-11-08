//! cuSOLVER surface. There is no dense/sparse solver backend on the Intel path: every
//! entry point PyTorch imports (`cusolverDn*`, `cusolverSp*`) returns
//! `CUSOLVER_STATUS_NOT_SUPPORTED` (9) (see `stubs.rs`), so the DLL loads and the first
//! solver call fails cleanly. Version queries are answered.
#![allow(non_snake_case)]

mod stubs;

pub const CUSOLVER_STATUS_SUCCESS: i32 = 0;
pub const CUSOLVER_STATUS_INVALID_VALUE: i32 = 3;
pub const CUSOLVER_STATUS_NOT_SUPPORTED: i32 = 9;

/// Reported cuSOLVER version (11.6.4 -> 11604).
pub const VERSION: i32 = 11604;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusolverGetVersion(version: *mut i32) -> i32 {
    if version.is_null() {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    unsafe { version.write(VERSION) };
    CUSOLVER_STATUS_SUCCESS
}

/// `libraryPropertyType`: MAJOR_VERSION 0, MINOR_VERSION 1, PATCH_LEVEL 2.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusolverGetProperty(kind: i32, value: *mut i32) -> i32 {
    if value.is_null() {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    let field = match kind {
        0 => VERSION / 1000,
        1 => (VERSION / 100) % 10,
        2 => VERSION % 100,
        _ => return CUSOLVER_STATUS_INVALID_VALUE,
    };
    unsafe { value.write(field) };
    CUSOLVER_STATUS_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;

    #[test]
    fn versions() {
        let mut v = 0;
        assert_eq!(unsafe { cusolverGetVersion(&mut v) }, 0);
        assert_eq!(v, 11604);
        assert_eq!(unsafe { cusolverGetVersion(null_mut()) }, CUSOLVER_STATUS_INVALID_VALUE);
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        unsafe {
            cusolverGetProperty(0, &mut major);
            cusolverGetProperty(1, &mut minor);
            cusolverGetProperty(2, &mut patch);
        }
        assert_eq!((major, minor, patch), (11, 6, 4));
        assert_eq!(unsafe { cusolverGetProperty(7, &mut patch) }, CUSOLVER_STATUS_INVALID_VALUE);
    }

    #[test]
    fn solvers_report_not_supported() {
        assert_eq!(stubs::cusolverDnCreate(), CUSOLVER_STATUS_NOT_SUPPORTED);
        assert_eq!(stubs::cusolverDnSgesvd(), CUSOLVER_STATUS_NOT_SUPPORTED);
        assert_eq!(stubs::cusolverDnDestroy(), CUSOLVER_STATUS_NOT_SUPPORTED);
    }
}

//! cuSPARSE surface. There is no sparse backend on the Intel path: handle creation,
//! descriptors and every kernel return `CUSPARSE_STATUS_NOT_SUPPORTED` (10) (see
//! `stubs.rs`), so applications load the DLL and fail cleanly at first use. Version and
//! error-string queries are answered.
#![allow(non_snake_case)]

mod stubs;

use std::ffi::c_char;

pub const CUSPARSE_STATUS_SUCCESS: i32 = 0;
pub const CUSPARSE_STATUS_INVALID_VALUE: i32 = 3;
pub const CUSPARSE_STATUS_NOT_SUPPORTED: i32 = 10;

/// Reported cuSPARSE version (12.5.9 -> 12509).
pub const VERSION: i32 = 12509;

pub fn status_name(status: i32) -> &'static std::ffi::CStr {
    match status {
        0 => c"CUSPARSE_STATUS_SUCCESS",
        1 => c"CUSPARSE_STATUS_NOT_INITIALIZED",
        2 => c"CUSPARSE_STATUS_ALLOC_FAILED",
        3 => c"CUSPARSE_STATUS_INVALID_VALUE",
        4 => c"CUSPARSE_STATUS_ARCH_MISMATCH",
        5 => c"CUSPARSE_STATUS_MAPPING_ERROR",
        6 => c"CUSPARSE_STATUS_EXECUTION_FAILED",
        7 => c"CUSPARSE_STATUS_INTERNAL_ERROR",
        8 => c"CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
        9 => c"CUSPARSE_STATUS_ZERO_PIVOT",
        10 => c"CUSPARSE_STATUS_NOT_SUPPORTED",
        11 => c"CUSPARSE_STATUS_INSUFFICIENT_RESOURCES",
        _ => c"unrecognized status code",
    }
}

pub fn status_message(status: i32) -> &'static std::ffi::CStr {
    match status {
        0 => c"success",
        1 => c"the library was not initialized",
        2 => c"resource allocation failed",
        3 => c"an invalid value was passed",
        4 => c"the device architecture is not supported",
        5 => c"mapping error",
        6 => c"the kernel failed to execute",
        7 => c"internal error",
        8 => c"the matrix type is not supported",
        9 => c"zero pivot",
        10 => c"the operation is not supported",
        11 => c"insufficient resources",
        _ => c"unrecognized status code",
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cusparseGetErrorName(status: i32) -> *const c_char {
    status_name(status).as_ptr()
}

#[unsafe(no_mangle)]
pub extern "system" fn cusparseGetErrorString(status: i32) -> *const c_char {
    status_message(status).as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetVersion(_handle: *mut std::ffi::c_void, version: *mut i32) -> i32 {
    if version.is_null() {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    unsafe { version.write(VERSION) };
    CUSPARSE_STATUS_SUCCESS
}

/// `libraryPropertyType`: MAJOR_VERSION 0, MINOR_VERSION 1, PATCH_LEVEL 2.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetProperty(kind: i32, value: *mut i32) -> i32 {
    if value.is_null() {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    let field = match kind {
        0 => VERSION / 1000,
        1 => (VERSION / 100) % 10,
        2 => VERSION % 100,
        _ => return CUSPARSE_STATUS_INVALID_VALUE,
    };
    unsafe { value.write(field) };
    CUSPARSE_STATUS_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::ptr::null_mut;

    #[test]
    fn names_and_strings() {
        let name = |s| unsafe { CStr::from_ptr(cusparseGetErrorName(s)) }.to_str().unwrap().to_owned();
        assert_eq!(name(0), "CUSPARSE_STATUS_SUCCESS");
        assert_eq!(name(10), "CUSPARSE_STATUS_NOT_SUPPORTED");
        assert_eq!(name(99), "unrecognized status code");
        assert!(!cusparseGetErrorString(7).is_null());
    }

    #[test]
    fn versions() {
        let mut v = 0;
        assert_eq!(unsafe { cusparseGetVersion(null_mut(), &mut v) }, 0);
        assert_eq!(v, 12509);
        assert_eq!(unsafe { cusparseGetVersion(null_mut(), null_mut()) }, CUSPARSE_STATUS_INVALID_VALUE);
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        unsafe {
            cusparseGetProperty(0, &mut major);
            cusparseGetProperty(1, &mut minor);
            cusparseGetProperty(2, &mut patch);
        }
        assert_eq!((major, minor, patch), (12, 5, 9));
    }

    #[test]
    fn everything_else_reports_not_supported() {
        assert_eq!(stubs::cusparseCreate(), CUSPARSE_STATUS_NOT_SUPPORTED);
        assert_eq!(stubs::cusparseSpMM(), CUSPARSE_STATUS_NOT_SUPPORTED);
        assert_eq!(stubs::cusparseCreateCsr(), CUSPARSE_STATUS_NOT_SUPPORTED);
    }
}

//! Level-2/1 BLAS built on the verified GEMM path, plus the small configuration entry
//! points PyTorch and cuBLAS clients probe (strided and f64/complex level-1/2 live in
//! `blas_dcz`). Everything unsupported keeps returning `CUBLAS_STATUS_NOT_SUPPORTED` (15).
use super::*;

/// How a unit-stride GEMV maps onto GEMM: `C(rows x 1) = op(A)(rows x cols) * x(cols x 1)`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) struct GemvAsGemm {
    pub rows: c_int,
    pub cols: c_int,
}

/// `m`/`n` are the dimensions of A as stored; `trans` is cublasOperation_t (0 N, 1 T, 2 C).
pub(crate) fn gemv_as_gemm(trans: c_int, m: c_int, n: c_int, incx: c_int, incy: c_int) -> Result<GemvAsGemm, c_int> {
    if m < 0 || n < 0 || !(0..=2).contains(&trans) {
        return Err(INVALID_VALUE);
    }
    if incx != 1 || incy != 1 {
        return Err(NOT_SUPPORTED);
    }
    Ok(if trans == 0 { GemvAsGemm { rows: m, cols: n } } else { GemvAsGemm { rows: n, cols: m } })
}

/// y = alpha * op(A) * x + beta * y. Unit-stride vectors run through the f32 GEMM kernels;
/// other increments (including negative ones) use the strided GEMV kernel in `blas_dcz`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgemv_v2(
    handle: *mut c_void,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: c_int,
    x: *const c_float,
    incx: c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: c_int,
) -> c_int {
    let shape = match gemv_as_gemm(trans, m, n, incx, incy) {
        Ok(shape) => shape,
        Err(NOT_SUPPORTED) => {
            return unsafe {
                super::blas_dcz::gemv(handle, kernels::ptxgen::Ty::S, trans, m, n, alpha.cast(), a as u64, lda, x as u64, incx, beta.cast(), y as u64, incy)
            };
        }
        Err(code) => return code,
    };
    if handle.is_null() {
        return NOT_INITIALIZED;
    }
    if shape.rows == 0 {
        return SUCCESS;
    }
    unsafe {
        cublasSgemm_v2(handle, trans, 0, shape.rows, 1, shape.cols, alpha, a, lda, x, shape.cols.max(1), beta, y, shape.rows)
    }
}

/// Dot product with any increments, the result on the host or (device pointer mode, as
/// PyTorch uses it) in device memory: the reduction kernel in `blas_dcz`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSdot_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const c_float,
    incx: c_int,
    y: *const c_float,
    incy: c_int,
    result: *mut c_float,
) -> c_int {
    unsafe { super::blas_dcz::dot(handle, kernels::ptxgen::Ty::S, n, x as u64, incx, y as u64, incy, result.cast(), false) }
}

/// `cublasEmulationStrategy_t`: DEFAULT (0) only.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetEmulationStrategy(handle: *mut c_void, strategy: *mut c_int) -> c_int {
    if handle.is_null() {
        return NOT_INITIALIZED;
    }
    if strategy.is_null() {
        return INVALID_VALUE;
    }
    unsafe { strategy.write(0) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetEmulationStrategy(handle: *mut c_void, strategy: c_int) -> c_int {
    if handle.is_null() {
        return NOT_INITIALIZED;
    }
    if strategy == 0 { SUCCESS } else { NOT_SUPPORTED }
}

/// `cublasMath_t`-independent environment switch: only the default mode exists.
#[unsafe(no_mangle)]
pub extern "system" fn cublasSetEnvironmentMode(mode: c_int) -> c_int {
    if mode == 0 { SUCCESS } else { NOT_SUPPORTED }
}

/// There are no CPU-instruction dispatch paths to disable; the previous mask is 0.
#[unsafe(no_mangle)]
pub extern "system" fn cublasLtDisableCpuInstructionsSetMask(_mask: u32) -> u32 {
    0
}

/// The CUDA Runtime version the library pairs with (`size_t`), the one `cudart_shim` reports.
#[unsafe(no_mangle)]
pub extern "system" fn cublasGetCudartVersion() -> usize {
    12090
}

/// BLAS error handler: report the routine and the invalid parameter position, as cuBLAS does.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasXerbla(name: *const c_char, info: c_int) {
    let name = if name.is_null() { "?".into() } else { unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy() };
    eprintln!(" ** On entry to {name} parameter number {info} had an illegal value");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn gemv_maps_onto_gemm_shapes() {
        assert_eq!(gemv_as_gemm(0, 5, 3, 1, 1), Ok(GemvAsGemm { rows: 5, cols: 3 }));
        assert_eq!(gemv_as_gemm(1, 5, 3, 1, 1), Ok(GemvAsGemm { rows: 3, cols: 5 }));
        assert_eq!(gemv_as_gemm(2, 5, 3, 1, 1), Ok(GemvAsGemm { rows: 3, cols: 5 }));
        assert_eq!(gemv_as_gemm(0, 5, 3, 2, 1), Err(NOT_SUPPORTED));
        assert_eq!(gemv_as_gemm(0, 5, 3, 1, -1), Err(NOT_SUPPORTED));
        assert_eq!(gemv_as_gemm(3, 5, 3, 1, 1), Err(INVALID_VALUE));
        assert_eq!(gemv_as_gemm(0, -1, 3, 1, 1), Err(INVALID_VALUE));
    }

    #[test]
    fn invalid_calls_return_statuses_without_a_driver() {
        unsafe {
            assert_eq!(cublasSgemv_v2(null_mut(), 0, 2, 2, null(), null(), 2, null(), 1, null(), null_mut(), 1), NOT_INITIALIZED);
            assert_eq!(cublasSgemv_v2(null_mut(), 0, 2, 2, null(), null(), 2, null(), 2, null(), null_mut(), 1), NOT_INITIALIZED);
            assert_eq!(cublasSdot_v2(null_mut(), 4, null(), 1, null(), 1, null_mut()), INVALID_VALUE);
            let mut out = 1.0;
            assert_eq!(cublasSdot_v2(null_mut(), 4, null(), 1, null(), 1, &mut out), NOT_INITIALIZED);
            assert_eq!(cublasGetEmulationStrategy(null_mut(), null_mut()), NOT_INITIALIZED);
            assert_eq!(cublasSetEmulationStrategy(null_mut(), 0), NOT_INITIALIZED);
        }
        assert_eq!(cublasSetEnvironmentMode(0), SUCCESS);
        assert_eq!(cublasSetEnvironmentMode(2), NOT_SUPPORTED);
        assert_eq!(cublasLtDisableCpuInstructionsSetMask(7), 0);
        assert_eq!(cublasGetCudartVersion(), 12090);
        unsafe { cublasXerbla(c"cublasDgemm".as_ptr(), 3) };
    }

    #[test]
    fn emulation_strategy_defaults() {
        let handle = Box::into_raw(Box::new(Handle {
            stream: null_mut(),
            math_mode: 0,
            pointer_mode: POINTER_MODE_HOST,
            workspace: (0, 0),
            sm_count_target: 0,
            scratch: 0,
        }));
        let mut strategy = 9;
        unsafe {
            assert_eq!(cublasGetEmulationStrategy(handle.cast(), &mut strategy), SUCCESS);
            assert_eq!(strategy, 0);
            assert_eq!(cublasSetEmulationStrategy(handle.cast(), 1), NOT_SUPPORTED);
            assert_eq!(cublasGetEmulationStrategy(handle.cast(), null_mut()), INVALID_VALUE);
            // zero-length work completes without touching a device
            let mut dot = 5.0;
            assert_eq!(cublasSdot_v2(handle.cast(), 0, null(), 1, null(), 1, &mut dot), SUCCESS);
            assert_eq!(dot, 0.0);
            assert_eq!(cublasSgemv_v2(handle.cast(), 0, 0, 4, null(), null(), 1, null(), 1, null(), null_mut(), 1), SUCCESS);
            drop(Box::from_raw(handle));
        }
    }
}

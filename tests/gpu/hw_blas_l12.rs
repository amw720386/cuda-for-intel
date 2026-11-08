//! cublasSgemv_v2 / cublasSdot_v2 through the cuBLAS shim (api-coverage), against an f64 CPU reference.
use super::hw_common::*;
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::null_mut;

type Create = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type Gemv = unsafe extern "system" fn(*mut c_void, i32, i32, i32, *const f32, u64, i32, u64, i32, *const f32, u64, i32) -> i32;
type Dot = unsafe extern "system" fn(*mut c_void, i32, u64, i32, u64, i32, *mut f32) -> i32;

#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_sgemv_sdot_numerics() {
    let lib = unsafe { libloading::Library::new(std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("cublas64_12.dll")) }.expect("cublas64_12.dll");
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuCtxSynchronize(),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let create: libloading::Symbol<Create> = unsafe { lib.get(b"cublasCreate_v2\0") }.unwrap();
    let gemv: libloading::Symbol<Gemv> = unsafe { lib.get(b"cublasSgemv_v2\0") }.unwrap();
    let dot: libloading::Symbol<Dot> = unsafe { lib.get(b"cublasSdot_v2\0") }.unwrap();
    let mut handle = null_mut();
    assert_eq!(unsafe { create(&mut handle) }, 0, "cublasCreate_v2");
    let mut rng = Rng(777);
    let mut rnd = |len: usize| -> Vec<f32> { (0..len).map(|_| rng.range(-64, 64) as f32 / 64.0).collect() };
    let up = |v: &[f32]| -> u64 {
        let mut d = 0u64;
        assert_eq!(unsafe { cuMemAlloc_v2(&mut d, v.len() * 4) }, 0);
        assert_eq!(unsafe { cuMemcpyHtoD_v2(d, v.as_ptr().cast(), v.len() * 4) }, 0);
        d
    };
    // y = alpha op(A) x + beta y, column-major A (m x n, lda = m)
    for &(m, n) in &[(1usize, 1usize), (7, 5), (64, 33), (129, 130), (896, 17)] {
        for trans in [0i32, 1] {
            let a = rnd(m * n);
            let (xl, yl) = if trans == 0 { (n, m) } else { (m, n) };
            let x = rnd(xl);
            let y0 = rnd(yl);
            let (alpha, beta) = (0.75f32, 0.5f32);
            let (da, dx, dy) = (up(&a), up(&x), up(&y0));
            let st = unsafe { gemv(handle, trans, m as i32, n as i32, &alpha, da, m as i32, dx, 1, &beta, dy, 1) };
            assert_eq!(st, 0, "Sgemv status m={m} n={n} trans={trans}");
            assert_eq!(unsafe { cuCtxSynchronize() }, 0);
            let mut out = vec![0f32; yl];
            ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dy, yl * 4));
            for i in 0..yl {
                let mut acc = 0f64;
                for l in 0..xl {
                    let aij = if trans == 0 { a[l * m + i] } else { a[i * m + l] };
                    acc += aij as f64 * x[l] as f64;
                }
                let want = alpha as f64 * acc + beta as f64 * y0[i] as f64;
                assert!((out[i] as f64 - want).abs() <= 1e-4 * want.abs() + 1e-4, "Sgemv m={m} n={n} trans={trans} i={i}: got {} want {want}", out[i]);
            }
        }
    }
    for &len in &[1usize, 5, 256, 1000, 4097] {
        let x = rnd(len);
        let y = rnd(len);
        let (dx, dy) = (up(&x), up(&y));
        let mut r = f32::NAN;
        let st = unsafe { dot(handle, len as i32, dx, 1, dy, 1, &mut r) };
        assert_eq!(st, 0, "Sdot status n={len}");
        let want: f64 = x.iter().zip(&y).map(|(a, b)| *a as f64 * *b as f64).sum();
        assert!((r as f64 - want).abs() <= 1e-4 * want.abs() + 1e-3, "Sdot n={len}: got {r} want {want}");
    }
    println!("hw_sgemv_sdot_numerics: ok");
}

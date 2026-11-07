//! Correctness probe for cublas_shim `cfi_hgemm_tn` (FP16 TN GEMM).
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

const SOURCE: &str = include_str!("../../crates/cublas_shim/src/hgemm_tn.ptx");
const KERNEL: &std::ffi::CStr = c"cfi_hgemm_tn";

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        sign
    } else if exp == 0xff {
        sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 }
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            sign
        } else if new_exp >= 31 {
            sign | 0x7c00
        } else {
            sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
        }
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn cublas_hgemm_tn_small_matches_cpu() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        cuCtxSynchronize(),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let source = std::ffi::CString::new(SOURCE).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, KERNEL.as_ptr()));

    // Column-major: A is k×m (lda=k), B is k×n (ldb=k), C is m×n (ldc=m).
    // C = A^T * B  with alpha=1, beta=0.
    const M: usize = 4;
    const N: usize = 3;
    const K: usize = 5;
    let mut a = vec![0u16; K * M];
    let mut b = vec![0u16; K * N];
    for i in 0..M {
        for kk in 0..K {
            a[kk + i * K] = f32_to_f16_bits((i + kk) as f32 * 0.1);
        }
    }
    for j in 0..N {
        for kk in 0..K {
            b[kk + j * K] = f32_to_f16_bits((j + 1) as f32 * 0.25 + kk as f32 * 0.05);
        }
    }
    let mut expected = vec![0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc = 0f32;
            for kk in 0..K {
                acc += f16_bits_to_f32(a[kk + i * K]) * f16_bits_to_f32(b[kk + j * K]);
            }
            expected[i + j * M] = acc;
        }
    }

    let mut da = 0u64;
    let mut db = 0u64;
    let mut dc = 0u64;
    ok!(cuMemAlloc_v2(&mut da, a.len() * 2));
    ok!(cuMemAlloc_v2(&mut db, b.len() * 2));
    ok!(cuMemAlloc_v2(&mut dc, M * N * 2));
    ok!(cuMemcpyHtoD_v2(da, a.as_ptr().cast(), a.len() * 2));
    ok!(cuMemcpyHtoD_v2(db, b.as_ptr().cast(), b.len() * 2));
    // Zero C
    let zeros = vec![0u16; M * N];
    ok!(cuMemcpyHtoD_v2(dc, zeros.as_ptr().cast(), zeros.len() * 2));

    let mut a_ptr = da;
    let mut b_ptr = db;
    let mut c_ptr = dc;
    let mut alpha = 1.0f32;
    let mut beta = 0.0f32;
    let mut m_u = M as u32;
    let mut n_u = N as u32;
    let mut k_u = K as u32;
    let mut lda = K as u32;
    let mut ldb = K as u32;
    let mut ldc = M as u32;
    let mut args: [*mut c_void; 11] = [
        (&mut a_ptr as *mut u64).cast(),
        (&mut b_ptr as *mut u64).cast(),
        (&mut c_ptr as *mut u64).cast(),
        (&mut alpha as *mut f32).cast(),
        (&mut beta as *mut f32).cast(),
        (&mut m_u as *mut u32).cast(),
        (&mut n_u as *mut u32).cast(),
        (&mut k_u as *mut u32).cast(),
        (&mut lda as *mut u32).cast(),
        (&mut ldb as *mut u32).cast(),
        (&mut ldc as *mut u32).cast(),
    ];
    ok!(cuLaunchKernel(
        kernel,
        1,
        1,
        1,
        16,
        16,
        1,
        0,
        null_mut(),
        args.as_mut_ptr(),
        null_mut(),
    ));
    ok!(cuCtxSynchronize());

    let mut out = vec![0u16; M * N];
    ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dc, out.len() * 2));
    for (idx, (&got_bits, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        let got = f16_bits_to_f32(got_bits);
        assert!(
            (got - exp).abs() < 1e-2,
            "mismatch at {idx}: got={got} expected={exp}"
        );
    }

    ok!(cuMemFree_v2(da));
    ok!(cuMemFree_v2(db));
    ok!(cuMemFree_v2(dc));
    ok!(cuCtxDestroy_v2(context));
}

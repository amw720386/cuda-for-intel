//! SGEMM TN with Qwen attention QK shapes: m=256 n=11 k=64 lda=128 ldb=896 ldc=256
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

const SOURCE: &str = include_str!("../../crates/cublas_shim/src/sgemm_tn.ptx");
const KERNEL: &std::ffi::CStr = c"cfi_sgemm_tn";

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn sgemm_tn_attn_qk_matches_cpu() {
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

    const M: usize = 256;
    const N: usize = 11;
    const K: usize = 64;
    const LDA: usize = 128;
    const LDB: usize = 896;
    const LDC: usize = 256;

    let mut a = vec![0f32; LDA * M];
    let mut b = vec![0f32; LDB * N];
    for i in 0..M {
        for kk in 0..K {
            a[kk + i * LDA] = (((i + kk) % 13) as f32) * 0.02 - 0.1;
        }
    }
    for j in 0..N {
        for kk in 0..K {
            b[kk + j * LDB] = (((j + 3 * kk) % 17) as f32) * 0.015 - 0.05;
        }
    }
    let mut expected = vec![0f32; LDC * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc = 0f32;
            for kk in 0..K {
                acc += a[kk + i * LDA] * b[kk + j * LDB];
            }
            expected[i + j * LDC] = acc;
        }
    }

    let mut da = 0u64;
    let mut db = 0u64;
    let mut dc = 0u64;
    ok!(cuMemAlloc_v2(&mut da, a.len() * 4));
    ok!(cuMemAlloc_v2(&mut db, b.len() * 4));
    ok!(cuMemAlloc_v2(&mut dc, expected.len() * 4));
    ok!(cuMemcpyHtoD_v2(da, a.as_ptr().cast(), a.len() * 4));
    ok!(cuMemcpyHtoD_v2(db, b.as_ptr().cast(), b.len() * 4));
    let zeros = vec![0f32; expected.len()];
    ok!(cuMemcpyHtoD_v2(dc, zeros.as_ptr().cast(), zeros.len() * 4));

    let mut a_ptr = da;
    let mut b_ptr = db;
    let mut c_ptr = dc;
    let mut alpha = 1.0f32;
    let mut beta = 0.0f32;
    let mut m = M as u32;
    let mut n = N as u32;
    let mut k = K as u32;
    let mut lda = LDA as u32;
    let mut ldb = LDB as u32;
    let mut ldc = LDC as u32;
    let mut args: [*mut c_void; 11] = [
        (&mut a_ptr as *mut u64).cast(),
        (&mut b_ptr as *mut u64).cast(),
        (&mut c_ptr as *mut u64).cast(),
        (&mut alpha as *mut f32).cast(),
        (&mut beta as *mut f32).cast(),
        (&mut m as *mut u32).cast(),
        (&mut n as *mut u32).cast(),
        (&mut k as *mut u32).cast(),
        (&mut lda as *mut u32).cast(),
        (&mut ldb as *mut u32).cast(),
        (&mut ldc as *mut u32).cast(),
    ];
    ok!(cuLaunchKernel(
        kernel,
        m.div_ceil(16).max(1),
        n.div_ceil(16).max(1),
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
    let mut out = vec![0f32; expected.len()];
    ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dc, out.len() * 4));
    let mut max_err = 0f32;
    for (idx, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        let e = (got - exp).abs();
        if e > max_err {
            max_err = e;
        }
        assert!(
            e < 1e-4,
            "sgemm attn mismatch at {idx}: got={got} expected={exp}"
        );
    }
    eprintln!("sgemm attn max_err={max_err}");
    ok!(cuMemFree_v2(da));
    ok!(cuMemFree_v2(db));
    ok!(cuMemFree_v2(dc));
    ok!(cuCtxDestroy_v2(context));
}

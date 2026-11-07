//! Authentic generic soft_max_f32 (ncols runtime) GPU correctness.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

const SOURCE: &str = include_str!("../../examples/soft_max_f32_generic_dump.ptx");
const KERNEL: &std::ffi::CStr =
    c"_Z12soft_max_f32ILb1ELi0ELi0EfEvPKfPKT2_S1_Pf15soft_max_params";

#[repr(C)]
struct SoftMaxParams {
    nheads: i64,
    n_head_log2: u32,
    _pad: u32,
    ncols: i64,
    nrows_x: i64,
    nrows_y: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb11: i64,
    nb12: i64,
    nb13: i64,
    ne12: i64,
    ne13: i64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn soft_max_generic_ncols12_matches_cpu() {
    assert_eq!(std::mem::size_of::<SoftMaxParams>(), 128);
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

    const NE0: usize = 12;
    const NE1: usize = 1;
    const NE2: usize = 2;
    const SCALE: f32 = 0.125;
    let mut x = vec![0f32; NE0 * NE1 * NE2];
    let mut mask = vec![0f32; NE0 * NE1];
    for i in 0..x.len() {
        x[i] = 0.03 * ((i % 23) as f32) - 0.2;
    }
    for i in 0..NE0 {
        mask[i] = if i < 11 { 0.0 } else { f32::NEG_INFINITY };
    }
    let mut expected = vec![0f32; x.len()];
    for i2 in 0..NE2 {
        for i1 in 0..NE1 {
            let src = &x[i2 * NE1 * NE0 + i1 * NE0..];
            let m = &mask[i1 * NE0..];
            let dst = &mut expected[i2 * NE1 * NE0 + i1 * NE0..];
            let mut maxv = f32::NEG_INFINITY;
            for i in 0..NE0 {
                maxv = maxv.max(src[i] * SCALE + m[i]);
            }
            let mut sum = 0f32;
            for i in 0..NE0 {
                dst[i] = (src[i] * SCALE + m[i] - maxv).exp();
                sum += dst[i];
            }
            for i in 0..NE0 {
                dst[i] /= sum;
            }
        }
    }

    let mut dx = 0u64;
    let mut dmask = 0u64;
    let mut ddst = 0u64;
    ok!(cuMemAlloc_v2(&mut dx, x.len() * 4));
    ok!(cuMemAlloc_v2(&mut dmask, mask.len() * 4));
    ok!(cuMemAlloc_v2(&mut ddst, expected.len() * 4));
    ok!(cuMemcpyHtoD_v2(dx, x.as_ptr().cast(), x.len() * 4));
    ok!(cuMemcpyHtoD_v2(dmask, mask.as_ptr().cast(), mask.len() * 4));
    let zeros = vec![0f32; expected.len()];
    ok!(cuMemcpyHtoD_v2(ddst, zeros.as_ptr().cast(), zeros.len() * 4));

    let mut params = SoftMaxParams {
        nheads: NE2 as i64,
        n_head_log2: 1,
        _pad: 0,
        ncols: NE0 as i64,
        nrows_x: (NE1 * NE2) as i64,
        nrows_y: NE1 as i64,
        ne00: NE0 as i64,
        ne01: NE1 as i64,
        ne02: NE2 as i64,
        ne03: 1,
        nb11: (NE0 * 4) as i64,
        nb12: (NE0 * NE1 * 4) as i64,
        nb13: (NE0 * NE1 * 4) as i64,
        ne12: 1,
        ne13: 1,
        scale: SCALE,
        max_bias: 0.0,
        m0: 1.0,
        m1: 1.0,
    };
    let mut x_ptr = dx;
    let mut mask_ptr = dmask;
    let mut sinks_ptr = 0u64;
    let mut dst_ptr = ddst;
    let mut args: [*mut c_void; 5] = [
        (&mut x_ptr as *mut u64).cast(),
        (&mut mask_ptr as *mut u64).cast(),
        (&mut sinks_ptr as *mut u64).cast(),
        (&mut dst_ptr as *mut u64).cast(),
        (&mut params as *mut SoftMaxParams).cast(),
    ];
    let shmem = (((NE0 + 31) & !31) + 32) * 4;
    ok!(cuLaunchKernel(
        kernel,
        NE1 as u32,
        NE2 as u32,
        1,
        32,
        1,
        1,
        shmem as u32,
        null_mut(),
        args.as_mut_ptr(),
        null_mut(),
    ));
    ok!(cuCtxSynchronize());
    let mut out = vec![0f32; expected.len()];
    ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ddst, out.len() * 4));
    let mut max_err = 0f32;
    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        let e = (got - exp).abs();
        if e > max_err {
            max_err = e;
        }
        assert!(
            e < 5e-4 || (got == 0.0 && exp == 0.0),
            "soft_max generic mismatch at {i}: got={got} expected={exp} max_err_so_far={max_err}"
        );
    }
    eprintln!("soft_max generic max_err={max_err} out[0]={}", out[0]);
    ok!(cuMemFree_v2(dx));
    ok!(cuMemFree_v2(dmask));
    ok!(cuMemFree_v2(ddst));
    ok!(cuCtxDestroy_v2(context));
}

use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const SOURCE: &str = include_str!("../../examples/ggml_rope_norm_f32.ptx");
const ROPE: &std::ffi::CStr = c"_Z13rope_norm_f32PKfPfiiiiiPKif";

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_rope_norm_f32() {
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
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let source = std::ffi::CString::new(SOURCE).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, ROPE.as_ptr()));

    const NE00: usize = 32;
    const NE01: usize = 4;
    const N_DIMS: usize = 32;
    const SEQ: usize = 2;
    let theta_scale = 10_000f32.powf(-2.0 / NE00 as f32);
    let positions: [i32; SEQ] = [0, 3];
    let mut input = vec![0.0f32; NE00 * NE01 * SEQ];
    let mut rng = 0xC0FF_EEu32;
    for v in &mut input {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = (rng >> 9) as f32 * (1.0 / 8_388_608.0) - 0.5;
    }
    let mut expected = input.clone();
    for i2 in 0..SEQ {
        for i1 in 0..NE01 {
            for i0 in (0..N_DIMS).step_by(2) {
                let theta = positions[i2] as f32 * theta_scale.powf(i0 as f32 * 0.5);
                let (s, c) = theta.sin_cos();
                let ix = i0 + i1 * NE00 + i2 * NE00 * NE01;
                let x0 = input[ix];
                let x1 = input[ix + 1];
                expected[ix] = x0 * c - x1 * s;
                expected[ix + 1] = x0 * s + x1 * c;
            }
        }
    }

    let bytes = input.len() * size_of::<f32>();
    let pos_bytes = positions.len() * size_of::<i32>();
    let (mut src, mut dst, mut pos) = (0u64, 0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut src, bytes));
    ok!(cuMemAlloc_v2(&mut dst, bytes));
    ok!(cuMemAlloc_v2(&mut pos, pos_bytes));
    ok!(cuMemcpyHtoD_v2(src, input.as_ptr().cast(), bytes));
    ok!(cuMemcpyHtoD_v2(pos, positions.as_ptr().cast(), pos_bytes));

    let mut ne00 = NE00 as i32;
    let mut ne01 = NE01 as i32;
    let mut s01 = NE00 as i32;
    let mut s02 = (NE00 * NE01) as i32;
    let mut n_dims = N_DIMS as i32;
    let mut theta = theta_scale;
    let mut params = [
        from_mut(&mut src).cast::<c_void>(),
        from_mut(&mut dst).cast::<c_void>(),
        from_mut(&mut ne00).cast(),
        from_mut(&mut ne01).cast(),
        from_mut(&mut s01).cast(),
        from_mut(&mut s02).cast(),
        from_mut(&mut n_dims).cast(),
        from_mut(&mut pos).cast(),
        from_mut(&mut theta).cast(),
    ];
    let rows = (NE01 * SEQ) as u32;
    ok!(cuLaunchKernel(
        kernel,
        rows,
        1,
        1,
        1,
        16,
        1,
        0,
        null_mut(),
        params.as_mut_ptr(),
        null_mut(),
    ));
    let mut actual = vec![0.0f32; input.len()];
    ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), dst, bytes));
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        let tol = 5.0e-3f32 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "mismatch at {i}: got={got} want={want} tol={tol}"
        );
    }
    ok!(cuMemFree_v2(src));
    ok!(cuMemFree_v2(dst));
    ok!(cuMemFree_v2(pos));
    ok!(cuCtxDestroy_v2(context));
}

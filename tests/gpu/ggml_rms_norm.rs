use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const SOURCE: &str = include_str!("../../examples/ggml_rms_norm_f32_32.ptx");
const RMS: &std::ffi::CStr = c"_Z12rms_norm_f32ILi32EEvPKfPfiiilllf";

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_rms_norm_f32_li32() {
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
    ok!(cuModuleGetFunction(&mut kernel, module, RMS.as_ptr()));

    const ROWS: usize = 4;
    const COLS: usize = 32;
    const EPS: f32 = 1.0e-5;
    let mut input = Vec::with_capacity(ROWS * COLS);
    let mut rng = 0xA11CEu32;
    for _ in 0..ROWS * COLS {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        input.push((rng >> 9) as f32 * (1.0 / 8_388_608.0) - 0.5);
    }
    let mut expected = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        let slice = &input[row * COLS..(row + 1) * COLS];
        let mean_sq = slice.iter().map(|x| x * x).sum::<f32>() / COLS as f32;
        // Match PTX rsqrt.approx.f32 precision loosely via f32 recip sqrt.
        let inv = (mean_sq + EPS).sqrt().recip();
        for (dst, src) in expected[row * COLS..(row + 1) * COLS]
            .iter_mut()
            .zip(slice)
        {
            *dst = *src * inv;
        }
    }

    let bytes = input.len() * size_of::<f32>();
    let (mut src, mut dst) = (0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut src, bytes));
    ok!(cuMemAlloc_v2(&mut dst, bytes));
    ok!(cuMemcpyHtoD_v2(src, input.as_ptr().cast(), bytes));

    let mut ncols = COLS as u32;
    let mut nchannels = 1u32;
    let mut nsamples = 1u32;
    let mut stride_row = COLS as i64;
    let mut stride_channel = (ROWS * COLS) as i64;
    let mut stride_sample = (ROWS * COLS) as i64;
    let mut eps = EPS;
    let mut params = [
        from_mut(&mut src).cast::<c_void>(),
        from_mut(&mut dst).cast::<c_void>(),
        from_mut(&mut ncols).cast(),
        from_mut(&mut nchannels).cast(),
        from_mut(&mut nsamples).cast(),
        from_mut(&mut stride_row).cast(),
        from_mut(&mut stride_channel).cast(),
        from_mut(&mut stride_sample).cast(),
        from_mut(&mut eps).cast(),
    ];
    ok!(cuLaunchKernel(
        kernel,
        ROWS as u32,
        1,
        1,
        32,
        1,
        1,
        0,
        null_mut(),
        params.as_mut_ptr(),
        null_mut(),
    ));
    let mut actual = vec![0.0f32; ROWS * COLS];
    ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), dst, bytes));
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        // rsqrt.approx is ~ULP-imprecise; allow a modest relative window.
        let tol = 2.0e-3f32 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "mismatch at {i}: got={got} want={want} tol={tol}"
        );
    }
    ok!(cuMemFree_v2(src));
    ok!(cuMemFree_v2(dst));
    ok!(cuCtxDestroy_v2(context));
}

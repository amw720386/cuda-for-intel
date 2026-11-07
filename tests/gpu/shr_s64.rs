use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn shr_s64_signed_div32() {
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
    let ptx = std::ffi::CString::new(include_str!("../../examples/shr_s64_probe.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"shr_s64_probe".as_ptr()));
    let input: Vec<i64> = (-16..48).collect();
    let n = input.len();
    let mut dx = 0u64;
    let mut dy = 0u64;
    ok!(cuMemAlloc_v2(&mut dx, n * 8));
    ok!(cuMemAlloc_v2(&mut dy, n * 8));
    ok!(cuMemcpyHtoD_v2(dx, input.as_ptr().cast(), n * 8));
    let mut n_u = n as u32;
    let mut params = [
        from_mut(&mut dx).cast::<c_void>(),
        from_mut(&mut dy).cast::<c_void>(),
        from_mut(&mut n_u).cast::<c_void>(),
    ];
    ok!(cuLaunchKernel(
        kernel,
        1,
        1,
        1,
        n_u,
        1,
        1,
        0,
        null_mut(),
        params.as_mut_ptr(),
        null_mut(),
    ));
    let sync = unsafe { cuCtxSynchronize() };
    assert_eq!(sync, 0, "sync -> {sync}");
    let mut got = vec![0i64; n];
    ok!(cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), dy, n * 8));
    for (i, (&x, &y)) in input.iter().zip(got.iter()).enumerate() {
        // PTX idiom: (x + (x<0 ? 31 : 0)) >> 5 — floor division by 32.
        let exp = if x >= 0 { x >> 5 } else { (x + 31) >> 5 };
        assert_eq!(y, exp, "i={i} x={x}");
    }
    ok!(cuCtxDestroy_v2(context));
}

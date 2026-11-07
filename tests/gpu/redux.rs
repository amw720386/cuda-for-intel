use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn redux_sync_add_sums_warp() {
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
    let ptx = std::ffi::CString::new(include_str!("../../examples/redux_add.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"redux_add".as_ptr()));

    for count in [32usize, 64] {
        let input: Vec<i32> = (0..count as i32).map(|i| (i & 31) - 15).collect();
        let bytes = count * 4;
        let (mut src, mut dst) = (0u64, 0u64);
        ok!(cuMemAlloc_v2(&mut src, bytes));
        ok!(cuMemAlloc_v2(&mut dst, bytes));
        ok!(cuMemcpyHtoD_v2(src, input.as_ptr().cast(), bytes));
        let mut n = count as u32;
        let mut params = [
            from_mut(&mut src).cast::<c_void>(),
            from_mut(&mut dst).cast::<c_void>(),
            from_mut(&mut n).cast::<c_void>(),
        ];
        let launch = unsafe {
            cuLaunchKernel(
                kernel,
                1,
                1,
                1,
                count as u32,
                1,
                1,
                0,
                null_mut(),
                params.as_mut_ptr(),
                null_mut(),
            )
        };
        assert_eq!(launch, 0, "launch count={count}");
        let sync = unsafe { cuCtxSynchronize() };
        assert_eq!(sync, 0, "sync count={count} -> {sync}");
        let mut actual = vec![0i32; count];
        ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), dst, bytes));
        for warp in 0..(count / 32) {
            let base = warp * 32;
            let expected: i32 = input[base..base + 32].iter().sum();
            for lane in 0..32 {
                assert_eq!(
                    actual[base + lane], expected,
                    "count={count} warp={warp} lane={lane}"
                );
            }
        }
        ok!(cuMemFree_v2(src));
        ok!(cuMemFree_v2(dst));
    }
    ok!(cuCtxDestroy_v2(context));
}

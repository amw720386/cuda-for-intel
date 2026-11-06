use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn shfl_sync_bfly_reverses_warp_lanes() {
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
    let ptx = std::ffi::CString::new(include_str!("../../examples/shfl_bfly.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"shfl_bfly".as_ptr()
    ));

    // Cover multiple warps and a partial last warp so subgroup width 8/16/32
    // hosts still exercise CUDA's fixed 32-lane butterfly semantics.
    for count in [32usize, 64, 96, 128] {
        let input: Vec<u32> = (0..count as u32).collect();
        let bytes = count * size_of::<u32>();
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
        ok!(cuLaunchKernel(
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
        ));
        let mut actual = vec![0u32; count];
        ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), dst, bytes));
        let expected: Vec<u32> = input
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let lane = (i as u32) & 31;
                let warp = (i as u32) & !31;
                input[(warp | (lane ^ 31)) as usize]
            })
            .collect();
        assert_eq!(actual, expected, "count={count}");
        ok!(cuMemFree_v2(src));
        ok!(cuMemFree_v2(dst));
    }
    ok!(cuCtxDestroy_v2(context));
}

use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const SOURCE: &str = include_str!("../../examples/ggml_dp4a_smoke.ptx");
const KERNEL: &std::ffi::CStr = c"_Z10dp4a_smokePKiS0_Pii";

fn host_dp4a(a: i32, b: i32, c: i32) -> i32 {
    let mut acc = c;
    for lane in 0..4 {
        let ai = ((a >> (8 * lane)) as i8) as i32;
        let bi = ((b >> (8 * lane)) as i8) as i32;
        acc = acc.wrapping_add(ai.wrapping_mul(bi));
    }
    acc
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_dp4a_smoke() {
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
    ok!(cuModuleGetFunction(&mut kernel, module, KERNEL.as_ptr()));

    const N: usize = 8;
    let a: [i32; N] = [
        0x0102_0304,
        0x7f80_81ffu32 as i32,
        0x0000_0000,
        0xffff_ffffu32 as i32,
        0x0a0b_0c0d,
        0x1122_3344,
        0x8080_8080u32 as i32,
        0x0101_0101,
    ];
    let b: [i32; N] = [
        0x0403_0201,
        0x01ff_807fu32 as i32,
        0x0101_0101,
        0x0202_0202,
        0x0100_ffffu32 as i32,
        0x4433_2211,
        0x0101_0101,
        0x8080_8080u32 as i32,
    ];
    let mut expected = [0i32; N];
    for i in 0..N {
        expected[i] = host_dp4a(a[i], b[i], 0);
    }

    let bytes = N * size_of::<i32>();
    let (mut da, mut db, mut ddst) = (0u64, 0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut da, bytes));
    ok!(cuMemAlloc_v2(&mut db, bytes));
    ok!(cuMemAlloc_v2(&mut ddst, bytes));
    ok!(cuMemcpyHtoD_v2(da, a.as_ptr().cast(), bytes));
    ok!(cuMemcpyHtoD_v2(db, b.as_ptr().cast(), bytes));

    let mut n = N as i32;
    let mut params = [
        from_mut(&mut da).cast::<c_void>(),
        from_mut(&mut db).cast(),
        from_mut(&mut ddst).cast(),
        from_mut(&mut n).cast(),
    ];
    ok!(cuLaunchKernel(
        kernel,
        1,
        1,
        1,
        N as u32,
        1,
        1,
        0,
        null_mut(),
        params.as_mut_ptr(),
        null_mut(),
    ));
    let mut actual = [0i32; N];
    ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), ddst, bytes));
    assert_eq!(actual, expected);
    ok!(cuMemFree_v2(da));
    ok!(cuMemFree_v2(db));
    ok!(cuMemFree_v2(ddst));
    ok!(cuCtxDestroy_v2(context));
}

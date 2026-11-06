use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const SOURCE: &str = include_str!("../../examples/ggml_half_roundtrip.ptx");
const KERNEL: &std::ffi::CStr = c"_Z14half_roundtripPK6__halfPfi";

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

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_half_roundtrip_f16() {
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

    const N: usize = 16;
    let values: [f32; N] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        3.1415927,
        -2.5,
        65504.0,
        1.0e-4,
        42.0,
        -42.0,
        0.125,
        1024.0,
        0.00390625,
        -0.00390625,
    ];
    let mut halfs = [0u16; N];
    let mut expected = [0.0f32; N];
    for i in 0..N {
        let h = f32_to_f16_bits(values[i]);
        halfs[i] = h;
        let sign = u32::from(h & 0x8000) << 16;
        let exp = u32::from((h >> 10) & 0x1f);
        let mant = u32::from(h & 0x3ff);
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut m = mant;
                let mut e = 1u32;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                sign | ((e + (127 - 15)) << 23) | (m << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (mant << 13)
        } else {
            sign | ((exp + (127 - 15)) << 23) | (mant << 13)
        };
        expected[i] = f32::from_bits(bits);
    }

    let src_bytes = N * size_of::<u16>();
    let dst_bytes = N * size_of::<f32>();
    let (mut src, mut dst) = (0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut src, src_bytes));
    ok!(cuMemAlloc_v2(&mut dst, dst_bytes));
    ok!(cuMemcpyHtoD_v2(src, halfs.as_ptr().cast(), src_bytes));

    let mut n = N as i32;
    let mut params = [
        from_mut(&mut src).cast::<c_void>(),
        from_mut(&mut dst).cast(),
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
    let mut actual = [0.0f32; N];
    ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), dst, dst_bytes));
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        let tol = 1.0e-6f32 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol || (got.to_bits() == want.to_bits()),
            "mismatch at {i}: got={got} ({:08x}) want={want} ({:08x})",
            got.to_bits(),
            want.to_bits()
        );
    }
    ok!(cuMemFree_v2(src));
    ok!(cuMemFree_v2(dst));
    ok!(cuCtxDestroy_v2(context));
}

use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

const F32_TO_F16: &str = include_str!("../../examples/convert_unary_half_dump.ptx");
const F16_TO_F32: &str = include_str!("../../examples/convert_unary_half_to_f32_dump.ptx");
const K_F32_TO_F16: &std::ffi::CStr = c"_Z23convert_unary_cont_vec4If6__halfEvPKvPT0_x";
const K_F16_TO_F32: &std::ffi::CStr = c"_Z23convert_unary_cont_vec4I6__halffEvPKvPT0_x";

fn f32_to_f16_bits_rn(v: f32) -> u16 {
    // Match PTX `cvt.rn.f16.f32` (round-to-nearest-even).
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let mut e = exp - 127 + 15;
    let mut m = mant;
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        m |= 0x80_0000;
        let shift = (1 - e) as u32;
        let rounded = (m + (1 << (shift - 1)) - if (m >> shift) & 1 == 0 { 1 } else { 0 }) >> shift;
        return sign | (rounded as u16);
    }
    if e >= 31 {
        return sign | 0x7c00;
    }
    // Round the 13 discarded mantissa bits.
    let round_bit = (m >> 12) & 1;
    let sticky = m & 0xfff;
    let mut keep = m >> 13;
    if round_bit != 0 && (sticky != 0 || (keep & 1) != 0) {
        keep += 1;
        if keep == 0x400 {
            keep = 0;
            e += 1;
            if e >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | (keep as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
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
    f32::from_bits(bits)
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn convert_unary_f32_half_roundtrip() {
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

    let src_to_h = std::ffi::CString::new(F32_TO_F16).unwrap();
    let mut mod_to_h = null_mut();
    ok!(cuModuleLoadData(&mut mod_to_h, src_to_h.as_ptr().cast()));
    let mut k_to_h = null_mut();
    ok!(cuModuleGetFunction(&mut k_to_h, mod_to_h, K_F32_TO_F16.as_ptr()));

    let src_to_f = std::ffi::CString::new(F16_TO_F32).unwrap();
    let mut mod_to_f = null_mut();
    ok!(cuModuleLoadData(&mut mod_to_f, src_to_f.as_ptr().cast()));
    let mut k_to_f = null_mut();
    ok!(cuModuleGetFunction(&mut k_to_f, mod_to_f, K_F16_TO_F32.as_ptr()));

    // Kernel processes vec4 elements; nelements is in vector units.
    const N_F32: usize = 64;
    const N_VEC: i64 = (N_F32 / 4) as i64;
    let mut input = vec![0f32; N_F32];
    for i in 0..N_F32 {
        input[i] = ((i % 19) as f32) * 0.07 - 0.4;
    }
    let mut expected_half = vec![0u16; N_F32];
    for i in 0..N_F32 {
        expected_half[i] = f32_to_f16_bits_rn(input[i]);
    }

    let mut d_f32 = 0u64;
    let mut d_f16 = 0u64;
    let mut d_back = 0u64;
    ok!(cuMemAlloc_v2(&mut d_f32, N_F32 * 4));
    ok!(cuMemAlloc_v2(&mut d_f16, N_F32 * 2));
    ok!(cuMemAlloc_v2(&mut d_back, N_F32 * 4));
    ok!(cuMemcpyHtoD_v2(d_f32, input.as_ptr().cast(), N_F32 * 4));

    let mut src = d_f32;
    let mut dst = d_f16;
    let mut ne = N_VEC;
    let mut args: [*mut c_void; 3] = [
        (&mut src as *mut u64).cast(),
        (&mut dst as *mut u64).cast(),
        (&mut ne as *mut i64).cast(),
    ];
    ok!(cuLaunchKernel(
        k_to_h,
        ((N_VEC as u32) + 255) / 256,
        1,
        1,
        256,
        1,
        1,
        0,
        null_mut(),
        args.as_mut_ptr(),
        null_mut(),
    ));
    ok!(cuCtxSynchronize());
    let mut got_half = vec![0u16; N_F32];
    ok!(cuMemcpyDtoH_v2(got_half.as_mut_ptr().cast(), d_f16, N_F32 * 2));
    for (i, (&got, &exp)) in got_half.iter().zip(expected_half.iter()).enumerate() {
        assert_eq!(got, exp, "f32->f16 mismatch at {i}: got={got:#x} exp={exp:#x}");
    }

    src = d_f16;
    dst = d_back;
    ne = N_VEC;
    args = [
        (&mut src as *mut u64).cast(),
        (&mut dst as *mut u64).cast(),
        (&mut ne as *mut i64).cast(),
    ];
    ok!(cuLaunchKernel(
        k_to_f,
        ((N_VEC as u32) + 255) / 256,
        1,
        1,
        256,
        1,
        1,
        0,
        null_mut(),
        args.as_mut_ptr(),
        null_mut(),
    ));
    ok!(cuCtxSynchronize());
    let mut got_f32 = vec![0f32; N_F32];
    ok!(cuMemcpyDtoH_v2(got_f32.as_mut_ptr().cast(), d_back, N_F32 * 4));
    for (i, (&got, &h)) in got_f32.iter().zip(got_half.iter()).enumerate() {
        let exp = f16_bits_to_f32(h);
        assert!(
            (got - exp).abs() < 1e-6 || (got.is_nan() && exp.is_nan()),
            "f16->f32 mismatch at {i}: got={got} exp={exp}"
        );
    }

    ok!(cuMemFree_v2(d_f32));
    ok!(cuMemFree_v2(d_f16));
    ok!(cuMemFree_v2(d_back));
    ok!(cuCtxDestroy_v2(context));
}

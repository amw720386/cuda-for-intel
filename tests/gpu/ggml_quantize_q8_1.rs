//! Correctness probe for authentic GGML `quantize_q8_1` (activation path for mmvq).
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const SOURCE: &str = include_str!("../../examples/quantize_q8_1_dump.ptx");
const KERNEL: &std::ffi::CStr = c"_Z13quantize_q8_1PKfPvxxxxxj5uint3";

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

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// GGML `init_fastdiv_values(d)` â†’ uint3 {mp, L, d}.
fn init_fastdiv_values(d: u32) -> [u32; 3] {
    assert!(d > 0);
    let mut l = 0u32;
    while l < 32 && (1u32 << l) < d {
        l += 1;
    }
    let mp = ((1u64 << 32) * ((1u64 << l) - u64::from(d)) / u64::from(d) + 1) as u32;
    [mp, l, d]
}

fn host_quantize_q8_1(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % 32, 0);
    let nblocks = x.len() / 32;
    let mut out = vec![0u8; nblocks * 36];
    for ib in 0..nblocks {
        let slice = &x[ib * 32..(ib + 1) * 32];
        let amax = slice.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let mut qs = [0i8; 32];
        let mut sum = 0i32;
        for (i, &v) in slice.iter().enumerate() {
            let q = if amax == 0.0 {
                0i8
            } else {
                (v / d).round() as i32 as i8
            };
            qs[i] = q;
            sum += i32::from(q);
        }
        let d_bits = f32_to_f16_bits(d).to_le_bytes();
        let s_bits = f32_to_f16_bits(d * sum as f32).to_le_bytes();
        let base = ib * 36;
        out[base] = d_bits[0];
        out[base + 1] = d_bits[1];
        out[base + 2] = s_bits[0];
        out[base + 3] = s_bits[1];
        for (i, q) in qs.iter().enumerate() {
            out[base + 4 + i] = *q as u8;
        }
    }
    out
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_quantize_q8_1_matches_host() {
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

    const NE: usize = 64; // two q8_1 blocks
    let mut x = vec![0.0f32; NE];
    let mut rng = 0xA11C_E0DEu32;
    for v in &mut x {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = (rng >> 9) as f32 * (1.0 / 8_388_608.0) - 0.5;
    }
    let expected = host_quantize_q8_1(&x);

    let mut dx = 0u64;
    let mut dy = 0u64;
    ok!(cuMemAlloc_v2(&mut dx, NE * 4));
    ok!(cuMemAlloc_v2(&mut dy, expected.len()));
    ok!(cuMemcpyHtoD_v2(dx, x.as_ptr().cast(), NE * 4));
    let zeros = vec![0u8; expected.len()];
    ok!(cuMemcpyHtoD_v2(dy, zeros.as_ptr().cast(), zeros.len()));

    let mut ne00 = NE as i64;
    let mut s01 = NE as i64;
    let mut s02 = NE as i64;
    let mut s03 = NE as i64;
    let mut ne0 = NE as i64;
    let mut ne1 = 1u32;
    let mut ne2 = init_fastdiv_values(1);
    let mut params = [
        from_mut(&mut dx).cast::<c_void>(),
        from_mut(&mut dy).cast::<c_void>(),
        from_mut(&mut ne00).cast(),
        from_mut(&mut s01).cast(),
        from_mut(&mut s02).cast(),
        from_mut(&mut s03).cast(),
        from_mut(&mut ne0).cast(),
        from_mut(&mut ne1).cast(),
        from_mut(&mut ne2).cast(),
    ];
    // Launch exactly NE threads so every lane in the active warps reaches the
    // warp reductions (no early-exit sibling warps in the workgroup).
    ok!(cuLaunchKernel(
        kernel,
        1,
        1,
        1,
        NE as u32,
        1,
        1,
        0,
        null_mut(),
        params.as_mut_ptr(),
        null_mut(),
    ));
    let sync = unsafe { cuCtxSynchronize() };
    assert_eq!(sync, 0, "cuCtxSynchronize -> {sync}");

    let mut got = vec![0u8; expected.len()];
    ok!(cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), dy, got.len()));

    for ib in 0..(NE / 32) {
        let base = ib * 36;
        let exp_d = f16_bits_to_f32(u16::from_le_bytes([expected[base], expected[base + 1]]));
        let got_d = f16_bits_to_f32(u16::from_le_bytes([got[base], got[base + 1]]));
        let exp_s = f16_bits_to_f32(u16::from_le_bytes([
            expected[base + 2],
            expected[base + 3],
        ]));
        let got_s = f16_bits_to_f32(u16::from_le_bytes([got[base + 2], got[base + 3]]));
        let exp_qs: Vec<i8> = expected[base + 4..base + 36]
            .iter()
            .map(|&b| b as i8)
            .collect();
        let got_qs: Vec<i8> = got[base + 4..base + 36].iter().map(|&b| b as i8).collect();
        eprintln!(
            "block {ib}: d got={got_d} exp={exp_d}; s got={got_s} exp={exp_s}; qs got={got_qs:?} exp={exp_qs:?}; hdr={:02x?}",
            &got[base..base + 4]
        );
        assert!(
            (got_d - exp_d).abs() < 1e-2,
            "block {ib} d: got={got_d} expected={exp_d}"
        );
        assert_eq!(got_qs, exp_qs, "block {ib} qs mismatch");
        assert!(
            (got_s - exp_s).abs() < 5e-2,
            "block {ib} s: got={got_s} expected={exp_s}"
        );
    }

    ok!(cuMemFree_v2(dx));
    ok!(cuMemFree_v2(dy));
    ok!(cuCtxDestroy_v2(context));
}


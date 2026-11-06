use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const SOURCE: &str = include_str!("../../examples/ggml_q8_0_auth.ptx");
const KERNEL: &std::ffi::CStr = c"_Z13q8_0_auth_f32PK10block_q8_0PKfPfii";

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

fn soft_half_to_f32(h: u16) -> f32 {
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

/// Authentic GGML `block_q8_0`: 2-byte half scale + 32 qs bytes (34 bytes, no pad).
fn pack_block(d: f32, qs: &[i8; 32]) -> [u8; 34] {
    let mut out = [0u8; 34];
    let half = f32_to_f16_bits(d).to_le_bytes();
    out[0] = half[0];
    out[1] = half[1];
    for (i, &q) in qs.iter().enumerate() {
        out[2 + i] = q as u8;
    }
    out
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_q8_0_auth_f32() {
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

    const NROWS: usize = 4;
    const NCOLS: usize = 64;
    const NBLOCKS: usize = NCOLS / 32;

    let mut y = vec![0.0f32; NCOLS];
    let mut rng = 0xA11C_E0DEu32;
    for v in &mut y {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = (rng >> 9) as f32 * (1.0 / 8_388_608.0) - 0.5;
    }

    let mut x_bytes = vec![0u8; NROWS * NBLOCKS * 34];
    let mut expected = vec![0.0f32; NROWS];
    for row in 0..NROWS {
        let mut sum = 0.0f32;
        for ib in 0..NBLOCKS {
            let mut qs = [0i8; 32];
            for q in &mut qs {
                rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *q = ((rng >> 24) as i8).wrapping_sub(64);
            }
            let scale = 0.05 + (row as f32) * 0.01 + (ib as f32) * 0.002;
            let packed = pack_block(scale, &qs);
            let base = (row * NBLOCKS + ib) * 34;
            x_bytes[base..base + 34].copy_from_slice(&packed);
            let d = soft_half_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
            let mut partial = 0.0f32;
            for (i, &q) in qs.iter().enumerate() {
                partial += q as f32 * y[ib * 32 + i];
            }
            sum += d * partial;
        }
        expected[row] = sum;
    }

    let x_len = x_bytes.len();
    let y_bytes = y.len() * size_of::<f32>();
    let dst_bytes = expected.len() * size_of::<f32>();
    let (mut dx, mut dy, mut ddst) = (0u64, 0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut dx, x_len));
    ok!(cuMemAlloc_v2(&mut dy, y_bytes));
    ok!(cuMemAlloc_v2(&mut ddst, dst_bytes));
    ok!(cuMemcpyHtoD_v2(dx, x_bytes.as_ptr().cast(), x_len));
    ok!(cuMemcpyHtoD_v2(dy, y.as_ptr().cast(), y_bytes));

    let mut ncols = NCOLS as i32;
    let mut nrows = NROWS as i32;
    let mut params = [
        from_mut(&mut dx).cast::<c_void>(),
        from_mut(&mut dy).cast(),
        from_mut(&mut ddst).cast(),
        from_mut(&mut ncols).cast(),
        from_mut(&mut nrows).cast(),
    ];
    ok!(cuLaunchKernel(
        kernel,
        1,
        1,
        1,
        NROWS as u32,
        1,
        1,
        0,
        null_mut(),
        params.as_mut_ptr(),
        null_mut(),
    ));
    let mut actual = vec![0.0f32; NROWS];
    ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), ddst, dst_bytes));
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        let tol = 5.0e-4f32 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "mismatch at row {i}: got={got} want={want} tol={tol}"
        );
    }
    ok!(cuMemFree_v2(dx));
    ok!(cuMemFree_v2(dy));
    ok!(cuMemFree_v2(ddst));
    ok!(cuCtxDestroy_v2(context));
}

//! The simulator outside `cargo test`: `src/ptx/sim.rs` is a test-only module of the translator
//! and reaches one helper of the test interpreter through `super::coverage_tests::interp`.
#[path = "ptx/sim.rs"]
pub mod sim;

pub mod coverage_tests {
    pub mod interp {
        /// IEEE binary32 to binary16, round to nearest even. Same as
        /// `src/ptx/coverage_tests.rs` `interp::f32_to_f16_bits`, which is test-only.
        pub fn f32_to_f16_bits(x: f32) -> u16 {
            let b = x.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = ((b >> 23) & 0xff) as i32;
            let man = b & 0x7f_ffff;
            if exp == 0xff {
                return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
            }
            let e = exp - 127 + 15;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
            if e <= 0 {
                if e < -10 {
                    return sign;
                }
                let mant = man | 0x80_0000;
                let shift = (14 - e) as u32;
                let mut half = mant >> shift;
                let rem = mant & ((1 << shift) - 1);
                let halfway = 1 << (shift - 1);
                if rem > halfway || (rem == halfway && half & 1 == 1) {
                    half += 1;
                }
                return sign | half as u16;
            }
            let mut h = ((e as u32) << 10) | (man >> 13);
            let rem = man & 0x1fff;
            if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
                h += 1;
            }
            sign | h as u16
        }
    }
}

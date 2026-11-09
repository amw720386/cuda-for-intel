//! fp8 pair conversions (OCP E4M3 and E5M2), lowered with integer arithmetic so the result
//! does not depend on the device's half-precision denormal handling.
//!
//! * `cvt.rn.satfinite[.relu].{e4m3x2,e5m2x2}.{f32,f16x2}` rounds to nearest even. Values
//!   beyond the largest finite fp8 value, and infinities, saturate to it (448 for E4M3,
//!   57344 for E5M2) with their sign; NaN becomes `0x7f`. `.relu` turns negative results
//!   (and -0) into +0.
//! * `cvt.rn[.relu].f16x2.{e4m3x2,e5m2x2}` is exact. An E4M3 NaN becomes the canonical f16
//!   NaN `0x7fff`; E5M2 is the upper byte of an f16, so its bits shift into place.
//!
//! The first source operand (or the upper half of a packed source) fills the upper byte
//! (or upper half) of the result, as for `cvt.rn.f16x2.f32`.

use super::super::Error;
use super::super::ext::Fp8Dir;
use super::{Emitter, Kind, Value};
use rspirv::spirv::Word;

type R<T> = Result<T, Error>;

/// Format constants: mantissa bits, exponent bias and the largest finite encoding.
struct Format {
    mantissa: u32,
    bias: u32,
    max_finite: u32,
}

fn format(e5m2: bool) -> Format {
    if e5m2 {
        Format { mantissa: 2, bias: 15, max_finite: 0x7b }
    } else {
        Format { mantissa: 3, bias: 7, max_finite: 0x7e }
    }
}

impl<'a> Emitter<'a> {
    pub(super) fn emit_fp8(&mut self, dir: Fp8Dir, e5m2: bool, relu: bool, dst: &'a str, src: [&str; 4]) -> R<()> {
        let ty = self.u32_type;
        let sixteen = self.u32_const(16);
        let eight = self.u32_const(8);
        let id = match dir {
            Fp8Dir::FromF32 | Fp8Dir::FromF16x2 => {
                // f32 bit patterns of the (high, low) lanes.
                let (hi, lo) = if dir == Fp8Dir::FromF32 {
                    (self.float_bits(src[0])?, self.float_bits(src[1])?)
                } else {
                    let packed = self.integer(src[0], Kind::B32)?;
                    let upper = self.builder.shift_right_logical(ty, None, packed, sixteen)?;
                    (self.f16_bits_to_f32_bits(upper)?, self.f16_bits_to_f32_bits(packed)?)
                };
                let hi = self.f32_bits_to_fp8(hi, e5m2, relu)?;
                let lo = self.f32_bits_to_fp8(lo, e5m2, relu)?;
                let hi = self.builder.shift_left_logical(ty, None, hi, eight)?;
                self.builder.bitwise_or(ty, None, hi, lo)?
            }
            Fp8Dir::ToF16x2 => {
                let packed = self.integer(src[0], Kind::B32)?;
                let mask = self.u32_const(0xff);
                let lo = self.builder.bitwise_and(ty, None, packed, mask)?;
                let upper = self.builder.shift_right_logical(ty, None, packed, eight)?;
                let hi = self.builder.bitwise_and(ty, None, upper, mask)?;
                let lo = self.fp8_to_f16_bits(lo, e5m2, relu)?;
                let hi = self.fp8_to_f16_bits(hi, e5m2, relu)?;
                let hi = self.builder.shift_left_logical(ty, None, hi, sixteen)?;
                self.builder.bitwise_or(ty, None, hi, lo)?
            }
        };
        self.write(dst, Value { id, kind: Kind::B32 })
    }

    /// Widens the f16 in the low 16 bits of `bits` to f32 bits (exact).
    fn f16_bits_to_f32_bits(&mut self, bits: Word) -> R<Word> {
        let bits16 = self.builder.u_convert(self.u16_type, None, bits)?;
        let half = self.builder.bitcast(self.f16_type, None, bits16)?;
        let single = self.builder.f_convert(self.f32_type, None, half)?;
        Ok(self.builder.bitcast(self.u32_type, None, single)?)
    }

    /// Rounds the f32 with bit pattern `u` to an fp8 byte (round to nearest even, saturating).
    fn f32_bits_to_fp8(&mut self, u: Word, e5m2: bool, relu: bool) -> R<Word> {
        let f = format(e5m2);
        let ty = self.u32_type;
        let b = self.bool_type;
        let c = |s: &mut Self, v: u32| s.u32_const(v);
        let c31 = c(self, 31);
        let sign = self.builder.shift_right_logical(ty, None, u, c31)?;
        let abs_mask = c(self, 0x7fff_ffff);
        let a = self.builder.bitwise_and(ty, None, u, abs_mask)?;
        let c23 = c(self, 23);
        let e = self.builder.shift_right_logical(ty, None, a, c23)?;
        let frac_mask = c(self, 0x007f_ffff);
        let frac = self.builder.bitwise_and(ty, None, a, frac_mask)?;
        let implicit = c(self, 0x0080_0000);
        let mant = self.builder.bitwise_or(ty, None, frac, implicit)?;
        // `t` is the f32 biased exponent of the smallest normal fp8 value.
        let t = c(self, 128 - f.bias);
        let below = self.builder.u_less_than(b, None, e, t)?;
        let zero = c(self, 0);
        let deficit = self.builder.i_sub(ty, None, t, e)?;
        let deficit = self.builder.select(ty, None, below, deficit, zero)?;
        let excess = self.builder.i_sub(ty, None, e, t)?;
        let excess = self.builder.select(ty, None, below, zero, excess)?;
        let m = c(self, f.mantissa);
        let base = self.builder.shift_left_logical(ty, None, excess, m)?;
        // Right shift that leaves the fp8 significand (subnormal steps add to it), capped
        // where every value rounds to zero.
        let normal_shift = c(self, 23 - f.mantissa);
        let shift = self.builder.i_add(ty, None, normal_shift, deficit)?;
        let cap = c(self, 25);
        let over = self.builder.u_greater_than(b, None, shift, cap)?;
        let shift = self.builder.select(ty, None, over, cap, shift)?;
        let q = self.builder.shift_right_logical(ty, None, mant, shift)?;
        let one = c(self, 1);
        let unit = self.builder.shift_left_logical(ty, None, one, shift)?;
        let low_mask = self.builder.i_sub(ty, None, unit, one)?;
        let rem = self.builder.bitwise_and(ty, None, mant, low_mask)?;
        let half = self.builder.shift_right_logical(ty, None, unit, one)?;
        let above_half = self.builder.u_greater_than(b, None, rem, half)?;
        let tie = self.builder.i_equal(b, None, rem, half)?;
        let odd_bit = self.builder.bitwise_and(ty, None, q, one)?;
        let odd = self.builder.i_not_equal(b, None, odd_bit, zero)?;
        let tie_up = self.builder.logical_and(b, None, tie, odd)?;
        let up = self.builder.logical_or(b, None, above_half, tie_up)?;
        let inc = self.builder.select(ty, None, up, one, zero)?;
        let enc = self.builder.i_add(ty, None, base, q)?;
        let enc = self.builder.i_add(ty, None, enc, inc)?;
        let max = c(self, f.max_finite);
        let too_big = self.builder.u_greater_than(b, None, enc, max)?;
        let enc = self.builder.select(ty, None, too_big, max, enc)?;
        let seven = c(self, 7);
        let sign_bit = self.builder.shift_left_logical(ty, None, sign, seven)?;
        let signed = self.builder.bitwise_or(ty, None, enc, sign_bit)?;
        let inf = c(self, 0x7f80_0000);
        let nan = self.builder.u_greater_than(b, None, a, inf)?;
        let result = if relu {
            let negative = self.builder.i_not_equal(b, None, sign, zero)?;
            self.builder.select(ty, None, negative, zero, signed)?
        } else {
            signed
        };
        let nan_bits = c(self, 0x7f);
        Ok(self.builder.select(ty, None, nan, nan_bits, result)?)
    }

    /// Widens the fp8 byte `x` (upper bits clear) to f16 bits (exact).
    fn fp8_to_f16_bits(&mut self, x: Word, e5m2: bool, relu: bool) -> R<Word> {
        let ty = self.u32_type;
        let b = self.bool_type;
        let zero = self.u32_const(0);
        let c80 = self.u32_const(0x80);
        let sign = self.builder.bitwise_and(ty, None, x, c80)?;
        let negative = self.builder.i_not_equal(b, None, sign, zero)?;
        let c7f = self.u32_const(0x7f);
        let abs = self.builder.bitwise_and(ty, None, x, c7f)?;
        let eight = self.u32_const(8);
        let sign16 = self.builder.shift_left_logical(ty, None, sign, eight)?;
        let (magnitude, nan) = if e5m2 {
            // E5M2 is the upper byte of the f16 with the same value.
            let magnitude = self.builder.shift_left_logical(ty, None, abs, eight)?;
            let c7c = self.u32_const(0x7c);
            (magnitude, self.builder.u_greater_than(b, None, abs, c7c)?)
        } else {
            let three = self.u32_const(3);
            let exponent = self.builder.shift_right_logical(ty, None, abs, three)?;
            let seven = self.u32_const(7);
            let mant = self.builder.bitwise_and(ty, None, abs, seven)?;
            // Normal: rebias the exponent by 15 - 7 and widen the mantissa by 7 bits.
            let shifted = self.builder.shift_left_logical(ty, None, abs, seven)?;
            let rebias = self.u32_const(8 << 10);
            let normal = self.builder.i_add(ty, None, shifted, rebias)?;
            // Subnormal `mant * 2^-9` is an f16 normal: exponent field `p + 6` where `p` is
            // the index of the leading one, mantissa `mant << (3 - p)` without it.
            let four = self.u32_const(4);
            let two = self.u32_const(2);
            let one = self.u32_const(1);
            let ge4 = self.builder.u_greater_than_equal(b, None, mant, four)?;
            let ge2 = self.builder.u_greater_than_equal(b, None, mant, two)?;
            let p_low = self.builder.select(ty, None, ge2, one, zero)?;
            let p = self.builder.select(ty, None, ge4, two, p_low)?;
            let six = self.u32_const(6);
            let field = self.builder.i_add(ty, None, p, six)?;
            let ten = self.u32_const(10);
            let field = self.builder.shift_left_logical(ty, None, field, ten)?;
            let up = self.builder.i_sub(ty, None, three, p)?;
            let frac = self.builder.shift_left_logical(ty, None, mant, up)?;
            let frac = self.builder.bitwise_and(ty, None, frac, seven)?;
            let frac = self.builder.shift_left_logical(ty, None, frac, seven)?;
            let sub = self.builder.bitwise_or(ty, None, field, frac)?;
            let has_mant = self.builder.i_not_equal(b, None, mant, zero)?;
            let sub = self.builder.select(ty, None, has_mant, sub, zero)?;
            let is_sub = self.builder.i_equal(b, None, exponent, zero)?;
            let magnitude = self.builder.select(ty, None, is_sub, sub, normal)?;
            (magnitude, self.builder.i_equal(b, None, abs, c7f)?)
        };
        let value = self.builder.bitwise_or(ty, None, magnitude, sign16)?;
        let value = if relu {
            self.builder.select(ty, None, negative, zero, value)?
        } else {
            value
        };
        let canonical = self.u32_const(0x7fff);
        // E4M3 NaN (and any NaN under `.relu`) is the canonical f16 NaN; plain E5M2 keeps its bits.
        let nan_value = if e5m2 && !relu {
            self.builder.bitwise_or(ty, None, magnitude, sign16)?
        } else {
            canonical
        };
        Ok(self.builder.select(ty, None, nan, nan_value, value)?)
    }
}

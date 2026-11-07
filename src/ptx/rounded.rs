//! Round binary32 division and square root using integer significands.
//! The target need only support integer SPIR-V, even without rounded FP32 divide/sqrt.
use super::Error;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{OpenclStd100Op, Word};

const SIGN: u32 = 0x8000_0000;
const INFINITY: u32 = 0x7f80_0000;
const NAN: u32 = 0x7fc0_0000;
const HIDDEN: u32 = 1 << 23;

struct Math<'a> {
    b: &'a mut Builder,
    u32: Word,
    boolean: Word,
    opencl: Word,
}

macro_rules! binary {
    ($($name:ident : $ty:ident => $op:ident),+ $(,)?) => {$(
        fn $name(&mut self, a: Word, b: Word) -> Result<Word, Error> {
            Ok(self.b.$op(self.$ty, None, a, b)?)
        }
    )+};
}

impl<'a> Math<'a> {
    fn new(b: &'a mut Builder, opencl: Word) -> Self {
        Self {
            u32: b.type_int(32, 0),
            boolean: b.type_bool(),
            b,
            opencl,
        }
    }

    binary! {
        add: u32 => i_add, sub: u32 => i_sub,
        and: u32 => bitwise_and, or: u32 => bitwise_or, xor: u32 => bitwise_xor,
        shl: u32 => shift_left_logical, shr: u32 => shift_right_logical,
        signed_shr: u32 => shift_right_arithmetic,
        eq: boolean => i_equal, lt: boolean => u_less_than, slt: boolean => s_less_than,
        both: boolean => logical_and, either: boolean => logical_or,
    }

    fn c(&mut self, value: u32) -> Word {
        self.b.constant_bit32(self.u32, value)
    }

    fn select(&mut self, condition: Word, yes: Word, no: Word) -> Result<Word, Error> {
        Ok(self.b.select(self.u32, None, condition, yes, no)?)
    }

    fn magnitude(&mut self, bits: Word) -> Result<Word, Error> {
        let mask = self.c(!SIGN);
        self.and(bits, mask)
    }

    // Return a normalized 24-bit significand and its signed biased exponent.
    fn normalize(&mut self, abs: Word) -> Result<(Word, Word), Error> {
        let (zero, one, eight, shift23) = (self.c(0), self.c(1), self.c(8), self.c(23));
        let (fraction_mask, hidden) = (self.c(HIDDEN - 1), self.c(HIDDEN));
        let exponent = self.shr(abs, shift23)?;
        let fraction = self.and(abs, fraction_mask)?;
        let subnormal = self.eq(exponent, zero)?;
        let nonzero_fraction = self.or(fraction, one)?;
        let leading = self.b.ext_inst(
            self.u32,
            None,
            self.opencl,
            OpenclStd100Op::clz as u32,
            [Operand::IdRef(nonzero_fraction)],
        )?;
        let sub_shift = self.sub(leading, eight)?;
        let shift = self.select(subnormal, sub_shift, zero)?;
        let leading_bit = self.select(subnormal, zero, hidden)?;
        let significand = self.or(fraction, leading_bit)?;
        let significand = self.shl(significand, shift)?;
        let sub_exp = self.sub(one, shift)?;
        let exponent = self.select(subnormal, sub_exp, exponent)?;
        Ok((significand, exponent))
    }

    fn digits(
        &mut self,
        steps: Word,
        initial_value: Word,
        initial_remainder: Word,
        step: impl FnOnce(&mut Self, Word, Word, Word) -> Result<(Word, Word), Error>,
    ) -> Result<(Word, Word), Error> {
        let (zero, one) = (self.c(0), self.c(1));
        let (entry, head, body, done) = (self.b.id(), self.b.id(), self.b.id(), self.b.id());
        let (next_index, next_value, next_remainder) = (self.b.id(), self.b.id(), self.b.id());
        self.b.branch(entry)?;
        self.b.begin_block(Some(entry))?;
        self.b.branch(head)?;
        self.b.begin_block(Some(head))?;
        let index = self
            .b
            .phi(self.u32, None, [(zero, entry), (next_index, body)])?;
        let value = self
            .b
            .phi(self.u32, None, [(initial_value, entry), (next_value, body)])?;
        let remainder = self.b.phi(
            self.u32,
            None,
            [(initial_remainder, entry), (next_remainder, body)],
        )?;
        let active = self.lt(index, steps)?;
        self.b.branch_conditional(active, body, done, [])?;
        self.b.begin_block(Some(body))?;
        let (value_out, remainder_out) = step(self, index, value, remainder)?;
        self.b.i_add(self.u32, Some(next_index), index, one)?;
        self.b.copy_object(self.u32, Some(next_value), value_out)?;
        self.b
            .copy_object(self.u32, Some(next_remainder), remainder_out)?;
        self.b.branch(head)?;
        self.b.begin_block(Some(done))?;
        Ok((value, remainder))
    }

    fn sqrt(&mut self, bits: Word) -> Result<Word, Error> {
        let (zero, one, shift23, bias) = (self.c(0), self.c(1), self.c(23), self.c(127));
        let (inf, nan) = (self.c(INFINITY), self.c(NAN));
        let abs = self.magnitude(bits)?;
        let (significand, biased) = self.normalize(abs)?;
        let exponent = self.sub(biased, bias)?;
        let odd = self.and(exponent, one)?;
        let shift = self.add(shift23, odd)?;
        let (two, mask, word_bits, steps) = (self.c(2), self.c(31), self.c(32), self.c(24));
        let low = self.shl(significand, shift)?;
        let upper_shift = self.sub(word_bits, shift)?;
        let high = self.shr(significand, upper_shift)?;

        // Consume the 48-bit radicand two bits at a time. The partial remainder
        // is small enough to keep in a single word throughout the loop.
        let (root, remainder) = self.digits(steps, zero, zero, |m, index, root, remainder| {
            let pair_index = m.sub(shift23, index)?;
            let position = m.shl(pair_index, one)?;
            let low_half = m.lt(position, word_bits)?;
            let word = m.select(low_half, low, high)?;
            let position = m.and(position, mask)?;
            let pair = m.shr(word, position)?;
            let three = m.c(3);
            let pair = m.and(pair, three)?;
            let shifted = m.shl(remainder, two)?;
            let scaled = m.or(shifted, pair)?;
            let shifted_root = m.shl(root, two)?;
            let trial = m.or(shifted_root, one)?;
            let below = m.lt(scaled, trial)?;
            let difference = m.sub(scaled, trial)?;
            let remainder = m.select(below, scaled, difference)?;
            let digit = m.select(below, zero, one)?;
            let shifted_root = m.shl(root, one)?;
            let root = m.or(shifted_root, digit)?;
            Ok((root, remainder))
        })?;
        // The midpoint square is root^2 + root + 1/4, so integer input cannot tie.
        let round_up = self.lt(root, remainder)?;
        let bump = self.select(round_up, one, zero)?;
        let rounded = self.add(root, bump)?;
        let half_exp = self.signed_shr(exponent, one)?;
        let bias_minus_one = self.c(126);
        let prefix = self.add(half_exp, bias_minus_one)?;
        let prefix = self.shl(prefix, shift23)?;
        let result = self.add(prefix, rounded)?;

        let is_infinite = self.eq(abs, inf)?;
        let result = self.select(is_infinite, inf, result)?;
        let negative = self.slt(bits, zero)?;
        let is_nan = self.lt(inf, abs)?;
        let invalid = self.either(negative, is_nan)?;
        let result = self.select(invalid, nan, result)?;
        let is_zero = self.eq(abs, zero)?;
        self.select(is_zero, bits, result)
    }

    fn div(&mut self, a: Word, b: Word) -> Result<Word, Error> {
        let (zero, one, shift23) = (self.c(0), self.c(1), self.c(23));
        let (inf, nan, sign_mask) = (self.c(INFINITY), self.c(NAN), self.c(SIGN));
        let (abs_a, abs_b) = (self.magnitude(a)?, self.magnitude(b)?);
        let (num, a_exp) = self.normalize(abs_a)?;
        let (den, b_exp) = self.normalize(abs_b)?;
        let adjust = self.lt(num, den)?;
        let adjust = self.select(adjust, one, zero)?;
        let num = self.shl(num, adjust)?;
        let exp = self.sub(a_exp, b_exp)?;
        let exp = self.sub(exp, adjust)?;

        // Quantize directly at the final exponent so subnormals are rounded only once.
        let (min_normal, half_min_subnormal, subnormal_scale) = (
            self.c((-126i32) as u32),
            self.c((-150i32) as u32),
            self.c(149),
        );
        let subnormal = self.slt(exp, min_normal)?;
        let underflow = self.slt(exp, half_min_subnormal)?;
        let half_range = self.eq(exp, half_min_subnormal)?;
        let too_small = self.either(underflow, half_range)?;
        let sub_shift = self.add(exp, subnormal_scale)?;
        let sub_shift = self.select(too_small, zero, sub_shift)?;
        let shift = self.select(subnormal, sub_shift, shift23)?;
        let half_shift = self.select(half_range, one, zero)?;
        let den = self.shl(den, half_shift)?;
        // All paths are evaluated, including the zero-denominator special case.
        let den_zero = self.eq(den, zero)?;
        let den = self.select(den_zero, one, den)?;
        // The ratio is below two. Start with its integer bit, then shift in
        // one fractional bit at a time. All remainders fit in 25 bits.
        let below_one = self.lt(num, den)?;
        let difference = self.sub(num, den)?;
        let quotient = self.select(below_one, zero, one)?;
        let remainder = self.select(below_one, num, difference)?;
        let (quotient, remainder) =
            self.digits(shift, quotient, remainder, |m, _, quotient, remainder| {
                let scaled = m.shl(remainder, one)?;
                let below = m.lt(scaled, den)?;
                let difference = m.sub(scaled, den)?;
                let remainder = m.select(below, scaled, difference)?;
                let digit = m.select(below, zero, one)?;
                let prefix = m.shl(quotient, one)?;
                Ok((m.or(prefix, digit)?, remainder))
            })?;
        let twice_remainder = self.shl(remainder, one)?;
        let above_half = self.lt(den, twice_remainder)?;
        let halfway = self.eq(den, twice_remainder)?;
        let odd = self.and(quotient, one)?;
        let odd = self.eq(odd, one)?;
        let tie_round_up = self.both(halfway, odd)?;
        let round_up = self.either(above_half, tie_round_up)?;
        let bump = self.select(round_up, one, zero)?;
        let rounded = self.add(quotient, bump)?;

        let (bias_minus_one, max_exp) = (self.c(126), self.c(127));
        let prefix = self.add(exp, bias_minus_one)?;
        let prefix = self.shl(prefix, shift23)?;
        let prefix = self.select(subnormal, zero, prefix)?;
        let result = self.add(prefix, rounded)?;
        let result = self.select(underflow, zero, result)?;
        let overflow = self.slt(max_exp, exp)?;
        let result = self.select(overflow, inf, result)?;

        let (a_zero, b_zero) = (self.eq(abs_a, zero)?, self.eq(abs_b, zero)?);
        let (a_inf, b_inf) = (self.eq(abs_a, inf)?, self.eq(abs_b, inf)?);
        let zero_result = self.either(a_zero, b_inf)?;
        let inf_result = self.either(a_inf, b_zero)?;
        let result = self.select(zero_result, zero, result)?;
        let result = self.select(inf_result, inf, result)?;
        let signs = self.xor(a, b)?;
        let sign = self.and(signs, sign_mask)?;
        let result = self.or(result, sign)?;
        let both_zero = self.both(a_zero, b_zero)?;
        let both_infinite = self.both(a_inf, b_inf)?;
        let invalid = self.either(both_zero, both_infinite)?;
        let (a_nan, b_nan) = (self.lt(inf, abs_a)?, self.lt(inf, abs_b)?);
        let input_nan = self.either(a_nan, b_nan)?;
        let nan_result = self.either(input_nan, invalid)?;
        self.select(nan_result, nan, result)
    }
}

pub(super) fn div(builder: &mut Builder, opencl: Word, a: Word, b: Word) -> Result<Word, Error> {
    Math::new(builder, opencl).div(a, b)
}

pub(super) fn sqrt(builder: &mut Builder, opencl: Word, value: Word) -> Result<Word, Error> {
    Math::new(builder, opencl).sqrt(value)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Mode {
    Zero,
    Down,
    Up,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Arith {
    Add,
    Sub,
    Mul,
}

impl<'a> Math<'a> {
    /// Directed-rounding add/sub/mul. SPIR-V cannot select a rounding mode for FAdd, so
    /// compute the round-to-nearest result plus its exact error (two-sum / FMA) and step
    /// one ulp when nearest landed on the wrong side. Assumes no flush-to-zero surprises.
    fn directed(&mut self, kind: Arith, mode: Mode, a: Word, b: Word) -> Result<Word, Error> {
        let f32t = self.b.type_float(32, None);
        let b = if kind == Arith::Sub {
            let sign = self.c(SIGN);
            self.xor(b, sign)?
        } else {
            b
        };
        let fa = self.b.bitcast(f32t, None, a)?;
        let fb = self.b.bitcast(f32t, None, b)?;
        let (fs, fe) = if kind == Arith::Mul {
            let p = self.b.f_mul(f32t, None, fa, fb)?;
            let np = self.b.f_negate(f32t, None, p)?;
            let e = self.b.ext_inst(
                f32t,
                None,
                self.opencl,
                OpenclStd100Op::fma as u32,
                [fa, fb, np].map(Operand::IdRef),
            )?;
            (p, e)
        } else {
            let s = self.b.f_add(f32t, None, fa, fb)?;
            let bb = self.b.f_sub(f32t, None, s, fa)?;
            let t = self.b.f_sub(f32t, None, s, bb)?;
            let ea = self.b.f_sub(f32t, None, fa, t)?;
            let eb = self.b.f_sub(f32t, None, fb, bb)?;
            (s, self.b.f_add(f32t, None, ea, eb)?)
        };
        let s = self.b.bitcast(self.u32, None, fs)?;
        let zero_f = self.b.constant_bit32(f32t, 0);
        let e_pos = self.b.f_ord_greater_than(self.boolean, None, fe, zero_f)?;
        let e_neg = self.b.f_ord_less_than(self.boolean, None, fe, zero_f)?;
        let (sign_c, one, zero) = (self.c(SIGN), self.c(1), self.c(0));
        let s_sign = self.and(s, sign_c)?;
        let s_neg = self.eq(s_sign, sign_c)?;
        let s_mag = self.magnitude(s)?;
        let s_zero = self.eq(s_mag, zero)?;
        let inc = self.add(s, one)?;
        let dec = self.sub(s, one)?;
        let min_pos = one;
        let min_neg = self.c(SIGN | 1);
        let stepped = match mode {
            Mode::Zero => {
                // Move magnitude toward zero when the error has the opposite sign.
                let need = self.b.select(self.boolean, None, s_neg, e_pos, e_neg)?;
                let need_nz = self.both_not(need, s_zero)?;
                self.select(need_nz, dec, s)?
            }
            Mode::Down => {
                // Positive: decrement (+0 becomes -min). Negative: grow magnitude.
                let pos_next = self.select(s_zero, min_neg, dec)?;
                let next = self.select(s_neg, inc, pos_next)?;
                self.select(e_neg, next, s)?
            }
            Mode::Up => {
                let neg_next = self.select(s_zero, min_pos, dec)?;
                let next = self.select(s_neg, neg_next, inc)?;
                self.select(e_pos, next, s)?
            }
        };
        // Finite operands that overflowed to infinity: toward-zero directions give max finite.
        let inf = self.c(INFINITY);
        let (a_mag, b_mag) = (self.magnitude(a)?, self.magnitude(b)?);
        let a_fin = self.lt(a_mag, inf)?;
        let b_fin = self.lt(b_mag, inf)?;
        let s_inf = self.eq(s_mag, inf)?;
        let fin = self.both(a_fin, b_fin)?;
        let overflow = self.both(s_inf, fin)?;
        let max_mag = self.c(INFINITY - 1);
        let max_val = self.or(s_sign, max_mag)?;
        let use_max = match mode {
            Mode::Zero => overflow,
            Mode::Down => self.both_not_neg(overflow, s_neg)?,
            Mode::Up => self.both(overflow, s_neg)?,
        };
        let result = self.select(use_max, max_val, stepped)?;
        // Exact cancellation of nonzero operands yields -0 when rounding down.
        if mode == Mode::Down && kind != Arith::Mul {
            let a_nz = self.lt(zero, a_mag)?;
            let cancel = self.both(a_nz, s_zero)?;
            let cancel = self.both(cancel, fin)?;
            let pos_zero = self.eq(s, zero)?;
            let cancel = self.both(cancel, pos_zero)?;
            let neg_zero = self.c(SIGN);
            return self.select(cancel, neg_zero, result);
        }
        Ok(result)
    }

    fn both_not(&mut self, a: Word, b: Word) -> Result<Word, Error> {
        let nb = self.b.logical_not(self.boolean, None, b)?;
        self.both(a, nb)
    }

    fn both_not_neg(&mut self, a: Word, b: Word) -> Result<Word, Error> {
        self.both_not(a, b)
    }
}

pub(super) fn directed(
    builder: &mut Builder,
    opencl: Word,
    kind: Arith,
    mode: Mode,
    a: Word,
    b: Word,
) -> Result<Word, Error> {
    Math::new(builder, opencl).directed(kind, mode, a, b)
}

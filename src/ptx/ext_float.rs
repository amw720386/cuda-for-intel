//! Floating-point lowering for `ext.rs`: f64, f16/bf16 (scalar and packed),
//! saturating and NaN-propagating f32, generic `cvt`, float compares.
//!
//! Half-precision values live in the low 16 bits of 32-bit register slots
//! (packed pairs fill the slot). Their arithmetic is performed in f32 and
//! rounded back; for add, sub and mul that is exact because f32 carries more
//! than twice the significand bits of f16/bf16. `fma` rounds twice, which can
//! differ from a single rounding in rare ties.

use super::super::ext::{ExtOp, FCmp, FOp, Mods, NumTy, Round, TestKind};
use super::super::Error;
use super::{Emitter, Kind, Value};
use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Decoration, FPRoundingMode, OpenclStd100Op, Word};

type R<T> = Result<T, Error>;

impl<'a> Emitter<'a> {
    pub(super) fn emit_float_ext(
        &mut self,
        op: ExtOp,
        dst: [&'a str; 2],
        src: [&'a str; 4],
    ) -> R<()> {
        self.fp32 = true;
        match op {
            ExtOp::Float { op, ty, mods } => self.float_arith(op, ty, mods, dst[0], src),
            ExtOp::Cvt { d, s, round, mods } => self.cvt(d, s, round, mods, dst[0], src),
            ExtOp::FSetp { cmp, ty } => {
                let a = self.fl_read(src[0], ty)?;
                let b = self.fl_read(src[1], ty)?;
                let id = self.float_compare(cmp, a[0], b[0])?;
                self.write(dst[0], Value { id, kind: Kind::Predicate })
            }
            ExtOp::FSet { cmp, d, s } => {
                let a = self.fl_read(src[0], s)?;
                let b = self.fl_read(src[1], s)?;
                let lane_count = a.len();
                let mut result = self.u32_const(0);
                for lane in 0..lane_count {
                    let hit = self.float_compare(cmp, a[lane], b[lane])?;
                    // Float destinations get 1.0, integer destinations all ones.
                    let (yes, width) = match d {
                        NumTy::F16x2 => (0x3c00u32, 16),
                        NumTy::Bf16x2 => (0x3f80, 16),
                        NumTy::F16 => (0x3c00, 16),
                        NumTy::Bf16 => (0x3f80, 16),
                        NumTy::F32 => (0x3f80_0000, 32),
                        NumTy::U16 | NumTy::S16 => (0xffff, 16),
                        // Packed sources produce one 16-bit mask per lane, also for .u32/.s32.
                        _ if lane_count == 2 => (0xffff, 16),
                        _ => (u32::MAX, 32),
                    };
                    let yes = self.u32_const(yes);
                    let zero = self.u32_const(0);
                    let value = self.builder.select(self.u32_type, None, hit, yes, zero)?;
                    let value = if width == 16 && lane_count == 2 && lane == 1 {
                        let sh = self.u32_const(16);
                        self.builder.shift_left_logical(self.u32_type, None, value, sh)?
                    } else {
                        value
                    };
                    result = self.builder.bitwise_or(self.u32_type, None, result, value)?;
                }
                self.write(dst[0], Value { id: result, kind: Kind::B32 })
            }
            ExtOp::Testp { kind, ty } => {
                let id = self.testp(kind, ty, src[0])?;
                self.write(dst[0], Value { id, kind: Kind::Predicate })
            }
            _ => unreachable!("non-float op routed to the float emitter"),
        }
    }

    fn f64_type(&mut self) -> Word {
        self.declare("fp64", Some(Capability::Float64), None);
        self.builder.type_float(64, None)
    }

    fn float_type_of(&mut self, ty: NumTy) -> Word {
        if ty == NumTy::F64 { self.f64_type() } else { self.f32_type }
    }

    fn fconst(&mut self, ty: NumTy, value: f64) -> Word {
        if ty == NumTy::F64 {
            let t = self.f64_type();
            self.builder.constant_bit64(t, value.to_bits())
        } else {
            self.builder.constant_bit32(self.f32_type, (value as f32).to_bits())
        }
    }

    /// Reads a float operand as native lanes: one value for f32/f64/f16/bf16,
    /// two (low lane first) for packed types. Half lanes are widened to f32.
    pub(super) fn fl_read(&mut self, operand: &str, ty: NumTy) -> R<Vec<Word>> {
        match ty {
            NumTy::F32 | NumTy::Tf32 => Ok(vec![self.float_value(operand)?]),
            NumTy::F64 => {
                let bits = self.integer(operand, Kind::B64)?;
                let t = self.f64_type();
                Ok(vec![self.builder.bitcast(t, None, bits)?])
            }
            NumTy::F16 | NumTy::Bf16 | NumTy::F16x2 | NumTy::Bf16x2 => {
                let bits = self.integer(operand, Kind::B32)?;
                let lanes = if ty.is_packed() { 2 } else { 1 };
                let mut out = Vec::new();
                for lane in 0..lanes {
                    let shifted = if lane == 0 {
                        bits
                    } else {
                        let sh = self.u32_const(16);
                        self.builder.shift_right_logical(self.u32_type, None, bits, sh)?
                    };
                    out.push(self.half_bits_to_f32(shifted, ty.is_bf16())?);
                }
                Ok(out)
            }
            _ => Err(Error::new("integer type used as a floating-point operand")),
        }
    }

    /// `bits` holds the half in its low 16 bits (upper bits ignored).
    fn half_bits_to_f32(&mut self, bits: Word, bf16: bool) -> R<Word> {
        if bf16 {
            let sh = self.u32_const(16);
            let up = self.builder.shift_left_logical(self.u32_type, None, bits, sh)?;
            return Ok(self.builder.bitcast(self.f32_type, None, up)?);
        }
        let bits16 = self.builder.u_convert(self.u16_type, None, bits)?;
        let half = self.builder.bitcast(self.f16_type, None, bits16)?;
        Ok(self.builder.f_convert(self.f32_type, None, half)?)
    }

    /// Rounds an f32 to half / bf16 bits (zero-extended in a u32).
    fn f32_to_half_bits(&mut self, x: Word, bf16: bool, mode: FPRoundingMode) -> R<Word> {
        if !bf16 {
            let half = self.builder.f_convert(self.f16_type, None, x)?;
            self.builder
                .decorate(half, Decoration::FPRoundingMode, [Operand::FPRoundingMode(mode)]);
            let bits16 = self.builder.bitcast(self.u16_type, None, half)?;
            return Ok(self.builder.u_convert(self.u32_type, None, bits16)?);
        }
        let ty = self.u32_type;
        let b = self.bool_type;
        let u = self.builder.bitcast(ty, None, x)?;
        let sixteen = self.u32_const(16);
        let one = self.u32_const(1);
        let top = self.builder.shift_right_logical(ty, None, u, sixteen)?;
        let lsb = self.builder.bitwise_and(ty, None, top, one)?;
        let bias = match mode {
            FPRoundingMode::RTZ => self.u32_const(0),
            FPRoundingMode::RTP | FPRoundingMode::RTN => {
                // Directed modes: add 0xffff when the discarded bits are non-zero in the
                // rounding direction, which is the sign-dependent ceil/floor.
                let low_mask = self.u32_const(0xffff);
                let low = self.builder.bitwise_and(ty, None, u, low_mask)?;
                let zero = self.u32_const(0);
                let inexact = self.builder.i_not_equal(b, None, low, zero)?;
                let c31 = self.u32_const(31);
                let sign_bit = self.builder.shift_right_logical(ty, None, u, c31)?;
                let negative = self.builder.i_not_equal(b, None, sign_bit, zero)?;
                let up_dir = matches!(mode, FPRoundingMode::RTP);
                let away = if up_dir {
                    self.builder.logical_not(b, None, negative)?
                } else {
                    negative
                };
                let go = self.builder.logical_and(b, None, inexact, away)?;
                let full = self.u32_const(0xffff);
                self.builder.select(ty, None, go, full, zero)?
            }
            _ => {
                let base = self.u32_const(0x7fff);
                self.builder.i_add(ty, None, base, lsb)?
            }
        };
        let sum = self.builder.i_add(ty, None, u, bias)?;
        let rounded = self.builder.shift_right_logical(ty, None, sum, sixteen)?;
        // NaN keeps a quiet payload instead of rounding into infinity.
        let is_nan = self.builder.f_unord_not_equal(b, None, x, x)?;
        let sign_mask = self.u32_const(0x8000);
        let sign = self.builder.bitwise_and(ty, None, top, sign_mask)?;
        let quiet = self.u32_const(0x7fff);
        let nan_bits = self.builder.bitwise_or(ty, None, sign, quiet)?;
        let low16 = self.u32_const(0xffff);
        let nan_bits = self.builder.bitwise_and(ty, None, nan_bits, low16)?;
        Ok(self.builder.select(ty, None, is_nan, nan_bits, rounded)?)
    }

    fn fl_write(&mut self, dst: &'a str, lanes: &[Word], ty: NumTy) -> R<()> {
        match ty {
            NumTy::F32 | NumTy::Tf32 => {
                let id = self.builder.bitcast(self.u32_type, None, lanes[0])?;
                self.write(dst, Value { id, kind: Kind::B32 })
            }
            NumTy::F64 => {
                let id = self.builder.bitcast(self.u64_type, None, lanes[0])?;
                self.write(dst, Value { id, kind: Kind::B64 })
            }
            _ => {
                let mut packed = self.u32_const(0);
                for (index, lane) in lanes.iter().enumerate() {
                    let bits = self.f32_to_half_bits(*lane, ty.is_bf16(), FPRoundingMode::RTE)?;
                    let bits = if index == 1 {
                        let sh = self.u32_const(16);
                        self.builder.shift_left_logical(self.u32_type, None, bits, sh)?
                    } else {
                        bits
                    };
                    packed = self.builder.bitwise_or(self.u32_type, None, packed, bits)?;
                }
                self.write(dst, Value { id: packed, kind: Kind::B32 })
            }
        }
    }

    fn ext_f(&mut self, fty: Word, op: OpenclStd100Op, args: &[Word]) -> R<Word> {
        Ok(self.builder.ext_inst(
            fty,
            None,
            self.opencl,
            op as u32,
            args.iter().copied().map(Operand::IdRef),
        )?)
    }

    fn apply_sat_relu(&mut self, ty: NumTy, fty: Word, x: Word, mods: Mods) -> R<Word> {
        let b = self.bool_type;
        let zero = self.fconst(ty, 0.0);
        let one = self.fconst(ty, 1.0);
        let mut x = x;
        if mods.relu {
            let negative = self.builder.f_ord_less_than(b, None, x, zero)?;
            x = self.builder.select(fty, None, negative, zero, x)?;
        }
        if mods.sat {
            let positive = self.builder.f_ord_greater_than(b, None, x, zero)?;
            let above = self.builder.f_ord_greater_than(b, None, x, one)?;
            let inner = self.builder.select(fty, None, positive, x, zero)?;
            x = self.builder.select(fty, None, above, one, inner)?;
        }
        Ok(x)
    }

    fn float_arith(&mut self, op: FOp, ty: NumTy, mods: Mods, dst: &'a str, src: [&str; 4]) -> R<()> {
        let b = self.bool_type;
        let compute_ty = if ty == NumTy::F64 { NumTy::F64 } else { NumTy::F32 };
        let fty = self.float_type_of(compute_ty);
        let arity = match op {
            FOp::Fma => 3,
            FOp::Neg | FOp::Abs | FOp::Ex2 | FOp::Tanh | FOp::Rcp | FOp::Sqrt | FOp::Rsqrt => 1,
            _ => 2,
        };
        let mut operands: Vec<Vec<Word>> = Vec::new();
        for slot in src.iter().take(arity) {
            operands.push(self.fl_read(slot, ty)?);
        }
        let lanes = operands[0].len();
        let mut results = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let x = operands[0][lane];
            let y = operands.get(1).map(|o| o[lane]);
            let z = operands.get(2).map(|o| o[lane]);
            let value = match op {
                FOp::Add => self.builder.f_add(fty, None, x, y.unwrap())?,
                FOp::Sub => self.builder.f_sub(fty, None, x, y.unwrap())?,
                FOp::Mul => self.builder.f_mul(fty, None, x, y.unwrap())?,
                FOp::Div => self.builder.f_div(fty, None, x, y.unwrap())?,
                FOp::Fma => self.ext_f(fty, OpenclStd100Op::fma, &[x, y.unwrap(), z.unwrap()])?,
                FOp::Min | FOp::Max => {
                    let y = y.unwrap();
                    let which = if op == FOp::Min { OpenclStd100Op::fmin } else { OpenclStd100Op::fmax };
                    let numeric = self.ext_f(fty, which, &[x, y])?;
                    if mods.nan {
                        let x_nan = self.builder.f_unord_not_equal(b, None, x, x)?;
                        let y_nan = self.builder.f_unord_not_equal(b, None, y, y)?;
                        let inner = self.builder.select(fty, None, y_nan, y, numeric)?;
                        self.builder.select(fty, None, x_nan, x, inner)?
                    } else {
                        numeric
                    }
                }
                FOp::Neg => self.builder.f_negate(fty, None, x)?,
                FOp::Abs => self.ext_f(fty, OpenclStd100Op::fabs, &[x])?,
                FOp::Ex2 => self.ext_f(fty, OpenclStd100Op::exp2, &[x])?,
                FOp::Tanh => self.ext_f(fty, OpenclStd100Op::tanh, &[x])?,
                FOp::Rsqrt => self.ext_f(fty, OpenclStd100Op::rsqrt, &[x])?,
                FOp::Sqrt => self.ext_f(fty, OpenclStd100Op::sqrt, &[x])?,
                FOp::Rcp => {
                    let one = self.fconst(compute_ty, 1.0);
                    self.builder.f_div(fty, None, one, x)?
                }
            };
            let value = if mods.sat || mods.relu {
                self.apply_sat_relu(compute_ty, fty, value, mods)?
            } else {
                value
            };
            results.push(value);
        }
        // Negation and absolute value of half types must not quiet NaN payloads or
        // flush subnormals: operate on the sign bit directly.
        if ty.is_half() && matches!(op, FOp::Neg | FOp::Abs) {
            let bits = self.integer(src[0], Kind::B32)?;
            let (lane_mask, sign) = (0x7fff_7fffu32, 0x8000_8000u32);
            let id = if op == FOp::Abs {
                let mask = self.u32_const(if ty.is_packed() { lane_mask } else { 0x7fff });
                self.builder.bitwise_and(self.u32_type, None, bits, mask)?
            } else {
                let mask = self.u32_const(if ty.is_packed() { sign } else { 0x8000 });
                self.builder.bitwise_xor(self.u32_type, None, bits, mask)?
            };
            let id = if ty.is_packed() {
                id
            } else {
                // Scalar halves keep the upper register bits clear.
                let low = self.u32_const(0xffff);
                self.builder.bitwise_and(self.u32_type, None, id, low)?
            };
            return self.write(dst, Value { id, kind: Kind::B32 });
        }
        self.fl_write(dst, &results, ty)
    }

    fn float_compare(&mut self, cmp: FCmp, a: Word, b_: Word) -> R<Word> {
        let t = self.bool_type;
        Ok(match cmp {
            FCmp::Eq => self.builder.f_ord_equal(t, None, a, b_)?,
            FCmp::Ne => self.builder.f_ord_not_equal(t, None, a, b_)?,
            FCmp::Lt => self.builder.f_ord_less_than(t, None, a, b_)?,
            FCmp::Le => self.builder.f_ord_less_than_equal(t, None, a, b_)?,
            FCmp::Gt => self.builder.f_ord_greater_than(t, None, a, b_)?,
            FCmp::Ge => self.builder.f_ord_greater_than_equal(t, None, a, b_)?,
            FCmp::Equ => self.builder.f_unord_equal(t, None, a, b_)?,
            FCmp::Neu => self.builder.f_unord_not_equal(t, None, a, b_)?,
            FCmp::Ltu => self.builder.f_unord_less_than(t, None, a, b_)?,
            FCmp::Leu => self.builder.f_unord_less_than_equal(t, None, a, b_)?,
            FCmp::Gtu => self.builder.f_unord_greater_than(t, None, a, b_)?,
            FCmp::Geu => self.builder.f_unord_greater_than_equal(t, None, a, b_)?,
            FCmp::Num | FCmp::Nan => {
                let a_ok = self.builder.f_ord_equal(t, None, a, a)?;
                let b_ok = self.builder.f_ord_equal(t, None, b_, b_)?;
                let both = self.builder.logical_and(t, None, a_ok, b_ok)?;
                if cmp == FCmp::Num { both } else { self.builder.logical_not(t, None, both)? }
            }
        })
    }

    fn testp(&mut self, kind: TestKind, ty: NumTy, operand: &str) -> R<Word> {
        let b = self.bool_type;
        let (wide, mant_bits, exp_mask) = if ty == NumTy::F64 {
            (true, 52u32, 0x7ffu64)
        } else {
            (false, 23, 0xff)
        };
        let int_ty = if wide { self.u64_type } else { self.u32_type };
        let bits = self.integer(operand, if wide { Kind::B64 } else { Kind::B32 })?;
        let c = |e: &mut Self, v: u64| e.const_of(wide, v);
        let sh = c(self, u64::from(mant_bits));
        let shifted = self.builder.shift_right_logical(int_ty, None, bits, sh)?;
        let em = c(self, exp_mask);
        let exp = self.builder.bitwise_and(int_ty, None, shifted, em)?;
        let mant_mask = c(self, (1u64 << mant_bits) - 1);
        let mant = self.builder.bitwise_and(int_ty, None, bits, mant_mask)?;
        let zero = c(self, 0);
        let exp_zero = self.builder.i_equal(b, None, exp, zero)?;
        let exp_ones = self.builder.i_equal(b, None, exp, em)?;
        let mant_zero = self.builder.i_equal(b, None, mant, zero)?;
        let not_exp_ones = self.builder.logical_not(b, None, exp_ones)?;
        let not_exp_zero = self.builder.logical_not(b, None, exp_zero)?;
        let mant_nonzero = self.builder.logical_not(b, None, mant_zero)?;
        Ok(match kind {
            TestKind::Finite => not_exp_ones,
            TestKind::Infinite => self.builder.logical_and(b, None, exp_ones, mant_zero)?,
            TestKind::NotANumber => self.builder.logical_and(b, None, exp_ones, mant_nonzero)?,
            TestKind::Number => {
                let nan = self.builder.logical_and(b, None, exp_ones, mant_nonzero)?;
                self.builder.logical_not(b, None, nan)?
            }
            TestKind::Normal => self.builder.logical_and(b, None, not_exp_zero, not_exp_ones)?,
            TestKind::Subnormal => self.builder.logical_and(b, None, exp_zero, mant_nonzero)?,
        })
    }

    fn rounding(round: Round) -> FPRoundingMode {
        match round {
            Round::Rz | Round::Rzi => FPRoundingMode::RTZ,
            Round::Rm | Round::Rmi => FPRoundingMode::RTN,
            Round::Rp | Round::Rpi => FPRoundingMode::RTP,
            _ => FPRoundingMode::RTE,
        }
    }

    fn read_int(&mut self, operand: &str, ty: NumTy) -> R<(Word, bool)> {
        // Returns the value extended to 32 or 64 bits and whether it is 64-bit.
        if ty.bits() == 64 {
            return Ok((self.integer(operand, Kind::B64)?, true));
        }
        let x = self.integer(operand, Kind::B32)?;
        let bits = ty.bits();
        if bits == 32 {
            return Ok((x, false));
        }
        let t = self.u32_type;
        let value = if ty.is_signed() {
            let sh = self.u32_const(32 - bits);
            let up = self.builder.shift_left_logical(t, None, x, sh)?;
            self.builder.shift_right_arithmetic(t, None, up, sh)?
        } else {
            let mask = self.u32_const((1u32 << bits) - 1);
            self.builder.bitwise_and(t, None, x, mask)?
        };
        Ok((value, false))
    }

    fn cvt(
        &mut self,
        d: NumTy,
        s: NumTy,
        round: Round,
        mods: Mods,
        dst: &'a str,
        src: [&str; 4],
    ) -> R<()> {
        let mode = Self::rounding(round);
        // tf32 keeps an f32 with the low 13 mantissa bits cleared.
        if d == NumTy::Tf32 {
            let ty = self.u32_type;
            let u = self.integer(src[0], Kind::B32)?;
            let bias = if round == Round::Rna {
                self.u32_const(0x1000)
            } else {
                let sh = self.u32_const(13);
                let one = self.u32_const(1);
                let lsb = self.builder.shift_right_logical(ty, None, u, sh)?;
                let lsb = self.builder.bitwise_and(ty, None, lsb, one)?;
                let base = self.u32_const(0x0fff);
                self.builder.i_add(ty, None, base, lsb)?
            };
            let sum = self.builder.i_add(ty, None, u, bias)?;
            let mask = self.u32_const(0xffff_e000);
            let rounded = self.builder.bitwise_and(ty, None, sum, mask)?;
            // Infinity and NaN pass through untouched.
            let x = self.builder.bitcast(self.f32_type, None, u)?;
            let finite = self.builder.f_ord_equal(self.bool_type, None, x, x)?;
            let inf = self.u32_const(0x7f80_0000);
            let magnitude = self.u32_const(0x7fff_ffff);
            let abs = self.builder.bitwise_and(ty, None, u, magnitude)?;
            let not_inf = self.builder.i_not_equal(self.bool_type, None, abs, inf)?;
            let keep = self.builder.logical_and(self.bool_type, None, finite, not_inf)?;
            let id = self.builder.select(ty, None, keep, rounded, u)?;
            return self.write(dst, Value { id, kind: Kind::B32 });
        }
        if d.is_float() {
            // Source lanes as native floats (or an integer converted below).
            let mut lanes: Vec<Word> = if s.is_float() {
                let first = self.fl_read(src[0], s)?;
                if d.is_packed() {
                    // cvt.rn.f16x2.f32 d, hi, lo
                    let lo = self.fl_read(src[1], s)?;
                    vec![lo[0], first[0]]
                } else {
                    first
                }
            } else {
                let (value, wide) = self.read_int(src[0], s)?;
                let target = if d == NumTy::F64 { NumTy::F64 } else { NumTy::F32 };
                let fty = self.float_type_of(target);
                let converted = if s.is_signed() {
                    self.builder.convert_s_to_f(fty, None, value)?
                } else {
                    self.builder.convert_u_to_f(fty, None, value)?
                };
                let _ = wide;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(mode)],
                );
                vec![converted]
            };
            // f64 -> narrower float: apply the requested rounding once.
            if s == NumTy::F64 && d != NumTy::F64 {
                let converted = self.builder.f_convert(self.f32_type, None, lanes[0])?;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(mode)],
                );
                lanes[0] = converted;
            }
            if d == NumTy::F64 && s != NumTy::F64 && s.is_float() {
                let t = self.f64_type();
                lanes[0] = self.builder.f_convert(t, None, lanes[0])?;
            }
            let compute = if d == NumTy::F64 { NumTy::F64 } else { NumTy::F32 };
            let fty = self.float_type_of(compute);
            if mods.sat || mods.relu {
                for lane in lanes.iter_mut() {
                    *lane = self.apply_sat_relu(compute, fty, *lane, mods)?;
                }
            }
            if d.is_half() {
                // Rounding mode applies to the final narrowing step.
                let mut packed = self.u32_const(0);
                for (index, lane) in lanes.iter().enumerate() {
                    let bits = self.f32_to_half_bits(*lane, d.is_bf16(), mode)?;
                    let bits = if index == 1 {
                        let sh = self.u32_const(16);
                        self.builder.shift_left_logical(self.u32_type, None, bits, sh)?
                    } else {
                        bits
                    };
                    packed = self.builder.bitwise_or(self.u32_type, None, packed, bits)?;
                }
                return self.write(dst, Value { id: packed, kind: Kind::B32 });
            }
            return self.fl_write(dst, &lanes, d);
        }
        // Integer destination.
        let dbits = d.bits();
        let dwide = dbits == 64;
        if s.is_float() {
            let x = self.fl_read(src[0], s)?[0];
            let compute = if s == NumTy::F64 { NumTy::F64 } else { NumTy::F32 };
            let fty = self.float_type_of(compute);
            let which = match round {
                Round::Rni | Round::Rn | Round::None => Some(OpenclStd100Op::rint),
                Round::Rzi | Round::Rz => Some(OpenclStd100Op::trunc),
                Round::Rmi | Round::Rm => Some(OpenclStd100Op::floor),
                Round::Rpi | Round::Rp => Some(OpenclStd100Op::ceil),
                Round::Rna => Some(OpenclStd100Op::round),
            };
            let rounded = match which {
                Some(op) => self.ext_f(fty, op, &[x])?,
                None => x,
            };
            let zero = self.fconst(compute, 0.0);
            let target = if dwide { self.u64_type } else { self.u32_type };
            let signed_wide = d.is_signed() || dbits < 32;
            let value = if signed_wide {
                self.builder.convert_f_to_s(target, None, rounded)?
            } else {
                let negative = self.builder.f_ord_less_than(self.bool_type, None, rounded, zero)?;
                let clamped = self.builder.select(fty, None, negative, zero, rounded)?;
                self.builder.convert_f_to_u(target, None, clamped)?
            };
            self.builder.decorate(
                value,
                Decoration::FPRoundingMode,
                [Operand::FPRoundingMode(FPRoundingMode::RTZ)],
            );
            self.builder.decorate(value, Decoration::SaturatedConversion, []);
            // A saturated OpenCL conversion maps NaN to 0. PTX (cvt, "NaN input results in"):
            // 0 only when the source is not .f64 and the destination is not .s64/.u64, otherwise
            // 1 << (bits - 1) (ZLUDA's cvt_u64_f16 vector). Narrow destinations from .f64 keep 0.
            let value = if dwide || (s == NumTy::F64 && dbits == 32) {
                let nan = self.builder.f_unord_not_equal(self.bool_type, None, x, x)?;
                let (min, ty) = if dwide {
                    (self.builder.constant_bit64(self.u64_type, 1u64 << 63), self.u64_type)
                } else {
                    (self.u32_const(1 << 31), self.u32_type)
                };
                self.builder.select(ty, None, nan, min, value)?
            } else {
                value
            };
            let id = if dbits < 32 {
                // Clamp the 32-bit saturated value into the narrow destination range.
                let t = self.u32_type;
                let (lo, hi) = Self::int_range(d);
                let lo = self.u32_const(lo as u32);
                let hi = self.u32_const(hi as u32);
                let v = self.opencl_int(t, OpenclStd100Op::s_max, &[value, lo])?;
                let v = self.opencl_int(t, OpenclStd100Op::s_min, &[v, hi])?;
                let mask = self.u32_const((1u32 << Self::reg_bits(d, dbits)) - 1);
                self.builder.bitwise_and(t, None, v, mask)?
            } else {
                value
            };
            let kind = if dwide { Kind::B64 } else { Kind::B32 };
            return self.write(dst, Value { id, kind });
        }
        // int -> int with saturation.
        let (value, wide) = self.read_int(src[0], s)?;
        let t64 = self.u64_type;
        let v64 = if wide {
            value
        } else if s.is_signed() {
            self.builder.s_convert(t64, None, value)?
        } else {
            self.builder.u_convert(t64, None, value)?
        };
        let mut v = v64;
        if s.is_signed() {
            if !d.is_signed() {
                let zero = self.builder.constant_bit64(t64, 0);
                v = self.opencl_int(t64, OpenclStd100Op::s_max, &[v, zero])?;
            } else if dbits < 64 {
                let (lo, _) = Self::int_range(d);
                let lo = self.builder.constant_bit64(t64, lo as u64);
                v = self.opencl_int(t64, OpenclStd100Op::s_max, &[v, lo])?;
            }
            if dbits < 64 {
                let (_, hi) = Self::int_range(d);
                let hi = self.builder.constant_bit64(t64, hi as u64);
                v = self.opencl_int(t64, OpenclStd100Op::s_min, &[v, hi])?;
            }
        } else if dbits < 64 || d.is_signed() {
            let hi = if dbits < 64 { Self::int_range(d).1 as u64 } else { i64::MAX as u64 };
            let hi = self.builder.constant_bit64(t64, hi);
            v = self.opencl_int(t64, OpenclStd100Op::u_min, &[v, hi])?;
        }
        if dwide {
            return self.write(dst, Value { id: v, kind: Kind::B64 });
        }
        let narrowed = self.builder.u_convert(self.u32_type, None, v)?;
        let id = if dbits < 32 {
            let mask = self.u32_const((1u32 << Self::reg_bits(d, dbits)) - 1);
            self.builder.bitwise_and(self.u32_type, None, narrowed, mask)?
        } else {
            narrowed
        };
        self.write(dst, Value { id, kind: Kind::B32 })
    }

    /// Bits of the destination register that carry the value: PTX sign- or zero-extends a
    /// narrow result to the 16-bit register width, so a signed 8-bit result occupies 16 bits.
    fn reg_bits(d: NumTy, dbits: u32) -> u32 {
        if dbits == 8 && d.is_signed() { 16 } else { dbits }
    }

    /// Inclusive value range of a narrow (<= 32-bit) integer type as i64.
    fn int_range(ty: NumTy) -> (i64, i64) {
        let bits = ty.bits();
        if ty.is_signed() {
            (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
        } else {
            (0, (1i64 << bits) - 1)
        }
    }

    /// Lane-wise half / bf16 addition of two packed registers (round to nearest even).
    pub(super) fn half2_add_bits(&mut self, a: Word, b: Word, bf16: bool) -> R<Word> {
        let t = self.u32_type;
        let mut packed = self.u32_const(0);
        for lane in 0..2u32 {
            let (x, y) = if lane == 0 {
                (a, b)
            } else {
                let sh = self.u32_const(16);
                (
                    self.builder.shift_right_logical(t, None, a, sh)?,
                    self.builder.shift_right_logical(t, None, b, sh)?,
                )
            };
            let fx = self.half_bits_to_f32(x, bf16)?;
            let fy = self.half_bits_to_f32(y, bf16)?;
            let sum = self.builder.f_add(self.f32_type, None, fx, fy)?;
            let bits = self.f32_to_half_bits(sum, bf16, FPRoundingMode::RTE)?;
            let bits = if lane == 1 {
                let sh = self.u32_const(16);
                self.builder.shift_left_logical(t, None, bits, sh)?
            } else {
                bits
            };
            packed = self.builder.bitwise_or(t, None, packed, bits)?;
        }
        Ok(packed)
    }
}

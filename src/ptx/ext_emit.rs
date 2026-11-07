//! SPIR-V lowering for the instructions parsed by `ext.rs`.
//!
//! This file is a child module of `spirv.rs` so that it can use the emitter's
//! private state; everything here is `impl Emitter`.

use super::super::ext::{
    AtomTy, ClockKind, Combine, ExtOp, PrmtMode, ReduxOp, ShflMode,
};
use super::super::{AtomicOp, Comparison, Error, Space};
use super::{Emitter, Kind, Value};
use rspirv::spirv::{
    Capability, GroupOperation, MemorySemantics, OpenclStd100Op, Scope, StorageClass, Word,
};

type R<T> = Result<T, Error>;

impl<'a> Emitter<'a> {
    pub(super) fn emit_ext(&mut self, op: ExtOp, dst: [&'a str; 2], src: [&'a str; 4]) -> R<()> {
        match op {
            ExtOp::Float { .. }
            | ExtOp::Cvt { .. }
            | ExtOp::FSetp { .. }
            | ExtOp::FSet { .. }
            | ExtOp::Testp { .. } => self.emit_float_ext(op, dst, src),
            ExtOp::Brev { wide } => {
                let kind = if wide { Kind::B64 } else { Kind::B32 };
                let a = self.integer(src[0], kind)?;
                let id = self.brev(a, wide)?;
                self.write(dst[0], Value { id, kind })
            }
            ExtOp::Bmsk { wrap } => {
                let ty = self.u32_type;
                let mut a = self.integer(src[0], Kind::B32)?;
                let mut b = self.integer(src[1], Kind::B32)?;
                if wrap {
                    let m = self.u32_const(31);
                    a = self.builder.bitwise_and(ty, None, a, m)?;
                    b = self.builder.bitwise_and(ty, None, b, m)?;
                } else {
                    let c32 = self.u32_const(32);
                    a = self.opencl_int(ty, OpenclStd100Op::u_min, &[a, c32])?;
                    b = self.opencl_int(ty, OpenclStd100Op::u_min, &[b, c32])?;
                }
                let ones = self.u32_const(u32::MAX);
                let low = self.shl_sat(ones, a)?;
                let end = self.builder.i_add(ty, None, a, b)?;
                let high = self.shl_sat(ones, end)?;
                let not_high = self.builder.not(ty, None, high)?;
                let id = self.builder.bitwise_and(ty, None, low, not_high)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Fns => {
                let mask = self.integer(src[0], Kind::B32)?;
                let base = self.integer(src[1], Kind::B32)?;
                let offset = self.integer(src[2], Kind::B32)?;
                let id = self.fns(mask, base, offset)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Szext { wrap, signed } => {
                let ty = self.u32_type;
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let c32 = self.u32_const(32);
                let zero = self.u32_const(0);
                let w = if wrap {
                    let m = self.u32_const(31);
                    self.builder.bitwise_and(ty, None, b, m)?
                } else {
                    self.opencl_int(ty, OpenclStd100Op::u_min, &[b, c32])?
                };
                let id = if signed {
                    let shift = self.builder.i_sub(ty, None, c32, w)?;
                    // w == 0 would shift by 32; the result is forced to zero below.
                    let is_zero = self.builder.i_equal(self.bool_type, None, w, zero)?;
                    let safe = self.builder.select(ty, None, is_zero, zero, shift)?;
                    let up = self.builder.shift_left_logical(ty, None, a, safe)?;
                    let down = self.builder.shift_right_arithmetic(ty, None, up, safe)?;
                    self.builder.select(ty, None, is_zero, zero, down)?
                } else {
                    let mask = self.low_mask(w)?;
                    self.builder.bitwise_and(ty, None, a, mask)?
                };
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Mul24 { hi, signed } | ExtOp::Mad24 { hi, signed } => {
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let product = self.mul24(a, b, hi, signed)?;
                let id = if matches!(op, ExtOp::Mad24 { .. }) {
                    let c = self.integer(src[2], Kind::B32)?;
                    self.builder.i_add(self.u32_type, None, product, c)?
                } else {
                    product
                };
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Sad { signed } => {
                let ty = self.u32_type;
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let c = self.integer(src[2], Kind::B32)?;
                let greater = if signed {
                    self.builder.s_greater_than(self.bool_type, None, a, b)?
                } else {
                    self.builder.u_greater_than(self.bool_type, None, a, b)?
                };
                let ab = self.builder.i_sub(ty, None, a, b)?;
                let ba = self.builder.i_sub(ty, None, b, a)?;
                let diff = self.builder.select(ty, None, greater, ab, ba)?;
                let id = self.builder.i_add(ty, None, diff, c)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Cnot { wide } => {
                let kind = if wide { Kind::B64 } else { Kind::B32 };
                let a = self.integer(src[0], kind)?;
                let (zero, one) = if wide {
                    (self.builder.constant_bit64(self.u64_type, 0), self.builder.constant_bit64(self.u64_type, 1))
                } else {
                    (self.u32_const(0), self.u32_const(1))
                };
                let is_zero = self.builder.i_equal(self.bool_type, None, a, zero)?;
                let ty = self.value_type(kind);
                let id = self.builder.select(ty, None, is_zero, one, zero)?;
                self.write(dst[0], Value { id, kind })
            }
            ExtOp::SatAddSub { sub } => {
                let t64 = self.u64_type;
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let a = self.builder.s_convert(t64, None, a)?;
                let b = self.builder.s_convert(t64, None, b)?;
                let sum = if sub {
                    self.builder.i_sub(t64, None, a, b)?
                } else {
                    self.builder.i_add(t64, None, a, b)?
                };
                let lo = self.builder.constant_bit64(t64, i64::from(i32::MIN) as u64);
                let hi = self.builder.constant_bit64(t64, i64::from(i32::MAX) as u64);
                let v = self.opencl_int(t64, OpenclStd100Op::s_max, &[sum, lo])?;
                let v = self.opencl_int(t64, OpenclStd100Op::s_min, &[v, hi])?;
                let id = self.builder.u_convert(self.u32_type, None, v)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::MadWide { signed } => {
                let t64 = self.u64_type;
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let c = self.integer(src[2], Kind::B64)?;
                let (a, b) = if signed {
                    (self.builder.s_convert(t64, None, a)?, self.builder.s_convert(t64, None, b)?)
                } else {
                    (self.builder.u_convert(t64, None, a)?, self.builder.u_convert(t64, None, b)?)
                };
                let product = self.builder.i_mul(t64, None, a, b)?;
                let id = self.builder.i_add(t64, None, product, c)?;
                self.write(dst[0], Value { id, kind: Kind::B64 })
            }
            ExtOp::Bfind { wide, signed, shiftamt } => {
                let id = self.bfind(src[0], wide, signed, shiftamt)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Clz64 => {
                let a = self.integer(src[0], Kind::B64)?;
                let count = self.opencl_int(self.u64_type, OpenclStd100Op::clz, &[a])?;
                let id = self.builder.u_convert(self.u32_type, None, count)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::MadHi { signed } => {
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let c = self.integer(src[2], Kind::B32)?;
                let which = if signed { OpenclStd100Op::s_mul_hi } else { OpenclStd100Op::u_mul_hi };
                let hi = self.opencl_int(self.u32_type, which, &[a, b])?;
                let id = self.builder.i_add(self.u32_type, None, hi, c)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::MulHi64 { signed } => {
                let a = self.integer(src[0], Kind::B64)?;
                let b = self.integer(src[1], Kind::B64)?;
                let which = if signed { OpenclStd100Op::s_mul_hi } else { OpenclStd100Op::u_mul_hi };
                let id = self.opencl_int(self.u64_type, which, &[a, b])?;
                self.write(dst[0], Value { id, kind: Kind::B64 })
            }
            ExtOp::Bfe64 { signed } => {
                let id = self.bfe64(src, signed)?;
                self.write(dst[0], Value { id, kind: Kind::B64 })
            }
            ExtOp::Bfi64 => {
                let id = self.bfi64(src)?;
                self.write(dst[0], Value { id, kind: Kind::B64 })
            }
            ExtOp::Prmt(mode) => {
                let ty = self.u32_type;
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let c = self.integer(src[2], Kind::B32)?;
                let three = self.u32_const(3);
                let n = self.builder.bitwise_and(ty, None, c, three)?;
                let mut selector = self.u32_const(0);
                for i in 0..4u32 {
                    let ci = self.u32_const(i);
                    let nibble = match mode {
                        PrmtMode::F4e => self.builder.i_add(ty, None, n, ci)?,
                        PrmtMode::B4e => {
                            let diff = self.builder.i_sub(ty, None, n, ci)?;
                            let seven = self.u32_const(7);
                            self.builder.bitwise_and(ty, None, diff, seven)?
                        }
                        PrmtMode::Rc8 => n,
                        PrmtMode::Ecl => self.opencl_int(ty, OpenclStd100Op::u_max, &[n, ci])?,
                        PrmtMode::Ecr => self.opencl_int(ty, OpenclStd100Op::u_min, &[n, ci])?,
                        PrmtMode::Rc16 => {
                            let one = self.u32_const(1);
                            let two = self.u32_const(2);
                            let odd = self.builder.bitwise_and(ty, None, n, one)?;
                            let base = self.builder.i_mul(ty, None, odd, two)?;
                            let parity = self.u32_const(i & 1);
                            self.builder.i_add(ty, None, base, parity)?
                        }
                    };
                    let shift = self.u32_const(4 * i);
                    let placed = self.builder.shift_left_logical(ty, None, nibble, shift)?;
                    selector = self.builder.bitwise_or(ty, None, selector, placed)?;
                }
                let id = self.prmt(a, b, selector)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Dp4a { signed } => {
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let c = self.integer(src[2], Kind::B32)?;
                let id = self.dot4(a, b, c, signed)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Dp2a { hi, signed } => {
                let ty = self.u32_type;
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let mut acc = self.integer(src[2], Kind::B32)?;
                for k in 0..2u32 {
                    let a_shift = self.u32_const(16 * k);
                    let lane = self.builder.shift_right_logical(ty, None, a, a_shift)?;
                    let sixteen = self.u32_const(16);
                    let lane = if signed[0] {
                        let up = self.builder.shift_left_logical(ty, None, lane, sixteen)?;
                        self.builder.shift_right_arithmetic(ty, None, up, sixteen)?
                    } else {
                        let m = self.u32_const(0xffff);
                        self.builder.bitwise_and(ty, None, lane, m)?
                    };
                    let byte_index = k + if hi { 2 } else { 0 };
                    let b_shift = self.u32_const(8 * byte_index);
                    let byte = self.builder.shift_right_logical(ty, None, b, b_shift)?;
                    let twenty4 = self.u32_const(24);
                    let byte = if signed[1] {
                        let up = self.builder.shift_left_logical(ty, None, byte, twenty4)?;
                        self.builder.shift_right_arithmetic(ty, None, up, twenty4)?
                    } else {
                        let m = self.u32_const(0xff);
                        self.builder.bitwise_and(ty, None, byte, m)?
                    };
                    let product = self.builder.i_mul(ty, None, lane, byte)?;
                    acc = self.builder.i_add(ty, None, acc, product)?;
                }
                self.write(dst[0], Value { id: acc, kind: Kind::B32 })
            }
            ExtOp::Shf { left, clamp } => {
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let c = self.integer(src[2], Kind::B32)?;
                let ty = self.u32_type;
                let n = if clamp {
                    let c32 = self.u32_const(32);
                    self.opencl_int(ty, OpenclStd100Op::u_min, &[c, c32])?
                } else {
                    let m = self.u32_const(31);
                    self.builder.bitwise_and(ty, None, c, m)?
                };
                let t64 = self.u64_type;
                let lo = self.builder.u_convert(t64, None, a)?;
                let hi = self.builder.u_convert(t64, None, b)?;
                let c32_64 = self.builder.constant_bit64(t64, 32);
                let hi = self.builder.shift_left_logical(t64, None, hi, c32_64)?;
                let value = self.builder.bitwise_or(t64, None, lo, hi)?;
                let n64 = self.builder.u_convert(t64, None, n)?;
                let id = if left {
                    let shifted = self.builder.shift_left_logical(t64, None, value, n64)?;
                    let high = self.builder.shift_right_logical(t64, None, shifted, c32_64)?;
                    self.builder.u_convert(ty, None, high)?
                } else {
                    let shifted = self.builder.shift_right_logical(t64, None, value, n64)?;
                    self.builder.u_convert(ty, None, shifted)?
                };
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Slct { wide, float_selector } => {
                let kind = if wide { Kind::B64 } else { Kind::B32 };
                let a = self.integer(src[0], kind)?;
                let b = self.integer(src[1], kind)?;
                let condition = if float_selector {
                    let bits = self.float_bits(src[2])?;
                    let value = self.builder.bitcast(self.f32_type, None, bits)?;
                    let zero = self.builder.constant_bit32(self.f32_type, 0);
                    self.builder
                        .f_ord_greater_than_equal(self.bool_type, None, value, zero)?
                } else {
                    let c = self.integer(src[2], Kind::B32)?;
                    let zero = self.u32_const(0);
                    self.builder.s_greater_than_equal(self.bool_type, None, c, zero)?
                };
                let ty = self.value_type(kind);
                let id = self.builder.select(ty, None, condition, a, b)?;
                self.write(dst[0], Value { id, kind })
            }
            ExtOp::SetpCombine { cmp, signed, wide, combine } => {
                let kind = if wide { Kind::B64 } else { Kind::B32 };
                let a = self.integer(src[0], kind)?;
                let b = self.integer(src[1], kind)?;
                let t = self.bool_type;
                let first = match (cmp, signed) {
                    (Comparison::Equal, _) => self.builder.i_equal(t, None, a, b)?,
                    (Comparison::NotEqual, _) => self.builder.i_not_equal(t, None, a, b)?,
                    (Comparison::Less, true) => self.builder.s_less_than(t, None, a, b)?,
                    (Comparison::Less, false) => self.builder.u_less_than(t, None, a, b)?,
                    (Comparison::LessEqual, true) => self.builder.s_less_than_equal(t, None, a, b)?,
                    (Comparison::LessEqual, false) => self.builder.u_less_than_equal(t, None, a, b)?,
                    (Comparison::Greater, true) => self.builder.s_greater_than(t, None, a, b)?,
                    (Comparison::Greater, false) => self.builder.u_greater_than(t, None, a, b)?,
                    (Comparison::GreaterEqual, true) => {
                        self.builder.s_greater_than_equal(t, None, a, b)?
                    }
                    (Comparison::GreaterEqual, false) => {
                        self.builder.u_greater_than_equal(t, None, a, b)?
                    }
                };
                let q = self.predicate_operand(src[2])?;
                let id = match combine {
                    Combine::And => self.builder.logical_and(t, None, first, q)?,
                    Combine::Or => self.builder.logical_or(t, None, first, q)?,
                    Combine::Xor => self.builder.logical_not_equal(t, None, first, q)?,
                };
                self.write(dst[0], Value { id, kind: Kind::Predicate })
            }
            ExtOp::PackB64 => {
                let lo = self.integer(src[0], Kind::B32)?;
                let hi = self.integer(src[1], Kind::B32)?;
                let t64 = self.u64_type;
                let lo = self.builder.u_convert(t64, None, lo)?;
                let hi = self.builder.u_convert(t64, None, hi)?;
                let sh = self.builder.constant_bit64(t64, 32);
                let hi = self.builder.shift_left_logical(t64, None, hi, sh)?;
                let id = self.builder.bitwise_or(t64, None, lo, hi)?;
                self.write(dst[0], Value { id, kind: Kind::B64 })
            }
            ExtOp::UnpackB64 => {
                let x = self.integer(src[0], Kind::B64)?;
                let t64 = self.u64_type;
                let lo = self.builder.u_convert(self.u32_type, None, x)?;
                let sh = self.builder.constant_bit64(t64, 32);
                let hi = self.builder.shift_right_logical(t64, None, x, sh)?;
                let hi = self.builder.u_convert(self.u32_type, None, hi)?;
                self.write(dst[0], Value { id: lo, kind: Kind::B32 })?;
                self.write(dst[1], Value { id: hi, kind: Kind::B32 })
            }
            ExtOp::UnpackB32 => {
                let x = self.integer(src[0], Kind::B32)?;
                let ty = self.u32_type;
                let mask = self.u32_const(0xffff);
                let lo = self.builder.bitwise_and(ty, None, x, mask)?;
                let sh = self.u32_const(16);
                let hi = self.builder.shift_right_logical(ty, None, x, sh)?;
                self.write(dst[0], Value { id: lo, kind: Kind::B32 })?;
                self.write(dst[1], Value { id: hi, kind: Kind::B32 })
            }
            ExtOp::WarpId => {
                let linear = self.linear_tid()?;
                let five = self.u32_const(5);
                let id = self.builder.shift_right_logical(self.u32_type, None, linear, five)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::NWarpId => {
                let total = self.block_size()?;
                let c31 = self.u32_const(31);
                let five = self.u32_const(5);
                let sum = self.builder.i_add(self.u32_type, None, total, c31)?;
                let id = self.builder.shift_right_logical(self.u32_type, None, sum, five)?;
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::Clock(kind) => {
                self.declare("shader_clock", Some(Capability::ShaderClockKHR), Some("SPV_KHR_shader_clock"));
                let scope = self.builder.constant_bit32(self.u32_type, Scope::Device as u32);
                let ticks = self.builder.read_clock_khr(self.u64_type, None, scope)?;
                match kind {
                    ClockKind::Clock64 => self.write(dst[0], Value { id: ticks, kind: Kind::B64 }),
                    ClockKind::Clock32 => {
                        let id = self.builder.u_convert(self.u32_type, None, ticks)?;
                        self.write(dst[0], Value { id, kind: Kind::B32 })
                    }
                }
            }
            ExtOp::CvtaCopy => {
                let value = self.read(src[0])?;
                if value.kind == Kind::B32 {
                    return Err(Error::new("cvta expects a 64-bit address"));
                }
                self.write(dst[0], value)
            }
            ExtOp::IsSpace(space) => {
                let value = self.read(src[0])?;
                let holds = match value.kind {
                    Kind::Address(actual) => actual == space,
                    // Untyped 64-bit values are device pointers in this subset.
                    Kind::B64 => space == Space::Global,
                    _ => false,
                };
                let id = if holds {
                    self.builder.constant_true(self.bool_type)
                } else {
                    self.builder.constant_false(self.bool_type)
                };
                self.write(dst[0], Value { id, kind: Kind::Predicate })
            }
            ExtOp::Redux { op, signed } => {
                self.declare("arith", Some(Capability::GroupNonUniformArithmetic), None);
                let ty = self.u32_type;
                let value = self.integer(src[0], Kind::B32)?;
                let scope = self.builder.constant_bit32(ty, Scope::Subgroup as u32);
                let reduce = GroupOperation::Reduce;
                let id = match (op, signed) {
                    (ReduxOp::Add, _) => {
                        self.builder.group_non_uniform_i_add(ty, None, scope, reduce, value, None)?
                    }
                    (ReduxOp::Min, true) => {
                        self.builder.group_non_uniform_s_min(ty, None, scope, reduce, value, None)?
                    }
                    (ReduxOp::Min, false) => {
                        self.builder.group_non_uniform_u_min(ty, None, scope, reduce, value, None)?
                    }
                    (ReduxOp::Max, true) => {
                        self.builder.group_non_uniform_s_max(ty, None, scope, reduce, value, None)?
                    }
                    (ReduxOp::Max, false) => {
                        self.builder.group_non_uniform_u_max(ty, None, scope, reduce, value, None)?
                    }
                    (ReduxOp::And, _) => self
                        .builder
                        .group_non_uniform_bitwise_and(ty, None, scope, reduce, value, None)?,
                    (ReduxOp::Or, _) => self
                        .builder
                        .group_non_uniform_bitwise_or(ty, None, scope, reduce, value, None)?,
                    (ReduxOp::Xor, _) => self
                        .builder
                        .group_non_uniform_bitwise_xor(ty, None, scope, reduce, value, None)?,
                };
                self.write(dst[0], Value { id, kind: Kind::B32 })
            }
            ExtOp::MatchAny { wide } => {
                let ty = self.u32_type;
                let (lo, hi) = self.match_operand(src[0], wide)?;
                let mut acc = self.u32_const(0);
                let zero = self.u32_const(0);
                for lane in 0..32u32 {
                    let index = self.u32_const(lane);
                    let other_lo = self.subgroup_shuffle(lo, index)?;
                    let mut same = self.builder.i_equal(self.bool_type, None, lo, other_lo)?;
                    if let Some(hi) = hi {
                        let other_hi = self.subgroup_shuffle(hi, index)?;
                        let same_hi = self.builder.i_equal(self.bool_type, None, hi, other_hi)?;
                        same = self.builder.logical_and(self.bool_type, None, same, same_hi)?;
                    }
                    let bit = self.u32_const(1 << lane);
                    let picked = self.builder.select(ty, None, same, bit, zero)?;
                    acc = self.builder.bitwise_or(ty, None, acc, picked)?;
                }
                self.write(dst[0], Value { id: acc, kind: Kind::B32 })
            }
            ExtOp::MatchAll { wide } => {
                self.declare("vote", Some(Capability::GroupNonUniformVote), None);
                let ty = self.u32_type;
                let (lo, hi) = self.match_operand(src[0], wide)?;
                let scope = self.builder.constant_bit32(ty, Scope::Subgroup as u32);
                let mut all = self.subgroup_all_equal(scope, lo)?;
                if let Some(hi) = hi {
                    let hi_all = self.subgroup_all_equal(scope, hi)?;
                    all = self.builder.logical_and(self.bool_type, None, all, hi_all)?;
                }
                if !dst[0].is_empty() {
                    let ones = self.u32_const(u32::MAX);
                    let zero = self.u32_const(0);
                    let id = self.builder.select(ty, None, all, ones, zero)?;
                    self.write(dst[0], Value { id, kind: Kind::B32 })?;
                }
                if !dst[1].is_empty() {
                    self.write(dst[1], Value { id: all, kind: Kind::Predicate })?;
                }
                Ok(())
            }
            ExtOp::VoteUni => {
                self.declare("vote", Some(Capability::GroupNonUniformVote), None);
                let predicate = self.predicate_operand(src[0])?;
                let scope = self.builder.constant_bit32(self.u32_type, Scope::Subgroup as u32);
                let ty = self.u32_type;
                let (one, zero) = (self.u32_const(1), self.u32_const(0));
                let as_word = self.builder.select(ty, None, predicate, one, zero)?;
                let id = self.subgroup_all_equal(scope, as_word)?;
                self.write(dst[0], Value { id, kind: Kind::Predicate })
            }
            ExtOp::Elect => {
                let scope = self.builder.constant_bit32(self.u32_type, Scope::Subgroup as u32);
                let elected = self.builder.group_non_uniform_elect(self.bool_type, None, scope)?;
                if !dst[0].is_empty() {
                    self.declare("ballot", Some(Capability::GroupNonUniformBallot), None);
                    let lane = self.lane_id()?;
                    let id = self.builder.group_non_uniform_broadcast_first(
                        self.u32_type,
                        None,
                        scope,
                        lane,
                    )?;
                    self.write(dst[0], Value { id, kind: Kind::B32 })?;
                }
                if !dst[1].is_empty() {
                    self.write(dst[1], Value { id: elected, kind: Kind::Predicate })?;
                }
                Ok(())
            }
            ExtOp::ShflDyn(mode) => {
                let value = self.integer(src[0], Kind::B32)?;
                let delta = self.integer(src[1], Kind::B32)?;
                let control = self.integer(src[2], Kind::B32)?;
                let (result, valid) = self.shuffle_dynamic(mode, value, delta, control)?;
                self.write(dst[0], Value { id: result, kind: Kind::B32 })?;
                if !dst[1].is_empty() {
                    self.write(dst[1], Value { id: valid, kind: Kind::Predicate })?;
                }
                Ok(())
            }
            ExtOp::Atom { op, ty, space, returns } => {
                self.atom64(op, ty, space, returns, dst[0], src)
            }
            ExtOp::AtomIncDec { dec, space, returns } => {
                self.atom_inc_dec(dec, space, returns, dst[0], src)
            }
        }
    }

    /// Reads a predicate operand, accepting `!%p`-free register names only.
    fn predicate_operand(&mut self, operand: &str) -> R<Word> {
        if operand == "1" {
            return Ok(self.builder.constant_true(self.bool_type));
        }
        if operand == "0" {
            return Ok(self.builder.constant_false(self.bool_type));
        }
        self.read(operand)?.require(Kind::Predicate, operand)
    }

    fn match_operand(&mut self, operand: &str, wide: bool) -> R<(Word, Option<Word>)> {
        if !wide {
            return Ok((self.integer(operand, Kind::B32)?, None));
        }
        let x = self.integer(operand, Kind::B64)?;
        let lo = self.builder.u_convert(self.u32_type, None, x)?;
        let sh = self.builder.constant_bit64(self.u64_type, 32);
        let hi = self.builder.shift_right_logical(self.u64_type, None, x, sh)?;
        let hi = self.builder.u_convert(self.u32_type, None, hi)?;
        Ok((lo, Some(hi)))
    }

    fn shl_sat(&mut self, x: Word, n: Word) -> R<Word> {
        let ty = self.u32_type;
        let c32 = self.u32_const(32);
        let zero = self.u32_const(0);
        let ok = self.builder.u_less_than(self.bool_type, None, n, c32)?;
        let safe = self.builder.select(ty, None, ok, n, zero)?;
        let shifted = self.builder.shift_left_logical(ty, None, x, safe)?;
        Ok(self.builder.select(ty, None, ok, shifted, zero)?)
    }

    /// `OpGroupNonUniformAllEqual` is not provided by the Intel OpenCL backend
    /// (undefined `__spirv_GroupNonUniformAllEqual`), so compare against the first active lane.
    fn subgroup_all_equal(&mut self, scope: Word, value: Word) -> R<Word> {
        self.declare("ballot", Some(Capability::GroupNonUniformBallot), None);
        let first = self
            .builder
            .group_non_uniform_broadcast_first(self.u32_type, None, scope, value)?;
        let same = self.builder.i_equal(self.bool_type, None, value, first)?;
        Ok(self.builder.group_non_uniform_all(self.bool_type, None, scope, same)?)
    }

    pub(super) fn linear_tid(&mut self) -> R<Word> {
        let ty = self.u32_type;
        let tid: Vec<Word> = ["x", "y", "z"]
            .iter()
            .map(|axis| self.builtin(&format!("%tid.{axis}")))
            .collect::<Result<_, _>>()?;
        let ntid: Vec<Word> = ["x", "y"]
            .iter()
            .map(|axis| self.builtin(&format!("%ntid.{axis}")))
            .collect::<Result<_, _>>()?;
        let inner = self.builder.i_mul(ty, None, tid[2], ntid[1])?;
        let inner = self.builder.i_add(ty, None, inner, tid[1])?;
        let linear = self.builder.i_mul(ty, None, inner, ntid[0])?;
        Ok(self.builder.i_add(ty, None, linear, tid[0])?)
    }

    fn block_size(&mut self) -> R<Word> {
        let ty = self.u32_type;
        let x = self.builtin("%ntid.x")?;
        let y = self.builtin("%ntid.y")?;
        let z = self.builtin("%ntid.z")?;
        let xy = self.builder.i_mul(ty, None, x, y)?;
        Ok(self.builder.i_mul(ty, None, xy, z)?)
    }

    pub(super) fn const_of(&mut self, wide: bool, value: u64) -> Word {
        if wide {
            self.builder.constant_bit64(self.u64_type, value)
        } else {
            self.builder.constant_bit32(self.u32_type, value as u32)
        }
    }

    fn brev(&mut self, mut x: Word, wide: bool) -> R<Word> {
        let ty = if wide { self.u64_type } else { self.u32_type };
        let steps: &[(u32, u64)] = if wide {
            &[
                (1, 0x5555_5555_5555_5555),
                (2, 0x3333_3333_3333_3333),
                (4, 0x0f0f_0f0f_0f0f_0f0f),
                (8, 0x00ff_00ff_00ff_00ff),
                (16, 0x0000_ffff_0000_ffff),
                (32, 0x0000_0000_ffff_ffff),
            ]
        } else {
            &[
                (1, 0x5555_5555),
                (2, 0x3333_3333),
                (4, 0x0f0f_0f0f),
                (8, 0x00ff_00ff),
                (16, 0x0000_ffff),
            ]
        };
        for &(shift, mask) in steps {
            let s = self.const_of(wide, u64::from(shift));
            let m = self.const_of(wide, mask);
            let down = self.builder.shift_right_logical(ty, None, x, s)?;
            let down = self.builder.bitwise_and(ty, None, down, m)?;
            let up = self.builder.bitwise_and(ty, None, x, m)?;
            let up = self.builder.shift_left_logical(ty, None, up, s)?;
            x = self.builder.bitwise_or(ty, None, down, up)?;
        }
        Ok(x)
    }

    /// `fns.b32`: position of the `offset`-th set bit of `mask` relative to `base`.
    fn fns(&mut self, mask: Word, base: Word, offset: Word) -> R<Word> {
        let ty = self.u32_type;
        let b = self.bool_type;
        let none = self.u32_const(u32::MAX);
        let zero = self.u32_const(0);
        let one = self.u32_const(1);
        let c32 = self.u32_const(32);
        let two = self.u32_const(2);
        let base_ok = self.builder.u_less_than(b, None, base, c32)?;
        // offset == 0: the base bit itself.
        let safe_base = self.builder.select(ty, None, base_ok, base, zero)?;
        let base_shifted = self.builder.shift_right_logical(ty, None, mask, safe_base)?;
        let base_bit = self.builder.bitwise_and(ty, None, base_shifted, one)?;
        let base_set = self.builder.i_not_equal(b, None, base_bit, zero)?;
        let base_hit = self.builder.logical_and(b, None, base_ok, base_set)?;
        let at_base = self.builder.select(ty, None, base_hit, base, none)?;
        // Bits strictly above / below the base.
        let upto = self.shl_sat(two, base)?;
        let up_low = self.builder.i_sub(ty, None, upto, one)?;
        let up_mask = self.builder.not(ty, None, up_low)?;
        let above = self.builder.bitwise_and(ty, None, mask, up_mask)?;
        let below_mask = self.shl_sat(one, base)?;
        let below_mask = self.builder.i_sub(ty, None, below_mask, one)?;
        let below_all = self.builder.select(ty, None, base_ok, below_mask, mask)?;
        let below = self.builder.bitwise_and(ty, None, mask, below_all)?;
        let negative = self.builder.s_less_than(b, None, offset, zero)?;
        let magnitude = {
            let neg = self.builder.i_sub(ty, None, zero, offset)?;
            self.builder.select(ty, None, negative, neg, offset)?
        };
        let mut up_result = none;
        let mut count = zero;
        for i in 0..32u32 {
            let shift = self.u32_const(i);
            let bit = self.builder.shift_right_logical(ty, None, above, shift)?;
            let bit = self.builder.bitwise_and(ty, None, bit, one)?;
            count = self.builder.i_add(ty, None, count, bit)?;
            let is_set = self.builder.i_not_equal(b, None, bit, zero)?;
            let reached = self.builder.i_equal(b, None, count, magnitude)?;
            let hit = self.builder.logical_and(b, None, is_set, reached)?;
            let index = self.u32_const(i);
            up_result = self.builder.select(ty, None, hit, index, up_result)?;
        }
        let mut down_result = none;
        let mut count = zero;
        for i in (0..32u32).rev() {
            let shift = self.u32_const(i);
            let bit = self.builder.shift_right_logical(ty, None, below, shift)?;
            let bit = self.builder.bitwise_and(ty, None, bit, one)?;
            count = self.builder.i_add(ty, None, count, bit)?;
            let is_set = self.builder.i_not_equal(b, None, bit, zero)?;
            let reached = self.builder.i_equal(b, None, count, magnitude)?;
            let hit = self.builder.logical_and(b, None, is_set, reached)?;
            let index = self.u32_const(i);
            down_result = self.builder.select(ty, None, hit, index, down_result)?;
        }
        let directed = self.builder.select(ty, None, negative, down_result, up_result)?;
        let is_zero = self.builder.i_equal(b, None, offset, zero)?;
        Ok(self.builder.select(ty, None, is_zero, at_base, directed)?)
    }

    fn mul24(&mut self, a: Word, b: Word, hi: bool, signed: bool) -> R<Word> {
        let ty = self.u32_type;
        let (a, b) = if signed {
            let eight = self.u32_const(8);
            let a = self.builder.shift_left_logical(ty, None, a, eight)?;
            let a = self.builder.shift_right_arithmetic(ty, None, a, eight)?;
            let b = self.builder.shift_left_logical(ty, None, b, eight)?;
            let b = self.builder.shift_right_arithmetic(ty, None, b, eight)?;
            (a, b)
        } else {
            let mask = self.u32_const(0x00ff_ffff);
            (
                self.builder.bitwise_and(ty, None, a, mask)?,
                self.builder.bitwise_and(ty, None, b, mask)?,
            )
        };
        if !hi {
            return Ok(self.builder.i_mul(ty, None, a, b)?);
        }
        let t64 = self.u64_type;
        let (a, b) = if signed {
            (self.builder.s_convert(t64, None, a)?, self.builder.s_convert(t64, None, b)?)
        } else {
            (self.builder.u_convert(t64, None, a)?, self.builder.u_convert(t64, None, b)?)
        };
        let product = self.builder.i_mul(t64, None, a, b)?;
        let sixteen = self.builder.constant_bit64(t64, 16);
        let shifted = if signed {
            self.builder.shift_right_arithmetic(t64, None, product, sixteen)?
        } else {
            self.builder.shift_right_logical(t64, None, product, sixteen)?
        };
        Ok(self.builder.u_convert(ty, None, shifted)?)
    }

    fn bfind(&mut self, source: &str, wide: bool, signed: bool, shiftamt: bool) -> R<Word> {
        let b = self.bool_type;
        let ty = self.u32_type;
        let none = self.u32_const(u32::MAX);
        let (leading, width, zero_test) = if wide {
            let mut x = self.integer(source, Kind::B64)?;
            let zero = self.builder.constant_bit64(self.u64_type, 0);
            if signed {
                let negative = self.builder.s_less_than(b, None, x, zero)?;
                let inverted = self.builder.not(self.u64_type, None, x)?;
                x = self.builder.select(self.u64_type, None, negative, inverted, x)?;
            }
            let is_zero = self.builder.i_equal(b, None, x, zero)?;
            let count = self.opencl_int(self.u64_type, OpenclStd100Op::clz, &[x])?;
            (self.builder.u_convert(ty, None, count)?, 63u32, is_zero)
        } else {
            let mut x = self.integer(source, Kind::B32)?;
            let zero = self.u32_const(0);
            if signed {
                let negative = self.builder.s_less_than(b, None, x, zero)?;
                let inverted = self.builder.not(ty, None, x)?;
                x = self.builder.select(ty, None, negative, inverted, x)?;
            }
            let is_zero = self.builder.i_equal(b, None, x, zero)?;
            (self.opencl_int(ty, OpenclStd100Op::clz, &[x])?, 31u32, is_zero)
        };
        let result = if shiftamt {
            leading
        } else {
            let top = self.u32_const(width);
            self.builder.i_sub(ty, None, top, leading)?
        };
        Ok(self.builder.select(ty, None, zero_test, none, result)?)
    }

    fn mask64(&mut self, len: Word) -> R<Word> {
        // len in 0..=255: all ones for len >= 64, zero for len == 0.
        let t = self.u64_type;
        let b = self.bool_type;
        let len64 = self.builder.u_convert(t, None, len)?;
        let c64 = self.builder.constant_bit64(t, 64);
        let zero = self.builder.constant_bit64(t, 0);
        let ones = self.builder.constant_bit64(t, u64::MAX);
        let small = self.builder.u_less_than(b, None, len64, c64)?;
        let safe = self.builder.select(t, None, small, len64, zero)?;
        let shifted = self.builder.shift_left_logical(t, None, ones, safe)?;
        let mask = self.builder.not(t, None, shifted)?;
        Ok(self.builder.select(t, None, small, mask, ones)?)
    }

    fn bfe64(&mut self, src: [&str; 4], signed: bool) -> R<Word> {
        let t = self.u64_type;
        let b = self.bool_type;
        let a = self.integer(src[0], Kind::B64)?;
        let pos = self.integer(src[1], Kind::B32)?;
        let len = self.integer(src[2], Kind::B32)?;
        let ff = self.u32_const(0xff);
        let pos = self.builder.bitwise_and(self.u32_type, None, pos, ff)?;
        let len = self.builder.bitwise_and(self.u32_type, None, len, ff)?;
        let mask = self.mask64(len)?;
        let pos64 = self.builder.u_convert(t, None, pos)?;
        let c63 = self.builder.constant_bit64(t, 63);
        let zero = self.builder.constant_bit64(t, 0);
        let one = self.builder.constant_bit64(t, 1);
        if !signed {
            let c64 = self.builder.constant_bit64(t, 64);
            let in_range = self.builder.u_less_than(b, None, pos64, c64)?;
            let safe = self.builder.select(t, None, in_range, pos64, zero)?;
            let shifted = self.builder.shift_right_logical(t, None, a, safe)?;
            let shifted = self.builder.select(t, None, in_range, shifted, zero)?;
            return Ok(self.builder.bitwise_and(t, None, shifted, mask)?);
        }
        let big = self.builder.u_less_than(b, None, c63, pos64)?;
        let pos63 = self.builder.select(t, None, big, c63, pos64)?;
        let shifted = self.builder.shift_right_arithmetic(t, None, a, pos63)?;
        let field = self.builder.bitwise_and(t, None, shifted, mask)?;
        let len64 = self.builder.u_convert(t, None, len)?;
        let last = self.builder.i_add(t, None, pos64, len64)?;
        let last = self.builder.i_sub(t, None, last, one)?;
        let over = self.builder.u_less_than(b, None, c63, last)?;
        let last = self.builder.select(t, None, over, c63, last)?;
        let sign = self.builder.shift_right_logical(t, None, a, last)?;
        let sign = self.builder.bitwise_and(t, None, sign, one)?;
        let len_zero = self.builder.i_equal(b, None, len64, zero)?;
        let sign = self.builder.select(t, None, len_zero, zero, sign)?;
        let negative = self.builder.i_not_equal(b, None, sign, zero)?;
        let inverse = self.builder.not(t, None, mask)?;
        let extended = self.builder.bitwise_or(t, None, field, inverse)?;
        Ok(self.builder.select(t, None, negative, extended, field)?)
    }

    fn bfi64(&mut self, src: [&str; 4]) -> R<Word> {
        let t = self.u64_type;
        let b = self.bool_type;
        let f = self.integer(src[0], Kind::B64)?;
        let a = self.integer(src[1], Kind::B64)?;
        let pos = self.integer(src[2], Kind::B32)?;
        let len = self.integer(src[3], Kind::B32)?;
        let ff = self.u32_const(0xff);
        let pos = self.builder.bitwise_and(self.u32_type, None, pos, ff)?;
        let len = self.builder.bitwise_and(self.u32_type, None, len, ff)?;
        let mask = self.mask64(len)?;
        let pos64 = self.builder.u_convert(t, None, pos)?;
        let c64 = self.builder.constant_bit64(t, 64);
        let zero = self.builder.constant_bit64(t, 0);
        let in_range = self.builder.u_less_than(b, None, pos64, c64)?;
        let safe = self.builder.select(t, None, in_range, pos64, zero)?;
        let field = self.builder.shift_left_logical(t, None, mask, safe)?;
        let moved = self.builder.shift_left_logical(t, None, f, safe)?;
        let moved = self.builder.bitwise_and(t, None, moved, field)?;
        let keep = self.builder.not(t, None, field)?;
        let keep = self.builder.bitwise_and(t, None, a, keep)?;
        let merged = self.builder.bitwise_or(t, None, keep, moved)?;
        Ok(self.builder.select(t, None, in_range, merged, a)?)
    }

    /// Byte permute of `{b, a}` with a selector word (only nibble bits 0-2 and 3 used).
    fn prmt(&mut self, a: Word, b: Word, sel: Word) -> R<Word> {
        let ty = self.u32_type;
        let t64 = self.u64_type;
        let a64 = self.builder.u_convert(t64, None, a)?;
        let b64 = self.builder.u_convert(t64, None, b)?;
        let c32 = self.builder.constant_bit64(t64, 32);
        let hi = self.builder.shift_left_logical(t64, None, b64, c32)?;
        let pair = self.builder.bitwise_or(t64, None, a64, hi)?;
        let mut result = self.u32_const(0);
        let zero = self.u32_const(0);
        for lane in 0..4u32 {
            let nibble_shift = self.u32_const(4 * lane);
            let nibble = self.builder.shift_right_logical(ty, None, sel, nibble_shift)?;
            let seven = self.u32_const(7);
            let index = self.builder.bitwise_and(ty, None, nibble, seven)?;
            let three = self.u32_const(3);
            let bit_index = self.builder.shift_left_logical(ty, None, index, three)?;
            let bit_index = self.builder.u_convert(t64, None, bit_index)?;
            let picked = self.builder.shift_right_logical(t64, None, pair, bit_index)?;
            let picked = self.builder.u_convert(ty, None, picked)?;
            let byte_mask = self.u32_const(0xff);
            let byte = self.builder.bitwise_and(ty, None, picked, byte_mask)?;
            let sign = self.builder.shift_right_logical(ty, None, byte, seven)?;
            let negated = self.builder.i_sub(ty, None, zero, sign)?;
            let smear = self.builder.bitwise_and(ty, None, negated, byte_mask)?;
            let eight = self.u32_const(8);
            let flag = self.builder.bitwise_and(ty, None, nibble, eight)?;
            let replicate = self.builder.i_not_equal(self.bool_type, None, flag, zero)?;
            let value = self.builder.select(ty, None, replicate, smear, byte)?;
            let lane_shift = self.u32_const(8 * lane);
            let placed = self.builder.shift_left_logical(ty, None, value, lane_shift)?;
            result = self.builder.bitwise_or(ty, None, result, placed)?;
        }
        Ok(result)
    }

    fn shuffle_dynamic(
        &mut self,
        mode: ShflMode,
        value: Word,
        delta: Word,
        control: Word,
    ) -> R<(Word, Word)> {
        let ty = self.u32_type;
        let lane = self.lane_id()?;
        let m31 = self.u32_const(31);
        let m8 = self.u32_const(8);
        let clamp = self.builder.bitwise_and(ty, None, control, m31)?;
        let seg = self.builder.shift_right_logical(ty, None, control, m8)?;
        let segment = self.builder.bitwise_and(ty, None, seg, m31)?;
        let min_lane = self.builder.bitwise_and(ty, None, lane, segment)?;
        let not_segment = self.builder.not(ty, None, segment)?;
        let not_segment = self.builder.bitwise_and(ty, None, not_segment, m31)?;
        let upper = self.builder.bitwise_and(ty, None, clamp, not_segment)?;
        let max_lane = self.builder.bitwise_or(ty, None, min_lane, upper)?;
        let (j, valid) = match mode {
            ShflMode::Up => {
                let room = self.builder.i_sub(ty, None, lane, min_lane)?;
                let valid = self.builder.u_less_than_equal(self.bool_type, None, delta, room)?;
                (self.builder.i_sub(ty, None, lane, delta)?, valid)
            }
            _ => {
                let j = match mode {
                    ShflMode::Idx => {
                        let low = self.builder.bitwise_and(ty, None, delta, not_segment)?;
                        self.builder.bitwise_or(ty, None, low, min_lane)?
                    }
                    ShflMode::Down => self.builder.i_add(ty, None, lane, delta)?,
                    _ => self.builder.bitwise_xor(ty, None, lane, delta)?,
                };
                let valid = self.builder.u_less_than_equal(self.bool_type, None, j, max_lane)?;
                (j, valid)
            }
        };
        let source = self.builder.select(ty, None, valid, j, lane)?;
        let result = self.subgroup_shuffle(value, source)?;
        Ok((result, valid))
    }

    /// Compare-and-swap loop applying `update(old) -> new` to a 32/64-bit word.
    /// Returns the value observed before the successful swap.
    fn cas_loop(
        &mut self,
        pointer: Word,
        wide: bool,
        scope: Word,
        mut update: impl FnMut(&mut Self, Word) -> R<Word>,
    ) -> R<Word> {
        let ty = if wide { self.u64_type } else { self.u32_type };
        let relaxed = self.builder.constant_bit32(self.u32_type, MemorySemantics::empty().bits());
        let slot_ptr = self.builder.type_pointer(None, StorageClass::Function, ty);
        let slot = self.builder.id();
        self.locals.push(super::SpirvInstruction::new(
            rspirv::spirv::Op::Variable,
            Some(slot_ptr),
            Some(slot),
            vec![rspirv::dr::Operand::StorageClass(StorageClass::Function)],
        ));
        let initial = self.builder.atomic_load(ty, None, pointer, scope, relaxed)?;
        self.builder.store(slot, initial, None, [])?;
        let header = self.builder.id();
        let body = self.builder.id();
        let cont = self.builder.id();
        let merge = self.builder.id();
        self.builder.branch(header)?;
        self.builder.begin_block(Some(header))?;
        self.builder.branch(body)?;
        self.builder.begin_block(Some(body))?;
        let old = self.builder.load(ty, None, slot, None, [])?;
        let new = update(self, old)?;
        let seen = self
            .builder
            .atomic_compare_exchange(ty, None, pointer, scope, relaxed, relaxed, new, old)?;
        self.builder.store(slot, seen, None, [])?;
        let done = self.builder.i_equal(self.bool_type, None, seen, old)?;
        self.builder.branch_conditional(done, merge, cont, [])?;
        self.builder.begin_block(Some(cont))?;
        self.builder.branch(header)?;
        self.builder.begin_block(Some(merge))?;
        Ok(self.builder.load(ty, None, slot, None, [])?)
    }

    fn atom_scope(&mut self, space: Space) -> Word {
        let kind = if space == Space::Global {
            self.device_atomics = true;
            Scope::Device
        } else {
            Scope::Workgroup
        };
        self.builder.constant_bit32(self.u32_type, kind as u32)
    }

    fn atom_inc_dec(
        &mut self,
        dec: bool,
        space: Space,
        returns: bool,
        dst: &'a str,
        src: [&str; 4],
    ) -> R<()> {
        if space == Space::Local {
            return Err(Error::new("atomics on local memory are not supported"));
        }
        let pointer = self.word_pointer(space, src[0])?;
        let limit = self.integer(src[1], Kind::B32)?;
        let scope = self.atom_scope(space);
        let b = self.bool_type;
        let old = self.cas_loop(pointer, false, scope, |e, old| {
            let ty = e.u32_type;
            let zero = e.u32_const(0);
            let one = e.u32_const(1);
            if dec {
                // old == 0 || old > limit ? limit : old - 1
                let is_zero = e.builder.i_equal(b, None, old, zero)?;
                let above = e.builder.u_greater_than(b, None, old, limit)?;
                let reset = e.builder.logical_or(b, None, is_zero, above)?;
                let minus = e.builder.i_sub(ty, None, old, one)?;
                Ok(e.builder.select(ty, None, reset, limit, minus)?)
            } else {
                // old >= limit ? 0 : old + 1
                let reset = e.builder.u_greater_than_equal(b, None, old, limit)?;
                let plus = e.builder.i_add(ty, None, old, one)?;
                Ok(e.builder.select(ty, None, reset, zero, plus)?)
            }
        })?;
        if returns {
            self.write(dst, Value { id: old, kind: Kind::B32 })?;
        }
        Ok(())
    }

    fn atom64(
        &mut self,
        op: AtomicOp,
        ty: AtomTy,
        space: Space,
        returns: bool,
        dst: &'a str,
        src: [&str; 4],
    ) -> R<()> {
        if space == Space::Local {
            return Err(Error::new("atomics on local memory are not supported"));
        }
        if matches!(ty, AtomTy::F16x2 | AtomTy::Bf16x2) {
            return self.atom_half2_add(ty == AtomTy::Bf16x2, space, returns, dst, src);
        }
        let wide = matches!(ty, AtomTy::U64 | AtomTy::S64 | AtomTy::F64);
        let kind = if wide { Kind::B64 } else { Kind::B32 };
        if wide {
            self.declare("int64_atomics", Some(Capability::Int64Atomics), None);
        }
        let bytes_ptr = self.addressed_byte_pointer(space, src[0])?;
        let storage = if space == Space::Global {
            StorageClass::CrossWorkgroup
        } else {
            StorageClass::Workgroup
        };
        let word_ty = if wide { self.u64_type } else { self.u32_type };
        let ptr_ty = self.builder.type_pointer(None, storage, word_ty);
        let pointer = self.builder.bitcast(ptr_ty, None, bytes_ptr)?;
        let scope = self.atom_scope(space);
        let relaxed = self.builder.constant_bit32(self.u32_type, MemorySemantics::empty().bits());
        let value = self.integer(src[1], kind)?;
        let b = self.bool_type;
        let old = match (op, ty) {
            (AtomicOp::Add, AtomTy::U64 | AtomTy::S64) => {
                self.builder.atomic_i_add(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Min, AtomTy::U64) => {
                self.builder.atomic_u_min(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Min, AtomTy::S64) => {
                self.builder.atomic_s_min(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Max, AtomTy::U64) => {
                self.builder.atomic_u_max(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Max, AtomTy::S64) => {
                self.builder.atomic_s_max(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::And, AtomTy::U64 | AtomTy::S64) => {
                self.builder.atomic_and(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Or, AtomTy::U64 | AtomTy::S64) => {
                self.builder.atomic_or(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Xor, AtomTy::U64 | AtomTy::S64) => {
                self.builder.atomic_xor(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Exch, _) => {
                self.builder.atomic_exchange(word_ty, None, pointer, scope, relaxed, value)?
            }
            (AtomicOp::Cas, _) => {
                let compare = self.integer(src[2], kind)?;
                self.builder.atomic_compare_exchange(
                    word_ty, None, pointer, scope, relaxed, relaxed, value, compare,
                )?
            }
            // Floating-point read-modify-write has no native Intel atomic: use a CAS loop.
            (AtomicOp::Add, AtomTy::F64) => {
                self.declare("fp64", Some(Capability::Float64), None);
                self.cas_loop(pointer, true, scope, |e, old| {
                    let f = e.builder.type_float(64, None);
                    let x = e.builder.bitcast(f, None, old)?;
                    let y = e.builder.bitcast(f, None, value)?;
                    let sum = e.builder.f_add(f, None, x, y)?;
                    Ok(e.builder.bitcast(e.u64_type, None, sum)?)
                })?
            }
            (AtomicOp::Min | AtomicOp::Max, AtomTy::F32 | AtomTy::F64) => {
                let is_min = op == AtomicOp::Min;
                let double = ty == AtomTy::F64;
                if double {
                    self.declare("fp64", Some(Capability::Float64), None);
                }
                self.cas_loop(pointer, double, scope, |e, old| {
                    let f = e.builder.type_float(if double { 64 } else { 32 }, None);
                    let x = e.builder.bitcast(f, None, old)?;
                    let y = e.builder.bitcast(f, None, value)?;
                    // Replace when the operand is strictly smaller/larger (or the old value is NaN).
                    let better = if is_min {
                        e.builder.f_ord_less_than(b, None, y, x)?
                    } else {
                        e.builder.f_ord_greater_than(b, None, y, x)?
                    };
                    let old_nan = e.builder.f_unord_not_equal(b, None, x, x)?;
                    let take = e.builder.logical_or(b, None, better, old_nan)?;
                    Ok(e.builder.select(word_ty, None, take, value, old)?)
                })?
            }
            _ => return Err(Error::new("unsupported PTX atomic operation")),
        };
        if returns {
            self.write(dst, Value { id: old, kind })?;
        }
        Ok(())
    }

    /// `atom.add.noftz.{f16x2,bf16x2}`: per-lane half add inside a CAS loop.
    fn atom_half2_add(
        &mut self,
        bf16: bool,
        space: Space,
        returns: bool,
        dst: &'a str,
        src: [&str; 4],
    ) -> R<()> {
        if space == Space::Local {
            return Err(Error::new("atomics on local memory are not supported"));
        }
        let pointer = self.word_pointer(space, src[0])?;
        let scope = self.atom_scope(space);
        let value = self.integer(src[1], Kind::B32)?;
        let ty = super::super::ext::NumTy::F16x2;
        let _ = ty;
        let old = self.cas_loop(pointer, false, scope, |e, old| {
            e.half2_add_bits(old, value, bf16)
        })?;
        if returns {
            self.write(dst, Value { id: old, kind: Kind::B32 })?;
        }
        Ok(())
    }
}

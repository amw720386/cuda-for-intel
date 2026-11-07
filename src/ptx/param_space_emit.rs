//! SPIR-V lowering for [`super::super::param_space`]. Child module of `spirv.rs`.

use super::super::param_space::{PLoad, PTy, ParamOp, layout};
use super::super::{Error, Param};
use super::{Emitter, Kind, Value};
use rspirv::spirv::{MemorySemantics, Scope, StorageClass, Word};
use std::collections::BTreeMap;

type R<T> = Result<T, Error>;

impl<'a> Emitter<'a> {
    /// Module-scope part: the workgroup array (declared before the function begins).
    pub(super) fn declare_param_array(&mut self, params: &[(&'a str, Param)]) {
        let (_, total) = layout(params);
        let length = self.u32_const(total.div_ceil(4).max(1));
        let array = self.builder.type_array(self.u32_type, length);
        let pointer = self.builder.type_pointer(None, StorageClass::Workgroup, array);
        let variable = self.builder.variable(pointer, None, StorageClass::Workgroup, None);
        self.param_array = Some(variable);
    }

    /// Copies every kernel parameter into a workgroup array laid out like the launch
    /// argument buffer. Local thread 0 stores, a barrier publishes. Must run in the
    /// entry block after the parameter values exist.
    pub(super) fn setup_param_array(&mut self, params: &[(&'a str, Param)]) -> R<()> {
        let (offsets, total) = layout(params);
        self.param_bases.clear();
        let mut words: BTreeMap<u32, Word> = BTreeMap::new();
        let mut put = |this: &mut Self, index: u32, value: Word, shift: u32| -> R<()> {
            let value = if shift == 0 {
                value
            } else {
                let amount = this.u32_const(shift);
                this.builder.shift_left_logical(this.u32_type, None, value, amount)?
            };
            let merged = match words.get(&index).copied() {
                Some(old) => this.builder.bitwise_or(this.u32_type, None, old, value)?,
                None => value,
            };
            words.insert(index, merged);
            Ok(())
        };
        for (&(name, param), &base) in params.iter().zip(&offsets) {
            self.param_bases.insert(name, base);
            let index = base / 4;
            match param {
                Param::Words { words: count, .. } => {
                    for k in 0..u32::from(count) {
                        let value = self
                            .params
                            .get(&(name, (k * 4) as usize))
                            .ok_or_else(|| Error::new("missing aggregate parameter word"))?
                            .id;
                        put(self, index + k, value, 0)?;
                    }
                }
                Param::U8 | Param::U16 | Param::U32 | Param::F32 => {
                    let value = self.params[&(name, 0)].id;
                    put(self, index, value, (base % 4) * 8)?;
                }
                Param::U64 | Param::GlobalPtr => {
                    let value = self.params[&(name, 0)];
                    let wide = if matches!(value.kind, Kind::Address(_)) {
                        self.builder.bitcast(self.u64_type, None, value.id)?
                    } else {
                        value.id
                    };
                    let lo = self.builder.u_convert(self.u32_type, None, wide)?;
                    let sh = self.builder.constant_bit64(self.u64_type, 32);
                    let top = self.builder.shift_right_logical(self.u64_type, None, wide, sh)?;
                    let hi = self.builder.u_convert(self.u32_type, None, top)?;
                    put(self, index, lo, 0)?;
                    put(self, index + 1, hi, 0)?;
                }
            }
        }
        let variable = self.param_array.expect("declared before the function starts");
        let _ = total;

        let (tx, ty, tz) = (self.builtin("%tid.x")?, self.builtin("%tid.y")?, self.builtin("%tid.z")?);
        let any = self.builder.bitwise_or(self.u32_type, None, tx, ty)?;
        let any = self.builder.bitwise_or(self.u32_type, None, any, tz)?;
        let zero = self.u32_const(0);
        let first = self.builder.i_equal(self.bool_type, None, any, zero)?;
        let (store, merge) = (self.builder.id(), self.builder.id());
        self.builder.branch_conditional(first, store, merge, [])?;
        self.builder.begin_block(Some(store))?;
        let first_word = self.param_word_pointer(variable)?;
        for (&index, &value) in &words {
            let at = self.builder.constant_bit64(self.u64_type, u64::from(index));
            let slot = self.builder.ptr_access_chain(self.shared_word_ptr, None, first_word, at, [])?;
            self.builder.store(slot, value, None, [])?;
        }
        self.builder.branch(merge)?;
        self.builder.begin_block(Some(merge))?;
        let scope = self.u32_const(Scope::Workgroup as u32);
        let semantics = self.u32_const(
            (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits(),
        );
        self.builder.control_barrier(scope, scope, semantics)?;
        Ok(())
    }

    pub(super) fn emit_param_space(&mut self, op: ParamOp<'a>) -> R<()> {
        match op {
            ParamOp::Addr { dst, name } => {
                let base = *self
                    .param_bases
                    .get(name)
                    .ok_or_else(|| Error::new(format!("unknown PTX parameter {name}")))?;
                let id = self.builder.constant_bit64(self.u64_type, u64::from(base));
                self.write(dst, Value { id, kind: Kind::B64 })
            }
            ParamOp::Load(load) => self.emit_param_load(load),
        }
    }

    fn param_word_pointer(&mut self, array: Word) -> R<Word> {
        let zero = self.u32_const(0);
        Ok(self.builder.access_chain(self.shared_word_ptr, None, array, [zero])?)
    }

    /// Word `index` (a u32 id) of the dynamic parameter array.
    fn param_array_word(&mut self, index: Word) -> R<Word> {
        let array = self
            .param_array
            .ok_or_else(|| Error::new("parameter space array was not set up"))?;
        let first = self.param_word_pointer(array)?;
        let index = self.builder.u_convert(self.u64_type, None, index)?;
        let slot = self.builder.ptr_access_chain(self.shared_word_ptr, None, first, index, [])?;
        Ok(self.builder.load(self.u32_type, None, slot, None, [])?)
    }

    /// Word at constant byte offset `off` of a named parameter.
    fn param_static_word(&mut self, name: &'a str, off: u32) -> R<Word> {
        let aligned = off & !3;
        if let Some(value) = self.params.get(&(name, aligned as usize)).copied()
            && value.kind == Kind::B32
        {
            return Ok(value.id);
        }
        let wide = match self.params.get(&(name, 0)).copied() {
            Some(Value { id, kind: Kind::B64 }) => id,
            Some(Value { id, kind: Kind::Address(_) }) => self.builder.bitcast(self.u64_type, None, id)?,
            _ => return Err(Error::new(format!("unknown PTX parameter word {name}+{off}"))),
        };
        match aligned {
            0 => Ok(self.builder.u_convert(self.u32_type, None, wide)?),
            4 => {
                let sh = self.builder.constant_bit64(self.u64_type, 32);
                let top = self.builder.shift_right_logical(self.u64_type, None, wide, sh)?;
                Ok(self.builder.u_convert(self.u32_type, None, top)?)
            }
            _ => Err(Error::new(format!("unknown PTX parameter word {name}+{off}"))),
        }
    }

    fn emit_param_load(&mut self, load: PLoad<'a>) -> R<()> {
        let u32t = self.u32_type;
        let step = load.ty.bytes();
        // Byte offset as a u32 id (dynamic) or a constant (static).
        let base_off = if load.dynamic {
            let value = self.read(load.base)?;
            let low = match value.kind {
                Kind::B32 => value.id,
                Kind::B64 => self.builder.u_convert(u32t, None, value.id)?,
                other => return Err(Error::new(format!("expected an integer for {}, got {other:?}", load.base))),
            };
            let imm = self.u32_const(load.off as u32);
            self.builder.i_add(u32t, None, low, imm)?
        } else {
            self.u32_const(load.off as u32)
        };
        for k in 0..usize::from(load.count) {
            let delta = self.u32_const(step * k as u32);
            let off = self.builder.i_add(u32t, None, base_off, delta)?;
            let (word_lo, word_hi, shift) = if load.dynamic {
                let two = self.u32_const(2);
                let index = self.builder.shift_right_logical(u32t, None, off, two)?;
                let lo = self.param_array_word(index)?;
                let hi = if load.ty == PTy::U64 {
                    let one = self.u32_const(1);
                    let next = self.builder.i_add(u32t, None, index, one)?;
                    Some(self.param_array_word(next)?)
                } else {
                    None
                };
                let three = self.u32_const(3);
                let eight = self.u32_const(8);
                let low = self.builder.bitwise_and(u32t, None, off, three)?;
                (lo, hi, self.builder.i_mul(u32t, None, low, eight)?)
            } else {
                let fixed = load.off as u32 + step * k as u32;
                let lo = self.param_static_word(load.base, fixed)?;
                let hi = if load.ty == PTy::U64 {
                    Some(self.param_static_word(load.base, fixed + 4)?)
                } else {
                    None
                };
                (lo, hi, self.u32_const((fixed & 3) * 8))
            };
            let (id, kind) = self.extract_param(load.ty, word_lo, word_hi, shift)?;
            let dst = load.dst[k];
            let want_wide = self.declarations.get(dst) == Some(&super::super::Width::B64);
            let (id, kind) = match (kind, want_wide) {
                (Kind::B32, true) => {
                    let wide = if load.ty.signed() {
                        self.builder.s_convert(self.u64_type, None, id)?
                    } else {
                        self.builder.u_convert(self.u64_type, None, id)?
                    };
                    (wide, Kind::B64)
                }
                (Kind::B64, false) => (self.builder.u_convert(u32t, None, id)?, Kind::B32),
                other => (id, other.0),
            };
            self.write(dst, Value { id, kind })?;
        }
        Ok(())
    }

    fn extract_param(&mut self, ty: PTy, lo: Word, hi: Option<Word>, shift: Word) -> R<(Word, Kind)> {
        let u32t = self.u32_type;
        Ok(match ty {
            PTy::U32 | PTy::S32 => (lo, Kind::B32),
            PTy::U8 | PTy::U16 => {
                let moved = self.builder.shift_right_logical(u32t, None, lo, shift)?;
                let mask = self.u32_const(if ty == PTy::U8 { 0xff } else { 0xffff });
                (self.builder.bitwise_and(u32t, None, moved, mask)?, Kind::B32)
            }
            PTy::S8 | PTy::S16 => {
                let width = if ty == PTy::S8 { 8 } else { 16 };
                let top = self.u32_const(32 - width);
                let up = self.builder.i_sub(u32t, None, top, shift)?;
                let left = self.builder.shift_left_logical(u32t, None, lo, up)?;
                let amount = self.u32_const(32 - width);
                (self.builder.shift_right_arithmetic(u32t, None, left, amount)?, Kind::B32)
            }
            PTy::U64 => {
                let hi = hi.expect("64-bit load has a high word");
                let lo = self.builder.u_convert(self.u64_type, None, lo)?;
                let hi = self.builder.u_convert(self.u64_type, None, hi)?;
                let sh = self.builder.constant_bit64(self.u64_type, 32);
                let hi = self.builder.shift_left_logical(self.u64_type, None, hi, sh)?;
                (self.builder.bitwise_or(self.u64_type, None, lo, hi)?, Kind::B64)
            }
        })
    }
}

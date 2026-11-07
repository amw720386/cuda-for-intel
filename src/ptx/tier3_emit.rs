//! SPIR-V for module-scope data (see `tier3_data.rs`): one `CrossWorkgroup`
//! variable per declaration, created on first use and shared by every entry.

use super::{Emitter, Error};
use super::super::GlobalData;
use rspirv::dr::Operand;
use rspirv::spirv::{Decoration, StorageClass, Word};

impl<'a> Emitter<'a> {
    pub(super) fn declare_globals(&mut self, globals: &[GlobalData<'a>]) -> Result<(), Error> {
        for data in globals {
            if self.globals.contains_key(data.name) {
                continue;
            }
            let words = data.bytes.len().div_ceil(4).max(1);
            let length = self.builder.constant_bit32(self.u32_type, words as u32);
            let array = self.builder.type_array(self.u32_type, length);
            let pointer = self.builder.type_pointer(None, StorageClass::CrossWorkgroup, array);
            let init = if data.bytes.iter().all(|b| *b == 0) {
                self.builder.constant_null(array)
            } else {
                let elements: Vec<Word> = (0..words)
                    .map(|i| {
                        let mut w = [0u8; 4];
                        for (k, b) in data.bytes[(4 * i).min(data.bytes.len())..(4 * i + 4).min(data.bytes.len())].iter().enumerate() {
                            w[k] = *b;
                        }
                        self.builder.constant_bit32(self.u32_type, u32::from_le_bytes(w))
                    })
                    .collect();
                self.builder.constant_composite(array, elements)
            };
            let variable = self.builder.variable(pointer, None, StorageClass::CrossWorkgroup, Some(init));
            self.builder.decorate(variable, Decoration::Alignment, [Operand::LiteralBit32(data.align.max(4))]);
            self.globals.insert(data.name, variable);
        }
        Ok(())
    }

    /// Byte pointer to a module-scope data symbol.
    pub(super) fn global_symbol(&mut self, name: &str) -> Option<Result<Word, Error>> {
        let variable = *self.globals.get(name)?;
        Some(self.builder.bitcast(self.global_byte_ptr, None, variable).map_err(Error::from))
    }
}

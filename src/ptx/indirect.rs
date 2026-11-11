//! Whether a kernel can run without declaring indirect USM access (`zeKernelSetIndirectAccess`).
//!
//! The Intel driver keeps an allocation resident for a dispatch when a kernel argument typed as a
//! global pointer points into it. Indirect access makes it keep every allocation of the declared
//! kinds resident instead; with host indirect access the runtime also treats the kernel as using
//! system memory, which feeds its per-walker L3 flush decision.
//!
//! A kernel is *direct* when every global or generic memory address it uses carries pointer
//! provenance only from parameters typed [`Param::GlobalPtr`] (or shared/local/module symbols).
//! Each register gets a provenance set, joined over all its definitions to a fixed point:
//! - `TRACKED`: derived from a pointer the driver tracks (a `GlobalPtr` parameter or a symbol);
//! - `UNKNOWN`: may hold an untracked pointer: a value loaded from memory (a pointer table, a
//!   module variable holding a pointer), a `.u64` parameter the translator kept as an integer, any
//!   word of a by-value aggregate, or the parameter space.
//!
//! Additive operations (add, sub, and, or, select, moves, conversions) carry the union of their
//! inputs. Scaling operations (mul, shifts, div, rem, the product part of mad) yield plain
//! integers: a loaded or `.u64` row index times a stride is an offset, not a pointer. An address is
//! direct when its set is exactly `TRACKED`; an address built from integers alone, or one that may
//! carry an `UNKNOWN` pointer, is not. The one assumption is that kernels do not scale a pointer and
//! then use the result as an offset into another allocation.
use super::ext::ExtOp;
use super::param_space::ParamOp;
use super::{Alu64Op, Block, Instruction, IntegerOp, Param, Param as ParamKind, Space};
use std::collections::HashMap;

const TRACKED: u8 = 1;
const UNKNOWN: u8 = 2;

fn root(operand: &str) -> &str {
    operand.split('+').next().unwrap_or(operand).trim()
}

/// `(space, address)` of each global/generic memory access by `inst`.
fn global_address<'a>(inst: &Instruction<'a>) -> Option<&'a str> {
    use Instruction::*;
    let (space, address) = match inst {
        Load32(space, _, addr)
        | Load64(space, _, addr)
        | Load16(space, _, addr)
        | Load8(space, _, addr)
        | Store32(space, addr, _)
        | Store8(space, addr, _)
        | Store16(space, addr, _)
        | Store64(space, addr, _)
        | StoreVecU16(space, addr, _)
        | StoreVec4U16(space, addr, _) => (*space, *addr),
        LoadSigned { space, addr, .. }
        | LoadVec { space, addr, .. }
        | LoadVecU16 { space, addr, .. }
        | LoadVecU8 { space, addr, .. }
        | StoreVec { space, addr, .. }
        | Atomic { space, addr, .. } => (*space, *addr),
        StoreVec4U8(addr, _) | AtomicAdd32(_, addr, _) => (Space::Global, *addr),
        Ext { op: ExtOp::Atom { space, .. } | ExtOp::AtomIncDec { space, .. }, src, .. } => (*space, src[0]),
        _ => return None,
    };
    (space == Space::Global).then_some(address)
}

fn operand(provenance: &HashMap<&str, u8>, operand: &str) -> u8 {
    let base = root(operand);
    if base.starts_with('%') {
        provenance.get(base).copied().unwrap_or(0)
    } else if base.is_empty() || base.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
        0 // an immediate
    } else {
        TRACKED // a module, shared or local symbol
    }
}

fn transfer(inst: &Instruction<'_>, sources: &[&str], params: &HashMap<&str, Param>, provenance: &HashMap<&str, u8>) -> u8 {
    use Instruction::*;
    let param = |name: &str| match params.get(name) {
        Some(ParamKind::GlobalPtr) => TRACKED,
        Some(ParamKind::U8 | ParamKind::U16 | ParamKind::U32 | ParamKind::F32) => 0,
        Some(ParamKind::U64 | ParamKind::Words { .. } | ParamKind::Staged { .. }) | None => UNKNOWN,
    };
    let of = |source: &str| operand(provenance, source);
    match inst {
        // A loaded value (also an atomic's old value or a matrix fragment).
        _ if inst.is_memory_load() => UNKNOWN,
        LdMatrix { .. } | ParamVec { .. } | ParamSpace(ParamOp::Load(_) | ParamOp::Addr { .. }) => UNKNOWN,
        Param(_, _, name, _) | ParamS32(_, name, _) | ParamS8(_, name) => param(name),
        SharedAddress(..) | SharedSymbol32(..) | LocalSymbol64(..) | GlobalAddress(..) => TRACKED,
        Builtin(..) | Special(..) | LaneId(_) => 0,
        Mul64(..)
        | ShiftLeft64(..)
        | ShiftRightLogical64(..)
        | ShiftRightArithmetic64(..)
        | Div64 { .. }
        | Rem64 { .. }
        | MulWide32 { .. } => 0,
        Integer32(
            IntegerOp::Mul
            | IntegerOp::ShiftLeft
            | IntegerOp::ShiftRightLogical
            | IntegerOp::ShiftRightArithmetic
            | IntegerOp::Div
            | IntegerOp::DivUnsigned
            | IntegerOp::Rem
            | IntegerOp::RemUnsigned
            | IntegerOp::MulHighUnsigned
            | IntegerOp::MulHighSigned,
            ..,
        ) => 0,
        Mad32(_, [_, _, addend]) | Alu64(Alu64Op::MadLo, _, [_, _, addend]) => of(addend),
        Ext { op: ExtOp::MadWide { .. }, src, .. } => of(src[2]),
        _ => sources.iter().fold(0, |set, source| set | of(source)),
    }
}

/// True when the kernel provably needs no indirect access (see the module documentation).
pub(super) fn direct_access(params: &[(&str, Param)], blocks: &[Block<'_>]) -> bool {
    let params: HashMap<&str, Param> = params.iter().copied().collect();
    let mut provenance: HashMap<&str, u8> = HashMap::new();
    let mut defined = Vec::new();
    loop {
        let mut changed = false;
        for inst in blocks.iter().flat_map(|block| &block.instructions) {
            let Some(sources) = inst.collect_definitions(&mut defined) else {
                continue;
            };
            let value = transfer(inst, sources, &params, &provenance);
            for &dst in &defined {
                let slot = provenance.entry(dst).or_insert(0);
                if *slot | value != *slot {
                    *slot |= value;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(global_address)
        .all(|address| operand(&provenance, address) == TRACKED)
}

#[cfg(test)]
mod tests {
    use super::super::translate;

    const HEADER: &str = ".version 8.0\n.target sm_80\n.address_size 64\n";

    fn direct(body: &str) -> bool {
        let translated = translate(&format!("{HEADER}{body}")).unwrap();
        assert_eq!(translated.kernels.len(), 1);
        translated.kernels[0].direct_access
    }

    #[test]
    fn pointer_parameters_with_thread_offsets_are_direct() {
        assert!(direct(
            ".visible .entry add(.param .u64 a, .param .u64 b, .param .u32 n)\n{\n\
             .reg .pred %p<2>;\n.reg .b32 %r<6>;\n.reg .f32 %f<4>;\n.reg .b64 %rd<8>;\n\
             ld.param.u64 %rd1, [a];\nld.param.u64 %rd2, [b];\nld.param.u32 %r1, [n];\n\
             cvta.to.global.u64 %rd3, %rd1;\ncvta.to.global.u64 %rd4, %rd2;\n\
             mov.u32 %r2, %ctaid.x;\nmov.u32 %r3, %ntid.x;\nmov.u32 %r4, %tid.x;\nmad.lo.s32 %r5, %r2, %r3, %r4;\n\
             setp.ge.u32 %p1, %r5, %r1;\n@%p1 bra DONE;\n\
             mul.wide.u32 %rd5, %r5, 4;\nadd.s64 %rd6, %rd3, %rd5;\nadd.s64 %rd7, %rd4, %rd5;\n\
             ld.global.f32 %f1, [%rd6];\nld.global.f32 %f2, [%rd7+4];\nadd.f32 %f3, %f1, %f2;\nst.global.f32 [%rd6], %f3;\n\
             DONE:\nret;\n}\n"
        ));
    }

    #[test]
    fn shared_memory_staging_stays_direct() {
        assert!(direct(
            ".visible .entry stage(.param .u64 a)\n{\n\
             .shared .align 4 .b8 buf[128];\n.reg .b32 %r<4>;\n.reg .f32 %f<2>;\n.reg .b64 %rd<6>;\n\
             ld.param.u64 %rd1, [a];\ncvta.to.global.u64 %rd2, %rd1;\nmov.u32 %r1, %tid.x;\n\
             mul.wide.u32 %rd3, %r1, 4;\nadd.s64 %rd4, %rd2, %rd3;\nld.global.f32 %f1, [%rd4];\n\
             shl.b32 %r2, %r1, 2;\nmov.u32 %r3, buf;\nadd.s32 %r3, %r3, %r2;\nst.shared.f32 [%r3], %f1;\n\
             bar.sync 0;\nld.shared.f32 %f1, [%r3];\nst.global.f32 [%rd4], %f1;\nret;\n}\n"
        ));
    }

    #[test]
    fn a_pointer_loaded_from_a_table_is_indirect() {
        assert!(!direct(
            ".visible .entry table(.param .u64 ptrs)\n{\n\
             .reg .b32 %r<2>;\n.reg .b64 %rd<6>;\n\
             ld.param.u64 %rd1, [ptrs];\ncvta.to.global.u64 %rd2, %rd1;\n\
             ld.global.u64 %rd3, [%rd2];\ncvta.to.global.u64 %rd4, %rd3;\n\
             mov.u32 %r1, 7;\nst.global.u32 [%rd4], %r1;\nret;\n}\n"
        ));
    }

    #[test]
    fn a_loaded_index_scaled_into_an_offset_is_direct() {
        assert!(direct(
            ".visible .entry rows(.param .u64 ids, .param .u64 src)\n{\n\
             .reg .b32 %r<3>;\n.reg .f32 %f<2>;\n.reg .b64 %rd<8>;\n\
             ld.param.u64 %rd1, [ids];\nld.param.u64 %rd2, [src];\n\
             cvta.to.global.u64 %rd3, %rd1;\ncvta.to.global.u64 %rd4, %rd2;\n\
             ld.global.u32 %r1, [%rd3];\nmul.wide.u32 %rd5, %r1, 4;\nadd.s64 %rd6, %rd4, %rd5;\n\
             ld.global.f32 %f1, [%rd6];\nst.global.f32 [%rd4], %f1;\nret;\n}\n"
        ));
    }

    #[test]
    fn a_u64_parameter_used_as_a_generic_address_is_indirect() {
        assert!(!direct(
            ".visible .entry generic(.param .u64 a)\n{\n\
             .reg .b32 %r<2>;\n.reg .b64 %rd<3>;\n\
             ld.param.u64 %rd1, [a];\nadd.s64 %rd2, %rd1, 8;\nmov.u32 %r1, 1;\nst.u32 [%rd2], %r1;\nret;\n}\n"
        ));
    }

    /// How many corpus entries qualify:
    /// `cargo test --lib direct_access_corpus_report -- --ignored --nocapture`
    /// (`CFI_CORPUS_DIR` overrides the default `tests/ptx_corpus`).
    #[test]
    #[ignore = "walks the PTX corpus"]
    fn direct_access_corpus_report() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else { return };
            let mut paths: Vec<_> = entries.filter_map(Result::ok).map(|entry| entry.path()).collect();
            paths.sort();
            for path in paths {
                if path.file_name().is_some_and(|name| name == "harness" || name == "target") {
                    continue;
                }
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "ptx") {
                    out.push(path);
                }
            }
        }
        let root = std::env::var_os("CFI_CORPUS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ptx_corpus"));
        let mut files = Vec::new();
        walk(&root, &mut files);
        let mut by_group: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        for file in &files {
            let Ok(source) = std::fs::read_to_string(file) else { continue };
            let Ok(translated) = super::super::translate_lenient(&source) else { continue };
            let group = file
                .strip_prefix(&root)
                .ok()
                .and_then(|path| path.components().next())
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .unwrap_or_default();
            let slot = by_group.entry(group).or_default();
            for kernel in &translated.kernels {
                slot.0 += 1;
                slot.1 += usize::from(kernel.direct_access);
                if std::env::var_os("CFI_DIRECT_LIST").is_some() {
                    println!("{} {} {}", if kernel.direct_access { "direct  " } else { "indirect" }, file.display(), kernel.name);
                }
            }
        }
        let (mut total, mut direct) = (0, 0);
        for (group, (kernels, qualified)) in &by_group {
            println!("{group:12} {qualified:5} / {kernels:5} entries need no indirect access");
            total += kernels;
            direct += qualified;
        }
        println!("{:12} {direct:5} / {total:5}", "total");
        assert!(total > 0, "no corpus entries under {}", root.display());
    }

    #[test]
    fn a_u64_stride_parameter_scaled_into_an_offset_is_direct() {
        assert!(direct(
            ".visible .entry rows(.param .u64 x, .param .u64 stride)\n{\n\
             .reg .b32 %r<2>;\n.reg .f32 %f<2>;\n.reg .b64 %rd<7>;\n\
             ld.param.u64 %rd1, [x];\nld.param.u64 %rd2, [stride];\ncvta.to.global.u64 %rd3, %rd1;\n\
             mov.u32 %r1, %ctaid.x;\ncvt.u64.u32 %rd4, %r1;\nmul.lo.s64 %rd5, %rd4, %rd2;\nshl.b64 %rd5, %rd5, 2;\n\
             add.s64 %rd6, %rd3, %rd5;\nld.global.f32 %f1, [%rd6];\nst.global.f32 [%rd6], %f1;\nret;\n}\n"
        ));
    }

    #[test]
    fn an_unscaled_u64_offset_or_an_integer_address_is_indirect() {
        // base + raw .u64 parameter: the parameter may be another pointer.
        assert!(!direct(
            ".visible .entry offset(.param .u64 x, .param .u64 off)\n{\n\
             .reg .b32 %r<2>;\n.reg .b64 %rd<5>;\n\
             ld.param.u64 %rd1, [x];\nld.param.u64 %rd2, [off];\ncvta.to.global.u64 %rd3, %rd1;\n\
             add.s64 %rd4, %rd3, %rd2;\nmov.u32 %r1, 1;\nst.global.u32 [%rd4], %r1;\nret;\n}\n"
        ));
        // An address synthesised from 32-bit halves carries no tracked pointer.
        assert!(!direct(
            ".visible .entry halves(.param .u32 lo, .param .u32 hi)\n{\n\
             .reg .b32 %r<4>;\n.reg .b64 %rd<5>;\n\
             ld.param.u32 %r1, [lo];\nld.param.u32 %r2, [hi];\ncvt.u64.u32 %rd1, %r1;\ncvt.u64.u32 %rd2, %r2;\n\
             shl.b64 %rd3, %rd2, 32;\nor.b64 %rd4, %rd3, %rd1;\nmov.u32 %r3, 1;\nst.global.u32 [%rd4], %r3;\nret;\n}\n"
        ));
    }

    #[test]
    fn data_loaded_and_stored_through_parameters_is_direct() {
        // Loaded values feed only arithmetic and stored data, never an address.
        assert!(direct(
            ".visible .entry scale(.param .u64 a, .param .u64 b)\n{\n\
             .reg .b32 %r<4>;\n.reg .b64 %rd<6>;\n\
             ld.param.u64 %rd1, [a];\nld.param.u64 %rd2, [b];\n\
             cvta.to.global.u64 %rd3, %rd1;\ncvta.to.global.u64 %rd4, %rd2;\n\
             ld.global.u32 %r1, [%rd3];\nmul.lo.s32 %r2, %r1, 3;\nst.global.u32 [%rd4], %r2;\n\
             atom.global.add.u32 %r3, [%rd4+4], %r2;\nret;\n}\n"
        ));
    }
}

//! Spelling and lowering rewrites for PTX produced by newer nvcc (CUDA 12.8, sm_120)
//! that the instruction-level translator does not take directly. All of them are
//! expressed as other PTX instructions, in the normalisation pass.

use super::{render, rewrite_inst, Ctx, Inst};

/// Lower an instruction produced by a rewrite through the whole rewrite pipeline again.
fn again(ctx: &mut Ctx, guard: &str, op: &str, operands: Vec<String>) -> Vec<String> {
    let inst = Inst { guard: guard.to_owned(), op: op.to_owned(), operands };
    rewrite_inst(ctx, &inst).unwrap_or_else(|| vec![inst.render()])
}
use std::sync::atomic::{AtomicU32, Ordering};

static LABELS: AtomicU32 = AtomicU32::new(0);

pub(super) fn rewrite(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let op = inst.op.as_str();
    let parts: Vec<&str> = op.split('.').collect();
    // mov.s32 / mov.s64 / mov.s16 are plain bit moves.
    if parts.len() == 2 && parts[0] == "mov" && matches!(parts[1], "s32" | "s64" | "s16") {
        let ty = format!("b{}", &parts[1][1..]);
        return Some(again(ctx, &inst.guard, &format!("mov.{ty}"), inst.operands.clone()));
    }
    // `X.approx.f32.ftz` -> `X.approx.ftz.f32`.
    if parts.len() >= 3 && parts[parts.len() - 1] == "ftz" && parts[parts.len() - 2] == "f32" && parts[0] != "cvt" {
        let mut p = parts.clone();
        p.swap(parts.len() - 1, parts.len() - 2);
        return Some(again(ctx, &inst.guard, &p.join("."), inst.operands.clone()));
    }
    // vote.ballot.sync.b32 d, p, mask -> vote.sync.ballot.b32
    if op == "vote.ballot.sync.b32" {
        return Some(again(ctx, &inst.guard, "vote.sync.ballot.b32", inst.operands.clone()));
    }
    // Non-coherent signed loads: plain loads.
    if op == "ld.global.nc.s32" {
        return Some(again(ctx, &inst.guard, "ld.global.s32", inst.operands.clone()));
    }
    // ldmatrix.sync.aligned.x4.m8n8[.trans].shared.b16 -> ...m8n8.x4...
    if parts.len() >= 5 && parts[0] == "ldmatrix" && parts[3].starts_with('x') && parts[4] == "m8n8" {
        let mut p = parts.clone();
        p.swap(3, 4);
        return Some(again(ctx, &inst.guard, &p.join("."), inst.operands.clone()));
    }
    if parts.len() >= 6 && parts[0] == "ldmatrix" && parts[3].starts_with('x') && parts[4] == "trans" && parts[5] == "m8n8" {
        let mut p = parts.clone();
        p.swap(3, 5);
        p.swap(4, 5);
        return Some(again(ctx, &inst.guard, &p.join("."), inst.operands.clone()));
    }
    // `mov.u32 %r, shared_symbol+N`: symbol address, then the displacement.
    if matches!(op, "mov.u32" | "mov.b32" | "mov.u64" | "mov.b64") && inst.operands.len() == 2 {
        let src = inst.operand(1).replace(' ', "");
        if let Some((name, off)) = src.split_once('+') {
            if let Ok(off) = off.parse::<i64>() {
                if ctx.t3.shared.iter().any(|n| *n == name) {
                    let dst = inst.operand(0).to_owned();
                    let mut l = again(ctx, &inst.guard, op, vec![dst.clone(), name.to_owned()]);
                    let add = if op.ends_with("64") { "add.s64" } else { "add.u32" };
                    l.push(render(&inst.guard, add, &[dst.clone(), dst, off.to_string()]));
                    return Some(l);
                }
            }
        }
    }
    // `[%reg-N]` / `[%reg+-N]`: fold the negative displacement into an address register.
    if matches!(parts[0], "ld" | "st" | "atom" | "red") {
        for (i, operand) in inst.operands.iter().enumerate() {
            let Some((base, off)) = address_parts(operand) else { continue };
            if off >= 0 || !base.starts_with('%') || !operand.contains('-') {
                continue;
            }
            let wide = ctx.width_of(&base) == Some(64);
            let mut operands = inst.operands.clone();
            let line = if wide {
                ctx.need64(1);
                operands[i] = "[ %__t64_0 ]".into();
                render(&inst.guard, "add.s64", &["%__t64_0".to_owned(), base, off.to_string()])
            } else {
                ctx.tc = ctx.tc.max(2);
                operands[i] = "[ %__tc_1 ]".into();
                render(&inst.guard, "add.s32", &["%__tc_1".to_owned(), base, off.to_string()])
            };
            let mut out = vec![line];
            out.extend(again(ctx, &inst.guard, op, operands));
            return Some(out);
        }
    }
    // movmatrix.sync.aligned.m8n8.trans.b16 d, s: transpose of an 8x8 b16 matrix whose rows are
    // spread two elements per lane (lane l holds row l/4, columns 2*(l%4), 2*(l%4)+1).
    // Lane l receives T[l/4][2(l%4)+{0,1}] = M[2(l%4)+{0,1}][l/4], which sits in lane
    // 8*(l%4) + (l>>3) (+4 for the odd row), half (l>>2)&1.
    if op == "movmatrix.sync.aligned.m8n8.trans.b16" && inst.operands.len() == 2 {
        ctx.tc = ctx.tc.max(8);
        let (d, src) = (inst.operand(0).trim().to_owned(), inst.operand(1).trim().to_owned());
        let g = |op: &str, o: &[&str]| render(&inst.guard, op, &o.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>());
        let mut out = vec![
            g("mov.u32", &["%__tc_0", "%laneid"]),
            g("and.b32", &["%__tc_1", "%__tc_0", "3"]),
            g("shl.b32", &["%__tc_1", "%__tc_1", "3"]),
            g("shr.u32", &["%__tc_2", "%__tc_0", "3"]),
            g("add.u32", &["%__tc_1", "%__tc_1", "%__tc_2"]),
            g("add.u32", &["%__tc_2", "%__tc_1", "4"]),
            g("shr.u32", &["%__tc_3", "%__tc_0", "2"]),
            g("and.b32", &["%__tc_3", "%__tc_3", "1"]),
            g("shl.b32", &["%__tc_3", "%__tc_3", "4"]),
        ];
        out.extend(again(ctx, &inst.guard, "shfl.sync.idx.b32", vec!["%__tc_4".into(), src.clone(), "%__tc_1".into(), "31".into(), "0xffffffff".into()]));
        out.extend(again(ctx, &inst.guard, "shfl.sync.idx.b32", vec!["%__tc_5".into(), src, "%__tc_2".into(), "31".into(), "0xffffffff".into()]));
        out.extend([
            g("shr.u32", &["%__tc_4", "%__tc_4", "%__tc_3"]),
            g("and.b32", &["%__tc_4", "%__tc_4", "65535"]),
            g("shr.u32", &["%__tc_5", "%__tc_5", "%__tc_3"]),
            g("shl.b32", &["%__tc_5", "%__tc_5", "16"]),
            g("or.b32", &[&d, "%__tc_4", "%__tc_5"]),
        ]);
        return Some(out);
    }
    // cvt to an 8-bit type keeps the low byte.
    if parts.len() == 3 && parts[0] == "cvt" && matches!(parts[1], "s8" | "u8") && matches!(parts[2], "s32" | "u32") && inst.operands.len() == 2 {
        ctx.tc = ctx.tc.max(1);
        return Some(vec![
            render(&inst.guard, "and.b32", &["%__tc_0".to_owned(), inst.operand(1).to_owned(), "255".to_owned()]),
            render(&inst.guard, "cvt.u16.u32", &[inst.operand(0).to_owned(), "%__tc_0".to_owned()]),
        ]);
    }
    // Half-precision atomic add as a compare-and-swap loop on the containing dword.
    if matches!(op, "atom.global.add.noftz.f16" | "atom.global.add.noftz.bf16" | "atom.add.noftz.f16" | "atom.add.noftz.bf16")
        && inst.operands.len() == 3
    {
        return Some(half_atomic_add(ctx, inst, op.ends_with("bf16")));
    }
    None
}

/// `[%rd5+16]` -> ("%rd5", 16)
fn address_parts(operand: &str) -> Option<(String, i64)> {
    let inner = operand.trim().strip_prefix('[')?.strip_suffix(']')?.replace(' ', "");
    match inner.find(['+', '-']) {
        Some(at) => Some((inner[..at].to_owned(), inner[at..].replace('+', "").parse().ok()?)),
        None => Some((inner, 0)),
    }
}

fn invert_guard(guard: &str) -> String {
    let t: Vec<&str> = guard.split_whitespace().collect();
    match t.as_slice() {
        ["@", "!", p] => format!("@ {p}"),
        ["@", p] => format!("@ ! {p}"),
        _ => String::new(),
    }
}

fn half_atomic_add(ctx: &mut Ctx, inst: &Inst, bf16: bool) -> Vec<String> {
    ctx.tc = ctx.tc.max(6);
    ctx.need64(3);
    ctx.needp(5);
    ctx.extra_decls.insert(".reg .f32 %__xf_<2> ;".into());
    ctx.extra_decls.insert(".reg .b16 %__xh_<2> ;".into());
    let n = LABELS.fetch_add(1, Ordering::Relaxed);
    let (lp, ld) = (format!("$cfi_cas{n}"), format!("$cfi_casend{n}"));
    let dst = inst.operand(0).trim().to_owned();
    let (base, off) = address_parts(inst.operand(1)).unwrap_or_else(|| (inst.operand(1).trim().to_owned(), 0));
    let val = inst.operand(2).trim().to_owned();
    let g = |op: &str, o: &[&str]| render("", op, &o.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>());
    let mut l: Vec<String> = Vec::new();
    if !inst.guard.is_empty() {
        l.push(render(&invert_guard(&inst.guard), "bra", &[ld.clone()]));
    }
    l.push(g("add.s64", &["%__t64_0", &base, &off.to_string()]));
    l.push(g("and.b64", &["%__t64_1", "%__t64_0", "-4"]));
    l.push(g("and.b64", &["%__t64_0", "%__t64_0", "2"]));
    l.push(g("cvt.u32.u64", &["%__tc_0", "%__t64_0"]));
    l.push(g("shl.b32", &["%__tc_0", "%__tc_0", "3"]));
    l.push(g("cvta.to.global.u64", &["%__t64_2", "%__t64_1"]));
    l.push(g("ld.global.u32", &["%__tc_1", "[ %__t64_2 ]"]));
    l.push(format!("{lp} :"));
    l.push(g("shr.u32", &["%__tc_4", "%__tc_1", "%__tc_0"]));
    l.push(g("and.b32", &["%__tc_4", "%__tc_4", "65535"]));
    l.push(g("cvt.u16.u32", &["%__xh_0", "%__tc_4"]));
    if bf16 {
        l.push(g("shl.b32", &["%__tc_5", "%__tc_4", "16"]));
        l.push(g("mov.b32", &["%__xf_0", "%__tc_5"]));
        l.push(g("cvt.u32.u16", &["%__tc_5", &val]));
        l.push(g("shl.b32", &["%__tc_5", "%__tc_5", "16"]));
        l.push(g("mov.b32", &["%__xf_1", "%__tc_5"]));
    } else {
        l.push(g("cvt.f32.f16", &["%__xf_0", "%__xh_0"]));
        l.push(g("cvt.f32.f16", &["%__xf_1", &val]));
    }
    l.push(g("add.f32", &["%__xf_0", "%__xf_0", "%__xf_1"]));
    l.push(g(if bf16 { "cvt.rn.bf16.f32" } else { "cvt.rn.f16.f32" }, &["%__xh_1", "%__xf_0"]));
    l.push(g("cvt.u32.u16", &["%__tc_2", "%__xh_1"]));
    l.push(g("shl.b32", &["%__tc_2", "%__tc_2", "%__tc_0"]));
    l.push(g("mov.b32", &["%__tc_5", "65535"]));
    l.push(g("shl.b32", &["%__tc_5", "%__tc_5", "%__tc_0"]));
    l.push(g("not.b32", &["%__tc_5", "%__tc_5"]));
    l.push(g("and.b32", &["%__tc_3", "%__tc_1", "%__tc_5"]));
    l.push(g("or.b32", &["%__tc_2", "%__tc_2", "%__tc_3"]));
    l.push(g("atom.global.cas.b32", &["%__tc_3", "[ %__t64_2 ]", "%__tc_1", "%__tc_2"]));
    l.push(g("setp.ne.u32", &["%__tp_4", "%__tc_3", "%__tc_1"]));
    l.push(g("mov.b32", &["%__tc_1", "%__tc_3"]));
    l.push(render("@ %__tp_4", "bra", &[lp]));
    if dst != "_" && !dst.is_empty() {
        l.push(g("mov.b32", &[&dst, "%__xh_0"]));
    }
    if !inst.guard.is_empty() {
        l.push(format!("{ld} :"));
    }
    l
}

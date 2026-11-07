//! `ld/st.{global,shared,local}.v{2,4}.{u8,s8,b8,u16,s16,b16}`: split into scalar accesses at
//! consecutive offsets (the sub-word scalar forms are already lowered). Child of `normalize`.

use super::{Ctx, Inst, render, rewrite_inst};

pub(super) fn rewrite(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let store = match parts.first().copied()? {
        "ld" => false,
        "st" => true,
        _ => return None,
    };
    let at = parts.iter().position(|p| matches!(*p, "v2" | "v4"))?;
    let ty = *parts.get(at + 1)?;
    let size: i64 = match ty {
        "u8" | "s8" | "b8" => 1,
        "u16" | "s16" | "b16" => 2,
        _ => return None,
    };
    if parts.len() != at + 2 {
        return None;
    }
    let count = if parts[at] == "v2" { 2 } else { 4 };
    let (vec_index, addr_index) = if store { (1, 0) } else { (0, 1) };
    let elems: Vec<&str> = inst
        .operand(vec_index)
        .trim()
        .strip_prefix('{')?
        .strip_suffix('}')?
        .split(',')
        .map(str::trim)
        .collect();
    if elems.len() != count {
        return None;
    }
    let inner = inst.operand(addr_index).trim().strip_prefix('[')?.strip_suffix(']')?.trim();
    let (base, off) = match inner.split_once('+') {
        Some((b, o)) => (b.trim(), o.trim().parse::<i64>().ok()?),
        None => (inner, 0),
    };
    let scalar = parts.iter().enumerate().filter(|&(i, _)| i != at).map(|(_, p)| *p).collect::<Vec<_>>().join(".");
    let mut out = Vec::new();
    for (i, elem) in elems.iter().enumerate() {
        let address = format!("[ {base}+{} ]", off + size * i as i64);
        let operands = if store { vec![address, (*elem).to_owned()] } else { vec![(*elem).to_owned(), address] };
        let line = Inst { guard: inst.guard.clone(), op: scalar.clone(), operands };
        out.extend(rewrite_inst(ctx, &line).unwrap_or_else(|| vec![render(&line.guard, &line.op, &line.operands)]));
    }
    Some(out)
}

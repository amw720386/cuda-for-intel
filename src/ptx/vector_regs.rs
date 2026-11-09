//! Vector registers and wide vector moves, lowered to scalar registers before inlining.
//!
//! * `.reg .v2/.v4 .T v` declares one scalar register per lane (`v__vx`, `v__vy`, ...).
//!   Ranges (`.reg .v2 .b16 h<4>`) become one scalar range per lane (`h0.y` is `h__vy0`).
//!   Lane operands (`v.x`, `v.g`) name the scalar; a whole-vector operand becomes the brace
//!   list `{ v__vx , v__vy }`, which the vector `ld`/`st`, `mov.b64`/`mov.b32` pack/unpack and
//!   `call` argument forms already accept. `.reg` vector parameters of `.func` headers are
//!   split the same way, so the inliner sees scalar parameters.
//! * `mov.vN.T d, s` becomes scalar moves with PTX's parallel semantics (sources are read
//!   before any destination is written, through scratch registers when they overlap).
//! * `ld/st.v8.T` becomes two `.v4` accesses at consecutive offsets.
//! * Fused declaration spellings (`.reg.v4.u16`, `.param.u64`) are split into the canonical
//!   token sequence first.
//!
//! Child of `normalize`; it runs on the statement list before `inline::expand`.

use super::Stmt;
use std::collections::HashMap;

/// Lane count and element type (`.u32`) of a vector register, or of every member of a range.
#[derive(Clone)]
struct VecReg {
    lanes: usize,
    ty: String,
    /// `Some(prefix)` when the register is a member of a declared range (`prefix<N>`).
    range_prefix: Option<String>,
}

const LANES: [&str; 4] = ["x", "y", "z", "w"];

fn lane_index(c: &str) -> Option<usize> {
    match c {
        "x" | "r" => Some(0),
        "y" | "g" => Some(1),
        "z" | "b" => Some(2),
        "w" | "a" => Some(3),
        _ => None,
    }
}

fn vector_lanes(token: &str) -> Option<usize> {
    match token {
        ".v2" => Some(2),
        ".v4" => Some(4),
        _ => None,
    }
}

pub(super) fn run(statements: Vec<Stmt>) -> Vec<Stmt> {
    let interesting = |t: &String| {
        t.starts_with(".reg.") || t.starts_with(".param.") || t.starts_with(".v2.") || t.starts_with(".v4.")
            || t.starts_with(".v8.") || t == ".v2" || t == ".v4" || t.contains(".v8.") || t.starts_with("mov.v")
    };
    if !statements.iter().any(|s| s.iter().any(interesting)) {
        return statements;
    }
    let statements: Vec<Stmt> = statements.into_iter().map(split_fused).collect();
    let mut out = Vec::with_capacity(statements.len());
    let mut counter = 0usize;
    let mut i = 0;
    while i < statements.len() {
        let stmt = &statements[i];
        let header = stmt.last().is_some_and(|t| t == "{")
            && stmt.iter().any(|t| t == ".func" || t == ".entry");
        if !header {
            // Module scope and declarations without a body.
            if stmt.iter().any(|t| t == ".func") && stmt.last().is_some_and(|t| t == ";") {
                let regs = collect(std::slice::from_ref(stmt));
                out.push(rewrite_header(stmt, &regs));
            } else {
                out.extend(rewrite_stmt(stmt, &HashMap::new(), &mut counter));
            }
            i += 1;
            continue;
        }
        // Function or entry: header plus body through the matching brace.
        let mut depth = 1i32;
        let mut end = i + 1;
        while end < statements.len() && depth > 0 {
            match statements[end].first().map(String::as_str) {
                Some("{") if statements[end].len() == 1 => depth += 1,
                Some("}") if statements[end].len() == 1 => depth -= 1,
                _ => {}
            }
            end += 1;
        }
        let scope = &statements[i..end];
        let regs = collect(scope);
        out.push(rewrite_header(stmt, &regs));
        for inner in &scope[1..] {
            out.extend(rewrite_stmt(inner, &regs, &mut counter));
        }
        i = end;
    }
    out
}

/// `.reg.v4.u16` -> `.reg .v4 .u16`; `.param.u64` -> `.param .u64`.
fn split_fused(stmt: Stmt) -> Stmt {
    if !stmt.iter().any(|t| {
        (t.starts_with(".reg.") || t.starts_with(".param.") || t.starts_with(".v2.") || t.starts_with(".v4."))
            && t.len() > 4
    }) {
        return stmt;
    }
    let mut out = Vec::with_capacity(stmt.len() + 2);
    for token in stmt {
        let fused = token.starts_with(".reg.")
            || token.starts_with(".param.")
            || token.starts_with(".v2.")
            || token.starts_with(".v4.");
        if fused {
            out.extend(token[1..].split('.').filter(|p| !p.is_empty()).map(|p| format!(".{p}")));
        } else {
            out.push(token);
        }
    }
    out
}

/// Vector registers declared anywhere in `stmts` (a header and its body), by name.
fn collect(stmts: &[Stmt]) -> HashMap<String, VecReg> {
    let mut regs = HashMap::new();
    for stmt in stmts {
        let mut i = 0;
        while i < stmt.len() {
            if stmt[i] != ".reg" {
                i += 1;
                continue;
            }
            let Some(lanes) = stmt.get(i + 1).and_then(|t| vector_lanes(t)) else {
                i += 1;
                continue;
            };
            let Some(ty) = stmt.get(i + 2).filter(|t| t.starts_with('.')) else {
                i += 1;
                continue;
            };
            let mut j = i + 3;
            while j < stmt.len() {
                let name = &stmt[j];
                if matches!(name.as_str(), ";" | ")" | "{") || name.starts_with('.') {
                    break;
                }
                if name == "," {
                    j += 1;
                    continue;
                }
                if stmt.get(j + 1).is_some_and(|t| t == "<") {
                    let count: usize = stmt.get(j + 2).and_then(|t| t.parse().ok()).unwrap_or(0);
                    for k in 0..count.min(4096) {
                        regs.insert(
                            format!("{name}{k}"),
                            VecReg { lanes, ty: ty.clone(), range_prefix: Some(name.clone()) },
                        );
                    }
                    // The prefix itself is not a register.
                    j += 4;
                    continue;
                }
                regs.insert(name.clone(), VecReg { lanes, ty: ty.clone(), range_prefix: None });
                j += 1;
            }
            i = j;
        }
    }
    regs
}

/// Scalar register holding lane `lane` of `name`.
fn scalar(name: &str, reg: &VecReg, lane: usize) -> String {
    match &reg.range_prefix {
        Some(prefix) => format!("{prefix}__v{}{}", LANES[lane], &name[prefix.len()..]),
        None => format!("{name}__v{}", LANES[lane]),
    }
}

fn lanes_of(name: &str, reg: &VecReg) -> Vec<String> {
    (0..reg.lanes).map(|lane| scalar(name, reg, lane)).collect()
}

/// `.reg .vN .T v` parameters of a `.func` header become N scalar `.reg .T` parameters.
fn rewrite_header(stmt: &Stmt, regs: &HashMap<String, VecReg>) -> Stmt {
    if !stmt.iter().any(|t| t == ".v2" || t == ".v4") {
        return stmt.clone();
    }
    let mut out = Vec::with_capacity(stmt.len());
    let mut i = 0;
    while i < stmt.len() {
        if stmt[i] == ".reg" && stmt.get(i + 1).and_then(|t| vector_lanes(t)).is_some() {
            if let (Some(ty), Some(name)) = (stmt.get(i + 2), stmt.get(i + 3)) {
                if let Some(reg) = regs.get(name) {
                    for (k, lane) in lanes_of(name, reg).into_iter().enumerate() {
                        if k > 0 {
                            out.push(",".into());
                        }
                        out.extend([".reg".into(), ty.clone(), lane]);
                    }
                    i += 4;
                    continue;
                }
            }
        }
        out.push(stmt[i].clone());
        i += 1;
    }
    out
}

fn rewrite_stmt(stmt: &Stmt, regs: &HashMap<String, VecReg>, counter: &mut usize) -> Vec<Stmt> {
    if stmt.first().is_some_and(|t| t == ".reg") {
        return rewrite_decl(stmt, regs);
    }
    let (guard_len, op) = match stmt.first().map(String::as_str) {
        Some("@") => {
            let n = if stmt.get(1).is_some_and(|t| t == "!") { 3 } else { 2 };
            (n, stmt.get(n).map_or("", String::as_str))
        }
        _ => (0, stmt.first().map_or("", String::as_str)),
    };
    if op.starts_with('.') || op == "{" || op == "}" {
        return vec![stmt.clone()];
    }
    let in_call = op == "call" || op.starts_with("call.");
    let mut tokens: Stmt = Vec::with_capacity(stmt.len());
    let mut bracket = 0;
    for (index, token) in stmt.iter().enumerate() {
        match token.as_str() {
            "[" => bracket += 1,
            "]" => bracket -= 1,
            _ => {}
        }
        if index <= guard_len || bracket > 0 {
            tokens.push(token.clone());
            continue;
        }
        if let Some(reg) = regs.get(token.as_str()) {
            let lanes = lanes_of(token, reg);
            if !in_call {
                tokens.push("{".into());
            }
            for (k, lane) in lanes.into_iter().enumerate() {
                if k > 0 {
                    tokens.push(",".into());
                }
                tokens.push(lane);
            }
            if !in_call {
                tokens.push("}".into());
            }
            continue;
        }
        if let Some((name, lane)) = token.rsplit_once('.') {
            if let (Some(reg), Some(lane)) = (regs.get(name), lane_index(lane)) {
                if lane < reg.lanes {
                    tokens.push(scalar(name, reg, lane));
                    continue;
                }
            }
        }
        tokens.push(token.clone());
    }
    let guard = &tokens[..guard_len];
    let op = tokens[guard_len].clone();
    let body = &tokens[guard_len + 1..];
    let parts: Vec<&str> = op.split('.').collect();
    if parts.first() == Some(&"mov") && parts.get(1).is_some_and(|p| matches!(*p, "v2" | "v4" | "v8")) {
        if let Some(lines) = parallel_move(guard, &parts, body, counter) {
            return lines;
        }
    }
    if matches!(parts.first(), Some(&"ld" | &"st")) && parts.contains(&"v8") {
        if let Some(lines) = split_v8(guard, &parts, body) {
            return lines;
        }
    }
    vec![tokens]
}

/// `.reg .vN .T a , b<4> ;` -> `.reg .T a__vx ;` ... and `.reg .T b__vx < 4 > ;` ...
fn rewrite_decl(stmt: &Stmt, regs: &HashMap<String, VecReg>) -> Vec<Stmt> {
    let Some(lanes) = stmt.get(1).and_then(|t| vector_lanes(t)) else {
        return vec![stmt.clone()];
    };
    let Some(ty) = stmt.get(2) else {
        return vec![stmt.clone()];
    };
    let mut out = Vec::new();
    let mut j = 3;
    while j < stmt.len() {
        let name = &stmt[j];
        if matches!(name.as_str(), ";" | ",") {
            j += 1;
            continue;
        }
        if stmt.get(j + 1).is_some_and(|t| t == "<") {
            let count = stmt.get(j + 2).cloned().unwrap_or_default();
            for lane in &LANES[..lanes] {
                out.push(vec![
                    ".reg".into(),
                    ty.clone(),
                    format!("{name}__v{lane}"),
                    "<".into(),
                    count.clone(),
                    ">".into(),
                    ";".into(),
                ]);
            }
            j += 4;
            continue;
        }
        if let Some(reg) = regs.get(name) {
            for lane in lanes_of(name, reg) {
                out.push(vec![".reg".into(), ty.clone(), lane, ";".into()]);
            }
        }
        j += 1;
    }
    out
}

/// Splits `{ a , b }` (or a single operand) into its elements.
fn elements(tokens: &[String]) -> Vec<String> {
    let inner = match tokens {
        [open, inner @ .., close] if open == "{" && close == "}" => inner,
        other => other,
    };
    let mut out = Vec::new();
    let mut current = String::new();
    for token in inner {
        if token == "," {
            out.push(std::mem::take(&mut current));
        } else {
            current.push_str(token);
        }
    }
    out.push(current);
    out
}

/// Top-level comma split of an operand list (brace and bracket groups stay whole).
fn operands(body: &[String]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0i32;
    for token in body {
        match token.as_str() {
            "{" | "[" | "(" => depth += 1,
            "}" | "]" | ")" => depth -= 1,
            _ => {}
        }
        if (token == "," || token == ";") && depth == 0 {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
        } else {
            current.push(token.clone());
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn line(guard: &[String], op: &str, operands: &[String]) -> Stmt {
    let mut out: Stmt = guard.to_vec();
    out.push(op.to_owned());
    for (k, operand) in operands.iter().enumerate() {
        if k > 0 {
            out.push(",".into());
        }
        out.push(operand.clone());
    }
    out.push(";".into());
    out
}

/// `mov.vN.T {d..}, {s..}`: every source is read before any destination is written.
fn parallel_move(guard: &[String], parts: &[&str], body: &[String], counter: &mut usize) -> Option<Vec<Stmt>> {
    let ty = parts.get(2)?;
    if parts.len() != 3 {
        return None;
    }
    let ops = operands(body);
    let [dst, src] = ops.as_slice() else { return None };
    let dst = elements(dst);
    let src = elements(src);
    let n: usize = parts[1][1..].parse().ok()?;
    if dst.len() != n || src.len() != n {
        return None;
    }
    let scalar_op = format!("mov.{ty}");
    let overlap = dst.iter().enumerate().any(|(k, d)| src.iter().enumerate().any(|(j, s)| j > k && s == d));
    let mut out = Vec::new();
    if !overlap {
        for (d, s) in dst.iter().zip(&src) {
            if d != s {
                out.push(line(guard, &scalar_op, &[d.clone(), s.clone()]));
            }
        }
        return Some(out);
    }
    *counter += 1;
    let temps: Vec<String> = (0..n).map(|k| format!("%__vmv{}_{k}", *counter)).collect();
    for temp in &temps {
        out.push(vec![".reg".into(), format!(".{ty}"), temp.clone(), ";".into()]);
    }
    for (t, s) in temps.iter().zip(&src) {
        out.push(line(guard, &scalar_op, &[t.clone(), s.clone()]));
    }
    for (d, t) in dst.iter().zip(&temps) {
        out.push(line(guard, &scalar_op, &[d.clone(), t.clone()]));
    }
    Some(out)
}

fn element_bytes(ty: &str) -> Option<i64> {
    Some(match ty {
        "b8" | "u8" | "s8" => 1,
        "b16" | "u16" | "s16" | "f16" | "bf16" => 2,
        "b32" | "u32" | "s32" | "f32" | "f16x2" | "bf16x2" => 4,
        "b64" | "u64" | "s64" | "f64" => 8,
        _ => return None,
    })
}

/// `ld/st.*.v8.T` -> two `.v4.T` accesses (the second at `+ 4 * sizeof(T)`).
fn split_v8(guard: &[String], parts: &[&str], body: &[String]) -> Option<Vec<Stmt>> {
    let store = parts[0] == "st";
    let at = parts.iter().position(|p| *p == "v8")?;
    let size = element_bytes(parts.get(at + 1)?)?;
    let op: Vec<&str> = parts.iter().map(|p| if *p == "v8" { "v4" } else { p }).collect();
    let op = op.join(".");
    let ops = operands(body);
    if ops.len() != 2 {
        return None;
    }
    let (list, addr) = if store { (&ops[1], &ops[0]) } else { (&ops[0], &ops[1]) };
    let regs = elements(list);
    if regs.len() != 8 {
        return None;
    }
    let inner = match addr.as_slice() {
        [open, inner @ .., close] if open == "[" && close == "]" => inner.concat(),
        _ => return None,
    };
    let (base, offset) = match inner.split_once('+') {
        Some((base, offset)) => (base.to_owned(), parse_offset(offset)?),
        None => (inner, 0),
    };
    let half = |h: usize| -> Stmt {
        let mut list = vec!["{".to_owned()];
        for (k, reg) in regs[4 * h..4 * h + 4].iter().enumerate() {
            if k > 0 {
                list.push(",".into());
            }
            list.push(reg.clone());
        }
        list.push("}".into());
        let address = vec!["[".to_owned(), format!("{base}+{}", offset + 4 * size * h as i64), "]".to_owned()];
        let mut out: Stmt = guard.to_vec();
        out.push(op.clone());
        if store {
            out.extend(address);
            out.push(",".into());
            out.extend(list);
        } else {
            out.extend(list);
            out.push(",".into());
            out.extend(address);
        }
        out.push(";".into());
        out
    };
    // A load must not overwrite its base register before the second half reads it.
    let first_clobbers = !store && regs[..4].iter().any(|r| *r == base);
    Some(if first_clobbers { vec![half(1), half(0)] } else { vec![half(0), half(1)] })
}

fn parse_offset(text: &str) -> Option<i64> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else {
        text.parse().ok()
    }
}

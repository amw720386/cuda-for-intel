//! Source-level canonicalisation that runs before the PTX parser.
//!
//! Real nvcc output (CUDA 11 to 13, sm_70 to sm_120) is full of spellings that
//! do not change the lowering: cache-hint qualifiers (`ld.global.L2::128B`),
//! memory-order qualifiers (`ld.relaxed.gpu`), `sm_90a` targets, float-typed
//! moves (`mov.f64`), `barrier.sync` and so on. This pass rewrites them to the
//! canonical forms the parser already understands, or expands them into short
//! sequences of already-supported instructions (`cp.async`, legacy warp
//! intrinsics, `.func` calls, ...). Anything it does not recognise is passed
//! through untouched, so it cannot make a previously accepted module fail.

use super::{Error, inline, parse};

pub(super) type Stmt = Vec<String>;

/// Entry point: returns canonical PTX text.
pub(super) fn run(source: &str) -> Result<String, Error> {
    let tokens = merge_double_colons(parse::tokens(source)?);
    let statements = split(tokens);
    let statements = inline::expand(statements)?;
    let mut ctx = Ctx::default();
    let mut out: Vec<String> = Vec::with_capacity(statements.len());
    // Index in `out` of the opening brace of the current entry, plus pending
    // scratch-register declarations that the rewrites of that entry need.
    let mut entry_open: Option<usize> = None;
    let mut depth = 0i32;
    for stmt in statements {
        if stmt.first().is_some_and(|t| t == "{") {
            depth += 1;
        } else if stmt.first().is_some_and(|t| t == "}") {
            depth -= 1;
            if depth == 0 {
                if let Some(open) = entry_open.take() {
                    let decls = ctx.take_decls();
                    if !decls.is_empty() {
                        let at = decl_insert_position(&out, open);
                        for (i, decl) in decls.into_iter().enumerate() {
                            out.insert(at + i, decl);
                        }
                    }
                }
                ctx.block_threads = None;
            }
        }
        if is_entry_header(&stmt) {
            ctx.block_threads = block_threads(&stmt);
            let text = fix_entry_header(&stmt);
            out.push(text);
            depth += 1;
            entry_open = Some(out.len() - 1);
            continue;
        }
        for text in rewrite(&mut ctx, &stmt) {
            out.push(text);
        }
    }
    let out = reorder_prologues(out);
    let mut text = out.join("\n");
    text.push('\n');
    Ok(text)
}

/// Puts each entry's declarations in the order the parser expects (`.local`,
/// `.reg`, `.shared`), hoisting `.reg` declarations that follow instructions.
fn reorder_prologues(lines: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].clone();
        let entry = line.contains(".entry") && line.ends_with('{');
        out.push(line);
        i += 1;
        if !entry {
            continue;
        }
        let start = i;
        let mut depth = 1i32;
        let mut end = i;
        while end < lines.len() {
            if lines[end] == "{" || lines[end].ends_with(" {") {
                depth += 1;
            } else if lines[end] == "}" {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            end += 1;
        }
        let mut locals = Vec::new();
        let mut regs = Vec::new();
        let mut shared = Vec::new();
        let mut body = Vec::new();
        for line in &lines[start..end] {
            // Only top-level declarations move; nested scopes were flattened earlier.
            if line.starts_with(".local") {
                locals.push(line.clone());
            } else if line.starts_with(".reg") {
                regs.extend(split_reg_list(line));
            } else if line.starts_with(".shared") {
                shared.push(line.clone());
            } else {
                body.push(line.clone());
            }
        }
        out.extend(locals);
        out.extend(regs);
        out.extend(shared);
        out.extend(body);
        i = end;
    }
    out
}

/// `.reg .u32 a , b ;` -> one declaration per name.
fn split_reg_list(line: &str) -> Vec<String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if !tokens.contains(&",") {
        return vec![line.to_owned()];
    }
    let ty = tokens.get(1).copied().unwrap_or(".b32");
    tokens[2..]
        .iter()
        .filter(|t| **t != "," && **t != ";")
        .map(|name| format!(".reg {ty} {name} ;"))
        .collect()
}

fn is_entry_header(stmt: &[String]) -> bool {
    stmt.last().is_some_and(|t| t == "{")
        && stmt.len() > 1
        && stmt.iter().any(|t| t == ".entry")
}

fn fix_entry_header(stmt: &[String]) -> String {
    let mut out: Vec<String> = Vec::with_capacity(stmt.len());
    let mut i = 0;
    while i < stmt.len() {
        let token = stmt[i].as_str();
        if token == ".param" {
            // `.param .s32 x` -> `.u32`; `.f64`/`.s64` -> `.u64`.
            out.push(".param".into());
            if let Some(ty) = stmt.get(i + 1) {
                out.push(
                    match ty.as_str() {
                        ".s32" => ".u32",
                        ".s64" | ".f64" => ".u64",
                        other => other,
                    }
                    .into(),
                );
                i += 2;
                continue;
            }
        }
        out.push(token.to_owned());
        i += 1;
    }
    out.join(" ")
}

fn block_threads(stmt: &[String]) -> Option<u32> {
    let at = stmt.iter().position(|t| t == ".maxntid" || t == ".reqntid")?;
    let mut total = 1u32;
    let mut i = at + 1;
    loop {
        total = total.checked_mul(stmt.get(i)?.parse::<u32>().ok()?)?;
        if stmt.get(i + 1).is_some_and(|t| t == ",") {
            i += 2;
        } else {
            return Some(total);
        }
    }
}

/// Position in `out` at which scratch `.reg` declarations of the entry opened
/// at `open` can be inserted: after the leading `.local`/`.reg` run.
fn decl_insert_position(out: &[String], open: usize) -> usize {
    let mut at = open + 1;
    while out
        .get(at)
        .is_some_and(|s| s.starts_with(".local") || s.starts_with(".reg"))
    {
        at += 1;
    }
    at
}

fn merge_double_colons(tokens: Vec<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let mut token = tokens[i].to_owned();
        i += 1;
        while i + 2 < tokens.len()
            && tokens[i] == ":"
            && tokens[i + 1] == ":"
            && !token.is_empty()
            && !matches!(token.as_str(), "(" | ")" | "," | ";" | "{" | "}" | "[" | "]")
        {
            token.push_str("::");
            token.push_str(tokens[i + 2]);
            i += 3;
        }
        out.push(token);
    }
    out
}

/// Splits a token stream into statements. Header directives without a
/// terminating semicolon (`.version`, `.target`, `.address_size`) are
/// statements of their own; `{` and `}` at statement start are statements.
fn split(tokens: Vec<String>) -> Vec<Stmt> {
    let mut out: Vec<Stmt> = Vec::new();
    let mut cur: Stmt = Vec::new();
    let mut iter = tokens.into_iter().peekable();
    while let Some(token) = iter.next() {
        if cur.is_empty() {
            match token.as_str() {
                "{" | "}" => {
                    out.push(vec![token]);
                    continue;
                }
                ".version" | ".address_size" => {
                    let mut stmt = vec![token];
                    if let Some(value) = iter.next() {
                        stmt.push(value);
                    }
                    out.push(stmt);
                    continue;
                }
                ".target" => {
                    let mut stmt = vec![token];
                    while let Some(next) = iter.peek() {
                        if next.starts_with('.') {
                            break;
                        }
                        stmt.push(iter.next().unwrap());
                    }
                    out.push(stmt);
                    continue;
                }
                _ => {}
            }
            if iter.peek().is_some_and(|t| t == ":") && !token.starts_with('.') && !token.starts_with('@') {
                let colon = iter.next().unwrap();
                out.push(vec![token, colon]);
                continue;
            }
        }
        let terminates_on_brace = {
            let has = |name: &str| cur.iter().any(|t| t == name);
            !cur.is_empty()
                && token == "{"
                && (has(".entry") || has(".func"))
                && !has(".global")
                && !has(".const")
                && !has(".shared")
                && !has(".local")
        };
        if terminates_on_brace {
            cur.push(token);
            out.push(std::mem::take(&mut cur));
            continue;
        }
        let semi = token == ";";
        cur.push(token);
        if semi {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[derive(Default)]
struct Ctx {
    block_threads: Option<u32>,
    /// Scratch registers requested by rewrites of the current entry.
    t32: u32,
    t64: u32,
    tp: u32,
}

impl Ctx {
    fn take_decls(&mut self) -> Vec<String> {
        let mut decls = Vec::new();
        if self.t32 > 0 {
            decls.push(format!(".reg .b32 %__t32_<{}> ;", self.t32));
        }
        if self.t64 > 0 {
            decls.push(format!(".reg .b64 %__t64_<{}> ;", self.t64));
        }
        if self.tp > 0 {
            decls.push(format!(".reg .pred %__tp_<{}> ;", self.tp));
        }
        self.t32 = 0;
        self.t64 = 0;
        self.tp = 0;
        decls
    }
    fn need32(&mut self, n: u32) {
        self.t32 = self.t32.max(n);
    }
    fn need64(&mut self, n: u32) {
        self.t64 = self.t64.max(n);
    }
    fn needp(&mut self, n: u32) {
        self.tp = self.tp.max(n);
    }
}

/// A parsed instruction statement.
struct Inst {
    /// `@ %p` or `@ ! %p`, empty when unguarded.
    guard: String,
    op: String,
    operands: Vec<String>,
}

impl Inst {
    fn parse(stmt: &[String]) -> Option<Inst> {
        let mut i = 0;
        let mut guard = String::new();
        if stmt.first().is_some_and(|t| t == "@") {
            i = 1;
            let mut parts = vec!["@".to_owned()];
            if stmt.get(i).is_some_and(|t| t == "!") {
                parts.push("!".into());
                i += 1;
            }
            parts.push(stmt.get(i)?.clone());
            i += 1;
            guard = parts.join(" ");
        }
        let op = stmt.get(i)?.clone();
        if op.starts_with('.') || op == "{" || op == "}" || op == ";" {
            return None;
        }
        if stmt.last().map(String::as_str) != Some(";") {
            return None;
        }
        let body = &stmt[i + 1..stmt.len() - 1];
        let mut operands = Vec::new();
        let mut current: Vec<&str> = Vec::new();
        let mut depth = 0i32;
        for token in body {
            match token.as_str() {
                "[" | "{" | "(" => depth += 1,
                "]" | "}" | ")" => depth -= 1,
                _ => {}
            }
            if token == "," && depth == 0 {
                operands.push(current.join(" "));
                current.clear();
            } else {
                current.push(token);
            }
        }
        if !current.is_empty() {
            operands.push(current.join(" "));
        }
        Some(Inst { guard, op, operands })
    }

    fn render(&self) -> String {
        render(&self.guard, &self.op, &self.operands)
    }

    fn operand(&self, index: usize) -> &str {
        self.operands.get(index).map_or("", String::as_str)
    }
}

fn render(guard: &str, op: &str, operands: &[String]) -> String {
    let mut text = String::new();
    if !guard.is_empty() {
        text.push_str(guard);
        text.push(' ');
    }
    text.push_str(op);
    if !operands.is_empty() {
        text.push(' ');
        text.push_str(&operands.join(" , "));
    }
    text.push_str(" ;");
    text
}

fn immediate(text: &str) -> Option<u64> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        text.parse::<u64>().ok()
    }
}

/// `0h3C00` / `0d3FF0000000000000` float literals become plain hex integers.
fn literal(token: &str) -> String {
    let hex = |body: &str, digits: usize| body.len() == digits && body.bytes().all(|b| b.is_ascii_hexdigit());
    if let Some(body) = token.strip_prefix("0h").or_else(|| token.strip_prefix("0H")) {
        if hex(body, 4) {
            return format!("0x{body}");
        }
    }
    if let Some(body) = token.strip_prefix("0d").or_else(|| token.strip_prefix("0D")) {
        if hex(body, 16) {
            return format!("0x{body}");
        }
    }
    token.to_owned()
}

fn plain(stmt: &[String]) -> String {
    stmt.iter().map(|t| literal(t)).collect::<Vec<_>>().join(" ")
}

fn rewrite(ctx: &mut Ctx, stmt: &[String]) -> Vec<String> {
    let Some(first) = stmt.first() else {
        return Vec::new();
    };
    match first.as_str() {
        ".version" => return vec![plain(stmt)],
        ".target" => {
            let target = stmt.get(1).map_or("sm_80", String::as_str);
            let digits: String = target
                .strip_prefix("sm_")
                .unwrap_or("80")
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            return vec![format!(".target sm_{}", if digits.is_empty() { "80" } else { &digits })];
        }
        ".pragma" => return Vec::new(),
        ".reg" => return vec![rewrite_reg(stmt)],
        _ => {}
    }
    let Some(inst) = Inst::parse(stmt) else {
        return vec![plain(stmt)];
    };
    match rewrite_inst(ctx, &inst) {
        Some(lines) => lines,
        None => vec![plain(stmt)],
    }
}

fn rewrite_reg(stmt: &[String]) -> String {
    let mut out: Vec<String> = stmt.iter().map(|t| literal(t)).collect();
    if let Some(ty) = out.get_mut(1) {
        let mapped = match ty.as_str() {
            ".f64" => Some(".b64"),
            ".bf16" => Some(".b16"),
            ".bf16x2" => Some(".b32"),
            ".tf32" => Some(".b32"),
            _ => None,
        };
        if let Some(mapped) = mapped {
            *ty = mapped.into();
        }
    }
    out.join(" ")
}

const FENCE: &str = "membar.gl ;";

/// Scalar float / half type suffixes to their bit-type equivalents.
fn bit_type(part: &str) -> &str {
    match part {
        "f64" | "s64" => "b64",
        "f16" | "bf16" => "b16",
        "f16x2" | "bf16x2" => "b32",
        other => other,
    }
}

fn rewrite_inst(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let op = inst.op.as_str();
    let family = op.split('.').next().unwrap_or(op);
    match family {
        "ld" | "ldu" | "st" => return rewrite_ldst(ctx, inst),
        "mov" => return rewrite_mov(inst),
        "selp" => {
            let parts: Vec<&str> = op.split('.').collect();
            if parts.len() == 2 {
                let ty = match parts[1] {
                    "f64" => Some("b64"),
                    "s16" | "u16" | "b16" | "f16" | "bf16" | "f16x2" | "bf16x2" => Some("b32"),
                    _ => None,
                };
                if let Some(ty) = ty {
                    let operands: Vec<String> = inst.operands.iter().map(|o| literal(o)).collect();
                    return Some(vec![render(&inst.guard, &format!("selp.{ty}"), &operands)]);
                }
            }
            return None;
        }
        "prefetch" | "prefetchu" | "discard" | "applypriority" | "nanosleep" | "pmevent"
        | "griddepcontrol" | "setmaxnreg" => return Some(Vec::new()),
        "barrier" if op.starts_with("barrier.cluster") => return Some(Vec::new()),
        "createpolicy" => {
            let dst = inst.operand(0);
            return Some(vec![render(&inst.guard, "mov.b64", &[dst.to_owned(), "0".into()])]);
        }
        "trap" | "exit" | "brkpt" => {
            if inst.operands.is_empty() {
                return Some(vec![render(&inst.guard, "ret", &[])]);
            }
            return None;
        }
        "fence" | "membar" => {
            let parts: Vec<&str> = op.split('.').collect();
            if parts.contains(&"proxy") || parts.contains(&"mbarrier_init") {
                return Some(Vec::new());
            }
            return Some(vec![render(&inst.guard, "membar.gl", &[])]);
        }
        "bar" | "barrier" => return rewrite_barrier(ctx, inst),
        "cp" => return rewrite_cp_async(ctx, inst),
        "vote" => {
            let parts: Vec<&str> = op.split('.').collect();
            if parts.get(1) != Some(&"sync") && inst.operands.len() == 2 {
                let mut parts = parts.clone();
                parts.insert(1, "sync");
                let mut operands = inst.operands.clone();
                operands.push("0xffffffff".into());
                return Some(vec![render(&inst.guard, &parts.join("."), &operands)]);
            }
            return None;
        }
        "shfl" => {
            let parts: Vec<&str> = op.split('.').collect();
            if parts.get(1) != Some(&"sync") && inst.operands.len() == 4 {
                let mut parts = parts.clone();
                parts.insert(1, "sync");
                let mut operands = inst.operands.clone();
                operands.push("0xffffffff".into());
                return Some(vec![render(&inst.guard, &parts.join("."), &operands)]);
            }
            return None;
        }
        _ => {}
    }
    // Literal canonicalisation for everything else (0h / 0d immediates).
    if inst.operands.iter().any(|o| literal(o) != *o) {
        let operands: Vec<String> = inst.operands.iter().map(|o| literal(o)).collect();
        return Some(vec![render(&inst.guard, &inst.op, &operands)]);
    }
    None
}

fn rewrite_mov(inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let ty = *parts.get(1)?;
    let dst = inst.operand(0);
    let src = literal(inst.operand(1));
    if inst.operands.len() != 2 {
        return None;
    }
    if ty == "pred" {
        return Some(match immediate(&src) {
            Some(0) => vec![render(&inst.guard, "setp.ne.u32", &[dst.into(), "0".into(), "0".into()])],
            Some(_) => vec![render(&inst.guard, "setp.eq.u32", &[dst.into(), "0".into(), "0".into()])],
            None => vec![render(&inst.guard, "and.pred", &[dst.into(), src.clone(), src])],
        });
    }
    if let Some(sreg) = special_register(&src) {
        // Registers the translator understands natively are left alone.
        let native = matches!(
            sreg,
            "tid" | "ntid" | "ctaid" | "nctaid" | "laneid"
        );
        if native {
            return None;
        }
        let wide = matches!(ty, "u64" | "b64");
        let line = |op: &str, ops: &[&str]| {
            render(
                &inst.guard,
                op,
                &ops.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>(),
            )
        };
        let mov = if wide { "mov.b64" } else { "mov.u32" };
        let lines = match sreg {
            "WARP_SZ" => vec![line(mov, &[dst, "32"])],
            "gridid" | "smid" | "pm0" | "pm1" | "pm2" | "pm3" | "pm4" | "pm5" | "pm6" | "pm7"
            | "envreg0" | "cluster_ctaid" | "cluster_ctarank" | "cluster_nctarank"
            | "cluster_ctaid.x" | "is_explicit_cluster" | "aggr_smem_size" => vec![line(mov, &[dst, "0"])],
            "nsmid" => vec![line(mov, &[dst, "1"])],
            "cluster_nctaid" => vec![line(mov, &[dst, "1"])],
            "lanemask_eq" => vec![
                line("mov.u32", &[dst, "%laneid"]),
                line("shl.b32", &[dst, "1", dst]),
            ],
            "lanemask_lt" | "lanemask_le" | "lanemask_ge" | "lanemask_gt" => {
                let base = if matches!(sreg, "lanemask_lt" | "lanemask_ge") { "1" } else { "2" };
                let mut lines = vec![
                    line("mov.u32", &[dst, "%laneid"]),
                    line("shl.b32", &[dst, base, dst]),
                    line("sub.u32", &[dst, dst, "1"]),
                ];
                if matches!(sreg, "lanemask_ge" | "lanemask_gt") {
                    lines.push(line("not.b32", &[dst, dst]));
                }
                lines
            }
            _ => return None,
        };
        return Some(lines);
    }
    let mapped = match ty {
        "b16" | "u16" | "s16" | "f16" | "bf16" | "f16x2" | "bf16x2" | "pred" => "mov.b32",
        "f64" | "s64" => "mov.b64",
        _ => return None,
    };
    if (mapped == "mov.b32" && ty == "b16") && src == inst.operand(1) {
        // `mov.b16 d, s` is only accepted by the parser as mov.b32.
    }
    Some(vec![render(&inst.guard, mapped, &[dst.into(), src])])
}

/// Returns the canonical special-register name (without the `%`).
fn special_register(token: &str) -> Option<&str> {
    let name = token.strip_prefix('%')?;
    let base = name.split('.').next().unwrap_or(name);
    match base {
        "tid" | "ntid" | "ctaid" | "nctaid" | "laneid" | "WARP_SZ" | "gridid" | "smid"
        | "nsmid" | "pm0" | "pm1" | "pm2" | "pm3" | "pm4" | "pm5" | "pm6" | "pm7" | "envreg0"
        | "is_explicit_cluster" | "lanemask_eq" | "lanemask_lt" | "lanemask_le"
        | "lanemask_ge" | "lanemask_gt" | "cluster_ctarank" | "cluster_nctarank"
        | "aggr_smem_size" | "cluster_ctaid" | "cluster_nctaid" => Some(base),
        _ => None,
    }
}

fn rewrite_barrier(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    // bar.sync / barrier.sync[.aligned]; arrive/red/warp forms are not CTA barriers.
    if parts.get(1) != Some(&"sync") || (parts.len() > 2 && parts[2] != "aligned") {
        return None;
    }
    if parts[0] == "bar" && parts.len() > 2 {
        return None;
    }
    match inst.operands.len() {
        1 => {}
        2 => {
            let count = immediate(inst.operand(1))?;
            if ctx.block_threads != Some(u32::try_from(count).ok()?) {
                return None;
            }
        }
        _ => return None,
    }
    Some(vec![render(&inst.guard, "bar.sync", &["0".into()])])
}

fn rewrite_ldst(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let is_store = inst.op.starts_with("st.");
    let parts: Vec<&str> = inst.op.split('.').collect();
    if parts.iter().any(|p| matches!(*p, "async" | "bulk" | "param" | "const")) {
        // ld.param.f64 and friends only need the type mapping.
        if parts.contains(&"param") {
            let mut changed = false;
            let mapped: Vec<String> = parts
                .iter()
                .map(|p| {
                    let m = match *p {
                        "f64" | "s64" => "u64",
                        "f16" | "bf16" => "u32",
                        _ => *p,
                    };
                    changed |= m != *p;
                    m.to_owned()
                })
                .collect();
            if changed {
                return Some(vec![render(&inst.guard, &mapped.join("."), &inst.operands)]);
            }
        }
        return None;
    }
    let mut out_parts: Vec<String> = vec![if parts[0] == "ldu" { "ld".into() } else { parts[0].to_owned() }];
    // Generic accesses (no state space) are resolved per pointer by the emitter.
    let spaced = parts[1..].iter().any(|p| {
        matches!(*p, "global" | "shared" | "local" | "param" | "const" | "shared::cta" | "global::cta")
    });
    let mut changed = parts[0] == "ldu" || !spaced;
    if !spaced {
        out_parts.push("global".into());
    }
    let mut acquire = false;
    let mut release = false;
    let mut cache_hint = false;
    let mut after_order = false;
    let mut vector = 1u32;
    let mut element = String::new();
    for part in &parts[1..] {
        let part = *part;
        if part.contains("cache_hint") {
            cache_hint = true;
            changed = true;
            continue;
        }
        if part.starts_with("L1::") || part.starts_with("L2::") || part.starts_with("L2") && part.contains("::") {
            changed = true;
            continue;
        }
        match part {
            "weak" | "volatile" | "mmio" | "cg" | "ca" | "cs" | "lu" | "lc" | "wb" | "wt" | "cv" => {
                changed = true;
                after_order = false;
                continue;
            }
            "relaxed" => {
                changed = true;
                after_order = true;
                continue;
            }
            "acquire" => {
                changed = true;
                acquire = true;
                after_order = true;
                continue;
            }
            "release" => {
                changed = true;
                release = true;
                after_order = true;
                continue;
            }
            "cta" | "gpu" | "sys" | "cluster" if after_order => {
                changed = true;
                continue;
            }
            "shared::cta" => {
                changed = true;
                out_parts.push("shared".into());
                continue;
            }
            "global::cta" => {
                out_parts.push("global".into());
                changed = true;
                continue;
            }
            "v2" => vector = 2,
            "v4" => vector = 4,
            "v8" => vector = 8,
            _ => {}
        }
        after_order = false;
        let mut mapped = bit_type(part);
        if is_store && matches!(part, "s8" | "s16" | "s32") {
            mapped = match part {
                "s8" => "b8",
                "s16" => "b16",
                _ => "b32",
            };
        }
        if !is_store && part == "s64" {
            mapped = "b64";
        }
        if mapped != part {
            changed = true;
        }
        if matches!(part, "b8" | "u8" | "s8" | "b16" | "u16" | "s16" | "b32" | "u32" | "s32" | "f32" | "b64" | "u64" | "s64" | "f64" | "f16" | "bf16" | "f16x2" | "bf16x2") {
            element = mapped.to_owned();
        }
        out_parts.push(mapped.to_owned());
    }
    let mut operands = inst.operands.clone();
    if cache_hint && operands.len() > if is_store { 2 } else { 2 } {
        operands.pop();
    }
    let wide_vector = vector > 1 && matches!(element.as_str(), "b64" | "u64");
    let mut lines = Vec::new();
    if release {
        lines.push(FENCE.to_owned());
    }
    if wide_vector {
        lines.extend(expand_wide_vector(inst, &out_parts, vector, &operands, is_store)?);
    } else if changed {
        lines.push(render(&inst.guard, &out_parts.join("."), &operands));
    } else {
        return None;
    }
    if acquire {
        lines.push(FENCE.to_owned());
    }
    let _ = ctx;
    Some(lines)
}

/// `ld.global.v2.u64 {a,b}, [p]` -> two scalar loads (and likewise for stores).
fn expand_wide_vector(
    inst: &Inst,
    parts: &[String],
    vector: u32,
    operands: &[String],
    is_store: bool,
) -> Option<Vec<String>> {
    let scalar: Vec<String> = parts
        .iter()
        .filter(|p| !matches!(p.as_str(), "v2" | "v4" | "v8"))
        .cloned()
        .collect();
    let scalar = scalar.join(".");
    let (list, address) = if is_store {
        (operands.get(1)?, operands.first()?)
    } else {
        (operands.first()?, operands.get(1)?)
    };
    let regs: Vec<String> = list
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .map(|r| r.trim().to_owned())
        .collect();
    if regs.len() != vector as usize {
        return None;
    }
    let inner = address.trim().trim_start_matches('[').trim_end_matches(']').trim();
    let (base, offset) = match inner.split_once('+') {
        Some((base, offset)) => (base.trim(), immediate(offset)?),
        None => (inner, 0),
    };
    let mut order: Vec<usize> = (0..regs.len()).collect();
    // A load must not clobber the base register before later elements read it.
    if !is_store {
        if let Some(clobber) = regs.iter().position(|r| r == base) {
            order.retain(|&i| i != clobber);
            order.push(clobber);
        }
    }
    Some(
        order
            .into_iter()
            .map(|i| {
                let addr = format!("[ {base}+{} ]", offset + 8 * i as u64);
                if is_store {
                    render(&inst.guard, &scalar, &[addr, regs[i].clone()])
                } else {
                    render(&inst.guard, &scalar, &[regs[i].clone(), addr])
                }
            })
            .collect(),
    )
}

fn rewrite_cp_async(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    if parts.get(1) != Some(&"async") {
        return None;
    }
    match parts.get(2).copied() {
        Some("commit_group" | "wait_group" | "wait_all") => return Some(Vec::new()),
        Some("bulk") if matches!(parts.get(3).copied(), Some("commit_group" | "wait_group")) => {
            return Some(Vec::new());
        }
        Some("ca" | "cg") => {}
        _ => return None,
    }
    let dst = inst.operands.first()?;
    let src = inst.operands.get(1)?;
    let size = immediate(inst.operands.get(2)?)?;
    let zfill = inst.operands.get(3);
    let (ld, st, regs): (&str, &str, &[&str]) = match size {
        4 => ("ld.global.u32", "st.shared.u32", &["%__t32_0"]),
        8 => ("ld.global.v2.u32", "st.shared.v2.u32", &["%__t32_0", "%__t32_1"]),
        16 => (
            "ld.global.v4.u32",
            "st.shared.v4.u32",
            &["%__t32_0", "%__t32_1", "%__t32_2", "%__t32_3"],
        ),
        _ => return None,
    };
    ctx.need32(regs.len() as u32);
    let list = if regs.len() == 1 {
        regs[0].to_owned()
    } else {
        format!("{{ {} }}", regs.join(" , "))
    };
    let mut lines = Vec::new();
    let mut load_guard = inst.guard.clone();
    if let Some(src_size) = zfill {
        ctx.needp(1);
        for reg in regs {
            lines.push(format!("mov.u32 {reg} , 0 ;"));
        }
        lines.push(format!("setp.ne.u32 %__tp_0 , {src_size} , 0 ;"));
        load_guard = "@ %__tp_0".to_owned();
        if !inst.guard.is_empty() {
            // Both the instruction guard and the size check must hold.
            ctx.needp(2);
            lines.push(format!("and.pred %__tp_1 , {} , %__tp_0 ;", inst.guard.trim_start_matches("@ ")));
            load_guard = "@ %__tp_1".to_owned();
        }
        if regs.len() == 1 {
            // Scalar form loads straight into the scratch register.
        }
    } else {
        for reg in regs {
            lines.push(format!("mov.u32 {reg} , 0 ;"));
        }
    }
    let ld_text = render(&load_guard, ld, &[list.clone(), src.clone()]);
    let st_text = render(&inst.guard, st, &[dst.clone(), list]);
    lines.push(ld_text);
    lines.push(st_text);
    Some(lines)
}

/// Removes plain `{ ... }` scopes left inside entry bodies (their `.reg`
/// declarations were already hoisted by the scope flattener).
pub(super) fn finish(text: &str) -> Result<String, Error> {
    let tokens: Vec<String> = parse::tokens(text)?.into_iter().map(str::to_owned).collect();
    let statements = split(tokens);
    let mut out = String::with_capacity(text.len());
    let mut depth = 0i32;
    let mut dead = false;
    for stmt in statements {
        let lone = |t: &str| stmt.len() == 1 && stmt[0] == t;
        let label = stmt.len() == 2 && stmt[1] == ":";
        if dead && !label && !lone("}") {
            continue;
        }
        dead = stmt.len() == 2 && stmt[0] == "ret" && stmt[1] == ";";
        if is_entry_header(&stmt) {
            depth = 1;
        } else if lone("{") && depth >= 1 {
            depth += 1;
            continue;
        } else if lone("}") && depth >= 1 {
            depth -= 1;
            if depth >= 1 {
                continue;
            }
        }
        out.push_str(&stmt.join(" "));
        out.push('\n');
    }
    Ok(out)
}

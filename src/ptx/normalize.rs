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

#[path = "tier3.rs"]
mod tier3;
#[path = "tier3_mma.rs"]
mod tier3_mma;

pub(super) type Stmt = Vec<String>;

/// Entry point: returns canonical PTX text.
pub(super) fn run(source: &str) -> Result<String, Error> {
    let source = tier3::strip_debug(source);
    let tokens = merge_addresses(merge_double_colons(parse::tokens(&source)?));
    let statements = vector_regs::run(split(tokens));
    tier3::check_unsupported(&statements)?;
    let (statements, dynamic) = extract_dynamic_shared(statements);
    let statements = inline::expand(statements)?;
    let mut entry_locals = tier3::entry_locals(&statements).into_iter();
    let mut ctx = Ctx::default();
    let mut out: Vec<String> = Vec::with_capacity(statements.len());
    // Index in `out` of the opening brace of the current entry, plus pending
    // scratch-register declarations that the rewrites of that entry need.
    let mut entry_open: Option<usize> = None;
    let mut depth = 0i32;
    let mut uses_dynamic = false;
    for stmt in statements {
        if let (false, Some((name, _))) = (uses_dynamic, &dynamic) {
            if stmt.iter().any(|t| t == name || t.strip_prefix(name.as_str()).is_some_and(|r| r.starts_with('+'))) {
                uses_dynamic = true;
            }
        }
        if stmt.first().is_some_and(|t| t == "{") {
            depth += 1;
        } else if stmt.first().is_some_and(|t| t == "}") {
            depth -= 1;
            if depth == 0 {
                // Trailing labels (`func_end0:`) after the last `ret` form an empty block:
                // drop the unreferenced ones, give a referenced one a `ret`.
                while entry_open.is_some() && out.last().is_some_and(|l| l.ends_with(" :")) {
                    let label = out.last().unwrap().trim_end_matches(" :").to_owned();
                    let used = out[entry_open.unwrap_or(0)..out.len() - 1].iter().any(|l| !l.ends_with(" :") && l.split_whitespace().any(|t| t == label));
                    if used {
                        out.push("ret ;".into());
                        break;
                    }
                    out.pop();
                }
                if let Some(open) = entry_open.take() {
                    if super::load16_merge_enabled() {
                        merge_u16_load_pairs(&mut out, open, &mut ctx);
                    }
                    let mut decls = ctx.take_decls();
                    if let (true, Some((name, align))) = (uses_dynamic, &dynamic) {
                        // Dynamic window = min(device max, env, 64 KiB) minus the entry's static
                        // shared arrays, so static + dynamic never exceeds the limit.
                        let fixed: u64 = out[open..].iter().map(|l| static_shared_bytes(l)).sum();
                        let window = (u64::from(super::dynamic_shared_bytes())
                            .saturating_sub(fixed.next_multiple_of(16)))
                        .max(16);
                        decls.push(format!(".shared .align {align} .b8 {name} [ {window} ] ; /*dyn*/"));
                    }
                    uses_dynamic = false;
                    if !decls.is_empty() {
                        let at = decl_insert_position(&out, open);
                        out.splice(at..at, decls);
                    }
                }
                ctx.block_threads = None;
            }
        }
        if is_entry_header(&stmt) {
            ctx.widths.clear();
            tier3::begin_entry(&mut ctx.t3, entry_locals.next().unwrap_or_default(), dynamic.as_ref().map(|(name, _)| name.as_str()));
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
        let mut shared_tail = Vec::new();
        let mut body = Vec::new();
        for line in &lines[start..end] {
            // Only top-level declarations move; nested scopes were flattened earlier.
            if line.starts_with(".local") {
                locals.push(line.clone());
            } else if line.starts_with(".reg") {
                regs.extend(split_reg_list(line));
            } else if line.starts_with(".shared") {
                // The dynamic window must follow the static arrays.
                if line.contains("/*dyn*/") {
                    shared_tail.push(line.clone());
                } else {
                    shared.push(line.clone());
                }
            } else {
                body.push(line.clone());
            }
        }
        out.extend(locals);
        out.extend(regs);
        out.extend(shared);
        out.extend(shared_tail);
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

/// Module-level `.extern .shared` arrays all alias the dynamic shared window.
/// They are removed here; every entry that uses one gets a single window
/// declaration placed after its static shared arrays.
fn extract_dynamic_shared(statements: Vec<Stmt>) -> (Vec<Stmt>, Option<(String, String)>) {
    let mut names: Vec<String> = Vec::new();
    let mut align = "16".to_owned();
    let mut kept = Vec::with_capacity(statements.len());
    for stmt in statements {
        let is_extern_shared = stmt.iter().take(3).any(|t| t == ".extern")
            && stmt.iter().take(4).any(|t| t == ".shared")
            && stmt.last().is_some_and(|t| t == ";");
        if !is_extern_shared {
            kept.push(stmt);
            continue;
        }
        if let Some(at) = stmt.iter().position(|t| t == "[") {
            if at > 0 {
                names.push(stmt[at - 1].clone());
            }
        }
        if let Some(at) = stmt.iter().position(|t| t == ".align") {
            if let Some(value) = stmt.get(at + 1) {
                align = value.clone();
            }
        }
    }
    let Some(first) = names.first().cloned() else {
        return (kept, None);
    };
    if names.len() > 1 {
        for stmt in &mut kept {
            for token in stmt.iter_mut() {
                for other in &names[1..] {
                    if token == other {
                        *token = first.clone();
                    } else if let Some(rest) = token.strip_prefix(other.as_str()) {
                        if rest.starts_with('+') {
                            *token = format!("{first}{rest}");
                        }
                    }
                }
            }
        }
    }
    (kept, Some((first, align)))
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
        if token == ".noreturn" {
            i += 1;
            continue;
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

/// `[ %rd1 + 8 ]` -> `[ %rd1+8 ]` (the parser expects one address token).
fn merge_addresses(tokens: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        out.push(tokens[i].clone());
        if tokens[i] == "[" {
            if let Some(close) = tokens[i + 1..].iter().take(8).position(|t| t == "]") {
                let inner = &tokens[i + 1..i + 1 + close];
                if close > 1 && !inner.iter().any(|t| matches!(t.as_str(), "," | "{" | "}" | ";")) {
                    out.push(inner.concat());
                    i += close + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
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
    /// Private 32-bit scratch registers of the cu128 rewrites (own namespace `%__tc_`).
    tc: u32,
    /// Extra scratch-register declarations requested by rewrites of the current entry.
    extra_decls: std::collections::BTreeSet<String>,
    /// Declared register widths of the current entry: exact names and range prefixes.
    widths: Vec<(String, bool, u32)>,
    /// Number of wmma statements seen, used to give each its own 64-bit address register.
    wmma_seq: u32,
    t3: tier3::State,
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
        if self.tc > 0 {
            decls.push(format!(".reg .b32 %__tc_<{}> ;", self.tc));
        }
        decls.extend(std::mem::take(&mut self.extra_decls));
        self.tc = 0;
        self.t32 = 0;
        self.t64 = 0;
        self.tp = 0;
        decls.extend(self.t3.take_decls());
        decls
    }
    /// Width in bits (32 or 64) of a declared register, if known.
    fn width_of(&self, name: &str) -> Option<u32> {
        if let Some((_, _, width)) = self.widths.iter().find(|(n, range, _)| !range && n == name) {
            return Some(*width);
        }
        let prefix = name.trim_end_matches(|c: char| c.is_ascii_digit());
        self.widths
            .iter()
            .find(|(n, range, _)| *range && n == prefix)
            .map(|(_, _, width)| *width)
    }

    fn note_register_declaration(&mut self, stmt: &[String]) {
        let Some(ty) = stmt.get(1) else { return };
        let width = match ty.as_str() {
            ".b64" | ".u64" | ".s64" | ".f64" => 64,
            ".pred" => return,
            _ => 32,
        };
        let mut i = 2;
        while i < stmt.len() {
            let name = &stmt[i];
            if name == ";" || name == "," {
                i += 1;
                continue;
            }
            if stmt.get(i + 1).is_some_and(|t| t == "<") {
                self.widths.push((name.clone(), true, width));
                i += 4;
            } else {
                self.widths.push((name.clone(), false, width));
                i += 1;
            }
        }
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
        ".local" | ".shared" => {
            if let Some(lines) = tier3::decl(ctx, stmt) {
                return lines;
            }
        }
        ".global" | ".const" | ".visible" | ".weak" | ".extern" | ".common" => {
            tier3::module_decl(ctx, stmt);
        }
        ".reg" => {
            ctx.note_register_declaration(stmt);
            return vec![rewrite_reg(stmt)];
        }
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

#[path = "vec_small.rs"]
mod vec_small;

#[path = "vector_regs.rs"]
mod vector_regs;

#[path = "rewrites_cu128.rs"]
mod rewrites_cu128;

fn rewrite_inst(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    if let Some(lines) = rewrites_cu128::rewrite(ctx, inst) {
        return Some(lines);
    }
    if let Some(lines) = vec_small::rewrite(ctx, inst) {
        return Some(lines);
    }
    if let Some(lines) = tier3::rewrite(ctx, inst) {
        return Some(lines);
    }
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
        "mad" | "rcp" | "div" | "sqrt" if op.ends_with(".f32") => {
            let parts: Vec<&str> = op.split('.').collect();
            let has = |m: &str| parts.contains(&m);
            let (name, operands): (&str, Vec<String>) = match parts[0] {
                "mad" if !has("sat") && !has("rz") && !has("rm") && !has("rp") => {
                    ("fma.rn.f32", inst.operands.clone())
                }
                "rcp" if has("rn") && inst.operands.len() == 2 => (
                    "div.rn.f32",
                    vec![inst.operands[0].clone(), "0f3F800000".into(), inst.operands[1].clone()],
                ),
                "div" if has("full") => ("div.approx.f32", inst.operands.clone()),
                "div" if has("rn") && has("ftz") => ("div.rn.f32", inst.operands.clone()),
                "sqrt" if has("rn") && has("ftz") => ("sqrt.rn.f32", inst.operands.clone()),
                _ => return None,
            };
            return Some(vec![render(&inst.guard, name, &operands)]);
        }
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
        "wmma" => return rewrite_wmma(ctx, inst),
        "cp" => return rewrite_cp_async(ctx, inst),
        "vote" => {
            let mut parts: Vec<&str> = op.split('.').collect();
            let mut operands = inst.operands.clone();
            let mut changed = false;
            if parts.get(1) != Some(&"sync") && operands.len() == 2 {
                parts.insert(1, "sync");
                operands.push("0xffffffff".into());
                changed = true;
            }
            // A negated predicate source (`!%p`) needs its own register.
            let mut lines = Vec::new();
            if let Some(source) = operands.get(1).cloned() {
                if let Some(name) = source.strip_prefix('!') {
                    let name = name.trim().to_owned();
                    ctx.needp(3);
                    lines.push(format!("not.pred %__tp_2 , {name} ;"));
                    operands[1] = "%__tp_2".into();
                    changed = true;
                }
            }
            if !changed {
                return None;
            }
            lines.push(render(&inst.guard, &parts.join("."), &operands));
            return Some(lines);
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
        let value = immediate(&src).or_else(|| src.parse::<i64>().ok().map(|v| v as u64));
        return Some(match value {
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
            | "envreg0" | "cluster_ctaid" | "cluster_ctarank"
            | "cluster_ctaid.x" | "is_explicit_cluster" | "aggr_smem_size" => vec![line(mov, &[dst, "0"])],
            "nsmid" => vec![line(mov, &[dst, "1"])],
            "cluster_nctaid" | "cluster_nctarank" => vec![line(mov, &[dst, "1"])],
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
            // nvcc loads 32-bit parameters straight into 64-bit registers
            // (`ld.param.u32 %rd3, [p]`): load into a scratch word and extend.
            if let ([_, "param", ty @ ("u32" | "s32" | "b32")], [dst, addr]) = (&parts[..], &inst.operands[..]) {
                if ctx.width_of(dst) == Some(64) && !is_store {
                    ctx.need32(1);
                    let load = render(&inst.guard, &format!("ld.param.{ty}"), &["%__t32_0".to_owned(), addr.clone()]);
                    let cvt = if *ty == "s32" { "cvt.s64.s32" } else { "cvt.u64.u32" };
                    return Some(vec![load, render(&inst.guard, cvt, &[dst.clone(), "%__t32_0".to_owned()])]);
                }
            }
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
    let tokens = parse::tokens(text)?;
    let mut out = String::with_capacity(text.len());
    let mut depth = 0i32;
    let mut dead = false;
    split_each(&tokens, |stmt| {
        let lone = |t: &str| stmt.len() == 1 && stmt[0] == t;
        let label = stmt.len() == 2 && stmt[1] == ":";
        if dead && !label && !lone("}") {
            return;
        }
        dead = stmt.len() == 2 && stmt[0] == "ret" && stmt[1] == ";";
        if stmt.last().is_some_and(|t| *t == "{") && stmt.len() > 1 && stmt.contains(&".entry") {
            depth = 1;
        } else if lone("{") && depth >= 1 {
            depth += 1;
            return;
        } else if lone("}") && depth >= 1 {
            depth -= 1;
            if depth >= 1 {
                return;
            }
        }
        for (i, token) in stmt.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(token);
        }
        out.push('\n');
    });
    Ok(out)
}

/// Borrowed-token twin of [`split`]: calls `f` with each statement without allocating one.
fn split_each<'a>(tokens: &[&'a str], mut f: impl FnMut(&[&'a str])) {
    let mut cur: Vec<&'a str> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];
        i += 1;
        if cur.is_empty() {
            match token {
                "{" | "}" => {
                    f(&[token]);
                    continue;
                }
                ".version" | ".address_size" => {
                    if let Some(&value) = tokens.get(i) {
                        i += 1;
                        f(&[token, value]);
                    } else {
                        f(&[token]);
                    }
                    continue;
                }
                ".target" => {
                    let start = i;
                    while i < tokens.len() && !tokens[i].starts_with('.') {
                        i += 1;
                    }
                    let mut stmt = vec![token];
                    stmt.extend_from_slice(&tokens[start..i]);
                    f(&stmt);
                    continue;
                }
                _ => {}
            }
            if tokens.get(i) == Some(&":") && !token.starts_with('.') && !token.starts_with('@') {
                i += 1;
                f(&[token, ":"]);
                continue;
            }
        }
        let terminates_on_brace = {
            let has = |name: &str| cur.iter().any(|t| *t == name);
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
            f(&cur);
            cur.clear();
            continue;
        }
        cur.push(token);
        if token == ";" {
            f(&cur);
            cur.clear();
        }
    }
    if !cur.is_empty() {
        f(&cur);
    }
}

// ---- wmma ---------------------------------------------------------------
//
// `wmma` fragments are opaque, so they are given the register layout of the
// existing `mma.sync.m16n8k16` emulation: the A fragment lives in the first four
// registers, B in the first four, and a 16x16 accumulator is two 16x8 halves.
// Only the m16n16k16 f16-input shape is supported.

struct WmmaShape {
    row: bool,
    space: &'static str,
    f16: bool,
}

fn wmma_shape(parts: &[&str]) -> Option<WmmaShape> {
    if !parts.contains(&"m16n16k16") {
        return None;
    }
    let row = if parts.contains(&"row") {
        true
    } else if parts.contains(&"col") {
        false
    } else {
        return None;
    };
    let space = if parts.contains(&"shared") || parts.contains(&"shared::cta") { "shared" } else { "global" };
    let f16 = parts.last() == Some(&"f16");
    Some(WmmaShape { row, space, f16 })
}

/// Registers holding `g`, `g+8`, `2t`, `2t+1`, `2t+8`, `2t+9` for the calling lane.
const G: &str = "%__t32_1";
const G8: &str = "%__t32_2";
const C0: &str = "%__t32_3";
const C1: &str = "%__t32_4";
const C8: &str = "%__t32_5";
const C9: &str = "%__t32_6";
const IDX: &str = "%__t32_7";
const A32: &str = "%__t32_8";
const LO: &str = "%__t32_9";
const HI: &str = "%__t32_10";
const A64: &str = "%__t64_0";

fn wmma_prologue(ctx: &mut Ctx) -> Vec<String> {
    ctx.need32(11);
    ctx.need64(1);
    vec![
        format!("mov.u32 {LO} , %laneid ;"),
        format!("shr.u32 {G} , {LO} , 2 ;"),
        format!("add.u32 {G8} , {G} , 8 ;"),
        format!("and.b32 {HI} , {LO} , 3 ;"),
        format!("shl.b32 {C0} , {HI} , 1 ;"),
        format!("add.u32 {C1} , {C0} , 1 ;"),
        format!("add.u32 {C8} , {C0} , 8 ;"),
        format!("add.u32 {C9} , {C0} , 9 ;"),
    ]
}

/// Emits address arithmetic for element (`row`, `col`); returns the register to use.
fn wmma_address(
    ctx: &Ctx,
    sum: &str,
    lines: &mut Vec<String>,
    shape: &WmmaShape,
    base: &str,
    stride: &str,
    row: &str,
    col: &str,
    elem: u32,
) -> String {
    let (base, offset) = match base.split_once('+') {
        Some((reg, off)) => (reg.trim(), immediate(off).unwrap_or(0)),
        None => (base, 0),
    };
    if shape.row {
        lines.push(format!("mad.lo.u32 {IDX} , {row} , {stride} , {col} ;"));
    } else {
        lines.push(format!("mad.lo.u32 {IDX} , {col} , {stride} , {row} ;"));
    }
    if ctx.width_of(base).unwrap_or(64) == 64 {
        lines.push(format!("mul.wide.u32 {A64} , {IDX} , {elem} ;"));
        if offset != 0 {
            lines.push(format!("add.s64 {A64} , {A64} , {offset} ;"));
        }
        lines.push(format!("add.s64 {sum} , {base} , {A64} ;"));
        sum.to_owned()
    } else {
        lines.push(format!("mul.lo.u32 {A32} , {IDX} , {elem} ;"));
        if offset != 0 {
            lines.push(format!("add.u32 {A32} , {A32} , {offset} ;"));
        }
        lines.push(format!("add.u32 {A32} , {base} , {A32} ;"));
        A32.to_owned()
    }
}

fn parse_regs(text: &str) -> Vec<String> {
    text.trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .map(|r| r.trim().to_owned())
        .filter(|r| !r.is_empty())
        .collect()
}

fn rewrite_wmma(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let shape = wmma_shape(&parts)?;
    let kind = *parts.get(1)?;
    let mut lines = wmma_prologue(ctx);
    ctx.wmma_seq += 1;
    ctx.need64(1 + ctx.wmma_seq);
    let sum = format!("%__t64_{}", ctx.wmma_seq);
    let guard = inst.guard.clone();
    let space = shape.space;
    match kind {
        "load" => {
            let which = *parts.get(2)?;
            let regs = parse_regs(inst.operand(0));
            let base = inst.operand(1).trim().trim_start_matches('[').trim_end_matches(']').trim().to_owned();
            let stride = inst.operands.get(2).cloned().unwrap_or_else(|| "16".into());
            match which {
                "a" | "b" if shape.f16 && regs.len() == 8 => {
                    // (row, col) element coordinates of the two halves of each register.
                    let a_coords = [
                        ((G, C0), (G, C1)),
                        ((G8, C0), (G8, C1)),
                        ((G, C8), (G, C9)),
                        ((G8, C8), (G8, C9)),
                    ];
                    // B(k, n): registers hold k pairs for n = g (left half) then n = g+8.
                    let b_coords = [
                        ((C0, G), (C1, G)),
                        ((C8, G), (C9, G)),
                        ((C0, G8), (C1, G8)),
                        ((C8, G8), (C9, G8)),
                    ];
                    let coords = if which == "a" { a_coords } else { b_coords };
                    // The pair is contiguous when it varies along the fast memory axis.
                    let pair_varies_col = which == "a";
                    let contiguous = pair_varies_col == shape.row;
                    for (reg, ((r0, c0), (r1, c1))) in regs.iter().zip(coords) {
                        if contiguous {
                            let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r0, c0, 2);
                            lines.push(render(&guard, &format!("ld.{space}.u32"), &[reg.clone(), format!("[ {addr} ]")]));
                        } else {
                            let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r0, c0, 2);
                            lines.push(render(&guard, &format!("ld.{space}.u16"), &[LO.into(), format!("[ {addr} ]")]));
                            let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r1, c1, 2);
                            lines.push(render(&guard, &format!("ld.{space}.u16"), &[HI.into(), format!("[ {addr} ]")]));
                            lines.push(format!("mov.b32 {reg} , {{ {LO} , {HI} }} ;"));
                        }
                    }
                    for reg in &regs[4..] {
                        lines.push(format!("mov.b32 {reg} , 0 ;"));
                    }
                    Some(lines)
                }
                "c" => {
                    let coords = [(G, C0, G, C1), (G8, C0, G8, C1), (G, C8, G, C9), (G8, C8, G8, C9)];
                    if regs.len() == 8 {
                        // f32 accumulator: left half regs 0-3, right half regs 4-7.
                        let order = [(G, C0), (G, C1), (G8, C0), (G8, C1), (G, C8), (G, C9), (G8, C8), (G8, C9)];
                        for (reg, (r, c)) in regs.iter().zip(order) {
                            let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r, c, 4);
                            lines.push(render(&guard, &format!("ld.{space}.u32"), &[reg.clone(), format!("[ {addr} ]")]));
                        }
                        Some(lines)
                    } else if regs.len() == 4 {
                        // f16 accumulator: register k packs (row, c0) and (row, c1).
                        let order = [(G, C0, G, C1), (G8, C0, G8, C1), (G, C8, G, C9), (G8, C8, G8, C9)];
                        let _ = coords;
                        for (reg, (r0, c0, r1, c1)) in regs.iter().zip(order) {
                            let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r0, c0, 2);
                            lines.push(render(&guard, &format!("ld.{space}.u16"), &[LO.into(), format!("[ {addr} ]")]));
                            let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r1, c1, 2);
                            lines.push(render(&guard, &format!("ld.{space}.u16"), &[HI.into(), format!("[ {addr} ]")]));
                            lines.push(format!("mov.b32 {reg} , {{ {LO} , {HI} }} ;"));
                        }
                        Some(lines)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        "store" => {
            let base = inst.operand(0).trim().trim_start_matches('[').trim_end_matches(']').trim().to_owned();
            let regs = parse_regs(inst.operand(1));
            let stride = inst.operands.get(2).cloned().unwrap_or_else(|| "16".into());
            if regs.len() == 8 {
                let order = [(G, C0), (G, C1), (G8, C0), (G8, C1), (G, C8), (G, C9), (G8, C8), (G8, C9)];
                for (reg, (r, c)) in regs.iter().zip(order) {
                    let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r, c, 4);
                    lines.push(render(&guard, &format!("st.{space}.u32"), &[format!("[ {addr} ]"), reg.clone()]));
                }
                Some(lines)
            } else if regs.len() == 4 {
                let order = [(G, C0, G, C1), (G8, C0, G8, C1), (G, C8, G, C9), (G8, C8, G8, C9)];
                for (reg, (r0, c0, r1, c1)) in regs.iter().zip(order) {
                    let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r0, c0, 2);
                    lines.push(render(&guard, &format!("st.{space}.u16"), &[format!("[ {addr} ]"), reg.clone()]));
                    lines.push(format!("shr.u32 {HI} , {reg} , 16 ;"));
                    let addr = wmma_address(ctx, &sum, &mut lines, &shape, &base, &stride, r1, c1, 2);
                    lines.push(render(&guard, &format!("st.{space}.u16"), &[format!("[ {addr} ]"), HI.into()]));
                }
                Some(lines)
            } else {
                None
            }
        }
        "mma" => {
            // wmma.mma.sync.aligned.ALAYOUT.BLAYOUT.m16n16k16.DTYPE.CTYPE {d}, {a}, {b}, {c}
            let d = parse_regs(inst.operand(0));
            let a = parse_regs(inst.operand(1));
            let b = parse_regs(inst.operand(2));
            let c = parse_regs(inst.operand(3));
            if a.len() != 8 || b.len() != 8 || d.len() != c.len() || !matches!(d.len(), 4 | 8) {
                return None;
            }
            let mut out = Vec::new();
            let wide = d.len() == 8;
            let c_f16 = !parts.ends_with(&["f32"]) && !wide;
            let _ = c_f16;
            ctx.need32(8);
            for half in 0..2usize {
                let af = format!("{{ {} , {} , {} , {} }}", a[0], a[1], a[2], a[3]);
                let bf = format!("{{ {} , {} }}", b[2 * half], b[2 * half + 1]);
                if wide {
                    let cf = format!("{{ {} }}", c[4 * half..4 * half + 4].join(" , "));
                    let df = format!("{{ {} }}", d[4 * half..4 * half + 4].join(" , "));
                    out.push(format!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {df} , {af} , {bf} , {cf} ;"));
                } else {
                    // f16 accumulators: widen to f32 around the mma.
                    let t = |k: usize| format!("%__t32_{}", k);
                    let src = [&c[2 * half], &c[2 * half + 1]];
                    for (k, reg) in src.iter().enumerate() {
                        out.push(format!("and.b32 {} , {reg} , 65535 ;", t(2 * k)));
                        out.push(format!("shr.u32 {} , {reg} , 16 ;", t(2 * k + 1)));
                        out.push(format!("cvt.f32.f16 {} , {} ;", t(2 * k), t(2 * k)));
                        out.push(format!("cvt.f32.f16 {} , {} ;", t(2 * k + 1), t(2 * k + 1)));
                    }
                    let cf = format!("{{ {} , {} , {} , {} }}", t(0), t(1), t(2), t(3));
                    let df = format!("{{ {} , {} , {} , {} }}", t(4), t(5), t(6), t(7));
                    out.push(format!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {df} , {af} , {bf} , {cf} ;"));
                    out.push(format!("cvt.rn.f16x2.f32 {} , {} , {} ;", d[2 * half + 1], t(7), t(6)));
                    out.push(format!("cvt.rn.f16x2.f32 {} , {} , {} ;", d[2 * half], t(5), t(4)));
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Bytes of a normalized `.shared ... name [ N ] ;` declaration line (0 for any other line).
fn static_shared_bytes(line: &str) -> u64 {
    if !line.trim_start().starts_with(".shared") {
        return 0;
    }
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.first() != Some(&".shared") || line.contains("/*dyn*/") {
        return 0;
    }
    let elem = tokens.iter().find_map(|t| match *t {
        ".b8" | ".u8" | ".s8" => Some(1),
        ".b16" | ".u16" | ".s16" | ".f16" => Some(2),
        ".b32" | ".u32" | ".s32" | ".f32" => Some(4),
        ".b64" | ".u64" | ".s64" | ".f64" => Some(8),
        _ => None,
    });
    let count = tokens
        .iter()
        .position(|&t| t == "[")
        .and_then(|i| tokens.get(i + 1))
        .and_then(|n| n.parse::<u64>().ok());
    match (elem, count) {
        (Some(e), Some(n)) => e * n,
        (Some(e), None) => e,
        _ => 0,
    }
}

/// Splits `ld.global[.nc].u16 %rs , [ %rd+off ] ;` into (op, destination, base, offset).
fn parse_u16_load(line: &str) -> Option<(&str, &str, &str, i64)> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() != 7 || !(tokens[0] == "ld.global.u16" || tokens[0] == "ld.global.nc.u16") {
        return None;
    }
    if tokens[2] != "," || tokens[3] != "[" || tokens[5] != "]" || tokens[6] != ";" {
        return None;
    }
    let (base, offset) = match tokens[4].split_once('+') {
        Some((base, offset)) => (base, offset.parse::<i64>().ok()?),
        None => (tokens[4], 0),
    };
    base.starts_with("%rd").then_some((tokens[0], tokens[1], base, offset))
}

/// Rewrites the two adjacent 16-bit loads that nvcc emits for unaligned 32-bit reads
/// (`ld.u16 lo; ld.u16 hi; mov.b32 d, {lo, hi}`, as in Q8_0 blocks) into two aligned
/// 32-bit loads and a funnel shift. The second dword is only fetched when the pair
/// straddles dwords, so no byte outside the original two loads' dwords is read.
fn merge_u16_load_pairs(out: &mut Vec<String>, open: usize, ctx: &mut Ctx) {
    let mut i = open + 1;
    while i < out.len() {
        let Some((op, dst1, base, off1)) = parse_u16_load(&out[i]) else {
            i += 1;
            continue;
        };
        let mut matched = None;
        for j in i + 1..(i + 9).min(out.len()) {
            let Some((_, dst2, base2, off2)) = parse_u16_load(&out[j]) else {
                // The base register must not change between the loads.
                if out[j].split_whitespace().nth(1) == Some(base) {
                    break;
                }
                continue;
            };
            if base2 == base && (off2 == off1 + 2 || off2 == off1 - 2) {
                matched = Some((j, dst2, off2));
                break;
            }
            if out[j].split_whitespace().nth(1) == Some(base) {
                break;
            }
        }
        let Some((j, dst2, off2)) = matched else {
            i += 1;
            continue;
        };
        let (lo, hi, low_offset) = if off2 > off1 { (dst1, dst2, off1) } else { (dst2, dst1, off2) };
        let pack = format!("mov.b32 ");
        let want = format!(" , {{ {lo} , {hi} }} ;");
        let Some(k) = (j + 1..(j + 17).min(out.len())).find(|&k| out[k].starts_with(&pack) && out[k].ends_with(&want)) else {
            i += 1;
            continue;
        };
        let used = |reg: &str| out[open..].iter().map(|l| l.split_whitespace().filter(|t| *t == reg).count()).sum::<usize>();
        let straight = out[i..=k].iter().all(|l| !l.ends_with(" :") && !l.contains(" bra ") && !l.starts_with('@') && !l.starts_with("ret") && !l.starts_with("bar")
            // The second load is hoisted to the first one: nothing in between may write memory
            // or order it (a store/atomic to the same dword would otherwise be read stale).
            && !["st.", "atom", "red.", "membar", "fence", "call", "cp.async", "multimem"].iter().any(|p| l.starts_with(p)));
        if !straight || used(lo) != 2 || used(hi) != 2 || low_offset % 2 != 0 {
            i += 1;
            continue;
        }
        let result = out[k].split_whitespace().nth(1).unwrap().to_owned();
        let t64 = |ctx: &mut Ctx| {
            ctx.t64 += 1;
            format!("%__t64_{}", ctx.t64 - 1)
        };
        let t32 = |ctx: &mut Ctx| {
            ctx.t32 += 1;
            format!("%__t32_{}", ctx.t32 - 1)
        };
        let (a, aligned, bit, next, w0, w1, sh, shifted, merged) =
            (t64(ctx), t64(ctx), t64(ctx), t64(ctx), t32(ctx), t32(ctx), t32(ctx), t32(ctx), t32(ctx));
        let wide = op.replace("u16", "u32");
        let mut seq = vec![
            if low_offset == 0 {
                format!("mov.b64 {a} , {base} ;")
            } else {
                format!("add.s64 {a} , {base} , {low_offset} ;")
            },
            format!("and.b64 {aligned} , {a} , 0xfffffffffffffffc ;"),
            format!("and.b64 {bit} , {a} , 2 ;"),
            format!("{wide} {w0} , [ {aligned} ] ;"),
            format!("shl.b64 {next} , {bit} , 1 ;"),
            format!("add.s64 {next} , {aligned} , {next} ;"),
            format!("{wide} {w1} , [ {next} ] ;"),
            format!("cvt.u32.u64 {sh} , {bit} ;"),
            format!("shl.b32 {shifted} , {sh} , 3 ;"),
            format!("shf.r.wrap.b32 {merged} , {w0} , {w1} , {shifted} ;"),
        ];
        out[k] = format!("mov.b32 {result} , {merged} ;");
        out.remove(j);
        let count = seq.len();
        out.splice(i..=i, seq.drain(..));
        i += count;
    }
}

#[cfg(test)]
mod load_pair_tests {
    use super::*;

    #[test]
    fn merges_adjacent_u16_loads_into_aligned_dwords() {
        let mut out: Vec<String> = [
            "{",
            "ld.global.nc.u16 %rs4 , [ %rd31 ] ;",
            "ld.global.nc.u16 %rs5 , [ %rd31+2 ] ;",
            "mov.b32 %r77 , { %rs4 , %rs5 } ;",
            "ret ;",
            "}",
        ]
        .map(String::from)
        .to_vec();
        let mut ctx = Ctx::default();
        merge_u16_load_pairs(&mut out, 0, &mut ctx);
        let text = out.join("\n");
        assert!(!text.contains(".u16"), "{text}");
        assert!(text.contains("ld.global.nc.u32"), "{text}");
        assert!(text.contains("shf.r.wrap.b32"), "{text}");
        assert!(text.contains("mov.b32 %r77 , %__t32_"), "{text}");
        assert_eq!(ctx.t64, 4);
    }

    #[test]
    fn leaves_pairs_alone_when_a_half_is_reused() {
        let mut out: Vec<String> = [
            "{",
            "ld.global.nc.u16 %rs4 , [ %rd31 ] ;",
            "ld.global.nc.u16 %rs5 , [ %rd31+2 ] ;",
            "mov.b32 %r77 , { %rs4 , %rs5 } ;",
            "cvt.f32.f16 %f1 , %rs4 ;",
            "}",
        ]
        .map(String::from)
        .to_vec();
        let before = out.clone();
        merge_u16_load_pairs(&mut out, 0, &mut Ctx::default());
        assert_eq!(out, before);
    }

    #[test]
    fn leaves_pairs_alone_when_a_store_sits_between_the_loads() {
        let mut out: Vec<String> = [
            "{",
            "ld.global.u16 %rs4 , [ %rd31 ] ;",
            "st.global.u32 [ %rd31 ] , %r9 ;",
            "ld.global.u16 %rs5 , [ %rd31+2 ] ;",
            "mov.b32 %r77 , { %rs4 , %rs5 } ;",
            "}",
        ]
        .map(String::from)
        .to_vec();
        let before = out.clone();
        merge_u16_load_pairs(&mut out, 0, &mut Ctx::default());
        assert_eq!(out, before);
    }

    /// Runs the kernel `out[t] = pack(ld.u16 [p+off], ld.u16 [p+off+2])` (or reversed order) with and
    /// without the rewrite on the SPIR-V simulator at every 2-byte alignment.
    #[test]
    fn merged_loads_equal_the_original_on_the_simulator() {
        use crate::ptx::sim::{Gpu, Launch, Sim};
        for (nc, reversed) in [(false, false), (true, false), (false, true), (true, true)] {
            let ld = if nc { "ld.global.nc.u16" } else { "ld.global.u16" };
            for off in (0..8i64).step_by(2) {
                let (o1, o2) = if reversed { (off + 2, off) } else { (off, off + 2) };
                let (r1, r2) = if reversed { ("%rs5", "%rs4") } else { ("%rs4", "%rs5") };
                let body: Vec<String> = [
                    "ld.param.u64 %rd1 , [ k_param_0 ] ;".to_string(),
                    "ld.param.u64 %rd2 , [ k_param_1 ] ;".to_string(),
                    "cvta.to.global.u64 %rd3 , %rd1 ;".to_string(),
                    "cvta.to.global.u64 %rd4 , %rd2 ;".to_string(),
                    format!("{ld} {r1} , [ %rd3+{o1} ] ;"),
                    format!("{ld} {r2} , [ %rd3+{o2} ] ;"),
                    "mov.b32 %r7 , { %rs4 , %rs5 } ;".to_string(),
                    "st.global.u32 [ %rd4 ] , %r7 ;".to_string(),
                    "ret ;".to_string(),
                ]
                .to_vec();
                let build = |lines: &[String], decls: &[String]| {
                    format!(
                        ".version 8.8\n.target sm_80\n.address_size 64\n.visible .entry k(\n.param .u64 k_param_0,\n.param .u64 k_param_1\n)\n{{\n.reg .b16 %rs<8>;\n.reg .b32 %r<8>;\n.reg .b64 %rd<8>;\n{}\n{}\n}}\n",
                        decls.join("\n"),
                        lines.join("\n")
                    )
                };
                // Base is 4-aligned (allocation) so `off` covers both dword alignments.
                let data: Vec<u8> = (0..32u8).map(|b| b.wrapping_mul(37).wrapping_add(11)).collect();
                let run = |src: &str| {
                    let t = crate::ptx::translate(src).unwrap();
                    let module = rspirv::dr::load_words(&t.words).unwrap();
                    let mut gpu = Gpu::default();
                    let inp = gpu.upload(&data);
                    let out = gpu.alloc(4);
                    Sim::new(&module).run(&mut gpu, &Launch::new([1, 1, 1], [1, 1, 1], vec![inp, out])).unwrap();
                    gpu.download(out, 4)
                };
                let mut merged = body.clone();
                let mut ctx = Ctx::default();
                let mut framed = vec!["{".to_string()];
                framed.extend(merged.drain(..));
                merge_u16_load_pairs(&mut framed, 0, &mut ctx);
                let decls = ctx.take_decls();
                framed.remove(0);
                assert!(!framed.iter().any(|l| l.contains(".u16")), "not merged: {framed:?}");
                let a = run(&build(&body, &[]));
                let b = run(&build(&framed, &decls));
                let want = [data[off as usize], data[off as usize + 1], data[off as usize + 2], data[off as usize + 3]];
                assert_eq!(a, want, "reference nc={nc} rev={reversed} off={off}");
                assert_eq!(b, want, "merged nc={nc} rev={reversed} off={off}");
            }
        }
    }
}

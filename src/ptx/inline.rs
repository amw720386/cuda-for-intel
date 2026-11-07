//! Inlines PTX `.func` definitions into their call sites.
//!
//! SPIR-V for the OpenCL environment could express real function calls, but the
//! translator's register model is per-entry, so the simplest correct lowering
//! is to splice each callee body into the caller: formal `ld.param` become
//! register moves from the actual `st.param` operands, `st.param` of the return
//! value become moves into the registers the caller later reads with
//! `ld.param`, and `ret` becomes a branch to the end of the inlined region.
//! Callee registers and labels are renamed per call site.

use super::Error;
use super::normalize::Stmt;
use std::collections::{HashMap, HashSet};

const MAX_DEPTH: usize = 12;
const MAX_STATEMENTS: usize = 12_000_000;

struct Func {
    returns: Vec<String>,
    params: Vec<String>,
    body: Vec<Stmt>,
}

pub(super) fn expand(statements: Vec<Stmt>) -> Result<Vec<Stmt>, Error> {
    if !statements.iter().any(is_func_statement) {
        // Fast path; also keeps modules without `.func` byte-for-byte identical.
        let calls = statements.iter().any(|s| s.first().is_some_and(|t| t.starts_with("call")));
        if !calls {
            return Ok(statements);
        }
    }
    let mut funcs: HashMap<String, Func> = HashMap::new();
    let mut rest: Vec<Stmt> = Vec::with_capacity(statements.len());
    let mut iter = statements.into_iter();
    while let Some(stmt) = iter.next() {
        if !is_func_statement(&stmt) {
            rest.push(stmt);
            continue;
        }
        if stmt.last().is_some_and(|t| t == ";") {
            // Declaration only (`.extern .func`, forward declaration).
            continue;
        }
        // Definition: header ends with `{`; take the body through the matching `}`.
        let mut body = Vec::new();
        let mut depth = 1i32;
        for inner in iter.by_ref() {
            match inner.first().map(String::as_str) {
                Some("{") if inner.len() == 1 => depth += 1,
                Some("}") if inner.len() == 1 => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            body.push(inner);
        }
        if let Some((name, func)) = parse_func(&stmt, body) {
            funcs.insert(name, func);
        }
    }
    let mut counter = 0usize;
    let mut budget = MAX_STATEMENTS;
    expand_stmts(&rest, &funcs, 0, &mut counter, &mut budget)
}

fn is_func_statement(stmt: &Stmt) -> bool {
    stmt.iter().take(4).any(|t| t == ".func") && !stmt.iter().any(|t| t == ".entry")
}

fn parse_func(header: &Stmt, body: Vec<Stmt>) -> Option<(String, Func)> {
    let at = header.iter().position(|t| t == ".func")?;
    let mut i = at + 1;
    let mut returns = Vec::new();
    if header.get(i).is_some_and(|t| t == "(") {
        let (names, next) = param_names(header, i)?;
        returns = names;
        i = next;
    }
    let name = header.get(i)?.clone();
    i += 1;
    let params = if header.get(i).is_some_and(|t| t == "(") {
        param_names(header, i)?.0
    } else {
        Vec::new()
    };
    Some((name, Func { returns, params, body }))
}

/// Names of the `.param` declarations in the parenthesised list at `open`.
fn param_names(tokens: &[String], open: usize) -> Option<(Vec<String>, usize)> {
    let mut names = Vec::new();
    let mut i = open + 1;
    let mut last: Option<String> = None;
    let mut bracket = 0;
    while i < tokens.len() {
        match tokens[i].as_str() {
            ")" => {
                if let Some(name) = last.take() {
                    names.push(name);
                }
                return Some((names, i + 1));
            }
            "," => {
                if let Some(name) = last.take() {
                    names.push(name);
                }
            }
            "[" => bracket += 1,
            "]" => bracket -= 1,
            token if bracket == 0 && !token.starts_with('.') => last = Some(token.to_owned()),
            _ => {}
        }
        i += 1;
    }
    None
}

#[derive(Default)]
struct Frame {
    /// `param name -> (byte offset -> operand)`.
    slots: HashMap<String, HashMap<u64, String>>,
}

fn expand_stmts(
    stmts: &[Stmt],
    funcs: &HashMap<String, Func>,
    depth: usize,
    counter: &mut usize,
    budget: &mut usize,
) -> Result<Vec<Stmt>, Error> {
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    let mut frame = Frame::default();
    let mut i = 0;
    while i < stmts.len() {
        let stmt = &stmts[i];
        i += 1;
        let (guard, rest) = split_guard(stmt);
        let op = rest.first().map(String::as_str).unwrap_or("");
        if op == ".param" && stmt.last().is_some_and(|t| t == ";") && depth_ok_param(stmt) {
            if let Some(name) = declared_param_name(stmt) {
                frame.slots.insert(name, HashMap::new());
                continue;
            }
        }
        if op.starts_with("st.param") {
            if let Some((name, stores)) = parse_param_store(rest) {
                if let Some(slots) = frame.slots.get_mut(&name) {
                    slots.extend(stores);
                    continue;
                }
            }
        }
        if op == "call" || op.starts_with("call.") {
            let Some(call) = parse_call(rest) else {
                out.push(stmt.clone());
                continue;
            };
            let Some(func) = funcs.get(&call.target) else {
                if let Some(special) = special_call(&call, &mut i, stmts, &mut frame, &mut out) {
                    special?;
                    continue;
                }
                out.push(stmt.clone());
                continue;
            };
            if depth >= MAX_DEPTH || func.body.iter().any(unsupported_in_callee) {
                out.push(stmt.clone());
                continue;
            }
            // Return-value consumers follow the call.
            let mut retmap: HashMap<u64, String> = HashMap::new();
            while i < stmts.len() {
                let (g, next) = split_guard(&stmts[i]);
                if !g.is_empty() || !next.first().is_some_and(|t| t.starts_with("ld.param")) {
                    break;
                }
                let Some((name, loads)) = parse_param_load(next) else {
                    break;
                };
                if !call.returns.contains(&name) {
                    break;
                }
                retmap.extend(loads);
                i += 1;
            }
            *counter += 1;
            let id = *counter;
            let body = expand_stmts(&func.body, funcs, depth + 1, counter, budget)?;
            let inlined = inline_body(func, &call, &frame, &retmap, &body, id)?;
            if *budget < inlined.len() {
                return Err(Error::new("PTX function inlining is too large"));
            }
            *budget -= inlined.len();
            if guard.is_empty() {
                out.extend(inlined);
            } else {
                let skip = format!("$L__inl_skip{id}");
                let mut inverted: Vec<String> = Vec::new();
                inverted.push("@".into());
                if guard.get(1).is_some_and(|t| t == "!") {
                    inverted.push(guard[2].clone());
                } else {
                    inverted.push("!".into());
                    inverted.push(guard[1].clone());
                }
                inverted.extend(["bra.uni".into(), skip.clone(), ";".into()]);
                out.push(inverted);
                out.extend(inlined);
                out.push(vec![skip, ":".into()]);
            }
            continue;
        }
        out.push(stmt.clone());
    }
    Ok(out)
}

fn depth_ok_param(stmt: &Stmt) -> bool {
    stmt.len() >= 3
}

fn declared_param_name(stmt: &Stmt) -> Option<String> {
    // `.param .b32 name ;` or `.param .align 8 .b8 name [ 16 ] ;`
    let mut last = None;
    for token in &stmt[1..stmt.len() - 1] {
        if token == "[" {
            break;
        }
        if !token.starts_with('.') && !token.chars().all(|c| c.is_ascii_digit()) {
            last = Some(token.clone());
        }
    }
    last
}

fn split_guard(stmt: &Stmt) -> (Vec<String>, &[String]) {
    if stmt.first().is_some_and(|t| t == "@") {
        let n = if stmt.get(1).is_some_and(|t| t == "!") { 3 } else { 2 };
        if stmt.len() > n {
            return (stmt[..n].to_vec(), &stmt[n..]);
        }
    }
    (Vec::new(), &stmt[..])
}

fn element_bytes(ty: &str) -> u64 {
    match ty {
        "b8" | "u8" | "s8" => 1,
        "b16" | "u16" | "s16" | "f16" | "bf16" => 2,
        "b64" | "u64" | "s64" | "f64" => 8,
        _ => 4,
    }
}

fn address_parts(tokens: &[String]) -> Option<(String, u64, usize)> {
    // `[ name+off ]` or `[ name ]`; returns (name, offset, tokens consumed).
    if tokens.first().map(String::as_str) != Some("[") {
        return None;
    }
    let close = tokens.iter().position(|t| t == "]")?;
    let inner = tokens[1..close].concat();
    let (name, offset) = match inner.split_once('+') {
        Some((name, off)) => (name.to_owned(), off.parse::<u64>().ok()?),
        None => (inner, 0),
    };
    Some((name, offset, close + 1))
}

fn operand_list(tokens: &[String]) -> Vec<String> {
    // Registers inside `{ a , b }` or a single operand.
    let mut regs = Vec::new();
    let mut current = Vec::new();
    for token in tokens {
        match token.as_str() {
            "{" | "}" => {}
            "," => regs.push(std::mem::take(&mut current).concat()),
            other => current.push(other.to_owned()),
        }
    }
    if !current.is_empty() {
        regs.push(current.concat());
    }
    regs
}

/// `st.param[.vN].T [ p+off ] , src ;`
fn parse_param_store(rest: &[String]) -> Option<(String, Vec<(u64, String)>)> {
    let op = rest.first()?;
    let (name, offset, used) = address_parts(&rest[1..])?;
    let after = &rest[1 + used..];
    if after.first().map(String::as_str) != Some(",") {
        return None;
    }
    let values = operand_list(&after[1..after.len().checked_sub(1)?]);
    let ty = op.rsplit('.').next()?;
    let size = element_bytes(ty);
    Some((
        name,
        values
            .into_iter()
            .enumerate()
            .map(|(k, v)| (offset + size * k as u64, v))
            .collect(),
    ))
}

/// `ld.param[.vN].T dst , [ p+off ] ;`
fn parse_param_load(rest: &[String]) -> Option<(String, Vec<(u64, String)>)> {
    let op = rest.first()?;
    let comma = rest.iter().position(|t| t == ",")?;
    // The destination may itself contain commas inside braces.
    let mut depth = 0;
    let mut split = None;
    for (index, token) in rest.iter().enumerate().skip(1) {
        match token.as_str() {
            "{" => depth += 1,
            "}" => depth -= 1,
            "," if depth == 0 => {
                split = Some(index);
                break;
            }
            _ => {}
        }
    }
    let split = split.unwrap_or(comma);
    let dsts = operand_list(&rest[1..split]);
    let (name, offset, _) = address_parts(&rest[split + 1..])?;
    let ty = op.rsplit('.').next()?;
    let size = element_bytes(ty);
    Some((
        name,
        dsts.into_iter()
            .enumerate()
            .map(|(k, v)| (offset + size * k as u64, v))
            .collect(),
    ))
}

struct Call {
    returns: Vec<String>,
    target: String,
    args: Vec<String>,
}

/// `call[.uni] [(rets) ,] target [, (args)] ;`
fn parse_call(rest: &[String]) -> Option<Call> {
    let mut i = 1;
    let mut returns = Vec::new();
    if rest.get(i).is_some_and(|t| t == "(") {
        let close = rest.iter().position(|t| t == ")")?;
        returns = rest[i + 1..close].iter().filter(|t| *t != ",").cloned().collect();
        i = close + 1;
        if rest.get(i).is_some_and(|t| t == ",") {
            i += 1;
        }
    }
    let target = rest.get(i)?.clone();
    if target.starts_with('%') || target == "(" {
        return None; // indirect call
    }
    i += 1;
    let mut args = Vec::new();
    if rest.get(i).is_some_and(|t| t == ",") && rest.get(i + 1).is_some_and(|t| t == "(") {
        let close = rest[i + 1..].iter().position(|t| t == ")")? + i + 1;
        args = rest[i + 2..close].iter().filter(|t| *t != ",").cloned().collect();
    }
    Some(Call { returns, target, args })
}

/// Library calls that nvcc emits for assertions and printing.
fn special_call(
    call: &Call,
    cursor: &mut usize,
    stmts: &[Stmt],
    frame: &mut Frame,
    out: &mut Vec<Stmt>,
) -> Option<Result<(), Error>> {
    let _ = (frame, &out);
    match call.target.as_str() {
        "__assertfail" | "free" => Some(Ok(())),
        "vprintf" => {
            // printf is not supported on this target; report zero characters.
            while *cursor < stmts.len() {
                let (g, next) = split_guard(&stmts[*cursor]);
                if !g.is_empty() || !next.first().is_some_and(|t| t.starts_with("ld.param")) {
                    break;
                }
                let Some((name, loads)) = parse_param_load(next) else { break };
                if !call.returns.contains(&name) {
                    break;
                }
                for (_, dst) in loads {
                    out.push(vec!["mov.b32".into(), dst, ",".into(), "0".into(), ";".into()]);
                }
                *cursor += 1;
            }
            Some(Ok(()))
        }
        _ => None,
    }
}

fn unsupported_in_callee(stmt: &Stmt) -> bool {
    // `.local` frames (e.g. libdevice slow paths with a `__local_depot`) are supported: the
    // inliner renames the depot per call site and hoists its declaration.
    matches!(stmt.first().map(String::as_str), Some(".shared" | ".global" | ".const"))
}

const SPECIAL_PREFIXES: &[&str] = &[
    "tid", "ntid", "ctaid", "nctaid", "laneid", "warpid", "nwarpid", "smid", "nsmid", "lanemask_",
    "clock", "globaltimer", "gridid", "WARP_SZ", "pm", "envreg", "cluster_", "is_explicit_cluster",
    "total_smem_size", "dynamic_smem_size", "aggr_smem_size",
];

fn rename_token(token: &str, suffix: &str, labels: &HashSet<String>) -> String {
    if labels.contains(token) {
        return format!("{token}{}", suffix.trim_end_matches('_'));
    }
    if !token.contains('%') {
        return token.to_owned();
    }
    let mut out = String::with_capacity(token.len() + suffix.len());
    let bytes: Vec<char> = token.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '%' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || matches!(bytes[end], '_' | '$')) {
                end += 1;
            }
            let name: String = bytes[start..end].iter().collect();
            out.push('%');
            // Prefix (not suffix) so `%r<5>` ranges and `%r1` members stay consistent.
            if !SPECIAL_PREFIXES.iter().any(|p| name.starts_with(p)) {
                out.push_str(suffix);
            }
            out.push_str(&name);
            i = end;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

fn mov_for(ty: &str) -> &'static str {
    if element_bytes(ty) == 8 { "mov.b64" } else { "mov.b32" }
}

fn inline_body(
    func: &Func,
    call: &Call,
    frame: &Frame,
    retmap: &HashMap<u64, String>,
    body: &[Stmt],
    id: usize,
) -> Result<Vec<Stmt>, Error> {
    let suffix = format!("__c{id}_");
    let end_label = format!("$L__inl_end{id}");
    let mut labels: HashSet<String> = body
        .iter()
        .filter(|s| s.len() == 2 && s[1] == ":")
        .map(|s| s[0].clone())
        .collect();
    // Callee `.local` arrays are renamed per call site and declared in the caller.
    let mut hoisted: Vec<Stmt> = Vec::new();
    for stmt in body {
        if stmt.first().is_some_and(|t| t == ".local") {
            if let Some(at) = stmt.iter().position(|t| t == "[") {
                if let Some(name) = at.checked_sub(1).and_then(|i| stmt.get(i)) {
                    labels.insert(name.clone());
                }
            }
        }
    }
    let mut out: Vec<Stmt> = vec![vec!["{".into()]];
    for stmt in body {
        if stmt.first().is_some_and(|t| t == ".local") {
            hoisted.push(stmt.iter().map(|t| rename_token(t, &suffix, &labels)).collect());
            continue;
        }
        let (guard, rest) = split_guard(stmt);
        let guard: Vec<String> = guard.iter().map(|t| rename_token(t, &suffix, &labels)).collect();
        let op = rest.first().map(String::as_str).unwrap_or("");
        if op.starts_with("ld.param") {
            if let Some((name, loads)) = parse_param_load(rest) {
                if let Some(position) = func.params.iter().position(|p| *p == name) {
                    let Some(actual) = call.args.get(position) else {
                        return Err(Error::new("PTX call has too few arguments"));
                    };
                    let slots = frame.slots.get(actual);
                    let ty = op.rsplit('.').next().unwrap_or("b32");
                    for (offset, dst) in loads {
                        let value = slots
                            .and_then(|s| s.get(&offset))
                            .cloned()
                            .ok_or_else(|| Error::new("PTX call argument was never stored"))?;
                        let mut line = guard.clone();
                        line.extend([
                            mov_for(ty).to_owned(),
                            rename_token(&dst, &suffix, &labels),
                            ",".into(),
                            value,
                            ";".into(),
                        ]);
                        out.push(line);
                    }
                    continue;
                }
            }
        }
        if op.starts_with("st.param") {
            if let Some((name, stores)) = parse_param_store(rest) {
                if func.returns.contains(&name) {
                    let ty = op.rsplit('.').next().unwrap_or("b32");
                    for (offset, value) in stores {
                        if let Some(dst) = retmap.get(&offset) {
                            let mut line = guard.clone();
                            line.extend([
                                mov_for(ty).to_owned(),
                                dst.clone(),
                                ",".into(),
                                rename_token(&value, &suffix, &labels),
                                ";".into(),
                            ]);
                            out.push(line);
                        }
                    }
                    continue;
                }
            }
        }
        if op == "ret" || op == "ret.uni" {
            let mut line = guard.clone();
            line.extend(["bra.uni".into(), end_label.clone(), ";".into()]);
            out.push(line);
            continue;
        }
        out.push(stmt.iter().map(|t| rename_token(t, &suffix, &labels)).collect());
    }
    out.push(vec![end_label, ":".into()]);
    out.push(vec!["}".into()]);
    if hoisted.is_empty() {
        return Ok(out);
    }
    hoisted.extend(out);
    Ok(hoisted)
}

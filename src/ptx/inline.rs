//! Inlines PTX `.func` definitions into their call sites.
//!
//! SPIR-V for the OpenCL environment could express real function calls, but the
//! translator's register model is per-entry, so the simplest correct lowering
//! is to splice each callee body into the caller: formal `ld.param` become
//! register moves from the actual `st.param` operands, `st.param` of the return
//! value become moves into the registers the caller later reads with
//! `ld.param`, and `ret` becomes a branch to the end of the inlined region.
//! Callee registers and labels are renamed per call site.
//!
//! `.reg` parameters and return values (`.func (.reg .u32 r) f (.reg .u32 a)`) become
//! per-call-site registers: moves from the actual arguments before the body, moves to the
//! actual return registers after it. Function-scope `.shared`, `.global` and `.const`
//! variables are static, so they move to module scope once per function.
//!
//! Function pointers follow the design of ZLUDA's `resolve_function_pointers` pass, adapted
//! to inlining: `mov.u64 %rd, f` (and `f` in a module-scope initializer) yields a small
//! non-zero token for `f`, and `call (r), %rd, (a), proto;` becomes a switch over the
//! address-taken functions whose signature matches the prototype (or the `.calltargets`
//! list), each arm an inlined call. Calls that cannot be inlined keep their text with a
//! `call.cfi_<reason>` opcode, which the parser reports as a precise unsupported error.

use super::Error;
use super::normalize::Stmt;
use std::collections::{HashMap, HashSet};

const MAX_DEPTH: usize = 12;
const MAX_STATEMENTS: usize = 12_000_000;

struct Func {
    returns: Vec<String>,
    params: Vec<String>,
    /// Register type (`.u32`, `.pred`) of each return value passed in a `.reg`, else `None`.
    return_regs: Vec<Option<String>>,
    /// Register type of each parameter passed in a `.reg`, else `None` (a `.param`).
    param_regs: Vec<Option<String>>,
    body: Vec<Stmt>,
    /// Function-pointer token (`mov.u64 %rd, f` yields it).
    token: u64,
}

/// A call prototype: `.reg` flags of the return values and parameters.
#[derive(Clone, PartialEq, Eq)]
struct Signature {
    returns: Vec<bool>,
    params: Vec<bool>,
}

impl Func {
    fn signature(&self) -> Signature {
        Signature {
            returns: self.return_regs.iter().map(Option::is_some).collect(),
            params: self.param_regs.iter().map(Option::is_some).collect(),
        }
    }
}

/// Module-wide facts the expansion consults.
struct Module {
    funcs: HashMap<String, Func>,
    /// Functions declared (`.extern .func f (...) ;` or a prototype) but not defined here.
    declared: HashSet<String>,
    /// Defined functions whose address is taken, in definition order.
    address_taken: Vec<String>,
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
    let mut declared: HashSet<String> = HashSet::new();
    let mut rest: Vec<Stmt> = Vec::with_capacity(statements.len());
    let mut iter = statements.into_iter();
    while let Some(stmt) = iter.next() {
        if !is_func_statement(&stmt) {
            rest.push(stmt);
            continue;
        }
        if stmt.last().is_some_and(|t| t == ";") {
            // Declaration only (`.extern .func`, forward declaration).
            if let Some((name, _)) = parse_func(&stmt, Vec::new()) {
                declared.insert(name);
            }
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
        let token = funcs.len() as u64 + 1;
        if let Some((name, mut func)) = parse_func(&stmt, body) {
            func.token = token;
            funcs.insert(name, func);
        }
    }
    for name in funcs.keys() {
        declared.remove(name);
    }
    let statics = hoist_statics(&mut funcs);
    if !statics.is_empty() {
        // Module scope, before the first kernel (or at the end when there is none).
        let at = rest.iter().position(|s| s.iter().any(|t| t == ".entry")).unwrap_or(rest.len());
        rest.splice(at..at, statics);
    }
    let mut address_taken: Vec<(u64, String)> = Vec::new();
    {
        let mut seen = |stmts: &[Stmt]| {
            for stmt in stmts {
                for name in taken_functions(stmt, &funcs) {
                    let token = funcs[name].token;
                    if !address_taken.iter().any(|(_, n)| n == name) {
                        address_taken.push((token, name.to_owned()));
                    }
                }
            }
        };
        seen(&rest);
        for func in funcs.values() {
            seen(&func.body);
        }
    }
    address_taken.sort();
    let module = Module { funcs, declared, address_taken: address_taken.into_iter().map(|(_, n)| n).collect() };
    let mut counter = 0usize;
    let mut budget = MAX_STATEMENTS;
    expand_stmts(rest, &module, 0, &mut counter, &mut budget)
}

/// Names of defined functions used as values in `stmt` (`mov.u64 %rd, f`, `.global .u64 t[] = {f}`).
fn taken_functions<'m>(stmt: &Stmt, funcs: &'m HashMap<String, Func>) -> Vec<&'m str> {
    let (_, rest) = split_guard(stmt);
    let op = rest.first().map_or("", String::as_str);
    let value_position = op.starts_with("mov.")
        || op.starts_with("cvta.")
        || op.starts_with("st.")
        || op.starts_with("selp.")
        || (matches!(op, ".global" | ".const" | ".visible" | ".weak") && stmt.iter().any(|t| t == "="));
    if !value_position {
        return Vec::new();
    }
    let start = if op.starts_with('.') { stmt.iter().position(|t| t == "=").unwrap_or(stmt.len()) } else { 0 };
    stmt[start..]
        .iter()
        .filter_map(|t| funcs.get_key_value(t.as_str()).map(|(name, _)| name.as_str()))
        .collect()
}

/// Replaces function names in value positions with their pointer tokens.
fn replace_function_values(stmt: &Stmt, module: &Module) -> Option<Stmt> {
    if taken_functions(stmt, &module.funcs).is_empty() {
        return None;
    }
    let (_, rest) = split_guard(stmt);
    let op = rest.first().map_or("", String::as_str);
    let start = if op.starts_with('.') { stmt.iter().position(|t| t == "=").unwrap_or(stmt.len()) } else { 0 };
    Some(
        stmt.iter()
            .enumerate()
            .map(|(k, t)| match module.funcs.get(t.as_str()) {
                Some(func) if k >= start => func.token.to_string(),
                _ => t.clone(),
            })
            .collect(),
    )
}

/// Moves function-scope `.shared` / `.global` / `.const` declarations to module scope, renamed
/// `<function>__<name>`; every call site of the function shares the one variable, as in PTX.
fn hoist_statics(funcs: &mut HashMap<String, Func>) -> Vec<Stmt> {
    let mut names: Vec<&String> = funcs.keys().collect();
    names.sort();
    let names: Vec<String> = names.into_iter().cloned().collect();
    let mut out = Vec::new();
    for fname in names {
        let func = funcs.get_mut(&fname).unwrap();
        if !func.body.iter().any(unsupported_in_callee) {
            continue;
        }
        let mut renames: HashMap<String, String> = HashMap::new();
        let mut kept = Vec::with_capacity(func.body.len());
        for stmt in std::mem::take(&mut func.body) {
            if !unsupported_in_callee(&stmt) {
                kept.push(stmt);
                continue;
            }
            let end = stmt.iter().position(|t| matches!(t.as_str(), "[" | ";" | "=" | "<")).unwrap_or(stmt.len());
            let Some(at) = stmt[..end].iter().rposition(|t| !t.starts_with('.') && !t.chars().all(|c| c.is_ascii_digit())) else {
                kept.push(stmt);
                continue;
            };
            let renamed = format!("{fname}__{}", stmt[at]);
            renames.insert(stmt[at].clone(), renamed.clone());
            let mut decl = stmt.clone();
            decl[at] = renamed;
            out.push(decl);
        }
        for stmt in &mut kept {
            for token in stmt.iter_mut() {
                if let Some(new) = rename_symbol(token, &renames) {
                    *token = new;
                }
            }
        }
        func.body = kept;
    }
    out
}

/// `sym`, `sym+8` and `sym-8` with `sym` renamed.
fn rename_symbol(token: &str, renames: &HashMap<String, String>) -> Option<String> {
    let split = token.find(['+', '-']).unwrap_or(token.len());
    let (name, tail) = token.split_at(split);
    renames.get(name).map(|new| format!("{new}{tail}"))
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
    let (returns, return_regs) = returns.into_iter().unzip();
    let (params, param_regs) = params.into_iter().unzip();
    Some((name, Func { returns, params, return_regs, param_regs, body, token: 0 }))
}

/// Names of the declarations in the parenthesised list at `open`, each with its register type
/// when it is a `.reg` (not a `.param`).
fn param_names(tokens: &[String], open: usize) -> Option<(Vec<(String, Option<String>)>, usize)> {
    let mut names = Vec::new();
    let mut i = open + 1;
    let mut last: Option<String> = None;
    let mut is_reg = false;
    let mut ty: Option<String> = None;
    let mut bracket = 0;
    let mut finish = |last: &mut Option<String>, is_reg: &mut bool, ty: &mut Option<String>| {
        if let Some(name) = last.take() {
            names.push((name, if *is_reg { Some(ty.clone().unwrap_or_else(|| ".b32".into())) } else { None }));
        }
        *is_reg = false;
        *ty = None;
    };
    while i < tokens.len() {
        match tokens[i].as_str() {
            ")" => {
                finish(&mut last, &mut is_reg, &mut ty);
                return Some((names, i + 1));
            }
            "," => finish(&mut last, &mut is_reg, &mut ty),
            "[" => bracket += 1,
            "]" => bracket -= 1,
            ".reg" => is_reg = true,
            token if token.starts_with(".reg.") => {
                is_reg = true;
                ty = token.rsplit('.').next().map(|t| format!(".{t}"));
            }
            token if bracket == 0 && token.starts_with('.') && !token.starts_with(".param") && !token.starts_with(".align") => {
                ty = Some(token.to_owned());
            }
            token if bracket == 0 && !token.starts_with('.') && !token.chars().all(|c| c.is_ascii_digit()) => {
                last = Some(token.to_owned())
            }
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

/// Expands calls in `stmts`. The statements are taken by value: one that passes through unchanged
/// is moved to the output instead of cloned (copying every statement of the module was most of
/// `inline::expand` on stock ggml-cuda). Only the statement just read is moved; the look-ahead
/// reads the next one, which is still in place.
fn expand_stmts(
    mut stmts: Vec<Stmt>,
    module: &Module,
    depth: usize,
    counter: &mut usize,
    budget: &mut usize,
) -> Result<Vec<Stmt>, Error> {
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    let mut frame = Frame::default();
    // `label: .callprototype ...` and `label: .calltargets ...` seen so far in this body.
    let mut prototypes: HashMap<String, Signature> = HashMap::new();
    let mut target_lists: HashMap<String, Vec<String>> = HashMap::new();
    let mut i = 0;
    while i < stmts.len() {
        let stmt = &stmts[i];
        i += 1;
        if stmt.len() == 2 && stmt[1] == ":" {
            if let Some(next) = stmts.get(i) {
                match next.first().map(String::as_str) {
                    Some(".callprototype") => {
                        if let Some(signature) = prototype_signature(next) {
                            prototypes.insert(stmt[0].clone(), signature);
                        }
                        i += 1;
                        continue;
                    }
                    Some(".calltargets") => {
                        let names = next[1..].iter().filter(|t| !matches!(t.as_str(), "," | ";")).cloned().collect();
                        target_lists.insert(stmt[0].clone(), names);
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
            }
        }
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
            let candidates: Vec<&str> = if call.pointer.is_some() {
                let signature = call.prototype.as_ref().and_then(|p| prototypes.get(p)).cloned();
                let listed = call.prototype.as_ref().and_then(|p| target_lists.get(p));
                module
                    .address_taken
                    .iter()
                    .filter(|name| match listed {
                        Some(list) => list.contains(name),
                        None => {
                            let func = &module.funcs[name.as_str()];
                            match &signature {
                                Some(signature) => func.signature() == *signature,
                                None => func.returns.len() == call.returns.len() && func.params.len() == call.args.len(),
                            }
                        }
                    })
                    .map(String::as_str)
                    .collect()
            } else {
                vec![call.target.as_str()]
            };
            if call.pointer.is_none() && !module.funcs.contains_key(&call.target) {
                if let Some(special) = special_call(&call, &mut i, &stmts, &mut frame, &mut out) {
                    special?;
                    continue;
                }
                let reason = if module.declared.contains(&call.target) { "extern" } else { "undefined" };
                out.push(mark_unsupported(stmt, reason));
                continue;
            }
            if candidates.is_empty() {
                out.push(mark_unsupported(stmt, "indirect"));
                continue;
            }
            if depth >= MAX_DEPTH {
                out.push(mark_unsupported(stmt, "recursive"));
                continue;
            }
            if candidates.iter().any(|name| module.funcs[*name].body.iter().any(unsupported_in_callee)) {
                out.push(mark_unsupported(stmt, "static"));
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
            let mut inlined: Vec<Stmt> = Vec::new();
            let count = candidates.len();
            if let Some(pointer) = call.pointer.as_ref().filter(|_| count > 1) {
                // Switch over the candidates; the last one is the fall-through arm.
                let predicate = format!("%__icp{id}");
                let end = format!("$L__icall_end{id}");
                inlined.push(vec!["{".into()]);
                inlined.push(vec![".reg".into(), ".pred".into(), predicate.clone(), ";".into()]);
                for (k, name) in candidates[..count - 1].iter().enumerate() {
                    let token = module.funcs[*name].token.to_string();
                    inlined.push(vec!["setp.eq.u64".into(), predicate.clone(), ",".into(), pointer.clone(), ",".into(), token, ";".into()]);
                    inlined.push(vec!["@".into(), predicate.clone(), "bra".into(), format!("$L__icall{id}_{k}"), ";".into()]);
                }
                for (k, name) in candidates.iter().enumerate().rev() {
                    if k + 1 < count {
                        inlined.push(vec![format!("$L__icall{id}_{k}"), ":".into()]);
                    }
                    *counter += 1;
                    let body = inline_one(&module.funcs[*name], &call, &frame, &retmap, module, depth, counter, budget)?;
                    inlined.extend(body);
                    inlined.push(vec!["bra.uni".into(), end.clone(), ";".into()]);
                }
                inlined.push(vec![end, ":".into()]);
                inlined.push(vec!["}".into()]);
            } else {
                *counter -= 1;
                inlined = inline_one(&module.funcs[candidates[0]], &call, &frame, &retmap, module, depth, counter, budget)?;
            }
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
        if let Some(replaced) = replace_function_values(stmt, module) {
            out.push(replaced);
            continue;
        }
        out.push(std::mem::take(&mut stmts[i - 1]));
    }
    Ok(out)
}

/// One inlined copy of `func` for `call`, with a fresh call-site id.
#[allow(clippy::too_many_arguments)]
fn inline_one(
    func: &Func,
    call: &Call,
    frame: &Frame,
    retmap: &HashMap<u64, String>,
    module: &Module,
    depth: usize,
    counter: &mut usize,
    budget: &mut usize,
) -> Result<Vec<Stmt>, Error> {
    *counter += 1;
    let id = *counter;
    let body = expand_stmts(func.body.clone(), module, depth + 1, counter, budget)?;
    inline_body(func, call, frame, retmap, &body, id)
}

/// Keeps a call that cannot be inlined, with an opcode the parser reports precisely.
fn mark_unsupported(stmt: &Stmt, reason: &str) -> Stmt {
    stmt.iter()
        .map(|t| if t == "call" || t.starts_with("call.") { format!("call.cfi_{reason}") } else { t.clone() })
        .collect()
}

/// `.callprototype (.param .b32 _) _ (.reg .u32 _, .param .b64 _) ;`
fn prototype_signature(stmt: &Stmt) -> Option<Signature> {
    let mut i = 1;
    let mut returns = Vec::new();
    if stmt.get(i).is_some_and(|t| t == "(") {
        let (names, next) = param_names(stmt, i)?;
        returns = names;
        i = next;
    }
    i += 1; // `_`
    let params = if stmt.get(i).is_some_and(|t| t == "(") { param_names(stmt, i)?.0 } else { Vec::new() };
    Some(Signature {
        returns: returns.iter().map(|(_, r)| r.is_some()).collect(),
        params: params.iter().map(|(_, r)| r.is_some()).collect(),
    })
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
    /// The register holding the callee of an indirect call.
    pointer: Option<String>,
    /// The `.callprototype` or `.calltargets` label of an indirect call.
    prototype: Option<String>,
}

/// `call[.uni] [(rets) ,] target [, (args)] [, proto] ;`
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
    if target == "(" {
        return None;
    }
    i += 1;
    let mut args = Vec::new();
    if rest.get(i).is_some_and(|t| t == ",") && rest.get(i + 1).is_some_and(|t| t == "(") {
        let close = rest[i + 1..].iter().position(|t| t == ")")? + i + 1;
        args = rest[i + 2..close].iter().filter(|t| *t != ",").cloned().collect();
        i = close + 1;
    }
    let pointer = target.starts_with('%').then(|| target.clone());
    let prototype = if rest.get(i).is_some_and(|t| t == ",") { rest.get(i + 1).filter(|t| *t != ";").cloned() } else { None };
    Some(Call { returns, target, args, pointer, prototype })
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

/// Shrinks the callee's `.reg %x<N>;` ranges to the registers the inlined copy actually uses
/// (and drops unused ranges). Without this every call site re-declares the callee's full
/// register file, so a large device function called hundreds of times exceeds the register cap.
fn trim_register_ranges(out: &mut Vec<Stmt>, suffix: &str) {
    let mut used: HashMap<String, usize> = HashMap::new();
    for stmt in out.iter() {
        if stmt.first().is_some_and(|t| t == ".reg") {
            continue;
        }
        for token in stmt {
            let mut rest = token.as_str();
            while let Some(at) = rest.find('%') {
                rest = &rest[at + 1..];
                let end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '$'))
                    .unwrap_or(rest.len());
                let name = &rest[..end];
                if let Some(local) = name.strip_prefix(suffix) {
                    let digits = local.len() - local.trim_end_matches(|c: char| c.is_ascii_digit()).len();
                    if digits > 0 {
                        let (prefix, number) = local.split_at(local.len() - digits);
                        if let Ok(n) = number.parse::<usize>() {
                            let slot = used.entry(prefix.to_owned()).or_insert(0);
                            *slot = (*slot).max(n + 1);
                        }
                    }
                }
                rest = &rest[end..];
            }
        }
    }
    out.retain_mut(|stmt| {
        if stmt.first().map(String::as_str) != Some(".reg") {
            return true;
        }
        let Some(open) = stmt.iter().position(|t| t == "<") else { return true };
        let Some(name) = open.checked_sub(1).and_then(|i| stmt.get(i)) else { return true };
        let Some(local) = name.strip_prefix('%').and_then(|n| n.strip_prefix(suffix)) else { return true };
        let count = used.get(local).copied().unwrap_or(0);
        if count == 0 {
            return false;
        }
        if let Some(slot) = stmt.get_mut(open + 1) {
            if slot.parse::<usize>().is_ok_and(|n| n > count) {
                *slot = count.to_string();
            }
        }
        true
    });
}

/// Move between registers of a `.reg` parameter's type.
fn reg_mov(ty: &str) -> &'static str {
    match ty {
        ".pred" => "mov.pred",
        _ if element_bytes(ty.trim_start_matches('.')) == 8 => "mov.b64",
        ".u16" | ".s16" | ".b16" | ".f16" | ".bf16" => "mov.b16",
        _ => "mov.b32",
    }
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
    // `.reg` parameters: a per-call-site register initialised from the actual argument.
    for (position, (name, reg)) in func.params.iter().zip(&func.param_regs).enumerate() {
        let Some(ty) = reg else { continue };
        let Some(actual) = call.args.get(position) else {
            return Err(Error::new("PTX call has too few arguments"));
        };
        let local = rename_token(name, &suffix, &labels);
        out.push(vec![".reg".into(), ty.clone(), local.clone(), ";".into()]);
        out.push(vec![reg_mov(ty).into(), local, ",".into(), actual.clone(), ";".into()]);
    }
    let mut reg_results: Vec<Stmt> = Vec::new();
    for (position, (name, reg)) in func.returns.iter().zip(&func.return_regs).enumerate() {
        let Some(ty) = reg else { continue };
        let local = rename_token(name, &suffix, &labels);
        out.push(vec![".reg".into(), ty.clone(), local.clone(), ";".into()]);
        if let Some(actual) = call.returns.get(position) {
            reg_results.push(vec![reg_mov(ty).into(), actual.clone(), ",".into(), local, ";".into()]);
        }
    }
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
    out.extend(reg_results);
    out.push(vec!["}".into()]);
    trim_register_ranges(&mut out, &suffix);
    if hoisted.is_empty() {
        return Ok(out);
    }
    hoisted.extend(out);
    Ok(hoisted)
}

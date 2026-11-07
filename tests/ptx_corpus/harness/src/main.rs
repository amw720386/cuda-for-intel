//! Scratch harness: compiles the translator sources (src/ptx.rs) outside the
//! main crate (which does not build on macOS) and runs every corpus file.
//!
//! usage: ptx-corpus-harness <corpus_dir> [--bless]
//! Output: one line per file: `<relative path>\t<pass|unsupported|error>\t<message>`
#![allow(dead_code)]
// src/ptx.rs and src/ptx/ are symlinks to the translator sources.
mod ptx;

use std::{fs, path::{Path, PathBuf}};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        // The harness crate lives inside the corpus tree and symlinks back into the repo.
        if p.file_name().is_some_and(|n| n == "harness" || n == "target") { continue; }
        if p.is_dir() { walk(&p, out) } else if p.extension().is_some_and(|e| e == "ptx") { out.push(p) }
    }
}

/// ZLUDA's kernels use bare register names (`.reg .u64 temp;`). nvcc always emits
/// `%name`. Prefix declared register names with `%` so those files exercise the
/// instruction set rather than the naming gap (recorded in docs/ptx-gap-report.md).
fn prefix_regs(src: &str) -> String {
    // (name, is_range): `temp<3>` declares temp0..temp2 and uses them as `temp0`.
    let mut names: Vec<(String, bool)> = Vec::new();
    for l in src.lines() {
        let t = l.trim_start();
        if let Some(rest) = t.strip_prefix(".reg") {
            let mut it = rest.trim_start().trim_end_matches(';').split_whitespace();
            let _ty = it.next();
            for tok in it {
                let tok = tok.trim_matches(',');
                let base = tok.split('<').next().unwrap_or(tok);
                if !base.is_empty() && !base.starts_with('%') { names.push((base.to_string(), tok.contains('<'))); }
            }
        }
    }
    if names.is_empty() { return src.to_string(); }
    names.sort_by_key(|n| std::cmp::Reverse(n.0.len()));
    let isid = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$';
    let mut out = String::new();
    for l in src.lines() {
        let code = l.split("//").next().unwrap();
        let comment = &l[code.len()..];
        let chars: Vec<char> = code.chars().collect();
        let mut i = 0;
        let mut res = String::new();
        while i < chars.len() {
            let prev_ok = i == 0 || !(isid(chars[i - 1]) || chars[i - 1] == '%' || chars[i - 1] == '.');
            let mut hit = None;
            if prev_ok {
                for (n, range) in &names {
                    let nc: Vec<char> = n.chars().collect();
                    if !chars[i..].starts_with(&nc) { continue; }
                    let mut end = i + nc.len();
                    if *range {
                        if chars.get(end) == Some(&'<') { hit = Some(nc.len()); break; }
                        let digits = chars[end..].iter().take_while(|c| c.is_ascii_digit()).count();
                        if digits == 0 { continue; }
                        end += digits;
                    }
                    if chars.get(end).map_or(true, |c| !isid(*c)) { hit = Some(end - i); break; }
                }
            }
            if let Some(len) = hit { res.push('%'); res.extend(&chars[i..i + len]); i += len; } else { res.push(chars[i]); i += 1; }
        }
        out.push_str(&res); out.push_str(comment); out.push('\n');
    }
    out
}

fn probe(path: &str) {
    std::panic::set_hook(Box::new(|_| {}));
    let raw = fs::read_to_string(path).unwrap();
    let prepared = ptx::prepare_module(&raw).unwrap_or_else(|e| { println!("prepare failed: {e}"); std::process::exit(1) });
    let names = ptx::entry_names(&prepared).map(|n| n.len()).unwrap_or(0);
    let results = ptx::probe_module(&prepared).unwrap_or_else(|e| { println!("parse failed: {e}"); std::process::exit(1) });
    let mut by_error: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let mut ok = 0;
    for (name, r) in &results { match r { Ok(()) => ok += 1, Err(e) => by_error.entry(e.clone()).or_default().push(name.clone()) } }
    println!("entries {names}, parsed {}, emitted ok {ok}", results.len());
    println!("parse-skipped: {}", names.saturating_sub(results.len()));
    let mut skipped: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for (n, e) in ptx::skipped_entries() { skipped.entry(e).or_default().push(n) }
    for (e, v) in &skipped { println!("SKIP {:5}  {}\n         e.g. {}", v.len(), e, v[0]); }
    for (e, v) in &by_error { println!("{:5}  {}\n         e.g. {}", v.len(), e, v[0]); }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--probe") { probe(&std::env::args().nth(2).expect("ptx file")); return; }
    let root = PathBuf::from(std::env::args().nth(1).expect("corpus dir"));
    let mut files = Vec::new();
    walk(&root, &mut files);
    std::panic::set_hook(Box::new(|_| {}));
    for f in files {
        let rel = f.strip_prefix(&root).unwrap().display().to_string();
        let raw = String::from_utf8_lossy(&fs::read(&f).unwrap()).into_owned();
        // The translator only accepts .version 4.1/8.8/9.0. Normalise the version so
        // the report shows the first *real* gap rather than the version gate; the
        // gate itself is recorded in the `strict` column.
        let strict = ptx::translate(&raw).is_ok();
        let text = {
            let mut out = String::new();
            let mut done = false;
            for l in raw.lines() {
                if !done && l.trim_start().starts_with(".version") { out.push_str(".version 8.8\n"); done = true; } else { out.push_str(l); out.push('\n'); }
            }
            prefix_regs(&out)
        };
        let r = std::panic::catch_unwind(|| {
            ptx::translate(&text).map(|_| ()).map_err(|e| {
                let m = e.to_string();
                if m.contains("no supported") {
                    // translate() hides per-entry reasons; ask diagnose_entry.
                    if let Ok(names) = ptx::entry_names(&text) {
                        if let Some(n) = names.first() {
                            if let Err(d) = ptx::diagnose_entry(&text, n) { return d.to_string(); }
                        }
                    }
                }
                m
            })
        });
        let (status, msg) = match r {
            Ok(Ok(())) => ("pass", String::new()),
            Ok(Err(m)) if m.contains("unsupported") || m.contains("no supported") => ("unsupported", m),
            Ok(Err(m)) => ("error", m),
            Err(p) => {
                let m = p.downcast_ref::<String>().cloned().or(p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                ("error", format!("panic: {m}"))
            }
        };
        println!("{rel}\t{status}\t{}\t{}", if strict {"strict-pass"} else {"strict-fail"}, msg.replace('\n', " "));
    }
}

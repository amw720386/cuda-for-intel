//! Scratch harness: compiles the translator sources (src/ptx.rs) outside the
//! main crate (which does not build on macOS) and runs every corpus file.
//!
//! usage: ptx-corpus-harness <corpus_dir> [--bless]
//! Output: one line per file: `<relative path>\t<pass|unsupported|error>\t<message>`
#![allow(dead_code)]
// src/ptx.rs and src/ptx/ are symlinks to the translator sources.
mod ptx;
// Shared with src/ptx/zluda_vector_tests.rs.
#[path = "ptx/zluda_source.rs"]
mod zluda_source;

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

fn probe(path: &str) {
    ptx::set_lenient_undefined(true);
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
    if std::env::var_os("PROBE_ALL").is_some() { for (e, v) in &by_error { for n in v { println!("FAIL\t{e}\t{n}"); } } }
}

/// `--hash file.ptx`: one line per entry, `name<TAB>fnv64(SPIR-V words)` or `name<TAB>ERR`.
fn hash(path: &str) {
    ptx::set_lenient_undefined(true);
    std::panic::set_hook(Box::new(|_| {}));
    let raw = fs::read_to_string(path).unwrap();
    let prepared = ptx::prepare_module(&raw).unwrap_or_else(|e| { println!("prepare failed: {e}"); std::process::exit(1) });
    let mut names: Vec<String> = ptx::entry_names(&prepared).map(|n| n.into_iter().map(String::from).collect()).unwrap_or_default();
    names.sort();
    for n in names {
        let r = std::panic::catch_unwind(|| ptx::translate_prepared_entry(&prepared, &n).map(|t| {
            let mut h: u64 = 0xcbf29ce484222325;
            for w in &t.words { for b in w.to_le_bytes() { h = (h ^ b as u64).wrapping_mul(0x100000001b3); } }
            h
        }));
        match r { Ok(Ok(h)) => println!("{n}\t{h:016x}"), _ => println!("{n}\tERR") }
    }
}

/// `--time <file> [threads]`: per-phase timing, then per-entry `name<TAB>hash|error<TAB>ms` on stdout (sorted).
fn time_file(path: &str, threads: usize, mode: &str) {
    use std::time::Instant;
    ptx::set_lenient_undefined(true);
    std::panic::set_hook(Box::new(|_| {}));
    let t = Instant::now();
    let raw = String::from_utf8_lossy(&fs::read(path).unwrap()).into_owned();
    let prepared = match ptx::prepare_module(&raw) { Ok(p) => p, Err(e) => { println!("PREPARE_FAIL\t{e}"); return } };
    eprintln!("prepare {:.2?} ({} bytes -> {})", t.elapsed(), raw.len(), prepared.len());
    let t = Instant::now();
    if mode == "lenient" {
        let t = Instant::now();
        let r = ptx::translate_lenient(&raw);
        match r { Ok(t2) => { let mut h: u64 = 0xcbf29ce484222325; for w in &t2.words { for b in w.to_le_bytes() { h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3); } } println!("LENIENT\t{h:016x}\t{}", t2.kernels.len()) } Err(e) => println!("LENIENT\tERR {e}") }
        eprintln!("translate_lenient whole module {:.2?}", t.elapsed());
        return;
    }
    let t = Instant::now();
    let r = match if mode == "lazy" { ptx::probe_lazy_hashed(&prepared, threads) } else if mode == "prepared" { ptx::probe_prepared_hashed(&prepared, threads) } else { ptx::probe_module_hashed(&prepared, threads) } { Ok(r) => r, Err(e) => { println!("PARSE_FAIL\t{e}"); return } };
    eprintln!("parse+translate all entries ({threads} threads) {:.2?}", t.elapsed());
    let mut worst: Vec<_> = r.iter().map(|(n, _, d)| (*d, n.clone())).collect();
    worst.sort(); worst.reverse();
    for (d, n) in worst.iter().take(3) { eprintln!("  slowest {:.2?} {}", d, &n[..n.len().min(70)]); }
    let mut lines: Vec<String> = r.iter().map(|(n, h, d)| match h {
        Ok(h) => format!("{n}\t{h:016x}\t{}", d.as_millis()),
        Err(e) => format!("{n}\tERR {}\t{}", e.replace('\n', " "), d.as_millis()),
    }).collect();
    lines.sort();
    for l in lines { println!("{l}"); }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--hash") { hash(&std::env::args().nth(2).expect("ptx file")); return; }
    if std::env::args().nth(1).as_deref() == Some("--time") {
        let th = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);
        let mode = std::env::var("MODE").unwrap_or_default();
        time_file(&std::env::args().nth(2).expect("ptx file"), th, &mode);
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("--prep") {
        let raw = fs::read_to_string(std::env::args().nth(2).unwrap()).unwrap();
        print!("{}", ptx::prepare_module(&raw).unwrap());
        return;
    }
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
        let text = zluda_source::normalize(&raw);
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

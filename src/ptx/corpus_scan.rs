//! Corpus coverage scan (ignored test). `CFI_CORPUS_DIR=<dir>` walks every `*.ptx` below it,
//! translates every entry, validates with `spirv-val` when `CFI_CORPUS_VAL=1`, and prints
//! instruction/modifier/state-space histograms plus the translate-error classes. The corpus
//! lives outside the repository (NVIDIA/llama.cpp/torch PTX must not be committed).
//! `CFI_CORPUS_OUT=<file>` receives the machine-readable report.
use super::{parse, prepare, spirv};
use std::collections::BTreeMap;
use std::fmt::Write as _;

fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut v: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    v.sort();
    for p in v {
        if p.file_name().is_some_and(|n| n == "harness" || n == "target" || n == ".git") {
            continue;
        }
        if p.is_dir() {
            walk(&p, out)
        } else if p.extension().is_some_and(|e| e == "ptx") {
            out.push(p)
        }
    }
}

/// Collapses digits so register-count style variation does not split error classes.
fn error_class(e: &str) -> String {
    let mut s = String::new();
    let mut prev_digit = false;
    for c in e.chars() {
        if c.is_ascii_digit() || c == '%' {
            if !prev_digit {
                s.push('#');
            }
            prev_digit = true;
        } else {
            s.push(c);
            prev_digit = false;
        }
    }
    s.chars().take(160).collect()
}

fn opcode_of(line: &str) -> Option<&str> {
    let mut t = line.trim();
    if t.is_empty() || t.starts_with("//") || t.starts_with('.') || t.starts_with('{') || t.starts_with('}') || t.ends_with(':') || t.starts_with('$') {
        return None;
    }
    if let Some(rest) = t.strip_prefix('@') {
        t = rest.split_whitespace().nth(1)?;
    }
    let op = t.split_whitespace().next()?;
    let op = op.trim_end_matches(';');
    if op.starts_with('%') || op.starts_with('(') || op.is_empty() { None } else { Some(op) }
}

#[test]
#[ignore = "reads the corpus named by CFI_CORPUS_DIR"]
fn corpus_scan() {
    let Some(dir) = std::env::var_os("CFI_CORPUS_DIR") else { return };
    let validate = std::env::var_os("CFI_CORPUS_VAL").is_some();
    let mut files = Vec::new();
    walk(std::path::Path::new(&dir), &mut files);
    let mut ops_full = BTreeMap::<String, usize>::new(); // full dotted opcode -> count
    let mut ops_base = BTreeMap::<String, usize>::new();
    let mut mods = BTreeMap::<String, usize>::new();
    let mut spaces = BTreeMap::<String, usize>::new();
    // opcode (base) -> error classes of failing entries that use it
    let mut failing_ops = BTreeMap::<String, usize>::new();
    let mut errs = BTreeMap::<String, (usize, String)>::new();
    let (mut nfiles, mut nentries, mut nok, mut nfail, mut nval_bad, mut nparse_fail) = (0, 0, 0, 0, 0, 0);
    let mut seen_hash = std::collections::HashSet::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        nfiles += 1;
        let mut ctx = format!("{}", f.file_name().unwrap().to_string_lossy());
        if !seen_hash.insert(text.len() as u64 * 1_000_003 + text.bytes().take(4096).map(u64::from).sum::<u64>()) {
            continue; // duplicate module
        }
        for line in text.lines() {
            if let Some(op) = opcode_of(line) {
                *ops_full.entry(op.to_string()).or_default() += 1;
                let mut parts = op.split('.');
                let base = parts.next().unwrap().to_string();
                *ops_base.entry(base.clone()).or_default() += 1;
                for m in parts {
                    *mods.entry(format!("{base}.{m}")).or_default() += 1;
                    if matches!(m, "global" | "shared" | "local" | "const" | "param" | "generic" | "tex" | "surf" | "cluster" | "async") {
                        *spaces.entry(format!("{base}.{m}")).or_default() += 1;
                    }
                }
            }
        }
        let prepared = match prepare(&text) {
            Ok(p) => p,
            Err(e) => {
                nparse_fail += 1;
                let c = error_class(&format!("prepare: {e}"));
                errs.entry(c).or_insert((0, ctx.clone())).0 += 1;
                continue;
            }
        };
        let (entries, vars) = match parse::parse_module(&prepared) {
            Ok(p) => p,
            Err(e) => {
                nparse_fail += 1;
                let c = error_class(&format!("parse: {e}"));
                errs.entry(c).or_insert((0, ctx.clone())).0 += 1;
                continue;
            }
        };
        for entry in entries {
            nentries += 1;
            let name = entry.name.to_owned();
            ctx = format!("{} :: {}", f.file_name().unwrap().to_string_lossy(), &name[..name.len().min(60)]);
            match spirv::emit(vec![entry], vars.device.clone()) {
                Ok(t) => {
                    nok += 1;
                    if validate {
                        let tmp = std::env::temp_dir().join("cfi_corpus_val.spv");
                        let bytes: Vec<u8> = t.words.iter().flat_map(|w| w.to_le_bytes()).collect();
                        std::fs::write(&tmp, bytes).unwrap();
                        let out = std::process::Command::new("spirv-val").arg("--target-env").arg("opencl2.2").arg(&tmp).output();
                        if let Ok(o) = out
                            && !o.status.success()
                        {
                            nval_bad += 1;
                            let c = error_class(&format!("spirv-val: {}", String::from_utf8_lossy(&o.stderr).lines().next().unwrap_or("")));
                            errs.entry(c).or_insert((0, ctx.clone())).0 += 1;
                        }
                    }
                }
                Err(e) => {
                    nfail += 1;
                    let c = error_class(&e.to_string());
                    errs.entry(c).or_insert((0, ctx.clone())).0 += 1;
                }
            }
        }
    }
    let mut r = String::new();
    writeln!(r, "SUMMARY files={nfiles} unique_entries={nentries} translated={nok} failed={nfail} module_parse_fail={nparse_fail} spirv_val_bad={nval_bad}").unwrap();
    for (k, v) in &ops_base { writeln!(r, "OP\t{k}\t{v}").unwrap(); }
    for (k, v) in &ops_full { writeln!(r, "OPFULL\t{k}\t{v}").unwrap(); }
    for (k, v) in &mods { writeln!(r, "MOD\t{k}\t{v}").unwrap(); }
    for (k, v) in &spaces { writeln!(r, "SPACE\t{k}\t{v}").unwrap(); }
    for (k, (n, ex)) in &errs { writeln!(r, "ERR\t{n}\t{k}\te.g. {ex}").unwrap(); }
    let _ = &mut failing_ops;
    match std::env::var_os("CFI_CORPUS_OUT") {
        Some(p) => std::fs::write(p, &r).unwrap(),
        None => eprintln!("{r}"),
    }
}

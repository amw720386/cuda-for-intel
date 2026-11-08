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


fn normalise(text: &str) -> String {
    // ZLUDA test kernels use bare register names and PTX 6.x headers; real toolchains emit %names and >= 7.0.
    let mut t = prefix_regs(text);
    for old in [".version 6.0", ".version 6.2", ".version 6.3", ".version 6.4", ".version 6.5"] {
        t = t.replace(old, ".version 7.0");
    }
    t
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
    super::set_lenient_undefined(true); // as the runtime does
    // CFI_CORPUS_VAL=N validates every Nth translated entry with spirv-val (1 = all; slow).
    let validate_every: usize = std::env::var("CFI_CORPUS_VAL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut validated = 0usize;
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
    let mut fails: Vec<String> = Vec::new();
    let mut seen_hash = std::collections::HashSet::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        let text = normalise(&text);
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
            ctx = format!("{} :: {}", f.file_name().unwrap().to_string_lossy(), &name[..name.len().min(400)]);
            match spirv::emit(vec![entry], vars.device.clone()) {
                Ok(t) => {
                    nok += 1;
                    if validate_every > 0 && { nok += 0; validated += 1; validated % validate_every == 0 } {
                        if let Some(msg) = super::fuzz_tests::validate(&t.words) {
                            nval_bad += 1;
                            let c = error_class(&format!("spirv-val: {msg}"));
                            errs.entry(c).or_insert((0, ctx.clone())).0 += 1;
                        }
                    }
                }
                Err(e) => {
                    nfail += 1;
                    fails.push(format!("FAIL\t{ctx}\t{e}"));
                    let c = error_class(&e.to_string());
                    errs.entry(c).or_insert((0, format!("{ctx} [{}]", e.to_string().chars().take(120).collect::<String>()))).0 += 1;
                }
            }
        }
    }
    let mut r = String::new();
    writeln!(r, "SUMMARY validated={}/{nok} files={nfiles} unique_entries={nentries} translated={nok} failed={nfail} module_parse_fail={nparse_fail} spirv_val_bad={nval_bad}", validated / validate_every.max(1)).unwrap();
    for (k, v) in &ops_base { writeln!(r, "OP\t{k}\t{v}").unwrap(); }
    for (k, v) in &ops_full { writeln!(r, "OPFULL\t{k}\t{v}").unwrap(); }
    for (k, v) in &mods { writeln!(r, "MOD\t{k}\t{v}").unwrap(); }
    for (k, v) in &spaces { writeln!(r, "SPACE\t{k}\t{v}").unwrap(); }
    for (k, (n, ex)) in &errs { writeln!(r, "ERR\t{n}\t{k}\te.g. {ex}").unwrap(); }
    for f in &fails { writeln!(r, "{f}").unwrap(); }
    let _ = &mut failing_ops;
    match std::env::var_os("CFI_CORPUS_OUT") {
        Some(p) => std::fs::write(p, &r).unwrap(),
        None => eprintln!("{r}"),
    }
}

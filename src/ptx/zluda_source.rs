//! Source normalisation for ZLUDA's test kernels, shared by the corpus harness
//! (tests/ptx_corpus/harness) and `zluda_vector_tests`. The translator itself uses
//! [`has_bare_regs`] and [`prefix_regs`] (see `prepare` in src/ptx.rs).

/// The translator accepts only `.version` 4.1/8.8/9.0; ZLUDA's files use older versions.
/// Rewrites the first `.version` line to 8.8 and prefixes bare register names.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn normalize(raw: &str) -> String {
    let mut out = String::new();
    let mut done = false;
    for l in raw.lines() {
        if !done && l.trim_start().starts_with(".version") {
            out.push_str(".version 8.8\n");
            done = true;
        } else {
            out.push_str(l);
            out.push('\n');
        }
    }
    prefix_regs(&out)
}

/// True when a `.reg` declaration (statement or `.func` parameter) names a register without the
/// `%` prefix. Only lines that mention `.reg` are looked at, so nvcc output costs one scan.
pub(crate) fn has_bare_regs(src: &str) -> bool {
    src.lines().filter(|l| l.contains(".reg")).any(|l| {
        let code = l.split("//").next().unwrap_or(l);
        let spaced = code.replace(['(', ')', ';', '{', '}', '=', ','], " ; ");
        let mut in_reg = false;
        for tok in spaced.split_whitespace() {
            if tok == ".reg" || tok.starts_with(".reg.") {
                in_reg = true;
            } else if tok == ";" || tok == ".param" || tok.starts_with(".param.") {
                in_reg = false;
            } else if in_reg && !tok.starts_with('.') && !tok.starts_with('%') {
                return true;
            }
        }
        false
    })
}

/// ZLUDA's kernels use bare register names (`.reg .u64 temp;`). nvcc always emits
/// `%name`. Prefix declared register names with `%` so those files exercise the
/// instruction set rather than the naming gap (recorded in docs/ptx-gap-report.md).
pub(crate) fn prefix_regs(src: &str) -> String {
    // (name, is_range): `temp<3>` declares temp0..temp2 and uses them as `temp0`.
    // Declarations are `.reg` statements and `.reg` parameters of `.func` headers; the type may be
    // several tokens (`.reg .v2 .u32 v`) or fused (`.reg.v4.u16 v`).
    let mut names: Vec<(String, bool)> = Vec::new();
    let code: String = src.lines().map(|l| l.split("//").next().unwrap()).collect::<Vec<_>>().join("\n");
    let spaced = code.replace(['(', ')', ';', '{', '}', '='], " ; ").replace(',', " , ");
    let mut in_reg = false;
    for tok in spaced.split_whitespace() {
        if tok == ".reg" || tok.starts_with(".reg.") {
            in_reg = true;
        } else if tok == ";" || tok == ".param" || tok.starts_with(".param.") {
            in_reg = false;
        } else if tok.starts_with('.') || tok == "," {
            // Type tokens precede the names.
        } else if in_reg {
            let base = tok.split('<').next().unwrap_or(tok);
            if !base.is_empty() && !base.starts_with('%') { names.push((base.to_string(), tok.contains('<'))); }
        }
    }
    names.sort();
    names.dedup();
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

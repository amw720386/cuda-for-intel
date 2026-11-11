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
    rewrite_bare(src, &names)
}

/// Prefixes every occurrence of a bare register name (longest names first; a range name also
/// matches `name<` and `name<digits>`) with `%`, outside `//` comments.
///
/// nvcc output reaches this too: cuda_fp16's inline asm declares `{.reg .f16 low,high; ...}` and
/// calls declare `.reg .b32 temp_param_reg;`, so `has_bare_regs` is true for real ggml modules
/// (10,496 `low,high` scopes in the largest stock one). This works on bytes, copies lines that
/// contain no name unchanged and copies the text between matches in slices. The earlier
/// per-character version, which also collected each name into a fresh `Vec<char>` at every
/// position, took 459 ms of the 1.2 s prepare of a 9 MB module. Matches only start at char
/// boundaries (a name's first byte is never a UTF-8 continuation byte), so the output is identical
/// to the char-level rule; `tests::byte_rewrite_matches_char_reference` checks that.
fn rewrite_bare(src: &str, names: &[(String, bool)]) -> String {
    let isid = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'$';
    let mut out = String::with_capacity(src.len() + src.len() / 16);
    for l in src.lines() {
        let code = l.split("//").next().unwrap();
        if !names.iter().any(|(n, _)| code.contains(n.as_str())) {
            out.push_str(l);
            out.push('\n');
            continue;
        }
        let comment = &l[code.len()..];
        let b = code.as_bytes();
        let (mut i, mut copied) = (0usize, 0usize);
        while i < b.len() {
            let prev_ok = i == 0 || !(isid(b[i - 1]) || b[i - 1] == b'%' || b[i - 1] == b'.');
            let mut hit = None;
            if prev_ok {
                for (n, range) in names {
                    let nb = n.as_bytes();
                    if !b[i..].starts_with(nb) {
                        continue;
                    }
                    let mut end = i + nb.len();
                    if *range {
                        if b.get(end) == Some(&b'<') {
                            hit = Some(nb.len());
                            break;
                        }
                        let digits = b[end..].iter().take_while(|c| c.is_ascii_digit()).count();
                        if digits == 0 {
                            continue;
                        }
                        end += digits;
                    }
                    if b.get(end).is_none_or(|c| !isid(*c)) {
                        hit = Some(end - i);
                        break;
                    }
                }
            }
            if let Some(len) = hit {
                out.push_str(&code[copied..i]);
                out.push('%');
                out.push_str(&code[i..i + len]);
                i += len;
                copied = i;
            } else {
                i += 1;
            }
        }
        out.push_str(&code[copied..]);
        out.push_str(comment);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The original per-character rewrite, kept as the reference for `rewrite_bare`.
    fn rewrite_bare_reference(src: &str, names: &[(String, bool)]) -> String {
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
                    for (n, range) in names {
                        let nc: Vec<char> = n.chars().collect();
                        if !chars[i..].starts_with(&nc) {
                            continue;
                        }
                        let mut end = i + nc.len();
                        if *range {
                            if chars.get(end) == Some(&'<') {
                                hit = Some(nc.len());
                                break;
                            }
                            let digits = chars[end..].iter().take_while(|c| c.is_ascii_digit()).count();
                            if digits == 0 {
                                continue;
                            }
                            end += digits;
                        }
                        if chars.get(end).is_none_or(|c| !isid(*c)) {
                            hit = Some(end - i);
                            break;
                        }
                    }
                }
                if let Some(len) = hit {
                    res.push('%');
                    res.extend(&chars[i..i + len]);
                    i += len;
                } else {
                    res.push(chars[i]);
                    i += 1;
                }
            }
            out.push_str(&res);
            out.push_str(comment);
            out.push('\n');
        }
        out
    }

    #[test]
    fn byte_rewrite_matches_char_reference() {
        let names = |v: &[(&str, bool)]| v.iter().map(|(n, r)| (n.to_string(), *r)).collect::<Vec<_>>();
        let cases: &[(&str, &[(&str, bool)])] = &[
            // nvcc's fp16 unpack and call scratch register.
            (
                "{\n.reg .f16 low,high;\nmov.b32 {low,high}, %r1;\n}\n.reg .b32 temp_param_reg;\nst.param.b32 [p], temp_param_reg; // temp_param_reg low\n",
                &[("temp_param_reg", false), ("high", false), ("low", false)],
            ),
            // Range names: `t<3>` declares t0..t2; `t` alone, `t<` and longer identifiers.
            ("add.u32 t0, t1, t12;\nmov.u32 tx, t;\n.reg .u32 t<3>;\nmov.u32 %t1, .t2, t2$a, _t1, t2;\n", &[("t", true)]),
            // A name that prefixes another, `$` and `_` boundaries, already prefixed or after '.'.
            ("mov.u32 abc, ab; mov.u32 ab$, %ab, .ab, ab_c, (ab), [ab+4], ab\n", &[("abc", false), ("ab", false)]),
            // Non-ASCII text around names, CRLF line ends, empty lines, no trailing newline.
            ("// é low\r\nmov low, é low ;\r\n\r\n\nlowé low\tlow", &[("low", false)]),
            // A range name and a plain name of the same text.
            ("mov.u32 v, v0, v<2>, v12x, xv;\n", &[("v", false), ("v", true)]),
            ("", &[("x", false)]),
        ];
        for (src, n) in cases {
            let n = names(n);
            assert_eq!(rewrite_bare(src, &n), rewrite_bare_reference(src, &n), "input {src:?}");
        }
        // And through the full entry point on nvcc-shaped text.
        let src = cases[0].0;
        assert!(has_bare_regs(src));
        assert!(prefix_regs(src).contains("mov.b32 {%low,%high}, %r1;"));
    }
}

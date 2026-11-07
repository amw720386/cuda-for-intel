//! Module-scope `.global` / `.const` data (lookup tables, `__device__` variables).
//!
//! The declaration is parsed into raw little-endian bytes and handed to the
//! emitter (`tier3_emit.rs`), which makes it a program-scope `CrossWorkgroup`
//! variable with an initialiser. Child module of `parse` so it can use `Parser`.

use super::{Error, Parser};
use crate::ptx::GlobalData;

fn element_size(ty: &str) -> Option<usize> {
    Some(match ty {
        ".b8" | ".u8" | ".s8" => 1,
        ".b16" | ".u16" | ".s16" | ".f16" | ".bf16" => 2,
        ".b32" | ".u32" | ".s32" | ".f32" | ".f16x2" => 4,
        ".b64" | ".u64" | ".s64" | ".f64" => 8,
        _ => return None,
    })
}

/// One initialiser element as raw bits.
fn element_bits(token: &str, ty: &str) -> Option<u64> {
    let t = token.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u64::from_str_radix(hex.trim_end_matches(['U', 'u', 'L', 'l']), 16).ok();
    }
    if let Some(hex) = t.strip_prefix("0f").or_else(|| t.strip_prefix("0F")) {
        return u64::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = t.strip_prefix("0d").or_else(|| t.strip_prefix("0D")) {
        return u64::from_str_radix(hex, 16).ok();
    }
    let plain = t.trim_end_matches(['U', 'u', 'L', 'l']);
    if let Ok(v) = plain.parse::<i64>() {
        return Some(v as u64);
    }
    if let Ok(v) = plain.parse::<u64>() {
        return Some(v);
    }
    let f = plain.parse::<f64>().ok()?;
    Some(match ty {
        ".f32" => u64::from((f as f32).to_bits()),
        _ => f.to_bits(),
    })
}

/// Parses `[.align N] .TYPE name [ [count] ] [ = init ] ;` after the `.global`/`.const` token.
/// Returns `None` (after consuming the declaration) for `.extern` or unparseable forms.
pub(super) fn parse_declaration<'a>(parser: &mut Parser<'a>, is_extern: bool) -> Result<Option<GlobalData<'a>>, Error> {
    let mut align = 0u32;
    let mut ty = "";
    let mut name = "";
    let mut count: Option<usize> = None;
    let mut init: Vec<&'a str> = Vec::new();
    let mut state = 0; // 0 = header, 1 = after '=' (initialiser)
    let mut depth = 0i32;
    loop {
        let token = parser.take()?;
        if state == 1 {
            match token {
                "{" => depth += 1,
                "}" => depth -= 1,
                ";" if depth == 0 => break,
                "," => {}
                other => init.push(other),
            }
            continue;
        }
        match token {
            ";" => break,
            "=" => state = 1,
            ".align" => align = parser.take()?.parse().unwrap_or(0),
            "[" => {
                let n = parser.take()?;
                if n == "]" {
                    count = Some(0);
                } else {
                    count = n.parse().ok();
                    parser.take()?; // ]
                }
            }
            t if t.starts_with('.') && element_size(t).is_some() => ty = t,
            t if t.starts_with('.') => {}
            t => {
                if let Some((n, rest)) = t.split_once('=') {
                    name = n;
                    if rest.is_empty() {
                        state = 1;
                    }
                } else {
                    name = t;
                }
            }
        }
    }
    let Some(size) = element_size(ty) else { return Ok(None) };
    if is_extern || name.is_empty() {
        return Ok(None);
    }
    let n = count.unwrap_or(1).max(init.len()).max(usize::from(count.is_none()));
    if count == Some(0) && init.is_empty() {
        return Ok(None); // `name[]` with no initialiser: extern-style, nothing to allocate
    }
    let mut bytes = vec![0u8; n * size];
    for (i, token) in init.iter().enumerate() {
        let bits = element_bits(token, ty)
            .ok_or_else(|| Error::new(format!("unsupported PTX initialiser element {token}")))?;
        bytes[i * size..(i + 1) * size].copy_from_slice(&bits.to_le_bytes()[..size]);
    }
    Ok(Some(GlobalData { name, align: align.max(size as u32), bytes }))
}

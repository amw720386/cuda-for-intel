//! Kernel substitution: hand-written Intel-native SPIR-V replacements for known ggml-cuda kernels.
//!
//! A kernel is recognised by its Itanium-mangled name (template arguments included) *and* the
//! layout of its parameter list; anything else keeps the translated PTX. The driver glue
//! (`cuda/subst_hook.rs`) builds the native module lazily, validates it once against a host
//! reference on the real device, and falls back to the translated kernel on any failure.
//!
//! Selection: `CUDA_FOR_INTEL_SUBST=none|auto|all|<comma list>` (default `auto` = measured per-kernel policy; `none` disables). `auto` includes mmq (Q8_0); `auto,-mmq` (or `-mmvq`, `-quantize_q8_1`) switches a family off. Native mmq differs from stock stream-k only in float add order, so greedy output can flip near-ties. List items are a
//! family (`mmvq`, `quantize_q8_1`) optionally narrowed to a type (`mmvq:q8_0`, `mmvq:q4_k`).
use crate::ptx::Param;

pub(crate) mod mmq;
pub(crate) mod mmvq;
pub(crate) mod probe;
pub(crate) mod quantize;
pub(crate) mod refs;
pub(crate) mod spv;

#[cfg(test)]
mod interp;
#[cfg(test)]
mod tests;

// ---- environment switch ----

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Switch {
    all: bool,
    /// `auto`: the per-kernel policy measured on hardware (see `auto_ok`).
    auto: bool,
    items: Vec<String>,
    /// `-family` / `-family:type` items: switched off even under `auto` or `all`.
    denied: Vec<String>,
}

impl Switch {
    pub(crate) fn parse(text: &str) -> Self {
        let mut s = Self::default();
        for item in text.split(',').map(|i| i.trim().to_ascii_lowercase()).filter(|i| !i.is_empty()) {
            match item.as_str() {
                "none" | "0" | "off" => {
                    s.all = false;
                    s.auto = false;
                    s.items.clear();
                    s.denied.clear();
                }
                "all" | "1" | "on" => s.all = true,
                "auto" => s.auto = true,
                _ if item.starts_with('-') && item.len() > 1 => s.denied.push(item[1..].to_string()),
                _ => s.items.push(item),
            }
        }
        s
    }

    /// Reads the variable on every call (cheap, and lets tests and the hardware comparison load
    /// the same kernel with and without substitution in one process).
    pub(crate) fn from_env() -> Self {
        // Unset means `auto` (measured policy); `CUDA_FOR_INTEL_SUBST=none` disables substitution.
        match std::env::var("CUDA_FOR_INTEL_SUBST") {
            Ok(text) => Self::parse(&text),
            Err(_) => Self::parse("auto"),
        }
    }

    pub(crate) fn allows(&self, family: &str, ty: Option<&str>) -> bool {
        self.all
            || self.items.iter().any(|i| {
                i == family || ty.is_some_and(|t| i.strip_prefix(family).and_then(|r| r.strip_prefix(':')) == Some(t))
            })
    }

    pub(crate) fn denies(&self, family: &str, ty: Option<&str>) -> bool {
        self.denied.iter().any(|d| {
            d == family || ty.is_some_and(|t| d.strip_prefix(family).and_then(|r| r.strip_prefix(':')) == Some(t))
        })
    }

    pub(crate) fn is_none(&self) -> bool {
        !self.all && !self.auto && self.items.is_empty()
    }
}

// ---- Itanium name parsing (only what ggml-cuda kernel names need) ----

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TArg {
    Int(i64),
    Enum(String, i64),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Mangled {
    pub base: String,
    pub targs: Vec<TArg>,
}

/// `_Z13mul_mat_vec_qIL9ggml_type2ELi1EEvPKv...` -> base `mul_mat_vec_q`, targs `[ggml_type(2), 1]`.
/// Returns `None` for names with other template argument forms (expressions, packs, ...).
pub(crate) fn demangle(name: &str) -> Option<Mangled> {
    let rest = name.strip_prefix("_Z")?;
    let (base, rest) = source_name(rest)?;
    let mut targs = Vec::new();
    if let Some(mut r) = rest.strip_prefix('I') {
        loop {
            if let Some(after) = r.strip_prefix('E') {
                let _ = after;
                break;
            }
            let after_l = r.strip_prefix('L')?;
            let (ty, tail) = if after_l.starts_with(|c: char| c.is_ascii_digit()) {
                let (n, t) = source_name(after_l)?;
                (Some(n), t)
            } else {
                (None, after_l.get(1..)?)
            };
            let neg = tail.starts_with('n');
            let tail = if neg { &tail[1..] } else { tail };
            let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                return None;
            }
            let value: i64 = digits.parse().ok()?;
            let value = if neg { -value } else { value };
            r = tail[digits.len()..].strip_prefix('E')?;
            targs.push(match ty {
                Some(t) => TArg::Enum(t, value),
                None => TArg::Int(value),
            });
        }
    }
    Some(Mangled { base, targs })
}

fn source_name(s: &str) -> Option<(String, &str)> {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n: usize = digits.parse().ok()?;
    let rest = &s[digits.len()..];
    (rest.len() >= n && n > 0).then(|| (rest[..n].to_string(), &rest[n..]))
}

// ---- kernel table ----

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Kernel {
    Mmvq { ty: u32, ncols: u32 },
    QuantizeQ81,
    /// `mul_mat_q<Q8_0>` (any mmq_x / need_check), non-MoE only.
    MmqQ80,
    /// `mul_mat_q_stream_k_fixup<Q8_0>`: skipped after a native `MmqQ80` launch.
    MmqFixupQ80,
}

fn ptr() -> Param {
    Param::GlobalPtr
}
fn u3() -> Param {
    Param::Words { words: 3, align: 4, trim: 0 }
}

/// The measured default: with weights loaded once per row the native mmvq beat or tied the
/// translated PTX on the Arc at every type and ncols 1..8 (Q8_0 1.02-1.88x, Q4_0 1.02-1.44x,
/// Q4_K 1.65-2.23x, Q6_K 1.65-2.54x; one dense run, hw_subst). quantize_q8_1 is byte-exact and
/// cheap. mmq (Q8_0) and its stream-k fixup are native in `auto` too: 330 of 330 shadow checks exact on the Arc, prefill about 4x faster; switch off with `auto,-mmq`.
pub(crate) fn auto_ok(kernel: Kernel) -> bool {
    match kernel {
        Kernel::Mmvq { .. } | Kernel::QuantizeQ81 | Kernel::MmqQ80 | Kernel::MmqFixupQ80 => true,
    }
}

/// Parameter layout of `mul_mat_vec_q` (23 words once the `uint3`s are expanded).
fn mmvq_layout() -> Vec<Param> {
    use Param::U32;
    vec![ptr(), ptr(), ptr(), ptr(), U32, u3(), U32, U32, U32, u3(), U32, U32, U32, u3(), U32, U32, U32]
}

fn quantize_layout() -> Vec<Param> {
    use Param::{U32, U64};
    vec![ptr(), ptr(), U64, U64, U64, U64, U64, U32, u3()]
}

/// `mul_mat_q`: six pointers (unused ones may be typed u64) and 17 words.
fn mmq_layout() -> Vec<Param> {
    let mut v = vec![ptr(); 6];
    v.extend(vec![Param::U32; 17]);
    v
}

/// `mul_mat_q_stream_k_fixup`: four pointers and nine words.
fn mmq_fixup_layout() -> Vec<Param> {
    let mut v = vec![ptr(); 4];
    v.extend(vec![Param::U32; 9]);
    v
}

/// FNV-1a over the debug rendering of the layout: a stable key for logs and the registry.
pub(crate) fn layout_hash(params: &[Param]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in format!("{params:?}").bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Exact parameter match, except that `mul_mat_vec_q`'s `ids` slot may be typed `u64` by the
/// translator when the instantiation never dereferences it (ncols_dst > 1): same 8 bytes.
fn layout_matches(kernel: Kernel, params: &[Param], expected: &[Param]) -> bool {
    if params == expected {
        return true;
    }
    if matches!(kernel, Kernel::MmqQ80 | Kernel::MmqFixupQ80) {
        // unused pointer slots (ids, expert bounds, ...) may be typed u64
        let lead = if kernel == Kernel::MmqQ80 { 6 } else { 4 };
        return params.len() == expected.len()
            && params.iter().zip(expected).enumerate().all(|(i, (a, b))| {
                a == b || (i < lead && matches!(a, Param::U64 | Param::GlobalPtr))
            });
    }
    matches!(kernel, Kernel::Mmvq { ncols, .. } if ncols > 1)
        && params.len() == expected.len()
        && params.get(2) == Some(&Param::U64)
        && params.iter().zip(expected).enumerate().all(|(i, (a, b))| i == 2 || a == b)
}

pub(crate) fn type_name(ty: u32) -> Option<&'static str> {
    Some(match ty {
        refs::Q4_0 => "q4_0",
        refs::Q8_0 => "q8_0",
        refs::Q4_K => "q4_k",
        refs::Q6_K => "q6_k",
        _ => return None,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Plan {
    pub kernel: Kernel,
    /// Stable id of the native module (kernel + variant knobs).
    pub key: String,
    /// `CUDA_FOR_INTEL_SUBST_NWARPS` override for mmvq (experiments; changes only cross-warp add order).
    pub nwarps: Option<u32>,
    pub layout_hash: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Decision {
    Substitute(Plan),
    /// Recognised name but the parameter layout differs: keep PTX, log why.
    Mismatch(String),
    NotApplicable,
}

pub(crate) fn decide(name: &str, params: &[Param], switch: &Switch) -> Decision {
    if switch.is_none() {
        return Decision::NotApplicable;
    }
    let Some(m) = demangle(name) else { return Decision::NotApplicable };
    let (kernel, expected, family, ty) = match (m.base.as_str(), m.targs.as_slice()) {
        ("mul_mat_vec_q", [TArg::Enum(e, ty), TArg::Int(ncols)]) if e == "ggml_type" => {
            let (Some(tn), true) = (type_name(*ty as u32), (1..=8).contains(ncols)) else {
                return Decision::NotApplicable;
            };
            (Kernel::Mmvq { ty: *ty as u32, ncols: *ncols as u32 }, mmvq_layout(), "mmvq", Some(tn))
        }
        ("mul_mat_q", [TArg::Enum(e, 8), TArg::Int(_), TArg::Int(_)]) if e == "ggml_type" => {
            (Kernel::MmqQ80, mmq_layout(), "mmq", Some("q8_0"))
        }
        ("mul_mat_q_stream_k_fixup", [TArg::Enum(e, 8), TArg::Int(_), TArg::Int(_)]) if e == "ggml_type" => {
            (Kernel::MmqFixupQ80, mmq_fixup_layout(), "mmq", Some("q8_0"))
        }
        ("quantize_q8_1", []) => (Kernel::QuantizeQ81, quantize_layout(), "quantize_q8_1", None),
        _ => return Decision::NotApplicable,
    };
    if switch.denies(family, ty) {
        return Decision::NotApplicable;
    }
    if !(switch.allows(family, ty) || (switch.auto && auto_ok(kernel))) {
        return Decision::NotApplicable;
    }
    if !layout_matches(kernel, params, &expected) {
        return Decision::Mismatch(format!(
            "{name}: parameter layout {:016x} differs from expected {:016x}",
            layout_hash(params),
            layout_hash(&expected)
        ));
    }
    let nwarps = match kernel {
        Kernel::Mmvq { .. } => std::env::var("CUDA_FOR_INTEL_SUBST_NWARPS").ok().and_then(|v| v.parse::<u32>().ok()),
        _ => None,
    }
    .filter(|w| w.is_power_of_two() && (1..=32).contains(w));
    let key = match kernel {
        Kernel::Mmvq { ty, ncols } => format!("mmvq:t{ty}:c{ncols}:w{}", nwarps.unwrap_or(0)),
        Kernel::QuantizeQ81 => "quantize_q8_1".to_string(),
        Kernel::MmqQ80 => "mmq:q8_0".to_string(),
        Kernel::MmqFixupQ80 => "mmq_fixup:q8_0".to_string(),
    };
    Decision::Substitute(Plan { kernel, key, nwarps, layout_hash: layout_hash(params) })
}

impl Plan {
    /// Warps per block the native kernel is built for, given ggml's default.
    pub(crate) fn nwarps_for(&self, ggml_default: u32) -> u32 {
        self.nwarps.unwrap_or(ggml_default)
    }

    pub(crate) fn entry_name(&self) -> String {
        match self.kernel {
            Kernel::Mmvq { ty, ncols } => {
                format!("cfi_mmvq_t{}_c{}_w{}", ty, ncols, self.nwarps_for(refs::Mmvq::ggml_nwarps(ncols as usize) as u32))
            }
            Kernel::QuantizeQ81 => "cfi_quantize_q8_1".to_string(),
            Kernel::MmqQ80 => "cfi_mmq_q8_0".to_string(),
            Kernel::MmqFixupQ80 => String::new(),
        }
    }

    /// Generates the SPIR-V; `dot` selects SPV_KHR_integer_dot_product over scalar emulation.
    /// A generator panic is reported as an error so the caller can fall back.
    pub(crate) fn build(&self, dot: bool) -> Result<Vec<u32>, String> {
        let plan = self.clone();
        std::panic::catch_unwind(move || match plan.kernel {
            Kernel::Mmvq { ty, ncols } => {
                let nwarps = plan.nwarps_for(refs::Mmvq::ggml_nwarps(ncols as usize) as u32);
                mmvq::build(mmvq::Cfg { ty, ncols, nwarps }, dot)
            }
            Kernel::QuantizeQ81 => quantize::build(dot),
            Kernel::MmqQ80 => mmq::build(dot),
            Kernel::MmqFixupQ80 => Vec::new(),
        })
        .map_err(|_| format!("native generator for {} panicked", self.key))
    }

    /// Maps the CUDA launch geometry to the native kernel's, or `None` when the launch does not
    /// have the shape the native kernel was written for (the translated kernel then runs).
    pub(crate) fn map_geometry(&self, grid: [u32; 3], block: [u32; 3]) -> Option<([u32; 3], [u32; 3])> {
        match self.kernel {
            Kernel::Mmvq { ncols, .. } => {
                let stock = refs::Mmvq::ggml_nwarps(ncols as usize) as u32;
                if block != [32, stock, 1] {
                    return None;
                }
                Some((grid, [32, self.nwarps_for(stock), 1]))
            }
            Kernel::QuantizeQ81 => (block == [256, 1, 1]).then_some((grid, block)),
            Kernel::MmqQ80 | Kernel::MmqFixupQ80 => None, // needs the argument values: `map_launch`
        }
    }

    /// Like `map_geometry` for kernels whose grid derives from argument values (`vals` holds
    /// each launch word, pointers as addresses).
    pub(crate) fn map_launch(&self, grid: [u32; 3], block: [u32; 3], vals: &[u64]) -> Option<([u32; 3], [u32; 3])> {
        match self.kernel {
            Kernel::MmqQ80 => {
                // stock: 32 x 8 threads; MoE (ids_dst / expert_bounds) stays on the translated kernel
                if block != [32, 8, 1] || vals.len() != 23 || vals[2] != 0 || vals[3] != 0 {
                    return None;
                }
                let (nrows, ncols_dst, nch, nsamples) = (vals[7], vals[8], vals[13], vals[18]);
                if nrows == 0 || ncols_dst == 0 || nch == 0 || nsamples == 0 || vals[6] % 128 != 0 {
                    return None;
                }
                let g = [
                    nrows.div_ceil(u64::from(mmq::TILE_I)) as u32,
                    ncols_dst.div_ceil(u64::from(mmq::TILE_J)) as u32,
                    (nch * nsamples) as u32,
                ];
                let _ = grid;
                Some((g, [16, 16, 1]))
            }
            _ => self.map_geometry(grid, block),
        }
    }
}

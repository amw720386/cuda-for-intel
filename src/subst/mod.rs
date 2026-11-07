//! Kernel substitution: hand-written Intel-native SPIR-V replacements for known ggml-cuda kernels.
//!
//! A kernel is recognised by its Itanium-mangled name (template arguments included) *and* the
//! layout of its parameter list; anything else keeps the translated PTX. The driver glue
//! (`cuda/subst_hook.rs`) builds the native module lazily, validates it once against a host
//! reference on the real device, and falls back to the translated kernel on any failure.
//!
//! Selection: `CUDA_FOR_INTEL_SUBST=none|all|<comma list>` (default `none`). List items are a
//! family (`mmvq`, `quantize_q8_1`) optionally narrowed to a type (`mmvq:q8_0`, `mmvq:q4_k`).
use crate::ptx::Param;

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
    items: Vec<String>,
}

impl Switch {
    pub(crate) fn parse(text: &str) -> Self {
        let mut s = Self::default();
        for item in text.split(',').map(|i| i.trim().to_ascii_lowercase()).filter(|i| !i.is_empty()) {
            match item.as_str() {
                "none" | "0" | "off" => {
                    s.all = false;
                    s.items.clear();
                }
                "all" | "1" | "on" => s.all = true,
                _ => s.items.push(item),
            }
        }
        s
    }

    /// Reads the variable on every call (cheap, and lets tests and the hardware comparison load
    /// the same kernel with and without substitution in one process).
    pub(crate) fn from_env() -> Self {
        Self::parse(&std::env::var("CUDA_FOR_INTEL_SUBST").unwrap_or_default())
    }

    pub(crate) fn allows(&self, family: &str, ty: Option<&str>) -> bool {
        self.all
            || self.items.iter().any(|i| {
                i == family || ty.is_some_and(|t| i.strip_prefix(family).and_then(|r| r.strip_prefix(':')) == Some(t))
            })
    }

    pub(crate) fn is_none(&self) -> bool {
        !self.all && self.items.is_empty()
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
}

fn ptr() -> Param {
    Param::GlobalPtr
}
fn u3() -> Param {
    Param::Words { words: 3, align: 4 }
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
        ("quantize_q8_1", []) => (Kernel::QuantizeQ81, quantize_layout(), "quantize_q8_1", None),
        _ => return Decision::NotApplicable,
    };
    if !switch.allows(family, ty) {
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
        Kernel::QuantizeQ81 => None,
    }
    .filter(|w| w.is_power_of_two() && (1..=32).contains(w));
    let key = match kernel {
        Kernel::Mmvq { ty, ncols } => format!("mmvq:t{ty}:c{ncols}:w{}", nwarps.unwrap_or(0)),
        Kernel::QuantizeQ81 => "quantize_q8_1".to_string(),
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
        }
    }
}

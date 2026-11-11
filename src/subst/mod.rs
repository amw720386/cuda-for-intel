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

pub(crate) mod fattn;
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

/// Native FlashAttention geometry, or `None` when the launch uses a feature the native kernel does
/// not implement (sinks, ALiBi, softcap, other head size, unaligned K/V) or has fewer queries than
/// one tile.
fn fattn_geometry(d: u32, v: &[u64]) -> Option<([u32; 3], [u32; 3])> {
    if v.len() != 37 {
        return None;
    }
    let f32_is_zero = |x: u64| (x as u32) & 0x7fff_ffff == 0;
    let (ne00, ne01, ne02, ne03) = (v[14], v[15], v[16], v[17]);
    let (ne11, ne12) = (v[22], v[23]);
    // Below a full tile (decode, GQA-batched decode) the stock kernel's split-KV parallelism wins:
    // one 16-query tile per head leaves most of the GPU idle (Llama-3.2-1B decode: 287 us native
    // vs 176 us translated).
    let ok = ne01 >= u64::from(fattn::qt())
        && v[4] == 0
        && f32_is_zero(v[9])
        && f32_is_zero(v[13])
        && ne00 == u64::from(d)
        && ne01 > 0
        && ne02 > 0
        && ne03 > 0
        && ne11 > 0
        && ne12 > 0
        && ne02 % ne12 == 0
        && (v[1] | v[2] | v[25] | v[28]) % 4 == 0
        && (v[0] | v[18] | v[19] | v[20]) % 4 == 0
        && (v[3] == 0 || ((v[3] | v[34]) % 2 == 0 && v[33] > 0));
    ok.then(|| ([ne01.div_ceil(u64::from(fattn::qt())) as u32, ne02 as u32, ne03 as u32], [32, fattn::nw(), 1]))
}

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

/// Which `mul_mat_vec_q` signature a kernel has.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum MmvqAbi {
    /// Ollama v0.12.6: `<type, ncols_dst>`, 23 launch words.
    Legacy,
    /// Ollama 0.40 (current ggml): `<type, ncols_dst, has_fusion, small_k, halve_iters>` with an
    /// 80-byte `ggml_cuda_mm_fusion_args_device` (20 words) after `ids` and a trailing `ids_stride`:
    /// 44 launch words.
    /// Only `has_fusion = false` is substituted; the hook drops the fusion words and `ids_stride`,
    /// so the native kernels see the legacy 23-word layout.
    Fused { small_k: bool },
}

/// Launch words of the 0.40 `mul_mat_vec_q` and the legacy words they map to.
pub(crate) const MMVQ_FUSED_WORDS: usize = 44;

impl MmvqAbi {
    /// Warps per block ggml compiled the stock kernel for (its `__launch_bounds__`).
    pub(crate) fn stock_nwarps(self, ty: u32, ncols: u32) -> u32 {
        match self {
            Self::Legacy => refs::Mmvq::ggml_nwarps(ncols as usize) as u32,
            // MMVQ_PARAMETERS_TURING (the compute_75 PTX that a reported 7.5 selects); checked
            // against the `.maxntid` of every Ollama 0.40.14 instantiation.
            Self::Fused { .. } => match ncols {
                1 if matches!(ty, 10..=14) => 2, // Q2_K .. Q6_K
                1..=4 => 4,
                5..=8 => 2,
                _ => 1,
            },
        }
    }

    /// Matrix rows per thread block (ggml's `calc_rows_per_block`): `small_k` gives each of the
    /// `nwarps` warps of an ncols_dst-1 block its own row.
    pub(crate) fn rows_per_block(self, ty: u32, ncols: u32) -> u32 {
        match self {
            Self::Fused { small_k: true } if ncols == 1 => self.stock_nwarps(ty, ncols),
            _ if ncols == 1 => 1,
            _ => 2,
        }
    }

    /// The legacy 23 launch words: the three leading pointers, then `dst` onward without the
    /// fusion struct words (3..23) and the trailing `ids_stride`.
    pub(crate) fn legacy_words<T: Copy>(self, words: &[T]) -> Option<Vec<T>> {
        match self {
            Self::Legacy => (words.len() == 23).then(|| words.to_vec()),
            Self::Fused { .. } => (words.len() == MMVQ_FUSED_WORDS)
                .then(|| words[..3].iter().chain(&words[23..MMVQ_FUSED_WORDS - 1]).copied().collect()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Kernel {
    Mmvq { ty: u32, ncols: u32, abi: MmvqAbi },
    QuantizeQ81,
    /// `mul_mat_q<ty>` (any mmq_x / need_check), non-MoE only.
    Mmq { ty: u32 },
    /// `mul_mat_q_stream_k_fixup<ty>`: skipped after a native `Mmq` launch.
    MmqFixup { ty: u32 },
    /// `flash_attn_ext_f16<d, d, ..., false, false>` (tensor-core FlashAttention).
    Fattn { d: u32 },
    /// `flash_attn_stream_k_fixup<d, ...>`: skipped after a native `Fattn` launch.
    FattnFixup,
}

fn ptr() -> Param {
    Param::GlobalPtr
}
fn u3() -> Param {
    Param::Words { words: 3, align: 4, trim: 0 }
}

/// mmq types enabled in `auto` (the rest need an explicit `mmq:<type>` / `mmq` / `all`); raised
/// only where hw_subst shows exact shadow checks and a speedup of at least 2x.
const AUTO_MMQ: [u32; 6] = [refs::Q8_0, refs::Q4_0, refs::Q5_0, refs::Q4_K, refs::Q5_K, refs::Q6_K];

/// The measured default: with weights loaded once per row the native mmvq beat or tied the
/// translated PTX on the Arc at every type and ncols 1..8 (Q8_0 1.02-1.88x, Q4_0 1.02-1.44x,
/// Q4_K 1.65-2.23x, Q6_K 1.65-2.54x; one dense run, hw_subst). quantize_q8_1 is byte-exact and
/// cheap. mmq (Q8_0) and its stream-k fixup are native in `auto` too: 330 of 330 shadow checks exact on the Arc, prefill about 4x faster; switch off with `auto,-mmq`.
pub(crate) fn auto_ok(kernel: Kernel) -> bool {
    match kernel {
        Kernel::Mmvq { .. } | Kernel::QuantizeQ81 => true,
        Kernel::Mmq { ty } | Kernel::MmqFixup { ty } => AUTO_MMQ.contains(&ty),
        // FA-on prefill +81-88 % end to end on the B390 with CPU-identical greedy text
        // (docs/hw-logs/perf-mmvq-2026-11-09); `auto,-fattn` switches it off
        Kernel::Fattn { .. } | Kernel::FattnFixup => true,
    }
}

/// Parameter layout of `mul_mat_vec_q` (23 words once the `uint3`s are expanded).
fn mmvq_layout() -> Vec<Param> {
    use Param::U32;
    vec![ptr(), ptr(), ptr(), ptr(), U32, u3(), U32, U32, U32, u3(), U32, U32, U32, u3(), U32, U32, U32]
}

/// Parameter layout of the 0.40 `mul_mat_vec_q` (see `MmvqAbi::Fused`).
fn mmvq_fused_layout() -> Vec<Param> {
    use Param::U32;
    let fusion = Param::Words { words: 20, align: 8, trim: 0 };
    vec![ptr(), ptr(), ptr(), fusion, ptr(), U32, u3(), U32, U32, U32, u3(), U32, U32, U32, u3(), U32, U32, U32, U32]
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

/// `flash_attn_ext_f16`: 8 pointers, 29 scalars (three of them int64).
fn fattn_layout() -> Vec<Param> {
    fattn::arg_sizes().iter().map(|&n| if n == 8 { Param::U64 } else { Param::U32 }).collect()
}

/// `flash_attn_stream_k_fixup`: dst, dst_fixup, ne01, ne02, ne03, ne11.
fn fattn_fixup_layout() -> Vec<Param> {
    vec![Param::U64, Param::U64, Param::U32, Param::U32, Param::U32, Param::U32]
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
fn param_bytes(p: &Param) -> Option<usize> {
    match p {
        Param::U32 | Param::F32 => Some(4),
        Param::U64 | Param::GlobalPtr => Some(8),
        _ => None,
    }
}

fn layout_matches(kernel: Kernel, params: &[Param], expected: &[Param]) -> bool {
    if matches!(kernel, Kernel::Fattn { .. } | Kernel::FattnFixup) {
        // pointers may be typed u64 and floats u32 by the translator: compare word sizes
        let sizes: Vec<Option<usize>> = params.iter().map(param_bytes).collect();
        let want: Vec<Option<usize>> = expected.iter().map(param_bytes).collect();
        return sizes == want;
    }
    if params == expected {
        return true;
    }
    if matches!(kernel, Kernel::Mmq { .. } | Kernel::MmqFixup { .. }) {
        // unused pointer slots (ids, expert bounds, ...) may be typed u64
        let lead = if matches!(kernel, Kernel::Mmq { .. }) { 6 } else { 4 };
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

fn mmvq_supported(ty: u32) -> bool {
    matches!(ty, refs::Q4_0 | refs::Q8_0 | refs::Q4_K | refs::Q6_K)
}

pub(crate) fn type_name(ty: u32) -> Option<&'static str> {
    Some(match ty {
        refs::Q4_0 => "q4_0",
        refs::Q5_0 => "q5_0",
        refs::Q8_0 => "q8_0",
        refs::Q4_K => "q4_k",
        refs::Q5_K => "q5_k",
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
    /// Template tile width (`mmq_x`) of mul_mat_q variants; 0 otherwise.
    pub aux: u32,
    /// Decode-fast mmvq (`CUDA_FOR_INTEL_MMVQ_FAST`, ncols_dst 1).
    pub fast: Option<mmvq::Fast>,
}

/// `CUDA_FOR_INTEL_MMVQ_FAST`: `0`/`off` keeps the stock-order native mmvq; `1`/`on` selects the
/// decode-fast variant with the per-type default; `<rows>x<warps>x<unroll>` (or comma separated)
/// picks one shape for every type; a trailing `w` selects the whole-dword group loads (Q8_0, Q4_K,
/// Q6_K; ignored for Q4_0).
pub(crate) fn mmvq_fast_from(text: Option<&str>, ty: u32) -> Option<mmvq::Fast> {
    let text = text?.trim().to_ascii_lowercase();
    match text.as_str() {
        "" | "0" | "off" | "none" => None,
        "1" | "on" | "auto" => Some(mmvq_fast_default(ty)),
        _ => {
            let (body, wide) = match text.strip_suffix('w') {
                Some(b) => (b, true),
                None => (text.as_str(), false),
            };
            let v: Vec<Option<u32>> = body.split([',', 'x']).map(|v| v.trim().parse::<u32>().ok()).collect();
            let [Some(rows), Some(warps), Some(unroll)] = v[..] else { return None };
            let f = mmvq::Fast { rows, warps, unroll, wide: wide && matches!(ty, refs::Q8_0 | refs::Q4_K | refs::Q6_K) };
            (rows.is_power_of_two() && warps.is_power_of_two() && rows * warps <= 32 && warps <= 8 && (1..=8).contains(&unroll))
                .then_some(f)
        }
    }
}

/// Per-type default shape of the decode-fast mmvq, from `hw_subst_mmvq_decode_shapes` on the Arc
/// B390: whole-dword group loads (Q8_0 / Q4_K / Q6_K) with 4 rows per workgroup and 2 accumulators
/// reach 106-109 GB/s on the output matrices (read ceiling 114 GB/s, `hw_perf_probe`) against
/// 36-53 GB/s for the stock-order native kernel. Q4_0 has no wide form.
pub(crate) fn mmvq_fast_default(ty: u32) -> mmvq::Fast {
    let wide = matches!(ty, refs::Q8_0 | refs::Q4_K | refs::Q6_K);
    mmvq::Fast { rows: 4, warps: 1, unroll: 2, wide }
}

fn mmvq_fast_env(ty: u32) -> Option<mmvq::Fast> {
    mmvq_fast_from(std::env::var("CUDA_FOR_INTEL_MMVQ_FAST").ok().as_deref(), ty)
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
            let (Some(tn), true) =
                (type_name(*ty as u32).filter(|_| mmvq_supported(*ty as u32)), (1..=8).contains(ncols))
            else {
                return Decision::NotApplicable;
            };
            (Kernel::Mmvq { ty: *ty as u32, ncols: *ncols as u32, abi: MmvqAbi::Legacy }, mmvq_layout(), "mmvq", Some(tn))
        }
        // halve_iters only changes the warp count of the GB10 table, never of TURING.
        ("mul_mat_vec_q", [TArg::Enum(e, ty), TArg::Int(ncols), TArg::Int(fusion), TArg::Int(small_k), TArg::Int(_halve)])
            if e == "ggml_type" =>
        {
            let (Some(tn), true, 0) =
                (type_name(*ty as u32).filter(|_| mmvq_supported(*ty as u32)), (1..=8).contains(ncols), *fusion)
            else {
                return Decision::NotApplicable;
            };
            let abi = MmvqAbi::Fused { small_k: *small_k != 0 };
            (Kernel::Mmvq { ty: *ty as u32, ncols: *ncols as u32, abi }, mmvq_fused_layout(), "mmvq", Some(tn))
        }
        ("mul_mat_q", [TArg::Enum(e, ty), TArg::Int(_), TArg::Int(_)]) if e == "ggml_type" && mmq::supported(*ty as u32) => {
            (Kernel::Mmq { ty: *ty as u32 }, mmq_layout(), "mmq", type_name(*ty as u32))
        }
        ("mul_mat_q_stream_k_fixup", [TArg::Enum(e, ty), TArg::Int(_), TArg::Int(_)])
            if e == "ggml_type" && mmq::supported(*ty as u32) =>
        {
            (Kernel::MmqFixup { ty: *ty as u32 }, mmq_fixup_layout(), "mmq", type_name(*ty as u32))
        }
        ("quantize_q8_1", []) => (Kernel::QuantizeQ81, quantize_layout(), "quantize_q8_1", None),
        (
            "flash_attn_ext_f16",
            [TArg::Int(dkq), TArg::Int(dv), TArg::Int(_), TArg::Int(_), TArg::Int(_), TArg::Int(_), TArg::Int(0), TArg::Int(0)],
        ) if dkq == dv && fattn::supported_d(*dkq as u32) => {
            (Kernel::Fattn { d: *dkq as u32 }, fattn_layout(), "fattn", None)
        }
        ("flash_attn_stream_k_fixup", [TArg::Int(_), TArg::Int(_), TArg::Int(_)]) => {
            (Kernel::FattnFixup, fattn_fixup_layout(), "fattn", None)
        }
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
    let fast = match kernel {
        Kernel::Mmvq { ty, ncols: 1, .. } => mmvq_fast_env(ty),
        _ => None,
    };
    let key = match kernel {
        Kernel::Mmvq { ty, ncols, .. } if fast.is_some() => {
            let f = fast.unwrap();
            format!("mmvq:t{ty}:c{ncols}:fast{}x{}x{}{}", f.rows, f.warps, f.unroll, if f.wide { "w" } else { "" })
        }
        Kernel::Mmvq { ty, ncols, abi } => {
            format!("mmvq:t{ty}:c{ncols}:w{}:r{}", nwarps.unwrap_or(0), abi.rows_per_block(ty, ncols))
        }
        Kernel::QuantizeQ81 => "quantize_q8_1".to_string(),
        Kernel::Mmq { ty } => format!("mmq:t{ty}"),
        Kernel::MmqFixup { ty } => format!("mmq_fixup:t{ty}"),
        Kernel::Fattn { d } => format!("fattn:d{d}:w{}", fattn::nw()),
        Kernel::FattnFixup => "fattn_fixup".to_string(),
    };
    let aux = match m.targs.as_slice() {
        [_, TArg::Int(x), _] if matches!(kernel, Kernel::Mmq { .. } | Kernel::MmqFixup { .. }) => *x as u32,
        _ => 0,
    };
    Decision::Substitute(Plan { kernel, key, nwarps, layout_hash: layout_hash(params), aux, fast })
}

impl Plan {
    /// Warps per block the native kernel is built for, given ggml's default.
    pub(crate) fn nwarps_for(&self, ggml_default: u32) -> u32 {
        self.nwarps.unwrap_or(ggml_default)
    }

    pub(crate) fn entry_name(&self) -> String {
        match self.kernel {
            Kernel::Mmvq { ty, .. } if self.fast.is_some() => {
                mmvq::fast_entry_name(ty, self.fast.unwrap())
            }
            Kernel::Mmvq { ty, ncols, abi } => format!(
                "cfi_mmvq_t{}_c{}_w{}_r{}",
                ty,
                ncols,
                self.nwarps_for(abi.stock_nwarps(ty, ncols)),
                abi.rows_per_block(ty, ncols)
            ),
            Kernel::QuantizeQ81 => "cfi_quantize_q8_1".to_string(),
            Kernel::Mmq { ty } => mmq::entry_name(ty),
            Kernel::MmqFixup { .. } | Kernel::FattnFixup => String::new(),
            Kernel::Fattn { d } => fattn::entry_name(d),
        }
    }

    /// Generates the SPIR-V; `dot` selects SPV_KHR_integer_dot_product over scalar emulation.
    /// A generator panic is reported as an error so the caller can fall back.
    pub(crate) fn build(&self, dot: bool) -> Result<Vec<u32>, String> {
        let plan = self.clone();
        std::panic::catch_unwind(move || match plan.kernel {
            Kernel::Mmvq { ty, .. } if plan.fast.is_some() => {
                mmvq::build_fast(ty, plan.fast.unwrap(), dot)
            }
            Kernel::Mmvq { ty, ncols, abi } => {
                let nwarps = plan.nwarps_for(abi.stock_nwarps(ty, ncols));
                mmvq::build(mmvq::Cfg { ty, ncols, nwarps, rpb: abi.rows_per_block(ty, ncols) }, dot)
            }
            Kernel::QuantizeQ81 => quantize::build(dot),
            Kernel::Mmq { ty } => mmq::build(ty, dot),
            Kernel::MmqFixup { .. } | Kernel::FattnFixup => Vec::new(),
            Kernel::Fattn { d } => fattn::build(d),
        })
        .map_err(|_| format!("native generator for {} panicked", self.key))
    }

    /// Maps the CUDA launch geometry to the native kernel's, or `None` when the launch does not
    /// have the shape the native kernel was written for (the translated kernel then runs).
    pub(crate) fn map_geometry(&self, grid: [u32; 3], block: [u32; 3]) -> Option<([u32; 3], [u32; 3])> {
        match self.kernel {
            Kernel::Mmvq { ty, ncols, abi } => {
                let stock = abi.stock_nwarps(ty, ncols);
                if block != [32, stock, 1] {
                    return None;
                }
                if let Some(f) = self.fast {
                    // ncols_dst 1 with one row per block: grid.x == nrows (more rows per block
                    // go through `map_launch`, which knows the row count)
                    if abi.rows_per_block(ty, ncols) != 1 {
                        return None;
                    }
                    return Some(([grid[0].div_ceil(f.rows), grid[1], grid[2]], [32, f.rows * f.warps, 1]));
                }
                Some((grid, [32, self.nwarps_for(stock), 1]))
            }
            Kernel::QuantizeQ81 => (block == [256, 1, 1]).then_some((grid, block)),
            Kernel::Mmq { .. } | Kernel::MmqFixup { .. } | Kernel::Fattn { .. } | Kernel::FattnFixup => None, // needs the argument values: `map_launch`
        }
    }

    /// A launch word the native kernel takes after the stock parameters (`nrows` for the
    /// decode-fast mmvq, from the stock grid).
    pub(crate) fn extra_word(&self, stock_grid: [u32; 3], vals: &[u64]) -> Option<u32> {
        match self.kernel {
            Kernel::Mmvq { .. } if self.fast.is_some() => self.mmvq_rows(stock_grid, vals),
            _ => None,
        }
    }

    /// Matrix rows an ncols_dst-1 `mul_mat_vec_q` launch covers: `grid.x` blocks of
    /// `rows_per_block` rows, bounded by `stride_col_dst` (legacy word 10, the row count of a
    /// one-column destination) when a block holds several rows.
    pub(crate) fn mmvq_rows(&self, grid: [u32; 3], vals: &[u64]) -> Option<u32> {
        let Kernel::Mmvq { ty, ncols: 1, abi } = self.kernel else { return None };
        match abi.rows_per_block(ty, 1) {
            1 => Some(grid[0]),
            rpb => {
                let bound = u32::try_from(*vals.get(10)?).ok()?;
                Some(grid[0].checked_mul(rpb)?.min(bound))
            }
        }
    }

    /// Like `map_geometry` for kernels whose grid derives from argument values (`vals` holds
    /// each launch word, pointers as addresses).
    pub(crate) fn map_launch(&self, grid: [u32; 3], block: [u32; 3], vals: &[u64]) -> Option<([u32; 3], [u32; 3])> {
        match self.kernel {
            Kernel::Mmq { .. } => {
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
            Kernel::Fattn { d } => {
                let _ = (grid, block);
                fattn_geometry(d, vals)
            }
            Kernel::Mmvq { ty, ncols, abi } if self.fast.is_some() && abi.rows_per_block(ty, ncols) != 1 => {
                let f = self.fast?;
                if block != [32, abi.stock_nwarps(ty, ncols), 1] {
                    return None;
                }
                let rows = self.mmvq_rows(grid, vals)?;
                Some(([rows.div_ceil(f.rows), grid[1], grid[2]], [32, f.rows * f.warps, 1]))
            }
            _ => self.map_geometry(grid, block),
        }
    }
}

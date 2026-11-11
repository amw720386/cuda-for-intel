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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Kernel {
    Mmvq { ty: u32, ncols: u32 },
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

/// `CUDA_FOR_INTEL_MMVQ_FAST`: unset, empty, `1`/`on`/`true`/`yes`/`auto` select the decode-fast
/// variant with the per-type default (the default since B390 decode went 32.3 -> 44.2 tok/s FA-off
/// and 37.7 -> 56.5 FA-on, docs/hw-logs/perf-mmvq-2026-11-09/owner-validation-2026-11-10.md);
/// `0`/`off`/`false`/`no`/`none` keeps the stock-order native mmvq;
/// `<rows>x<warps>x<unroll>` (or comma separated) picks one shape for every type; a trailing `w`
/// selects the whole-dword group loads (Q8_0, Q4_K, Q6_K; ignored for Q4_0). An unparsable or
/// out-of-range shape keeps the stock-order kernel (`mmvq_fast_env` logs it once).
pub(crate) fn mmvq_fast_from(text: Option<&str>, ty: u32) -> Option<mmvq::Fast> {
    let text = text.unwrap_or("").trim().to_ascii_lowercase();
    match text.as_str() {
        t if mmvq_fast_is_opt_out(t) => None,
        "" | "1" | "on" | "true" | "yes" | "auto" => Some(mmvq_fast_default(ty)),
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

/// Per-context native module key of an mmvq plan: every knob that changes the generated SPIR-V
/// (fast shape including `wide`, or the stock-order warp count) is part of it, so two variants
/// never share a cached module.
pub(crate) fn mmvq_key(ty: u32, ncols: u32, fast: Option<mmvq::Fast>, nwarps: Option<u32>) -> String {
    match fast {
        Some(f) => format!("mmvq:t{ty}:c{ncols}:fast{}x{}x{}{}", f.rows, f.warps, f.unroll, if f.wide { "w" } else { "" }),
        None => format!("mmvq:t{ty}:c{ncols}:w{}", nwarps.unwrap_or(0)),
    }
}

fn mmvq_fast_env(ty: u32) -> Option<mmvq::Fast> {
    let text = std::env::var("CUDA_FOR_INTEL_MMVQ_FAST").ok();
    let fast = mmvq_fast_from(text.as_deref(), ty);
    if fast.is_none() && text.as_deref().is_some_and(|t| !mmvq_fast_is_opt_out(t)) {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::cuda::export_log_always(format_args!(
                "subst: CUDA_FOR_INTEL_MMVQ_FAST={:?} not understood; using the stock-order mmvq",
                text.unwrap_or_default()
            ));
        }
    }
    fast
}

/// The explicit opt-out values of `CUDA_FOR_INTEL_MMVQ_FAST` (anything else that yields no shape is
/// a typo, logged by `mmvq_fast_env`).
pub(crate) fn mmvq_fast_is_opt_out(text: &str) -> bool {
    matches!(text.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no" | "none")
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
            (Kernel::Mmvq { ty: *ty as u32, ncols: *ncols as u32 }, mmvq_layout(), "mmvq", Some(tn))
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
        Kernel::Mmvq { ty, ncols: 1 } => mmvq_fast_env(ty),
        _ => None,
    };
    let key = match kernel {
        Kernel::Mmvq { ty, ncols } => mmvq_key(ty, ncols, fast, nwarps),
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
            Kernel::Mmvq { ty, ncols } => {
                format!("cfi_mmvq_t{}_c{}_w{}", ty, ncols, self.nwarps_for(refs::Mmvq::ggml_nwarps(ncols as usize) as u32))
            }
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
            Kernel::Mmvq { ty, ncols } => {
                let nwarps = plan.nwarps_for(refs::Mmvq::ggml_nwarps(ncols as usize) as u32);
                mmvq::build(mmvq::Cfg { ty, ncols, nwarps }, dot)
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
            Kernel::Mmvq { ncols, .. } => {
                let stock = refs::Mmvq::ggml_nwarps(ncols as usize) as u32;
                if block != [32, stock, 1] {
                    return None;
                }
                if let Some(f) = self.fast {
                    // stock ncols_dst 1: one row per block, grid.x == nrows
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
    pub(crate) fn extra_word(&self, stock_grid: [u32; 3]) -> Option<u32> {
        match self.kernel {
            Kernel::Mmvq { .. } if self.fast.is_some() => Some(stock_grid[0]),
            _ => None,
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
            _ => self.map_geometry(grid, block),
        }
    }
}

//! Offline conformance of real ggml-cuda kernels (stock Ollama PTX) through the translator and the
//! SPIR-V simulator. PTX is read from `CFI_GGML_PTX_DIR` (extracted locally, never committed);
//! every test returns early when it is unset.
use super::ggml_ref::*;
use super::sim::{Gpu, Launch, Sim};
use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub(super) fn ptx_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("CFI_GGML_PTX_DIR").map(Into::into)
}

/// `entry name -> blob path` for every `.entry` of every PTX file in the directory.
fn index() -> &'static HashMap<String, std::path::PathBuf> {
    static INDEX: OnceLock<HashMap<String, std::path::PathBuf>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut map = HashMap::new();
        let dir = ptx_dir().expect("CFI_GGML_PTX_DIR");
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            if path.extension().is_none_or(|x| x != "ptx") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix(".entry ") {
                    map.entry(rest.trim_end_matches('(').to_owned()).or_insert_with(|| path.clone());
                }
            }
        }
        map
    })
}

fn prepared(path: &std::path::Path) -> std::sync::Arc<String> {
    static CACHE: OnceLock<Mutex<HashMap<std::path::PathBuf, std::sync::Arc<String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(p) = cache.lock().unwrap().get(path) {
        return p.clone();
    }
    let text = std::fs::read_to_string(path).unwrap();
    let p = std::sync::Arc::new(prepare_source(&text).unwrap());
    cache.lock().unwrap().insert(path.to_owned(), p.clone());
    p
}

/// Like `translate_entry` but keeps module-scope tables (`kvalues_iq4nl`, ...).
fn translate_with_globals(prepared: &str, name: &str) -> Result<Translation, Error> {
    let (entries, vars) = parse::parse_module(prepared)?;
    let entry = entries.into_iter().find(|e| e.name == name).ok_or_else(|| Error::new("PTX entry is absent"))?;
    let used: Vec<_> = vars.device.iter().filter(|v| prepared_mentions(prepared, name, &v.name)).cloned().collect();
    spirv::emit(vec![entry], used)
}

/// Whether the entry body text mentions the global (keeps modules small).
fn prepared_mentions(prepared: &str, entry: &str, var: &str) -> bool {
    let start = prepared.find(&format!(".entry {entry}(")).unwrap_or(0);
    let body = &prepared[start..];
    let end = body.find("\n}\n").unwrap_or(body.len());
    body[..end].contains(var)
}

static DP4A_LOCK: Mutex<()> = Mutex::new(());

/// A translated kernel ready to run in the simulator.
pub(super) struct Kernel {
    pub meta: KernelMetadata,
    pub module: rspirv::dr::Module,
    pub words: Vec<u32>,
    pub has_dot_op: bool,
}

/// Translates a stock kernel with native (`dot == true`) or scalar dp4a lowering.
pub(super) fn load_kernel(name: &str, dot: bool) -> Kernel {
    let path = index().get(name).unwrap_or_else(|| panic!("entry {name} not found in PTX dump"));
    let src = prepared(path);
    let _guard = DP4A_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let before = dot_product_enabled();
    set_dot_product(dot);
    let translated = translate_with_globals(&src, name);
    set_dot_product(before);
    let t = translated.unwrap_or_else(|e| panic!("{name}: {e}"));
    coverage_tests::validate(&t.words);
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let has_dot_op = module
        .all_inst_iter()
        .any(|i| matches!(i.class.opcode, rspirv::spirv::Op::SDot | rspirv::spirv::Op::UDot | rspirv::spirv::Op::SUDot));
    assert!(dot || !has_dot_op, "scalar lowering must not emit dot-product instructions");
    let mut t = t;
    Kernel { meta: t.kernels.remove(0), module, words: t.words, has_dot_op }
}

#[derive(Clone, Debug)]
pub(super) enum Arg {
    P(u64),
    U(u32),
    F(f32),
    W(Vec<u32>),
}

impl Kernel {
    pub fn run(&self, gpu: &mut Gpu, grid: [u32; 3], block: [u32; 3], args: &[Arg]) -> Result<(), String> {
        let mut flat = Vec::new();
        for (a, p) in args.iter().zip(&self.meta.params) {
            match (a, p) {
                (Arg::P(v), Param::GlobalPtr | Param::U64) => flat.push(*v),
                (Arg::U(v), Param::U32) => flat.push(u64::from(*v)),
                (Arg::F(v), Param::F32) => flat.push(u64::from(v.to_bits())),
                (Arg::W(w), Param::Words { words, .. }) => {
                    assert_eq!(w.len(), usize::from(*words));
                    flat.extend(w.iter().map(|&x| u64::from(x)));
                }
                other => panic!("argument/parameter mismatch {other:?}"),
            }
        }
        assert_eq!(args.len(), self.meta.params.len(), "argument count");
        if let Some(max) = self.meta.max_threads {
            assert!(block[0] * block[1] * block[2] <= max, "block exceeds .maxntid");
        }
        let mut sim = Sim::new(&self.module);
        sim.run(gpu, &Launch::new(grid, block, flat))
    }
}

/// `init_fastdiv_values` from ggml-cuda/common.cuh: {mp, L, d}.
pub(super) fn fastdiv(d: u32) -> Vec<u32> {
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < u64::from(d) {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - u64::from(d))) / u64::from(d) + 1) as u32;
    vec![mp, l, d]
}

pub(super) fn mmvq_name(t: Q, ncols: u32) -> String {
    format!("_Z13mul_mat_vec_qIL9ggml_type{}ELi{ncols}EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj", t.ggml_type())
}

pub(super) struct MmvqCase {
    pub weights: Vec<Vec<u8>>, // per row: blocks
    pub y: Vec<Vec<u8>>,       // per column: q8_1 blocks
}

pub(super) fn gen_case(t: Q, nrows: usize, ncols: usize, k: usize, seed: u64, pow2: bool) -> MmvqCase {
    let mut rng = Rng(seed | 1);
    let weights = (0..nrows)
        .map(|_| (0..k / t.qk()).flat_map(|_| random_block(t, &mut rng, pow2)).collect())
        .collect();
    let y = (0..ncols)
        .map(|_| (0..k / 32).flat_map(|_| random_q8_1(&mut rng, pow2, if pow2 { 7 } else { 127 })).collect())
        .collect();
    MmvqCase { weights, y }
}

/// Runs mul_mat_vec_q and returns dst[col][row].
pub(super) fn run_mmvq(t: Q, case: &MmvqCase, k: usize, dot: bool) -> Result<Vec<Vec<f32>>, String> {
    let (nrows, ncols) = (case.weights.len(), case.y.len());
    let new_name = format!(
        "_Z13mul_mat_vec_qIL9ggml_type{}ELi{ncols}ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
        t.ggml_type()
    );
    let fused_abi = index().contains_key(&new_name);
    let name = if fused_abi { new_name } else { mmvq_name(t, ncols as u32) };
    let kern = load_kernel(&name, dot);
    assert_eq!(kern.has_dot_op, dot, "mmvq must use dp4a");
    let rows_per_block = if ncols == 1 { 1 } else { 2 };
    let nwarps = (kern.meta.max_threads.unwrap() / 32) as u32;
    let mut gpu = Gpu::default();
    let wbytes: Vec<u8> = case.weights.concat();
    let ybytes: Vec<u8> = case.y.concat();
    let (w, y) = (gpu.upload(&wbytes), gpu.upload(&ybytes));
    let dst = gpu.alloc(4 * nrows * ncols);
    let head = if fused_abi {
        vec![Arg::P(w), Arg::P(y), Arg::P(0), Arg::W(vec![0; 20]), Arg::P(dst)]
    } else {
        vec![Arg::P(w), Arg::P(y), Arg::P(0), Arg::P(dst)]
    };
    let mut args = head;
    args.extend([
        Arg::U(k as u32),
        Arg::W(fastdiv(1)),
        Arg::U((k / t.qk()) as u32),
        Arg::U((k / 32) as u32),
        Arg::U(nrows as u32),
        Arg::W(fastdiv(1)),
        Arg::U(0),
        Arg::U(0),
        Arg::U(0),
        Arg::W(fastdiv(1)),
        Arg::U(0),
        Arg::U(0),
        Arg::U(0),
    ]);
    if fused_abi {
        args.push(Arg::U(0)); // ids_stride
    }
    kern.run(&mut gpu, [(nrows / rows_per_block) as u32, 1, 1], [32, nwarps, 1], &args)?;
    let flat = gpu.download_f32(dst, nrows * ncols);
    Ok((0..ncols).map(|j| flat[j * nrows..(j + 1) * nrows].to_vec()).collect())
}

pub(super) fn check_mmvq(t: Q, ncols: usize, nrows: usize, k: usize, seed: u64, pow2: bool, dot: bool) {
    let case = gen_case(t, nrows, ncols, k, seed, pow2);
    let got = run_mmvq(t, &case, k, dot).unwrap_or_else(|e| panic!("{} ncols={ncols}: {e}", t.name()));
    assert!(got.iter().flatten().any(|&v| v != 0.0), "all-zero output");
    for j in 0..ncols {
        for r in 0..nrows {
            let (want, mag) = dot_ref(t, &case.weights[r], &case.y[j]);
            let g = f64::from(got[j][r]);
            let tol = if pow2 { 0.0 } else { 2e-5 * mag + 1e-9 };
            assert!(
                (g - want).abs() <= tol,
                "mmvq {} ncols={ncols} dot={dot} pow2={pow2} seed={seed} col {j} row {r}: got {g} want {want} (tol {tol})",
                t.name()
            );
        }
    }
}

/// Reduction length that crosses the kernel's per-iteration block count at least once.
fn k_for(t: Q) -> usize {
    t.qk() * if t.qk() == 256 { 6 } else { 70 }
}

fn mmvq_matrix(t: Q, ncols_list: &[usize]) {
    if ptx_dir().is_none() {
        return;
    }
    for &ncols in ncols_list {
        for (pow2, seed) in [(true, 11u64), (false, 23)] {
            for dot in [false, true] {
                check_mmvq(t, ncols, 4, k_for(t), seed + ncols as u64, pow2, dot);
            }
        }
    }
}

macro_rules! mmvq_tests {
    ($($name:ident: $q:expr,)*) => {$(
        #[test]
        fn $name() { mmvq_matrix($q, &[1, 2, 3, 4, 5, 6, 7, 8]); }
    )*};
}

mmvq_tests! {
    mmvq_q6_k: Q::Q6K,
    mmvq_q5_k: Q::Q5K,
    mmvq_q4_k: Q::Q4K,
    mmvq_q3_k: Q::Q3K,
    mmvq_q2_k: Q::Q2K,
    mmvq_iq4_nl: Q::IQ4NL,
    mmvq_iq4_xs: Q::IQ4XS,
    mmvq_q4_0: Q::Q4_0,
    mmvq_q5_0: Q::Q5_0,
    mmvq_q8_0: Q::Q8_0,
    mmvq_q4_1: Q::Q4_1,
    mmvq_q5_1: Q::Q5_1,
}

#[test]
#[ignore = "bring-up: CFI_GGML_PTX_DIR, CFI_DUMP_FILE, CFI_DUMP_ENTRY"]
fn dump_entry_spirv() {
    let (Some(file), Some(entry), Some(out)) = (
        std::env::var_os("CFI_DUMP_FILE"),
        std::env::var("CFI_DUMP_ENTRY").ok(),
        std::env::var_os("CFI_DUMP_OUT"),
    ) else {
        return;
    };
    let text = std::fs::read_to_string(file).unwrap();
    let prepared = prepare_source(&text).unwrap();
    let t = translate_entry(&prepared, &entry).unwrap();
    let bytes: Vec<u8> = t.words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(out, bytes).unwrap();
}

fn rand_floats(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| ((rng.unit() * 2.0 - 1.0) as f32) * scale).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// quantize_q8_1 over a [ne3][ne2][ne1][ne00] tensor, padded to `ne0` columns.
fn check_quantize_q8_1(ne00: usize, ne0: usize, ne1: usize, ne2: usize, ne3: usize, seed: u64) {
    let kern = load_kernel("_Z13quantize_q8_1PKfPvxxxxxj5uint3", false);
    let mut rng = Rng(seed);
    let mut x = rand_floats(&mut rng, ne00 * ne1 * ne2 * ne3, 3.0);
    // exercise an all-zero group and a group with one large outlier
    for v in &mut x[..32.min(ne00)] {
        *v = 0.0;
    }
    if ne00 >= 64 {
        x[40] = 100.0;
    }
    let mut gpu = Gpu::default();
    let xp = gpu.upload(&f32_bytes(&x));
    let out_bytes = ne0 / 32 * ne1 * ne2 * ne3 * Q8_1_BYTES;
    let yp = gpu.alloc(out_bytes);
    let args = vec![
        Arg::P(xp),
        Arg::P(yp),
        Arg::P(ne00 as u64),
        Arg::P(ne00 as u64),
        Arg::P((ne00 * ne1) as u64),
        Arg::P((ne00 * ne1 * ne2) as u64),
        Arg::P(ne0 as u64),
        Arg::U(ne1 as u32),
        Arg::W(fastdiv(ne2 as u32)),
    ];
    kern.run(&mut gpu, [ne0.div_ceil(256) as u32, ne1 as u32, (ne2 * ne3) as u32], [256, 1, 1], &args).unwrap();
    let got = gpu.download(yp, out_bytes);
    for row in 0..ne1 * ne2 * ne3 {
        for g in 0..ne0 / 32 {
            let mut grp = [0f32; 32];
            for (i, v) in grp.iter_mut().enumerate() {
                let c = g * 32 + i;
                *v = if c < ne00 { x[row * ne00 + c] } else { 0.0 };
            }
            let want = quantize_q8_1_group(&grp, index().keys().any(|k| k.contains("fusion_args_device")));
            let off = (row * (ne0 / 32) + g) * Q8_1_BYTES;
            assert_eq!(&got[off..off + Q8_1_BYTES], &want[..], "quantize_q8_1 row {row} group {g} (ne00={ne00} ne0={ne0})");
        }
    }
}

#[test]
fn quantize_q8_1_matches_reference() {
    if ptx_dir().is_none() {
        return;
    }
    check_quantize_q8_1(256, 256, 1, 1, 1, 3);
    check_quantize_q8_1(96, 512, 3, 2, 1, 4);
    check_quantize_q8_1(300, 512, 2, 2, 2, 5);
}

/// Random quantised rows, `nblk` blocks of `t` per row.
fn gen_rows(t: Q, rows: usize, nblk: usize, seed: u64, pow2: bool) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    (0..rows * nblk).flat_map(|_| random_block(t, &mut rng, pow2)).collect()
}

fn compare_dequant(what: &str, got: &[f32], want: &[f32], pow2: bool, half_out: bool) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (g, w) = if half_out { (f16_val(f16_bits(*g)), f16_val(f16_bits(*w))) } else { (*g, *w) };
        let tol = if pow2 { 0.0 } else if half_out { w.abs() * 1.2e-3 + 1e-7 } else { w.abs() * 1e-6 + 1e-9 };
        assert!((g - w).abs() <= tol, "{what}: element {i} got {g} want {w}");
    }
}

fn read_out(gpu: &Gpu, addr: u64, n: usize, half_out: bool) -> Vec<f32> {
    if half_out {
        (0..n as u64).map(|i| f16_val(gpu.mem.read(addr + 2 * i, 2) as u16)).collect()
    } else {
        gpu.download_f32(addr, n)
    }
}

fn dequant_block_name(t: Q, half_out: bool) -> String {
    let ty = if half_out { "I6__halfE" } else { "IfE" };
    let n = t.name();
    match t {
        Q::Q4_0 | Q::Q4_1 => format!("_Z21dequantize_block_{n}{ty}vPKvPT_i"),
        Q::Q2K | Q::Q3K | Q::Q4K | Q::Q5K | Q::Q6K => format!("_Z21dequantize_block_{n}{ty}vPKvPT_"),
        Q::IQ4NL | Q::IQ4XS => format!("_Z23dequantize_block_{n}{ty}vPKvPT_"),
        _ => unreachable!(),
    }
}

fn check_dequant_block(t: Q, nblk: usize, half_out: bool, pow2: bool, seed: u64) {
    let kern = load_kernel(&dequant_block_name(t, half_out), false);
    let data = gen_rows(t, 1, nblk, seed, pow2);
    let k = nblk * t.qk();
    let mut gpu = Gpu::default();
    let src = gpu.upload(&data);
    let esz = if half_out { 2 } else { 4 };
    let dst = gpu.alloc(k * esz);
    let (grid, block, extra): (u32, u32, Option<u32>) = match t {
        Q::Q2K | Q::Q3K | Q::Q5K | Q::Q6K => (nblk as u32, 64, None),
        Q::Q4K | Q::IQ4XS => (nblk as u32, 32, None),
        Q::IQ4NL => ((k / 256) as u32, 32, None),
        Q::Q4_0 | Q::Q4_1 => (k.div_ceil(256) as u32, 32, Some(nblk as u32)),
        _ => unreachable!(),
    };
    let mut args = vec![Arg::P(src), Arg::P(dst)];
    args.extend(extra.map(Arg::U));
    kern.run(&mut gpu, [grid, 1, 1], [block, 1, 1], &args).unwrap();
    let got = read_out(&gpu, dst, k, half_out);
    let want = dequant_row(t, &data);
    compare_dequant(&format!("dequantize_block_{} half={half_out} pow2={pow2}", t.name()), &got, &want, pow2, half_out);
}

#[test]
fn dequantize_block_kernels_match_reference() {
    if ptx_dir().is_none() {
        return;
    }
    for t in [Q::Q6K, Q::Q5K, Q::Q4K, Q::Q3K, Q::Q2K, Q::IQ4NL, Q::IQ4XS, Q::Q4_0, Q::Q4_1] {
        let nblk = if t.qk() == 256 { 3 } else { 24 };
        for half_out in [false, true] {
            for (pow2, seed) in [(true, 7u64), (false, 8)] {
                check_dequant_block(t, nblk, half_out, pow2, seed);
            }
        }
    }
}

/// Finds the single entry whose name starts with `prefix` and contains every `needle`.
fn find_entry(prefix: &str, needles: &[&str], suffix: &str) -> String {
    let hits: Vec<&String> = index()
        .keys()
        .filter(|k| k.starts_with(prefix) && k.ends_with(suffix) && needles.iter().all(|n| k.contains(n)))
        .collect();
    assert_eq!(hits.len(), 1, "entries for {prefix} {needles:?} {suffix}: {hits:?}");
    hits[0].clone()
}

fn to_half_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f16_bits(*x).to_le_bytes()).collect()
}

/// dequantize_block<32,2,...> (convert.cu): generic path for q4_0/q4_1/q5_0/q5_1/q8_0.
fn check_dequantize_block_generic(t: Q, half_out: bool, pow2: bool, seed: u64) {
    let name = find_entry(
        "_Z16dequantize_blockILi32ELi",
        &[&format!("dequantize_{}E", t.name())],
        if half_out { "6__halfEvS2_PT2_xxxxxx" } else { "EfEvS2_PT2_xxxxxx" },
    );
    let kern = load_kernel(&name, false);
    let (ne00, ne01, ne02, ne03) = (96usize, 3usize, 2usize, 1usize);
    let bpr = ne00 / t.qk();
    let rows = ne01 * ne02 * ne03;
    let data = gen_rows(t, rows, bpr, seed, pow2);
    let mut gpu = Gpu::default();
    let src = gpu.upload(&data);
    let esz = if half_out { 2 } else { 4 };
    let n = ne00 * rows;
    let dst = gpu.alloc(n * esz);
    let args = vec![
        Arg::P(src),
        Arg::P(dst),
        Arg::P(ne00 as u64),
        Arg::P(ne01 as u64),
        Arg::P(ne02 as u64),
        Arg::P(bpr as u64),
        Arg::P((bpr * ne01) as u64),
        Arg::P((bpr * ne01 * ne02) as u64),
    ];
    kern.run(&mut gpu, [ne00.div_ceil(512) as u32, ne01 as u32, (ne02 * ne03) as u32], [256, 1, 1], &args).unwrap();
    let got = read_out(&gpu, dst, n, half_out);
    compare_dequant(&format!("dequantize_block<{}> half={half_out}", t.name()), &got, &dequant_row(t, &data), pow2, half_out);
}

#[test]
fn dequantize_block_generic_matches_reference() {
    if ptx_dir().is_none() {
        return;
    }
    for t in [Q::Q4_0, Q::Q4_1, Q::Q5_0, Q::Q5_1, Q::Q8_0] {
        for half_out in [false, true] {
            for (pow2, seed) in [(true, 5u64), (false, 6)] {
                check_dequantize_block_generic(t, half_out, pow2, seed);
            }
        }
    }
}

/// get_rows over [ne12][ne11][ne10] row ids; `rows_src` is bytes of ne01 rows per (i11,i12).
#[allow(clippy::too_many_arguments)]
fn run_get_rows(
    kern: &Kernel,
    src_bytes: &[u8],
    row_bytes: usize,
    ne00: usize,
    ne01: usize,
    ne10: usize,
    ne11: usize,
    ne12: usize,
    ids: &[i32],
    dst_elem: usize,
) -> Vec<u8> {
    let mut gpu = Gpu::default();
    let src = gpu.upload(src_bytes);
    let idp = gpu.upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let n = ne00 * ne10 * ne11 * ne12;
    let dst = gpu.alloc(n * dst_elem);
    let args = vec![
        Arg::P(src),
        Arg::P(idp),
        Arg::P(dst),
        Arg::P(ne00 as u64),
        Arg::P(ne11 as u64),
        Arg::P(ne12 as u64),
        Arg::P(ne00 as u64),                         // s1
        Arg::P((ne00 * ne10) as u64),                // s2
        Arg::P((ne00 * ne10 * ne11) as u64),         // s3
        Arg::P(row_bytes as u64),                    // nb01
        Arg::P((row_bytes * ne01) as u64),           // nb02
        Arg::P((row_bytes * ne01 * ne11) as u64),    // nb03
        Arg::P(1),                                   // s10
        Arg::P(ne10 as u64),                         // s11
        Arg::P((ne10 * ne11) as u64),                // s12
    ];
    let by = ne00.div_ceil(2 * 256).max(1) as u32;
    kern.run(&mut gpu, [ne10 as u32, by, (ne11 * ne12) as u32], [256, 1, 1], &args).unwrap();
    gpu.download(dst, n * dst_elem)
}

fn gen_ids(rng: &mut Rng, count: usize, ne01: usize) -> Vec<i32> {
    (0..count).map(|_| rng.below(ne01 as u64) as i32).collect()
}

#[test]
fn get_rows_quantized_matches_reference() {
    if ptx_dir().is_none() {
        return;
    }
    let (ne00, ne01, ne10, ne11, ne12) = (96usize, 5usize, 3usize, 2usize, 2usize);
    for t in [Q::Q4_0, Q::Q4_1, Q::Q5_0, Q::Q5_1, Q::Q8_0] {
        for half_out in [false, true] {
            for (pow2, seed) in [(true, 9u64), (false, 10)] {
                let name = find_entry(
                    "_Z10k_get_rowsILi32ELi",
                    &[&format!("dequantize_{}E", t.name())],
                    if half_out { "6__halfEvS2_PKiPT2_xxxyyyyyyyyy" } else { "EfEvS2_PKiPT2_xxxyyyyyyyyy" },
                );
                let kern = load_kernel(&name, false);
                let bpr = ne00 / t.qk();
                let data = gen_rows(t, ne01 * ne11 * ne12, bpr, seed, pow2);
                let ids = gen_ids(&mut Rng(seed ^ 77), ne10 * ne11 * ne12, ne01);
                let got = run_get_rows(&kern, &data, bpr * t.block_bytes(), ne00, ne01, ne10, ne11, ne12, &ids, if half_out { 2 } else { 4 });
                let got: Vec<f32> = if half_out {
                    got.chunks(2).map(|c| f16_val(u16::from_le_bytes([c[0], c[1]]))).collect()
                } else {
                    got.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
                };
                let mut want = Vec::new();
                for i12 in 0..ne12 {
                    for i11 in 0..ne11 {
                        for i10 in 0..ne10 {
                            let id = ids[i12 * ne10 * ne11 + i11 * ne10 + i10] as usize;
                            let rb = bpr * t.block_bytes();
                            let off = ((i12 * ne11 + i11) * ne01 + id) * rb;
                            want.extend(dequant_row(t, &data[off..off + rb]));
                        }
                    }
                }
                compare_dequant(&format!("k_get_rows {} half={half_out}", t.name()), &got, &want, pow2, half_out);
            }
        }
    }
}

#[test]
fn get_rows_float_matches_reference() {
    if ptx_dir().is_none() {
        return;
    }
    let (ne00, ne01, ne10, ne11, ne12) = (300usize, 7usize, 3usize, 2usize, 2usize);
    let mut rng = Rng(99);
    let vals = rand_floats(&mut rng, ne00 * ne01 * ne11 * ne12, 5.0);
    let ids = gen_ids(&mut rng, ne10 * ne11 * ne12, ne01);
    // (kernel suffix, source encoding, destination f16?)
    for (tag, src_half, dst_half) in [("IffE", false, false), ("I6__halffE", true, false), ("If6__halfE", false, true), ("I6__halfS0_E", true, true)] {
        let name = format!("_Z16k_get_rows_float{tag}vPKT_PKiPT0_xxxyyyyyyyyy");
        let kern = load_kernel(&name, false);
        let src_bytes = if src_half { to_half_bytes(&vals) } else { f32_bytes(&vals) };
        let esz = if src_half { 2 } else { 4 };
        let src_vals: Vec<f32> = if src_half { vals.iter().map(|v| f16_val(f16_bits(*v))).collect() } else { vals.clone() };
        let got = run_get_rows(&kern, &src_bytes, ne00 * esz, ne00, ne01, ne10, ne11, ne12, &ids, if dst_half { 2 } else { 4 });
        let got: Vec<f32> = if dst_half {
            got.chunks(2).map(|c| f16_val(u16::from_le_bytes([c[0], c[1]]))).collect()
        } else {
            got.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let mut want = Vec::new();
        for i12 in 0..ne12 {
            for i11 in 0..ne11 {
                for i10 in 0..ne10 {
                    let id = ids[i12 * ne10 * ne11 + i11 * ne10 + i10] as usize;
                    let off = ((i12 * ne11 + i11) * ne01 + id) * ne00;
                    want.extend(src_vals[off..off + ne00].iter().map(|&v| if dst_half { f16_val(f16_bits(v)) } else { v }));
                }
            }
        }
        compare_dequant(&format!("k_get_rows_float{tag}"), &got, &want, true, dst_half);
    }
}

/// Hardware repro shape (m=512, n=4, k=2048) for Q6_K mmvq; slow, run explicitly.
#[test]
#[ignore = "slow: m=512 n=4 k=2048 Q6_K mmvq on both dp4a lowerings"]
fn mmvq_q6_k_hardware_shape() {
    if ptx_dir().is_none() {
        return;
    }
    for dot in [false, true] {
        check_mmvq(Q::Q6K, 4, 512, 2048, 77, false, dot);
    }
}

#[test]
#[ignore = "bring-up probe"]
fn probe_mmq_meta() {
    if ptx_dir().is_none() {
        return;
    }
    for (t, x) in [(Q::Q8_0, 8), (Q::Q4K, 8), (Q::Q6K, 8), (Q::Q8_0, 128)] {
        let k = load_kernel(&format!("_Z9mul_mat_qIL9ggml_type{}ELi{x}ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii", t.ggml_type()), false);
        eprintln!("MMQ {} x={x} shared={} maxthreads={:?} params={}", t.name(), k.meta.shared_bytes, k.meta.max_threads, k.meta.params.len());
    }
}

/// Translates a self-contained PTX source with the requested dp4a lowering.
fn load_source(src: &str, dot: bool) -> Kernel {
    let _guard = DP4A_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let before = dot_product_enabled();
    set_dot_product(dot);
    let translated = translate(src);
    set_dot_product(before);
    let mut t = translated.unwrap();
    coverage_tests::validate(&t.words);
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let has_dot_op = module
        .all_inst_iter()
        .any(|i| matches!(i.class.opcode, rspirv::spirv::Op::SDot | rspirv::spirv::Op::UDot | rspirv::spirv::Op::SUDot));
    Kernel { meta: t.kernels.remove(0), module, words: t.words, has_dot_op }
}

/// Q6_K-style `dp4a(x', u) + dp4a(0xE0E0E0E0, u)` over four columns: both lowerings must equal
/// sum((x'_i - 32) * u_i). Hardware reproducer: examples/dp4a_const_operand.ptx.
#[test]
fn dp4a_constant_operand_matches_reference_in_both_lowerings() {
    let src = include_str!("../../examples/dp4a_const_operand.ptx");
    let n = 64usize;
    let mut rng = Rng(5);
    let x: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
    let u: Vec<u32> = (0..4 * n).map(|_| rng.next() as u32).collect();
    for dot in [false, true] {
        let kern = load_source(src, dot);
        assert_eq!(kern.has_dot_op, dot);
        let mut gpu = Gpu::default();
        let xp = gpu.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        let up = gpu.upload(&u.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        let op = gpu.alloc(4 * 4 * n);
        kern.run(&mut gpu, [1, 1, 1], [64, 1, 1], &[Arg::P(xp), Arg::P(up), Arg::P(op), Arg::U(n as u32)]).unwrap();
        for j in 0..4 {
            for t in 0..n {
                let xv = (x[t] & 0x0f0f_0f0f) | ((x[t] << 4) & 0x3030_3030);
                let mut want = 0i32;
                for b in 0..4 {
                    let q = (((xv >> (8 * b)) & 0xff) as u8 as i8 as i32) + i32::from((0xE0u8 as i8) as i8);
                    let uu = ((u[j * n + t] >> (8 * b)) & 0xff) as u8 as i8 as i32;
                    want += q * uu;
                }
                let got = gpu.mem.read(op + 4 * (j * n + t) as u64, 4) as u32 as i32;
                assert_eq!(got, want, "dot={dot} col {j} lane {t}");
            }
        }
    }
}

/// block_q8_1_mmq (144 bytes, 128 values) from four consecutive q8_1 groups of one column.
fn mmq_block(layout: &str, groups: &[&[u8]]) -> Vec<u8> {
    let mut b = vec![0u8; 144];
    for (g, grp) in groups.iter().enumerate() {
        b[16 + 32 * g..16 + 32 * g + 32].copy_from_slice(&grp[4..36]);
        match layout {
            "d4" => b[4 * g..4 * g + 4].copy_from_slice(&f16_val(u16::from_le_bytes([grp[0], grp[1]])).to_le_bytes()),
            "ds4" => b[4 * g..4 * g + 4].copy_from_slice(&grp[0..4]),
            _ => unreachable!(),
        }
    }
    b
}

fn check_mmq(t: Q, mmq_x: usize, nrows: usize, ncols: usize, k: usize, nsm: u32, seed: u64, pow2: bool) {
    let layout = match t {
        Q::Q4_0 | Q::Q4_1 | Q::Q5_1 | Q::Q4K | Q::Q5K => "ds4",
        Q::Q2K => panic!("D2S6 layout not modelled"),
        _ => "d4",
    };
    let case = gen_case(t, nrows, ncols, k, seed, pow2);
    let tn = t.ggml_type();
    let nc_flag = u8::from(nrows % 128 != 0);
    let main = load_kernel(&format!("_Z9mul_mat_qIL9ggml_type{tn}ELi{mmq_x}ELb{nc_flag}EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii"), false);
    let fix = load_kernel(&format!("_Z24mul_mat_q_stream_k_fixupIL9ggml_type{tn}ELi{mmq_x}ELb{nc_flag}EEvPKiS2_PfPKfiiiiiiiii"), false);
    let mut gpu = Gpu::default();
    let w = gpu.upload(&case.weights.concat());
    // y: [chunk][col] blocks of 144 bytes
    let mut y = Vec::new();
    for chunk in 0..k / 128 {
        for col in 0..ncols {
            let groups: Vec<&[u8]> = (0..4).map(|g| &case.y[col][(chunk * 4 + g) * Q8_1_BYTES..][..Q8_1_BYTES]).collect();
            y.extend(mmq_block(layout, &groups));
        }
    }
    y.extend(vec![0u8; 128 * 144]); // ggml pads src1 by mmq_x_max blocks
    let yp = gpu.upload(&y);
    let dst = gpu.alloc(4 * nrows * ncols);
    let fixup = gpu.alloc(4 * nsm as usize * mmq_x * 128);
    let (nr, nc, kk) = (nrows as u32, ncols as u32, k as u32);
    let margs = vec![
        Arg::P(w), Arg::P(yp), Arg::P(0), Arg::P(0), Arg::P(dst), Arg::P(fixup),
        Arg::U(kk), Arg::U(nr), Arg::U(nc), Arg::U((k / t.qk()) as u32), Arg::U(nc), Arg::U(nr),
        Arg::U(1), Arg::U(1), Arg::U(0), Arg::U(0), Arg::U(0), Arg::U(1), Arg::U(1), Arg::U(0), Arg::U(0), Arg::U(0), Arg::U(nc),
    ];
    main.run(&mut gpu, [nsm, 1, 1], [32, 8, 1], &margs).unwrap();
    let fargs = vec![
        Arg::P(0), Arg::P(0), Arg::P(dst), Arg::P(fixup),
        Arg::U(kk), Arg::U(nr), Arg::U(nc), Arg::U(nr), Arg::U(1), Arg::U(0), Arg::U(1), Arg::U(0), Arg::U(nc),
    ];
    fix.run(&mut gpu, [nsm, 1, 1], [32, 8, 1], &fargs).unwrap();
    let got = gpu.download_f32(dst, nrows * ncols);
    let opts = DotOpts { half_mul: false, exact_sum: layout == "d4" || t == Q::Q4_0, k_min_by_sum: true, half_scales: true };
    for j in 0..ncols {
        for r in 0..nrows {
            let (want, mag) = dot_ref_opts(t, &case.weights[r], &case.y[j], opts);
            let g = f64::from(got[j * nrows + r]);
            let tol = if pow2 { 1e-7 * mag + 1e-12 } else { 3e-5 * mag + 1e-9 };
            assert!((g - want).abs() <= tol, "mmq {} x={mmq_x} nsm={nsm} pow2={pow2} col {j} row {r}: got {g} want {want}", t.name());
        }
    }
}

fn mmq_matrix(t: Q) {
    if ptx_dir().is_none() {
        return;
    }
    for (x, nrows, nsm) in [(8usize, 128usize, 3u32), (32, 192, 5)] {
        let ncols = if x == 8 { 8 } else { x / 2 + 3 };
        for (pow2, seed) in [(true, 21u64), (false, 22)] {
            check_mmq(t, x, nrows, ncols, 512, nsm, seed, pow2);
        }
    }
}

macro_rules! mmq_tests {
    ($($name:ident: $q:expr,)*) => {$(
        #[test]
        fn $name() { mmq_matrix($q); }
    )*};
}

mmq_tests! {
    mmq_q8_0: Q::Q8_0,
    mmq_q4_k: Q::Q4K,
    mmq_q6_k: Q::Q6K,
    mmq_q5_k: Q::Q5K,
    mmq_q3_k: Q::Q3K,
    mmq_q4_0: Q::Q4_0,
    mmq_q5_0: Q::Q5_0,
    mmq_q4_1: Q::Q4_1,
    mmq_q5_1: Q::Q5_1,
    mmq_iq4_nl: Q::IQ4NL,
    mmq_iq4_xs: Q::IQ4XS,
}

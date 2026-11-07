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
    let translated = translate_entry(&src, name);
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
    let kern = load_kernel(&mmvq_name(t, ncols as u32), dot);
    assert_eq!(kern.has_dot_op, dot, "mmvq must use dp4a");
    let rows_per_block = if ncols == 1 { 1 } else { 2 };
    let nwarps = (kern.meta.max_threads.unwrap() / 32) as u32;
    let mut gpu = Gpu::default();
    let wbytes: Vec<u8> = case.weights.concat();
    let ybytes: Vec<u8> = case.y.concat();
    let (w, y) = (gpu.upload(&wbytes), gpu.upload(&ybytes));
    let dst = gpu.alloc(4 * nrows * ncols);
    let args = vec![
        Arg::P(w),
        Arg::P(y),
        Arg::P(0),
        Arg::P(dst),
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
    ];
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

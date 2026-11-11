use super::interp::{self, ArgVal, Dev};
use super::probe::{self, MmvqCase, PArg};
use super::{mmvq, quantize};
use super::refs::{self, Mmvq};

struct Sim<'a> {
    dev: Dev,
    words: &'a [u32],
}

impl probe::Device for Sim<'_> {
    fn alloc(&mut self, bytes: usize) -> Result<u64, String> {
        Ok(self.dev.alloc(bytes))
    }
    fn upload(&mut self, addr: u64, data: &[u8]) -> Result<(), String> {
        self.dev.upload(addr, data);
        Ok(())
    }
    fn download(&mut self, addr: u64, len: usize) -> Result<Vec<u8>, String> {
        Ok(self.dev.download(addr, len))
    }
    fn run(&mut self, grid: [u32; 3], block: [u32; 3], args: &[PArg]) -> Result<(), String> {
        let a: Vec<ArgVal> = args
            .iter()
            .map(|a| match *a {
                PArg::Ptr(p) => ArgVal::Ptr(p),
                PArg::U32(v) => ArgVal::U32(v),
                PArg::U64(v) => ArgVal::U64(v),
            })
            .collect();
        interp::run(&mut self.dev, self.words, &a, grid, block);
        Ok(())
    }
}

fn problem(ty: u32, ncols_dst: usize, nrows: usize, ncols_x: usize, nch: usize, ratio: u32, nsamples: usize, use_ids: bool, seed: u64) -> MmvqCase {
    probe::mmvq_case(ty, ncols_dst, nrows, ncols_x, nch, ratio, nsamples, use_ids, seed)
}

fn run_native(pr: &MmvqCase, cfg: mmvq::Cfg) -> Vec<f32> {
    let words = mmvq::build(cfg, true);
    let mut sim = Sim { dev: Dev::default(), words: &words };
    pr.run(&mut sim, cfg.nwarps as usize).unwrap()
}

fn check(pr: &MmvqCase, nwarps: Option<usize>) {
    let p = &pr.p;
    let nw = nwarps.unwrap_or(p.nwarps);
    let want = pr.reference(nw);
    let cfg = mmvq::Cfg { ty: p.ty, ncols: p.ncols_dst as u32, nwarps: nw as u32, rpb: p.rows_per_block() as u32, fusion: false };
    let got = run_native(pr, cfg);
    let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
    let nmse = refs::nmse(&got, &want);
    assert!(
        bad == 0 || nmse < 1e-10,
        "type {} ncols {} nwarps {}: {bad}/{} outputs differ bitwise, nmse {nmse}",
        p.ty,
        p.ncols_dst,
        nw,
        got.len()
    );
    assert!(want.iter().any(|v| *v != 0.0), "reference all zero");
}

#[test]
fn mmvq_matches_reference_all_types_ncols1() {
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        check(&problem(ty, 1, 5, 512, 1, 1, 1, false, 7 + u64::from(ty)), None);
    }
}

#[test]
fn mmvq_matches_reference_multi_column_and_odd_rows() {
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        for ncols in [2, 5, 8] {
            check(&problem(ty, ncols, 5, 512, 1, 1, 1, false, 11 + u64::from(ty) + ncols as u64), None);
        }
    }
}

#[test]
fn mmvq_channels_samples_and_ids() {
    for ty in [refs::Q8_0, refs::Q4_K] {
        check(&problem(ty, 1, 3, 512, 4, 2, 2, false, 99), None);
        check(&problem(ty, 1, 3, 512, 5, 1, 1, true, 100), None);
    }
}

#[test]
fn mmvq_nwarps_override_is_numerically_close() {
    let pr = problem(refs::Q8_0, 1, 4, 1024, 1, 1, 1, false, 5);
    let want = pr.reference(4);
    let got = run_native(&pr, mmvq::Cfg::legacy(refs::Q8_0, 1, 8));
    assert!(refs::nmse(&got, &want) < 1e-5);
}

/// Runs `spirv-val` (SPIRV_VAL or PATH) on a module; silently skipped when not installed.
pub(super) fn validate(words: &[u32]) {
    use rspirv::binary::Assemble;
    use rspirv::dr::Operand;
    use rspirv::spirv::ExecutionMode;
    use std::io::Write;
    use std::process::{Command, Stdio};
    let validator = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    let mut module = rspirv::dr::load_words(words).unwrap();
    module
        .execution_modes
        .retain(|i| !matches!(i.operands.get(1), Some(Operand::ExecutionMode(ExecutionMode::SubgroupSize))));
    let bytes: Vec<u8> = module.assemble().iter().flat_map(|w| w.to_le_bytes()).collect();
    let Ok(mut child) = Command::new(&validator)
        .args(["--target-env", "spv1.3", "-"])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return;
    };
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&bytes));
    let out = child.wait_with_output().unwrap();
    let _ = writer.join();
    assert!(out.status.success(), "spirv-val rejected the module: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn every_mmvq_variant_validates() {
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        for ncols in 1..=8u32 {
            let nwarps = Mmvq::ggml_nwarps(ncols as usize) as u32;
            validate(&mmvq::build(mmvq::Cfg::legacy(ty, ncols, nwarps), true));
        }
    }
    validate(&mmvq::build(mmvq::Cfg::legacy(refs::Q8_0, 1, 4), false));
}

#[test]
fn quantize_q8_1_matches_reference() {
    let case = probe::quantize_case(96, 128, 3, 2, 2, 42);
    let words = quantize::build(true);
    validate(&words);
    let mut sim = Sim { dev: Dev::default(), words: &words };
    assert_eq!(case.run(&mut sim).unwrap(), case.reference());
}

#[test]
fn device_probe_accepts_native_kernels_and_rejects_corrupt_ones() {
    use super::{Kernel, Plan, Switch, decide};
    let plan = |kernel, key: &str| Plan { kernel, key: key.into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        let p = plan(Kernel::Mmvq { ty, ncols: 3, abi: super::MmvqAbi::Legacy }, "m");
        let words = p.build(true).unwrap();
        let mut sim = Sim { dev: Dev::default(), words: &words };
        probe::validate(&p, &mut sim).unwrap();
    }
    let p = plan(Kernel::QuantizeQ81, "q");
    let words = p.build(true).unwrap();
    probe::validate(&p, &mut Sim { dev: Dev::default(), words: &words }).unwrap();
    // a different kernel's code must fail the check
    let wrong = plan(Kernel::Mmvq { ty: refs::Q8_0, ncols: 1, abi: super::MmvqAbi::Legacy }, "m").build(true).unwrap();
    let p = plan(Kernel::Mmvq { ty: refs::Q4_0, ncols: 1, abi: super::MmvqAbi::Legacy }, "m");
    assert!(probe::validate(&p, &mut Sim { dev: Dev::default(), words: &wrong }).is_err());
    let _ = (decide, Switch::default());
}

#[test]
fn demangles_ggml_kernel_names() {
    use super::{Mangled, TArg, demangle};
    let m = demangle("_Z13mul_mat_vec_qIL9ggml_type2ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj").unwrap();
    assert_eq!(
        m,
        Mangled { base: "mul_mat_vec_q".into(), targs: vec![TArg::Enum("ggml_type".into(), 2), TArg::Int(1)] }
    );
    let m = demangle("_Z13quantize_q8_1PKfPvxxxxxj5uint3").unwrap();
    assert_eq!((m.base.as_str(), m.targs.len()), ("quantize_q8_1", 0));
    let m = demangle("_Z9mul_mat_qIL9ggml_type12ELi8ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii").unwrap();
    assert_eq!(m.targs, vec![TArg::Enum("ggml_type".into(), 12), TArg::Int(8), TArg::Int(1)]);
    // expression template arguments are not understood
    assert!(demangle("_Z11k_bin_bcastIXadL_Z6op_addEEEvv").is_none());
    assert!(demangle("plain_name").is_none());
}

#[test]
fn switch_parsing_and_decisions() {
    use super::{Decision, Switch, decide, layout_hash, mmvq_layout, quantize_layout};
    assert!(Switch::parse("").is_none());
    assert!(Switch::parse("none").is_none());
    assert!(Switch::parse("mmvq,none").is_none());
    let s = Switch::parse("MMVQ:Q8_0, quantize_q8_1");
    assert!(s.allows("mmvq", Some("q8_0")) && !s.allows("mmvq", Some("q4_0")) && s.allows("quantize_q8_1", None));
    assert!(Switch::parse("all").allows("mmvq", Some("q6_k")));
    let name = "_Z13mul_mat_vec_qIL9ggml_type8ELi4EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj";
    let layout = mmvq_layout();
    match decide(name, &layout, &Switch::parse("mmvq")) {
        Decision::Substitute(p) => assert_eq!(p.kernel, super::Kernel::Mmvq { ty: 8, ncols: 4, abi: super::MmvqAbi::Legacy }),
        other => panic!("{other:?}"),
    }
    assert_eq!(decide(name, &layout, &Switch::parse("mmvq:q4_0")), Decision::NotApplicable);
    assert_eq!(decide(name, &layout, &Switch::default()), Decision::NotApplicable);
    // wrong layout -> mismatch, never a substitution
    let mut bad = layout.clone();
    bad.pop();
    assert!(matches!(decide(name, &bad, &Switch::parse("all")), Decision::Mismatch(_)));
    // unsupported type or too many columns
    let q5 = "_Z13mul_mat_vec_qIL9ggml_type6ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj";
    assert_eq!(decide(q5, &layout, &Switch::parse("all")), Decision::NotApplicable);
    let q = "_Z13quantize_q8_1PKfPvxxxxxj5uint3";
    assert!(matches!(decide(q, &quantize_layout(), &Switch::parse("quantize_q8_1")), Decision::Substitute(_)));
    assert_ne!(layout_hash(&mmvq_layout()), layout_hash(&quantize_layout()));
}

#[test]
fn geometry_mapping_rejects_unexpected_launches() {
    use super::{Kernel, Plan};
    let p = Plan { kernel: Kernel::Mmvq { ty: 8, ncols: 1, abi: super::MmvqAbi::Legacy }, key: "k".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
    assert_eq!(p.map_geometry([10, 1, 1], [32, 4, 1]), Some(([10, 1, 1], [32, 4, 1])));
    assert_eq!(p.map_geometry([10, 1, 1], [32, 2, 1]), None);
    let wide = Plan { nwarps: Some(8), ..p };
    assert_eq!(wide.map_geometry([10, 1, 1], [32, 4, 1]), Some(([10, 1, 1], [32, 8, 1])));
}


/// Runs the stock ggml-cuda PTX entry (when `CFI_STOCK_PTX_DIR` points at the extracted blobs)
/// through the PTX translator and the interpreter, and compares it with the native kernel.
fn translated_words(blob_file: &str, entry: &str) -> Option<Vec<u32>> {
    translated(blob_file, entry).map(|t| t.0)
}

fn translated(blob_file: &str, entry: &str) -> Option<(Vec<u32>, Vec<crate::ptx::Param>)> {
    let dir = std::env::var_os("CFI_STOCK_PTX_DIR")?;
    let source = std::fs::read_to_string(std::path::Path::new(&dir).join(blob_file)).ok()?;
    let prepared = crate::ptx::prepare_source(&source).ok()?;
    crate::ptx::set_dot_product(true);
    let t = crate::ptx::translate_entry(&prepared, entry).unwrap_or_else(|e| panic!("{entry}: {e}"));
    Some((t.words, t.kernels[0].params.clone()))
}

#[test]
fn translated_stock_mmvq_matches_native_on_the_interpreter() {
    for (ty, ncols, nch, ratio, nsamples) in [
        (refs::Q8_0, 1u32, 1usize, 1u32, 1usize),
        (refs::Q4_0, 2, 1, 1, 1),
        (refs::Q4_K, 1, 1, 1, 1),
        (refs::Q6_K, 1, 1, 1, 1),
        (refs::Q8_0, 4, 2, 2, 2),
    ] {
        let entry = format!("_Z13mul_mat_vec_qIL9ggml_type{ty}ELi{ncols}EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj");
        let Some((stock, params)) = translated("blob29.ptx", &entry) else { return };
        let name = &entry;
        assert!(
            matches!(super::decide(name, &params, &super::Switch::parse("all")), super::Decision::Substitute(_)),
            "{entry}: registry rejects the real layout {params:?}"
        );
        let pr = problem(ty, ncols as usize, 6, 512, nch, ratio, nsamples, false, 77);
        let nw = pr.p.nwarps;
        let native = run_native(&pr, mmvq::Cfg::legacy(ty, ncols, nw as u32));
        let mut sim = Sim { dev: Dev::default(), words: &stock };
        let translated = pr.run(&mut sim, nw).unwrap();
        let nmse = refs::nmse(&native, &translated);
        let bad = native.iter().zip(&translated).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("type {ty} ncols {ncols} ch {nch}/{ratio} samples {nsamples}: native vs translated stock PTX: nmse {nmse}, {bad} bitwise differences");
        assert!(nmse < 1e-10);
    }
}

#[test]
fn translated_stock_quantize_matches_native_on_the_interpreter() {
    let name = "_Z13quantize_q8_1PKfPvxxxxxj5uint3";
    let dir = std::env::var_os("CFI_STOCK_PTX_DIR");
    let Some(dir) = dir else { return };
    let file = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| std::fs::read_to_string(p).is_ok_and(|s| s.contains(&format!(".entry {name}("))));
    let Some(file) = file else { return };
    let Some((stock, params)) = translated(file.file_name().unwrap().to_str().unwrap(), name) else { return };
    assert_eq!(params, super::quantize_layout());
    let case = probe::quantize_case(96, 128, 3, 2, 2, 9);
    let mut sim = Sim { dev: Dev::default(), words: &stock };
    let translated_out = case.run(&mut sim).unwrap();
    let native_words = quantize::build(true);
    let mut sim = Sim { dev: Dev::default(), words: &native_words };
    assert_eq!(case.run(&mut sim).unwrap(), translated_out, "native quantize_q8_1 must equal the translated stock kernel");
}

/// Every `mul_mat_vec_q` instantiation for the supported types in the stock PTX must be accepted
/// by the registry (catches parameter-layout variants such as an unused `ids` typed u64).
#[test]
fn registry_accepts_every_stock_mmvq_instantiation() {
    let Some(dir) = std::env::var_os("CFI_STOCK_PTX_DIR") else { return };
    let source = std::fs::read_to_string(std::path::Path::new(&dir).join("blob29.ptx")).unwrap();
    let prepared = crate::ptx::prepare_source(&source).unwrap();
    let mut accepted = 0;
    for ty in [refs::Q4_0, refs::Q8_0, refs::Q4_K, refs::Q6_K] {
        for ncols in 1..=8 {
            let entry = format!("_Z13mul_mat_vec_qIL9ggml_type{ty}ELi{ncols}EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj");
            let t = crate::ptx::translate_entry(&prepared, &entry).unwrap_or_else(|e| panic!("{entry}: {e}"));
            let d = super::decide(&entry, &t.kernels[0].params, &super::Switch::parse("all"));
            assert!(matches!(d, super::Decision::Substitute(_)), "{entry}: {d:?} / {:?}", t.kernels[0].params);
            accepted += 1;
        }
    }
    assert_eq!(accepted, 32);
}

#[test]
fn mmq_q8_0_matches_reference() {
    let words = super::mmq::build(refs::Q8_0, true);
    validate(&words);
    let plan = super::Plan { kernel: super::Kernel::Mmq { ty: refs::Q8_0, abi: super::MmqAbi::Legacy }, key: "m".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
    // ragged rows/cols, channels with a ratio, two samples, k not a multiple of the 4-block stage
    for (ncols_x, nrows, ncols, nch, ratio, nsamples) in [(256, 70, 40, 2, 1, 1), (384, 65, 9, 4, 2, 2), (128, 3, 1, 1, 1, 1)] {
        let case = probe::mmq_case(ncols_x, nrows, ncols, nch, ratio, nsamples, 5 + ncols as u64);
        let want = case.reference();
        let mut sim = Sim { dev: Dev::default(), words: &words };
        let got = case.run(&mut sim, &plan).unwrap();
        let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert!(bad == 0, "mmq {ncols_x}x{nrows}x{ncols}: {bad} of {} differ, nmse {}", got.len(), refs::nmse(&got, &want));
    }
}

/// The translated stock `mul_mat_q<Q8_0>` (MMA path, mma.sync emulation) against the native kernel.
#[test]
fn translated_stock_mmq_q8_0_matches_native_on_the_interpreter() {
    let entry = "_Z9mul_mat_qIL9ggml_type8ELi8ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii";
    crate::ptx::set_dynamic_shared_bytes(65536);
    let Some((stock, params)) = translated("blob98.ptx", entry) else { return };
    let plan = super::Plan { kernel: super::Kernel::Mmq { ty: refs::Q8_0, abi: super::MmqAbi::Legacy }, key: "m".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
    assert!(matches!(
        super::decide(entry, &params, &super::Switch::parse("mmq")),
        super::Decision::Substitute(_)
    ));
    let case = probe::mmq_case(256, 70, 8, 1, 1, 1, 21);
    // stock launch: one stream-k CTA covers the single tile (no fixup involved)
    let mut dev = Dev::default();
    let xa = dev.put(&case.x);
    let ya = dev.put(&case.y);
    let n = case.dst_len();
    let da = dev.alloc(n * 4);
    let tmp = dev.alloc(1 << 16);
    let mut args: Vec<ArgVal> = case
        .args(xa, ya, da)
        .iter()
        .map(|a| match *a {
            PArg::Ptr(p) => ArgVal::Ptr(p),
            PArg::U32(v) => ArgVal::U32(v),
            PArg::U64(v) => ArgVal::U64(v),
        })
        .collect();
    args[5] = ArgVal::Ptr(tmp);
    interp::run(&mut dev, &stock, &args, [1, 1, 1], [32, 8, 1]);
    let translated_out: Vec<f32> =
        dev.download(da, n * 4).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let words = super::mmq::build(refs::Q8_0, true);
    let mut sim = Sim { dev: Dev::default(), words: &words };
    let native = case.run(&mut sim, &plan).unwrap();
    let nmse = refs::nmse(&native, &translated_out);
    let bad = native.iter().zip(&translated_out).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    println!("mmq q8_0 native vs translated stock: nmse {nmse}, {bad} bitwise differences");
    assert!(nmse < 1e-9);
}

#[test]
fn auto_policy_follows_the_measurements() {
    use super::{Decision, Switch, decide, mmq_layout, mmvq_layout, quantize_layout};
    let auto = Switch::parse("auto");
    assert!(!auto.is_none());
    let name = |ty: u32, n: u32| format!("_Z13mul_mat_vec_qIL9ggml_type{ty}ELi{n}EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj");
    let yes = |ty, n| matches!(decide(&name(ty, n), &mmvq_layout(), &auto), Decision::Substitute(_));
    for ty in [2, 8, 12, 14] {
        assert!(yes(ty, 1), "ncols 1 native for type {ty}");
    }
    assert!(yes(8, 2) && yes(2, 4) && yes(8, 8) && yes(12, 8) && yes(14, 2) && yes(14, 4));
    assert!(matches!(decide("_Z13quantize_q8_1PKfPvxxxxxj5uint3", &quantize_layout(), &auto), Decision::Substitute(_)));
    let mmq = "_Z9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii";
    assert!(matches!(decide(mmq, &mmq_layout(), &auto), Decision::Substitute(_)), "mmq is in auto");
    for ty in [2, 6, 12, 13, 14] {
        let e = format!("_Z9mul_mat_qIL9ggml_type{ty}ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
        assert!(matches!(decide(&e, &mmq_layout(), &auto), Decision::Substitute(_)), "mmq type {ty} in auto");
    }
    let q41 = "_Z9mul_mat_qIL9ggml_type3ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii";
    assert_eq!(decide(q41, &mmq_layout(), &auto), Decision::NotApplicable);
    // `-family` switches one family off
    let no_mmq = Switch::parse("auto,-mmq");
    assert_eq!(decide(mmq, &mmq_layout(), &no_mmq), Decision::NotApplicable);
    assert!(matches!(decide(&name(8, 1), &mmvq_layout(), &no_mmq), Decision::Substitute(_)));
    assert_eq!(decide(&name(8, 1), &mmvq_layout(), &Switch::parse("all,-mmvq:q8_0")), Decision::NotApplicable);
    assert!(Switch::parse("auto,-mmq,none").is_none());
}

/// Multi-tile, ragged shapes (several row/column tiles, channels, samples) against the stock
/// translated kernel launched with one CTA per tile (no stream-k split, no fixup).
/// K must be a multiple of 256: the stock kernel consumes 8 Q8_0 blocks per step and ggml zero-pads y
/// to that, so a shorter K reads unrelated y data (not a kernel bug).
/// Slow (minutes) on the debug interpreter: run with `--ignored` and CFI_STOCK_PTX_DIR.
#[test]
#[ignore = "slow: multi-tile stock mmq on the interpreter; mismatch under investigation"]
fn translated_stock_mmq_q8_0_multi_tile_shapes() {
    crate::ptx::set_dynamic_shared_bytes(65536);
    let plan = super::Plan { kernel: super::Kernel::Mmq { ty: refs::Q8_0, abi: super::MmqAbi::Legacy }, key: "m".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
    let words = super::mmq::build(refs::Q8_0, true);
    let mut failed = false;
    // (ncols_x, nrows, ncols, mmq_x, nch, ratio, nsamples)
    for (ncols_x, nrows, ncols, mmq_x, nch, ratio, nsamples) in [(256usize, 130usize, 9usize, 8usize, 1usize, 1usize, 1usize), (256, 130, 9, 8, 2, 2, 2)] {
        let entry = format!("_Z9mul_mat_qIL9ggml_type8ELi{mmq_x}ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
        let Some(stock) = translated_words("blob98.ptx", &entry) else { return };
        let case = probe::mmq_case(ncols_x, nrows, ncols, nch, ratio, nsamples, 31 + ncols as u64);
        let n = case.dst_len();
        let mut dev = Dev::default();
        let (xa, ya) = (dev.put(&case.x), dev.put(&case.y));
        let da = dev.alloc(n * 4);
        let tmp = dev.alloc(1 << 18);
        let mut args: Vec<ArgVal> = case
            .args(xa, ya, da)
            .iter()
            .map(|a| match *a {
                PArg::Ptr(p) => ArgVal::Ptr(p),
                PArg::U32(v) => ArgVal::U32(v),
                PArg::U64(v) => ArgVal::U64(v),
            })
            .collect();
        args[5] = ArgVal::Ptr(tmp);
        let tiles = (ncols.div_ceil(mmq_x) * nrows.div_ceil(128) * nch * nsamples) as u32;
        interp::run(&mut dev, &stock, &args, [tiles, 1, 1], [32, 8, 1]);
        let translated_out: Vec<f32> =
            dev.download(da, n * 4).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut sim = Sim { dev: Dev::default(), words: &words };
        let native = case.run(&mut sim, &plan).unwrap();
        let reference = case.reference();
        println!(
            "{ncols_x}x{nrows}x{ncols} mmq_x {mmq_x} ch {nch}/{ratio} s {nsamples}: native-vs-stock {:.2e}, stock-vs-ref {:.2e}, native-vs-ref {:.2e}",
            refs::nmse(&native, &translated_out),
            refs::nmse(&translated_out, &reference),
            refs::nmse(&native, &reference)
        );
        // where does the stock result differ from the reference? (row, col) of the first few
        let mut shown = 0;
        for col in 0..ncols {
            for row in 0..nrows {
                let i = col * nrows + row;
                if translated_out[i].to_bits() != reference[i].to_bits() && shown < 12 {
                    println!("  stock differs at row {row} col {col}: stock {} ref {} native {}", translated_out[i], reference[i], native[i]);
                    shown += 1;
                }
            }
        }
        failed |= refs::nmse(&native, &translated_out) >= 1e-9;
    }
    assert!(!failed);
}

const MMQ_TYPES: [(u32, &str); 6] = [
    (refs::Q4_0, "blob91.ptx"),
    (refs::Q5_0, "blob94.ptx"),
    (refs::Q8_0, "blob98.ptx"),
    (refs::Q4_K, "blob93.ptx"),
    (refs::Q5_K, "blob96.ptx"),
    (refs::Q6_K, "blob97.ptx"),
];

/// Native mmq against the host reference, every supported type: ragged rows/cols, channels with a
/// ratio, two samples, k not a multiple of the stage (legacy types only).
#[test]
fn mmq_all_types_match_reference_bitwise() {
    for (ty, _) in MMQ_TYPES {
        let words = super::mmq::build(ty, true);
        validate(&words);
        let plan = super::Plan { kernel: super::Kernel::Mmq { ty, abi: super::MmqAbi::Legacy }, key: "m".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
        let shapes: &[(usize, usize, usize, usize, usize, usize)] =
            if refs::mmq_groups(ty) == 1 { &[(256, 70, 40, 2, 1, 1), (384, 65, 9, 4, 2, 2), (128, 3, 1, 1, 1, 1)] } else { &[(512, 70, 40, 2, 1, 1), (256, 65, 9, 4, 2, 2), (768, 3, 1, 1, 1, 1)] };
        for &(ncols_x, nrows, ncols, nch, ratio, nsamples) in shapes {
            let case = probe::mmq_case_ty(ty, ncols_x, nrows, ncols, nch, ratio, nsamples, 5 + ncols as u64 + u64::from(ty));
            let want = case.reference();
            let mut sim = Sim { dev: Dev::default(), words: &words };
            let got = case.run(&mut sim, &plan).unwrap();
            let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert!(bad == 0, "type {ty} mmq {ncols_x}x{nrows}x{ncols}: {bad} of {} differ, nmse {}", got.len(), refs::nmse(&got, &want));
        }
    }
}

/// The translated stock `mul_mat_q<ty>` (MMA path) against the native kernel, one CTA per tile
/// (needs `CFI_STOCK_PTX_DIR`).
#[test]
fn translated_stock_mmq_all_types_match_native_on_the_interpreter() {
    crate::ptx::set_dynamic_shared_bytes(65536);
    for (ty, blob) in MMQ_TYPES {
        let entry = format!("_Z9mul_mat_qIL9ggml_type{ty}ELi8ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
        let Some((stock, params)) = translated(blob, &entry) else { return };
        let plan = super::Plan { kernel: super::Kernel::Mmq { ty, abi: super::MmqAbi::Legacy }, key: "m".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
        assert!(matches!(super::decide(&entry, &params, &super::Switch::parse("mmq")), super::Decision::Substitute(_)), "type {ty}");
        let case = probe::mmq_case_ty(ty, 512, 70, 8, 1, 1, 1, 21 + u64::from(ty));
        let mut dev = Dev::default();
        let xa = dev.put(&case.x);
        let ya = dev.put(&case.y);
        let n = case.dst_len();
        let da = dev.alloc(n * 4);
        let tmp = dev.alloc(1 << 16);
        let mut args: Vec<ArgVal> = case
            .args(xa, ya, da)
            .iter()
            .map(|a| match *a {
                PArg::Ptr(p) => ArgVal::Ptr(p),
                PArg::U32(v) => ArgVal::U32(v),
                PArg::U64(v) => ArgVal::U64(v),
            })
            .collect();
        args[5] = ArgVal::Ptr(tmp);
        interp::run(&mut dev, &stock, &args, [1, 1, 1], [32, 8, 1]);
        let translated_out: Vec<f32> =
            dev.download(da, n * 4).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let words = super::mmq::build(ty, true);
        let mut sim = Sim { dev: Dev::default(), words: &words };
        let native = case.run(&mut sim, &plan).unwrap();
        let nmse = refs::nmse(&native, &translated_out);
        let bad = native.iter().zip(&translated_out).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("mmq type {ty} native vs translated stock: nmse {nmse:.3e}, {bad} of {} bitwise differences", n);
        assert!(nmse < 1e-9, "type {ty}");
        assert_eq!(bad, 0, "type {ty}: not bit-identical");
    }
}

fn run_fast(pr: &MmvqCase, f: mmvq::Fast) -> Vec<f32> {
    let words = mmvq::build_fast(pr.p.ty, f, true);
    let mut sim = Sim { dev: Dev::default(), words: &words };
    pr.run_fast(&mut sim, f.rows, f.warps).unwrap()
}

fn fast(rows: u32, warps: u32, unroll: u32) -> mmvq::Fast {
    mmvq::Fast { rows, warps, unroll, wide: false }
}

#[test]
fn mmvq_fast_wide_matches_its_reference_bitwise() {
    for ty in [refs::Q8_0, refs::Q4_K, refs::Q6_K] {
        for (f, ncols_x) in [(fast(4, 1, 2), 1024usize), (fast(2, 2, 4), 2304), (fast(1, 4, 1), 512), (fast(4, 1, 8), 4864)] {
            let ncols_x = if ty == refs::Q8_0 { ncols_x } else { ncols_x.div_ceil(256) * 256 };
            let f = mmvq::Fast { wide: true, ..f };
            for (nch, ratio, ns, ids) in [(1usize, 1u32, 1usize, false), (4, 2, 2, false), (5, 1, 1, true)] {
                let pr = problem(ty, 1, 11, ncols_x, nch, ratio, ns, ids, 1234 + ncols_x as u64 + u64::from(ty));
                let got = run_fast(&pr, f);
                let want = pr.reference_fast(f);
                let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
                assert_eq!(bad, 0, "type {ty} {f:?} nch {nch} ids {ids}: {bad}/{} outputs differ", got.len());
                let nmse = refs::nmse(&got, &pr.reference(4));
                assert!(nmse < 1e-10, "type {ty} {f:?}: nmse vs stock {nmse}");
            }
        }
        validate(&mmvq::build_fast(ty, mmvq::Fast { rows: 4, warps: 1, unroll: 4, wide: true }, true));
        validate(&mmvq::build_fast(ty, mmvq::Fast { rows: 2, warps: 2, unroll: 2, wide: true }, false));
    }
    assert_eq!(super::mmvq_fast_from(Some("4x1x4w"), refs::Q8_0), Some(mmvq::Fast { rows: 4, warps: 1, unroll: 4, wide: true }));
    assert_eq!(super::mmvq_fast_from(Some("4x1x4w"), refs::Q4_0), Some(fast(4, 1, 4)));
}

#[test]
fn mmvq_fast_matches_its_reference_bitwise_and_stock_closely() {
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        // 11 rows (tail workgroup), 1024 / 2304 columns (partial last unrolled step)
        for (f, ncols_x) in [(fast(4, 1, 2), 1024usize), (fast(2, 2, 1), 2304), (fast(8, 1, 3), 2304), (fast(1, 4, 4), 512), (fast(4, 2, 8), 2304)] {
            let pr = problem(ty, 1, 11, ncols_x, 1, 1, 1, false, 31 + u64::from(ty) + u64::from(f.rows));
            let got = run_fast(&pr, f);
            let want = pr.reference_fast(f);
            let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
            assert_eq!(bad, 0, "type {ty} {f:?}: {bad}/{} outputs differ", got.len());
            let stock = pr.reference(4);
            let nmse = refs::nmse(&got, &stock);
            assert!(nmse < 1e-10, "type {ty}: nmse vs stock order {nmse}");
            assert!(want.iter().all(|v| *v != 0.0));
        }
    }
}

#[test]
fn mmvq_fast_channels_samples_and_ids() {
    for ty in [refs::Q8_0, refs::Q4_K, refs::Q6_K] {
        for (nch, ratio, ns, ids) in [(4usize, 2u32, 2usize, false), (5, 1, 1, true)] {
            for f in [fast(4, 1, 2), fast(2, 4, 2)] {
                let pr = problem(ty, 1, 3, 512, nch, ratio, ns, ids, 77 + u64::from(ty));
                let got = run_fast(&pr, f);
                let want = pr.reference_fast(f);
                assert!(got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()), "type {ty} nch {nch} ids {ids} {f:?}");
                assert!(refs::nmse(&got, &pr.reference(4)) < 1e-10);
            }
        }
    }
}

#[test]
fn mmvq_fast_variants_validate_and_parse() {
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        validate(&mmvq::build_fast(ty, fast(4, 1, 2), true));
        validate(&mmvq::build_fast(ty, fast(2, 4, 1), false));
    }
    let q8 = refs::Q8_0;
    assert_eq!(super::mmvq_fast_from(None, q8), None);
    assert_eq!(super::mmvq_fast_from(Some("0"), q8), None);
    assert_eq!(super::mmvq_fast_from(Some("1"), q8), Some(super::mmvq_fast_default(q8)));
    assert_eq!(super::mmvq_fast_from(Some("8,1,3"), q8), Some(fast(8, 1, 3)));
    assert_eq!(super::mmvq_fast_from(Some("2x4x1"), q8), Some(fast(2, 4, 1)));
    assert_eq!(super::mmvq_fast_from(Some("3x1x1"), q8), None);
    assert_eq!(super::mmvq_fast_from(Some("4x1x9"), q8), None);
    assert_eq!(super::mmvq_fast_from(Some("4x2"), q8), None);
}

#[test]
fn fattn_matches_exact_attention_on_the_interpreter() {
    use super::fattn;
    for d in [64u32, 128] {
        let words = fattn::build(d);
        validate(&words);
        let plan = super::Plan { kernel: super::Kernel::Fattn { d }, key: "f".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
        for (nq, nh, nhk, nkv, ns, causal) in [(19usize, 4usize, 2usize, 70usize, 1usize, true), (16, 2, 1, 33, 2, true), (16, 2, 2, 32, 1, false), (17, 1, 1, 160, 1, true)] {
            let case = probe::fattn_case(d as usize, nq, nh, nhk, nkv, ns, causal, 7 + nq as u64);
            let mut sim = Sim { dev: Dev::default(), words: &words };
            let got = case.run(&mut sim, &plan).unwrap();
            let want = case.reference();
            let nmse = refs::nmse(&got, &want);
            assert!(nmse < 1e-10, "d {d} nq {nq} nh {nh}/{nhk} nkv {nkv} ns {ns}: nmse {nmse}");
        }
    }
    // feature gates: sinks, ALiBi and softcap keep the translated kernel
    let plan = super::Plan { kernel: super::Kernel::Fattn { d: 64 }, key: "f".into(), nwarps: None, layout_hash: 0, aux: 0, fast: None };
    let case = probe::fattn_case(64, 16, 2, 2, 32, 1, true, 1);
    let base = probe::launch_values(&case.args(16, 16, 16, 16, 16));
    assert!(plan.map_launch([1, 1, 1], [32, 4, 1], &base).is_some());
    for (i, v) in [(4usize, 64u64), (9, 1.0f32.to_bits() as u64), (13, 2.0f32.to_bits() as u64), (14, 128), (15, 1)] {
        let mut bad = base.clone();
        bad[i] = v;
        assert!(plan.map_launch([1, 1, 1], [32, 4, 1], &bad).is_none(), "word {i}");
    }
}

#[test]
fn fattn_names_and_layout_are_recognised() {
    use crate::ptx::Param;
    let name = "_Z18flash_attn_ext_f16ILi64ELi64ELi32ELi1ELi4ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP6float2ffffjfiiiiiiiiiiiiiliiliiiiil";
    let mut params = vec![Param::GlobalPtr; 8];
    params.extend(super::fattn::arg_sizes()[8..].iter().map(|&n| if n == 8 { Param::U64 } else { Param::U32 }));
    params[8] = Param::F32;
    let sw = super::Switch::parse("auto,fattn");
    match super::decide(name, &params, &sw) {
        super::Decision::Substitute(p) => assert_eq!(p.kernel, super::Kernel::Fattn { d: 64 }),
        other => panic!("{other:?}"),
    }
    // in auto by default, removable with -fattn
    assert!(matches!(super::decide(name, &params, &super::Switch::parse("auto")), super::Decision::Substitute(_)));
    assert_eq!(super::decide(name, &params, &super::Switch::parse("auto,-fattn")), super::Decision::NotApplicable);
    let softcap = name.replace("ELb0ELb0EE", "ELb1ELb0EE");
    assert_eq!(super::decide(&softcap, &params, &sw), super::Decision::NotApplicable);
    let fixup = "_Z25flash_attn_stream_k_fixupILi64ELi32ELi1EEvPfPK6float2iiii";
    let fp = vec![Param::GlobalPtr, Param::GlobalPtr, Param::U32, Param::U32, Param::U32, Param::U32];
    match super::decide(fixup, &fp, &sw) {
        super::Decision::Substitute(p) => assert_eq!(p.kernel, super::Kernel::FattnFixup),
        other => panic!("{other:?}"),
    }
}

// ---- Ollama 0.40 `mul_mat_vec_q<type, ncols, has_fusion, small_k, halve_iters>` ----

fn fused_name(ty: u32, ncols: u32, fusion: u32, small_k: u32, halve: u32) -> String {
    format!(
        "_Z13mul_mat_vec_qIL9ggml_type{ty}ELi{ncols}ELb{fusion}ELb{small_k}ELb{halve}EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"
    )
}

#[test]
fn fused_abi_mmvq_registry_decisions() {
    use super::{Decision, Kernel, MmvqAbi, Switch, decide, mmvq_fused_layout};
    let all = Switch::parse("all");
    let layout = mmvq_fused_layout();
    match decide(&fused_name(8, 1, 0, 1, 0), &layout, &all) {
        Decision::Substitute(p) => {
            assert_eq!(p.kernel, Kernel::Mmvq { ty: 8, ncols: 1, abi: MmvqAbi::Fused { small_k: true, fusion: false } })
        }
        other => panic!("{other:?}"),
    }
    match decide(&fused_name(14, 4, 0, 0, 1), &layout, &all) {
        Decision::Substitute(p) => {
            assert_eq!(p.kernel, Kernel::Mmvq { ty: 14, ncols: 4, abi: MmvqAbi::Fused { small_k: false, fusion: false } })
        }
        other => panic!("{other:?}"),
    }
    // has_fusion kernels are substituted too; unsupported epilogues are rejected per launch
    // (`fusion_launch_words`), and the decode-fast kernel is never used for them.
    match decide(&fused_name(8, 1, 1, 0, 0), &layout, &all) {
        Decision::Substitute(p) => {
            assert_eq!(p.kernel, Kernel::Mmvq { ty: 8, ncols: 1, abi: MmvqAbi::Fused { small_k: false, fusion: true } });
            assert!(p.fast.is_none());
            assert!(p.entry_name().ends_with("_f"), "{}", p.entry_name());
        }
        other => panic!("{other:?}"),
    }
    // Unsupported type (IQ4_NL) and a legacy layout under the new name.
    assert_eq!(decide(&fused_name(20, 1, 0, 0, 0), &layout, &all), Decision::NotApplicable);
    assert!(matches!(decide(&fused_name(8, 1, 0, 0, 0), &super::mmvq_layout(), &all), Decision::Mismatch(_)));
    // The legacy name keeps the legacy ABI.
    let legacy = "_Z13mul_mat_vec_qIL9ggml_type8ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj";
    match decide(legacy, &super::mmvq_layout(), &all) {
        Decision::Substitute(p) => assert_eq!(p.kernel, Kernel::Mmvq { ty: 8, ncols: 1, abi: MmvqAbi::Legacy }),
        other => panic!("{other:?}"),
    }
}

#[test]
fn fused_abi_geometry_follows_the_turing_table() {
    use super::{Decision, MmvqAbi, Switch, decide, mmvq_fused_layout};
    let (plain, small) = (MmvqAbi::Fused { small_k: false, fusion: false }, MmvqAbi::Fused { small_k: true, fusion: false });
    // `.maxntid` of the Ollama 0.40.14 compute_75 instantiations
    for (ty, ncols, warps) in [(8, 1, 4), (2, 1, 4), (12, 1, 2), (14, 1, 2), (8, 4, 4), (12, 3, 4), (8, 5, 2), (14, 8, 2)] {
        assert_eq!(plain.stock_nwarps(ty, ncols), warps, "type {ty} ncols {ncols}");
        assert_eq!(small.stock_nwarps(ty, ncols), warps, "type {ty} ncols {ncols}");
    }
    assert_eq!((plain.rows_per_block(8, 1), small.rows_per_block(8, 1), small.rows_per_block(14, 1)), (1, 4, 2));
    assert_eq!((small.rows_per_block(8, 2), MmvqAbi::Legacy.rows_per_block(8, 1)), (2, 1));
    // the hook's 47 -> 23 word normalisation
    let words: Vec<u32> = (0..44).collect();
    let legacy = small.legacy_words(&words).unwrap();
    assert_eq!(legacy.len(), 23);
    assert_eq!(&legacy[..4], &[0, 1, 2, 23]);
    assert_eq!(legacy[22], 42); // stride_sample_dst; 43 (ids_stride) is dropped
    assert!(small.legacy_words(&words[..43]).is_none());
    // stock block shape is checked against the TURING table, rows from stride_col_dst
    let Decision::Substitute(p) = decide(&fused_name(12, 1, 0, 1, 0), &mmvq_fused_layout(), &Switch::parse("all")) else {
        panic!()
    };
    assert_eq!(p.map_launch([7, 1, 1], [32, 2, 1], &[0; 23]), Some(([7, 1, 1], [32, 2, 1])));
    assert_eq!(p.map_launch([7, 1, 1], [32, 4, 1], &[0; 23]), None);
    let mut vals = vec![0u64; 23];
    vals[10] = 13; // 13 rows in 7 blocks of 2
    assert_eq!(p.mmvq_rows([7, 1, 1], &vals), Some(13));
    // The decode-fast kernel (CUDA_FOR_INTEL_MMVQ_FAST) covers exactly the matrix rows: 13 rows
    // in 4 stock small_k blocks of 4 become 4 workgroups of 4 rows, and nrows is passed as 13.
    let fast_plan = super::Plan {
        kernel: super::Kernel::Mmvq { ty: 8, ncols: 1, abi: small },
        key: "f".into(),
        nwarps: None,
        layout_hash: 0,
        aux: 0,
        fast: Some(fast(4, 1, 2)),
    };
    assert_eq!(fast_plan.map_launch([4, 1, 1], [32, 4, 1], &vals), Some(([4, 1, 1], [32, 4, 1])));
    assert_eq!(fast_plan.extra_word([4, 1, 1], &vals), Some(13));
    assert_eq!(fast_plan.map_geometry([4, 1, 1], [32, 4, 1]), None, "needs the row count");
    let legacy_fast = super::Plan { kernel: super::Kernel::Mmvq { ty: 8, ncols: 1, abi: MmvqAbi::Legacy }, ..fast_plan };
    assert_eq!(legacy_fast.extra_word([13, 1, 1], &[]), Some(13));
}

/// The small_k decomposition (rows per block = nwarps for one column) against the reference,
/// on a row count that is not a multiple of the rows per block and K below one block pass.
#[test]
fn small_k_native_mmvq_matches_reference() {
    use super::MmvqAbi;
    let abi = MmvqAbi::Fused { small_k: true, fusion: false };
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        let ncols_x = if matches!(ty, refs::Q4_K | refs::Q6_K) { 512 } else { 896 };
        let mut pr = problem(ty, 1, 13, ncols_x, 2, 1, 2, false, 0x5a11 + u64::from(ty));
        pr.p.nwarps = abi.stock_nwarps(ty, 1) as usize;
        pr.p.rpb = abi.rows_per_block(ty, 1) as usize;
        let cfg = mmvq::Cfg { ty, ncols: 1, nwarps: pr.p.nwarps as u32, rpb: pr.p.rpb as u32, fusion: false };
        let want = pr.reference(pr.p.nwarps);
        let got = run_native(&pr, cfg);
        let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
        assert_eq!(bad, 0, "type {ty}: {bad}/{} differ, nmse {}", got.len(), refs::nmse(&got, &want));
        assert!(want.iter().any(|v| *v != 0.0));
    }
}

/// Ollama 0.40.14 stock PTX (`CFI_STOCK_PTX_DIR_040` with the compute_75 module holding
/// mul_mat_vec_q as `blob40.ptx`, extracted locally, never committed): the registry accepts the
/// real parameter layout of every supported instantiation without fusion, and the TURING table
/// matches each kernel's launch bounds. The numeric check against the stock kernel is
/// `ptx::ggml_conformance::mmvq_fused_abi_stock_matches_reference_bitwise`.
#[test]
fn stock_0_40_mmvq_instantiations_are_accepted() {
    use super::{Decision, Kernel, MmvqAbi, Switch, decide};
    let Some(dir) = std::env::var_os("CFI_STOCK_PTX_DIR_040") else { return };
    let source = std::fs::read_to_string(std::path::Path::new(&dir).join("blob40.ptx")).unwrap();
    let prepared = crate::ptx::prepare_source(&source).unwrap();
    crate::ptx::set_dot_product(true);
    let mut accepted = 0;
    for ty in [refs::Q4_0, refs::Q8_0, refs::Q4_K, refs::Q6_K] {
        for (ncols, fusion) in (1..=8u32).map(|n| (n, 0)).chain([(1, 1)]) {
            for (small_k, halve) in [(0, 0), (1, 0), (0, 1)] {
                if ncols > 1 && (small_k, halve) != (0, 0) {
                    continue;
                }
                let entry = fused_name(ty, ncols, fusion, small_k, halve);
                let t = crate::ptx::translate_entry(&prepared, &entry).unwrap_or_else(|e| panic!("{entry}: {e}"));
                let params = t.kernels[0].params.clone();
                let Decision::Substitute(plan) = decide(&entry, &params, &Switch::parse("all")) else {
                    panic!("{entry}: registry rejects the real layout {params:?}")
                };
                let Kernel::Mmvq { abi, .. } = plan.kernel else { unreachable!() };
                assert_eq!(abi, MmvqAbi::Fused { small_k: small_k == 1, fusion: fusion == 1 });
                assert_eq!(Some(abi.stock_nwarps(ty, ncols) * 32), t.kernels[0].max_threads, "{entry}");
                accepted += 1;
            }
        }
    }
    assert_eq!(accepted, 4 * (8 + 2 + 3));
}

/// ggml's fused epilogue on the native kernel: bias only (q/k/v projections), a SwiGLU gate
/// (FFN gate/up), and both, against the composed host reference, with and without small_k.
#[test]
fn fused_native_mmvq_matches_reference() {
    use super::MmvqAbi;
    use super::probe::Fusion;
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        for (ncols, small_k) in [(1u32, true), (1, false), (2, false)] {
            for (gate, x_bias, gate_bias) in [(false, true, false), (true, false, false), (true, true, true)] {
                let abi = MmvqAbi::Fused { small_k, fusion: true };
                let ncols_x = match (small_k, ty) {
                    (true, refs::Q4_K | refs::Q6_K) => 512,
                    (true, _) => 896,
                    _ => 1024,
                };
                let seed = 0xf0 + u64::from(ty) * 7 + u64::from(ncols) * 3 + u64::from(small_k);
                let mut pr = problem(ty, ncols as usize, 13, ncols_x, 2, 1, 2, false, seed);
                pr.p.nwarps = abi.stock_nwarps(ty, ncols) as usize;
                pr.p.rpb = abi.rows_per_block(ty, ncols) as usize;
                let mut rng = refs::Rng(seed ^ 0x9e37);
                let nblocks = pr.x.len() / refs::traits(ty).unwrap().block_bytes;
                let n = pr.dst_len();
                let f = Fusion {
                    gate: gate.then(|| refs::random_blocks(ty, nblocks, &mut rng)),
                    x_bias: x_bias.then(|| (0..n).map(|_| rng.unit()).collect()),
                    gate_bias: gate_bias.then(|| (0..n).map(|_| rng.unit()).collect()),
                };
                let cfg = mmvq::Cfg { ty, ncols, nwarps: pr.p.nwarps as u32, rpb: pr.p.rpb as u32, fusion: true };
                let words = mmvq::build(cfg, true);
                validate(&words);
                let mut sim = Sim { dev: Dev::default(), words: &words };
                let got = pr.run_fusion(&mut sim, pr.p.nwarps, &f).unwrap();
                let want = pr.reference_fusion(pr.p.nwarps, &f);
                let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
                let nmse = refs::nmse(&got, &want);
                assert!(
                    bad == 0 || nmse < 1e-12,
                    "type {ty} ncols {ncols} small_k {small_k} gate {gate} bias {x_bias}/{gate_bias}: {bad}/{n} differ, nmse {nmse}"
                );
                assert!(want.iter().any(|v| *v != 0.0));
            }
        }
    }
}

#[test]
fn fusion_launch_words_reject_what_the_native_epilogue_lacks() {
    let mut w = vec![0u64; super::MMVQ_FUSED_WORDS];
    let set_ptr = |w: &mut Vec<u64>, i: usize, v: u64| {
        w[i] = v & 0xffff_ffff;
        w[i + 1] = v >> 32;
    };
    set_ptr(&mut w, 3, 0x1_0000_1000); // x_bias
    assert_eq!(super::fusion_launch_words(&w), Some([0x1_0000_1000, 0, 0, 0]), "bias only, any glu op");
    set_ptr(&mut w, 5, 0x2_0000_2000); // gate
    assert_eq!(super::fusion_launch_words(&w), None, "gate with REGLU (0)");
    w[13] = u64::from(mmvq::GLU_SWIGLU);
    assert_eq!(super::fusion_launch_words(&w), Some([0x1_0000_1000, 0x2_0000_2000, 0, 2]));
    let mut moe = w.clone();
    moe[2] = 0x3000; // ids
    assert_eq!(super::fusion_launch_words(&moe), None);
    let mut shared = w.clone();
    set_ptr(&mut shared, 15, 0x4000); // shared_up
    assert_eq!(super::fusion_launch_words(&shared), None);
    let mut scaled = w.clone();
    set_ptr(&mut scaled, 9, 0x5000); // x_scale (NVFP4)
    assert_eq!(super::fusion_launch_words(&scaled), None);
    assert_eq!(super::fusion_launch_words(&w[..43]), None);
}

// ---- Ollama 0.40 `mul_mat_q<type, mmq_x, need_check, ggml_prec, has_fusion>` ----

/// Translated 0.40 stock entry from `CFI_STOCK_PTX_DIR_040` (any `.ptx` file holding it).
fn translated_040(entry: &str) -> Option<(Vec<u32>, Vec<crate::ptx::Param>)> {
    let dir = std::env::var_os("CFI_STOCK_PTX_DIR_040")?;
    for e in std::fs::read_dir(&dir).ok()? {
        let path = e.ok()?.path();
        let Ok(source) = std::fs::read_to_string(&path) else { continue };
        if !source.contains(&format!(".entry {entry}(")) {
            continue;
        }
        let prepared = crate::ptx::prepare_source(&source).ok()?;
        crate::ptx::set_dot_product(true);
        let t = crate::ptx::translate_entry(&prepared, entry).unwrap_or_else(|e| panic!("{entry}: {e}"));
        return Some((t.words, t.kernels[0].params.clone()));
    }
    None
}

fn mmq_040_name(ty: u32, mmq_x: u32, need_check: u32, fusion: u32) -> String {
    format!(
        "_Z9mul_mat_qIL9ggml_type{ty}ELi{mmq_x}ELb{need_check}EL9ggml_prec30ELb{fusion}EEvPKcS3_PKiS5_S5_PfS6_PKf11ggml_glu_opf5uint3iiiiiSA_SA_iiiSA_SA_iiiSA_"
    )
}

/// 0.40 launch words for a legacy `MmqCase` (inverse of the hook's 39 -> 23 mapping): `tiles`
/// is the value of the trailing uint3.
fn mmq_040_args(case: &probe::MmqCase, x: u64, y: u64, dst: u64, tmp: u64, last: u32) -> Vec<ArgVal> {
    let p = &case.p;
    let qk = 32 * refs::mmq_groups(case.ty);
    let u = |v: usize| ArgVal::U32(v as u32);
    let fd = |d: usize| refs::fastdiv_values(d as u32).map(ArgVal::U32);
    let mut a = vec![
        ArgVal::Ptr(x),
        ArgVal::Ptr(0),
        ArgVal::Ptr(y),
        ArgVal::Ptr(0),
        ArgVal::Ptr(0),
        ArgVal::Ptr(dst),
        ArgVal::Ptr(tmp),
        ArgVal::Ptr(0),
        ArgVal::U32(0),
        ArgVal::F32(0.0),
    ];
    a.extend(fd(p.ncols_x / qk));
    a.extend([u(p.nrows_x), u(p.ncols_dst), u(p.stride_row_x), u(p.ncols_y), u(p.stride_col_dst)]);
    a.extend(fd(p.channel_ratio));
    a.extend(fd(p.nchannels_y));
    a.extend([u(p.stride_channel_x), u(p.stride_channel_y), u(p.stride_channel_dst)]);
    a.extend(fd(p.sample_ratio));
    a.extend(fd(p.nsamples_y));
    a.extend([u(p.stride_sample_x), u(p.stride_sample_y), u(p.stride_sample_dst)]);
    a.extend(refs::fastdiv_values(last).map(ArgVal::U32));
    a
}

fn argval_words(args: &[ArgVal]) -> Vec<u64> {
    args.iter()
        .map(|a| match *a {
            ArgVal::Ptr(v) | ArgVal::U64(v) => v,
            ArgVal::U32(v) => u64::from(v),
            ArgVal::F32(v) => u64::from(v.to_bits()),
        })
        .collect()
}

#[test]
fn mmq_v040_words_map_to_the_legacy_launch() {
    for ty in [refs::Q8_0, refs::Q4_K] {
        let case = probe::mmq_case_ty(ty, 512, 70, 9, 4, 2, 2, 3);
        let v040 = argval_words(&mmq_040_args(&case, 0x1000, 0x2000, 0x3000, 0x4000, 2));
        assert_eq!(v040.len(), super::MMQ_V040_WORDS);
        let mut legacy = probe::launch_values(&case.args(0x1000, 0x2000, 0x3000));
        legacy[5] = 0x4000; // tmp_fixup
        assert_eq!(super::mmq_legacy_words(ty, &v040).map(|w| w.to_vec()), Some(legacy), "type {ty}");
        // what the native kernel lacks stays on the translated kernel
        for (slot, what) in [(1, "x_gate"), (7, "y_scale"), (3, "ids_dst"), (4, "expert_bounds")] {
            let mut w = v040.clone();
            w[slot] = 0x9000;
            assert_eq!(super::mmq_legacy_words(ty, &w), None, "{what}");
        }
        assert_eq!(super::mmq_legacy_words(ty, &v040[..38]), None);
    }
}

#[test]
fn mmq_v040_registry_decisions() {
    use super::{Decision, Kernel, MmqAbi, Switch, decide, mmq_fixup_v040_layout, mmq_v040_layout};
    let mmq = Switch::parse("mmq");
    match decide(&mmq_040_name(8, 64, 0, 0), &mmq_v040_layout(), &mmq) {
        Decision::Substitute(p) => {
            assert_eq!(p.kernel, Kernel::Mmq { ty: 8, abi: MmqAbi::V040 });
            assert_eq!((p.aux, p.key.as_str()), (64, "mmq:t8"), "same native module as v0.12.6");
        }
        other => panic!("{other:?}"),
    }
    // fused gate (has_fusion) and the FP4 activation path (ggml_prec 40) stay translated
    assert_eq!(decide(&mmq_040_name(8, 64, 0, 1), &mmq_v040_layout(), &mmq), Decision::NotApplicable);
    let prec40 = mmq_040_name(8, 64, 0, 0).replace("ggml_prec30", "ggml_prec40");
    assert_eq!(decide(&prec40, &mmq_v040_layout(), &mmq), Decision::NotApplicable);
    assert!(matches!(decide(&mmq_040_name(8, 64, 0, 0), &super::mmq_layout(), &mmq), Decision::Mismatch(_)));
    let fixup = "_Z24mul_mat_q_stream_k_fixupIL9ggml_type12ELi64ELb1EL9ggml_prec30EEvPKiS3_PfS4_5uint3iiiS5_iS5_iS5_";
    match decide(fixup, &mmq_fixup_v040_layout(), &mmq) {
        Decision::Substitute(p) => assert_eq!(p.kernel, Kernel::MmqFixup { ty: 12, abi: MmqAbi::V040 }),
        other => panic!("{other:?}"),
    }
    // auto enables the same types as for v0.12.6
    let auto = Switch::parse("auto");
    assert!(matches!(decide(&mmq_040_name(8, 64, 0, 0), &mmq_v040_layout(), &auto), Decision::Substitute(_)));
}

/// The translated Ollama 0.40 stock `mul_mat_q` (one stream-k CTA covering one tile, the shape
/// the v0.12.6 counterpart test uses) against the native kernel fed the hook's legacy words, for
/// every supported type, and the registry against the real parameter layouts (needs
/// `CFI_STOCK_PTX_DIR_040` with the 0.40 mul_mat_q modules, extracted locally, never committed).
#[test]
fn translated_stock_0_40_mmq_matches_native() {
    use super::{Decision, Kernel, MmqAbi, Switch, decide};
    crate::ptx::set_dynamic_shared_bytes(65536);
    let mut ran = 0;
    for (ty, _) in MMQ_TYPES {
        let entry = mmq_040_name(ty, 8, 1, 0);
        let Some((stock, params)) = translated_040(&entry) else { continue };
        let Decision::Substitute(plan) = decide(&entry, &params, &Switch::parse("mmq")) else {
            panic!("{entry}: registry rejects the real layout {params:?}")
        };
        assert_eq!(plan.kernel, Kernel::Mmq { ty, abi: MmqAbi::V040 });
        let ncols_x = 256 * refs::mmq_groups(ty);
        let case = probe::mmq_case_ty(ty, ncols_x, 70, 8, 1, 1, 1, 21 + u64::from(ty));
        let mut dev = Dev::default();
        let (xa, ya) = (dev.put(&case.x), dev.put(&case.y));
        let n = case.dst_len();
        let da = dev.alloc(n * 4);
        let tmp = dev.alloc(1 << 16);
        let args = mmq_040_args(&case, xa, ya, da, tmp, 1);
        // the hook's mapping of these exact words is what the native kernel receives
        let legacy = super::mmq_legacy_words(ty, &argval_words(&args)).expect("supported launch");
        interp::run(&mut dev, &stock, &args, [1, 1, 1], [32, 8, 1]);
        let stock_out: Vec<f32> = dev.download(da, n * 4).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let words = super::mmq::build(ty, true);
        let mut sim = Sim { dev: Dev::default(), words: &words };
        let native = case.run(&mut sim, &plan).unwrap();
        assert_eq!(&legacy[6..22], &probe::launch_values(&case.args(0, 0, 0))[6..22], "{entry}: word mapping");
        let reference = case.reference();
        let nmse = refs::nmse(&native, &stock_out);
        let bad = native.iter().zip(&stock_out).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("{entry}: native vs stock nmse {nmse:.2e}, {bad}/{n} bitwise differences; stock vs reference {:.2e}", refs::nmse(&stock_out, &reference));
        assert!(nmse < 1e-9, "{entry}: nmse {nmse}");
        assert!(stock_out.iter().any(|v| *v != 0.0));
        ran += 1;
    }
    if std::env::var_os("CFI_STOCK_PTX_DIR_040").is_some() {
        assert_eq!(ran, MMQ_TYPES.len());
    }
}

/// The runtime's self-check of a fused plan must run the fused kernel with real fusion inputs
/// (27 launch words), not the plain 23: otherwise the epilogue is never checked on hardware.
#[test]
fn probe_validates_fused_plans_with_their_epilogue() {
    use super::{Decision, Switch, decide, mmvq_fused_layout};
    for (ty, small_k) in [(refs::Q8_0, 1), (refs::Q4_K, 0)] {
        let Decision::Substitute(plan) = decide(&fused_name(ty, 1, 1, small_k, 0), &mmvq_fused_layout(), &Switch::parse("all")) else {
            panic!()
        };
        let words = plan.build(true).unwrap();
        probe::validate(&plan, &mut Sim { dev: Dev::default(), words: &words }).unwrap_or_else(|e| panic!("type {ty}: {e}"));
        // a kernel without the epilogue must fail the fused probe
        let Decision::Substitute(plain) = decide(&fused_name(ty, 1, 0, small_k, 0), &mmvq_fused_layout(), &Switch::parse("all")) else {
            panic!()
        };
        let wrong = plain.build(true).unwrap();
        assert!(probe::validate(&plan, &mut Sim { dev: Dev::default(), words: &wrong }).is_err(), "type {ty}");
    }
}

/// The decode-fast kernel with ggml's fused epilogue (bias only, SwiGLU gate, both) against the
/// fast reference composed with the epilogue, bit for bit, wide and narrow, with tail rows.
#[test]
fn fused_fast_mmvq_matches_reference() {
    use super::probe::Fusion;
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        let wide = matches!(ty, refs::Q8_0 | refs::Q4_K | refs::Q6_K);
        for f in [mmvq::Fast { rows: 4, warps: 1, unroll: 2, wide }, mmvq::Fast { rows: 2, warps: 2, unroll: 1, wide: false }] {
            for (gate, x_bias, gate_bias) in [(false, true, false), (true, false, false), (true, true, true)] {
                let ncols_x = if matches!(ty, refs::Q4_K | refs::Q6_K) { 1024 } else { 896 };
                let seed = 0x7a + u64::from(ty) + u64::from(f.rows);
                let pr = problem(ty, 1, 11, ncols_x, 2, 1, 2, false, seed);
                let mut rng = refs::Rng(seed ^ 0x55);
                let nblocks = pr.x.len() / refs::traits(ty).unwrap().block_bytes;
                let n = pr.dst_len();
                let fu = Fusion {
                    gate: gate.then(|| refs::random_blocks(ty, nblocks, &mut rng)),
                    x_bias: x_bias.then(|| (0..n).map(|_| rng.unit()).collect()),
                    gate_bias: gate_bias.then(|| (0..n).map(|_| rng.unit()).collect()),
                };
                let words = mmvq::build_fast_with(ty, f, true, true);
                validate(&words);
                let mut sim = Sim { dev: Dev::default(), words: &words };
                let got = pr.run_fast_fusion(&mut sim, f, &fu).unwrap();
                let want = pr.reference_fast_fusion(f, &fu);
                let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
                let nmse = refs::nmse(&got, &want);
                assert!(bad == 0 || nmse < 1e-12, "type {ty} {f:?} gate {gate} bias {x_bias}/{gate_bias}: {bad}/{n} differ, nmse {nmse}");
                // against the stock-order fused reference: same values to rounding
                let stock = pr.reference_fusion(pr.p.nwarps, &fu);
                assert!(refs::nmse(&got, &stock) < 1e-9, "type {ty} {f:?}: vs stock order");
            }
        }
    }
}

#[test]
fn probe_validates_fused_fast_plans() {
    use super::{Decision, Switch, decide, mmvq_fused_layout};
    let fast = mmvq::Fast { rows: 4, warps: 1, unroll: 2, wide: true };
    for ty in [refs::Q8_0, refs::Q6_K] {
        let Decision::Substitute(mut plan) = decide(&fused_name(ty, 1, 1, 0, 0), &mmvq_fused_layout(), &Switch::parse("all")) else {
            panic!()
        };
        plan.fast = Some(fast);
        assert!(plan.entry_name().ends_with("_wide_f"), "{}", plan.entry_name());
        let words = plan.build(true).unwrap();
        probe::validate(&plan, &mut Sim { dev: Dev::default(), words: &words }).unwrap_or_else(|e| panic!("type {ty}: {e}"));
        let plain = mmvq::build_fast(ty, fast, true);
        assert!(probe::validate(&plan, &mut Sim { dev: Dev::default(), words: &plain }).is_err(), "plain kernel must fail");
    }
}

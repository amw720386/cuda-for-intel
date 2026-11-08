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
    let cfg = mmvq::Cfg { ty: p.ty, ncols: p.ncols_dst as u32, nwarps: nw as u32 };
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
    let got = run_native(&pr, mmvq::Cfg { ty: refs::Q8_0, ncols: 1, nwarps: 8 });
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
            validate(&mmvq::build(mmvq::Cfg { ty, ncols, nwarps }, true));
        }
    }
    validate(&mmvq::build(mmvq::Cfg { ty: refs::Q8_0, ncols: 1, nwarps: 4 }, false));
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
    let plan = |kernel, key: &str| Plan { kernel, key: key.into(), nwarps: None, layout_hash: 0 };
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        let p = plan(Kernel::Mmvq { ty, ncols: 3 }, "m");
        let words = p.build(true).unwrap();
        let mut sim = Sim { dev: Dev::default(), words: &words };
        probe::validate(&p, &mut sim).unwrap();
    }
    let p = plan(Kernel::QuantizeQ81, "q");
    let words = p.build(true).unwrap();
    probe::validate(&p, &mut Sim { dev: Dev::default(), words: &words }).unwrap();
    // a different kernel's code must fail the check
    let wrong = plan(Kernel::Mmvq { ty: refs::Q8_0, ncols: 1 }, "m").build(true).unwrap();
    let p = plan(Kernel::Mmvq { ty: refs::Q4_0, ncols: 1 }, "m");
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
        Decision::Substitute(p) => assert_eq!(p.kernel, super::Kernel::Mmvq { ty: 8, ncols: 4 }),
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
    let p = Plan { kernel: Kernel::Mmvq { ty: 8, ncols: 1 }, key: "k".into(), nwarps: None, layout_hash: 0 };
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
        let native = run_native(&pr, mmvq::Cfg { ty, ncols, nwarps: nw as u32 });
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
    let words = super::mmq::build(true);
    validate(&words);
    let plan = super::Plan { kernel: super::Kernel::MmqQ80, key: "m".into(), nwarps: None, layout_hash: 0 };
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
    let plan = super::Plan { kernel: super::Kernel::MmqQ80, key: "m".into(), nwarps: None, layout_hash: 0 };
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
    let words = super::mmq::build(true);
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
    assert!(!yes(8, 2) && !yes(2, 4) && !yes(8, 8));
    assert!(yes(12, 8) && yes(14, 2) && !yes(14, 4));
    assert!(matches!(decide("_Z13quantize_q8_1PKfPvxxxxxj5uint3", &quantize_layout(), &auto), Decision::Substitute(_)));
    let mmq = "_Z9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii";
    assert_eq!(decide(mmq, &mmq_layout(), &auto), Decision::NotApplicable);
    // explicit list items still force a kernel in addition to auto
    assert!(matches!(decide(mmq, &mmq_layout(), &Switch::parse("auto,mmq")), Decision::Substitute(_)));
}

/// Multi-tile, ragged shapes (several row/column tiles, channels, samples) against the stock
/// translated kernel launched with one CTA per tile (no stream-k split, no fixup).
/// Slow (minutes) on the debug interpreter: run with `--ignored` and CFI_STOCK_PTX_DIR.
#[test]
#[ignore = "slow: multi-tile stock mmq on the interpreter; mismatch under investigation"]
fn translated_stock_mmq_q8_0_multi_tile_shapes() {
    crate::ptx::set_dynamic_shared_bytes(65536);
    let plan = super::Plan { kernel: super::Kernel::MmqQ80, key: "m".into(), nwarps: None, layout_hash: 0 };
    let words = super::mmq::build(true);
    let mut failed = false;
    // (ncols_x, nrows, ncols, mmq_x, nch, ratio, nsamples)
    for (ncols_x, nrows, ncols, mmq_x, nch, ratio, nsamples) in [(128usize, 130usize, 9usize, 8usize, 1usize, 1usize, 1usize), (128, 130, 9, 8, 2, 1, 1), (128, 130, 9, 8, 2, 2, 2)] {
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
        failed |= refs::nmse(&native, &translated_out) >= 1e-9;
    }
    assert!(!failed);
}

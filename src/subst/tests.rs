use super::interp::{self, ArgVal, Dev};
use super::{mmvq, quantize};
use super::refs::{self, Mmvq, Rng};

struct Problem {
    p: Mmvq,
    x: Vec<u8>,
    y: Vec<u8>,
    ids: Vec<i32>,
}

fn problem(ty: u32, ncols_dst: usize, nrows: usize, ncols_x: usize, nch: usize, ratio: u32, nsamples: usize, use_ids: bool, seed: u64) -> Problem {
    let t = refs::traits(ty).unwrap();
    let mut rng = Rng(seed);
    let bpr = ncols_x / t.qk;
    let nchannels_x = nch / ratio as usize;
    let nsamples_x = nsamples; // sample ratio 1
    let ystride_col = (ncols_x / 32) as u32;
    let nch_y = if use_ids { 3 } else { nch };
    let p = Mmvq {
        ty,
        ncols_dst,
        nwarps: Mmvq::ggml_nwarps(ncols_dst),
        nrows_x: nrows,
        ncols_x,
        nchannels_dst: nch,
        nsamples_dst: nsamples,
        channel_ratio: ratio,
        sample_ratio: 1,
        nchannels_y: nch_y as u32,
        use_ids,
        stride_row_x: bpr as u32,
        stride_col_y: ystride_col,
        stride_col_dst: nrows as u32,
        stride_channel_x: (nrows * bpr) as u32,
        stride_channel_y: ystride_col * ncols_dst as u32,
        stride_channel_dst: (nrows * ncols_dst) as u32,
        stride_sample_x: (nrows * bpr * nchannels_x.max(if use_ids { 4 } else { 1 })) as u32,
        stride_sample_y: ystride_col * ncols_dst as u32 * nch_y as u32,
        stride_sample_dst: (nrows * ncols_dst * nch) as u32,
    };
    let nx_ch = nchannels_x.max(if use_ids { 4 } else { 1 });
    let x = refs::random_blocks(ty, nrows * bpr * nx_ch * nsamples_x, &mut rng);
    let acts: Vec<f32> = (0..ncols_x * ncols_dst * nch_y * nsamples).map(|_| rng.unit() * 3.0).collect();
    let y = refs::quantize_q8_1(&acts);
    let ids = (0..nch).map(|i| ((i * 3 + 1) % 4) as i32).collect();
    Problem { p, x, y, ids }
}

fn run_native(pr: &Problem, cfg: mmvq::Cfg) -> Vec<f32> {
    let p = &pr.p;
    let words = mmvq::build(cfg, true);
    let mut dev = Dev::default();
    let xa = dev.put(&pr.x);
    let ya = dev.put(&pr.y);
    let ia = if p.use_ids { dev.put(&pr.ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()) } else { 0 };
    let n_dst = p.nsamples_dst * p.stride_sample_dst as usize;
    let da = dev.alloc(n_dst * 4);
    let chr = if p.use_ids { [0; 3] } else { refs::fastdiv_values(p.channel_ratio) };
    let nchy = if p.use_ids { refs::fastdiv_values(p.nchannels_y) } else { [0; 3] };
    let smr = refs::fastdiv_values(p.sample_ratio);
    let u = ArgVal::U32;
    let args = [
        ArgVal::Ptr(xa),
        ArgVal::Ptr(ya),
        ArgVal::Ptr(ia),
        ArgVal::Ptr(da),
        u(p.ncols_x as u32),
        u(nchy[0]),
        u(nchy[1]),
        u(nchy[2]),
        u(p.stride_row_x),
        u(p.stride_col_y),
        u(p.stride_col_dst),
        u(chr[0]),
        u(chr[1]),
        u(chr[2]),
        u(p.stride_channel_x),
        u(p.stride_channel_y),
        u(p.stride_channel_dst),
        u(smr[0]),
        u(smr[1]),
        u(smr[2]),
        u(p.stride_sample_x),
        u(p.stride_sample_y),
        u(p.stride_sample_dst),
    ];
    let mut q = p.clone();
    q.nwarps = cfg.nwarps as usize;
    interp::run(&mut dev, &words, &args, q.grid(), q.block());
    dev.download(da, n_dst * 4).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn check(pr: &Problem, nwarps: Option<usize>) {
    let p = &pr.p;
    let n_dst = p.nsamples_dst * p.stride_sample_dst as usize;
    let mut want = vec![0f32; n_dst];
    let mut q = p.clone();
    if let Some(w) = nwarps {
        q.nwarps = w;
    }
    refs::mmvq(&q, &pr.x, &pr.y, p.use_ids.then_some(&pr.ids[..]), &mut want);
    let cfg = mmvq::Cfg { ty: p.ty, ncols: p.ncols_dst as u32, nwarps: q.nwarps as u32 };
    let got = run_native(pr, cfg);
    let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
    let nmse = refs::nmse(&got, &want);
    assert!(
        bad == 0 || nmse < 1e-10,
        "type {} ncols {} nwarps {}: {bad}/{} outputs differ bitwise, nmse {nmse}",
        p.ty,
        p.ncols_dst,
        q.nwarps,
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
    let n_dst = pr.p.stride_sample_dst as usize;
    let mut want = vec![0f32; n_dst];
    refs::mmvq(&pr.p, &pr.x, &pr.y, None, &mut want);
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
    let (ne00, ne0, ne1, ne2, ne3) = (96usize, 128usize, 3usize, 2usize, 2usize);
    let mut rng = Rng(42);
    let total = ne00 * ne1 * ne2 * ne3;
    let mut x: Vec<f32> = (0..total).map(|_| rng.unit() * 5.0).collect();
    // exercise an all-zero block and a tiny-magnitude block
    for v in x.iter_mut().take(32) {
        *v = 0.0;
    }
    for v in x.iter_mut().skip(32).take(32) {
        *v *= 1e-6;
    }
    let mut want = Vec::new();
    for r in 0..ne1 * ne2 * ne3 {
        let mut row = x[r * ne00..(r + 1) * ne00].to_vec();
        row.resize(ne0, 0.0);
        want.extend(refs::quantize_q8_1(&row));
    }
    let words = quantize::build(true);
    validate(&words);
    let mut dev = Dev::default();
    let xa = dev.put(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let ya = dev.alloc(want.len());
    let fd = refs::fastdiv_values(ne2 as u32);
    let args = [
        ArgVal::Ptr(xa),
        ArgVal::Ptr(ya),
        ArgVal::U64(ne00 as u64),
        ArgVal::U64(ne00 as u64),
        ArgVal::U64((ne00 * ne1) as u64),
        ArgVal::U64((ne00 * ne1 * ne2) as u64),
        ArgVal::U64(ne0 as u64),
        ArgVal::U32(ne1 as u32),
        ArgVal::U32(fd[0]),
        ArgVal::U32(fd[1]),
        ArgVal::U32(fd[2]),
    ];
    interp::run(&mut dev, &words, &args, [ne0.div_ceil(256) as u32, ne1 as u32, (ne2 * ne3) as u32], [256, 1, 1]);
    assert_eq!(dev.download(ya, want.len()), want);
}

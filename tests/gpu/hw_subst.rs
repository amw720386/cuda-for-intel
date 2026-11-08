//! One dense hardware run for kernel substitution (`CUDA_FOR_INTEL_SUBST`): native Intel SPIR-V
//! kernels versus the translated stock ggml-cuda PTX, on random data, plus a bandwidth benchmark.
//!
//! Needs the stock Ollama v0.12.6 `ggml-cuda.dll` (`CFI_STOCK_GGML_DLL`), a built nvcuda.dll and an
//! Intel GPU. Run (one slot):
//!   set CFI_STOCK_GGML_DLL=<path>\ggml-cuda.dll
//!   cargo test --test gpu hw_subst -- --ignored --nocapture --test-threads=1
//! Also runs Q8_0 mul_mat_q (prefill) native vs translated. Optional: CFI_SUBST_REPS (benchmark launches, default 40), CFI_SUBST_NWARPS_SWEEP=1 (also time
//! the native kernels with 8 warps per block), CFI_SUBST_TYPES=q8_0,q4_0,q4_k,q6_k.
use super::hw_ggml_stock::extract_all;
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;
use std::time::Instant;

#[path = "../../src/subst/refs.rs"]
mod refs;
use refs::{Mmvq, Rng};

const SHAPE_ROWS: usize = 300; // deliberately not a multiple of the 2-row tile of ncols > 1 kernels
const SHAPE_COLS: usize = 1024;

struct Api {
    launch: unsafe extern "system" fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut Handle) -> i32,
    sync: unsafe extern "system" fn() -> i32,
    alloc: unsafe extern "system" fn(*mut u64, usize) -> i32,
    htod: unsafe extern "system" fn(u64, *const c_void, usize) -> i32,
    dtoh: unsafe extern "system" fn(*mut c_void, u64, usize) -> i32,
}

fn bytes_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A device buffer filled from `data`.
fn upload(api: &Api, data: &[u8]) -> u64 {
    let mut dev = 0u64;
    assert_eq!(unsafe { (api.alloc)(&mut dev, data.len().max(16)) }, 0);
    assert_eq!(unsafe { (api.htod)(dev, data.as_ptr().cast(), data.len()) }, 0);
    dev
}

fn download(api: &Api, dev: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    assert_eq!(unsafe { (api.dtoh)(out.as_mut_ptr().cast(), dev, len) }, 0);
    out
}

/// Argument storage for `mul_mat_vec_q` (17 CUDA parameters, uint3 as 12 bytes).
fn mmvq_params(p: &Mmvq, x: u64, y: u64, ids: u64, dst: u64) -> Vec<Vec<u8>> {
    let w = |v: u32| v.to_le_bytes().to_vec();
    let fd = |v: [u32; 3]| bytes_u32(&v);
    let chr = if p.use_ids { [0; 3] } else { refs::fastdiv_values(p.channel_ratio) };
    let nchy = if p.use_ids { refs::fastdiv_values(p.nchannels_y) } else { [0; 3] };
    vec![
        x.to_le_bytes().to_vec(),
        y.to_le_bytes().to_vec(),
        ids.to_le_bytes().to_vec(),
        dst.to_le_bytes().to_vec(),
        w(p.ncols_x as u32),
        fd(nchy),
        w(p.stride_row_x),
        w(p.stride_col_y),
        w(p.stride_col_dst),
        fd(chr),
        w(p.stride_channel_x),
        w(p.stride_channel_y),
        w(p.stride_channel_dst),
        fd(refs::fastdiv_values(p.sample_ratio)),
        w(p.stride_sample_x),
        w(p.stride_sample_y),
        w(p.stride_sample_dst),
    ]
}

fn launch(api: &Api, f: Handle, grid: [u32; 3], block: [u32; 3], params: &[Vec<u8>]) -> i32 {
    let mut pointers: Vec<*mut c_void> = params.iter().map(|p| p.as_ptr().cast_mut().cast()).collect();
    unsafe {
        (api.launch)(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, null_mut(), pointers.as_mut_ptr(), null_mut())
    }
}

struct Case {
    p: Mmvq,
    x: Vec<u8>,
    y: Vec<u8>,
    ids: Vec<i32>,
}

#[allow(clippy::too_many_arguments)]
fn make_case(ty: u32, ncols: usize, nrows: usize, ncols_x: usize, nch: usize, ratio: u32, nsamples: usize, use_ids: bool, seed: u64) -> Case {
    let t = refs::traits(ty).unwrap();
    let mut rng = Rng(seed);
    let bpr = ncols_x / t.qk;
    let nchannels_x = nch / ratio as usize;
    let nx_ch = nchannels_x.max(if use_ids { 4 } else { 1 });
    let ysc = (ncols_x / 32) as u32;
    let nch_y = if use_ids { 3 } else { nch };
    let p = Mmvq {
        ty,
        ncols_dst: ncols,
        nwarps: Mmvq::ggml_nwarps(ncols),
        nrows_x: nrows,
        ncols_x,
        nchannels_dst: nch,
        nsamples_dst: nsamples,
        channel_ratio: ratio,
        sample_ratio: 1,
        nchannels_y: nch_y as u32,
        use_ids,
        stride_row_x: bpr as u32,
        stride_col_y: ysc,
        stride_col_dst: nrows as u32,
        stride_channel_x: (nrows * bpr) as u32,
        stride_channel_y: ysc * ncols as u32,
        stride_channel_dst: (nrows * ncols) as u32,
        stride_sample_x: (nrows * bpr * nx_ch) as u32,
        stride_sample_y: ysc * ncols as u32 * nch_y as u32,
        stride_sample_dst: (nrows * ncols * nch) as u32,
    };
    let x = refs::random_blocks(ty, nrows * bpr * nx_ch * nsamples, &mut rng);
    let acts: Vec<f32> = (0..ncols_x * ncols * nch_y * nsamples).map(|_| rng.unit() * 3.0).collect();
    let y = refs::quantize_q8_1(&acts);
    let ids = (0..nch).map(|i| ((i * 3 + 1) % 4) as i32).collect();
    Case { p, x, y, ids }
}

fn make_mmq_case(nrows: usize, ncols_x: usize, ncols: usize, nch: usize, ratio: usize, nsamples: usize, seed: u64) -> MmqCase {
    let mut rng = Rng(seed);
    let nb = ncols_x / 32;
    let nch_x = nch / ratio;
    let ychan = ncols_x / 128 * ncols * 36;
    let p = refs::Mmq {
        ncols_x,
        nrows_x: nrows,
        ncols_dst: ncols,
        nchannels_y: nch,
        nsamples_y: nsamples,
        channel_ratio: ratio,
        sample_ratio: 1,
        stride_row_x: nb,
        ncols_y: ncols,
        stride_col_dst: nrows,
        stride_channel_x: nrows * nb,
        stride_channel_y: ychan,
        stride_channel_dst: nrows * ncols,
        stride_sample_x: nrows * nb * nch_x,
        stride_sample_y: ychan * nch,
        stride_sample_dst: nrows * ncols * nch,
    };
    let x = refs::random_blocks(refs::Q8_0, nrows * nb * nch_x * nsamples, &mut rng);
    let mut y = Vec::new();
    for _ in 0..nch * nsamples {
        let acts: Vec<f32> = (0..ncols_x * ncols).map(|_| rng.unit() * 2.0).collect();
        y.extend(refs::quantize_mmq_d4(&acts, ncols_x, ncols));
    }
    MmqCase { p, x, y }
}

struct MmqCase {
    p: refs::Mmq,
    x: Vec<u8>,
    y: Vec<u8>,
}

/// `mul_mat_q` parameters: six pointers (ids, expert bounds unused) and 17 words.
fn mmq_params(p: &refs::Mmq, x: u64, y: u64, dst: u64, tmp: u64) -> Vec<Vec<u8>> {
    let w = |v: usize| (v as u32).to_le_bytes().to_vec();
    let q = |v: u64| v.to_le_bytes().to_vec();
    vec![
        q(x), q(y), q(0), q(0), q(dst), q(tmp),
        w(p.ncols_x), w(p.nrows_x), w(p.ncols_dst), w(p.stride_row_x), w(p.ncols_y), w(p.stride_col_dst),
        w(p.channel_ratio), w(p.nchannels_y), w(p.stride_channel_x), w(p.stride_channel_y), w(p.stride_channel_dst),
        w(p.sample_ratio), w(p.nsamples_y), w(p.stride_sample_x), w(p.stride_sample_y), w(p.stride_sample_dst),
        w(p.ncols_dst),
    ]
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn type_label(ty: u32) -> &'static str {
    match ty {
        refs::Q4_0 => "q4_0",
        refs::Q8_0 => "q8_0",
        refs::Q4_K => "q4_k",
        _ => "q6_k",
    }
}

#[test]
#[ignore = "requires CFI_STOCK_GGML_DLL, a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_subst_native_vs_translated() {
    let path = std::env::var("CFI_STOCK_GGML_DLL").expect("CFI_STOCK_GGML_DLL");
    let reps: usize = std::env::var("CFI_SUBST_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let sweep = std::env::var_os("CFI_SUBST_NWARPS_SWEEP").is_some();
    let want_types: Vec<String> = std::env::var("CFI_SUBST_TYPES")
        .unwrap_or_else(|_| "q8_0,q4_0,q4_k,q6_k".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let blobs = extract_all(&path);
    let mmvq_blob = blobs.iter().find(|b| b.contains(".entry _Z13mul_mat_vec_qIL9ggml_type8ELi1E")).expect("mmvq blob");
    let quant_blob = blobs.iter().find(|b| b.contains(".entry _Z13quantize_q8_1PKfPvxxxxxj5uint3(")).expect("quantize blob");
    let _ = std::fs::remove_file("cuda-for-intel-export.log");

    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxSynchronize(),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
    );
    let launch_fn = unsafe {
        dll.get::<unsafe extern "system" fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut Handle) -> i32>(b"cuLaunchKernel\0")
    }
    .expect("cuLaunchKernel");
    let api = Api {
        launch: *launch_fn,
        sync: *cuCtxSynchronize,
        alloc: *cuMemAlloc_v2,
        htod: *cuMemcpyHtoD_v2,
        dtoh: *cuMemcpyDtoH_v2,
    };
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));

    // Load each library twice: with substitution off (translated PTX) and on (native).
    let load = |blob: &str, switch: &str, nwarps: Option<u32>| -> Handle {
        // SAFETY: single-threaded test; the driver reads the variables when it installs entries.
        unsafe {
            if switch.is_empty() {
                std::env::remove_var("CUDA_FOR_INTEL_SUBST");
            } else {
                std::env::set_var("CUDA_FOR_INTEL_SUBST", switch);
            }
            match nwarps {
                Some(w) => std::env::set_var("CUDA_FOR_INTEL_SUBST_NWARPS", w.to_string()),
                None => std::env::remove_var("CUDA_FOR_INTEL_SUBST_NWARPS"),
            }
        }
        let src = std::ffi::CString::new(blob).unwrap();
        let mut module = null_mut();
        let status = unsafe { cuModuleLoadData(&mut module, src.as_ptr().cast()) };
        assert_eq!(status, 0, "cuModuleLoadData");
        module
    };
    let function = |module: Handle, name: &str| -> Handle {
        let c = std::ffi::CString::new(name).unwrap();
        let mut f = null_mut();
        let status = unsafe { cuModuleGetFunction(&mut f, module, c.as_ptr()) };
        assert_eq!(status, 0, "cuModuleGetFunction({name})");
        f
    };
    let plain_mmvq = load(mmvq_blob, "", None);
    let native_mmvq = load(mmvq_blob, "mmvq", None);
    let wide_mmvq = sweep.then(|| load(mmvq_blob, "mmvq", Some(8)));
    let mmq_blob = blobs.iter().find(|b| b.contains(".entry _Z9mul_mat_qIL9ggml_type8ELi64ELb0E")).expect("mmq blob");
    let skip_mmq = std::env::var_os("CFI_SUBST_SKIP_MMQ").is_some();
    let (plain_mmq, native_mmq) = if skip_mmq { (null_mut(), null_mut()) } else { (load(mmq_blob, "", None), load(mmq_blob, "mmq", None)) };
    let plain_quant = load(quant_blob, "", None);
    let native_quant = load(quant_blob, "quantize_q8_1", None);
    unsafe {
        std::env::remove_var("CUDA_FOR_INTEL_SUBST");
        std::env::remove_var("CUDA_FOR_INTEL_SUBST_NWARPS");
    }

    let mut worst_nmse = 0f64;
    let mut failures = Vec::new();
    println!("== correctness: native vs translated vs host reference (rows {SHAPE_ROWS}, cols {SHAPE_COLS})");
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        if !want_types.iter().any(|t| t == type_label(ty)) {
            continue;
        }
        for ncols in 1..=8usize {
            let entry = format!("_Z13mul_mat_vec_qIL9ggml_type{ty}ELi{ncols}EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj");
            let (fp, fnat) = (function(plain_mmvq, &entry), function(native_mmvq, &entry));
            // plain shape, then channels/samples (ratio 2, 2 samples); ids only exists for ncols == 1
            let mut shapes = vec![(1usize, 1u32, 1usize, false), (4, 2, 2, false)];
            if ncols == 1 {
                shapes.push((5, 1, 1, true));
            }
            for (nch, ratio, nsamples, use_ids) in shapes {
                let c = make_case(ty, ncols, SHAPE_ROWS, SHAPE_COLS, nch, ratio, nsamples, use_ids, 0xC0DE + ty as u64 * 16 + ncols as u64);
                let n = c.p.nsamples_dst * c.p.stride_sample_dst as usize;
                let mut want = vec![0f32; n];
                refs::mmvq(&c.p, &c.x, &c.y, use_ids.then_some(&c.ids[..]), &mut want);
                let (xa, ya) = (upload(&api, &c.x), upload(&api, &c.y));
                let ia = if use_ids { upload(&api, &bytes_u32(&c.ids.iter().map(|v| *v as u32).collect::<Vec<_>>())) } else { 0 };
                let mut results = Vec::new();
                for f in [fp, fnat] {
                    let da = upload(&api, &vec![0u8; n * 4]);
                    let params = mmvq_params(&c.p, xa, ya, ia, da);
                    assert_eq!(launch(&api, f, c.p.grid(), c.p.block(), &params), 0);
                    assert_eq!(unsafe { (api.sync)() }, 0);
                    results.push(floats(&download(&api, da, n * 4)));
                }
                let (translated, native) = (&results[0], &results[1]);
                let vs_ref = refs::nmse(native, &want);
                let vs_translated = refs::nmse(native, translated);
                let ptx_vs_ref = refs::nmse(translated, &want);
                let bitwise = native.iter().zip(translated).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                worst_nmse = worst_nmse.max(vs_translated).max(vs_ref);
                let tag = format!("{} ncols {ncols} ch {nch}/{ratio} samples {nsamples} ids {use_ids}", type_label(ty));
                println!("{tag}: native-vs-translated nmse {vs_translated:.2e} ({bitwise} bitwise diffs), native-vs-ref {vs_ref:.2e}, translated-vs-ref {ptx_vs_ref:.2e}");
                if !(vs_translated < 1e-9 && vs_ref < 1e-9) {
                    failures.push(tag);
                }
            }
        }
    }

    println!("== quantize_q8_1 (byte exact)");
    {
        let (ne00, ne0, ne1, ne2, ne3) = (1000usize, 1024usize, 7usize, 3usize, 2usize);
        let mut rng = Rng(0x71);
        let x: Vec<f32> = (0..ne00 * ne1 * ne2 * ne3).map(|_| rng.unit() * 8.0).collect();
        let mut want = Vec::new();
        for r in 0..ne1 * ne2 * ne3 {
            let mut row = x[r * ne00..(r + 1) * ne00].to_vec();
            row.resize(ne0, 0.0);
            want.extend(refs::quantize_q8_1(&row));
        }
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xa = upload(&api, &xb);
        let name = "_Z13quantize_q8_1PKfPvxxxxxj5uint3";
        let mut outs = Vec::new();
        for module in [plain_quant, native_quant] {
            let f = function(module, name);
            let ya = upload(&api, &vec![0u8; want.len()]);
            let q = |v: usize| (v as u64).to_le_bytes().to_vec();
            let params = vec![
                xa.to_le_bytes().to_vec(),
                ya.to_le_bytes().to_vec(),
                q(ne00),
                q(ne00),
                q(ne00 * ne1),
                q(ne00 * ne1 * ne2),
                q(ne0),
                (ne1 as u32).to_le_bytes().to_vec(),
                bytes_u32(&refs::fastdiv_values(ne2 as u32)),
            ];
            assert_eq!(launch(&api, f, [ne0.div_ceil(256) as u32, ne1 as u32, (ne2 * ne3) as u32], [256, 1, 1], &params), 0);
            assert_eq!(unsafe { (api.sync)() }, 0);
            outs.push(download(&api, ya, want.len()));
        }
        let (translated, native) = (&outs[0], &outs[1]);
        let t_bad = translated.iter().zip(&want).filter(|(a, b)| a != b).count();
        let n_bad = native.iter().zip(&want).filter(|(a, b)| a != b).count();
        let nt_bad = native.iter().zip(translated).filter(|(a, b)| a != b).count();
        println!("quantize_q8_1: translated-vs-ref {t_bad} bytes differ, native-vs-ref {n_bad}, native-vs-translated {nt_bad}");
        if n_bad != 0 || nt_bad != 0 {
            failures.push("quantize_q8_1".into());
        }
    }

    println!("== mul_mat_q Q8_0 (prefill): native vs translated vs host reference");
    if !skip_mmq {
        let cases = [
            // (nrows, ncols_x, ncols, mmq_x, need_check, nch, ratio, nsamples)
            (256usize, 1024usize, 64usize, 64usize, 0u32, 1usize, 1usize, 1usize),
            (300, 1024, 40, 64, 1, 2, 1, 1),
            (384, 512, 17, 32, 1, 4, 2, 2),
        ];
        for (nrows, ncols_x, ncols, mmq_x, need_check, nch, ratio, nsamples) in cases {
            let entry = format!("_Z9mul_mat_qIL9ggml_type8ELi{mmq_x}ELb{need_check}EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
            let c = make_mmq_case(nrows, ncols_x, ncols, nch, ratio, nsamples, 7 + nrows as u64);
            let n = c.p.nsamples_y * c.p.stride_sample_dst;
            let mut want = vec![0f32; n];
            refs::mmq_q8_0(&c.p, &c.x, &c.y, &mut want);
            let (xa, ya) = (upload(&api, &c.x), upload(&api, &c.y));
            let tmp = upload(&api, &vec![0u8; 1 << 22]);
            let mut outs = Vec::new();
            for module in [plain_mmq, native_mmq] {
                let f = function(module, &entry);
                let da = upload(&api, &vec![0u8; n * 4]);
                let params = mmq_params(&c.p, xa, ya, da, tmp);
                // one stock CTA per tile: no stream-k split, no fixup kernel
                let tiles = (c.p.ncols_dst.div_ceil(mmq_x) * nrows.div_ceil(128) * nch * nsamples) as u32;
                assert_eq!(launch(&api, f, [tiles, 1, 1], [32, 8, 1], &params), 0);
                assert_eq!(unsafe { (api.sync)() }, 0);
                outs.push(floats(&download(&api, da, n * 4)));
            }
            let (t, nat) = (&outs[0], &outs[1]);
            let (a, b, d) = (refs::nmse(nat, &want), refs::nmse(nat, t), refs::nmse(t, &want));
            println!("rows {nrows} k {ncols_x} cols {ncols} mmq_x {mmq_x} ch {nch}/{ratio} s {nsamples}: native-vs-ref {a:.2e}, native-vs-translated {b:.2e}, translated-vs-ref {d:.2e}");
            if !(a < 1e-9 && b < 1e-9) {
                failures.push(format!("mmq {entry}"));
            }
        }
        // bandwidth-ish throughput: 4096 x 4096 weights, 512 prompt columns
        let (nrows, ncols_x, ncols, mmq_x) = (4096usize, 4096usize, 512usize, 64usize);
        let entry = format!("_Z9mul_mat_qIL9ggml_type8ELi{mmq_x}ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
        let c = make_mmq_case(nrows, ncols_x, ncols, 1, 1, 1, 3);
        let (xa, ya) = (upload(&api, &c.x), upload(&api, &c.y));
        let tmp = upload(&api, &vec![0u8; 1 << 22]);
        let da = upload(&api, &vec![0u8; nrows * ncols * 4]);
        let params = mmq_params(&c.p, xa, ya, da, tmp);
        let tiles = (ncols.div_ceil(mmq_x) * nrows.div_ceil(128)) as u32;
        let flops = 2.0 * nrows as f64 * ncols as f64 * ncols_x as f64;
        println!("type  translated TOPS  native TOPS  speedup  (m={nrows} k={ncols_x} n={ncols}, stock one CTA per tile)");
        let mut rates = Vec::new();
        for module in [plain_mmq, native_mmq] {
            let f = function(module, &entry);
            for _ in 0..2 {
                assert_eq!(launch(&api, f, [tiles, 1, 1], [32, 8, 1], &params), 0);
            }
            assert_eq!(unsafe { (api.sync)() }, 0);
            let start = Instant::now();
            for _ in 0..reps {
                assert_eq!(launch(&api, f, [tiles, 1, 1], [32, 8, 1], &params), 0);
            }
            assert_eq!(unsafe { (api.sync)() }, 0);
            rates.push(flops * reps as f64 / start.elapsed().as_secs_f64() / 1e12);
        }
        println!("q8_0  {:15.3}  {:11.3}  {:6.2}x", rates[0], rates[1], rates[1] / rates[0]);
    }

    println!("== mul_mat_vec_q bandwidth (rows 8192, cols 4096, {reps} launches each)");
    println!("type  ncols  translated GB/s  native GB/s  speedup{}", if sweep { "  native-8warps GB/s" } else { "" });
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        if !want_types.iter().any(|t| t == type_label(ty)) {
            continue;
        }
        for ncols in [1usize, 2, 4, 8] {
            let entry = format!("_Z13mul_mat_vec_qIL9ggml_type{ty}ELi{ncols}EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj");
            let c = make_case(ty, ncols, 8192, 4096, 1, 1, 1, false, 99);
            let (xa, ya) = (upload(&api, &c.x), upload(&api, &c.y));
            let da = upload(&api, &vec![0u8; ncols * 8192 * 4]);
            let params = mmvq_params(&c.p, xa, ya, 0, da);
            let bytes = (c.x.len() + c.y.len()) as f64;
            let mut time = |f: Handle, block: [u32; 3]| -> f64 {
                for _ in 0..3 {
                    assert_eq!(launch(&api, f, c.p.grid(), block, &params), 0);
                }
                assert_eq!(unsafe { (api.sync)() }, 0);
                let start = Instant::now();
                for _ in 0..reps {
                    assert_eq!(launch(&api, f, c.p.grid(), block, &params), 0);
                }
                assert_eq!(unsafe { (api.sync)() }, 0);
                bytes * reps as f64 / start.elapsed().as_secs_f64() / 1e9
            };
            let t = time(function(plain_mmvq, &entry), c.p.block());
            let n = time(function(native_mmvq, &entry), c.p.block());
            let wide = wide_mmvq.map(|m| time(function(m, &entry), c.p.block()));
            println!(
                "{:5} {ncols:5}  {t:15.1}  {n:11.1}  {:6.2}x{}",
                type_label(ty),
                n / t,
                wide.map(|w| format!("  {w:18.1}")).unwrap_or_default()
            );
        }
    }

    let log = std::fs::read_to_string("cuda-for-intel-export.log").unwrap_or_default();
    let ready = log.lines().filter(|l| l.starts_with("subst: native") && l.contains("ready")).count();
    let rejected: Vec<&str> = log.lines().filter(|l| l.contains("subst:") && (l.contains("rejected") || l.contains("failed") || l.contains("keeping translated"))).collect();
    println!("native kernels ready: {ready}; rejected/failed: {}", rejected.len());
    for line in &rejected {
        println!("  {line}");
    }
    println!("worst NMSE {worst_nmse:.2e}");
    assert!(ready > 0, "no native kernel was installed; check CUDA_FOR_INTEL_SUBST handling and cuda-for-intel-export.log");
    assert!(rejected.is_empty(), "the driver rejected native kernels (they ran as translated PTX)");
    assert!(failures.is_empty(), "native results differ from reference: {failures:?}");
}

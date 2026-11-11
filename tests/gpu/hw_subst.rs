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
    // ggml pads y's K to a multiple of 512 with zeros; the stock kernel consumes 8 Q8_0 blocks per step
    let kpad = ncols_x.next_multiple_of(512);
    let ychan = kpad / 128 * ncols * 36;
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
    let mut x = refs::random_blocks(refs::Q8_0, nrows * nb * nch_x * nsamples, &mut rng);
    x.extend(vec![0u8; 34 * 8]); // the stock kernel may read up to 7 blocks past the last row
    let mut y = Vec::new();
    for _ in 0..nch * nsamples {
        let mut acts = vec![0f32; kpad * ncols];
        for c in 0..ncols {
            for k in 0..ncols_x {
                acts[c * kpad + k] = rng.unit() * 2.0;
            }
        }
        y.extend(refs::quantize_mmq_d4(&acts, kpad, ncols));
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
        cuLibraryLoadData(*mut Handle, *const c_void, *mut c_void, *mut c_void, u32, *mut c_void, *mut c_void, u32),
        cuLibraryGetKernel(*mut Handle, Handle, *const c_char),
        cuKernelGetFunction(*mut Handle, Handle),
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

    // Libraries load lazily (one translated module per resolved kernel), so a huge blob such as the
    // whole mul_mat_q set is never compiled at once. Substitution is decided when a kernel is
    // resolved, hence the switch is set around each `function` call; each library is resolved under
    // exactly one setting (load it twice for plain and native).
    let load = |blob: &str| -> Handle {
        let src = std::ffi::CString::new(blob).unwrap();
        let mut library = null_mut();
        let status = unsafe {
            cuLibraryLoadData(&mut library, src.as_ptr().cast(), null_mut(), null_mut(), 0, null_mut(), null_mut(), 0)
        };
        assert_eq!(status, 0, "cuLibraryLoadData");
        library
    };
    let function = |library: Handle, name: &str, switch: &str, nwarps: Option<u32>| -> Handle {
        // SAFETY: single-threaded test; the driver reads the variables when it resolves the kernel.
        unsafe {
            if switch.is_empty() {
                // unset now means `auto`; the translated reference needs an explicit `none`
                std::env::set_var("CUDA_FOR_INTEL_SUBST", "none");
            } else {
                std::env::set_var("CUDA_FOR_INTEL_SUBST", switch);
            }
            match nwarps {
                // NWARPS shapes the stock-order kernel only; ncols 1 would otherwise take the
                // decode-fast default (`CUDA_FOR_INTEL_MMVQ_FAST`), which ignores it
                Some(w) => {
                    std::env::set_var("CUDA_FOR_INTEL_SUBST_NWARPS", w.to_string());
                    std::env::set_var("CUDA_FOR_INTEL_MMVQ_FAST", "0");
                }
                None => std::env::remove_var("CUDA_FOR_INTEL_SUBST_NWARPS"),
            }
        }
        let c = std::ffi::CString::new(name).unwrap();
        let mut kernel = null_mut();
        assert_eq!(unsafe { cuLibraryGetKernel(&mut kernel, library, c.as_ptr()) }, 0, "cuLibraryGetKernel({name})");
        let mut f = null_mut();
        assert_eq!(unsafe { cuKernelGetFunction(&mut f, kernel) }, 0, "cuKernelGetFunction({name})");
        unsafe {
            std::env::remove_var("CUDA_FOR_INTEL_SUBST");
            std::env::remove_var("CUDA_FOR_INTEL_SUBST_NWARPS");
            if nwarps.is_some() {
                std::env::remove_var("CUDA_FOR_INTEL_MMVQ_FAST");
            }
        }
        f
    };
    let plain_mmvq = load(mmvq_blob);
    let native_mmvq = load(mmvq_blob);
    let wide_mmvq = sweep.then(|| load(mmvq_blob));
    let mmq_blob = blobs.iter().find(|b| b.contains(".entry _Z9mul_mat_qIL9ggml_type8ELi64ELb0E")).expect("mmq blob");
    let (plain_mmq, native_mmq) = (load(mmq_blob), load(mmq_blob));
    let (plain_quant, native_quant) = (load(quant_blob), load(quant_blob));

    let mut worst_nmse = 0f64;
    let mut failures = Vec::new();
    println!("== correctness: native vs translated vs host reference (rows {SHAPE_ROWS}, cols {SHAPE_COLS})");
    for ty in [refs::Q8_0, refs::Q4_0, refs::Q4_K, refs::Q6_K] {
        if !want_types.iter().any(|t| t == type_label(ty)) {
            continue;
        }
        for ncols in 1..=8usize {
            let entry = format!("_Z13mul_mat_vec_qIL9ggml_type{ty}ELi{ncols}EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj");
            let (fp, fnat) = (function(plain_mmvq, &entry, "", None), function(native_mmvq, &entry, "mmvq", None));
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
        for (module, switch) in [(plain_quant, ""), (native_quant, "quantize_q8_1")] {
            let f = function(module, name, switch, None);
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
    {
        let cases = [
            // (nrows, ncols_x, ncols, mmq_x, need_check, nch, ratio, nsamples)
            (256usize, 1024usize, 64usize, 64usize, 0u32, 1usize, 1usize, 1usize),
            (300, 1024, 40, 64, 1, 2, 1, 1),
            (384, 512, 17, 32, 1, 4, 2, 2),
        ];
        // Qwen2.5-0.5B shapes (k = 896 is not a multiple of the stock kernel's 256-value step)
        let mut cases = cases.to_vec();
        for (m, k) in [(896usize, 896usize), (896, 4864), (4864, 896), (1024, 896)] {
            for n in [1usize, 2, 7, 16, 33, 64] {
                let mmq_x = n.next_multiple_of(8).clamp(8, 64);
                cases.push((m, k, n, mmq_x, if m % 128 == 0 { 0 } else { 1 }, 1, 1, 1));
            }
        }
        for (nrows, ncols_x, ncols, mmq_x, need_check, nch, ratio, nsamples) in cases {
            let entry = format!("_Z9mul_mat_qIL9ggml_type8ELi{mmq_x}ELb{need_check}EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
            let c = make_mmq_case(nrows, ncols_x, ncols, nch, ratio, nsamples, 7 + nrows as u64);
            let n = c.p.nsamples_y * c.p.stride_sample_dst;
            let mut want = vec![0f32; n];
            refs::mmq_q8_0(&c.p, &c.x, &c.y, &mut want);
            let (xa, ya) = (upload(&api, &c.x), upload(&api, &c.y));
            let tmp = upload(&api, &vec![0u8; 1 << 22]);
            let mut outs = Vec::new();
            for (module, switch) in [(plain_mmq, ""), (native_mmq, "mmq")] {
                let f = function(module, &entry, switch, None);
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
        for (module, switch) in [(plain_mmq, ""), (native_mmq, "mmq")] {
            let f = function(module, &entry, switch, None);
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
            let t = time(function(plain_mmvq, &entry, "", None), c.p.block());
            let n = time(function(native_mmvq, &entry, "mmvq", None), c.p.block());
            let wide = wide_mmvq.map(|m| time(function(m, &entry, "mmvq", Some(8)), c.p.block()));
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

/// `mul_mat_q` for every native type (Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K): native vs translated
/// stock vs host reference on Llama-3.2-1B / Qwen2.5-1.5B / Qwen2.5-0.5B prefill shapes, then
/// translated and native throughput at 4096 x 4096 x 512. One GPU slot:
///   set CFI_STOCK_GGML_DLL=<path>\ggml-cuda.dll
///   cargo test --test gpu hw_subst_mmq -- --ignored --nocapture --test-threads=1
/// Optional: CFI_SUBST_REPS (default 20), CFI_SUBST_TYPES=q4_0,q5_0,q8_0,q4_k,q5_k,q6_k.
#[test]
#[ignore = "requires CFI_STOCK_GGML_DLL, a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_subst_mmq_types() {
    let path = std::env::var("CFI_STOCK_GGML_DLL").expect("CFI_STOCK_GGML_DLL");
    let reps: usize = std::env::var("CFI_SUBST_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let want_types: Vec<String> = std::env::var("CFI_SUBST_TYPES")
        .unwrap_or_else(|_| "q8_0,q4_0,q5_0,q4_k,q5_k,q6_k".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let blobs = extract_all(&path);
    let _ = std::fs::remove_file("cuda-for-intel-export.log");
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxSynchronize(),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLibraryLoadData(*mut Handle, *const c_void, *mut c_void, *mut c_void, u32, *mut c_void, *mut c_void, u32),
        cuLibraryGetKernel(*mut Handle, Handle, *const c_char),
        cuKernelGetFunction(*mut Handle, Handle),
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
    let load = |blob: &str| -> Handle {
        let src = std::ffi::CString::new(blob).unwrap();
        let mut library = null_mut();
        let status = unsafe {
            cuLibraryLoadData(&mut library, src.as_ptr().cast(), null_mut(), null_mut(), 0, null_mut(), null_mut(), 0)
        };
        assert_eq!(status, 0, "cuLibraryLoadData");
        library
    };
    let function = |library: Handle, name: &str, switch: &str| -> Handle {
        // SAFETY: single-threaded test; the driver reads the variable when it resolves the kernel.
        unsafe { std::env::set_var("CUDA_FOR_INTEL_SUBST", if switch.is_empty() { "none" } else { switch }) };
        let c = std::ffi::CString::new(name).unwrap();
        let mut kernel = null_mut();
        assert_eq!(unsafe { cuLibraryGetKernel(&mut kernel, library, c.as_ptr()) }, 0, "cuLibraryGetKernel({name})");
        let mut f = null_mut();
        assert_eq!(unsafe { cuKernelGetFunction(&mut f, kernel) }, 0, "cuKernelGetFunction({name})");
        unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
        f
    };
    let mut failures = Vec::new();
    let mut summary = Vec::new();
    let types: [(u32, &str); 6] = [
        (refs::Q8_0, "q8_0"),
        (refs::Q4_0, "q4_0"),
        (refs::Q5_0, "q5_0"),
        (refs::Q4_K, "q4_k"),
        (refs::Q5_K, "q5_k"),
        (refs::Q6_K, "q6_k"),
    ];
    for (ty, label) in types {
        if !want_types.iter().any(|t| t == label) {
            continue;
        }
        let Some(blob) = blobs.iter().find(|b| b.contains(&format!(".entry _Z9mul_mat_qIL9ggml_type{ty}ELi64ELb0E"))) else {
            println!("{label}: no mul_mat_q blob");
            continue;
        };
        let (plain, native) = (load(blob), load(blob));
        let bytes = refs::mmq_block_bytes(ty);
        let per_block = 32 * refs::mmq_groups(ty);
        let mk = |nrows: usize, ncols_x: usize, ncols: usize, seed: u64| -> (refs::Mmq, Vec<u8>, Vec<u8>) {
            let mut rng = Rng(seed);
            let nb = ncols_x / per_block;
            let kpad = ncols_x.next_multiple_of(512);
            let ychan = kpad / 128 * ncols * 36;
            let p = refs::Mmq {
                ncols_x,
                nrows_x: nrows,
                ncols_dst: ncols,
                nchannels_y: 1,
                nsamples_y: 1,
                channel_ratio: 1,
                sample_ratio: 1,
                stride_row_x: nb,
                ncols_y: ncols,
                stride_col_dst: nrows,
                stride_channel_x: nrows * nb,
                stride_channel_y: ychan,
                stride_channel_dst: nrows * ncols,
                stride_sample_x: nrows * nb,
                stride_sample_y: ychan,
                stride_sample_dst: nrows * ncols,
            };
            let mut x = refs::random_blocks(ty, nrows * nb, &mut rng);
            x.extend(vec![0u8; bytes * 16]);
            let mut acts = vec![0f32; kpad * ncols];
            for c in 0..ncols {
                for k in 0..ncols_x {
                    acts[c * kpad + k] = rng.unit() * 2.0;
                }
            }
            (p, x, refs::quantize_mmq(ty, &acts, kpad, ncols))
        };
        println!("== mul_mat_q {label}: native vs translated vs host reference");
        // Llama-3.2-1B (2048/8192/512), Qwen2.5-1.5B (1536/8960/256), Qwen2.5-0.5B (896/4864) shapes
        let mut cases: Vec<(usize, usize, usize)> = Vec::new();
        for (m, k) in [(2048usize, 2048usize), (8192, 2048), (2048, 8192), (512, 2048), (1536, 1536), (8960, 1536), (1536, 8960), (256, 1536), (300, 1024)] {
            for n in [7usize, 33, 64] {
                cases.push((m, k, n));
            }
        }
        if per_block == 32 {
            cases.push((896, 896, 16));
            cases.push((4864, 896, 33));
        }
        let mut worst = (0f64, 0f64, 0usize);
        for (nrows, ncols_x, ncols) in cases {
            let mmq_x = ncols.next_multiple_of(8).clamp(8, 64);
            let need_check = u32::from(nrows % 128 != 0);
            let entry = format!("_Z9mul_mat_qIL9ggml_type{ty}ELi{mmq_x}ELb{need_check}EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
            let (p, x, y) = mk(nrows, ncols_x, ncols, 7 + nrows as u64 + u64::from(ty));
            let n = p.stride_sample_dst;
            let mut want = vec![0f32; n];
            refs::mmq_ref(ty, &p, &x, &y, &mut want);
            let (xa, ya) = (upload(&api, &x), upload(&api, &y));
            let tmp = upload(&api, &vec![0u8; 1 << 22]);
            let mut outs = Vec::new();
            for (module, switch) in [(plain, ""), (native, "mmq")] {
                let f = function(module, &entry, switch);
                let da = upload(&api, &vec![0u8; n * 4]);
                let params = mmq_params(&p, xa, ya, da, tmp);
                let tiles = (ncols.div_ceil(mmq_x) * nrows.div_ceil(128)) as u32;
                assert_eq!(launch(&api, f, [tiles, 1, 1], [32, 8, 1], &params), 0);
                assert_eq!(unsafe { (api.sync)() }, 0);
                outs.push(floats(&download(&api, da, n * 4)));
            }
            let (t, nat) = (&outs[0], &outs[1]);
            let (a, b, d) = (refs::nmse(nat, &want), refs::nmse(nat, t), refs::nmse(t, &want));
            let bitwise = nat.iter().zip(t).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("{label} rows {nrows} k {ncols_x} cols {ncols} mmq_x {mmq_x}: native-vs-ref {a:.2e}, native-vs-translated {b:.2e} ({bitwise} bitwise), translated-vs-ref {d:.2e}");
            worst = (worst.0.max(a), worst.1.max(b), worst.2 + bitwise);
            if !(a < 1e-9 && b < 1e-9) {
                failures.push(format!("{label} {entry} {nrows}x{ncols_x}x{ncols}"));
            }
        }
        // throughput: 4096 x 4096 weights, 512 prompt columns
        let (nrows, ncols_x, ncols, mmq_x) = (4096usize, 4096usize, 512usize, 64usize);
        let entry = format!("_Z9mul_mat_qIL9ggml_type{ty}ELi{mmq_x}ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii");
        let (p, x, y) = mk(nrows, ncols_x, ncols, 3);
        let (xa, ya) = (upload(&api, &x), upload(&api, &y));
        let tmp = upload(&api, &vec![0u8; 1 << 22]);
        let da = upload(&api, &vec![0u8; nrows * ncols * 4]);
        let params = mmq_params(&p, xa, ya, da, tmp);
        let tiles = (ncols.div_ceil(mmq_x) * nrows.div_ceil(128)) as u32;
        let flops = 2.0 * nrows as f64 * ncols as f64 * ncols_x as f64;
        let mut rates = Vec::new();
        for (module, switch) in [(plain, ""), (native, "mmq")] {
            let f = function(module, &entry, switch);
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
        println!("{label}: translated {:.3} TOPS, native {:.3} TOPS, speedup {:.2}x", rates[0], rates[1], rates[1] / rates[0]);
        summary.push(format!(
            "{label:6} speedup {:6.2}x  worst native-vs-ref {:.2e}  native-vs-translated {:.2e}  bitwise-diff floats {}",
            rates[1] / rates[0],
            worst.0,
            worst.1,
            worst.2
        ));
    }
    println!("== summary (mmq native vs translated stock, 4096x4096x512, one CTA per tile)");
    for l in &summary {
        println!("{l}");
    }
    let log = std::fs::read_to_string("cuda-for-intel-export.log").unwrap_or_default();
    let rejected: Vec<&str> = log.lines().filter(|l| l.contains("subst:") && (l.contains("rejected") || l.contains("failed") || l.contains("keeping translated"))).collect();
    for line in &rejected {
        println!("  {line}");
    }
    assert!(rejected.is_empty(), "the driver rejected native kernels (they ran as translated PTX)");
    assert!(failures.is_empty(), "native results differ from reference: {failures:?}");
}

/// Fuzz of `quantize_q8_1`: native vs translated stock kernel vs host reference on a few million
/// realistic activations (wide per-row magnitudes, outliers, zero blocks). Reports how many
/// `d`, `sum` and quant bytes differ and prints examples. One GPU slot:
///   cargo test --test gpu hw_subst_quantize_fuzz -- --ignored --nocapture --test-threads=1
#[test]
#[ignore = "requires CFI_STOCK_GGML_DLL, a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_subst_quantize_fuzz() {
    let path = std::env::var("CFI_STOCK_GGML_DLL").expect("CFI_STOCK_GGML_DLL");
    let blobs = extract_all(&path);
    let quant_blob = blobs.iter().find(|b| b.contains(".entry _Z13quantize_q8_1PKfPvxxxxxj5uint3(")).expect("quantize blob");
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxSynchronize(),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLibraryLoadData(*mut Handle, *const c_void, *mut c_void, *mut c_void, u32, *mut c_void, *mut c_void, u32),
        cuLibraryGetKernel(*mut Handle, Handle, *const c_char),
        cuKernelGetFunction(*mut Handle, Handle),
    );
    let launch_fn = unsafe {
        dll.get::<unsafe extern "system" fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut Handle) -> i32>(b"cuLaunchKernel\0")
    }
    .expect("cuLaunchKernel");
    let api = Api { launch: *launch_fn, sync: *cuCtxSynchronize, alloc: *cuMemAlloc_v2, htod: *cuMemcpyHtoD_v2, dtoh: *cuMemcpyDtoH_v2 };
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let load = |blob: &str| -> Handle {
        let src = std::ffi::CString::new(blob).unwrap();
        let mut library = null_mut();
        assert_eq!(unsafe { cuLibraryLoadData(&mut library, src.as_ptr().cast(), null_mut(), null_mut(), 0, null_mut(), null_mut(), 0) }, 0);
        library
    };
    let function = |library: Handle, switch: &str| -> Handle {
        unsafe { std::env::set_var("CUDA_FOR_INTEL_SUBST", switch) };
        let c = std::ffi::CString::new("_Z13quantize_q8_1PKfPvxxxxxj5uint3").unwrap();
        let mut kernel = null_mut();
        assert_eq!(unsafe { cuLibraryGetKernel(&mut kernel, library, c.as_ptr()) }, 0);
        let mut f = null_mut();
        assert_eq!(unsafe { cuKernelGetFunction(&mut f, kernel) }, 0);
        unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
        f
    };
    let (plain, native) = (load(quant_blob), load(quant_blob));
    let (f_plain, f_native) = (function(plain, "none"), function(native, "quantize_q8_1"));
    let mut rng = Rng(0xFEED);
    let (ne0, ne1) = (4096usize, 512usize);
    let mut total = (0usize, 0usize, 0usize, 0usize, 0usize); // blocks, d diffs, sum diffs, q diffs, q diffs vs host ref (translated)
    let mut shown = 0;
    for round in 0..6 {
        let mut x = vec![0f32; ne0 * ne1];
        for r in 0..ne1 {
            let scale = 10f32.powf(rng.unit() * 2.5 - 0.5);
            for c in 0..ne0 {
                let g = rng.unit() + rng.unit() + rng.unit();
                let mut v = g * scale;
                if rng.u32() % 512 == 0 {
                    v *= 40.0;
                }
                if (c / 32) % 17 == (round + r) % 17 && round % 2 == 1 {
                    v = 0.0;
                }
                x[r * ne0 + c] = v;
            }
        }
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xa = upload(&api, &xb);
        let nbytes = ne0 * ne1 / 32 * refs::Q8_1_BYTES;
        let mut outs = Vec::new();
        for f in [f_plain, f_native] {
            let ya = upload(&api, &vec![0u8; nbytes]);
            let q = |v: usize| (v as u64).to_le_bytes().to_vec();
            let params = vec![
                xa.to_le_bytes().to_vec(),
                ya.to_le_bytes().to_vec(),
                q(ne0),
                q(ne0),
                q(ne0 * ne1),
                q(ne0 * ne1),
                q(ne0),
                (ne1 as u32).to_le_bytes().to_vec(),
                bytes_u32(&refs::fastdiv_values(1)),
            ];
            assert_eq!(launch(&api, f, [ne0.div_ceil(256) as u32, ne1 as u32, 1], [256, 1, 1], &params), 0);
            assert_eq!(unsafe { (api.sync)() }, 0);
            outs.push(download(&api, ya, nbytes));
        }
        let (t, n) = (&outs[0], &outs[1]);
        for b in 0..nbytes / refs::Q8_1_BYTES {
            let (tb, nb) = (&t[b * 36..b * 36 + 36], &n[b * 36..b * 36 + 36]);
            total.0 += 1;
            let dd = tb[0..2] != nb[0..2];
            let sd = tb[2..4] != nb[2..4];
            let qd = (0..32).filter(|i| tb[4 + i] != nb[4 + i]).count();
            total.1 += usize::from(dd);
            total.2 += usize::from(sd);
            total.3 += qd;
            if (dd || sd || qd > 0) && shown < 12 {
                shown += 1;
                let xs = &x[b * 32..b * 32 + 32];
                let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
                let first = (0..32).find(|i| tb[4 + i] != nb[4 + i]);
                println!(
                    "diff block {b}: d translated {:04x}{:04x} native {:04x}{:04x}; sum t {:02x}{:02x} n {:02x}{:02x}; q diffs {qd}; amax {amax:e}; first q diff {:?}",
                    tb[1], tb[0], nb[1], nb[0], tb[3], tb[2], nb[3], nb[2],
                    first.map(|i| (xs[i], tb[4 + i] as i8, nb[4 + i] as i8, xs[i] / (amax / 127.0)))
                );
            }
        }
        let want = {
            let mut w = Vec::new();
            for r in 0..ne1 {
                w.extend(refs::quantize_q8_1(&x[r * ne0..(r + 1) * ne0]));
            }
            w
        };
        total.4 += t.iter().zip(&want).filter(|(a, b)| a != b).count();
    }
    println!("quantize fuzz: {} blocks; native-vs-translated: d differs in {}, sum in {}, quant bytes {}; translated-vs-host-ref differing bytes {}", total.0, total.1, total.2, total.3, total.4);
    // d and the block sum are exact. Quant bytes may differ by 1 LSB where x/d is within ~1e-6 of a
    // rounding tie (the native division is not correctly rounded there); measured 16 of 12.6M bytes.
    assert_eq!(total.1 + total.2, 0, "native quantize_q8_1 d/sum differ from the translated kernel");
    assert!(total.3 * 100_000 < total.0 * 32, "native quantize_q8_1 quant bytes differ more than 1e-5");
}

/// Decode-shape mmvq (ncols_dst 1) for the three test models: translated stock PTX, the stock-order
/// native kernel and the decode-fast variants (`CUDA_FOR_INTEL_MMVQ_FAST`), each checked against
/// the host reference and timed over enough weight copies that the matrix is not cache resident
/// (decode streams every weight once per token). One GPU slot:
///   set CFI_STOCK_GGML_DLL=<path>\ggml-cuda.dll
///   cargo test --release --test gpu hw_subst_mmvq_decode -- --ignored --nocapture --test-threads=1
/// Optional: CFI_SUBST_REPS (timed launches per variant, default 60), CFI_MMVQ_FAST_LIST
/// (variants `rows x warps x unroll`, default `4x1x2,4x1x4,4x1x8,8x1x4,2x2x4,1x4x2,1x4x4,4x2x4`).
#[test]
#[ignore = "requires CFI_STOCK_GGML_DLL, a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_subst_mmvq_decode_shapes() {
    let path = std::env::var("CFI_STOCK_GGML_DLL").expect("CFI_STOCK_GGML_DLL");
    let reps: usize = std::env::var("CFI_SUBST_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let variants: Vec<String> = std::env::var("CFI_MMVQ_FAST_LIST")
        .unwrap_or_else(|_| "4x1x2,4x1x4,4x1x8,8x1x4,2x2x4,1x4x2,1x4x4,4x2x4".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let blobs = extract_all(&path);
    let mmvq_blob = blobs.iter().find(|b| b.contains(".entry _Z13mul_mat_vec_qIL9ggml_type8ELi1E")).expect("mmvq blob");
    let _ = std::fs::remove_file("cuda-for-intel-export.log");
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxSynchronize(),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLibraryLoadData(*mut Handle, *const c_void, *mut c_void, *mut c_void, u32, *mut c_void, *mut c_void, u32),
        cuLibraryGetKernel(*mut Handle, Handle, *const c_char),
        cuKernelGetFunction(*mut Handle, Handle),
    );
    let launch_fn = unsafe {
        dll.get::<unsafe extern "system" fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut Handle) -> i32>(b"cuLaunchKernel\0")
    }
    .expect("cuLaunchKernel");
    let api = Api { launch: *launch_fn, sync: *cuCtxSynchronize, alloc: *cuMemAlloc_v2, htod: *cuMemcpyHtoD_v2, dtoh: *cuMemcpyDtoH_v2 };
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    // one library per setting: substitution is decided when a kernel is first resolved
    let resolve = |subst: &str, fast: Option<&str>, entry: &str| -> Handle {
        let src = std::ffi::CString::new(mmvq_blob.as_str()).unwrap();
        let mut library = null_mut();
        assert_eq!(unsafe { cuLibraryLoadData(&mut library, src.as_ptr().cast(), null_mut(), null_mut(), 0, null_mut(), null_mut(), 0) }, 0);
        unsafe {
            std::env::set_var("CUDA_FOR_INTEL_SUBST", subst);
            match fast {
                Some(f) => std::env::set_var("CUDA_FOR_INTEL_MMVQ_FAST", f),
                None => std::env::remove_var("CUDA_FOR_INTEL_MMVQ_FAST"),
            }
        }
        let c = std::ffi::CString::new(entry).unwrap();
        let mut kernel = null_mut();
        assert_eq!(unsafe { cuLibraryGetKernel(&mut kernel, library, c.as_ptr()) }, 0);
        let mut f = null_mut();
        assert_eq!(unsafe { cuKernelGetFunction(&mut f, kernel) }, 0);
        unsafe {
            std::env::remove_var("CUDA_FOR_INTEL_SUBST");
            std::env::remove_var("CUDA_FOR_INTEL_MMVQ_FAST");
        }
        f
    };
    // (label, type, rows, cols): Qwen2.5-0.5B Q8_0, Llama-3.2-1B Q4_K_M, Qwen2.5-1.5B Q4_K_M
    let shapes: [(&str, u32, usize, usize); 14] = [
        ("qwen05 q/o", refs::Q8_0, 896, 896),
        ("qwen05 k/v", refs::Q8_0, 128, 896),
        ("qwen05 gate/up", refs::Q8_0, 4864, 896),
        ("qwen05 down", refs::Q8_0, 896, 4864),
        ("qwen05 lm_head", refs::Q8_0, 151936, 896),
        ("llama1b q/o", refs::Q4_K, 2048, 2048),
        ("llama1b k", refs::Q4_K, 512, 2048),
        ("llama1b gate/up", refs::Q4_K, 8192, 2048),
        ("llama1b down", refs::Q6_K, 2048, 8192),
        ("llama1b v", refs::Q6_K, 512, 2048),
        ("llama1b lm_head", refs::Q6_K, 128256, 2048),
        ("qwen15 gate/up", refs::Q4_K, 8960, 1536),
        ("qwen15 down", refs::Q6_K, 1536, 8960),
        ("qwen15 lm_head", refs::Q6_K, 151936, 1536),
    ];
    let mut failures = Vec::new();
    let mut header = format!("{:16} {:5} {:>7} {:>6} {:>4} | {:>15} {:>15}", "shape", "type", "rows", "cols", "cps", "translated", "native");
    for v in &variants {
        header += &format!(" {:>15}", format!("f{v}"));
    }
    println!("== mul_mat_vec_q ncols_dst 1, us per launch / GB/s (weights + activations), {reps} launches");
    println!("{header}");
    for (label, ty, nrows, ncols_x) in shapes {
        let entry = format!("_Z13mul_mat_vec_qIL9ggml_type{ty}ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj");
        let c = make_case(ty, 1, nrows, ncols_x, 1, 1, 1, false, 0xDEC0 + ty as u64 + nrows as u64);
        let mut want = vec![0f32; nrows];
        refs::mmvq(&c.p, &c.x, &c.y, None, &mut want);
        // enough copies of the weights to exceed the last-level cache
        let copies = (256usize << 20).div_ceil(c.x.len()).clamp(1, 64);
        let xs: Vec<u64> = (0..copies).map(|_| upload(&api, &c.x)).collect();
        let ya = upload(&api, &c.y);
        let da = upload(&api, &vec![0u8; nrows * 4]);
        let params: Vec<Vec<Vec<u8>>> = xs.iter().map(|&xa| mmvq_params(&c.p, xa, ya, 0, da)).collect();
        let bytes = (c.x.len() + c.y.len()) as f64;
        let mut cells = Vec::new();
        let mut kinds: Vec<(String, &str, Option<String>)> = vec![("translated".into(), "none", None), ("native".into(), "mmvq", Some("0".into()))];
        for v in &variants {
            kinds.push((format!("fast {v}"), "mmvq", Some(v.clone())));
        }
        let mut native_out = Vec::new();
        for (kind, subst, fast) in &kinds {
            let f = resolve(subst, fast.as_deref(), &entry);
            assert_eq!(launch(&api, f, c.p.grid(), c.p.block(), &params[0]), 0);
            assert_eq!(unsafe { (api.sync)() }, 0);
            let got = floats(&download(&api, da, nrows * 4));
            let nmse = refs::nmse(&got, &want);
            if kind == "native" {
                native_out = got.clone();
            }
            if !(nmse < 1e-9) {
                failures.push(format!("{label} {kind}: nmse {nmse:.2e}"));
            }
            for i in 0..3 {
                assert_eq!(launch(&api, f, c.p.grid(), c.p.block(), &params[i % copies]), 0);
            }
            assert_eq!(unsafe { (api.sync)() }, 0);
            let start = Instant::now();
            for i in 0..reps {
                assert_eq!(launch(&api, f, c.p.grid(), c.p.block(), &params[i % copies]), 0);
            }
            assert_eq!(unsafe { (api.sync)() }, 0);
            let s = start.elapsed().as_secs_f64() / reps as f64;
            cells.push(format!(" {:>15}", format!("{:.1}/{:.1}", s * 1e6, bytes / s / 1e9)));
        }
        let _ = native_out;
        println!("{label:16} {:5} {nrows:7} {ncols_x:6} {copies:4} |{}", type_label(ty), cells.concat());
    }
    let log = std::fs::read_to_string("cuda-for-intel-export.log").unwrap_or_default();
    let rejected: Vec<&str> = log.lines().filter(|l| l.contains("subst:") && (l.contains("rejected") || l.contains("failed"))).collect();
    for l in &rejected {
        println!("  {l}");
    }
    assert!(rejected.is_empty(), "native kernels rejected");
    assert!(failures.is_empty(), "{failures:?}");
}

/// FlashAttention test problem in ggml's contiguous layouts (mirrors `src/subst/probe.rs`).
struct FaCase {
    d: usize,
    nq: usize,
    nh: usize,
    nhk: usize,
    nkv: usize,
    nq_pad: usize,
    scale: f32,
    q: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
    mask: Vec<u16>,
}

/// Keys at `nvis..nkv` are padding (-inf); query `i` sees keys `j <= nvis - nq + i` (causal).
#[allow(clippy::too_many_arguments)]
fn fa_case(d: usize, nq: usize, nh: usize, nhk: usize, nkv: usize, nvis: usize, seed: u64) -> FaCase {
    let mut rng = Rng(seed);
    let q = (0..d * nq * nh).map(|_| rng.unit()).collect();
    let k = (0..d * nkv * nhk).map(|_| refs::f32_to_f16(rng.unit())).collect();
    let v = (0..d * nkv * nhk).map(|_| refs::f32_to_f16(rng.unit())).collect();
    let nq_pad = nq.div_ceil(16) * 16;
    let mut mask = vec![0u16; nkv * nq_pad];
    for i in 0..nq_pad {
        for j in 0..nkv {
            mask[i * nkv + j] = if j < nvis && j + nq <= nvis + i { refs::f32_to_f16(rng.unit() * 0.5) } else { 0xfc00 };
        }
    }
    FaCase { d, nq, nh, nhk, nkv, nq_pad, scale: 1.0 / (d as f32).sqrt(), q, k, v, mask }
}

impl FaCase {
    fn reference(&self) -> Vec<f32> {
        let (d, nq, nh, nkv) = (self.d, self.nq, self.nh, self.nkv);
        let gqa = nh / self.nhk;
        let mut out = vec![0f32; d * nh * nq];
        for h in 0..nh {
            let kh = h / gqa;
            for i in 0..nq {
                let qrow = &self.q[(h * nq + i) * d..][..d];
                let sc: Vec<f64> = (0..nkv)
                    .map(|j| {
                        let krow = &self.k[(kh * nkv + j) * d..][..d];
                        let dot: f64 = (0..d).map(|t| f64::from(qrow[t] * self.scale) * f64::from(refs::f16_to_f32(krow[t]))).sum();
                        dot + f64::from(refs::f16_to_f32(self.mask[i * nkv + j]))
                    })
                    .collect();
                let mx = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                let l: f64 = p.iter().sum();
                for t in 0..d {
                    let o: f64 = p.iter().enumerate().map(|(j, pj)| pj * f64::from(refs::f16_to_f32(self.v[(kh * nkv + j) * d + t]))).sum();
                    out[(i * nh + h) * d + t] = (o / l) as f32;
                }
            }
        }
        out
    }

    /// The 37 launch words of `flash_attn_ext_f16`.
    fn words(&self, q: u64, k: u64, v: u64, mask: u64, dst: u64) -> Vec<Vec<u8>> {
        let (d, nq, nh, nkv, nhk, nqp) = (self.d as u32, self.nq as u32, self.nh as u32, self.nkv as u32, self.nhk as u32, self.nq_pad as u32);
        let p = |x: u64| x.to_le_bytes().to_vec();
        let u = |x: u32| x.to_le_bytes().to_vec();
        let f = |x: f32| x.to_bits().to_le_bytes().to_vec();
        vec![
            p(q), p(k), p(v), p(mask), p(0), p(0), p(dst), p(0),
            f(self.scale), f(0.0), f(1.0), f(1.0), u(0), f(0.0),
            u(d), u(nq), u(nh), u(1), u(d * 4), u(d * 4 * nq), u(d * 4 * nq * nh),
            u(d), u(nkv), u(nhk), u(1), u(d * 2), u(d * 2 * nkv), p(u64::from(d * 2 * nkv * nhk)),
            u(d * 2), u(d * 2 * nkv), p(u64::from(d * 2 * nkv * nhk)),
            u(nkv), u(nqp), u(1), u(nkv * 2), u(nkv * 2 * nqp), p(u64::from(nkv * 2 * nqp)),
        ]
    }
}

/// Native FlashAttention (`CUDA_FOR_INTEL_SUBST=fattn`, `flash_attn_ext_f16`) on the device against
/// exact f64 attention at the test models' prefill and decode shapes, with kernel timing. One slot:
///   set CFI_STOCK_GGML_DLL=<path>\ggml-cuda.dll
///   cargo test --release --test gpu hw_subst_fattn -- --ignored --nocapture --test-threads=1
#[test]
#[ignore = "requires CFI_STOCK_GGML_DLL, a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_subst_fattn_native_vs_reference() {
    use std::io::Write as _;
    let path = std::env::var("CFI_STOCK_GGML_DLL").expect("CFI_STOCK_GGML_DLL");
    let blobs = extract_all(&path);
    // The driver keeps the export log open for the whole process: never delete it here (later lines
    // would go to the unlinked file); count the lines this test adds instead.
    let log_before = std::fs::read_to_string("cuda-for-intel-export.log").map(|l| l.len()).unwrap_or(0);
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxSynchronize(),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLibraryLoadData(*mut Handle, *const c_void, *mut c_void, *mut c_void, u32, *mut c_void, *mut c_void, u32),
        cuLibraryGetKernel(*mut Handle, Handle, *const c_char),
        cuKernelGetFunction(*mut Handle, Handle),
    );
    let launch_fn = unsafe {
        dll.get::<unsafe extern "system" fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut Handle) -> i32>(b"cuLaunchKernel\0")
    }
    .expect("cuLaunchKernel");
    let api = Api { launch: *launch_fn, sync: *cuCtxSynchronize, alloc: *cuMemAlloc_v2, htod: *cuMemcpyHtoD_v2, dtoh: *cuMemcpyDtoH_v2 };
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    // Some instantiations are architecture-guard stubs in this PTX (refused at load): try candidates.
    let resolve = |entry: &str| -> Option<Handle> {
        let blob = blobs.iter().find(|b| b.contains(&format!(".entry {entry}(")))?;
        let src = std::ffi::CString::new(blob.as_str()).unwrap();
        let mut library = null_mut();
        assert_eq!(unsafe { cuLibraryLoadData(&mut library, src.as_ptr().cast(), null_mut(), null_mut(), 0, null_mut(), null_mut(), 0) }, 0);
        unsafe { std::env::set_var("CUDA_FOR_INTEL_SUBST", "fattn") };
        let c = std::ffi::CString::new(entry).unwrap();
        let mut kernel = null_mut();
        let ok = unsafe { cuLibraryGetKernel(&mut kernel, library, c.as_ptr()) } == 0;
        let mut f = null_mut();
        let ok = ok && unsafe { cuKernelGetFunction(&mut f, kernel) } == 0;
        unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
        ok.then_some(f)
    };
    // every non-softcap, non-MLA instantiation for head size d, the preferred template first
    let entries_for = |d: u32, want: (u32, u32, u32, u32)| -> Vec<String> {
        let prefix = format!(".entry _Z18flash_attn_ext_f16ILi{d}ELi{d}E");
        let mut out: Vec<String> = Vec::new();
        for b in &blobs {
            let mut from = 0;
            while let Some(i) = b[from..].find(&prefix) {
                let start = from + i + 7;
                let rest = &b[start..];
                let name = &rest[..rest.find('(').unwrap_or(0)];
                if name.contains("ELb0ELb0E") && !out.iter().any(|n| n == name) {
                    out.push(name.to_string());
                }
                from = start;
            }
        }
        let tag = format!("ELi{}ELi{}ELi{}ELi{}E", want.0, want.1, want.2, want.3);
        out.sort_by_key(|n| !n.contains(&tag));
        out
    };
    // (label, d, nq, nh, nh_kv, nkv, stock template (ncols1, ncols2, nwarps, ntiles))
    let shapes = [
        ("qwen05 prefill", 64usize, 631usize, 14usize, 2usize, 768usize, (32u32, 1u32, 4u32, 2u32)),
        ("llama1b prefill", 64, 631, 32, 8, 768, (32, 1, 4, 2)),
        ("qwen15 prefill", 128, 631, 12, 2, 768, (32, 1, 4, 2)),
    ];
    let mut failures = Vec::new();
    println!("== flash_attn_ext_f16 native: nmse vs exact attention, ms per launch");
    for (label, d, nq, nh, nhk, nkv, (c1, c2, nw, nt)) in shapes {
        let Some((entry, f)) = entries_for(d as u32, (c1, c2, nw, nt)).into_iter().find_map(|e| resolve(&e).map(|f| (e, f))) else {
            println!("{label}: no loadable stock instantiation, skipped");
            continue;
        };
        let _ = entry;
        let case = fa_case(d, nq, nh, nhk, nkv, 631, 0xfa + nq as u64);
        let b16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let qa = upload(&api, &case.q.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let ka = upload(&api, &b16(&case.k));
        let va = upload(&api, &b16(&case.v));
        let ma = upload(&api, &b16(&case.mask));
        let n = d * nh * nq;
        let da = upload(&api, &vec![0u8; n * 4]);
        let words = case.words(qa, ka, va, ma, da);
        // stock geometry is irrelevant to the native kernel (it maps its own grid)
        assert_eq!(launch(&api, f, [8, 1, 1], [32, nw, 1], &words), 0);
        assert_eq!(unsafe { (api.sync)() }, 0);
        let got = floats(&download(&api, da, n * 4));
        let want = case.reference();
        let nmse = refs::nmse(&got, &want);
        let reps = 20;
        let start = Instant::now();
        for _ in 0..reps {
            assert_eq!(launch(&api, f, [8, 1, 1], [32, nw, 1], &words), 0);
        }
        assert_eq!(unsafe { (api.sync)() }, 0);
        let ms = start.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let flops = 4.0 * (d * nq * nh * nkv) as f64;
        println!("{label:16} d {d:3} nq {nq:4} heads {nh}/{nhk} kv {nkv}: nmse {nmse:.2e}, {ms:.3} ms ({:.0} GFLOP/s dense)", flops / ms / 1e6);
        let _ = std::io::stdout().flush();
        if !(nmse < 1e-8) {
            failures.push(format!("{label}: nmse {nmse:.2e}"));
        }
    }
    let log = std::fs::read_to_string("cuda-for-intel-export.log").unwrap_or_default();
    let new = log.get(log_before..).unwrap_or("");
    for l in new.lines().filter(|l| l.contains("fattn")) {
        println!("  {l}");
    }
    let rejected = new.lines().filter(|l| l.contains("subst: native fattn") && (l.contains("rejected") || l.contains("failed"))).count();
    assert_eq!(rejected, 0, "native fattn was rejected");
    assert!(failures.is_empty(), "{failures:?}");
}

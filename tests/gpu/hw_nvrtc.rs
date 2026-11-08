//! One dense hardware run of the NVRTC shim: CUDA C++ -> PTX (clang NVPTX) -> Driver API on the
//! Intel GPU, every kernel checked against a Rust reference. Failures are collected and reported
//! together so a single GPU session shows the full picture.
//!
//! Run: NVRTC_SHIM_CLANG=<path to clang.exe> cargo test --test gpu hw_nvrtc -- --ignored --nocapture
use super::hw_common::*;
use nvrtc64_120_0::compiler::{self, Request};

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn rnd(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| (rng.range(-2000, 2000) as f32) / 1000.0).collect()
}
fn nvrtc(src: &str, names: &[&str]) -> compiler::Output {
    let req = Request { source: src.into(), name: "hw.cu".into(), name_expressions: names.iter().map(|s| (*s).to_owned()).collect(), ..Default::default() };
    compiler::compile(&req).unwrap_or_else(|e| panic!("nvrtc: {}", e.log()))
}

type Check = Box<dyn Fn() -> Result<(), String>>;

#[test]
#[ignore = "requires clang with NVPTX (NVRTC_SHIM_CLANG), a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_nvrtc_all_kernels() {
    let mut results: Vec<(&str, Result<(), String>)> = Vec::new();
    let mut run = |name: &'static str, f: Check| results.push((name, f()));

    run("vector_add", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/vadd.cu"), &[]);
        let mut r = Rng(1);
        let n = 5000usize;
        let (a, b) = (rnd(&mut r, n), rnd(&mut r, n));
        let mut bufs = vec![f32s(&a), f32s(&b), vec![0u8; n * 4]];
        launch_args(&out.ptx, "vadd", [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], 0, &mut bufs,
            &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), u32_arg(n as u32)])?;
        let c = as_f32(&bufs[2]);
        (0..n).find(|&i| c[i] != a[i] + b[i]).map_or(Ok(()), |i| Err(format!("i={i}: {} vs {}", c[i], a[i] + b[i])))
    }));

    run("shared_reduce", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/reduce.cu"), &[]);
        let mut r = Rng(2);
        let (n, bs) = (3000usize, 128usize);
        let blocks = n.div_ceil(bs);
        let x = rnd(&mut r, n);
        let mut bufs = vec![f32s(&x), vec![0u8; blocks * 4]];
        launch_args(&out.ptx, "block_sum", [blocks as u32, 1, 1], [bs as u32, 1, 1], 0, &mut bufs,
            &[Arg::Buf(0), Arg::Buf(1), u32_arg(n as u32)])?;
        let got = as_f32(&bufs[1]);
        for b in 0..blocks {
            let want: f32 = x[b * bs..((b + 1) * bs).min(n)].iter().sum();
            if (got[b] - want).abs() > 1e-2 {
                return Err(format!("block {b}: {} vs {want}", got[b]));
            }
        }
        Ok(())
    }));

    run("warp_shuffle_vote", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/warp.cu"), &[]);
        let mut r = Rng(3);
        let x = rnd(&mut r, 256);
        let mut bufs = vec![f32s(&x), vec![0u8; 32], vec![0u8; 32], vec![0u8; 32]];
        launch_args(&out.ptx, "warp_ops", [1, 1, 1], [256, 1, 1], 0, &mut bufs, &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3)])?;
        let (s, m, v) = (as_f32(&bufs[1]), as_f32(&bufs[2]), u32s_of(&bufs[3]));
        for w in 0..8 {
            let lane = &x[w * 32..(w + 1) * 32];
            let ws: f32 = lane.iter().sum();
            let wm = lane.iter().cloned().fold(f32::MIN, f32::max);
            let wv = lane.iter().filter(|&&q| q > 0.0).count() as u32;
            if (s[w] - ws).abs() > 1e-3 || m[w] != wm || v[w] != wv {
                return Err(format!("warp {w}: sum {} vs {ws}, max {} vs {wm}, votes {} vs {wv}", s[w], m[w], v[w]));
            }
        }
        Ok(())
    }));

    run("atomics", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/atomics.cu"), &[]);
        let n = 4096usize;
        let keys: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % 13).collect();
        let vals: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.5).collect();
        let mut bufs = vec![bytes_of_u32(&keys), f32s(&vals), vec![0u8; 32], vec![0u8; 4]];
        launch_args(&out.ptx, "atomics", [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], 0, &mut bufs,
            &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3), u32_arg(n as u32)])?;
        let mut want = [0u32; 8];
        keys.iter().for_each(|k| want[(k & 7) as usize] += 1);
        if u32s_of(&bufs[2]) != want {
            return Err(format!("hist {:?} vs {:?}", u32s_of(&bufs[2]), want));
        }
        let total: f32 = vals.iter().sum();
        let got = as_f32(&bufs[3])[0];
        if (got - total).abs() > 1e-2 { Err(format!("total {got} vs {total}")) } else { Ok(()) }
    }));

    run("templates_name_expressions", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/templ.cu"), &["axpy<float>", "ns::scale<3>"]);
        let n = 300usize;
        let mut r = Rng(5);
        let (x, y) = (rnd(&mut r, n), rnd(&mut r, n));
        let mut bufs = vec![f32s(&x), f32s(&y)];
        launch_args(&out.ptx, &out.lowered["axpy<float>"], [3, 1, 1], [128, 1, 1], 0, &mut bufs,
            &[Arg::Bytes(2.5f32.to_le_bytes().to_vec()), Arg::Buf(0), Arg::Buf(1), u32_arg(n as u32)])?;
        let got = as_f32(&bufs[1]);
        // NVRTC contracts a*x+y into an fma by default (--fmad=true): accept the fused or unfused result.
        if let Some(i) = (0..n).find(|&i| got[i] != 2.5f32.mul_add(x[i], y[i]) && got[i] != 2.5 * x[i] + y[i]) {
            return Err(format!("axpy i={i}: {} vs {}", got[i], 2.5 * x[i] + y[i]));
        }
        let mut bufs = vec![f32s(&x)];
        launch_args(&out.ptx, &out.lowered["ns::scale<3>"], [3, 1, 1], [128, 1, 1], 0, &mut bufs, &[Arg::Buf(0), u32_arg(n as u32)])?;
        let got = as_f32(&bufs[0]);
        (0..n).find(|&i| got[i] != 3.0 * x[i]).map_or(Ok(()), |i| Err(format!("scale i={i}")))
    }));

    run("half_math", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/half.cu"), &[]);
        let n = 512usize;
        let mut r = Rng(6);
        let mk = |r: &mut Rng| -> Vec<u16> { rnd(r, n).iter().map(|&v| f32_to_f16(v)).collect() };
        let (x, w, b) = (mk(&mut r), mk(&mut r), mk(&mut r));
        let bytes = |v: &[u16]| v.iter().flat_map(|q| q.to_le_bytes()).collect::<Vec<u8>>();
        let mut bufs = vec![bytes(&x), bytes(&w), bytes(&b), vec![0u8; n * 2], vec![0u8; n * 4]];
        launch_args(&out.ptx, "half_ops", [(n as u32).div_ceil(128), 1, 1], [128, 1, 1], 0, &mut bufs,
            &[Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3), Arg::Buf(4), u32_arg(n as u32)])?;
        let e = as_f32(&bufs[4]);
        for i in 0..n {
            let got = u16::from_le_bytes([bufs[3][2 * i], bufs[3][2 * i + 1]]);
            let want = f32_to_f16(f16_to_f32(x[i]).mul_add(f16_to_f32(w[i]), f16_to_f32(b[i])));
            if (i32::from(got) - i32::from(want)).abs() > 1 {
                return Err(format!("hfma i={i}: {got:#x} vs {want:#x}"));
            }
            let we = f16_to_f32(x[i]).exp();
            if (e[i] - we).abs() > we * 1e-4 {
                return Err(format!("exp i={i}: {} vs {we}", e[i]));
            }
        }
        Ok(())
    }));

    run("cooperative_groups_tile_reduce", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/cg.cu"), &[]);
        let mut r = Rng(7);
        let x = rnd(&mut r, 128);
        let mut bufs = vec![f32s(&x), vec![0u8; 16]];
        launch_args(&out.ptx, "cg_sum", [1, 1, 1], [128, 1, 1], 0, &mut bufs, &[Arg::Buf(0), Arg::Buf(1)])?;
        let got = as_f32(&bufs[1]);
        for w in 0..4 {
            let want: f32 = x[w * 32..(w + 1) * 32].iter().sum();
            if (got[w] - want).abs() > 1e-3 {
                return Err(format!("tile {w}: {} vs {want}", got[w]));
            }
        }
        Ok(())
    }));

    run("math_functions", Box::new(|| {
        let out = nvrtc(include_str!("../../crates/nvrtc_shim/tests/kernels/mathfn.cu"), &[]);
        let n = 200usize;
        let x: Vec<f32> = (0..n).map(|i| 0.05 + i as f32 * 0.05).collect();
        let mut bufs = vec![f32s(&x), vec![0u8; n * 20]];
        launch_args(&out.ptx, "mathfn", [2, 1, 1], [128, 1, 1], 0, &mut bufs, &[Arg::Buf(0), Arg::Buf(1), u32_arg(n as u32)])?;
        let o = as_f32(&bufs[1]);
        for i in 0..n {
            let v = x[i];
            for (j, w) in [v.sqrt(), (-v).exp(), v.ln(), v.sin(), v.tanh()].iter().enumerate() {
                if (o[5 * i + j] - w).abs() > 2e-5 * w.abs().max(1.0) {
                    return Err(format!("fn {j} x={v}: {} vs {w}", o[5 * i + j]));
                }
            }
        }
        Ok(())
    }));

    let mut failed = 0;
    for (name, r) in &results {
        match r {
            Ok(()) => println!("hw_nvrtc PASS {name}"),
            Err(e) => {
                failed += 1;
                println!("hw_nvrtc FAIL {name}: {e}");
            }
        }
    }
    assert_eq!(failed, 0, "{failed} of {} NVRTC kernels failed on hardware", results.len());
}

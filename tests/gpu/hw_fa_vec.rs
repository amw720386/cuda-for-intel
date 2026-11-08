//! Regression: ggml `flash_attn_ext_vec<64,1,f16,f16>` (decode, Qwen2.5-0.5B shapes) must write the
//! softmax denominator into `dst_meta` when several parallel blocks split the KV range and most
//! blocks see only masked positions. Mirrors the first decode step of a real Ollama run:
//! 14 query heads, 2 KV heads, KV padded to 256 with a few visible positions, 4 parallel blocks.
//!
//!   cargo test --test gpu hw_fa_vec -- --ignored --nocapture --test-threads=1
use super::hw_common::{Arg, Rng, f16_to_f32, f32_to_f16, i64_arg, launch_args, u32_arg};

const KERNEL: &str = "_Z18flash_attn_ext_vecILi64ELi1EL9ggml_type1ELS0_1ELb0EEvPKcS2_S2_S2_S2_PKiPfP6float2ffffjfiiiiiiiiiiiiixiixiiiiix";
const PTX: &str = include_str!("fa_vec_f16_d64.ptx");
const D: usize = 64;
const HEADS: usize = 14;
const KV_HEADS: usize = 2;
const KV_PAD: usize = 256;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn half_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect()
}

fn unit(rng: &mut Rng) -> f32 {
    (rng.u32() as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0
}

struct Out {
    tmp: Vec<f32>,
    meta: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn run(visible: usize, parallel_blocks: u32, q_scale: f32, ptx_src: &str) -> Out {
    let mut rng = Rng(0x5eed_u64 + visible as u64);
    let q: Vec<f32> = (0..HEADS * D).map(|_| unit(&mut rng) * q_scale).collect();
    // Ollama layout: position stride KV_HEADS*D, head stride D (f16 elements).
    let k: Vec<f32> = (0..KV_PAD * KV_HEADS * D).map(|_| unit(&mut rng) * 3.0).collect();
    let v: Vec<f32> = (0..KV_PAD * KV_HEADS * D).map(|_| unit(&mut rng)).collect();
    let mut mask = vec![f32::NEG_INFINITY; 64 * KV_PAD];
    mask[..visible].fill(0.0);
    let k16: Vec<f32> = k.iter().map(|x| f16_to_f32(f32_to_f16(*x))).collect();
    let v16: Vec<f32> = v.iter().map(|x| f16_to_f32(f32_to_f16(*x))).collect();
    let pb = parallel_blocks as usize;
    let mut bufs = vec![
        f32_bytes(&q),
        half_bytes(&k),
        half_bytes(&v),
        half_bytes(&mask),
        vec![0u8; 4 * HEADS * pb * D],
        vec![0u8; 8 * HEADS * pb],
    ];
    let (kvp, d2) = (KV_PAD as u32, (D * 2) as u32);
    let f = |x: f32| Arg::Bytes(x.to_le_bytes().to_vec());
    let args = [
        Arg::Buf(0), Arg::Buf(1), Arg::Buf(2), Arg::Buf(3), Arg::Bytes(vec![0; 8]), Arg::Bytes(vec![0; 8]), Arg::Buf(4), Arg::Buf(5),
        f(0.125), f(0.0), f(1.0), f(1.0), u32_arg(8), f(0.0),
        u32_arg(D as u32), u32_arg(1), u32_arg(HEADS as u32), u32_arg(1),
        u32_arg((D * HEADS * 4) as u32), u32_arg((D * 4) as u32), u32_arg((D * HEADS * 4) as u32),
        u32_arg(D as u32), u32_arg(kvp), u32_arg(KV_HEADS as u32), u32_arg(1),
        u32_arg(d2 * KV_HEADS as u32), u32_arg(d2), i64_arg((d2 as usize * KV_PAD * KV_HEADS) as i64),
        u32_arg(d2 * KV_HEADS as u32), u32_arg(d2), i64_arg((d2 as usize * KV_PAD * KV_HEADS) as i64),
        u32_arg(kvp), u32_arg(1), u32_arg(1),
        u32_arg(kvp * 2), u32_arg(kvp * 2 * 64), i64_arg((kvp * 2 * 64) as i64),
    ];
    let ptx = format!(".version 8.0\n.target sm_80\n.address_size 64\n{ptx_src}");
    launch_args(&ptx, KERNEL, [1, parallel_blocks, HEADS as u32], [32, 4, 1], 0, &mut bufs, &args).expect("launch");
    let to_f32 = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    Out { tmp: to_f32(&bufs[4]), meta: to_f32(&bufs[5]), q, k: k16, v: v16 }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn flash_attn_vec_writes_softmax_denominator() {
    let mut failures = Vec::new();
    for &(visible, pb) in &[(4usize, 4u32), (12, 4), (40, 4), (4, 1), (200, 4)] {
        let Out { tmp, meta, q, k, v } = run(visible, pb, 4.0, PTX);
        let pb = pb as usize;
        for h in 0..HEADS {
            let kv = h / (HEADS / KV_HEADS);
            let s: Vec<f64> = (0..visible)
                .map(|p| (0..D).map(|d| f64::from(q[h * D + d]) * f64::from(k[(p * KV_HEADS + kv) * D + d])).sum::<f64>() * 0.125)
                .collect();
            let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = s.iter().map(|x| (x - max).exp()).sum();
            if pb == 1 {
                // A single block normalises itself: tmp is the attention output.
                let want: f64 = s.iter().enumerate().map(|(p, x)| (x - max).exp() * f64::from(v[(p * KV_HEADS + kv) * D])).sum::<f64>() / total;
                let got = f64::from(tmp[h * D]);
                if (got - want).abs() > 5e-3 * (1.0 + want.abs()) {
                    failures.push(format!("visible={visible} pb=1 head={h}: out0 {got} want {want}"));
                }
                continue;
            }
            // Denominator over the blocks' (max, sum) pairs, rescaled to the global max.
            let gmax = (0..pb).map(|b| meta[(h * pb + b) * 2]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = (0..pb)
                .map(|b| f64::from(meta[(h * pb + b) * 2 + 1]) * f64::from((meta[(h * pb + b) * 2] - gmax).exp()))
                .sum();
            if !(sum.is_finite() && (sum - total).abs() <= 1e-2 * total) {
                failures.push(format!("visible={visible} pb={pb} head={h}: denominator {sum} want {total} (max {gmax} vs {max})"));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures, first: {:?}", failures.len(), &failures[..failures.len().min(6)]);
}

/// Bring-up probe: patched kernel variants, prints head-0 `dst_meta` per variant.
#[test]
#[ignore = "bring-up probe"]
fn flash_attn_vec_probe_variants() {
    let m = "mov.f32 %f544, 0f00000000;";
    let variants: Vec<(&str, Vec<(&str, &str)>)> = vec![
        ("base", vec![]),
        ("init f538=0 both sites", vec![(m, "mov.f32 %f544, 0f00000000;\nmov.f32 %f538, 0f00000000;")]),
        ("init f538=1.0 both sites", vec![(m, "mov.f32 %f544, 0f00000000;\nmov.f32 %f538, 0f3F800000;")]),
        ("init f538=0 first site only", vec![("@@1", "")]),
        ("init f538=0 second site only", vec![("@@2", "")]),
        ("init f538 at kernel start", vec![("mov.u64 %SPL, __local_depot0;", "mov.u64 %SPL, __local_depot0;\nmov.f32 %f538, 0f00000000;")]),
    ];
    for (name, patches) in variants {
        let mut ptx = PTX.to_owned();
        for (old, new) in patches {
            if old == "@@1" || old == "@@2" {
                let add = "mov.f32 %f538, 0f00000000;";
                let first = ptx.find(m).unwrap();
                let second = first + m.len() + ptx[first + m.len()..].find(m).unwrap();
                let at = if old == "@@1" { first } else { second } + m.len();
                ptx.insert_str(at, &format!("\n{add}"));
                continue;
            }
            assert!(ptx.contains(old), "{old}");
            ptx = ptx.replace(old, new);
        }
        let out = run(12, 4, 4.0, &ptx);
        let m: Vec<f32> = (0..4).flat_map(|b| [out.meta[b * 2], out.meta[b * 2 + 1]]).collect();
        eprintln!("PROBE {name}: head0 meta {m:?}");
    }
}

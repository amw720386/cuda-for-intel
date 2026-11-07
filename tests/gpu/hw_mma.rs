//! mma.sync emulation and ldmatrix fragment layouts vs the PTX ISA layouts.
use super::hw_common::*;

#[derive(Clone, Copy, Debug)]
enum Shape {
    I8M8N8K16,
    I8M16N8K16,
    I8M16N8K32,
    F16M16N8K8,
    F16M16N8K16,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::I8M8N8K16 => "m8n8k16",
            Shape::I8M16N8K16 | Shape::F16M16N8K16 => "m16n8k16",
            Shape::I8M16N8K32 => "m16n8k32",
            Shape::F16M16N8K8 => "m16n8k8",
        }
    }
    fn is_int(self) -> bool {
        matches!(self, Shape::I8M8N8K16 | Shape::I8M16N8K16 | Shape::I8M16N8K32)
    }
    fn m(self) -> usize {
        if matches!(self, Shape::I8M8N8K16) { 8 } else { 16 }
    }
    fn k(self) -> usize {
        match self {
            Shape::I8M8N8K16 | Shape::I8M16N8K16 | Shape::F16M16N8K16 => 16,
            Shape::I8M16N8K32 => 32,
            Shape::F16M16N8K8 => 8,
        }
    }
    /// 32-bit registers per lane for A, B, C/D.
    fn regs(self) -> (usize, usize, usize) {
        match self {
            Shape::I8M8N8K16 => (1, 1, 2),
            Shape::I8M16N8K16 => (2, 1, 4),
            Shape::I8M16N8K32 => (4, 2, 4),
            Shape::F16M16N8K8 => (2, 1, 4),
            Shape::F16M16N8K16 => (4, 2, 4),
        }
    }
    /// (row, col) of A element `i` (element index across the lane's packed registers).
    fn a_pos(self, lane: usize, i: usize) -> (usize, usize) {
        let (g, t) = (lane / 4, lane % 4);
        match self {
            Shape::I8M8N8K16 => (g, t * 4 + i),
            Shape::I8M16N8K16 => (g + 8 * (i / 4), t * 4 + i % 4),
            Shape::I8M16N8K32 => (g + 8 * ((i / 4) % 2), t * 4 + i % 4 + 16 * (i / 8)),
            Shape::F16M16N8K8 => (g + 8 * (i / 2), t * 2 + i % 2),
            Shape::F16M16N8K16 => (g + 8 * ((i / 2) % 2), t * 2 + i % 2 + 8 * (i / 4)),
        }
    }
    fn b_pos(self, lane: usize, i: usize) -> (usize, usize) {
        let (g, t) = (lane / 4, lane % 4);
        match self {
            Shape::I8M8N8K16 | Shape::I8M16N8K16 => (t * 4 + i, g),
            Shape::I8M16N8K32 => (t * 4 + i % 4 + 16 * (i / 4), g),
            Shape::F16M16N8K8 => (t * 2 + i, g),
            Shape::F16M16N8K16 => (t * 2 + i % 2 + 8 * (i / 2), g),
        }
    }
    fn c_pos(self, lane: usize, i: usize) -> (usize, usize) {
        let (g, t) = (lane / 4, lane % 4);
        (g + 8 * (i / 2), t * 2 + i % 2)
    }
}

fn mma_ptx(shape: Shape, atype: &str, btype: &str) -> String {
    let (na, nb, nc) = shape.regs();
    let ty = if shape.is_int() {
        format!("s32.{atype}.{btype}.s32")
    } else {
        "f32.f16.f16.f32".to_string()
    };
    let mut s = prologue("mma_k", 4, 0, "");
    s += "\tmov.u32 %r0, %tid.x;\n";
    let list = |base: usize, n: usize| {
        format!(
            "{{{}}}",
            (0..n).map(|i| format!("%r{}", base + i)).collect::<Vec<_>>().join(",")
        )
    };
    for (buf, base, n) in [(1usize, 10usize, na), (2, 20, nb), (3, 30, nc)] {
        s += &format!(
            "\tmul.wide.u32 %rd20, %r0, {};\n\tadd.s64 %rd21, %rd{buf}, %rd20;\n",
            n * 4
        );
        for i in 0..n {
            s += &format!("\tld.global.u32 %r{}, [%rd21+{}];\n", base + i, i * 4);
        }
    }
    s += &format!(
        "\tmma.sync.aligned.{}.row.col.{ty} {}, {}, {}, {};\n",
        shape.name(),
        list(50, nc),
        list(10, na),
        list(20, nb),
        list(30, nc)
    );
    // D goes to buffer 4 (appended): reuse param order a,b,c,d => 4 buffers (d is the 4th)
    s += &format!("\tmul.wide.u32 %rd20, %r0, {};\n\tadd.s64 %rd21, %rd4, %rd20;\n", nc * 4);
    for i in 0..nc {
        s += &format!("\tst.global.u32 [%rd21+{}], %r{};\n", i * 4, 50 + i);
    }
    s += "\tret;\n}\n";
    s
}

fn pack_elems(bytes_per: usize, elems: &[u32]) -> Vec<u32> {
    let per = 4 / bytes_per;
    elems
        .chunks(per)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |w, (i, &e)| w | ((e & ((1u64 << (8 * bytes_per)) - 1) as u32) << (8 * bytes_per * i)))
        })
        .collect()
}

fn run_int(shape: Shape, a_signed: bool, b_signed: bool, rng: &mut Rng, iters: usize) {
    let (m, k) = (shape.m(), shape.k());
    let (na, nb, nc) = shape.regs();
    let ptx = mma_ptx(shape, if a_signed { "s8" } else { "u8" }, if b_signed { "s8" } else { "u8" });
    for iter in 0..iters {
        let extreme = iter % 4 == 0;
        let mut elem = |signed: bool| -> i32 {
            if extreme {
                if signed { [-128, 127, -1, 0][rng.range(0, 3) as usize] } else { [0, 255, 128, 1][rng.range(0, 3) as usize] }
            } else if signed {
                rng.range(-128, 127)
            } else {
                rng.range(0, 255)
            }
        };
        let a: Vec<Vec<i32>> = (0..m).map(|_| (0..k).map(|_| elem(a_signed)).collect()).collect();
        let b: Vec<Vec<i32>> = (0..k).map(|_| (0..8).map(|_| elem(b_signed)).collect()).collect();
        let c: Vec<Vec<i32>> = (0..m).map(|_| (0..8).map(|_| rng.u32() as i32 >> if extreme { 0 } else { 12 }).collect()).collect();
        let (mut ab, mut bb, mut cb) = (vec![], vec![], vec![]);
        for lane in 0..32 {
            let ea: Vec<u32> = (0..na * 4).map(|i| { let (r, c) = shape.a_pos(lane, i); a[r][c] as u32 }).collect();
            let eb: Vec<u32> = (0..nb * 4).map(|i| { let (r, c) = shape.b_pos(lane, i); b[r][c] as u32 }).collect();
            ab.extend(pack_elems(1, &ea));
            bb.extend(pack_elems(1, &eb));
            cb.extend((0..nc).map(|i| { let (r, cc) = shape.c_pos(lane, i); c[r][cc] as u32 }));
        }
        let mut bufs = vec![bytes_of_u32(&ab), bytes_of_u32(&bb), bytes_of_u32(&cb), vec![0u8; 32 * nc * 4]];
        launch(&ptx, "mma_k", 1, 32, &mut bufs, &[]).unwrap();
        let d = u32s_of(&bufs[3]);
        for lane in 0..32 {
            for i in 0..nc {
                let (r, cc) = shape.c_pos(lane, i);
                let want = (0..k).fold(c[r][cc], |acc, kk| acc.wrapping_add(a[r][kk].wrapping_mul(b[kk][cc]))) as u32;
                assert_eq!(d[lane * nc + i], want, "{shape:?} s={a_signed}{b_signed} iter {iter} lane {lane} reg {i} (row {r} col {cc})");
            }
        }
    }
}

fn run_f16(shape: Shape, rng: &mut Rng, iters: usize) {
    let (m, k) = (shape.m(), shape.k());
    let (na, nb, nc) = shape.regs();
    let ptx = mma_ptx(shape, "", "");
    for iter in 0..iters {
        // multiples of 1/8 in [-4, 4): all products and sums are exact in f32.
        let mut elem = || rng.range(-32, 31) as f32 / 8.0;
        let a: Vec<Vec<f32>> = (0..m).map(|_| (0..k).map(|_| elem()).collect()).collect();
        let b: Vec<Vec<f32>> = (0..k).map(|_| (0..8).map(|_| elem()).collect()).collect();
        let c: Vec<Vec<f32>> = (0..m).map(|_| (0..8).map(|_| rng.range(-1000, 1000) as f32).collect()).collect();
        let (mut ab, mut bb, mut cb) = (vec![], vec![], vec![]);
        for lane in 0..32 {
            let pack = |vals: Vec<f32>| -> Vec<u32> {
                vals.chunks(2).map(|h| f32_to_f16(h[0]) as u32 | ((f32_to_f16(h[1]) as u32) << 16)).collect()
            };
            ab.extend(pack((0..na * 2).map(|i| { let (r, cc) = shape.a_pos(lane, i); a[r][cc] }).collect()));
            bb.extend(pack((0..nb * 2).map(|i| { let (r, cc) = shape.b_pos(lane, i); b[r][cc] }).collect()));
            cb.extend((0..nc).map(|i| { let (r, cc) = shape.c_pos(lane, i); c[r][cc].to_bits() }));
        }
        let mut bufs = vec![bytes_of_u32(&ab), bytes_of_u32(&bb), bytes_of_u32(&cb), vec![0u8; 32 * nc * 4]];
        launch(&ptx, "mma_k", 1, 32, &mut bufs, &[]).unwrap();
        let d = u32s_of(&bufs[3]);
        for lane in 0..32 {
            for i in 0..nc {
                let (r, cc) = shape.c_pos(lane, i);
                let want = (0..k).fold(c[r][cc], |acc, kk| acc + a[r][kk] * b[kk][cc]);
                let got = f32::from_bits(d[lane * nc + i]);
                assert_eq!(got, want, "{shape:?} iter {iter} lane {lane} reg {i} (row {r} col {cc})");
            }
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_mma_int8_all_shapes_all_signedness() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for shape in [Shape::I8M8N8K16, Shape::I8M16N8K16, Shape::I8M16N8K32] {
        for (sa, sb) in [(true, true), (false, false), (true, false), (false, true)] {
            run_int(shape, sa, sb, &mut rng, 40);
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_mma_f16_f32_accumulate() {
    let mut rng = Rng(0xfeed_beef_0bad_f00d);
    for shape in [Shape::F16M16N8K8, Shape::F16M16N8K16] {
        run_f16(shape, &mut rng, 60);
    }
}

#[test]
fn f16_helpers_round_trip() {
    for k in -32..32 {
        let v = k as f32 / 8.0;
        assert_eq!(f16_to_f32(f32_to_f16(v)), v);
    }
    assert_eq!(f32_to_f16(1.0), 0x3c00);
    assert_eq!(f32_to_f16(-2.0), 0xc000);
}

// ---------------------------------------------------------------- ldmatrix

fn ldmatrix_ptx(count: u8, trans: bool, generic: bool) -> String {
    let t = if trans { ".trans" } else { "" };
    let sh = if generic { "" } else { ".shared" };
    let mut s = prologue("ldm_k", 3, 0, "\t.shared .align 16 .b8 smem[1024];\n");
    s += "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r9, smem;\n";
    // Stage 1024 bytes: 8 words per lane, lane-strided.
    for i in 0..8 {
        s += &format!(
            "\tmad.lo.s32 %r1, {i}, 32, %r0;\n\tmul.wide.u32 %rd20, %r1, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tld.global.u32 %r2, [%rd21];\n\tshl.b32 %r5, %r1, 2;\n\tadd.s32 %r5, %r5, %r9;\n\tst.shared.u32 [%r5], %r2;\n"
        );
    }
    s += "\tbar.sync 0;\n";
    // Per-lane row address (byte offset into smem) from buffer 2.
    s += "\tmul.wide.u32 %rd20, %r0, 4;\n\tadd.s64 %rd21, %rd2, %rd20;\n\tld.global.u32 %r3, [%rd21];\n";
    let operand;
    if generic {
        s += "\tmov.b64 %rd22, smem;\n\tcvta.shared.u64 %rd23, %rd22;\n\tcvt.u64.u32 %rd24, %r3;\n\tadd.s64 %rd25, %rd23, %rd24;\n";
        operand = "%rd25";
    } else {
        s += "\tadd.s32 %r4, %r9, %r3;\n";
        operand = "%r4";
    }
    let regs: Vec<String> = (0..count).map(|i| format!("%r{}", 10 + i)).collect();
    s += &format!(
        "\tldmatrix.sync.aligned.m8n8.x{count}{t}{sh}.b16 {{{}}}, [{operand}];\n",
        regs.join(",")
    );
    s += &format!("\tmul.wide.u32 %rd20, %r0, {};\n\tadd.s64 %rd21, %rd3, %rd20;\n", count as usize * 4);
    for i in 0..count as usize {
        s += &format!("\tst.global.u32 [%rd21+{}], %r{};\n", i * 4, 10 + i);
    }
    s += "\tret;\n}\n";
    s
}

fn run_ldmatrix(count: u8, trans: bool, shared_q: bool, rng: &mut Rng) {
    let ptx = ldmatrix_ptx(count, trans, shared_q);
    for iter in 0..6 {
        let smem: Vec<u32> = (0..256).map(|_| rng.u32()).collect();
        // Row addresses: distinct 16-byte aligned rows (rows may be anywhere in 1024 bytes).
        let mut offs = vec![0u32; 32];
        for lane in 0..8 * count as usize {
            offs[lane] = 16 * (rng.u32() % 64);
        }
        for lane in 8 * count as usize..32 {
            offs[lane] = 0xdead_0000; // must be ignored
        }
        let mut bufs = vec![bytes_of_u32(&smem), bytes_of_u32(&offs), vec![0u8; 32 * count as usize * 4]];
        launch(&ptx, "ldm_k", 1, 32, &mut bufs, &[]).unwrap();
        let out = u32s_of(&bufs[2]);
        let half = |byte: u32| -> u32 {
            let w = smem[(byte / 4) as usize];
            (w >> (16 * ((byte / 2) % 2))) & 0xffff
        };
        for lane in 0..32usize {
            for k in 0..count as usize {
                let row = |r: usize| offs[8 * k + r];
                let want = if !trans {
                    smem[((row(lane / 4) + 4 * (lane as u32 % 4)) / 4) as usize]
                } else {
                    let (t, g) = (lane % 4, lane / 4);
                    half(row(2 * t) + 2 * g as u32) | (half(row(2 * t + 1) + 2 * g as u32) << 16)
                };
                assert_eq!(
                    out[lane * count as usize + k], want,
                    "x{count} trans={trans} shared_q={shared_q} iter {iter} lane {lane} reg {k}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_ldmatrix_all_counts_and_trans() {
    let mut rng = Rng(0x0dd_ba11_cafe_1234);
    for count in [1u8, 2, 4] {
        for trans in [false, true] {
            run_ldmatrix(count, trans, false, &mut rng);
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_ldmatrix_generic_pointer_without_shared_qualifier() {
    let mut rng = Rng(0x5eed_5eed_5eed_5eed);
    for count in [1u8, 2, 4] {
        for trans in [false, true] {
            run_ldmatrix(count, trans, true, &mut rng);
        }
    }
}

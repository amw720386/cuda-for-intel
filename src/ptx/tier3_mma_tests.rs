//! Numeric tests for the tier-3 tensor-core emulation, run on the 32-lane SPIR-V
//! interpreter. Operands are small integers so every type is exact and the
//! reference is a plain integer matrix product.
use super::coverage_tests::{interp, kernel_with_shared, lower};

fn f2h(x: f32) -> u16 {
    interp::f32_to_f16_bits(x)
}

fn f2bf(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn h2f(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) & 1, i32::from((h >> 10) & 0x1f), f32::from(h & 0x3ff));
    let magnitude = match exp {
        0 => man * 2f32.powi(-24),
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ty {
    F16,
    Bf16,
    Tf32,
    S8,
    U8,
    F64,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::F16 => "f16",
            Ty::Bf16 => "bf16",
            Ty::Tf32 => "tf32",
            Ty::S8 => "s8",
            Ty::U8 => "u8",
            Ty::F64 => "f64",
        }
    }
    fn bytes(self) -> usize {
        match self {
            Ty::F16 | Ty::Bf16 => 2,
            Ty::Tf32 => 4,
            Ty::S8 | Ty::U8 => 1,
            Ty::F64 => 8,
        }
    }
    fn encode(self, v: i32) -> u64 {
        match self {
            Ty::F16 => u64::from(f2h(v as f32)),
            Ty::Bf16 => u64::from(f2bf(v as f32)),
            Ty::Tf32 => u64::from((v as f32).to_bits()),
            Ty::S8 | Ty::U8 => (v as u64) & 0xff,
            Ty::F64 => (v as f64).to_bits(),
        }
    }
    fn int(self) -> bool {
        matches!(self, Ty::S8 | Ty::U8)
    }
}

fn a_at(ty: Ty, r: usize, c: usize) -> i32 {
    let v = ((r * 7 + c * 3 + 1) % 9) as i32;
    if ty == Ty::U8 { v } else { v - 4 }
}

fn b_at(ty: Ty, r: usize, c: usize) -> i32 {
    let v = ((r * 5 + c * 11 + 2) % 7) as i32;
    if ty == Ty::U8 { v } else { v - 3 }
}

/// (a regs, b regs, c regs) of the PTX fragment for a wmma shape.
fn counts(ty: Ty, m: usize, n: usize, k: usize, half_acc: bool) -> (usize, usize, usize) {
    let a = m * k * ty.bytes() / 4 / 32 * if ty == Ty::F64 { 2 } else { 1 };
    let b = k * n * ty.bytes() / 4 / 32 * if ty == Ty::F64 { 2 } else { 1 };
    let c = if ty == Ty::F64 { 2 } else if half_acc { m * n / 64 } else { m * n / 32 };
    (a.max(1), b.max(1), c)
}

/// Runs wmma.load/mma/store for one shape and compares with the reference.
fn check_wmma(ty: Ty, m: usize, n: usize, k: usize, a_row: bool, b_row: bool, half_acc: bool) {
    let (na, nb, nc) = counts(ty, m, n, k, half_acc);
    let wide = ty == Ty::F64;
    let acc = if ty.int() { "s32" } else if wide { "f64" } else if half_acc { "f16" } else { "f32" };
    let regs = format!(
        "    .reg .b32 %wa<16>;\n    .reg .b32 %wb<16>;\n    .reg .b32 %wc<16>;\n    .reg .b32 %wd<16>;\n    .reg .b64 %xa<4>;\n    .reg .b64 %xb<4>;\n    .reg .b64 %xc<4>;\n    .reg .b64 %xd<4>;"
    );
    let prefix = |p: &str, count: usize| (0..count).map(|i| format!("%{}{i}", if wide { format!("x{p}") } else { format!("w{p}") })).collect::<Vec<_>>().join(", ");
    let (al, bl) = (if a_row { "row" } else { "col" }, if b_row { "row" } else { "col" });
    let shape = format!("m{m}n{n}k{k}");
    let lda = if a_row { k } else { m };
    let ldb = if b_row { n } else { k };
    let zero: String = (0..nc).map(|i| if wide { format!("    mov.b64 %xc{i}, 0;\n") } else { format!("    mov.b32 %wc{i}, 0;\n") }).collect();
    let mma_types = if ty.int() {
        format!("s32.{0}.{0}.s32", ty.name())
    } else if wide {
        "f64.f64.f64.f64".to_owned()
    } else if matches!(ty, Ty::F16) {
        if half_acc { "f16.f16".to_owned() } else { "f32.f32".to_owned() }
    } else {
        format!("f32.{0}.{0}.f32", ty.name())
    };
    let body = format!(
        r#"    add.s64 %rd1, %rd15, 65536;
    add.s64 %rd2, %rd1, 8192;
    wmma.load.a.sync.aligned.{al}.{shape}.global.{t} {{{a}}}, [%rd1], {lda};
    wmma.load.b.sync.aligned.{bl}.{shape}.global.{t} {{{b}}}, [%rd2], {ldb};
{zero}    wmma.mma.sync.aligned.{al}.{bl}.{shape}.{mma} {{{d}}}, {{{a}}}, {{{b}}}, {{{c}}};
    wmma.store.d.sync.aligned.row.{shape}.global.{acc} [%rd15], {{{d}}}, {n};"#,
        t = ty.name(),
        mma = mma_types,
        a = prefix("a", na),
        b = prefix("b", nb),
        c = prefix("c", nc),
        d = prefix("d", nc),
    );
    let module = lower(&kernel_with_shared(&regs, "", &body));
    let mut machine = interp::Machine { params: vec![interp::GLOBAL, 0], ..Default::default() };
    let input = interp::GLOBAL + 0x10000;
    let bytes = ty.bytes();
    for r in 0..m {
        for c in 0..k {
            let at = if a_row { r * k + c } else { c * m + r };
            machine.write(input + (bytes * at) as u64, bytes as u32, ty.encode(a_at(ty, r, c)));
        }
    }
    for r in 0..k {
        for c in 0..n {
            let at = if b_row { r * n + c } else { c * k + r };
            machine.write(input + 8192 + (bytes * at) as u64, bytes as u32, ty.encode(b_at(ty, r, c)));
        }
    }
    interp::run_lanes(&module, &mut machine, 32);
    for r in 0..m {
        for c in 0..n {
            let want: i64 = (0..k).map(|x| i64::from(a_at(ty, r, x)) * i64::from(b_at(ty, x, c))).sum();
            let got: i64 = if wide {
                f64::from_bits(machine.read(interp::GLOBAL + 8 * (r * n + c) as u64, 8)) as i64
            } else if half_acc {
                h2f(machine.read(interp::GLOBAL + 2 * (r * n + c) as u64, 2) as u16) as i64
            } else if ty.int() {
                machine.read(interp::GLOBAL + 4 * (r * n + c) as u64, 4) as u32 as i32 as i64
            } else {
                f32::from_bits(machine.read(interp::GLOBAL + 4 * (r * n + c) as u64, 4) as u32) as i64
            };
            assert_eq!(got, want, "{ty:?} {shape} D[{r}][{c}] a_row={a_row} b_row={b_row} half={half_acc}");
        }
    }
}

#[test]
fn wmma_bf16_all_shapes() {
    for (m, n) in [(16, 16), (32, 8), (8, 32)] {
        check_wmma(Ty::Bf16, m, n, 16, true, false, false);
        check_wmma(Ty::Bf16, m, n, 16, false, true, false);
    }
}

#[test]
fn wmma_f16_other_shapes_and_accumulators() {
    for (m, n) in [(32, 8), (8, 32)] {
        check_wmma(Ty::F16, m, n, 16, true, false, false);
        check_wmma(Ty::F16, m, n, 16, false, true, false);
        check_wmma(Ty::F16, m, n, 16, true, true, true);
    }
}

#[test]
fn wmma_tf32_f64_and_integers() {
    check_wmma(Ty::Tf32, 16, 16, 8, true, false, false);
    check_wmma(Ty::Tf32, 16, 16, 8, false, true, false);
    check_wmma(Ty::F64, 8, 8, 4, true, false, false);
    check_wmma(Ty::F64, 8, 8, 4, false, true, false);
    for ty in [Ty::S8, Ty::U8] {
        for (m, n) in [(16, 16), (32, 8), (8, 32)] {
            check_wmma(ty, m, n, 16, true, false, false);
            check_wmma(ty, m, n, 16, false, true, false);
        }
    }
}

/// Direct mma.sync check with fragments laid out as in the PTX ISA tables.
fn check_mma_sync(ty: Ty, shape: &str, half_acc: bool) {
    let (m, n) = (16usize, 8usize);
    let (k, wide) = match shape {
        "m16n8k16" => (16, false),
        "m16n8k8" => (8, false),
        "m16n8k4" => (4, false),
        _ => (4, true),
    };
    let (m, n) = if wide { (8, 8) } else { (m, n) };
    let e = if wide { 1 } else { 4 / ty.bytes() };
    let rows_regs = if m == 16 { 2 } else { 1 };
    let hn = k / (4 * e);
    let na = rows_regs * hn;
    let nb = hn;
    let nc = if wide { 2 } else if half_acc { 2 } else { 4 };
    let acc = if wide { "f64" } else if half_acc { "f16" } else { "f32" };
    let types = if wide {
        "f64.f64.f64.f64".to_owned()
    } else {
        format!("{acc}.{0}.{0}.{acc}", ty.name())
    };
    let regs = "    .reg .b32 %wa<16>;\n    .reg .b32 %wb<16>;\n    .reg .b32 %wc<16>;\n    .reg .b32 %wd<16>;\n    .reg .b64 %xa<4>;\n    .reg .b64 %xb<4>;\n    .reg .b64 %xc<4>;\n    .reg .b64 %xd<4>;";
    let pre = if wide { "x" } else { "w" };
    let list = |p: &str, count: usize| (0..count).map(|i| format!("%{pre}{p}{i}")).collect::<Vec<_>>().join(", ");
    let ld = if wide { "ld.global.u64" } else { "ld.global.u32" };
    let w = if wide { 8 } else { 4 };
    // Per-lane fragments live at in + lane * 64 (a), +16*w.. etc.
    let mut loads = String::new();
    loads.push_str("    mov.u32 %r1, %laneid;\n    mul.wide.u32 %rd1, %r1, 128;\n    add.s64 %rd2, %rd15, 65536;\n    add.s64 %rd2, %rd2, %rd1;\n");
    for i in 0..na {
        loads.push_str(&format!("    {ld} %{pre}a{i}, [%rd2+{}];\n", w * i));
    }
    for i in 0..nb {
        loads.push_str(&format!("    {ld} %{pre}b{i}, [%rd2+{}];\n", 32 + w * i));
    }
    for i in 0..nc {
        loads.push_str(&format!("    mov.b{} %{pre}c{i}, 0;\n", if wide { 64 } else { 32 }));
    }
    let mut stores = String::new();
    for i in 0..nc {
        stores.push_str(&format!(
            "    mul.wide.u32 %rd3, %r1, {};\n    add.s64 %rd4, %rd15, %rd3;\n    st.global.{} [%rd4+{}], %{pre}d{i};\n",
            nc * w,
            if wide { "u64" } else { "u32" },
            w * i
        ));
    }
    let body = format!(
        "{loads}    mma.sync.aligned.{shape}.row.col.{types} {{{}}}, {{{}}}, {{{}}}, {{{}}};\n{stores}",
        list("d", nc), list("a", na), list("b", nb), list("c", nc)
    );
    let module = lower(&kernel_with_shared(regs, "", &body));
    let mut machine = interp::Machine { params: vec![interp::GLOBAL, 0], ..Default::default() };
    let input = interp::GLOBAL + 0x10000;
    for lane in 0..32usize {
        let (g, t) = (lane / 4, lane % 4);
        let base = input + (lane * 128) as u64;
        let pack = |vals: Vec<u64>| -> u64 {
            vals.iter().enumerate().fold(0u64, |acc, (i, v)| acc | (v << (8 * ty.bytes() * i)))
        };
        for idx in 0..na {
            let (r, h) = (idx % rows_regs, idx / rows_regs);
            let vals: Vec<u64> = (0..e).map(|i| ty.encode(a_at(ty, g + 8 * r, e * t + i + 4 * e * h))).collect();
            let word = if wide { vals[0] } else { pack(vals) };
            machine.write(base + (w * idx) as u64, w as u32, word);
        }
        for h in 0..nb {
            let vals: Vec<u64> = (0..e).map(|i| ty.encode(b_at(ty, e * t + i + 4 * e * h, g))).collect();
            let word = if wide { vals[0] } else { pack(vals) };
            machine.write(base + 32 + (w * h) as u64, w as u32, word);
        }
    }
    interp::run_lanes(&module, &mut machine, 32);
    for lane in 0..32usize {
        let (g, t) = (lane / 4, lane % 4);
        for r in 0..rows_regs {
            for j in 0..2 {
                let (row, col) = (g + 8 * r, 2 * t + j);
                let want: i64 = (0..k).map(|x| i64::from(a_at(ty, row, x)) * i64::from(b_at(ty, x, col))).sum();
                let slot = |i: usize| machine.read(interp::GLOBAL + (lane * nc * w + w * i) as u64, w as u32);
                let got = if wide {
                    f64::from_bits(slot(r * 2 + j)) as i64
                } else if half_acc {
                    let word = slot(r) as u32;
                    h2f(((word >> (16 * j)) & 0xffff) as u16) as i64
                } else {
                    f32::from_bits(slot(r * 2 + j) as u32) as i64
                };
                assert_eq!(got, want, "{ty:?} {shape} lane {lane} D[{row}][{col}]");
            }
        }
    }
}

#[test]
fn mma_sync_bf16_tf32_f16acc_and_f64() {
    check_mma_sync(Ty::Bf16, "m16n8k16", false);
    check_mma_sync(Ty::Bf16, "m16n8k8", false);
    check_mma_sync(Ty::F16, "m16n8k16", true);
    check_mma_sync(Ty::F16, "m16n8k8", true);
    check_mma_sync(Ty::Tf32, "m16n8k8", false);
    check_mma_sync(Ty::Tf32, "m16n8k4", false);
    check_mma_sync(Ty::F64, "m8n8k4", false);
}

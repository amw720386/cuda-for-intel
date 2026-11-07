//! Tier-3 tensor-core emulation: `mma.sync` shapes the native lowering does
//! not cover (bf16, tf32, f16 accumulators, f64) and `wmma` for every shape.
//!
//! The native `Instruction::Mma` handles f16 inputs with f32 accumulators and
//! the 8-bit integer shapes. Everything else is expanded here, as PTX text, into
//! subgroup shuffles and scalar f32/f64 fused multiply-adds: each lane gathers
//! the A row and B column fragments of its output elements from the four lanes
//! that hold them and accumulates in fp32 (or fp64). Products of bf16/f16/tf32
//! operands are exact in f32, so the only difference from hardware is the
//! summation order.
//!
//! `wmma` fragments are opaque, so they reuse the register layout of the
//! `mma.sync` tile they are built from (see `Layout`). The one shape the
//! tier-2 normaliser already covers (m16n16k16 with f16 inputs) is left alone.

use super::{Ctx, Inst, render};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Elem {
    F16,
    Bf16,
    Tf32,
    S8,
    U8,
    F64,
}

impl Elem {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "tf32" => Self::Tf32,
            "s8" => Self::S8,
            "u8" => Self::U8,
            "f64" => Self::F64,
            _ => return None,
        })
    }
    /// Elements per 32-bit fragment register.
    fn per_reg(self) -> u32 {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::S8 | Self::U8 => 4,
            Self::Tf32 | Self::F64 => 1,
        }
    }
    fn bytes(self) -> u32 {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::S8 | Self::U8 => 1,
            Self::Tf32 => 4,
            Self::F64 => 8,
        }
    }
    fn is_int(self) -> bool {
        matches!(self, Self::S8 | Self::U8)
    }
}

/// Scratch allocator (separate from tier3's: this module needs many registers).
struct Gen<'a> {
    ctx: &'a mut Ctx,
    lines: Vec<String>,
    n32: u32,
    n64: u32,
    np: u32,
}

impl<'a> Gen<'a> {
    fn new(ctx: &'a mut Ctx) -> Self {
        Gen { ctx, lines: Vec::new(), n32: 0, n64: 0, np: 0 }
    }
    fn r32(&mut self) -> String {
        self.n32 += 1;
        format!("%__m32_{}", self.n32 - 1)
    }
    fn r64(&mut self) -> String {
        self.n64 += 1;
        format!("%__m64_{}", self.n64 - 1)
    }
    fn rp(&mut self) -> String {
        self.np += 1;
        format!("%__mp_{}", self.np - 1)
    }
    fn p(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }
    fn finish(self) -> Vec<String> {
        let t = &mut self.ctx.t3;
        t.m32 = t.m32.max(self.n32);
        t.m64 = t.m64.max(self.n64);
        t.mp = t.mp.max(self.np);
        self.lines
    }
}

fn parse_regs(text: &str) -> Vec<String> {
    text.trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .map(|r| r.trim().to_owned())
        .filter(|r| !r.is_empty())
        .collect()
}

pub(super) fn rewrite(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let family = inst.op.split('.').next()?;
    match family {
        "mma" if inst.op.starts_with("mma.sync.aligned.") => mma_sync(ctx, inst),
        "wmma" => wmma(ctx, inst),
        _ => None,
    }
}

// ---- lane geometry ---------------------------------------------------------

struct Lane {
    /// `laneid >> 2` and `laneid & 3`.
    g: String,
    tl: String,
    /// Constant 31 held in a register (the dynamic shuffle form needs a register control).
    ctl: String,
}

fn lane(g: &mut Gen) -> Lane {
    let (l, gg, tl, ctl) = (g.r32(), g.r32(), g.r32(), g.r32());
    g.p(format!("mov.u32 {l} , %laneid ;"));
    g.p(format!("shr.u32 {gg} , {l} , 2 ;"));
    g.p(format!("and.b32 {tl} , {l} , 3 ;"));
    g.p(format!("mov.b32 {ctl} , 31 ;"));
    Lane { g: gg, tl, ctl }
}

fn shuffle32(g: &mut Gen, lane: &Lane, dst: &str, src: &str, from: &str) {
    g.p(format!("shfl.sync.idx.b32 {dst} , {src} , {from} , {} , 0xffffffff ;", lane.ctl));
}

fn shuffle64(g: &mut Gen, lane: &Lane, dst: &str, src: &str, from: &str) {
    let (lo, hi, slo, shi) = (g.r32(), g.r32(), g.r32(), g.r32());
    g.p(format!("mov.b64 {{ {lo} , {hi} }} , {src} ;"));
    shuffle32(g, lane, &slo, &lo, from);
    shuffle32(g, lane, &shi, &hi, from);
    g.p(format!("mov.b64 {dst} , {{ {slo} , {shi} }} ;"));
}

/// Unpacks element `i` of a fragment register into an f32 register `out`.
fn unpack_f32(g: &mut Gen, kind: Elem, reg: &str, i: u32, out: &str) {
    match (kind, i) {
        (Elem::F16, 0) => {
            g.p(format!("and.b32 {out} , {reg} , 65535 ;"));
            g.p(format!("cvt.f32.f16 {out} , {out} ;"));
        }
        (Elem::F16, _) => {
            g.p(format!("shr.u32 {out} , {reg} , 16 ;"));
            g.p(format!("cvt.f32.f16 {out} , {out} ;"));
        }
        (Elem::Bf16, 0) => g.p(format!("shl.b32 {out} , {reg} , 16 ;")),
        (Elem::Bf16, _) => g.p(format!("and.b32 {out} , {reg} , 0xffff0000 ;")),
        _ => g.p(format!("and.b32 {out} , {reg} , 0xffffe000 ;")),
    }
}

/// Accumulates one 16x8 (or 8x8 when `rows == 1`) tile in f32 or f64.
/// `a`: fragment registers indexed `r + rows * h`; `b`: one register per k half;
/// `acc`: `rows * 2` registers in (r, j) order, already holding C.
fn emulate_tile(g: &mut Gen, kind: Elem, rows: usize, hn: usize, a: &[String], b: &[String], acc: &[String]) {
    let ln = lane(g);
    let e = kind.per_reg();
    let wide = kind == Elem::F64;
    let (la, lb0, lb1) = (g.r32(), g.r32(), g.r32());
    let (base_a, base_b0, base_b1) = (g.r32(), g.r32(), g.r32());
    g.p(format!("shl.b32 {base_a} , {} , 2 ;", ln.g));
    g.p(format!("shl.b32 {base_b0} , {} , 3 ;", ln.tl));
    g.p(format!("add.u32 {base_b1} , {base_b0} , 4 ;"));
    let ra = |g: &mut Gen| if wide { g.r64() } else { g.r32() };
    let fa: Vec<String> = a.iter().map(|_| ra(g)).collect();
    let fb0: Vec<String> = b.iter().map(|_| ra(g)).collect();
    let fb1: Vec<String> = b.iter().map(|_| ra(g)).collect();
    // Unpacked f32 operands: [reg][element].
    let ua: Vec<Vec<String>> = a.iter().map(|_| (0..e).map(|_| g.r32()).collect()).collect();
    let ub0: Vec<Vec<String>> = b.iter().map(|_| (0..e).map(|_| g.r32()).collect()).collect();
    let ub1: Vec<Vec<String>> = b.iter().map(|_| (0..e).map(|_| g.r32()).collect()).collect();
    let k_steps = if wide { 4 } else { 4 };
    for t in 0..k_steps {
        g.p(format!("add.u32 {la} , {base_a} , {t} ;"));
        g.p(format!("add.u32 {lb0} , {base_b0} , {t} ;"));
        g.p(format!("add.u32 {lb1} , {base_b1} , {t} ;"));
        for (src, dst) in a.iter().zip(&fa) {
            if wide { shuffle64(g, &ln, dst, src, &la) } else { shuffle32(g, &ln, dst, src, &la) }
        }
        for (src, (d0, d1)) in b.iter().zip(fb0.iter().zip(&fb1)) {
            if wide {
                shuffle64(g, &ln, d0, src, &lb0);
                shuffle64(g, &ln, d1, src, &lb1);
            } else {
                shuffle32(g, &ln, d0, src, &lb0);
                shuffle32(g, &ln, d1, src, &lb1);
            }
        }
        if wide {
            for r in 0..rows {
                g.p(format!("fma.rn.f64 {} , {} , {} , {} ;", acc[2 * r], fa[r], fb0[0], acc[2 * r]));
                g.p(format!("fma.rn.f64 {} , {} , {} , {} ;", acc[2 * r + 1], fa[r], fb1[0], acc[2 * r + 1]));
            }
            continue;
        }
        for (regs, outs) in [(&fa, &ua), (&fb0, &ub0), (&fb1, &ub1)] {
            for (reg, out) in regs.iter().zip(outs) {
                for (i, o) in out.iter().enumerate() {
                    unpack_f32(g, kind, reg, i as u32, o);
                }
            }
        }
        for r in 0..rows {
            for j in 0..2 {
                for h in 0..hn {
                    let idx = r + rows * h;
                    let bsrc = if j == 0 { &ub0[h] } else { &ub1[h] };
                    for i in 0..e as usize {
                        let o = &acc[2 * r + j];
                        g.p(format!("fma.rn.f32 {o} , {} , {} , {o} ;", ua[idx][i], bsrc[i]));
                    }
                }
            }
        }
    }
}

/// f32 accumulator registers from `c` (f32 registers, or f16x2 packs when `half`).
fn load_acc(g: &mut Gen, c: &[String], half: bool, rows: usize) -> Vec<String> {
    let mut acc = Vec::new();
    if half {
        for r in 0..rows {
            let (lo, hi) = (g.r32(), g.r32());
            g.p(format!("and.b32 {lo} , {} , 65535 ;", c[r]));
            g.p(format!("shr.u32 {hi} , {} , 16 ;", c[r]));
            g.p(format!("cvt.f32.f16 {lo} , {lo} ;"));
            g.p(format!("cvt.f32.f16 {hi} , {hi} ;"));
            acc.push(lo);
            acc.push(hi);
        }
    } else {
        for reg in c {
            let t = g.r32();
            g.p(format!("mov.b32 {t} , {reg} ;"));
            acc.push(t);
        }
    }
    acc
}

fn store_acc(g: &mut Gen, d: &[String], acc: &[String], half: bool, rows: usize) {
    if half {
        for r in 0..rows {
            g.p(format!("cvt.rn.f16x2.f32 {} , {} , {} ;", d[r], acc[2 * r + 1], acc[2 * r]));
        }
    } else {
        for (reg, a) in d.iter().zip(acc) {
            g.p(format!("mov.b32 {reg} , {a} ;"));
        }
    }
}

// ---- mma.sync ---------------------------------------------------------------

fn mma_sync(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let mut p = parts[3..].to_vec();
    let shape = p.remove(0);
    p.retain(|x| *x != "row" && *x != "col" && *x != "satfinite");
    let [dty, aty, bty, cty] = p[..] else { return None };
    if aty != bty || inst.operands.len() != 4 {
        return None;
    }
    let kind = Elem::parse(aty)?;
    let (d, a, b, c) = (
        parse_regs(inst.operand(0)),
        parse_regs(inst.operand(1)),
        parse_regs(inst.operand(2)),
        parse_regs(inst.operand(3)),
    );
    let half_acc = dty == "f16" && cty == "f16";
    let mut g = Gen::new(ctx);
    match (shape, kind, dty, cty) {
        ("m16n8k16" | "m16n8k8", Elem::Bf16, "f32", "f32") | ("m16n8k16" | "m16n8k8", Elem::F16, "f16", "f16") => {
            let hn = if shape == "m16n8k16" { 2 } else { 1 };
            if a.len() != 2 * hn || b.len() != hn || d.len() != if half_acc { 2 } else { 4 } || c.len() != d.len() {
                return None;
            }
            let acc = load_acc(&mut g, &c, half_acc, 2);
            emulate_tile(&mut g, kind, 2, hn, &a, &b, &acc);
            store_acc(&mut g, &d, &acc, half_acc, 2);
        }
        ("m16n8k8" | "m16n8k4", Elem::Tf32, "f32", "f32") => {
            let hn = if shape == "m16n8k8" { 2 } else { 1 };
            if a.len() != 2 * hn || b.len() != hn || d.len() != 4 || c.len() != 4 {
                return None;
            }
            let acc = load_acc(&mut g, &c, false, 2);
            emulate_tile(&mut g, kind, 2, hn, &a, &b, &acc);
            store_acc(&mut g, &d, &acc, false, 2);
        }
        ("m8n8k4", Elem::F64, "f64", "f64") => {
            if a.len() != 1 || b.len() != 1 || d.len() != 2 || c.len() != 2 {
                return None;
            }
            let acc: Vec<String> = c.iter().map(|_| g.r64()).collect();
            for (t, reg) in acc.iter().zip(&c) {
                g.p(format!("mov.b64 {t} , {reg} ;"));
            }
            emulate_tile(&mut g, kind, 1, 1, &a, &b, &acc);
            for (reg, t) in d.iter().zip(&acc) {
                g.p(format!("mov.b64 {reg} , {t} ;"));
            }
        }
        _ => return None,
    }
    let mut lines = g.finish();
    if !inst.guard.is_empty() {
        // The emulation is lane-uniform; a guard would have to wrap every line.
        lines = lines.into_iter().map(|l| format!("{} {l}", inst.guard)).collect();
    }
    Some(lines)
}

// ---- wmma -------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Shape {
    m: u32,
    n: u32,
    k: u32,
}

impl Shape {
    fn parse(parts: &[&str]) -> Option<Self> {
        for p in parts {
            let rest = p.strip_prefix('m')?;
            let (m, rest) = rest.split_once('n')?;
            let (n, k) = rest.split_once('k')?;
            return Some(Shape { m: m.parse().ok()?, n: n.parse().ok()?, k: k.parse().ok()? });
        }
        None
    }
}

/// Register layout of a wmma operand in terms of mma tiles.
struct Layout {
    tiles_m: u32,
    tiles_n: u32,
    /// rows per tile: 2 (16 rows) or 1 (8 rows).
    rows: u32,
    kind: Elem,
    /// k halves per tile (a register holds `per_reg` consecutive k).
    hn: u32,
}

fn layout(shape: Shape, kind: Elem) -> Option<Layout> {
    let rows = match shape.m {
        8 => 1,
        16 | 32 => 2,
        _ => return None,
    };
    let tiles_m = if shape.m == 32 { 2 } else { 1 };
    if shape.n % 8 != 0 {
        return None;
    }
    let tiles_n = shape.n / 8;
    // k covered by one tile register group: per_reg * 4 lanes * hn.
    let per_tile_k = kind.per_reg() * 4;
    if shape.k % per_tile_k != 0 {
        return None;
    }
    let hn = shape.k / per_tile_k;
    if kind == Elem::F64 && (shape.m, shape.n, shape.k) != (8, 8, 4) {
        return None;
    }
    Some(Layout { tiles_m, tiles_n, rows, kind, hn })
}

impl Layout {
    fn a_per_tile(&self) -> u32 {
        if self.kind.is_int() { self.rows } else { self.rows * self.hn }
    }
    fn b_per_tile(&self) -> u32 {
        if self.kind.is_int() { 1 } else { self.hn }
    }
}

fn address(g: &mut Gen, base: &str, offset: i64, wide: bool, row_major: bool, stride: &str, row: &str, col: &str, bytes: u32) -> String {
    let idx = g.r32();
    if row_major {
        g.p(format!("mad.lo.u32 {idx} , {row} , {stride} , {col} ;"));
    } else {
        g.p(format!("mad.lo.u32 {idx} , {col} , {stride} , {row} ;"));
    }
    if wide {
        // Address sums get their own registers: a register keeps one kind for the whole entry.
        let off = g.r64();
        g.p(format!("mul.wide.u32 {off} , {idx} , {bytes} ;"));
        if offset != 0 {
            g.p(format!("add.s64 {off} , {off} , {offset} ;"));
        }
        g.ctx.t3.addr_seq += 1;
        let a = format!("%__ma_{}", g.ctx.t3.addr_seq - 1);
        g.ctx.t3.addr_max = g.ctx.t3.addr_max.max(g.ctx.t3.addr_seq);
        g.p(format!("add.s64 {a} , {base} , {off} ;"));
        a
    } else {
        let a = g.r32();
        g.p(format!("mul.lo.u32 {a} , {idx} , {bytes} ;"));
        if offset != 0 {
            g.p(format!("add.u32 {a} , {a} , {offset} ;"));
        }
        g.p(format!("add.u32 {a} , {base} , {a} ;"));
        a
    }
}

/// (row expression, column expression) of one fragment element.
type At = (String, String);

fn constant(g: &mut Gen, v: u32) -> String {
    let r = g.r32();
    g.p(format!("mov.b32 {r} , {v} ;"));
    r
}

fn add_const(g: &mut Gen, a: &str, v: u32) -> String {
    if v == 0 {
        return a.to_owned();
    }
    let r = g.r32();
    g.p(format!("add.u32 {r} , {a} , {v} ;"));
    r
}

fn wmma(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let shape = Shape::parse(parts.iter().copied().filter(|p| p.starts_with('m') && p.contains('n') && p.contains('k')).collect::<Vec<_>>().as_slice())?;
    let kind_of_op = *parts.get(1)?;
    // Element type of the operands: the load type for a/b, the mma operand type for mma.
    let elem_text = match kind_of_op {
        "load" | "store" => match *parts.get(2)? {
            "a" | "b" => parts.last().copied()?,
            _ => "",
        },
        "mma" => {
            // wmma.mma.sync.aligned.row.col.SHAPE.D.A.B.C  or  ...SHAPE.D.C (f16)
            let tail: Vec<&str> = parts.iter().copied().skip_while(|p| !p.starts_with('m') || !p.contains('k')).skip(1).collect();
            match tail[..] {
                [_, a, _, _] => a,
                _ => "f16",
            }
        }
        _ => return None,
    };
    let kind = if elem_text.is_empty() {
        // c/d fragments: the accumulator type decides; element kind comes from the other operands.
        None
    } else {
        Some(Elem::parse(elem_text)?)
    };
    // The tier-2 normaliser already handles m16n16k16 with f16 inputs and f16/f32 accumulators.
    if (shape.m, shape.n, shape.k) == (16, 16, 16) {
        let acc_int = parts.last().is_some_and(|t| *t == "s32");
        let tier2 = match kind_of_op {
            "load" | "store" => match kind {
                Some(k) => k == Elem::F16,
                None => !acc_int,
            },
            _ => kind == Some(Elem::F16),
        };
        if tier2 {
            return None;
        }
    }
    match kind_of_op {
        "load" | "store" => wmma_memory(ctx, inst, &parts, shape, kind),
        "mma" => wmma_mma(ctx, inst, &parts, shape),
        _ => None,
    }
}

fn wmma_memory(ctx: &mut Ctx, inst: &Inst, parts: &[&str], shape: Shape, kind: Option<Elem>) -> Option<Vec<String>> {
    let is_store = parts[1] == "store";
    let which = if is_store { "d" } else { *parts.get(2)? };
    let row_major = if parts.contains(&"row") {
        true
    } else if parts.contains(&"col") {
        false
    } else {
        return None;
    };
    let space = if parts.contains(&"shared") || parts.contains(&"shared::cta") { "shared" } else { "global" };
    let acc_ty = *parts.last()?;
    let (regs, base_text) = if is_store {
        (parse_regs(inst.operand(1)), inst.operand(0))
    } else {
        (parse_regs(inst.operand(0)), inst.operand(1))
    };
    let base_text = base_text.trim().trim_start_matches('[').trim_end_matches(']').trim().to_owned();
    let stride = inst.operands.get(2).cloned().unwrap_or_else(|| shape.n.to_string());
    let (base, offset) = match base_text.split_once('+') {
        Some((reg, off)) => (reg.trim().to_owned(), super::immediate(off).unwrap_or(0) as i64),
        None => (base_text.clone(), 0),
    };
    let wide = ctx.width_of(&base).unwrap_or(64) == 64;
    let mut g = Gen::new(ctx);
    let ln = lane(&mut g);
    let stride_r = {
        let r = g.r32();
        g.p(format!("mov.b32 {r} , {stride} ;"));
        r
    };
    let guard = inst.guard.clone();
    // Fragment elements as (reg index, [coordinates of each packed element]).
    let frag: Vec<Vec<At>>;
    let elem_bytes: u32;
    let elem_kind: Elem;
    match which {
        "a" | "b" => {
            let k = kind?;
            let lay = layout(shape, k)?;
            elem_kind = k;
            elem_bytes = k.bytes();
            let mut v: Vec<Vec<At>> = Vec::new();
            let e = k.per_reg();
            if which == "a" {
                for ti in 0..lay.tiles_m {
                    for idx in 0..lay.a_per_tile() {
                        // idx -> (r, h); integer kinds have one k group and idx = r.
                        let (r, h) = if k.is_int() { (idx, 0) } else { (idx % lay.rows, idx / lay.rows) };
                        let row = add_const(&mut g, &ln.g, 16 * ti + 8 * r);
                        let mut elems = Vec::new();
                        for i in 0..e {
                            // k = e*t + i + 4e*h
                            let kk = {
                                let t = g.r32();
                                g.p(format!("mul.lo.u32 {t} , {} , {e} ;", ln.tl));
                                add_const(&mut g, &t, i + 4 * e * h)
                            };
                            elems.push((row.clone(), kk));
                        }
                        v.push(elems);
                    }
                }
            } else {
                for tj in 0..lay.tiles_n {
                    for h in 0..lay.b_per_tile() {
                        let col = add_const(&mut g, &ln.g, 8 * tj);
                        let mut elems = Vec::new();
                        for i in 0..e {
                            let kk = {
                                let t = g.r32();
                                g.p(format!("mul.lo.u32 {t} , {} , {e} ;", ln.tl));
                                add_const(&mut g, &t, i + 4 * e * h)
                            };
                            elems.push((kk, col.clone()));
                        }
                        v.push(elems);
                    }
                }
            }
            frag = v;
        }
        _ => {
            // Accumulator: f32 / s32 / f16 (packed pairs) / f64.
            let (rows, tiles_m, tiles_n) = match shape.m {
                8 => (1, 1, shape.n / 8),
                16 => (2, 1, shape.n / 8),
                32 => (2, 2, shape.n / 8),
                _ => return None,
            };
            let f64acc = acc_ty == "f64";
            let half = acc_ty == "f16";
            elem_kind = if f64acc { Elem::F64 } else if half { Elem::F16 } else { Elem::Tf32 };
            elem_bytes = if f64acc { 8 } else if half { 2 } else { 4 };
            let mut v: Vec<Vec<At>> = Vec::new();
            for ti in 0..tiles_m {
                for tj in 0..tiles_n {
                    for r in 0..rows {
                        let row = add_const(&mut g, &ln.g, 16 * ti + 8 * r);
                        let c0 = {
                            let t = g.r32();
                            g.p(format!("shl.b32 {t} , {} , 1 ;", ln.tl));
                            add_const(&mut g, &t, 8 * tj)
                        };
                        let c1 = add_const(&mut g, &c0, 1);
                        if half {
                            v.push(vec![(row.clone(), c0), (row.clone(), c1)]);
                        } else {
                            v.push(vec![(row.clone(), c0)]);
                            v.push(vec![(row.clone(), c1)]);
                        }
                    }
                }
            }
            frag = v;
        }
    }
    if frag.len() > regs.len() {
        return None;
    }
    let pack_ty = |bytes: u32| match bytes {
        1 => "u8",
        2 => "u16",
        4 => "u32",
        _ => "u64",
    };
    for (reg, elems) in regs.iter().zip(&frag) {
        let n = elems.len() as u32;
        let mut addrs = Vec::new();
        for (r, c) in elems {
            addrs.push(address(&mut g, &base, offset, wide, row_major, &stride_r, r, c, elem_bytes));
        }
        let mem = |a: &String| format!("[{a}]");
        let ty = pack_ty(elem_bytes);
        if n == 1 {
            if is_store {
                g.p(render(&guard, &format!("st.{space}.{ty}"), &[mem(&addrs[0]), reg.clone()]));
            } else {
                g.p(render(&guard, &format!("ld.{space}.{ty}"), &[reg.clone(), mem(&addrs[0])]));
            }
            continue;
        }
        if is_store {
            let t = g.r32();
            for (i, a) in addrs.iter().enumerate() {
                match elem_bytes {
                    2 if i == 0 => g.p(render(&guard, &format!("st.{space}.u16"), &[mem(a), reg.clone()])),
                    2 => {
                        g.p(format!("shr.u32 {t} , {reg} , 16 ;"));
                        g.p(render(&guard, &format!("st.{space}.u16"), &[mem(a), t.clone()]));
                    }
                    _ => {
                        g.p(format!("bfe.u32 {t} , {reg} , {} , 8 ;", 8 * i));
                        g.p(render(&guard, &format!("st.{space}.u8"), &[mem(a), t.clone()]));
                    }
                }
            }
        } else if elem_bytes == 2 {
            let (lo, hi) = (g.r32(), g.r32());
            g.p(render(&guard, &format!("ld.{space}.u16"), &[lo.clone(), mem(&addrs[0])]));
            g.p(render(&guard, &format!("ld.{space}.u16"), &[hi.clone(), mem(&addrs[1])]));
            g.p(format!("shl.b32 {hi} , {hi} , 16 ;"));
            g.p(format!("or.b32 {reg} , {lo} , {hi} ;"));
        } else {
            g.p(format!("mov.b32 {reg} , 0 ;"));
            let t = g.r32();
            for (i, a) in addrs.iter().enumerate() {
                g.p(render(&guard, &format!("ld.{space}.u8"), &[t.clone(), mem(a)]));
                g.p(format!("bfi.b32 {reg} , {t} , {reg} , {} , 8 ;", 8 * i));
            }
        }
    }
    // Unused trailing registers (padding in wider fragments) read as zero.
    if !is_store {
        for reg in &regs[frag.len()..] {
            if elem_kind == Elem::F64 {
                g.p(format!("mov.b64 {reg} , 0 ;"));
            } else {
                g.p(format!("mov.b32 {reg} , 0 ;"));
            }
        }
    }
    Some(g.finish())
}

fn wmma_mma(ctx: &mut Ctx, inst: &Inst, parts: &[&str], shape: Shape) -> Option<Vec<String>> {
    // Operand and accumulator types from the tail of the opcode.
    let tail: Vec<&str> = {
        let at = parts.iter().position(|p| p.starts_with('m') && p.contains('n') && p.contains('k') && p.len() >= 5)?;
        parts[at + 1..].iter().copied().filter(|p| *p != "satfinite").collect()
    };
    let (dty, aty, cty) = match tail[..] {
        [d, c] => (d, "f16", c),
        [d, a, b, c] if a == b => (d, a, c),
        _ => return None,
    };
    let kind = Elem::parse(aty)?;
    let lay = layout(shape, kind)?;
    let (d, a, b, c) = (
        parse_regs(inst.operand(0)),
        parse_regs(inst.operand(1)),
        parse_regs(inst.operand(2)),
        parse_regs(inst.operand(3)),
    );
    let half_acc = dty == "f16" && cty == "f16";
    if (dty == "f16") != (cty == "f16") && !kind.is_int() {
        // Mixed f16/f32 accumulate types: unsupported combination.
        return None;
    }
    let (rows, tiles_m, tiles_n) = (lay.rows as usize, lay.tiles_m as usize, lay.tiles_n as usize);
    let a_t = lay.a_per_tile() as usize;
    let b_t = lay.b_per_tile() as usize;
    let c_t = if kind == Elem::F64 { 2 } else if half_acc { rows } else { rows * 2 };
    if a.len() < a_t * tiles_m || b.len() < b_t * tiles_n || c.len() != d.len() || c.len() != c_t * tiles_m * tiles_n {
        return None;
    }
    let mut g = Gen::new(ctx);
    for ti in 0..tiles_m {
        for tj in 0..tiles_n {
            let ar = &a[ti * a_t..(ti + 1) * a_t];
            let br = &b[tj * b_t..(tj + 1) * b_t];
            let at = (ti * tiles_n + tj) * c_t;
            let cr = &c[at..at + c_t];
            let dr = &d[at..at + c_t];
            if kind.is_int() {
                // The integer tiles map onto the native 8-bit mma.
                let sig = format!("{aty}.{aty}");
                if rows == 2 {
                    g.p(format!(
                        "mma.sync.aligned.m16n8k16.row.col.s32.{sig}.s32 {{ {} }} , {{ {} }} , {{ {} }} , {{ {} }} ;",
                        dr.join(" , "), ar.join(" , "), br.join(" , "), cr.join(" , ")
                    ));
                } else {
                    g.p(format!(
                        "mma.sync.aligned.m8n8k16.row.col.s32.{sig}.s32 {{ {} }} , {{ {} }} , {{ {} }} , {{ {} }} ;",
                        dr.join(" , "), ar.join(" , "), br.join(" , "), cr.join(" , ")
                    ));
                }
                continue;
            }
            if kind == Elem::F16 && !half_acc && shape.k == 16 {
                // f16 inputs with f32 accumulators map onto the native m16n8k16 mma.
                let mut aa: Vec<String> = ar.to_vec();
                let mut cc: Vec<String> = cr.to_vec();
                let mut dd: Vec<String> = dr.to_vec();
                if rows == 1 {
                    // 8-row tile: rows 8..15 are dummies (zeros in, ignored out).
                    let zero = g.r32();
                    g.p(format!("mov.b32 {zero} , 0 ;"));
                    let scratch: Vec<String> = (0..2).map(|_| g.r32()).collect();
                    aa = vec![ar[0].clone(), zero.clone(), ar[1].clone(), zero.clone()];
                    cc = vec![cr[0].clone(), cr[1].clone(), zero.clone(), zero.clone()];
                    dd = vec![dr[0].clone(), dr[1].clone(), scratch[0].clone(), scratch[1].clone()];
                }
                g.p(format!(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{ {} }} , {{ {} }} , {{ {} }} , {{ {} }} ;",
                    dd.join(" , "), aa.join(" , "), br.join(" , "), cc.join(" , ")
                ));
                continue;
            }
            if kind == Elem::F64 {
                let acc: Vec<String> = cr.iter().map(|_| g.r64()).collect();
                for (t, reg) in acc.iter().zip(cr) {
                    g.p(format!("mov.b64 {t} , {reg} ;"));
                }
                emulate_tile(&mut g, kind, 1, 1, ar, br, &acc);
                for (reg, t) in dr.iter().zip(&acc) {
                    g.p(format!("mov.b64 {reg} , {t} ;"));
                }
                continue;
            }
            let acc = load_acc(&mut g, cr, half_acc, rows);
            emulate_tile(&mut g, kind, rows, lay.hn as usize, ar, br, &acc);
            store_acc(&mut g, dr, &acc, half_acc, rows);
        }
    }
    let mut lines = g.finish();
    if !inst.guard.is_empty() {
        lines = lines.into_iter().map(|l| format!("{} {l}", inst.guard)).collect();
    }
    Some(lines)
}

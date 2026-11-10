//! Texture and surface instructions on texture/surface object handles (`tex`, `txq`, `suld.b`,
//! `sust.b`, `suq`), emulated on linear memory. Child module of `tier3`.
//!
//! A handle is the device address of a descriptor (`crate::texdesc`). Each instruction loads
//! the descriptor fields it needs, does CUDA's addressing (normalised coordinates, wrap,
//! clamp, mirror, border, per axis) and filtering (nearest, or linear with CUDA's 8-bit
//! fixed-point weights) in ordinary arithmetic, and reads texels with global loads. Every
//! texel address is clamped into the resource before the load, so no load can leave it; an
//! out-of-range texel is replaced by the border colour (or 0) afterwards.
//!
//! Not handled here (the module is rejected with a clear message): `.texref`/`.surfref`
//! module variables, layered (`a1d`/`a2d`), cube and multisample geometries, `tld4`, `tex.grad`,
//! formatted surface stores (`sust.p`), surface reductions (`sured`), and residency predicates.

use super::{B, Inst, State};
use crate::ptx::texdesc as d;

const F_ONE: &str = "0f3F800000";
const F_HALF: &str = "0f3F000000";
const F_MINUS_ONE: &str = "0fBF800000";

/// `{ %a , %b }` -> `["%a", "%b"]`.
fn braced(text: &str) -> Vec<String> {
    text.trim().trim_start_matches('{').trim_end_matches('}').split(',').map(|r| r.trim().to_owned()).filter(|r| !r.is_empty()).collect()
}

/// `[ %h , { %x , %y } ]` or `[ %h ]` -> handle register and coordinates.
fn handle_and_coords(text: &str) -> Option<(String, Vec<String>)> {
    let inner = text.trim().strip_prefix('[')?.strip_suffix(']')?.trim();
    let (handle, rest) = match inner.find(',') {
        Some(at) => (inner[..at].trim(), inner[at + 1..].trim()),
        None => (inner, ""),
    };
    // A `.texref` / `.surfref` name instead of a register is the texture-reference form.
    if !handle.starts_with('%') || handle.contains(|c: char| c.is_whitespace() || c == '+') {
        return None;
    }
    let coords = if rest.is_empty() { Vec::new() } else { braced(rest) };
    Some((handle.to_owned(), coords))
}

fn geometry(part: &str) -> Option<usize> {
    match part {
        "1d" => Some(1),
        "2d" => Some(2),
        "3d" => Some(3),
        _ => None,
    }
}

/// Parsed `tex[.level].{1d,2d,3d}.v4.{f32,s32,u32}.{f32,s32}`.
struct Tex {
    dims: usize,
    float_out: bool,
    float_coords: bool,
    dst: Vec<String>,
    handle: String,
    coords: Vec<String>,
}

fn parse_tex(inst: &Inst) -> Option<Tex> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let mut i = 1;
    let level = parts.get(1) == Some(&"level");
    if level {
        i += 1;
    }
    let dims = geometry(parts.get(i)?)?;
    if parts.get(i + 1) != Some(&"v4") || parts.len() != i + 4 {
        return None;
    }
    let float_out = match parts[i + 2] {
        "f32" => true,
        "s32" | "u32" => false,
        _ => return None,
    };
    let float_coords = match parts[i + 3] {
        "f32" => true,
        "s32" => false,
        _ => return None,
    };
    // `tex.level` carries the level of detail as a third operand; there are no mipmaps, so
    // level 0 is the only level.
    if inst.operands.len() != if level { 3 } else { 2 } {
        return None;
    }
    let dst = braced(inst.operand(0));
    if dst.len() != 4 || dst.iter().any(|r| !r.starts_with('%')) || inst.operand(0).contains('|') {
        return None;
    }
    let (handle, coords) = handle_and_coords(inst.operand(1))?;
    if coords.len() < dims || coords.len() > 4 {
        return None;
    }
    Some(Tex { dims, float_out, float_coords, dst, handle, coords })
}

/// Parsed `suld.b.{1d,2d,3d}[.v2,.v4].{b8,b16,b32,b64}[.{trap,clamp,zero}]` and the `sust.b` form.
struct Surf {
    store: bool,
    dims: usize,
    bits: u32,
    regs: Vec<String>,
    handle: String,
    coords: Vec<String>,
    clamp: bool,
}

fn parse_surf(inst: &Inst) -> Option<Surf> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let store = parts[0] == "sust";
    if parts.get(1) != Some(&"b") {
        return None;
    }
    let dims = geometry(parts.get(2)?)?;
    let mut i = 3;
    let count = match parts.get(i) {
        Some(&"v2") => 2,
        Some(&"v4") => 4,
        _ => 1,
    };
    if count > 1 {
        i += 1;
    }
    let bits = match *parts.get(i)? {
        "b8" => 8,
        "b16" => 16,
        "b32" => 32,
        "b64" => 64,
        _ => return None,
    };
    i += 1;
    let clamp = match parts.get(i) {
        None | Some(&"trap") | Some(&"zero") => false,
        Some(&"clamp") => true,
        _ => return None,
    };
    if parts.len() > i + 1 || inst.operands.len() != 2 {
        return None;
    }
    let (addr, data) = if store { (inst.operand(0), inst.operand(1)) } else { (inst.operand(1), inst.operand(0)) };
    let regs = if data.trim().starts_with('{') { braced(data) } else { vec![data.trim().to_owned()] };
    if regs.len() != count {
        return None;
    }
    let (handle, coords) = handle_and_coords(addr)?;
    if coords.len() < dims {
        return None;
    }
    Some(Surf { store, dims, bits, regs, handle, coords, clamp })
}

/// `txq.<query>.b32 %r, [%h]` / `suq.<query>.b32 %r, [%h]`.
fn parse_query(inst: &Inst) -> Option<(String, String, String)> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    if parts.len() != 3 || parts[2] != "b32" || inst.operands.len() != 2 {
        return None;
    }
    let query = parts[1];
    let known = match parts[0] {
        "txq" => matches!(
            query,
            "width" | "height" | "depth" | "channel_data_type" | "channel_order" | "normalized_coords" | "filter_mode"
                | "addr_mode_0" | "addr_mode_1" | "addr_mode_2" | "array_size" | "num_mipmap_levels" | "num_samples"
        ),
        _ => matches!(query, "width" | "height" | "depth" | "channel_data_type" | "channel_order" | "array_size" | "memory_layout"),
    };
    let (handle, coords) = handle_and_coords(inst.operand(1))?;
    (known && coords.is_empty() && inst.operand(0).starts_with('%')).then(|| (query.to_owned(), inst.operand(0).to_owned(), handle))
}

/// Whether `inst` is a texture/surface form this module lowers (`check_unsupported` asks).
pub(super) fn supported(inst: &Inst) -> bool {
    match inst.op.split('.').next().unwrap_or("") {
        "tex" => parse_tex(inst).is_some(),
        "suld" | "sust" => parse_surf(inst).is_some(),
        "txq" | "suq" => parse_query(inst).is_some(),
        _ => false,
    }
}

/// Emission helpers over the tier-3 scratch allocator.
struct G<'a> {
    b: B<'a>,
    handle: String,
    fields: Vec<(u32, String)>,
}

impl G<'_> {
    fn p(&mut self, line: String) {
        self.b.p(line);
    }
    fn field32(&mut self, off: u32) -> String {
        if let Some((_, r)) = self.fields.iter().find(|(o, _)| *o == off) {
            return r.clone();
        }
        let t = self.b.t32();
        self.p(format!("ld.global.u32 {t} , [{}+{off}] ;", self.handle));
        self.fields.push((off, t.clone()));
        t
    }
    fn field64(&mut self, off: u32) -> String {
        if let Some((_, r)) = self.fields.iter().find(|(o, _)| *o == off) {
            return r.clone();
        }
        let t = self.b.t64();
        self.p(format!("ld.global.u64 {t} , [{}+{off}] ;", self.handle));
        self.fields.push((off, t.clone()));
        t
    }
    fn op(&mut self, op: &str, args: &[&str]) -> String {
        // The destination type is the last suffix, or the first of the two types of a `cvt`.
        let parts: Vec<&str> = op.split('.').collect();
        let ty = if parts[0] == "cvt" { parts[parts.len() - 2] } else { parts[parts.len() - 1] };
        let t = if ty.ends_with("64") { self.b.t64() } else { self.b.t32() };
        self.p(format!("{op} {t} , {} ;", args.join(" , ")));
        t
    }
    fn setp(&mut self, cmp: &str, a: &str, b: &str) -> String {
        let t = self.b.tp();
        self.p(format!("setp.{cmp} {t} , {a} , {b} ;"));
        t
    }
    fn pred(&mut self, op: &str, a: &str, b: &str) -> String {
        let t = self.b.tp();
        self.p(format!("{op}.pred {t} , {a} , {b} ;"));
        t
    }
    fn sel(&mut self, a: &str, b: &str, p: &str) -> String {
        let t = self.b.t32();
        self.p(format!("selp.b32 {t} , {a} , {b} , {p} ;"));
        t
    }
    fn sel64(&mut self, a: &str, b: &str, p: &str) -> String {
        let t = self.b.t64();
        self.p(format!("selp.b64 {t} , {a} , {b} , {p} ;"));
        t
    }
    /// Value of a 32-bit integer `x == k` test.
    fn is(&mut self, x: &str, k: u32) -> String {
        self.setp("eq.u32", x, &k.to_string())
    }
}

/// Runs `body` with branch-around for a guarded instruction (the handle may be garbage when the
/// guard is false, so even the descriptor loads must not run).
fn with_guard(st: &mut State, inst: &Inst, handle: &str, body: impl FnOnce(&mut G)) -> Vec<String> {
    let label = if inst.guard.is_empty() {
        None
    } else {
        st.labels += 1;
        Some(format!("$L__t3_tex{}", st.labels))
    };
    let mut g = G { b: B::new(st), handle: handle.to_owned(), fields: Vec::new() };
    if let Some(label) = &label {
        let inverse = match inst.guard.strip_prefix("@ ! ") {
            Some(p) => format!("@ {p}"),
            None => format!("@ ! {}", inst.guard.trim_start_matches("@ ")),
        };
        g.p(format!("{inverse} bra {label} ;"));
    }
    body(&mut g);
    let mut lines = g.b.finish();
    if let Some(label) = label {
        lines.push(format!("{label} :"));
    }
    lines
}

/// One axis of a texel index: the in-range index to load and whether the texel is outside
/// the resource under border semantics.
fn fix_index(g: &mut G, i: &str, size: &str, mode: &str, linear_res: &str) -> (String, String) {
    let wm1 = g.op("sub.s32", &[size, "1"]);
    let c = g.op("max.s32", &[i, "0"]);
    let c = g.op("min.s32", &[&c, &wm1]);
    // wrap: i mod size, made non-negative
    let r = g.op("rem.s32", &[i, size]);
    let neg = g.setp("lt.s32", &r, "0");
    let r2 = g.op("add.s32", &[&r, size]);
    let r = g.sel(&r2, &r, &neg);
    // mirror: reflect once across each edge, then clamp
    let below = g.setp("lt.s32", i, "0");
    let flipped = g.op("not.b32", &[i]);
    let m = g.sel(&flipped, i, &below);
    let twice = g.op("shl.b32", &[size, "1"]);
    let far = g.op("sub.s32", &[&twice, "1"]);
    let far = g.op("sub.s32", &[&far, &m]);
    let above = g.setp("ge.s32", &m, size);
    let m = g.sel(&far, &m, &above);
    let m = g.op("max.s32", &[&m, "0"]);
    let m = g.op("min.s32", &[&m, &wm1]);
    let is_wrap = g.is(mode, 0);
    let is_mirror = g.is(mode, 2);
    let idx = g.sel(&r, &c, &is_wrap);
    let idx = g.sel(&m, &idx, &is_mirror);
    let hi = g.setp("ge.s32", i, size);
    let outside = g.pred("or", &below, &hi);
    let border = g.is(mode, 3);
    let lin = g.setp("ne.u32", linear_res, "0");
    let border = g.pred("or", &border, &lin);
    let oob = g.pred("and", &outside, &border);
    (idx, oob)
}

/// The four 32-bit words of the element at `addr` (zero above the element size).
fn fetch_words(g: &mut G, addr: &str, sizes: &[String; 5]) -> [String; 4] {
    let w: [String; 4] = std::array::from_fn(|_| g.b.t32());
    for r in &w {
        g.p(format!("mov.b32 {r} , 0 ;"));
    }
    let [p1, p2, p4, p8, p16] = sizes;
    g.p(format!("@ {p1} ld.global.u8 {} , [{addr}] ;", w[0]));
    g.p(format!("@ {p2} ld.global.u16 {} , [{addr}] ;", w[0]));
    g.p(format!("@ {p4} ld.global.u32 {} , [{addr}] ;", w[0]));
    g.p(format!("@ {p8} ld.global.u32 {} , [{addr}] ;", w[0]));
    g.p(format!("@ {p8} ld.global.u32 {} , [{addr}+4] ;", w[1]));
    // p8 means "8 or 16 bytes"; the upper half needs 16.
    g.p(format!("@ {p16} ld.global.u32 {} , [{addr}+8] ;", w[2]));
    g.p(format!("@ {p16} ld.global.u32 {} , [{addr}+12] ;", w[3]));
    w
}

/// Raw bits of channel `c` (zero-extended; zero for channels the format lacks).
fn channel(g: &mut G, w: &[String; 4], c: usize, cb: &str, nch: &str) -> String {
    let v1 = g.op("shr.u32", &[&w[0], &(8 * c).to_string()]);
    let v1 = g.op("and.b32", &[&v1, "255"]);
    let v2 = g.op("shr.u32", &[&w[c / 2], &(16 * (c % 2)).to_string()]);
    let v2 = g.op("and.b32", &[&v2, "65535"]);
    let q1 = g.is(cb, 1);
    let q2 = g.is(cb, 2);
    let raw = g.sel(&v2, &w[c], &q2);
    let raw = g.sel(&v1, &raw, &q1);
    let present = g.setp("gt.u32", nch, &c.to_string());
    g.sel(&raw, "0", &present)
}

/// Per-instruction conversion constants derived from the kind and the read mode.
struct Conv {
    signed8: String,
    signed16: String,
    signed: String,
    normalize: String,
    is_half: String,
    is_f32: String,
    denominator: String,
}

fn conversion(g: &mut G, kind: &str, flags: &str) -> Conv {
    let signed8 = g.is(kind, d::Kind::S8 as u32);
    let signed16 = g.is(kind, d::Kind::S16 as u32);
    let s32 = g.is(kind, d::Kind::S32 as u32);
    let signed = g.pred("or", &signed8, &signed16);
    let signed = g.pred("or", &signed, &s32);
    let is_half = g.is(kind, d::Kind::F16 as u32);
    let is_f32 = g.is(kind, d::Kind::F32 as u32);
    // 8- and 16-bit integer channels read as normalised floats unless READ_AS_INTEGER.
    let narrow = g.setp("lt.u32", kind, "2"); // U8, U16
    let narrow_s = g.pred("or", &signed8, &signed16);
    let narrow = g.pred("or", &narrow, &narrow_s);
    let ri = g.op("and.b32", &[flags, &d::READ_AS_INTEGER.to_string()]);
    let as_float = g.setp("eq.u32", &ri, "0");
    let normalize = g.pred("and", &narrow, &as_float);
    let u8_ = g.is(kind, d::Kind::U8 as u32);
    let u16_ = g.is(kind, d::Kind::U16 as u32);
    // 255, 65535, 127, 32767 as f32
    // selp.b32 takes the f32 bit patterns of 255, 1, 65535, 127 and 32767.
    let den = g.sel("1132396544", "1065353216", &u8_);
    let den = g.sel("1199570688", &den, &u16_);
    let den = g.sel("1123942400", &den, &signed8);
    let den = g.sel("1191181824", &den, &signed16);
    Conv { signed8, signed16, signed, normalize, is_half, is_f32, denominator: den }
}

/// Sign-extends 8- and 16-bit signed channels.
fn extend(g: &mut G, raw: &str, cv: &Conv) -> String {
    let s8 = g.op("shl.b32", &[raw, "24"]);
    let s8 = g.op("shr.s32", &[&s8, "24"]);
    let s16 = g.op("shl.b32", &[raw, "16"]);
    let s16 = g.op("shr.s32", &[&s16, "16"]);
    let x = g.sel(&s16, raw, &cv.signed16);
    g.sel(&s8, &x, &cv.signed8)
}

/// IEEE half bits -> f32 bits, with integer arithmetic (exact).
fn half_to_float(g: &mut G, h: &str) -> String {
    let sign = g.op("and.b32", &[h, "32768"]);
    let sign = g.op("shl.b32", &[&sign, "16"]);
    let e = g.op("shr.u32", &[h, "10"]);
    let e = g.op("and.b32", &[&e, "31"]);
    let m = g.op("and.b32", &[h, "1023"]);
    let mbits = g.op("shl.b32", &[&m, "13"]);
    let ne = g.op("add.u32", &[&e, "112"]);
    let ne = g.op("shl.b32", &[&ne, "23"]);
    let normal = g.op("or.b32", &[&ne, &mbits]);
    let special = g.op("or.b32", &[&mbits, "2139095040"]); // 0x7f800000
    let is_special = g.is(&e, 31);
    let v = g.sel(&special, &normal, &is_special);
    // subnormal: m * 2^-24
    let mf = g.op("cvt.rn.f32.u32", &[&m]);
    let sub = g.op("mul.f32", &[&mf, "0f33800000"]);
    let is_sub = g.is(&e, 0);
    let v = g.sel(&sub, &v, &is_sub);
    g.op("or.b32", &[&v, &sign])
}

/// Channel bits -> f32 result (float destinations).
fn to_float(g: &mut G, raw: &str, cv: &Conv) -> String {
    let x = extend(g, raw, cv);
    let fs = g.op("cvt.rn.f32.s32", &[&x]);
    let fu = g.op("cvt.rn.f32.u32", &[&x]);
    let f = g.sel(&fs, &fu, &cv.signed);
    let n = g.op("div.rn.f32", &[&f, &cv.denominator]);
    let n = g.op("max.f32", &[&n, F_MINUS_ONE]);
    let v = g.sel(&n, &f, &cv.normalize);
    let h = half_to_float(g, raw);
    let v = g.sel(&h, &v, &cv.is_half);
    g.sel(raw, &v, &cv.is_f32)
}

/// Channel bits -> integer result (s32/u32 destinations): sign-extended for signed kinds.
fn to_int(g: &mut G, raw: &str, cv: &Conv) -> String {
    extend(g, raw, cv)
}

fn lerp(g: &mut G, a: &str, b: &str, w: &str) -> String {
    let om = g.op("sub.f32", &[F_ONE, w]);
    let x = g.op("mul.f32", &[a, &om]);
    g.op("fma.rn.f32", &[b, w, &x])
}

fn tex(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let t = parse_tex(inst)?;
    Some(with_guard(st, inst, &t.handle, |g| {
        let base = g.field64(d::BASE);
        let sizes: Vec<String> = [d::WIDTH, d::HEIGHT, d::DEPTH][..t.dims].iter().map(|&o| g.field32(o)).collect();
        let kind = g.field32(d::KIND);
        let nch = g.field32(d::CHANNELS);
        let esz = g.field32(d::ELEMENT_BYTES);
        let cb = g.field32(d::CHANNEL_BYTES);
        let flags = g.field32(d::FLAGS);
        let linear_res = g.field32(d::LINEAR);
        let modes: Vec<String> = (0..t.dims).map(|k| g.field32(d::ADDRESS_MODE + 4 * k as u32)).collect();
        let normalized = g.op("and.b32", &[&flags, &d::NORMALIZED_COORDINATES.to_string()]);
        let pn = g.setp("ne.u32", &normalized, "0");
        // Wrap and mirror need normalised coordinates; CUDA clamps otherwise.
        let eff: Vec<String> = modes
            .iter()
            .map(|m| {
                let border = g.is(m, 3);
                let other = g.sel("3", "1", &border);
                g.sel(m, &other, &pn)
            })
            .collect();
        let filtering = t.float_out && t.float_coords;
        let lin = if filtering {
            let f = g.field32(d::FILTER);
            Some(g.is(&f, 1))
        } else {
            None
        };
        // Per axis: base index i0 and weight a (linear only).
        let mut i0 = Vec::new();
        let mut weight = Vec::new();
        for k in 0..t.dims {
            let x = t.coords[k].clone();
            if !t.float_coords {
                i0.push(x);
                continue;
            }
            let sf = g.op("cvt.rn.f32.u32", &[&sizes[k]]);
            let fl = g.op("cvt.rmi.f32.f32", &[&x]);
            let fr = g.op("sub.f32", &[&x, &fl]);
            let fli = g.op("cvt.rmi.s32.f32", &[&x]);
            let odd = g.op("and.b32", &[&fli, "1"]);
            let podd = g.setp("ne.u32", &odd, "0");
            let mir = g.op("sub.f32", &[F_ONE, &fr]);
            let mir = g.sel(&mir, &fr, &podd);
            let pw = g.is(&modes[k], 0);
            let pm = g.is(&modes[k], 2);
            let xn = g.sel(&fr, &x, &pw);
            let xn = g.sel(&mir, &xn, &pm);
            let xs = g.op("mul.f32", &[&xn, &sf]);
            let big_x = g.sel(&xs, &x, &pn);
            let xb = match &lin {
                Some(lin) => {
                    let shifted = g.op("sub.f32", &[&big_x, F_HALF]);
                    g.sel(&shifted, &big_x, lin)
                }
                None => big_x,
            };
            i0.push(g.op("cvt.rmi.s32.f32", &[&xb]));
            if lin.is_some() {
                let f0 = g.op("cvt.rmi.f32.f32", &[&xb]);
                let a = g.op("sub.f32", &[&xb, &f0]);
                let a = g.op("mul.f32", &[&a, "0f43800000"]); // 256
                let a = g.op("cvt.rni.f32.f32", &[&a]);
                weight.push(g.op("mul.f32", &[&a, "0f3B800000"])); // 1/256
            }
        }
        let pitch = if t.dims >= 2 { Some(g.field64(d::PITCH)) } else { None };
        let slice = if t.dims >= 3 { Some(g.field64(d::SLICE)) } else { None };
        let e1 = g.is(&esz, 1);
        let e2 = g.is(&esz, 2);
        let e4 = g.is(&esz, 4);
        let e8 = g.setp("ge.u32", &esz, "8");
        let e16 = g.is(&esz, 16);
        let size_preds = [e1, e2, e4, e8, e16];
        let cv = conversion(g, &kind, &flags);
        let border: Vec<String> = if t.float_out { (0..4).map(|c| g.field32(d::BORDER + 4 * c)).collect() } else { vec!["0".into(); 4] };
        let esz64 = g.op("cvt.u64.u32", &[&esz]);
        let corners = if filtering { 1usize << t.dims } else { 1 };
        let mut texels: Vec<[String; 4]> = Vec::new();
        for corner in 0..corners {
            let mut addr = base.clone();
            let mut oob: Option<String> = None;
            for k in 0..t.dims {
                let i = if corner >> k & 1 == 1 { g.op("add.s32", &[&i0[k], "1"]) } else { i0[k].clone() };
                let (idx, out) = fix_index(g, &i, &sizes[k], &eff[k], &linear_res);
                oob = Some(match oob {
                    Some(prev) => g.pred("or", &prev, &out),
                    None => out,
                });
                let idx64 = g.op("cvt.u64.u32", &[&idx]);
                let stride = match k {
                    0 => esz64.clone(),
                    1 => pitch.clone().unwrap(),
                    _ => slice.clone().unwrap(),
                };
                let off = g.op("mul.lo.u64", &[&idx64, &stride]);
                addr = g.op("add.u64", &[&addr, &off]);
            }
            let oob = oob.unwrap();
            let words = fetch_words(g, &addr, &size_preds);
            let values: [String; 4] = std::array::from_fn(|c| {
                let raw = channel(g, &words, c, &cb, &nch);
                let v = if t.float_out { to_float(g, &raw, &cv) } else { to_int(g, &raw, &cv) };
                g.sel(&border[c], &v, &oob)
            });
            texels.push(values);
        }
        let result: [String; 4] = std::array::from_fn(|c| {
            if corners == 1 {
                return texels[0][c].clone();
            }
            // Reduce along x, then y, then z: corner index bit k is axis k.
            let mut layer: Vec<String> = texels.iter().map(|t| t[c].clone()).collect();
            for w in &weight {
                layer = layer.chunks(2).map(|pair| lerp(g, &pair[0], &pair[1], w)).collect();
            }
            g.sel(&layer[0], &texels[0][c], lin.as_ref().unwrap())
        });
        for (dst, v) in t.dst.iter().zip(&result) {
            g.p(format!("mov.b32 {dst} , {v} ;"));
        }
    }))
}

fn surf(st: &mut State, inst: &Inst, widths: &dyn Fn(&str) -> Option<u32>) -> Option<Vec<String>> {
    let s = parse_surf(inst)?;
    // A 64-bit component needs 64-bit registers, and the others 32-bit (16-bit registers live
    // in 32-bit slots).
    if s.regs.iter().any(|r| r.starts_with('%') && widths(r).is_some_and(|w| (w == 64) != (s.bits == 64))) {
        return None;
    }
    let bytes = s.bits / 8;
    let size = bytes * s.regs.len() as u32;
    Some(with_guard(st, inst, &s.handle, |g| {
        let base = g.field64(d::BASE);
        let esz = g.field32(d::ELEMENT_BYTES);
        let width = g.field32(d::WIDTH);
        let row = g.op("mul.lo.u32", &[&width, &esz]);
        let x_max = g.op("sub.s32", &[&row, &size.to_string()]);
        let mut oob: Option<String> = None;
        let mut addr = base;
        for k in 0..s.dims {
            let c = s.coords[k].clone();
            let (limit, stride) = match k {
                0 => (x_max.clone(), None),
                1 => {
                    let h = g.field32(d::HEIGHT);
                    (g.op("sub.s32", &[&h, "1"]), Some(g.field64(d::PITCH)))
                }
                _ => {
                    let dd = g.field32(d::DEPTH);
                    (g.op("sub.s32", &[&dd, "1"]), Some(g.field64(d::SLICE)))
                }
            };
            let lo = g.setp("lt.s32", &c, "0");
            let hi = g.setp("gt.s32", &c, &limit);
            let out = g.pred("or", &lo, &hi);
            oob = Some(match oob {
                Some(prev) => g.pred("or", &prev, &out),
                None => out,
            });
            let cl = g.op("max.s32", &[&c, "0"]);
            let cl = g.op("min.s32", &[&cl, &limit]);
            let c64 = g.op("cvt.u64.u32", &[&cl]);
            let off = match stride {
                None => c64,
                Some(stride) => g.op("mul.lo.u64", &[&c64, &stride]),
            };
            addr = g.op("add.u64", &[&addr, &off]);
        }
        // .clamp accesses the nearest element; .zero (and .trap, which has no trap here) drops
        // out-of-range stores and reads zero.
        let oob = if s.clamp { None } else { oob };
        let ty = format!("u{}", s.bits);
        for (i, r) in s.regs.iter().enumerate() {
            let off = i as u32 * bytes;
            let at = if off == 0 { format!("[{addr}]") } else { format!("[{addr}+{off}]") };
            if s.store {
                let guard = oob.as_ref().map(|p| format!("@ ! {p} ")).unwrap_or_default();
                g.p(format!("{guard}st.global.{ty} {at} , {r} ;"));
            } else if r.starts_with('%') {
                let t = if s.bits == 64 { g.b.t64() } else { g.b.t32() };
                g.p(format!("ld.global.{ty} {t} , {at} ;"));
                match &oob {
                    Some(p) if s.bits == 64 => {
                        let v = g.sel64("0", &t, p);
                        g.p(format!("mov.b64 {r} , {v} ;"));
                    }
                    Some(p) => {
                        let v = g.sel("0", &t, p);
                        g.p(format!("mov.b32 {r} , {v} ;"));
                    }
                    None => g.p(format!("mov.b{} {r} , {t} ;", if s.bits == 64 { 64 } else { 32 })),
                }
            }
        }
    }))
}

fn query(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let (q, dst, handle) = parse_query(inst)?;
    Some(with_guard(st, inst, &handle, |g| {
        let v = match q.as_str() {
            "width" => g.field32(d::WIDTH),
            "height" => g.field32(d::HEIGHT),
            "depth" => g.field32(d::DEPTH),
            "channel_data_type" => g.field32(d::CU_FORMAT),
            "channel_order" => g.field32(d::CHANNELS),
            "filter_mode" => g.field32(d::FILTER),
            "addr_mode_0" => g.field32(d::ADDRESS_MODE),
            "addr_mode_1" => g.field32(d::ADDRESS_MODE + 4),
            "addr_mode_2" => g.field32(d::ADDRESS_MODE + 8),
            "normalized_coords" => {
                let f = g.field32(d::FLAGS);
                let n = g.op("shr.u32", &[&f, "1"]);
                g.op("and.b32", &[&n, "1"])
            }
            "num_mipmap_levels" => "1".into(),
            // array_size, num_samples, memory_layout: no layered, multisample or block-linear resources.
            _ => "0".into(),
        };
        g.p(format!("mov.b32 {dst} , {v} ;"));
    }))
}

/// `widths` gives the declared width of the instruction's registers (`Ctx::width_of`).
pub(super) fn rewrite(st: &mut State, inst: &Inst, widths: &dyn Fn(&str) -> Option<u32>) -> Option<Vec<String>> {
    match inst.op.split('.').next().unwrap_or("") {
        "tex" => tex(st, inst),
        "suld" | "sust" => surf(st, inst, widths),
        "txq" | "suq" => query(st, inst),
        _ => None,
    }
}

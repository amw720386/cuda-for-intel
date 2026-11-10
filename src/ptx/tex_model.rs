//! Host model of CUDA texture sampling and the test kernels, shared by the simulator tests
//! (`tex_tests.rs`) and the hardware tests (`tests/gpu/hw_texture.rs`). Independent of the
//! translator's lowering (`tier3_tex.rs`): it is the specification both are checked against.
use super::texdesc::{self, Desc, Kind};

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Host model of one texture: descriptor plus texel bytes (row pitch and slice pitch as in the
/// descriptor, base address 0).
pub struct Model {
    pub desc: Desc,
    pub data: Vec<u8>,
}

fn half_to_f32(h: u32) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = (h >> 10) & 31;
    let m = h & 1023;
    match e {
        0 => s * m as f32 * 2f32.powi(-24),
        31 => {
            if m == 0 {
                s * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => s * (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

impl Model {
    pub fn channel_raw(&self, x: usize, y: usize, z: usize, c: usize) -> u32 {
        let d = &self.desc;
        if c as u32 >= d.channels {
            return 0;
        }
        let cb = d.channel_bytes() as usize;
        let at = z * d.slice as usize + y * d.pitch as usize + x * d.element_bytes() as usize + c * cb;
        let mut v = 0u32;
        for i in 0..cb {
            v |= u32::from(self.data[at + i]) << (8 * i);
        }
        v
    }

    /// The value CUDA returns for one channel of one in-range texel.
    fn convert(&self, raw: u32, float_out: bool) -> u32 {
        let d = &self.desc;
        let kind = d.kind;
        let signed = |bits: u32| ((raw << (32 - bits)) as i32) >> (32 - bits);
        let int_value: i64 = match kind {
            3 => i64::from(signed(8)),
            4 => i64::from(signed(16)),
            5 => i64::from(raw as i32),
            _ => i64::from(raw),
        };
        if !float_out {
            return int_value as u32;
        }
        let as_int = d.flags & texdesc::READ_AS_INTEGER != 0;
        let f = match kind {
            7 => f32::from_bits(raw),
            6 => half_to_f32(raw),
            0 | 1 | 3 | 4 if !as_int => {
                let den = [255.0f32, 65535.0, 0.0, 127.0, 32767.0][kind as usize];
                (int_value as f32 / den).max(-1.0)
            }
            3..=5 => int_value as f32,
            _ => raw as f32,
        };
        f.to_bits()
    }

    /// In-range index and out-of-range flag along one axis (CUDA address modes).
    fn index(&self, i: i64, size: i64, mode: u32) -> (usize, bool) {
        let outside = i < 0 || i >= size;
        let idx = match mode {
            0 => i.rem_euclid(size),
            2 => {
                let m = if i < 0 { -i - 1 } else { i };
                let m = if m >= size { 2 * size - 1 - m } else { m };
                m.clamp(0, size - 1)
            }
            _ => i.clamp(0, size - 1),
        };
        (idx as usize, outside && (mode == 3 || self.desc.linear))
    }

    /// `tex.{dims}d.v4.{f32|s32}.{f32|s32}` at `coords`.
    pub fn sample(&self, dims: usize, coords: [u32; 3], float_coords: bool, float_out: bool) -> [u32; 4] {
        let d = &self.desc;
        let sizes = [d.width as i64, d.height as i64, d.depth as i64];
        let normalized = d.flags & texdesc::NORMALIZED_COORDINATES != 0;
        let mode = |k: usize| {
            let m = d.address_mode[k];
            if normalized { m } else if m == 3 { 3 } else { 1 }
        };
        let linear = float_out && float_coords && d.filter == 1;
        let mut i0 = [0i64; 3];
        let mut w = [0f32; 3];
        for k in 0..dims {
            if !float_coords {
                i0[k] = coords[k] as i32 as i64;
                continue;
            }
            let x = f32::from_bits(coords[k]);
            let fr = x - x.floor();
            let am = d.address_mode[k];
            let mut xn = x;
            if am == 0 {
                xn = fr;
            }
            if am == 2 {
                xn = if (x.floor() as i32) & 1 != 0 { 1.0 - fr } else { fr };
            }
            let big = if normalized { xn * sizes[k] as f32 } else { x };
            let xb = if linear { big - 0.5 } else { big };
            i0[k] = xb.floor() as i32 as i64;
            w[k] = ((xb - xb.floor()) * 256.0).round_ties_even() * (1.0 / 256.0);
        }
        let corners = if linear { 1 << dims } else { 1 };
        let mut texels = Vec::new();
        for corner in 0..corners {
            let mut idx = [0usize; 3];
            let mut oob = false;
            for k in 0..dims {
                let (i, o) = self.index(i0[k] + (corner >> k & 1) as i64, sizes[k], mode(k));
                idx[k] = i;
                oob |= o;
            }
            let v: [u32; 4] = std::array::from_fn(|c| {
                if oob {
                    if float_out { d.border[c].to_bits() } else { 0 }
                } else {
                    self.convert(self.channel_raw(idx[0], idx[1], idx[2], c), float_out)
                }
            });
            texels.push(v);
        }
        if !linear {
            return texels[0];
        }
        std::array::from_fn(|c| {
            let mut layer: Vec<f32> = texels.iter().map(|t| f32::from_bits(t[c])).collect();
            for wk in &w[..dims] {
                layer = layer.chunks(2).map(|p| p[1].mul_add(*wk, p[0] * (1.0 - wk))).collect();
            }
            layer[0].to_bits()
        })
    }
}

pub fn tex_kernel(dims: usize, float_coords: bool, float_out: bool) -> String {
    let cty = if float_coords { "f32" } else { "s32" };
    let dty = if float_out { "f32" } else { "s32" };
    let c: Vec<&str> = if float_coords { vec!["%f1", "%f2", "%f3", "%f4"] } else { vec!["%r1", "%r2", "%r3", "%r4"] };
    let coord_list = match dims {
        1 => c[0].to_string(),
        2 => format!("{}, {}", c[0], c[1]),
        _ => format!("{}, {}, {}, {}", c[0], c[1], c[2], c[3]),
    };
    let ld = if float_coords { "f32" } else { "s32" };
    let d: Vec<&str> = if float_out { vec!["%f5", "%f6", "%f7", "%f8"] } else { vec!["%r5", "%r6", "%r7", "%r8"] };
    format!(
        ".version 7.0\n.target sm_80\n.address_size 64\n\
.visible .entry tk(.param .u64 tex, .param .u64 coords, .param .u64 out)\n{{\n\
.reg .b32 %r<10>;\n.reg .f32 %f<10>;\n.reg .b64 %rd<10>;\n\
ld.param.u64 %rd1, [tex];\nld.param.u64 %rd2, [coords];\nld.param.u64 %rd3, [out];\n\
mov.u32 %r9, %tid.x;\nmul.wide.u32 %rd4, %r9, 16;\nadd.s64 %rd5, %rd2, %rd4;\nadd.s64 %rd6, %rd3, %rd4;\n\
ld.global.{ld} {c0}, [%rd5];\nld.global.{ld} {c1}, [%rd5+4];\nld.global.{ld} {c2}, [%rd5+8];\nmov.b32 {c3}, {c2};\n\
tex.{dims}d.v4.{dty}.{cty} {{{d0}, {d1}, {d2}, {d3}}}, [%rd1, {{{coord_list}}}];\n\
st.global.{dty} [%rd6], {d0};\nst.global.{dty} [%rd6+4], {d1};\nst.global.{dty} [%rd6+8], {d2};\nst.global.{dty} [%rd6+12], {d3};\n\
ret;\n}}\n",
        c0 = c[0],
        c1 = c[1],
        c2 = c[2],
        c3 = c[3],
        d0 = d[0],
        d1 = d[1],
        d2 = d[2],
        d3 = d[3],
    )
}

pub const KINDS: [(Kind, u32); 8] = [
    (Kind::U8, 0x01),
    (Kind::U16, 0x02),
    (Kind::U32, 0x03),
    (Kind::S8, 0x08),
    (Kind::S16, 0x09),
    (Kind::S32, 0x0a),
    (Kind::F16, 0x10),
    (Kind::F32, 0x20),
];

/// A random texture of `dims` dimensions; rows padded beyond the element width.
pub fn random_model(rng: &mut Rng, dims: usize, kind: Kind, cu: u32, channels: u32) -> Model {
    random_model_with(rng, dims, kind, cu, channels, false)
}

/// `dense`: rows exactly `width * element bytes` apart (the layout of a CUDA array here).
pub fn random_model_with(rng: &mut Rng, dims: usize, kind: Kind, cu: u32, channels: u32, dense: bool) -> Model {
    let width = 1 + rng.below(9) as u32;
    let height = if dims >= 2 { 1 + rng.below(6) as u32 } else { 1 };
    let depth = if dims >= 3 { 1 + rng.below(4) as u32 } else { 1 };
    let mut desc = Desc {
        width,
        height,
        depth,
        kind: kind as u32,
        channels,
        dims: dims as u32,
        cu_format: cu,
        resource_type: 0,
        ..Desc::default()
    };
    let esz = desc.element_bytes() as u64;
    desc.pitch = if dense { u64::from(width) * esz } else { (u64::from(width) * esz).next_multiple_of(16) + 16 * rng.below(2) };
    desc.slice = desc.pitch * u64::from(height);
    let bytes = (desc.slice * u64::from(depth)) as usize;
    let mut data = vec![0u8; bytes];
    for (i, b) in data.iter_mut().enumerate() {
        *b = rng.next() as u8;
        // Keep float channels finite and moderate: clear high exponent bits.
        let cb = desc.channel_bytes() as usize;
        if kind == Kind::F32 && i % 4 == 3 {
            *b = (*b & 0x80) | 0x3f;
        }
        if kind == Kind::F16 && cb == 2 && i % 2 == 1 {
            *b = (*b & 0x83) | 0x38;
        }
    }
    desc.border = [rng.unit(), -rng.unit(), 2.0, 0.25];
    Model { desc, data }
}


/// Random coordinates: in and around the resource, with exact texel centres and edges.
pub fn coords_for(rng: &mut Rng, m: &Model, dims: usize, float_coords: bool, n: usize) -> Vec<[u32; 3]> {
    let sizes = [m.desc.width, m.desc.height, m.desc.depth];
    let normalized = m.desc.flags & texdesc::NORMALIZED_COORDINATES != 0;
    (0..n)
        .map(|_| {
            std::array::from_fn(|k| {
                if k >= dims {
                    return 0;
                }
                let s = sizes[k] as f32;
                if float_coords {
                    let span = if normalized { 4.0 } else { 4.0 * s };
                    let v = rng.unit() * span - span * 0.375;
                    // Also hit exact texel centres and edges.
                    let v = match rng.below(4) {
                        0 => (v * 2.0).round() / 2.0,
                        _ => v,
                    };
                    v.to_bits()
                } else {
                    (rng.below(3 * sizes[k] as u64) as i32 - sizes[k] as i32) as u32
                }
            })
        })
        .collect()
}


/// Thread 0 of the kernel loads from surface A at coords[0] into out and stores vals[0] into B
/// at the same coordinates (`suld.b` / `sust.b` with `.zero` or `.clamp`).
pub fn surf_kernel(dims: usize, bits: u32, count: usize, clamp: bool) -> String {
    let mode = if clamp { "clamp" } else { "zero" };
    let vec = match count {
        1 => String::new(),
        n => format!(".v{n}"),
    };
    let (reg, decl) = if bits == 64 { ("%rd", "") } else { ("%r", "") };
    let _ = decl;
    let regs: Vec<String> = (0..count).map(|i| format!("{reg}{}", 20 + i)).collect();
    let data = if count == 1 { regs[0].clone() } else { format!("{{{}}}", regs.join(", ")) };
    let coords = match dims {
        1 => "%r1".to_string(),
        2 => "%r1, %r2".to_string(),
        _ => "%r1, %r2, %r3, %r3".to_string(),
    };
    let elem = if bits == 64 { 8 } else { 4 };
    let mut loads = String::new();
    let mut stores = String::new();
    for (i, r) in regs.iter().enumerate() {
        let w = if bits == 64 { "u64" } else { "u32" };
        loads.push_str(&format!("ld.global.{w} {r}, [%rd7+{}];\n", i * elem));
        stores.push_str(&format!("st.global.{w} [%rd8+{}], {r};\n", i * elem));
    }
    // Thread i: load from surface A at coords[i], store to out[i]; then store vals[i] into B.
    format!(
        ".version 7.0\n.target sm_80\n.address_size 64\n\
.visible .entry sk(.param .u64 a, .param .u64 b, .param .u64 coords, .param .u64 vals, .param .u64 out)\n{{\n\
.reg .b32 %r<40>;\n.reg .b64 %rd<40>;\n\
ld.param.u64 %rd1, [a];\nld.param.u64 %rd2, [b];\nld.param.u64 %rd3, [coords];\nld.param.u64 %rd4, [vals];\nld.param.u64 %rd5, [out];\n\
mov.u32 %r9, %tid.x;\nmul.wide.u32 %rd6, %r9, 16;\nadd.s64 %rd6, %rd3, %rd6;\n\
ld.global.s32 %r1, [%rd6];\nld.global.s32 %r2, [%rd6+4];\nld.global.s32 %r3, [%rd6+8];\n\
mul.wide.u32 %rd9, %r9, 32;\nadd.s64 %rd7, %rd4, %rd9;\nadd.s64 %rd8, %rd5, %rd9;\n\
suld.b.{dims}d{vec}.b{bits}.{mode} {data}, [%rd1, {{{coords}}}];\n{stores}\
{loads}sust.b.{dims}d{vec}.b{bits}.{mode} [%rd2, {{{coords}}}], {data};\nret;\n}}\n"
    )
}


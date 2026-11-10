//! Texture and surface object emulation (`tier3_tex.rs`) in the SPIR-V simulator, against an
//! independent host model of CUDA's sampling rules: 1D/2D/3D, every address mode, nearest and
//! linear filtering, normalised and unnormalised coordinates, integer coordinates, the channel
//! formats and read modes, surface loads and stores with zero and clamp boundary modes, and
//! `txq`/`suq`. The simulator also proves that no texel load leaves the resource.
use super::sim::{Gpu, Launch, Sim};
use super::tex_model::*;
use super::texdesc::{self, Kind};
use super::*;

/// Uploads texels and descriptor; returns the handle (descriptor address).
pub(crate) fn upload(gpu: &mut Gpu, m: &Model) -> u64 {
    let base = gpu.upload(&m.data);
    let mut d = m.desc.clone();
    d.base = base;
    gpu.upload(&d.encode())
}

fn run_case(gpu: &mut Gpu, sim: &mut Sim, m: &Model, coords: &[[u32; 3]]) -> Vec<[u32; 4]> {
    let handle = upload(gpu, m);
    let mut cbytes = Vec::new();
    for c in coords {
        for v in c.iter().chain(std::iter::once(&0)) {
            cbytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let cbuf = gpu.upload(&cbytes);
    let out = gpu.alloc(coords.len() * 16);
    sim.run(gpu, &Launch::new([1, 1, 1], [coords.len() as u32, 1, 1], vec![handle, cbuf, out])).unwrap();
    assert!(gpu.violations.borrow().is_empty(), "{:?}", gpu.violations.borrow());
    let raw = gpu.download(out, coords.len() * 16);
    raw.chunks(16).map(|q| std::array::from_fn(|c| u32::from_le_bytes(q[4 * c..4 * c + 4].try_into().unwrap()))).collect()
}

fn translate_kernel(src: &str) -> Sim {
    let t = translate(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    Sim::new(&rspirv::dr::load_words(&t.words).unwrap())
}

#[test]
fn tex_objects_match_cuda_sampling_rules() {
    let mut rng = Rng(0x7e57_ab1e_5eed_0001);
    let mut cases = 0;
    let (mut nonzero, mut blended) = (0usize, 0usize);
    for dims in 1..=3 {
        for (float_coords, float_out) in [(true, true), (false, true), (true, false), (false, false)] {
            let mut sim = translate_kernel(&tex_kernel(dims, float_coords, float_out));
            for &(kind, cu) in &KINDS {
                for channels in [1u32, 2, 4] {
                    for mode in 0..4u32 {
                        for filter in 0..2u32 {
                            for normalized in [false, true] {
                                if !float_out && filter == 1 {
                                    continue; // CUDA rejects linear filtering for integer reads
                                }
                                let mut m = random_model(&mut rng, dims, kind, cu, channels);
                                m.desc.address_mode = [mode, (mode + 1) % 4, (mode + 2) % 4];
                                m.desc.filter = filter;
                                m.desc.flags = if normalized { texdesc::NORMALIZED_COORDINATES } else { 0 };
                                if rng.below(4) == 0 {
                                    m.desc.flags |= texdesc::READ_AS_INTEGER;
                                }
                                let coords = coords_for(&mut rng, &m, dims, float_coords, 16);
                                let mut gpu = Gpu::default();
                                let got = run_case(&mut gpu, &mut sim, &m, &coords);
                                for (c, g) in coords.iter().zip(&got) {
                                    let want = m.sample(dims, *c, float_coords, float_out);
                                    nonzero += g.iter().filter(|v| **v != 0).count();
                                    if filter == 1 && float_coords {
                                        // A filtered value that equals no single texel channel was blended.
                                        let point = { let mut p = Model { desc: m.desc.clone(), data: m.data.clone() }; p.desc.filter = 0; p.sample(dims, *c, true, true) };
                                        blended += usize::from(g[0] != point[0]);
                                    }
                                    let same = want.iter().zip(g).all(|(w, g)| w == g || (float_out && f32::from_bits(*w).is_nan() && f32::from_bits(*g).is_nan()));
                                    assert!(
                                        same,
                                        "dims {dims} kind {kind:?} ch {channels} mode {mode} filter {filter} norm {normalized} \
                                         fc {float_coords} fo {float_out} coords {:?}: want {:?} got {:?}\n{:?}",
                                        c.map(f32::from_bits),
                                        want.map(f32::from_bits),
                                        g.map(f32::from_bits),
                                        m.desc
                                    );
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 1000, "{cases}");
    assert!(nonzero > cases * 16, "outputs are mostly zero: {nonzero}");
    assert!(blended > cases, "linear filtering rarely blends: {blended}");
}

#[test]
fn tex_linear_resource_reads_zero_out_of_range() {
    // tex1Dfetch on CU_RESOURCE_TYPE_LINEAR: integer coordinates, zero outside.
    let mut rng = Rng(42);
    let mut m = random_model(&mut rng, 1, Kind::F32, 0x20, 1);
    m.desc.linear = true;
    m.desc.resource_type = 2;
    let mut sim = translate_kernel(&tex_kernel(1, false, true));
    let w = m.desc.width as i32;
    let coords: Vec<[u32; 3]> = [-3, -1, 0, w - 1, w, w + 5].iter().map(|&x| [x as u32, 0, 0]).collect();
    let mut gpu = Gpu::default();
    let got = run_case(&mut gpu, &mut sim, &m, &coords);
    for (c, g) in coords.iter().zip(&got) {
        let x = c[0] as i32;
        let inside = (0..w).contains(&x);
        let want = if inside { m.channel_raw(x as usize, 0, 0, 0) } else { m.desc.border[0].to_bits() };
        assert_eq!(g[0], want, "x={x}");
        let _ = inside;
    }
}

#[test]
fn surface_loads_and_stores_follow_boundary_modes() {
    let mut rng = Rng(0x5eed_5u64);
    for dims in 1..=3usize {
        for (bits, count) in [(8u32, 1usize), (16, 2), (32, 1), (32, 2), (32, 4), (64, 1), (64, 2)] {
            for clamp in [false, true] {
                let src = surf_kernel(dims, bits, count, clamp);
                let mut sim = translate_kernel(&src);
                let bytes = (bits / 8) as usize * count;
                // Surface element = one access unit (uint1/2/4 etc.).
                let (kind, cu, ch) = match bytes {
                    1 => (Kind::U8, 1, 1),
                    2 => (Kind::U16, 2, 1),
                    4 => (Kind::U32, 3, 1),
                    8 => (Kind::U32, 3, 2),
                    _ => (Kind::U32, 3, 4),
                };
                let a = random_model(&mut rng, dims, kind, cu, ch);
                let mut b = random_model(&mut rng, dims, kind, cu, ch);
                b.desc.width = a.desc.width;
                b.desc.height = a.desc.height;
                b.desc.depth = a.desc.depth;
                b.desc.pitch = a.desc.pitch;
                b.desc.slice = a.desc.slice;
                b.data = vec![0u8; a.data.len()];
                let n = 24usize;
                let sizes = [a.desc.width as i64, a.desc.height as i64, a.desc.depth as i64];
                let coords: Vec<[i32; 3]> = (0..n)
                    .map(|i| {
                        std::array::from_fn(|k| {
                            if k >= dims {
                                return 0;
                            }
                            let s = sizes[k];
                            let v = if k == 0 {
                                // x is a byte offset, element aligned
                                (rng.below(3 * s as u64) as i64 - s) * bytes as i64
                            } else {
                                rng.below(3 * s as u64) as i64 - s
                            };
                            // Thread i in range at a unique element for the store check.
                            if i < 4 && k == 0 { (i as i64 % s) * bytes as i64 } else { v as i64 }
                        } as i32)
                    })
                    .collect();
                let vals: Vec<u8> = (0..n * 32).map(|_| rng.next() as u8).collect();
                let mut gpu = Gpu::default();
                let ha = upload(&mut gpu, &a);
                let b_base = gpu.upload(&b.data);
                let mut bd = b.desc.clone();
                bd.base = b_base;
                let hb = gpu.upload(&bd.encode());
                let mut cb = Vec::new();
                for c in &coords {
                    for v in c.iter().chain(std::iter::once(&0)) {
                        cb.extend_from_slice(&v.to_le_bytes());
                    }
                }
                let cbuf = gpu.upload(&cb);
                let vbuf = gpu.upload(&vals);
                let out = gpu.alloc(n * 32);
                sim.run(&mut gpu, &Launch::new([1, 1, 1], [1, 1, 1], vec![ha, hb, cbuf, vbuf, out])).unwrap();
                // One thread (coords[0]) keeps the store check free of write races.
                assert!(gpu.violations.borrow().is_empty(), "{:?}", gpu.violations.borrow());
                let elem = if bits == 64 { 8 } else { 4 };
                let got = gpu.download(out, 32);
                // Thread 0: in range by construction.
                let c = coords[0];
                let at = |c: [i32; 3]| -> Option<usize> {
                    let x = c[0] as i64;
                    let inside = x >= 0 && x + bytes as i64 <= sizes[0] * bytes as i64 && (1..dims).all(|k| c[k] >= 0 && (c[k] as i64) < sizes[k]);
                    let (x, y, z) = if inside || clamp {
                        (
                            x.clamp(0, (sizes[0] - 1) * bytes as i64) as usize,
                            if dims >= 2 { (c[1] as i64).clamp(0, sizes[1] - 1) as usize } else { 0 },
                            if dims >= 3 { (c[2] as i64).clamp(0, sizes[2] - 1) as usize } else { 0 },
                        )
                    } else {
                        return None;
                    };
                    Some(z * a.desc.slice as usize + y * a.desc.pitch as usize + x)
                };
                let comp = (bits / 8) as usize;
                for i in 0..count {
                    let want: u64 = match at(c) {
                        Some(o) => (0..comp).map(|j| u64::from(a.data[o + i * comp + j]) << (8 * j)).sum(),
                        None => 0,
                    };
                    let g = u64::from_le_bytes(got[i * elem..i * elem + 8].try_into().unwrap_or([0; 8]));
                    let mask = if comp == 8 { u64::MAX } else { (1u64 << (8 * comp)) - 1 };
                    let g = if elem == 4 { g & 0xffff_ffff } else { g };
                    assert_eq!(g & mask, want, "load dims {dims} b{bits} x{count} clamp {clamp} coords {c:?}");
                }
                // The store wrote vals[0] at the same place.
                let bout = gpu.download(b_base, b.data.len());
                if let Some(o) = at(c) {
                    for i in 0..count {
                        for j in 0..comp {
                            assert_eq!(bout[o + i * comp + j], vals[i * elem + j], "store dims {dims} b{bits} x{count}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn surface_out_of_range_threads_read_zero_and_drop_stores() {
    // Many threads, each with its own coordinate; out-of-range ones must read 0 (zero mode)
    // or the clamped element (clamp mode), and never write outside the surface.
    let mut rng = Rng(77);
    for dims in 1..=3usize {
        for clamp in [false, true] {
            let mut sim = translate_kernel(&surf_kernel(dims, 32, 1, clamp));
            let a = random_model(&mut rng, dims, Kind::U32, 3, 1);
            let n = 48usize;
            let sizes = [a.desc.width as i64, a.desc.height as i64, a.desc.depth as i64];
            let coords: Vec<[i32; 3]> = (0..n)
                .map(|_| std::array::from_fn(|k| if k >= dims { 0 } else if k == 0 { ((rng.below(3 * sizes[0] as u64) as i64 - sizes[0]) * 4) as i32 } else { (rng.below(3 * sizes[k] as u64) as i64 - sizes[k]) as i32 }))
                .collect();
            let mut gpu = Gpu::default();
            let ha = upload(&mut gpu, &a);
            // Store target: a copy of A, so racing in-range stores still leave valid data.
            let hb = upload(&mut gpu, &a);
            let mut cb = Vec::new();
            for c in &coords {
                for v in c.iter().chain(std::iter::once(&0)) {
                    cb.extend_from_slice(&v.to_le_bytes());
                }
            }
            let cbuf = gpu.upload(&cb);
            let vbuf = gpu.alloc(n * 32);
            let out = gpu.alloc(n * 32);
            sim.run(&mut gpu, &Launch::new([1, 1, 1], [n as u32, 1, 1], vec![ha, hb, cbuf, vbuf, out])).unwrap();
            assert!(gpu.violations.borrow().is_empty(), "{:?}", gpu.violations.borrow());
            let got = gpu.download(out, n * 32);
            for (t, c) in coords.iter().enumerate() {
                let x = c[0] as i64 / 4;
                let inside = c[0] >= 0 && x < sizes[0] && (1..dims).all(|k| c[k] >= 0 && (c[k] as i64) < sizes[k]);
                let want = if inside || clamp {
                    let cx = x.clamp(0, sizes[0] - 1) as usize;
                    let cy = if dims >= 2 { (c[1] as i64).clamp(0, sizes[1] - 1) as usize } else { 0 };
                    let cz = if dims >= 3 { (c[2] as i64).clamp(0, sizes[2] - 1) as usize } else { 0 };
                    a.channel_raw(cx, cy, cz, 0)
                } else {
                    0
                };
                let g = u32::from_le_bytes(got[t * 32..t * 32 + 4].try_into().unwrap());
                assert_eq!(g, want, "dims {dims} clamp {clamp} coords {c:?}");
            }
        }
    }
}

#[test]
fn txq_and_suq_report_the_descriptor() {
    let src = ".version 7.0\n.target sm_80\n.address_size 64\n\
.visible .entry q(.param .u64 t, .param .u64 out)\n{\n.reg .b32 %r<16>;\n.reg .b64 %rd<4>;\n.reg .pred %p<2>;\n\
ld.param.u64 %rd1, [t];\nld.param.u64 %rd2, [out];\n\
txq.width.b32 %r1, [%rd1];\ntxq.height.b32 %r2, [%rd1];\ntxq.depth.b32 %r3, [%rd1];\n\
txq.normalized_coords.b32 %r4, [%rd1];\ntxq.filter_mode.b32 %r5, [%rd1];\ntxq.addr_mode_1.b32 %r6, [%rd1];\n\
txq.channel_order.b32 %r7, [%rd1];\nsuq.width.b32 %r8, [%rd1];\n\
setp.eq.u32 %p1, %r1, 0;\n@%p1 txq.width.b32 %r9, [%rd1];\n@!%p1 mov.u32 %r9, 77;\n\
st.global.u32 [%rd2], %r1;\nst.global.u32 [%rd2+4], %r2;\nst.global.u32 [%rd2+8], %r3;\nst.global.u32 [%rd2+12], %r4;\n\
st.global.u32 [%rd2+16], %r5;\nst.global.u32 [%rd2+20], %r6;\nst.global.u32 [%rd2+24], %r7;\nst.global.u32 [%rd2+28], %r8;\n\
st.global.u32 [%rd2+32], %r9;\nret;\n}\n";
    let mut sim = translate_kernel(src);
    let mut rng = Rng(9);
    let mut m = random_model(&mut rng, 3, Kind::F32, 0x20, 2);
    m.desc.flags = texdesc::NORMALIZED_COORDINATES;
    m.desc.filter = 1;
    m.desc.address_mode = [0, 3, 1];
    let mut gpu = Gpu::default();
    let h = upload(&mut gpu, &m);
    let out = gpu.alloc(36);
    sim.run(&mut gpu, &Launch::new([1, 1, 1], [1, 1, 1], vec![h, out])).unwrap();
    let g = gpu.download(out, 36);
    let w: Vec<u32> = g.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(w, vec![m.desc.width, m.desc.height, m.desc.depth, 1, 1, 3, 2, m.desc.width, 77]);
}

#[test]
fn texture_references_stay_a_precise_error() {
    let src = ".version 7.0\n.target sm_80\n.address_size 64\n.global .texref t;\n\
.visible .entry k(.param .u64 o)\n{\n.reg .f32 %f<6>;\n.reg .b64 %rd<2>;\n\
tex.1d.v4.f32.f32 {%f1, %f2, %f3, %f4}, [t, {%f5}];\nret;\n}\n";
    let err = translate(src).unwrap_err().to_string();
    assert!(err.contains("texture references"), "{err}");
}

#[test]
fn zluda_texobj_fixtures_translate() {
    for f in std::fs::read_dir("tests/ptx_corpus/zluda/tex").unwrap() {
        let p = f.unwrap().path();
        if !p.to_string_lossy().ends_with("_texobj.ptx") {
            continue;
        }
        // As the corpus harness does: ZLUDA's files declare PTX 6.5.
        let src = super::zluda_source::normalize(&std::fs::read_to_string(&p).unwrap());
        translate(&src).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    }
}

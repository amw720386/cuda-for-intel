//! CUDA arrays, texture objects and surface objects on the GPU, against the host sampling model
//! shared with the simulator tests (`src/ptx/tex_model.rs`): array creation, descriptors and
//! 1D/2D/3D copies to and from arrays; `tex` over arrays, pitched 2D memory and linear memory
//! for every address mode, both filter modes, normalised coordinates and the channel formats;
//! ZLUDA's 12 texture-object fixtures; `suld`/`sust` over surface objects.
use super::tex_model::*;
use super::texdesc::{self, Kind};
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{CString, c_void};
use std::ptr::{from_mut, null, null_mut};

#[repr(C)]
#[derive(Default, Debug, PartialEq)]
struct Array3D {
    width: usize,
    height: usize,
    depth: usize,
    format: u32,
    channels: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Default, Debug, PartialEq)]
struct Array2D {
    width: usize,
    height: usize,
    format: u32,
    channels: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ResourceDesc {
    res_type: u32,
    res: [u64; 16],
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TextureDesc {
    address_mode: [u32; 3],
    filter_mode: u32,
    flags: u32,
    max_anisotropy: u32,
    mipmap_filter_mode: u32,
    bias: f32,
    min_clamp: f32,
    max_clamp: f32,
    border: [f32; 4],
    reserved: [i32; 12],
}

#[repr(C)]
#[derive(Default)]
struct Memcpy3D {
    src_x: usize,
    src_y: usize,
    src_z: usize,
    src_lod: usize,
    src_type: u32,
    src_host: usize,
    src_device: u64,
    src_array: usize,
    reserved0: usize,
    src_pitch: usize,
    src_height: usize,
    dst_x: usize,
    dst_y: usize,
    dst_z: usize,
    dst_lod: usize,
    dst_type: u32,
    dst_host: usize,
    dst_device: u64,
    dst_array: usize,
    reserved1: usize,
    dst_pitch: usize,
    dst_height: usize,
    width: usize,
    height: usize,
    depth: usize,
}

#[repr(C)]
#[derive(Default)]
struct Memcpy2D {
    src_x: usize,
    src_y: usize,
    src_type: u32,
    src_host: usize,
    src_device: u64,
    src_array: usize,
    src_pitch: usize,
    dst_x: usize,
    dst_y: usize,
    dst_type: u32,
    dst_host: usize,
    dst_device: u64,
    dst_array: usize,
    dst_pitch: usize,
    width: usize,
    height: usize,
}

const HOST: u32 = 1;
const DEVICE: u32 = 2;
const ARRAY: u32 = 3;

/// Bits equal, or floats within 2 ulp (the model and the GPU round identically on every step,
/// so this is slack only).
fn close(want: u32, got: u32, float: bool) -> bool {
    if want == got {
        return true;
    }
    if !float {
        return false;
    }
    let (w, g) = (f32::from_bits(want), f32::from_bits(got));
    (w.is_nan() && g.is_nan()) || ((want as i32).wrapping_sub(got as i32)).unsigned_abs() <= 2
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_texture_arrays_textures_and_surfaces() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSynchronize(),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const i8),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        cuArray3DCreate_v2(*mut Handle, *const Array3D),
        cuArrayCreate_v2(*mut Handle, *const Array2D),
        cuArray3DGetDescriptor_v2(*mut Array3D, Handle),
        cuArrayGetDescriptor_v2(*mut Array2D, Handle),
        cuArrayDestroy(Handle),
        cuMemcpy3D_v2(*const Memcpy3D),
        cuMemcpy2D_v2(*const Memcpy2D),
        cuMemcpyHtoA_v2(Handle, usize, *const c_void, usize),
        cuMemcpyAtoH_v2(*mut c_void, Handle, usize, usize),
        cuMemcpyAtoD_v2(u64, Handle, usize, usize),
        cuMemcpyDtoA_v2(Handle, usize, u64, usize),
        cuMemcpyAtoA_v2(Handle, usize, Handle, usize, usize),
        cuTexObjectCreate(*mut u64, *const ResourceDesc, *const TextureDesc, *const c_void),
        cuTexObjectDestroy(u64),
        cuTexObjectGetResourceDesc(*mut ResourceDesc, u64),
        cuTexObjectGetTextureDesc(*mut TextureDesc, u64),
        cuSurfObjectCreate(*mut u64, *const ResourceDesc),
        cuSurfObjectDestroy(u64),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let mut rng = Rng(0x0dd_ba11);

    let load = |src: &str, entry: &str| -> Handle {
        let text = CString::new(src).unwrap();
        let mut module = null_mut();
        ok!(cuModuleLoadData(&mut module, text.as_ptr().cast()));
        let name = CString::new(entry).unwrap();
        let mut f = null_mut();
        ok!(cuModuleGetFunction(&mut f, module, name.as_ptr()));
        f
    };
    let upload = |bytes: &[u8]| -> u64 {
        let mut d = 0u64;
        ok!(cuMemAlloc_v2(&mut d, bytes.len().max(16)));
        ok!(cuMemcpyHtoD_v2(d, bytes.as_ptr().cast(), bytes.len()));
        d
    };
    let download = |d: u64, n: usize| -> Vec<u8> {
        let mut v = vec![0u8; n];
        ok!(cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), d, n));
        v
    };
    let launch = |f: Handle, threads: u32, args: &mut [*mut c_void]| {
        ok!(cuLaunchKernel(f, 1, 1, 1, threads, 1, 1, 0, null_mut(), args.as_mut_ptr(), null_mut()));
        ok!(cuCtxSynchronize());
    };
    // A CUDA array holding the model's texels (the model must be dense).
    let make_array = |m: &Model| -> Handle {
        let d = &m.desc;
        let desc = Array3D {
            width: d.width as usize,
            height: if d.dims >= 2 { d.height as usize } else { 0 },
            depth: if d.dims >= 3 { d.depth as usize } else { 0 },
            format: d.cu_format,
            channels: d.channels,
            flags: 0,
        };
        let mut a = null_mut();
        ok!(cuArray3DCreate_v2(&mut a, &desc));
        let row = d.width as usize * d.element_bytes() as usize;
        let copy = Memcpy3D {
            src_type: HOST,
            src_host: m.data.as_ptr().addr(),
            src_pitch: d.pitch as usize,
            src_height: d.height as usize,
            dst_type: ARRAY,
            dst_array: a.addr(),
            width: row,
            height: d.height as usize,
            depth: d.depth as usize,
            ..Memcpy3D::default()
        };
        ok!(cuMemcpy3D_v2(&copy));
        a
    };

    // ---- arrays: descriptors and copies --------------------------------------------------
    let mut array_cases = 0;
    for dims in 1..=3usize {
        for &(kind, cu) in &KINDS {
            for channels in [1u32, 2, 4] {
                let m = random_model_with(&mut rng, dims, kind, cu, channels, true);
                let a = make_array(&m);
                let mut got = Array3D::default();
                ok!(cuArray3DGetDescriptor_v2(&mut got, a));
                assert_eq!((got.width, got.format, got.channels), (m.desc.width as usize, cu, channels));
                assert_eq!(got.height, if dims >= 2 { m.desc.height as usize } else { 0 });
                assert_eq!(got.depth, if dims >= 3 { m.desc.depth as usize } else { 0 });
                // array -> host (pitched, with an x/y offset on the host side)
                let row = m.desc.width as usize * m.desc.element_bytes() as usize;
                let host_pitch = row + 24;
                let mut back = vec![0u8; host_pitch * m.desc.height as usize * m.desc.depth as usize + 8];
                let copy = Memcpy3D {
                    src_type: ARRAY,
                    src_array: a.addr(),
                    dst_type: HOST,
                    dst_host: back.as_mut_ptr().addr(),
                    dst_x: 8,
                    dst_pitch: host_pitch,
                    dst_height: m.desc.height as usize,
                    width: row,
                    height: m.desc.height as usize,
                    depth: m.desc.depth as usize,
                    ..Memcpy3D::default()
                };
                ok!(cuMemcpy3D_v2(&copy));
                for z in 0..m.desc.depth as usize {
                    for y in 0..m.desc.height as usize {
                        let s = z * m.desc.slice as usize + y * m.desc.pitch as usize;
                        let t = (z * m.desc.height as usize + y) * host_pitch + 8;
                        assert_eq!(&back[t..t + row], &m.data[s..s + row], "dims {dims} {kind:?} x{channels} z{z} y{y}");
                    }
                }
                // 1D linear copies: AtoH, HtoA, AtoD, DtoA, AtoA
                let total = m.data.len();
                let mut flat = vec![0u8; total];
                ok!(cuMemcpyAtoH_v2(flat.as_mut_ptr().cast(), a, 0, total));
                assert_eq!(flat, m.data);
                let dev = upload(&vec![0u8; total]);
                ok!(cuMemcpyAtoD_v2(dev, a, 0, total));
                assert_eq!(download(dev, total), m.data);
                let mut b = null_mut();
                let desc = Array3D { width: total, height: 0, depth: 0, format: 0x01, channels: 1, flags: 0 };
                ok!(cuArray3DCreate_v2(&mut b, &desc));
                ok!(cuMemcpyDtoA_v2(b, 0, dev, total));
                let mut c = null_mut();
                ok!(cuArray3DCreate_v2(&mut c, &desc));
                ok!(cuMemcpyAtoA_v2(c, 1, b, 0, total - 1));
                let mut tail = vec![0u8; total - 1];
                ok!(cuMemcpyAtoH_v2(tail.as_mut_ptr().cast(), c, 1, total - 1));
                assert_eq!(tail, m.data[..total - 1]);
                assert_eq!(unsafe { cuMemcpyAtoH_v2(tail.as_mut_ptr().cast(), c, 2, total - 1) }, 1, "past the end");
                ok!(cuMemcpyHtoA_v2(c, 0, m.data.as_ptr().cast(), total));
                ok!(cuMemcpyAtoH_v2(flat.as_mut_ptr().cast(), c, 0, total));
                assert_eq!(flat, m.data);
                for h in [a, b, c] {
                    ok!(cuArrayDestroy(h));
                }
                ok!(cuMemFree_v2(dev));
                array_cases += 1;
            }
        }
    }
    // The 2D descriptor API and cuMemcpy2D host <-> array.
    let mut a2 = null_mut();
    ok!(cuArrayCreate_v2(&mut a2, &Array2D { width: 13, height: 5, format: 0x20, channels: 2 }));
    let mut d2 = Array2D::default();
    ok!(cuArrayGetDescriptor_v2(&mut d2, a2));
    assert_eq!(d2, Array2D { width: 13, height: 5, format: 0x20, channels: 2 });
    let src: Vec<u8> = (0..13 * 8 * 5 + 40).map(|i| i as u8).collect();
    let copy = Memcpy2D { src_type: HOST, src_host: src.as_ptr().addr(), src_pitch: 13 * 8 + 8, dst_type: ARRAY, dst_array: a2.addr(), width: 13 * 8, height: 5, ..Memcpy2D::default() };
    ok!(cuMemcpy2D_v2(&copy));
    let mut flat = vec![0u8; 13 * 8 * 5];
    ok!(cuMemcpyAtoH_v2(flat.as_mut_ptr().cast(), a2, 0, flat.len()));
    for y in 0..5 {
        assert_eq!(&flat[y * 104..y * 104 + 104], &src[y * 112..y * 112 + 104]);
    }
    ok!(cuArrayDestroy(a2));
    assert_eq!(unsafe { cuArrayDestroy(a2) }, 400);
    println!("arrays: {array_cases} layouts round-tripped");

    // ---- texture objects -----------------------------------------------------------------
    let mut tex_cases = 0;
    let mut mismatches = Vec::new();
    for dims in 1..=3usize {
        for (fc, fo) in [(true, true), (false, true), (true, false), (false, false)] {
            let f = load(&tex_kernel(dims, fc, fo), "tk");
            for &(kind, cu) in &KINDS {
                for channels in [1u32, 4] {
                    for mode in 0..4u32 {
                        for filter in 0..2u32 {
                            for normalized in [false, true] {
                                if !fo && filter == 1 {
                                    continue;
                                }
                                if filter == 1 && matches!(kind, Kind::U32 | Kind::S32) {
                                    continue; // rejected at create time
                                }
                                // Resource kind: array (dense), or pitched / linear memory.
                                let pitched = dims == 2 && rng.below(2) == 0;
                                let linear = dims == 1 && !fc && rng.below(2) == 0;
                                let mut m = random_model_with(&mut rng, dims, kind, cu, channels, !pitched);
                                m.desc.address_mode = [mode, (mode + 1) % 4, (mode + 2) % 4];
                                m.desc.filter = filter;
                                m.desc.flags = if normalized { texdesc::NORMALIZED_COORDINATES } else { 0 };
                                let mut res = ResourceDesc { res_type: 0, res: [0; 16], flags: 0 };
                                let (mut array, mut memory) = (null_mut(), 0u64);
                                if pitched || linear {
                                    memory = upload(&m.data);
                                    res.res[0] = memory;
                                    res.res[1] = u64::from(cu) | u64::from(channels) << 32;
                                    if linear {
                                        res.res_type = 2;
                                        res.res[2] = m.data.len() as u64;
                                        m.desc.linear = true;
                                        m.desc.width = (m.data.len() as u64 / u64::from(m.desc.element_bytes())) as u32;
                                    } else {
                                        res.res_type = 3;
                                        res.res[2] = u64::from(m.desc.width);
                                        res.res[3] = u64::from(m.desc.height);
                                        res.res[4] = m.desc.pitch;
                                    }
                                } else {
                                    array = make_array(&m);
                                    res.res[0] = array.addr() as u64;
                                }
                                let td = TextureDesc { address_mode: m.desc.address_mode, filter_mode: filter, flags: m.desc.flags, border: m.desc.border, ..TextureDesc::default() };
                                let mut tex = 0u64;
                                ok!(cuTexObjectCreate(&mut tex, &res, &td, null()));
                                let mut back = TextureDesc::default();
                                ok!(cuTexObjectGetTextureDesc(&mut back, tex));
                                assert_eq!((back.address_mode, back.filter_mode), (td.address_mode, td.filter_mode));
                                let mut rb = ResourceDesc { res_type: 9, res: [0; 16], flags: 0 };
                                ok!(cuTexObjectGetResourceDesc(&mut rb, tex));
                                assert_eq!((rb.res_type, rb.res[0]), (res.res_type, res.res[0]));
                                let coords = coords_for(&mut rng, &m, dims, fc, 64);
                                let mut cb = Vec::new();
                                for c in &coords {
                                    for v in c.iter().chain(std::iter::once(&0)) {
                                        cb.extend_from_slice(&v.to_le_bytes());
                                    }
                                }
                                let mut cdev = upload(&cb);
                                let mut odev = upload(&vec![0u8; coords.len() * 16]);
                                let mut t = tex;
                                launch(f, coords.len() as u32, &mut [from_mut(&mut t).cast(), from_mut(&mut cdev).cast(), from_mut(&mut odev).cast()]);
                                let out = download(odev, coords.len() * 16);
                                for (i, c) in coords.iter().enumerate() {
                                    let want = m.sample(dims, *c, fc, fo);
                                    let got: [u32; 4] = std::array::from_fn(|k| u32::from_le_bytes(out[i * 16 + 4 * k..i * 16 + 4 * k + 4].try_into().unwrap()));
                                    if !want.iter().zip(&got).all(|(w, g)| close(*w, *g, fo)) && mismatches.len() < 10 {
                                        mismatches.push(format!(
                                            "dims {dims} {kind:?} x{channels} mode {mode} filter {filter} norm {normalized} fc {fc} fo {fo} pitched {pitched} linear {linear} at {:?}: want {:?} got {:?}",
                                            c.map(f32::from_bits),
                                            want,
                                            got
                                        ));
                                    }
                                }
                                ok!(cuTexObjectDestroy(tex));
                                assert_eq!(unsafe { cuTexObjectDestroy(tex) }, 1);
                                if !array.is_null() {
                                    ok!(cuArrayDestroy(array));
                                }
                                if memory != 0 {
                                    ok!(cuMemFree_v2(memory));
                                }
                                ok!(cuMemFree_v2(cdev));
                                ok!(cuMemFree_v2(odev));
                                tex_cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{} texture mismatches, first:\n{}", mismatches.len(), mismatches.join("\n"));
    println!("texture objects: {tex_cases} configurations x 64 coordinates match the model");

    // ---- ZLUDA's 12 texture-object fixtures -------------------------------------------------
    let fixtures: [(&str, &str); 12] = [
        ("tex_read_1d_float_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_1d_float_texobj.ptx")),
        ("tex_read_1d_float_s32coord_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_1d_float_s32coord_texobj.ptx")),
        ("tex_read_1d_int_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_1d_int_texobj.ptx")),
        ("tex_read_1d_int_s32coord_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_1d_int_s32coord_texobj.ptx")),
        ("tex_read_2d_float_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_2d_float_texobj.ptx")),
        ("tex_read_2d_float_s32coord_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_2d_float_s32coord_texobj.ptx")),
        ("tex_read_2d_int_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_2d_int_texobj.ptx")),
        ("tex_read_2d_int_s32coord_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_2d_int_s32coord_texobj.ptx")),
        ("tex_read_3d_float_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_3d_float_texobj.ptx")),
        ("tex_read_3d_float_s32coord_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_3d_float_s32coord_texobj.ptx")),
        ("tex_read_3d_int_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_3d_int_texobj.ptx")),
        ("tex_read_3d_int_s32coord_texobj", include_str!("../ptx_corpus/zluda/tex/test_ptx__tex_read_3d_int_s32coord_texobj.ptx")),
    ];
    for (entry, src) in fixtures {
        let dims = entry.as_bytes()[9] as usize - b'0' as usize;
        let fo = entry.contains("float");
        let fc = !entry.contains("s32coord");
        // ZLUDA's files declare PTX 6.5; the translator takes 8.8 (as src/ptx/zluda_source.rs rewrites it).
        let src = src.replacen(".version 6.5", ".version 8.8", 1);
        let f = load(&src, entry);
        // Float fixtures read a float texture, int fixtures an int texture (ZLUDA's set-up).
        let (kind, cu) = if fo { (Kind::F32, 0x20) } else { (Kind::S32, 0x0a) };
        for filter in 0..if fo && fc { 2 } else { 1 } {
            let mut m = random_model_with(&mut rng, dims, kind, cu, 4, true);
            m.desc.filter = filter;
            m.desc.address_mode = [1, 1, 1];
            let array = make_array(&m);
            let mut res = ResourceDesc { res_type: 0, res: [0; 16], flags: 0 };
            res.res[0] = array.addr() as u64;
            let td = TextureDesc { address_mode: [1; 3], filter_mode: filter, flags: if fo { 0 } else { texdesc::READ_AS_INTEGER }, ..TextureDesc::default() };
            m.desc.flags = td.flags;
            let mut tex = 0u64;
            ok!(cuTexObjectCreate(&mut tex, &res, &td, null()));
            for c in coords_for(&mut rng, &m, dims, fc, 12) {
                let mut out = upload(&[0u8; 16]);
                let mut t = tex;
                let mut cs = c;
                let mut args: Vec<*mut c_void> = vec![from_mut(&mut out).cast(), from_mut(&mut t).cast()];
                for v in cs.iter_mut().take(dims) {
                    args.push(from_mut(v).cast());
                }
                launch(f, 1, &mut args);
                let got = download(out, 16);
                let got: [u32; 4] = std::array::from_fn(|k| u32::from_le_bytes(got[4 * k..4 * k + 4].try_into().unwrap()));
                let want = m.sample(dims, c, fc, fo);
                assert!(want.iter().zip(&got).all(|(w, g)| close(*w, *g, fo)), "{entry} filter {filter} at {c:?}: want {want:?} got {got:?}");
                ok!(cuMemFree_v2(out));
            }
            ok!(cuTexObjectDestroy(tex));
            ok!(cuArrayDestroy(array));
        }
    }
    println!("ZLUDA texture-object fixtures: 12 of 12 match the model");

    // ---- surface objects ---------------------------------------------------------------------
    let mut surf_cases = 0;
    for dims in 1..=3usize {
        for (bits, count) in [(8u32, 1usize), (16, 2), (32, 1), (32, 4), (64, 2)] {
            for clamp in [false, true] {
                let f = load(&surf_kernel(dims, bits, count, clamp), "sk");
                let bytes = (bits / 8) as usize * count;
                let (kind, cu, ch) = match bytes {
                    1 => (Kind::U8, 1, 1),
                    2 => (Kind::U16, 2, 1),
                    4 => (Kind::U32, 3, 1),
                    8 => (Kind::U32, 3, 2),
                    _ => (Kind::U32, 3, 4),
                };
                let m = random_model_with(&mut rng, dims, kind, cu, ch, true);
                let a = make_array(&m);
                let zero = Model { desc: m.desc.clone(), data: vec![0u8; m.data.len()] };
                let b = make_array(&zero);
                let surface = |arr: Handle| {
                    let mut r = ResourceDesc { res_type: 0, res: [0; 16], flags: 0 };
                    r.res[0] = arr.addr() as u64;
                    let mut s = 0u64;
                    ok!(cuSurfObjectCreate(&mut s, &r));
                    s
                };
                let (sa, sb) = (surface(a), surface(b));
                let sizes = [m.desc.width as i64, m.desc.height as i64, m.desc.depth as i64];
                for trial in 0..8 {
                    let c: [i32; 3] = std::array::from_fn(|k| {
                        if k >= dims {
                            0
                        } else if trial < 4 {
                            (rng.below(sizes[k] as u64) as i64 * if k == 0 { bytes as i64 } else { 1 }) as i32
                        } else {
                            ((rng.below(3 * sizes[k] as u64) as i64 - sizes[k]) * if k == 0 { bytes as i64 } else { 1 }) as i32
                        }
                    });
                    let mut cb = Vec::new();
                    for v in c.iter().chain(std::iter::once(&0)) {
                        cb.extend_from_slice(&v.to_le_bytes());
                    }
                    let vals: Vec<u8> = (0..32).map(|_| rng.next() as u8).collect();
                    let (mut pa, mut pb) = (sa, sb);
                    let mut cdev = upload(&cb);
                    let mut vdev = upload(&vals);
                    let mut odev = upload(&[0u8; 32]);
                    launch(f, 1, &mut [from_mut(&mut pa).cast(), from_mut(&mut pb).cast(), from_mut(&mut cdev).cast(), from_mut(&mut vdev).cast(), from_mut(&mut odev).cast()]);
                    let got = download(odev, 32);
                    let inside = c[0] >= 0 && (c[0] as i64) + bytes as i64 <= sizes[0] * bytes as i64 && (1..dims).all(|k| c[k] >= 0 && (c[k] as i64) < sizes[k]);
                    let at = (inside || clamp).then(|| {
                        let x = (c[0] as i64).clamp(0, (sizes[0] - 1) * bytes as i64) as usize;
                        let y = if dims >= 2 { (c[1] as i64).clamp(0, sizes[1] - 1) as usize } else { 0 };
                        let z = if dims >= 3 { (c[2] as i64).clamp(0, sizes[2] - 1) as usize } else { 0 };
                        z * m.desc.slice as usize + y * m.desc.pitch as usize + x
                    });
                    let comp = (bits / 8) as usize;
                    let elem = if bits == 64 { 8 } else { 4 };
                    for i in 0..count {
                        let want: u64 = at.map_or(0, |o| (0..comp).map(|j| u64::from(m.data[o + i * comp + j]) << (8 * j)).sum());
                        let mut g = [0u8; 8];
                        g[..elem].copy_from_slice(&got[i * elem..i * elem + elem]);
                        let mask = if comp == 8 { u64::MAX } else { (1u64 << (8 * comp)) - 1 };
                        assert_eq!(u64::from_le_bytes(g) & mask, want, "suld dims {dims} b{bits} x{count} clamp {clamp} at {c:?}");
                    }
                    if let Some(o) = at {
                        let mut flat = vec![0u8; m.data.len()];
                        ok!(cuMemcpyAtoH_v2(flat.as_mut_ptr().cast(), b, 0, flat.len()));
                        for i in 0..count {
                            for j in 0..comp {
                                assert_eq!(flat[o + i * comp + j], vals[i * elem + j], "sust dims {dims} b{bits} x{count} clamp {clamp} at {c:?}");
                            }
                        }
                    }
                    for d in [cdev, vdev, odev] {
                        ok!(cuMemFree_v2(d));
                    }
                    surf_cases += 1;
                }
                ok!(cuSurfObjectDestroy(sa));
                ok!(cuSurfObjectDestroy(sb));
                ok!(cuArrayDestroy(a));
                ok!(cuArrayDestroy(b));
            }
        }
    }
    println!("surface objects: {surf_cases} load/store pairs match the model");
    ok!(cuCtxDestroy_v2(context));
}

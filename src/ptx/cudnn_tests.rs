//! cuDNN shim kernels: translation, spirv-val and execution in the SPIR-V simulator
//! against the naive CPU references in `crates/cudnn_shim/src/refmath.rs`.
use super::cudnn_ptxgen::{self as pg, Elem};
use super::cudnn_refmath::{self as rm, ConvGeom, Kind};
use super::sim::{Gpu, Launch, Sim};
use super::*;

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x7f_ffff;
    if value == 0.0 {
        return sign;
    }
    assert!((-14..=15).contains(&exp), "test values must be normal f16");
    let mut half = (((exp + 15) as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

fn encode(e: Elem, v: &[f32]) -> Vec<u8> {
    match e {
        Elem::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Elem::F16 => v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect(),
        Elem::Bf16 => v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect(),
    }
}

fn decode(e: Elem, b: &[u8]) -> Vec<f32> {
    match e {
        Elem::F32 => b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        Elem::F16 => b
            .chunks(2)
            .map(|c| super::sim::f16_to_f32(u16::from_le_bytes([c[0], c[1]]) as u32))
            .collect(),
        Elem::Bf16 => b
            .chunks(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
    }
}

fn spirv_val(words: &[u32], name: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let validator = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    let Ok(mut child) = Command::new(&validator)
        .args(["--target-env", "opencl2.2", "-"])
        .stdin(Stdio::piped())
        .spawn()
    else {
        eprintln!("spirv-val not found; skipping validation of {name}");
        return;
    };
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let _ = child.stdin.take().unwrap().write_all(&bytes);
    assert!(child.wait().unwrap().success(), "invalid SPIR-V for {name}");
}

struct Arg;
impl Arg {
    fn f(v: f32) -> u64 {
        u64::from(v.to_bits())
    }
}

fn run(src: &str, name: &str, gpu: &mut Gpu, grid: [u32; 3], block: [u32; 3], args: Vec<u64>) {
    let t = translate(src).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(t.kernels[0].name, name);
    assert_eq!(t.kernels[0].params.len(), args.len(), "{name} arg count");
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let mut sim = Sim::new(&module);
    sim.run(gpu, &Launch::new(grid, block, args)).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert!(gpu.violations.borrow().is_empty(), "{name}: {:?}", gpu.violations.borrow());
}

#[test]
fn cudnn_kernels_translate_and_validate() {
    for (name, src, n) in pg::all_entries() {
        let t = translate(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(t.kernels[0].name, name);
        assert_eq!(t.kernels[0].params.len(), n, "{name}");
        spirv_val(&t.words, &name);
    }
}

fn conv_geom(flip: bool, nhwc: bool, groups: u32, dil: u32, stride: u32, pad: u32) -> ConvGeom {
    let (n, c, h, w, k, r, s) = (2u32, 4, 6, 5, 4, 3, 2);
    let p = rm::out_dim(h, pad, r, dil, stride).unwrap();
    let q = rm::out_dim(w, pad, s, dil, stride).unwrap();
    let cg = c / groups;
    let (xs, ys, ws) = if nhwc {
        (ConvGeom::packed_nhwc(c, h, w), ConvGeom::packed_nhwc(k, p, q), [r * s * cg, 1, s * cg, cg])
    } else {
        (ConvGeom::packed_nchw(c, h, w), ConvGeom::packed_nchw(k, p, q), [cg * r * s, r * s, s, 1])
    };
    ConvGeom {
        n, c, h, w, k, r, s, p, q, pad_h: pad, pad_w: pad, str_h: stride, str_w: stride,
        dil_h: dil, dil_w: dil, groups, flip, xs, ws, ys,
    }
}

fn sample(len: usize, seed: usize) -> Vec<f32> {
    // multiples of 1/8 in [-1, 1): exact in f16 and bf16, sums stay exact enough in f32
    (0..len).map(|i| (((i * 5 + seed * 3) % 16) as f32 - 8.0) / 8.0).collect()
}

fn check_conv(g: &ConvGeom, kind: Kind, e: Elem, tol: f32) {
    let (x, w, y0) = (sample(g.x_len(), 1), sample(g.w_len(), 2), sample(g.y_len(), 3));
    let (alpha, beta) = (1.5f32, 0.5f32);
    let mut gpu = Gpu::default();
    let (a_host, b_host, out_host, out_len, rk, gk) = match kind {
        Kind::Fwd => (x.clone(), w.clone(), y0.clone(), g.y_len(), Kind::Fwd, pg::ConvKind::Fwd),
        Kind::BwdData => (y0.clone(), w.clone(), x.clone(), g.x_len(), Kind::BwdData, pg::ConvKind::BwdData),
        Kind::BwdFilter => (x.clone(), y0.clone(), w.clone(), g.w_len(), Kind::BwdFilter, pg::ConvKind::BwdFilter),
    };
    // Round operands through the element type so the reference sees what the kernel sees.
    let rt = |v: &[f32]| decode(e, &encode(e, v));
    let (a_host, b_host, out_host) = (rt(&a_host), rt(&b_host), rt(&out_host));
    let a = gpu.upload(&encode(e, &a_host));
    let b = gpu.upload(&encode(e, &b_host));
    let out = gpu.upload(&encode(e, &out_host));
    let mut args = vec![a, b, out, Arg::f(alpha), Arg::f(beta)];
    args.extend(g.kernel_args(rk).iter().map(|v| u64::from(*v)));
    let total = g.total(rk);
    let name = pg::conv_entry(gk, e);
    run(&pg::generate_conv(gk, e), &name, &mut gpu, [total.div_ceil(256), 1, 1], [256, 1, 1], args);
    let got = decode(e, &gpu.download(out, out_len * (1 << e.shift())));
    let mut want = out_host.clone();
    match kind {
        Kind::Fwd => rm::conv_fwd(g, &a_host, &b_host, &mut want, alpha, beta),
        Kind::BwdData => rm::conv_bwd_data(g, &b_host, &a_host, &mut want, alpha, beta),
        Kind::BwdFilter => rm::conv_bwd_filter(g, &a_host, &b_host, &mut want, alpha, beta),
    }
    for (i, (g_, w_)) in got.iter().zip(&rt(&want)).enumerate() {
        assert!((g_ - w_).abs() <= tol * (1.0 + w_.abs()), "{name} elem {i}: got {g_} want {w_} ({g:?})");
    }
}

#[test]
fn sim_conv_f32_all_kinds_and_layouts() {
    let cases = [
        (false, false, 1, 1, 1, 0),
        (false, true, 1, 1, 1, 1),
        (true, false, 2, 1, 2, 1),
        (false, true, 2, 2, 1, 2),
        (true, true, 4, 2, 2, 1), // depthwise
        (false, false, 1, 1, 2, 0),
    ];
    for (flip, nhwc, groups, dil, stride, pad) in cases {
        let g = conv_geom(flip, nhwc, groups, dil, stride, pad);
        assert!(g.validate(), "{g:?}");
        for kind in [Kind::Fwd, Kind::BwdData, Kind::BwdFilter] {
            check_conv(&g, kind, Elem::F32, 1e-4);
        }
    }
}

#[test]
fn sim_conv_half_types() {
    let g = conv_geom(false, true, 2, 1, 1, 1);
    for e in [Elem::F16, Elem::Bf16] {
        for kind in [Kind::Fwd, Kind::BwdData, Kind::BwdFilter] {
            check_conv(&g, kind, e, 1e-2);
        }
    }
}

#[test]
fn sim_add_tensor_bias_broadcast() {
    let (n, c, h, w) = (2u32, 3u32, 2u32, 4u32);
    let total = n * c * h * w;
    let bias = sample(c as usize, 4);
    let y0 = sample(total as usize, 7);
    for e in Elem::all() {
        let mut gpu = Gpu::default();
        let a = gpu.upload(&encode(e, &bias));
        let y = gpu.upload(&encode(e, &y0));
        let mut args = vec![a, y, Arg::f(2.0), Arg::f(0.5)];
        // dims, A strides (bias broadcast over n,h,w), C strides (NCHW packed), total
        let cs = ConvGeom::packed_nchw(c, h, w);
        for v in [n, c, h, w, 0, 1, 0, 0, cs[0], cs[1], cs[2], cs[3], total] {
            args.push(u64::from(v));
        }
        run(&pg::generate_add_tensor(e), &pg::add_tensor_entry(e), &mut gpu, [1, 1, 1], [256, 1, 1], args);
        let got = decode(e, &gpu.download(y, total as usize * (1 << e.shift())));
        for i in 0..total as usize {
            let ch = (i / (h * w) as usize) % c as usize;
            let want = 2.0 * bias[ch] + 0.5 * y0[i];
            assert!((got[i] - want).abs() < 1e-2, "{e:?} {i}: {} vs {want}", got[i]);
        }
    }
}

fn bn_data(n: usize, c: usize, hw: usize) -> Vec<f32> {
    (0..n * c * hw).map(|i| ((i * 7) % 23) as f32 / 8.0 - 1.0 + (i / (hw * c)) as f32 * 0.25).collect()
}

#[test]
fn sim_batchnorm_train_infer_backward() {
    let (n, c, hw, wdim) = (3usize, 2usize, 6usize, 3u32);
    let x = bn_data(n, c, hw);
    let scale = vec![1.5f32, 0.75];
    let bias = vec![0.25f32, -0.5];
    let eps = 1e-5f32;
    let (want_y, want_mean, want_inv, want_uvar) = rm::bn_train(&x, n, c, hw, &scale, &bias, eps);
    let st = [(c * hw) as u32, hw as u32, wdim, 1];
    let dims = [n as u32, c as u32, hw as u32, wdim];
    let e = Elem::F32;
    let mut gpu = Gpu::default();
    let xd = gpu.upload(&encode(e, &x));
    let yd = gpu.alloc(x.len() * 4);
    let sd = gpu.upload(&encode(e, &scale));
    let bd = gpu.upload(&encode(e, &bias));
    let rmean = gpu.upload(&encode(e, &[1.0, 2.0]));
    let rvar = gpu.upload(&encode(e, &[3.0, 4.0]));
    let smean = gpu.alloc(8);
    let sinv = gpu.alloc(8);
    let factor = 0.25f32;
    let mut args = vec![xd, yd, sd, bd, rmean, rvar, smean, sinv, Arg::f(1.0), Arg::f(0.0), Arg::f(factor), Arg::f(eps)];
    args.extend(dims.iter().chain(&st).map(|v| u64::from(*v)));
    run(&pg::generate_bn_train(e), &pg::bn_train_entry(e), &mut gpu, [c as u32, 1, 1], [pg::BN_BLOCK, 1, 1], args);
    let close = |a: &[f32], b: &[f32], what: &str| {
        for (i, (p, q)) in a.iter().zip(b).enumerate() {
            assert!((p - q).abs() < 2e-4 * (1.0 + q.abs()), "{what}[{i}]: {p} vs {q}");
        }
    };
    close(&gpu.download_f32(yd, x.len()), &want_y, "y");
    close(&gpu.download_f32(smean, c), &want_mean, "saveMean");
    close(&gpu.download_f32(sinv, c), &want_inv, "saveInvVar");
    let rm0 = [1.0f32, 2.0];
    let rv0 = [3.0f32, 4.0];
    let want_rm: Vec<f32> = (0..c).map(|i| (1.0 - factor) * rm0[i] + factor * want_mean[i]).collect();
    let want_rv: Vec<f32> = (0..c).map(|i| (1.0 - factor) * rv0[i] + factor * want_uvar[i]).collect();
    close(&gpu.download_f32(rmean, c), &want_rm, "runMean");
    close(&gpu.download_f32(rvar, c), &want_rv, "runVar");

    // inference with the saved statistics reproduces training output (var = 1/invstd^2 - eps)
    let var: Vec<f32> = want_inv.iter().map(|i| 1.0 / (i * i) - eps).collect();
    let mean_d = gpu.upload(&encode(e, &want_mean));
    let var_d = gpu.upload(&encode(e, &var));
    let y2 = gpu.alloc(x.len() * 4);
    let mut args = vec![xd, y2, sd, bd, mean_d, var_d, Arg::f(1.0), Arg::f(0.0), Arg::f(eps)];
    args.extend(dims.iter().chain(&st).map(|v| u64::from(*v)));
    let total = (n * c * hw) as u32;
    run(&pg::generate_bn_infer(e), &pg::bn_infer_entry(e), &mut gpu, [total.div_ceil(256), 1, 1], [256, 1, 1], args);
    close(&gpu.download_f32(y2, x.len()), &want_y, "infer y");

    // backward with saved stats and with recomputed stats (null pointers)
    let dy: Vec<f32> = (0..x.len()).map(|i| ((i * 3) % 7) as f32 / 4.0 - 0.75).collect();
    let (wdx, wds, wdb) = rm::bn_bwd(&x, &dy, n, c, hw, &scale, &want_mean, &want_inv);
    for use_saved in [true, false] {
        let dyd = gpu.upload(&encode(e, &dy));
        let dxd = gpu.alloc(x.len() * 4);
        let dsd = gpu.alloc(8);
        let dbd = gpu.alloc(8);
        let (m, i) = if use_saved { (smean, sinv) } else { (0, 0) };
        let mut args = vec![xd, dyd, dxd, sd, dsd, dbd, m, i, Arg::f(1.0), Arg::f(0.0), Arg::f(1.0), Arg::f(0.0), Arg::f(eps)];
        args.extend(dims.iter().chain(&st).map(|v| u64::from(*v)));
        run(&pg::generate_bn_bwd(e), &pg::bn_bwd_entry(e), &mut gpu, [c as u32, 1, 1], [pg::BN_BLOCK, 1, 1], args);
        close(&gpu.download_f32(dxd, x.len()), &wdx, "dx");
        close(&gpu.download_f32(dsd, c), &wds, "dscale");
        close(&gpu.download_f32(dbd, c), &wdb, "dbias");
    }
}

#[test]
fn sim_relu_inplace_strided() {
    // NHWC-strided 4-D tensor: logical (2,3,2,2), channels-last strides.
    let (n, c, h, w) = (2u32, 3u32, 2u32, 2u32);
    let st = ConvGeom::packed_nhwc(c, h, w);
    let total = n * c * h * w;
    let y0 = sample(total as usize, 11);
    for e in Elem::all() {
        let mut gpu = Gpu::default();
        let y = gpu.upload(&encode(e, &y0));
        let mut args = vec![y];
        for v in [n, c, h, w, st[0], st[1], st[2], st[3], total] {
            args.push(u64::from(v));
        }
        run(&pg::generate_relu(e), &pg::relu_entry(e), &mut gpu, [1, 1, 1], [256, 1, 1], args);
        let got = decode(e, &gpu.download(y, total as usize * (1 << e.shift())));
        for (i, (g, v)) in got.iter().zip(&y0).enumerate() {
            assert_eq!(*g, v.max(0.0), "{e:?} elem {i}");
        }
    }
}

// --- Explicit-GEMM path: planner + im2col/col2im kernels in the simulator, GEMM on the CPU --------

mod gemm_path {
    use super::super::cudnn_gemmconv::{self as gc, Buf};
    use super::*;

    pub struct SimExec<'a> {
        pub gpu: &'a mut Gpu,
        pub e: Elem,
        pub g: ConvGeom,
        pub ops: [u64; 3],
        pub ws: u64,
        pub gemms: usize,
    }
    impl SimExec<'_> {
        fn addr(&self, b: Buf, off: usize) -> u64 {
            let esz = 1u64 << self.e.shift();
            match b {
                Buf::A => self.ops[0] + off as u64 * esz,
                Buf::B => self.ops[1] + off as u64 * esz,
                Buf::Out => self.ops[2] + off as u64 * esz,
                Buf::Ws(byte) => self.ws + byte as u64 + off as u64 * esz,
            }
        }
    }
    impl gc::Exec for SimExec<'_> {
        type Err = String;
        fn im2col(&mut self, src: Buf, dst: Buf, dst_off: usize, p: &gc::Im2col) -> Result<(), String> {
            let g = self.g;
            let mut a = vec![self.addr(src, 0), self.addr(dst, dst_off)];
            let cg = g.c / g.groups;
            for v in [
                p.total, p.n0, p.nl, cg, g.h, g.w, g.r, g.s, g.p, g.q, g.pad_h, g.pad_w, g.str_h, g.str_w, g.dil_h, g.dil_w, p.order,
                g.xs[0], g.xs[1], g.xs[2], g.xs[3], p.col_strides[0], p.col_strides[1], p.col_strides[2], p.col_strides[3],
            ] {
                a.push(u64::from(v));
            }
            run(&pg::generate_im2col(self.e, p.kc_fast), &pg::im2col_entry(self.e, p.kc_fast), self.gpu, [p.total.div_ceil(64), 1, 1], [64, 1, 1], a);
            Ok(())
        }
        fn col2im(&mut self, src: Buf, src_off: usize, dst: Buf, p: &gc::Col2im) -> Result<(), String> {
            let g = self.g;
            let mut a = vec![self.addr(src, src_off), self.addr(dst, 0), Arg::f(p.alpha), Arg::f(p.beta)];
            let cg = g.c / g.groups;
            for v in [
                p.total, p.n0, p.nl, cg, g.h, g.w, g.r, g.s, g.p, g.q, g.pad_h, g.pad_w, g.str_h, g.str_w, g.dil_h, g.dil_w, p.order,
                g.xs[0], g.xs[1], g.xs[2], g.xs[3], p.col_strides[0], p.col_strides[1], p.col_strides[2], p.col_strides[3], g.c,
            ] {
                a.push(u64::from(v));
            }
            run(&pg::generate_col2im(self.e, p.c_fast), &pg::col2im_entry(self.e, p.c_fast), self.gpu, [p.total.div_ceil(64), 1, 1], [64, 1, 1], a);
            Ok(())
        }
        fn permute(&mut self, src: Buf, so: usize, ss: [u32; 4], dst: Buf, dof: usize, ds: [u32; 4], dims: [u32; 4]) -> Result<(), String> {
            let total: u32 = dims.iter().product();
            let mut a = vec![self.addr(src, so), self.addr(dst, dof), Arg::f(1.0), Arg::f(0.0)];
            for v in dims.iter().chain(&ss).chain(&ds) {
                a.push(u64::from(*v));
            }
            a.push(u64::from(total));
            run(&pg::generate_add_tensor(self.e), &pg::add_tensor_entry(self.e), self.gpu, [total.div_ceil(64), 1, 1], [64, 1, 1], a);
            Ok(())
        }
        fn gemm(&mut self, q: &gc::Gemm) -> Result<(), String> {
            self.gemms += 1;
            let esz = 1usize << self.e.shift();
            for z in 0..q.batch as usize {
                let (a0, b0, c0) = (
                    self.addr(q.a.0, q.a.1 + z * q.sa),
                    self.addr(q.b.0, q.b.1 + z * q.sb),
                    self.addr(q.c.0, q.c.1 + z * q.sc),
                );
                let (m, n, k) = (q.m as usize, q.n as usize, q.k as usize);
                let rd = |gpu: &Gpu, base: u64, idx: usize| decode(self.e, &gpu.download(base + (idx * esz) as u64, esz))[0];
                for j in 0..n {
                    for i in 0..m {
                        let mut acc = 0.0f32;
                        for l in 0..k {
                            let av = if q.ta { rd(self.gpu, a0, l + i * q.lda as usize) } else { rd(self.gpu, a0, i + l * q.lda as usize) };
                            let bv = if q.tb { rd(self.gpu, b0, j + l * q.ldb as usize) } else { rd(self.gpu, b0, l + j * q.ldb as usize) };
                            acc += av * bv;
                        }
                        let ci = i + j * q.ldc as usize;
                        let mut v = q.alpha * acc;
                        if q.beta != 0.0 {
                            v += q.beta * rd(self.gpu, c0, ci);
                        }
                        let bytes = encode(self.e, &[v]);
                        for (o, b) in bytes.iter().enumerate() {
                            self.gpu.mem.write(c0 + (ci * esz + o) as u64, 1, u64::from(*b));
                        }
                    }
                }
            }
            Ok(())
        }
    }

    pub fn geom(nhwc_x: bool, nhwc_y: bool, w_order: u32, groups: u32, stride: u32, dil: u32, pad: u32, n: u32) -> ConvGeom {
        let (c, h, w, k, r, s) = (16u32, 5u32, 4u32, 16u32, 3u32, 2u32);
        let p = rm::out_dim(h, pad, r, dil, stride).unwrap();
        let q = rm::out_dim(w, pad, s, dil, stride).unwrap();
        let cg = c / groups;
        let xs = if nhwc_x { ConvGeom::packed_nhwc(c, h, w) } else { ConvGeom::packed_nchw(c, h, w) };
        let ys = if nhwc_y { ConvGeom::packed_nhwc(k, p, q) } else { ConvGeom::packed_nchw(k, p, q) };
        let ws = if w_order == 1 { [r * s * cg, 1, s * cg, cg] } else { [cg * r * s, r * s, s, 1] };
        ConvGeom { n, c, h, w, k, r, s, p, q, pad_h: pad, pad_w: pad, str_h: stride, str_w: stride, dil_h: dil, dil_w: dil, groups, flip: false, xs, ws, ys }
    }

    pub fn check(g: &ConvGeom, kind: Kind, e: Elem, tol: f32, ws_images: u32) {
        let (x, w, y0) = (sample(g.x_len(), 1), sample(g.w_len(), 2), sample(g.y_len(), 3));
        let (alpha, beta) = (1.5f32, 0.5f32);
        let rt = |v: &[f32]| decode(e, &encode(e, v));
        let (x, w, y0) = (rt(&x), rt(&w), rt(&y0));
        let (a_host, b_host, out_host, out_len) = match kind {
            Kind::Fwd => (x.clone(), w.clone(), y0.clone(), g.y_len()),
            Kind::BwdData => (y0.clone(), w.clone(), x.clone(), g.x_len()),
            Kind::BwdFilter => (x.clone(), y0.clone(), w.clone(), g.w_len()),
        };
        let mut gpu = Gpu::default();
        let a = gpu.upload(&encode(e, &a_host));
        let b = gpu.upload(&encode(e, &b_host));
        let out = gpu.upload(&encode(e, &out_host));
        let full = gc::workspace_bytes(g, kind, e);
        assert!(full > 0, "no workspace for {g:?}");
        // `ws_images` > 0 restricts the workspace to that many images to exercise chunking
        let ws_len = if ws_images > 0 { full / g.n as usize * ws_images as usize } else { full };
        let ws = gpu.alloc(ws_len + 4096);
        let mut ex = SimExec { gpu: &mut gpu, e, g: *g, ops: [a, b, out], ws, gemms: 0 };
        let used = gc::run(&mut ex, g, kind, e, alpha, beta, ws_len).unwrap();
        assert!(used, "path not taken for {g:?}");
        let got = decode(e, &gpu.download(out, out_len * (1 << e.shift())));
        let mut want = out_host.clone();
        match kind {
            Kind::Fwd => rm::conv_fwd(g, &a_host, &b_host, &mut want, alpha, beta),
            Kind::BwdData => rm::conv_bwd_data(g, &b_host, &a_host, &mut want, alpha, beta),
            Kind::BwdFilter => rm::conv_bwd_filter(g, &a_host, &b_host, &mut want, alpha, beta),
        }
        for (i, (g_, w_)) in got.iter().zip(&rt(&want)).enumerate() {
            assert!((g_ - w_).abs() <= tol * (1.0 + w_.abs()), "{kind:?} {e:?} elem {i}: got {g_} want {w_} ({g:?})");
        }
    }
}

#[test]
fn sim_gemm_path_all_layouts_f32() {
    // (x nhwc, y nhwc, filter order, groups, stride, dilation, pad, n)
    let cases = [
        (false, false, 0, 1, 1, 1, 1, 2),
        (true, true, 1, 1, 1, 1, 1, 2),
        (false, true, 1, 1, 2, 1, 0, 3),
        (true, false, 0, 1, 1, 2, 2, 2),
        (false, false, 0, 2, 1, 1, 1, 2),
        (true, true, 1, 2, 2, 1, 1, 3),
    ];
    for (xn, yn, ord, gr, st, dil, pad, n) in cases {
        let g = gemm_path::geom(xn, yn, ord, gr, st, dil, pad, n);
        assert!(g.validate(), "{g:?}");
        for kind in [Kind::Fwd, Kind::BwdData, Kind::BwdFilter] {
            gemm_path::check(&g, kind, Elem::F32, 1e-4, 0);
            if n > 2 {
                gemm_path::check(&g, kind, Elem::F32, 1e-4, 2); // chunks of 2 images, last one partial
            }
        }
    }
}

#[test]
fn sim_gemm_path_half_types() {
    let g = gemm_path::geom(true, true, 1, 1, 1, 1, 1, 2);
    let g2 = gemm_path::geom(false, false, 0, 1, 2, 1, 1, 2);
    for e in [Elem::F16, Elem::Bf16] {
        for gg in [&g, &g2] {
            for kind in [Kind::Fwd, Kind::BwdData, Kind::BwdFilter] {
                gemm_path::check(gg, kind, e, 3e-2, 0);
            }
        }
    }
}

#[test]
fn sim_bwd_filter_reduction_kernel() {
    for e in [Elem::F32, Elem::F16] {
        for (nhwc, groups) in [(false, 1), (true, 4)] {
            let g = conv_geom(false, nhwc, groups, 1, 1, 1);
            let (x, dy, w0) = (sample(g.x_len(), 1), sample(g.y_len(), 3), sample(g.w_len(), 2));
            let rt = |v: &[f32]| decode(e, &encode(e, v));
            let (x, dy, w0) = (rt(&x), rt(&dy), rt(&w0));
            let mut gpu = Gpu::default();
            let (a, b, out) = (gpu.upload(&encode(e, &x)), gpu.upload(&encode(e, &dy)), gpu.upload(&encode(e, &w0)));
            let mut args = vec![a, b, out, Arg::f(1.5), Arg::f(0.5)];
            args.extend(g.kernel_args(Kind::BwdFilter).iter().map(|v| u64::from(*v)));
            let total = g.total(Kind::BwdFilter);
            run(&pg::generate_conv_bwd_filter_red(e), &pg::conv_bwd_filter_red_entry(e), &mut gpu, [total, 1, 1], [256, 1, 1], args);
            let got = decode(e, &gpu.download(out, g.w_len() * (1 << e.shift())));
            let mut want = w0.clone();
            rm::conv_bwd_filter(&g, &x, &dy, &mut want, 1.5, 0.5);
            for (i, (g_, w_)) in got.iter().zip(&rt(&want)).enumerate() {
                assert!((g_ - w_).abs() <= 1e-2 * (1.0 + w_.abs()), "{e:?} elem {i}: got {g_} want {w_}");
            }
        }
    }
}

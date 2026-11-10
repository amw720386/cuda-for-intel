//! cuFFT shim on the GPU (`cufft64_11.dll`): C2C/R2C/C2R in f32 and Z2Z/D2Z/Z2D in f64,
//! power-of-two, mixed-radix and prime sizes, 1D/2D/3D, batched, in-place and
//! out-of-place, the torch path (`cufftXtMakePlanMany` + user work area + stream +
//! `cufftXtExec`) and an advanced layout, each against a host f64 DFT.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::{null, null_mut};

type C = (f64, f64);

/// Separable DFT of a dense `[batch][n...]` complex array, along every axis, in f64.
fn dft(n: &[usize], batch: usize, x: &[C], inverse: bool) -> Vec<C> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut a = x.to_vec();
    let per: usize = n.iter().product();
    for (d, &len) in n.iter().enumerate() {
        let inner: usize = n[d + 1..].iter().product();
        let w: Vec<C> = (0..len)
            .map(|t| {
                let ang = sign * 2.0 * std::f64::consts::PI * t as f64 / len as f64;
                (ang.cos(), ang.sin())
            })
            .collect();
        let mut out = vec![(0.0, 0.0); a.len()];
        for b in 0..batch * per / (len * inner) {
            for i in 0..inner {
                let base = b * len * inner + i;
                for k in 0..len {
                    let mut acc = (0.0, 0.0);
                    for j in 0..len {
                        let (wr, wi) = w[(j * k) % len];
                        let v = a[base + j * inner];
                        acc.0 += v.0 * wr - v.1 * wi;
                        acc.1 += v.0 * wi + v.1 * wr;
                    }
                    out[base + k * inner] = acc;
                }
            }
        }
        a = out;
    }
    a
}

fn sample(len: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2001) as f64 / 1000.0 - 1.0
        })
        .collect()
}

fn rel_err(got: &[C], want: &[C]) -> f64 {
    let scale = want.iter().map(|v| v.0.hypot(v.1)).fold(1e-30, f64::max);
    got.iter().zip(want).map(|(x, y)| (x.0 - y.0).hypot(x.1 - y.1)).fold(0.0, f64::max) / scale
}

fn pairs(v: &[f64]) -> Vec<C> {
    v.chunks(2).map(|c| (c[0], c[1])).collect()
}

#[derive(Clone, Copy, PartialEq)]
enum K {
    C2C,
    R2C,
    C2R,
}

struct Api {
    lib: libloading::Library,
}

type PlanMany = unsafe extern "system" fn(*mut i32, i32, *const i32, *const i32, i32, i32, *const i32, i32, i32, i32, i32) -> i32;
type Exec2 = unsafe extern "system" fn(i32, *mut c_void, *mut c_void) -> i32;
type Exec3 = unsafe extern "system" fn(i32, *mut c_void, *mut c_void, i32) -> i32;
type XtMake = unsafe extern "system" fn(
    i32, i32, *const i64, *const i64, i64, i64, i32, *const i64, i64, i64, i32, i64, *mut usize, i32,
) -> i32;

impl Api {
    fn f<T: Copy>(&self, name: &str) -> T {
        *unsafe { self.lib.get::<T>(format!("{name}\0").as_bytes()) }.unwrap_or_else(|_| panic!("{name}"))
    }
}

fn type_code(k: K, double: bool) -> i32 {
    match (k, double) {
        (K::C2C, false) => 0x29,
        (K::R2C, false) => 0x2a,
        (K::C2R, false) => 0x2c,
        (K::C2C, true) => 0x69,
        (K::R2C, true) => 0x6a,
        (K::C2R, true) => 0x6c,
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll, cufft64_11.dll and an Intel Level Zero GPU"]
fn hw_cufft_transforms_match_host_dft() {
    let dir = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    let api = Api { lib: unsafe { libloading::Library::new(dir.join("cufft64_11.dll")) }.expect("cufft64_11.dll") };
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemFree_v2(u64),
        cuCtxSynchronize(),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuStreamDestroy_v2(Handle),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));

    let plan_many: PlanMany = api.f("cufftPlanMany");
    let destroy: unsafe extern "system" fn(i32) -> i32 = api.f("cufftDestroy");
    let enc = |double: bool, v: &[f64]| -> Vec<u8> {
        if double { v.iter().flat_map(|x| x.to_le_bytes()).collect() } else { v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect() }
    };
    let dec = |double: bool, b: &[u8]| -> Vec<f64> {
        if double {
            b.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
        } else {
            b.chunks(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap()))).collect()
        }
    };
    // Uploads `input` scalars, runs `exec` on (in, out) device pointers, returns `out_len` scalars.
    let device_run = |double: bool, input: &[f64], out_len: usize, in_place: bool, exec: &dyn Fn(u64, u64) -> i32| -> Vec<f64> {
        let rs = if double { 8 } else { 4 };
        let mut bytes = enc(double, input);
        if in_place {
            bytes.resize(bytes.len().max(out_len * rs), 0);
        }
        let (mut din, mut dout) = (0u64, 0u64);
        ok!(cuMemAlloc_v2(&mut din, bytes.len()));
        ok!(cuMemcpyHtoD_v2(din, bytes.as_ptr().cast(), bytes.len()));
        if in_place {
            dout = din;
        } else {
            ok!(cuMemAlloc_v2(&mut dout, out_len * rs));
        }
        assert_eq!(exec(din, dout), 0, "exec");
        ok!(cuCtxSynchronize());
        let mut back = vec![0u8; out_len * rs];
        ok!(cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), dout, back.len()));
        ok!(cuMemFree_v2(din));
        if !in_place {
            ok!(cuMemFree_v2(dout));
        }
        dec(double, &back)
    };

    let mut cases = 0;
    // --- complex-to-complex, basic layout
    let c2c: &[(&[i32], i32, bool, bool)] = &[
        (&[256], 3, false, false),
        (&[4096], 2, false, false),
        (&[1000], 2, false, true),
        (&[105], 4, false, false),
        (&[97], 3, false, false),
        (&[64, 48], 1, false, false),
        (&[8, 12, 10], 2, false, true),
        (&[512], 2, true, false),
        (&[360], 3, true, true),
        (&[4099], 1, true, false),
        (&[30, 14], 2, true, false),
        (&[6, 10, 9], 1, true, false),
        // Bluestein (prime factor > 63; chirp-z convolution of a power-of-two length)
        (&[10007], 1, false, false),
        (&[4099], 2, false, true),
        (&[2062], 1, true, false),
        (&[131, 6], 1, false, false),
        // four-step (longer than one shared-memory pass: 2048 f32, 1024 f64)
        (&[16384], 1, false, false),
        (&[8192], 1, true, true),
        (&[3, 6000], 1, false, false),
    ];
    for &(n, batch, double, in_place) in c2c {
        let nu: Vec<usize> = n.iter().map(|&v| v as usize).collect();
        let total = batch as usize * nu.iter().product::<usize>();
        let mut x = sample(2 * total, total as u64);
        if !double {
            x = x.iter().map(|v| f64::from(*v as f32)).collect();
        }
        let mut plan = 0;
        assert_eq!(unsafe { plan_many(&mut plan, n.len() as i32, n.as_ptr(), null(), 1, 0, null(), 1, 0, type_code(K::C2C, double), batch) }, 0);
        let exec: Exec3 = api.f(if double { "cufftExecZ2Z" } else { "cufftExecC2C" });
        for dir in [-1, 1] {
            let got = device_run(double, &x, 2 * total, in_place, &|i, o| unsafe { exec(plan, i as *mut c_void, o as *mut c_void, dir) });
            let e = rel_err(&pairs(&got), &dft(&nu, batch as usize, &pairs(&x), dir == 1));
            let tol = if double { 1e-12 } else if total > 4096 { 2e-5 } else { 1e-5 };
            println!("  C2C {n:?} batch {batch} {} dir {dir} in_place {in_place}: rel err {e:.2e}", if double { "f64" } else { "f32" });
            assert!(e < tol, "C2C {n:?} double={double} dir={dir}: {e}");
            cases += 1;
        }
        assert_eq!(unsafe { destroy(plan) }, 0);
    }

    // --- real transforms, basic layout: R2C then C2R round trip
    let real: &[(&[i32], i32, bool, bool)] = &[
        (&[512], 3, false, false),
        (&[100], 2, false, true),
        (&[32, 48], 2, false, false),
        (&[6, 8, 10], 1, false, true),
        (&[1024], 2, true, true),
        (&[75], 3, true, false),
        (&[12, 20], 1, true, true),
    ];
    for &(n, batch, double, in_place) in real {
        let nu: Vec<usize> = n.iter().map(|&v| v as usize).collect();
        let last = *nu.last().unwrap();
        let m2 = last / 2 + 1;
        let total = batch as usize * nu.iter().product::<usize>();
        let rows = total / last;
        let mut x = sample(total, 17 + total as u64);
        if !double {
            x = x.iter().map(|v| f64::from(*v as f32)).collect();
        }
        let full = dft(&nu, batch as usize, &x.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(), false);
        let want: Vec<C> = full.iter().enumerate().filter(|(i, _)| i % last < m2).map(|(_, v)| *v).collect();
        let (mut fwd, mut inv) = (0, 0);
        assert_eq!(unsafe { plan_many(&mut fwd, n.len() as i32, n.as_ptr(), null(), 1, 0, null(), 1, 0, type_code(K::R2C, double), batch) }, 0);
        assert_eq!(unsafe { plan_many(&mut inv, n.len() as i32, n.as_ptr(), null(), 1, 0, null(), 1, 0, type_code(K::C2R, double), batch) }, 0);
        let r2c: Exec2 = api.f(if double { "cufftExecD2Z" } else { "cufftExecR2C" });
        let c2r: Exec2 = api.f(if double { "cufftExecZ2D" } else { "cufftExecC2R" });
        // in place: real rows padded to 2 * m2 scalars
        let input: Vec<f64> = if in_place {
            let mut p = vec![0.0; rows * 2 * m2];
            for r in 0..rows {
                p[r * 2 * m2..r * 2 * m2 + last].copy_from_slice(&x[r * last..(r + 1) * last]);
            }
            p
        } else {
            x.clone()
        };
        let y = device_run(double, &input, 2 * rows * m2, in_place, &|i, o| unsafe { r2c(fwd, i as *mut c_void, o as *mut c_void) });
        let e = rel_err(&pairs(&y), &want);
        let tol = if double { 1e-12 } else { 1e-5 };
        let p = if double { "f64" } else { "f32" };
        println!("  R2C {n:?} batch {batch} {p} in_place {in_place}: rel err {e:.2e}");
        assert!(e < tol, "R2C {n:?} double={double}: {e}");
        let out_len = if in_place { rows * 2 * m2 } else { total };
        let z = device_run(double, &y, out_len, in_place, &|i, o| unsafe { c2r(inv, i as *mut c_void, o as *mut c_void) });
        let per = (total / batch as usize) as f64;
        let mut e2: f64 = 0.0;
        for r in 0..rows {
            for c in 0..last {
                let got = if in_place { z[r * 2 * m2 + c] } else { z[r * last + c] };
                e2 = e2.max((got / per - x[r * last + c]).abs());
            }
        }
        println!("  C2R {n:?} batch {batch} {p} in_place {in_place}: round-trip err {e2:.2e}");
        assert!(e2 < tol * 10.0, "C2R {n:?} double={double}: {e2}");
        cases += 2;
        assert_eq!(unsafe { destroy(fwd) }, 0);
        assert_eq!(unsafe { destroy(inv) }, 0);
    }

    // --- torch's path: cufftCreate, no auto allocation, XtMakePlanMany with an advanced
    // layout (transform over a non-last dim: istride = batch, idist = 1), user work area,
    // a non-default stream, cufftXtExec.
    {
        let create: unsafe extern "system" fn(*mut i32) -> i32 = api.f("cufftCreate");
        let set_auto: unsafe extern "system" fn(i32, i32) -> i32 = api.f("cufftSetAutoAllocation");
        let set_work: unsafe extern "system" fn(i32, *mut c_void) -> i32 = api.f("cufftSetWorkArea");
        let set_stream: unsafe extern "system" fn(i32, Handle) -> i32 = api.f("cufftSetStream");
        let xt_make: XtMake = api.f("cufftXtMakePlanMany");
        let xt_exec: Exec3 = api.f("cufftXtExec");
        let (n, batch) = (200usize, 5usize);
        let mut plan = 0;
        assert_eq!(unsafe { create(&mut plan) }, 0);
        assert_eq!(unsafe { set_auto(plan, 0) }, 0);
        let (nn, m2) = ([n as i64], (n / 2 + 1) as i64);
        let mut ws = 0usize;
        // R2C f32: real input [n][batch] (interleaved), complex output [batch][m2]
        assert_eq!(unsafe { xt_make(plan, 1, nn.as_ptr(), nn.as_ptr(), batch as i64, 1, 0, [m2].as_ptr(), 1, m2, 4, batch as i64, &mut ws, 4) }, 0);
        assert!(ws > 0);
        let mut work = 0u64;
        ok!(cuMemAlloc_v2(&mut work, ws));
        assert_eq!(unsafe { set_work(plan, work as *mut c_void) }, 0);
        let mut stream = null_mut();
        ok!(cuStreamCreate(&mut stream, 1));
        assert_eq!(unsafe { set_stream(plan, stream) }, 0);
        let x: Vec<f64> = sample(n * batch, 99).iter().map(|v| f64::from(*v as f32)).collect();
        let xb = enc(false, &x);
        let (mut din, mut dout) = (0u64, 0u64);
        ok!(cuMemAlloc_v2(&mut din, xb.len()));
        ok!(cuMemAlloc_v2(&mut dout, batch * m2 as usize * 8));
        ok!(cuMemcpyHtoD_v2(din, xb.as_ptr().cast(), xb.len()));
        assert_eq!(unsafe { xt_exec(plan, din as *mut c_void, dout as *mut c_void, -1) }, 0);
        ok!(cuStreamSynchronize(stream));
        let mut back = vec![0u8; batch * m2 as usize * 8];
        ok!(cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), dout, back.len()));
        let got = pairs(&dec(false, &back));
        let mut worst: f64 = 0.0;
        for b in 0..batch {
            let xs: Vec<C> = (0..n).map(|i| (x[i * batch + b], 0.0)).collect();
            let want = dft(&[n], 1, &xs, false);
            worst = worst.max(rel_err(&got[b * m2 as usize..(b + 1) * m2 as usize], &want[..m2 as usize]));
        }
        println!("  XtMakePlanMany/XtExec R2C n {n} batch {batch} strided input, user work area, stream: rel err {worst:.2e}");
        assert!(worst < 1e-5);
        // the plan's exec type is fixed: a mismatched exec function is rejected
        let z2z: Exec3 = api.f("cufftExecZ2Z");
        assert_eq!(unsafe { z2z(plan, din as *mut c_void, dout as *mut c_void, -1) }, 3, "INVALID_TYPE");
        assert_eq!(unsafe { destroy(plan) }, 0);
        ok!(cuStreamDestroy_v2(stream));
        for p in [din, dout, work] {
            ok!(cuMemFree_v2(p));
        }
        cases += 1;
    }

    // --- cufftPlan1d/2d/3d entry points (default auto allocation)
    {
        let plan1d: unsafe extern "system" fn(*mut i32, i32, i32, i32) -> i32 = api.f("cufftPlan1d");
        let plan2d: unsafe extern "system" fn(*mut i32, i32, i32, i32) -> i32 = api.f("cufftPlan2d");
        let plan3d: unsafe extern "system" fn(*mut i32, i32, i32, i32, i32) -> i32 = api.f("cufftPlan3d");
        let c2c: Exec3 = api.f("cufftExecC2C");
        for (dims, batch) in [(vec![48usize], 2usize), (vec![16, 20], 1), (vec![4, 6, 8], 1)] {
            let mut plan = 0;
            let d: Vec<i32> = dims.iter().map(|&v| v as i32).collect();
            let st = unsafe {
                match d.len() {
                    1 => plan1d(&mut plan, d[0], 0x29, batch as i32),
                    2 => plan2d(&mut plan, d[0], d[1], 0x29),
                    _ => plan3d(&mut plan, d[0], d[1], d[2], 0x29),
                }
            };
            assert_eq!(st, 0);
            let total = batch * dims.iter().product::<usize>();
            let x: Vec<f64> = sample(2 * total, 5).iter().map(|v| f64::from(*v as f32)).collect();
            let got = device_run(false, &x, 2 * total, false, &|i, o| unsafe { c2c(plan, i as *mut c_void, o as *mut c_void, -1) });
            let e = rel_err(&pairs(&got), &dft(&dims, batch, &pairs(&x), false));
            println!("  cufftPlan{}d {dims:?}: rel err {e:.2e}", dims.len());
            assert!(e < 1e-5);
            assert_eq!(unsafe { destroy(plan) }, 0);
            cases += 1;
        }
    }
    ok!(cuCtxDestroy_v2(ctx));
    println!("cufft: {cases} transforms match the host DFT");
}

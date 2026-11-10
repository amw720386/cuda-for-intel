//! PTX for the shim's f64 and complex BLAS (and the f32 level-1/2 routines that need
//! strides). The Driver API translates it to SPIR-V like any application PTX, the route the
//! f32/f16 GEMM kernels in `../ptxgen.rs` take; those kernels are not touched by this file.
//!
//! One module per element type [`Ty`] holds:
//! * `gemms`/`gemmp` - 16x16 shared-memory tiled `C = alpha op(A) op(B) + beta C`, one
//!   output per thread, accumulation in the element precision. op() is reduced on the host
//!   to element strides plus a conjugate flag per operand, so one kernel covers N, T, C and
//!   CONJG. `s` reads batch bases as `param + z * stride`, `p` from device pointer tables.
//! * `gemv` - one thread per element of y, any (signed) increments.
//! * `dot` - one 256-thread block, grid-stride partial sums and a shared-memory tree;
//!   optional conjugation of x (`dotc`). Writes the result to a device address.
//! * `nrm2` - one block: max |component| pass, then a scaled sum of squares, so the result
//!   neither overflows nor underflows.
//! * `axpy`, `scal` - one thread per element.
//! * `trsms`/`trsmp` - in-place triangular solve, one thread per right-hand-side vector.
//!
//! The `h` and `b` (f16, bf16) modules hold `dot` only, accumulating in f32 (`cublasDotEx`).
//!
//! Every parameter is a `.u64` slot (the launcher passes 8-byte values). Scalars travel as
//! their bit patterns (an f32 in the low word); a nonzero `*_ptr` parameter makes the kernel
//! load the scalar from device memory instead (`CUBLAS_POINTER_MODE_DEVICE`), so no host
//! round trip is needed. Increments and strides are in elements and may be negative (two's
//! complement); the host pre-offsets base pointers the way BLAS defines negative increments.
//! A zero alpha never multiplies A, and a zero beta never reads C or y, so NaNs there do not
//! leak into the result (BLAS semantics). No thread leaves before a barrier.

use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    S,
    D,
    C,
    Z,
    H,
    B,
}

impl Ty {
    pub const BLAS: [Ty; 4] = [Ty::S, Ty::D, Ty::C, Ty::Z];
    pub const ALL: [Ty; 6] = [Ty::S, Ty::D, Ty::C, Ty::Z, Ty::H, Ty::B];

    pub fn complex(self) -> bool {
        matches!(self, Ty::C | Ty::Z)
    }
    pub fn double(self) -> bool {
        matches!(self, Ty::D | Ty::Z)
    }
    /// PTX type of the arithmetic.
    pub fn ft(self) -> &'static str {
        if self.double() { "f64" } else { "f32" }
    }
    /// Bytes of one real component of the arithmetic type.
    pub fn comp(self) -> u32 {
        if self.double() { 8 } else { 4 }
    }
    /// Bytes of one stored element.
    pub fn size(self) -> u32 {
        match self {
            Ty::S => 4,
            Ty::D | Ty::C => 8,
            Ty::Z => 16,
            Ty::H | Ty::B => 2,
        }
    }
    /// Bytes of one element in the arithmetic type (shared-memory slot).
    pub fn slot(self) -> u32 {
        self.comp() * if self.complex() { 2 } else { 1 }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Ty::S => "s",
            Ty::D => "d",
            Ty::C => "c",
            Ty::Z => "z",
            Ty::H => "h",
            Ty::B => "b",
        }
    }
    fn zero(self) -> &'static str {
        if self.double() {
            "0d0000000000000000"
        } else {
            "0f00000000"
        }
    }
    fn one(self) -> &'static str {
        if self.double() {
            "0d3FF0000000000000"
        } else {
            "0f3F800000"
        }
    }
    fn inf(self) -> &'static str {
        if self.double() {
            "0d7FF0000000000000"
        } else {
            "0f7F800000"
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kernel {
    GemmS,
    GemmP,
    Gemv,
    Dot,
    Nrm2,
    Axpy,
    Scal,
    TrsmS,
    TrsmP,
    Copy,
    Amax,
    Geam,
    Getrf,
    Laswp,
    GemmBS,
    GemmBP,
    Nrm2P,
    Geqrf,
    GetrfT,
}

/// Threads of the one-dimensional kernels (`gemv`, `dot`, `nrm2`, `axpy`, `scal`).
pub const BLOCK: u32 = 256;
/// Threads of `trsm` blocks.
pub const TRSM_BLOCK: u32 = 64;
/// GEMM tile edge (blocks are TILE x TILE).
pub const TILE: u32 = 16;
/// Threads of the one-work-group-per-matrix factorizations (`getrf`, `geqrf`).
pub const LU_BLOCK: u32 = 64;

/// Register block of the `gemmb*` kernels: each of the 16 x 16 threads computes
/// `rb x rb` outputs, so a work-group covers a `16 rb` square of C.
pub fn gemm_rb(ty: Ty) -> u32 {
    let o = GEMM_RB.load(std::sync::atomic::Ordering::Relaxed);
    if o != 0 {
        return o;
    }
    match ty {
        Ty::Z => 2,
        _ => 4,
    }
}

/// Override of [`gemm_rb`] for every type (0: the defaults); set before the kernels load.
pub static GEMM_RB: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

impl Kernel {
    pub const BLAS: [Kernel; 19] = [
        Kernel::GemmS,
        Kernel::GemmP,
        Kernel::Gemv,
        Kernel::Dot,
        Kernel::Nrm2,
        Kernel::Axpy,
        Kernel::Scal,
        Kernel::TrsmS,
        Kernel::TrsmP,
        Kernel::Copy,
        Kernel::Amax,
        Kernel::Geam,
        Kernel::Getrf,
        Kernel::Laswp,
        Kernel::GemmBS,
        Kernel::GemmBP,
        Kernel::Nrm2P,
        Kernel::Geqrf,
        Kernel::GetrfT,
    ];

    pub fn tag(self) -> &'static str {
        match self {
            Kernel::GemmS => "gemms",
            Kernel::GemmP => "gemmp",
            Kernel::Gemv => "gemv",
            Kernel::Dot => "dot",
            Kernel::Nrm2 => "nrm2",
            Kernel::Axpy => "axpy",
            Kernel::Scal => "scal",
            Kernel::TrsmS => "trsms",
            Kernel::TrsmP => "trsmp",
            Kernel::Copy => "copy",
            Kernel::Amax => "amax",
            Kernel::Geam => "geam",
            Kernel::Getrf => "getrf",
            Kernel::Laswp => "laswp",
            Kernel::GemmBS => "gemmbs",
            Kernel::GemmBP => "gemmbp",
            Kernel::Nrm2P => "nrm2p",
            Kernel::Geqrf => "geqrf",
            Kernel::GetrfT => "getrft",
        }
    }

    pub fn params(self) -> usize {
        match self {
            Kernel::GemmS | Kernel::GemmP | Kernel::GemmBS | Kernel::GemmBP => 31,
            Kernel::Gemv | Kernel::TrsmS | Kernel::TrsmP => 16,
            Kernel::Dot | Kernel::Scal => 7,
            Kernel::Nrm2 => 6,
            Kernel::Nrm2P => 4,
            Kernel::Axpy => 8,
            Kernel::Copy => 6,
            Kernel::Amax => 6,
            Kernel::Geam => 21,
            Kernel::Getrf | Kernel::GetrfT => 7,
            Kernel::Laswp => 6,
            Kernel::Geqrf => 10,
        }
    }

    /// Block shape of a launch.
    pub fn block(self) -> [u32; 3] {
        match self {
            Kernel::GemmS | Kernel::GemmP | Kernel::GemmBS | Kernel::GemmBP => [TILE, TILE, 1],
            Kernel::TrsmS | Kernel::TrsmP | Kernel::Laswp | Kernel::GetrfT => [TRSM_BLOCK, 1, 1],
            Kernel::Getrf | Kernel::Geqrf => [LU_BLOCK, 1, 1],
            _ => [BLOCK, 1, 1],
        }
    }
}

/// Kernels in the module of `ty`.
pub fn kernels(ty: Ty) -> &'static [Kernel] {
    match ty {
        Ty::H | Ty::B => &[Kernel::Dot],
        _ => &Kernel::BLAS,
    }
}

pub fn entry_name(k: Kernel, ty: Ty) -> String {
    format!("cfi_bx_{}_{}", k.tag(), ty.tag())
}

const HEAD: &str = ".version 7.0\n.target sm_60\n.address_size 64\n\n";

/// The PTX module holding every kernel of `ty`.
pub fn module(ty: Ty) -> String {
    let mut s = String::from(HEAD);
    for &k in kernels(ty) {
        s += &body(k, ty);
        s.push('\n');
    }
    s
}

/// One kernel as a stand-alone module (tests).
pub fn single(k: Kernel, ty: Ty) -> String {
    format!("{HEAD}{}", body(k, ty))
}

/// (name, stand-alone source, parameter count) of every kernel of every type.
pub fn all_entries() -> Vec<(String, String, usize)> {
    let mut v = Vec::new();
    for ty in Ty::ALL {
        for &k in kernels(ty) {
            v.push((entry_name(k, ty), single(k, ty), k.params()));
        }
    }
    v
}

fn body(k: Kernel, ty: Ty) -> String {
    match k {
        Kernel::GemmS => gemm(ty, false),
        Kernel::GemmP => gemm(ty, true),
        Kernel::Gemv => gemv(ty),
        Kernel::Dot => dot(ty),
        Kernel::Nrm2 => nrm2(ty),
        Kernel::Axpy => axpy(ty),
        Kernel::Scal => scal(ty),
        Kernel::TrsmS => trsm(ty, false),
        Kernel::TrsmP => trsm(ty, true),
        Kernel::Copy => copy(ty),
        Kernel::Amax => amax(ty),
        Kernel::Geam => geam(ty),
        Kernel::Getrf => getrf(ty),
        Kernel::Laswp => laswp(ty),
        Kernel::GemmBS => gemm_blocked(ty, false),
        Kernel::GemmBP => gemm_blocked(ty, true),
        Kernel::Nrm2P => nrm2_partial(ty),
        Kernel::Geqrf => geqrf(ty),
        Kernel::GetrfT => getrf_thread(ty),
    }
}

/// A complex (or real) value held in `%f<re>` and `%f<re+1>` (the second unused for reals).
#[derive(Clone, Copy)]
struct V(u32);

/// Kernel text builder. Parameter `i` is loaded into `%rd<i>`; `%f60` holds zero and
/// `%f61..%f63` are scratch of the arithmetic helpers.
struct K {
    s: String,
    ty: Ty,
}

impl K {
    fn new(name: &str, nparams: usize, ty: Ty, shared: u32) -> K {
        let p: Vec<String> = (0..nparams)
            .map(|i| format!(".param .u64 {name}_p{i}"))
            .collect();
        let mut s = format!(".visible .entry {name}(\n{}\n)\n{{\n", p.join(",\n"));
        let t = ty.ft();
        let _ = writeln!(
            s,
            ".reg .pred %p<24>;\n.reg .b16 %rs<4>;\n.reg .b32 %r<64>;\n.reg .b64 %rd<128>;\n.reg .{t} %f<128>;"
        );
        if shared > 0 {
            let _ = writeln!(s, ".shared .align 16 .b8 smem[{shared}];");
        }
        // kernels use %rd29 and up as scratch
        assert!(
            nparams <= 31,
            "{name}: parameters would overlap scratch registers"
        );
        for i in 0..nparams {
            let _ = writeln!(s, "ld.param.u64 %rd{i}, [{name}_p{i}];");
        }
        let _ = writeln!(s, "mov.{t} %f60, {};", ty.zero());
        K { s, ty }
    }

    fn l(&mut self, line: &str) {
        self.s += line;
        self.s.push('\n');
    }

    fn t(&self) -> &'static str {
        self.ty.ft()
    }

    fn cx(&self) -> bool {
        self.ty.complex()
    }

    /// Global load of the element at `%rd<addr>` into `v` (widened to f32 for h/b).
    fn ld(&mut self, addr: u32, v: V) {
        let (t, c) = (self.t(), self.ty.comp());
        match self.ty {
            Ty::H => self.l(&format!("ld.global.u16 %rs1, [%rd{addr}];\ncvt.f32.f16 %f{}, %rs1;", v.0)),
            Ty::B => self.l(&format!(
                "ld.global.u16 %rs1, [%rd{addr}];\ncvt.u32.u16 %r60, %rs1;\nshl.b32 %r60, %r60, 16;\nmov.b32 %f{}, %r60;",
                v.0
            )),
            _ => {
                self.l(&format!("ld.global.{t} %f{}, [%rd{addr}];", v.0));
                if self.cx() {
                    self.l(&format!("ld.global.{t} %f{}, [%rd{addr}+{c}];", v.0 + 1));
                }
            }
        }
    }

    /// Global store of `v` at `%rd<addr>` (rounded to nearest even for h/b).
    fn st(&mut self, addr: u32, v: V) {
        let (t, c) = (self.t(), self.ty.comp());
        match self.ty {
            Ty::H => self.l(&format!(
                "cvt.rn.f16.f32 %rs2, %f{};\nst.global.u16 [%rd{addr}], %rs2;",
                v.0
            )),
            Ty::B => {
                let f = v.0;
                self.l(&format!(
                    "mov.b32 %r60, %f{f};\nsetp.ne.f32 %p23, %f{f}, %f{f};\nshr.u32 %r61, %r60, 16;\nand.b32 %r61, %r61, 1;\nadd.s32 %r61, %r61, 32767;\nadd.s32 %r60, %r60, %r61;\nshr.u32 %r60, %r60, 16;\nselp.u32 %r60, 32704, %r60, %p23;\ncvt.u16.u32 %rs2, %r60;\nst.global.u16 [%rd{addr}], %rs2;"
                ));
            }
            _ => {
                self.l(&format!("st.global.{t} [%rd{addr}], %f{};", v.0));
                if self.cx() {
                    self.l(&format!("st.global.{t} [%rd{addr}+{c}], %f{};", v.0 + 1));
                }
            }
        }
    }

    /// Shared-memory load of `v` (arithmetic type) at the 32-bit address `%r<addr>`.
    fn lds(&mut self, addr: u32, v: V) {
        let (t, c) = (self.t(), self.ty.comp());
        self.l(&format!("ld.shared.{t} %f{}, [%r{addr}];", v.0));
        if self.cx() {
            self.l(&format!("ld.shared.{t} %f{}, [%r{addr}+{c}];", v.0 + 1));
        }
    }

    fn sts(&mut self, addr: u32, v: V) {
        let (t, c) = (self.t(), self.ty.comp());
        self.l(&format!("st.shared.{t} [%r{addr}], %f{};", v.0));
        if self.cx() {
            self.l(&format!("st.shared.{t} [%r{addr}+{c}], %f{};", v.0 + 1));
        }
    }

    fn zero(&mut self, v: V) {
        let t = self.t();
        self.l(&format!("mov.{t} %f{}, %f60;", v.0));
        if self.cx() {
            self.l(&format!("mov.{t} %f{}, %f60;", v.0 + 1));
        }
    }

    fn mov(&mut self, d: V, a: V) {
        let t = self.t();
        self.l(&format!("mov.{t} %f{}, %f{};", d.0, a.0));
        if self.cx() {
            self.l(&format!("mov.{t} %f{}, %f{};", d.0 + 1, a.0 + 1));
        }
    }

    /// `v = p ? v : 0`.
    fn keep_if(&mut self, v: V, p: u32) {
        let t = self.t();
        self.l(&format!("selp.{t} %f{0}, %f{0}, %f60, %p{p};", v.0));
        if self.cx() {
            self.l(&format!("selp.{t} %f{0}, %f{0}, %f60, %p{p};", v.0 + 1));
        }
    }

    /// `v = conj(v)` when `%p<p>` (no-op for reals).
    fn conj_if(&mut self, v: V, p: u32) {
        if self.cx() {
            let t = self.t();
            self.l(&format!(
                "neg.{t} %f62, %f{0};\nselp.{t} %f{0}, %f62, %f{0}, %p{p};",
                v.0 + 1
            ));
        }
    }

    fn neg(&mut self, v: V) {
        let t = self.t();
        self.l(&format!("neg.{t} %f{0}, %f{0};", v.0));
        if self.cx() {
            self.l(&format!("neg.{t} %f{0}, %f{0};", v.0 + 1));
        }
    }

    /// `acc += a * b`.
    fn fma(&mut self, acc: V, a: V, b: V) {
        let t = self.t();
        let (ar, br, cr) = (a.0, b.0, acc.0);
        self.l(&format!("fma.rn.{t} %f{cr}, %f{ar}, %f{br}, %f{cr};"));
        if self.cx() {
            let (ai, bi, ci) = (ar + 1, br + 1, cr + 1);
            self.l(&format!(
                "neg.{t} %f63, %f{ai};\nfma.rn.{t} %f{cr}, %f63, %f{bi}, %f{cr};\nfma.rn.{t} %f{ci}, %f{ar}, %f{bi}, %f{ci};\nfma.rn.{t} %f{ci}, %f{ai}, %f{br}, %f{ci};"
            ));
        }
    }

    /// `d = a * b` (`d` distinct from `a` and `b`).
    fn mul(&mut self, d: V, a: V, b: V) {
        let t = self.t();
        let (ar, br, dr) = (a.0, b.0, d.0);
        self.l(&format!("mul.rn.{t} %f{dr}, %f{ar}, %f{br};"));
        if self.cx() {
            let (ai, bi, di) = (ar + 1, br + 1, dr + 1);
            self.l(&format!(
                "neg.{t} %f63, %f{ai};\nfma.rn.{t} %f{dr}, %f63, %f{bi}, %f{dr};\nmul.rn.{t} %f{di}, %f{ar}, %f{bi};\nfma.rn.{t} %f{di}, %f{ai}, %f{br}, %f{di};"
            ));
        }
    }

    fn add(&mut self, d: V, a: V) {
        let t = self.t();
        self.l(&format!("add.rn.{t} %f{0}, %f{0}, %f{1};", d.0, a.0));
        if self.cx() {
            self.l(&format!(
                "add.rn.{t} %f{0}, %f{0}, %f{1};",
                d.0 + 1,
                a.0 + 1
            ));
        }
    }

    /// `acc /= d` (`acc`, `d` distinct; scratch `%f8..%f11` for complex values).
    fn div(&mut self, acc: V, d: V) {
        let t = self.t();
        let (a, b) = (acc.0, d.0);
        if self.cx() {
            // (a + bi) / (c + di) = ((ac + bd) + (bc - ad) i) / (c^2 + d^2)
            self.l(&format!(
                "mul.rn.{t} %f8, %f{b}, %f{b};\nfma.rn.{t} %f8, %f{b1}, %f{b1}, %f8;\nmul.rn.{t} %f9, %f{a}, %f{b};\nfma.rn.{t} %f9, %f{a1}, %f{b1}, %f9;\nmul.rn.{t} %f10, %f{a1}, %f{b};\nneg.{t} %f11, %f{a};\nfma.rn.{t} %f10, %f11, %f{b1}, %f10;\ndiv.rn.{t} %f{a}, %f9, %f8;\ndiv.rn.{t} %f{a1}, %f10, %f8;",
                a1 = a + 1,
                b1 = b + 1
            ));
        } else {
            self.l(&format!("div.rn.{t} %f{a}, %f{a}, %f{b};"));
        }
    }

    /// `%f<dst>` = |re| + |im| of `v` (BLAS `abs1`).
    fn abs1(&mut self, dst: u32, v: V) {
        let t = self.t();
        self.l(&format!("abs.{t} %f{dst}, %f{};", v.0));
        if self.cx() {
            self.l(&format!(
                "abs.{t} %f63, %f{};\nadd.rn.{t} %f{dst}, %f{dst}, %f63;",
                v.0 + 1
            ));
        }
    }

    /// `%p<p>` = (v == 0).
    fn is_zero(&mut self, p: u32, v: V) {
        let t = self.t();
        self.l(&format!("setp.eq.{t} %p{p}, %f{}, %f60;", v.0));
        if self.cx() {
            self.l(&format!(
                "setp.eq.{t} %p22, %f{}, %f60;\nand.pred %p{p}, %p{p}, %p22;",
                v.0 + 1
            ));
        }
    }

    /// Scalar from parameters `re`/`im` (bit patterns), or from device memory at `ptr` when
    /// that parameter is nonzero. With `real_only`, a nonzero parameter means the device
    /// scalar is real (`Csscal`): only its first component is loaded.
    fn scalar(&mut self, v: V, re: u32, im: u32, ptr: u32, real_only: Option<u32>, stem: &str) {
        let (t, c) = (self.t(), self.ty.comp());
        let skip = format!("$L_{stem}_host");
        for (reg, p) in [(v.0, re), (v.0 + 1, im)]
            .into_iter()
            .take(if self.cx() { 2 } else { 1 })
        {
            if self.ty.double() {
                self.l(&format!("mov.b64 %f{reg}, %rd{p};"));
            } else {
                self.l(&format!(
                    "cvt.u32.u64 %r59, %rd{p};\nmov.b32 %f{reg}, %r59;"
                ));
            }
        }
        self.l(&format!(
            "setp.eq.u64 %p20, %rd{ptr}, 0;\n@%p20 bra {skip};\ncvta.to.global.u64 %rd120, %rd{ptr};\nld.global.{t} %f{}, [%rd120];",
            v.0
        ));
        if self.cx() {
            if let Some(flag) = real_only {
                self.l(&format!(
                    "setp.ne.u64 %p21, %rd{flag}, 0;\n@%p21 bra {skip};"
                ));
            }
            self.l(&format!("ld.global.{t} %f{}, [%rd120+{c}];", v.0 + 1));
        }
        self.l(&format!("{skip}:"));
    }

    /// `%rd<dst>` = &elem[base + idx] (`%rd<idx>` in elements).
    fn elem(&mut self, dst: u32, base: u32, idx: u32) {
        let sz = self.ty.size();
        self.l(&format!("mad.lo.u64 %rd{dst}, %rd{idx}, {sz}, %rd{base};"));
    }

    /// `%rd<dst>` = global 1-D thread index for blocks of `block` threads; `%r2` = tid.x.
    fn gid(&mut self, dst: u32, block: u32) {
        self.l(&format!(
            "mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %tid.x;\nmul.wide.u32 %rd{dst}, %r1, {block};\ncvt.u64.u32 %rd121, %r2;\nadd.u64 %rd{dst}, %rd{dst}, %rd121;"
        ));
    }

    /// `%rd<dst>` = batch `%rd<z>` base of operand parameter `%rd<param>`: an entry of a
    /// device pointer table, or `param + z * stride` elements.
    fn batch_base(&mut self, dst: u32, param: u32, stride: u32, z: u32, tables: bool) {
        let sz = self.ty.size();
        if tables {
            self.l(&format!(
                "cvta.to.global.u64 %rd122, %rd{param};\nmad.lo.u64 %rd122, %rd{z}, 8, %rd122;\nld.global.u64 %rd{dst}, [%rd122];\ncvta.to.global.u64 %rd{dst}, %rd{dst};"
            ));
        } else {
            self.l(&format!(
                "cvta.to.global.u64 %rd123, %rd{param};\nmul.lo.u64 %rd122, %rd{z}, %rd{stride};\nmad.lo.u64 %rd{dst}, %rd122, {sz}, %rd123;"
            ));
        }
    }

    /// Shared tree reduction of `v` over BLOCK threads (`%r2` = tid, `%r11` = own slot,
    /// smem base in `%r10`); `max` reduces real values with `max` instead of adding.
    /// Ends with a barrier; slot 0 holds the result.
    fn tree(&mut self, v: V, scratch: V, max: bool, stem: &str) {
        let slot = if max { self.ty.comp() } else { self.ty.slot() };
        let t = self.t();
        let st = |k: &mut K, addr: u32, v: V| {
            if max {
                k.l(&format!("st.shared.{t} [%r{addr}], %f{};", v.0));
            } else {
                k.sts(addr, v);
            }
        };
        let ld = |k: &mut K, addr: u32, v: V| {
            if max {
                k.l(&format!("ld.shared.{t} %f{}, [%r{addr}];", v.0));
            } else {
                k.lds(addr, v);
            }
        };
        st(self, 11, v);
        self.l("bar.sync 0;");
        let mut s = BLOCK / 2;
        while s >= 1 {
            let skip = format!("$L_{stem}_{s}");
            self.l(&format!("setp.ge.u32 %p19, %r2, {s};\n@%p19 bra {skip};"));
            ld(self, 11, v);
            self.l(&format!("add.u32 %r12, %r11, {};", s * slot));
            ld(self, 12, scratch);
            if max {
                self.l(&format!("max.{t} %f{0}, %f{0}, %f{1};", v.0, scratch.0));
            } else {
                self.add(v, scratch);
            }
            st(self, 11, v);
            self.l(&format!("{skip}:\nbar.sync 0;"));
            s /= 2;
        }
    }

    /// Shared-memory load of `v` at `%r<addr> + off` bytes.
    fn lds_off(&mut self, addr: u32, off: u32, v: V) {
        let (t, c) = (self.t(), self.ty.comp());
        let at = |o: u32| {
            if o == 0 {
                format!("[%r{addr}]")
            } else {
                format!("[%r{addr}+{o}]")
            }
        };
        self.l(&format!("ld.shared.{t} %f{}, {};", v.0, at(off)));
        if self.cx() {
            self.l(&format!("ld.shared.{t} %f{}, {};", v.0 + 1, at(off + c)));
        }
    }

    /// `v = conj(v)` (no-op for reals).
    fn conj(&mut self, v: V) {
        if self.cx() {
            let t = self.t();
            self.l(&format!("neg.{t} %f{0}, %f{0};", v.0 + 1));
        }
    }

    /// Shared tree sum of the real `%f<r>` over `nt` threads (`%r2` = tid, `%r11` = own
    /// slot of one real component); thread 0 ends with the total in `%f<r>` and slot 0.
    /// Scratch `%f<s>`, `%r12`. Ends with a barrier.
    fn real_sum_tree(&mut self, r: u32, s: u32, nt: u32, stem: &str) {
        let (t, c) = (self.t(), self.ty.comp());
        self.l(&format!("st.shared.{t} [%r11], %f{r};\nbar.sync 0;"));
        let mut h = nt / 2;
        while h >= 1 {
            self.l(&format!(
                "setp.ge.u32 %p19, %r2, {h};\n@%p19 bra $L_{stem}_{h};\nadd.u32 %r12, %r11, {};\nld.shared.{t} %f{s}, [%r12];\nadd.rn.{t} %f{r}, %f{r}, %f{s};\nst.shared.{t} [%r11], %f{r};\n$L_{stem}_{h}:\nbar.sync 0;",
                h * c
            ));
            h /= 2;
        }
    }

    /// Replace the best (`%f0`, index `%rd40`) by the candidate (`%f<cf>`, `%rd<ci>`) when
    /// the best is none (`%rd42`), or the candidate is valid and larger (smaller when
    /// `%p7`; `%p8` = !`%p7`), or equal with a smaller index. Scratch `%p4..%p6`, `%p9..%p11`.
    fn pick(&mut self, cf: u32, ci: u32) {
        let t = self.t();
        self.l(&format!(
            "setp.eq.u64 %p4, %rd40, %rd42;\nsetp.ne.u64 %p9, %rd{ci}, %rd42;\nsetp.gt.{t} %p5, %f{cf}, %f0;\nsetp.lt.{t} %p6, %f{cf}, %f0;\nand.pred %p5, %p5, %p8;\nand.pred %p6, %p6, %p7;\nor.pred %p5, %p5, %p6;\nsetp.eq.{t} %p10, %f{cf}, %f0;\nsetp.lt.u64 %p11, %rd{ci}, %rd40;\nand.pred %p10, %p10, %p11;\nor.pred %p5, %p5, %p10;\nand.pred %p5, %p5, %p9;\nor.pred %p5, %p5, %p4;\nselp.{t} %f0, %f{cf}, %f0, %p5;\nselp.u64 %rd40, %rd{ci}, %rd40, %p5;"
        ));
    }

    /// Shared arg-best tree of (`%f0`, `%rd40`) over `nt` threads with [`K::pick`]: values
    /// at `%r11` (one real component per thread), indices at `%r13` (8 bytes per thread).
    /// Slot 0 ends with the block's best. Scratch `%f6`, `%rd41`, `%r12`, `%r14`. Ends with
    /// a barrier.
    fn argbest_tree(&mut self, nt: u32, stem: &str) {
        let (t, c) = (self.t(), self.ty.comp());
        self.l(&format!(
            "st.shared.{t} [%r11], %f0;\nst.shared.u64 [%r13], %rd40;\nbar.sync 0;"
        ));
        let mut h = nt / 2;
        while h >= 1 {
            self.l(&format!(
                "setp.ge.u32 %p19, %r2, {h};\n@%p19 bra $L_{stem}_{h};\nadd.u32 %r12, %r11, {};\nadd.u32 %r14, %r13, {};\nld.shared.{t} %f6, [%r12];\nld.shared.u64 %rd41, [%r14];",
                h * c,
                h * 8
            ));
            self.pick(6, 41);
            self.l(&format!(
                "st.shared.{t} [%r11], %f0;\nst.shared.u64 [%r13], %rd40;\n$L_{stem}_{h}:\nbar.sync 0;"
            ));
            h /= 2;
        }
    }

    fn done(mut self) -> String {
        self.l("$L_out:\nret;\n}");
        self.s
    }
}

/// Load op(X)(row, col) of a gemm operand into `v`, honouring its storage mode.
///
/// `row`/`col` are `%rd` registers holding clamped indices into op(X); `si`/`sl` the
/// parameter registers of its strides (`op(X)(r,c) = X[r*si + c*sl]`); `mode`, `unit`, `tr`
/// parameter registers of [`Mode`] code, unit diagonal and "op(X) is transposed". Mode 0
/// (plain) takes a uniform fast path. Scratch: `%rd60..%rd69`, `%p0`, `%p14..%p19`.
#[allow(clippy::too_many_arguments)]
fn operand(
    k: &mut K,
    base: u32,
    addr: u32,
    row: u32,
    col: u32,
    si: u32,
    sl: u32,
    mode: u32,
    unit: u32,
    tr: u32,
    v: V,
    stem: &str,
) {
    let t = k.t();
    k.l(&format!(
        "setp.eq.u64 %p14, %rd{mode}, 0;\n@%p14 bra $L_{stem}_plain;"
    ));
    // stored coordinates and strides
    k.l(&format!(
        "setp.ne.u64 %p14, %rd{tr}, 0;\nselp.u64 %rd60, %rd{col}, %rd{row}, %p14;\nselp.u64 %rd61, %rd{row}, %rd{col}, %p14;\nselp.u64 %rd62, %rd{sl}, %rd{si}, %p14;\nselp.u64 %rd63, %rd{si}, %rd{sl}, %p14;"
    ));
    // inside = lower ? sr >= sc : sr <= sc   (odd modes are lower)
    k.l(&format!(
        "setp.ge.u64 %p15, %rd60, %rd61;\nsetp.le.u64 %p16, %rd60, %rd61;\nand.b64 %rd64, %rd{mode}, 1;\nsetp.ne.u64 %p17, %rd64, 0;\nand.pred %p18, %p17, %p15;\nnot.pred %p0, %p17;\nand.pred %p0, %p0, %p16;\nor.pred %p18, %p18, %p0;"
    ));
    // symmetric / Hermitian: read the mirrored element outside the stored triangle
    k.l(&format!(
        "setp.ge.u64 %p0, %rd{mode}, 3;\nnot.pred %p19, %p18;\nand.pred %p19, %p19, %p0;\nselp.u64 %rd65, %rd61, %rd60, %p19;\nselp.u64 %rd66, %rd60, %rd61, %p19;\nmul.lo.u64 %rd67, %rd65, %rd62;\nmad.lo.u64 %rd67, %rd66, %rd63, %rd67;"
    ));
    k.elem(addr, base, 67);
    k.ld(addr, v);
    // Hermitian: conjugate mirrored elements, real diagonal
    k.l(&format!(
        "setp.ge.u64 %p0, %rd{mode}, 5;\nand.pred %p17, %p19, %p0;\nsetp.eq.u64 %p15, %rd65, %rd66;"
    ));
    k.conj_if(v, 17);
    if k.cx() {
        k.l(&format!(
            "and.pred %p16, %p15, %p0;\nselp.{t} %f{0}, %f60, %f{0}, %p16;",
            v.0 + 1
        ));
    }
    // triangular: zero outside the triangle, optional unit diagonal
    k.l(&format!(
        "setp.ge.u64 %p0, %rd{mode}, 1;\nsetp.le.u64 %p17, %rd{mode}, 2;\nand.pred %p0, %p0, %p17;\nnot.pred %p17, %p18;\nand.pred %p17, %p17, %p0;\nnot.pred %p17, %p17;"
    ));
    k.keep_if(v, 17);
    k.l(&format!("setp.ne.u64 %p17, %rd{unit}, 0;\nand.pred %p17, %p17, %p0;\nand.pred %p17, %p17, %p15;\nselp.{t} %f{0}, {1}, %f{0}, %p17;", v.0, k.ty.one()));
    if k.cx() {
        k.l(&format!("selp.{t} %f{0}, %f60, %f{0}, %p17;", v.0 + 1));
    }
    k.l(&format!("bra $L_{stem}_done;\n$L_{stem}_plain:\nmul.lo.u64 %rd67, %rd{row}, %rd{si};\nmad.lo.u64 %rd67, %rd{col}, %rd{sl}, %rd67;"));
    k.elem(addr, base, 67);
    k.ld(addr, v);
    k.l(&format!("$L_{stem}_done:"));
}

/// Storage mode of a gemm operand (parameter values; stored-triangle coordinates).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Plain = 0,
    TriLower = 1,
    TriUpper = 2,
    SymLower = 3,
    SymUpper = 4,
    HermLower = 5,
    HermUpper = 6,
}

/// Parameters: 0 A, 1 B, 2 C, 3/4 alpha re/im, 5/6 beta re/im, 7 alpha_ptr, 8 beta_ptr,
/// 9 m, 10 n, 11 k, 12 a_si, 13 a_sl, 14 b_sl, 15 b_sj, 16 ldc, 17 sa, 18 sb, 19 sc,
/// 20 conj_a, 21 conj_b, 22 c_si, 23 tri_c (0 all, 1 store i >= j, 2 store i <= j),
/// 24 herm_c (real diagonal), 25/26/27 mode/unit/tr of A, 28/29/30 mode/unit/tr of B.
/// `op(A)(i,l) = A[i*a_si + l*a_sl]`, `op(B)(l,j) = B[l*b_sl + j*b_sj]`,
/// `C(i,j) = C[i*c_si + j*ldc]`; [`Mode`] reinterprets an operand as triangular,
/// symmetric or Hermitian storage. Launch: block (16,16), grid (ceil(m/16), ceil(n/16), batch).
fn gemm(ty: Ty, tables: bool) -> String {
    let k_ = if tables { Kernel::GemmP } else { Kernel::GemmS };
    let name = entry_name(k_, ty);
    let slot = ty.slot();
    let half = TILE * TILE * slot;
    let mut k = K::new(&name, k_.params(), ty, 2 * half);
    k.l("mov.u32 %r1, %ctaid.z;\ncvt.u64.u32 %rd59, %r1;");
    k.batch_base(31, 0, 17, 59, tables);
    k.batch_base(32, 1, 18, 59, tables);
    k.batch_base(33, 2, 19, 59, tables);
    k.l("mov.u32 %r2, %tid.x;\nmov.u32 %r3, %tid.y;\nmov.u32 %r4, %ctaid.x;\nmov.u32 %r5, %ctaid.y;");
    k.l("mul.wide.u32 %rd34, %r4, 16;\ncvt.u64.u32 %rd36, %r2;\nadd.u64 %rd34, %rd34, %rd36;"); // i
    k.l("mul.wide.u32 %rd35, %r5, 16;\ncvt.u64.u32 %rd37, %r3;\nadd.u64 %rd35, %rd35, %rd37;"); // j
    k.l("setp.ne.u64 %p1, %rd20, 0;\nsetp.ne.u64 %p2, %rd21, 0;");
    k.l(&format!(
        "mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r3, 16, %r2;\nmad.lo.u32 %r12, %r11, {slot}, %r10;\nadd.u32 %r13, %r12, {half};"
    ));
    k.l(&format!(
        "mad.lo.u32 %r14, %r2, {slot}, %r10;\nmul.lo.u32 %r15, %r3, {};\nadd.u32 %r15, %r15, %r10;\nadd.u32 %r15, %r15, {half};",
        16 * slot
    ));
    let acc = V(0);
    k.zero(acc);
    k.l("mov.u64 %rd41, 0;\nsetp.eq.u64 %p3, %rd11, 0;\n@%p3 bra $L_epi;");
    k.l("sub.u64 %rd38, %rd9, 1;\nsub.u64 %rd39, %rd10, 1;\nsub.u64 %rd40, %rd11, 1;");
    k.l("$L_k:");
    // op(A)(i, kk+ty) -> tileA[ty][tx]
    k.l("cvt.u64.u32 %rd42, %r3;\nadd.u64 %rd42, %rd41, %rd42;");
    k.l("setp.lt.u64 %p4, %rd34, %rd9;\nsetp.lt.u64 %p5, %rd42, %rd11;\nselp.u64 %rd43, %rd34, %rd38, %p4;\nselp.u64 %rd44, %rd42, %rd40, %p5;");
    let a = V(2);
    operand(&mut k, 31, 46, 43, 44, 12, 13, 25, 26, 27, a, "a");
    k.l("and.pred %p6, %p4, %p5;");
    k.keep_if(a, 6);
    k.conj_if(a, 1);
    k.sts(12, a);
    // op(B)(kk+tx, j) -> tileB[ty][tx]
    k.l("cvt.u64.u32 %rd47, %r2;\nadd.u64 %rd47, %rd41, %rd47;");
    k.l("setp.lt.u64 %p7, %rd47, %rd11;\nsetp.lt.u64 %p8, %rd35, %rd10;\nselp.u64 %rd48, %rd47, %rd40, %p7;\nselp.u64 %rd49, %rd35, %rd39, %p8;");
    let b = V(4);
    operand(&mut k, 32, 51, 48, 49, 14, 15, 28, 29, 30, b, "b");
    k.l("and.pred %p9, %p7, %p8;");
    k.keep_if(b, 9);
    k.conj_if(b, 2);
    k.sts(13, b);
    k.l("bar.sync 0;");
    for l in 0..TILE {
        k.l(&format!(
            "add.u32 %r16, %r14, {};\nadd.u32 %r17, %r15, {};",
            l * TILE * slot,
            l * slot
        ));
        k.lds(16, V(6));
        k.lds(17, V(8));
        k.fma(acc, V(6), V(8));
    }
    k.l("bar.sync 0;");
    k.l("add.u64 %rd41, %rd41, 16;\nsetp.lt.u64 %p3, %rd41, %rd11;\n@%p3 bra $L_k;");
    k.l("$L_epi:");
    k.l("setp.ge.u64 %p10, %rd34, %rd9;\n@%p10 bra $L_out;\nsetp.ge.u64 %p11, %rd35, %rd10;\n@%p11 bra $L_out;");
    // triangle-only store (syrk, herk, syr2k, ...)
    k.l("setp.eq.u64 %p14, %rd23, 1;\nsetp.lt.u64 %p15, %rd34, %rd35;\nand.pred %p14, %p14, %p15;\n@%p14 bra $L_out;");
    k.l("setp.eq.u64 %p14, %rd23, 2;\nsetp.gt.u64 %p15, %rd34, %rd35;\nand.pred %p14, %p14, %p15;\n@%p14 bra $L_out;");
    let (alpha, beta, v, c) = (V(20), V(22), V(24), V(26));
    k.scalar(alpha, 3, 4, 7, None, "alpha");
    k.scalar(beta, 5, 6, 8, None, "beta");
    k.mul(v, alpha, acc);
    k.is_zero(12, alpha);
    k.l("not.pred %p12, %p12;");
    k.keep_if(v, 12);
    k.l("mul.lo.u64 %rd52, %rd34, %rd22;\nmad.lo.u64 %rd52, %rd35, %rd16, %rd52;");
    k.elem(53, 33, 52);
    k.is_zero(13, beta);
    k.l("@%p13 bra $L_store;");
    k.ld(53, c);
    k.fma(v, beta, c);
    k.l("$L_store:");
    if ty.complex() {
        let t = ty.ft();
        k.l(&format!(
            "setp.ne.u64 %p14, %rd24, 0;\nsetp.eq.u64 %p15, %rd34, %rd35;\nand.pred %p14, %p14, %p15;\nselp.{t} %f25, %f60, %f25, %p14;"
        ));
    }
    k.st(53, v);
    k.done()
}

/// Parameters: 0 A, 1 x, 2 y, 3/4 alpha, 5/6 beta, 7 alpha_ptr, 8 beta_ptr, 9 rows,
/// 10 cols, 11 a_si, 12 a_sl, 13 incx, 14 incy, 15 conj. `y[r*incy] = alpha * sum_l
/// op(A)(r,l) x[l*incx] + beta y[r*incy]`. Launch: grid ceil(rows/256), block 256.
fn gemv(ty: Ty) -> String {
    let name = entry_name(Kernel::Gemv, ty);
    let mut k = K::new(&name, Kernel::Gemv.params(), ty, 0);
    k.gid(30, BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd9;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd32, %rd1;\ncvta.to.global.u64 %rd33, %rd2;");
    k.l("setp.ne.u64 %p2, %rd15, 0;");
    let acc = V(0);
    k.zero(acc);
    k.l("mul.lo.u64 %rd34, %rd30, %rd11;\nmov.u64 %rd35, 0;\nsetp.eq.u64 %p3, %rd10, 0;\n@%p3 bra $L_epi;");
    k.l("$L_l:");
    k.l("mad.lo.u64 %rd36, %rd35, %rd12, %rd34;");
    k.elem(37, 31, 36);
    k.ld(37, V(2));
    k.conj_if(V(2), 2);
    k.l("mul.lo.u64 %rd38, %rd35, %rd13;");
    k.elem(39, 32, 38);
    k.ld(39, V(4));
    k.fma(acc, V(2), V(4));
    k.l("add.u64 %rd35, %rd35, 1;\nsetp.lt.u64 %p3, %rd35, %rd10;\n@%p3 bra $L_l;");
    k.l("$L_epi:");
    let (alpha, beta, v, y) = (V(20), V(22), V(24), V(26));
    k.scalar(alpha, 3, 4, 7, None, "alpha");
    k.scalar(beta, 5, 6, 8, None, "beta");
    k.mul(v, alpha, acc);
    k.is_zero(12, alpha);
    k.l("not.pred %p12, %p12;");
    k.keep_if(v, 12);
    k.l("mul.lo.u64 %rd40, %rd30, %rd14;");
    k.elem(41, 33, 40);
    k.is_zero(13, beta);
    k.l("@%p13 bra $L_store;");
    k.ld(41, y);
    k.fma(v, beta, y);
    k.l("$L_store:");
    k.st(41, v);
    k.done()
}

/// Parameters: 0 x, 1 y (0: all ones), 2 out (device), 3 n, 4 incx, 5 incy, 6 conj.
/// Block `b` writes `sum conj?(x[e*incx]) * y[e*incy]` over its grid-stride share of `e` to
/// `out[b]`: one block gives the dot product; several give partial sums that a one-block
/// launch with `y = 0` adds up. Launch: grid G, block 256.
fn dot(ty: Ty) -> String {
    let name = entry_name(Kernel::Dot, ty);
    let mut k = K::new(&name, Kernel::Dot.params(), ty, BLOCK * ty.slot());
    k.gid(34, BLOCK);
    k.l(&format!("mov.u32 %r3, %nctaid.x;\nmul.wide.u32 %rd39, %r3, {BLOCK};"));
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd32, %rd1;\ncvta.to.global.u64 %rd33, %rd2;");
    k.l("setp.ne.u64 %p2, %rd6, 0;\nsetp.eq.u64 %p3, %rd1, 0;");
    let acc = V(0);
    k.zero(acc);
    k.l("$L_e:\nsetp.ge.u64 %p1, %rd34, %rd3;\n@%p1 bra $L_red;");
    k.l("mul.lo.u64 %rd35, %rd34, %rd4;");
    k.elem(36, 31, 35);
    k.ld(36, V(2));
    k.conj_if(V(2), 2);
    k.l("@%p3 bra $L_one;\nmul.lo.u64 %rd37, %rd34, %rd5;");
    k.elem(38, 32, 37);
    k.ld(38, V(4));
    k.fma(acc, V(2), V(4));
    k.l("bra $L_next;\n$L_one:");
    k.add(acc, V(2));
    k.l("$L_next:\nadd.u64 %rd34, %rd34, %rd39;\nbra $L_e;");
    k.l("$L_red:");
    k.l(&format!(
        "mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r2, {}, %r10;",
        ty.slot()
    ));
    k.tree(acc, V(2), false, "sum");
    k.l("setp.ne.u32 %p4, %r2, 0;\n@%p4 bra $L_out;");
    k.lds(10, acc);
    k.l("cvt.u64.u32 %rd40, %r1;");
    k.elem(41, 33, 40);
    k.st(41, acc);
    k.done()
}

/// Parameters: 0 x, 1 out (device, real), 2 n, 3 incx, 4 part, 5 nparts. `*out = ||x||_2`.
/// With `nparts > 0`, `part` holds the `nrm2p` pairs of a first launch: the result is
/// `sqrt(sum)` unless the sum overflowed or the largest component is small enough for
/// squares to underflow; then (and with `nparts == 0`) this block recomputes it from x as
/// `scale * sqrt(sum (x/scale)^2)` with `scale = max |component|`. One block of 256.
fn nrm2(ty: Ty) -> String {
    let name = entry_name(Kernel::Nrm2, ty);
    let t = ty.ft();
    let c = ty.comp();
    let mut k = K::new(&name, Kernel::Nrm2.params(), ty, BLOCK * c);
    k.l("mov.u32 %r2, %tid.x;\ncvt.u64.u32 %rd30, %r2;");
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd33, %rd1;");
    k.l(&format!("mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r2, {c}, %r10;"));
    // combine the partials
    k.l("setp.eq.u64 %p1, %rd5, 0;\n@%p1 bra $L_full;\ncvta.to.global.u64 %rd37, %rd4;");
    k.l(&format!(
        "mov.{t} %f0, %f60;\nmov.{t} %f8, %f60;\nmov.u64 %rd34, %rd30;\n$L_p:\nsetp.ge.u64 %p1, %rd34, %rd5;\n@%p1 bra $L_pred;\nmad.lo.u64 %rd38, %rd34, {}, %rd37;\nld.global.{t} %f2, [%rd38];\nld.global.{t} %f3, [%rd38+{c}];\nmax.{t} %f0, %f0, %f2;\nadd.rn.{t} %f8, %f8, %f3;\nadd.u64 %rd34, %rd34, {BLOCK};\nbra $L_p;\n$L_pred:",
        2 * c
    ));
    k.tree(V(0), V(4), true, "pmax");
    k.l(&format!("ld.shared.{t} %f6, [%r10];\nbar.sync 0;"));
    k.real_sum_tree(8, 9, BLOCK, "psum");
    k.l(&format!("ld.shared.{t} %f8, [%r10];\nbar.sync 0;"));
    // usable: finite sum, and max == 0 or max^2 far above the underflow range
    let small = if ty.double() {
        "0d26F0000000000000" // 2^-400
    } else {
        "0f2B800000" // 2^-40
    };
    k.l(&format!(
        "setp.lt.{t} %p2, %f8, {inf};\nsetp.ge.{t} %p3, %f6, {small};\nsetp.eq.{t} %p4, %f6, %f60;\nor.pred %p3, %p3, %p4;\nand.pred %p2, %p2, %p3;\n@!%p2 bra $L_full;\nsetp.ne.u32 %p4, %r2, 0;\n@%p4 bra $L_out;\nsqrt.rn.{t} %f10, %f8;\nst.global.{t} [%rd33], %f10;\nbra $L_out;",
        inf = ty.inf()
    ));
    k.l("$L_full:");
    // pass 1: max |component|
    k.l("mov.u64 %rd34, %rd30;");
    k.l(&format!("mov.{t} %f0, %f60;"));
    k.l("$L_m:\nsetp.ge.u64 %p1, %rd34, %rd2;\n@%p1 bra $L_mred;\nmul.lo.u64 %rd35, %rd34, %rd3;");
    k.elem(36, 31, 35);
    k.ld(36, V(2));
    k.l(&format!("abs.{t} %f4, %f2;\nmax.{t} %f0, %f0, %f4;"));
    if ty.complex() {
        k.l(&format!("abs.{t} %f5, %f3;\nmax.{t} %f0, %f0, %f5;"));
    }
    k.l(&format!(
        "add.u64 %rd34, %rd34, {BLOCK};\nbra $L_m;\n$L_mred:"
    ));
    k.tree(V(0), V(4), true, "max");
    k.l(&format!("ld.shared.{t} %f6, [%r10];\nbar.sync 0;"));
    // inv = scale == 0 ? 0 : 1/scale
    k.l(&format!(
        "setp.eq.{t} %p5, %f6, %f60;\nmov.{t} %f7, {};\ndiv.rn.{t} %f7, %f7, %f6;\nselp.{t} %f7, %f60, %f7, %p5;",
        ty.one()
    ));
    // pass 2: sum of squares of scaled components
    k.l(&format!("mov.u64 %rd34, %rd30;\nmov.{t} %f8, %f60;"));
    k.l("$L_s:\nsetp.ge.u64 %p1, %rd34, %rd2;\n@%p1 bra $L_sred;\nmul.lo.u64 %rd35, %rd34, %rd3;");
    k.elem(36, 31, 35);
    k.ld(36, V(2));
    k.l(&format!(
        "mul.rn.{t} %f9, %f2, %f7;\nfma.rn.{t} %f8, %f9, %f9, %f8;"
    ));
    if ty.complex() {
        k.l(&format!(
            "mul.rn.{t} %f9, %f3, %f7;\nfma.rn.{t} %f8, %f9, %f9, %f8;"
        ));
    }
    k.l(&format!(
        "add.u64 %rd34, %rd34, {BLOCK};\nbra $L_s;\n$L_sred:"
    ));
    k.real_sum_tree(8, 9, BLOCK, "sum");
    k.l("setp.ne.u32 %p4, %r2, 0;\n@%p4 bra $L_out;");
    k.l(&format!(
        "sqrt.rn.{t} %f10, %f8;\nmul.rn.{t} %f10, %f10, %f6;\nsetp.eq.{t} %p6, %f6, {};\nselp.{t} %f10, %f6, %f10, %p6;\nst.global.{t} [%rd33], %f10;",
        ty.inf()
    ));
    k.done()
}

/// Parameters: 0 x, 1 y, 2/3 alpha, 4 alpha_ptr, 5 n, 6 incx, 7 incy.
/// `y[e*incy] += alpha x[e*incx]`; alpha 0 leaves y untouched. Grid ceil(n/256).
fn axpy(ty: Ty) -> String {
    let name = entry_name(Kernel::Axpy, ty);
    let mut k = K::new(&name, Kernel::Axpy.params(), ty, 0);
    k.gid(30, BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd5;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd32, %rd1;");
    let alpha = V(20);
    k.scalar(alpha, 2, 3, 4, None, "alpha");
    k.is_zero(12, alpha);
    k.l("@%p12 bra $L_out;");
    k.l("mul.lo.u64 %rd34, %rd30, %rd6;");
    k.elem(35, 31, 34);
    k.ld(35, V(2));
    k.l("mul.lo.u64 %rd36, %rd30, %rd7;");
    k.elem(37, 32, 36);
    k.ld(37, V(4));
    k.fma(V(4), alpha, V(2));
    k.st(37, V(4));
    k.done()
}

/// Parameters: 0 x, 1/2 alpha, 3 alpha_ptr, 4 n, 5 incx, 6 real_alpha.
/// `x[e*incx] *= alpha`. Grid ceil(n/256).
fn scal(ty: Ty) -> String {
    let name = entry_name(Kernel::Scal, ty);
    let mut k = K::new(&name, Kernel::Scal.params(), ty, 0);
    k.gid(30, BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd4;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd31, %rd0;");
    let alpha = V(20);
    k.scalar(alpha, 1, 2, 3, Some(6), "alpha");
    if ty.complex() {
        // a real device scalar has no imaginary part to load
        let t = ty.ft();
        k.l(&format!(
            "setp.ne.u64 %p13, %rd6, 0;\nselp.{t} %f21, %f60, %f21, %p13;"
        ));
    }
    k.l("mul.lo.u64 %rd34, %rd30, %rd5;");
    k.elem(35, 31, 34);
    k.ld(35, V(2));
    k.mul(V(4), alpha, V(2));
    k.st(35, V(4));
    k.done()
}

/// Parameters: 0 A, 1 B, 2/3 alpha, 4 alpha_ptr, 5 len, 6 nvec, 7 a_rs, 8 a_cs, 9 b_vs,
/// 10 b_es, 11 lower, 12 unit, 13 conj, 14 sa, 15 sb. Solves `T x = alpha b` in place for
/// every vector `b = B[v*b_vs + e*b_es]`, `T(r,c) = conj?(A[r*a_rs + c*a_cs])` lower or
/// upper triangular of order `len`. Launch: grid (ceil(nvec/64), batch), block 64.
fn trsm(ty: Ty, tables: bool) -> String {
    let k_ = if tables { Kernel::TrsmP } else { Kernel::TrsmS };
    let name = entry_name(k_, ty);
    let t = ty.ft();
    let mut k = K::new(&name, k_.params(), ty, 0);
    k.gid(30, TRSM_BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd6;\n@%p1 bra $L_out;");
    k.l("mov.u32 %r3, %ctaid.y;\ncvt.u64.u32 %rd29, %r3;");
    k.batch_base(31, 0, 14, 29, tables);
    k.batch_base(32, 1, 15, 29, tables);
    k.l("mul.lo.u64 %rd34, %rd30, %rd9;");
    k.elem(33, 32, 34);
    k.l("setp.ne.u64 %p2, %rd11, 0;\nsetp.ne.u64 %p3, %rd12, 0;\nsetp.ne.u64 %p4, %rd13, 0;");
    let alpha = V(20);
    k.scalar(alpha, 2, 3, 4, None, "alpha");
    k.l("sub.u64 %rd35, %rd5, 1;\nmov.u64 %rd36, 0;\nsetp.eq.u64 %p6, %rd5, 0;\n@%p6 bra $L_out;");
    k.l("$L_row:");
    k.l("sub.u64 %rd37, %rd35, %rd36;\nselp.u64 %rd38, %rd36, %rd37, %p2;");
    k.l("mul.lo.u64 %rd39, %rd38, %rd10;");
    k.elem(40, 33, 39);
    k.ld(40, V(2));
    let acc = V(0);
    k.mul(acc, alpha, V(2));
    k.l("mul.lo.u64 %rd41, %rd38, %rd7;\nmov.u64 %rd42, 0;\nsetp.ge.u64 %p5, %rd42, %rd36;\n@%p5 bra $L_diag;");
    k.l("$L_col:");
    k.l("sub.u64 %rd43, %rd35, %rd42;\nselp.u64 %rd44, %rd42, %rd43, %p2;");
    k.l("mad.lo.u64 %rd45, %rd44, %rd8, %rd41;");
    k.elem(46, 31, 45);
    k.ld(46, V(4));
    k.conj_if(V(4), 4);
    k.neg(V(4));
    k.l("mul.lo.u64 %rd47, %rd44, %rd10;");
    k.elem(48, 33, 47);
    k.ld(48, V(6));
    k.fma(acc, V(4), V(6));
    k.l("add.u64 %rd42, %rd42, 1;\nsetp.lt.u64 %p5, %rd42, %rd36;\n@%p5 bra $L_col;");
    k.l("$L_diag:\n@%p3 bra $L_st;");
    k.l("mad.lo.u64 %rd45, %rd38, %rd8, %rd41;");
    k.elem(46, 31, 45);
    k.ld(46, V(4));
    k.conj_if(V(4), 4);
    if ty.complex() {
        // (a + bi) / (c + di) = ((ac + bd) + (bc - ad) i) / (c^2 + d^2)
        k.l(&format!(
            "mul.rn.{t} %f8, %f4, %f4;\nfma.rn.{t} %f8, %f5, %f5, %f8;\nmul.rn.{t} %f9, %f0, %f4;\nfma.rn.{t} %f9, %f1, %f5, %f9;\nmul.rn.{t} %f10, %f1, %f4;\nneg.{t} %f11, %f0;\nfma.rn.{t} %f10, %f11, %f5, %f10;\ndiv.rn.{t} %f0, %f9, %f8;\ndiv.rn.{t} %f1, %f10, %f8;"
        ));
    } else {
        k.l(&format!("div.rn.{t} %f0, %f0, %f4;"));
    }
    k.l("$L_st:");
    k.st(40, acc);
    k.l("add.u64 %rd36, %rd36, 1;\nsetp.lt.u64 %p6, %rd36, %rd5;\n@%p6 bra $L_row;");
    k.done()
}

/// Parameters: 0 x, 1 y, 2 n, 3 incx, 4 incy, 5 swap. `y[e*incy] = x[e*incx]`, or exchange
/// the two when `swap`. Grid ceil(n/256).
fn copy(ty: Ty) -> String {
    let name = entry_name(Kernel::Copy, ty);
    let mut k = K::new(&name, Kernel::Copy.params(), ty, 0);
    k.gid(30, BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd2;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd32, %rd1;");
    k.l("mul.lo.u64 %rd34, %rd30, %rd3;");
    k.elem(35, 31, 34);
    k.l("mul.lo.u64 %rd36, %rd30, %rd4;");
    k.elem(37, 32, 36);
    k.ld(35, V(2));
    k.l("setp.eq.u64 %p2, %rd5, 0;\n@%p2 bra $L_put;");
    k.ld(37, V(4));
    k.st(35, V(4));
    k.l("$L_put:");
    k.st(37, V(2));
    k.done()
}

/// Parameters: 0 x, 1 out (device), 2 n, 3 incx, 4 min, 5 mode. The first element with the
/// largest (smallest when `min`) `|re| + |im|`. Mode 0: one block over x, `*out` = its
/// 1-based index (i32), 0 when `n == 0`. Mode 1: block `b` covers its grid-stride share
/// and writes (value, 0-based u64 index) to `out + 16 b`. Mode 2: one block over the `n`
/// pairs at x from mode 1, `*out` as in mode 0. Ties go to the smaller index throughout.
fn amax(ty: Ty) -> String {
    let name = entry_name(Kernel::Amax, ty);
    let t = ty.ft();
    let c = ty.comp();
    let mut k = K::new(&name, Kernel::Amax.params(), ty, BLOCK * (c + 8));
    k.gid(34, BLOCK);
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd33, %rd1;");
    k.l("setp.eq.u64 %p12, %rd5, 1;\nsetp.eq.u64 %p13, %rd5, 2;\nsetp.ne.u64 %p7, %rd4, 0;\nnot.pred %p8, %p7;");
    k.l(&format!(
        "cvt.u64.u32 %rd30, %r2;\nselp.u64 %rd34, %rd34, %rd30, %p12;\nmov.u32 %r3, %nctaid.x;\nmul.wide.u32 %rd39, %r3, {BLOCK};\nmov.u64 %rd38, {BLOCK};\nselp.u64 %rd39, %rd39, %rd38, %p12;"
    ));
    // best value %f0, best index %rd40 (%rd42 = u64 max = none)
    k.l(&format!(
        "mov.{t} %f0, %f60;\nmov.u64 %rd42, 0;\nnot.b64 %rd42, %rd42;\nmov.u64 %rd40, %rd42;"
    ));
    k.l("$L_e:\nsetp.ge.u64 %p1, %rd34, %rd2;\n@%p1 bra $L_red;\n@%p13 bra $L_pair;\nmul.lo.u64 %rd35, %rd34, %rd3;");
    k.elem(36, 31, 35);
    k.ld(36, V(2));
    k.abs1(4, V(2));
    k.l("mov.u64 %rd43, %rd34;\nbra $L_cmp;");
    k.l(&format!(
        "$L_pair:\nmad.lo.u64 %rd36, %rd34, 16, %rd31;\nld.global.{t} %f4, [%rd36];\nld.global.u64 %rd43, [%rd36+8];\n$L_cmp:"
    ));
    k.pick(4, 43);
    k.l("add.u64 %rd34, %rd34, %rd39;\nbra $L_e;\n$L_red:");
    // shared: values at smem[tid*c], indices at smem[BLOCK*c + tid*8]
    k.l(&format!(
        "mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r2, {c}, %r10;\nmad.lo.u32 %r13, %r2, 8, %r10;\nadd.u32 %r13, %r13, {};",
        BLOCK * c
    ));
    k.argbest_tree(BLOCK, "a");
    k.l("setp.ne.u32 %p4, %r2, 0;\n@%p4 bra $L_out;");
    k.l(&format!(
        "@!%p12 bra $L_index;\ncvt.u64.u32 %rd44, %r1;\nmad.lo.u64 %rd45, %rd44, 16, %rd33;\nst.global.{t} [%rd45], %f0;\nst.global.u64 [%rd45+8], %rd40;\nbra $L_out;"
    ));
    k.l("$L_index:\nadd.u64 %rd40, %rd40, 1;\ncvt.u32.u64 %r21, %rd40;\nst.global.u32 [%rd33], %r21;");
    k.done()
}

/// Parameters: 0 A, 1 B, 2 C, 3/4 alpha, 5/6 beta, 7 alpha_ptr, 8 beta_ptr, 9 m, 10 n,
/// 11 a_si, 12 a_sj, 13 b_si, 14 b_sj, 15 ldc, 16 conj_a, 17 conj_b, 18 mode, 19 c_si,
/// 20 unused. Mode 0: `C(i,j) = alpha op(A)(i,j) + beta op(B)(i,j)` (a zero alpha or beta
/// skips that operand, so C may alias it). Mode 1: `C(i,j) = op(A)(i,j) * op(B)(i,j)`
/// (`dgmm`, B a vector given by its strides). Grid (ceil(m/256), min(n, 65535)); columns
/// beyond the grid loop.
fn geam(ty: Ty) -> String {
    let name = entry_name(Kernel::Geam, ty);
    let mut k = K::new(&name, Kernel::Geam.params(), ty, 0);
    k.gid(30, BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd9;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd32, %rd1;\ncvta.to.global.u64 %rd33, %rd2;");
    k.l("setp.ne.u64 %p2, %rd16, 0;\nsetp.ne.u64 %p3, %rd17, 0;\nsetp.ne.u64 %p4, %rd18, 0;");
    let (alpha, beta) = (V(20), V(22));
    k.scalar(alpha, 3, 4, 7, None, "alpha");
    k.scalar(beta, 5, 6, 8, None, "beta");
    k.is_zero(12, alpha);
    k.is_zero(13, beta);
    k.l("mov.u32 %r5, %ctaid.y;\ncvt.u64.u32 %rd34, %r5;\nmov.u32 %r6, %nctaid.y;\ncvt.u64.u32 %rd35, %r6;");
    k.l("$L_col:\nsetp.ge.u64 %p1, %rd34, %rd10;\n@%p1 bra $L_out;");
    let (va, vb, v) = (V(2), V(4), V(24));
    k.zero(va);
    k.zero(vb);
    k.l("@%p12 bra $L_noa;\nmul.lo.u64 %rd36, %rd30, %rd11;\nmad.lo.u64 %rd36, %rd34, %rd12, %rd36;");
    k.elem(37, 31, 36);
    k.ld(37, va);
    k.conj_if(va, 2);
    k.l("$L_noa:");
    // mode 1 always reads B
    k.l("not.pred %p5, %p4;\nand.pred %p5, %p5, %p13;\n@%p5 bra $L_nob;\nmul.lo.u64 %rd38, %rd30, %rd13;\nmad.lo.u64 %rd38, %rd34, %rd14, %rd38;");
    k.elem(39, 32, 38);
    k.ld(39, vb);
    k.conj_if(vb, 3);
    k.l("$L_nob:");
    k.l("@%p4 bra $L_mul;");
    k.mul(v, alpha, va);
    k.l("@%p12 bra $L_add_b;\nbra $L_add_b2;\n$L_add_b:");
    k.zero(v);
    k.l("$L_add_b2:\n@%p13 bra $L_st;");
    k.fma(v, beta, vb);
    k.l("bra $L_st;\n$L_mul:");
    k.mul(v, va, vb);
    k.l("$L_st:\nmul.lo.u64 %rd40, %rd30, %rd19;\nmad.lo.u64 %rd40, %rd34, %rd15, %rd40;");
    k.elem(41, 33, 40);
    k.st(41, v);
    k.l("add.u64 %rd34, %rd34, %rd35;\nbra $L_col;");
    k.done()
}

/// Parameters: 0 A table, 1 P (device i32, n per matrix; 0 = no pivoting), 2 info (device
/// i32 per matrix), 3 n, 4 lda, 5 batch, 6 pivot. In-place LU with partial pivoting
/// (LAPACK `getf2` order: pivot on max |re| + |im|, first index on ties, swap whole rows,
/// scale the column unless the pivot is zero, rank-1 update). One work-group of
/// `LU_BLOCK` threads per matrix: a shared arg-max finds the pivot, threads split the
/// columns of the swap and the rows of the scaling and update. `info` = first zero pivot
/// (1-based) or 0. Launch: grid batch, block `LU_BLOCK`.
fn getrf(ty: Ty) -> String {
    let name = entry_name(Kernel::Getrf, ty);
    let c = ty.comp();
    let nt = LU_BLOCK;
    let mut k = K::new(&name, Kernel::Getrf.params(), ty, nt * (c + 8));
    let t = k.t();
    k.l("mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %tid.x;\ncvt.u64.u32 %rd30, %r1;\ncvt.u64.u32 %rd29, %r2;");
    k.l("setp.ge.u64 %p1, %rd30, %rd5;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd40, %rd0;\nmad.lo.u64 %rd40, %rd30, 8, %rd40;\nld.global.u64 %rd31, [%rd40];\ncvta.to.global.u64 %rd31, %rd31;");
    k.l("cvta.to.global.u64 %rd32, %rd1;\nmul.lo.u64 %rd41, %rd30, %rd3;\nmad.lo.u64 %rd32, %rd41, 4, %rd32;");
    k.l("cvta.to.global.u64 %rd33, %rd2;\nmad.lo.u64 %rd33, %rd30, 4, %rd33;");
    k.l(&format!(
        "mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r2, {c}, %r10;\nadd.u32 %r16, %r10, {};\nmad.lo.u32 %r13, %r2, 8, %r16;",
        nt * c
    ));
    k.l("setp.ne.u64 %p2, %rd6, 0;\nmov.u32 %r20, 0;\nmov.u64 %rd34, 0;\nmov.u64 %rd42, 0;\nnot.b64 %rd42, %rd42;\nsetp.eq.u64 %p7, %rd42, 0;\nnot.pred %p8, %p7;");
    k.l("setp.eq.u64 %p3, %rd3, 0;\n@%p3 bra $L_done;");
    k.l("$L_k:\nmul.lo.u64 %rd35, %rd34, %rd4;\n@!%p2 bra $L_nopiv;");
    // per-thread best of rows k + tid + nt q, then the block's
    k.l(&format!(
        "mov.{t} %f0, %f60;\nmov.u64 %rd40, %rd42;\nadd.u64 %rd37, %rd34, %rd29;\n$L_ps:\nsetp.ge.u64 %p4, %rd37, %rd3;\n@%p4 bra $L_pe;\nadd.u64 %rd43, %rd35, %rd37;"
    ));
    k.elem(44, 31, 43);
    k.ld(44, V(2));
    k.abs1(4, V(2));
    k.pick(4, 37);
    k.l(&format!("add.u64 %rd37, %rd37, {nt};\nbra $L_ps;\n$L_pe:"));
    k.argbest_tree(nt, "piv");
    k.l("ld.shared.u64 %rd36, [%r16];");
    k.l("setp.ne.u32 %p6, %r2, 0;\n@%p6 bra $L_pw;\nadd.u64 %rd44, %rd36, 1;\ncvt.u32.u64 %r21, %rd44;\nmad.lo.u64 %rd45, %rd34, 4, %rd32;\nst.global.u32 [%rd45], %r21;\n$L_pw:");
    // swap rows k and p, threads over columns
    k.l("setp.eq.u64 %p6, %rd36, %rd34;\n@%p6 bra $L_nopiv;\nmov.u64 %rd38, %rd29;\n$L_sw:\nsetp.ge.u64 %p9, %rd38, %rd3;\n@%p9 bra $L_swd;\nmul.lo.u64 %rd46, %rd38, %rd4;\nadd.u64 %rd47, %rd46, %rd34;\nadd.u64 %rd48, %rd46, %rd36;");
    k.elem(49, 31, 47);
    k.elem(50, 31, 48);
    k.ld(49, V(2));
    k.ld(50, V(4));
    k.st(49, V(4));
    k.st(50, V(2));
    k.l(&format!("add.u64 %rd38, %rd38, {nt};\nbra $L_sw;\n$L_swd:\nbar.sync 0;"));
    k.l("$L_nopiv:\nadd.u64 %rd43, %rd35, %rd34;");
    k.elem(44, 31, 43);
    let d = V(14);
    k.ld(44, d);
    k.is_zero(10, d);
    k.l("@!%p10 bra $L_nz;\nsetp.eq.u32 %p11, %r20, 0;\nadd.u64 %rd44, %rd34, 1;\ncvt.u32.u64 %r22, %rd44;\nselp.u32 %r20, %r22, %r20, %p11;\n$L_nz:");
    // rows below the pivot: scale l(i,k), then a(i,j) -= l(i,k) a(k,j)
    k.l("add.u64 %rd37, %rd34, 1;\nadd.u64 %rd37, %rd37, %rd29;\n$L_r:\nsetp.ge.u64 %p4, %rd37, %rd3;\n@%p4 bra $L_rd;\nadd.u64 %rd43, %rd35, %rd37;");
    k.elem(44, 31, 43);
    k.ld(44, V(2));
    k.l("@%p10 bra $L_ns;");
    k.div(V(2), d);
    k.st(44, V(2));
    k.l("$L_ns:");
    k.neg(V(2));
    k.l("add.u64 %rd38, %rd34, 1;\n$L_j:\nsetp.ge.u64 %p5, %rd38, %rd3;\n@%p5 bra $L_jd;\nmul.lo.u64 %rd46, %rd38, %rd4;\nadd.u64 %rd47, %rd46, %rd34;\nadd.u64 %rd48, %rd46, %rd37;");
    k.elem(49, 31, 47);
    k.elem(50, 31, 48);
    k.ld(49, V(4));
    k.ld(50, V(6));
    k.fma(V(6), V(2), V(4));
    k.st(50, V(6));
    k.l(&format!("add.u64 %rd38, %rd38, 1;\nbra $L_j;\n$L_jd:\nadd.u64 %rd37, %rd37, {nt};\nbra $L_r;\n$L_rd:\nbar.sync 0;"));
    k.l("add.u64 %rd34, %rd34, 1;\nsetp.lt.u64 %p3, %rd34, %rd3;\n@%p3 bra $L_k;");
    k.l("$L_done:\nsetp.ne.u32 %p6, %r2, 0;\n@%p6 bra $L_out;\nsetp.eq.u64 %p3, %rd2, 0;\n@%p3 bra $L_out;\nst.global.u32 [%rd33], %r20;");
    k.done()
}

/// Parameters: 0 B table, 1 P (device i32, n per matrix), 2 n, 3 nrhs, 4 ldb, 5 reverse.
/// Applies the row interchanges of `getrf` to every column of B (`laswp`), in order or
/// (`reverse`) backwards. Launch: grid (ceil(nrhs/64), batch), block 64; tables and P
/// pre-offset per launch.
fn laswp(ty: Ty) -> String {
    let name = entry_name(Kernel::Laswp, ty);
    let mut k = K::new(&name, Kernel::Laswp.params(), ty, 0);
    k.gid(30, TRSM_BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd3;\n@%p1 bra $L_out;");
    k.l("mov.u32 %r3, %ctaid.y;\ncvt.u64.u32 %rd29, %r3;");
    k.l("cvta.to.global.u64 %rd40, %rd0;\nmad.lo.u64 %rd40, %rd29, 8, %rd40;\nld.global.u64 %rd31, [%rd40];\ncvta.to.global.u64 %rd31, %rd31;");
    k.l("cvta.to.global.u64 %rd32, %rd1;\nmul.lo.u64 %rd41, %rd29, %rd2;\nmad.lo.u64 %rd32, %rd41, 4, %rd32;");
    k.l("mul.lo.u64 %rd35, %rd30, %rd4;\nsetp.ne.u64 %p2, %rd5, 0;\nmov.u64 %rd34, 0;\nsetp.eq.u64 %p3, %rd2, 0;\n@%p3 bra $L_out;");
    k.l("$L_k:\nsub.u64 %rd36, %rd2, 1;\nsub.u64 %rd36, %rd36, %rd34;\nselp.u64 %rd37, %rd36, %rd34, %p2;");
    k.l("mad.lo.u64 %rd45, %rd37, 4, %rd32;\nld.global.u32 %r21, [%rd45];\ncvt.u64.u32 %rd38, %r21;\nsub.u64 %rd38, %rd38, 1;");
    k.l("setp.eq.u64 %p4, %rd38, %rd37;\n@%p4 bra $L_n;\nadd.u64 %rd46, %rd35, %rd37;\nadd.u64 %rd47, %rd35, %rd38;");
    k.elem(48, 31, 46);
    k.elem(49, 31, 47);
    k.ld(48, V(2));
    k.ld(49, V(4));
    k.st(48, V(4));
    k.st(49, V(2));
    k.l("$L_n:\nadd.u64 %rd34, %rd34, 1;\nsetp.lt.u64 %p3, %rd34, %rd2;\n@%p3 bra $L_k;");
    k.done()
}

/// Parameters: 0 x, 1 part (device, real), 2 n, 3 incx. Block `b` writes the largest
/// |component| and the unscaled sum of squares of its grid-stride share of x to
/// `part[2b]` and `part[2b+1]`. Launch: grid G, block 256.
fn nrm2_partial(ty: Ty) -> String {
    let name = entry_name(Kernel::Nrm2P, ty);
    let t = ty.ft();
    let c = ty.comp();
    let mut k = K::new(&name, Kernel::Nrm2P.params(), ty, BLOCK * c);
    k.gid(34, BLOCK);
    k.l(&format!("mov.u32 %r3, %nctaid.x;\nmul.wide.u32 %rd39, %r3, {BLOCK};"));
    k.l("cvta.to.global.u64 %rd31, %rd0;\ncvta.to.global.u64 %rd33, %rd1;");
    k.l(&format!("mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r2, {c}, %r10;"));
    k.l(&format!("mov.{t} %f0, %f60;\nmov.{t} %f8, %f60;"));
    k.l("$L_e:\nsetp.ge.u64 %p1, %rd34, %rd2;\n@%p1 bra $L_red;\nmul.lo.u64 %rd35, %rd34, %rd3;");
    k.elem(36, 31, 35);
    k.ld(36, V(2));
    k.l(&format!(
        "abs.{t} %f4, %f2;\nmax.{t} %f0, %f0, %f4;\nfma.rn.{t} %f8, %f2, %f2, %f8;"
    ));
    if ty.complex() {
        k.l(&format!(
            "abs.{t} %f5, %f3;\nmax.{t} %f0, %f0, %f5;\nfma.rn.{t} %f8, %f3, %f3, %f8;"
        ));
    }
    k.l("add.u64 %rd34, %rd34, %rd39;\nbra $L_e;\n$L_red:");
    k.tree(V(0), V(4), true, "max");
    k.real_sum_tree(8, 9, BLOCK, "sum");
    k.l("setp.ne.u32 %p4, %r2, 0;\n@%p4 bra $L_out;");
    k.l(&format!(
        "cvt.u64.u32 %rd40, %r1;\nmad.lo.u64 %rd41, %rd40, {}, %rd33;\nst.global.{t} [%rd41], %f0;\nst.global.{t} [%rd41+{c}], %f8;",
        2 * c
    ));
    k.done()
}

/// Householder update of the columns `%rd38, %rd38 + nt, ... < %rd<count>` of the matrix at
/// `%rd<base>` (leading dimension `%rd<ld>`) by the reflector in column j of A (`%rd34` = j,
/// `%rd35` = j lda, v(j) = 1 implicit, v(i) = A(i, j) below), with `-conj(tau)` folded in:
/// `x -= conj(tau) v (v^H x)`. Uses `%f2..%f5`, `%f36..%f39`, `%rd37`, `%rd39`, `%rd43..%rd48`.
fn qr_apply(k: &mut K, base: u32, ld: u32, count: u32, stem: &str) {
    let nt = LU_BLOCK;
    k.l(&format!(
        "$L_{stem}:\nsetp.ge.u64 %p5, %rd38, %rd{count};\n@%p5 bra $L_{stem}_d;\nmul.lo.u64 %rd39, %rd38, %rd{ld};\nadd.u64 %rd43, %rd39, %rd34;"
    ));
    k.elem(44, base, 43);
    let w = V(36);
    k.ld(44, w);
    k.l(&format!(
        "add.u64 %rd37, %rd34, 1;\n$L_{stem}_a:\nsetp.ge.u64 %p6, %rd37, %rd4;\n@%p6 bra $L_{stem}_ad;\nadd.u64 %rd45, %rd35, %rd37;"
    ));
    k.elem(46, 31, 45);
    k.ld(46, V(2));
    k.conj(V(2));
    k.l("add.u64 %rd47, %rd39, %rd37;");
    k.elem(48, base, 47);
    k.ld(48, V(4));
    k.fma(w, V(2), V(4));
    k.l(&format!(
        "add.u64 %rd37, %rd37, 1;\nbra $L_{stem}_a;\n$L_{stem}_ad:"
    ));
    let nw = V(38);
    k.mul(nw, V(34), w);
    k.neg(nw);
    k.ld(44, V(4));
    k.add(V(4), nw);
    k.st(44, V(4));
    k.l(&format!(
        "add.u64 %rd37, %rd34, 1;\n$L_{stem}_b:\nsetp.ge.u64 %p6, %rd37, %rd4;\n@%p6 bra $L_{stem}_bd;\nadd.u64 %rd45, %rd35, %rd37;"
    ));
    k.elem(46, 31, 45);
    k.ld(46, V(2));
    k.l("add.u64 %rd47, %rd39, %rd37;");
    k.elem(48, base, 47);
    k.ld(48, V(4));
    k.fma(V(4), V(2), nw);
    k.st(48, V(4));
    k.l(&format!(
        "add.u64 %rd37, %rd37, 1;\nbra $L_{stem}_b;\n$L_{stem}_bd:\nadd.u64 %rd38, %rd38, {nt};\nbra $L_{stem};\n$L_{stem}_d:"
    ));
}

/// One thread per matrix variant of [`getrf`] (same parameters) for small orders, where a
/// work-group per matrix would leave most threads idle: LAPACK `getf2` order, serially.
/// Launch: grid ceil(batch/64), block 64.
fn getrf_thread(ty: Ty) -> String {
    let name = entry_name(Kernel::GetrfT, ty);
    let t = ty.ft();
    let mut k = K::new(&name, Kernel::GetrfT.params(), ty, 0);
    k.gid(30, TRSM_BLOCK);
    k.l("setp.ge.u64 %p1, %rd30, %rd5;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd40, %rd0;\nmad.lo.u64 %rd40, %rd30, 8, %rd40;\nld.global.u64 %rd31, [%rd40];\ncvta.to.global.u64 %rd31, %rd31;");
    k.l("cvta.to.global.u64 %rd32, %rd1;\nmul.lo.u64 %rd41, %rd30, %rd3;\nmad.lo.u64 %rd32, %rd41, 4, %rd32;");
    k.l("cvta.to.global.u64 %rd33, %rd2;\nmad.lo.u64 %rd33, %rd30, 4, %rd33;");
    k.l("setp.ne.u64 %p2, %rd6, 0;\nmov.u32 %r20, 0;\nmov.u64 %rd34, 0;");
    k.l("setp.eq.u64 %p3, %rd3, 0;\n@%p3 bra $L_done;");
    k.l("$L_k:");
    k.l("mul.lo.u64 %rd35, %rd34, %rd4;"); // k*lda
    k.l("@!%p2 bra $L_nopiv;");
    // pivot search over rows k..n of column k
    k.l("mov.u64 %rd36, %rd34;\nadd.u64 %rd42, %rd35, %rd34;");
    k.elem(43, 31, 42);
    k.ld(43, V(2));
    k.abs1(12, V(2));
    k.l("add.u64 %rd37, %rd34, 1;\n$L_ps:\nsetp.ge.u64 %p4, %rd37, %rd3;\n@%p4 bra $L_pe;\nadd.u64 %rd42, %rd35, %rd37;");
    k.elem(43, 31, 42);
    k.ld(43, V(2));
    k.abs1(13, V(2));
    k.l(&format!("setp.gt.{t} %p5, %f13, %f12;\nselp.{t} %f12, %f13, %f12, %p5;\nselp.u64 %rd36, %rd37, %rd36, %p5;\nadd.u64 %rd37, %rd37, 1;\nbra $L_ps;\n$L_pe:"));
    k.l("add.u64 %rd44, %rd36, 1;\ncvt.u32.u64 %r21, %rd44;\nmad.lo.u64 %rd45, %rd34, 4, %rd32;\nst.global.u32 [%rd45], %r21;");
    // swap rows k and p
    k.l("setp.eq.u64 %p6, %rd36, %rd34;\n@%p6 bra $L_nopiv;\nmov.u64 %rd38, 0;\n$L_sw:\nmul.lo.u64 %rd46, %rd38, %rd4;\nadd.u64 %rd47, %rd46, %rd34;\nadd.u64 %rd48, %rd46, %rd36;");
    k.elem(49, 31, 47);
    k.elem(50, 31, 48);
    k.ld(49, V(2));
    k.ld(50, V(4));
    k.st(49, V(4));
    k.st(50, V(2));
    k.l("add.u64 %rd38, %rd38, 1;\nsetp.lt.u64 %p7, %rd38, %rd3;\n@%p7 bra $L_sw;");
    k.l("$L_nopiv:");
    // pivot d = A(k,k); zero pivot: record, skip the scaling
    k.l("add.u64 %rd42, %rd35, %rd34;");
    k.elem(43, 31, 42);
    let d = V(14);
    k.ld(43, d);
    k.is_zero(8, d);
    k.l("@!%p8 bra $L_scale;\nsetp.eq.u32 %p9, %r20, 0;\nadd.u64 %rd44, %rd34, 1;\ncvt.u32.u64 %r22, %rd44;\nselp.u32 %r20, %r22, %r20, %p9;\nbra $L_upd;");
    k.l("$L_scale:\nadd.u64 %rd37, %rd34, 1;\n$L_sc:\nsetp.ge.u64 %p4, %rd37, %rd3;\n@%p4 bra $L_upd;\nadd.u64 %rd42, %rd35, %rd37;");
    k.elem(43, 31, 42);
    k.ld(43, V(2));
    k.div(V(2), d);
    k.st(43, V(2));
    k.l("add.u64 %rd37, %rd37, 1;\nbra $L_sc;");
    // trailing update A(i,j) -= A(i,k) A(k,j)
    k.l("$L_upd:\nadd.u64 %rd38, %rd34, 1;\n$L_uj:\nsetp.ge.u64 %p4, %rd38, %rd3;\n@%p4 bra $L_next;\nmul.lo.u64 %rd46, %rd38, %rd4;\nadd.u64 %rd47, %rd46, %rd34;");
    k.elem(49, 31, 47);
    k.ld(49, V(4));
    k.neg(V(4));
    k.l("add.u64 %rd37, %rd34, 1;\n$L_ui:\nsetp.ge.u64 %p5, %rd37, %rd3;\n@%p5 bra $L_ujn;\nadd.u64 %rd42, %rd35, %rd37;\nadd.u64 %rd48, %rd46, %rd37;");
    k.elem(43, 31, 42);
    k.elem(50, 31, 48);
    k.ld(43, V(2));
    k.ld(50, V(6));
    k.fma(V(6), V(2), V(4));
    k.st(50, V(6));
    k.l("add.u64 %rd37, %rd37, 1;\nbra $L_ui;\n$L_ujn:\nadd.u64 %rd38, %rd38, 1;\nbra $L_uj;");
    k.l("$L_next:\nadd.u64 %rd34, %rd34, 1;\nsetp.lt.u64 %p3, %rd34, %rd3;\n@%p3 bra $L_k;");
    k.l("$L_done:\nsetp.eq.u64 %p3, %rd2, 0;\n@%p3 bra $L_out;\nst.global.u32 [%rd33], %r20;");
    k.done()
}

/// Parameters: 0 A table, 1 Tau table (0: none), 2 C table (0: none), 3 info (device i32
/// per matrix, 0: none), 4 m, 5 n, 6 nrhs, 7 lda, 8 ldc, 9 batch. Householder QR in place
/// (LAPACK `geqr2`/`larfg`: `A = H_1 ... H_k`, `H_j = I - tau_j v_j v_j^H`, R in the upper
/// triangle, v below the diagonal, tau to the Tau table). With a C table the reflectors
/// are also applied to C (`C = Q^H C`, m x nrhs) and `R X = C(0:n, :)` is solved in place
/// (`gelsBatched`, m >= n): `info` = first zero diagonal of R (1-based, then no solve).
/// One work-group of `LU_BLOCK` threads per matrix: a shared sum for the column norm,
/// thread 0 forms the reflector, threads split rows (scaling) and columns (update, solve).
/// Launch: grid batch, block `LU_BLOCK`.
fn geqrf(ty: Ty) -> String {
    let name = entry_name(Kernel::Geqrf, ty);
    let c = ty.comp();
    let nt = LU_BLOCK;
    let mut k = K::new(&name, Kernel::Geqrf.params(), ty, nt * c + 4 * c);
    let t = k.t();
    let cx = ty.complex();
    k.l("mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %tid.x;\ncvt.u64.u32 %rd30, %r1;\ncvt.u64.u32 %rd29, %r2;");
    k.l("setp.ge.u64 %p1, %rd30, %rd9;\n@%p1 bra $L_out;");
    k.l("cvta.to.global.u64 %rd40, %rd0;\nmad.lo.u64 %rd40, %rd30, 8, %rd40;\nld.global.u64 %rd31, [%rd40];\ncvta.to.global.u64 %rd31, %rd31;");
    k.l("setp.ne.u64 %p2, %rd1, 0;\nsetp.ne.u64 %p3, %rd2, 0;");
    k.l("@!%p2 bra $L_t1;\ncvta.to.global.u64 %rd40, %rd1;\nmad.lo.u64 %rd40, %rd30, 8, %rd40;\nld.global.u64 %rd32, [%rd40];\ncvta.to.global.u64 %rd32, %rd32;\n$L_t1:");
    k.l("@!%p3 bra $L_t2;\ncvta.to.global.u64 %rd40, %rd2;\nmad.lo.u64 %rd40, %rd30, 8, %rd40;\nld.global.u64 %rd33, [%rd40];\ncvta.to.global.u64 %rd33, %rd33;\n$L_t2:");
    k.l(&format!(
        "mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r2, {c}, %r10;\nadd.u32 %r14, %r10, {};",
        nt * c
    ));
    k.l("min.u64 %rd36, %rd4, %rd5;\nmov.u64 %rd34, 0;\nsetp.eq.u64 %p4, %rd36, 0;\n@%p4 bra $L_solve;");
    k.l("$L_j:\nmul.lo.u64 %rd35, %rd34, %rd7;");
    // squared norm of A(j+1:m, j)
    k.l(&format!(
        "mov.{t} %f0, %f60;\nadd.u64 %rd37, %rd34, 1;\nadd.u64 %rd37, %rd37, %rd29;\n$L_q:\nsetp.ge.u64 %p5, %rd37, %rd4;\n@%p5 bra $L_qd;\nadd.u64 %rd43, %rd35, %rd37;"
    ));
    k.elem(44, 31, 43);
    k.ld(44, V(2));
    k.l(&format!("fma.rn.{t} %f0, %f2, %f2, %f0;"));
    if cx {
        k.l(&format!("fma.rn.{t} %f0, %f3, %f3, %f0;"));
    }
    k.l(&format!("add.u64 %rd37, %rd37, {nt};\nbra $L_q;\n$L_qd:"));
    k.real_sum_tree(0, 1, nt, "ssq");
    // thread 0: reflector of (alpha, x) = A(j:m, j); tau %f30/31, 1/(alpha - beta) %f32/33
    k.l("setp.ne.u32 %p6, %r2, 0;\n@%p6 bra $L_hb;\nadd.u64 %rd43, %rd35, %rd34;");
    k.elem(44, 31, 43);
    let alpha = V(4);
    k.ld(44, alpha);
    let (tau, s) = (V(30), V(32));
    k.zero(tau);
    k.zero(s);
    k.l(&format!("mov.{t} %f32, {};\nsetp.eq.{t} %p7, %f0, %f60;", ty.one()));
    if cx {
        k.l(&format!("setp.eq.{t} %p8, %f5, %f60;\nand.pred %p7, %p7, %p8;"));
    }
    k.l("@%p7 bra $L_hw;");
    // beta = -sign(alpha_re) sqrt(|alpha|^2 + ||x||^2)
    k.l(&format!(
        "fma.rn.{t} %f12, %f4, %f4, %f0;"
    ));
    if cx {
        k.l(&format!("fma.rn.{t} %f12, %f5, %f5, %f12;"));
    }
    k.l(&format!(
        "sqrt.rn.{t} %f12, %f12;\nneg.{t} %f13, %f12;\nsetp.ge.{t} %p8, %f4, %f60;\nselp.{t} %f12, %f13, %f12, %p8;\nsub.rn.{t} %f30, %f12, %f4;\ndiv.rn.{t} %f30, %f30, %f12;\nsub.rn.{t} %f16, %f4, %f12;"
    ));
    if cx {
        k.l(&format!(
            "neg.{t} %f31, %f5;\ndiv.rn.{t} %f31, %f31, %f12;\nmul.rn.{t} %f18, %f16, %f16;\nfma.rn.{t} %f18, %f5, %f5, %f18;\ndiv.rn.{t} %f32, %f16, %f18;\nneg.{t} %f33, %f5;\ndiv.rn.{t} %f33, %f33, %f18;"
        ));
    } else {
        k.l(&format!("div.rn.{t} %f32, %f32, %f16;"));
    }
    k.l(&format!("mov.{t} %f4, %f12;"));
    if cx {
        k.l(&format!("mov.{t} %f5, %f60;"));
    }
    k.st(44, alpha);
    k.l("$L_hw:\n@!%p2 bra $L_ht;");
    k.elem(45, 32, 34);
    k.st(45, tau);
    k.l("$L_ht:");
    k.l(&format!(
        "st.shared.{t} [%r14], %f30;\nst.shared.{t} [%r14+{}], %f32;",
        2 * c
    ));
    if cx {
        k.l(&format!(
            "st.shared.{t} [%r14+{c}], %f31;\nst.shared.{t} [%r14+{}], %f33;",
            3 * c
        ));
    }
    k.l("$L_hb:\nbar.sync 0;");
    k.l(&format!(
        "ld.shared.{t} %f30, [%r14];\nld.shared.{t} %f32, [%r14+{}];",
        2 * c
    ));
    if cx {
        k.l(&format!(
            "ld.shared.{t} %f31, [%r14+{c}];\nld.shared.{t} %f33, [%r14+{}];",
            3 * c
        ));
    }
    // v = x / (alpha - beta)
    k.l("add.u64 %rd37, %rd34, 1;\nadd.u64 %rd37, %rd37, %rd29;\n$L_v:\nsetp.ge.u64 %p5, %rd37, %rd4;\n@%p5 bra $L_vd;\nadd.u64 %rd43, %rd35, %rd37;");
    k.elem(44, 31, 43);
    k.ld(44, V(2));
    k.mul(V(6), V(2), s);
    k.st(44, V(6));
    k.l(&format!("add.u64 %rd37, %rd37, {nt};\nbra $L_v;\n$L_vd:\nbar.sync 0;"));
    // apply H^H = I - conj(tau) v v^H to the trailing columns of A and to C
    k.mov(V(34), tau);
    k.conj(V(34));
    k.l("add.u64 %rd38, %rd34, 1;\nadd.u64 %rd38, %rd38, %rd29;");
    qr_apply(&mut k, 31, 7, 5, "ua");
    k.l("@!%p3 bra $L_cskip;\nmov.u64 %rd38, %rd29;");
    qr_apply(&mut k, 33, 8, 6, "uc");
    k.l("$L_cskip:\nbar.sync 0;");
    k.l("add.u64 %rd34, %rd34, 1;\nsetp.lt.u64 %p4, %rd34, %rd36;\n@%p4 bra $L_j;");
    k.l("$L_solve:\n@!%p3 bra $L_out;");
    // info: first zero diagonal entry of R
    k.l("mov.u32 %r20, 0;\nmov.u64 %rd37, 0;\n$L_z:\nsetp.ge.u64 %p5, %rd37, %rd5;\n@%p5 bra $L_zd;\nmad.lo.u64 %rd43, %rd37, %rd7, %rd37;");
    k.elem(44, 31, 43);
    k.ld(44, V(2));
    k.is_zero(9, V(2));
    k.l("@%p9 bra $L_zf;\nadd.u64 %rd37, %rd37, 1;\nbra $L_z;\n$L_zf:\nadd.u64 %rd38, %rd37, 1;\ncvt.u32.u64 %r20, %rd38;\n$L_zd:");
    k.l("setp.ne.u32 %p6, %r2, 0;\nsetp.eq.u64 %p5, %rd3, 0;\nor.pred %p6, %p6, %p5;\n@%p6 bra $L_iw;\ncvta.to.global.u64 %rd45, %rd3;\nmad.lo.u64 %rd45, %rd30, 4, %rd45;\nst.global.u32 [%rd45], %r20;\n$L_iw:");
    k.l("setp.ne.u32 %p5, %r20, 0;\n@%p5 bra $L_out;");
    // back substitution R x = c for the right-hand sides tid, tid + nt, ...
    k.l("mov.u64 %rd38, %rd29;\n$L_r:\nsetp.ge.u64 %p5, %rd38, %rd6;\n@%p5 bra $L_out;\nmul.lo.u64 %rd39, %rd38, %rd8;\nmov.u64 %rd37, %rd5;");
    k.l("$L_i:\nsetp.eq.u64 %p6, %rd37, 0;\n@%p6 bra $L_rn;\nsub.u64 %rd37, %rd37, 1;\nadd.u64 %rd43, %rd39, %rd37;");
    k.elem(44, 33, 43);
    k.ld(44, V(2));
    k.l("add.u64 %rd40, %rd37, 1;\n$L_l:\nsetp.ge.u64 %p7, %rd40, %rd5;\n@%p7 bra $L_ld;\nmad.lo.u64 %rd46, %rd40, %rd7, %rd37;");
    k.elem(47, 31, 46);
    k.ld(47, V(4));
    k.neg(V(4));
    k.l("add.u64 %rd48, %rd39, %rd40;");
    k.elem(49, 33, 48);
    k.ld(49, V(6));
    k.fma(V(2), V(4), V(6));
    k.l("add.u64 %rd40, %rd40, 1;\nbra $L_l;\n$L_ld:\nmad.lo.u64 %rd46, %rd37, %rd7, %rd37;");
    k.elem(47, 31, 46);
    k.ld(47, V(4));
    k.div(V(2), V(4));
    k.st(44, V(2));
    k.l(&format!("bra $L_i;\n$L_rn:\nadd.u64 %rd38, %rd38, {nt};\nbra $L_r;"));
    k.done()
}

/// Register-blocked variant of [`gemm`] (same parameters and operand modes): the 16 x 16
/// threads of a work-group compute a `16 rb` square of C (`rb` = [`gemm_rb`]), thread
/// (tx, ty) the outputs `(i0 + tx + 16 r, j0 + ty + 16 c)`, from 16-deep shared tiles of
/// op(A) and op(B) (each thread loads `rb` elements of each). Launch: block (16,16), grid
/// (ceil(m/(16 rb)), ceil(n/(16 rb)), batch).
fn gemm_blocked(ty: Ty, tables: bool) -> String {
    let k_ = if tables { Kernel::GemmBP } else { Kernel::GemmBS };
    let name = entry_name(k_, ty);
    let slot = ty.slot();
    let rb = gemm_rb(ty);
    let bm = TILE * rb;
    let half = TILE * bm * slot;
    let lg = bm.trailing_zeros();
    let mut k = K::new(&name, k_.params(), ty, 2 * half);
    k.l("mov.u32 %r1, %ctaid.z;\ncvt.u64.u32 %rd59, %r1;");
    k.batch_base(31, 0, 17, 59, tables);
    k.batch_base(32, 1, 18, 59, tables);
    k.batch_base(33, 2, 19, 59, tables);
    k.l("mov.u32 %r2, %tid.x;\nmov.u32 %r3, %tid.y;\nmov.u32 %r4, %ctaid.x;\nmov.u32 %r5, %ctaid.y;");
    k.l(&format!(
        "mul.wide.u32 %rd34, %r4, {bm};\nmul.wide.u32 %rd35, %r5, {bm};\ncvt.u64.u32 %rd36, %r2;\ncvt.u64.u32 %rd37, %r3;"
    ));
    k.l("setp.ne.u64 %p1, %rd20, 0;\nsetp.ne.u64 %p2, %rd21, 0;");
    k.l(&format!(
        "mov.u32 %r10, smem;\nmad.lo.u32 %r11, %r3, 16, %r2;\nadd.u32 %r13, %r10, {half};\nmad.lo.u32 %r14, %r2, {slot}, %r10;\nmad.lo.u32 %r15, %r3, {slot}, %r13;"
    ));
    let acc = |r: u32, c: u32| V(64 + 2 * (r * rb + c));
    for r in 0..rb {
        for c in 0..rb {
            k.zero(acc(r, c));
        }
    }
    k.l("mov.u64 %rd41, 0;\nsetp.eq.u64 %p3, %rd11, 0;\n@%p3 bra $L_epi;");
    k.l("sub.u64 %rd38, %rd9, 1;\nsub.u64 %rd39, %rd10, 1;\nsub.u64 %rd40, %rd11, 1;");
    k.l("$L_k:");
    for q in 0..rb {
        // op(A) tile element e = tid + 256 q: row e % bm, depth e / bm, stored at slot e
        k.l(&format!(
            "add.u32 %r20, %r11, {};\nand.b32 %r21, %r20, {};\nshr.u32 %r22, %r20, {lg};\ncvt.u64.u32 %rd42, %r21;\nadd.u64 %rd42, %rd34, %rd42;\ncvt.u64.u32 %rd43, %r22;\nadd.u64 %rd43, %rd41, %rd43;",
            256 * q,
            bm - 1
        ));
        k.l("setp.lt.u64 %p4, %rd42, %rd9;\nsetp.lt.u64 %p5, %rd43, %rd11;\nselp.u64 %rd44, %rd42, %rd38, %p4;\nselp.u64 %rd45, %rd43, %rd40, %p5;");
        operand(&mut k, 31, 46, 44, 45, 12, 13, 25, 26, 27, V(2), &format!("a{q}"));
        k.l("and.pred %p6, %p4, %p5;");
        k.keep_if(V(2), 6);
        k.conj_if(V(2), 1);
        k.l(&format!("mad.lo.u32 %r23, %r20, {slot}, %r10;"));
        k.sts(23, V(2));
    }
    for q in 0..rb {
        // op(B) tile element e: depth e % 16, column e / 16, stored at depth * bm + column
        k.l(&format!(
            "add.u32 %r20, %r11, {};\nand.b32 %r21, %r20, 15;\nshr.u32 %r22, %r20, 4;\ncvt.u64.u32 %rd42, %r21;\nadd.u64 %rd42, %rd41, %rd42;\ncvt.u64.u32 %rd43, %r22;\nadd.u64 %rd43, %rd35, %rd43;",
            256 * q
        ));
        k.l("setp.lt.u64 %p7, %rd42, %rd11;\nsetp.lt.u64 %p8, %rd43, %rd10;\nselp.u64 %rd48, %rd42, %rd40, %p7;\nselp.u64 %rd49, %rd43, %rd39, %p8;");
        operand(&mut k, 32, 51, 48, 49, 14, 15, 28, 29, 30, V(2), &format!("b{q}"));
        k.l("and.pred %p9, %p7, %p8;");
        k.keep_if(V(2), 9);
        k.conj_if(V(2), 2);
        k.l(&format!(
            "mad.lo.u32 %r23, %r21, {bm}, %r22;\nmad.lo.u32 %r23, %r23, {slot}, %r13;"
        ));
        k.sts(23, V(2));
    }
    k.l("bar.sync 0;");
    for l in 0..TILE {
        for r in 0..rb {
            k.lds_off(14, (l * bm + 16 * r) * slot, V(96 + 2 * r));
        }
        for c in 0..rb {
            k.lds_off(15, (l * bm + 16 * c) * slot, V(104 + 2 * c));
        }
        for r in 0..rb {
            for c in 0..rb {
                k.fma(acc(r, c), V(96 + 2 * r), V(104 + 2 * c));
            }
        }
    }
    k.l("bar.sync 0;");
    k.l("add.u64 %rd41, %rd41, 16;\nsetp.lt.u64 %p3, %rd41, %rd11;\n@%p3 bra $L_k;");
    k.l("$L_epi:");
    let (alpha, beta, v, cv) = (V(20), V(22), V(24), V(26));
    k.scalar(alpha, 3, 4, 7, None, "alpha");
    k.scalar(beta, 5, 6, 8, None, "beta");
    k.is_zero(12, alpha);
    k.l("not.pred %p12, %p12;");
    k.is_zero(13, beta);
    for r in 0..rb {
        for c in 0..rb {
            let skip = format!("$L_e{r}_{c}");
            k.l(&format!(
                "add.u64 %rd70, %rd34, %rd36;\nadd.u64 %rd70, %rd70, {};\nadd.u64 %rd71, %rd35, %rd37;\nadd.u64 %rd71, %rd71, {};",
                16 * r,
                16 * c
            ));
            k.l(&format!(
                "setp.ge.u64 %p10, %rd70, %rd9;\nsetp.ge.u64 %p11, %rd71, %rd10;\nor.pred %p10, %p10, %p11;\n@%p10 bra {skip};"
            ));
            // triangle-only store (syrk, herk, syr2k, ...)
            k.l(&format!(
                "setp.eq.u64 %p14, %rd23, 1;\nsetp.lt.u64 %p15, %rd70, %rd71;\nand.pred %p14, %p14, %p15;\n@%p14 bra {skip};\nsetp.eq.u64 %p14, %rd23, 2;\nsetp.gt.u64 %p15, %rd70, %rd71;\nand.pred %p14, %p14, %p15;\n@%p14 bra {skip};"
            ));
            k.mul(v, alpha, acc(r, c));
            k.keep_if(v, 12);
            k.l("mul.lo.u64 %rd52, %rd70, %rd22;\nmad.lo.u64 %rd52, %rd71, %rd16, %rd52;");
            k.elem(53, 33, 52);
            k.l(&format!("@%p13 bra {skip}_st;"));
            k.ld(53, cv);
            k.fma(v, beta, cv);
            k.l(&format!("{skip}_st:"));
            if ty.complex() {
                let t = ty.ft();
                k.l(&format!(
                    "setp.ne.u64 %p14, %rd24, 0;\nsetp.eq.u64 %p15, %rd70, %rd71;\nand.pred %p14, %p14, %p15;\nselp.{t} %f25, %f60, %f25, %p14;"
                ));
            }
            k.st(53, v);
            k.l(&format!("{skip}:"));
        }
    }
    k.done()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_have_declared_params_and_unique_names() {
        let e = all_entries();
        assert_eq!(e.len(), 4 * 19 + 2);
        let mut names: Vec<&String> = e.iter().map(|(n, _, _)| n).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), e.len());
        for (name, src, n) in &e {
            assert!(src.contains(&format!(".entry {name}(")));
            assert_eq!(src.matches(".param .u64").count(), *n, "{name}");
            assert_eq!(src.matches("$L_out:").count(), 1, "{name}");
        }
    }

    #[test]
    fn gemm_barriers_precede_every_exit() {
        for ty in Ty::BLAS {
            for k in [Kernel::GemmS, Kernel::GemmP, Kernel::GemmBS, Kernel::GemmBP] {
                let s = single(k, ty);
                assert_eq!(s.matches("bar.sync 0;").count(), 2);
                assert!(s.find("bra $L_out").map_or(true, |o| o > s.rfind("bar.sync").unwrap()));
                assert_eq!(
                    s.contains("ld.global.u64 %rd31"),
                    matches!(k, Kernel::GemmP | Kernel::GemmBP)
                );
                assert_eq!(s.contains(".shared .align 16 .b8 smem["), true);
            }
        }
        // complex tiles hold two components per element
        assert!(single(Kernel::GemmS, Ty::Z).contains("smem[8192]"));
        assert!(single(Kernel::GemmS, Ty::S).contains("smem[2048]"));
    }

    #[test]
    fn reductions_reach_every_barrier() {
        for ty in Ty::ALL {
            let s = single(Kernel::Dot, ty);
            // 1 + 8 tree levels, no early exit before the last one
            assert_eq!(s.matches("bar.sync 0;").count(), 9);
            assert!(s.find("bra $L_out").unwrap() > s.rfind("bar.sync").unwrap());
        }
        for ty in Ty::BLAS {
            // partials: max tree 9 + 1, sum tree 9 + 1; fallback: 9 + 1 + 9
            let s = single(Kernel::Nrm2, ty);
            assert_eq!(s.matches("bar.sync 0;").count(), 20 + 19);
            let s = single(Kernel::Nrm2P, ty);
            assert_eq!(s.matches("bar.sync 0;").count(), 18);
            assert!(s.find("bra $L_out").unwrap() > s.rfind("bar.sync").unwrap());
        }
    }

    #[test]
    fn type_specific_snippets() {
        assert!(single(Kernel::Dot, Ty::H).contains("cvt.f32.f16"));
        assert!(single(Kernel::Dot, Ty::B).contains("selp.u32 %r60, 32704"));
        assert!(single(Kernel::GemmS, Ty::Z).contains("ld.global.f64 %f3, [%rd46+8];"));
        assert!(!single(Kernel::GemmS, Ty::D).contains("%rd46+8"));
        assert!(single(Kernel::TrsmS, Ty::C).contains("div.rn.f32 %f1"));
        assert!(single(Kernel::Scal, Ty::Z).contains("setp.ne.u64 %p21, %rd6, 0;"));
    }
}

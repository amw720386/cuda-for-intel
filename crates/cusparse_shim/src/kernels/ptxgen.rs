//! PTX for the cuSPARSE shim. The Driver API translates it to SPIR-V like any application
//! PTX (the route the cuBLAS and cuFFT shims take). Dependency-free so
//! `src/ptx/cusparse_tests.rs` can include it.
//!
//! One module per [`Variant`] (value type, offset index width, column index width) holds:
//! * `gather` - one thread per (row, column of C): `C(r,j) = alpha * sum_e val[e] * B(col[e], j)
//!   + beta * C(r,j)` over a CSR row. Scalar CSR SpMV/SpMM for short rows.
//! * `vgather` - the same with one 64-thread block per (row, column of C) and a shared-memory
//!   tree reduction. Vector CSR kernel for long rows.
//! * `scatter` - `C(d,j) += alpha * val[e] * B(s,j)` with a compare-and-swap float add, for the
//!   transposed CSR (mode 0, one thread per row), COO (mode 1) and transposed COO (mode 2,
//!   one thread per nonzero). Runs after `scale`.
//! * `scale` - `C = beta * C` (beta 0 writes zeros, so NaNs in C do not survive).
//! * `sddmm` - `val[e] = alpha * (op(A) op(B))(r, col[e]) + beta * val[e]`, one thread per row.
//! * `bsr` - block-CSR times dense, one thread per (row, column of C).
//!
//! The index module (32-bit only) holds `coo2csr`, `csr2coo` and `identity`.
//!
//! Every parameter is a `.u64` slot (the launcher passes 8-byte values); scalars travel as
//! their bit patterns, and a nonzero `*_ptr` parameter makes the kernel load the scalar from
//! device memory (`CUSPARSE_POINTER_MODE_DEVICE`). Dense operands are addressed as
//! `base + (i * si + j * sj) * size`, so one kernel covers both orders and `op(B)`.
//! Batch `b = %ctaid.y` advances each pointer by its batch stride (in elements).

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Variant {
    pub double: bool,
    /// 64-bit row offsets (CSR) or row indices (COO).
    pub ptr64: bool,
    /// 64-bit column indices.
    pub idx64: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kernel {
    Gather,
    VGather,
    Scatter,
    Scale,
    Sddmm,
    Bsr,
}

pub const KERNELS: [Kernel; 6] = [Kernel::Gather, Kernel::VGather, Kernel::Scatter, Kernel::Scale, Kernel::Sddmm, Kernel::Bsr];

/// Threads per block of `vgather` (a power of two; the reduction is unrolled for it).
pub const VBLOCK: u32 = 64;

impl Kernel {
    pub fn tag(self) -> &'static str {
        match self {
            Kernel::Gather => "gather",
            Kernel::VGather => "vgather",
            Kernel::Scatter => "scatter",
            Kernel::Scale => "scale",
            Kernel::Sddmm => "sddmm",
            Kernel::Bsr => "bsr",
        }
    }

    pub fn params(self) -> usize {
        match self {
            Kernel::Gather | Kernel::VGather | Kernel::Sddmm => 20,
            Kernel::Scatter => 19,
            Kernel::Scale => 8,
            Kernel::Bsr => 18,
        }
    }
}

impl Variant {
    pub fn suffix(self) -> String {
        format!(
            "{}_p{}_i{}",
            if self.double { "f64" } else { "f32" },
            if self.ptr64 { 64 } else { 32 },
            if self.idx64 { 64 } else { 32 }
        )
    }

    pub fn all() -> Vec<Variant> {
        let mut v = Vec::new();
        for double in [false, true] {
            for ptr64 in [false, true] {
                for idx64 in [false, true] {
                    v.push(Variant { double, ptr64, idx64 });
                }
            }
        }
        v
    }
}

pub fn entry_name(k: Kernel, v: Variant) -> String {
    format!("cfi_sp_{}_{}", k.tag(), v.suffix())
}

pub const COO2CSR: &str = "cfi_sp_coo2csr";
pub const CSR2COO: &str = "cfi_sp_csr2coo";
pub const IDENTITY: &str = "cfi_sp_identity";
/// Parameter count of each index kernel.
pub const INDEX_PARAMS: usize = 5;

const HEAD: &str = ".version 7.0\n.target sm_60\n.address_size 64\n\n";

/// Kernel text builder: every parameter is a u64 loaded into `%rd<i>`.
struct K {
    s: String,
    t: &'static str,
    sz: u32,
    label: u32,
}

impl K {
    fn new(name: &str, nparams: usize, double: bool) -> K {
        let p: Vec<String> = (0..nparams).map(|i| format!(".param .u64 {name}_p{i}")).collect();
        let mut s = format!(".visible .entry {name}(\n{}\n)\n{{\n", p.join(",\n"));
        let t = if double { "f64" } else { "f32" };
        s += &format!(".reg .pred %p<16>;\n.reg .b32 %r<32>;\n.reg .b64 %rd<96>;\n.reg .{t} %f<24>;\n");
        for i in 0..nparams {
            s += &format!("ld.param.u64 %rd{i}, [{name}_p{i}];\n");
        }
        K { s, t, sz: if double { 8 } else { 4 }, label: 0 }
    }

    fn l(&mut self, line: &str) {
        self.s += line;
        self.s.push('\n');
    }

    fn label(&mut self, stem: &str) -> String {
        self.label += 1;
        format!("$L_{stem}_{}", self.label)
    }

    fn zero(&self) -> &'static str {
        if self.t == "f64" { "0d0000000000000000" } else { "0f00000000" }
    }

    /// `%rd<g>` = global thread index (64-bit), `%rd<b>` = batch index.
    fn gid(&mut self, g: u32, b: u32) {
        self.l("mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %ntid.x;\nmov.u32 %r3, %tid.x;");
        self.l(&format!("mul.wide.u32 %rd{g}, %r1, %r2;\ncvt.u64.u32 %rd95, %r3;\nadd.u64 %rd{g}, %rd{g}, %rd95;"));
        self.l(&format!("mov.u32 %r4, %ctaid.y;\ncvt.u64.u32 %rd{b}, %r4;"));
    }

    /// Converts the generic pointer in `%rd<p>` to global and advances it by
    /// `batch * %rd<stride> * size` bytes (`stride` may be `None` for no batching).
    fn ptr(&mut self, p: u32, batch: u32, stride: Option<u32>, size: u32) {
        self.l(&format!("cvta.to.global.u64 %rd{p}, %rd{p};"));
        if let Some(st) = stride {
            self.l(&format!("mul.lo.u64 %rd94, %rd{batch}, %rd{st};\nmad.lo.u64 %rd{p}, %rd94, {size}, %rd{p};"));
        }
    }

    /// `%rd<dst>` = index array `%rd<arr>`[`%rd<i>`] (u32 or u64 elements) minus `%rd<base>`.
    fn ld_index(&mut self, dst: u32, arr: u32, i: u32, wide: bool, base: u32) {
        if wide {
            self.l(&format!("mad.lo.u64 %rd94, %rd{i}, 8, %rd{arr};\nld.global.u64 %rd{dst}, [%rd94];"));
        } else {
            self.l(&format!("mad.lo.u64 %rd94, %rd{i}, 4, %rd{arr};\nld.global.u32 %r5, [%rd94];\ncvt.u64.u32 %rd{dst}, %r5;"));
        }
        self.l(&format!("sub.u64 %rd{dst}, %rd{dst}, %rd{base};"));
    }

    /// `%rd<dst>` = address of element `%rd<i>` of value array `%rd<arr>`.
    fn val_addr(&mut self, dst: u32, arr: u32, i: u32) {
        let sz = self.sz;
        self.l(&format!("mad.lo.u64 %rd{dst}, %rd{i}, {sz}, %rd{arr};"));
    }

    /// `%rd<dst>` = address of dense element (`%rd<i>`, `%rd<j>`) with strides `%rd<si>`, `%rd<sj>`.
    fn dense_addr(&mut self, dst: u32, arr: u32, i: u32, j: u32, si: u32, sj: u32) {
        let sz = self.sz;
        self.l(&format!("mul.lo.u64 %rd93, %rd{i}, %rd{si};\nmad.lo.u64 %rd93, %rd{j}, %rd{sj}, %rd93;\nmad.lo.u64 %rd{dst}, %rd93, {sz}, %rd{arr};"));
    }

    /// `%f<dst>` = scalar from bits `%rd<bits>`, or loaded from `%rd<ptr>` when it is nonzero.
    fn scalar(&mut self, dst: u32, bits: u32, ptr: u32) {
        let t = self.t;
        if t == "f64" {
            self.l(&format!("mov.b64 %f{dst}, %rd{bits};"));
        } else {
            self.l(&format!("cvt.u32.u64 %r6, %rd{bits};\nmov.b32 %f{dst}, %r6;"));
        }
        let skip = self.label("sc");
        self.l(&format!("setp.eq.u64 %p10, %rd{ptr}, 0;\n@%p10 bra {skip};\ncvta.to.global.u64 %rd92, %rd{ptr};\nld.global.{t} %f{dst}, [%rd92];\n{skip}:"));
    }

    /// `[%rd<addr>] = alpha * %f<acc> (+ beta * [%rd<addr>] when beta != 0)`.
    fn store_axpby(&mut self, addr: u32, acc: u32, alpha: u32, beta: u32) {
        let (t, z) = (self.t, self.zero());
        let skip = self.label("nobeta");
        self.l(&format!("mul.{t} %f{acc}, %f{acc}, %f{alpha};\nsetp.eq.{t} %p11, %f{beta}, {z};\n@%p11 bra {skip};"));
        self.l(&format!("ld.global.{t} %f20, [%rd{addr}];\nfma.rn.{t} %f{acc}, %f{beta}, %f20, %f{acc};\n{skip}:\nst.global.{t} [%rd{addr}], %f{acc};"));
    }

    /// Atomic `[%rd<addr>] += %f<v>` by compare-and-swap (integer atomics only).
    fn atomic_add(&mut self, addr: u32, v: u32) {
        let t = self.t;
        let lp = self.label("cas");
        if t == "f64" {
            self.l(&format!("ld.global.u64 %rd90, [%rd{addr}];\n{lp}:\nmov.b64 %f21, %rd90;\nadd.f64 %f21, %f21, %f{v};\nmov.b64 %rd91, %f21;"));
            self.l(&format!("atom.global.cas.b64 %rd89, [%rd{addr}], %rd90, %rd91;\nsetp.ne.u64 %p12, %rd89, %rd90;\nmov.u64 %rd90, %rd89;\n@%p12 bra {lp};"));
        } else {
            self.l(&format!("ld.global.u32 %r10, [%rd{addr}];\n{lp}:\nmov.b32 %f21, %r10;\nadd.f32 %f21, %f21, %f{v};\nmov.b32 %r11, %f21;"));
            self.l(&format!("atom.global.cas.b32 %r12, [%rd{addr}], %r10, %r11;\nsetp.ne.u32 %p12, %r12, %r10;\nmov.u32 %r10, %r12;\n@%p12 bra {lp};"));
        }
    }

    fn finish(mut self) -> String {
        self.l("$L_done:\nret;\n}");
        self.s
    }
}

// Parameter slots shared by gather and vgather.
// 0 off, 1 col, 2 val, 3 b, 4 c, 5 rows, 6 n, 7 base, 8 bsi, 9 bsj, 10 csi, 11 csj,
// 12 alpha, 13 beta, 14 alpha_ptr, 15 beta_ptr, 16 off_bs, 17 cv_bs, 18 b_bs, 19 c_bs
fn gather_prologue(k: &mut K, v: Variant) {
    let (psz, isz, sz) = (if v.ptr64 { 8 } else { 4 }, if v.idx64 { 8 } else { 4 }, k.sz);
    k.ptr(0, 21, Some(16), psz);
    k.ptr(1, 21, Some(17), isz);
    k.ptr(2, 21, Some(17), sz);
    k.ptr(3, 21, Some(18), sz);
    k.ptr(4, 21, Some(19), sz);
}

fn gather(v: Variant) -> String {
    let mut k = K::new(&entry_name(Kernel::Gather, v), Kernel::Gather.params(), v.double);
    let (t, z) = (k.t, k.zero());
    k.gid(20, 21);
    k.l("mul.lo.u64 %rd22, %rd5, %rd6;\nsetp.ge.u64 %p1, %rd20, %rd22;\n@%p1 bra $L_done;");
    gather_prologue(&mut k, v);
    // r = rd23, j = rd24
    k.l("div.u64 %rd23, %rd20, %rd6;\nrem.u64 %rd24, %rd20, %rd6;\nadd.u64 %rd25, %rd23, 1;");
    k.ld_index(26, 0, 23, v.ptr64, 7); // lo
    k.ld_index(27, 0, 25, v.ptr64, 7); // hi
    k.l(&format!("mov.{t} %f1, {z};"));
    k.l("$L_loop:\nsetp.ge.u64 %p2, %rd26, %rd27;\n@%p2 bra $L_end;");
    k.ld_index(28, 1, 26, v.idx64, 7);
    k.val_addr(29, 2, 26);
    k.l(&format!("ld.global.{t} %f2, [%rd29];"));
    k.dense_addr(30, 3, 28, 24, 8, 9);
    k.l(&format!("ld.global.{t} %f3, [%rd30];\nfma.rn.{t} %f1, %f2, %f3, %f1;\nadd.u64 %rd26, %rd26, 1;\nbra $L_loop;\n$L_end:"));
    k.scalar(4, 12, 14);
    k.scalar(5, 13, 15);
    k.dense_addr(31, 4, 23, 24, 10, 11);
    k.store_axpby(31, 1, 4, 5);
    k.finish()
}

fn vgather(v: Variant) -> String {
    let name = entry_name(Kernel::VGather, v);
    let mut k = K::new(&name, Kernel::VGather.params(), v.double);
    let (t, z, sz) = (k.t, k.zero(), k.sz);
    // shared scratch, one value per lane (32-bit shared addresses, as the cuBLAS shim uses)
    k.s = k.s.replacen("{\n", &format!("{{\n.shared .align 8 .b8 {name}_red[{}];\n", VBLOCK * sz), 1);
    // one block per (row, j): g = ctaid.x, lane = tid.x
    k.l("mov.u32 %r1, %ctaid.x;\ncvt.u64.u32 %rd20, %r1;\nmov.u32 %r4, %ctaid.y;\ncvt.u64.u32 %rd21, %r4;\nmov.u32 %r3, %tid.x;\ncvt.u64.u32 %rd32, %r3;");
    k.l("mul.lo.u64 %rd22, %rd5, %rd6;\nsetp.ge.u64 %p1, %rd20, %rd22;\n@%p1 bra $L_done;");
    gather_prologue(&mut k, v);
    k.l("div.u64 %rd23, %rd20, %rd6;\nrem.u64 %rd24, %rd20, %rd6;\nadd.u64 %rd25, %rd23, 1;");
    k.ld_index(26, 0, 23, v.ptr64, 7);
    k.ld_index(27, 0, 25, v.ptr64, 7);
    k.l(&format!("add.u64 %rd26, %rd26, %rd32;\nmov.{t} %f1, {z};"));
    k.l("$L_loop:\nsetp.ge.u64 %p2, %rd26, %rd27;\n@%p2 bra $L_end;");
    k.ld_index(28, 1, 26, v.idx64, 7);
    k.val_addr(29, 2, 26);
    k.l(&format!("ld.global.{t} %f2, [%rd29];"));
    k.dense_addr(30, 3, 28, 24, 8, 9);
    k.l(&format!("ld.global.{t} %f3, [%rd30];\nfma.rn.{t} %f1, %f2, %f3, %f1;\nadd.u64 %rd26, %rd26, {VBLOCK};\nbra $L_loop;\n$L_end:"));
    // tree reduction in shared memory
    k.l(&format!("mov.u32 %r13, {name}_red;\nmad.lo.u32 %r14, %r3, {sz}, %r13;\nst.shared.{t} [%r14], %f1;\nbar.sync 0;"));
    let mut s = VBLOCK / 2;
    while s >= 1 {
        let skip = k.label("red");
        k.l(&format!(
            "setp.ge.u32 %p3, %r3, {s};\n@%p3 bra {skip};\nld.shared.{t} %f6, [%r14];\nld.shared.{t} %f7, [%r14+{}];\nadd.{t} %f6, %f6, %f7;\nst.shared.{t} [%r14], %f6;\n{skip}:\nbar.sync 0;",
            s * sz
        ));
        s /= 2;
    }
    k.l("setp.ne.u32 %p4, %r3, 0;\n@%p4 bra $L_done;");
    k.l(&format!("ld.shared.{t} %f1, [%r13];"));
    k.scalar(4, 12, 14);
    k.scalar(5, 13, 15);
    k.dense_addr(31, 4, 23, 24, 10, 11);
    k.store_axpby(31, 1, 4, 5);
    k.finish()
}

// 0 p0, 1 idx, 2 val, 3 b, 4 c, 5 count, 6 n, 7 base, 8 mode, 9 bsi, 10 bsj, 11 csi, 12 csj,
// 13 alpha, 14 alpha_ptr, 15 p0_bs, 16 cv_bs, 17 b_bs, 18 c_bs
fn scatter(v: Variant) -> String {
    let mut k = K::new(&entry_name(Kernel::Scatter, v), Kernel::Scatter.params(), v.double);
    let t = k.t;
    let (psz, isz, sz) = (if v.ptr64 { 8 } else { 4 }, if v.idx64 { 8 } else { 4 }, k.sz);
    k.gid(20, 21);
    k.l("mul.lo.u64 %rd22, %rd5, %rd6;\nsetp.ge.u64 %p1, %rd20, %rd22;\n@%p1 bra $L_done;");
    k.ptr(0, 21, Some(15), psz);
    k.ptr(1, 21, Some(16), isz);
    k.ptr(2, 21, Some(16), sz);
    k.ptr(3, 21, Some(17), sz);
    k.ptr(4, 21, Some(18), sz);
    // t = rd23, j = rd24
    k.l("div.u64 %rd23, %rd20, %rd6;\nrem.u64 %rd24, %rd20, %rd6;");
    k.l("setp.ne.u64 %p5, %rd8, 0;\n@%p5 bra $L_coo;");
    k.l("add.u64 %rd25, %rd23, 1;");
    k.ld_index(26, 0, 23, v.ptr64, 7);
    k.ld_index(27, 0, 25, v.ptr64, 7);
    k.l("bra $L_range;\n$L_coo:\nmov.u64 %rd26, %rd23;\nadd.u64 %rd27, %rd23, 1;\n$L_range:");
    k.scalar(4, 13, 14);
    k.l("$L_loop:\nsetp.ge.u64 %p2, %rd26, %rd27;\n@%p2 bra $L_done;");
    k.ld_index(28, 1, 26, v.idx64, 7); // x = idx[e]
    // (s, d): mode 0 (r, x); mode 1 (x, y); mode 2 (y, x), y = p0[e]
    k.l("mov.u64 %rd35, %rd23;\nmov.u64 %rd36, %rd28;\n@!%p5 bra $L_sd;");
    k.ld_index(29, 0, 26, v.ptr64, 7);
    k.l("setp.eq.u64 %p6, %rd8, 1;\nselp.b64 %rd35, %rd28, %rd29, %p6;\nselp.b64 %rd36, %rd29, %rd28, %p6;\n$L_sd:");
    k.val_addr(30, 2, 26);
    k.l(&format!("ld.global.{t} %f2, [%rd30];"));
    k.dense_addr(31, 3, 35, 24, 9, 10);
    k.l(&format!("ld.global.{t} %f3, [%rd31];\nmul.{t} %f2, %f2, %f4;\nmul.{t} %f2, %f2, %f3;"));
    k.dense_addr(37, 4, 36, 24, 11, 12);
    k.atomic_add(37, 2);
    k.l("add.u64 %rd26, %rd26, 1;\nbra $L_loop;");
    k.finish()
}

// 0 c, 1 rows, 2 n, 3 csi, 4 csj, 5 beta, 6 beta_ptr, 7 c_bs
fn scale(v: Variant) -> String {
    let mut k = K::new(&entry_name(Kernel::Scale, v), Kernel::Scale.params(), v.double);
    let (t, z, sz) = (k.t, k.zero(), k.sz);
    k.gid(20, 21);
    k.l("mul.lo.u64 %rd22, %rd1, %rd2;\nsetp.ge.u64 %p1, %rd20, %rd22;\n@%p1 bra $L_done;");
    k.ptr(0, 21, Some(7), sz);
    k.l("div.u64 %rd23, %rd20, %rd2;\nrem.u64 %rd24, %rd20, %rd2;");
    k.dense_addr(25, 0, 23, 24, 3, 4);
    k.scalar(1, 5, 6);
    k.l(&format!("mov.{t} %f2, {z};\nsetp.eq.{t} %p2, %f1, {z};\n@%p2 bra $L_st;\nld.global.{t} %f2, [%rd25];\nmul.{t} %f2, %f2, %f1;\n$L_st:\nst.global.{t} [%rd25], %f2;"));
    k.finish()
}

// 0 off, 1 col, 2 val, 3 a, 4 b, 5 rows, 6 k, 7 base, 8 asi, 9 asj, 10 bsi, 11 bsj,
// 12 alpha, 13 beta, 14 alpha_ptr, 15 beta_ptr, 16 off_bs, 17 cv_bs, 18 a_bs, 19 b_bs
fn sddmm(v: Variant) -> String {
    let mut k = K::new(&entry_name(Kernel::Sddmm, v), Kernel::Sddmm.params(), v.double);
    let (t, z, sz) = (k.t, k.zero(), k.sz);
    let (psz, isz) = (if v.ptr64 { 8 } else { 4 }, if v.idx64 { 8 } else { 4 });
    k.gid(20, 21);
    k.l("setp.ge.u64 %p1, %rd20, %rd5;\n@%p1 bra $L_done;");
    k.ptr(0, 21, Some(16), psz);
    k.ptr(1, 21, Some(17), isz);
    k.ptr(2, 21, Some(17), sz);
    k.ptr(3, 21, Some(18), sz);
    k.ptr(4, 21, Some(19), sz);
    k.l("add.u64 %rd25, %rd20, 1;");
    k.ld_index(26, 0, 20, v.ptr64, 7);
    k.ld_index(27, 0, 25, v.ptr64, 7);
    k.scalar(4, 12, 14);
    k.scalar(5, 13, 15);
    k.l("$L_loop:\nsetp.ge.u64 %p2, %rd26, %rd27;\n@%p2 bra $L_done;");
    k.ld_index(28, 1, 26, v.idx64, 7);
    k.l(&format!("mov.{t} %f1, {z};\nmov.u64 %rd38, 0;\n$L_dot:\nsetp.ge.u64 %p3, %rd38, %rd6;\n@%p3 bra $L_dotend;"));
    k.dense_addr(30, 3, 20, 38, 8, 9);
    k.dense_addr(31, 4, 38, 28, 10, 11);
    k.l(&format!("ld.global.{t} %f2, [%rd30];\nld.global.{t} %f3, [%rd31];\nfma.rn.{t} %f1, %f2, %f3, %f1;\nadd.u64 %rd38, %rd38, 1;\nbra $L_dot;\n$L_dotend:"));
    k.val_addr(29, 2, 26);
    k.store_axpby(29, 1, 4, 5);
    k.l("add.u64 %rd26, %rd26, 1;\nbra $L_loop;");
    k.finish()
}

// 0 ptr, 1 col, 2 val, 3 b, 4 c, 5 mb, 6 n, 7 base, 8 bd, 9 dir, 10 bsi, 11 bsj, 12 csi, 13 csj,
// 14 alpha, 15 beta, 16 alpha_ptr, 17 beta_ptr
fn bsr(v: Variant) -> String {
    let mut k = K::new(&entry_name(Kernel::Bsr, v), Kernel::Bsr.params(), v.double);
    let (t, z) = (k.t, k.zero());
    k.gid(20, 21);
    k.l("mul.lo.u64 %rd22, %rd5, %rd8;\nmul.lo.u64 %rd22, %rd22, %rd6;\nsetp.ge.u64 %p1, %rd20, %rd22;\n@%p1 bra $L_done;");
    for p in 0..5 {
        k.ptr(p, 21, None, 0);
    }
    // j = rd24, t = rd23 -> ri = rd39, ib = rd40, row = rd41
    k.l("div.u64 %rd23, %rd20, %rd6;\nrem.u64 %rd24, %rd20, %rd6;\nrem.u64 %rd39, %rd23, %rd8;\ndiv.u64 %rd40, %rd23, %rd8;\nmad.lo.u64 %rd41, %rd40, %rd8, %rd39;\nadd.u64 %rd25, %rd40, 1;");
    k.ld_index(26, 0, 40, v.ptr64, 7);
    k.ld_index(27, 0, 25, v.ptr64, 7);
    k.l(&format!("mov.{t} %f1, {z};\nmul.lo.u64 %rd42, %rd8, %rd8;\nsetp.ne.u64 %p7, %rd9, 0;"));
    k.l("$L_blk:\nsetp.ge.u64 %p2, %rd26, %rd27;\n@%p2 bra $L_end;");
    k.ld_index(28, 1, 26, v.idx64, 7);
    k.l("mul.lo.u64 %rd43, %rd26, %rd42;\nmul.lo.u64 %rd44, %rd28, %rd8;\nmov.u64 %rd45, 0;");
    k.l("$L_in:\nsetp.ge.u64 %p3, %rd45, %rd8;\n@%p3 bra $L_inend;");
    // vidx = e*bd*bd + (dir == row ? ri*bd + cj : cj*bd + ri)
    k.l("mad.lo.u64 %rd46, %rd39, %rd8, %rd45;\nmad.lo.u64 %rd47, %rd45, %rd8, %rd39;\nselp.b64 %rd46, %rd47, %rd46, %p7;\nadd.u64 %rd46, %rd46, %rd43;\nadd.u64 %rd48, %rd44, %rd45;");
    k.val_addr(29, 2, 46);
    k.dense_addr(30, 3, 48, 24, 10, 11);
    k.l(&format!("ld.global.{t} %f2, [%rd29];\nld.global.{t} %f3, [%rd30];\nfma.rn.{t} %f1, %f2, %f3, %f1;\nadd.u64 %rd45, %rd45, 1;\nbra $L_in;\n$L_inend:\nadd.u64 %rd26, %rd26, 1;\nbra $L_blk;\n$L_end:"));
    k.scalar(4, 14, 16);
    k.scalar(5, 15, 17);
    k.dense_addr(31, 4, 41, 24, 12, 13);
    k.store_axpby(31, 1, 4, 5);
    k.finish()
}

fn kernel(kind: Kernel, v: Variant) -> String {
    match kind {
        Kernel::Gather => gather(v),
        Kernel::VGather => vgather(v),
        Kernel::Scatter => scatter(v),
        Kernel::Scale => scale(v),
        Kernel::Sddmm => sddmm(v),
        Kernel::Bsr => bsr(v),
    }
}

/// The module for one variant (all six entries).
pub fn module(v: Variant) -> String {
    let mut s = HEAD.to_owned();
    for k in KERNELS {
        s += &kernel(k, v);
        s.push('\n');
    }
    s
}

// Index kernels: 0 in, 1 nnz, 2 m, 3 out, 4 base (identity: 0 out, 1 n).
fn coo2csr() -> String {
    // csr[i] = lower_bound(coo[0..nnz], i + base) + base for i in 0..=m (signed compare)
    let mut k = K::new(COO2CSR, INDEX_PARAMS, false);
    k.gid(20, 21);
    k.l("setp.gt.u64 %p1, %rd20, %rd2;\n@%p1 bra $L_done;\nadd.u64 %rd22, %rd20, %rd4;\ncvt.u32.u64 %r7, %rd22;");
    k.ptr(0, 21, None, 0);
    k.ptr(3, 21, None, 0);
    k.l("mov.u64 %rd23, 0;\nmov.u64 %rd24, %rd1;\n$L_bs:\nsetp.ge.u64 %p2, %rd23, %rd24;\n@%p2 bra $L_found;");
    k.l("add.u64 %rd25, %rd23, %rd24;\nshr.u64 %rd25, %rd25, 1;\nmad.lo.u64 %rd26, %rd25, 4, %rd0;\nld.global.s32 %r8, [%rd26];");
    k.l("setp.lt.s32 %p3, %r8, %r7;\nadd.u64 %rd27, %rd25, 1;\nselp.b64 %rd23, %rd27, %rd23, %p3;\nselp.b64 %rd24, %rd24, %rd25, %p3;\nbra $L_bs;");
    k.l("$L_found:\nadd.u64 %rd23, %rd23, %rd4;\ncvt.u32.u64 %r9, %rd23;\nmad.lo.u64 %rd28, %rd20, 4, %rd3;\nst.global.u32 [%rd28], %r9;");
    k.finish()
}

fn csr2coo() -> String {
    let mut k = K::new(CSR2COO, INDEX_PARAMS, false);
    k.gid(20, 21);
    k.l("setp.ge.u64 %p1, %rd20, %rd2;\n@%p1 bra $L_done;");
    k.ptr(0, 21, None, 0);
    k.ptr(3, 21, None, 0);
    k.l("add.u64 %rd25, %rd20, 1;");
    k.ld_index(26, 0, 20, false, 4);
    k.ld_index(27, 0, 25, false, 4);
    k.l("add.u64 %rd22, %rd20, %rd4;\ncvt.u32.u64 %r7, %rd22;");
    k.l("$L_loop:\nsetp.ge.u64 %p2, %rd26, %rd27;\n@%p2 bra $L_done;\nmad.lo.u64 %rd28, %rd26, 4, %rd3;\nst.global.u32 [%rd28], %r7;\nadd.u64 %rd26, %rd26, 1;\nbra $L_loop;");
    k.finish()
}

fn identity() -> String {
    let mut k = K::new(IDENTITY, INDEX_PARAMS, false);
    k.gid(20, 21);
    k.l("setp.ge.u64 %p1, %rd20, %rd1;\n@%p1 bra $L_done;");
    k.ptr(0, 21, None, 0);
    k.l("cvt.u32.u64 %r7, %rd20;\nmad.lo.u64 %rd28, %rd20, 4, %rd0;\nst.global.u32 [%rd28], %r7;");
    k.finish()
}

pub fn index_module() -> String {
    format!("{HEAD}{}\n{}\n{}\n", coo2csr(), csr2coo(), identity())
}

/// `(entry name, single-entry module source, parameter count)` for every kernel.
pub fn all_entries() -> Vec<(String, String, usize)> {
    let mut out = Vec::new();
    for v in Variant::all() {
        for k in KERNELS {
            out.push((entry_name(k, v), format!("{HEAD}{}", kernel(k, v)), k.params()));
        }
    }
    out.push((COO2CSR.to_owned(), format!("{HEAD}{}", coo2csr()), INDEX_PARAMS));
    out.push((CSR2COO.to_owned(), format!("{HEAD}{}", csr2coo()), INDEX_PARAMS));
    out.push((IDENTITY.to_owned(), format!("{HEAD}{}", identity()), INDEX_PARAMS));
    out
}

//! Native XMX (Intel DPAS) GEMM kernels for the cuBLAS shim.
//!
//! The kernels are emitted directly as SPIR-V (see [`build`]) because PTX has
//! no way to express `OpSubgroupMatrixMultiplyAccumulateINTEL`
//! (`SPV_INTEL_subgroup_matrix_multiply_accumulate`). The PTX `mma.sync`
//! lowering was rejected in `docs/xmx-design.md` (branch `xmx-mma`); this is the
//! route it recommends: SIMD16 kernels that load operands straight into the
//! layout DPAS wants, so no fragment conversion exists at all.
//!
//! # Formulation
//!
//! cuBLAS matrices are column-major, so `C(i,j)` lives at `C[i + j*ldc]` and
//! lane `i` of a subgroup writing consecutive `i` coalesces. DPAS yields
//! `R[r][lane]` (rows `r` = component, lane = column), therefore the kernel
//! computes `C^T = op(B)^T * op(A)^T`:
//!
//! * DPAS "A" (`M'=8` rows = 8 columns `j` of C, `K'=16`): lane `l` holds the
//!   `<8 x i16>` `op(B)(kk+l, j0+r)`, `r = 0..8`.
//! * DPAS "B" (`K'=16` x `N'=16` = 16 rows `i` of C): lane `i` holds `<8 x i32>`
//!   VNNI pairs `op(A)(i, kk+2p) | op(A)(i, kk+2p+1) << 16`.
//! * accumulator/result `<8 x f32>`: component `r` of lane `i` is `C(i, j0+r)`.
//!
//! One subgroup (16 invocations, `SubgroupSize 16`, `LocalSize 16x1x1`) owns an
//! `IB*16 x JB*8` tile of C (default 32 x 32 = eight accumulators). The K loop
//! steps 16 at a time and needs no masking; one masked tail iteration handles
//! `k % 16`. Rows/columns beyond `m`/`n` are loaded with clamped indices (the
//! results land in lanes/components that are never stored; DPAS columns and rows
//! are independent), so loads never leave the allocation.
//!
//! Operand loads are plain ordinary loads (`SPV_INTEL_2d_block_io` is not
//! assumed); two variants use wider loads when the host proves alignment:
//! `pair_a` (A stored K-contiguous, one dword per VNNI pair) and `vec_b` (B
//! stored K-strided with eight contiguous columns, one 16-byte load).
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FPRoundingMode, FunctionControl, LoopControl, MatrixMultiplyAccumulateOperands, MemoryAccess,
    MemoryModel, OpenclStd100Op, SelectionControl, StorageClass, Word,
};

pub const EXTENSION: &str = "SPV_INTEL_subgroup_matrix_multiply_accumulate";
/// DPAS shape for 16-bit inputs on Xe2/Xe3: M' (rows per op), K', N' (= subgroup size).
pub const DPAS_M: u32 = 8;
pub const DPAS_K: u32 = 16;
pub const DPAS_N: u32 = 16;

type R<T> = Result<T, rspirv::dr::Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    F16,
    Bf16,
}

/// Element type of C and of the bias vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    F16,
    Bf16,
    F32,
}

impl Ty {
    pub fn bytes(self) -> u64 {
        if self == Ty::F32 { 4 } else { 2 }
    }
    fn tag(self) -> &'static str {
        match self {
            Ty::F16 => "f16",
            Ty::Bf16 => "bf16",
            Ty::F32 => "f32",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    None,
    Relu,
    Gelu,
}

/// How the multiply-accumulate operand mask is spelled. Hardware validation
/// picks the first variant whose self-test passes (see `xmx_host`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flags {
    /// `MatrixAPackedFloat16 | MatrixBPackedFloat16` (or the BFloat16 pair).
    Packed,
    /// No operand mask at all.
    Bare,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cfg {
    pub input: Input,
    pub out: Ty,
    /// Operands are device pointer tables (`cublasGemmBatchedEx`) instead of strided.
    pub ptrs: bool,
    /// beta == 0: C is never read.
    pub beta0: bool,
    pub act: Act,
    /// Bias vector element type (indexed by row i), if any.
    pub bias: Option<Ty>,
    /// C-tile height in 16-row DPAS blocks.
    pub ib: u32,
    /// C-tile width in 8-column DPAS blocks.
    pub jb: u32,
    pub pair_a: bool,
    pub vec_b: bool,
    pub flags: Flags,
}

impl Cfg {
    pub fn tile_m(&self) -> u32 {
        self.ib * DPAS_N
    }
    pub fn tile_n(&self) -> u32 {
        self.jb * DPAS_M
    }
    pub fn name(&self) -> String {
        let mut n = format!(
            "cfi_xmx_{}{}{}_{}x{}",
            match self.input {
                Input::F16 => "f16",
                Input::Bf16 => "bf16",
            },
            self.out.tag(),
            if self.ptrs { "p" } else { "s" },
            self.ib,
            self.jb
        );
        if self.beta0 {
            n.push_str("_b0");
        }
        match self.act {
            Act::None => {}
            Act::Relu => n.push_str("_relu"),
            Act::Gelu => n.push_str("_gelu"),
        }
        if let Some(b) = self.bias {
            n.push_str("_bias");
            n.push_str(b.tag());
        }
        if self.pair_a {
            n.push_str("_pa");
        }
        if self.vec_b {
            n.push_str("_vb");
        }
        if self.flags == Flags::Bare {
            n.push_str("_bare");
        }
        n
    }
}

/// Kernel parameters, in order. Identical to the PTX GEMM kernels for 0..=15;
/// 16 is the bias pointer (0 when the variant has no bias).
pub const PARAMS: [&str; 17] = [
    "u64", "u64", "u64", "f32", "f32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32",
    "u64", "u64", "u64", "u64",
];

struct G {
    b: Builder,
    cfg: Cfg,
    ocl: Word,
    t_bool: Word,
    t_u16: Word,
    t_u32: Word,
    t_u64: Word,
    t_f32: Word,
    t_f16: Word,
    t_v4u32: Word,
    t_v8u16: Word,
    t_v8u32: Word,
    t_v8f32: Word,
    p_u16: Word,
    p_u32: Word,
    p_u64: Word,
    p_f32: Word,
    p_v4u32: Word,
}

/// Values shared by every block of the kernel body.
struct Ctx {
    lane: Word,
    j0: Word,
    n: Word,
    k: Word,
    a_sl2: Word,
    b_sl2: Word,
    b_sj2: Word,
    bbase: Word,
    a_ic: Vec<Word>, // per ib: A pointer-int at the (clamped) row of this lane
}

impl G {
    fn c32(&mut self, v: u32) -> Word {
        self.b.constant_bit32(self.t_u32, v)
    }
    fn c64(&mut self, v: u64) -> Word {
        self.b.constant_bit64(self.t_u64, v)
    }
    fn cf(&mut self, v: f32) -> Word {
        self.b.constant_bit32(self.t_f32, v.to_bits())
    }
    fn ext(&mut self, ty: Word, op: OpenclStd100Op, args: &[Word]) -> R<Word> {
        self.b.ext_inst(ty, None, self.ocl, op as u32, args.iter().map(|&a| Operand::IdRef(a)))
    }
    fn to64(&mut self, v: Word) -> R<Word> {
        self.b.u_convert(self.t_u64, None, v)
    }
    fn load(&mut self, ty: Word, ptr_ty: Word, addr: Word, align: u32) -> R<Word> {
        let p = self.b.convert_u_to_ptr(ptr_ty, None, addr)?;
        self.b.load(ty, None, p, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(align)])
    }
    fn store(&mut self, ptr_ty: Word, addr: Word, v: Word, align: u32) -> R<()> {
        let p = self.b.convert_u_to_ptr(ptr_ty, None, addr)?;
        self.b.store(p, v, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(align)])
    }
    /// 64-bit `base + idx * scale` where `idx` is a u32 value and `scale` a u64 value.
    fn addr(&mut self, base: Word, idx32: Word, scale: Word) -> R<Word> {
        let i = self.to64(idx32)?;
        let o = self.b.i_mul(self.t_u64, None, i, scale)?;
        self.b.i_add(self.t_u64, None, base, o)
    }

    /// Element at `addr` of type `ty`, converted to f32.
    fn load_f32(&mut self, ty: Ty, addr: Word) -> R<Word> {
        Ok(match ty {
            Ty::F32 => self.load(self.t_f32, self.p_f32, addr, 4)?,
            Ty::F16 => {
                let bits = self.load(self.t_u16, self.p_u16, addr, 2)?;
                let h = self.b.bitcast(self.t_f16, None, bits)?;
                self.b.f_convert(self.t_f32, None, h)?
            }
            Ty::Bf16 => {
                let bits = self.load(self.t_u16, self.p_u16, addr, 2)?;
                let w = self.b.u_convert(self.t_u32, None, bits)?;
                let s = self.c32(16);
                let w = self.b.shift_left_logical(self.t_u32, None, w, s)?;
                self.b.bitcast(self.t_f32, None, w)?
            }
        })
    }

    /// Round the f32 `v` to `ty` and store it at `addr`.
    fn store_ty(&mut self, ty: Ty, addr: Word, v: Word) -> R<()> {
        match ty {
            Ty::F32 => self.store(self.p_f32, addr, v, 4),
            Ty::F16 => {
                let h = self.b.f_convert(self.t_f16, None, v)?;
                self.b.decorate(
                    h,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(FPRoundingMode::RTE)],
                );
                let bits = self.b.bitcast(self.t_u16, None, h)?;
                self.store(self.p_u16, addr, bits, 2)
            }
            Ty::Bf16 => {
                let x = self.b.bitcast(self.t_u32, None, v)?;
                let nan = self.b.is_nan(self.t_bool, None, v)?;
                let s16 = self.c32(16);
                let one = self.c32(1);
                let hi = self.b.shift_right_logical(self.t_u32, None, x, s16)?;
                let lsb = self.b.bitwise_and(self.t_u32, None, hi, one)?;
                let k = self.c32(0x7fff);
                let r = self.b.i_add(self.t_u32, None, lsb, k)?;
                let r = self.b.i_add(self.t_u32, None, x, r)?;
                let r = self.b.shift_right_logical(self.t_u32, None, r, s16)?;
                let qn = self.c32(0x7fc0);
                let r = self.b.select(self.t_u32, None, nan, qn, r)?;
                let bits = self.b.u_convert(self.t_u16, None, r)?;
                self.store(self.p_u16, addr, bits, 2)
            }
        }
    }

    /// DPAS "A" operand for column block `jb`: lane l holds `op(B)(kk+l, j0+8jb+r)`.
    fn a_operand(&mut self, c: &Ctx, jb: u32, kk: Word, rem: Option<Word>) -> R<Word> {
        let one = self.c32(1);
        let l = self.b.i_add(self.t_u32, None, kk, c.lane)?;
        let km1 = self.b.i_sub(self.t_u32, None, c.k, one)?;
        let lc = if rem.is_some() { self.ext(self.t_u32, OpenclStd100Op::u_min, &[l, km1])? } else { l };
        let jbase = {
            let o = self.c32(jb * DPAS_M);
            self.b.i_add(self.t_u32, None, c.j0, o)?
        };
        if self.cfg.vec_b && rem.is_none() {
            // n % 8 == 0 and b_sj == 1: eight contiguous columns, one 16-byte load.
            let eight = self.c32(8);
            let nm8 = self.b.i_sub(self.t_u32, None, c.n, eight)?;
            let js = self.ext(self.t_u32, OpenclStd100Op::u_min, &[jbase, nm8])?;
            let two = self.c64(2);
            let a = self.addr(c.bbase, lc, c.b_sl2)?;
            let a = self.addr(a, js, two)?;
            let v = self.load(self.t_v4u32, self.p_v4u32, a, 16)?;
            return self.b.bitcast(self.t_v8u16, None, v);
        }
        let lrow = self.addr(c.bbase, lc, c.b_sl2)?;
        let nm1 = self.b.i_sub(self.t_u32, None, c.n, one)?;
        let mut comps = Vec::with_capacity(8);
        for r in 0..DPAS_M {
            let rr = self.c32(r);
            let j = self.b.i_add(self.t_u32, None, jbase, rr)?;
            let jc = self.ext(self.t_u32, OpenclStd100Op::u_min, &[j, nm1])?;
            let a = self.addr(lrow, jc, c.b_sj2)?;
            let mut v = self.load(self.t_u16, self.p_u16, a, 2)?;
            if let Some(rem) = rem {
                let ok = self.b.u_less_than(self.t_bool, None, c.lane, rem)?;
                let z = self.b.constant_bit32(self.t_u16, 0);
                v = self.b.select(self.t_u16, None, ok, v, z)?;
            }
            comps.push(v);
        }
        self.b.composite_construct(self.t_v8u16, None, comps)
    }

    /// DPAS "B" operand for row block `ib`: lane i holds the VNNI pairs of
    /// `op(A)(i0+16ib+i, kk..kk+16)`.
    fn b_operand(&mut self, c: &Ctx, ib: u32, kk: Word, rem: Option<Word>) -> R<Word> {
        let arow = c.a_ic[ib as usize];
        let one = self.c32(1);
        let km1 = self.b.i_sub(self.t_u32, None, c.k, one)?;
        let mut comps = Vec::with_capacity(8);
        for p in 0..8u32 {
            if self.cfg.pair_a && rem.is_none() {
                // a_sl == 1, even a_si, 4-byte aligned base: one dword is one VNNI pair.
                let off = self.c32(2 * p);
                let l = self.b.i_add(self.t_u32, None, kk, off)?;
                let two = self.c64(2);
                let a = self.addr(arow, l, two)?;
                comps.push(self.load(self.t_u32, self.p_u32, a, 4)?);
                continue;
            }
            let mut halves = [0; 2];
            for h in 0..2u32 {
                let off = self.c32(2 * p + h);
                let l = self.b.i_add(self.t_u32, None, kk, off)?;
                let lc = if rem.is_some() {
                    self.ext(self.t_u32, OpenclStd100Op::u_min, &[l, km1])?
                } else {
                    l
                };
                let a = self.addr(arow, lc, c.a_sl2)?;
                let mut v = self.load(self.t_u16, self.p_u16, a, 2)?;
                if let Some(rem) = rem {
                    let ok = self.b.u_less_than(self.t_bool, None, off, rem)?;
                    let z = self.b.constant_bit32(self.t_u16, 0);
                    v = self.b.select(self.t_u16, None, ok, v, z)?;
                }
                halves[h as usize] = self.b.u_convert(self.t_u32, None, v)?;
            }
            let s16 = self.c32(16);
            let hi = self.b.shift_left_logical(self.t_u32, None, halves[1], s16)?;
            comps.push(self.b.bitwise_or(self.t_u32, None, halves[0], hi)?);
        }
        self.b.composite_construct(self.t_v8u32, None, comps)
    }

    fn dpas(&mut self, a: Word, b: Word, acc: Word, id: Option<Word>) -> R<Word> {
        let k = self.c32(DPAS_K);
        let flags = match (self.cfg.flags, self.cfg.input) {
            (Flags::Bare, _) => None,
            (Flags::Packed, Input::F16) => Some(
                MatrixMultiplyAccumulateOperands::MATRIX_A_PACKED_FLOAT16_INTEL
                    | MatrixMultiplyAccumulateOperands::MATRIX_B_PACKED_FLOAT16_INTEL,
            ),
            (Flags::Packed, Input::Bf16) => Some(
                MatrixMultiplyAccumulateOperands::MATRIX_A_PACKED_B_FLOAT16_INTEL
                    | MatrixMultiplyAccumulateOperands::MATRIX_B_PACKED_B_FLOAT16_INTEL,
            ),
        };
        self.b.subgroup_matrix_multiply_accumulate_intel(self.t_v8f32, id, k, a, b, acc, flags)
    }

    /// One K step of 16 for every (jb, ib) accumulator. `ids` pre-assigns the result ids.
    fn step(
        &mut self,
        c: &Ctx,
        kk: Word,
        rem: Option<Word>,
        accs: &[Word],
        ids: Option<&[Word]>,
    ) -> R<Vec<Word>> {
        let (ib_n, jb_n) = (self.cfg.ib, self.cfg.jb);
        let bops: Vec<Word> =
            (0..ib_n).map(|ib| self.b_operand(c, ib, kk, rem)).collect::<R<_>>()?;
        let mut out = Vec::with_capacity(accs.len());
        for jb in 0..jb_n {
            let a = self.a_operand(c, jb, kk, rem)?;
            for ib in 0..ib_n {
                let idx = (jb * ib_n + ib) as usize;
                out.push(self.dpas(a, bops[ib as usize], accs[idx], ids.map(|v| v[idx]))?);
            }
        }
        Ok(out)
    }
}

/// Build the SPIR-V words of one kernel variant.
pub fn build(cfg: &Cfg) -> R<Vec<u32>> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    let need_f16 = cfg.out == Ty::F16 || cfg.bias == Some(Ty::F16);
    let mut caps = vec![
        Capability::Addresses,
        Capability::Kernel,
        Capability::Int16,
        Capability::Int64,
        Capability::Vector16,
        Capability::SubgroupDispatch,
        Capability::SubgroupMatrixMultiplyAccumulateINTEL,
    ];
    if need_f16 {
        caps.push(Capability::Float16);
    }
    for c in caps {
        b.capability(c);
    }
    b.extension(EXTENSION);
    b.memory_model(AddressingModel::Physical64, MemoryModel::OpenCL);
    let ocl = b.ext_inst_import("OpenCL.std");
    let t_bool = b.type_bool();
    let t_u16 = b.type_int(16, 0);
    let t_u32 = b.type_int(32, 0);
    let t_u64 = b.type_int(64, 0);
    let t_f32 = b.type_float(32, None);
    let t_f16 = if need_f16 { b.type_float(16, None) } else { 0 };
    let t_v3u64 = b.type_vector(t_u64, 3);
    let t_v4u32 = b.type_vector(t_u32, 4);
    let t_v8u16 = b.type_vector(t_u16, 8);
    let t_v8u32 = b.type_vector(t_u32, 8);
    let t_v8f32 = b.type_vector(t_f32, 8);
    let gp = |b: &mut Builder, t| b.type_pointer(None, StorageClass::CrossWorkgroup, t);
    let (p_u16, p_u32, p_u64, p_f32, p_v4u32) = (
        gp(&mut b, t_u16),
        gp(&mut b, t_u32),
        gp(&mut b, t_u64),
        gp(&mut b, t_f32),
        gp(&mut b, t_v4u32),
    );
    let in_ptr = b.type_pointer(None, StorageClass::Input, t_v3u64);
    let wgid = b.variable(in_ptr, None, StorageClass::Input, None);
    b.decorate(wgid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let lid = b.variable(in_ptr, None, StorageClass::Input, None);
    b.decorate(lid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);

    let mut g = G {
        b, cfg: *cfg, ocl, t_bool, t_u16, t_u32, t_u64, t_f32, t_f16, t_v4u32, t_v8u16, t_v8u32,
        t_v8f32, p_u16, p_u32, p_u64, p_f32, p_v4u32,
    };
    let void = g.b.type_void();
    let ptys: Vec<Word> = PARAMS
        .iter()
        .map(|t| match *t {
            "u64" => t_u64,
            "u32" => t_u32,
            _ => t_f32,
        })
        .collect();
    let fty = g.b.type_function(void, ptys.iter().copied());
    let f = g.b.begin_function(void, None, FunctionControl::NONE, fty)?;
    let p: Vec<Word> = ptys.iter().map(|&t| g.b.function_parameter(t)).collect::<R<_>>()?;
    g.b.entry_point(ExecutionModel::Kernel, f, cfg.name(), [wgid, lid]);
    g.b.execution_mode(f, ExecutionMode::SubgroupSize, [DPAS_N]);
    g.b.execution_mode(f, ExecutionMode::LocalSize, [DPAS_N, 1, 1]);
    let (pa, pb, pc, alpha, beta) = (p[0], p[1], p[2], p[3], p[4]);
    let (m, n, k, a_si, a_sl, b_sl, b_sj, ldc) = (p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12]);
    let (sa, sb, sc, pbias) = (p[13], p[14], p[15], p[16]);

    let entry = g.b.id();
    let header = g.b.id();
    let body = g.b.id();
    let merge = g.b.id();
    let tail = g.b.id();
    let after = g.b.id();
    g.b.begin_block(Some(entry))?;

    let wg = g.b.load(t_v3u64, None, wgid, None, [])?;
    let li = g.b.load(t_v3u64, None, lid, None, [])?;
    let wx = g.b.composite_extract(t_u64, None, wg, [0])?;
    let wy = g.b.composite_extract(t_u64, None, wg, [1])?;
    let wz = g.b.composite_extract(t_u64, None, wg, [2])?;
    let lane64 = g.b.composite_extract(t_u64, None, li, [0])?;
    let lane = g.b.u_convert(t_u32, None, lane64)?;
    let wx32 = g.b.u_convert(t_u32, None, wx)?;
    let wy32 = g.b.u_convert(t_u32, None, wy)?;
    let tm = g.c32(cfg.tile_m());
    let tn = g.c32(cfg.tile_n());
    let i0 = g.b.i_mul(t_u32, None, wx32, tm)?;
    let j0 = g.b.i_mul(t_u32, None, wy32, tn)?;

    // Per-batch base pointers (as integers).
    let (abase, bbase, cbase) = if cfg.ptrs {
        let eight = g.c64(8);
        let off = g.b.i_mul(t_u64, None, wz, eight)?;
        let mut v = [0; 3];
        for (slot, table) in [pa, pb, pc].into_iter().enumerate() {
            let at = g.b.i_add(t_u64, None, table, off)?;
            v[slot] = g.load(t_u64, p_u64, at, 8)?;
        }
        (v[0], v[1], v[2])
    } else {
        let two = g.c64(2);
        let cs = g.c64(cfg.out.bytes());
        let mut v = [0; 3];
        for (slot, (base, stride, scale)) in
            [(pa, sa, two), (pb, sb, two), (pc, sc, cs)].into_iter().enumerate()
        {
            let o = g.b.i_mul(t_u64, None, wz, stride)?;
            let o = g.b.i_mul(t_u64, None, o, scale)?;
            v[slot] = g.b.i_add(t_u64, None, base, o)?;
        }
        (v[0], v[1], v[2])
    };
    let two = g.c64(2);
    let scale2 = |g: &mut G, v: Word| -> R<Word> {
        let w = g.to64(v)?;
        g.b.i_mul(t_u64, None, w, two)
    };
    let a_sl2 = scale2(&mut g, a_sl)?;
    let a_si2 = scale2(&mut g, a_si)?;
    let b_sl2 = scale2(&mut g, b_sl)?;
    let b_sj2 = scale2(&mut g, b_sj)?;
    let one = g.c32(1);
    let mm1 = g.b.i_sub(t_u32, None, m, one)?;
    let mut a_ic = Vec::new();
    for ib in 0..cfg.ib {
        let off = g.c32(ib * DPAS_N);
        let i = g.b.i_add(t_u32, None, i0, off)?;
        let i = g.b.i_add(t_u32, None, i, lane)?;
        let ic = g.ext(t_u32, OpenclStd100Op::u_min, &[i, mm1])?;
        a_ic.push(g.addr(abase, ic, a_si2)?);
    }
    let ctx = Ctx { lane, j0, n, k, a_sl2, b_sl2, b_sj2, bbase, a_ic };
    g.b.branch(header)?;

    // Main loop over full 16-wide K steps.
    let nacc = (cfg.ib * cfg.jb) as usize;
    g.b.begin_block(Some(header))?;
    let zero = g.b.constant_null(t_v8f32);
    let kk_next = g.b.id();
    let acc_next: Vec<Word> = (0..nacc).map(|_| g.b.id()).collect();
    let zero32 = g.c32(0);
    let kk = g.b.phi(t_u32, None, [(zero32, entry), (kk_next, body)])?;
    let accs_h: Vec<Word> = (0..nacc)
        .map(|i| g.b.phi(t_v8f32, None, [(zero, entry), (acc_next[i], body)]))
        .collect::<R<_>>()?;
    let sixteen = g.c32(DPAS_K);
    let kk16 = g.b.i_add(t_u32, None, kk, sixteen)?;
    let full = g.b.u_less_than_equal(t_bool, None, kk16, k)?;
    g.b.loop_merge(merge, body, LoopControl::NONE, [])?;
    g.b.branch_conditional(full, body, merge, [])?;

    g.b.begin_block(Some(body))?;
    // The results carry the ids the header phis already reference.
    g.step(&ctx, kk, None, &accs_h, Some(&acc_next))?;
    g.b.i_add(t_u32, Some(kk_next), kk, sixteen)?;
    g.b.branch(header)?;

    // Tail: k % 16 leftover columns of K, masked.
    g.b.begin_block(Some(merge))?;
    let has_tail = g.b.u_less_than(t_bool, None, kk, k)?;
    g.b.selection_merge(after, SelectionControl::NONE)?;
    g.b.branch_conditional(has_tail, tail, after, [])?;

    g.b.begin_block(Some(tail))?;
    let rem = g.b.i_sub(t_u32, None, k, kk)?;
    let tail_accs = g.step(&ctx, kk, Some(rem), &accs_h, None)?;
    g.b.branch(after)?;

    g.b.begin_block(Some(after))?;
    let accs: Vec<Word> = (0..nacc)
        .map(|i| g.b.phi(t_v8f32, None, [(accs_h[i], merge), (tail_accs[i], tail)]))
        .collect::<R<_>>()?;

    // Epilogue: C(i, j) = act(alpha * acc + beta * C + bias[i]).
    let ldc64 = g.to64(ldc)?;
    for jb in 0..cfg.jb {
        for ib in 0..cfg.ib {
            let acc = accs[(jb * cfg.ib + ib) as usize];
            let ioff = g.c32(ib * DPAS_N);
            let i = g.b.i_add(t_u32, None, i0, ioff)?;
            let i = g.b.i_add(t_u32, None, i, lane)?;
            let i_ok = g.b.u_less_than(t_bool, None, i, m)?;
            for r in 0..DPAS_M {
                let v = g.b.composite_extract(t_f32, None, acc, [r])?;
                let joff = g.c32(jb * DPAS_M + r);
                let j = g.b.i_add(t_u32, None, j0, joff)?;
                let j_ok = g.b.u_less_than(t_bool, None, j, n)?;
                let ok = g.b.logical_and(t_bool, None, i_ok, j_ok)?;
                let then = g.b.id();
                let join = g.b.id();
                g.b.selection_merge(join, SelectionControl::NONE)?;
                g.b.branch_conditional(ok, then, join, [])?;
                g.b.begin_block(Some(then))?;
                let j64 = g.to64(j)?;
                let i64v = g.to64(i)?;
                let col = g.b.i_mul(t_u64, None, j64, ldc64)?;
                let idx = g.b.i_add(t_u64, None, col, i64v)?;
                let sz = g.c64(cfg.out.bytes());
                let byte = g.b.i_mul(t_u64, None, idx, sz)?;
                let caddr = g.b.i_add(t_u64, None, cbase, byte)?;
                let mut x = g.b.f_mul(t_f32, None, v, alpha)?;
                if !cfg.beta0 {
                    let cv = g.load_f32(cfg.out, caddr)?;
                    x = g.ext(t_f32, OpenclStd100Op::fma, &[beta, cv, x])?;
                }
                if let Some(bt) = cfg.bias {
                    let bsz = g.c64(bt.bytes());
                    let a = g.addr(pbias, i, bsz)?;
                    let bv = g.load_f32(bt, a)?;
                    x = g.b.f_add(t_f32, None, x, bv)?;
                }
                match cfg.act {
                    Act::None => {}
                    Act::Relu => {
                        let z = g.cf(0.0);
                        x = g.ext(t_f32, OpenclStd100Op::fmax, &[x, z])?;
                    }
                    Act::Gelu => {
                        let x2 = g.b.f_mul(t_f32, None, x, x)?;
                        let x3 = g.b.f_mul(t_f32, None, x2, x)?;
                        let cub = g.cf(crate::ptxgen::GELU_CUBIC);
                        let inner = g.ext(t_f32, OpenclStd100Op::fma, &[x3, cub, x])?;
                        let s2p = g.cf(crate::ptxgen::GELU_SQRT_2_OVER_PI);
                        let inner = g.b.f_mul(t_f32, None, inner, s2p)?;
                        let th = g.ext(t_f32, OpenclStd100Op::tanh, &[inner])?;
                        let onef = g.cf(1.0);
                        let th = g.b.f_add(t_f32, None, th, onef)?;
                        let half = g.cf(0.5);
                        let hx = g.b.f_mul(t_f32, None, x, half)?;
                        x = g.b.f_mul(t_f32, None, hx, th)?;
                    }
                }
                g.store_ty(cfg.out, caddr, x)?;
                g.b.branch(join)?;
                g.b.begin_block(Some(join))?;
            }
        }
    }
    g.b.ret()?;
    g.b.end_function()?;
    use rspirv::binary::Assemble;
    Ok(g.b.module().assemble())
}

/// ASCII container understood by the driver's SPIR-V pass-through
/// (`src/ptx/passthrough.rs`), loadable with `cuModuleLoadData`.
pub fn container(cfg: &Cfg, words: &[u32]) -> String {
    let mut s = String::from("//CFI-SPIRV 1\n");
    s.push_str(&format!("//ENTRY {} {}\n", cfg.name(), PARAMS.join(",")));
    s.push_str(&format!("//THREADS {DPAS_N}\n//WORDS\n"));
    for chunk in words.chunks(8) {
        let line: Vec<String> = chunk.iter().map(|w| format!("{w:08x}")).collect();
        s.push_str(&line.join(" "));
        s.push('\n');
    }
    s
}

#[cfg(test)]
pub(crate) mod gen_tests {
    use super::*;

    pub fn all_cfgs() -> Vec<Cfg> {
        let mut v = Vec::new();
        for input in [Input::F16, Input::Bf16] {
            for out in [Ty::F16, Ty::Bf16, Ty::F32] {
                for ptrs in [false, true] {
                    for beta0 in [false, true] {
                        for act in [Act::None, Act::Relu, Act::Gelu] {
                            for bias in [None, Some(Ty::F32), Some(Ty::F16), Some(Ty::Bf16)] {
                                v.push(Cfg {
                                    input, out, ptrs, beta0, act, bias, ib: 2, jb: 4,
                                    pair_a: false, vec_b: false, flags: Flags::Packed,
                                });
                            }
                        }
                    }
                }
            }
        }
        for (pair_a, vec_b, flags) in [(true, false, Flags::Packed), (false, true, Flags::Packed), (true, true, Flags::Bare)] {
            v.push(Cfg { input: Input::F16, out: Ty::F32, ptrs: false, beta0: false, act: Act::None, bias: None, ib: 2, jb: 4, pair_a, vec_b, flags });
        }
        v.push(Cfg { input: Input::Bf16, out: Ty::Bf16, ptrs: true, beta0: true, act: Act::Gelu, bias: Some(Ty::F16), ib: 4, jb: 2, pair_a: false, vec_b: false, flags: Flags::Packed });
        v
    }

    /// Runs `spirv-val` (SPIRV-Tools) when installed. The tool predates or
    /// restricts the Intel extension in some releases; `--target-env spv1.3`
    /// is used and an "unknown capability/extension" diagnostic is the only
    /// accepted failure (the structural checks run before it).
    fn validate(words: &[u32], tag: &str) {
        let Ok(tool) = std::process::Command::new("spirv-val").arg("--version").output() else { return };
        if !tool.status.success() {
            return;
        }
        let dir = std::env::temp_dir().join("cfi-xmx-spv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{tag}.spv"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        let out = std::process::Command::new("spirv-val").args(["--target-env", "spv1.3"]).arg(&path).output().unwrap();
        assert!(out.status.success(), "spirv-val rejected {tag}: {}", String::from_utf8_lossy(&out.stderr));
    }

    #[test]
    fn every_variant_builds_and_validates() {
        for cfg in all_cfgs() {
            let words = build(&cfg).unwrap();
            assert_eq!(words[0], 0x0723_0203);
            validate(&words, &cfg.name());
        }
    }

    #[test]
    fn container_has_entry_and_words() {
        let cfg = all_cfgs()[0];
        let words = build(&cfg).unwrap();
        let text = container(&cfg, &words);
        assert!(text.starts_with("//CFI-SPIRV 1\n//ENTRY cfi_xmx_"));
        assert_eq!(text.split("//WORDS\n").nth(1).unwrap().split_whitespace().count(), words.len());
    }
}

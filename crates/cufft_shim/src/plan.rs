//! Pure plan logic for the cuFFT shim: type decoding, data layouts (basic and advanced),
//! the Stockham stage schedule, twiddle tables and work-area size. Nothing here touches the
//! driver, so it is unit-tested on the host and shared with the SPIR-V simulator tests
//! (`src/ptx/cufft_tests.rs`), which run the generated kernels with exactly these parameters.
//!
//! Execution model (see `ptxgen.rs` for the kernels):
//! 1. `gather`: input (any cuFFT layout; complex, real, or Hermitian half) -> a dense complex
//!    work buffer `[batch][n0][n1][n2]`. A Hermitian half is expanded by symmetry, so C2R
//!    runs as a full complex inverse whose real part is the result.
//! 2. the passes of every transformed dimension (`Pass`), ping-ponging between the two
//!    dense halves of the work area:
//!    * `local`: a dimension of length <= `local_cap` whose factors are all <= 63 runs in
//!      one launch, a work-group transforming whole lines in shared memory (every Stockham
//!      stage between two barriers);
//!    * four-step: a longer last dimension `n = n1 n2` with both factors local-sized runs as
//!      two `local` passes (n1-point transforms with a fused `W_n^(j2 k1)` twiddle, then
//!      n2-point transforms whose store transposes the result into place);
//!    * Bluestein: a length with a prime factor > 63 becomes a chirp-z convolution of
//!      power-of-two length `m >= 2n - 1` in two extra work buffers (`chirp` multiplies by
//!      `exp(-i pi j^2 / n)`, a forward m-point FFT, a product with the FFT of the chirp, an
//!      inverse m-point FFT and the output chirp with the 1/m scale);
//!    * otherwise one global `stage` launch per radix (mixed-radix Stockham autosort, one
//!      output per thread).
//! 3. `scatter`: dense work buffer -> output layout (complex, the non-redundant half for
//!    R2C, or the real part for C2R).
//!
//! In-place transforms are safe because the input is fully read into the work area before
//! anything is written to the output.

/// cuFFT status codes.
pub const SUCCESS: i32 = 0;
pub const INVALID_PLAN: i32 = 1;
pub const ALLOC_FAILED: i32 = 2;
pub const INVALID_TYPE: i32 = 3;
pub const INVALID_VALUE: i32 = 4;
pub const INTERNAL_ERROR: i32 = 5;
pub const EXEC_FAILED: i32 = 6;
pub const SETUP_FAILED: i32 = 7;
pub const INVALID_SIZE: i32 = 8;
pub const NOT_SUPPORTED: i32 = 16;

/// Largest dense element count handled (kernel indices are 32-bit).
pub const MAX_ELEMS: u64 = (1 << 31) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    C2C,
    R2C,
    C2R,
}

/// A cuFFT transform type: kind plus precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FftType {
    pub kind: Kind,
    pub double: bool,
}

impl FftType {
    /// `cufftType`: R2C 0x2a, C2R 0x2c, C2C 0x29, D2Z 0x6a, Z2D 0x6c, Z2Z 0x69.
    pub fn from_cufft(t: i32) -> Option<FftType> {
        let (kind, double) = match t {
            0x29 => (Kind::C2C, false),
            0x2a => (Kind::R2C, false),
            0x2c => (Kind::C2R, false),
            0x69 => (Kind::C2C, true),
            0x6a => (Kind::R2C, true),
            0x6c => (Kind::C2R, true),
            _ => return None,
        };
        Some(FftType { kind, double })
    }

    /// `cufftXtMakePlanMany` types (`cudaDataType`): R_32F 0, R_64F 1, R_16F 2, C_32F 4,
    /// C_64F 5, C_16F 6, R_16BF 14, C_16BF 15. Half and bfloat16 are `NOT_SUPPORTED`.
    pub fn from_xt(input: i32, output: i32, exec: i32) -> Result<FftType, i32> {
        let half = |t: i32| matches!(t, 2 | 6 | 14 | 15);
        if half(input) || half(output) || half(exec) {
            return Err(NOT_SUPPORTED);
        }
        let t = match (input, output, exec) {
            (4, 4, 4) => 0x29,
            (0, 4, 4) => 0x2a,
            (4, 0, 4) => 0x2c,
            (5, 5, 5) => 0x69,
            (1, 5, 5) => 0x6a,
            (5, 1, 5) => 0x6c,
            _ => return Err(INVALID_TYPE),
        };
        Ok(FftType::from_cufft(t).unwrap())
    }

    /// Bytes of one real scalar.
    pub fn real_size(self) -> u64 {
        if self.double { 8 } else { 4 }
    }

    pub fn complex_size(self) -> u64 {
        2 * self.real_size()
    }
}

/// Element strides of one side of the transform, normalized to rank 3:
/// `[batch, d0, d1, d2]`, in elements of that side (real or complex).
pub type Strides = [i64; 4];

/// One Stockham pass: radix `r` over dimension length `n` whose elements are `sd` apart in
/// the dense work layout, with `p` the product of the radices already applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    pub n: u32,
    pub sd: u32,
    pub r: u32,
    pub p: u32,
    /// `n / (p * r)`: converts a twiddle exponent modulo `p * r` into one modulo `n`.
    pub twstep: u32,
    /// Offset (in complex elements) of the `n`-point twiddle table in the twiddle buffer.
    pub tw_offset: u32,
}

/// Elements a `local` work-group holds per shared-memory buffer (two buffers): the longest
/// transform that runs in one pass.
pub fn local_cap(double: bool) -> u32 {
    if double { 1024 } else { 2048 }
}

/// Largest radix of a `local` stage; a larger prime factor takes Bluestein's algorithm.
pub const MAX_LOCAL_RADIX: u32 = 63;

/// Direction of a pass: the execution's, or fixed (Bluestein's inner transforms).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Exec,
    Forward,
    Inverse,
}

impl Dir {
    pub fn inverse(self, exec_inverse: bool) -> bool {
        match self {
            Dir::Exec => exec_inverse,
            Dir::Forward => false,
            Dir::Inverse => true,
        }
    }
}

/// A table in the twiddle buffer (complex elements from `offset`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// `W_n^t = exp(-2 pi i t / n)`, t < n.
    Roots { n: u32 },
    /// Bluestein input chirp `exp(-+ i pi j^2 / n)`, j < n (+ for the inverse).
    ChirpIn { n: u32, inverse: bool },
    /// The FFT of the conjugate chirp, wrapped to length m.
    ChirpB { n: u32, m: u32, inverse: bool },
    /// The output chirp divided by m, k < n.
    ChirpOut { n: u32, m: u32, inverse: bool },
}

impl Table {
    pub fn len(self) -> u32 {
        match self {
            Table::Roots { n } | Table::ChirpIn { n, .. } | Table::ChirpOut { n, .. } => n,
            Table::ChirpB { m, .. } => m,
        }
    }
}

/// A `local` pass: `nl` lines of length `l`, line `t` = (a, b) = (t / ni, t % ni) read from
/// `a sa + b sb + j sj` and written (output o) to `a da + b db + o dj` (elements of the
/// dense buffers). `radices` packs the stage radices 6 bits each (first stage lowest);
/// `tw` is the offset of the `W_l` table, `tw2` (four-step) of a `W_n` table applied to
/// output o of line b as `W_n^(b o)`. A work-group holds `g` lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Local {
    pub nl: u32,
    pub l: u32,
    pub ni: u32,
    pub sa: u32,
    pub sb: u32,
    pub sj: u32,
    pub da: u32,
    pub db: u32,
    pub dj: u32,
    pub radices: u64,
    pub g: u32,
    pub tw: u32,
    pub tw2: Option<u32>,
}

/// A `chirp` pass over a `[outer][nd][sd]` destination from a `[outer][ns][sd]` source:
/// `dst(o, j, i) = (j < ns ? src(o, j, i) : 0) * table[j]`, the table at `tab[0]` (forward
/// execution) or `tab[1]` (inverse).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chirp {
    pub nd: u32,
    pub ns: u32,
    pub sd: u32,
    pub tab: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Stage(Stage),
    Local(Local),
    Chirp(Chirp),
}

/// One launch between work buffers 0 and 1 (dense, `total` elements each) and 2 and 3
/// (Bluestein, `ybuf` elements each). `total` is the launch's element count (output
/// elements; lines times length for `local`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub op: Op,
    pub src: u8,
    pub dst: u8,
    pub dir: Dir,
    pub total: u64,
}

/// Stage radices of a `local` transform: 8s, then 4, 2, then the odd factors.
pub fn local_factors(mut n: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for r in [8, 4, 2] {
        while n % r == 0 {
            out.push(r);
            n /= r;
        }
    }
    out.extend(factorize(n).into_iter().filter(|&f| f > 1));
    out
}

fn local_ok(n: u32, double: bool) -> bool {
    n <= local_cap(double) && local_factors(n).iter().all(|&r| r <= MAX_LOCAL_RADIX) && local_factors(n).len() <= 10
}

fn pack(radices: &[u32]) -> u64 {
    radices.iter().enumerate().fold(0u64, |acc, (i, &r)| acc | (u64::from(r) << (6 * i)))
}

/// Builds the pass list and twiddle tables of a plan.
struct Builder {
    double: bool,
    tables: Vec<(Table, u32)>,
    next: u32,
    passes: Vec<Pass>,
    ybuf: u64,
}

impl Builder {
    fn table(&mut self, t: Table) -> u32 {
        if let Some(&(_, off)) = self.tables.iter().find(|(x, _)| *x == t) {
            return off;
        }
        let off = self.next;
        self.tables.push((t, off));
        self.next += t.len();
        off
    }

    fn local(&mut self, l: u32, src: u8, dst: u8, dir: Dir, mut p: Local) {
        let cap = local_cap(self.double);
        p.l = l;
        p.radices = pack(&local_factors(l));
        // lines per work-group: a power of two (the kernel splits indices with shifts)
        p.g = 1 << (cap / l).max(1).ilog2();
        p.tw = self.table(Table::Roots { n: l });
        self.passes.push(Pass { op: Op::Local(p), src, dst, dir, total: u64::from(p.nl) * u64::from(l) });
    }

    /// Transforms the dimension of length `len` (elements `sd` apart, `outer` blocks) of
    /// buffer `cur` (scratch partner `other`); returns the buffer holding the result.
    fn dim(&mut self, len: u32, sd: u32, outer: u32, cur: u8, other: u8, dir: Dir, bluestein: bool) -> u8 {
        let cap = local_cap(self.double);
        let total = u64::from(outer) * u64::from(len) * u64::from(sd);
        let largest = factorize(len).into_iter().max().unwrap_or(1);
        if bluestein && largest > MAX_LOCAL_RADIX {
            return self.bluestein(len, sd, outer, cur, other, dir);
        }
        let plain = Local { nl: outer * sd, l: 0, ni: sd, sa: len * sd, sb: 1, sj: sd, da: len * sd, db: 1, dj: sd, radices: 0, g: 0, tw: 0, tw2: None };
        if local_ok(len, self.double) {
            self.local(len, cur, other, dir, plain);
            return other;
        }
        if sd == 1 {
            // four-step: the most balanced split n1 n2 with both factors local-sized
            let split = (1..=cap).filter(|&n1| len % n1 == 0 && local_ok(n1, self.double) && local_ok(len / n1, self.double)).min_by_key(|&n1| n1.abs_diff(len / n1));
            if let Some(n1) = split {
                let n2 = len / n1;
                let tw2 = Some(self.table(Table::Roots { n: len }));
                self.local(n1, cur, other, dir, Local { nl: outer * n2, ni: n2, sa: len, sb: 1, sj: n2, da: len, db: 1, dj: n2, tw2, ..plain });
                self.local(n2, other, cur, dir, Local { nl: outer * n1, ni: n1, sa: len, sb: n2, sj: 1, da: len, db: 1, dj: n1, ..plain });
                return cur;
            }
        }
        let tw_offset = self.table(Table::Roots { n: len });
        let (mut a, mut b) = (cur, other);
        let mut p = 1;
        for r in factorize(len) {
            let s = Stage { n: len, sd, r, p, twstep: len / (p * r), tw_offset };
            self.passes.push(Pass { op: Op::Stage(s), src: a, dst: b, dir, total });
            std::mem::swap(&mut a, &mut b);
            p *= r;
        }
        a
    }

    fn bluestein(&mut self, len: u32, sd: u32, outer: u32, cur: u8, other: u8, dir: Dir) -> u8 {
        let m = (2 * len - 1).next_power_of_two();
        let ytot = u64::from(outer) * u64::from(m) * u64::from(sd);
        self.ybuf = self.ybuf.max(ytot);
        let pair = |b: &mut Builder, f: &dyn Fn(bool) -> Table| [b.table(f(false)), b.table(f(true))];
        let tin = pair(self, &|inverse| Table::ChirpIn { n: len, inverse });
        let tb = pair(self, &|inverse| Table::ChirpB { n: len, m, inverse });
        let tout = pair(self, &|inverse| Table::ChirpOut { n: len, m, inverse });
        assert_eq!(dir, Dir::Exec, "Bluestein inside Bluestein");
        self.passes.push(Pass { op: Op::Chirp(Chirp { nd: m, ns: len, sd, tab: tin }), src: cur, dst: 2, dir, total: ytot });
        let r = self.dim(m, sd, outer, 2, 3, Dir::Forward, false);
        self.passes.push(Pass { op: Op::Chirp(Chirp { nd: m, ns: m, sd, tab: tb }), src: r, dst: r, dir, total: ytot });
        let r2 = self.dim(m, sd, outer, r, 5 - r, Dir::Inverse, false);
        let total = u64::from(outer) * u64::from(len) * u64::from(sd);
        self.passes.push(Pass { op: Op::Chirp(Chirp { nd: len, ns: m, sd, tab: tout }), src: r2, dst: other, dir, total });
        other
    }
}

/// In-place iterative radix-2 FFT (forward, unnormalized) of a power-of-two length.
fn fft_pow2(a: &mut [C]) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (u, v) = (a[start + k], a[start + k + len / 2]);
                let t = (v.0 * w.0 - v.1 * w.1, v.0 * w.1 + v.1 * w.0);
                a[start + k] = (u.0 + t.0, u.1 + t.1);
                a[start + k + len / 2] = (u.0 - t.0, u.1 - t.1);
            }
        }
        len <<= 1;
    }
}

/// `exp(-+ i pi j^2 / n)` with `j^2` reduced modulo `2n` first.
fn chirp(j: u32, n: u32, inverse: bool) -> C {
    let r = (u64::from(j) * u64::from(j)) % (2 * u64::from(n));
    let a = std::f64::consts::PI * r as f64 / f64::from(n) * if inverse { 1.0 } else { -1.0 };
    (a.cos(), a.sin())
}

impl Table {
    /// The table's values (f64).
    pub fn values(self) -> Vec<C> {
        match self {
            Table::Roots { n } => (0..n)
                .map(|t| {
                    let a = -2.0 * std::f64::consts::PI * f64::from(t) / f64::from(n);
                    (a.cos(), a.sin())
                })
                .collect(),
            Table::ChirpIn { n, inverse } => (0..n).map(|j| chirp(j, n, inverse)).collect(),
            Table::ChirpOut { n, m, inverse } => (0..n)
                .map(|k| {
                    let c = chirp(k, n, inverse);
                    (c.0 / f64::from(m), c.1 / f64::from(m))
                })
                .collect(),
            Table::ChirpB { n, m, inverse } => {
                let mut b = vec![(0.0, 0.0); m as usize];
                for j in 0..n {
                    let c = chirp(j, n, inverse);
                    let v = (c.0, -c.1);
                    b[j as usize] = v;
                    if j > 0 {
                        b[(m - j) as usize] = v;
                    }
                }
                fft_pow2(&mut b);
                b
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Desc {
    pub ty: FftType,
    pub rank: usize,
    /// Logical transform dimensions, padded on the left with 1s to rank 3.
    pub n: [u32; 3],
    pub batch: u32,
    pub input: Strides,
    pub output: Strides,
    /// Basic layout (no `inembed`/`onembed`): for an in-place real transform cuFFT pads the
    /// real side's last dimension to `2 * (n / 2 + 1)`; these strides replace the real side's.
    pub in_place_real: Option<Strides>,
    pub passes: Vec<Pass>,
    /// The tables of the twiddle buffer with their offsets (complex elements).
    pub tables: Vec<(Table, u32)>,
    /// Elements of each Bluestein buffer (2 and 3), 0 without Bluestein.
    pub ybuf: u64,
    /// The work buffer holding the transform after the last pass (0 or 1).
    pub result: u8,
    /// C2C with dense input and output: the first pass reads the input (buffer 4) and the
    /// last writes the output (buffer 5), with no gather or scatter launch.
    pub direct: bool,
}

/// Radices for an `n`-point Stockham pass, largest-first for powers of two. A prime factor
/// above 7 becomes one direct-DFT stage of that size (correct, O(n * p)).
pub fn factorize(mut n: u32) -> Vec<u32> {
    let mut out = Vec::new();
    while n % 4 == 0 {
        out.push(4);
        n /= 4;
    }
    while n % 2 == 0 {
        out.push(2);
        n /= 2;
    }
    let mut f = 3;
    while n > 1 {
        if f * f > n {
            out.push(n);
            break;
        }
        while n % f == 0 {
            out.push(f);
            n /= f;
        }
        f += 2;
    }
    out
}

fn side_strides(rank: usize, embed: &[i64], stride: i64, dist: i64) -> Strides {
    let mut s = [0i64; 4];
    s[0] = dist;
    let mut acc = stride;
    for d in (0..rank).rev() {
        s[1 + 3 - rank + d] = acc;
        acc = acc.saturating_mul(embed[d]);
    }
    s
}

/// The logical dims of each side: complex sides of R2C/C2R hold `n_last / 2 + 1`.
fn side_dims(n: &[i64], kind: Kind, input: bool) -> Vec<i64> {
    let mut d = n.to_vec();
    let half = match kind {
        Kind::C2C => false,
        Kind::R2C => !input,
        Kind::C2R => input,
    };
    if half {
        let last = d.len() - 1;
        d[last] = d[last] / 2 + 1;
    }
    d
}

/// Builds a plan description from `cufftPlanMany`-style arguments (64-bit). `inembed` or
/// `onembed` = `None` selects the basic layout (stride and dist are then ignored).
#[allow(clippy::too_many_arguments)]
pub fn describe(
    rank: i64,
    n: &[i64],
    inembed: Option<&[i64]>,
    istride: i64,
    idist: i64,
    onembed: Option<&[i64]>,
    ostride: i64,
    odist: i64,
    ty: FftType,
    batch: i64,
) -> Result<Desc, i32> {
    if !(1..=3).contains(&rank) || n.len() != rank as usize {
        return Err(INVALID_SIZE);
    }
    let rank = rank as usize;
    if batch < 1 || n.iter().any(|&x| x < 1 || x > u32::MAX as i64) {
        return Err(INVALID_SIZE);
    }
    let total = n.iter().try_fold(batch as u64, |acc, &x| acc.checked_mul(x as u64));
    if total.is_none_or(|t| t > MAX_ELEMS) {
        return Err(INVALID_SIZE);
    }
    let basic = |input: bool| -> (Vec<i64>, i64, i64) {
        let d = side_dims(n, ty.kind, input);
        let dist = d.iter().product();
        (d, 1, dist)
    };
    let side = |embed: Option<&[i64]>, stride: i64, dist: i64, input: bool| -> Result<Strides, i32> {
        let (e, s, di) = match embed {
            Some(e) => {
                if e.len() < rank || stride < 1 || dist < 0 || e[1..rank].iter().any(|&x| x < 1) {
                    return Err(INVALID_VALUE);
                }
                (e[..rank].to_vec(), stride, dist)
            }
            None => basic(input),
        };
        Ok(side_strides(rank, &e, s, di))
    };
    // cuFFT: if either embed array is NULL, both sides use the basic layout.
    let advanced = inembed.is_some() && onembed.is_some();
    let (inembed, onembed) = if advanced { (inembed, onembed) } else { (None, None) };
    let input = side(inembed, istride, idist, true)?;
    let output = side(onembed, ostride, odist, false)?;
    let in_place_real = match ty.kind {
        Kind::C2C => None,
        _ if advanced => None,
        _ => {
            let mut e = n.to_vec();
            e[rank - 1] = 2 * (n[rank - 1] / 2 + 1);
            let dist = e.iter().product();
            Some(side_strides(rank, &e, 1, dist))
        }
    };
    let mut dims = [1u32; 3];
    for (d, &x) in n.iter().enumerate() {
        dims[3 - rank + d] = x as u32;
    }
    let mut b = Builder { double: ty.double, tables: Vec::new(), next: 0, passes: Vec::new(), ybuf: 0 };
    let mut cur = 0u8;
    for d in 0..3 {
        let len = dims[d];
        if len == 1 {
            continue;
        }
        let sd: u32 = dims[d + 1..].iter().product();
        let outer: u32 = batch as u32 * dims[..d].iter().product::<u32>();
        cur = b.dim(len, sd, outer, cur, 1 - cur, Dir::Exec, true);
    }
    let total = batch as u64 * dims.iter().map(|&x| x as u64).product::<u64>();
    if b.ybuf > MAX_ELEMS || total > MAX_ELEMS || b.next as u64 > MAX_ELEMS {
        return Err(INVALID_SIZE);
    }
    let mut passes = b.passes;
    let dense = |s: &Strides| {
        let mut want = 1i64;
        for d in (0..3).rev() {
            if dims[d] > 1 && s[1 + d] != want {
                return false;
            }
            want *= dims[d] as i64;
        }
        batch == 1 || s[0] == want
    };
    let direct = ty.kind == Kind::C2C && !passes.is_empty() && dense(&input) && dense(&output);
    let mut result = cur;
    if direct {
        passes[0].src = 4;
        passes.last_mut().unwrap().dst = 5;
        result = 5;
    }
    Ok(Desc { ty, rank, n: dims, batch: batch as u32, input, output, in_place_real, passes, tables: b.tables, ybuf: b.ybuf, result, direct })
}

impl Desc {
    /// Dense complex elements in one work buffer.
    pub fn total(&self) -> u64 {
        self.batch as u64 * self.n.iter().map(|&x| x as u64).product::<u64>()
    }

    /// Bytes of the work area: two dense complex buffers, plus two Bluestein buffers.
    pub fn work_size(&self) -> u64 {
        (2 * self.total() + 2 * self.ybuf) * self.ty.complex_size()
    }

    /// Element offset of work buffer `i` (0..4) in the work area.
    pub fn buf_offset(&self, i: u8) -> u64 {
        match i {
            0 => 0,
            1 => self.total(),
            2 => 2 * self.total(),
            _ => 2 * self.total() + self.ybuf,
        }
    }

    /// Device address of work buffer `i` (4 and 5: the user's input and output).
    pub fn buf(&self, work: u64, i: u8, input: u64, output: u64) -> u64 {
        match i {
            4 => input,
            5 => output,
            _ => work + self.buf_offset(i) * self.ty.complex_size(),
        }
    }

    /// Complex elements of the twiddle buffer.
    pub fn tw_len(&self) -> u32 {
        self.tables.iter().map(|t| t.0.len()).sum()
    }

    /// The twiddle buffer (every table, computed in f64) as interleaved (re, im) pairs.
    pub fn twiddles(&self) -> Vec<f64> {
        let mut out = vec![0.0; 2 * self.tw_len() as usize];
        for &(t, off) in &self.tables {
            for (i, v) in t.values().into_iter().enumerate() {
                out[2 * (off as usize + i)] = v.0;
                out[2 * (off as usize + i) + 1] = v.1;
            }
        }
        out
    }

    pub fn twiddle_bytes(&self) -> Vec<u8> {
        let tw = self.twiddles();
        if self.ty.double {
            tw.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            tw.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect()
        }
    }

    /// Input and output strides for one execution. `in_place` selects the padded real layout
    /// of a basic-layout real transform.
    pub fn exec_strides(&self, in_place: bool) -> (Strides, Strides) {
        match (self.in_place_real, in_place, self.ty.kind) {
            (Some(p), true, Kind::R2C) => (p, self.output),
            (Some(p), true, Kind::C2R) => (self.input, p),
            _ => (self.input, self.output),
        }
    }

    /// Kernel launch parameters, in the order `ptxgen` declares them (all widened to u64).
    pub fn gather_args(&self, src: u64, dst: u64, strides: &Strides) -> Vec<u64> {
        let mode = match self.ty.kind {
            Kind::C2C => 0,
            Kind::R2C => 1,
            Kind::C2R => 2,
        };
        let [n0, n1, n2] = self.n;
        let mut a = vec![src, dst, self.total(), n0 as u64, n1 as u64, n2 as u64];
        a.extend(strides.iter().map(|&s| s as u64));
        a.push(mode);
        a
    }

    pub fn stage_args(&self, s: &Stage, total: u64, src: u64, dst: u64, tw: u64, inverse: bool) -> Vec<u64> {
        let tw = tw + s.tw_offset as u64 * self.ty.complex_size();
        vec![src, dst, tw, total, s.n as u64, s.sd as u64, s.r as u64, s.p as u64, s.twstep as u64, inverse as u64]
    }

    /// `local` launch parameters (see `ptxgen::local`).
    pub fn local_args(&self, l: &Local, src: u64, dst: u64, tw: u64, inverse: bool) -> Vec<u64> {
        let cs = self.ty.complex_size();
        let tw2 = l.tw2.map_or(0, |o| tw + u64::from(o) * cs);
        let mut a = vec![src, dst, tw + u64::from(l.tw) * cs, tw2];
        a.extend([l.nl, l.l, l.ni, l.sa, l.sb, l.sj, l.da, l.db, l.dj].map(u64::from));
        a.extend([l.radices, inverse as u64, u64::from(l.g), u64::from(l.g.trailing_zeros())]);
        a
    }

    /// Work-groups of a `local` launch.
    pub fn local_groups(l: &Local) -> u32 {
        l.nl.div_ceil(l.g)
    }

    /// `chirp` launch parameters (see `ptxgen::chirp`).
    pub fn chirp_args(&self, c: &Chirp, total: u64, src: u64, dst: u64, tw: u64, inverse: bool) -> Vec<u64> {
        let tab = tw + u64::from(c.tab[inverse as usize]) * self.ty.complex_size();
        vec![src, dst, tab, total, u64::from(c.nd), u64::from(c.ns), u64::from(c.sd)]
    }

    /// Output elements written by `scatter`.
    pub fn out_total(&self) -> u64 {
        let m2 = if self.ty.kind == Kind::R2C { self.n[2] / 2 + 1 } else { self.n[2] };
        self.batch as u64 * self.n[0] as u64 * self.n[1] as u64 * m2 as u64
    }

    pub fn scatter_args(&self, src: u64, dst: u64, strides: &Strides) -> Vec<u64> {
        let [n0, n1, n2] = self.n;
        let m2 = if self.ty.kind == Kind::R2C { n2 / 2 + 1 } else { n2 };
        let mut a = vec![src, dst, self.out_total(), n0 as u64, n1 as u64, m2 as u64, n2 as u64];
        a.extend(strides.iter().map(|&s| s as u64));
        a.push((self.ty.kind == Kind::C2R) as u64);
        a
    }
}

// ---------------------------------------------------------------------------
// Host model of the three kernels (same index math as ptxgen), plus a naive DFT.
// ---------------------------------------------------------------------------

pub type C = (f64, f64);

/// Host execution of the kernel pipeline on f64 scalars. `input`/`output` are flat scalar
/// arrays indexed as the device buffers are (complex element `e` at scalars `2e, 2e+1`).
pub fn execute_host(d: &Desc, input: &[f64], output: &mut [f64], inverse: bool, in_place: bool) {
    let (is, os) = d.exec_strides(in_place);
    let total = d.total() as usize;
    let [n0, n1, n2] = d.n.map(|x| x as usize);
    let off = |s: &Strides, b: usize, x0: usize, x1: usize, x2: usize| -> usize {
        (b as i64 * s[0] + x0 as i64 * s[1] + x1 as i64 * s[2] + x2 as i64 * s[3]) as usize
    };
    // gather
    let a: Vec<C> = (0..total)
        .map(|g| {
            let (mut x2, t) = (g % n2, g / n2);
            let (mut x1, t) = (t % n1, t / n1);
            let (mut x0, b) = (t % n0, t / n0);
            match d.ty.kind {
                Kind::C2C => {
                    let o = off(&is, b, x0, x1, x2);
                    (input[2 * o], input[2 * o + 1])
                }
                Kind::R2C => (input[off(&is, b, x0, x1, x2)], 0.0),
                Kind::C2R => {
                    let mirror = x2 > n2 / 2;
                    if mirror {
                        x2 = n2 - x2;
                        x1 = (n1 - x1) % n1;
                        x0 = (n0 - x0) % n0;
                    }
                    let o = off(&is, b, x0, x1, x2);
                    (input[2 * o], if mirror { -input[2 * o + 1] } else { input[2 * o + 1] })
                }
            }
        })
        .collect();
    // passes over the four work buffers
    let tw: Vec<C> = d.twiddles().chunks(2).map(|c| (c[0], c[1])).collect();
    // a direct plan's buffers 4 and 5 (the user's) hold the dense input and output
    let mut bufs = vec![a.clone(), vec![(0.0, 0.0); total], vec![(0.0, 0.0); d.ybuf as usize], vec![(0.0, 0.0); d.ybuf as usize], a, vec![(0.0, 0.0); total]];
    for p in &d.passes {
        let inv = p.dir.inverse(inverse);
        let src = bufs[p.src as usize].clone();
        let dst = &mut bufs[p.dst as usize];
        match p.op {
            Op::Stage(s) => {
                let tw_flat: Vec<f64> = tw.iter().flat_map(|v| [v.0, v.1]).collect();
                stage_host(&s, &src[..p.total as usize], &mut dst[..p.total as usize], &tw_flat, inv);
            }
            Op::Local(l) => local_host(&l, &src, dst, &tw, inv),
            Op::Chirp(c) => {
                let tab = &tw[c.tab[inverse as usize] as usize..];
                for g in 0..p.total as usize {
                    let (inner, t) = (g % c.sd as usize, g / c.sd as usize);
                    let (j, o) = (t % c.nd as usize, t / c.nd as usize);
                    dst[g] = if j < c.ns as usize { cmul(src[(o * c.ns as usize + j) * c.sd as usize + inner], tab[j]) } else { (0.0, 0.0) };
                }
            }
        }
    }
    let a = std::mem::take(&mut bufs[d.result as usize]);
    // scatter
    let m2 = if d.ty.kind == Kind::R2C { n2 / 2 + 1 } else { n2 };
    for g in 0..d.out_total() as usize {
        let (x2, t) = (g % m2, g / m2);
        let (x1, t) = (t % n1, t / n1);
        let (x0, bb) = (t % n0, t / n0);
        let v = a[((bb * n0 + x0) * n1 + x1) * n2 + x2];
        let o = off(&os, bb, x0, x1, x2);
        if d.ty.kind == Kind::C2R {
            output[o] = v.0;
        } else {
            output[2 * o] = v.0;
            output[2 * o + 1] = v.1;
        }
    }
}

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// A `local` pass, as the kernel computes it: every Stockham stage of the `l`-point
/// transform per line, then the optional four-step twiddle, then the strided store.
pub fn local_host(l: &Local, src: &[C], dst: &mut [C], tw: &[C], inverse: bool) {
    let n = l.l as usize;
    let roots = &tw[l.tw as usize..];
    let w = |t: usize| if inverse { (roots[t].0, -roots[t].1) } else { roots[t] };
    for line in 0..l.nl as usize {
        let (a, b) = (line / l.ni as usize, line % l.ni as usize);
        let mut x: Vec<C> = (0..n).map(|j| src[a * l.sa as usize + b * l.sb as usize + j * l.sj as usize]).collect();
        let mut y = vec![(0.0, 0.0); n];
        let (mut rad, mut p) = (l.radices, 1usize);
        while rad & 63 != 0 {
            let r = (rad & 63) as usize;
            rad >>= 6;
            let (pr, m) = (p * r, n / r);
            let twstep = n / pr;
            for (o, out) in y.iter_mut().enumerate() {
                let rem = o % pr;
                let i = (o / pr) * p + rem % p;
                let mut acc = (0.0, 0.0);
                for q in 0..r {
                    let v = cmul(x[i + q * m], w(((q * rem) % pr) * twstep));
                    acc = (acc.0 + v.0, acc.1 + v.1);
                }
                *out = acc;
            }
            std::mem::swap(&mut x, &mut y);
            p = pr;
        }
        for (o, &v) in x.iter().enumerate() {
            let v = match l.tw2 {
                Some(off) => {
                    let t = tw[off as usize + b * o];
                    cmul(v, if inverse { (t.0, -t.1) } else { t })
                }
                None => v,
            };
            dst[a * l.da as usize + b * l.db as usize + o * l.dj as usize] = v;
        }
    }
}

/// One Stockham stage, exactly as the `stage` kernel computes it (one output per index).
pub fn stage_host(s: &Stage, src: &[C], dst: &mut [C], tw: &[f64], inverse: bool) {
    let (n, sd, r, p) = (s.n as usize, s.sd as usize, s.r as usize, s.p as usize);
    let pr = p * r;
    let m = n / r;
    for (g, out) in dst.iter_mut().enumerate() {
        let (inner, t) = (g % sd, g / sd);
        let (o, outer) = (t % n, t / n);
        let base = outer * n * sd + inner;
        let rem = o % pr;
        let i = (o / pr) * p + rem % p;
        let mut acc = (0.0, 0.0);
        let mut e = 0;
        for q in 0..r {
            let x = src[base + (i + q * m) * sd];
            let ti = 2 * (s.tw_offset as usize + e * s.twstep as usize);
            let w = (tw[ti], if inverse { -tw[ti + 1] } else { tw[ti + 1] });
            acc.0 += x.0 * w.0 - x.1 * w.1;
            acc.1 += x.0 * w.1 + x.1 * w.0;
            e += rem;
            if e >= pr {
                e -= pr;
            }
        }
        *out = acc;
    }
}

/// Naive multi-dimensional DFT over a dense `[batch][n0][n1][n2]` complex array.
pub fn naive_dft(n: [usize; 3], batch: usize, x: &[C], inverse: bool) -> Vec<C> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let per = n[0] * n[1] * n[2];
    let mut out = vec![(0.0, 0.0); x.len()];
    for b in 0..batch {
        for k in 0..per {
            let k3 = [k / (n[1] * n[2]), (k / n[2]) % n[1], k % n[2]];
            let mut acc = (0.0, 0.0);
            for j in 0..per {
                let j3 = [j / (n[1] * n[2]), (j / n[2]) % n[1], j % n[2]];
                let mut ph = 0.0;
                for d in 0..3 {
                    ph += ((j3[d] * k3[d]) % n[d]) as f64 / n[d] as f64;
                }
                let a = sign * 2.0 * std::f64::consts::PI * ph;
                let v = x[b * per + j];
                acc.0 += v.0 * a.cos() - v.1 * a.sin();
                acc.1 += v.0 * a.sin() + v.1 * a.cos();
            }
            out[b * per + k] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(code: i32) -> FftType {
        FftType::from_cufft(code).unwrap()
    }

    #[test]
    fn factorization_covers_n() {
        for n in 1..2000u32 {
            let f = factorize(n);
            assert_eq!(f.iter().product::<u32>(), n.max(1), "{n}");
        }
        assert_eq!(factorize(1024), vec![4, 4, 4, 4, 4]);
        assert_eq!(factorize(105), vec![3, 5, 7]);
        assert_eq!(factorize(4099), vec![4099]);
        assert_eq!(factorize(400), vec![4, 4, 5, 5]);
    }

    #[test]
    fn pass_selection() {
        let z = |n: &[i64]| describe(n.len() as i64, n, None, 0, 0, None, 0, 0, t(0x69), 1).unwrap();
        let kinds = |d: &Desc| d.passes.iter().map(|p| match p.op { Op::Stage(_) => 'S', Op::Local(_) => 'L', Op::Chirp(_) => 'C' }).collect::<String>();
        assert_eq!(kinds(&z(&[1000])), "L");
        // 4096 > 1024 (f64 cap): four-step 64 x 64
        assert_eq!(kinds(&z(&[4096])), "LL");
        // prime 4099: chirp, 16384-point FFT (four-step), product, inverse, chirp
        assert_eq!(kinds(&z(&[4099])), "CLLCLLC");
        let d = z(&[4099]);
        assert_eq!(d.ybuf, 16384);
        assert_eq!(d.work_size(), (2 * 4099 + 2 * 16384) * 16);
        // f32 holds 2048 per pass
        let c = describe(1, &[2048], None, 0, 0, None, 0, 0, t(0x29), 1).unwrap();
        assert_eq!(kinds(&c), "L");
        // a long non-last dimension falls back to global stages
        assert_eq!(kinds(&z(&[2048, 2])), "SSSSSSL");
        assert_eq!(local_factors(2048), vec![8, 8, 8, 4]);
        assert_eq!(local_factors(1000), vec![8, 5, 5, 5]);
    }

    #[test]
    fn bluestein_tables_reproduce_the_dft() {
        // the chirp-z identity on the host with the tables' values
        let (n, m) = (11u32, 32u32);
        for inverse in [false, true] {
            let x: Vec<C> = (0..n).map(|j| (f64::from(j) * 0.3 - 1.0, 0.5 - f64::from(j % 3))).collect();
            let cin = Table::ChirpIn { n, inverse }.values();
            let bf = Table::ChirpB { n, m, inverse }.values();
            let cout = Table::ChirpOut { n, m, inverse }.values();
            let mut a = vec![(0.0, 0.0); m as usize];
            for j in 0..n as usize {
                a[j] = cmul(x[j], cin[j]);
            }
            fft_pow2(&mut a);
            for (v, b) in a.iter_mut().zip(&bf) {
                *v = cmul(*v, *b);
            }
            // inverse FFT = conj(FFT(conj))
            for v in a.iter_mut() {
                v.1 = -v.1;
            }
            fft_pow2(&mut a);
            let got: Vec<C> = (0..n as usize).map(|k| cmul((a[k].0, -a[k].1), cout[k])).collect();
            let want = naive_dft([1, 1, n as usize], 1, &x, inverse);
            assert!(max_err(&got, &want) < 1e-12, "inverse {inverse}");
        }
    }

    #[test]
    fn types_decode() {
        assert_eq!(t(0x2a), FftType { kind: Kind::R2C, double: false });
        assert_eq!(t(0x6c), FftType { kind: Kind::C2R, double: true });
        assert!(FftType::from_cufft(0x2b).is_none());
        assert_eq!(FftType::from_xt(0, 4, 4).unwrap(), t(0x2a));
        assert_eq!(FftType::from_xt(5, 1, 5).unwrap(), t(0x6c));
        assert_eq!(FftType::from_xt(6, 6, 6), Err(NOT_SUPPORTED));
        assert_eq!(FftType::from_xt(4, 5, 4), Err(INVALID_TYPE));
    }

    #[test]
    fn invalid_sizes() {
        let c2c = t(0x29);
        assert_eq!(describe(0, &[], None, 1, 1, None, 1, 1, c2c, 1), Err(INVALID_SIZE));
        assert_eq!(describe(1, &[0], None, 1, 1, None, 1, 1, c2c, 1), Err(INVALID_SIZE));
        assert_eq!(describe(1, &[8], None, 1, 1, None, 1, 1, c2c, 0), Err(INVALID_SIZE));
        assert_eq!(describe(4, &[2, 2, 2, 2], None, 1, 1, None, 1, 1, c2c, 1), Err(INVALID_SIZE));
        assert_eq!(describe(1, &[1 << 20], None, 1, 1, None, 1, 1, c2c, 1 << 12), Err(INVALID_SIZE));
    }

    #[test]
    fn basic_layouts_and_work_size() {
        let d = describe(2, &[6, 10], None, 0, 0, None, 0, 0, t(0x2a), 3).unwrap();
        assert_eq!(d.n, [1, 6, 10]);
        assert_eq!(d.input, [60, 0, 10, 1]);
        assert_eq!(d.output, [36, 0, 6, 1]);
        assert_eq!(d.in_place_real, Some([72, 0, 12, 1]));
        assert_eq!(d.work_size(), 2 * 180 * 8);
        assert_eq!(d.out_total(), 3 * 6 * 6);
        let z = describe(3, &[2, 3, 4], None, 0, 0, None, 0, 0, t(0x69), 1).unwrap();
        assert_eq!(z.input, [24, 12, 4, 1]);
        assert_eq!(z.work_size(), 2 * 24 * 16);
        assert_eq!(z.tables, vec![(Table::Roots { n: 2 }, 0), (Table::Roots { n: 3 }, 2), (Table::Roots { n: 4 }, 5)]);
        // every dimension fits one shared-memory pass
        assert_eq!(z.passes.len(), 3);
        assert!(z.passes.iter().all(|p| matches!(p.op, Op::Local(_))));
        assert!(z.direct);
        assert_eq!((z.passes[0].src, z.result), (4, 5));
        assert!(!d.direct);
        // advanced layout: interleaved batches (stride = batch, dist = 1), as torch emits
        let a = describe(1, &[8], Some(&[8]), 4, 1, Some(&[5]), 1, 5, t(0x2a), 4).unwrap();
        assert_eq!(a.input, [1, 0, 0, 4]);
        assert_eq!(a.output, [5, 0, 0, 1]);
        assert_eq!(a.in_place_real, None);
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

    fn max_err(a: &[C], b: &[C]) -> f64 {
        let scale = b.iter().map(|v| v.0.hypot(v.1)).fold(1e-30, f64::max);
        a.iter().zip(b).map(|(x, y)| (x.0 - y.0).hypot(x.1 - y.1)).fold(0.0, f64::max) / scale
    }

    /// C2C through the host model of the kernels against the naive DFT, both directions.
    #[test]
    fn host_pipeline_c2c_matches_naive() {
        // local, Bluestein (127, 2 * 131, 4099 and a non-last 67), four-step (4096 > the f64
        // local cap of 1024), and mixed
        let shapes: &[(&[i64], i64)] = &[
            (&[1], 2),
            (&[2], 1),
            (&[8], 3),
            (&[12], 2),
            (&[105], 1),
            (&[97], 2),
            (&[64], 1),
            (&[6, 10], 2),
            (&[4, 3, 5], 1),
            (&[16, 9], 1),
            (&[127], 2),
            (&[262], 1),
            (&[67, 6], 1),
            (&[5, 71], 1),
            (&[4096], 1),
            (&[3000], 1),
            (&[4099], 1),
        ];
        for &(n, batch) in shapes {
            let d = describe(n.len() as i64, n, None, 0, 0, None, 0, 0, t(0x69), batch).unwrap();
            let total = d.total() as usize;
            let x = sample(2 * total, n.iter().sum::<i64>() as u64);
            let xc: Vec<C> = x.chunks(2).map(|c| (c[0], c[1])).collect();
            let n3 = d.n.map(|v| v as usize);
            for inverse in [false, true] {
                let mut y = vec![0.0; 2 * total];
                execute_host(&d, &x, &mut y, inverse, false);
                let yc: Vec<C> = y.chunks(2).map(|c| (c[0], c[1])).collect();
                let e = max_err(&yc, &naive_dft(n3, batch as usize, &xc, inverse));
                assert!(e < 1e-12, "{n:?} batch {batch} inverse {inverse}: err {e}");
            }
        }
    }

    /// R2C then C2R, out-of-place and in-place (padded basic layout), against the naive DFT.
    #[test]
    fn host_pipeline_real_transforms() {
        for (n, batch) in [(&[8i64][..], 2i64), (&[15], 3), (&[6, 7], 2), (&[3, 4, 6], 1), (&[1], 1)] {
            let d = describe(n.len() as i64, n, None, 0, 0, None, 0, 0, t(0x6a), batch).unwrap();
            let total = d.total() as usize;
            let n3 = d.n.map(|v| v as usize);
            let x = sample(total, 7 + n[0] as u64);
            let xc: Vec<C> = x.iter().map(|&v| (v, 0.0)).collect();
            let want = naive_dft(n3, batch as usize, &xc, false);
            let m2 = n3[2] / 2 + 1;
            let half = |full: &[C]| -> Vec<C> { full.iter().enumerate().filter(|(i, _)| i % n3[2] < m2).map(|(_, v)| *v).collect() };
            // out-of-place R2C
            let mut y = vec![0.0; 2 * d.out_total() as usize];
            execute_host(&d, &x, &mut y, false, false);
            let yc: Vec<C> = y.chunks(2).map(|c| (c[0], c[1])).collect();
            assert!(max_err(&yc, &half(&want)) < 1e-12, "R2C {n:?}");
            // C2R back: unnormalized, so expect x * N
            let inv = describe(n.len() as i64, n, None, 0, 0, None, 0, 0, t(0x6c), batch).unwrap();
            let mut z = vec![0.0; total];
            execute_host(&inv, &y, &mut z, true, false);
            let per = (n3[0] * n3[1] * n3[2]) as f64;
            for (a, b) in z.iter().zip(&x) {
                assert!((a / per - b).abs() < 1e-12, "C2R {n:?}");
            }
            // in-place: real rows padded to 2 * m2 scalars
            let rows = total / n3[2];
            let mut buf = vec![0.0; rows * 2 * m2];
            for r in 0..rows {
                buf[r * 2 * m2..r * 2 * m2 + n3[2]].copy_from_slice(&x[r * n3[2]..(r + 1) * n3[2]]);
            }
            let src = buf.clone();
            execute_host(&d, &src, &mut buf, false, true);
            let yc: Vec<C> = buf.chunks(2).map(|c| (c[0], c[1])).collect();
            assert!(max_err(&yc, &half(&want)) < 1e-12, "in-place R2C {n:?}");
            let src = buf.clone();
            execute_host(&inv, &src, &mut buf, true, true);
            for r in 0..rows {
                for c in 0..n3[2] {
                    assert!((buf[r * 2 * m2 + c] / per - x[r * n3[2] + c]).abs() < 1e-12, "in-place C2R {n:?}");
                }
            }
        }
    }

    /// Advanced layout: strided input with batches interleaved, output padded rows.
    #[test]
    fn host_pipeline_advanced_layout() {
        let (n, batch) = (12usize, 3usize);
        let d = describe(1, &[n as i64], Some(&[n as i64]), batch as i64, 1, Some(&[16]), 2, 40, t(0x69), batch as i64).unwrap();
        let x = sample(2 * n * batch, 3);
        let mut y = vec![0.0; 2 * (40 * batch)];
        execute_host(&d, &x, &mut y, false, false);
        for b in 0..batch {
            let xc: Vec<C> = (0..n).map(|i| (x[2 * (i * batch + b)], x[2 * (i * batch + b) + 1])).collect();
            let want = naive_dft([1, 1, n], 1, &xc, false);
            let got: Vec<C> = (0..n).map(|k| (y[2 * (b * 40 + 2 * k)], y[2 * (b * 40 + 2 * k) + 1])).collect();
            assert!(max_err(&got, &want) < 1e-12);
        }
    }
}

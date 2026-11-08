# PTX translator coverage

Status of the PTX to SPIR-V translator (`src/ptx/`) as measured on the translator-hardening
branch. Three sources of evidence:

1. **Corpus scan** (`scripts/ptx-corpus-scan.sh`, test `ptx::corpus_scan`): every `.entry` of a
   local PTX corpus is translated (with the same lenient undefined-register mode the runtime
   uses) and instruction, modifier and state-space histograms are collected. The corpus is
   third-party code (Ollama `ggml-cuda` fatbins, PyTorch cu128 frames, ZLUDA and CUTLASS-style
   instruction tests) and is kept **outside the repository**; only the scan code is committed.
2. **Differential fuzzing** (`src/ptx/fuzz_tests.rs`, `cargo test --lib fuzz_tests`): one PTX
   instruction per kernel, executed on the SPIR-V simulator for random and edge-case inputs and
   compared with a reference written from the PTX ISA (about 250 instruction forms,
   about 45 000 comparisons per run).
3. **Hardware tests** that the simulator cannot decide: `tests/gpu/hw_translator_edge.rs`
   (written, not yet run on hardware).

## Corpus numbers

| Item | Count |
| --- | --- |
| PTX files scanned | 1331 |
| Unique entries (duplicate modules removed) | 24 070 |
| Entries translated | 24 049 (99.91 %) |
| Entries rejected with an error | 21 |
| Modules that could not be parsed at all | 181 (see below) |
| Translated entries sampled through `spirv-val` (every 400th) | 60, no real validation error |

The 14 `spirv-val` messages in the sample were validator-environment noise, not translator
defects: 10 were "SPIR-V 1.3 under the OpenCL 2.2 environment", 3 were `Int64Atomics` (needs
`cl_khr_int64_base_atomics`), and 1 was an OpTypeFunction with more than 255 parameters (a
PyTorch kernel with a very large argument list). The scan now validates with `--target-env
spv1.3` and ignores the two capability messages.

Most frequent instructions (whole corpus, occurrences): ld 4.2 M, bra 4.1 M, add 3.6 M, setp
3.4 M, mov 2.9 M, mul 2.6 M, cvt 1.8 M, mad 1.3 M, shr 1.2 M, st 1.2 M, and 1.1 M, fma 1.1 M,
selp 0.6 M, mma 0.26 M, shfl 86 k, bar 71 k, vote 30 k, atom 11 k, dp4a 10 k.

Modules that fail to parse (181): 51 contain no supported `.entry` (only `.func` helpers or
unsupported entries), 16 use PTX ISA older than 7.0 (the translator accepts 7.0 to 9.x and 4.1),
12 use `.texref`, about 20 use features listed under "Unsupported", and the rest are
syntax forms from the ZLUDA instruction tests (see below).

## Supported

Verified by the fuzzer against a PTX-ISA reference unless noted.

| Area | Forms |
| --- | --- |
| 32/64-bit integer arithmetic | add, sub, mul.lo/hi/wide, mad.lo/hi/wide (hi.u64 excepted), mul24, mad24, min, max, abs, neg, not, cnot, and/or/xor, add/sub .cc/.c carry chains, .sat |
| Division | div/rem u32, s32, u64, s64 with **defined** results for zero divisor and INT_MIN / -1 (see "Defined edge cases") |
| Shifts | shl, shr.u/s 32 and 64 bit; counts at or above the width saturate (0, or sign fill) |
| Bit manipulation | popc, clz, brev, bfind (u32/s32/u64), bfe/bfi (32 and 64), prmt (default and f4e, b4e, rc8, ecl, ecr, rc16), lop3, shf.l/r clamp and wrap, szext, bmsk, sad, dp4a (all signedness), dp2a lo/hi |
| Predicates | setp with all integer and float comparisons including NaN-aware (ltu, equ, nan, num...), selp, combine with .and/.or |
| f32 | add, sub, mul, fma with .rn/.rz/.rm/.rp (+ .sat), div.rn, sqrt.rn, rcp.rn, min/max (PTX NaN and signed-zero rules, .NaN), abs, neg, copysign, cvt.rzi/rni/rmi/rpi |
| f64 | add, mul, fma, div, sqrt, min, max, abs, neg (.rn) |
| Conversions | f32/f64 to s32/u32/s64/u64 (always saturating, NaN to 0) with rn/rz/rm/rp integer rounding; s32/u32/s64/u64 to f32 with all four rounding modes; f64 to f32 with all four modes; f32 to f16 (rn, rz) and f16/bf16 to f32; bf16 rn; cvt.sat to s8/u8/s16/u16/s32/u32 |
| Atomics (global and shared) | add (u32, s32, f32, u64), min/max (u32, s32, u64, s64), and/or/xor, inc/dec, exch, cas.b32, red.add |
| Warp | shfl.sync idx/up/down/bfly with arbitrary clamp/segment operand and predicate output (membermask must be 0xffffffff), vote.sync ballot/all/any/uni, activemask, redux.sync add/min/max/and/or/xor, match.sync any/all, bar.warp.sync, membar/fence |
| Control flow | predicated and unconditional bra, nested divergent loops, break/continue, early `ret` of some threads, diamonds that reconverge, conditionally defined registers, uniform `bar.sync` in loops |
| Memory | ld/st/atom for global, shared, local, param, const; vector ld/st; module-level globals; dynamic shared memory |
| Matrix | mma.sync (f16, bf16, tf32, int8 shapes covered by `tier3_mma_tests`), wmma, ldmatrix, stmatrix |

## Approximated (translates, result can differ from NVIDIA)

| Feature | Behaviour | Corpus occurrences |
| --- | --- | --- |
| `.ftz` on f32 add/sub/mul/fma/div/sqrt/cvt/setp/min/max | **Ignored**: denormals are not flushed. Results differ from NVIDIA only when an operand or the result is a denormal. The fuzzer compares `.ftz` forms only for non-denormal cases. | add.ftz 162 k, mul.ftz 226 k, fma.ftz 322 k, ex2.ftz 27 k |
| `fma.rz/rm/rp.f32` | Exact except when 0 < \|a*b\| < 2^-100 (the error-free transform loses the product error); then the result may be 1 ulp off. Overflow now returns +-MAX like hardware. | fma.rz 2.7 k, fma.rm 10.6 k |
| `.approx` transcendental (ex2, lg2, sin, cos, tanh, rcp, rsqrt, div.approx, sqrt.approx) | Mapped to OpenCL builtins whose accuracy is whatever the Intel runtime provides; they are not bit-identical to the NVIDIA approximations. | ex2.approx 28 k, rcp.approx 16 k, div.approx 6.5 k, tanh 1.8 k, sin/cos 196 |
| `trap`, `brkpt`, `exit` without operands | Lowered to `ret`: the kernel does not abort. | trap 1049, exit 678 |
| `__assertfail`, `vprintf`, `nanosleep`, `prefetch`, `discard`, `applypriority`, `pmevent`, `griddepcontrol`, `setmaxnreg` | Dropped (device asserts and printf have no effect). | __assertfail 46 k, vprintf 1.2 k, nanosleep 793 |
| `fence`/`membar` of any scope | Always a device-scope fence (stronger than required). | membar 2.7 k |
| `vote.sync`, `activemask`, `match.sync`, `redux.sync` membermask | The mask operand is ignored: the operation covers the active lanes of the subgroup. Identical for full-warp masks (all uses in the corpus); differs only for deliberate partial masks. | vote 30 k |
| `createpolicy` | Result register set to 0. | rare |
| Partial-warp `bar.sync` (some threads exit first) | Accepted when the exit condition is warp-uniform; PTX treats exited threads as arrived. Whether the Intel runtime agrees is covered by `hw_barrier_with_exited_threads_completes` (not yet run). | common (`if (tid >= n) return;`) |
| Registers read before any write | PTX leaves them undefined. The translator stores a real `mov r, 0` at entry (the flash-attention lesson) and counts each one (`ptx::undefined_register_count()`, set `CFI_WARN_UNDEFINED=1` to print them). Predicates are zero-initialised as false. | 40 entries in the PyTorch Bernoulli kernels alone |
| Uninitialised shared memory | Not zeroed (like hardware); contents after launch are whatever the device had. Kernels reading it before writing are racy on NVIDIA too. | n/a |

## Defined edge cases (previously undefined behaviour in SPIR-V)

| Case | Result now |
| --- | --- |
| `div.u32/u64` by 0 | all ones |
| `rem.u32/u64` by 0 | the dividend |
| `div.s32/s64` by 0 | -1 |
| `rem.s32/s64` by 0 | the dividend |
| `div.s32 INT_MIN, -1` | INT_MIN |
| `rem.s32 INT_MIN, -1` | 0 |
| `shfl.sync` lane/offset operand above 31 | only bits [4:0] are used (was: out-of-range lane) |
| `cvt.sat.s8.*` result | sign-extended to 16 bits (was zero-extended) |

## Unsupported (clear translate error naming the instruction)

Every item below fails translation with `unsupported PTX instruction: <text>` (or a specific
message); none is silently miscompiled.

| Feature | Corpus occurrences / note |
| --- | --- |
| Texture and surface: `tex`, `tld4`, `txq`, `suld`, `sust`, `sured`, `.texref`, `.surfref`, `.samplerref` | 22 entries in the ZLUDA tests. Decision: **clean unsupported error**. The OpenCL-environment SPIR-V target has no bindless image objects. |
| `div.rz/rm/rp.f32`, `sqrt.rz/rm/rp.f32` | none in the corpus (only `.rn`, `.approx`, `.full` are used by compilers) |
| `mad.hi.u64` | none in the corpus |
| `atom.cas.b128`, `atom.global.cas` with vector operands | 1 |
| fp8 / fp4 / packed conversions (`cvt.rn.satfinite.e4m3x2`, `e2m1x2`, `cvt.rs`, `cvt.rn.f16x2.e4m3x2`) | 7 (sm_89/sm_100 tests) |
| Hopper/Blackwell: `wgmma`, `tcgen05`, `mbarrier.*`, `cp.async.bulk(.tensor)`, `st.async`, `red.async`, `mapa`, `getctarank`, cluster `ld`/`barrier.cluster`, `multimem` | wgmma/tcgen05/mbarrier: 11 entries |
| `mma.sync` int4/int1/m8n8k4/sparse forms, `ldmatrix` `.m16n16.b8` | 6 |
| Partial-warp `membermask` (other than 0xffffffff) on shfl.sync (generic modes and bfly) | clear error; ggml and PyTorch use full masks |
| Indirect calls, `call` to non-inlinable device functions, `.func` with `.param` aggregates | about 8 entries |
| `.v8` / 256-bit vector `ld`/`st`, `.reg .v4` | 6 (ZLUDA tests) |
| `.local` symbols addressed as `[sym+reg]` in a few forms; address arithmetic on two pointers | 4 |
| Kernels whose parameter aggregate is larger than the translator's limit | 2 |

The remaining unparsed modules (about 100) are ZLUDA instruction tests that use non-nvcc syntax
(bare register names are handled by the scan, but operand forms like `[in_addr]` without a
register prefix, `.reg.u64 temp` style declarations and unreachable-block tests are not).

## Fuzzer status

| Module | Forms | Result |
| --- | --- | --- |
| `int_ops` | 73 32-bit integer forms | all match |
| `int64_ops` | 38 64-bit forms | all match; `mad.hi.u64` reported unsupported |
| `float_ops` | 140 f32/f64/cvt/f16/bf16 forms | all match; 12 directed div/sqrt forms reported unsupported |
| `threads` | atomics (global/shared, 8 ops, cas, exch, inc, dec, f32, u64, s64), shfl (4 modes x 13 control values, register operands, predicate form), vote, activemask, redux, match, divergence | all match |

### Bugs found and fixed by this work

1. Integer `div`/`rem` by zero and `INT_MIN / -1` were SPIR-V undefined behaviour (the
   simulator trapped; hardware behaviour is device-dependent). Now guarded, with NVIDIA results.
2. `fma.rz/rm/rp.f32` returned infinity instead of +-MAX when the exact result overflowed.
3. `shfl.sync.*` used all 32 bits of the lane operand; PTX uses bits [4:0].
4. `cvt.sat.s8.*` produced a zero-extended 16-bit register.
5. Simulator only: conversions to f32 ignored `FPRoundingMode` decorations, so directed
   rounding of int/f64 to f32 conversions could not be checked; it now models RTZ/RTN/RTP.

### Open questions for hardware

See `tests/gpu/hw_translator_edge.rs` (all `#[ignore]`, not yet run): integer division edge
cases, shfl operand masking, barrier with exited threads, directed rounding and overflow,
`.ftz` behaviour probe, `cvt.sat.s8` sign extension.

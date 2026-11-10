# PTX conformance and gap report

Scope: what `src/ptx.rs` + `src/ptx/*` (PTX to OpenCL-environment SPIR-V) accepts today, measured on a checked-in corpus, and which gaps matter most for GGML/llama.cpp and PyTorch kernels.

Reproduce: `tests/ptx_corpus/run.sh` (builds `tests/ptx_corpus/harness`, a scratch crate that compiles `src/ptx.rs` through symlinks, because the main crate does not build on macOS). `run.sh --bless` rewrites `tests/ptx_corpus/expected_status.tsv`. Columns: path, status, strict (would the unmodified file translate), first-failure message.

## Method and caveats

- Corpus (623 files): `zluda/` 230 kernels derived from ZLUDA's test suite (MIT OR Apache-2.0, header and licenses kept; `import_zluda.py` regenerates), `handwritten/` 349 single-feature probes (`gen_handwritten.py`), `ggml/` symlink to `examples/*.ptx` (44 files, real nvcc output of stock GGML kernels).
- Status is the translator's **first** failure per file. A file marked unsupported may hide further gaps behind the first one. Rerun after each fix.
- The harness normalises two things so the report shows real gaps, not gates: `.version` is rewritten to 8.8 (translator accepts only 4.1/8.8/9.0, 575 of 575 files fail strictly otherwise) and bare register names in ZLUDA files get a `%` prefix (translator requires `%`; nvcc always emits it, so this is not a practical gap).
- Probes with uninitialised-register hazards are avoided by initialising every register first. Probe operand shapes (for example mma fragment arity) are plausible, not validated by ptxas, and the translator rejects on opcode before checking them.
- Counts "kernels that need X" in section 2 are estimates from my knowledge of llama.cpp `ggml-cuda` and PyTorch (marked "memory"). No llama.cpp or PyTorch checkout was available offline.

## 1. Corpus result

Totals: pass 94, unsupported 450, error 79 (of 623).

| Family | pass / unsupported / error |
| --- | --- |
| ggml (real nvcc PTX) | 41 / 3 / 0 (3 fail on an empty module directive, see 3.1) |
| handwritten atomics | 1 / 32 / 0 |
| handwritten barrier/fence/cluster | 1 / 22 / 3 |
| handwritten bitmanip (prmt, dp4a, lop3, bfe, bfi, fns, clz, popc, ...) | 5 / 53 / 0 |
| handwritten cp.async, bulk, mbarrier, st.async | 0 / 16 / 6 |
| handwritten cvt | 7 / 15 / 2 |
| handwritten float_math | 19 / 15 / 0 |
| handwritten half/bf16 arithmetic | 0 / 16 / 0 |
| handwritten mma + wgmma + tcgen05 | 0 / 16 / 2 |
| handwritten wmma | 0 / 3 / 0 |
| handwritten ldmatrix/stmatrix | 0 / 7 / 1 |
| handwritten shfl | 3 / 11 / 0 |
| handwritten vote/match | 0 / 10 / 0 |
| handwritten redux | 1 / 7 / 1 |
| handwritten sreg | 7 / 2 / 21 |
| handwritten tex/surf | 0 / 9 / 0 |
| zluda/* (all families) | 11 / 195 / 24 |

Interpretation: the translator is a targeted GGML-subset compiler. It is complete for the kernels that currently reach it (all 41 stock GGML dumps that were handed to it pass), and is deliberately narrow elsewhere. The 3 ggml failures are the next stock kernels (`k_set_rows`, `rope_neox` f32/half).

## 2. Ranked gaps (by how many GGML / llama.cpp / PyTorch kernels need them)

Rank 1 is "fix first". Effort: S hours, M days, L weeks. Evidence column cites the corpus probe that fails.

| # | Feature | Who needs it (memory estimates) | Corpus evidence | Effort |
| --- | --- | --- | --- | --- |
| 1 | **Module-level directives and `.func` / `call`** (`.func`, `.global` data, `.extern`, `__assertfail`, `vprintf`) | Every PyTorch kernel TU with device helpers and any TU where nvcc does not inline (`.func`, `__assertfail`, `vprintf`); most non-trivial TUs. The 3 ggml dump failures are NOT this (see 3.1) | `zluda/control/call*`, `handwritten/memory/func_call_internal` | M-L (inlining pass) |
| 2 | **`::` qualifiers in the tokenizer** (`ld.global.L2::128B`, `L1::no_allocate`, `ld.shared::cluster`, `fence.proxy.async.shared::cta`, `cp.async.cg...L2::128B`) | nvcc emits `L2::128B`/`::cache_hint` prefetch qualifiers for many sm_80+ loads in GGML/PyTorch; blocks otherwise-supported loads | 8 probes fail with `invalid PTX identifier ...L2` | S |
| 3 | **Predicated non-branch instructions** (`@%p add...`, `@!%p ld`, `@%p ret`); translator accepts only `@p bra.uni` | Virtually all real kernels at -O2 (guarded loads/stores, selects lowered to predication) | `handwritten/control/pred_guard_neg`, `ret_in_branch` | M |
| 4 | **Generic-space `ld`/`st`** (no `.global`/`.shared`), `cvta` on non-global, `mov` of addresses | PyTorch/CUTLASS generic pointers, any kernel taking `T*` from mixed spaces; ZLUDA corpus is dominated by it (27+19+19+17+11 failures on plain `ld.f32/u32/b32/u64/s32`) | `zluda/memory/*`, `zluda/integer/*` | M |
| 5 | **f64** (`.reg .f64`, `.param .f64`, all f64 arithmetic/cvt, `fma.rn.f64`, `sqrt.rn.f64`) | PyTorch double tensors, `torch.linalg`, reductions in double (hundreds of kernel instantiations); GGML uses double rarely (a few softmax/norm accumulators in ggml-cuda, ~5%) | 15 files fail on `unsupported PTX register type .f64`; capability also needs Level Zero fp64 per device (Iris Xe lacks native fp64 on some SKUs; emulation needed) | M, device-gated |
| 6 | **f16 / bf16 / f16x2 / bf16x2 registers and arithmetic** (`.reg .f16`, `add/mul/fma.rn.f16x2`, `bf16`, `ex2.approx.f16x2`, `min/max.NaN`, `cvt.rn.f16x2.f32`, `cvt.rn.bf16x2.f32`) | PyTorch half/bf16 (very common: every elementwise, softmax, layernorm, embedding for fp16/bf16 models); GGML: f16 `mul_mat_vec_f`, `flash-attn`, `rope` half, `convert`, `cpy`, `get_rows` (~60+ kernels; some already handled through b16 loads) | 16 handwritten and 8 ZLUDA files; only `cvt` between f32 and f16 loads pass | M |
| 7 | **`mma.sync` shapes** m16n8k16 f16/bf16 to f32 (and f16 acc), m16n8k8 f16/bf16/tf32, m16n8k32 s8 to s32, m8n8k4 (sm_70) | GGML: `mmq.cuh` (all quantised types, int8 `m16n8k32`, ~40 instances), `mma.cuh`, `fattn-mma-f16` (m16n8k16 f16, ~50 instances), `mmf` ; PyTorch: GEMM-free fused attention and CUTLASS-based kernels (hundreds). The single biggest throughput item; without it prompt processing falls back to dp4a/scalar paths | 16 probes + 5 ZLUDA mma files fail | L (needs sub-group matrix, `SPV_INTEL_subgroup_matrix_multiply_accumulate`, or scalar emulation first) |
| 8 | **`ldmatrix` (x1/x2/x4, `.trans`) / `stmatrix`** | Always paired with mma.sync: `fattn-mma`, `mmq`, `mma.cuh` loads (~90 GGML instances), CUTLASS | 7 probes + 2 ZLUDA fail | M (with #7) |
| 9 | **`vote.sync` (ballot/all/any/uni), `activemask`, `match.any/all.sync`** | GGML: `quantize`, `softmax`, `topk/argsort`, `ssm`, warp-wide predicates (~25 kernels); PyTorch: reductions, scan, `nonzero`, sort, unique (~100+) | 10 vote probes + 11 ZLUDA files fail; note `shfl.sync` and `redux.sync` basic forms already pass | S-M |
| 10 | **`shfl.sync` predicate-out forms and `up/down/idx` modes** (only the `bfly` and some forms translate) | warp reductions in nearly every kernel using `warp_reduce_*`; GGML `norm`, `softmax`, `mmvq` (already needed bfly only), scan kernels use `up`; PyTorch `WarpShuffle` | 11 of 14 shfl files fail (`up`, `down`, `idx`, and `|p` output forms) | S |
| 11 | **`prmt`** (default and `.f4e/.b4e/.rc8/.ecl/.ecr/.rc16`) | GGML dequant/mmq int8 packing (`__byte_perm`, ~30 kernels); PyTorch fp8 / int4 pack | 8 prmt probes fail | S |
| 12 | **`lop3.b32`, `bfe`, `bfi`, `fns`, `bmsk`, `szext`, `shf`, `bfind`, `brev`, `clz`, `popc`, `sad`, `mul24/mad24`, `addc/madc` carry chain, `mad.hi`, `mul.hi.u64`** | `lop3` is emitted by ptxas-bound nvcc for any 3-input boolean expression (GGML dequant masks `0x0f0f0f0f` mixes, `mmq`); `popc`/`clz`/`bfe` in PyTorch (sort, topk, hashing, RNG Philox uses `mul.hi.u32`/`mad.wide`); `shf` in rotation/hash. Probably 100s of kernels | 53 handwritten + 21 ZLUDA fail; only a handful pass | M (each is small; schedule as a batch) |
| 13 | **`dp4a` / `dp2a`** | Already a stock GGML dependency: tests/gpu/ggml_dp4a.rs shows `dp4a.s32.s32` handled. Variants `u32.u32`, `s32.u32` and `dp2a.lo/hi` still fail in the probes. GGML MMVQ/MMQ use `__dp4a` heavily (~200 instantiations) | 5 probes | S |
| 14 | **Atomics** (`atom`/`red` on f32/f64/u64/f16x2/bf16x2, `cas.b64/.b128`, `exch`, `min/max`, `inc/dec`, scopes/sem `.relaxed.gpu`, shared-space) | PyTorch: `scatter_add`, `index_add`, embedding backward, histogram, batchnorm (several hundred kernels). GGML: `atomicAdd(float)` in `mmq` stream-K fixup and `softmax`/`flash-attn` tile fixups (~10) | 32 of 33 handwritten + 5 ZLUDA fail; `atom.global.add.u32` is the one that passes | M |
| 15 | **`bar.sync`/`barrier.sync` variants** (id register, thread count, `.aligned`, `bar.arrive`, `bar.red.popc/and/or`, `bar.warp.sync`), `membar`/`fence.*`, `ld.acquire`/`st.release` | `bar.sync 0` is in nearly every reduction kernel. Only `bar.sync 0` works today. GGML `softmax`, `norm`, `mmq`, `fattn` need `bar.sync`; `bar.red.*` is `__syncthreads_count/and/or` (PyTorch only) | 22 handwritten barrier probes + 3 ZLUDA fail | S (bar.sync) to M |
| 16 | **Special registers** (`%laneid`, `%warpid`, `%nwarpid`, `%smid`, `%nsmid`, `%gridid`, `%clock`, `%clock64`, `%globaltimer`, `%lanemask_*`, `%WARP_SZ`, cluster regs, `%total_smem_size`, `%dynamic_smem_size`) | `%laneid` (`threadIdx.x & 31` is more common but `__lane_id()` appears in PyTorch scan/sort; `%lanemask_lt` in `__ffs`-style compaction; `%clock64` in profiling-only code); `%dynamic_smem_size` rare | 21 sreg probes fail with "undefined register" (the sreg lookup falls through) | S |
| 17 | **`cp.async` (ca/cg 4/8/16, zfill, `commit_group`, `wait_group`, `wait_all`, `mbarrier.arrive`)** | GGML: `fattn-mma`, `mmq`, `mmf` use cp.async to stage tiles on sm_80+ builds (~100 instances); PyTorch: CUTLASS/FlashAttention-2, xformers (hundreds). Lowering as synchronous copy + no-op group ops is legal and cheap | 22 probes + 2 ZLUDA fail | S (sync fallback), L (true async) |
| 18 | **`wmma.load/mma/store`** | GGML `fattn-wmma-f16` fallback (sm_70/AMD-style; ~30 instances), PyTorch legacy tensor-core kernels; supersede by #7 | 3 probes | L |
| 19 | **Unsupported targets / ISA versions**: only `.target sm_52/75/80`-style names, `.version 4.1/8.8/9.0`. sm_90, sm_90a, sm_100a, sm_120 and the many in-between PTX versions are rejected (`expected a PTX sm target`, `unsupported PTX version`) | **Blocker for everything**: any stock fatbin PTX from CUDA 12.x/13.0 (versions 8.0-8.7, 9.0 only for CUDA 13.0) and PyTorch wheels (PTX embedded for last arch, `compute_90` or `compute_120`) | 575/575 files fail strictly on version; 7 probes on target | S |
| 20 | **`tex`/`tld4`/`txq`/`suld`/`sust`/`sured`** | PyTorch image ops (`grid_sample` legacy, `torch.nn.functional.interpolate` older paths use plain loads; texture mostly unused in current PyTorch), GGML none. Low | 9 probes + 24 ZLUDA tex files fail | L (needs image objects) |
| 21 | **Hopper/Blackwell**: `wgmma`, `tcgen05`, `cp.async.bulk(.tensor)`, TMA, `mbarrier.*`, `st.async`, `red.async`, `elect.sync`, `setmaxnreg`, `getctarank`/`mapa`, `griddepcontrol`, cluster barriers, `multimem` | Only on sm_90+/sm_100 code paths of FlashAttention-3/CUTLASS 3 (PyTorch SDPA FA3, `torch._scaled_mm`, cuBLASLt-class kernels), GGML sm_90 paths are `mma.sync` based (does not emit these). Not needed if the Intel target advertises sm_80 and fatbins carry sm_80 PTX | 36 probes fail by design | skip (hardware-class gap; advertise sm_80) |
| 22 | **`createpolicy`/`applypriority`/`discard`/`prefetch`/`ld.global.L2::cache_hint`** | cache-hint operands are performance-only: lower to plain access. emitted by CUTLASS and recent GGML builds on sm_80+ | 6 probes | S |
| 23 | **fp8/fp4/tf32 converts** (`cvt.rn.satfinite.e4m3x2.f32`, `e2m1x2`, `cvt.rna.tf32.f32`) | PyTorch `float8_e4m3fn` casts (sm_89+), GGML none (MXFP4 dequant uses LUT + `prmt`, not hardware cvt) | 4 probes + 5 ZLUDA files | M |
| 24 | **Misc**: `trap`/`brkpt`/`nanosleep`/`exit`, video instructions (`vadd2/vabsdiff4/vset4/vmax`, used by `__vcmpeq4`, `__vabsdiff4` etc. in GGML `mmvq` int8 paths? rare), `isspacep`, `testp`, `slct`, `cnot`, `.v2.b16`/`.v4` register vector types, `mov.b64 {a,b}` pack/unpack, `mov.pred` | each small; `trap` is the `GGML_ABORT` path and `__trap()` appears in PyTorch assert paths; the vector-register and pack/unpack forms appear in nearly every half2/bf162 kernel | spread across families | S each |

Reading the order: ranks 1-4 and 19 are structural (they block most real TUs, independent of feature breadth). Ranks 6-8 are the performance cliff for model workloads. The ZLUDA corpus is hand-written PTX that mostly uses generic loads and bare names, so it overstates the importance of #4 for nvcc output and understates #3.

## 3. Details worth knowing

### 3.1 Module directive failure on three stock GGML dumps
`ggml/k_set_rows_dump.ptx`, `rope_neox_f32_dump.ptx`, `rope_neox_half_dump.ptx` fail with `unsupported PTX module directive ` and an empty directive token. The files start with `.version 8.8 / .target sm_80 / .address_size 64` and a plain `.entry`, with no `.func`/`.global`, so the cause is a stray token (the parser's NUL-only skip at `src/ptx/parse.rs` ~line 1105 suggests trailing NULs or similar from the dump process) rather than a missing feature. Treat it as a fixture/tokenizer defect, to be confirmed by whoever owns `src/ptx*`; no feature work is implied.

### 3.2 Tokenizer
`::` is not a valid identifier character in the tokenizer. Adding it unblocks eight otherwise-trivial probes in a single change (rank 2).

### 3.3 What passes
Integer `add/sub/mul/mad` core forms, `ld/st.global/shared/param`, `cvta.to.global`, `setp`, `selp`, f32 arithmetic including `fma.rn.f32`, `sqrt.rn.f32`, `rcp.approx.ftz`, `ex2.approx`, `lg2.approx`, `sin/cos.approx`, `bar.sync 0` (but not `bar.sync id, count` or a register id), `shfl.sync.bfly.b32`, `redux.sync.add.u32` (GGML-style), `atom.global.add.u32`, `dp4a.s32.s32`, `cvt` f32<->f16 loads, special registers `tid/ntid/ctaid/nctaid`. Authoritative list: `grep pass tests/ptx_corpus/expected_status.tsv`.

### 3.4 Sources for item 2 estimates
Counts are rough instantiation counts for llama.cpp `ggml/src/ggml-cuda` (mmq.cu, mmvq.cu, fattn-*.cu, mma.cuh, softmax.cu, norm.cu, rope.cu, quantize.cu, convert.cu, cpy.cu, getrows.cu, set-rows.cu, ssm-*.cu, sum*.cu) and PyTorch `aten/src/ATen/native/cuda` plus CUTLASS/FlashAttention extensions. Treat as ordering signals, not measurements. To make them measured, build GGML PTX with `scripts/build-ggml-ptx.py`-style `nvcc --ptx` for all `.cu` units (needs a CUDA host, not run here) and drop the output under `tests/ptx_corpus/ggml_full/`; `run.sh` will pick it up and `awk` over `expected_status.tsv` gives true per-feature counts.

## 4. Suggested order of attack

1. Accept PTX versions 7.x-9.x and sm_70-sm_120 target names (rank 19) and the `::` tokenizer fix (rank 2). Rerun the corpus.
2. Fix the three stock GGML directive failures (3.1); this unblocks the Ollama path.
3. Predicated instructions and generic-space `ld/st` (ranks 3-4), then `bar.sync` variants, special registers, `shfl.sync` modes, `vote`, `prmt`, `lop3`/`bfe`/`bfi`/`popc`/`clz` (ranks 9-16): small, high-yield.
4. f16/bf16/f16x2 register types (rank 6), then atomics (rank 14).
5. `mma.sync` + `ldmatrix` + `cp.async` as one project (ranks 7, 8, 17); begin with scalar emulation to get correctness, then Intel subgroup-matrix extensions.
6. f64 (rank 5) gated on device capability.

## 5. Tier-2 update (branch `ptx-tier2`)

Corpus pass count: 94 in the original report, 203 on `integration` (ptx-coverage and audit-leftovers merged) and 435 after the tier-2 work (of 623; the remainder is dominated by texture/surface, Hopper/Blackwell, fp8/fp4, video SIMD instructions, module-scope `.global`/`.const` data, named barriers with partial thread counts and ZLUDA fixtures that use bare register names).

New files (kept apart from the hand-written parser to limit merge conflicts):

- `src/ptx/normalize.rs`: token-level canonicalisation before parsing: `sm_90a`/`sm_100a`/`sm_120` targets and any `.version`, `::` qualifiers (`L2::128B`, `shared::cta`, `cache_hint`), cache and memory-order qualifiers on `ld`/`st` (acquire/release add a fence), float-typed `mov`/`ld`/`st`/`selp`, `barrier.sync`/`bar.sync` with register or named ids (a thread count is accepted only when it equals `.reqntid`/`.maxntid`), legacy `vote`/`shfl`, special registers, `exit`/`trap`/`brkpt` as `ret`, `prefetch`/`discard`/`createpolicy` and cluster/PDL no-ops, `cp.async` as a synchronous copy (zero-fill form supported) with no-op commit/wait, 64-bit vector accesses split into scalars, dynamic `.extern .shared` aliasing, `mad.f32`/`rcp.rn`/`div.full` aliases, and `wmma` m16n16k16 (f16 inputs, f32 or f16 accumulators) lowered onto the `mma.sync.m16n8k16` emulation with its own fragment layout.
- `src/ptx/inline.rs`: `.func` definitions are inlined at call sites (registers and labels renamed per site, `.param` frames turned into moves, recursion rejected). `__assertfail` and `free` calls are dropped, `vprintf` returns 0.
- `src/ptx/ext.rs`, `ext_emit.rs`, `ext_float.rs`: new `Instruction::Ext`: brev, bmsk, fns, szext, mul24/mad24, sad, cnot, bfind, clz.b64, mad.hi, mul.hi.u64, bfe/bfi.b64, all `prmt` modes, dp4a/dp2a combinations, shf, slct, `setp` with bool combine, `mov.b64`/`mov.b32` pack/unpack, `%warpid`/`%nwarpid`/`%clock*`, `redux.sync` (add/min/max/and/or/xor), `match.any/all.sync`, `vote.sync.uni`, `elect.sync`, `shfl.sync` with register control, 64-bit atomics (add/min/max/and/or/xor/exch/cas), `atom.inc/dec`, f32/f64 min/max/add atomics and `f16x2`/`bf16x2` add (CAS loops), f64 arithmetic and compares, f16/bf16 scalar and packed arithmetic, saturating and NaN-propagating f32, unordered f32 compares, `testp`, and a generic `cvt` (f64, bf16, packed halves, tf32, directed rounding, saturating integer narrowing).
- `src/ptx/tier2_tests.rs`: executable tests on the SPIR-V interpreter (extended with float arithmetic, compares, conversions and OpenCL.std math) plus `spirv-val` validation.

Known limits: f16/bf16 `fma` rounds twice (via f32); f64 only supports round-to-nearest; `bar.sync` with a partial thread count, `bar.red`, `bar.arrive`, mbarrier, TMA, tex/surf, wgmma/tcgen05, fp8/fp4 conversions, video SIMD ops and carry-chain `add.cc`/`addc` remain unsupported; `%clock` needs `SPV_KHR_shader_clock` on the device.

## 6. Tier-3 update (branch `ptx-tier3`)

Corpus (623 files): pass 520, unsupported 76, error 27 (tier 2: 435 / 116 / 72). Reproduce with `tests/ptx_corpus/run.sh`. Unit tests (`cargo test --release --manifest-path tests/ptx_corpus/harness/Cargo.toml`, now runs on macOS through symlinks) pass: 124, including numeric checks on the 32-lane SPIR-V interpreter and `spirv-val` on every translated module.

Corpus fixes: the harness now maps bare register ranges (`temp<3>` / `temp0`) and skips its own crate directory; the generator emits valid shared/local addresses and mma fragment arities, and no longer gives a destination-first register a type-conflicting initialiser. The 72 tier-2 "error" cases were mostly fixture artefacts (bare names, global pointers used as shared addresses); the remaining 27 are ZLUDA parser torture tests (`call`/`func_ptr`/malformed labels, in-place `cvta`), Hopper cluster spellings, and `ld.param.v2` on a scalar parameter.

New files: `src/ptx/tier3.rs` (text lowering hooked first in `normalize::rewrite_inst`), `tier3_mma.rs` (tensor cores), `tier3_data.rs` (parser for module data), `tier3_emit.rs` (SPIR-V for module data), `tier3_tests.rs`, `tier3_mma_tests.rs`. Small hooks: `normalize.rs` (calls, entry bookkeeping), `parse.rs`/`spirv.rs`/`ptx.rs` (module data), `ext.rs` (`mov.u64 %clock64`, which an earlier arm shadowed).

Closed gaps:

- **Carry chains**: `add.cc`, `addc`, `sub.cc`, `subc`, `madc`, `mad.lo/hi.cc` (u32/s32/u64/s64) through a per-entry `CC.CF` register; subtraction uses borrow semantics.
- **Video**: `vadd/vsub/vabsdiff/vmin/vmax/vavrg/vset/vshl/vshr` with `.b0-.b3/.h0/.h1` selectors, `.sat` and `.add/.min/.max`, and the SIMD forms `v*2` / `v*4`.
- **Warp collectives**: partial immediate `membermask` on `shfl.sync`/`vote.sync` (SPIR-V subgroup ops act on the active lanes), masked ballot, `bfly` with segment masks, `%laneid`-based code.
- **Barriers**: `bar.sync id, count` (not equal to `.reqntid`), `bar.arrive`, `bar.red.popc/and/or`, `barrier.cta.*`, `bar.warp.sync`. Counted barriers use shared arrival counters and a polled generation word (atomic reads, so the loop cannot be hoisted); reductions use a shared counter between full barriers.
- **Tensor cores**: `mma.sync` m16n8k8/k16 bf16, f16 with f16 accumulators, m16n8k4/k8 tf32, m8n8k4 f64 by subgroup-shuffle emulation (exact products, f32/f64 accumulate); `wmma` load/mma/store for m16n16k16, m32n8k16, m8n32k16 with f16 (f32 or f16 acc), bf16, s8/u8; m16n16k8 tf32; m8n8k4 f64; `stmatrix` (b16, x1/x2/x4, `.trans`). m16n16k16 f16 stays on the tier-2 path and fragment layouts match it.
- **Memory**: several `.local` arrays merged into one depot, `[sym+off]` and 64-bit accesses on `.local`/`.shared` symbols, typed (`.b32`, `.f32`) and unaligned declarations, `ld.param.v2/v4` (byval structs), `ld.const`/`cvta.const`, `.noreturn`, trailing labels, debug sections stripped (`.file`, `.loc`, `.section`), generic-space atomics, module-scope `.global`/`.const` arrays with initialisers as program-scope `CrossWorkgroup` variables (lookup tables, `__device__` variables; they persist across launches like CUDA's).
- **Numerics**: `{add,sub,mul,fma,div,sqrt,rcp}.{rz,rm,rp}.f64` via error-free transformations (one-ulp step), `fma.rn` on f16/bf16/f16x2/bf16x2 now rounds once from f64, `cvt.pack.sat`, same-width integer `cvt`.
- **Misc**: `setp.p|q`, `setp.cmp.and/or/xor` with negated predicate, `_` sinks, literal-predicate `selp`, `U` literal suffixes, special registers as plain operands, `%clock64`/`%globaltimer`, `%dynamic_smem_size`/`%total_smem_size` (constant 49152, the window the translator allocates), `WARP_SZ`.
- **tex/surf/sured/txq**: fail early with `unsupported PTX instruction: ... need bindless images`; no bindless image extension is exposed by the OpenCL-environment SPIR-V target, so image2d lowering is not possible without a host-side handle table.

Still open: `mma.sp` sparse (metadata layout not validated), s4/b1 `mma`, legacy `mma.m8n8k4` f16 (sm_70), `redux.f32`, fp4/fp6 converts, `atom.cas.b128`, `multimem`, mbarrier/TMA/wgmma/tcgen05 (Hopper/Blackwell, by design), `cvta.to.shared` of an untyped generic pointer, ZLUDA control-flow torture files. Semantics worth knowing: counted barriers need all participating subgroups resident (true on Xe), `%dynamic_smem_size` is not the launch value, f64 directed rounding does not model overflow to the largest finite value, and tensor-core results differ from hardware only in summation order.

## 7. vosen/ptx_tests (roadmap item 1, branch `ptx-tests`)

ZLUDA's per-instruction suite now runs on the simulator (`tests/ptx_tests/run.sh`) and on the B390
(`scripts/hwvalidate/ptx_tests.ps1`): 668 of 787 cases pass on the GPU by default and 780 with `CFI_EXACT_FTZ=1`.
The cases left and the 12 translator bugs it found are in `hw-logs/ptx-tests-2026-11-09.md`; `tests/ptx_tests/expected_status.tsv`
gives each remaining case its reason.


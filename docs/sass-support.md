# SASS and PTX-less binary support

## Status
S1 is implemented: a SASS-only fatbin or a standalone cubin ELF (e_machine 190) prints a one-time error naming the SASS architectures and this page, and still returns CUDA_ERROR_NO_BINARY_FOR_GPU (209). `scripts/ptxscan.py` is in the repo.

The error names the module that carries the image, once per module: `cfi_fatbin::offending_module` resolves the DLL whose mapped range holds the fatbin (`GetModuleHandleExW(FROM_ADDRESS)`; `dladdr` on Linux). For a heap copy, it uses the first caller on the stack that is neither our module nor a Windows system DLL (`RtlCaptureStackBackTrace`). Example: "`cublas64_12.dll` carries SASS only (sm_50, ...) and no PTX; use cuda-for-intel's stand-in for it (scripts/cfi.ps1 install) or start the program with cfi-run". Other modules, such as `torch_cuda.dll`, get the "use a PTX build" advice. Unit test: `cfi-fatbin` `sass_only_diagnostic_names_the_carrying_module_once` (synthetic SASS-only frame). Not yet confirmed on the box with torch's stock cuBLAS (roadmap item 5, GPU part).

## Verdict
Translating SASS (cubin) to SPIR-V is **technically possible but not realistic** as a general path: ~6-12+ person-months for a narrow, unverified subset, per-arch (sm_75/80/86/89/90 each have different encodings), no PTX-style spec, and a ToS problem. The **realistic path is to avoid needing SASS**: PTX-carrying builds, own library stand-ins, and a clear diagnostic. Cheapest useful step: a PTX-less detector + diagnostic (prototype done, ~1 day to wire into the shim).

## 1. How common are PTX-less binaries (relevant apps)
Measured with `ptxscan.py` (scripts/ptxscan.py; read-only fatbin header walk) and the earlier agent's notes (the PyTorch PTX availability survey):

| Binary | PTX? | Notes |
|---|---|---|
| Ollama / ggml-cuda.dll (v0.12.6 cuda13) | **PTX in 117/117 fatbins** (sm_75..121), the scan found no SASS entries in that file | Works today; the shim's main target |
| conda-forge libtorch win cuda128 torch_cuda.dll | PTX sm_120 in 308/308 + SASS sm_50..120 | OK (PTX-carrying) |
| PyTorch Linux cu129/cu130 wheels | PTX (compute_120) | OK |
| PyTorch cu126, cu128 (both OSes), all Windows cu121+ wheels | **SASS only** | Blocked. cu128 verified SASS only |
| PyTorch Win cu118 (2.0-2.7.1) | PTX compute_37 | OK, easy ISA |
| cuBLAS / cuBLASLt / cuDNN / cuFFT / cuSOLVER / cuSPARSE / NCCL | **SASS only** (closed source, PTX deliberately stripped; cuBLAS may also carry some NVRTC-JIT'd kernels but not as shipped PTX) | Shim must not load them; it already ships `cublas_shim` (crates/cublas_shim) and a cudnn stand-in (cudnn crate), so these are *avoided*, not translated |
| llama.cpp release builds | PTX for the highest arch in the list typically (CMAKE_CUDA_ARCHITECTURES, "-real" suffix removes it); verify per build with ptxscan | Mostly OK |
| Triton / torch.compile / NVRTC output | Generated as PTX first, then ptxas'd; the PTX is available at the NVRTC/driver boundary | OK if NVRTC is real NVIDIA's (PTX output) |
| Third-party wheels (flash-attn, xformers, bitsandbytes, vLLM kernels) | Mostly SASS only (built with explicit `-gencode arch=..,code=sm_XX`) | Blocked unless rebuilt with +PTX |

So the failure set is: stock cu126/cu128 PyTorch wheels, third-party CUDA extensions, and any closed-source NVIDIA library the shim has not replaced. For libraries the answer is "replace" (oneMKL/own GEMM), not "translate"; SASS from cuBLAS is also the hardest SASS (hand-tuned, tensor core / TMA / wgmma heavy).

## 2. SASS ISA situation
- Not documented by NVIDIA (only an instruction *list* per arch in the CUDA Binary Utilities doc, no encodings/semantics). Encodings change per major arch: sm_70/75 (128-bit, scoreboard in control bits), sm_80/86/89 (Ampere/Ada mostly Turing-like with new ops: LDGSTS=cp.async, HMMA.16816, BF16), sm_90 (Hopper: WGMMA, TMA UTMALDG, clusters/DSMEM, warpgroup sync), sm_100+ (tcgen05). Pre-Volta (Maxwell/Pascal) is a different 64-bit-with-control-word format (MaxAs/KeplerAs era), irrelevant here.
- Tools: `cuobjdump`/`nvdisasm` ship in the CUDA Toolkit (free download). They give disassembly + CFG only. CuAssembler (cloudcores, forks OpenPPL / bogan-FMA): unofficial *assembler* (Pascal..Ampere, sm_60-86; Ada/Hopper unconfirmed), built by reverse-engineering with nvdisasm as an oracle. MaxAs/KeplerAs are Maxwell/Kepler only. Rust crate `sass-assembler` exists (small). JEB 5.31 (commercial) has an experimental SASS decompiler sm_70..121.
- Lifters (research, 2026): **CuLifter** (arXiv 2604.27486) lifts SASS to typed LLVM IR, 11,977 kernels lift to valid IR, >90% of a testable set executed correctly on CPU; type recovery from the untyped register file is the hard part; no code link found; tensor core handling not stated. **NVLift** (purseclab/Sass-LLVM-Lifter, NDSS BAR 2026): sm_75 only, proof of concept, ~25 instructions, no license. Conclusion: nobody has a production SASS->SPIR-V/PTX path; a lifter to LLVM IR would then need llvm-spirv, and the lifted code would carry explicit SASS-isms (warp-synchronous assumptions, 32-wide warps, scoreboards, predicate regs, uniform regs, shared-mem banking, `HMMA` fragment layouts) that Intel Xe (subgroup 16/32, XMX) would have to remap. Also nothing recovers *PTX* from SASS: ptxas discards it.
- Hard parts specific to us: tensor-core fragment layouts (HMMA/WGMMA to XMX), TMA/async barriers, SASS-level `SHFL/VOTE/MATCH` semantics, ABI (param bank c[0x0][...] offsets, kernel param layout recovered from `.nv.info` in cubin), cubin loading (ELF sections `.nv.constant0`, `.nv.shared`, relocations), and per-arch coverage tests (no oracle on this hardware other than a real NVIDIA GPU).

**Legal/ToS (not legal advice):** The CUDA Toolkit EULA (sec. 1.2 restrictions) forbids reverse engineering, decompiling or disassembling "any portion of the output generated using SDK elements for the purpose of translating such output to target a non-NVIDIA platform"; this is exactly SASS-to-Intel translation of binaries compiled by nvcc. The same EULA restricts redistributing/distributing modified NVIDIA libraries (cuBLAS/cuDNN etc. are redistributable only under the attachment terms, unmodified). Disassembling the closed libraries to port them is therefore the highest-risk area; PTX is a documented, public ISA with explicit forward-compat JIT intent, and translating it is a much safer footing (the same reason ZLUDA focused on PTX; note ZLUDA's AMD era was challenged by NVIDIA's 2024 EULA wording). Interoperability exceptions vary by jurisdiction. Need counsel before any shipping of SASS lifting.

## 3. Practical alternatives (ranked)
1. **Use PTX-carrying builds (zero shim work)**: cu129/cu130 Linux wheels; conda-forge torch (win/linux, cuda128+); Win cu118 wheels for early triage; Ollama builds (already PTX). Already documented in TORCH-PTX-AVAILABILITY.md. Ship a "known-good wheels" table + download helper.
2. **Rebuild with PTX**: `TORCH_CUDA_ARCH_LIST="8.0+PTX"` (or `-gencode arch=compute_80,code=compute_80`) for torch, flash-attn, xformers, bitsandbytes, llama.cpp (`CMAKE_CUDA_ARCHITECTURES=80` without `-real`). Cost: 2-6 h build per project on a CPU box; we publish "intel-shim" wheel set. Choose a low virtual arch (compute_75/80) so PTX uses fewer new features than sm_120 PTX (ISA 8.7), which is easier to translate (no wgmma/tcgen05).
3. **Cubin->PTX fallback trick**: does not exist (PTX not embedded). What *does* help: some libs/apps embed PTX *and* cubin side by side, in which case just prefer PTX (shim already does). Apps that call NVRTC/`cuModuleLoadData` with PTX strings (Triton, torch JIT fusers) are fine.
4. **CUDA_FORCE_PTX_JIT=1**: this is a *driver* env var that makes the real NVIDIA driver ignore SASS and JIT PTX. In our shim it is effectively already the behavior (we only read PTX). Its value: with the real driver on an NVIDIA box it tells you immediately whether an app has usable PTX (run the app with it set; if it fails with "no kernel image", the app is PTX-less). Good test hook for CI, and a hint to users. It does not create PTX.
5. **Replace libraries**: cuBLAS/cuBLASLt/cuDNN/cuFFT via own implementations (done for cuBLAS/cudnn partially; cuFFT, cuSOLVER, cuSPARSE, NCCL on demand), backed by oneMKL/oneDNN. This is where most SASS-only pain actually is for PyTorch.
6. **Torch-side kernel regeneration**: for PTX-less torch ATen, as stopgap build `libtorch_cuda` once in CI per release with +PTX and publish.
7. SASS lifting: last resort (below).

## 4. Staged plan
| Stage | What | Effort |
|---|---|---|
| S0 (done here) | `ptxscan.py` detector, validated on ggml-cuda.dll (117 fatbins PTX) and conda-forge torch_cuda.dll (308/308 PTX + SASS), synthetic SASS-only fatbin returns rc 2 + diagnostic | 0.5 d |
| S1 | Port the check into the shim: in `src/cuda.rs::load_module_from_fatbin`, `src/cuda/library.rs` (cuLibraryLoad), `crates/cudart_shim` registration, and `cuModuleLoadData` paths. When `extract_ptx` is None, ALWAYS (not only when trace is on) log once per module: "module has SASS for sm_XX but no PTX; cuda-for-intel needs PTX. Rebuild with +PTX / use a PTX wheel (docs/sass-support.md)", and detect standalone cubin ELF (e_machine 190). Keep returning `CUDA_ERROR_NO_BINARY_FOR_GPU` (209). Add `cfi_fatbin::summarize()` (kinds/arches) reusing `describe`. Tests: unit fixtures with SASS-only frame. | 1 d |
| S2 | "Known-good" tooling: `scripts/check-binary` (ptxscan) in repo + wheel table in docs; CI job that scans pinned wheels; optional preflight on `import torch` via env hint | 1-2 d |
| S3 | Reproducible PTX-carrying builds: CI recipe for torch (cu126/cu128 source, `TORCH_CUDA_ARCH_LIST=7.5+PTX;8.0+PTX`... only highest PTX is used by JIT), flash-attn, bitsandbytes, llama.cpp; publish artifacts. Needs a CPU build box (big, ~64 GB RAM/32 cores, 3-5 h each for torch) | 1-2 wk incl. debugging |
| S4 | Library stand-ins for what torch actually calls: cuFFT, cuSOLVER-lite, cuSPARSE-lite on oneMKL | 2-6 wk, demand driven |
| S5 (not recommended) | SASS lift for one arch (sm_80/86, no wgmma), cubin loader + `.nv.info` param ABI + SASS->SPIR-V via own IR or LLVM IR + llvm-spirv, HMMA to XMX; fuzz vs. real GPU oracle | 6-12+ person-months for kernels ATen/ggml-level; closed libs several more; plus legal exposure |

## Recommended first step
S1 (+S0): make the shim print an actionable one-time error on PTX-less modules and ship ptxscan as `scripts/ptxscan.py`. Then S3 for torch (PTX build artifacts) since that moves the biggest app (PyTorch cu126/cu128 users) with the least risk.

## Caveats
- ptxscan only reads fatbin entry headers (kind, arch); it does not distinguish "PTX that the shim can translate" from "PTX that exists". Windows/PE and Linux/ELF both work (scans for fatbin magic); compressed or encrypted (some obfuscated/--compress-all) fatbin layouts are still handled because only headers are read; cubins embedded NOT as nvcc fatbin (e.g. raw cubin blobs in cuBLAS's own packaging) are only flagged if they're ELF at file start.
- Web claims (CuLifter, NVLift, JEB, CuAssembler arch coverage) are from search summaries; EULA wording is quoted from memory of the CUDA Toolkit EULA sec. 1.2; verify against the current text before relying on it.

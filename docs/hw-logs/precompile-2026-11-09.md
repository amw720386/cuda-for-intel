# cfi-precompile on the Arc B390, 2026-11-09 (roadmap item 6)

Build: branch `precompile`, release, driver 32.0.101.9033.
- Stock Ollama v0.12.6, from an untouched tree, under `cfi-run` (redirect to the install's `nvcuda.dll` and cuBLAS shim). The
  companion DLL was set up as in [ollama.md](../ollama.md), with `OLLAMA_FLASH_ATTENTION=0` and `NEO_CACHE_PERSISTENT=0`.
- Model: Qwen2.5-0.5B-Instruct Q8_0.
- Script: `scripts/precompile-ttft.ps1` (GPU-queue slot held).
- "First request" is the time of the first `/api/generate` after the server starts. It covers the model load, a 307-token
  prefill and 1 generated token, and it is the cold first-token time. Every run uses its own fresh cache directories.

| Run | Caches before the run | First request (s) | of which load (s) | Second request (s) | Output |
| --- | --- | ---: | ---: | ---: | --- |
| cold | empty | 115.5 (112.0 in an earlier run) | 62.0 | 0.31 | " lorem" |
| warm | from the cold run | 1.64, 1.64, 1.89 (three runs) | 1.30-1.57 | 0.23 | same |
| **after `cfi-precompile`** | empty, then precompiled | **2.36** | 2.05 | 0.23 | same |
| precompiled, second start | from the precompiled run | 1.72 | 1.34 | 0.23 | same |

Precompile command (211 s wall on the box's 16 logical cores):

```
cfi-precompile --install-dir <install>\bin --gguf qwen.gguf --native-skip flash_attn <ollama>\lib\ollama\cuda_v13
skipped 2 NVIDIA libraries that cfi-run replaces (--all-files scans them)
ggml-cuda.dll: 117 fatbin frames, 114 new PTX modules
1 files, 114 PTX modules, 3307 entries (532 architecture-guard stubs skipped, 1551 selected for a native build)
translator: dynamic shared window 65536 dot product false; SPIR-V cache on
  rejected x1: expected Local address for %__t64_0, got Address(Global)
done in 206.0s: 3306 translated, 0 already in the SPIR-V cache, 1 rejected by the translator, 723 with module variables (not SPIR-V cached); native: 1548 built, 23 already cached, 0 failed, 10 substitute kernels built
cuBLAS XMX kernels: 8 built or cached in 4.7s
```

Cache sizes after the precompile: SPIR-V 2,583 files, 2,359 MB; native 1,566 files, 319 MB. A cold run leaves 23 SPIR-V
files (31 MB) and 29 native files (39 MB).

Findings:
- Almost all of the cold time is the Intel compiler. Four `mul_mat_q<Q8_0>` variants are about 2.1 million SPIR-V words each
  and compile in 23-30 s apiece (`ze_ms` in the export log). Every remaining kernel compiles in under 0.8 s.
- After the precompile, every module of the first request came from the native cache (`ze_ms` <= 17) and the substitute
  kernels were ready in 5 ms. The remaining 0.5-0.7 s over a warm start is in Ollama's model load. It was not traced further:
  the warm figures themselves vary by 0.25 s, and the cause may be the first read of fresh cache files.
- Result: 115.5 s down to 2.36 s, which is 1.25 to 1.44 times a warm start, against the roadmap's "within 20 %" criterion.
- A first attempt without a type selection was compiling all 3,307 entries. It had done 669 entries (1.4 GB) after 18
  minutes, when it was stopped. The mmq kernels of all 21 ggml types do not fit the 4 GB default native-cache cap, hence
  `--gguf`/`--ggml-types`.
- Bug found: the replacement for the stream-k fixup kernel has an empty SPIR-V module. Handing that to `zeModuleCreate` made the
  Intel driver print "invalid magic number" and exit the process with code 8. The precompiler now checks the SPIR-V magic.

# PTX -> SPIR-V translation cost of stock ggml-cuda, 2026-11-10

CPU-only measurements of the translator on the second test box (Arc 140V / Lunar Lake, see
`second-box-bringup-2026-11-10.md`). That box's GPU driver does not load, so nothing here touches Level Zero. The translator
is the same code on every GPU, so these numbers also hold for the Arc B390 box's cold start, scaled by its CPU.

## Setup

| item | value |
|---|---|
| CPU | Intel Core Ultra 7 268V, 8 cores / 8 threads, 31.6 GB RAM, Windows 11 23H2 (22631) |
| input | stock Ollama v0.12.6 `lib/ollama/cuda_v13/ggml-cuda.dll`, SHA-256 `4328F98A...` (114 PTX modules, 3,307 entries after 532 architecture-guard stubs) |
| command | `cfi-precompile --translate-only --quiet --report <tsv> <cuda_v13>` with an empty `CFI_CACHE_DIR`, default threads (8) |
| script | `scripts/profile/bench-translate.sh` (cold cache per run, wall time, digest of the cache contents) |

Every run translates 3,306 entries (1 rejected: `ssm_conv_f32`, "expected Local address ... got Address(Global)") into
2,369 MB of SPIR-V cache.

## Where the time goes (one thread)

Measured with `tests/ptx_corpus/harness --time <module.ptx> 1` and `CFI_PREP_TIMING=1` over the 117 PTX records that
`scan_stock_dll_translation` dumps from the same DLL (212 MB of PTX):

| phase | total | largest module (9.1 MB PTX, 64 entries) |
|---|---:|---:|
| `prepare_module` (once per module) | 28.5 s | 1.21 s |
| - of which `normalize::run` | 17.3 s | 0.71 s |
| - of which `rewrite_half_scopes` | 8.8 s | 0.44 s |
| parse + SPIR-V emission of every entry | 57.8 s | 1.62 s (slowest entry `mul_mat_q<q4_k,128>` 134 ms) |

About 86 s of single-thread work, but the 8-thread tool took 65-76 s, so only about 1.3x of eight cores were used.

## Finding 1: `--report` charged module preparation to entries

`translate_ms` started before the shared `prepare_module` call, so every entry that ran or waited for a module's prepare was
charged for it: tiny kernels such as `mul_mat_q_stream_k_fixup` showed about 1.76 s, and the column summed to about 425 s for
a 65 s run. With a separate `prepare_ms` column (fix/precompile-prepare-accounting) one run reads: `translate_ms` sum 132.9 s,
`prepare_ms` sum 46.9 s over 113 modules (higher than the one-thread figures because 8 threads share memory bandwidth).

## Finding 2: workers convoyed on each module's prepare

Jobs were sorted big-module-first, which kept each module's entries together: all workers reached a module at once and
waited for the one thread that prepared it. Moving the first entry of every module to the front (perf/precompile-interleave-prepare)
lets the modules prepare concurrently. Cold runs, alternating A/B, 8 threads:

| build | run 1 | run 2 | run 3 | median |
|---|---:|---:|---:|---:|
| main `11b70b5` | 76.0 s | 65.3 s | 65.9 s | 65.9 s |
| interleaved prepare | 43.6 s | 43.3 s | 43.2 s | 43.3 s (-34%) |

The SPIR-V cache digest was identical in all six runs (the translator output does not change). The first main run is
slower than the morning's 66.2 / 65.2 / 65.0 s baseline on the same build: this is a power-limited laptop part, which is why
the runs alternate.

## Next

The remaining 43 s are about 86 s of work on 8 cores plus writing 2.4 GB. The translator's own CPU work (`normalize::run`,
entry parse and emission) is the next target; it also shortens the runtime's lazy first-launch translation on both GPUs.

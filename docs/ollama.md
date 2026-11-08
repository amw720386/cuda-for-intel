# Running stock Ollama v0.12.6 on an Intel GPU

Verified setup: Windows x64, Intel Arc B390 (Xe3), unchanged `ollama.exe` and unchanged
`lib\ollama\cuda_v13\ggml-cuda.dll` from Ollama v0.12.6. Evidence: `docs/hw-logs/ollama-stock-v0.12.6.md`
(8 greedy tokens identical to the CPU backend) and `docs/hw-logs/stock-scan.md` (12 prompts GPU vs CPU).
Other Ollama versions and other GPUs are untested.

The quick way is `scripts/setup-windows.ps1` followed by `scripts/run-ollama.ps1`
(see below). The sections after that describe exactly what the scripts do.

## What you must supply yourself

This repository never contains third-party binaries and you should not redistribute them.

| File | Source | SHA-256 |
| --- | --- | --- |
| Ollama v0.12.6 (`ollama-windows-amd64.zip`, extracted) | the official [release page](https://github.com/ollama/ollama/releases/tag/v0.12.6) | `lib\ollama\cuda_v13\ggml-cuda.dll` must be `4328F98AAB69D9AF0F875210F769EBBB0854D65F783E9FF6B45E63F387F38E1A` |
| `nvcudart_hybrid64.dll` (NVIDIA-signed companion) | the ZLUDA repository at commit `f4df215edbc3b6698a1c61ef7a8688fdff3872be`, path `zluda/bin/nvcudart_hybrid64.dll` (a Git LFS object) | `BF430D65B863C49BAB525001A0712992D17F52F4B5A205536160A337CF8B22B3` |
| a GGUF model | for example Qwen2.5-0.5B-Instruct Q8_0 from Hugging Face | the verified run used model SHA-256 `CA59CA7F13D0E15A8CFA77BD17E65D24F6844B554A7B6C12E07A5F89FF76844E` |

Redistribution caveat: `nvcudart_hybrid64.dll` is NVIDIA proprietary software. It is not bundled in
this repository and must not be committed or attached to releases. Each user downloads it
themselves and is responsible for NVIDIA's license terms. Ollama's binaries are likewise obtained from Ollama.

## Why the companion DLL is needed

The stock CUDA 13 plug-in links the CUDA Runtime statically. At start-up that static runtime asks
the driver for private export tables (`cuGetExportTable`), among them `HYBRID_CUDART`, which hands
back a resolver living in NVIDIA's `nvcudart_hybrid64.dll`. The resolver is a closed-source
function table (523 entries) that the Runtime thunks call into; its layout and the integrity digest
that the Runtime checks are not published. This project therefore loads the genuine signed companion from an explicit
absolute directory (`CUDA_FOR_INTEL_HYBRID_DIR`) and returns its resolver, instead of faking the table.
The renamed 13.0.88 redistributable runtime does not have the required export, so it cannot replace the
companion. The driver accepts only the two file names `nvcudart_hybrid64.dll` and `nvcudart_hybrid64a.dll`.
Details: `docs/agent-handoff.md`, "The resolver and primary context".

Without the companion, `cuGetExportTable` fails with `CUDA_ERROR_FILE_NOT_FOUND` (301) when
`CUDA_FOR_INTEL_HYBRID_DIR` is unset, or `CUDA_ERROR_INVALID_VALUE` (1) when the path is not absolute
or the file name is not one of the two above.

## Automatic staging

```powershell
pwsh -File scripts\setup-windows.ps1 -TargetDir C:\cfi `
     -HybridDll D:\downloads\nvcudart_hybrid64.dll `
     -OllamaDir D:\downloads\ollama-windows-amd64
pwsh -File scripts\check-install.ps1 -TargetDir C:\cfi
pwsh -File scripts\run-ollama.ps1    -TargetDir C:\cfi
```

The setup script builds with `cargo build --release --locked`, refuses a companion or
`ggml-cuda.dll` whose SHA-256 differs from the pinned values (override with `-AllowUnknownHash`,
unsupported), copies the Ollama tree (your download is not modified) and patches the copy as described next.
The scripts run in Windows PowerShell 5.1 (`powershell -ExecutionPolicy Bypass -File ...`) as well as in
PowerShell 7 (`pwsh`). `cargo` must be on `PATH` (install Rust with the MSVC x64 toolchain first).

The staged Ollama listens on **127.0.0.1:11435**, not on Ollama's default 11434 (`setup-windows.ps1 -Port`
changes it). An installed Ollama (tray app or service) normally owns 11434, and a client that used that port
would talk to the other instance and never touch the Intel GPU. `run-ollama.ps1` refuses to start when the
port is busy. Keep the server window open and use a second shell for the client:

```powershell
. C:\cfi\cfi-env.ps1                       # sets PATH, OLLAMA_HOST, OLLAMA_MODELS, CUDA_FOR_INTEL_*
Set-Content Modelfile 'FROM D:\models\qwen2.5-0.5b-instruct-q8_0.gguf'
ollama-cfi.exe create mymodel -f Modelfile  # register the GGUF once
ollama-cfi.exe run mymodel "Say hi."
# or the HTTP API:
Invoke-RestMethod http://127.0.0.1:11435/api/generate -Method Post -ContentType application/json `
  -Body '{"model":"mymodel","prompt":"Say hi.","stream":false,"options":{"num_predict":8,"temperature":0}}'
```

The first request JIT-compiles kernels and can take minutes (the compiled kernels are cached in
`C:\cfi\kernel-cache`). To stop the server press Ctrl+C in its window. The server and its runner processes
are named `ollama-cfi.exe`, so stop them by that name (`Stop-Process -Name ollama-cfi`); that never touches an
installed Ollama.

## Manual staging (what the script does)

Build: `cargo build --release --locked -p cuda-for-intel -p cublas_shim`. Outputs are
`target\release\nvcuda.dll` and `target\release\cublas64_12.dll`.

1. Copy the Ollama tree to a private directory (keep the original).
2. In `lib\ollama\cuda_v13\`:
   - rename the original `cublas64_13.dll` to `cublas64_13.dll.orig` and install the built `cublas64_12.dll`
     under the name `cublas64_13.dll`. The one shim DLL exports both the cuBLAS and the cuBLASLt entry points.
   - move `cublasLt64_13.dll` aside (the verified run removed it).
   - copy the built `nvcuda.dll` next to `ggml-cuda.dll`. The shim imports `cuCtxGetCurrent`, so Windows maps
     this driver before the plug-in's CUDA constructors run.
3. Place `nvcudart_hybrid64.dll` in a directory of its own and set `CUDA_FOR_INTEL_HYBRID_DIR` to its absolute path.
4. Environment, as used in the verified run:

   | Variable | Value | Reason |
   | --- | --- | --- |
   | `CUDA_FOR_INTEL_HYBRID_DIR` | absolute path of the companion directory | see above |
   | `OLLAMA_LIBRARY_PATH` | `<ollama>\lib\ollama;<ollama>\lib\ollama\cuda_v13` | make the runner load the patched plug-in |
   | `PATH` | `<ollama>`, `<ollama>\lib\ollama`, `<ollama>\lib\ollama\cuda_v13` first | DLL resolution |
   | `OLLAMA_HOST` | `127.0.0.1:11435` | not Ollama's default port, which an installed Ollama usually owns |
   | `OLLAMA_MODELS` | `<target>\models` | private model store |
   | `CUDA_FOR_INTEL_CACHE_DIR` | `<target>\kernel-cache` | reuse compiled kernels between runs |
   | `OLLAMA_VULKAN` | `0` | keep Ollama's own Vulkan backend from taking the GPU |
   | `OLLAMA_FLASH_ATTENTION` | `0` | flash-attention kernels do not all translate yet (`docs/hw-logs/stock-scan.md`) |
   | `OLLAMA_NUM_PARALLEL` | `1` | as in the verified run |
   | `OLLAMA_LOAD_TIMEOUT` | `60m` | the first load JIT-compiles large kernels, minutes |

5. Choose a port nobody else uses (`OLLAMA_HOST=127.0.0.1:11435`) and run `ollama.exe serve` from a copy named
   `ollama-cfi.exe`, so it neither collides with nor can be confused with an installed Ollama.
   Creating a model from a GGUF and calling `/api/generate` worked in the verified run
   with options `num_gpu 99`, `num_ctx 128`, `num_batch 1`, `temperature 0`.

## What was verified

- Qwen2.5-0.5B-Instruct Q8_0, prompt `Say hi.`, raw, temperature 0, 8 tokens: identical text on GPU and with `num_gpu 0`.
- 12 prompts, greedy, 24 tokens: 10 identical. Two open-ended prompts diverge (`docs/hw-logs/stock-scan.md`):
  attributed in the log to float reordering at near-ties, not confirmed by a logprob comparison.
- Other models, longer generation and multi-user serving: not verified.

## Performance expectations

First use compiles kernels with the Intel compiler, minutes for the large quantised matrix-multiply kernels.
Set `CUDA_FOR_INTEL_CACHE_DIR` (or leave the default, `%LOCALAPPDATA%\cuda-for-intel`) so later processes
reuse them; the verified run took 12.1 s to load and 1.06 s for 8 tokens. With logging off the logs recorded 30 to 46 tok/s
decode on the GPU against 119 tok/s on the CPU backend (`docs/hw-logs/perf-matrix.md`). The GPU path is not currently faster than the CPU
for this model; see "Performance guidance" in the README.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Crash with `0xc0000005` during plug-in registration | wrong or missing companion (null resolver); check the hash and `CUDA_FOR_INTEL_HYBRID_DIR` |
| `/info` shows no CUDA device | `nvcuda.dll` not beside `ggml-cuda.dll`, or `OLLAMA_LIBRARY_PATH` not set |
| Load never finishes | JIT compile of the first large kernels; wait, watch the CPU, set `OLLAMA_LOAD_TIMEOUT=60m` |
| `no kernel image` / error 209 | an architecture-guard stub kernel; see README, Troubleshooting |
| Different `ggml-cuda.dll` hash | another Ollama version; the stock and llama.cpp GGML ABIs differ, keep each `ggml-base.dll` with its own plug-in |

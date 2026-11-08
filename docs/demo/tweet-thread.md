# Tweet thread draft

Attach: 1 = docs/demo/pipeline.svg screen-recorded or architecture.png, 3 = architecture.png, 4 = prefill-chart.png.

1/ 🟦 CUDA apps on an Intel Arc GPU, no NVIDIA card. A Rust shim implements the CUDA Driver API, turns the PTX inside CUDA programs into SPIR-V and runs it through Level Zero. Stock Ollama is generating tokens on an Arc B390 this way. 🧵

2/ 🔧 How it works: the app and NVIDIA's own CUDA runtime stay unchanged. Our nvcuda.dll replacement catches the Driver API calls, reads the PTX, translates it to SPIR-V and launches it on the GPU through Level Zero.

3/ 🧩 Architecture: app -> CUDA runtime -> Rust nvcuda shim -> PTX parser -> PTX-to-SPIR-V translator -> Level Zero -> Arc. Plus a kernel substitution layer, on by default: a few hot ggml kernels get swapped for hand-written Intel-native SPIR-V, with a fallback to the translated kernel if anything mismatches.

4/ ⚡ What substitution does to prompt processing (prefill tok/s, Arc B390, stock Ollama v0.12.6, 631-token prompt, flash attention off), off -> on:
Qwen2.5-0.5B Q8_0: 1818 -> 6767 (3.7x)
Llama-3.2-1B Q4_K_M: 817 -> 4877 (6.0x)
Qwen2.5-1.5B Q4_K_M: 627 -> 3727 (5.9x)

5/ 🔍 Honest scope: experimental, Windows, PTX-only programs (no SASS), tested on Arc B390. Flash attention is being fixed, so it stays off for now. These are prefill numbers only.

6/ 🛠️ Code, docs and the compatibility matrix with logs for every claim: https://github.com/amw720386/cuda-for-intel ⭐

## Alt text
- architecture.png: Dark diagram of eight stacked boxes from top to bottom: CUDA app, genuine CUDA runtime, Rust nvcuda shim, PTX parser, PTX to SPIR-V translator, kernel substitution layer, Level Zero, Intel Arc GPU.
- prefill-chart.png: Bar chart of prefill tokens per second with kernel substitution off versus on. Qwen2.5-0.5B Q8_0: 1818 to 6767. Llama-3.2-1B Q4_K_M: 817 to 4877. Qwen2.5-1.5B Q4_K_M: 627 to 3727.
- pipeline.svg: Animation of a blue PTX packet travelling from the CUDA app through the shim to the translator, turning green as SPIR-V, then passing Level Zero to the Arc GPU.

# Copy to <CFI_WORK>\hwval\config.ps1 on the box and edit. Read by run_all.ps1 (dot-sourced); never committed with real values.
# Paths are relative to the box user's profile via $env:USERPROFILE so nothing machine-specific lives in the repo.
$w = Join-Path $env:USERPROFILE 'cfi-work'
$env:CFI_OLLAMA_RUN      = "$w\stock\runcfi"                       # staged stock Ollama tree (ollama.exe + lib\ollama\cuda_v13)
$env:CFI_MODELS          = "$w\stock\qwen.gguf,$w\stock\llama32-1b-q4km.gguf,$w\stock\third-model.gguf"   # 3 GGUFs
$env:CFI_F16_MODEL       = "$w\stock\f16-model.gguf"               # f16/bf16 GGUF for the XMX A/B (optional)
$env:CFI_HYBRID_DIR      = "$w\stock\hybrid"                       # dir with nvcudart_hybrid64.dll
$env:CFI_STOCK_GGML_DLL  = "$w\stock\runcfi\lib\ollama\cuda_v13\ggml-cuda.dll"
$env:CFI_CORPUS_DIR      = "$w\corpus"                             # PTX corpus subset (optional)
$env:CFI_TORCH_PY        = "$w\torch\torch222\Scripts\python.exe"  # torch venv python (optional)

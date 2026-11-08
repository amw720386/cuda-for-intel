# short slot: no-workspace conv1 with full tracing
$w = Join-Path $env:USERPROFILE 'cfi-work'; $d = "$w\cd3"; $g = "$d\gate"; $run = "$d\run"
$py = "$w\torch\torch222\Scripts\python.exe"
& "$w\gpu-queue.ps1" -Action acquire -Id cudnn3-dbg2 -MaxWaitSec 14400
try {
  $env:CFI_BENCH_ROOT = $run; $env:CFI_BENCH_BATCH = '2'; $env:CFI_CUDNN_NO_WS = '1'; $env:CFI_CUDNN_TRACE = '1'; $env:CUDA_FOR_INTEL_TRACE = '1'
  & $py "$d\bench_conv.py" cudnn "$g\nowsdbg.jsonl" "r18_conv1_7x7s2|fp32|nchw" > "$g\nowsdbg.out" 2> "$g\nowsdbg.err"
  "rc=$LASTEXITCODE"
} finally { & "$w\gpu-queue.ps1" -Action release -Id cudnn3-dbg2 }

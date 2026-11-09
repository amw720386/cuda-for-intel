# Hardware gate in ONE queue slot: ResNet18 (112 and 224 px, batch 4, cudnn enabled), direct-path repro,
# then the ignored GPU suite. Logs go to cd3\gate. Usage: gate.ps1 [-SkipSuite]
param([switch]$SkipSuite)
$w = Join-Path $env:USERPROFILE 'cfi-work'; $d = "$w\cd3"; $g = "$d\gate"; $run = "$d\run"
$py = "$w\torch\torch222\Scripts\python.exe"
New-Item -ItemType Directory -Force $g | Out-Null
& "$w\gpu-queue.ps1" -Action acquire -Id cudnn3-gate -MaxWaitSec 14400
$lock = "$w\gpu.lock"
$hb = Start-Job -ScriptBlock { param($l) while ($true) { try { (Get-Item $l).LastWriteTime = Get-Date } catch {}; Start-Sleep 45 } } -ArgumentList $lock
try {
  $env:CFI_BENCH_ROOT = $run
  foreach ($sz in 112, 224) {
    $env:CFI_BENCH_BATCH = '4'; $env:CFI_RESNET_SIZE = "$sz"
    foreach ($m in 'cudnn', 'aten') {
      $out = "$g\resnet_${m}_$sz"
      & $py "$d\test_resnet18.py" $m "$out.json" > "$out.out" 2> "$out.err"
      "resnet $m $sz rc=$LASTEXITCODE" | Tee-Object -Append "$g\summary.txt"
    }
  }
  # large shapes with NO workspace advertised: callers pass none, the shim must use its internal scratch (formerly 'no engine')
  $env:CFI_BENCH_BATCH = '8'; $env:CFI_CUDNN_NO_WS = '1'
  & $py "$d\bench_conv.py" cudnn "$g\nows_big.jsonl" "fp32|nchw" > "$g\nows_big.out" 2> "$g\nows_big.err"
  "no-workspace sweep (fp32 nchw) rc=$LASTEXITCODE" | Tee-Object -Append "$g\summary.txt"
  Remove-Item Env:CFI_CUDNN_NO_WS -EA 0
  # default path, same shapes
  & $py "$d\bench_conv.py" cudnn "$g\cudnn_big.jsonl" "fp32|nchw" > "$g\cudnn_big.out" 2> "$g\cudnn_big.err"
  "default sweep (fp32 nchw) rc=$LASTEXITCODE" | Tee-Object -Append "$g\summary.txt"
  if (-not $SkipSuite) {
    Set-Location "$d\src"
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"; $env:CARGO_TARGET_DIR = "$d\target"
    $env:CUDA_FOR_INTEL_HYBRID_DIR = "$run\hybrid"
    $env:CFI_HW_WATCHDOG_S = '600'
    cargo test --release -j4 --test gpu -- --ignored --test-threads=1 --skip hw_nvrtc > "$g\gpu_suite.out" 2> "$g\gpu_suite.err"
    "gpu suite rc=$LASTEXITCODE" | Tee-Object -Append "$g\summary.txt"
  }
} finally {
  Stop-Job $hb -EA 0; Remove-Job $hb -Force -EA 0
  & "$w\gpu-queue.ps1" -Action release -Id cudnn3-gate
}
'done' | Out-File "$g\done.txt"

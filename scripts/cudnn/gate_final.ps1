# Final cuDNN hardware gate in ONE queue slot. Everything runs to exit by itself (timeouts, no process killing needed).
# Prereq: work dir $env:USERPROFILE\cfi-work\cd4 with src\ (this repo), run\ (nvcuda.dll, cudnn64_8.dll built with --features v8, hybrid\)
#         and stock\ggml-cuda.dll path in $env:CFI_STOCK_GGML_DLL. Logs: cd4\gate.   Usage: gate_final.ps1 [-SkipSuite]
param([switch]$SkipSuite, [switch]$NoQueue, [string]$Dir = '', [string]$Gate = '')
$w = Join-Path $env:USERPROFILE 'cfi-work'; $d = if ($Dir) { $Dir } else { "$w\cd4" }; $g = if ($Gate) { $Gate } else { "$d\gate" }; $run = "$d\run"; $src = "$d\src\scripts\cudnn"
$py = "$w\torch\torch222\Scripts\python.exe"
New-Item -ItemType Directory -Force $g | Out-Null
if (-not $NoQueue) { & "$w\gpu-queue.ps1" -Action acquire -Id cudnn4-gate -MaxWaitSec 14400 }
$lock = "$w\gpu.lock"
$hb = if ($NoQueue) { $null } else { Start-Job -ScriptBlock { param($l) while ($true) { try { (Get-Item $l).LastWriteTime = Get-Date } catch {}; Start-Sleep 45 } } -ArgumentList $lock }
function Step($name, $exe, $argv, $out, $timeoutMin) {
  $p = Start-Process $exe -ArgumentList $argv -RedirectStandardOutput "$out.out" -RedirectStandardError "$out.err" -PassThru -NoNewWindow
  if (-not $p.WaitForExit($timeoutMin * 60000)) { "$name TIMEOUT (left to finish on its own)" | Tee-Object -Append "$g\summary.txt"; return }
  "$name rc=$($p.ExitCode)" | Tee-Object -Append "$g\summary.txt"
}
try {
  $env:CFI_BENCH_ROOT = $run
  foreach ($m in 'helper', 'raw', 'off') { Step "callers $m" $py "`"$src\test_callers.py`" $m `"$g\callers_$m.json`"" "$g\callers_$m" 40 }
  $env:CFI_BENCH_BATCH = '4'
  foreach ($sz in 112, 224) {
    $env:CFI_RESNET_SIZE = "$sz"
    foreach ($m in 'cudnn', 'aten') { Step "resnet18 $m $sz" $py "`"$src\test_resnet18.py`" $m `"$g\resnet_${m}_$sz.json`"" "$g\resnet_${m}_$sz" 20 }
  }
  $env:CFI_BENCH_BATCH = '8'
  foreach ($m in 'aten', 'cudnn') { Step "bench $m" $py "`"$src\bench_conv.py`" $m `"$g\bench_$m.jsonl`"" "$g\bench_$m" 40 }
  $env:CFI_CUDNN_NO_WS = '1'
  Step "no-workspace sweep" $py "`"$src\bench_conv.py`" cudnn `"$g\nows.jsonl`" fp32|nchw" "$g\nows" 25
  Remove-Item Env:CFI_CUDNN_NO_WS -EA 0
  if (-not $SkipSuite) {
    Set-Location "$d\src"
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"; $env:CARGO_TARGET_DIR = "$d\target"
    $env:CUDA_FOR_INTEL_HYBRID_DIR = "$run\hybrid"; $env:CFI_HW_WATCHDOG_S = '900'
    cargo test --release -j4 --test gpu -- --ignored --test-threads=1 --skip hw_stock_ggml_kernels_compile > "$g\gpu_suite.out" 2> "$g\gpu_suite.err"
    "gpu suite rc=$LASTEXITCODE" | Tee-Object -Append "$g\summary.txt"
  }
} finally {
  if ($hb) { Stop-Job $hb -EA 0; Remove-Job $hb -Force -EA 0 }
  if (-not $NoQueue) { & "$w\gpu-queue.ps1" -Action release -Id cudnn4-gate }
}
'done' | Out-File "$g\done.txt"

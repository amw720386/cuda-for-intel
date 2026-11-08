$w='C:\Users\USER\cfi-work'; $d="$w\cd3"; $g="$d\gate"
& $w\gpu-queue.ps1 -Action acquire -Id cudnn3-hw -MaxWaitSec 14400
$lock="$w\gpu.lock"
$hb = Start-Job -ScriptBlock { param($l) while ($true) { try { (Get-Item $l).LastWriteTime = Get-Date } catch {}; Start-Sleep 45 } } -ArgumentList $lock
try {
  Set-Location "$d\src"
  $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"; $env:CARGO_TARGET_DIR = "$d\target"
  $env:CUDA_FOR_INTEL_HYBRID_DIR = "$d\run\hybrid"; $env:CFI_HW_WATCHDOG_S='90'; $env:CFI_CUDNN_TRACE='1'
  cargo test --release -j4 --test gpu hw_cudnn -- --ignored --nocapture --test-threads=1 > "$g\hwc.out" 2> "$g\hwc.err"
  "hw_cudnn rc=$LASTEXITCODE"
} finally { Stop-Job $hb -EA 0; Remove-Job $hb -Force -EA 0; & $w\gpu-queue.ps1 -Action release -Id cudnn3-hw }

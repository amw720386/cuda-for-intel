<#
.SYNOPSIS
  CUDA VMM hardware gate (docs/vmm.md, gpu-day-checklist item 9). Runs ON a Windows GPU box.

.DESCRIPTION
  Runs tests/gpu/vmm.rs (GGML pool growth, kernel across chunks, copies across the chunk
  boundary, physical aliasing, teardown) once per Level Zero device ordinal, three times
  each, and writes the `VMM:` summary lines plus the full output to <Out>\vmm-<ordinal>.log.
  It does not change any default: CUDA_FOR_INTEL_VMM stays unset here (the test calls the
  entry points directly).

.EXAMPLE
  powershell -NoProfile -File scripts\hwvalidate\vmm.ps1 -Ordinals 0 -Out C:\cfi-out
#>
[CmdletBinding()]
param(
  [int[]]$Ordinals = @(0),
  [int]$Repeats = 3,
  [string]$Out = '.\vmm-out'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
New-Item -ItemType Directory -Force $Out | Out-Null
Push-Location $repo
try {
  cargo build --release --locked -j 2 -p cuda-for-intel
  if ($LASTEXITCODE -ne 0) { throw 'build failed' }
  $failed = 0
  foreach ($ordinal in $Ordinals) {
    $log = Join-Path $Out "vmm-$ordinal.log"
    Remove-Item -ErrorAction SilentlyContinue $log
    $env:CUDA_FOR_INTEL_DEVICE = "$ordinal"
    for ($i = 1; $i -le $Repeats; $i++) {
      $text = cargo test --release --locked -j 2 --test gpu vmm_pool_growth -- --ignored --nocapture --test-threads=1 2>&1 | Out-String
      $code = $LASTEXITCODE
      Add-Content $log "=== ordinal $ordinal run $i exit $code ===`n$text"
      $line = ($text -split "`n" | Where-Object { $_ -match '^VMM:' }) -join ' '
      Write-Host "ordinal $ordinal run ${i}: exit $code $line"
      if ($code -ne 0) { $failed++ }
    }
  }
  Remove-Item Env:CUDA_FOR_INTEL_DEVICE -ErrorAction SilentlyContinue
  if ($failed) { Write-Host "VMM GATE: FAIL ($failed runs)"; exit 1 }
  Write-Host 'VMM GATE: PASS'
} finally {
  Pop-Location
}

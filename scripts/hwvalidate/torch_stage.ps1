<#
.SYNOPSIS
  Torch ladder (scripts/torch/progressive.py, levels L0-L7) as a sub-step of run_all.ps1 stage 9. The caller already holds
  the GPU-queue slot, so this script takes no lock. Nothing is killed: the harness gives every level its own timeout and the
  outer wait below is only a backstop that reports a timeout.
.PARAMETER Work     cfi-work dir
.PARAMETER Src      repo root (the synced tree)
.PARAMETER Results  run_all results dir (output goes to <Results>\torch)
.PARAMETER Levels   comma separated levels (default all)
#>
param([Parameter(Mandatory)][string]$Work, [Parameter(Mandatory)][string]$Src, [Parameter(Mandatory)][string]$Results,
      [string]$Levels = 'L0,L1,L2,L3,L4,L5,L6,L7', [int]$LevelTimeoutSec = 900)
$ErrorActionPreference = 'Continue'
$py = if ($env:CFI_TORCH_PY) { $env:CFI_TORCH_PY } else { Join-Path $Work 'torch\torch222\Scripts\python.exe' }
if (-not (Test-Path $py)) { Write-Host 'no torch python (set CFI_TORCH_PY)'; exit 0 }
$nv = Join-Path $Work 'hwval\target\release\nvcuda.dll'
$out = Join-Path $Results 'torch'; New-Item -ItemType Directory -Force $out | Out-Null
$env:CUDA_MODULE_LOADING = 'LAZY'; $env:CFI_NVCUDA = $nv
if ($env:CFI_HYBRID_DIR) { $env:CUDA_FOR_INTEL_HYBRID_DIR = $env:CFI_HYBRID_DIR }
Remove-Item Env:CFI_FAKE_CPU -ErrorAction SilentlyContinue
$n = ($Levels -split ',').Count
$outer = $LevelTimeoutSec * ($n + 4) * 3 + 600
$p = Start-Process -PassThru -NoNewWindow -FilePath $py -WorkingDirectory $Src -ArgumentList @((Join-Path $Src 'scripts\torch\progressive.py'), '--levels', $Levels, '--timeout', $LevelTimeoutSec, '--out', $out) -RedirectStandardOutput (Join-Path $out 'stdout.txt') -RedirectStandardError (Join-Path $out 'stderr.txt')
$null = $p.Handle   # without touching the handle before exit, ExitCode reads back null and every run looked like rc=0
if (-not $p.WaitForExit($outer * 1000)) { Write-Host 'outer timeout: harness still running, not killed'; exit 3 }
$p.WaitForExit()
exit $p.ExitCode

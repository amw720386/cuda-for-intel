<#
.SYNOPSIS
  Runs scripts/torch/progressive.py on the shared GPU: takes the gpu-queue slot, keeps a lock heartbeat,
  releases the slot in finally. Results: <Out>\L*.json, results.json, matrix.md (paste into docs/pytorch-status.md).
.PARAMETER Work     work dir that holds gpu-queue.ps1 and gpu.lock (the queue's directory)
.PARAMETER Python   python.exe of the venv with torch 2.2.2+cu118 (+ transformers for L6)
.PARAMETER Nvcuda   full path of the built nvcuda.dll (loaded before torch)
.PARAMETER Hybrid   directory with nvcudart_hybrid64.dll (CUDA_FOR_INTEL_HYBRID_DIR)
.PARAMETER Out      output directory (default: <Work>\torch-progressive)
.PARAMETER Model    optional local HF model directory for L6 (GPT-2 or Qwen2.5-0.5B)
.PARAMETER Levels   comma separated levels (default all)
.PARAMETER LevelTimeoutSec  per-level timeout enforced by the harness (default 900; L6 gets 4x)
.PARAMETER Profile  set CFI_PROFILE=1 and write <Out>\cfi-profile.tsv (summarise with scripts/profile/summarize.py)
#>
param([Parameter(Mandatory)][string]$Work, [Parameter(Mandatory)][string]$Python, [Parameter(Mandatory)][string]$Nvcuda,
      [Parameter(Mandatory)][string]$Hybrid, [string]$Out = '', [string]$Model = '', [string]$Levels = 'L0,L1,L2,L3,L4,L5,L6,L7',
      [int]$LevelTimeoutSec = 900, [string]$Id = 'torch-progressive', [int]$MaxWaitSec = 7200, [switch]$Profile)
$ErrorActionPreference = 'Continue'
$queue = Join-Path $Work 'gpu-queue.ps1'
$lock  = Join-Path $Work 'gpu.lock'
if (-not $Out) { $Out = Join-Path $Work 'torch-progressive' }
New-Item -ItemType Directory -Force $Out | Out-Null
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$script = Join-Path $PSScriptRoot 'progressive.py'

& $queue -Action acquire -Id $Id -MaxWaitSec $MaxWaitSec
"slot acquired $(Get-Date -f s)" | Out-File (Join-Path $Out 'progress.txt')
$hb = Start-Job -ArgumentList $lock -ScriptBlock { param($l) while ($true) { try { (Get-Item $l).LastWriteTime = Get-Date } catch {}; Start-Sleep 45 } }
$rc = 1
try {
  $env:CUDA_MODULE_LOADING = 'LAZY'
  $env:CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid
  $env:CFI_NVCUDA = $Nvcuda
  if ($Model) { $env:CFI_HF_MODEL = $Model }
  if ($Profile) { $env:CFI_PROFILE = '1'; $env:CFI_PROFILE_OUT = (Join-Path $Out 'cfi-profile.tsv') }
  Remove-Item Env:CFI_FAKE_CPU -ErrorAction SilentlyContinue
  # The harness runs each level in a child process with its own timeout; this outer limit is a backstop.
  $nLevels = ($Levels -split ',').Count
  $outer = $LevelTimeoutSec * ($nLevels + 4) * 3 + 600
  $p = Start-Process -PassThru -NoNewWindow -FilePath $Python -WorkingDirectory $repo `
        -ArgumentList @($script, '--levels', $Levels, '--timeout', $LevelTimeoutSec, '--out', $Out) `
        -RedirectStandardOutput (Join-Path $Out 'stdout.txt') -RedirectStandardError (Join-Path $Out 'stderr.txt')
  if (-not $p.WaitForExit($outer * 1000)) { Stop-Process -Id $p.Id -Force; "outer timeout" | Out-File -Append (Join-Path $Out 'progress.txt') }
  else { $rc = $p.ExitCode }
  "harness done rc=$rc $(Get-Date -f s)" | Out-File -Append (Join-Path $Out 'progress.txt')
} finally {
  Stop-Job $hb -ErrorAction SilentlyContinue
  if ($p -and -not $p.HasExited) { & taskkill /PID $p.Id /T /F 2>&1 | Out-Null }
  & $queue -Action release -Id $Id
  "released $(Get-Date -f s)" | Out-File -Append (Join-Path $Out 'progress.txt')
}
exit $rc

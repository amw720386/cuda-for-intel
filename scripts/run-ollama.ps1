<#
.SYNOPSIS
  Starts the staged stock Ollama v0.12.6 on an Intel GPU through CUDA for Intel.

.DESCRIPTION
  Requires a tree staged by scripts/setup-windows.ps1 with -OllamaDir and -HybridDll.
  Dot-sources <TargetDir>\cfi-env.ps1 (PATH, OLLAMA_LIBRARY_PATH, CUDA_FOR_INTEL_HYBRID_DIR, ...),
  then runs a renamed copy of ollama.exe (ollama-cfi.exe). The copy keeps this instance apart
  from an installed Ollama (tray app or service) that would otherwise own the same process name
  and port. The first GPU request JIT-compiles large kernels and can take minutes
  (the compiled kernels are cached afterwards).

.PARAMETER TargetDir
  The directory given to setup-windows.ps1.

.PARAMETER Arguments
  Arguments for ollama.exe. Default: serve.

.PARAMETER Trace
  Sets CUDA_FOR_INTEL_TRACE=1 (API trace on stderr).

.EXAMPLE
  pwsh -File scripts\run-ollama.ps1 -TargetDir C:\cfi
  # in a second shell, with the same environment:
  #   . C:\cfi\cfi-env.ps1; ollama-cfi.exe run <model>
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$TargetDir,
    [string[]]$Arguments = @('serve'),
    [switch]$Trace
)

$ErrorActionPreference = 'Stop'

$target = (Resolve-Path -LiteralPath $TargetDir).Path
$envFile = Join-Path $target 'cfi-env.ps1'
$ollamaDir = Join-Path $target 'ollama'
if (-not (Test-Path -LiteralPath $envFile -PathType Leaf) -or -not (Test-Path -LiteralPath $ollamaDir -PathType Container)) {
    throw "No staged Ollama in $target. Run scripts\setup-windows.ps1 -TargetDir ... -OllamaDir ... -HybridDll ... first."
}

. $envFile

if (-not $env:CUDA_FOR_INTEL_HYBRID_DIR -or -not (Test-Path -LiteralPath (Join-Path $env:CUDA_FOR_INTEL_HYBRID_DIR 'nvcudart_hybrid64.dll'))) {
    throw 'CUDA_FOR_INTEL_HYBRID_DIR does not contain nvcudart_hybrid64.dll.'
}
if ($Trace) { $env:CUDA_FOR_INTEL_TRACE = '1' }

$running = Get-Process -Name 'ollama', 'ollama app' -ErrorAction SilentlyContinue
if ($running) {
    Write-Warning 'An Ollama process is already running. Quit it (tray icon, Exit) so port 11434 is free for this instance.'
}

$exe = Join-Path $ollamaDir 'ollama-cfi.exe'
Copy-Item -LiteralPath (Join-Path $ollamaDir 'ollama.exe') -Destination $exe -Force

Write-Host "Starting $exe $($Arguments -join ' ')"
Write-Host "CUDA_FOR_INTEL_HYBRID_DIR=$env:CUDA_FOR_INTEL_HYBRID_DIR"
Write-Host "OLLAMA_LIBRARY_PATH=$env:OLLAMA_LIBRARY_PATH"
& $exe @Arguments
exit $LASTEXITCODE

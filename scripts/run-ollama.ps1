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

# Refuse to start when the port is taken: otherwise `ollama run` and the HTTP API would quietly
# talk to the other Ollama instance (an installed tray app or service) and the GPU is never used.
$hostPort = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { '127.0.0.1:11434' }
$portNumber = [int]($hostPort -replace '^.*:', '')
$busy = $false
try {
    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $portNumber)
    $listener.Start()
    $listener.Stop()
} catch {
    $busy = $true
}
if ($busy) {
    throw "Port $portNumber ($hostPort) is already in use, probably by an installed Ollama. Stop that instance, or re-run scripts\setup-windows.ps1 with another -Port."
}

$exe = Join-Path $ollamaDir 'ollama-cfi.exe'
if (-not (Get-Process -Name 'ollama-cfi' -ErrorAction SilentlyContinue)) {
    Copy-Item -LiteralPath (Join-Path $ollamaDir 'ollama.exe') -Destination $exe -Force
}

Write-Host "Starting $exe $($Arguments -join ' ')"
Write-Host "CUDA_FOR_INTEL_HYBRID_DIR=$env:CUDA_FOR_INTEL_HYBRID_DIR"
Write-Host "OLLAMA_LIBRARY_PATH=$env:OLLAMA_LIBRARY_PATH"
& $exe @Arguments
exit $LASTEXITCODE

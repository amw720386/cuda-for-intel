<#
.SYNOPSIS
  Builds CUDA for Intel and stages it into a target directory (Windows x64).

.DESCRIPTION
  1. Builds the DLLs with `cargo build --release --locked`.
  2. Copies them into -TargetDir under the file names applications expect.
  3. Optionally stages a stock Ollama v0.12.6 tree (-OllamaDir) and the NVIDIA-signed
     nvcudart_hybrid64.dll companion (-HybridDll). Both are third-party files that this
     repository never contains: you supply them, and the script verifies their SHA-256.
  4. Writes <TargetDir>\cfi-env.ps1, which sets the environment variables the launchers use.

  Nothing outside -TargetDir is modified. The Ollama tree is copied, not changed in place.

.PARAMETER TargetDir
  Directory that receives the staged files. Created if missing.

.PARAMETER HybridDll
  Path to the signed nvcudart_hybrid64.dll (obtain it yourself; see docs/ollama.md).
  Required for stock Ollama, genuine NVIDIA CUDA Runtime 13.x programs and PyTorch.

.PARAMETER OllamaDir
  Extracted ollama-windows-amd64.zip of Ollama v0.12.6 (the folder holding ollama.exe).
  Copied to <TargetDir>\ollama and patched there.

.PARAMETER SkipBuild
  Use the DLLs already in <repo>\target\release.

.PARAMETER WithCudartShim
  Also build and stage cudart64_12.dll (the project's own minimal CUDA Runtime, used by the
  controlled llama.cpp bring-up builds; not needed with the genuine NVIDIA runtime).

.PARAMETER AllowUnknownHash
  Continue (with a warning) when a third-party file does not match the pinned SHA-256.
  Off by default: an unexpected hash usually means an unsupported Ollama version.

.EXAMPLE
  pwsh -File scripts\setup-windows.ps1 -TargetDir C:\cfi -HybridDll D:\dl\nvcudart_hybrid64.dll `
       -OllamaDir D:\dl\ollama-windows-amd64
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$TargetDir,
    [string]$HybridDll,
    [string]$OllamaDir,
    [switch]$SkipBuild,
    [switch]$WithCudartShim,
    [switch]$AllowUnknownHash
)

$ErrorActionPreference = 'Stop'

# Pinned third-party hashes (see docs/ollama.md). The files themselves are NVIDIA / Ollama property.
$HybridSha256 = 'BF430D65B863C49BAB525001A0712992D17F52F4B5A205536160A337CF8B22B3'
$OllamaCudaV13Sha256 = '4328F98AAB69D9AF0F875210F769EBBB0854D65F783E9FF6B45E63F387F38E1A'

$repo = Split-Path -Parent $PSScriptRoot

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Assert-Hash([string]$Path, [string]$Expected, [string]$What) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$What not found: $Path" }
    $actual = Get-Sha256 $Path
    if ($actual -eq $Expected) {
        Write-Host "OK    $What SHA-256 matches ($Expected)"
        return
    }
    $message = "$What SHA-256 is $actual, expected $Expected"
    if ($AllowUnknownHash) { Write-Warning $message; return }
    throw "$message. Use the exact file listed in docs/ollama.md, or pass -AllowUnknownHash at your own risk."
}

# ---------------------------------------------------------------- build
if (-not $SkipBuild) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'cargo not found. Install Rust (https://rustup.rs) with the MSVC x64 toolchain.'
    }
    $packages = @('-p', 'cuda-for-intel', '-p', 'cublas_shim')
    if ($WithCudartShim) { $packages += @('-p', 'cudart_shim') }
    Write-Host "cargo build --release --locked $($packages -join ' ')"
    Push-Location $repo
    try {
        & cargo build --release --locked @packages
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}

$release = Join-Path $repo 'target\release'
$nvcuda = Join-Path $release 'nvcuda.dll'
$cublas = Join-Path $release 'cublas64_12.dll'
$cudart = Join-Path $release 'cudart64_12.dll'
foreach ($required in @($nvcuda, $cublas)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing build output: $required"
    }
}
if ($WithCudartShim -and -not (Test-Path -LiteralPath $cudart -PathType Leaf)) {
    throw "Missing build output: $cudart"
}

# ---------------------------------------------------------------- stage
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
$target = (Resolve-Path -LiteralPath $TargetDir).Path
$bin = Join-Path $target 'bin'
New-Item -ItemType Directory -Force -Path $bin | Out-Null

# One cublas shim DLL exports both the cublas and the cublasLt names, so it is copied under both.
Copy-Item -LiteralPath $nvcuda -Destination (Join-Path $bin 'nvcuda.dll') -Force
Copy-Item -LiteralPath $cublas -Destination (Join-Path $bin 'cublas64_12.dll') -Force
Copy-Item -LiteralPath $cublas -Destination (Join-Path $bin 'cublasLt64_12.dll') -Force
if ($WithCudartShim) {
    Copy-Item -LiteralPath $cudart -Destination (Join-Path $bin 'cudart64_12.dll') -Force
}
Write-Host "Staged project DLLs in $bin"

$hybridDir = $null
if ($HybridDll) {
    Assert-Hash $HybridDll $HybridSha256 'nvcudart_hybrid64.dll'
    $hybridDir = Join-Path $target 'hybrid'
    New-Item -ItemType Directory -Force -Path $hybridDir | Out-Null
    Copy-Item -LiteralPath $HybridDll -Destination (Join-Path $hybridDir 'nvcudart_hybrid64.dll') -Force
    Write-Host "Staged companion in $hybridDir"
}

$ollamaStage = $null
if ($OllamaDir) {
    if (-not $hybridDir) { throw 'Stock Ollama needs the companion: pass -HybridDll as well.' }
    $src = (Resolve-Path -LiteralPath $OllamaDir).Path
    $srcCuda = Join-Path $src 'lib\ollama\cuda_v13'
    Assert-Hash (Join-Path $srcCuda 'ggml-cuda.dll') $OllamaCudaV13Sha256 'Ollama cuda_v13\ggml-cuda.dll'
    if (-not (Test-Path -LiteralPath (Join-Path $src 'ollama.exe') -PathType Leaf)) {
        throw "ollama.exe not found in $src"
    }

    $ollamaStage = Join-Path $target 'ollama'
    Write-Host "Copying Ollama tree to $ollamaStage (this can take a while)"
    New-Item -ItemType Directory -Force -Path $ollamaStage | Out-Null
    Copy-Item -Path (Join-Path $src '*') -Destination $ollamaStage -Recurse -Force

    # Patch the COPY: the shim takes the place of cuBLAS, nvcuda.dll sits beside the plug-in.
    $stageCuda = Join-Path $ollamaStage 'lib\ollama\cuda_v13'
    foreach ($name in @('cublas64_13.dll', 'cublasLt64_13.dll')) {
        $original = Join-Path $stageCuda $name
        if (Test-Path -LiteralPath $original -PathType Leaf) {
            Move-Item -LiteralPath $original -Destination "$original.orig" -Force
        }
    }
    Copy-Item -LiteralPath $cublas -Destination (Join-Path $stageCuda 'cublas64_13.dll') -Force
    Copy-Item -LiteralPath $nvcuda -Destination (Join-Path $stageCuda 'nvcuda.dll') -Force
    Write-Host 'Patched cuda_v13: shim installed as cublas64_13.dll, cublasLt64_13.dll set aside, nvcuda.dll added'
}

# ---------------------------------------------------------------- environment script
$envLines = @(
    '# Generated by scripts/setup-windows.ps1. Dot-source it:  . .\cfi-env.ps1',
    '$here = Split-Path -Parent $MyInvocation.MyCommand.Path',
    '$env:CUDA_FOR_INTEL_ROOT = $here',
    '$env:PATH = (Join-Path $here ''bin'') + '';'' + $env:PATH'
)
if ($hybridDir) {
    $envLines += '$env:CUDA_FOR_INTEL_HYBRID_DIR = Join-Path $here ''hybrid'''
}
if ($ollamaStage) {
    $envLines += @(
        '$ollama = Join-Path $here ''ollama''',
        '$libs = (Join-Path $ollama ''lib\ollama'') + '';'' + (Join-Path $ollama ''lib\ollama\cuda_v13'')',
        '$env:OLLAMA_LIBRARY_PATH = $libs',
        '$env:PATH = $ollama + '';'' + $libs + '';'' + $env:PATH',
        '$env:OLLAMA_VULKAN = ''0''',
        '$env:OLLAMA_FLASH_ATTENTION = ''0''',
        '$env:OLLAMA_NUM_PARALLEL = ''1''',
        '$env:OLLAMA_LOAD_TIMEOUT = ''60m'''
    )
}
$envFile = Join-Path $target 'cfi-env.ps1'
Set-Content -LiteralPath $envFile -Value $envLines -Encoding UTF8
Write-Host "Wrote $envFile"

Write-Host ''
Write-Host 'Done. Next steps:'
Write-Host "  pwsh -File scripts\check-install.ps1 -TargetDir `"$target`""
if ($ollamaStage) {
    Write-Host "  pwsh -File scripts\run-ollama.ps1 -TargetDir `"$target`""
}

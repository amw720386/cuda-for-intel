<#
.SYNOPSIS
  One-command installer / launcher for CUDA for Intel (Windows x64, experimental).

.DESCRIPTION
  Commands:
    check      Prerequisite check only (Intel GPU + driver, Level Zero loader, VC runtime, Ollama).
    install    check, place the shim DLLs, write cfi-env.ps1 and a manifest, run the self-test.
    selftest   Vector add through the installed shim: prints PASS or FAIL with next steps.
    doctor     Redacted diagnostic report for bug reports.
    env        Print the line that loads the environment into your current shell.
    run        Start a program through cfi-run.exe: cfi_redirect.dll maps NVIDIA CUDA library loads
               (nvcuda, cublas*, cublasLt*, cudnn*, nvml, nvrtc*, cufft*, cusparse*, cusolver*) to
               the install's bin directory, also for absolute-path loads and in child processes.
               Nothing in the program's directory is changed. Example:
                 pwsh -File scripts\cfi.ps1 run -- python -c "import torch; print(torch.cuda.is_available())"
    precompile Translate and compile every kernel of the given programs/DLLs ahead of time with
               cfi-precompile.exe, so the first run does not pay for it. Fills the SPIR-V cache
               and the native-binary cache that the installed nvcuda.dll reads. Run it in the same
               environment as the program (dot-source cfi-env.ps1 first if you use it). Example:
                 pwsh -File scripts\cfi.ps1 precompile -- "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama\cuda_v13"
    uninstall  Remove what install placed and restore anything it backed up.

  Safety model: install writes ONLY inside -InstallDir (default %LOCALAPPDATA%\cuda-for-intel\install).
  It never writes to System32, never to an existing Ollama or CUDA directory, and never sets
  machine or user environment variables: the environment is a script you dot-source per shell.
  If a destination file already exists and is not one this installer placed, it is copied to
  <name>.bak-<timestamp> first; uninstall puts it back. -DryRun prints the plan and changes nothing.

  DLL source (first that applies): -BinDir (folder with nvcuda.dll, cublas64_12.dll),
  -ZipPath / -ZipUrl (+ mandatory -ZipSha256), the repository's target\release, or a local
  `cargo build --release --locked` if cargo is available.

  Stock Ollama staging needs third-party files you supply; use scripts\setup-windows.ps1 for that
  (docs/ollama.md). It also works on a copy and never modifies your Ollama install.

.EXAMPLE
  pwsh -File scripts\cfi.ps1 install
  . "$env:LOCALAPPDATA\cuda-for-intel\install\cfi-env.ps1"
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [ValidateSet('check', 'install', 'selftest', 'doctor', 'env', 'uninstall', 'run', 'precompile')] [string]$Command = 'check',
    [string]$InstallDir,
    [string]$BinDir,
    [string]$ZipPath,
    [string]$ZipUrl,
    [string]$ZipSha256,
    [switch]$SkipSelfTest,
    [switch]$DryRun,
    [switch]$Force,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$RunArgs
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'cfi-common.ps1')
if (-not $InstallDir) { $InstallDir = Get-CfiDefaultInstallDir }
$repo = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $InstallDir 'manifest.json'
$shimNames = @('nvcuda.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
# Placed when the source has them (cargo build -p cfi_redirect -p cfi_launcher -p cfi_precompile);
# needed by `run` and `precompile`.
$launcherNames = @('cfi_redirect.dll', 'cfi-run.exe', 'cfi-precompile.exe')

function Invoke-Prereqs {
    $problems = 0
    Write-Host '--- prerequisites ---'
    if (-not [Environment]::Is64BitOperatingSystem) { Write-Host 'FAIL  64-bit Windows is required'; $problems++ } else { Write-Host 'OK    64-bit Windows' }
    $gpus = @(Get-IntelGpus | Where-Object { $_.IsIntel })
    if ($gpus.Count -eq 0) {
        Write-Host 'FAIL  no Intel GPU detected. Remote desktop sessions and VMs without passthrough expose none.'
        $problems++
    } else {
        foreach ($g in $gpus) {
            if ($g.Problem) { Write-Safe "FAIL  Intel GPU: $($g.Name), driver $($g.Driver): Device Manager reports $(Get-PnpProblemText $g.Problem). Level Zero finds no device until the driver loads."; $problems++ }
            else { Write-Safe "OK    Intel GPU: $($g.Name), driver $($g.Driver)" }
        }
        Write-Host '      Only Arc B390 (driver 32.0.101.9033) is a tested part; see docs/compatibility.md.'
    }
    $ze = Find-LevelZeroLoader
    if ($ze) { Write-Host 'OK    Level Zero loader (ze_loader.dll) present' }
    else { Write-Host 'FAIL  Level Zero loader missing: install the current Intel graphics driver'; $problems++ }
    $vc = Test-VcRuntime
    if ($vc) { Write-Host "OK    VC++ runtime x64 $vc" }
    else { Write-Host 'FAIL  VC++ 2015-2022 x64 runtime missing: install the Microsoft redistributable'; $problems++ }
    $ov = Get-OllamaVersion $null
    if ($ov) {
        if ($ov -like '0.12.6*') { Write-Host "OK    Ollama $ov (tested version)" }
        else { Write-Host "WARN  Ollama $ov found; only v0.12.6 is tested. Your install is not modified by this script." }
    } else { Write-Host 'INFO  Ollama not on PATH (optional)' }
    return $problems
}

function Get-DllSource {
    if ($BinDir) { return @{ Dir = (Resolve-Path -LiteralPath $BinDir).Path; Origin = 'BinDir' } }
    if ($ZipUrl -or $ZipPath) {
        if (-not $ZipSha256) { throw '-ZipSha256 is required with -ZipPath/-ZipUrl (pin the hash from the release notes).' }
        $zip = $ZipPath
        if ($ZipUrl) {
            if ($ZipUrl -notmatch '^https://') { throw '-ZipUrl must be https.' }
            $zip = Join-Path ([IO.Path]::GetTempPath()) ('cfi-' + [guid]::NewGuid().ToString('N') + '.zip')
            if ($DryRun) { Write-Host "[dry-run] would download $ZipUrl"; return @{ Dir = $null; Origin = 'zip-dryrun' } }
            Invoke-WebRequest -Uri $ZipUrl -OutFile $zip -UseBasicParsing
        }
        if ($DryRun) { Write-Host "[dry-run] would verify and extract $zip"; return @{ Dir = $null; Origin = 'zip-dryrun' } }
        $actual = Get-Sha256Hex $zip
        if ($actual -ne $ZipSha256.ToUpperInvariant()) { throw "Zip SHA-256 mismatch: got $actual, expected $ZipSha256. Not extracting." }
        $dst = Join-Path ([IO.Path]::GetTempPath()) ('cfi-x-' + [guid]::NewGuid().ToString('N'))
        Expand-Archive -LiteralPath $zip -DestinationPath $dst -Force
        $hit = Get-ChildItem -LiteralPath $dst -Recurse -Filter 'nvcuda.dll' | Select-Object -First 1
        if (-not $hit) { throw 'nvcuda.dll not found in the archive.' }
        return @{ Dir = $hit.DirectoryName; Origin = 'zip' }
    }
    $rel = Join-Path $repo 'target\release'
    if (Test-Path -LiteralPath (Join-Path $rel 'nvcuda.dll')) { return @{ Dir = $rel; Origin = 'target\release' } }
    if (Get-Command cargo -ErrorAction SilentlyContinue) {
        if ($DryRun) { Write-Host '[dry-run] would run: cargo build --release --locked -p cuda-for-intel -p cublas_shim -p cfi_redirect -p cfi_launcher'; return @{ Dir = $null; Origin = 'build-dryrun' } }
        Write-Host 'Building from source (cargo build --release --locked) ...'
        Push-Location $repo
        try { & cargo build --release --locked -p cuda-for-intel -p cublas_shim -p cfi_redirect -p cfi_launcher; if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" } }
        finally { Pop-Location }
        return @{ Dir = $rel; Origin = 'cargo' }
    }
    throw 'No DLLs to install. Pass -BinDir <folder>, or -ZipPath/-ZipUrl with -ZipSha256, or install Rust (rustup, MSVC x64) and rerun.'
}

function Install-Cfi {
    $problems = Invoke-Prereqs
    if ($problems -gt 0 -and -not $Force) {
        Write-Host ''
        Write-Host "$problems prerequisite problem(s) above. Fix them, or pass -Force to install anyway."
        exit 2
    }
    $src = Get-DllSource
    $bin = Join-Path $InstallDir 'bin'
    Write-Host ''
    Write-Safe "--- plan: place shim DLLs in $bin (source: $($src.Origin)) ---"
    if ($DryRun) {
        foreach ($n in $shimNames) { Write-Host "[dry-run] would place $n" }
        foreach ($n in $launcherNames) { Write-Host "[dry-run] would place $n if the source has it" }
        Write-Host "[dry-run] would write cfi-env.ps1 and manifest.json in the install directory; nothing changed."
        return
    }
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $entries = @()
    $old = $null
    if (Test-Path -LiteralPath $manifestPath) { $old = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json }
    $ours = @{}
    if ($old) { foreach ($e in $old.files) { $ours[$e.path] = $e.sha256 } }
    foreach ($n in $shimNames) {
        $from = Join-Path $src.Dir $n
        if ($n -eq 'cublasLt64_12.dll') { $from = Join-Path $src.Dir 'cublas64_12.dll' }  # one DLL exports both
        if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { throw "Missing $from" }
        $to = Join-Path $bin $n
        $backup = $null
        if (Test-Path -LiteralPath $to -PathType Leaf) {
            $isOurs = $ours.ContainsKey($to) -and ($ours[$to] -eq (Get-Sha256Hex $to))
            if (-not $isOurs) {
                $backup = "$to.bak-$stamp"
                Copy-Item -LiteralPath $to -Destination $backup
                Write-Safe "backup  existing $n -> $(Split-Path -Leaf $backup)"
            }
        }
        Copy-Item -LiteralPath $from -Destination $to -Force
        $entries += [pscustomobject]@{ path = $to; sha256 = (Get-Sha256Hex $to); backup = $backup }
        Write-Host "placed  $n"
    }
    foreach ($n in $launcherNames) {
        $from = Join-Path $src.Dir $n
        if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { Write-Host "skip    $n (not in the source; 'run' needs it)"; continue }
        $to = Join-Path $bin $n
        Copy-Item -LiteralPath $from -Destination $to -Force
        $entries += [pscustomobject]@{ path = $to; sha256 = (Get-Sha256Hex $to); backup = $null }
        Write-Host "placed  $n"
    }
    $envFile = Join-Path $InstallDir 'cfi-env.ps1'
    if (Test-Path -LiteralPath $envFile) { Copy-Item -LiteralPath $envFile -Destination "$envFile.bak-$stamp" }
    @(
        '# Generated by scripts/cfi.ps1 install. Dot-source it per shell:  . .\cfi-env.ps1',
        '$here = Split-Path -Parent $MyInvocation.MyCommand.Path',
        '$env:CUDA_FOR_INTEL_ROOT = $here',
        '$env:PATH = (Join-Path $here ''bin'') + '';'' + $env:PATH',
        '# Translator substitution policy: auto (default), none (pure PTX translation), or a list. docs/troubleshooting.md',
        'if (-not $env:CUDA_FOR_INTEL_SUBST) { $env:CUDA_FOR_INTEL_SUBST = ''auto'' }',
        '$env:CUDA_FOR_INTEL_CACHE_DIR = Join-Path $here ''kernel-cache''',
        '$env:CFI_CACHE_DIR = Join-Path $here ''spv-cache'''
    ) | Set-Content -LiteralPath $envFile -Encoding UTF8
    $entries += [pscustomobject]@{ path = $envFile; sha256 = (Get-Sha256Hex $envFile); backup = $null }
    [pscustomobject]@{ marker = $script:CfiVersionMarker; installed = (Get-Date).ToString('s'); origin = $src.Origin; files = $entries } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host ''
    Write-Safe "Installed. Load the environment in a shell with:   . `"$envFile`""
    if ($SkipSelfTest) { return }
    Write-Host ''
    Write-Host '--- self-test (uses the GPU briefly) ---'
    & (Join-Path $PSScriptRoot 'cfi-selftest.ps1') -InstallDir $InstallDir
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host 'Next: docs/ollama.md for stock Ollama, docs/pytorch.md for PyTorch, docs/troubleshooting.md if anything fails.'
    }
    exit $code
}

function Uninstall-Cfi {
    if (-not (Test-Path -LiteralPath $manifestPath)) { Write-Safe "Nothing to uninstall: no manifest in $InstallDir"; return }
    $m = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    foreach ($e in $m.files) {
        if ($DryRun) { Write-Safe "[dry-run] would remove $($e.path); restore backup: $([bool]$e.backup)"; continue }
        if (Test-Path -LiteralPath $e.path -PathType Leaf) {
            if ((Get-Sha256Hex $e.path) -ne $e.sha256) {
                Write-Safe "keep    $($e.path) (modified since install; left in place)"
                continue
            }
            Remove-Item -LiteralPath $e.path -Force
            Write-Safe "removed $($e.path)"
        }
        if ($e.backup -and (Test-Path -LiteralPath $e.backup)) {
            Move-Item -LiteralPath $e.backup -Destination $e.path -Force
            Write-Safe "restored $($e.path) from backup"
        }
    }
    if ($DryRun) { return }
    Remove-Item -LiteralPath $manifestPath -Force
    foreach ($d in @((Join-Path $InstallDir 'bin'), $InstallDir)) {
        if ((Test-Path -LiteralPath $d) -and -not (Get-ChildItem -LiteralPath $d -Force)) { Remove-Item -LiteralPath $d -Force }
    }
    Write-Host 'Uninstalled. Caches (kernel-cache, spv-cache) were left in the install directory if present; delete them by hand if you want the space back.'
}

function Invoke-Run {
    $bin = Join-Path $InstallDir 'bin'
    $exe = Join-Path $bin 'cfi-run.exe'
    $prog = @($RunArgs)
    if ($prog.Count -gt 0 -and $prog[0] -eq '--') { $prog = @($prog | Select-Object -Skip 1) }
    if ($prog.Count -eq 0) { throw 'usage: cfi.ps1 run [-InstallDir DIR] -- PROGRAM [ARGS...]' }
    if ($DryRun) { Write-Safe "[dry-run] would run: $exe --install-dir $bin -- $($prog -join ' ')"; return }
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "cfi-run.exe not found in $bin; rerun 'cfi.ps1 install' from a build that includes cfi_launcher and cfi_redirect." }
    & $exe --install-dir $bin -- @prog
    exit $LASTEXITCODE
}

function Invoke-Precompile {
    $bin = Join-Path $InstallDir 'bin'
    $exe = Join-Path $bin 'cfi-precompile.exe'
    $rest = @($RunArgs)
    if ($rest.Count -gt 0 -and $rest[0] -eq '--') { $rest = @($rest | Select-Object -Skip 1) }
    if ($rest.Count -eq 0) { throw 'usage: cfi.ps1 precompile [-InstallDir DIR] -- [cfi-precompile options] FILE-OR-DIR...' }
    if ($DryRun) { Write-Safe "[dry-run] would run: $exe --install-dir $bin $($rest -join ' ')"; return }
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "cfi-precompile.exe not found in $bin; rerun 'cfi.ps1 install' from a build that includes cfi_precompile." }
    & $exe --install-dir $bin @rest
    exit $LASTEXITCODE
}

switch ($Command) {
    'check'     { $p = Invoke-Prereqs; if ($p -gt 0) { exit 2 } }
    'install'   { Install-Cfi }
    'selftest'  { & (Join-Path $PSScriptRoot 'cfi-selftest.ps1') -InstallDir $InstallDir; exit $LASTEXITCODE }
    'doctor'    { & (Join-Path $PSScriptRoot 'cfi-doctor.ps1') -InstallDir $InstallDir }
    'env'       { Write-Safe ". `"$(Join-Path $InstallDir 'cfi-env.ps1')`"" }
    'uninstall' { Uninstall-Cfi }
    'run'       { Invoke-Run }
    'precompile' { Invoke-Precompile }
}

<#
.SYNOPSIS
  Smoke-tests a CUDA for Intel build: device query, then a vector add on the GPU.
  Prints PASS or FAIL per step with a remediation hint.

.DESCRIPTION
  Step 1 runs `cargo run --release --example devices` (Level Zero driver and device report).
  Step 2 runs the GPU vector-add test (tests/gpu, `vector_add`-style test filter) against the
  release nvcuda.dll built from this checkout. Exit code 0 only if every step passed.

  Hold the GPU lock (see CONTRIBUTING.md) if other people or agents share the machine.

.PARAMETER TargetDir
  Optional. A directory staged by setup-windows.ps1; when given, its bin\nvcuda.dll is
  checked to be byte-identical to the build output.

.PARAMETER SkipBuild
  Do not run `cargo build`; use target\release as it is.
#>
[CmdletBinding()]
param(
    [string]$TargetDir,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$failed = 0

function Report([bool]$Ok, [string]$Name, [string]$Hint) {
    if ($Ok) {
        Write-Host "PASS  $Name"
    } else {
        Write-Host "FAIL  $Name"
        Write-Host "      hint: $Hint"
        $script:failed++
    }
}

function Invoke-Step([string[]]$CargoArguments) {
    # Windows PowerShell 5.1 turns any stderr line of a native command (cargo prints its
    # warnings there) into a terminating error under $ErrorActionPreference = 'Stop'.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & cargo @CargoArguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    [pscustomobject]@{ Code = $code; Text = ($output -join "`n") }
}

Push-Location $repo
try {
    $haveCargo = [bool](Get-Command cargo -ErrorAction SilentlyContinue)
    Report $haveCargo 'cargo is installed' 'Install Rust from https://rustup.rs with the MSVC x64 toolchain, then reopen the shell.'
    if (-not $haveCargo) { exit 1 }

    if (-not $SkipBuild) {
        $build = Invoke-Step @('build', '--release', '--locked', '-p', 'cuda-for-intel', '-p', 'cublas_shim')
        Report ($build.Code -eq 0) 'release build' "Read the cargo error below. A missing linker means the Visual Studio C++ build tools are not installed.`n$($build.Text)"
        if ($build.Code -ne 0) { exit 1 }
    }

    $dll = Join-Path $repo 'target\release\nvcuda.dll'
    Report (Test-Path -LiteralPath $dll -PathType Leaf) 'target\release\nvcuda.dll exists' 'Run scripts\setup-windows.ps1 or cargo build --release --locked.'

    if ($TargetDir) {
        $staged = Join-Path $TargetDir 'bin\nvcuda.dll'
        $same = (Test-Path -LiteralPath $staged) -and (Test-Path -LiteralPath $dll) -and
                ((Get-FileHash -LiteralPath $staged).Hash -eq (Get-FileHash -LiteralPath $dll).Hash)
        Report $same 'staged nvcuda.dll matches the build' 'Re-run scripts\setup-windows.ps1 so the staged DLL is current.'
    }

    # Step 1: device query.
    $devices = Invoke-Step @('run', '--release', '--locked', '--example', 'devices')
    $text = $devices.Text
    Write-Host $text
    $hint = 'No Level Zero device found. Install the current Intel graphics driver (it ships the Level Zero loader), check Device Manager shows an Intel GPU without a warning icon, and re-run. Integrated and Arc GPUs are listed; a remote desktop session or a virtual machine without GPU passthrough exposes none.'
    Report (($devices.Code -eq 0) -and ($text -notmatch '(?m)^SKIP') -and ($text -match '(?i)UUID')) 'device query (examples/devices.rs)' $hint

    # Step 2: tiny vector add through the built DLL (opt-in hardware test).
    $add = Invoke-Step @('test', '--release', '--locked', '--test', 'gpu', 'compiled_ptx_vector_add', '--', '--ignored', '--nocapture', '--test-threads=1')
    $passed = ($add.Code -eq 0) -and ($add.Text -match 'test result: ok\. [1-9]\d* passed')
    if (-not $passed) { Write-Host $add.Text }
    $hint2 = 'If the output mentions DEVICE_LOST or a timeout, the GPU reset (TDR): close other GPU-heavy programs and retry. If it names an unsupported feature, the driver is too old: update it. Set CUDA_FOR_INTEL_TRACE=1 and CUDA_FOR_INTEL_LOG=1 and re-run for a trace. See README.md, Troubleshooting.'
    Report $passed 'vector add on the GPU (tests/gpu)' $hint2
} finally {
    Pop-Location
}

if ($failed -gt 0) {
    Write-Host "`n$failed check(s) failed."
    exit 1
}
Write-Host "`nAll checks passed."
exit 0

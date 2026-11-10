<#
.SYNOPSIS
  Prints a diagnostic report that is safe to paste into a GitHub issue.

.DESCRIPTION
  Reports OS build, Intel GPU and driver, Level Zero loader, VC runtime, Ollama version, the
  installed shim DLLs (size, short SHA-256, build info including TRANSLATOR_VERSION), the
  CUDA_FOR_INTEL_* / CFI_* switches in effect and cache sizes.
  Read-only. It does not open the GPU. Output is passed through a redactor that replaces the
  user name, computer name, profile path, IPv4 addresses and e-mail-like strings.
  Review the text before posting anyway.

.PARAMETER InstallDir
  Install directory created by cfi.ps1 install (default: the per-user default).
.PARAMETER BinDir
  Look for the shim DLLs here instead (for example target\release).
#>
[CmdletBinding()]
param([string]$InstallDir, [string]$BinDir)
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'cfi-common.ps1')
if (-not $InstallDir) { $InstallDir = Get-CfiDefaultInstallDir }

function Row([string]$k, [string]$v) { Write-Safe ('{0,-26} {1}' -f $k, $v) }

Write-Host '=== cuda-for-intel doctor ==='
Row 'report-time-utc' ([DateTime]::UtcNow.ToString('yyyy-MM-dd HH:mm'))
$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
if ($os) { Row 'os' "$($os.Caption) build $($os.BuildNumber) $($os.OSArchitecture)" }
Row 'powershell' $PSVersionTable.PSVersion.ToString()

Write-Host ''
Write-Host '--- GPU and driver ---'
$gpus = @(Get-IntelGpus)
if ($gpus.Count -eq 0) { Row 'gpu' 'none detected (remote desktop session or VM without passthrough?)' }
foreach ($g in $gpus) {
    $date = ''
    if ($g.DriverDate) { $date = $g.DriverDate.ToString('yyyy-MM-dd') }
    Row 'gpu' "$($g.Name) | PCI $($g.Vendor):$($g.Device) | driver $($g.Driver) $date | intel=$($g.IsIntel)"
    if ($g.Problem) { Row 'gpu-device-status' "PROBLEM $(Get-PnpProblemText $g.Problem): the GPU is not usable, cuInit returns 100. See Device Manager and docs/troubleshooting.md" }
    else { Row 'gpu-device-status' 'working' }
}
$ze = Find-LevelZeroLoader
if ($ze) {
    $zv = (Get-Item -LiteralPath $ze).VersionInfo.FileVersion
    Row 'level-zero-loader' "found, version $zv"
} else {
    Row 'level-zero-loader' 'NOT FOUND (install or update the Intel graphics driver)'
}
$vc = Test-VcRuntime
if ($vc) { Row 'vc-runtime-x64' $vc } else { Row 'vc-runtime-x64' 'NOT FOUND (install the Microsoft Visual C++ 2015-2022 x64 redistributable)' }
$ov = Get-OllamaVersion $null
if ($ov) { Row 'ollama-on-path' "$ov (tested: 0.12.6)" } else { Row 'ollama-on-path' 'not found (optional)' }

Write-Host ''
Write-Host '--- shim DLLs ---'
$bins = @()
if ($BinDir) { $bins += $BinDir } else { $bins += (Join-Path $InstallDir 'bin') }
$manifest = Join-Path $InstallDir 'manifest.json'
if (-not $BinDir) {
    if (Test-Path -LiteralPath $manifest) { Row 'install-manifest' 'present' } else { Row 'install-manifest' 'absent (not installed with cfi.ps1 install)' }
}
$found = $false
foreach ($b in $bins) {
    foreach ($name in @('nvcuda.dll', 'cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll')) {
        $p = Join-Path $b $name
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { continue }
        $found = $true
        $len = (Get-Item -LiteralPath $p).Length
        $sha = (Get-Sha256Hex $p).Substring(0, 16)
        $info = ''
        if ($name -eq 'nvcuda.dll') { $info = ' | ' + (Get-ShimBuildInfo $p) }
        Row $name "$len bytes | sha256 $sha...$info"
    }
}
if (-not $found) { Row 'shim' 'no shim DLLs found (run: pwsh -File scripts\cfi.ps1 install)' }
$sys = Join-Path $env:SystemRoot 'System32\nvcuda.dll'
if (Test-Path -LiteralPath $sys) { Row 'system-nvcuda.dll' 'present in System32 (NVIDIA driver; this project never touches it)' }

Write-Host ''
Write-Host '--- environment switches in effect ---'
Row 'SUBST (effective)' $(if ($env:CUDA_FOR_INTEL_SUBST) { $env:CUDA_FOR_INTEL_SUBST } else { 'auto (unset = auto)' })
$vars = Get-ChildItem Env: | Where-Object { $_.Name -like 'CUDA_FOR_INTEL_*' -or $_.Name -like 'CFI_*' -or $_.Name -eq 'CUDA_MODULE_LOADING' } | Sort-Object Name
if (-not $vars) { Row 'switches' 'none set (all defaults)' }
foreach ($v in $vars) { Row $v.Name $v.Value }
$hy = $env:CUDA_FOR_INTEL_HYBRID_DIR
if ($hy) {
    $hp = Join-Path $hy 'nvcudart_hybrid64.dll'
    if (Test-Path -LiteralPath $hp) { Row 'hybrid-companion' ('present, sha256 ' + (Get-Sha256Hex $hp).Substring(0, 16) + '...') }
    else { Row 'hybrid-companion' 'HYBRID_DIR set but nvcudart_hybrid64.dll missing' }
}

Write-Host ''
Write-Host '--- caches ---'
$spv = if ($env:CFI_CACHE_DIR) { $env:CFI_CACHE_DIR } else { Join-Path $env:LOCALAPPDATA 'cuda-for-intel\spvcache' }
$nat = if ($env:CUDA_FOR_INTEL_CACHE_DIR) { $env:CUDA_FOR_INTEL_CACHE_DIR } else { Join-Path $env:LOCALAPPDATA 'cuda-for-intel\cache\v1' }
foreach ($pair in @(@('translation-cache', $spv), @('native-cache', $nat))) {
    $d = $pair[1]
    if (Test-Path -LiteralPath $d) {
        $files = @(Get-ChildItem -LiteralPath $d -Recurse -File -ErrorAction SilentlyContinue)
        $mb = [math]::Round((($files | Measure-Object Length -Sum).Sum) / 1MB, 1)
        Row $pair[0] "$($files.Count) files, $mb MB, at $d"
    } else { Row $pair[0] "empty/absent, at $d" }
}
Write-Host ''
Write-Host 'Paste everything above into your issue. Check it once for anything private.'

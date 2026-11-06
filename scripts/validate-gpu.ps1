[CmdletBinding()]
param(
    [int[]]$Device,
    [ValidateRange(0, 10)]
    [int]$PerformancePasses = 2,
    [string]$LogPath
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path -Parent $PSScriptRoot
$log = if ($LogPath) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LogPath)
} else {
    Join-Path $root 'docs/gpu-validation.log'
}
$variables = @(
    'CUDA_FOR_INTEL_DEVICE', 'CUDA_FOR_INTEL_BATCH_CAP', 'CUDA_FOR_INTEL_BATCH_US',
    'CUDA_FOR_INTEL_BATCH_TIMER', 'CUDA_FOR_INTEL_BENCH_CASES',
    'CUDA_FOR_INTEL_BENCH_LOADS', 'CUDA_FOR_INTEL_BENCH_SPACED'
)
$saved = @{}
foreach ($variable in $variables) {
    $saved[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
}
$timerAvailable = [Environment]::OSVersion.Version.Build -ge 17134

function Write-Log([string]$Message) {
    $Message | Tee-Object -FilePath $log -Append
}

function Invoke-Cargo([string[]]$Arguments) {
    Write-Host "cargo $($Arguments -join ' ')"
    Add-Content -LiteralPath $log "cargo $($Arguments -join ' ')"
    cargo @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "cargo exited with code $LASTEXITCODE. See $log." }
}

function Invoke-Tests([string[]]$Arguments, [string[]]$Skip) {
    $argumentsWithSkips = $Arguments + @('--', '--ignored', '--nocapture', '--test-threads=1')
    foreach ($test in $Skip) {
        $argumentsWithSkips += @('--skip', $test)
    }
    $result = @(Invoke-Cargo $argumentsWithSkips)
    $result | Write-Output
    foreach ($line in $result) {
        if ($line -match '^\s*Running .+ \((.+\.exe)\)$') {
            $binary = $Matches[1]
            if (Test-Path -LiteralPath $binary -PathType Leaf) {
                Write-Log "SHA256 $((Get-FileHash -LiteralPath $binary).Hash) $binary"
            }
        }
    }
    if (-not ($result -match '^test result: ok\. [1-9]\d* passed;')) {
        throw "No test executed. See $log."
    }
}

Push-Location $root
try {
    New-Item -ItemType Directory -Path (Split-Path -Parent $log) -Force | Out-Null
    Set-Content -LiteralPath $log 'Intel Level Zero GPU validation'
    Write-Log "Started $([DateTimeOffset]::UtcNow.ToString('o'))"
    if ((Test-Path -LiteralPath '.git') -and (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Log "Source revision $(git rev-parse HEAD)"
        foreach ($change in git status --short --untracked-files=normal) { Write-Log "Working tree $change" }
    }
    foreach ($artifact in Get-Item -Path 'Cargo.lock', 'examples/*.ptx') {
        Write-Log "SHA256 $((Get-FileHash -LiteralPath $artifact.FullName).Hash) $($artifact.Name)"
    }
    rustc -Vv | ForEach-Object { Write-Log "$_" }
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        Write-Log "Windows $($os.Caption) build=$($os.BuildNumber) visible-memory-KiB=$($os.TotalVisibleMemorySize)"
        foreach ($cpu in Get-CimInstance Win32_Processor) {
            Write-Log "CPU $($cpu.Name) cores=$($cpu.NumberOfCores) logical=$($cpu.NumberOfLogicalProcessors)"
        }
        foreach ($adapter in Get-CimInstance Win32_VideoController) {
            Write-Log "Windows adapter $($adapter.Name) driver=$($adapter.DriverVersion) PNP=$($adapter.PNPDeviceID)"
        }
    } catch {
        Write-Log "SKIP Windows inventory: $($_.Exception.Message)"
    }
    $loader = Join-Path $env:SystemRoot 'System32/ze_loader.dll'
    if (Test-Path -LiteralPath $loader) {
        Write-Log "System Level Zero loader $((Get-Item -LiteralPath $loader).VersionInfo.FileVersion)"
    }
    $inventory = @(Invoke-Cargo @('run', '--locked', '--example', 'devices', '--', '--validation'))
    $inventory | Write-Output
    $ordinals = @($inventory | Where-Object { $_ -match '^DEVICE\t\d+$' } |
        ForEach-Object { [int]($_ -split "`t")[1] })
    if ($Device.Count -eq 0) { $Device = $ordinals }
    if ($Device.Count -eq 0) {
        Write-Log 'SKIP hardware validation: no Intel Level Zero root GPU found'
        return
    }
    foreach ($ordinal in $Device) {
        if ($ordinal -notin $ordinals) { throw "Unknown Intel Level Zero device ordinal $ordinal." }
    }
    $target = cargo metadata --locked --no-deps --format-version 1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed' }
    $dll = Join-Path $target.target_directory 'debug/nvcuda.dll'

    foreach ($ordinal in $Device) {
        $env:CUDA_FOR_INTEL_DEVICE = "$ordinal"
        $skip = @()
        foreach ($line in $inventory) {
            if ($line -match "^SKIP\t$ordinal\t([^\t]+)\t(.+)$") {
                $skip += $Matches[1]
                Write-Log "SKIP device $ordinal $($Matches[1]): $($Matches[2])"
            }
        }
        if ('*' -in $skip) { continue }
        if (-not $timerAvailable) {
            $skip += @('gated_recovery_initialization_and_destruction_retry',
                'gated_recovery_worker_error_and_concurrent_teardown',
                'gated_recovery_buffered_and_retired_dependencies')
            Write-Log "SKIP device $ordinal high-resolution timer: Windows build predates 1803"
            Write-Log "SKIP device $ordinal high-resolution worker recovery cases"
        }
        foreach ($backend in @('immediate', 'batch')) {
            $features = @()
            if ($backend -eq 'batch') {
                $features = @('--features', 'experimental-batching')
                $env:CUDA_FOR_INTEL_BATCH_CAP = '32'
                $env:CUDA_FOR_INTEL_BATCH_US = '250'
                $env:CUDA_FOR_INTEL_BATCH_TIMER = if ($timerAvailable) { 'high-resolution' } else { 'condvar' }
            } else {
                foreach ($name in $variables | Where-Object { $_ -like '*_BATCH_*' }) {
                    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
                }
            }
            Write-Log "RUN device $ordinal backend $backend $env:CUDA_FOR_INTEL_BATCH_TIMER"
            Invoke-Cargo (@('build', '--locked') + $features)
            Write-Log "SHA256 $((Get-FileHash -LiteralPath $dll).Hash) $dll"
            Invoke-Tests (@('test', '--locked', '--test', 'gpu') + $features) $skip
            Invoke-Tests (@('test', '--locked', '--lib', 'gated_') + $features) ($skip + 'gated_m10_' + 'gated_forge_copy_queue_timestamps')
        }
        if ($PerformancePasses -gt 0 -and $timerAvailable) {
            $env:CUDA_FOR_INTEL_BENCH_CASES = 'A1,C250,A2'
            $env:CUDA_FOR_INTEL_BENCH_LOADS = 'idle'
            [Environment]::SetEnvironmentVariable('CUDA_FOR_INTEL_BENCH_SPACED', $null, 'Process')
            $perfTests = @('gated_m10_device_execution', 'gated_m10_forge', 'gated_m10_isolated_latency_under_load', 'gated_m10_throughput')
            $runnable = @($perfTests | Where-Object {
                $test = $_
                -not @($skip | Where-Object { $test.Contains($_) }).Count
            })
            if ($runnable.Count -eq 0) {
                Write-Log "SKIP device $ordinal release comparison: no supported workload"
            } else {
                for ($pass = 1; $pass -le $PerformancePasses; $pass++) {
                    Write-Log "RUN device $ordinal release comparison pass $pass"
                    Invoke-Tests @('test', '--release', '--locked', '--lib', 'gated_m10_') ($skip + 'gated_m10_host_timer')
                }
            }
        } elseif ($PerformancePasses -gt 0) {
            Write-Log "SKIP device $ordinal release comparison: high-resolution timer unavailable"
        }
    }
    Invoke-Cargo @('build', '--locked')
} finally {
    foreach ($variable in $variables) {
        [Environment]::SetEnvironmentVariable($variable, $saved[$variable], 'Process')
    }
    Pop-Location
}

<#
.SYNOPSIS
  Windows (no GPU) acceptance test for cfi-run + cfi_redirect.dll (docs/zluda-parity.md roadmap item 3).

.DESCRIPTION
  Builds the launcher, the redirect DLL and three test binaries, then lays out
    app\      cfi-redirect-probe.exe, cublas64_12.dll and cublas64_11.dll ("NVIDIA's", a dummy),
              cfi_consumer.dll (imports cublas64_12.dll by name, like torch_cuda.dll)
    install\  cublas64_12.dll (ours: the same dummy, told apart by path)
  and checks:
    1. control: without cfi-run every cuBLAS load resolves to app\ (the test can fail);
    2. under cfi-run, in the process and in one CreateProcessW child: the static import, a
       LoadLibraryW of the full path app\cublas64_12.dll and a LoadLibraryExW of app\cublas64_11.dll
       all resolve to install\ (or its aliases\ copy);
    3. no file in app\ was added, removed or modified.
  Exit 0 = pass.
#>
[CmdletBinding()]
param([string]$BuildProfile = 'debug', [switch]$NoBuild)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$target = Join-Path $repo "target\$BuildProfile"
if (-not $NoBuild) {
    $flags = @('build', '--locked', '-p', 'cfi_redirect', '-p', 'cfi_launcher', '-p', 'cfi_redirect_probe', '-p', 'cfi_redirect_dummy', '-p', 'cfi_redirect_consumer')
    if ($BuildProfile -eq 'release') { $flags += '--release' }
    Push-Location $repo
    try { & cargo @flags; if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" } } finally { Pop-Location }
}
$root = Join-Path ([IO.Path]::GetTempPath()) ('cfi-redirect-' + [guid]::NewGuid().ToString('N'))
$app = Join-Path $root 'app with space'
$inst = Join-Path $root 'install'
New-Item -ItemType Directory -Force -Path $app, $inst | Out-Null
$dummy = Join-Path $target 'cfi_redirect_dummy.dll'
Copy-Item $dummy (Join-Path $app 'cublas64_12.dll')
Copy-Item $dummy (Join-Path $app 'cublas64_11.dll')
Copy-Item (Join-Path $target 'cfi_redirect_consumer.dll') (Join-Path $app 'cfi_consumer.dll')
Copy-Item (Join-Path $target 'cfi-redirect-probe.exe') (Join-Path $app 'cfi-redirect-probe.exe')
Copy-Item $dummy (Join-Path $inst 'cublas64_12.dll')
$probe = Join-Path $app 'cfi-redirect-probe.exe'
$run = Join-Path $target 'cfi-run.exe'

function Get-Snapshot {
    Get-ChildItem -LiteralPath $app -Recurse -Force | Sort-Object FullName | ForEach-Object {
        $h = if ($_.PSIsContainer) { 'dir' } else { (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash }
        '{0}|{1}|{2}|{3}' -f $_.FullName, $_.Length, $_.LastWriteTimeUtc.Ticks, $h
    }
}
$before = Get-Snapshot
$failed = 0

Write-Host '--- 1. control run (no cfi-run): must load the app directory copies'
& $probe $app '-'
if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL control'; $failed++ }

Write-Host '--- 2. under cfi-run, with one CreateProcessW child'
$env:CFI_REDIRECT_LOG = '1'
& $run --verbose --install-dir $inst --redirect-dll (Join-Path $target 'cfi_redirect.dll') -- $probe $app $inst --spawn-child
$code = $LASTEXITCODE
Remove-Item Env:\CFI_REDIRECT_LOG
if ($code -ne 0) { Write-Host "FAIL redirect (exit $code)"; $failed++ }

Write-Host '--- 3. app directory unchanged'
$after = Get-Snapshot
$diff = Compare-Object @($before) @($after)
if ($diff) { $diff | ForEach-Object { Write-Host "changed: $($_.InputObject) ($($_.SideIndicator))" }; $failed++ }
else { Write-Host "OK    $(@($after).Count) entries unchanged" }
if (Test-Path -LiteralPath (Join-Path $inst 'aliases\cublas64_11.dll')) { Write-Host 'OK    alias written under install\aliases' }

Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
if ($failed) { Write-Host "redirect-ci: FAIL ($failed)"; exit 1 }
Write-Host 'redirect-ci: PASS'

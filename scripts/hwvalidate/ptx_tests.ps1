<#
.SYNOPSIS
  vosen/ptx_tests on the Arc GPU against our nvcuda.dll (roadmap item 1, docs/zluda-parity.md). Runs ON the Windows box.

.DESCRIPTION
  Builds ptx_tests (MIT OR Apache-2.0) at the commit tests/ptx_tests/run.sh pins, with tests/ptx_tests/ptx_tests.patch
  (PTX_TESTS_ELEMENTS sampling, per-test panic isolation), using stable Rust with RUSTC_BOOTSTRAP=1 (ptx_tests needs
  nightly features). Then, inside one gpu-queue.ps1 slot, runs tests/ptx_tests/run.py twice, default and with
  CFI_EXACT_FTZ=1, and compares each with tests/ptx_tests/expected_status.tsv. Translation caching is off
  (CFI_NO_CACHE=1), so every module goes through the current translator.

  Results: <Out>\ptx_tests_<mode>.tsv (name, status, first failing input) and <Out>\ptx_tests_<mode>.summary.
  On Windows ptx_tests lists 787 cases (6 directed sqrt cases need the `rug` crate, which it skips on Windows).

.EXAMPLE
  powershell -NoProfile -File scripts\hwvalidate\ptx_tests.ps1 -Nvcuda target\release\nvcuda.dll
  powershell -NoProfile -File scripts\hwvalidate\ptx_tests.ps1 -Nvcuda target\release\nvcuda.dll -Elements 0   # all 2^32 inputs
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$Nvcuda,
  [int]$Elements = 4194304,
  [int]$Jobs = 6,
  [string]$Work = (Join-Path $env:USERPROFILE 'cfi-work'),
  [string]$Out = '',
  [string]$Filter = ''
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Nvcuda = (Resolve-Path $Nvcuda).Path
if ($Out -eq '') { $Out = Join-Path $Work 'ptx_tests' }
New-Item -ItemType Directory -Force $Out | Out-Null
$rev = (Select-String -Path (Join-Path $repo 'tests\ptx_tests\run.sh') -Pattern '^rev=(\w+)').Matches[0].Groups[1].Value
$up = Join-Path $Out 'upstream'
if (-not (Test-Path (Join-Path $up '.git'))) { git clone --quiet https://github.com/vosen/ptx_tests $up }
git -C $up fetch --quiet origin $rev
git -C $up checkout --quiet --force $rev
git -C $up apply (Join-Path $repo 'tests\ptx_tests\ptx_tests.patch')
$env:RUSTUP_TOOLCHAIN = 'stable'; $env:RUSTC_BOOTSTRAP = '1'; $env:CARGO_TARGET_DIR = Join-Path $Out 'target'
cargo build --release --quiet --manifest-path (Join-Path $up 'Cargo.toml')
if ($LASTEXITCODE -ne 0) { throw "ptx_tests build failed" }
Remove-Item Env:\RUSTUP_TOOLCHAIN, Env:\RUSTC_BOOTSTRAP, Env:\CARGO_TARGET_DIR
$exe = Join-Path $Out 'target\release\ptx_tests.exe'
$py = (Get-Command python).Source

$queue = Join-Path $Work 'gpu-queue.ps1'; $lock = Join-Path $Work 'gpu.lock'; $id = 'ptx-tests'
$got = (& powershell -NoProfile -File $queue -Action acquire -Id $id -MaxWaitSec 3600 2>&1) -join ' '
if ($got -notmatch 'acquired') { "no GPU slot: $got"; exit 3 }
$hb = Start-Job -ArgumentList $lock, $id -ScriptBlock { param($l, $m) while ($true) { try { if ((Get-Content "$l\owner" -EA SilentlyContinue) -eq $m) { (Get-Item $l).LastWriteTime = Get-Date } } catch {}; Start-Sleep 45 } }
$rc = 0
try {
  $env:CFI_NO_CACHE = '1'
  foreach ($mode in 'default', 'exact-ftz') {
    $a = @('-I', (Join-Path $repo 'tests\ptx_tests\run.py'), '--ptx-tests', $exe, '--cuda', $Nvcuda, '--elements', "$Elements",
           '--jobs', "$Jobs", '--timeout', '300', '--out', (Join-Path $Out "ptx_tests_$mode.tsv"))
    if ($mode -eq 'exact-ftz') { $a += '--exact-ftz' }
    if ($Filter -ne '') { $a += @('--filter', $Filter) }
    & $py @a 2>&1 | Tee-Object -FilePath (Join-Path $Out "ptx_tests_$mode.summary")
    if ($LASTEXITCODE -ne 0) { $rc = 1 }
  }
} finally {
  Remove-Item Env:\CFI_NO_CACHE -EA SilentlyContinue
  Stop-Job $hb -EA SilentlyContinue; Remove-Job $hb -Force -EA SilentlyContinue
  & powershell -NoProfile -File $queue -Action release -Id $id | Out-Null
}
exit $rc

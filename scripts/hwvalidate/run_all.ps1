<#
.SYNOPSIS
  One-command hardware validation of the integrated `hw-validate` tree. Runs ON the Windows Arc box.

.DESCRIPTION
  Stages 1-9 run inside ONE gpu-queue.ps1 slot (heartbeat job, release in `finally`). Stage 0 (build, parse checks,
  environment) runs before the slot is taken because it needs no GPU.

  Every stage writes C:-relative results under one results dir:
    <Results>\stage<N>.json       machine-readable stage record (also the resume marker)
    <Results>\logs\...            raw stdout/stderr of every child
    <Results>\results.json        aggregate of all stage records
    <Results>\summary.md          human summary (what passed, what to blame, how to isolate)
  A stage whose stage<N>.json exists is skipped unless -Force (all) or -RerunFailed (only non-green stages).

  Safety: no taskkill / Stop-Process anywhere. Every child is started with a timeout and must exit by itself (GPU tests
  run with CFI_HW_WATCHDOG_S so a hang aborts the test process). If a child still lives after timeout + 120 s grace the run
  is "poisoned": remaining GPU stages are marked skipped and the lock is released. The only termination is the .NET Kill()
  on the Ollama server process tree WE started (Ollama has no shutdown command), after asking it to unload the model.

  Configuration comes from parameters or environment (no machine names in this file):
    CFI_WORK (default %USERPROFILE%\cfi-work), CFI_OLLAMA_RUN (staged stock Ollama tree), CFI_MODELS (comma list of 3 GGUFs),
    CFI_F16_MODEL (f16/bf16 GGUF for the XMX A/B), CFI_HYBRID_DIR, CFI_STOCK_GGML_DLL, CFI_CORPUS_DIR, CFI_TORCH_PY.

.EXAMPLE
  powershell -NoProfile -File scripts\hwvalidate\run_all.ps1
  powershell -NoProfile -File scripts\hwvalidate\run_all.ps1 -Stages 3,6 -Force
#>
[CmdletBinding()]
param(
    [string]$Work = $(if ($env:CFI_WORK) { $env:CFI_WORK } else { Join-Path $env:USERPROFILE 'cfi-work' }),
    [string]$Src = '',
    [string]$Results = '',
    [string]$Stages = '0,1,2,3,4,5,6,7,8,9',
    [switch]$Force,
    [switch]$RerunFailed,
    [string]$OllamaRun = $env:CFI_OLLAMA_RUN,
    [string]$Models = $env:CFI_MODELS,
    [string]$F16Model = $env:CFI_F16_MODEL,
    [string]$Hybrid = $env:CFI_HYBRID_DIR,
    [string]$StockGgml = $env:CFI_STOCK_GGML_DLL,
    [string]$Corpus = $env:CFI_CORPUS_DIR,
    [string]$TorchPy = $env:CFI_TORCH_PY,
    [string]$Id = 'hwvalidate',
    [int]$MinFreeGB = 30,
    [int]$QueueWaitSec = 14400,
    [double]$MaxHours = 8,
    [double]$TimeoutScale = 1.0
)
$ErrorActionPreference = 'Continue'
if (-not $Src) { $Src = Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }
$Src = (Resolve-Path -LiteralPath $Src).Path
$hw = Join-Path $Work 'hwval'
if (-not $Results) { $Results = Join-Path $hw 'results' }
$logs = Join-Path $Results 'logs'
New-Item -ItemType Directory -Force $Results, $logs, $hw | Out-Null
# optional per-box settings (CFI_* variables), kept on the box and never committed: <Work>\hwval\config.ps1 (see config.example.ps1)
$cfgFile = Join-Path $hw 'config.ps1'
if (Test-Path $cfgFile) { . $cfgFile }
if (-not $OllamaRun) { $OllamaRun = $env:CFI_OLLAMA_RUN }
if (-not $Models) { $Models = $env:CFI_MODELS }
if (-not $F16Model) { $F16Model = $env:CFI_F16_MODEL }
if (-not $Hybrid) { $Hybrid = $env:CFI_HYBRID_DIR }
if (-not $StockGgml) { $StockGgml = $env:CFI_STOCK_GGML_DLL }
if (-not $Corpus) { $Corpus = $env:CFI_CORPUS_DIR }
if (-not $TorchPy) { $TorchPy = $env:CFI_TORCH_PY }
$target = Join-Path $hw 'target'
$rel = Join-Path $target 'release'
$queue = Join-Path $Work 'gpu-queue.ps1'
$lockDir = Join-Path $Work 'gpu.lock'
$wanted = @($Stages -split '[,; ]+' | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
$runStart = Get-Date
$script:Poisoned = $null
$script:Cur = $null
$script:StageDeadline = $null
$script:Ports = @{}

# ------------------------------------------------------------------ helpers
function Protect-Text([string]$t) {
    if ($null -eq $t) { return $t }
    foreach ($pair in @(@(($env:USERPROFILE -replace '\\', '\\'), '%USERPROFILE%'), @($env:USERPROFILE, '%USERPROFILE%'), @($env:USERNAME, '<user>'), @($env:COMPUTERNAME, '<host>'))) {
        if ($pair[0] -and $pair[0].Length -ge 3) { $t = $t -ireplace [regex]::Escape($pair[0]), $pair[1] }
    }
    return ($t -replace '\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(\d{1,3}\.){3}\d{1,3}\b', '<ip>')
}
function Save-Json($obj, [string]$path) {
    $json = $obj | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText($path, (Protect-Text $json), (New-Object Text.UTF8Encoding($false)))
}
function Reset-CfiEnv {
    Get-ChildItem Env: | Where-Object { $_.Name -match '^(CFI_|CUDA_FOR_INTEL|NEO_|OLLAMA_)' } | ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
    $env:CARGO_TARGET_DIR = $target
    $cb = Join-Path $env:USERPROFILE '.cargo\bin'
    if ($env:Path -notlike "*$cb*") { $env:Path = "$env:Path;$cb" }
    $env:CFI_HW_WATCHDOG_S = '600'          # hardware tests abort themselves if a launch hangs
    if ($StockGgml) { $env:CFI_STOCK_GGML_DLL = $StockGgml }
    if ($Hybrid) { $env:CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid }
}
function Remaining-Sec { if ($script:StageDeadline) { return [int](($script:StageDeadline - (Get-Date)).TotalSeconds) } else { return 86400 } }
function Add-Result([string]$Name, [string]$Status, [string]$Detail = '', $Rc = $null, [double]$Secs = 0, [string]$Log = '', [string]$Blame = '', [string]$Isolate = '', $Data = $null) {
    $o = [ordered]@{ name = $Name; status = $Status; detail = $Detail; rc = $Rc; secs = [math]::Round($Secs, 1); log = $Log; blame = $Blame; isolate = $Isolate }
    if ($Data) { $o.data = $Data }
    $script:Cur.items += , $o
    Write-Host ("[{0}] {1} {2}" -f $Status.ToUpper(), $Name, $Detail)
}
# Run one child to completion with a timeout; the child must exit by itself.
function Invoke-Child([string]$Name, [string]$File, [string]$ArgLine, [int]$TimeoutSec, [hashtable]$ChildEnv = @{}, [string]$Cwd = '') {
    $safe = ($Name -replace '[^A-Za-z0-9._-]', '_')
    $log = Join-Path $logs $safe
    $res = @{ name = $Name; rc = $null; timeout = $false; secs = 0; log = $log; skipped = $false }
    if ($script:Poisoned) { $res.skipped = $true; return $res }
    $limit = [math]::Min([int]($TimeoutSec * $TimeoutScale), (Remaining-Sec))
    if ($limit -le 5) { $res.timeout = $true; $res.skipped = $true; return $res }
    $saved = @{}
    foreach ($k in $ChildEnv.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process'); [Environment]::SetEnvironmentVariable($k, [string]$ChildEnv[$k], 'Process') }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $sp = @{ FilePath = $File; RedirectStandardOutput = "$log.out"; RedirectStandardError = "$log.err"; PassThru = $true; NoNewWindow = $true }
        if ($ArgLine) { $sp.ArgumentList = $ArgLine }
        if ($Cwd) { $sp.WorkingDirectory = $Cwd }
        $p = Start-Process @sp
        $null = $p.Handle
        if (-not $p.WaitForExit($limit * 1000)) {
            $res.timeout = $true
            if (-not $p.WaitForExit(120000)) { $script:Poisoned = "$Name (pid $($p.Id)) still running after timeout + grace" }
        } else { $p.WaitForExit(); $res.rc = $p.ExitCode }
    } catch { $res.rc = -1; "launch failure: $($_.Exception.Message)" | Set-Content "$log.err" }
    finally {
        $sw.Stop(); $res.secs = $sw.Elapsed.TotalSeconds
        foreach ($k in $ChildEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
    }
    return $res
}
function Read-Out($r) { $t = ''; foreach ($e in '.out', '.err') { if (Test-Path "$($r.log)$e") { $t += (Get-Content "$($r.log)$e" -Raw -ErrorAction SilentlyContinue) + "`n" } }; return $t }

# cargo integration test (tests/gpu.rs): returns passed/failed counts and the failed test names.
# -Expect fail means the run MUST fail (used to prove a disable switch really changes behaviour).
function Invoke-GpuTest([string]$Name, [string[]]$Filters, [int]$TimeoutMin, [hashtable]$TEnv = @{}, [string[]]$Skips = @(), [string]$Expect = 'pass', [string]$Blame = '', [string]$Isolate = '') {
    $a = "test --release -j 2 --test gpu -- --ignored --test-threads=1 --nocapture"
    foreach ($f in $Filters) { $a += " $f" }
    foreach ($s in $Skips) { $a += " --skip $s" }
    $r = Invoke-Child $Name 'cargo' $a ($TimeoutMin * 60) $TEnv $Src
    if ($r.skipped) { Add-Result $Name 'skip' ($(if ($script:Poisoned) { "poisoned: $script:Poisoned" } else { 'stage time budget exhausted' })) -Blame $Blame; return }
    $t = Read-Out $r
    $passed = 0; $failed = 0
    foreach ($m in [regex]::Matches($t, 'test result: \w+\. (\d+) passed; (\d+) failed')) { $passed += [int]$m.Groups[1].Value; $failed += [int]$m.Groups[2].Value }
    $names = @([regex]::Matches($t, '(?m)^test (\S+) \.\.\. FAILED') | ForEach-Object { $_.Groups[1].Value })
    $detail = "passed=$passed failed=$failed"
    if ($names.Count) { $detail += ' failed_tests=' + ($names -join ',') }
    if ($r.timeout) { Add-Result $Name 'timeout' "$detail (limit ${TimeoutMin}m)" $null $r.secs $r.log $Blame $Isolate; return }
    if ($passed + $failed -eq 0 -and $r.rc -eq 0) { Add-Result $Name 'fail' 'no test matched the filter (typo or module not compiled)' $r.rc $r.secs $r.log $Blame $Isolate; return }
    if ($Expect -eq 'fail') {
        if ($r.rc -ne 0 -and $failed -ge 1) { Add-Result $Name 'pass' "failed as required ($detail)" $r.rc $r.secs $r.log $Blame $Isolate }
        else { Add-Result $Name 'fail' "expected the test to FAIL with this switch, but it did not ($detail)" $r.rc $r.secs $r.log $Blame $Isolate }
        return
    }
    if ($r.rc -eq 0 -and $failed -eq 0) { Add-Result $Name 'pass' $detail $r.rc $r.secs $r.log $Blame $Isolate }
    else { Add-Result $Name 'fail' $detail $r.rc $r.secs $r.log $Blame $Isolate }
}

# ---- Ollama helpers (used by stages 2, 3, 6, 8) -------------------------------------------------
function Get-OllamaTree {
    # A process still running from this tree (e.g. a server another script failed to stop) locks its DLLs; nothing is
    # killed, the next private tree (ollama_run_2, _3, ...) is used instead.
    foreach ($n in 1..6) {
        $priv = Join-Path $hw $(if ($n -eq 1) { 'ollama_run' } else { "ollama_run_$n" })
        if (-not (Test-Path (Join-Path $priv 'ollama.exe'))) {
            if (-not $OllamaRun -or -not (Test-Path (Join-Path $OllamaRun 'ollama.exe'))) { return $null }
            & robocopy $OllamaRun $priv /E /XF *.orig /NFL /NDL /NJH /NJS /NP | Out-Null
        }
        $g = Join-Path $priv 'lib\ollama\cuda_v13'
        New-Item -ItemType Directory -Force $g | Out-Null
        try {
            Copy-Item (Join-Path $rel 'nvcuda.dll') $g -Force -ErrorAction Stop
            Copy-Item (Join-Path $rel 'cublas64_12.dll') (Join-Path $g 'cublas64_13.dll') -Force -ErrorAction Stop
            Copy-Item (Join-Path $rel 'cublas64_12.dll') (Join-Path $g 'cublasLt64_13.dll') -Force -ErrorAction Stop
        } catch { Write-Host "ollama tree $priv is in use ($($_.Exception.Message)); trying the next one"; continue }
        $x = Join-Path $priv 'ollama-hv.exe'
        if (-not (Test-Path $x)) { Copy-Item (Join-Path $priv 'ollama.exe') $x }
        return $priv
    }
    return $null
}
function Get-DescendantIds([int]$ParentId) {
    $ids = @()
    foreach ($c in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -ErrorAction SilentlyContinue)) { $ids += [int]$c.ProcessId; $ids += Get-DescendantIds ([int]$c.ProcessId) }
    return $ids
}
function Start-Ollama([string]$Tag, [string]$Tree, [hashtable]$OEnv) {
    $g = Join-Path $Tree 'lib\ollama\cuda_v13'
    $port = Get-Random -Minimum 21000 -Maximum 29000
    $wd = Join-Path $hw "wd_$($Tag -replace '[^A-Za-z0-9._-]','_')"
    if (Test-Path $wd) { Remove-Item $wd -Recurse -Force -ErrorAction SilentlyContinue }
    $cache = Join-Path $hw "cache_$($Tag -replace '[^A-Za-z0-9._-]','_')"
    if (Test-Path $cache) { Remove-Item $cache -Recurse -Force -ErrorAction SilentlyContinue }
    New-Item -ItemType Directory -Force $wd, "$cache\kc", "$cache\spv" | Out-Null
    Reset-CfiEnv
    $env:PATH = "$Tree;$Tree\lib\ollama;$g;$env:PATH"
    $env:OLLAMA_LIBRARY_PATH = "$Tree\lib\ollama;$g"; $env:OLLAMA_VULKAN = '0'
    $env:OLLAMA_MODELS = Join-Path $hw 'models'; $env:OLLAMA_HOST = "127.0.0.1:$port"; $env:OLLAMA_NUM_PARALLEL = '1'
    $env:OLLAMA_FLASH_ATTENTION = '0'; $env:OLLAMA_LOAD_TIMEOUT = '120m'; $env:OLLAMA_KEEP_ALIVE = '10m'
    $env:CUDA_FOR_INTEL_CACHE_DIR = "$cache\kc"; $env:CFI_CACHE_DIR = "$cache\spv"; $env:NEO_CACHE_PERSISTENT = '0'
    $env:CUDA_FOR_INTEL_LOG = '1'
    foreach ($k in $OEnv.Keys) { [Environment]::SetEnvironmentVariable($k, [string]$OEnv[$k], 'Process') }
    $exe = Join-Path $Tree 'ollama-hv.exe'
    $srv = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $wd -FilePath $exe -ArgumentList 'serve' -RedirectStandardOutput "$wd\o.txt" -RedirectStandardError "$wd\e.txt"
    $up = $false
    for ($i = 0; $i -lt 40 -and -not $up; $i++) { Start-Sleep 2; $l = Get-Content "$wd\e.txt" -Raw -ErrorAction SilentlyContinue; if ($l -match "Listening on 127\.0\.0\.1:$port") { $up = $true } elseif ($srv.HasExited) { break } }
    return @{ proc = $srv; wd = $wd; port = $port; exe = $exe; up = $up; tag = $Tag }
}
function Stop-Ollama($S) {
    # ask the server to unload the model first (the runner then exits by itself), then end the process tree WE started.
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$($S.port)/api/generate" -Method Post -Body (@{ model = $S.model; keep_alive = 0 } | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 30 | Out-Null } catch {}
    Start-Sleep 5
    $ids = @(); try { $ids = Get-DescendantIds $S.proc.Id } catch {}
    foreach ($id in $ids) { try { [Diagnostics.Process]::GetProcessById($id).Kill() } catch {} }
    try { if (-not $S.proc.HasExited) { $S.proc.Kill() } } catch {}
    [void]$S.proc.WaitForExit(30000)
    Start-Sleep 3
}
function Invoke-OllamaCreate($S, [string]$Gguf, [string]$Name) {
    "FROM $Gguf" | Set-Content "$($S.wd)\Mf"
    $S.model = $Name
    $r = Invoke-Child "create_$($S.tag)" $S.exe "create $Name -f `"$($S.wd)\Mf`"" 900
    return ($r.rc -eq 0)
}
function Invoke-Gen($S, [string]$Prompt, [int]$Predict, [int]$NumGpu = 99, [int]$NumCtx = 2048) {
    $body = @{ model = $S.model; prompt = $Prompt; stream = $false; raw = $true; options = @{ num_predict = $Predict; temperature = 0; seed = 0; top_k = 1; repeat_penalty = 1.0; num_gpu = $NumGpu; num_ctx = $NumCtx; num_batch = 512 } } | ConvertTo-Json -Depth 4
    $to = [math]::Max(60, [math]::Min(5400, (Remaining-Sec)))
    return Invoke-RestMethod -Uri "http://127.0.0.1:$($S.port)/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec $to
}
function Get-Names { return @($Models -split '[,;]' | Where-Object { $_ -and (Test-Path $_) }) }
function Mk-Words([int]$n) { return ((@('lorem') * $n) -join ' ') }
# $V changes only the first token: every measured request needs a prompt the KV prefix cache has not seen, otherwise
# Ollama skips most of the prefill and the reported prefill tok/s is meaningless.
function Long-Prompt([int]$words, [int]$V = 1) { return ("$V " + (Mk-Words $words) + "`nCount from 1 to 300 separated by commas: 1, 2, 3,") }
function First-Diff([string]$a, [string]$b) { $m = [math]::Min($a.Length, $b.Length); for ($k = 0; $k -lt $m; $k++) { if ($a[$k] -ne $b[$k]) { return $k } }; if ($a.Length -ne $b.Length) { return $m }; return -1 }
function Err-Lines($S) { return @(Select-String -Path "$($S.wd)\e.txt" -Pattern 'CUDA error|device lost|ZE_RESULT|panic|LaunchFailed' -ErrorAction SilentlyContinue).Count }

# ---- stage runner -------------------------------------------------------------------------------
function Run-Stage([int]$N, [string]$Title, [int]$BudgetMin, [scriptblock]$Body) {
    if ($wanted -notcontains $N) { return }
    $file = Join-Path $Results "stage$N.json"
    if ((Test-Path $file) -and -not $Force) {
        $old = Get-Content $file -Raw | ConvertFrom-Json
        if (-not ($RerunFailed -and $old.status -notin @('pass'))) { Write-Host "== stage $N ($Title): result exists, skipping (use -Force / -RerunFailed)"; return }
    }
    Write-Host "== stage $N : $Title"
    $script:Cur = @{ stage = $N; title = $Title; started = (Get-Date).ToString('s'); items = @() }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    if ($script:Poisoned) { Add-Result "stage$N" 'skip' "poisoned: $script:Poisoned" }
    elseif (((Get-Date) - $runStart).TotalHours -gt $MaxHours) { Add-Result "stage$N" 'skip' "run time budget of $MaxHours h exhausted" }
    else {
        $script:StageDeadline = (Get-Date).AddMinutes($BudgetMin * $TimeoutScale)
        try { & $Body } catch { Add-Result "stage$N" 'fail' ("script error: " + $_.Exception.Message) }
        $script:StageDeadline = $null
    }
    $sw.Stop()
    $st = 'pass'
    $its = @($script:Cur.items)
    if ($its | Where-Object { $_.status -in 'fail', 'timeout' }) { $st = 'fail' }
    elseif ($its | Where-Object { $_.status -eq 'review' }) { $st = 'review' }
    elseif ($its.Count -eq 0 -or -not ($its | Where-Object { $_.status -in 'pass', 'info' })) { $st = 'skip' }
    $script:Cur.status = $st; $script:Cur.secs = [math]::Round($sw.Elapsed.TotalSeconds, 0); $script:Cur.budget_min = $BudgetMin
    Save-Json $script:Cur $file
}

# ---- pending-stage computation (decides whether the lock is needed) -----------------------------
function Test-Pending([int]$N) {
    if ($wanted -notcontains $N) { return $false }
    $file = Join-Path $Results "stage$N.json"
    if (-not (Test-Path $file) -or $Force) { return $true }
    if ($RerunFailed) { $old = Get-Content $file -Raw | ConvertFrom-Json; return ($old.status -ne 'pass') }
    return $false
}

# ---- report writer ------------------------------------------------------------------------------
function Write-Reports {
    $recs = @(); foreach ($f in (Get-ChildItem $Results -Filter 'stage*.json' -ErrorAction SilentlyContinue | Sort-Object { [int]($_.BaseName -replace '\D', '') })) { $recs += (Get-Content $f.FullName -Raw | ConvertFrom-Json) }
    $rev = ''; $rf = Join-Path $Src 'REVISION'; if (Test-Path $rf) { $rev = (Get-Content $rf -Raw).Trim() } else { try { $rev = (& git -C $Src rev-parse --short HEAD 2>$null) } catch {} }
    $overall = 'pass'
    foreach ($r in $recs) { if ($r.status -eq 'fail') { $overall = 'fail' } elseif ($r.status -eq 'review' -and $overall -eq 'pass') { $overall = 'review' } }
    $agg = [ordered]@{ generated = (Get-Date).ToString('s'); revision = $rev; overall = $overall; poisoned = $script:Poisoned; stages = $recs }
    Save-Json $agg (Join-Path $Results 'results.json')
    $md = New-Object Text.StringBuilder
    [void]$md.AppendLine("# hw-validate results"); [void]$md.AppendLine("")
    [void]$md.AppendLine("revision: $rev  overall: **$overall**  generated: $($agg.generated)"); if ($script:Poisoned) { [void]$md.AppendLine("POISONED: $script:Poisoned") }
    [void]$md.AppendLine(""); [void]$md.AppendLine("| stage | title | status | secs |"); [void]$md.AppendLine("|---|---|---|---|")
    foreach ($r in $recs) { [void]$md.AppendLine("| $($r.stage) | $($r.title) | $($r.status) | $($r.secs) |") }
    [void]$md.AppendLine("")
    foreach ($r in $recs) {
        [void]$md.AppendLine("## stage $($r.stage): $($r.title) - $($r.status)")
        foreach ($i in $r.items) {
            [void]$md.AppendLine("- [$($i.status)] $($i.name): $($i.detail)")
            if ($i.status -in 'fail', 'timeout', 'review') {
                if ($i.blame) { [void]$md.AppendLine("  - blame: $($i.blame)") }
                if ($i.isolate) { [void]$md.AppendLine("  - isolate: $($i.isolate)") }
                if ($i.log) { [void]$md.AppendLine("  - log: logs\$(Split-Path $i.log -Leaf).out/.err") }
            }
        }
        [void]$md.AppendLine("")
    }
    [IO.File]::WriteAllText((Join-Path $Results 'summary.md'), (Protect-Text $md.ToString()), (New-Object Text.UTF8Encoding($false)))
    # one small archive for scripts/hwvalidate/collect.sh (single scp)
    try {
        $list = @('results.json', 'summary.md') + @(Get-ChildItem $Results -Filter 'stage*.json' | ForEach-Object Name) + @(Get-ChildItem $logs -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -lt 262144 } | ForEach-Object { 'logs/' + $_.Name })
        $list | Set-Content (Join-Path $Results 'collect.lst') -Encoding ascii
        & tar czf (Join-Path $Results 'collect.tgz') -C $Results -T (Join-Path $Results 'collect.lst') 2>&1 | Out-Null
    } catch {}
    return $overall
}

$gpuPending = @(1..9 | Where-Object { Test-Pending $_ })
$lockHeld = $false; $hb = $null
try {
    # ================================================================== stage 0 (no GPU slot needed)
    Run-Stage 0 'environment + build' 75 {
        Reset-CfiEnv
        $free = [math]::Round((Get-PSDrive -Name ($Work.Substring(0, 1))).Free / 1GB, 1)
        Add-Result 'disk-free-gb' $(if ($free -ge $MinFreeGB) { 'pass' } else { 'fail' }) "$free GB free (need $MinFreeGB)"
        $stale = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(ollama|ollama-hv|ollama-x|ollama-divc|cfi_logits|cargo|rustc|python|llama)' } | ForEach-Object { "$($_.Name):$($_.Id)" })
        Add-Result 'stale-processes' 'info' $(if ($stale.Count) { 'present (not touched, may belong to others): ' + ($stale -join ' ') } else { 'none' })
        if (Test-Path $queue) { Add-Result 'gpu-queue' 'info' ((& powershell -NoProfile -File $queue -Action status 2>&1) -join ' | ') } else { Add-Result 'gpu-queue' 'fail' "gpu-queue.ps1 not found in the work dir" }
        # parse-check every script
        $bad = @()
        foreach ($f in (Get-ChildItem (Join-Path $Src 'scripts') -Filter *.ps1 -Recurse)) {
            $tk = $null; $er = $null; [void][System.Management.Automation.Language.Parser]::ParseFile($f.FullName, [ref]$tk, [ref]$er)
            if ($er -and $er.Count) { $bad += "$($f.Name):$($er[0].Message)" }
        }
        Add-Result 'ps1-parse-check' $(if ($bad.Count) { 'fail' } else { 'pass' }) $(if ($bad.Count) { $bad -join '; ' } else { 'all scripts parse' }) -Blame $(if ($bad.Count) { 'branch owning the script' } else { '' })
        $r = Invoke-Child 'cargo-build-release' 'cargo' 'build --release --workspace' 3000 @{} $Src
        Add-Result 'cargo-build-release' $(if ($r.rc -eq 0) { 'pass' } elseif ($r.timeout) { 'timeout' } else { 'fail' }) "rc=$($r.rc)" $r.rc $r.secs $r.log -Blame 'merge/any branch (compile error)'
        $r = Invoke-Child 'cargo-test-build' 'cargo' 'test --release --no-run --test gpu' 3000 @{} $Src
        Add-Result 'cargo-test-build' $(if ($r.rc -eq 0) { 'pass' } elseif ($r.timeout) { 'timeout' } else { 'fail' }) "rc=$($r.rc)" $r.rc $r.secs $r.log -Blame 'merge/any branch (compile error)'
        $sha = @(); foreach ($d in 'nvcuda.dll', 'cublas64_12.dll') { $p = Join-Path $rel $d; if (Test-Path $p) { $sha += "$d=$((Get-FileHash $p -Algorithm SHA256).Hash.Substring(0,16))" } else { $sha += "$d=MISSING" } }
        Add-Result 'dll-sha256' $(if ($sha -match 'MISSING') { 'fail' } else { 'info' }) ($sha -join ' ')
        $bi = Select-String -Path (Join-Path $Src 'src\cuda\xcache.rs') -Pattern 'TRANSLATOR_VERSION: u32 = (\d+)' | Select-Object -First 1
        if ((@($script:Cur.items) | Where-Object { $_.name -in 'cargo-build-release', 'cargo-test-build' -and $_.status -ne 'pass' })) { $script:Poisoned = 'stage 0 build failed: nothing to test' }
        Add-Result 'translator-version' 'info' $(if ($bi) { "source TRANSLATOR_VERSION=$($bi.Matches[0].Groups[1].Value) (expected 5 on hw-validate)" } else { 'not found' })
    }
    # ================================================================== lock for stages 1-9
    if ($gpuPending.Count -gt 0 -and -not $script:Poisoned) {
        if (-not (Test-Path $queue)) { throw "gpu-queue.ps1 not found at the work dir; refusing to run GPU stages without the lock" }
        Write-Host "acquiring GPU slot ($Id) ..."
        $out = (& powershell -NoProfile -File $queue -Action acquire -Id $Id -MaxWaitSec $QueueWaitSec 2>&1) -join ' '
        if ($out -notmatch 'acquired') { throw "could not acquire the GPU slot: $out" }
        $lockHeld = $true
        $hb = Start-Job -ArgumentList $lockDir, $Id -ScriptBlock { param($l, $me) while ($true) { try { if ((Get-Content "$l\owner" -ErrorAction SilentlyContinue) -eq $me) { (Get-Item $l).LastWriteTime = Get-Date } } catch {}; Start-Sleep 45 } }
    }

    # ================================================================== stage 1 baseline (main-equivalent switches)
    Run-Stage 1 'baseline gate (FAST_DIV=1, XMX_BLK=0 = main behaviour)' 120 {
        Reset-CfiEnv
        $be = @{ CUDA_FOR_INTEL_FAST_DIV = '1'; CFI_XMX_BLK = '0' }
        $why = 'translator-hardening or merge (baseline runs with the fix-divergence and xmx-speed features switched off)'
        Invoke-GpuTest 'baseline-hw_subst' @('hw_subst') 40 $be -Blame $why -Isolate 'rerun the named test; git stash the translator-hardening merge to bisect'
        Invoke-GpuTest 'baseline-hw_nvrtc' @('hw_nvrtc') 30 $be -Blame $why
        Invoke-GpuTest 'baseline-hw_fa_vec' @('hw_fa_vec') 20 $be -Blame $why
        $skip = @('hw_stock_ggml_kernels_compile', 'hw_subst', 'hw_nvrtc', 'hw_fa_vec', 'hw_div_sqrt', 'hw_translator_edge', 'hw_blas_l12', 'api_coverage', 'hw_xmx', 'hw_cudnn')
        Invoke-GpuTest 'baseline-rest-of-suite' @() 60 $be $skip -Blame $why
    }
    # ================================================================== stage 2 FA text check
    Run-Stage 2 'FA Ollama text check (CPU vs shim FA off vs FA on)' 90 {
        $tree = Get-OllamaTree; $ms = Get-Names
        if (-not $tree) { Add-Result 'fa-text' 'skip' 'CFI_OLLAMA_RUN not set or no ollama.exe there'; return }
        if ($ms.Count -eq 0) { Add-Result 'fa-text' 'skip' 'CFI_MODELS not set / files missing'; return }
        $short = 'The capital of France is Paris. Describe the city of Paris in detail.'
        $facts = @('The capital of France is Paris and its river is the Seine.', 'Water boils at one hundred degrees Celsius at sea level pressure.', 'The Pacific is the largest ocean and covers about a third of the planet.', 'Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar.', 'The printing press was introduced in Europe by Gutenberg around 1440.', 'Mount Everest is the highest mountain above sea level in the world.', 'Honey bees communicate the location of flowers with a waggle dance.', 'The speed of light in vacuum is close to three hundred thousand kilometers per second.')
        $sb = New-Object Text.StringBuilder; for ($i = 0; $i -lt 40; $i++) { [void]$sb.Append($facts[$i % 8] + " (note $i) ") }
        [void]$sb.Append("`nQuestion: What is the capital of France? Answer in a complete sentence, then describe the city briefly.`nAnswer:")
        $prompts = @(@('short', $short), @('long', $sb.ToString()))
        foreach ($m in $ms) {
            $mn = [IO.Path]::GetFileNameWithoutExtension($m); $texts = @{}
            foreach ($mode in @(@('cpu', 0, '0'), @('gpu_fa0', 99, '0'), @('gpu_fa1', 99, '1'))) {
                $S = Start-Ollama "fa_${mn}_$($mode[0])" $tree @{ OLLAMA_FLASH_ATTENTION = $mode[2] }
                try {
                    if (-not $S.up) { Add-Result "fa-$mn-$($mode[0])" 'fail' 'ollama server did not start' -Blame 'environment/shim load'; continue }
                    if (-not (Invoke-OllamaCreate $S $m "hv-fa-$mn")) { Add-Result "fa-$mn-$($mode[0])" 'fail' 'ollama create failed'; continue }
                    foreach ($p in $prompts) {
                        try { $r = Invoke-Gen $S $p[1] 48 $mode[1]; $texts["$($mode[0])_$($p[0])"] = $r.response; Write-Host "### $mn $($mode[0]) $($p[0]): $(($r.response -replace '\s+',' '))" } catch { $texts["$($mode[0])_$($p[0])"] = "ERROR: $($_.Exception.Message)" }
                    }
                } finally { Stop-Ollama $S; $e = Err-Lines $S; if ($e) { Add-Result "fa-$mn-$($mode[0])-stderr" 'review' "$e error lines in server log" -Log $S.wd } }
            }
            foreach ($p in 'short', 'long') {
                $c = [string]$texts["cpu_$p"]; $a = [string]$texts["gpu_fa0_$p"]; $b = [string]$texts["gpu_fa1_$p"]
                if ($b -like 'ERROR:*' -or $a -like 'ERROR:*' -or -not $b) { Add-Result "fa-$mn-$p" 'fail' "request error fa0=[$a] fa1=[$b]" -Blame 'cudnn-final/xmx-speed/fix-divergence (any GPU path)' -Isolate 'CFI_XMX=0, CFI_XMX_BLK=0, CUDA_FOR_INTEL_FAST_DIV=1'; continue }
                $d1 = First-Diff $a $b; $d2 = First-Diff $c $b
                $st = if ($d2 -lt 0) { 'pass' } elseif ($d1 -lt 0) { 'review' } else { 'review' }
                $txt = ($b -replace '\s+', ' '); if ($txt.Length -gt 90) { $txt = $txt.Substring(0, 90) }
                Add-Result "fa-$mn-$p" $st "fa1==fa0:$($d1 -lt 0) fa1==cpu:$($d2 -lt 0) first_diff_vs_cpu=$d2 text='$txt'" -Blame $(if ($st -ne 'pass') { 'read the text: coherent diverging text is expected fp drift; gibberish = FA/translator regression' } else { '' }) -Data @{ cpu = $c; fa0 = $a; fa1 = $b }
            }
        }
    }
    # ================================================================== stage 3 divergence
    Run-Stage 3 'divergence (fix-divergence)' 120 {
        Reset-CfiEnv
        $ce = { param($tag) @{ CUDA_FOR_INTEL_CACHE_DIR = (Join-Path $hw "cache_div_$tag\kc"); CFI_CACHE_DIR = (Join-Path $hw "cache_div_$tag\spv") } }
        foreach ($t in 'fast', 'rn') { New-Item -ItemType Directory -Force (Join-Path $hw "cache_div_$t\kc"), (Join-Path $hw "cache_div_$t\spv") | Out-Null; Remove-Item (Join-Path $hw "cache_div_$t\*") -Recurse -Force -ErrorAction SilentlyContinue }
        $e1 = & $ce 'fast'; $e1.CUDA_FOR_INTEL_FAST_DIV = '1'
        Invoke-GpuTest 'div_sqrt-FAST_DIV=1-must-fail' @('hw_div_sqrt_rn_f32_correctly_rounded') 15 $e1 -Expect 'fail' -Blame 'fix-divergence (opt-out switch not effective, or test cannot detect inexact div)'
        $e2 = & $ce 'rn'
        Invoke-GpuTest 'div_sqrt-default-must-pass' @('hw_div_sqrt_rn_f32_correctly_rounded') 15 $e2 -Blame 'fix-divergence (correctly-rounded flag not honoured by the driver)' -Isolate 'CUDA_FOR_INTEL_FAST_DIV=1 (opts out); then rerun'
        # quantize fuzz: test asserts d/sum identical and quant byte mismatch < 3.2e-4 of blocks
        Invoke-GpuTest 'quantize_fuzz-default' @('hw_subst_quantize_fuzz') 30 (& $ce 'rn') -Blame 'fix-divergence (div change moved quantize_q8_1 away from the native kernel)' -Isolate 'CUDA_FOR_INTEL_FAST_DIV=1'
        # auto-vs-none logit comparison
        $ps = Join-Path $Src 'scripts\divergence\run_on_box.ps1'
        $ms = Get-Names; $tree = Get-OllamaTree; $dw = Join-Path $Work 'div'
        if (-not (Test-Path $ps)) { Add-Result 'auto-vs-none-logits' 'skip' 'scripts\divergence\run_on_box.ps1 missing' }
        elseif (-not $tree -or $ms.Count -eq 0 -or -not $Hybrid) { Add-Result 'auto-vs-none-logits' 'skip' 'needs CFI_OLLAMA_RUN, CFI_MODELS and CFI_HYBRID_DIR' }
        else {
            New-Item -ItemType Directory -Force $dw | Out-Null
            $have = Test-Path (Join-Path $dw 'run\cfi_logits.exe')
            $a = "-NoProfile -File `"$ps`" -Work `"$dw`" -OllamaRun `"$tree`" -Model `"$($ms[0])`" -Hybrid `"$Hybrid`" -NoQueue -ResultsName results_hwv -NPredict 64"
            if (-not $have) { $a += ' -SkipTool' }
            $r = Invoke-Child 'auto-vs-none-logits' 'powershell' $a 5400
            $sum = Join-Path $dw 'results_hwv\scan_summary.txt'
            $rows = if (Test-Path $sum) { @(Get-Content $sum) } else { @() }
            $div = @($rows | Where-Object { $_ -match 'diverge_char=(\d+)' -and $_ -notmatch 'diverge_char=-1' }).Count
            $err = @($rows | Where-Object { $_ -match ' ERROR' }).Count
            if ($r.timeout) { Add-Result 'auto-vs-none-logits' 'timeout' '' $null $r.secs $r.log -Blame 'fix-divergence' }
            elseif (-not $rows.Count -or $err -gt 0) { Add-Result 'auto-vs-none-logits' 'fail' "scan produced no usable rows (errors=$err) rc=$($r.rc)" $r.rc $r.secs $r.log -Blame 'environment/shim load' }
            else {
                Copy-Item $sum (Join-Path $logs 'divergence_scan_summary.txt') -Force -ErrorAction SilentlyContinue
                Add-Result 'auto-vs-none-logits' 'info' "prompts=$($rows.Count - 1) diverging_auto_vs_none=$div tool_logits=$have (fewer diverging prompts than before the fix = improvement; see logs\divergence_scan_summary.txt)" $r.rc $r.secs $r.log
            }
        }
    }
    # ================================================================== stage 4 translator edge
    Run-Stage 4 'translator edge (translator-hardening)' 40 {
        Reset-CfiEnv
        Invoke-GpuTest 'hw_translator_edge' @('hw_translator_edge') 20 -Blame 'translator-hardening' -Isolate 'no runtime switch: bisect by reverting translator-hardening and rebuilding (TRANSLATOR_VERSION bump invalidates caches)'
        $r = Invoke-Child 'translator-unit-tests' 'cargo' 'test --release --lib -- ptx:: xcache' 1500 @{} $Src
        $t = Read-Out $r; $m = [regex]::Match($t, 'test result: \w+\. (\d+) passed; (\d+) failed')
        Add-Result 'translator-unit-tests' $(if ($r.rc -eq 0) { 'pass' } elseif ($r.timeout) { 'timeout' } else { 'fail' }) $m.Value $r.rc $r.secs $r.log -Blame 'translator-hardening'
        if ($Corpus -and (Test-Path $Corpus)) {
            $r = Invoke-Child 'corpus-scan-subset' 'cargo' 'test --release --lib corpus_scan -- --ignored --nocapture' 1800 @{ CFI_CORPUS_DIR = $Corpus; CFI_CORPUS_VAL = '50' } $Src
            Add-Result 'corpus-scan-subset' $(if ($r.rc -eq 0) { 'pass' } elseif ($r.timeout) { 'timeout' } else { 'fail' }) "rc=$($r.rc) (validates every 50th entry with spirv-val when it is on PATH)" $r.rc $r.secs $r.log -Blame 'translator-hardening'
        } else { Add-Result 'corpus-scan-subset' 'skip' 'CFI_CORPUS_DIR not set' }
    }
    # ================================================================== stage 5 api coverage
    Run-Stage 5 'api coverage (api-coverage)' 40 {
        Reset-CfiEnv
        Invoke-GpuTest 'api_coverage' @('api_coverage::') 20 -Blame 'api-coverage (cudart/driver additions)'
        Invoke-GpuTest 'sgemv-sdot-numerics' @('hw_blas_l12') 15 -Blame 'api-coverage (cublas_shim extra.rs)'
        $r = Invoke-Child 'shim-crates-unit-tests' 'cargo' 'test --release --workspace --exclude cuda-for-intel' 1500 @{} $Src
        Add-Result 'shim-crates-unit-tests' $(if ($r.rc -eq 0) { 'pass' } elseif ($r.timeout) { 'timeout' } else { 'fail' }) "rc=$($r.rc)" $r.rc $r.secs $r.log -Blame 'api-coverage (nvml/cufft/cusparse/cusolver/cublas shims)'
    }
    # ================================================================== stage 6 XMX
    Run-Stage 6 'XMX 2D block loads (xmx-speed)' 90 {
        Reset-CfiEnv
        Invoke-GpuTest 'hw_xmx_blk' @('hw_xmx_blk') 30 -Blame 'xmx-speed (block-load path)' -Isolate 'CFI_XMX_BLK=0 (back to gather loads); CFI_XMX=0 (no XMX)'
        Invoke-GpuTest 'hw_xmx_shapes' @('hw_xmx_shapes') 30 @{ CFI_XMX_BENCH_MODES = '0,1,2' } -Blame 'xmx-speed' -Isolate 'CFI_XMX_BLK=0'
        Invoke-GpuTest 'hw_xmx_probe-trace' @('hw_xmx_probe') 10 @{ CFI_XMX_TRACE = '1' } -Blame 'xmx-speed (driver capability probe)'
        $tr = (Read-Out @{ log = (Join-Path $logs 'hw_xmx_probe-trace') }); $n = ([regex]::Matches($tr, 'XMX', 'IgnoreCase')).Count
        Add-Result 'xmx-trace-lines' 'info' "$n lines mention XMX in the trace output" -Log (Join-Path $logs 'hw_xmx_probe-trace')
        Invoke-GpuTest 'hw_xmx_gemm-other' @('hw_xmx_gemm') 40 @{} @('hw_xmx_blk', 'hw_xmx_shapes', 'hw_xmx_perf') -Blame 'xmx-speed (xmx-gqa-fix included)'
        # Ollama f16 A/B
        $tree = Get-OllamaTree; $gg = if ($F16Model -and (Test-Path $F16Model)) { $F16Model } else { (Get-Names | Select-Object -First 1) }
        if (-not $tree -or -not $gg) { Add-Result 'ollama-f16-ab' 'skip' 'needs CFI_OLLAMA_RUN and CFI_F16_MODEL (or CFI_MODELS)'; return }
        $mn = [IO.Path]::GetFileNameWithoutExtension($gg); $texts = @{}; $n = $null
        foreach ($cfg in @(@('xmx0', @{ CFI_XMX = '0' }), @('default', @{}), @('blk0', @{ CFI_XMX_BLK = '0' }), @('trace', @{ CFI_XMX_TRACE = '1' }))) {
            $S = Start-Ollama "xmx_$($cfg[0])" $tree $cfg[1]
            try {
                if (-not $S.up -or -not (Invoke-OllamaCreate $S $gg "hv-xmx-$($cfg[0])")) { Add-Result "ollama-$($cfg[0])" 'fail' 'server/create failed'; continue }
                if (-not $n) { $c = (Invoke-Gen $S (Long-Prompt 600) 1).prompt_eval_count; $n = 600 + (631 - $c) } else { [void](Invoke-Gen $S (Long-Prompt $n) 1) }
                $r = Invoke-Gen $S (Long-Prompt $n 3) 48
                $pp = [math]::Round($r.prompt_eval_count * 1e9 / [math]::Max([double]1, [double]$r.prompt_eval_duration), 1); $dd = [math]::Round($r.eval_count * 1e9 / [math]::Max([double]1, [double]$r.eval_duration), 1)
                $texts[$cfg[0]] = $r.response
                $x = ''; if ($cfg[0] -eq 'trace') { $x = " xmx_trace_lines=$(@(Select-String -Path "$($S.wd)\e.txt" -Pattern 'xmx' -ErrorAction SilentlyContinue).Count)" }
                Add-Result "ollama-$mn-$($cfg[0])" 'info' "prefill_tps=$pp decode_tps=$dd tokens=$($r.prompt_eval_count)+$($r.eval_count)$x" -Data @{ prefill_tps = $pp; decode_tps = $dd }
            } catch { Add-Result "ollama-$($cfg[0])" 'fail' $_.Exception.Message -Blame 'xmx-speed' -Isolate 'CFI_XMX=0' }
            finally { Stop-Ollama $S; $e = Err-Lines $S; if ($e) { Add-Result "ollama-$($cfg[0])-stderr" 'review' "$e error lines in server log" -Log $S.wd } }
        }
        foreach ($k in 'default', 'blk0', 'trace') {
            if ($texts.ContainsKey('xmx0') -and $texts.ContainsKey($k)) {
                $d = First-Diff $texts['xmx0'] $texts[$k]
                Add-Result "text-xmx0-vs-$k" $(if ($d -lt 0) { 'pass' } else { 'review' }) "first_diff_char=$d (f16 accumulation order differs with XMX; coherent text = fine)" -Blame 'xmx-speed' -Isolate 'CFI_XMX=0 / CFI_XMX_BLK=0'
            }
        }
    }
    # ================================================================== stage 7 cuDNN
    Run-Stage 7 'cuDNN (cudnn-final)' 120 {
        Reset-CfiEnv
        Invoke-GpuTest 'hw_cudnn' @('hw_cudnn') 40 -Blame 'cudnn-final' -Isolate 'CFI_CUDNN is not a runtime switch: omit cudnn64_*.dll from the run dir to disable the shim'
        $py = if ($TorchPy) { $TorchPy } else { Join-Path $Work 'torch\torch222\Scripts\python.exe' }
        $cd = Join-Path $hw 'cd'; $run = Join-Path $cd 'run'
        if (-not (Test-Path $py)) { Add-Result 'cudnn-gate-final' 'skip' 'no torch python (CFI_TORCH_PY)'; return }
        New-Item -ItemType Directory -Force $run | Out-Null
        Copy-Item (Join-Path $rel 'nvcuda.dll') $run -Force
        $r = Invoke-Child 'cudnn-v8-build' 'cargo' "build --release -p cudnn_shim --features v8 --target-dir `"$hw\target_v8`"" 2400 @{} $Src
        if ($r.rc -ne 0) { Add-Result 'cudnn-v8-build' 'fail' "rc=$($r.rc)" $r.rc $r.secs $r.log -Blame 'cudnn-final'; return }
        Copy-Item "$hw\target_v8\release\cudnn64_9.dll" (Join-Path $run 'cudnn64_8.dll') -Force
        Copy-Item "$hw\target_v8\release\cudnn64_9.dll" (Join-Path $run 'cudnn64_9.dll') -Force
        Copy-Item (Join-Path $rel 'cublas64_12.dll') $run -Force
        if ($Hybrid) { New-Item -ItemType Directory -Force "$run\hybrid" | Out-Null; Copy-Item "$Hybrid\*" "$run\hybrid" -Force -Recurse }
        $srcLink = Join-Path $cd 'src'
        if (-not (Test-Path $srcLink)) { New-Item -ItemType Junction -Path $srcLink -Target $Src | Out-Null }
        $gate = Join-Path $cd 'gate'; Remove-Item "$gate\*" -Force -ErrorAction SilentlyContinue
        $r = Invoke-Child 'cudnn-gate-final' 'powershell' "-NoProfile -File `"$Src\scripts\cudnn\gate_final.ps1`" -NoQueue -SkipSuite -Dir `"$cd`" -Gate `"$gate`"" 7200 @{}
        $sumf = Join-Path $gate 'summary.txt'; $lines = if (Test-Path $sumf) { @(Get-Content $sumf) } else { @() }
        $bad = @($lines | Where-Object { $_ -match 'TIMEOUT' -or $_ -match 'rc=(?!0\b)\d+' })
        Copy-Item $sumf (Join-Path $logs 'cudnn_gate_summary.txt') -Force -ErrorAction SilentlyContinue
        if ($r.timeout) { Add-Result 'cudnn-gate-final' 'timeout' '' $null $r.secs $r.log -Blame 'cudnn-final' }
        elseif (-not $lines.Count) { Add-Result 'cudnn-gate-final' 'fail' 'gate produced no summary' $r.rc $r.secs $r.log -Blame 'environment (torch python / DLLs)' }
        else { Add-Result 'cudnn-gate-final' $(if ($bad.Count) { 'fail' } else { 'pass' }) ("steps=$($lines.Count) bad=$($bad.Count) " + ($bad -join '; ')) $r.rc $r.secs $r.log -Blame 'cudnn-final' -Isolate 'remove cudnn64_*.dll from the run dir (torch falls back to aten)' }
    }
    # ================================================================== stage 8 perf matrix
    Run-Stage 8 'perf matrix (631 prefill, 48 decode)' 240 {
        $tree = Get-OllamaTree; $ms = Get-Names
        if (-not $tree -or $ms.Count -eq 0) { Add-Result 'perf-matrix' 'skip' 'needs CFI_OLLAMA_RUN and CFI_MODELS'; return }
        $cfgs = @(@('fa0_ld0', '0', '0'), @('fa0_ld1', '0', '1'), @('fa1_ld0', '1', '0'), @('fa1_ld1', '1', '1'))
        foreach ($m in $ms) {
            $mn = [IO.Path]::GetFileNameWithoutExtension($m); $n = $null; $cpuText = $null
            # CPU reference (also calibrates the prompt to 631 tokens)
            $S = Start-Ollama "perf_${mn}_cpu" $tree @{}
            try {
                if ($S.up -and (Invoke-OllamaCreate $S $m "hv-perf-$mn")) {
                    $c = (Invoke-Gen $S (Long-Prompt 600) 1 0).prompt_eval_count; $n = 600 + (631 - $c)
                    $cpuText = (Invoke-Gen $S (Long-Prompt $n 3) 48 0).response   # same prompt as GPU run 1
                    Add-Result "perf-$mn-cpu-ref" 'info' "calibrated words=$n"
                } else { Add-Result "perf-$mn-cpu-ref" 'fail' 'server/create failed' }
            } catch { Add-Result "perf-$mn-cpu-ref" 'fail' $_.Exception.Message } finally { Stop-Ollama $S }
            if (-not $n) { continue }
            foreach ($cfg in $cfgs) {
                $S = Start-Ollama "perf_${mn}_$($cfg[0])" $tree @{ OLLAMA_FLASH_ATTENTION = $cfg[1]; CFI_OPT_LD16 = $cfg[2] }
                try {
                    if (-not $S.up -or -not (Invoke-OllamaCreate $S $m "hv-perf-$mn")) { Add-Result "perf-$mn-$($cfg[0])" 'fail' 'server/create failed'; continue }
                    [void](Invoke-Gen $S (Long-Prompt $n 2) 8)         # warm: compile all kernels for these shapes
                    $best = $null
                    foreach ($i in 1..2) {
                        $r = Invoke-Gen $S (Long-Prompt $n (2 + $i)) 48
                        $pp = [math]::Round($r.prompt_eval_count * 1e9 / [math]::Max([double]1, [double]$r.prompt_eval_duration), 1); $dd = [math]::Round($r.eval_count * 1e9 / [math]::Max([double]1, [double]$r.eval_duration), 1)
                        if ($i -eq 1) { $gpuText = $r.response }
                        if (-not $best -or $dd -gt $best.dd) { $best = @{ pp = $pp; dd = $dd; pe = $r.prompt_eval_count; ev = $r.eval_count } }
                    }
                    $d = First-Diff $cpuText $gpuText
                    $st = if ($d -lt 0) { 'pass' } else { 'review' }
                    Add-Result "perf-$mn-$($cfg[0])" $st "prefill_tps=$($best.pp) decode_tps=$($best.dd) tokens=$($best.pe)+$($best.ev) text_eq_cpu=$($d -lt 0) first_diff=$d" -Blame $(if ($st -ne 'pass') { 'compare the 4 configs: only ld1 differs -> perf-depth LD16; only fa1 differs -> FA; all differ -> fix-divergence/xmx/translator' } else { '' }) -Isolate 'CFI_OPT_LD16=0, OLLAMA_FLASH_ATTENTION=0, CFI_XMX=0, CUDA_FOR_INTEL_FAST_DIV=1' -Data @{ prefill_tps = $best.pp; decode_tps = $best.dd; model = $mn; config = $cfg[0] }
                } catch { Add-Result "perf-$mn-$($cfg[0])" 'fail' $_.Exception.Message -Blame 'any GPU path' }
                finally { Stop-Ollama $S; $e = Err-Lines $S; if ($e) { Add-Result "perf-$mn-$($cfg[0])-stderr" 'review' "$e error lines in server log (LaunchFailed/device lost?)" -Log $S.wd } }
            }
        }
    }
    # ================================================================== stage 9 installer (+ torch if hw-prep present)
    Run-Stage 9 'installer self-test (+ torch)' 40 {
        Reset-CfiEnv
        $cfi = Join-Path $Src 'scripts\cfi.ps1'; $scr = Join-Path $hw 'install_scratch'
        if (Test-Path $scr) { Remove-Item $scr -Recurse -Force -ErrorAction SilentlyContinue }
        $ba = "-NoProfile -File `"$cfi`""
        $all = ''
        $steps = @(
            @('install-dryrun', "$ba install -DryRun -InstallDir `"$scr`" -BinDir `"$rel`"", 300),
            @('install-real', "$ba install -InstallDir `"$scr`" -BinDir `"$rel`" -SkipSelfTest", 600),
            @('doctor', "$ba doctor -InstallDir `"$scr`"", 300),
            @('selftest', "$ba selftest -InstallDir `"$scr`"", 900))
        foreach ($s in $steps) {
            $r = Invoke-Child "cfi-$($s[0])" 'powershell' $s[1] $s[2]
            $all += (Read-Out $r)
            Add-Result "cfi-$($s[0])" $(if ($r.rc -eq 0) { 'pass' } elseif ($r.timeout) { 'timeout' } else { 'fail' }) "rc=$($r.rc)" $r.rc $r.secs $r.log -Blame 'packaging/installer (main)' -Isolate 'run the same cfi.ps1 command by hand'
        }
        $leaks = @()
        foreach ($pair in @(@('username', $env:USERNAME), @('hostname', $env:COMPUTERNAME))) { if ($pair[1] -and $pair[1].Length -ge 3 -and $all -match [regex]::Escape($pair[1])) { $leaks += $pair[0] } }
        if ($all -match '\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(\d{1,3}\.){3}\d{1,3}\b') { $leaks += 'ipv4' }
        Add-Result 'installer-output-leak-grep' $(if ($leaks.Count) { 'fail' } else { 'pass' }) $(if ($leaks.Count) { 'output contains: ' + ($leaks -join ',') + ' (see logs; doctor/install must print no username/host/IP)' } else { 'no username/host/IP in install, doctor or selftest output' }) -Blame 'packaging/installer (cfi-doctor.ps1, Write-Safe)'
        $r = Invoke-Child 'cfi-uninstall' 'powershell' "$ba uninstall -InstallDir `"$scr`"" 300
        $gone = -not (Test-Path (Join-Path $scr 'manifest.json'))
        Add-Result 'cfi-uninstall-roundtrip' $(if ($r.rc -eq 0 -and $gone) { 'pass' } else { 'fail' }) "rc=$($r.rc) manifest_removed=$gone" $r.rc $r.secs $r.log -Blame 'packaging/installer'
        $tp = Join-Path $Src 'scripts\hwvalidate\torch_stage.ps1'
        if (Test-Path $tp) {
            $r = Invoke-Child 'torch-stage' 'powershell' "-NoProfile -File `"$tp`" -Work `"$Work`" -Src `"$Src`" -Results `"$Results`"" 5400
            Add-Result 'torch-stage' $(if ($r.rc -eq 0) { 'pass' } elseif ($r.timeout) { 'timeout' } else { 'fail' }) "rc=$($r.rc) (details in its own logs)" $r.rc $r.secs $r.log -Blame 'hw-prep / cudnn-final / api-coverage'
        } else { Add-Result 'torch-stage' 'skip' 'scripts\hwvalidate\torch_stage.ps1 not present (hw-prep not merged)' }
    }
}
catch { Write-Host "FATAL: $($_.Exception.Message)"; $script:Fatal = $_.Exception.Message }
finally {
    if ($hb) { Stop-Job $hb -ErrorAction SilentlyContinue; Remove-Job $hb -Force -ErrorAction SilentlyContinue }
    if ($lockHeld) { & powershell -NoProfile -File $queue -Action release -Id $Id | Out-Null }
    $overall = Write-Reports
    Write-Host "overall: $overall   results: $Results\summary.md"
}

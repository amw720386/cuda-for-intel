<#
.SYNOPSIS
  Warm prefill/decode tok/s of stock Ollama on the shim, several env configs per model, with a greedy text check against the CPU
  backend. Runs ON the box; takes the GPU-queue slot once per model (released between models so other jobs get turns).

.DESCRIPTION
  Per model: a CPU-backend server calibrates a ~631-token prompt and records the reference text (48 greedy tokens). Then, per
  config, one GPU server: warm-up request (prompt variant 2, compiles every kernel), then measured requests with variants 3 and 4
  (never seen by the KV prefix cache, so the whole prompt is evaluated). Reports the best of the measured runs and whether the
  variant-3 text equals the CPU text (first differing character otherwise), plus a free-text story prompt compared the same way
  (`story_diff`, -1 = identical).
  Configs: `name=K1:V1;K2:V2` entries separated by `|`, e.g. `base=|fast=CUDA_FOR_INTEL_MMVQ_FAST:1`. Every config runs with
  OLLAMA_FLASH_ATTENTION 0 and 1 unless -Fa says otherwise.

.EXAMPLE
  powershell -NoProfile -File bench-decode.ps1 -Tree <ollama tree with the shim> -Models a.gguf,b.gguf -Out <dir> -Configs 'base=|fast=CUDA_FOR_INTEL_MMVQ_FAST:1'
#>
param(
    [Parameter(Mandatory)][string]$Tree,
    [Parameter(Mandatory)][string]$Models,
    [Parameter(Mandatory)][string]$Out,
    [string]$Configs = 'base=',
    [string]$Fa = '0,1',
    [int]$Predict = 48,
    [int]$Runs = 2,
    [string]$Work = $(if ($env:CFI_WORK) { $env:CFI_WORK } else { Join-Path $env:USERPROFILE 'cfi-work' }),
    [string]$Hybrid = $env:CFI_HYBRID_DIR,
    [string]$Id = 'cfi-bench',
    [switch]$NoCpu
)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force $Out | Out-Null
$queue = Join-Path $Work 'gpu-queue.ps1'; $lockDir = Join-Path $Work 'gpu.lock'
$exe = Join-Path $Tree 'ollama-bench.exe'
if (-not (Test-Path $exe)) { Copy-Item (Join-Path $Tree 'ollama.exe') $exe }

function Get-DescendantIds([int]$ParentId) {
    $ids = @()
    foreach ($c in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -ErrorAction SilentlyContinue)) { $ids += [int]$c.ProcessId; $ids += Get-DescendantIds ([int]$c.ProcessId) }
    return $ids
}
function Start-Srv([string]$Tag, [hashtable]$E) {
    $port = Get-Random -Minimum 21000 -Maximum 29000
    $wd = Join-Path $Out "wd_$Tag"; Remove-Item $wd -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $wd, "$wd\kc", "$wd\spv" | Out-Null
    $g = Join-Path $Tree 'lib\ollama\cuda_v13'
    $base = @{ PATH = "$Tree;$Tree\lib\ollama;$g;$env:PATH"; OLLAMA_LIBRARY_PATH = "$Tree\lib\ollama;$g"; OLLAMA_VULKAN = '0'
        OLLAMA_MODELS = (Join-Path $Out 'models'); OLLAMA_HOST = "127.0.0.1:$port"; OLLAMA_NUM_PARALLEL = '1'; OLLAMA_LOAD_TIMEOUT = '120m'
        OLLAMA_KEEP_ALIVE = '30m'; CUDA_FOR_INTEL_CACHE_DIR = "$wd\kc"; CFI_CACHE_DIR = "$wd\spv"; NEO_CACHE_PERSISTENT = '0'; CUDA_FOR_INTEL_LOG = '1' }
    if ($Hybrid) { $base.CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid }
    $saved = @{}
    foreach ($h in @($base, $E)) { foreach ($k in $h.Keys) { if (-not $saved.ContainsKey($k)) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process') }; [Environment]::SetEnvironmentVariable($k, [string]$h[$k], 'Process') } }
    $p = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $wd -FilePath $exe -ArgumentList 'serve' -RedirectStandardOutput "$wd\o.txt" -RedirectStandardError "$wd\e.txt"
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
    $up = $false
    for ($i = 0; $i -lt 40 -and -not $up; $i++) { Start-Sleep 2; if ((Get-Content "$wd\e.txt" -Raw -ErrorAction SilentlyContinue) -match "Listening on 127\.0\.0\.1:$port") { $up = $true } elseif ($p.HasExited) { break } }
    return @{ proc = $p; port = $port; wd = $wd; up = $up; model = $null }
}
function Stop-Srv($S) {
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$($S.port)/api/generate" -Method Post -Body (@{ model = $S.model; keep_alive = 0 } | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 30 | Out-Null } catch {}
    Start-Sleep 3
    foreach ($id in @(Get-DescendantIds $S.proc.Id)) { try { [Diagnostics.Process]::GetProcessById($id).Kill() } catch {} }
    try { if (-not $S.proc.HasExited) { $S.proc.Kill() } } catch {}
    [void]$S.proc.WaitForExit(30000); Start-Sleep 2
}
function Create($S, [string]$Gguf, [string]$Name) {
    "FROM $Gguf" | Set-Content "$($S.wd)\Mf"; $S.model = $Name
    $env:OLLAMA_HOST = "127.0.0.1:$($S.port)"
    $p = Start-Process -PassThru -NoNewWindow -FilePath $exe -ArgumentList "create $Name -f `"$($S.wd)\Mf`"" -RedirectStandardOutput "$($S.wd)\c.out" -RedirectStandardError "$($S.wd)\c.err"
    $null = $p.Handle
    [void]$p.WaitForExit(900000); return ($p.ExitCode -eq 0)
}
function Gen($S, [string]$Prompt, [int]$N, [int]$NumGpu = 99) {
    $body = @{ model = $S.model; prompt = $Prompt; stream = $false; raw = $true; options = @{ num_predict = $N; temperature = 0; seed = 0; top_k = 1; repeat_penalty = 1.0; num_gpu = $NumGpu; num_ctx = 2048; num_batch = 512 } } | ConvertTo-Json -Depth 4
    return Invoke-RestMethod -Uri "http://127.0.0.1:$($S.port)/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 3600
}
$story = "Write a short story about a lighthouse keeper who finds a message in a bottle. Describe the weather, the sea and what the message says."
function Long-Prompt([int]$words, [int]$V) { return ("$V " + ((@('lorem') * $words) -join ' ') + "`nCount from 1 to 300 separated by commas: 1, 2, 3,") }
function First-Diff([string]$a, [string]$b) { $m = [math]::Min($a.Length, $b.Length); for ($k = 0; $k -lt $m; $k++) { if ($a[$k] -ne $b[$k]) { return $k } }; if ($a.Length -ne $b.Length) { return $m }; return -1 }
function Tps($r) { return @([math]::Round($r.prompt_eval_count * 1e9 / [math]::Max([double]1, [double]$r.prompt_eval_duration), 1), [math]::Round($r.eval_count * 1e9 / [math]::Max([double]1, [double]$r.eval_duration), 1)) }

$cfgs = @()
foreach ($c in ($Configs -split '\|')) {
    $name, $rest = $c -split '=', 2
    $h = @{}
    foreach ($kv in ($rest -split ';')) { if ($kv -match '^([^:]+):(.*)$') { $h[$matches[1]] = $matches[2] } }
    $cfgs += , @($name, $h)
}
$results = @()
foreach ($m in ($Models -split ',')) {
    $mn = [IO.Path]::GetFileNameWithoutExtension($m)
    $n = 600; $cpuText = $null; $cpuStory = $null
    if (-not $NoCpu) {
        $S = Start-Srv "${mn}_cpu" @{}
        try {
            if ($S.up -and (Create $S $m "b-$mn")) {
                $c = (Gen $S (Long-Prompt 600 1) 1 0).prompt_eval_count; $n = 600 + (631 - $c)
                $cpuText = (Gen $S (Long-Prompt $n 3) $Predict 0).response
                $cpuStory = (Gen $S $story $Predict 0).response
            }
        } catch { Write-Host "cpu ref failed: $($_.Exception.Message)" } finally { Stop-Srv $S }
    }
    $o = (& powershell -NoProfile -File $queue -Action acquire -Id $Id -MaxWaitSec 7200 2>&1) -join ' '
    if ($o -notmatch 'acquired') { Write-Host "no GPU slot: $o"; continue }
    $hb = Start-Job -ArgumentList $lockDir, $Id -ScriptBlock { param($l, $me) while ($true) { try { if ((Get-Content "$l\owner" -ErrorAction SilentlyContinue) -eq $me) { (Get-Item $l).LastWriteTime = Get-Date } } catch {}; Start-Sleep 45 } }
    try {
        foreach ($faOn in ($Fa -split ',')) {
            foreach ($cfg in $cfgs) {
                $E = @{ OLLAMA_FLASH_ATTENTION = $faOn } + $cfg[1]
                $tag = "${mn}_fa${faOn}_$($cfg[0])"
                $S = Start-Srv $tag $E
                $rec = [ordered]@{ model = $mn; fa = $faOn; config = $cfg[0] }
                try {
                    if (-not $S.up -or -not (Create $S $m "b-$mn")) { $rec.error = 'server/create failed'; continue }
                    [void](Gen $S (Long-Prompt $n 2) 8)
                    $pp = @(); $dd = @(); $text = $null
                    foreach ($i in 1..$Runs) {
                        $r = Gen $S (Long-Prompt $n (2 + $i)) $Predict
                        $t = Tps $r; $pp += $t[0]; $dd += $t[1]
                        if ($i -eq 1) { $text = $r.response; $rec.tokens = "$($r.prompt_eval_count)+$($r.eval_count)" }
                    }
                    $rec.prefill = ($pp | Measure-Object -Maximum).Maximum; $rec.decode = ($dd | Measure-Object -Maximum).Maximum
                    $rec.prefill_all = $pp -join '/'; $rec.decode_all = $dd -join '/'
                    $rec.text_eq_cpu = if ($cpuText) { (First-Diff $cpuText $text) -lt 0 } else { $null }
                    $rec.first_diff = if ($cpuText) { First-Diff $cpuText $text } else { $null }
                    $rec.text = $text
                    if ($cpuStory) {
                        $st = (Gen $S $story $Predict).response
                        $rec.story_first_diff = First-Diff $cpuStory $st
                        $rec.story = $st
                    }
                } catch { $rec.error = $_.Exception.Message } finally {
                    Stop-Srv $S
                    $rec.server_errors = @(Select-String -Path "$($S.wd)\e.txt" -Pattern 'CUDA error|device lost|ZE_RESULT|panic|LaunchFailed' -ErrorAction SilentlyContinue).Count
                    $rec.subst_rejected = @(Select-String -Path "$($S.wd)\cuda-for-intel-export.log" -Pattern 'subst:.*(rejected|failed|MISMATCH)' -ErrorAction SilentlyContinue).Count
                }
                Write-Host ("{0,-14} fa{1} {2,-10} prefill {3,8} decode {4,6} (runs {5} / {6}) text_eq_cpu={7} first_diff={8} story_diff={9} err={10}" -f $mn, $faOn, $cfg[0], $rec.prefill, $rec.decode, $rec.prefill_all, $rec.decode_all, $rec.text_eq_cpu, $rec.first_diff, $rec.story_first_diff, $rec.server_errors)
                $results += [pscustomobject]$rec
            }
        }
    } finally {
        Stop-Job $hb -ErrorAction SilentlyContinue; Remove-Job $hb -Force -ErrorAction SilentlyContinue
        & powershell -NoProfile -File $queue -Action release -Id $Id | Out-Null
    }
    if ($cpuText) { $results += [pscustomobject]@{ model = $mn; config = 'cpu'; text = $cpuText; story = $cpuStory } }
}
($results | ConvertTo-Json -Depth 4) | Set-Content (Join-Path $Out 'bench.json') -Encoding utf8
'BENCH_DONE'

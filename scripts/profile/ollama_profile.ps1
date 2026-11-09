<#
.SYNOPSIS
  CFI_PROFILE snapshots of stock Ollama prefill and decode (checklist item 7). Runs ON the box and takes the GPU-queue slot
  itself unless -NoQueue (the caller already holds it).

.DESCRIPTION
  For every model and flash-attention setting, one Ollama server with the shim and CFI_PROFILE=1 serves this sequence:
    warm   (long prompt, 8 tokens: compiles every kernel for the shapes)
    short  (15-token prompt, 1 token)       -> prefill of a short prompt
    long   (~630-token prompt, 1 token)     -> prefill of a long prompt
    decode (short prompt, 64 tokens)        -> decode
  CFI_PROFILE_OUT is cumulative per runner process and is rewritten every CFI_PROFILE_FLUSH_MS, so the script copies it
  after each request (<tag>_<step>.tsv). Diff consecutive snapshots on the Mac with
  `python3 scripts/profile/summarize.py B.tsv --base A.tsv`. A second, unprofiled server first gives the real tok/s of the same
  sequence twice (timings.json: `cold.*` includes kernel translation on first use, `warm.*` is the repeat).
  Nothing is killed except the Ollama server tree this script started (after asking it to unload the model).

.EXAMPLE
  powershell -NoProfile -File scripts\profile\ollama_profile.ps1 -Tree <ollama tree with the shim> -Models a.gguf,b.gguf -Out <dir>
#>
param(
    [Parameter(Mandatory)][string]$Tree,
    [Parameter(Mandatory)][string]$Models,
    [Parameter(Mandatory)][string]$Out,
    [string]$Work = $(if ($env:CFI_WORK) { $env:CFI_WORK } else { Join-Path $env:USERPROFILE 'cfi-work' }),
    [string]$Hybrid = $env:CFI_HYBRID_DIR,
    [string]$Fa = '0,1',
    [hashtable]$ExtraEnv = @{},
    [string]$Id = 'cfi-profile',
    [switch]$NoQueue,
    [switch]$NoUnprofiled
)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force $Out | Out-Null
$queue = Join-Path $Work 'gpu-queue.ps1'; $lockDir = Join-Path $Work 'gpu.lock'

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
        OLLAMA_KEEP_ALIVE = '30m'; CUDA_FOR_INTEL_CACHE_DIR = "$wd\kc"; CFI_CACHE_DIR = "$wd\spv"; NEO_CACHE_PERSISTENT = '0' }
    if ($Hybrid) { $base.CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid }
    $saved = @{}
    foreach ($h in @($base, $ExtraEnv, $E)) { foreach ($k in $h.Keys) { if (-not $saved.ContainsKey($k)) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process') }; [Environment]::SetEnvironmentVariable($k, [string]$h[$k], 'Process') } }
    $exe = Join-Path $Tree 'ollama.exe'
    $p = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $wd -FilePath $exe -ArgumentList 'serve' -RedirectStandardOutput "$wd\o.txt" -RedirectStandardError "$wd\e.txt"
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
    $up = $false
    for ($i = 0; $i -lt 40 -and -not $up; $i++) { Start-Sleep 2; if ((Get-Content "$wd\e.txt" -Raw -ErrorAction SilentlyContinue) -match "Listening on 127\.0\.0\.1:$port") { $up = $true } elseif ($p.HasExited) { break } }
    return @{ proc = $p; port = $port; wd = $wd; up = $up; exe = $exe; model = $null; e = $E; saved = $base }
}
function Stop-Srv($S) {
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$($S.port)/api/generate" -Method Post -Body (@{ model = $S.model; keep_alive = 0 } | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 30 | Out-Null } catch {}
    Start-Sleep 5
    foreach ($id in @(Get-DescendantIds $S.proc.Id)) { try { [Diagnostics.Process]::GetProcessById($id).Kill() } catch {} }
    try { if (-not $S.proc.HasExited) { $S.proc.Kill() } } catch {}
    [void]$S.proc.WaitForExit(30000); Start-Sleep 3
}
function Create($S, [string]$Gguf, [string]$Name) {
    "FROM $Gguf" | Set-Content "$($S.wd)\Mf"; $S.model = $Name
    $env:OLLAMA_HOST = "127.0.0.1:$($S.port)"
    $p = Start-Process -PassThru -NoNewWindow -FilePath $S.exe -ArgumentList "create $Name -f `"$($S.wd)\Mf`"" -RedirectStandardOutput "$($S.wd)\c.out" -RedirectStandardError "$($S.wd)\c.err"
    [void]$p.WaitForExit(900000); return ($p.ExitCode -eq 0)
}
function Gen($S, [string]$Prompt, [int]$Predict) {
    $body = @{ model = $S.model; prompt = $Prompt; stream = $false; raw = $true; options = @{ num_predict = $Predict; temperature = 0; seed = 0; top_k = 1; num_gpu = 99; num_ctx = 2048; num_batch = 512 } } | ConvertTo-Json -Depth 4
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$($S.port)/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 5400
    $sw.Stop()
    return [ordered]@{ wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 2); prompt_tokens = $r.prompt_eval_count; prompt_s = [math]::Round($r.prompt_eval_duration / 1e9, 3)
        prefill_tps = [math]::Round($r.prompt_eval_count * 1e9 / [math]::Max([double]1, [double]$r.prompt_eval_duration), 1); eval_tokens = $r.eval_count
        decode_tps = [math]::Round($r.eval_count * 1e9 / [math]::Max([double]1, [double]$r.eval_duration), 1); load_s = [math]::Round($r.load_duration / 1e9, 2) }
}
$short = 'The capital of France is Paris. Describe the city of Paris in detail.'
$long = '1 ' + ((@('lorem') * 600) -join ' ') + "`nCount from 1 to 300 separated by commas: 1, 2, 3,"
$steps = @(@('warm', $long, 8), @('short', $short, 1), @('long', $long, 1), @('decode', $short, 64))

$held = $false; $hb = $null
try {
    if (-not $NoQueue) {
        $o = (& powershell -NoProfile -File $queue -Action acquire -Id $Id -MaxWaitSec 14400 2>&1) -join ' '
        if ($o -notmatch 'acquired') { throw "could not acquire the GPU slot: $o" }
        $held = $true
        $hb = Start-Job -ArgumentList $lockDir, $Id -ScriptBlock { param($l, $me) while ($true) { try { if ((Get-Content "$l\owner" -ErrorAction SilentlyContinue) -eq $me) { (Get-Item $l).LastWriteTime = Get-Date } } catch {}; Start-Sleep 45 } }
    }
    $all = [ordered]@{}
    foreach ($m in ($Models -split ',')) {
        $mn = [IO.Path]::GetFileNameWithoutExtension($m)
        foreach ($fa in ($Fa -split ',')) {
            $tag = "${mn}_fa$fa"
            $modes = @(@('prof', @{ CFI_PROFILE = '1'; CFI_PROFILE_OUT = "$Out\$tag.live.tsv"; CFI_PROFILE_FLUSH_MS = '100'; OLLAMA_FLASH_ATTENTION = $fa }))
            if (-not $NoUnprofiled) { $modes = @(@('plain', @{ OLLAMA_FLASH_ATTENTION = $fa })) + $modes }
            foreach ($mode in $modes) {
                $S = Start-Srv "${tag}_$($mode[0])" $mode[1]
                $rec = [ordered]@{}
                try {
                    if (-not $S.up -or -not (Create $S $m "prof-$mn")) { $rec.error = 'server/create failed'; continue }
                    foreach ($round in @('cold', 'warm')) {
                        if ($mode[0] -eq 'prof' -and $round -eq 'warm') { break }
                        foreach ($st in $steps) {
                            try { $rec["$round.$($st[0])"] = Gen $S $st[1] $st[2] } catch { $rec["$round.$($st[0])"] = @{ error = $_.Exception.Message } }
                            if ($mode[0] -eq 'prof') { Start-Sleep 2; Copy-Item "$Out\$tag.live.tsv" "$Out\${tag}_$($st[0]).tsv" -Force -ErrorAction SilentlyContinue }
                            Write-Host "$tag $($mode[0]) $round $($st[0]): $(($rec["$round.$($st[0])"] | ConvertTo-Json -Compress))"
                        }
                    }
                } finally { Stop-Srv $S; $rec.server_errors = @(Select-String -Path "$($S.wd)\e.txt" -Pattern 'CUDA error|device lost|ZE_RESULT|panic|LaunchFailed' -ErrorAction SilentlyContinue).Count; $all["$tag.$($mode[0])"] = $rec }
            }
        }
    }
    ($all | ConvertTo-Json -Depth 6) | Set-Content (Join-Path $Out 'timings.json') -Encoding utf8
} finally {
    if ($hb) { Stop-Job $hb -ErrorAction SilentlyContinue; Remove-Job $hb -Force -ErrorAction SilentlyContinue }
    if ($held) { & powershell -NoProfile -File $queue -Action release -Id $Id | Out-Null }
}
'PROFILE_DONE'

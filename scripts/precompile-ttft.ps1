<#
.SYNOPSIS
  Cold first-token time of stock Ollama under cfi-run: empty caches, after cfi-precompile, and warm. Runs ON the GPU machine
  and takes the GPU-queue slot itself unless -NoQueue.
.DESCRIPTION
  Modes (comma list, run in order):
    cold     empty SPIR-V and native caches; one request (load + ~310-token prefill + 1 token), then a second request
    warm     the cold run's caches again (a later run)
    pre      empty caches, cfi-precompile -PreArgs <Tree>\lib\ollama\cuda_v13, then the cold request
    prewarm  the precompiled caches again
  Servers start through cfi-run.exe on a random port with a private model store in -Out; only the server tree this script
  started is stopped. Results go to <Out>\results.jsonl, precompile output to <Out>\precompile_<Tag>.txt.
.EXAMPLE
  powershell -File scripts\precompile-ttft.ps1 -Tree <stock ollama> -Bin <install>\bin -Gguf <model.gguf> -Out <dir> -Hybrid <companion dir> -PreArgs "--gguf <model.gguf> --native-skip flash_attn"
#>
param([string]$Tag = 'm', [Parameter(Mandatory)][string]$Gguf, [Parameter(Mandatory)][string]$Tree, [Parameter(Mandatory)][string]$Bin,
  [Parameter(Mandatory)][string]$Out, [string]$Hybrid = $env:CUDA_FOR_INTEL_HYBRID_DIR, [string]$Modes = 'cold,warm,pre,prewarm',
  [switch]$NoQueue, [string]$PreArgs = '', [string]$Work = $(if ($env:CFI_WORK) { $env:CFI_WORK } else { Join-Path $env:USERPROFILE 'cfi-work' }))
$ProgressPreference = 'SilentlyContinue'; $ErrorActionPreference = 'Continue'
$w = $Work; $me = $Out; $bin = $Bin; $tree = $Tree
$out = $Out; New-Item -ItemType Directory -Force $out, "$me\models" | Out-Null
$queue = "$w\gpu-queue.ps1"; $lockDir = "$w\gpu.lock"; $id = 'precomp-agent'
function Get-DescendantIds([int]$ParentId) { $ids = @(); foreach ($c in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -EA SilentlyContinue)) { $ids += [int]$c.ProcessId; $ids += Get-DescendantIds ([int]$c.ProcessId) }; $ids }
function Set-Env([hashtable]$E) { foreach ($k in $E.Keys) { [Environment]::SetEnvironmentVariable($k, [string]$E[$k], 'Process') } }
function Run-Server([string]$Name, [string]$Cache) {
  $port = Get-Random -Minimum 21000 -Maximum 29000
  $wd = "$out\wd_${Tag}_$Name"; Remove-Item $wd -Recurse -Force -EA 0; New-Item -ItemType Directory -Force $wd | Out-Null
  $g = "$tree\lib\ollama\cuda_v13"
  Set-Env @{ OLLAMA_LIBRARY_PATH = "$tree\lib\ollama;$g"; OLLAMA_VULKAN = '0'; OLLAMA_MODELS = "$me\models"; OLLAMA_HOST = "127.0.0.1:$port"
    OLLAMA_NUM_PARALLEL = '1'; OLLAMA_LOAD_TIMEOUT = '120m'; OLLAMA_KEEP_ALIVE = '30m'; OLLAMA_FLASH_ATTENTION = '0'
    CUDA_FOR_INTEL_CACHE_DIR = "$Cache\kc"; CFI_CACHE_DIR = "$Cache\spv"; NEO_CACHE_PERSISTENT = '0'; CUDA_FOR_INTEL_LOG = '1'; CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $wd -FilePath "$bin\cfi-run.exe" -ArgumentList "--install-dir $bin -- `"$tree\ollama.exe`" serve" -RedirectStandardOutput "$wd\o.txt" -RedirectStandardError "$wd\e.txt"
  $up = $false; for ($i = 0; $i -lt 60 -and -not $up; $i++) { Start-Sleep -Milliseconds 500; if ((Get-Content "$wd\e.txt" -Raw -EA 0) -match "Listening on") { $up = $true } }
  $serverUp = $sw.Elapsed.TotalSeconds
  $model = "pc-$Tag"
  if (-not (Test-Path "$me\models\created_$Tag")) {
    "FROM $Gguf" | Set-Content "$wd\Mf"; $env:OLLAMA_HOST = "127.0.0.1:$port"
    & "$tree\ollama.exe" create $model -f "$wd\Mf" 2>&1 | Out-Null; if ($LASTEXITCODE -eq 0) { 'x' | Set-Content "$me\models\created_$Tag" }
  }
  $prompt = 'The capital of France is Paris. ' + ((@('lorem') * 300) -join ' ')
  $body = @{ model = $model; prompt = $prompt; stream = $false; raw = $true; options = @{ num_predict = 1; temperature = 0; seed = 0; top_k = 1; num_gpu = 99; num_ctx = 2048; num_batch = 512 } } | ConvertTo-Json -Depth 4
  $r = $null; $err = ''
  $t = [Diagnostics.Stopwatch]::StartNew()
  try { $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 5400 } catch { $err = $_.Exception.Message }
  $first = $t.Elapsed.TotalSeconds
  # a second, different prompt on the now-loaded model: steady state
  $body2 = $body.Replace('France is Paris', 'Italy is Rome')
  $t2 = [Diagnostics.Stopwatch]::StartNew(); try { Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/generate" -Method Post -Body $body2 -ContentType 'application/json' -TimeoutSec 900 | Out-Null } catch {}
  $second = $t2.Elapsed.TotalSeconds
  try { Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/generate" -Method Post -Body (@{ model = $model; keep_alive = 0 } | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 30 | Out-Null } catch {}
  Start-Sleep 3
  foreach ($d in @(Get-DescendantIds $p.Id)) { try { [Diagnostics.Process]::GetProcessById($d).Kill() } catch {} }
  try { if (-not $p.HasExited) { $p.Kill() } } catch {}
  [void]$p.WaitForExit(30000); Start-Sleep 2
  $logs = Get-ChildItem $wd -Filter cuda-for-intel-export.log -Recurse -EA 0 | Select-Object -First 1
  $lazy = 0; $hit = 0
  if ($logs) { $lazy = @(Select-String -Path $logs.FullName -Pattern 'lazy entry translated|module load: .*translate_ms=[1-9]' -EA 0).Count }
  $rec = [ordered]@{ tag = $Tag; mode = $Name; server_up_s = [math]::Round($serverUp, 1); first_request_s = [math]::Round($first, 2); second_request_s = [math]::Round($second, 2)
    load_s = if ($r) { [math]::Round($r.load_duration / 1e9, 2) } else { $null }; prompt_tokens = if ($r) { $r.prompt_eval_count } else { $null }
    response = if ($r) { [string]$r.response } else { $null }; error = $err
    server_errors = @(Select-String -Path "$wd\e.txt" -Pattern 'CUDA error|device lost|ZE_RESULT|panic' -EA 0).Count }
  ($rec | ConvertTo-Json -Compress) | Tee-Object -Append "$out\results.jsonl"
}
function Precompile([string]$Cache) {
  Set-Env @{ CUDA_FOR_INTEL_CACHE_DIR = "$Cache\kc"; CFI_CACHE_DIR = "$Cache\spv"; NEO_CACHE_PERSISTENT = '0'; CUDA_FOR_INTEL_LOG = $null; OLLAMA_FLASH_ATTENTION = $null }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $a = @('--install-dir', $bin, '--report', "$out\report_$Tag.tsv") + @($PreArgs -split ' ' | Where-Object { $_ }) + @("$tree\lib\ollama\cuda_v13")
  & "$bin\cfi-precompile.exe" @a 2>&1 | Tee-Object "$out\precompile_$Tag.txt"
  "precompile exit=$LASTEXITCODE wall_s=$([math]::Round($sw.Elapsed.TotalSeconds,1))" | Tee-Object -Append "$out\precompile_$Tag.txt"
}
$held = $false; $hb = $null
try {
  if (-not $NoQueue) {
    $o = (& powershell -NoProfile -File $queue -Action acquire -Id $id -MaxWaitSec 3600 2>&1) -join ' '
    if ($o -notmatch 'acquired') { throw "no GPU slot: $o" }
    $held = $true
    $hb = Start-Job -ArgumentList $lockDir, $id -ScriptBlock { param($l, $m) while ($true) { try { if ((Get-Content "$l\owner" -EA 0) -eq $m) { (Get-Item $l).LastWriteTime = Get-Date } } catch {}; Start-Sleep 45 } }
  }
  foreach ($m in ($Modes -split ',')) {
    switch ($m) {
      'cold'    { $c = "$out\cache_${Tag}_cold"; Remove-Item $c -Recurse -Force -EA 0; Run-Server 'cold' $c }
      'warm'    { Run-Server 'warm' "$out\cache_${Tag}_cold" }
      'pre'     { $c = "$out\cache_${Tag}_pre"; Remove-Item $c -Recurse -Force -EA 0; Precompile $c; Run-Server 'precompiled' $c }
      'prewarm' { Run-Server 'precompiled-warm' "$out\cache_${Tag}_pre" }
      'preonly' { $c = "$out\cache_${Tag}_pre"; Remove-Item $c -Recurse -Force -EA 0; Precompile $c }
    }
  }
} finally {
  if ($hb) { Stop-Job $hb -EA 0; Remove-Job $hb -Force -EA 0 }
  if ($held) { & powershell -NoProfile -File $queue -Action release -Id $id | Out-Null }
}
'TTFT_DONE'

<#
.SYNOPSIS
  Box-side driver (Windows, Intel Arc): scans prompts with stock Ollama in SUBST=auto and SUBST=none, picks the first
  prompt whose greedy text differs, and dumps per-step logits for it with cfi_logits.exe in both modes (free-run and
  teacher-forced on the other mode's tokens). Holds the shared GPU queue slot for the whole job.
.PARAMETER Work     work dir that contains run\ (cfi_logits.exe + llama.cpp DLLs + shim nvcuda.dll/cublas64_13.dll) and receives results
.PARAMETER OllamaRun staged stock Ollama tree (ollama.exe + lib\ollama\cuda_v13 with the shim DLLs)
.PARAMETER Model    the Q8_0 GGUF
.PARAMETER Hybrid   dir with nvcudart_hybrid64.dll
.PARAMETER Id       gpu-queue id
.PARAMETER Prompts  optional text file with one prompt per line (default: built-in list)
.PARAMETER Fixed    index (0-based) of the prompt to analyse even if the modes agree on it (default: first diverging)
#>
param([Parameter(Mandatory)][string]$Work, [Parameter(Mandatory)][string]$OllamaRun, [Parameter(Mandatory)][string]$Model,
      [Parameter(Mandatory)][string]$Hybrid, [string]$Id = 'divergence', [string]$Prompts = '', [int]$NPredict = 96, [int]$Fixed = -1)
$ErrorActionPreference = 'Continue'
$queue = Join-Path (Split-Path $Work -Parent) 'gpu-queue.ps1'
$lock  = Join-Path (Split-Path $Work -Parent) 'gpu.lock'
$res = Join-Path $Work 'results'; New-Item -ItemType Directory -Force $res | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
if ($Prompts) { $plist = [IO.File]::ReadAllLines($Prompts, $utf8) | ? { $_ } } else {
  $plist = @('Write a short story about a robot learning to paint.','The quick brown fox jumps over the lazy dog. The quick brown fox',
   'Explain why the sky is blue in one sentence:','def fibonacci(n):','The three primary colors are red, yellow, and',
   'Once upon a time in a small village there lived a young girl who loved to','Q: What is 17 plus 25? A:','Translate to French: Good morning, how are you?',
   'Once upon a time, in a distant galaxy,','Write a poem about the ocean.','Describe the city of Paris in detail.','List five tips for staying healthy:',
   'The history of the Roman Empire began','Write a short story about a dragon who is afraid of fire.','Explain how a car engine works.',
   'What are the benefits of exercise?','Tell me a joke about programmers.','Write a haiku about autumn, then explain it.',
   'Once there was a little boat that wanted to see the world','Dear Sir or Madam, I am writing to complain about') }

& powershell -NoProfile -File $queue -Action acquire -Id $Id -MaxWaitSec 14400 | Out-Null
$hb = Start-Job -ArgumentList $lock -ScriptBlock { param($l) while ($true) { try { (Get-Item $l).LastWriteTime = Get-Date } catch {}; Start-Sleep 60 } }
try {
  function Set-ModeEnv($mode, $cacheDir) {
    Get-ChildItem Env: | ? { $_.Name -match 'CFI|CUDA_FOR_INTEL|NEO_' } | % { Remove-Item "Env:$($_.Name)" }
    $env:CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid; $env:CUDA_FOR_INTEL_CACHE_DIR = $cacheDir; $env:CUDA_FOR_INTEL_LOG = '1'
    if ($mode -eq 'none') { $env:CUDA_FOR_INTEL_SUBST = 'none' }
  }
  # ---------- A: stock Ollama scan ----------
  $g = "$OllamaRun\lib\ollama\cuda_v13"; $exe = "$OllamaRun\ollama-divc.exe"
  if (-not (Test-Path $exe)) { Copy-Item "$OllamaRun\ollama.exe" $exe }
  $port = 11557
  foreach ($mode in 'auto','none') {
    $wd = "$res\ollama_$mode"; New-Item -ItemType Directory -Force "$wd", "$Work\kc_ollama_$mode" | Out-Null
    Set-ModeEnv $mode "$Work\kc_ollama_$mode"
    $env:PATH = "$OllamaRun;$OllamaRun\lib\ollama;$g;$env:PATH"; $env:OLLAMA_LIBRARY_PATH = "$OllamaRun\lib\ollama;$g"; $env:OLLAMA_VULKAN = '0'
    $env:OLLAMA_MODELS = "$(Split-Path $Model -Parent)\models"; $env:OLLAMA_HOST = "127.0.0.1:$port"; $env:OLLAMA_NUM_PARALLEL = '1'
    $env:OLLAMA_FLASH_ATTENTION = '0'; $env:OLLAMA_LOAD_TIMEOUT = '120m'; $env:OLLAMA_KEEP_ALIVE = '30m'
    Get-Process ollama-divc -EA 0 | Stop-Process -Force; Start-Sleep 2
    $srv = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $wd -FilePath $exe -ArgumentList 'serve' -RedirectStandardOutput "$wd\o.txt" -RedirectStandardError "$wd\e.txt"
    Start-Sleep 8
    "FROM $Model" | Set-Content "$wd\Mf"
    & $exe create "divc-$mode" -f "$wd\Mf" 2>&1 | Out-Null
    for ($i = 0; $i -lt $plist.Count; $i++) {
      $body = @{model = "divc-$mode"; prompt = $plist[$i]; stream = $false; raw = $true; options = @{num_predict = $NPredict; temperature = 0; seed = 0; num_gpu = 99; num_ctx = 1024; num_batch = 512}} | ConvertTo-Json -Depth 4
      try { $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/api/generate" -Method Post -Body $utf8.GetBytes($body) -ContentType 'application/json; charset=utf-8' -TimeoutSec 5400
            $ms = New-Object IO.MemoryStream; $r.RawContentStream.CopyTo($ms); [IO.File]::WriteAllBytes("$wd\p$i.json", $ms.ToArray()) } catch { "FAIL $mode $i $($_.Exception.Message)" | Add-Content "$res\scan_errors.txt" }
    }
    Stop-Process -Id $srv.Id -Force -EA 0; Get-Process ollama-divc -EA 0 | Stop-Process -Force
    if (Test-Path "$wd\cuda-for-intel-export.log") { Copy-Item "$wd\cuda-for-intel-export.log" "$res\ollama_${mode}_export.log" }
  }
  # compare
  $rows = @(); $pick = -1
  for ($i = 0; $i -lt $plist.Count; $i++) {
    $a = $null; $n = $null
    try { $a = ($utf8.GetString([IO.File]::ReadAllBytes("$res\ollama_auto\p$i.json")) | ConvertFrom-Json).response; $n = ($utf8.GetString([IO.File]::ReadAllBytes("$res\ollama_none\p$i.json")) | ConvertFrom-Json).response } catch {}
    if ($a -eq $null -or $n -eq $null) { $rows += "$i ERROR"; continue }
    $d = -1; $m = [Math]::Min($a.Length, $n.Length); for ($k = 0; $k -lt $m; $k++) { if ($a[$k] -ne $n[$k]) { $d = $k; break } }
    if ($d -lt 0 -and $a.Length -ne $n.Length) { $d = $m }
    $rows += "$i diverge_char=$d len_auto=$($a.Length) len_none=$($n.Length) :: $($plist[$i])"
    if ($d -ge 0 -and $pick -lt 0 -and $d -ge 60) { $pick = $i }
  }
  if ($pick -lt 0) { for ($i = 0; $i -lt $plist.Count; $i++) { if ($rows[$i] -match 'diverge_char=(\d+)' -and [int]$Matches[1] -ge 0 -and $rows[$i] -notmatch 'diverge_char=-1') { $pick = $i; break } } }
  if ($Fixed -ge 0) { $pick = $Fixed }
  $rows | Set-Content "$res\scan_summary.txt"; "picked=$pick" | Add-Content "$res\scan_summary.txt"
  # ---------- B: logits through llama.cpp (same shim, same mode switch) ----------
  if ($pick -ge 0) {
    [IO.File]::WriteAllText("$res\prompt.txt", $plist[$pick], $utf8)
    $tool = "$Work\run\cfi_logits.exe"
    function Run-Tool($mode, $kind, $idsFile, $tag) {
      Set-ModeEnv $mode "$Work\kc_llama_$mode"; New-Item -ItemType Directory -Force "$Work\kc_llama_$mode" | Out-Null
      $env:PATH = "$Work\run;$env:PATH"
      $wd = "$Work\run"; Push-Location $wd
      Remove-Item cuda-for-intel-export.log -EA 0
      $p = Start-Process -PassThru -Wait -NoNewWindow -FilePath $tool -ArgumentList $Model, '99', $kind, "$NPredict", "$res\prompt.txt", $idsFile, "$res\$tag" -RedirectStandardError "$res\$tag.err" -RedirectStandardOutput "$res\$tag.out"
      "$tag exit=$($p.ExitCode)" | Add-Content "$res\tool_runs.txt"
      if (Test-Path cuda-for-intel-export.log) { Copy-Item cuda-for-intel-export.log "$res\$tag.export.log" -Force }
      Pop-Location
    }
    Run-Tool 'auto' 'gen' '-' 'llama_auto_gen'
    Run-Tool 'none' 'gen' '-' 'llama_none_gen'
    # ids files for forcing: generated ids line of the other run
    foreach ($pair in @(@('auto','none'), @('none','auto'))) {
      $mode = $pair[0]; $other = $pair[1]
      $ids = (Get-Content "$res\llama_${other}_gen.ids.txt")[2]
      Set-Content "$res\forced_from_$other.txt" $ids
      Run-Tool $mode 'force' "$res\forced_from_$other.txt" "llama_${mode}_force_$other"
      # also force the mode's own sequence (repeatability / mode-independent step alignment)
    }
    foreach ($mode in 'auto','none') { $ids = (Get-Content "$res\llama_${mode}_gen.ids.txt")[2]; Set-Content "$res\forced_from_$mode.txt" $ids }
    Run-Tool 'auto' 'force' "$res\forced_from_auto.txt" 'llama_auto_force_auto'
  }
} finally {
  Stop-Job $hb -EA 0; Remove-Job $hb -Force -EA 0
  & powershell -NoProfile -File $queue -Action release -Id $Id | Out-Null
}
'DONE' | Set-Content "$res\DONE.txt"

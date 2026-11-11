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
.PARAMETER ResultsName subdirectory of Work that receives results (default results)
.PARAMETER ShimDir optional dir holding the nvcuda.dll and cublas64_13.dll to test (copied into private copies of the Ollama tree and of run\); default: the DLLs already staged
.PARAMETER VariantList comma separated subset of variants to run (default all)
.PARAMETER SkipTool only run the Ollama scan
.PARAMETER NoQueue the caller (scripts/hwvalidate/run_all.ps1) already holds the GPU queue slot: do not acquire/release or heartbeat here
.PARAMETER FastDiv set CUDA_FOR_INTEL_FAST_DIV=1 in every run (control for the correctly rounded division default)
.PARAMETER Fixed    index (0-based) of the prompt to analyse even if the modes agree on it (default: first diverging)
#>
param([Parameter(Mandatory)][string]$Work, [Parameter(Mandatory)][string]$OllamaRun, [Parameter(Mandatory)][string]$Model,
      [Parameter(Mandatory)][string]$Hybrid, [string]$Id = 'divergence', [string]$Prompts = '', [int]$NPredict = 96, [int]$Fixed = -1, [string]$ResultsName = 'results', [string]$ShimDir = '', [switch]$SkipTool, [string]$VariantList = '', [switch]$NoQueue, [switch]$FastDiv)
$ErrorActionPreference = 'Continue'
$queue = Join-Path (Split-Path $Work -Parent) 'gpu-queue.ps1'
$lock  = Join-Path (Split-Path $Work -Parent) 'gpu.lock'
$res = Join-Path $Work $ResultsName; New-Item -ItemType Directory -Force $res | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
if ($Prompts) { $plist = @([IO.File]::ReadAllLines($Prompts, $utf8) | ? { $_ }) } else {
  $plist = @('Write a short story about a robot learning to paint.','The quick brown fox jumps over the lazy dog. The quick brown fox',
   'Explain why the sky is blue in one sentence:','def fibonacci(n):','The three primary colors are red, yellow, and',
   'Once upon a time in a small village there lived a young girl who loved to','Q: What is 17 plus 25? A:','Translate to French: Good morning, how are you?',
   'Once upon a time, in a distant galaxy,','Write a poem about the ocean.','Describe the city of Paris in detail.','List five tips for staying healthy:',
   'The history of the Roman Empire began','Write a short story about a dragon who is afraid of fire.','Explain how a car engine works.',
   'What are the benefits of exercise?','Tell me a joke about programmers.','Write a haiku about autumn, then explain it.',
   'Once there was a little boat that wanted to see the world','Dear Sir or Madam, I am writing to complain about') }

if ($ShimDir) {
  $priv = Join-Path $Work 'ollama_priv'
  if (-not (Test-Path $priv)) { robocopy $OllamaRun $priv /E /XF *.orig /NFL /NDL /NJH /NJS /NP | Out-Null }
  $OllamaRun = $priv
  Copy-Item "$ShimDir\nvcuda.dll" "$OllamaRun\lib\ollama\cuda_v13\nvcuda.dll" -Force
  Copy-Item "$ShimDir\cublas64_13.dll" "$OllamaRun\lib\ollama\cuda_v13\cublas64_13.dll" -Force
  $runDir = Join-Path $Work 'run_priv'
  if (-not (Test-Path $runDir)) { New-Item -ItemType Directory $runDir | Out-Null; Copy-Item "$Work\run\*" $runDir -Recurse -Exclude *.log }
  Copy-Item "$ShimDir\nvcuda.dll" $runDir -Force; Copy-Item "$ShimDir\cublas64_13.dll" $runDir -Force; Copy-Item "$ShimDir\cublas64_13.dll" "$runDir\cublasLt64_13.dll" -Force
} else { $runDir = Join-Path $Work 'run' }
Get-FileHash "$OllamaRun\lib\ollama\cuda_v13\nvcuda.dll" -Algorithm SHA256 | % { "nvcuda.dll sha256=$($_.Hash)" } | Set-Content "$res\provenance.txt"
Get-FileHash "$OllamaRun\lib\ollama\cuda_v13\ggml-cuda.dll" -Algorithm SHA256 | % { "ollama ggml-cuda.dll sha256=$($_.Hash)" } | Add-Content "$res\provenance.txt"
Get-FileHash $Model -Algorithm SHA256 | % { "model sha256=$($_.Hash)" } | Add-Content "$res\provenance.txt"
if (-not $NoQueue) { & powershell -NoProfile -File $queue -Action acquire -Id $Id -MaxWaitSec 14400 | Out-Null }
$hb = if ($NoQueue) { $null } else { Start-Job -ArgumentList $lock,$Id -ScriptBlock { param($l,$me) while ($true) { try { if ((Get-Content "$l\owner" -EA 0) -eq $me) { (Get-Item $l).LastWriteTime = Get-Date } } catch {}; Start-Sleep 60 } } }
try {
  function Set-ModeEnv($mode, $cacheDir) {
    Get-ChildItem Env: | ? { $_.Name -match 'CFI|CUDA_FOR_INTEL|NEO_' } | % { Remove-Item "Env:$($_.Name)" }
    $env:CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid; $env:CUDA_FOR_INTEL_CACHE_DIR = $cacheDir; $env:CUDA_FOR_INTEL_LOG = '1'
    if ($FastDiv) { $env:CUDA_FOR_INTEL_FAST_DIV = '1' }   # control run: modules built without the correctly rounded div/sqrt flag
    if ($mode -eq 'none') { $env:CUDA_FOR_INTEL_SUBST = 'none' }
  }
  # ---------- A: stock Ollama scan ----------
  $g = "$OllamaRun\lib\ollama\cuda_v13"; $exe = "$OllamaRun\ollama-divc.exe"
  if (-not (Test-Path $exe)) { Copy-Item "$OllamaRun\ollama.exe" $exe }
  $port = Get-Random -Minimum 21000 -Maximum 29000
  foreach ($mode in 'auto','none') {
    $wd = "$res\ollama_$mode"; New-Item -ItemType Directory -Force "$wd", "$Work\kc_ollama_$mode" | Out-Null
    Set-ModeEnv $mode "$Work\kc_ollama_$mode"
    $env:PATH = "$OllamaRun;$OllamaRun\lib\ollama;$g;$env:PATH"; $env:OLLAMA_LIBRARY_PATH = "$OllamaRun\lib\ollama;$g"; $env:OLLAMA_VULKAN = '0'
    $env:OLLAMA_MODELS = "$(Split-Path $Model -Parent)\models"; $env:OLLAMA_HOST = "127.0.0.1:$port"; $env:OLLAMA_NUM_PARALLEL = '1'
    $env:OLLAMA_FLASH_ATTENTION = '0'; $env:OLLAMA_LOAD_TIMEOUT = '120m'; $env:OLLAMA_KEEP_ALIVE = '30m'
    if (@(Get-Process ollama-divc -EA 0 | Where-Object { $_.Path -and $_.Path -like "$OllamaRun\*" }).Count) { "ABORT $mode : a stale ollama-divc is running; refusing to start another" | Add-Content "$res\scan_errors.txt"; continue }
    $srv = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $wd -FilePath $exe -ArgumentList 'serve' -RedirectStandardOutput "$wd\o.txt" -RedirectStandardError "$wd\e.txt"
    Start-Sleep 8
    # guard: our own server log must say it listens on our port (a foreign server on the port would make ours fail with "bind")
    $okSrv = $false
    for ($w = 0; $w -lt 30 -and -not $okSrv; $w++) { $log = Get-Content "$wd\e.txt" -Raw -EA 0; if ($log -match "Listening on 127.0.0.1:$port ") { $okSrv = $true } elseif ($log -match 'bind:') { break } else { Start-Sleep 2 } }
    if (-not $okSrv -or $srv.HasExited) { "ABORT $mode : our ollama-divc did not start listening on $port" | Add-Content "$res\scan_errors.txt"; continue }
    "FROM $Model" | Set-Content "$wd\Mf"
    & $exe create "divc-$mode" -f "$wd\Mf" 2>&1 | Out-Null
    for ($i = 0; $i -lt $plist.Count; $i++) {
      $body = @{model = "divc-$mode"; prompt = $plist[$i]; stream = $false; raw = $true; options = @{num_predict = $NPredict; temperature = 0; seed = 0; top_k = 1; top_p = 1.0; repeat_penalty = 1.0; num_gpu = 99; num_ctx = 1024; num_batch = 512}} | ConvertTo-Json -Depth 4
      try { $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/api/generate" -Method Post -Body $utf8.GetBytes($body) -ContentType 'application/json; charset=utf-8' -TimeoutSec 5400
            $ms = New-Object IO.MemoryStream; $r.RawContentStream.CopyTo($ms); [IO.File]::WriteAllBytes("$wd\p$i.json", $ms.ToArray()) } catch { "FAIL $mode $i $($_.Exception.Message)" | Add-Content "$res\scan_errors.txt" }
    }
    # Unload first (the runner then exits by itself), then end the process tree WE started: killing only the server
    # left its runner child alive, and the next mode aborted on the stale ollama-divc.
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/generate" -Method Post -Body (@{ model = "divc-$mode"; keep_alive = 0 } | ConvertTo-Json) -ContentType 'application/json' -TimeoutSec 30 | Out-Null } catch {}
    Start-Sleep 5
    foreach ($c in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($srv.Id)" -EA 0)) { try { [Diagnostics.Process]::GetProcessById([int]$c.ProcessId).Kill() } catch {} }
    try { if (-not $srv.HasExited) { $srv.Kill() } } catch {}; [void]$srv.WaitForExit(30000); Start-Sleep 3
    if (Test-Path "$wd\cuda-for-intel-export.log") { Copy-Item "$wd\cuda-for-intel-export.log" "$res\ollama_${mode}_export.log" }
  }
  # compare
  $rows = @(); $pick = -1; $bestD = 0
  for ($i = 0; $i -lt $plist.Count; $i++) {
    $a = $null; $n = $null
    try { $a = ($utf8.GetString([IO.File]::ReadAllBytes("$res\ollama_auto\p$i.json")) | ConvertFrom-Json).response; $n = ($utf8.GetString([IO.File]::ReadAllBytes("$res\ollama_none\p$i.json")) | ConvertFrom-Json).response } catch {}
    if ($a -eq $null -or $n -eq $null) { $rows += "$i ERROR"; continue }
    $d = -1; $m = [Math]::Min($a.Length, $n.Length); for ($k = 0; $k -lt $m; $k++) { if ($a[$k] -ne $n[$k]) { $d = $k; break } }
    if ($d -lt 0 -and $a.Length -ne $n.Length) { $d = $m }
    $rows += "$i diverge_char=$d len_auto=$($a.Length) len_none=$($n.Length) :: $($plist[$i])"
    if ($d -ge 20 -and ($pick -lt 0 -or [Math]::Abs($d - 160) -lt [Math]::Abs($bestD - 160))) { $pick = $i; $bestD = $d }
  }
  if ($pick -lt 0) { for ($i = 0; $i -lt $plist.Count; $i++) { if ($rows[$i] -match 'diverge_char=(\d+)' -and [int]$Matches[1] -ge 0 -and $rows[$i] -notmatch 'diverge_char=-1') { $pick = $i; break } } }
  if ($Fixed -ge 0) { $pick = $Fixed }
  $rows | Set-Content "$res\scan_summary.txt"; "picked=$pick" | Add-Content "$res\scan_summary.txt"
  # ---------- B: logits through llama.cpp (same shim, same mode switch) ----------
  if ($pick -ge 0 -and -not $SkipTool) {
    [IO.File]::WriteAllText("$res\prompt.txt", $plist[$pick], $utf8)
    $tool = "$runDir\cfi_logits.exe"
    # variant name -> CUDA_FOR_INTEL_SUBST value ('' = unset = default auto)
    $variants = [ordered]@{ auto = ''; none = 'none'; cpu = 'none'; auto_nommq = 'auto,-mmq'; only_mmq = 'mmq'; only_mmvq = 'mmvq'; only_quant = 'quantize_q8_1'; mmvq_nw1 = 'mmvq'; mmvq_nw2 = 'mmvq'; mmvq_nw8 = 'mmvq' }
    $nwarps = @{ mmvq_nw1 = '1'; mmvq_nw2 = '2'; mmvq_nw8 = '8' }   # CUDA_FOR_INTEL_SUBST_NWARPS: changes only the cross-warp add order of mmvq
    if ($VariantList) { $keep = $VariantList.Split(',') + @('auto','none'); foreach ($k in @($variants.Keys)) { if ($keep -notcontains $k) { $variants.Remove($k) } } }
    function Run-Tool($var, $kind, $idsFile, $tag) {
      $kc = "$Work\kc_llama_$var"; New-Item -ItemType Directory -Force $kc | Out-Null
      Set-ModeEnv 'auto' $kc
      if ($variants[$var]) { $env:CUDA_FOR_INTEL_SUBST = $variants[$var] }
      # NWARPS shapes only the stock-order mmvq; decode (ncols 1) defaults to the fast variant, which ignores it
      if ($nwarps[$var]) { $env:CUDA_FOR_INTEL_SUBST_NWARPS = $nwarps[$var]; $env:CUDA_FOR_INTEL_MMVQ_FAST = '0' }
      $env:PATH = "$runDir;$env:PATH"
      Push-Location $runDir
      Remove-Item cuda-for-intel-export.log -EA 0
      $p = Start-Process -PassThru -Wait -NoNewWindow -FilePath $tool -ArgumentList $Model, $(if ($var -eq 'cpu') { '0' } else { '99' }), $kind, "$NPredict", "$res\prompt.txt", $idsFile, "$res\$tag" -RedirectStandardError "$res\$tag.err" -RedirectStandardOutput "$res\$tag.out"
      "$tag exit=$($p.ExitCode) SUBST=[$($variants[$var])]" | Add-Content "$res\tool_runs.txt"
      if (Test-Path cuda-for-intel-export.log) { Select-String -Path cuda-for-intel-export.log -Pattern '^subst' | % { $_.Line } | Set-Content "$res\$tag.subst.log" }
      Pop-Location
    }
    Run-Tool 'auto' 'gen' '-' 'llama_auto_gen'
    Run-Tool 'none' 'gen' '-' 'llama_none_gen'
    foreach ($seq in 'auto','none') { (Get-Content "$res\llama_${seq}_gen.ids.txt")[2] | Set-Content "$res\forced_from_$seq.txt" }
    foreach ($seq in 'none','auto') { foreach ($var in $variants.Keys) {
      if ($var -eq $seq) { continue }   # the free run already is "variant $var on its own sequence"
      Run-Tool $var 'force' "$res\forced_from_$seq.txt" "llama_${var}_force_$seq"
    } }
  }
} finally {
  if ($hb) { Stop-Job $hb -EA 0; Remove-Job $hb -Force -EA 0 }
  if (-not $NoQueue) { & powershell -NoProfile -File $queue -Action release -Id $Id | Out-Null }
}
'DONE' | Set-Content "$res\DONE.txt"

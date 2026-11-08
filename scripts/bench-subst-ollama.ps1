# Prefill/decode benchmark of stock Ollama v0.12.6 through the shim for several
# CUDA_FOR_INTEL_SUBST settings. One server per setting; a warm-up request first (kernel JIT),
# then timed requests with a unique first word (no prompt-cache reuse), ~631-token prompt,
# flash attention off, greedy. Hold the GPU queue slot around the whole run.
#   powershell -NoProfile -File scripts\bench-subst-ollama.ps1 -RunDir <ollama run dir with the
#   shim dlls in lib\ollama\cuda_v13> -Gguf <model.gguf> -Tag l1q4 -Configs 'none','auto','auto,mmq'
param(
    [Parameter(Mandatory)][string]$RunDir,
    [Parameter(Mandatory)][string]$Gguf,
    [Parameter(Mandatory)][string]$Tag,
    [string[]]$Configs = @('none', 'auto'),
    [int]$TargetTokens = 631,
    [int]$Runs = 3,
    [int]$Predict = 48,
    [int]$Port = 11577,
    [string]$Hybrid = '',
    [string]$OutDir = '.\subst-bench-out',
    [int]$Check = 0
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$g = "$RunDir\lib\ollama\cuda_v13"
$facts = @(
    'The capital of France is Paris and its river is the Seine.',
    'Water boils at one hundred degrees Celsius at sea level pressure.',
    'The Pacific is the largest ocean and covers about a third of the planet.',
    'Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar.',
    'The printing press was introduced in Europe by Gutenberg around 1440.',
    'Mount Everest is the highest mountain above sea level in the world.',
    'Honey bees communicate the location of flowers with a waggle dance.',
    'The speed of light in vacuum is close to three hundred thousand kilometers per second.')
function Make-Prompt([int]$words, [int]$run) {
    $sb = [Text.StringBuilder]::new()
    [void]$sb.Append("Record$($run + 100): ")
    $n = 0; $i = 0
    while ($n -lt $words) {
        $s = $facts[$i % $facts.Count] + " (note $i) "
        [void]$sb.Append($s); $n += $s.Split(' ').Count; $i++
    }
    [void]$sb.Append("`nQuestion: What is the capital of France? Answer in a complete sentence, then give a short description of the city.`nAnswer:")
    $sb.ToString()
}
function Invoke-Gen($model, $prompt, $predict) {
    $body = @{model = $model; prompt = $prompt; stream = $false; raw = $true; options = @{num_predict = $predict; temperature = 0; seed = 7; num_gpu = 99; num_ctx = 2048; num_batch = 512 } } | ConvertTo-Json -Depth 4
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 5400
}
$words = 330
$results = @()
foreach ($cfg in $Configs) {
    Get-ChildItem Env: | Where-Object { $_.Name -match '^(CFI|CUDA_FOR_INTEL|NEO_)' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:PATH = "$RunDir;$RunDir\lib\ollama;$g;$env:PATH"
    $env:OLLAMA_LIBRARY_PATH = "$RunDir\lib\ollama;$g"; $env:OLLAMA_VULKAN = '0'
    if ($Hybrid) { $env:CUDA_FOR_INTEL_HYBRID_DIR = $Hybrid }
    $env:OLLAMA_MODELS = "$OutDir\models"; $env:OLLAMA_HOST = "127.0.0.1:$Port"
    $env:OLLAMA_NUM_PARALLEL = '1'; $env:OLLAMA_FLASH_ATTENTION = '0'
    $env:OLLAMA_LOAD_TIMEOUT = '120m'; $env:OLLAMA_KEEP_ALIVE = '5m'
    $env:CUDA_FOR_INTEL_CACHE_DIR = "$OutDir\kc"; $env:CFI_CACHE_DIR = "$OutDir\spv"
    $env:CUDA_FOR_INTEL_LOG = '1'
    $env:CUDA_FOR_INTEL_SUBST = $cfg
    if ($Check -gt 0) { $env:CUDA_FOR_INTEL_SUBST_CHECK = "$Check" }
    $wd = "$OutDir\wd_$($Tag)_$($cfg -replace '[^a-z0-9]', '_')"
    if (Test-Path $wd) { Remove-Item $wd -Recurse -Force }
    New-Item -ItemType Directory -Force $wd, "$OutDir\kc", "$OutDir\spv" | Out-Null
    $exe = "$RunDir\ollama.exe"
    $srv = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $wd -FilePath $exe -ArgumentList 'serve' -RedirectStandardOutput "$wd\o.txt" -RedirectStandardError "$wd\e.txt"
    try {
        Start-Sleep 8
        "FROM $Gguf" | Set-Content "$wd\Mf"
        & $exe create "kq-$Tag" -f "$wd\Mf" 2>&1 | Out-Null
        # calibrate the prompt length once (first config)
        for ($k = 0; $k -lt 4; $k++) {
            $r = Invoke-Gen "kq-$Tag" (Make-Prompt $words 90) 1
            $d = $TargetTokens - $r.prompt_eval_count
            "calibrate words=$words -> $($r.prompt_eval_count) tokens"
            if ([math]::Abs($d) -le 3) { break }
            $words += [int]($d * $words / [math]::Max(1, $r.prompt_eval_count))
        }
        for ($i = 0; $i -lt $Runs + 1; $i++) {
            $r = Invoke-Gen "kq-$Tag" (Make-Prompt $words $i) $Predict
            if ($i -eq 0) { continue } # warm-up for this configuration
            $pf = $r.prompt_eval_count / ($r.prompt_eval_duration / 1e9)
            $dc = $r.eval_count / ($r.eval_duration / 1e9)
            $line = [pscustomobject]@{tag = $Tag; config = $cfg; run = $i; prompt_tokens = $r.prompt_eval_count; prefill_tps = [math]::Round($pf, 1); gen_tokens = $r.eval_count; decode_tps = [math]::Round($dc, 2) }
            $line | ConvertTo-Json -Compress
            $results += $line
            if ($i -eq 1) { $r.response | Set-Content -Encoding utf8 "$OutDir\resp_$($Tag)_$($cfg -replace '[^a-z0-9]', '_').txt" }
        }
    }
    finally {
        Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
        Get-Process | Where-Object { $_.Path -like "$RunDir*" } | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep 3
        Copy-Item "$wd\cuda-for-intel-export.log" "$OutDir\export_$($Tag)_$($cfg -replace '[^a-z0-9]', '_').log" -ErrorAction SilentlyContinue
    }
}
$results | Group-Object config | ForEach-Object {
    $m = $_.Group | Measure-Object prefill_tps, decode_tps -Average
    "{0,-12} prefill {1,8:N1} tok/s  decode {2,7:N2} tok/s" -f $_.Name, ($m | Where-Object Property -eq prefill_tps).Average, ($m | Where-Object Property -eq decode_tps).Average
}

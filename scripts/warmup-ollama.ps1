<#
.SYNOPSIS
  Fills the CUDA for Intel kernel caches for a model by running a few tokens at several
  prompt sizes against a running Ollama server (see docs/first-run-latency.md).
.EXAMPLE
  $env:CUDA_FOR_INTEL_WARMUP = "1"   # before starting the server: translate+compile every kernel
  .\scripts\warmup-ollama.ps1 -Model llama3.2:3b
#>
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [string]$Server = "http://127.0.0.1:11434",
  [int[]]$PromptWords = @(4, 64, 512),
  [int]$Tokens = 8
)
$ErrorActionPreference = "Stop"
foreach ($words in $PromptWords) {
  $prompt = (@("warm") * $words) -join " "
  $body = @{ model = $Model; prompt = $prompt; stream = $false; options = @{ num_predict = $Tokens; temperature = 0 } } | ConvertTo-Json -Depth 4
  $t = [Diagnostics.Stopwatch]::StartNew()
  # First request loads the model and compiles on demand; no client timeout on purpose.
  Invoke-RestMethod -Uri "$Server/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 7200 | Out-Null
  "{0,5} prompt words: {1:n1}s" -f $words, $t.Elapsed.TotalSeconds
}

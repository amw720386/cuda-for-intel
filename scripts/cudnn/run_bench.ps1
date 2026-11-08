# Run bench_conv.py for the given modes under the GPU queue. Usage: run_bench.ps1 -Tag base -Modes aten,cudnn [-Filter x] [-Batch 8] [-TimeoutMin 25]
param([string]$Tag='run', [string[]]$Modes=@('aten','cudnn'), [string]$Filter='', [int]$Batch=8, [int]$TimeoutMin=25)
$Modes = @($Modes | ForEach-Object { $_ -split ',' })
$w = Join-Path $env:USERPROFILE 'cfi-work'; $d = "$w\cd3"; $run = "$d\run"
$py = "$w\torch\torch222\Scripts\python.exe"
$env:CFI_BENCH_ROOT = $run; $env:CFI_BENCH_BATCH = "$Batch"
& "$w\gpu-queue.ps1" -Action acquire -Id "cudnn3-$Tag" -MaxWaitSec 7200
try {
  foreach ($m in $Modes) {
    $o = "$d\res_${Tag}_$m.jsonl"; Remove-Item $o -EA 0
    $p = Start-Process $py -ArgumentList $(if ($Filter) { "`"$d\bench_conv.py`" $m `"$o`" $Filter" } else { "`"$d\bench_conv.py`" $m `"$o`"" }) -RedirectStandardOutput "$d\log_${Tag}_$m.out" -RedirectStandardError "$d\log_${Tag}_$m.err" -PassThru -NoNewWindow
    if (-not $p.WaitForExit($TimeoutMin * 60000)) { "TIMEOUT $m (pid $($p.Id) left running)" | Out-File -Append "$d\log_${Tag}_$m.out" }
  }
} finally { & "$w\gpu-queue.ps1" -Action release -Id "cudnn3-$Tag" }
'done' | Out-File "$d\done_$Tag.txt"

param(
    [string]$Endpoint = 'http://127.0.0.1:11513',
    [ValidateSet('CPU', 'Vulkan')][string]$Backend,
    [int]$Samples = 9
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes(5)

function Get-Percentiles([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    [ordered]@{
        median = $sorted[[int][Math]::Ceiling($sorted.Count / 2.0) - 1]
        p95 = $sorted[[int][Math]::Ceiling($sorted.Count * 0.95) - 1]
    }
}

try {
    $results = @(for ($i = 0; $i -lt $Samples + 2; $i++) {
        $fact = 'The capital of France is Paris. Its river is the Seine. '
        $pad = (($fact * 16).ToCharArray() | Select-Object -First 800) -join ''
        # A unique first word limits prompt-cache reuse between samples.
        $prompt = "Record$($i + 100): $pad`nQuestion: What is the capital of France? Answer in a complete sentence, then give a short description."
        $request = @{
            model = 'qwen2.5:0.5b'
            prompt = $prompt
            stream = $true
            keep_alive = '5m'
            options = @{
                num_ctx = 2048
                num_predict = 48
                temperature = 0
                seed = 42
            }
        }
        if ($Backend -eq 'CPU') { $request.options.num_gpu = 0 }

        $message = [System.Net.Http.HttpRequestMessage]::new('POST', "$Endpoint/api/generate")
        $message.Content = [System.Net.Http.StringContent]::new(
            ($request | ConvertTo-Json -Depth 4 -Compress), [Text.Encoding]::UTF8, 'application/json')
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $response = $null
        $reader = $null
        $ttft = $null
        $result = $null
        $output = [Text.StringBuilder]::new()
        try {
            $response = $client.SendAsync($message, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
            [void]$response.EnsureSuccessStatusCode()
            $reader = [IO.StreamReader]::new($response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
            while ($null -ne ($line = $reader.ReadLine())) {
                $chunk = ConvertFrom-Json $line
                if ($chunk.error) { throw $chunk.error }
                if ($chunk.response) {
                    if ($null -eq $ttft) { $ttft = $clock.Elapsed.TotalMilliseconds }
                    [void]$output.Append($chunk.response)
                }
                if ($chunk.done) { $result = $chunk; break }
            }
        } finally {
            if ($reader) { $reader.Dispose() }
            if ($response) { $response.Dispose() }
            $message.Dispose()
        }
        if (!$result -or !$ttft -or $result.prompt_eval_duration -le 0 -or $result.eval_duration -le 0) {
            throw "Incomplete generation sample $i"
        }
        if ($output.ToString() -notmatch '\bParis\b') { throw "Unexpected output in sample ${i}: $output" }

        [pscustomobject]@{
            warmup = $i -lt 2
            prompt_tokens = $result.prompt_eval_count
            generated_tokens = $result.eval_count
            prompt_tps = $result.prompt_eval_count * 1e9 / $result.prompt_eval_duration
            generation_tps = $result.eval_count * 1e9 / $result.eval_duration
            ttft_ms = $ttft
            load_ms = $result.load_duration / 1e6
            output = $output.ToString()
        }
    })
    $timed = @($results | Where-Object { !$_.warmup })
    $ps = Invoke-RestMethod "$Endpoint/api/ps"
    if (@($ps.models).Count -ne 1) { throw 'Expected exactly one resident model on the benchmark server' }
    if (($Backend -eq 'Vulkan') -ne ($ps.models[0].size_vram -gt 0)) {
        throw "Model placement does not match $Backend"
    }
    [ordered]@{
        backend = $Backend
        version = (Invoke-RestMethod "$Endpoint/api/version").version
        prompt_tps = Get-Percentiles $timed.prompt_tps
        generation_tps = Get-Percentiles $timed.generation_tps
        ttft_ms = Get-Percentiles $timed.ttft_ms
        model = $ps.models[0]
        samples = $results
    } | ConvertTo-Json -Depth 8
} finally {
    $client.Dispose()
}

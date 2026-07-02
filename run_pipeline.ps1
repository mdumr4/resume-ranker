$ErrorActionPreference = "Stop"

Write-Host "Starting llama-server.exe in the background..."
$llamaProcess = Start-Process -FilePath "bin\llama\llama-server.exe" -ArgumentList "-m models\qwen2.5-0.5b-instruct-q4_k_m.gguf -c 16384 -np 16 --threads 22 --port 8080" -PassThru -NoNewWindow -RedirectStandardOutput "llama-server.log" -RedirectStandardError "llama-server-err.log"

Write-Host "Waiting for llama-server to be ready..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient("127.0.0.1", 8080)
        $tcp.Close()
        $ready = $true
        break
    } catch {
        # Not ready yet
    }
}

if (-not $ready) {
    Write-Host "Error: llama-server failed to start!"
    Stop-Process -Id $llamaProcess.Id -Force
    exit 1
}

Write-Host "Server is ready! Running the pipeline..."
try {
    $env:PYTHONIOENCODING='utf-8'
    uv run python -u src\optimize_rank.py
} finally {
    Write-Host "Pipeline finished. Terminating llama-server..."
    Stop-Process -Id $llamaProcess.Id -Force
}

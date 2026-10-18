$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$venv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { throw 'Сначала выполните setup.bat' }
Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -eq $venv -and $_.CommandLine -match '\bscada_rag\s+serve\b'
} | ForEach-Object { & taskkill.exe /PID $_.ProcessId /T /F }
$ollama = Get-Command 'ollama.exe' -ErrorAction SilentlyContinue
if ($ollama) { & $ollama.Source stop qwen2.5:3b }
Write-Host 'Локальный сервер перезапускается...'
& $venv -m scada_rag serve --open
exit $LASTEXITCODE

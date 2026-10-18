$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'

function Run-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Ошибка выполнения: $Exe $Arguments (код $LASTEXITCODE)" }
}

function Find-Program {
    param([string]$Name, [string[]]$Candidates)
    $found = Get-Command $Name -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    foreach ($candidate in $Candidates) { if (Test-Path $candidate) { return $candidate } }
    return $null
}

Write-Host 'SCADA RAG: первоначальная установка. Требуется интернет.'
$py = Find-Program 'py.exe' @()
if ($py) {
    & $py -3.11 -c 'import sys; print(sys.version)' 2>$null
    if ($LASTEXITCODE -eq 0) { $pythonExe = $py; $pythonArgs = @('-3.11') }
    else {
        & $py -3.12 -c 'import sys; print(sys.version)' 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonExe = $py; $pythonArgs = @('-3.12') }
    }
}
if (-not $pythonExe) {
    $candidate = Find-Program 'python.exe' @()
    if ($candidate) {
        & $candidate -c 'import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<(3,13) else 1)'
        if ($LASTEXITCODE -eq 0) { $pythonExe=$candidate; $pythonArgs=@() }
    }
}
if (-not $pythonExe) {
    throw 'Установите Python 3.11 (64-bit) с python.org, отметьте Add Python to PATH и повторите setup.bat.'
}
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Run-Checked $pythonExe ($pythonArgs + @('-m','venv','.venv'))
}
$venv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Run-Checked $venv @('-m','pip','install','--upgrade','pip')
Run-Checked $venv @('-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cpu')
Run-Checked $venv @('-m','pip','install','-r','requirements.txt')
Run-Checked $venv @('-m','pip','install','-e','.')

$ollama = Find-Program 'ollama.exe' @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe", "$env:ProgramFiles\Ollama\ollama.exe")
$tesseract = Find-Program 'tesseract.exe' @("$env:ProgramFiles\Tesseract-OCR\tesseract.exe", "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe")
$winget = Find-Program 'winget.exe' @()
if (-not $ollama) {
    if (-not $winget) { throw 'Установите Ollama с https://ollama.com/download/windows и повторите setup.bat.' }
    Run-Checked $winget @('install','--id','Ollama.Ollama','--exact','--accept-package-agreements','--accept-source-agreements')
    $ollama = Find-Program 'ollama.exe' @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe", "$env:ProgramFiles\Ollama\ollama.exe")
}
if (-not $tesseract) {
    if (-not $winget) { throw 'Установите Tesseract-OCR (UB Mannheim) и повторите setup.bat.' }
    Run-Checked $winget @('install','--id','UB-Mannheim.TesseractOCR','--exact','--accept-package-agreements','--accept-source-agreements')
    $tesseract = Find-Program 'tesseract.exe' @("$env:ProgramFiles\Tesseract-OCR\tesseract.exe", "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe")
}
if (-not $ollama -or -not $tesseract) { throw 'Не удалось найти Ollama или Tesseract после установки. Перезапустите терминал и повторите setup.bat.' }
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null }
catch { Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Minimized; Start-Sleep -Seconds 4 }
Run-Checked $ollama @('pull','qwen2.5:3b')
Write-Host 'Загрузка документов, OCR и перевод ANRE. Первый запуск может занять значительное время.'
Run-Checked $venv @('-m','scada_rag','prepare')
Run-Checked $venv @('-m','unittest','discover','-s','tests','-v')
Run-Checked $venv @('-m','scada_rag','doctor')

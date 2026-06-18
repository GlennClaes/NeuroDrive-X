param(
    [string]$Python = "py",
    [string]$PythonVersion = "-3.11",
    [string]$VenvDir = "",
    [switch]$InstallFrontend
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($VenvDir)) {
    $VenvDir = Join-Path $ProjectRoot ".venv"
}

Write-Host "[NeuroDrive X] Creating Python virtual environment at $VenvDir"
& $Python $PythonVersion -m venv $VenvDir

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip setuptools wheel
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "analytics\training_logs\plots") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "analytics\training_logs\replays") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "ai\models\checkpoints") | Out-Null

if ($env:CARLA_PYTHON_EGG) {
    $envFile = Join-Path $ProjectRoot ".env.carla.ps1"
    "`$env:PYTHONPATH = `"$env:CARLA_PYTHON_EGG;`$env:PYTHONPATH`"" | Set-Content -Path $envFile -Encoding UTF8
    Write-Host "[NeuroDrive X] Wrote CARLA Python path helper to $envFile"
}

if ($InstallFrontend) {
    $FrontendRoot = Join-Path $ProjectRoot "dashboard\frontend"
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm was not found. Install Node.js LTS, then rerun scripts\setup.ps1 -InstallFrontend."
    }
    Push-Location $FrontendRoot
    try {
        npm install
    }
    finally {
        Pop-Location
    }
}

Write-Host "[NeuroDrive X] Setup complete."


param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8080,
    [switch]$Reload,
    [string]$VenvDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($VenvDir)) { $VenvDir = Join-Path $ProjectRoot ".venv" }
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { throw "Virtual environment not found. Run scripts\setup.ps1 first." }

$args = @(
    (Join-Path $ProjectRoot "dashboard\app.py"),
    "--host", $HostName,
    "--port", "$Port"
)
if ($Reload) { $args += "--reload" }

Write-Host "[NeuroDrive X] FastAPI backend: http://$HostName`:$Port"
& $PythonExe @args


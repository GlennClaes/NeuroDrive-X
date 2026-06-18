param(
    [string]$Model = "",
    [int]$Episodes = 0,
    [string]$Town = "",
    [string]$Weather = "",
    [switch]$Headless,
    [string]$VenvDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($VenvDir)) { $VenvDir = Join-Path $ProjectRoot ".venv" }
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { throw "Virtual environment not found. Run scripts\setup.ps1 first." }

$CarlaEnv = Join-Path $ProjectRoot ".env.carla.ps1"
if (Test-Path $CarlaEnv) { . $CarlaEnv }

$args = @(
    (Join-Path $ProjectRoot "ai\evaluate.py"),
    "--carla-config", (Join-Path $ProjectRoot "configs\carla.yaml"),
    "--training-config", (Join-Path $ProjectRoot "configs\training.yaml")
)
if ($Model) { $args += @("--model", $Model) }
if ($Episodes -gt 0) { $args += @("--episodes", "$Episodes") }
if ($Town) { $args += @("--town", $Town) }
if ($Weather) { $args += @("--weather", $Weather) }
if ($Headless) { $args += "--headless" }

& $PythonExe @args


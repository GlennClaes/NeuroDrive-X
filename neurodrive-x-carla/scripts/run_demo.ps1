param(
    [string]$Model = "",
    [string]$Town = "Town05",
    [string]$Weather = "ClearNoon",
    [int]$MaxSteps = 5000,
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
    (Join-Path $ProjectRoot "ai\inference.py"),
    "--carla-config", (Join-Path $ProjectRoot "configs\carla.yaml"),
    "--training-config", (Join-Path $ProjectRoot "configs\training.yaml"),
    "--town", $Town,
    "--weather", $Weather,
    "--max-steps", "$MaxSteps"
)
if ($Model) { $args += @("--model", $Model) }
if ($Headless) { $args += "--headless" }

& $PythonExe @args


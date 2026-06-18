param(
    [string]$Model = "",
    [string]$Town = "Town05",
    [string]$Weather = "ClearNoon",
    [int]$MaxSteps = 5000,
    [string]$ExtraArgs = "",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"
$demoArgs = @("--town", $Town, "--weather", $Weather, "--max-steps", "$MaxSteps")
if ($Model) { $demoArgs += @("--model", $Model) }
if ($ExtraArgs) { $demoArgs += $ExtraArgs }
$env:DEMO_ARGS = ($demoArgs -join " ")

$args = @("compose", "-f", $ComposeFile, "--profile", "demo", "up")
if ($Build) { $args += "--build" }
$args += @("--abort-on-container-exit", "demo")

Write-Host "[NeuroDrive X] Running Docker demo"
& docker @args


param(
    [int]$Timesteps = 0,
    [string]$Town = "",
    [string]$Weather = "",
    [string]$ExtraArgs = "",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"
$trainArgs = @()
if ($Timesteps -gt 0) { $trainArgs += @("--timesteps", "$Timesteps") }
if ($Town) { $trainArgs += @("--town", $Town) }
if ($Weather) { $trainArgs += @("--weather", $Weather) }
if ($ExtraArgs) { $trainArgs += $ExtraArgs }
$env:TRAIN_ARGS = ($trainArgs -join " ")

$args = @("compose", "-f", $ComposeFile, "--profile", "train", "up")
if ($Build) { $args += "--build" }
$args += @("--abort-on-container-exit", "trainer")

Write-Host "[NeuroDrive X] Running Docker training"
& docker @args


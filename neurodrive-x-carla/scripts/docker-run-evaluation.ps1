param(
    [string]$Model = "",
    [int]$Episodes = 0,
    [string]$Town = "",
    [string]$Weather = "",
    [string]$ExtraArgs = "",
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"
$evalArgs = @()
if ($Model) { $evalArgs += @("--model", $Model) }
if ($Episodes -gt 0) { $evalArgs += @("--episodes", "$Episodes") }
if ($Town) { $evalArgs += @("--town", $Town) }
if ($Weather) { $evalArgs += @("--weather", $Weather) }
if ($ExtraArgs) { $evalArgs += $ExtraArgs }
$env:EVAL_ARGS = ($evalArgs -join " ")

$args = @("compose", "-f", $ComposeFile, "--profile", "evaluate", "up")
if ($Build) { $args += "--build" }
$args += @("--abort-on-container-exit", "evaluator")

Write-Host "[NeuroDrive X] Running Docker evaluation"
& docker @args


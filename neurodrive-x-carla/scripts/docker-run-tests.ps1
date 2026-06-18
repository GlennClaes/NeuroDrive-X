param(
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"
$args = @("compose", "-f", $ComposeFile, "--profile", "test", "up")
if ($Build) { $args += "--build" }
$args += @("--abort-on-container-exit", "tests")

Write-Host "[NeuroDrive X] Running Docker test suite"
& docker @args


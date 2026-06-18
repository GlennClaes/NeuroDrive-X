param(
    [switch]$NoCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"
$args = @("compose", "-f", $ComposeFile, "--profile", "all", "build")
if ($NoCache) { $args += "--no-cache" }

Write-Host "[NeuroDrive X] Building all Docker images"
& docker @args


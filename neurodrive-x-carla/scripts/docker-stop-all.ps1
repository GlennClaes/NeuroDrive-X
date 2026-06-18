Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"

Write-Host "[NeuroDrive X] Stopping Docker stack"
& docker compose -f $ComposeFile --profile all down --remove-orphans


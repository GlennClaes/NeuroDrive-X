param(
    [string]$ImageTag = "latest",
    [string]$QualityLevel = "Low"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"
$env:CARLA_IMAGE_TAG = $ImageTag
$env:CARLA_QUALITY_LEVEL = $QualityLevel

Write-Host "[NeuroDrive X] Starting CARLA simulator container"
& docker compose -f $ComposeFile --profile simulator up -d carla


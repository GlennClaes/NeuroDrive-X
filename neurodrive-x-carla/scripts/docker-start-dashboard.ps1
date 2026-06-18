param(
    [switch]$WithFrontendDev,
    [switch]$Build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"

if ($WithFrontendDev) {
    $args = @("compose", "-f", $ComposeFile, "--profile", "frontend", "up", "-d")
    if ($Build) { $args += "--build" }
    $args += @("dashboard", "frontend-dev")
    Write-Host "[NeuroDrive X] Starting FastAPI dashboard and Vite frontend containers"
    & docker @args
    Write-Host "[NeuroDrive X] React dev UI: http://localhost:5173"
}
else {
    $args = @("compose", "-f", $ComposeFile, "--profile", "dashboard", "up", "-d")
    if ($Build) { $args += "--build" }
    $args += "dashboard"
    Write-Host "[NeuroDrive X] Starting production dashboard container"
    & docker @args
    Write-Host "[NeuroDrive X] Dashboard: http://localhost:8080"
}


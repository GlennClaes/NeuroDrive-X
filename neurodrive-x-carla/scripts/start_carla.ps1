param(
    [switch]$Docker,
    [switch]$Headless,
    [string]$Image = "carlasim/carla:latest",
    [string]$CarlaRoot = "",
    [int]$Port = 2000,
    [string]$QualityLevel = "Low"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Docker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker was not found. Install Docker Desktop or run local CARLA with -CarlaRoot."
    }
    $HostPortEnd = $Port + 2
    $dockerArgs = @("run", "--rm", "-p", "$Port-$HostPortEnd`:2000-2002", $Image)
    if ($Headless) {
        $dockerArgs += @("./CarlaUE4.sh", "-RenderOffScreen", "-quality-level=$QualityLevel", "-carla-rpc-port=2000")
    }
    Write-Host "[NeuroDrive X] Starting CARLA Docker image $Image"
    & docker @dockerArgs
    exit $LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($CarlaRoot)) {
    $CarlaRoot = if ($env:CARLA_ROOT) { $env:CARLA_ROOT } else { Join-Path $HOME "CARLA_0.9.15" }
}

$CarlaExe = Join-Path $CarlaRoot "CarlaUE4.exe"
$CarlaShell = Join-Path $CarlaRoot "CarlaUE4.sh"
if (Test-Path $CarlaExe) {
    $args = @("-carla-rpc-port=$Port", "-quality-level=$QualityLevel")
    if ($Headless) { $args += "-RenderOffScreen" }
    Write-Host "[NeuroDrive X] Starting local CARLA at $CarlaExe"
    & $CarlaExe @args
}
elseif (Test-Path $CarlaShell) {
    $args = @("-carla-rpc-port=$Port", "-quality-level=$QualityLevel")
    if ($Headless) { $args += "-RenderOffScreen" }
    Write-Host "[NeuroDrive X] Starting local CARLA shell launcher at $CarlaShell"
    & $CarlaShell @args
}
else {
    throw "CARLA executable not found. Set -CarlaRoot or use scripts\start_carla.ps1 -Docker -Headless."
}

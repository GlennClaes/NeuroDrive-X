param(
    [string]$Service = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\docker-compose.yml"

if ($Service) {
    & docker compose -f $ComposeFile logs -f $Service
}
else {
    & docker compose -f $ComposeFile logs -f
}


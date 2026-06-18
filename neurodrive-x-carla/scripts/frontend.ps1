param(
    [switch]$Install,
    [switch]$Build,
    [switch]$Preview
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendRoot = Join-Path $ProjectRoot "dashboard\frontend"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js LTS to run the React dashboard."
}

Push-Location $FrontendRoot
try {
    if ($Install -or -not (Test-Path "node_modules")) {
        npm install
    }
    if ($Build) {
        npm run build
    }
    elseif ($Preview) {
        npm run preview
    }
    else {
        Write-Host "[NeuroDrive X] React dashboard: http://localhost:5173"
        Write-Host "[NeuroDrive X] Start scripts\dashboard.ps1 in another PowerShell for the FastAPI backend."
        npm run dev
    }
}
finally {
    Pop-Location
}


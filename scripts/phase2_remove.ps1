<#
.SYNOPSIS
    Safe Removal of Phase 2 9Router Vision Nuclio Function.

.DESCRIPTION
    Safely deletes ONLY the ninerouter-vision Nuclio function using nuctl.

    SAFETY GUARANTEES:
    - NEVER executes 'docker compose down -v'.
    - NEVER touches CVAT volumes, databases, tasks, or other functions.
    - Operates exclusively on the ninerouter-vision function container.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 2 Function Removal" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Determine nuctl command
$nuctlCmd = $null
$useWsl = $false

if (Get-Command nuctl -ErrorAction SilentlyContinue) {
    $nuctlCmd = "nuctl"
} else {
    $wslCheck = wsl -d Ubuntu which nuctl 2>$null
    if ($wslCheck) {
        $nuctlCmd = "wsl -d Ubuntu nuctl"
        $useWsl = $true
    }
}

if (-not $nuctlCmd) {
    Write-Error "nuctl CLI was not found in PATH or WSL Ubuntu. Cannot delete function."
    exit 1
}

Write-Host "Checking if 'ninerouter-vision' is currently deployed..." -ForegroundColor Yellow
$fnExists = $false
if ($useWsl) {
    $fnList = wsl -d Ubuntu nuctl get function --platform local 2>$null
    if ($fnList -match "ninerouter-vision") { $fnExists = $true }
} else {
    $fnList = nuctl get function --platform local 2>$null
    if ($fnList -match "ninerouter-vision") { $fnExists = $true }
}

if (-not $fnExists) {
    Write-Host "Function 'ninerouter-vision' is not currently deployed. Nothing to remove." -ForegroundColor Green
    exit 0
}

Write-Host "Deleting Nuclio function 'ninerouter-vision'..." -ForegroundColor Yellow

if ($useWsl) {
    wsl -d Ubuntu nuctl delete function ninerouter-vision --platform local
} else {
    nuctl delete function ninerouter-vision --platform local
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] 'ninerouter-vision' function has been removed cleanly." -ForegroundColor Green
    Write-Host "CVAT core services, database, tasks, and volumes remain completely untouched." -ForegroundColor Gray
} else {
    Write-Error "Failed to delete function 'ninerouter-vision'."
    exit 1
}

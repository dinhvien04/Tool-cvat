<#
.SYNOPSIS
    Safe Removal of Phase 3 9Router Vision Nuclio Functions.

.DESCRIPTION
    Safely deletes ONLY the selected 9Router Nuclio functions using nuctl:
    - ninerouter-vision (box)
    - ninerouter-vision-mask (mask)
    - ninerouter-vision-box-mask (box-mask)

    SAFETY GUARANTEES:
    - NEVER executes 'docker compose down -v'.
    - NEVER touches CVAT volumes, databases, tasks, or other functions.
    - Operates exclusively on the specified 9Router function containers.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("mask", "box", "box-mask", "both", "all")]
    [string]$Target = "all",

    [Parameter(Mandatory = $false)]
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 3 Function Removal" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Gray

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

$targetsToRemove = @()
switch ($Target) {
    "mask" { $targetsToRemove = @("ninerouter-vision-mask") }
    "box" { $targetsToRemove = @("ninerouter-vision") }
    "box-mask" { $targetsToRemove = @("ninerouter-vision-box-mask") }
    "both" { $targetsToRemove = @("ninerouter-vision", "ninerouter-vision-mask") }
    "all" { $targetsToRemove = @("ninerouter-vision", "ninerouter-vision-mask", "ninerouter-vision-box-mask") }
}

# Fetch running functions
$fnList = $null
if ($useWsl) {
    $fnList = wsl -d Ubuntu nuctl get function --platform local 2>$null
} else {
    $fnList = nuctl get function --platform local 2>$null
}

foreach ($fnName in $targetsToRemove) {
    if ($fnList -match [regex]::Escape($fnName)) {
        Write-Host "Deleting Nuclio function '$fnName'..." -ForegroundColor Yellow
        if ($useWsl) {
            wsl -d Ubuntu nuctl delete function $fnName --platform local
        } else {
            nuctl delete function $fnName --platform local
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  -> Function '$fnName' removed cleanly." -ForegroundColor Green
        } else {
            Write-Error "Failed to delete function '$fnName'."
        }
    } else {
        Write-Host "Function '$fnName' is not currently deployed. Skipping." -ForegroundColor Gray
    }
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " Function Removal Complete." -ForegroundColor Green
Write-Host " CVAT core services, database, tasks, and volumes remain completely untouched." -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Green

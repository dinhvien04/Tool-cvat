<#
.SYNOPSIS
    Safe Removal of Phase 3B 9Router Vision Nuclio Functions.

.DESCRIPTION
    Safely deletes ONLY the selected 9Router Nuclio functions using nuctl:
    - ninerouter-vision-31 (31)
    - ninerouter-vision-box-mask (box-mask)
    - ninerouter-vision-mask (mask)
    - ninerouter-vision (box)

    SAFETY GUARANTEES:
    - NEVER executes 'docker compose down -v'.
    - NEVER touches CVAT volumes, databases, tasks, or other functions.
    - Operates exclusively on the specified 9Router function containers.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("31", "box-mask", "mask", "box", "all")]
    [string]$Target = "31",

    [Parameter(Mandatory = $false)]
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 3B Function Removal" -ForegroundColor Cyan
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
    "31" { $targetsToRemove = @("ninerouter-vision-31") }
    "box-mask" { $targetsToRemove = @("ninerouter-vision-box-mask") }
    "mask" { $targetsToRemove = @("ninerouter-vision-mask") }
    "box" { $targetsToRemove = @("ninerouter-vision") }
    "all" {
        $targetsToRemove = @("ninerouter-vision-31", "ninerouter-vision-box-mask", "ninerouter-vision-mask", "ninerouter-vision")
    }
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
            Write-Host "  -> Warning: nuctl returned exit code $LASTEXITCODE for '$fnName'." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Function '$fnName' is not deployed in Nuclio. Skipping." -ForegroundColor Gray
    }
}

Write-Host "`nRemoval process complete." -ForegroundColor Green

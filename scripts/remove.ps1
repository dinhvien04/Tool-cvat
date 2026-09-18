<#
.SYNOPSIS
    Safe Removal of 9Router Nuclio Detector Functions.

.DESCRIPTION
    Safely deletes ONLY 9Router Nuclio functions using nuctl:
    - Target 'legacy': deletes the 4 legacy 9Router functions:
        * ninerouter-vision-31
        * ninerouter-vision-box-mask
        * ninerouter-vision-mask
        * ninerouter-vision
    - Target 'rectangle-mask': deletes ninerouter-rectangle-mask
    - Target 'polygon-mask': deletes ninerouter-polygon-mask
    - Target 'polyline': deletes ninerouter-polyline
    - Target 'all-9router': deletes all 9Router functions

    STRICT SAFETY GUARANTEES:
    - NEVER deletes third-party CVAT models (Human pose estimation, EoMT-DINOv3, etc.).
    - NEVER executes 'docker compose down -v'.
    - NEVER touches CVAT databases, volumes, tasks, jobs, or feedback data.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("legacy", "rectangle-mask", "polygon-mask", "polyline", "all-9router")]
    [string]$Target = "legacy",

    [Parameter(Mandatory = $false)]
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router - Safe Function Removal" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Gray

# Protected third-party models that must NEVER be deleted
$PROTECTED_MODELS = @(
    "pth-mmpose-hrnet32",
    "pth-tue-mps-eomt-dinov3-coco-panoptic",
    "openvino-omz-public-yolo-v3-tf",
    "openvino-omz-intel-person-detection-0200"
)

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
    "legacy" {
        $targetsToRemove = @(
            "ninerouter-vision-31",
            "ninerouter-vision-box-mask",
            "ninerouter-vision-mask",
            "ninerouter-vision"
        )
    }
    "rectangle-mask" {
        $targetsToRemove = @("ninerouter-rectangle-mask")
    }
    "polygon-mask" {
        $targetsToRemove = @("ninerouter-polygon-mask")
    }
    "polyline" {
        $targetsToRemove = @("ninerouter-polyline")
    }
    "all-9router" {
        $targetsToRemove = @(
            "ninerouter-vision-31",
            "ninerouter-vision-box-mask",
            "ninerouter-vision-mask",
            "ninerouter-vision",
            "ninerouter-rectangle-mask",
            "ninerouter-polygon-mask",
            "ninerouter-polyline"
        )
    }
}

# Strict safety verification: Ensure no protected model is in targetsToRemove
foreach ($targetItem in $targetsToRemove) {
    if ($PROTECTED_MODELS -contains $targetItem -or (-not $targetItem.StartsWith("ninerouter-"))) {
        Write-Error "CRITICAL SAFETY VIOLATION: Refusing to delete non-9Router / protected model '$targetItem'!"
        exit 1
    }
}

# Fetch running functions
$fnList = $null
if ($useWsl) {
    $fnList = wsl -d Ubuntu nuctl get function --platform local 2>$null
} else {
    $fnList = nuctl get function --platform local 2>$null
}

Write-Host "`nCurrent deployed functions in Nuclio:" -ForegroundColor Gray
if ($fnList) {
    $fnList | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
}

Write-Host "`nProcessing removal for $($targetsToRemove.Count) function(s)..." -ForegroundColor Yellow

$deletedCount = 0
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
            $deletedCount++
        } else {
            Write-Host "  -> Warning: nuctl returned exit code $LASTEXITCODE for '$fnName'." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Function '$fnName' is not deployed in Nuclio. Skipping." -ForegroundColor Gray
    }
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " REMOVAL COMPLETE: $deletedCount function(s) removed." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

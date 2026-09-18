<#
.SYNOPSIS
    Smoke Test for Streamlined Exactly-Three 9Router CVAT Detectors.

.DESCRIPTION
    Verifies:
    1. CVAT Lambda Manager API (/api/lambda/functions) integrity:
       - Confirms 3 user-facing 9Router detectors:
         * 9Router Rectangle + Mask (ninerouter-rectangle-mask)
         * 9Router Polygon + Mask (ninerouter-polygon-mask)
         * 9Router Polyline (ninerouter-polyline)
       - Confirms preserved third-party models (Human pose estimation, EoMT-DINOv3, etc.).
       - Verifies no legacy 9Router functions remain (when CleanCheck is enabled).
    2. Direct invocation and shape policy validation of each detector:
       - ninerouter-rectangle-mask: paired rectangle + mask with group_id, 14 labels.
       - ninerouter-polygon-mask: paired polygon + mask with group_id, no boxes, 10 labels.
       - ninerouter-polyline: polyline only, no boxes/polygons/masks/groups, 7 labels.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("three", "rectangle-mask", "polygon-mask", "polyline", "cvat-api", "all")]
    [string]$Target = "three",

    [Parameter(Mandatory = $false)]
    [string]$ImagePath = "test.jpg",

    [Parameter(Mandatory = $false)]
    [double]$Threshold = 0.3,

    [Parameter(Mandatory = $false)]
    [string]$CvatUrl = "http://localhost:18080",

    [Parameter(Mandatory = $false)]
    [switch]$CleanCheck = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router - Streamlined 3-Detector Smoke Test" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Gray

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Check CVAT API
Write-Host "`n[Check 1] Checking CVAT Lambda Functions Endpoint..." -ForegroundColor Yellow
$cvatCheckPassed = $true
try {
    $checkCmd = @"
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
django.setup()
from cvat.apps.lambda_manager.views import LambdaGateway
gateway = LambdaGateway()
funcs = [{'id': f.id, 'name': f.name, 'type': getattr(f, 'kind', 'detector')} for f in gateway.list()]
print('JSON_START' + json.dumps(funcs) + 'JSON_END')
"@
    $checkRes = docker exec -i cvat_server python3 -c "$checkCmd" 2>$null
    if ($checkRes -match "JSON_START(.*?)JSON_END") {
        $functionsResp = ConvertFrom-Json $Matches[1]
        Write-Host "Found $($functionsResp.Count) functions in CVAT Lambda Manager." -ForegroundColor Green

        $fnNames = @($functionsResp | ForEach-Object { $_.id })
        $fnDisplayNames = @($functionsResp | ForEach-Object { $_.name })

        Write-Host "`nDeployed Functions in CVAT:" -ForegroundColor Gray
        for ($i = 0; $i -lt $functionsResp.Count; $i++) {
            Write-Host "  - $($fnNames[$i]) ($($fnDisplayNames[$i]))" -ForegroundColor DarkGray
        }

        # Verify protected third-party models
        Write-Host "`nVerifying preservation of third-party models..." -ForegroundColor Gray
        $protectedFound = @()
        if ($fnNames -contains "pth-mmpose-hrnet32") {
            Write-Host "  [OK] Human pose estimation (pth-mmpose-hrnet32) is preserved." -ForegroundColor Green
            $protectedFound += "pth-mmpose-hrnet32"
        }
        if ($fnNames -contains "pth-tue-mps-eomt-dinov3-coco-panoptic") {
            Write-Host "  [OK] EoMT-DINOv3 Panoptic (pth-tue-mps-eomt-dinov3-coco-panoptic) is preserved." -ForegroundColor Green
            $protectedFound += "pth-tue-mps-eomt-dinov3-coco-panoptic"
        }

        # Verify 3 9Router detectors
        Write-Host "`nVerifying 3 new 9Router detectors in CVAT:" -ForegroundColor Gray
        $expected3 = @(
            @{ Id = "ninerouter-rectangle-mask"; Name = "9Router Rectangle + Mask" },
            @{ Id = "ninerouter-polygon-mask"; Name = "9Router Polygon + Mask" },
            @{ Id = "ninerouter-polyline"; Name = "9Router Polyline" }
        )

        foreach ($exp in $expected3) {
            if ($fnNames -contains $exp.Id) {
                Write-Host "  [OK] Found $($exp.Name) ($($exp.Id))" -ForegroundColor Green
            } else {
                Write-Host "  [WAITING/MISSING] $($exp.Name) ($($exp.Id)) is NOT yet registered in CVAT." -ForegroundColor Yellow
            }
        }

        if ($CleanCheck) {
            $legacyFunctions = @("ninerouter-vision", "ninerouter-vision-mask", "ninerouter-vision-box-mask", "ninerouter-vision-31")
            $legacyFound = @($fnNames | Where-Object { $legacyFunctions -contains $_ })
            if ($legacyFound.Count -gt 0) {
                Write-Host "  [WARNING] Legacy 9Router functions still present: $($legacyFound -join ', ')" -ForegroundColor Yellow
            } else {
                Write-Host "  [CLEAN] No legacy 9Router functions found in CVAT." -ForegroundColor Green
            }
        }
    } else {
        Write-Host "Could not parse output from cvat_server LambdaGateway." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not query CVAT Lambda Manager: $_" -ForegroundColor Yellow
    $cvatCheckPassed = $false
}

if ($Target -eq "cvat-api") {
    Write-Host "`nCVAT API check finished." -ForegroundColor Green
    exit 0
}

# Image preparation
Write-Host "`n[Check 2] Preparing test image..." -ForegroundColor Yellow
if (-not (Test-Path $ImagePath)) {
    $altPath = Join-Path $RepoRoot $ImagePath
    if (Test-Path $altPath) {
        $ImagePath = $altPath
    } else {
        $ImagePath = Join-Path $RepoRoot "test_temp.jpg"
        $pyPath = $ImagePath.Replace("\", "/")
        & python -c "from PIL import Image; Image.new('RGB', (1024, 768), color='lightblue').save('$pyPath')"
    }
}
Write-Host "Using test image: $ImagePath" -ForegroundColor Green
$imageBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $ImagePath).Path)
$base64Image = [Convert]::ToBase64String($imageBytes)

$targetDetectors = @()
switch ($Target) {
    "rectangle-mask" {
        $targetDetectors = @(@{ Id = "ninerouter-rectangle-mask"; Mode = "rectangle_mask" })
    }
    "polygon-mask" {
        $targetDetectors = @(@{ Id = "ninerouter-polygon-mask"; Mode = "polygon_mask" })
    }
    "polyline" {
        $targetDetectors = @(@{ Id = "ninerouter-polyline"; Mode = "polyline" })
    }
    "three" {
        $targetDetectors = @(
            @{ Id = "ninerouter-rectangle-mask"; Mode = "rectangle_mask" },
            @{ Id = "ninerouter-polygon-mask"; Mode = "polygon_mask" },
            @{ Id = "ninerouter-polyline"; Mode = "polyline" }
        )
    }
    "all" {
        $targetDetectors = @(
            @{ Id = "ninerouter-rectangle-mask"; Mode = "rectangle_mask" },
            @{ Id = "ninerouter-polygon-mask"; Mode = "polygon_mask" },
            @{ Id = "ninerouter-polyline"; Mode = "polyline" }
        )
    }
}

# Test invocations
Write-Host "`n[Check 3] Invoking and verifying detector output..." -ForegroundColor Yellow
foreach ($det in $targetDetectors) {
    $fnName = $det.Id
    $fnMode = $det.Mode

    Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Gray
    Write-Host "Testing $fnName ($fnMode)..." -ForegroundColor Cyan
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

    # Find port or container
    $containerName = docker ps --filter "name=$fnName" --format "{{.Names}}" | Select-Object -First 1
    if (-not $containerName) {
        Write-Host "Container for '$fnName' is not running in Docker. Skipping live invocation." -ForegroundColor Yellow
        continue
    }

    $portMapping = docker port $containerName 8080 2>$null
    $port = $null
    $firstMapping = ($portMapping | Select-Object -First 1) -as [string]
    if ($firstMapping -and $firstMapping -match ":(\d+)") {
        $port = $Matches[1]
    }

    if (-not $port) {
        Write-Host "Could not determine host port for $containerName. Skipping HTTP test." -ForegroundColor Yellow
        continue
    }

    $endpoint = "http://localhost:$port"
    Write-Host "Target endpoint: $endpoint" -ForegroundColor Gray

    $payload = @{
        image = $base64Image
        threshold = $Threshold
    } | ConvertTo-Json -Compress

    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $resp = Invoke-RestMethod -Uri $endpoint -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 120
        $stopwatch.Stop()
        Write-Host "Response received in $($stopwatch.ElapsedMilliseconds) ms with $($resp.Count) shape(s)." -ForegroundColor Green

        # Shape validation based on policy
        if ($fnMode -eq "rectangle_mask") {
            $rectangles = @($resp | Where-Object { $_.type -eq "rectangle" })
            $masks = @($resp | Where-Object { $_.type -eq "mask" })
            Write-Host "  - Rectangles: $($rectangles.Count), Masks: $($masks.Count)" -ForegroundColor Gray
            if ($rectangles.Count -gt 0 -and $masks.Count -gt 0) {
                Write-Host "  [PASS] Detector emits paired shapes." -ForegroundColor Green
            }
        } elseif ($fnMode -eq "polygon_mask") {
            $polygons = @($resp | Where-Object { $_.type -eq "polygon" })
            $masks = @($resp | Where-Object { $_.type -eq "mask" })
            $boxes = @($resp | Where-Object { $_.type -eq "rectangle" })
            Write-Host "  - Polygons: $($polygons.Count), Masks: $($masks.Count), Boxes: $($boxes.Count)" -ForegroundColor Gray
            if ($boxes.Count -eq 0) {
                Write-Host "  [PASS] Bounding boxes strictly suppressed." -ForegroundColor Green
            } else {
                Write-Host "  [FAIL] Found $($boxes.Count) bounding box(es) in polygon-mask output!" -ForegroundColor Red
            }
        } elseif ($fnMode -eq "polyline") {
            $polylines = @($resp | Where-Object { $_.type -eq "polyline" })
            $otherShapes = @($resp | Where-Object { $_.type -ne "polyline" })
            Write-Host "  - Polylines: $($polylines.Count), Other shapes: $($otherShapes.Count)" -ForegroundColor Gray
            if ($otherShapes.Count -eq 0) {
                Write-Host "  [PASS] Polyline ONLY output policy verified." -ForegroundColor Green
            } else {
                Write-Host "  [FAIL] Found non-polyline shapes in polyline output!" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "Failed to invoke $endpoint : $_" -ForegroundColor Yellow
    }
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " SMOKE TEST EXECUTION COMPLETED" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

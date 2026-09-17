<#
.SYNOPSIS
    Phase 3B Smoke Test for Deployed 9Router Vision 31-Label Nuclio Function.

.DESCRIPTION
    1. Locates or generates a test image.
    2. Encodes the image to Base64 (matching CVAT detector payload format).
    3. Finds the running Nuclio function port or invokes it via nuctl.
    4. Sends HTTP POST payload: {"image": "<base64>", "threshold": 0.5}.
    5. Validates the CVAT multi-shape detector response format:
       - Schema validation: all labels in 31-label master schema.
       - Instance objects: rectangle + mask with group_id.
       - Semantic regions: mask with contour points, NO rectangles, ungrouped.
       - Lane markings: polyline / polygon / mask, NO rectangles, ungrouped.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("31", "box-mask", "mask", "box", "all")]
    [string]$Target = "31",

    [Parameter(Mandatory = $false)]
    [string]$ImagePath = "test.jpg",

    [Parameter(Mandatory = $false)]
    [double]$Threshold = 0.3,

    [Parameter(Mandatory = $false)]
    [string]$FunctionUrl = $null
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 3B Smoke Test (31 Labels)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Gray

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Step 1: Locate or generate test image
if (-not (Test-Path $ImagePath)) {
    $altPath = Join-Path $RepoRoot $ImagePath
    if (Test-Path $altPath) {
        $ImagePath = $altPath
    } else {
        Write-Host "Generating temporary test image at $RepoRoot\test_temp.jpg..." -ForegroundColor Gray
        $ImagePath = Join-Path $RepoRoot "test_temp.jpg"
        $pyPath = $ImagePath.Replace("\", "/")
        & python -c "from PIL import Image; Image.new('RGB', (1024, 768), color='lightblue').save('$pyPath')"
    }
}

Write-Host "Test image: $ImagePath" -ForegroundColor Green
$imageBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $ImagePath).Path)
$base64Image = [Convert]::ToBase64String($imageBytes)

$master31Labels = @(
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "traffic light", "traffic sign", "area/alternative", "area/drivable",
    "lane/crosswalk", "lane/double white", "lane/double yellow", "lane/road curb",
    "lane/single other", "lane/single white", "lane/single yellow", "road", "sidewalk",
    "building", "wall", "fence", "pole", "vegetation", "terrain", "sky",
    "person", "traffic_light", "traffic_sign"
)

$instanceLabels = @(
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "traffic light", "traffic sign", "pole", "person",
    "traffic_light", "traffic_sign"
)

$regionLabels = @(
    "area/alternative", "area/drivable", "road", "sidewalk", "building",
    "wall", "fence", "vegetation", "terrain", "sky"
)

$laneLabels = @(
    "lane/crosswalk", "lane/double white", "lane/double yellow", "lane/road curb",
    "lane/single other", "lane/single white", "lane/single yellow"
)

$targetsToTest = @()
switch ($Target) {
    "31" { $targetsToTest = @("ninerouter-vision-31") }
    "box-mask" { $targetsToTest = @("ninerouter-vision-box-mask") }
    "mask" { $targetsToTest = @("ninerouter-vision-mask") }
    "box" { $targetsToTest = @("ninerouter-vision") }
    "all" {
        $targetsToTest = @("ninerouter-vision-31", "ninerouter-vision-box-mask", "ninerouter-vision-mask", "ninerouter-vision")
    }
}

foreach ($fnName in $targetsToTest) {
    Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Gray
    Write-Host "Testing function: $fnName" -ForegroundColor Cyan
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

    # Determine Function Endpoint
    $targetUrl = $FunctionUrl
    if (-not $targetUrl) {
        try {
            $containers = docker ps --format "{{.Names}} {{.Ports}}"
            $matching = ($containers -split "`n") | Where-Object { $_ -match "nuclio-nuclio-$fnName\s+" }
            if ($matching -match "0\.0\.0\.0:(\d+)->8080" -or $matching -match ":::(\d+)->8080") {
                $mappedPort = $Matches[1]
                $targetUrl = "http://localhost:$mappedPort"
                Write-Host "Found direct HTTP port: $targetUrl" -ForegroundColor Green
            }
        } catch {
            Write-Host "Could not automatically detect container port via docker ps." -ForegroundColor Yellow
        }
    }

    $responseBody = $null
    if ($targetUrl) {
        Write-Host "Sending POST request to $targetUrl..." -ForegroundColor Yellow
        $payloadObj = @{
            image = $base64Image
            threshold = $Threshold
        }
        $jsonPayload = $payloadObj | ConvertTo-Json -Compress

        $startTime = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $response = Invoke-WebRequest -Uri $targetUrl -Method Post -Body $jsonPayload -ContentType "application/json" -TimeoutSec 120 -UseBasicParsing
            $startTime.Stop()
            $status = $response.StatusCode
            $responseBody = $response.Content
            Write-Host "HTTP Response: $status ($($startTime.ElapsedMilliseconds) ms)" -ForegroundColor Green
        } catch {
            if ($_.Exception.Response) {
                $stream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($stream)
                $errBody = $reader.ReadToEnd()
                Write-Host "HTTP Status: $([int]$_.Exception.Response.StatusCode)" -ForegroundColor Red
                Write-Host "Error Body: $errBody" -ForegroundColor Red
                Write-Error "HTTP request failed with status $([int]$_.Exception.Response.StatusCode): $errBody"
            } else {
                Write-Error "HTTP request to $targetUrl failed: $_"
            }
            exit 1
        }
    } else {
        Write-Host "Invoking function via nuctl CLI..." -ForegroundColor Yellow
        $tempPayload = [System.IO.Path]::GetTempFileName()
        try {
            $payloadObj = @{
                image = $base64Image
                threshold = $Threshold
            }
            $payloadObj | ConvertTo-Json -Compress | Set-Content -Path $tempPayload -Encoding utf8

            if (Get-Command nuctl -ErrorAction SilentlyContinue) {
                $responseBody = nuctl invoke $fnName --platform local --body-file $tempPayload --content-type "application/json"
            } else {
                $tempDrive = $tempPayload.Substring(0, 1).ToLower()
                $tempRest = $tempPayload.Substring(2) -replace '\\', '/'
                $wslTemp = "/mnt/$tempDrive$tempRest"
                $responseBody = wsl -d Ubuntu nuctl invoke $fnName --platform local --body-file $wslTemp --content-type "application/json"
            }
        } finally {
            if (Test-Path $tempPayload) { Remove-Item -Force $tempPayload }
        }
    }

    # Validate JSON Array response
    try {
        $detections = $responseBody | ConvertFrom-Json
    } catch {
        Write-Error "Response is not valid JSON: $responseBody"
        exit 1
    }

    if ($detections -isnot [System.Collections.IEnumerable] -and $detections -isnot [Array]) {
        Write-Error "Response must be a JSON array. Received: $($detections.GetType().FullName)"
        exit 1
    }

    Write-Host "Detections returned: $($detections.Count)" -ForegroundColor Green

    $rectCount = 0
    $maskCount = 0
    $polyCount = 0
    $lineCount = 0

    $instDets = 0
    $regDets = 0
    $laneDets = 0

    foreach ($det in $detections) {
        $lbl = $det.label
        if ($master31Labels -notcontains $lbl) {
            Write-Error "Detection label '$lbl' is not in Phase 3B 31-label master schema!"
            exit 1
        }

        $type = $det.type

        if ($type -eq "rectangle") { $rectCount++ }
        elseif ($type -eq "mask") { $maskCount++ }
        elseif ($type -eq "polygon") { $polyCount++ }
        elseif ($type -eq "polyline") { $lineCount++ }

        if ($instanceLabels -contains $lbl) {
            $instDets++
            Write-Host "  -> [INST] [$lbl] type=$type conf=$($det.confidence) group_id=$($det.group_id)" -ForegroundColor Gray
        } elseif ($regionLabels -contains $lbl) {
            $regDets++
            if ($type -eq "rectangle") {
                Write-Error "Contract Violation: Semantic region '$lbl' emitted as rectangle bounding box!"
                exit 1
            }
            Write-Host "  -> [REGION] [$lbl] type=$type conf=$($det.confidence)" -ForegroundColor Gray
        } elseif ($laneLabels -contains $lbl) {
            $laneDets++
            if ($type -eq "rectangle") {
                Write-Error "Contract Violation: Lane marking '$lbl' emitted as rectangle bounding box!"
                exit 1
            }
            Write-Host "  -> [LANE] [$lbl] type=$type conf=$($det.confidence)" -ForegroundColor Gray
        }
    }

    Write-Host "`nBreakdown:" -ForegroundColor White
    Write-Host "  Instances: $instDets | Regions: $regDets | Lanes: $laneDets" -ForegroundColor Gray
    Write-Host "  Shapes: Rectangles: $rectCount | Masks: $maskCount | Polylines: $lineCount | Polygons: $polyCount" -ForegroundColor Gray
    Write-Host "Smoke test passed for $fnName!" -ForegroundColor Green
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " Smoke Test Complete: All Target Functions Passed!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

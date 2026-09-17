<#
.SYNOPSIS
    Phase 3 Smoke Test for Deployed 9Router Vision Nuclio Functions.

.DESCRIPTION
    1. Locates or generates a synthetic test image.
    2. Encodes the image to Base64 (matching CVAT payload format).
    3. Finds the running Nuclio function port or invokes it via nuctl.
    4. Sends HTTP POST payload: {"image": "<base64>", "threshold": 0.5}.
    5. Validates the CVAT detector response format:
       - Mask detector: JSON array of objects with type="mask", mask=[p0, p1, ..., xmin, ymin, xmax, ymax]
       - Box detector: JSON array of objects with type="rectangle", points=[x1, y1, x2, y2]
       - Box+Mask detector: JSON array of objects with both rectangle and mask shapes, paired by group_id
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("mask", "box", "box-mask", "all")]
    [string]$Target = "mask",

    [Parameter(Mandatory = $false)]
    [string]$ImagePath = "test.jpg",

    [Parameter(Mandatory = $false)]
    [double]$Threshold = 0.5,

    [Parameter(Mandatory = $false)]
    [string]$FunctionUrl = $null
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 3 Smoke Test" -ForegroundColor Cyan
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
        Write-Host "Generating temporary synthetic test image at $RepoRoot\test_temp.jpg..." -ForegroundColor Gray
        $ImagePath = Join-Path $RepoRoot "test_temp.jpg"
        & python -c "from PIL import Image; Image.new('RGB', (640, 480), color='lightblue').save('$ImagePath')"
    }
}

Write-Host "Test image: $ImagePath" -ForegroundColor Green
$imageBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $ImagePath).Path)
$base64Image = [Convert]::ToBase64String($imageBytes)

$supportedLabels = @(
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "traffic light", "traffic sign", "person", "traffic_light", "traffic_sign"
)

$targetsToTest = @()
switch ($Target) {
    "mask" { $targetsToTest = @("ninerouter-vision-mask") }
    "box" { $targetsToTest = @("ninerouter-vision") }
    "box-mask" { $targetsToTest = @("ninerouter-vision-box-mask") }
    "all" { $targetsToTest = @("ninerouter-vision-mask", "ninerouter-vision", "ninerouter-vision-box-mask") }
}

foreach ($fnName in $targetsToTest) {
    Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Gray
    Write-Host "Testing function: $fnName" -ForegroundColor Cyan
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

    # Determine Function Endpoint
    $targetUrl = $FunctionUrl
    if (-not $targetUrl) {
        try {
            $portStr = docker ps --filter "name=$fnName" --format "{{.Ports}}"
            if ($portStr -match "0\.0\.0\.0:(\d+)->8080") {
                $mappedPort = $Matches[1]
                $targetUrl = "http://localhost:$mappedPort"
                Write-Host "Found direct HTTP port: $targetUrl" -ForegroundColor Green
            } elseif ($portStr -match ":::(\d+)->8080") {
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
                $wslTemp = (wsl -d Ubuntu wslpath -u ($tempPayload -replace '\\', '/')).Trim()
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

    foreach ($det in $detections) {
        if ($supportedLabels -notcontains $det.label) {
            Write-Error "Detection label '$($det.label)' is not in supported CVAT detector labels."
            exit 1
        }

        if ($fnName -eq "ninerouter-vision-mask") {
            if ($det.type -ne "mask") {
                Write-Error "Expected type='mask' for mask detector, got '$($det.type)'"
                exit 1
            }
            if (-not $det.mask -or $det.mask.Count -lt 5) {
                Write-Error "Mask detector shape must contain 'mask' array with at least 5 elements ([crop_pixels..., xmin, ymin, xmax, ymax]). Got: $($det.mask.Count)"
                exit 1
            }
            $xmin = $det.mask[$det.mask.Count - 4]
            $ymin = $det.mask[$det.mask.Count - 3]
            $xmax = $det.mask[$det.mask.Count - 2]
            $ymax = $det.mask[$det.mask.Count - 1]
            Write-Host "  -> [MASK] [$($det.label)] conf=$($det.confidence) bbox=[$xmin, $ymin, $xmax, $ymax] crop_pixels=$($det.mask.Count - 4)" -ForegroundColor Gray
        } elseif ($fnName -eq "ninerouter-vision") {
            if ($det.type -ne "rectangle") {
                Write-Error "Expected type='rectangle' for box detector, got '$($det.type)'"
                exit 1
            }
            if (-not $det.points -or $det.points.Count -ne 4) {
                Write-Error "Box points must be [x1, y1, x2, y2]. Received: $($det.points -join ', ')"
                exit 1
            }
            Write-Host "  -> [RECT] [$($det.label)] conf=$($det.confidence) points=[$($det.points -join ', ')]" -ForegroundColor Gray
        } elseif ($fnName -eq "ninerouter-vision-box-mask") {
            if ($det.type -notin @("rectangle", "mask")) {
                Write-Error "Expected type 'rectangle' or 'mask' for Box+Mask detector, got '$($det.type)'"
                exit 1
            }
            Write-Host "  -> [$($det.type.ToUpper())] [$($det.label)] conf=$($det.confidence) group_id=$($det.group_id)" -ForegroundColor Gray
        }
    }
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " Phase 3 Smoke Test PASSED! All tested functions conform to contract." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

<#
.SYNOPSIS
    Phase 2 Smoke Test for Deployed 9Router Vision Nuclio Detector.

.DESCRIPTION
    1. Locates or generates a test image (e.g. test.jpg).
    2. Encodes the image to Base64 (without Data URL prefix, matching CVAT).
    3. Finds the running Nuclio function port or invokes it via nuctl.
    4. Sends HTTP POST payload: {"image": "<base64>", "threshold": 0.5}.
    5. Validates the CVAT detector response format:
       - Status 200 OK
       - JSON Array of objects
       - Each object has "confidence", "label", "points" ([x1, y1, x2, y2]), "type": "rectangle"
       - Labels belong to the 13 supported CVAT detector labels.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$ImagePath = "test.jpg",

    [Parameter(Mandatory = $false)]
    [double]$Threshold = 0.5,

    [Parameter(Mandatory = $false)]
    [string]$FunctionUrl = $null
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 2 Smoke Test" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

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

# Step 2: Determine Function Endpoint
$targetUrl = $FunctionUrl
if (-not $targetUrl) {
    Write-Host "Detecting running ninerouter-vision container port..." -ForegroundColor Gray
    try {
        $portStr = docker ps --filter "name=ninerouter-vision" --format "{{.Ports}}"
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

# Step 3: Invoke Function
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
    # Create temp payload file
    $tempPayload = [System.IO.Path]::GetTempFileName()
    try {
        $payloadObj = @{
            image = $base64Image
            threshold = $Threshold
        }
        $payloadObj | ConvertTo-Json -Compress | Set-Content -Path $tempPayload -Encoding utf8

        # Run nuctl invoke
        if (Get-Command nuctl -ErrorAction SilentlyContinue) {
            $responseBody = nuctl invoke ninerouter-vision --platform local --body-file $tempPayload --content-type "application/json"
        } else {
            $wslTemp = (wsl -d Ubuntu wslpath -u ($tempPayload -replace '\\', '/')).Trim()
            $responseBody = wsl -d Ubuntu nuctl invoke ninerouter-vision --platform local --body-file $wslTemp --content-type "application/json"
        }
    } finally {
        if (Test-Path $tempPayload) { Remove-Item -Force $tempPayload }
    }
}

# Step 4: Validate CVAT Detector Response Format
Write-Host "`nValidating CVAT Detector response contract..." -ForegroundColor Yellow
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

Write-Host "Detections count: $($detections.Count)" -ForegroundColor Green

$supportedLabels = @(
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "traffic light", "traffic sign", "person", "traffic_light", "traffic_sign"
)

$validCount = 0
foreach ($det in $detections) {
    if ($det.type -ne "rectangle") {
        Write-Error "Invalid detection type: $($det.type). Expected 'rectangle'."
        exit 1
    }
    if ($supportedLabels -notcontains $det.label) {
        Write-Error "Detection label '$($det.label)' is not in supported CVAT detector labels."
        exit 1
    }
    if (-not $det.points -or $det.points.Count -ne 4) {
        Write-Error "Detection points must be [x1, y1, x2, y2]. Received: $($det.points -join ', ')"
        exit 1
    }
    if ($det.PSObject.Properties['confidence'] -and $det.confidence -ne $null) {
        $conf = [double]$det.confidence
        if ($conf -lt 0.0 -or $conf -gt 1.0) {
            Write-Error "Detection confidence must be in range [0.0, 1.0]. Received: $conf"
            exit 1
        }
        $confDisplay = $det.confidence
    } else {
        $confDisplay = "<omitted>"
    }
    $validCount++
    Write-Host "  -> [$($det.label)] confidence=$confDisplay points=[$($det.points -join ', ')]" -ForegroundColor Gray
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " Smoke Test PASSED! ($validCount valid detections verified)" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

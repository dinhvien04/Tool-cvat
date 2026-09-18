<#
.SYNOPSIS
    Deploy Exactly Three 9Router CVAT AI Detectors.

.DESCRIPTION
    Deploys the streamlined 3-function 9Router detector suite:
    1. 9Router Rectangle + Mask (ninerouter-rectangle-mask) - 14 foreground instances
    2. 9Router Polygon + Mask (ninerouter-polygon-mask) - 10 semantic regions
    3. 9Router Polyline (ninerouter-polyline) - 7 lane demarcations

    SAFETY GUARANTEES:
    - NEVER deletes volumes or databases.
    - NEVER runs 'docker compose down -v'.
    - Third-party models (Human pose estimation, EoMT-DINOv3, etc.) are strictly protected.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("three", "rectangle-mask", "polygon-mask", "polyline", "all")]
    [string]$Target = "three",

    [Parameter(Mandatory = $false)]
    [string]$CvatRoot = $env:CVAT_ROOT,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterUrl = "http://host.docker.internal:20128",

    [Parameter(Mandatory = $false)]
    [string]$VisionModel = $env:VISION_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$RectangleMaskModel = $env:RECTANGLE_MASK_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$PolygonMaskModel = $env:POLYGON_MASK_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$PolylineModel = $env:POLYLINE_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterKey = $env:NINEROUTER_KEY,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterTimeout = "180.0",

    [Parameter(Mandatory = $false)]
    [string]$CvatWebhookSecret = $env:CVAT_WEBHOOK_SECRET
)

$ErrorActionPreference = "Stop"

if (-not $VisionModel) { $VisionModel = "ag/gemini-3.8-flash-low" }
if (-not $RectangleMaskModel) { $RectangleMaskModel = "ag/gemini-3.8-flash-low" }
if (-not $PolygonMaskModel) { $PolygonMaskModel = "ag/gemini-3.8-flash-low" }
if (-not $PolylineModel) { $PolylineModel = "ag/gemini-3.8-flash-low" }

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router - Deploy Streamlined 3-Detector Suite" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Gray

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Determine functions to deploy
$functionsToDeploy = @()
switch ($Target) {
    "rectangle-mask" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-rectangle-mask"; DisplayName = "9Router Rectangle + Mask"; Mode = "rectangle_mask" }
        )
    }
    "polygon-mask" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-polygon-mask"; DisplayName = "9Router Polygon + Mask"; Mode = "polygon_mask" }
        )
    }
    "polyline" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-polyline"; DisplayName = "9Router Polyline"; Mode = "polyline" }
        )
    }
    "three" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-rectangle-mask"; DisplayName = "9Router Rectangle + Mask"; Mode = "rectangle_mask" },
            @{ Name = "ninerouter-polygon-mask"; DisplayName = "9Router Polygon + Mask"; Mode = "polygon_mask" },
            @{ Name = "ninerouter-polyline"; DisplayName = "9Router Polyline"; Mode = "polyline" }
        )
    }
    "all" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-rectangle-mask"; DisplayName = "9Router Rectangle + Mask"; Mode = "rectangle_mask" },
            @{ Name = "ninerouter-polygon-mask"; DisplayName = "9Router Polygon + Mask"; Mode = "polygon_mask" },
            @{ Name = "ninerouter-polyline"; DisplayName = "9Router Polyline"; Mode = "polyline" }
        )
    }
}

# Step 1: Locate CVAT root safely
Write-Host "`n[Step 1/4] Locating CVAT repository..." -ForegroundColor Yellow
$candidates = @()
if ($CvatRoot) { $candidates += $CvatRoot }
$candidates += "D:\cvat"
$candidates += "C:\cvat"
$candidates += "$env:USERPROFILE\cvat"

$resolvedCvatRoot = $null
foreach ($cand in $candidates) {
    if (Test-Path "$cand\docker-compose.yml") {
        $resolvedCvatRoot = (Resolve-Path $cand).Path
        break
    }
}

if (-not $resolvedCvatRoot) {
    Write-Error "Could not locate CVAT installation directory. Please set CVAT_ROOT."
    exit 1
}
Write-Host "CVAT root directory: $resolvedCvatRoot" -ForegroundColor Green

# Step 2: Ensure CVAT serverless services are active
Write-Host "`n[Step 2/4] Ensuring CVAT serverless stack is running..." -ForegroundColor Yellow
$serverlessCompose = Join-Path $resolvedCvatRoot "components\serverless\docker-compose.serverless.yml"
$serverlessOverride = Join-Path $resolvedCvatRoot "docker-compose.serverless.override.yml"

Push-Location $resolvedCvatRoot
try {
    $composeFiles = @("-f", "docker-compose.yml", "-f", "components/serverless/docker-compose.serverless.yml")
    if (Test-Path $serverlessOverride) {
        $composeFiles += @("-f", "docker-compose.serverless.override.yml")
    }
    docker compose @composeFiles up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start CVAT serverless containers."
        exit 1
    }
    Write-Host "CVAT serverless stack is active." -ForegroundColor Green
} finally {
    Pop-Location
}

# Step 3: Resolve dynamic vision model
Write-Host "`n[Step 3/4] Resolving active vision model from 9Router..." -ForegroundColor Yellow
$hostUrl = "http://127.0.0.1:20128"
if ($NineRouterUrl -match "host\.docker\.internal") {
    $modelsCheckUrl = "$hostUrl/v1/models"
} else {
    $modelsCheckUrl = "$NineRouterUrl/v1/models"
}

$resolvedVisionModel = $null
try {
    $authHeaders = @{}
    if ($NineRouterKey -and $NineRouterKey.Trim() -ne "") {
        $authHeaders["Authorization"] = "Bearer $NineRouterKey"
    }
    $modelsResp = Invoke-RestMethod -Uri $modelsCheckUrl -Headers $authHeaders -Method Get -TimeoutSec 5
    $models = $modelsResp.data
    $visionModels = @($models | Where-Object {
        $_.capabilities.vision -eq $true -or
        ($_.id -match "(vision|vl|gemini|claude|gpt-4o|4o-mini|qwen-vl)" -and $_.capabilities.vision -ne $false)
    })

    if ($VisionModel -and $VisionModel.Trim() -ne "") {
        $resolvedVisionModel = $VisionModel
    } else {
        $repoPath = ($RepoRoot -replace '\\', '/')
        $probeCmd = "import os, sys; sys.path.insert(0, '$repoPath'); from app.client import NineRouterClient; c = NineRouterClient(base_url='$hostUrl', api_key=os.getenv('NINEROUTER_KEY')); m = c.resolve_segmentation_model(preferred_model=None); print('OK:' + m)"
        $segOut = python -c "$probeCmd" 2>$null
        if ($segOut -match "OK:(.+)") {
            $resolvedVisionModel = $Matches[1].Trim()
        } else {
            $resolvedVisionModel = $visionModels[0].id
        }
    }
} catch {
    Write-Error "Failed to query 9Router models at $($modelsCheckUrl): $_"
    exit 1
}
Write-Host "Resolved vision model: $resolvedVisionModel" -ForegroundColor Green

# Step 4: Deploy Nuclio Functions
Write-Host "`n[Step 4/4] Deploying $($functionsToDeploy.Count) detector function(s)..." -ForegroundColor Yellow

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
    Write-Error "nuctl CLI was not found. Please install nuctl or ensure it is available in WSL Ubuntu."
    exit 1
}

$networkName = "cvat_cvat"
$checkNet = docker network ls --filter "name=cvat_cvat" -q
if (-not $checkNet) {
    $networkName = "cvat"
}
Write-Host "Attaching to Docker network: $networkName" -ForegroundColor Gray

# Ensure shared feedback directory
$hostFeedbackDir = Join-Path $RepoRoot ".tool-cvat"
$hostExamplesDir = Join-Path $hostFeedbackDir "examples"
if (-not (Test-Path $hostFeedbackDir)) { New-Item -ItemType Directory -Path $hostFeedbackDir -Force | Out-Null }
if (-not (Test-Path $hostExamplesDir)) { New-Item -ItemType Directory -Path $hostExamplesDir -Force | Out-Null }
$forwardHostFeedbackDir = $hostFeedbackDir -replace '\\', '/'

# Zero-drift sync of modules
Write-Host "Syncing canonical modules across serverless targets..." -ForegroundColor Gray
python (Join-Path $ScriptDir "sync_serverless_modules.py")

foreach ($fn in $functionsToDeploy) {
    $fnName = $fn.Name
    $fnDisplay = $fn.DisplayName
    $fnMode = $fn.Mode
    $nuclioDir = Join-Path $RepoRoot "serverless\$fnName\nuclio"
    $functionYaml = Join-Path $nuclioDir "function.yaml"

    Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Gray
    Write-Host "Deploying $fnDisplay ($fnName)..." -ForegroundColor Cyan
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

    # Resolve detector-specific model with fallback to VISION_MODEL
    $fnModel = $resolvedVisionModel
    $detectorEnvVar = ""
    if ($fnName -eq "ninerouter-rectangle-mask") {
        if ($RectangleMaskModel -and $RectangleMaskModel.Trim() -ne "") {
            $fnModel = $RectangleMaskModel.Trim()
        }
        $detectorEnvVar = "RECTANGLE_MASK_MODEL=$fnModel"
    } elseif ($fnName -eq "ninerouter-polygon-mask") {
        if ($PolygonMaskModel -and $PolygonMaskModel.Trim() -ne "") {
            $fnModel = $PolygonMaskModel.Trim()
        }
        $detectorEnvVar = "POLYGON_MASK_MODEL=$fnModel"
    } elseif ($fnName -eq "ninerouter-polyline") {
        if ($PolylineModel -and $PolylineModel.Trim() -ne "") {
            $fnModel = $PolylineModel.Trim()
        }
        $detectorEnvVar = "POLYLINE_MODEL=$fnModel"
    }
    Write-Host "Active model for $($fnName): $fnModel" -ForegroundColor Green

    # Validate specification
    $specValidator = Join-Path $ScriptDir "validate_function_spec.py"
    & python $specValidator --yaml-path $functionYaml
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Specification validation FAILED for $functionYaml. Aborting deployment."
        exit 1
    }

    if ($useWsl) {
        $driveLetter = $nuclioDir.Substring(0, 1).ToLower()
        $restOfPath = $nuclioDir.Substring(2) -replace '\\', '/'
        $wslNuclioDir = "/mnt/$driveLetter$restOfPath"
        $wslFunctionYaml = "$wslNuclioDir/function.yaml"

        $env:WSLENV_BACKUP = $env:WSLENV
        if ($NineRouterKey) {
            $env:NINEROUTER_KEY = $NineRouterKey
            $env:WSLENV = if ($env:WSLENV) { "$($env:WSLENV):NINEROUTER_KEY" } else { "NINEROUTER_KEY" }
        }

        $extraEnvBash = ""
        if ($CvatWebhookSecret) {
            $extraEnvBash += "--env `"CVAT_WEBHOOK_SECRET=$CvatWebhookSecret`" "
        }
        if ($detectorEnvVar) {
            $extraEnvBash += "--env `"$detectorEnvVar`" "
        }

        $bashScript = @"
set -e
extraArgs=()
if [ -n "`$NINEROUTER_KEY" ]; then
    extraArgs+=(--env "NINEROUTER_KEY=`$NINEROUTER_KEY")
fi
nuctl deploy $fnName \
    --project-name cvat \
    --path "$wslNuclioDir" \
    --file "$wslFunctionYaml" \
    --platform local \
    --platform-config '{"attributes": {"network": "$networkName"}}' \
    --env "NINEROUTER_URL=$NineRouterUrl" \
    --env "VISION_MODEL=$fnModel" \
    --env "DETECTION_MODE=$fnMode" \
    --env "NINEROUTER_TIMEOUT=$NineRouterTimeout" \
    --env "FEEDBACK_DATA_DIR=/opt/nuclio/feedback" \
    --env "FEEDBACK_DB_PATH=/opt/nuclio/feedback/feedback.sqlite3" \
    $extraEnvBash \
    "`${extraArgs[@]}"
"@
        $bashScriptUnix = $bashScript -replace "`r`n", "`n"
        $b64Script = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($bashScriptUnix))

        try {
            Write-Host "Invoking nuctl deploy $fnName in WSL..." -ForegroundColor Gray
            & wsl -d Ubuntu bash -c "echo '$b64Script' | base64 -d | bash"
        } finally {
            $env:NINEROUTER_KEY = $null
            $env:WSLENV = $env:WSLENV_BACKUP
            $env:WSLENV_BACKUP = $null
        }
    } else {
        $deployArgs = @(
            "deploy", $fnName,
            "--project-name", "cvat",
            "--path", $nuclioDir,
            "--file", $functionYaml,
            "--platform", "local",
            "--platform-config", "{`"attributes`": {`"network`": `"$networkName`"}}",
            "--env", "NINEROUTER_URL=$NineRouterUrl",
            "--env", "VISION_MODEL=$fnModel",
            "--env", "DETECTION_MODE=$fnMode",
            "--env", "NINEROUTER_TIMEOUT=$NineRouterTimeout",
            "--env", "FEEDBACK_DATA_DIR=/opt/nuclio/feedback",
            "--env", "FEEDBACK_DB_PATH=/opt/nuclio/feedback/feedback.sqlite3"
        )
        if ($detectorEnvVar) { $deployArgs += @("--env", $detectorEnvVar) }
        if ($NineRouterKey) { $deployArgs += @("--env", "NINEROUTER_KEY=$NineRouterKey") }
        if ($CvatWebhookSecret) { $deployArgs += @("--env", "CVAT_WEBHOOK_SECRET=$CvatWebhookSecret") }
        & nuctl @deployArgs
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Deployment of $fnName failed."
        exit 1
    }
    Write-Host "[SUCCESS] $fnDisplay ($fnName) deployed." -ForegroundColor Green
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

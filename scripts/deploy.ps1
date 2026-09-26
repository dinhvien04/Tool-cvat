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
    [ValidateSet("three", "rectangle-mask", "polygon-mask", "polyline", "rectangle-tracker", "three-with-tracker", "week2", "human-pose-17", "face-vf50", "buddha", "buddha-multilimbs", "active-all", "all")]
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
    [string]$RectangleTrackerModel = $env:RECTANGLE_TRACKER_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$Pose17Model = $env:POSE17_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$Pose17RefineModel = $env:POSE17_REFINE_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$Vf50Model = $env:VF50_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$Vf50RefineModel = $env:VF50_REFINE_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$BuddhaModel = $env:BUDDHA_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$BuddhaRefineModel = $env:BUDDHA_REFINE_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$BuddhaAllowModelFallback = $env:BUDDHA_ALLOW_MODEL_FALLBACK,

    [Parameter(Mandatory = $false)]
    [string]$ToolCvatBuildSha = $env:TOOL_CVAT_BUILD_SHA,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterKey = $env:NINEROUTER_KEY,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterTimeout = "45.0",

    [Parameter(Mandatory = $false)]
    [string]$RectangleTrackerTimeout = "12.0",

    [Parameter(Mandatory = $false)]
    [string]$CvatWebhookSecret = $env:CVAT_WEBHOOK_SECRET,

    [Parameter(Mandatory = $false)]
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

if (-not $VisionModel) { $VisionModel = "ag/gemini-3.8-flash-low" }
if (-not $RectangleMaskModel) { $RectangleMaskModel = "ag/gemini-3.8-flash-low" }
if (-not $PolygonMaskModel) { $PolygonMaskModel = "ag/gemini-3.8-flash-low" }
if (-not $PolylineModel) { $PolylineModel = "ag/gemini-3.8-flash-low" }
if (-not $RectangleTrackerModel) { $RectangleTrackerModel = "ag/gemini-3.8-flash-low" }
if (-not $Pose17Model) { $Pose17Model = $VisionModel }
if (-not $Vf50Model) { $Vf50Model = $VisionModel }

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router - Deploy Streamlined Detector Suite" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Gray

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

$resolvedBuildSha = $ToolCvatBuildSha
if (-not $resolvedBuildSha -or $resolvedBuildSha.Trim() -eq "" -or $resolvedBuildSha -eq "unknown") {
    try {
        $gitOut = (git -C $RepoRoot rev-parse HEAD 2>$null)
        if ($LASTEXITCODE -eq 0 -and $gitOut -and $gitOut.Trim() -ne "") {
            $resolvedBuildSha = $gitOut.Trim()
        } else {
            $resolvedBuildSha = "unknown"
        }
    } catch {
        $resolvedBuildSha = "unknown"
    }
}
Write-Host "Deployment build revision SHA: $resolvedBuildSha" -ForegroundColor Gray
if ($Strict -and (-not $resolvedBuildSha -or $resolvedBuildSha -eq "unknown")) {
    Write-Error "Strict mode is enabled, but TOOL_CVAT_BUILD_SHA could not be resolved from environment or git HEAD."
    exit 1
}

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
    "rectangle-tracker" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-rectangle-tracker"; DisplayName = "9Router Rectangle Tracker"; Mode = "rectangle_tracker" }
        )
    }
    "three-with-tracker" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-rectangle-mask"; DisplayName = "9Router Rectangle + Mask"; Mode = "rectangle_mask" },
            @{ Name = "ninerouter-polygon-mask"; DisplayName = "9Router Polygon + Mask"; Mode = "polygon_mask" },
            @{ Name = "ninerouter-polyline"; DisplayName = "9Router Polyline"; Mode = "polyline" },
            @{ Name = "ninerouter-rectangle-tracker"; DisplayName = "9Router Rectangle Tracker"; Mode = "rectangle_tracker" }
        )
    }
    "human-pose-17" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-human-pose-17"; DisplayName = "9Router Human Pose 17"; Mode = "human_pose_17" }
        )
    }
    "face-vf50" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-face-vf50"; DisplayName = "9Router Face Landmark VF-50"; Mode = "face_vf50" }
        )
    }
    "buddha" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-buddha-multilimbs"; DisplayName = "9Router Buddha Multi-Limb Pose"; Mode = "buddha_multilimbs" }
        )
    }
    "buddha-multilimbs" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-buddha-multilimbs"; DisplayName = "9Router Buddha Multi-Limb Pose"; Mode = "buddha_multilimbs" }
        )
    }
    "week2" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-human-pose-17"; DisplayName = "9Router Human Pose 17"; Mode = "human_pose_17" },
            @{ Name = "ninerouter-face-vf50"; DisplayName = "9Router Face Landmark VF-50"; Mode = "face_vf50" }
        )
    }
    "active-all" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-rectangle-mask"; DisplayName = "9Router Rectangle + Mask"; Mode = "rectangle_mask" },
            @{ Name = "ninerouter-polygon-mask"; DisplayName = "9Router Polygon + Mask"; Mode = "polygon_mask" },
            @{ Name = "ninerouter-polyline"; DisplayName = "9Router Polyline"; Mode = "polyline" },
            @{ Name = "ninerouter-human-pose-17"; DisplayName = "9Router Human Pose 17"; Mode = "human_pose_17" },
            @{ Name = "ninerouter-face-vf50"; DisplayName = "9Router Face Landmark VF-50"; Mode = "face_vf50" },
            @{ Name = "ninerouter-buddha-multilimbs"; DisplayName = "9Router Buddha Multi-Limb Pose"; Mode = "buddha_multilimbs" },
            @{ Name = "ninerouter-rectangle-tracker"; DisplayName = "9Router Rectangle Tracker"; Mode = "rectangle_tracker" }
        )
    }
    "all" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-rectangle-mask"; DisplayName = "9Router Rectangle + Mask"; Mode = "rectangle_mask" },
            @{ Name = "ninerouter-polygon-mask"; DisplayName = "9Router Polygon + Mask"; Mode = "polygon_mask" },
            @{ Name = "ninerouter-polyline"; DisplayName = "9Router Polyline"; Mode = "polyline" },
            @{ Name = "ninerouter-human-pose-17"; DisplayName = "9Router Human Pose 17"; Mode = "human_pose_17" },
            @{ Name = "ninerouter-face-vf50"; DisplayName = "9Router Face Landmark VF-50"; Mode = "face_vf50" },
            @{ Name = "ninerouter-buddha-multilimbs"; DisplayName = "9Router Buddha Multi-Limb Pose"; Mode = "buddha_multilimbs" },
            @{ Name = "ninerouter-rectangle-tracker"; DisplayName = "9Router Rectangle Tracker"; Mode = "rectangle_tracker" }
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

$winNuctlWorks = $false
if (Get-Command nuctl -ErrorAction SilentlyContinue) {
    try {
        & nuctl get function 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $winNuctlWorks = $true
        }
    } catch {}
}

if ($winNuctlWorks) {
    $nuctlCmd = "nuctl"
} else {
    $wslCheck = wsl -d Ubuntu which nuctl 2>$null
    if ($wslCheck) {
        $nuctlCmd = "wsl -d Ubuntu nuctl"
        $useWsl = $true
    } elseif (Get-Command nuctl -ErrorAction SilentlyContinue) {
        $nuctlCmd = "nuctl"
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
$syncTargetParam = switch ($Target) {
    "week2" { "week2" }
    "human-pose-17" { "human-pose-17" }
    "face-vf50" { "face-vf50" }
    "buddha" { "buddha" }
    "buddha-multilimbs" { "buddha-multilimbs" }
    "three" { "three" }
    "rectangle-mask" { "rectangle-mask" }
    "polygon-mask" { "polygon-mask" }
    "polyline" { "polyline" }
    default { "all" }
}
Write-Host "Syncing canonical modules across serverless targets (target: $syncTargetParam)..." -ForegroundColor Gray
python (Join-Path $ScriptDir "sync_serverless_modules.py") --target $syncTargetParam
if ($LASTEXITCODE -ne 0) {
    Write-Error "Module synchronization failed. Aborting deployment."
    exit 1
}
python (Join-Path $ScriptDir "sync_serverless_modules.py") --check --target $syncTargetParam
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pre-deployment drift check failed. Serverless copies are not aligned with root modules!"
    exit 1
}

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
    $fnTimeout = $NineRouterTimeout
    $detectorEnvVar = ""
    $extraPoseRefine = $null
    $extraVf50Refine = $null
    $extraBuddhaRefine = $null
    $extraBuddhaFallback = $null
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
    } elseif ($fnName -eq "ninerouter-rectangle-tracker") {
        if ($RectangleTrackerModel -and $RectangleTrackerModel.Trim() -ne "") {
            $fnModel = $RectangleTrackerModel.Trim()
        }
        $detectorEnvVar = "RECTANGLE_TRACKER_MODEL=$fnModel"
        $fnTimeout = $RectangleTrackerTimeout
    } elseif ($fnName -eq "ninerouter-human-pose-17") {
        if ($Pose17Model -and $Pose17Model.Trim() -ne "") {
            $fnModel = $Pose17Model.Trim()
        }
        $detectorEnvVar = "POSE17_MODEL=$fnModel"
        if ($Pose17RefineModel -and $Pose17RefineModel.Trim() -ne "") {
            $extraPoseRefine = $Pose17RefineModel.Trim()
        }
    } elseif ($fnName -eq "ninerouter-face-vf50") {
        if ($Vf50Model -and $Vf50Model.Trim() -ne "") {
            $fnModel = $Vf50Model.Trim()
        }
        $detectorEnvVar = "VF50_MODEL=$fnModel"
        if ($Vf50RefineModel -and $Vf50RefineModel.Trim() -ne "") {
            $extraVf50Refine = $Vf50RefineModel.Trim()
        }
    } elseif ($fnName -eq "ninerouter-buddha-multilimbs") {
        if ($BuddhaModel -and $BuddhaModel.Trim() -ne "") {
            $fnModel = $BuddhaModel.Trim()
        }
        $detectorEnvVar = "BUDDHA_MODEL=$fnModel"
        if ($BuddhaRefineModel -and $BuddhaRefineModel.Trim() -ne "") {
            $extraBuddhaRefine = $BuddhaRefineModel.Trim()
        }
        if ($BuddhaAllowModelFallback -and $BuddhaAllowModelFallback.Trim() -ne "") {
            $extraBuddhaFallback = $BuddhaAllowModelFallback.Trim()
        }
        $fnTimeout = "180.0"
    }
    Write-Host "Active model for $($fnName): $fnModel" -ForegroundColor Green
    Write-Host "Request timeout for $($fnName): $fnTimeout s" -ForegroundColor Gray

    # Ensure feedback volume hostPath dynamically matches actual repository location
    if (Test-Path $functionYaml) {
        $yamlRaw = Get-Content $functionYaml -Raw
        if ($yamlRaw -match "(hostPath:\s*[\r\n]+\s*path:\s*)['`"]?([^'`"\r\n]+)['`"]?") {
            $existingPath = $Matches[2].Trim()
            if ($existingPath -ne $forwardHostFeedbackDir) {
                Write-Host "Normalizing host feedback path in $functionYaml to '$forwardHostFeedbackDir'..." -ForegroundColor DarkGray
                $yamlRaw = $yamlRaw -replace "(hostPath:\s*[\r\n]+\s*path:\s*)['`"]?[^'`"\r\n]+['`"]?", "`$1'$forwardHostFeedbackDir'"
                [System.IO.File]::WriteAllText($functionYaml, $yamlRaw, (New-Object System.Text.UTF8Encoding($false)))
            }
        }
    }

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
        if ($CvatWebhookSecret) {
            $env:CVAT_WEBHOOK_SECRET = $CvatWebhookSecret
            $env:WSLENV = if ($env:WSLENV) { "$($env:WSLENV):CVAT_WEBHOOK_SECRET" } else { "CVAT_WEBHOOK_SECRET" }
        }

        $extraEnvBash = ""
        if ($detectorEnvVar) {
            $extraEnvBash += "--env `"$detectorEnvVar`" "
        }
        if ($extraPoseRefine) {
            $extraEnvBash += "--env `"POSE17_REFINE_MODEL=$extraPoseRefine`" "
        }
        if ($extraVf50Refine) {
            $extraEnvBash += "--env `"VF50_REFINE_MODEL=$extraVf50Refine`" "
        }
        if ($extraBuddhaRefine) {
            $extraEnvBash += "--env `"BUDDHA_REFINE_MODEL=$extraBuddhaRefine`" "
        }
        if ($extraBuddhaFallback) {
            $extraEnvBash += "--env `"BUDDHA_ALLOW_MODEL_FALLBACK=$extraBuddhaFallback`" "
        }
        if ($resolvedBuildSha -and $resolvedBuildSha -ne "unknown") {
            $extraEnvBash += "--env `"TOOL_CVAT_BUILD_SHA=$resolvedBuildSha`" "
        }

        $bashScript = @"
set -e
extraArgs=()
if [ -n "`$NINEROUTER_KEY" ]; then
    extraArgs+=(--env "NINEROUTER_KEY=`$NINEROUTER_KEY")
fi
if [ -n "`$CVAT_WEBHOOK_SECRET" ]; then
    extraArgs+=(--env "CVAT_WEBHOOK_SECRET=`$CVAT_WEBHOOK_SECRET")
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
    --env "NINEROUTER_TIMEOUT=$fnTimeout" \
    --env "FEEDBACK_DATA_DIR=/opt/nuclio/feedback" \
    --env "FEEDBACK_DB_PATH=/opt/nuclio/feedback/feedback.sqlite3" \
    $extraEnvBash \
    "`${extraArgs[@]}"
"@
        $bashScriptUnix = $bashScript -replace "`r`n", "`n"
        $tmpDeployScript = Join-Path $hostFeedbackDir "deploy_${fnName}.sh"
        $tmpDriveLetter = $tmpDeployScript.Substring(0, 1).ToLower()
        $tmpRestPath = $tmpDeployScript.Substring(2) -replace '\\', '/'
        $wslTmpScript = "/mnt/$tmpDriveLetter$tmpRestPath"

        # Write script file directly without BOM to avoid Win32 32,767 character command length limits and PowerShell pipeline BOM corruption
        [System.IO.File]::WriteAllText($tmpDeployScript, $bashScriptUnix, (New-Object System.Text.UTF8Encoding($false)))

        try {
            Write-Host "Invoking nuctl deploy $fnName in WSL..." -ForegroundColor Gray
            & wsl -d Ubuntu bash "$wslTmpScript"
        } finally {
            Remove-Item -Path $tmpDeployScript -Force -ErrorAction SilentlyContinue
            $env:NINEROUTER_KEY = $null
            $env:CVAT_WEBHOOK_SECRET = $null
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
            "--env", "NINEROUTER_TIMEOUT=$fnTimeout",
            "--env", "FEEDBACK_DATA_DIR=/opt/nuclio/feedback",
            "--env", "FEEDBACK_DB_PATH=/opt/nuclio/feedback/feedback.sqlite3"
        )
        if ($detectorEnvVar) { $deployArgs += @("--env", $detectorEnvVar) }
        if ($extraPoseRefine) { $deployArgs += @("--env", "POSE17_REFINE_MODEL=$extraPoseRefine") }
        if ($extraVf50Refine) { $deployArgs += @("--env", "VF50_REFINE_MODEL=$extraVf50Refine") }
        if ($extraBuddhaRefine) { $deployArgs += @("--env", "BUDDHA_REFINE_MODEL=$extraBuddhaRefine") }
        if ($extraBuddhaFallback) { $deployArgs += @("--env", "BUDDHA_ALLOW_MODEL_FALLBACK=$extraBuddhaFallback") }
        if ($resolvedBuildSha -and $resolvedBuildSha -ne "unknown") { $deployArgs += @("--env", "TOOL_CVAT_BUILD_SHA=$resolvedBuildSha") }
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

if ($Target -in @("week2", "human-pose-17", "face-vf50", "active-all", "all")) {
    Write-Host "`n[Post-deploy] Verifying Week-2 runtime status and spec fingerprints..." -ForegroundColor Yellow
    $statusArgs = @((Join-Path $ScriptDir "week2_runtime_status.py"))
    if ($Strict) {
        $statusArgs += "--strict"
    } else {
        $statusArgs += "--check"
    }
    python @statusArgs
    if ($LASTEXITCODE -ne 0) {
        if ($Strict) {
            Write-Error "Post-deployment strict verification failed. Spec drift or build SHA mismatch detected!"
            exit 1
        } else {
            Write-Warning "Post-deployment verification reported unready containers or spec drift. Run with -Strict for zero-drift enforcement."
        }
    }
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

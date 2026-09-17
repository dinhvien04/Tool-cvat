<#
.SYNOPSIS
    Phase 3B Deployment Script for CVAT x 9Router AI Tools (Full 31-Label Multi-Shape Detector).

.DESCRIPTION
    1. Executes Phase 3B preflight validation for selected target(s).
    2. Ensures CVAT serverless compose stack is active.
    3. For selected function (31, box, mask, box-mask, all):
       - Syncs shared core, app, and config modules to Nuclio build context.
       - Dynamically resolves active vision model from 9Router.
       - Deploys the lightweight function using nuctl.
       - Injects runtime settings and API keys via secure in-memory execution.
    4. Verifies deployment health and prints CVAT UI instructions.

    SAFETY:
    - NEVER deletes volumes or databases.
    - NEVER runs 'docker compose down -v'.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("31", "box", "mask", "box-mask", "all")]
    [string]$Target = "31",

    [Parameter(Mandatory = $false)]
    [string]$CvatRoot = $env:CVAT_ROOT,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterUrl = "http://host.docker.internal:20128",

    [Parameter(Mandatory = $false)]
    [string]$VisionModel = $env:VISION_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterKey = $env:NINEROUTER_KEY,

    [Parameter(Mandatory = $false)]
    [switch]$SkipPreflight = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 3B 31-Label Multi-Shape Deploy" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target: $Target" -ForegroundColor Gray

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Determine functions to deploy
$functionsToDeploy = @()
switch ($Target) {
    "31" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-vision-31"; DisplayName = "9Router Vision 31-Label (Multi-Shape)" }
        )
    }
    "box" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-vision"; DisplayName = "9Router Vision (Bounding Box)" }
        )
    }
    "mask" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-vision-mask"; DisplayName = "9Router Vision Mask (Instance Segmentation)" }
        )
    }
    "box-mask" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-vision-box-mask"; DisplayName = "9Router Vision Box+Mask (Unified)" }
        )
    }
    "all" {
        $functionsToDeploy = @(
            @{ Name = "ninerouter-vision"; DisplayName = "9Router Vision (Bounding Box)" },
            @{ Name = "ninerouter-vision-mask"; DisplayName = "9Router Vision Mask (Instance Segmentation)" },
            @{ Name = "ninerouter-vision-box-mask"; DisplayName = "9Router Vision Box+Mask (Unified)" },
            @{ Name = "ninerouter-vision-31"; DisplayName = "9Router Vision 31-Label (Multi-Shape)" }
        )
    }
}

# Step 1: Preflight check
if (-not $SkipPreflight) {
    Write-Host "`n[Step 1/5] Running Phase 3B pre-flight checks..." -ForegroundColor Yellow
    $preflightScript = Join-Path $ScriptDir "phase3b_preflight.ps1"
    $preflightHostUrl = if ($NineRouterUrl -match "host\.docker\.internal") { "http://127.0.0.1:20128" } else { $NineRouterUrl }
    $preflightArgs = @("-ExecutionPolicy", "Bypass", "-File", $preflightScript, "-Target", $Target)
    if ($CvatRoot) { $preflightArgs += @("-CvatRoot", $CvatRoot) }
    if ($preflightHostUrl) { $preflightArgs += @("-NineRouterUrl", $preflightHostUrl) }
    if ($VisionModel) { $preflightArgs += @("-VisionModel", $VisionModel) }
    if ($NineRouterKey) { $preflightArgs += @("-NineRouterKey", $NineRouterKey) }
    & powershell @preflightArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Pre-flight checks failed. Please resolve reported issues before deploying."
        exit 1
    }
} else {
    Write-Host "`n[Step 1/5] Pre-flight checks skipped by user." -ForegroundColor Gray
}

# Step 2: Locate CVAT root safely
Write-Host "`n[Step 2/5] Locating CVAT repository..." -ForegroundColor Yellow
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

# Step 3: Ensure CVAT serverless services are active
Write-Host "`n[Step 3/5] Checking CVAT serverless compose stack..." -ForegroundColor Yellow
$serverlessCompose = Join-Path $resolvedCvatRoot "components\serverless\docker-compose.serverless.yml"
$serverlessOverride = Join-Path $resolvedCvatRoot "docker-compose.serverless.override.yml"

Push-Location $resolvedCvatRoot
try {
    $composeFiles = @("-f", "docker-compose.yml", "-f", "components/serverless/docker-compose.serverless.yml")
    if (Test-Path $serverlessOverride) {
        $composeFiles += @("-f", "docker-compose.serverless.override.yml")
    }

    Write-Host "Ensuring CVAT and Nuclio dashboard containers are running..." -ForegroundColor Gray
    docker compose @composeFiles up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start CVAT serverless containers."
        exit 1
    }
    Write-Host "CVAT stack is active." -ForegroundColor Green
} finally {
    Pop-Location
}

# Step 4: Resolve dynamic vision model
Write-Host "`n[Step 4/5] Resolving active vision model from 9Router..." -ForegroundColor Yellow
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

Write-Host "Resolved active vision model: $resolvedVisionModel" -ForegroundColor Green

# Step 5: Deploy each selected Nuclio Function
Write-Host "`n[Step 5/5] Deploying Nuclio functions ($($functionsToDeploy.Count) target(s))..." -ForegroundColor Yellow

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

foreach ($fn in $functionsToDeploy) {
    $fnName = $fn.Name
    $fnDisplay = $fn.DisplayName
    $nuclioDir = Join-Path $RepoRoot "serverless\$fnName\nuclio"
    $functionYaml = Join-Path $nuclioDir "function.yaml"

    Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Gray
    Write-Host "Deploying $fnDisplay ($fnName)..." -ForegroundColor Cyan
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

    # Sync shared application modules to Nuclio build context
    Write-Host "Syncing app, core, and config to $nuclioDir..." -ForegroundColor Gray
    if (Test-Path "$nuclioDir\app") { Remove-Item -Recurse -Force "$nuclioDir\app" }
    if (Test-Path "$nuclioDir\core") { Remove-Item -Recurse -Force "$nuclioDir\core" }
    if (Test-Path "$nuclioDir\config") { Remove-Item -Recurse -Force "$nuclioDir\config" }

    Copy-Item -Path "$RepoRoot\app" -Destination "$nuclioDir\app" -Recurse -Force
    Copy-Item -Path "$RepoRoot\core" -Destination "$nuclioDir\core" -Recurse -Force
    Copy-Item -Path "$RepoRoot\config" -Destination "$nuclioDir\config" -Recurse -Force

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
    --env "VISION_MODEL=$resolvedVisionModel" \
    --env "NINEROUTER_TIMEOUT=120.0" \
    "`${extraArgs[@]}"
"@
        $bashScriptUnix = $bashScript -replace "`r`n", "`n"
        $b64Script = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($bashScriptUnix))

        try {
            Write-Host "Invoking nuctl deploy $fnName inside WSL (in-memory secure execution)..." -ForegroundColor Gray
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
            "--env", "NINEROUTER_URL=$NineRouterUrl",
            "--env", "VISION_MODEL=$resolvedVisionModel",
            "--env", "NINEROUTER_TIMEOUT=120.0",
            "--platform-config", "{`"attributes`": {`"network`": `"$networkName`"}}"
        )
        if ($NineRouterKey) {
            $deployArgs += @("--env", "NINEROUTER_KEY=$NineRouterKey")
        }
        & nuctl @deployArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Deployment of $fnName failed via native nuctl."
            exit 1
        }
    }

    Write-Host "Successfully deployed $fnName." -ForegroundColor Green

    # Inject runtime environment variables into container if needed
    $containerName = "nuclio-nuclio-$fnName"
    Write-Host "Verifying container: $containerName..." -ForegroundColor Gray
    Start-Sleep -Seconds 2
    $containerStatus = docker inspect $containerName --format '{{.State.Status}}' 2>$null
    if ($containerStatus -eq "running") {
        Write-Host "Container $containerName is running." -ForegroundColor Green
    } else {
        Write-Host "Container $containerName status: $containerStatus" -ForegroundColor Yellow
    }
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " Deployment Complete!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "The 31-label multi-shape detector is now registered in CVAT AI Tools." -ForegroundColor White
Write-Host "Active Model: $resolvedVisionModel" -ForegroundColor White
Write-Host "To test via smoke test: .\scripts\phase3b_smoke_test.ps1" -ForegroundColor Yellow
Write-Host "======================================================================`n" -ForegroundColor Cyan

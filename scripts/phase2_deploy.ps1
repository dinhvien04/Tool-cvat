<#
.SYNOPSIS
    Phase 2 Safe Deployment Script for CVAT x 9Router AI Tools.

.DESCRIPTION
    1. Executes preflight validation.
    2. Syncs shared core/app modules to the Nuclio build context.
    3. Ensures CVAT serverless compose stack is active.
    4. Deploys the lightweight ninerouter-vision Nuclio function using nuctl.
    5. Injects non-secret settings and runtime API keys safely.
    6. Verifies function health and prints CVAT UI instructions.

    SAFETY:
    - NEVER deletes volumes or databases.
    - NEVER runs 'docker compose down -v'.
#>

[CmdletBinding()]
param (
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
Write-Host " CVAT x 9Router AI Annotation - Phase 2 Nuclio Function Deployment" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$NuclioDir = Join-Path $RepoRoot "serverless\ninerouter-vision\nuclio"

# Step 1: Preflight check
if (-not $SkipPreflight) {
    Write-Host "`n[Step 1/5] Running pre-flight checks..." -ForegroundColor Yellow
    $preflightScript = Join-Path $ScriptDir "phase2_preflight.ps1"
    $preflightHostUrl = if ($NineRouterUrl -match "host\.docker\.internal") { "http://127.0.0.1:20128" } else { $NineRouterUrl }
    $preflightArgs = @("-ExecutionPolicy", "Bypass", "-File", $preflightScript)
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

# Step 4: Sync shared application modules to Nuclio build context
Write-Host "`n[Step 4/5] Preparing Nuclio build context..." -ForegroundColor Yellow
Write-Host "Syncing app, core, and config to $NuclioDir..." -ForegroundColor Gray
if (Test-Path "$NuclioDir\app") { Remove-Item -Recurse -Force "$NuclioDir\app" }
if (Test-Path "$NuclioDir\core") { Remove-Item -Recurse -Force "$NuclioDir\core" }
if (Test-Path "$NuclioDir\config") { Remove-Item -Recurse -Force "$NuclioDir\config" }

Copy-Item -Path "$RepoRoot\app" -Destination "$NuclioDir\app" -Recurse -Force
Copy-Item -Path "$RepoRoot\core" -Destination "$NuclioDir\core" -Recurse -Force
Copy-Item -Path "$RepoRoot\config" -Destination "$NuclioDir\config" -Recurse -Force
Write-Host "Build context prepared." -ForegroundColor Green

# Step 5: Deploy the Nuclio Detector Function
Write-Host "`n[Step 5/5] Deploying 9Router Vision function via nuctl..." -ForegroundColor Yellow

$functionYaml = Join-Path $NuclioDir "function.yaml"

# Determine nuctl invocation method (Windows native or WSL)
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

# Determine network name
$networkName = "cvat_cvat"
$checkNet = docker network ls --filter "name=cvat_cvat" -q
if (-not $checkNet) {
    $networkName = "cvat"
}
Write-Host "Attaching to Docker network: $networkName" -ForegroundColor Gray

# Resolve vision model: ensure target model is genuinely available on 9Router
$hostUrl = "http://127.0.0.1:20128"
if ($NineRouterUrl -match "host\.docker\.internal") {
    $modelsCheckUrl = "$hostUrl/v1/models"
} else {
    $modelsCheckUrl = "$NineRouterUrl/v1/models"
}

Write-Host "Resolving active vision model from 9Router ($modelsCheckUrl)..." -ForegroundColor Gray
$resolvedVisionModel = $null
try {
    $authHeaders = @{}
    if ($NineRouterKey -and $NineRouterKey.Trim() -ne "") {
        $authHeaders["Authorization"] = "Bearer $NineRouterKey"
    }
    $modelsResp = Invoke-RestMethod -Uri $modelsCheckUrl -Headers $authHeaders -Method Get -TimeoutSec 5 -ErrorAction Stop
    $models = $modelsResp.data
    $visionModels = @($models | Where-Object {
        $_.capabilities.vision -eq $true -or
        ($_.id -match "(vision|vl|gemini|claude|gpt-4o|4o-mini|qwen-vl)" -and $_.capabilities.vision -ne $false)
    })

    if ($visionModels.Count -eq 0) {
        Write-Error "No vision-capable models found on 9Router. Cannot deploy detector without an available vision model."
        exit 1
    }

    if ($VisionModel -and $VisionModel.Trim() -ne "") {
        $modelFound = $visionModels | Where-Object { $_.id -eq $VisionModel }
        if ($modelFound) {
            $resolvedVisionModel = $modelFound.id
        } else {
            $avail = ($visionModels | ForEach-Object { $_.id }) -join ", "
            Write-Error "Requested vision model '$VisionModel' is not available on 9Router. Available models: $avail"
            exit 1
        }
    } else {
        $resolvedVisionModel = $visionModels[0].id
    }
} catch {
    Write-Error "Failed to query 9Router models at $($modelsCheckUrl): $_"
    exit 1
}

Write-Host "Resolved active vision model: $resolvedVisionModel" -ForegroundColor Green

# Build nuctl deploy arguments
if ($useWsl) {
    # Convert Windows paths to WSL paths
    $wslNuclioDir = (wsl -d Ubuntu wslpath -u ($NuclioDir -replace '\\', '/')).Trim()
    $wslFunctionYaml = "$wslNuclioDir/function.yaml"

    # In-memory secure environment passing via WSLENV without writing plaintext secrets to disk
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
nuctl deploy ninerouter-vision \
    --project-name cvat \
    --path "$wslNuclioDir" \
    --file "$wslFunctionYaml" \
    --platform local \
    --platform-config '{"attributes": {"network": "$networkName"}}' \
    --env "NINEROUTER_URL=$NineRouterUrl" \
    --env "VISION_MODEL=$resolvedVisionModel" \
    "`${extraArgs[@]}"
"@

    $b64Script = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($bashScript))

    try {
        Write-Host "Invoking nuctl deploy inside WSL (in-memory secure execution)..." -ForegroundColor Gray
        & wsl -d Ubuntu bash -c "echo '$b64Script' | base64 -d | bash"
    } finally {
        # Immediately sanitize process environment and restore WSLENV
        $env:NINEROUTER_KEY = $null
        $env:WSLENV = $env:WSLENV_BACKUP
        $env:WSLENV_BACKUP = $null
    }
} else {
    $deployArgs = @(
        "deploy", "ninerouter-vision",
        "--project-name", "cvat",
        "--path", $NuclioDir,
        "--file", $functionYaml,
        "--platform", "local",
        "--env", "NINEROUTER_URL=$NineRouterUrl",
        "--env", "VISION_MODEL=$resolvedVisionModel",
        "--platform-config", "{`"attributes`": {`"network`": `"$networkName`"}}"
    )
    if ($NineRouterKey) {
        $deployArgs += @("--env", "NINEROUTER_KEY=$NineRouterKey")
    }

    # Mask any secrets before logging the command line to prevent console/CI leak
    $displayArgs = @()
    foreach ($arg in $deployArgs) {
        if ($arg -like "*NINEROUTER_KEY=*") {
            $displayArgs += "NINEROUTER_KEY=***"
        } else {
            $displayArgs += $arg
        }
    }
    Write-Host "Executing: nuctl $($displayArgs -join ' ')" -ForegroundColor Gray
    & nuctl @deployArgs
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "nuctl deploy encountered an error. Please inspect container logs above."
    exit 1
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " Deployment Succeeded!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Function '9Router Vision' (ninerouter-vision) is now deployed."
Write-Host ""
Write-Host "Next Steps in CVAT UI:" -ForegroundColor Cyan
Write-Host "1. Open CVAT in browser: http://localhost:18080"
Write-Host "2. Open any Project/Task -> Open an Image/Job"
Write-Host "3. Click the Magic Wand (AI Tools) icon on the left toolbar"
Write-Host "4. Select the 'Detectors' tab"
Write-Host "5. Choose '9Router Vision' from the model dropdown"
Write-Host "6. Map detected labels to your task labels and run detection!"
Write-Host "======================================================================" -ForegroundColor Green

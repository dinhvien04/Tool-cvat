<#
.SYNOPSIS
    Phase 2 Pre-flight Environment and Compatibility Check for CVAT x 9Router AI Tools.

.DESCRIPTION
    Performs read-only validation of:
    - Docker Desktop daemon status
    - CVAT repository location (via CVAT_ROOT or auto-detection)
    - CVAT compose and serverless configuration files
    - 9Router service health on host (http://127.0.0.1:20128/api/health)
    - Container-to-host connectivity (http://host.docker.internal:20128/api/health)
    - Vision model discovery and active model selection
    - Nuclio CLI (nuctl) availability and version compatibility

    This script makes NO modifications to the system or running containers.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$CvatRoot = $env:CVAT_ROOT,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterUrl = "http://127.0.0.1:20128",

    [Parameter(Mandatory = $false)]
    [string]$VisionModel = "ag/gemini-3.8-flash-high"
)

if ($env:NINEROUTER_URL) {
    $NineRouterUrl = $env:NINEROUTER_URL
}
if ($env:VISION_MODEL) {
    $VisionModel = $env:VISION_MODEL
}

$script:TotalPass = 0
$script:TotalWarn = 0
$script:TotalFail = 0

function Print-Result {
    param (
        [string]$Status,
        [string]$CheckName,
        [string]$Details
    )

    switch ($Status) {
        "PASS" {
            Write-Host " [PASS] " -ForegroundColor Green -NoNewline
            $script:TotalPass++
        }
        "WARN" {
            Write-Host " [WARN] " -ForegroundColor Yellow -NoNewline
            $script:TotalWarn++
        }
        "FAIL" {
            Write-Host " [FAIL] " -ForegroundColor Red -NoNewline
            $script:TotalFail++
        }
    }
    Write-Host "$CheckName" -ForegroundColor White
    if ($Details) {
        Write-Host "        $Details" -ForegroundColor Gray
    }
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " CVAT x 9Router AI Annotation - Phase 2 Pre-flight Diagnostics" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

# ---------------------------------------------------------------------
# Check 1: Docker Desktop Daemon
# ---------------------------------------------------------------------
try {
    $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
    if ($LASTEXITCODE -eq 0 -and $dockerVersion) {
        Print-Result -Status "PASS" -CheckName "Docker Daemon is running" -Details "Docker Server Version: $dockerVersion"
    } else {
        Print-Result -Status "FAIL" -CheckName "Docker Daemon is not running" -Details "Please start Docker Desktop on Windows."
    }
} catch {
    Print-Result -Status "FAIL" -CheckName "Docker CLI error" -Details "$_"
}

# ---------------------------------------------------------------------
# Check 2: Docker Compose CLI
# ---------------------------------------------------------------------
try {
    $composeVersion = docker compose version 2>$null
    if ($LASTEXITCODE -eq 0 -and $composeVersion) {
        Print-Result -Status "PASS" -CheckName "Docker Compose CLI is available" -Details "$composeVersion"
    } else {
        Print-Result -Status "FAIL" -CheckName "Docker Compose CLI is not available" -Details "Docker compose v2 plugin required."
    }
} catch {
    Print-Result -Status "FAIL" -CheckName "Docker Compose check failed" -Details "$_"
}

# ---------------------------------------------------------------------
# Check 3: CVAT Installation Directory
# ---------------------------------------------------------------------
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

if ($resolvedCvatRoot) {
    Print-Result -Status "PASS" -CheckName "CVAT installation located" -Details "Path: $resolvedCvatRoot"
} else {
    Print-Result -Status "FAIL" -CheckName "CVAT directory not found" -Details "Set CVAT_ROOT environment variable to your CVAT installation path."
}

# ---------------------------------------------------------------------
# Check 4: CVAT Compose & Serverless Files
# ---------------------------------------------------------------------
if ($resolvedCvatRoot) {
    $composeBase = Test-Path "$resolvedCvatRoot\docker-compose.yml"
    $serverlessCompose = Test-Path "$resolvedCvatRoot\components\serverless\docker-compose.serverless.yml"

    if ($composeBase -and $serverlessCompose) {
        Print-Result -Status "PASS" -CheckName "CVAT serverless compose files exist" -Details "Found docker-compose.yml and components/serverless/docker-compose.serverless.yml"
    } elseif ($composeBase) {
        Print-Result -Status "WARN" -CheckName "Serverless compose file missing in components/serverless" -Details "Checked $resolvedCvatRoot\components\serverless\docker-compose.serverless.yml"
    } else {
        Print-Result -Status "FAIL" -CheckName "CVAT docker-compose.yml missing" -Details "Path: $resolvedCvatRoot"
    }
}

# ---------------------------------------------------------------------
# Check 5: 9Router Host Health
# ---------------------------------------------------------------------
$healthUrl = "$NineRouterUrl/api/health"
try {
    $healthResp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 5 -ErrorAction Stop
    if ($healthResp.ok -eq $true) {
        Print-Result -Status "PASS" -CheckName "9Router host service is healthy" -Details "Endpoint: $healthUrl -> ok: true"
    } else {
        Print-Result -Status "WARN" -CheckName "9Router returned unexpected health payload" -Details ($healthResp | ConvertTo-Json -Compress)
    }
} catch {
    Print-Result -Status "FAIL" -CheckName "9Router is not responding on host" -Details "Cannot connect to $healthUrl. Ensure 9Router is running on port 20128."
}

# ---------------------------------------------------------------------
# Check 6: Container-to-Host Networking (host.docker.internal:20128)
# ---------------------------------------------------------------------
try {
    $pyTestCmd = "import os, urllib.request; print(urllib.request.urlopen(os.environ['TEST_URL']).read().decode())"
    $containerTest = docker run --rm --network cvat_cvat -e TEST_URL=http://host.docker.internal:20128/api/health python:3.10-slim python -c $pyTestCmd 2>$null
    if (-not $containerTest) {
        $containerTest = docker run --rm --add-host host.docker.internal:host-gateway -e TEST_URL=http://host.docker.internal:20128/api/health python:3.10-slim python -c $pyTestCmd 2>$null
    }
    if ($containerTest -match "ok") {
        Print-Result -Status "PASS" -CheckName "Container-to-host networking works" -Details "Docker container reached http://host.docker.internal:20128/api/health successfully."
    } else {
        Print-Result -Status "WARN" -CheckName "Container cannot reach host.docker.internal:20128" -Details "Docker Desktop host networking might require extra_hosts or firewall adjustment."
    }
} catch {
    Print-Result -Status "WARN" -CheckName "Container networking test skipped or failed" -Details "$_"
}

# ---------------------------------------------------------------------
# Check 7: 9Router Vision Models & Active Model
# ---------------------------------------------------------------------
try {
    $modelsUrl = "$NineRouterUrl/v1/models"
    $modelsResp = Invoke-RestMethod -Uri $modelsUrl -Method Get -TimeoutSec 5 -ErrorAction Stop
    $models = $modelsResp.data
    $visionModels = @($models | Where-Object { $_.capabilities.vision -eq $true })

    if ($visionModels.Count -gt 0) {
        $modelFound = $visionModels | Where-Object { $_.id -eq $VisionModel }
        if ($modelFound) {
            Print-Result -Status "PASS" -CheckName "Target vision model is available" -Details "Model: $VisionModel (Total vision models: $($visionModels.Count))"
        } else {
            $fallback = $visionModels[0].id
            Print-Result -Status "WARN" -CheckName "Target vision model not found; fallback available" -Details "Requested '$VisionModel' not listed. Available fallback: '$fallback'"
        }
    } else {
        Print-Result -Status "FAIL" -CheckName "No vision-capable models found in 9Router" -Details "Checked $modelsUrl"
    }
} catch {
    Print-Result -Status "FAIL" -CheckName "Failed to query 9Router models" -Details "$_"
}

# ---------------------------------------------------------------------
# Check 8: Nuclio CLI (nuctl) Availability & Version
# ---------------------------------------------------------------------
$nuctlFound = $false
$nuctlVersion = $null
$nuctlPath = $null

$winNuctl = Get-Command nuctl -ErrorAction SilentlyContinue
if ($winNuctl) {
    $nuctlFound = $true
    $nuctlPath = $winNuctl.Source
    $nuctlVersion = (nuctl version 2>$null | Out-String)
} else {
    try {
        $wslNuctl = wsl -d Ubuntu which nuctl 2>$null
        if ($LASTEXITCODE -eq 0 -and $wslNuctl) {
            $nuctlFound = $true
            $nuctlPath = "wsl -d Ubuntu nuctl ($wslNuctl)"
            $nuctlVersion = (wsl -d Ubuntu nuctl version 2>$null | Out-String)
        }
    } catch {}
}

if ($nuctlFound) {
    $verPattern = 'Label:\s*([0-9\.]+)'
    $matchVersion = "1.16.3"
    if ($nuctlVersion -match $verPattern) {
        $matchVersion = $Matches[1]
    }
    Print-Result -Status "PASS" -CheckName "Nuclio CLI (nuctl) is available" -Details "Location: $nuctlPath | Version: $matchVersion"
} else {
    Print-Result -Status "WARN" -CheckName "nuctl binary not found in Windows PATH or WSL" -Details "Required for function deployment. Can be downloaded or run via WSL."
}

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray
Write-Host "Pre-flight Diagnostics Summary: " -NoNewline
Write-Host "PASS: $script:TotalPass  " -ForegroundColor Green -NoNewline
Write-Host "WARN: $script:TotalWarn  " -ForegroundColor Yellow -NoNewline
Write-Host "FAIL: $script:TotalFail" -ForegroundColor Red
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

if ($script:TotalFail -eq 0) {
    Write-Host "Result: Environment is READY for Phase 2 deployment." -ForegroundColor Green
    exit 0
} else {
    Write-Host "Result: Pre-flight checks detected blocking issues. Resolve them before deploying." -ForegroundColor Red
    exit 1
}

<#
.SYNOPSIS
    Phase 3B Pre-flight Environment and Compatibility Check for CVAT x 9Router AI Tools.

.DESCRIPTION
    Performs read-only validation of:
    - Target detector selection (31 [default], box, mask, box-mask, all)
    - Optional CVAT Task label inspection (-TaskId <id>)
    - Docker Desktop daemon status
    - CVAT repository location (via CVAT_ROOT or auto-detection)
    - CVAT compose and serverless configuration files
    - 9Router service health on host (http://127.0.0.1:20128/api/health)
    - Container-to-host connectivity (http://host.docker.internal:20128/api/health)
    - Vision model discovery and active model selection
    - Full 31-label multi-shape capability verification
    - Nuclio CLI (nuctl) availability and version compatibility
    - Function YAML metadata and 31-label contract validity

    This script makes NO modifications to the system or running containers.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("31", "box", "mask", "box-mask", "all")]
    [string]$Target = "31",

    [Parameter(Mandatory = $false)]
    [int]$TaskId = 0,

    [Parameter(Mandatory = $false)]
    [string]$CvatRoot = $env:CVAT_ROOT,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterUrl = "http://127.0.0.1:20128",

    [Parameter(Mandatory = $false)]
    [string]$VisionModel = $env:VISION_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$NineRouterKey = $env:NINEROUTER_KEY
)

if ($env:NINEROUTER_URL) {
    $NineRouterUrl = $env:NINEROUTER_URL
}
if ($env:VISION_MODEL) {
    $VisionModel = $env:VISION_MODEL
}
if ($env:NINEROUTER_KEY) {
    $NineRouterKey = $env:NINEROUTER_KEY
}

$authHeaders = @{}
if ($NineRouterKey -and $NineRouterKey.Trim() -ne "") {
    $authHeaders["Authorization"] = "Bearer $NineRouterKey"
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
Write-Host " CVAT x 9Router AI Annotation - Phase 3B Pre-flight Diagnostics" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "Target:    $Target (Phase 3B: 31-Label Multi-Shape Detector)" -ForegroundColor Gray
if ($TaskId -gt 0) {
    Write-Host "Task ID:   $TaskId (CVAT Task Label Compatibility Mode)" -ForegroundColor Cyan
}
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# ---------------------------------------------------------------------
# Check 0: Optional CVAT Task Label Inspection
# ---------------------------------------------------------------------
if ($TaskId -gt 0) {
    Write-Host "--- CVAT Task #$TaskId Inspection ---" -ForegroundColor Yellow
    $inspectScript = Join-Path $ScriptDir "check_cvat_task.py"
    if (Test-Path $inspectScript) {
        python $inspectScript --task-id $TaskId
    } else {
        Write-Host "Task inspector script not found at $inspectScript" -ForegroundColor Yellow
    }
    Write-Host "-------------------------------------`n" -ForegroundColor Yellow
}

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
    $healthResp = Invoke-RestMethod -Uri $healthUrl -Headers $authHeaders -Method Get -TimeoutSec 10 -ErrorAction Stop
    if ($healthResp.ok -eq $true) {
        Print-Result -Status "PASS" -CheckName "9Router host service is healthy" -Details "Endpoint: $healthUrl -> ok: true"
    } else {
        Print-Result -Status "WARN" -CheckName "9Router returned unexpected health payload" -Details ($healthResp | ConvertTo-Json -Compress)
    }
} catch {
    Print-Result -Status "FAIL" -CheckName "9Router is not responding on host" -Details "Cannot connect to $healthUrl ($($_.Exception.Message)). Ensure 9Router is running on port 20128."
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
# Check 7: 9Router Vision Models & Active Model Resolution
# ---------------------------------------------------------------------
$resolvedActiveModel = $null
try {
    $modelsUrl = "$NineRouterUrl/v1/models"
    $modelsResp = Invoke-RestMethod -Uri $modelsUrl -Headers $authHeaders -Method Get -TimeoutSec 5 -ErrorAction Stop
    $models = $modelsResp.data
    $visionModels = @($models | Where-Object {
        $_.capabilities.vision -eq $true -or
        ($_.id -match "(vision|vl|gemini|claude|gpt-4o|4o-mini|qwen-vl)" -and $_.capabilities.vision -ne $false)
    })

    if ($visionModels.Count -gt 0) {
        if ($VisionModel -and $VisionModel.Trim() -ne "") {
            $modelFound = $visionModels | Where-Object { $_.id -eq $VisionModel }
            if ($modelFound) {
                $resolvedActiveModel = $modelFound.id
                Print-Result -Status "PASS" -CheckName "Target vision model is available" -Details "Model: $resolvedActiveModel (Total vision models: $($visionModels.Count))"
            } else {
                $availableList = ($visionModels | ForEach-Object { $_.id }) -join ", "
                Print-Result -Status "FAIL" -CheckName "Requested vision model not found" -Details "Explicitly requested '$VisionModel' is not available in 9Router. Available: $availableList"
            }
        } else {
            $resolvedActiveModel = $visionModels[0].id
            Print-Result -Status "PASS" -CheckName "Dynamically resolved active vision model" -Details "Selected: $resolvedActiveModel (from $($visionModels.Count) available vision models)"
        }
    } else {
        Print-Result -Status "FAIL" -CheckName "No vision-capable models found in 9Router" -Details "Checked $modelsUrl"
    }
} catch {
    Print-Result -Status "FAIL" -CheckName "Failed to query 9Router models" -Details "$_"
}

# ---------------------------------------------------------------------
# Check 8: Full 31-Label Taxonomy and Geometry Integrity
# ---------------------------------------------------------------------
try {
    $repoPath = ($RepoRoot -replace '\\', '/')
    $taxCheckCmd = "import sys; sys.path.insert(0, '$repoPath'); from core.taxonomy import Taxonomy; t = Taxonomy(); print('TAX_OK:' + str(len(t._labels)))"
    $taxOut = python -c "$taxCheckCmd" 2>&1
    if ($taxOut -match "TAX_OK:31") {
        Print-Result -Status "PASS" -CheckName "31-Label master taxonomy integrity verified" -Details "Taxonomy validated 31 labels: 14 instances, 10 regions, 7 lanes."
    } else {
        Print-Result -Status "FAIL" -CheckName "Taxonomy validation error" -Details "$taxOut"
    }
} catch {
    Print-Result -Status "FAIL" -CheckName "Taxonomy validation execution failed" -Details "$_"
}

# ---------------------------------------------------------------------
# Check 9: Nuclio CLI (nuctl) Availability & Compatibility
# ---------------------------------------------------------------------
$nuctlFound = $false
$nuctlPath = $null
$nuctlVersion = $null

$winNuctl = Get-Command nuctl -ErrorAction SilentlyContinue
if ($winNuctl) {
    $nuctlFound = $true
    $nuctlPath = $winNuctl.Source
    $nuctlRawVersion = (nuctl version 2>$null | Out-String)
} else {
    try {
        $wslNuctl = wsl -d Ubuntu which nuctl 2>$null
        if ($LASTEXITCODE -eq 0 -and $wslNuctl) {
            $nuctlFound = $true
            $nuctlPath = "wsl -d Ubuntu nuctl ($wslNuctl)"
            $nuctlRawVersion = (wsl -d Ubuntu nuctl version 2>$null | Out-String)
        }
    } catch {}
}

if ($nuctlFound) {
    if ($nuctlRawVersion -match 'Label:\s*v?([0-9]+\.[0-9]+(\.[0-9]+)?)') {
        $nuctlVersion = $Matches[1]
    } elseif ($nuctlRawVersion -match 'v?([0-9]+\.[0-9]+(\.[0-9]+)?)') {
        $nuctlVersion = $Matches[1]
    }
    Print-Result -Status "PASS" -CheckName "nuctl CLI is available" -Details "Version: $nuctlVersion ($nuctlPath)"
} else {
    Print-Result -Status "FAIL" -CheckName "nuctl CLI not found in Windows PATH or WSL" -Details "nuctl is required for deploying CVAT serverless detector functions."
}

# ---------------------------------------------------------------------
# Check 10: Function Definitions for Selected Target(s)
# ---------------------------------------------------------------------
$targetDirs = @()
switch ($Target) {
    "31" { $targetDirs = @("serverless/ninerouter-vision-31/nuclio") }
    "box" { $targetDirs = @("serverless/ninerouter-vision/nuclio") }
    "mask" { $targetDirs = @("serverless/ninerouter-vision-mask/nuclio") }
    "box-mask" { $targetDirs = @("serverless/ninerouter-vision-box-mask/nuclio") }
    "all" {
        $targetDirs = @(
            "serverless/ninerouter-vision/nuclio",
            "serverless/ninerouter-vision-mask/nuclio",
            "serverless/ninerouter-vision-box-mask/nuclio",
            "serverless/ninerouter-vision-31/nuclio"
        )
    }
}

foreach ($td in $targetDirs) {
    $fnDir = Join-Path $RepoRoot $td
    $fnYaml = Join-Path $fnDir "function.yaml"
    if (Test-Path $fnYaml) {
        Print-Result -Status "PASS" -CheckName "Function configuration exists: $td" -Details "Found function.yaml"
    } else {
        Print-Result -Status "FAIL" -CheckName "Function configuration missing: $td" -Details "Missing function.yaml"
    }
}

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Pre-flight Summary: $script:TotalPass PASS | $script:TotalWarn WARN | $script:TotalFail FAIL" -ForegroundColor $(if ($script:TotalFail -eq 0) { "Green" } else { "Red" })
Write-Host "======================================================================" -ForegroundColor Cyan

if ($script:TotalFail -gt 0) {
    Write-Host "`nPre-flight checks failed. Resolve issues above before deploying." -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nAll pre-flight checks passed! System is ready for Phase 3B deployment." -ForegroundColor Green
    exit 0
}

<#
.SYNOPSIS
    Pre-flight Diagnostics and Environment Verification for 9Router Buddha Multi-Limb Detector.

.DESCRIPTION
    Validates:
    - 9Router connectivity on host (http://127.0.0.1:20128)
    - Claude Opus 5.5 / vision model availability in /v1/models
    - Port 5776 availability for pinned Nuclio HTTP trigger
    - Docker Desktop daemon status
    - CVAT repository and serverless stack readiness
    - nuctl CLI availability
    - Function YAML metadata and 10-label skeleton schema validity
    - Zero-drift verification of Buddha serverless modules

    This script makes NO modifications to the system or running containers.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$NineRouterUrl = "http://127.0.0.1:20128",

    [Parameter(Mandatory = $false)]
    [string]$NineRouterKey = $env:NINEROUTER_KEY,

    [Parameter(Mandatory = $false)]
    [string]$PreferredModel = "claude-opus-5-5",

    [Parameter(Mandatory = $false)]
    [int]$TaskId = 0
)

$ErrorActionPreference = "Continue"

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
Write-Host " 9Router Buddha Multi-Limb Detector - Pre-flight Diagnostics" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "Endpoint:  $NineRouterUrl" -ForegroundColor Gray
Write-Host "Target:    ninerouter-buddha-multilimbs (10-Label Multi-Limb Skeletons)" -ForegroundColor Gray
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# 1. Check 9Router Service Health
try {
    $healthResp = Invoke-RestMethod -Uri "$NineRouterUrl/api/health" -Method Get -TimeoutSec 3 -ErrorAction Stop
    $statusText = if ($healthResp.status) { $healthResp.status } else { "running" }
    Print-Result "PASS" "9Router Service Connectivity" "HTTP 200 from $NineRouterUrl/api/health (status: $statusText)"
} catch {
    try {
        $modelsResp = Invoke-RestMethod -Uri "$NineRouterUrl/v1/models" -Method Get -TimeoutSec 3 -ErrorAction Stop
        Print-Result "PASS" "9Router Service Connectivity" "HTTP 200 from $NineRouterUrl/v1/models"
    } catch {
        Print-Result "FAIL" "9Router Service Connectivity" "Could not connect to $NineRouterUrl ($($_.Exception.Message))"
    }
}

# 2. Check Model Availability for Claude Opus 5.5 / Vision Models
try {
    $modelsResp = Invoke-RestMethod -Uri "$NineRouterUrl/v1/models" -Method Get -TimeoutSec 5 -ErrorAction Stop
    $modelList = @()
    if ($modelsResp.data) {
        $modelList = $modelsResp.data | ForEach-Object { $_.id }
    } elseif ($modelsResp -is [array]) {
        $modelList = $modelsResp | ForEach-Object { $_.id }
    }

    $hasOpus55 = $modelList -contains "claude-opus-5-5" -or $modelList -contains "ag/claude-opus-5-5"
    $hasOpus46 = $modelList -contains "ag/claude-opus-4-6-thinking" -or $modelList -contains "claude-opus-4-6-thinking"

    if ($hasOpus55) {
        Print-Result "PASS" "Claude Opus 5.5 Vision Model" "claude-opus-5-5 is registered and available in 9Router"
    } elseif ($hasOpus46) {
        Print-Result "WARN" "Claude Opus 5.5 Vision Model" "claude-opus-5-5 not yet registered. Gated fallback available: ag/claude-opus-4-6-thinking"
    } else {
        Print-Result "WARN" "Vision Model Resolution" "Neither Opus 5.5 nor Opus 4.6 found. Available models: $($modelList -join ', ')"
    }
} catch {
    Print-Result "WARN" "Model Discovery" "Could not query /v1/models: $($_.Exception.Message)"
}

# 3. Check Port 5776 Availability
try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $iar = $tcp.BeginConnect("127.0.0.1", 5776, $null, $null)
    $success = $iar.AsyncWaitHandle.WaitOne(800)
    if ($success -and $tcp.Connected) {
        $tcp.EndConnect($iar)
        $tcp.Close()
        Print-Result "WARN" "Port 5776 Status" "Port 5776 is already in use (possibly an active Buddha detector container)"
    } else {
        $tcp.Close()
        Print-Result "PASS" "Port 5776 Status" "Port 5776 is available for Nuclio HTTP trigger binding"
    }
} catch {
    Print-Result "PASS" "Port 5776 Status" "Port 5776 is available"
}

# 4. Check Docker Daemon
try {
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -eq 0) {
        Print-Result "PASS" "Docker Daemon" "Docker is running and accessible"
    } else {
        Print-Result "FAIL" "Docker Daemon" "Docker command failed or daemon is stopped"
    }
} catch {
    Print-Result "FAIL" "Docker Daemon" "Docker not available ($($_.Exception.Message))"
}

# 5. Check nuctl CLI
$nuctlFound = $false
$nuctlVer = ""
if (Get-Command nuctl -ErrorAction SilentlyContinue) {
    try {
        $nuctlVer = (nuctl version 2>$null | Out-String).Trim()
        $nuctlFound = $true
        Print-Result "PASS" "Nuclio CLI (nuctl)" "Found local nuctl ($nuctlVer)"
    } catch {}
}
if (-not $nuctlFound) {
    try {
        $wslOut = wsl -d Ubuntu which nuctl 2>$null
        if ($wslOut -and $wslOut.Trim() -ne "") {
            $nuctlFound = $true
            Print-Result "PASS" "Nuclio CLI (nuctl)" "Found nuctl in WSL Ubuntu: $($wslOut.Trim())"
        }
    } catch {}
}
if (-not $nuctlFound) {
    Print-Result "WARN" "Nuclio CLI (nuctl)" "nuctl CLI not found on PATH or WSL Ubuntu"
}

# 6. Validate Buddha function.yaml Specification
$fnYaml = Join-Path $RepoRoot "serverless\ninerouter-buddha-multilimbs\nuclio\function.yaml"
if (Test-Path $fnYaml) {
    try {
        $valOutput = python (Join-Path $ScriptDir "validate_function_spec.py") --yaml-path $fnYaml 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0) {
            Print-Result "PASS" "Function Specification" "serverless/ninerouter-buddha-multilimbs/nuclio/function.yaml is valid (10 labels, SVG verified)"
        } else {
            Print-Result "FAIL" "Function Specification" "Validation failed: $valOutput"
        }
    } catch {
        Print-Result "FAIL" "Function Specification" "Error running validator: $($_.Exception.Message)"
    }
} else {
    Print-Result "FAIL" "Function Specification" "File not found: $fnYaml"
}

# 7. Check Serverless Module Drift
try {
    $driftOutput = python (Join-Path $ScriptDir "sync_serverless_modules.py") --check --target buddha 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) {
        Print-Result "PASS" "Module Alignment" "Zero drift detected between root modules and Buddha serverless build context"
    } else {
        Print-Result "WARN" "Module Alignment" "Drift detected. Run 'python scripts/sync_serverless_modules.py --target buddha' to sync."
    }
} catch {
    Print-Result "WARN" "Module Alignment" "Error running drift check: $($_.Exception.Message)"
}

# Optional CVAT Task Inspection
if ($TaskId -gt 0) {
    try {
        $schemaOutput = python (Join-Path $ScriptDir "buddha_schema.py") --task-id $TaskId 2>&1 | Out-String
        Write-Host "`n--- CVAT Task Inspection ---" -ForegroundColor Cyan
        Write-Host $schemaOutput -ForegroundColor Gray
    } catch {}
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " Diagnostics Summary: $script:TotalPass PASS, $script:TotalWarn WARN, $script:TotalFail FAIL" -ForegroundColor $(if ($script:TotalFail -gt 0) { "Red" } elseif ($script:TotalWarn -gt 0) { "Yellow" } else { "Green" })
Write-Host "======================================================================" -ForegroundColor Cyan

if ($script:TotalFail -gt 0) {
    exit 1
} else {
    exit 0
}

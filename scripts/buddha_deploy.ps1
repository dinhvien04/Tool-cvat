<#
.SYNOPSIS
    Deploy 9Router Buddha Multi-Limb CVAT AI Detector.

.DESCRIPTION
    Wraps scripts/deploy.ps1 with pre-configured parameters targeting 'buddha'.
    Deploys 'ninerouter-buddha-multilimbs' to CVAT Serverless with pinned HTTP port 5776.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$BuddhaModel = $env:BUDDHA_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$BuddhaRefineModel = $env:BUDDHA_REFINE_MODEL,

    [Parameter(Mandatory = $false)]
    [string]$BuddhaAllowModelFallback = "1",

    [Parameter(Mandatory = $false)]
    [string]$NineRouterUrl = "http://host.docker.internal:20128",

    [Parameter(Mandatory = $false)]
    [switch]$Strict
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$deployScript = Join-Path $ScriptDir "deploy.ps1"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Deploying 9Router Buddha Multi-Limb Detector to CVAT" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$params = @{
    Target = "buddha"
    NineRouterUrl = $NineRouterUrl
    BuddhaAllowModelFallback = $BuddhaAllowModelFallback
}

if ($BuddhaModel) { $params["BuddhaModel"] = $BuddhaModel }
if ($BuddhaRefineModel) { $params["BuddhaRefineModel"] = $BuddhaRefineModel }
if ($Strict) { $params["Strict"] = $true }

& $deployScript @params

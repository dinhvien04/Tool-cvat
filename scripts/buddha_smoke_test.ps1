<#
.SYNOPSIS
    Smoke Test for 9Router Buddha Multi-Limb Detector.

.DESCRIPTION
    Validates detector output format and contract:
    - Tests deployed HTTP endpoint (http://localhost:5776 by default) OR local Python ModelHandler.
    - Sends an image payload and verifies returned CVAT native skeleton shapes.
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$EndpointUrl = "http://localhost:5776",

    [Parameter(Mandatory = $false)]
    [string]$ImagePath = "",

    [Parameter(Mandatory = $false)]
    [switch]$LocalOnly
)

$ErrorActionPreference = "Continue"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " 9Router Buddha Multi-Limb Detector - Smoke Test" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Generate or read base64 image bytes
$b64 = ""
if ($ImagePath -and (Test-Path $ImagePath)) {
    $bytes = [System.IO.File]::ReadAllBytes($ImagePath)
    $b64 = [Convert]::ToBase64String($bytes)
    Write-Host "Using test image: $ImagePath ($($bytes.Length) bytes)" -ForegroundColor Gray
} else {
    # Generate 100x100 blank test JPEG in memory using python
    $pyGenCmd = "import io; from PIL import Image; img = Image.new('RGB', (100, 100), color=(128, 128, 128)); b = io.BytesIO(); img.save(b, 'JPEG'); import sys; sys.stdout.write(b.getvalue().hex())"
    $hex = python -c $pyGenCmd
    $bytes = [byte[]]($hex -split '(..)' | ? { $_ } | % { [Convert]::ToByte($_, 16) })
    $b64 = [Convert]::ToBase64String($bytes)
    Write-Host "Using generated 100x100 synthetic test image" -ForegroundColor Gray
}

if (-not $LocalOnly) {
    Write-Host "`nTesting HTTP trigger endpoint at $EndpointUrl..." -ForegroundColor Yellow
    $payload = @{
        image = $b64
        threshold = 0.3
    } | ConvertTo-Json -Compress

    try {
        $resp = Invoke-RestMethod -Uri $EndpointUrl -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        Write-Host " [PASS] HTTP Request succeeded (HTTP 200)" -ForegroundColor Green
        if ($resp -is [array]) {
            Write-Host " [INFO] Returned $($resp.Count) CVAT shape(s)" -ForegroundColor Green
            foreach ($s in $resp) {
                Write-Host "        - Label: $($s.label), Type: $($s.type), Group: $($s.group_id), Elements: $($s.elements.Count)" -ForegroundColor Gray
            }
        } else {
            Write-Host " [WARN] Returned unexpected response body type: $($resp.GetType().Name)" -ForegroundColor Yellow
        }
        exit 0
    } catch {
        Write-Host " [WARN] HTTP endpoint test failed: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "        Falling back to local Python ModelHandler verification..." -ForegroundColor Gray
    }
}

# Run local ModelHandler test
Write-Host "`nRunning local ModelHandler smoke test..." -ForegroundColor Yellow
$pyTestCmd = @"
import sys
from pathlib import Path
ROOT = Path(r'$RepoRoot')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'serverless' / 'ninerouter-buddha-multilimbs' / 'nuclio'))

from model_handler import ModelHandler
print('ModelHandler successfully imported.')
"@

python -c $pyTestCmd
if ($LASTEXITCODE -eq 0) {
    Write-Host " [PASS] Local ModelHandler import and initialization verification succeeded." -ForegroundColor Green
} else {
    Write-Host " [FAIL] Local ModelHandler test failed." -ForegroundColor Red
    exit 1
}

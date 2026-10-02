# Operations, Deployment & Testing Playbook

**Scope**: CLI Operations, Nuctl Deployment, Safe Teardown, Test Execution & Troubleshooting  
**Target Audience**: AI Agents executing tasks or developers operating the cluster  
**Last Updated**: October 2026

---

## 1. Environment & Prerequisites

### 1.1. Python Environment
- Python 3.11 or 3.12.
- Core dependencies: `pip install -r requirements.txt` (installs `pillow`, `requests`, `pyyaml`, `pytest`).
- Zero local heavy ML packages: Do NOT install `torch`, `torchvision`, `ultralytics`, `tensorflow`, or `onnxruntime`.

### 1.2. 9Router Remote Gateway Credentials
The pipeline requires access to the 9Router inference gateway. Pass the API key via environment variable:

```powershell
# Windows PowerShell
$env:NINEROUTER_KEY = "your-9router-api-key"
$env:NINEROUTER_URL = "https://9router-production-e47a.up.railway.app"
```

```bash
# Linux / WSL
export NINEROUTER_KEY="your-9router-api-key"
export NINEROUTER_URL="https://9router-production-e47a.up.railway.app"
```

*Security Invariant*: API keys are automatically masked (`sk-...xyz`) across all logs via `NineRouterClient.mask_api_key()`. When invoking deployments into WSL Ubuntu, keys are forwarded in-memory via `WSLENV=NINEROUTER_KEY/u` without writing plaintext secrets to disk.

### 1.3. CVAT & Serverless Requirements
- Running CVAT 2.75.1 instance (typically at `http://localhost:18080`).
- Running Nuclio dashboard (typically at `http://localhost:8070` or managed by CVAT Docker Compose).
- `nuctl` CLI installed in system `PATH` or available within WSL Ubuntu (`wsl -d Ubuntu nuctl`).

---

## 2. CLI Execution & Feedback Operations (`main.py`)

### 2.1. Standalone Inference
Run inference directly on local images without requiring CVAT or Docker:

```powershell
# Run default box and mask inference
python main.py --image path/to/image.jpg

# Run specific detection mode
python main.py --image path/to/image.jpg --mode box
python main.py --image path/to/image.jpg --mode polygon_mask
python main.py --image path/to/image.jpg --mode full_31

# Override model or gateway URL
python main.py --image test.jpg --model ag/gemini-3.8-flash-medium
```

Output artifacts land in `output/`:
- `output/result_bbox.jpg`: Visualized detections with color-coded bounding boxes.
- `output/predictions.json`: Normalized CVAT shapes ready for inspection.
- `output/run_report.json`: Metrics including token usage, latency, and feedback rules applied.

### 2.2. Human-in-the-Loop Feedback Commands
Inspect, synchronize, and evaluate annotator corrections:

```powershell
# Synchronize corrections from a completed CVAT Job
python main.py feedback-sync --job 24 --url http://localhost:18080

# View feedback statistics (diff counts, storage usage, active rules)
python main.py feedback-stats

# List derived text rules and associated labels
python main.py feedback-list --limit 20

# Run evaluation on an image with feedback rules injected
python main.py feedback-eval --image test.jpg

# Clear stored feedback safely
python main.py feedback-clear --rules   # Clear derived text rules only
python main.py feedback-clear --crops   # Clear cached image crops only
python main.py feedback-clear --all     # Full reset of feedback database
```

---

## 3. Serverless Deployment Playbook (`scripts/deploy.ps1`)

Deploy detector functions into CVAT Serverless with a single PowerShell script:

### 3.1. Common Deployment Commands

```powershell
# 1. Deploy the 5 active detectors individually
.\scripts\deploy.ps1 -Target box
.\scripts\deploy.ps1 -Target polygon
.\scripts\deploy.ps1 -Target mask
.\scripts\deploy.ps1 -Target box-3d
.\scripts\deploy.ps1 -Target human-pose-17

# 2. Deploy all active detectors at once
.\scripts\deploy.ps1 -Target active-all

# 3. Deploy legacy 3-detector suite
.\scripts\deploy.ps1 -Target three
```

### 3.2. Automated Pre-Flight Safety Checks Inside `deploy.ps1`
Before deploying any container, `deploy.ps1` executes automated safety checks:
1. **Locate CVAT Root**: Finds CVAT docker-compose configuration.
2. **Resolve Docker / WSL**: Detects Docker context and selects native Windows `nuctl` or `wsl -d Ubuntu nuctl`.
3. **Verify Build SHA**: Generates `TOOL_CVAT_BUILD_SHA` from git HEAD to track versioning.
4. **Module Drift Pre-Check**: Automatically runs `python scripts/sync_serverless_modules.py --check`. If root modules and serverless copies have drifted, deployment halts immediately.
5. **In-Memory Credential Forwarding**: Passes `$NineRouterKey` securely.

---

## 4. Safe Function Removal Playbook (`scripts/remove.ps1`)

To remove or update serverless detector functions, ALWAYS use `scripts/remove.ps1`:

```powershell
# Safely remove a specific detector function
.\scripts\remove.ps1 -Target box
.\scripts\remove.ps1 -Target human-pose-17

# Safely remove all 9Router functions
.\scripts\remove.ps1 -Target all-9router
```

### Safety Guarantees:
- **Third-Party Model Protection**: Explicitly forbids deleting models from other vendors (`pth-mmpose-hrnet32`, `openvino-*`, `pth-tue-mps-eomt-dinov3-*`).
- **Volume & Database Protection**: **NEVER** runs `docker compose down -v`. Databases and annotator tasks are completely untouched.

---

## 5. Source-of-Truth Drift Management

Because serverless containers require self-contained directories for Docker builds, canonical root files (`app/`, `core/`, `config/`) are mirrored to `serverless/*/nuclio/`.

### 5.1. How to Synchronize Modules
Whenever any file in `app/`, `core/`, or `config/` is modified, run the synchronizer:

```powershell
# Synchronize all serverless targets (115 files)
python scripts/sync_serverless_modules.py

# Verify zero drift (returns exit code 0 if clean, 1 if drift exists)
python scripts/sync_serverless_modules.py --check
```

### 5.2. Automated Drift Verification Test
```powershell
pytest tests/test_source_truth_drift.py -v
```
This test confirms:
- Identical SHA-256 byte parity between root files and serverless targets.
- Target manifest scoping (ensuring `week2` files only go to `human-pose-17`, etc.).
- Clean import smoke tests in isolated contexts.

---

## 6. Testing Guide & Verification Workflows

### 6.1. Running the Core Test Suites

```powershell
# 1. Source drift verification (MUST PASS FIRST)
python -m pytest tests/test_source_truth_drift.py -v

# 2. CVAT Data Safety Harness (Read-only protection for jobs 12, 13, 15)
python -m pytest tests/test_cvat_data_safety_harness.py -v

# 3. Active Detector Contract Tests
python -m pytest tests/test_box_detector.py -v
python -m pytest tests/test_box3d_detector.py -v
python -m pytest tests/test_pose17_model_handler.py -v

# 4. Geometry and Policy Partition Tests
python -m pytest tests/test_three_policies_strict.py -v
python -m pytest tests/test_geometry.py -v
python -m pytest tests/test_line_geometry.py -v

# 5. Feedback Loop & SQLite WAL Engine Tests
python -m pytest tests/test_feedback_learning.py -v
```

### 6.2. Smoke Testing Live Deployed Functions
After deploying via `deploy.ps1`, verify that CVAT detectors respond properly:

```powershell
# Run smoke test script against active endpoints
.\scripts\smoke_test.ps1

# Run live inference check against Nuclio HTTP RPC
python scripts/test_live_inference_nuclio.py --function ninerouter-box
```

---

## 7. Troubleshooting & Common Pitfalls

| Symptom / Error | Root Cause | Remediation Step |
|---|---|---|
| `test_source_truth_drift.py` fails with `content_mismatch` | An edit was made to a root file in `core/` or `app/` without updating `serverless/*/nuclio/` | Run `python scripts/sync_serverless_modules.py` and commit changes. |
| `DataSafetyViolationError: Protected job ID 12` | A script or test attempted to write annotations to protected production tasks/jobs | Point the test to a disposable task ID or use `DisposableTaskHarness`. Never bypass safety guards. |
| CVAT returns `400 Bad Request` when applying detections | Shape type mismatch or unauthorized label/attribute sent | Verify detector `function.yaml` spec matches exact output shape (e.g. `rectangle` vs `mask`). Check `docs/agent/03_DATA_CONTRACTS_AND_SCHEMAS.md`. |
| Human Pose 17 port conflict | Port `5997` already bound by another container | Check running containers: `docker ps --filter "publish=5997"`. Stop stale containers before re-deploying. |
| SQLite `database is locked` error | Concurrent writes without WAL mode | Verify SQLite configuration uses `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=10000` (enforced in `app/feedback.py`). |
| Vision request timeout (`408` / `ReadTimeout`) | Complex scene or network latency | Increase `NINEROUTER_TIMEOUT` in `function.yaml` (default: 45.0s for small models, 90.0s for box detector). |

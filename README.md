# CVAT × 9Router AI Annotation & Serverless Detectors

Production-grade integration connecting local **9Router** vision models (Gemini 3.8/3.7 Flash, Claude Sonnet, etc.) with **CVAT Community** (under **AI Tools -> Detectors**).

Supports:
- **Phase 1**: Local 2D Bounding Box AI annotation pipeline and CLI orchestration.
- **Phase 2**: Lightweight Nuclio detector integration (`ninerouter-vision`) for CVAT AI Tools.
- **Phase 3**: Unified **Bounding Box + Instance Mask / Segmentation** with zero local heavy ML dependencies, dual-mode and tri-detector deployment (`ninerouter-vision`, `ninerouter-vision-mask`, `ninerouter-vision-box-mask`).

---

## ⚡ Zero Local Heavy ML Directive (Strictly Enforced)

This project strictly adheres to a zero-heavy-ML architecture:
- **NO local model weights**: No multi-GB PyTorch/Torchvision, CUDA, TensorFlow, Ultralytics, SAM, SAM2, or ONNX runtimes.
- **Remote inference**: All perception runs through remote vision models orchestrated by local 9Router (`http://127.0.0.1:20128` or `http://host.docker.internal:20128`).
- **Pure Pillow geometry**: Polygon contour parsing, denormalization, rasterization to binary masks, and CVAT 1D flat list encoding run entirely via Python standard library and `Pillow` (`PIL.Image`, `PIL.ImageDraw`).
- **Featherweight containers**: Nuclio functions build in seconds from `python:3.11-slim` (< 200MB base), saving workstation RAM/VRAM.

---

## 🏗️ Architecture & Component Layout

```
Tool-cvat/
├── app/                              # Core application logic
│   ├── client.py                     # 9Router OpenAI-compatible client (streaming disabled, auth)
│   ├── config.py                     # Environment and label configuration loaders
│   ├── image_ops.py                  # Pillow loading, resizing, bbox and mask alpha overlays
│   ├── parser.py                     # Schema validation, coordinate denormalization, CVAT mask encoding
│   ├── pipeline.py                   # Local orchestration pipeline
│   └── service.py                    # Unified in-memory annotation service (modes: box, mask, box_and_mask)
├── core/
│   ├── geometry.py                   # Pure Pillow vector-to-raster & CVAT 1D flat list mask engine
│   └── vision_contract.py            # Prompt engineering, schemas, label definitions, coordinate rules
├── config/
│   ├── labels.yaml                   # 31 master CVAT labels
│   └── cvat_labels.json              # 13 rectangular bounding box candidate labels
├── docs/
│   ├── PHASE2_CVAT_AI_TOOLS.md       # Phase 2 CVAT AI Tools documentation
│   └── PHASE3_BOX_MASK.md            # Phase 3 Box + Instance Mask comprehensive guide
├── scripts/                          # Automated deployment & diagnostic scripts
│   ├── phase2_preflight.ps1          # Phase 2 pre-flight checks
│   ├── phase2_deploy.ps1             # Phase 2 safe deployment
│   ├── phase3_preflight.ps1          # Phase 3 comprehensive pre-flight validation
│   ├── phase3_deploy.ps1             # Phase 3 safe deployment (-Target mask|box|box-mask|both|all)
│   ├── phase3_smoke_test.ps1         # Phase 3 detector verification
│   └── phase3_remove.ps1             # Phase 3 safe container removal
├── serverless/                       # CVAT Nuclio detector functions
│   ├── ninerouter-vision/            # Box detector (type: rectangle)
│   ├── ninerouter-vision-mask/       # Mask detector (type: mask)
│   └── ninerouter-vision-box-mask/   # Unified Box + Mask detector (type: any)
├── tests/                            # 245 automated unit, integration, and security tests
├── main.py                           # CLI entry point
└── requirements.txt                  # Minimal lightweight dependencies (requests, Pillow, pyyaml)
```

---

## 🎯 Nuclio Detectors Overview

| Detector Name | Label Spec Type | Shape Types Returned | CVAT Feature |
|---|:---:|---|---|
| `ninerouter-vision` | `rectangle` | Bounding Box (`rectangle`) | Standard 2D bounding box detection |
| `ninerouter-vision-mask` | `mask` | Instance Mask (`mask`) | Native CVAT binary mask instance segmentation |
| `ninerouter-vision-box-mask` | `any` | **Both** `rectangle` and `mask` | Paired bounding box + mask with shared `group_id` & "Convert masks to polygons" |

---

## 🚀 Quick Start (Local CLI)

### 1. Installation
```bash
git clone https://github.com/dinhvien04/Tool-cvat.git
cd Tool-cvat
pip install -r requirements.txt
```

### 2. Discover Vision Models
Query your local 9Router instance to inspect available models:
```bash
python -m app.models
```

### 3. Run Inference Pipeline
```bash
# Unified Box + Mask (Default)
python main.py --image test.jpg --mode box_and_mask

# Pure Bounding Box
python main.py --image test.jpg --mode box

# Pure Instance Mask
python main.py --image test.jpg --mode mask
```

Output artifacts are generated in `output/`:
- `result_bbox.jpg`: Original image with bounding boxes and semi-transparent mask tints.
- `predictions.json`: Detailed detections with normalized/pixel coordinates and CVAT shapes.
- `raw_response.txt`: Verbatim vision model JSON output.
- `run_report.json`: Execution metadata, timing breakdown, and label distributions.

---

## 🐳 CVAT Serverless Deployment (PowerShell)

### Step 1: Pre-Flight Diagnostics
Ensure Docker, CVAT stack, and 9Router connectivity are healthy:
```powershell
.\scripts\phase3_preflight.ps1 -Target all
```

### Step 2: Deploy to CVAT
Deploy desired detectors into CVAT's Nuclio engine:
```powershell
# Deploy Unified Box+Mask detector
.\scripts\phase3_deploy.ps1 -Target box-mask

# Deploy Mask-only detector
.\scripts\phase3_deploy.ps1 -Target mask

# Deploy all 3 detectors
.\scripts\phase3_deploy.ps1 -Target all
```

### Step 3: Smoke Test Deployment
Send a test inference request to verify running containers:
```powershell
.\scripts\phase3_smoke_test.ps1 -Target box-mask
```

### Step 4: Use Inside CVAT
1. Open CVAT in browser: `http://localhost:18080`.
2. Open any Task or Job.
3. In the left toolbar, click **AI Tools** -> **Detectors**.
4. Select **9Router Vision Box+Mask** (or **9Router Vision Mask**).
5. Map task labels to model labels.
6. (Optional) Toggle **Convert masks to polygons** if vector contours are preferred.
7. Click **Annotate**. Bounding boxes and instance masks appear instantly on canvas!

---

## 🔒 Security, Safety & Privacy Hygiene

1. **Zero Database Destructive Actions**: Scripts never delete CVAT databases, tasks, jobs, or volumes (no `docker compose down -v`).
2. **Credential Hygiene**:
   - Secrets are masked across all logging (`sk-...xyz`).
   - WSL deployments transfer `$NineRouterKey` in-memory via `WSLENV` without writing plaintext secrets to disk.
3. **DoS & Resource Exhaustion Defense**:
   - 32MB maximum request body size guard in handlers and `function.yaml`.
   - Polygon vertex count limit (10,000 vertices/polygon) prevents CPU rasterization starvation.
   - Per-frame detection limit (500 objects/image) prevents memory exhaustion.
   - Decompression bomb protection (`Image.MAX_IMAGE_PIXELS = 89_478_485`).
4. **Honest Confidence Score Policy**: No false 1.0 confidence score fallbacks.

---

## 🧪 Testing

The repository maintains 100% test pass rate across 245 automated tests:

```bash
python -m pytest
```

Test breakdown:
- `tests/test_geometry.py`: 31 tests (Pillow rasterization, CVAT 1D flat list, Shoelace area, round-trip fidelity, coordinate ordering invariants).
- `tests/test_phase3_parser.py`: 8 tests (Box+mask parsing, strict schema validation, missing mask fallbacks).
- `tests/test_phase3_service.py`: 7 tests (modes: `box`, `mask`, `box_and_mask`, CVAT shape schemas, `group_id` instance pairing).
- `tests/test_phase3_nuclio_handlers.py`: 20 tests (Nuclio handlers, 32MB payload guards, error masking, YAML contracts).
- `tests/test_phase3_security_reliability.py`: 21 tests (DoS mitigation, memory limits, vertex boundaries, key sanitization).
- Baseline suites: 158 tests covering Phase 1 CLI and Phase 2 detectors.

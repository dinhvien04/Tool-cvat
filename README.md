# CVAT × 9Router AI Annotation & Serverless Detectors

Production-grade integration connecting local **9Router** vision models (Gemini 3.8/3.7 Flash, Claude Sonnet, etc.) with **CVAT Community** (under **AI Tools -> Detectors**).

Supports:
- **Phase 1**: Local 2D Bounding Box AI annotation pipeline and CLI orchestration.
- **Phase 2**: Lightweight Nuclio detector integration (`ninerouter-vision`) for CVAT AI Tools.
- **Phase 3**: Unified **Bounding Box + Instance Mask / Segmentation** with zero local heavy ML dependencies, dual-mode and tri-detector deployment (`ninerouter-vision`, `ninerouter-vision-mask`, `ninerouter-vision-box-mask`).
- **Phase 3B**: Full **31-Label Multi-Shape Annotation Assistant** tailored to autonomous driving & road-scene projects (BDD100K, Cityscapes, Mapillary). Intelligent 3-tier geometry routing:
  - **Instance Objects (14 labels)**: Paired `rectangle` + `mask` with shared `group_id`.
  - **Semantic Regions (10 labels)**: Native `mask` with polygon boundary `points` (enables CVAT "Convert masks to polygons"), bounding boxes suppressed, ungrouped (`group_id=None`).
  - **Lane Markings (7 labels)**: Zero-heavy-ML centerline extraction via spatial covariance PCA into `polyline` / `polygon` / `mask`, bounding boxes suppressed, ungrouped.

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
│   └── service.py                    # Unified annotation service (modes: box, mask, box_and_mask, full_31)
├── core/
│   ├── geometry.py                   # Pure Pillow vector-to-raster & CVAT 1D flat list mask engine
│   ├── line_geometry.py              # Zero-heavy-ML centerline extraction (PCA covariance, DP simplification)
│   ├── taxonomy.py                   # 31-label schema manager & task compatibility evaluator
│   └── vision_contract.py            # Prompt engineering, schemas, label definitions, coordinate rules
├── config/
│   ├── labels.yaml                   # 31 master CVAT labels
│   ├── label_geometry.yaml           # Shape routing rules (instances, regions, lanes)
│   ├── label_semantics.yaml          # Ambiguity policies & semantic descriptions
│   └── cvat_labels.json              # 13 rectangular bounding box candidate labels
├── docs/
│   ├── PHASE2_CVAT_AI_TOOLS.md       # Phase 2 CVAT AI Tools documentation
│   ├── PHASE3_BOX_MASK.md            # Phase 3 Box + Instance Mask comprehensive guide
│   └── PHASE3B_FULL_31_LABELS.md     # Phase 3B Full 31-Label Multi-Shape complete guide
├── scripts/                          # Automated deployment & diagnostic scripts
│   ├── check_cvat_task.py            # Direct CVAT task label inspection via Django ORM
│   ├── phase2_preflight.ps1          # Phase 2 pre-flight checks
│   ├── phase2_deploy.ps1             # Phase 2 safe deployment
│   ├── phase3_preflight.ps1          # Phase 3 comprehensive pre-flight validation
│   ├── phase3_deploy.ps1             # Phase 3 safe deployment (-Target mask|box|box-mask|both|all)
│   ├── phase3_smoke_test.ps1         # Phase 3 detector verification
│   ├── phase3_remove.ps1             # Phase 3 safe container removal
│   ├── phase3b_preflight.ps1         # Phase 3B 31-label pre-flight validation (-TaskId inspection)
│   ├── phase3b_deploy.ps1            # Phase 3B safe deployment with in-memory secret handling
│   ├── phase3b_smoke_test.ps1        # Phase 3B multi-shape verification on real road scene frames
│   └── phase3b_remove.ps1            # Phase 3B safe function removal (zero volume impact)
├── serverless/                       # CVAT Nuclio detector functions
│   ├── ninerouter-vision/            # Box detector (type: rectangle)
│   ├── ninerouter-vision-mask/       # Mask detector (type: mask)
│   ├── ninerouter-vision-box-mask/   # Unified Box + Mask detector (type: any)
│   └── ninerouter-vision-31/         # Full 31-Label Multi-Shape detector (type: any)
├── tests/                            # 393 automated unit, integration, and security tests
├── main.py                           # CLI entry point
└── requirements.txt                  # Minimal lightweight dependencies (requests, Pillow, pyyaml)
```

---

## 🎯 Nuclio Detectors Overview

| Detector Name | Label Spec Type | Shape Types Returned | Primary Use Case |
|---|:---:|---|---|
| `ninerouter-vision` | `rectangle` | Bounding Box (`rectangle`) | Standard 2D bounding box detection (13 labels) |
| `ninerouter-vision-mask` | `mask` | Instance Mask (`mask`) | Native CVAT binary mask instance segmentation (13 labels) |
| `ninerouter-vision-box-mask` | `any` | **Both** `rectangle` and `mask` | Paired bounding box + mask with shared `group_id` (13 labels) |
| `ninerouter-vision-31` | `any` | `rectangle`, `mask`, `polygon`, `polyline` | **Full 31-label autonomous driving multi-shape annotation** (paired instances, suppressed-bbox regions, thin lane polylines) |

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
# Phase 3B Preflight (supports optional -TaskId inspection)
.\scripts\phase3b_preflight.ps1 -Target 31 -TaskId 14

# Phase 3 Preflight (all detectors)
.\scripts\phase3_preflight.ps1 -Target all
```

### Step 2: Check CVAT Task Compatibility
Inspect any task's label schema directly via Django ORM inside `cvat_server`:
```powershell
python scripts\check_cvat_task.py --task-id 14
```

### Step 3: Deploy to CVAT
Deploy desired detectors into CVAT's Nuclio engine:
```powershell
# Deploy Full 31-Label Multi-Shape detector (Phase 3B)
.\scripts\phase3b_deploy.ps1 -Target 31

# Deploy Unified Box+Mask detector (Phase 3)
.\scripts\phase3_deploy.ps1 -Target box-mask

# Deploy Mask-only detector (Phase 3)
.\scripts\phase3_deploy.ps1 -Target mask

# Deploy all detectors
.\scripts\phase3b_deploy.ps1 -Target all
```

### Step 4: Smoke Test Deployment
Send a test inference request to verify running containers:
```powershell
# Test Phase 3B 31-Label detector on test image or real task frame
.\scripts\phase3b_smoke_test.ps1 -Target 31
.\scripts\phase3b_smoke_test.ps1 -Target 31 -ImagePath task14_frame0.jpg
```

### Step 5: Use Inside CVAT
1. Open CVAT in browser: `http://localhost:18080`.
2. Open any Task or Job (e.g. Task #14 `easy_semantic`).
3. In the left toolbar, click **AI Tools** -> **Detectors**.
4. Select **9Router Vision 31-Label Multi-Shape Detector** (or `ninerouter-vision-31`).
5. Matching task labels are automatically mapped thanks to `"type": "any"` in `function.yaml`.
6. (Recommended) Toggle **Convert masks to polygons** (`conv_mask_to_poly`) to edit segmented regions as vector polygons.
7. Click **Annotate**. Bounding boxes, instance masks, semantic regions, and lane polylines appear instantly on canvas!

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

The repository maintains 100% test pass rate across 393 automated tests:

```bash
python -m pytest
```

Test breakdown:
- `tests/test_phase3b_taxonomy.py`: 43 tests (31-label master schema, exact grouping, ambiguity policies, Django ORM task evaluator).
- `tests/test_line_geometry.py`: 22 tests (2D spatial covariance PCA aspect ratio, medial pair resampling, Douglas-Peucker simplification).
- `tests/test_phase3b_service.py`: 10 tests (3-tier shape routing, ROI coordinate translation, instance pairing, lane centerlines).
- `tests/test_phase3b_duplicate_suppression.py`: 10 tests (region rectangle suppression, lane bounding box suppression, IoU overlap deduplication).
- `tests/test_phase3b_nuclio_handler.py`: 11 tests (31-label function definition, 32MB payload guard, timeout handling, handler execution).
- `tests/test_phase3b_geometry.py`: 12 tests (vector conversion, trailing bbox verification, zero-heavy-ML constraints).
- `tests/test_phase3b_parser.py`: 5 tests (multi-shape response contracts, contour normalization).
- `tests/test_geometry.py`: 34 tests (Pillow rasterization, CVAT 1D flat list, Shoelace area, round-trip fidelity, coordinate ordering invariants).
- `tests/test_phase3_parser.py`: 8 tests (Box+mask parsing, strict schema validation, missing mask fallbacks).
- `tests/test_phase3_service.py`: 10 tests (modes: `box`, `mask`, `box_and_mask`, CVAT shape schemas, `group_id` instance pairing).
- `tests/test_phase3_nuclio_handlers.py`: 20 tests (Nuclio handlers, 32MB payload guards, error masking, YAML contracts).
- `tests/test_phase3_security_reliability.py`: 21 tests (DoS mitigation, memory limits, vertex boundaries, key sanitization).
- Baseline suites: 187 tests covering Phase 1 CLI, Phase 2 detectors, and vision contract invariants.
- `tests/test_phase3_service.py`: 7 tests (modes: `box`, `mask`, `box_and_mask`, CVAT shape schemas, `group_id` instance pairing).
- `tests/test_phase3_nuclio_handlers.py`: 20 tests (Nuclio handlers, 32MB payload guards, error masking, YAML contracts).
- `tests/test_phase3_security_reliability.py`: 21 tests (DoS mitigation, memory limits, vertex boundaries, key sanitization).
- Baseline suites: 158 tests covering Phase 1 CLI and Phase 2 detectors.

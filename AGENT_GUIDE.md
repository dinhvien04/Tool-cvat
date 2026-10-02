# AGENT_GUIDE — The AI Agent Handbook for Tool-cvat

> **FOR AI AGENTS**: Read this document first. It is the primary operating handbook for `Tool-cvat`. You do NOT need to read all source code files; follow the links and instructions in this suite to navigate, develop, debug, test, and deploy safely.

---

## 1. System Mission & Architecture in 60 Seconds

`Tool-cvat` bridges **CVAT Community Edition (v2.75.1)** with remote Multimodal Large Language Models (Gemini 3.8/3.7, Claude Sonnet/Opus) via **9Router**, a unified OpenAI-compatible vision gateway (`https://9router-production-e47a.up.railway.app`).

Instead of monolithic models, perception is decomposed into **5 specialized, lightweight serverless functions** running on Nuclio:
1. **`ninerouter-box`**: 14 foreground object classes (`type: rectangle`) with NMS overlap suppression.
2. **`ninerouter-polygon`**: 10 background semantic regions (`type: polygon`).
3. **`ninerouter-mask`**: 14 foreground object classes (`type: mask`) serialized as CVAT 1D Flat Lists.
4. **`ninerouter-box-3d`**: 8 vehicle classes (`type: cuboid`) generating 16-coordinate 3D cuboids.
5. **`ninerouter-human-pose-17`**: 17-keypoint COCO topology (`type: skeleton`) pinned to HTTP port `5997` with viewer-perspective laterality and two-pass adaptive crop refinement.

A built-in **Active Learning Feedback Loop** uses SQLite (`.tool-cvat/feedback.sqlite3` in WAL mode) to record baseline AI predictions, reconcile annotator corrections via a bipartite IoU diff engine, synthesize actionable text prompt rules, and inject visual few-shot crops into subsequent inference runs.

---

## 2. Non-Negotiable Core Invariants

Any agent touching this repository MUST observe these 5 core invariants:

1. **Zero Local Heavy ML Directive**:
   Strictly **NO** local PyTorch, TorchVision, CUDA, TensorFlow, Ultralytics, SAM/SAM-2, or ONNX Runtime. All geometry, contour extraction, Shoelace area calculation, and mask rasterization use pure Python standard libraries and `Pillow`.
2. **CVAT Data Safety Guard**:
   Tasks and Jobs **`12`, `13`, and `15`** are strictly **PROTECTED and READ-ONLY**. Never mutate or delete them. Never run destructive database commands or `docker compose down -v`.
3. **Zero-Drift Synchronization Rule**:
   The root directories (`app/`, `core/`, `config/`) are canonical. Serverless copies (`serverless/*/nuclio/`) must stay 100% byte-for-byte identical. Always run `python scripts/sync_serverless_modules.py` after editing root files.
4. **Visibility & Occlusion Flag Contract**:
   `0` = Outside frame (`outside=True, occluded=False`); `1` = Occluded (`outside=False, occluded=True`, renders dashed bone lines); `2` = Visible (`outside=False, occluded=False`). The state `outside=True AND occluded=True` is strictly forbidden.
5. **Viewer-Perspective Laterality**:
   All keypoints and landmarks are defined from the display screen viewpoint (annotator's perspective). Even IDs (2, 4, 6, ...) = Viewer Right ($x$ larger); Odd IDs (3, 5, 7, ...) = Viewer Left ($x$ smaller).

---

## 3. Quick Decision Matrix: "What Are You Trying to Do?"

| Task / Objective | Action to Take | Primary Reference Document |
|---|---|---|
| **Understand the pipeline flow from CVAT UI to 9Router** | Read the lifecycle and RPC trigger sequence | [`docs/agent/01_ARCHITECTURE_AND_PIPELINE.md`](docs/agent/01_ARCHITECTURE_AND_PIPELINE.md) |
| **Inspect detector specs, ports, models, or prompts** | Look up the 5 active detectors or historical suites | [`docs/agent/02_SERVERLESS_DETECTORS.md`](docs/agent/02_SERVERLESS_DETECTORS.md) |
| **Work with coordinates, labels, masks, cuboids, or skeletons** | Check mathematical formulas, schemas, and invariants | [`docs/agent/03_DATA_CONTRACTS_AND_SCHEMAS.md`](docs/agent/03_DATA_CONTRACTS_AND_SCHEMAS.md) |
| **Locate a specific class, method, or configuration** | Browse the comprehensive module catalog | [`docs/agent/04_CODEBASE_MODULE_MAP.md`](docs/agent/04_CODEBASE_MODULE_MAP.md) |
| **Run tests, deploy functions, or troubleshoot errors** | Follow the operations and deployment playbook | [`docs/agent/05_OPERATIONS_AND_TESTING.md`](docs/agent/05_OPERATIONS_AND_TESTING.md) |
| **Modify `core/` or `app/` files** | Make edits in root, then run `python scripts/sync_serverless_modules.py` | [`docs/agent/05_OPERATIONS_AND_TESTING.md#5`](docs/agent/05_OPERATIONS_AND_TESTING.md#5-source-of-truth-drift-management) |

---

## 4. Documentation Suite Index

The `docs/agent/` directory provides complete architectural documentation designed specifically for AI agents:

- **[`01_ARCHITECTURE_AND_PIPELINE.md`](docs/agent/01_ARCHITECTURE_AND_PIPELINE.md)**:
  * High-level system topology (Host $\leftrightarrow$ Nuclio $\leftrightarrow$ 9Router).
  * End-to-end HTTP RPC execution flow.
  * Human-in-the-loop active learning loop and storage governance.
- **[`02_SERVERLESS_DETECTORS.md`](docs/agent/02_SERVERLESS_DETECTORS.md)**:
  * Detailed specs for `ninerouter-box`, `polygon`, `mask`, `box-3d`, and `human-pose-17`.
  * Runtime environment variables, image sizes, and timeout configurations.
  * Historical detector context (Buddha multi-limb, VF50, 31-label).
- **[`03_DATA_CONTRACTS_AND_SCHEMAS.md`](docs/agent/03_DATA_CONTRACTS_AND_SCHEMAS.md)**:
  * Master 31-label taxonomy and 3-tier policy partition (Box/Mask, Polygon/Mask, Polyline).
  * Normalization formulas ($[0, 1000]$ integer to pixel space).
  * Serialization formats for Rectangles, Polygons, Flat List Masks, 3D Cuboids, and Skeletons.
  * Native occlusion contract and viewer-perspective laterality tables.
- **[`04_CODEBASE_MODULE_MAP.md`](docs/agent/04_CODEBASE_MODULE_MAP.md)**:
  * File-by-file catalog of `core/`, `app/`, `config/`, `scripts/`, and `serverless/`.
  * Class hierarchies, function signatures, and cross-module dependencies.
- **[`05_OPERATIONS_AND_TESTING.md`](docs/agent/05_OPERATIONS_AND_TESTING.md)**:
  * CLI commands for inference and feedback sync (`main.py`).
  * Deployment and teardown playbooks (`deploy.ps1`, `remove.ps1`).
  * Drift verification, test suite execution, and troubleshooting guide.

---

## 5. Essential Command Cheat Sheet

```powershell
# --- Drift Synchronization (Run after modifying core/ or app/) ---
python scripts/sync_serverless_modules.py
python scripts/sync_serverless_modules.py --check

# --- Essential Testing ---
python -m pytest tests/test_source_truth_drift.py -v
python -m pytest tests/test_cvat_data_safety_harness.py -v
python -m pytest tests/test_box_detector.py -v

# --- Standalone CLI Inference ---
python main.py --image test.jpg --mode box

# --- Serverless Deployment (PowerShell) ---
.\scripts\deploy.ps1 -Target box
.\scripts\deploy.ps1 -Target human-pose-17
.\scripts\deploy.ps1 -Target active-all

# --- Safe Function Teardown ---
.\scripts\remove.ps1 -Target box
.\scripts\remove.ps1 -Target all-9router
```

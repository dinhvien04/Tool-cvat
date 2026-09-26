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
- **Week 2**: Native **Skeleton & Keypoint Detectors** for in-cabin perception:
  - `ninerouter-human-pose-17`: 17-keypoint COCO human pose estimation with viewer-perspective laterality and 4-case occlusion crop refinement.
  - `ninerouter-face-vf50`: 50-landmark VinFast facial geometry partitioned across 7 component skeletons (`longmaytrai`, `longmayphai`, `songmui`, `mattrai`, `matphai`, `moingoai`, `moitrong`).
- **Buddha Multi-Limb**: Hierarchical multi-pass detector (`ninerouter-buddha-multilimbs`) for **complex multi-armed Buddhist iconography** (thousand-armed Avalokiteshvara/Guanyin). Outputs 10 native CVAT skeleton labels: single central body (`person`), 7 facial skeletons, dynamic 3-joint radial arms (`buddha_arm`) with deterministic clockwise angular ordering and vector cosine deduplication, and dynamic 21-joint mudra hands (`buddha_hand`) with bipartite 1:1 arm linking. Trigger pinned to HTTP port 5776.

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
│   ├── cvat_sync.py                  # CVAT REST API client, webhook HMAC verification & feedback sync
│   ├── feedback.py                   # Image hashing, SQLite correction storage & CorrectionDiffEngine
│   ├── image_ops.py                  # Pillow loading, resizing, bbox and mask alpha overlays
│   ├── parser.py                     # Schema validation, coordinate denormalization, CVAT mask encoding
│   ├── pipeline.py                   # Local orchestration pipeline
│   ├── retrieval.py                  # Dynamic few-shot selection & prompt extension builder
│   └── service.py                    # Unified annotation service (modes: box, mask, box_and_mask, full_31)
├── core/
│   ├── buddha_contract.py            # Buddha multi-limb coordinate reprojection, deduplication, ordering, CVAT skeletons
│   ├── geometry.py                   # Pure Pillow vector-to-raster & CVAT 1D flat list mask engine
│   ├── line_geometry.py              # Zero-heavy-ML centerline extraction (PCA covariance, DP simplification)
│   ├── pose_face_schema.py           # Week-2 Pose17 & VF-50 canonical schemas, laterality, quality gates
│   ├── taxonomy.py                   # 31-label schema manager & task compatibility evaluator
│   ├── vision_contract.py            # Prompt engineering, schemas, label definitions, coordinate rules
│   └── week2_schema.py               # Week-2 YAML schema loader, visibility normalization, SVG generation
├── config/
│   ├── buddha_multilimbs.yaml        # Authoritative schema for ninerouter-buddha-multilimbs detector
│   ├── cvat_labels.json              # 13 rectangular bounding box candidate labels
│   ├── label_geometry.yaml           # Shape routing rules (instances, regions, lanes)
│   ├── label_semantics.yaml          # Ambiguity policies & semantic descriptions
│   ├── labels.yaml                   # 31 master CVAT labels
│   ├── learning.yaml                 # Correction memory thresholds & storage quotas
│   ├── week2_pose17.yaml             # Week-2 Pose17 canonical configuration
│   └── week2_vf50.yaml               # Week-2 VF50 canonical configuration
├── docs/
│   ├── BUDDHA_MULTILIMB.md           # Buddha Multi-Limb Pose & Hand Landmark Annotation comprehensive guide
│   ├── PHASE2_CVAT_AI_TOOLS.md       # Phase 2 CVAT AI Tools documentation
│   ├── PHASE3_BOX_MASK.md            # Phase 3 Box + Instance Mask comprehensive guide
│   ├── PHASE3B_FULL_31_LABELS.md     # Phase 3B Full 31-Label Multi-Shape & Correction Memory guide
│   ├── WEEK2_OCCLUSION.md            # Week-2 Occlusion & Crop Refinement specification
│   └── WEEK2_SCHEMA_PROVENANCE.md    # Week-2 Pose17 & VF50 schema provenance & laterality contract
├── scripts/                          # Automated deployment & diagnostic scripts
│   ├── buddha_deploy.ps1             # Buddha multi-limb detector safe deployment (port 5776)
│   ├── buddha_preflight.ps1          # Buddha multi-limb pre-flight diagnostics & model availability
│   ├── buddha_schema.py              # Buddha 10-label schema inspector, summary table & JSON builder
│   ├── buddha_smoke_test.ps1         # Buddha smoke test (HTTP port 5776 or local ModelHandler)
│   ├── build_buddha_yaml.py          # Script to compile serverless function.yaml from buddha_multilimbs.yaml
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
│   ├── phase3b_remove.ps1            # Phase 3B safe function removal (zero volume impact)
│   ├── sync_serverless_modules.py    # Zero-drift synchronizer for root modules into serverless build contexts
│   └── validate_function_spec.py     # Cross-detector Nuclio function.yaml validator
├── serverless/                       # CVAT Nuclio detector functions
│   ├── ninerouter-buddha-multilimbs/ # 10-Label Buddha Multi-Limb Pose & Hand detector (port 5776)
│   ├── ninerouter-face-vf50/         # 50-Landmark VinFast 7-component face detector (port 5775)
│   ├── ninerouter-human-pose-17/     # 17-Keypoint COCO Pose detector with occlusion merging (port 5774)
│   ├── ninerouter-polyline/          # Pure polyline detector for lane markings
│   ├── ninerouter-polygon-mask/      # Pure polygon mask detector
│   ├── ninerouter-rectangle-mask/    # Paired rectangle + mask detector
│   ├── ninerouter-vision/            # Box detector (type: rectangle)
│   ├── ninerouter-vision-mask/       # Mask detector (type: mask)
│   ├── ninerouter-vision-box-mask/   # Unified Box + Mask detector (type: any)
│   └── ninerouter-vision-31/         # Full 31-Label Multi-Shape detector & Webhook sync (type: any)
├── tests/                            # 1,105 automated unit, integration, and security tests
├── main.py                           # CLI entry point (inference + feedback-*)
└── requirements.txt                  # Minimal lightweight dependencies (requests, Pillow, pyyaml)
```

---

## 🎯 Nuclio Detectors Overview

| Detector Name | Label Spec Type | Shape Types Returned | Primary Use Case | Pinned Port |
|---|:---:|---|---|:---:|
| `ninerouter-vision` | `rectangle` | Bounding Box (`rectangle`) | Standard 2D bounding box detection (13 labels) | Dynamic |
| `ninerouter-vision-mask` | `mask` | Instance Mask (`mask`) | Native CVAT binary mask instance segmentation (13 labels) | Dynamic |
| `ninerouter-vision-box-mask` | `any` | **Both** `rectangle` and `mask` | Paired bounding box + mask with shared `group_id` (13 labels) | Dynamic |
| `ninerouter-vision-31` | `any` | `rectangle`, `mask`, `polygon`, `polyline` | **Full 31-label autonomous driving multi-shape annotation** (paired instances, suppressed-bbox regions, thin lane polylines) | Dynamic |
| `ninerouter-human-pose-17` | `skeleton` | Native `skeleton` (17 keypoints) | Cabin driver/passenger pose estimation with viewer laterality & 4-case occlusion merging | `5774` |
| `ninerouter-face-vf50` | `skeleton` | Native `skeleton` (7 skeletons, 50 pts) | VinFast 7-component facial landmark topology (`longmaytrai`..`moitrong`) | `5775` |
| `ninerouter-buddha-multilimbs` | `skeleton` | Native `skeleton` (10 skeletons) | **Thousand-armed Avalokiteshvara & multi-limb sacred iconography** (body pose17, 7 face skeletons, dynamic radial arms, 21-joint mudra hands) | `5776` |

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

# Buddha Multi-Limb Preflight (validates port 5776, 10-label schema & Opus 5.5)
.\scripts\buddha_preflight.ps1 -TaskId 20

# Phase 3 Preflight (all detectors)
.\scripts\phase3_preflight.ps1 -Target all
```

### Step 2: Check CVAT Task Compatibility
Inspect any task's label schema directly via Django ORM inside `cvat_server`:
```powershell
python scripts\check_cvat_task.py --task-id 14

# Buddha 10-label schema inspector & summary
python scripts\buddha_schema.py --summary
```

### Step 3: Deploy to CVAT
Deploy desired detectors into CVAT's Nuclio engine:
```powershell
# Deploy Buddha Multi-Limb detector (Port 5776, 10 Skeletons)
.\scripts\buddha_deploy.ps1

# Deploy Human Pose 17 detector (Port 5774)
.\scripts\deploy.ps1 -Target pose17

# Deploy Face Landmark VF-50 detector (Port 5775)
.\scripts\deploy.ps1 -Target vf50

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
# Test Buddha Multi-Limb detector on port 5776
.\scripts\buddha_smoke_test.ps1

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

## 🧠 Human-in-the-Loop Correction Memory & Adaptive Prompting

The full 31-label detector (`ninerouter-vision-31`) is intrinsically **correction-aware**. When human annotators edit, correct, or delete AI-generated annotations in CVAT, the system reconciles the changes, records them into local memory, derives actionable rules, and injects dynamic multimodal visual few-shot guidance into future inference requests — **without local retraining or heavy ML dependencies**.

### How the Feedback Loop Works

```
  1. AI Baseline Inference
     [Image] ──> [ninerouter-vision-31] ──> [Shapes Generated]
                         │
                         └──> Saved to Persistent SQLite (/opt/nuclio/feedback/feedback.sqlite3)
                              Indexed by Canonical Visual Fingerprint ({w}x{h}_ + SHA256 of RGB buffer)

  2. Human Review & Editing in CVAT
     Annotator corrects mislabeled classes (car -> truck), adjusts contours,
     adds missed objects, or removes false positives.

  3. Synchronization (Automated Webhook or CLI)
     CVAT Webhook (`update:job`) ──> [Dual-Dispatch Nuclio Handler]
     OR manual CLI: python main.py feedback-sync --job <id>

  4. Geometric Bipartite Matching (CorrectionDiffEngine)
     Pairwise IoU matching classifies events into 9 precise categories:
     - RELABEL (cross-label IoU >= 0.60, e.g. car -> truck)
     - BOX_MOVE / BOX_RESIZE
     - MASK_EDIT / REGION_EDIT / LANE_EDIT
     - DELETE_FALSE_POSITIVE / ADD_MISSING / NO_CHANGE (IoU >= 0.92)

  5. Rule Derivation (SQL Aggregation)
     Patterns with count >= min_samples (default 3) are synthesized into rules:
     "When detecting high-profile heavy cargo vehicles, prefer truck over car."

  6. Real Multimodal Few-Shot Prompting (CorrectionRetrievalEngine)
     Future inferences retrieve active rules + 2-3 visual JPEG crops (<=512px)
     interleaved as Base64 Data URLs with expected structured JSON annotations,
     dynamically prepending them into the vision model message content.
```

### Shared Persistent Database & Docker Bind Mount

The feedback database and visual crops persist across container rebuilds and host reboots via a Docker host bind mount:
- **Host Location**: `.tool-cvat/feedback.sqlite3` and `.tool-cvat/examples/`
- **Container Mount**: `/opt/nuclio/feedback` bound to host `.tool-cvat` via `spec.volumes` in `function.yaml`:
  ```yaml
  spec:
    volumes:
      - volume:
          name: feedback-data
          hostPath:
            path: 'D:/tool-cvat/.tool-cvat'
        volumeMount:
          name: feedback-data
          mountPath: '/opt/nuclio/feedback'
  ```
- **Environment Variables**:
  - `FEEDBACK_DATA_DIR=/opt/nuclio/feedback`
  - `FEEDBACK_DB_PATH=/opt/nuclio/feedback/feedback.sqlite3`
  - WAL mode enabled (`PRAGMA journal_mode=WAL`) for concurrent host CLI and container access.

### CLI Feedback Management

Manage and inspect correction memory directly from the terminal:

```bash
# 1. Setup or inspect automated CVAT Webhooks
python main.py feedback-webhook-setup --url http://localhost:18080 --target-url http://nuclio-nuclio-ninerouter-vision-31:8080 --secret toolcvat_webhook_secret_2026
python main.py feedback-webhook-setup --url http://localhost:18080 --list

# 2. Synchronize annotations manually from a CVAT job
python main.py feedback-sync --job 15 [--url http://localhost:18080] [--token <token>]

# 3. View per-label statistics, active rules, and visual examples across all 31 labels
python main.py feedback-stats

# 4. Inspect recent individual correction records
python main.py feedback-list --limit 20 [--label car] [--type RELABEL]

# 5. Toggle correction memory globally
python main.py feedback-enable
python main.py feedback-disable

# 6. Clear stored records, crops, or derived rules
python main.py feedback-clear --crops   # Clears only image crops
python main.py feedback-clear --rules   # Clears only derived rules
python main.py feedback-clear --all     # Full reset of feedback database

# 7. Comparative evaluation (side-by-side baseline vs. adaptive prompting)
python main.py feedback-eval --image test.jpg
```

### Dual-Dispatch Nuclio Handler & Secure Webhooks

The serverless function (`ninerouter-vision-31`) handles both CVAT inference requests and CVAT webhooks within a single container:
- **Inference**: Receives `{"image": "<base64>", "threshold": 0.5}`, checks active rules & visual crops, saves baseline prediction with canonical fingerprint, and returns shapes with `visual_examples_used` and `text_rules_used` instrumentation.
- **Webhook**: Receives CVAT `update:job` or `update:task` events, verifies HMAC-SHA256 signatures against raw request body bytes via case-insensitive headers (`X-Signature-256`, `X-CVAT-Signature`), and automatically reconciles job annotations when a job is marked `completed`.
- Configurable environment variables: `CVAT_WEBHOOK_SECRET`, `CVAT_URL`, `CVAT_TOKEN`, `FEEDBACK_DB_PATH`, `FEEDBACK_DATA_DIR`.

---

## 🧘 Buddha Multi-Limb Pose & Hand Landmark Annotation (`ninerouter-buddha-multilimbs`)

Targeting complex Buddhist and Hindu sacred iconography (such as the thousand-armed Avalokiteshvara / Quán Thế Âm Bồ Tát Thiên Thủ Thiên Nhãn), this detector overcomes the architectural limits of conventional 17-point human pose models by combining a single central body with dynamic 3-joint radial arms and 21-joint mudra hands.

### Core Architecture & Multi-Pass Hierarchy
- **Pass 1: Global Discovery:** Identifies central torso anchor $(c_x, c_y)$, face ROI, body pose17 skeleton, and candidate arm/hand crops.
- **Pass 2: Arm Refine:** Extracts 15% padded crops around arm candidates to localize `root`, `elbow`, and `wrist` joints with high resolution.
- **Pass 3: Hand Refine:** Extracts 20% padded crops around wrists to resolve 21-joint mudra hand landmarks, executing concurrently with failure isolation.
- **Pass 4: Face VF50:** Extracts face crops to localize 50 landmarks across 7 VinFast facial component skeletons (`longmaytrai`..`moitrong`).
- **Pass 5: Body Pose17:** Refines central body pose coordinates, preventing radial arm interference.
- **Pass 6: Shape Merging & Validation:** Runs vector cosine forearm deduplication ($\ge 0.92$), deterministic clockwise angular sorting from 12 o'clock, greedy minimum-distance bipartite hand-to-arm matching, and CVAT native skeleton packaging.

```
[Input Image] ──> [Pass 1: Global Discovery] ──> [Passes 2-5: High-Res Crops] ──> [Pass 6: Merging] ──> [10 CVAT Skeletons]
```

### Key Technical Attributes
- **Zero Local Heavy ML:** Pure Python standard library and Pillow geometry. Remote perception via 9Router (defaulting to Claude Opus 5.5 High Thinking).
- **Pinned Port 5776:** Nuclio HTTP trigger bound strictly to host port `5776` with `maxWorkers: 1`.
- **CVAT Native Skeleton Contract:** 10 independent skeletons (`person`, 7 facial components, `buddha_arm`, `buddha_hand`) with child `points` sublabels, valid 1-based SVG topologies, and normalized visibility states.
- **Detailed Documentation:** Refer to [`docs/BUDDHA_MULTILIMB.md`](docs/BUDDHA_MULTILIMB.md) for complete mathematical derivations, coordinate transform proofs, and operational guidelines.

---

## 🔒 Security, Safety & Privacy Hygiene

1. **Zero Database Destructive Actions**: Scripts never delete CVAT databases, tasks, jobs, or volumes (no `docker compose down -v`).
2. **Credential & Privacy Hygiene**:
   - Secrets are masked across all logging (`sk-...xyz`).
   - WSL deployments transfer `$NineRouterKey` in-memory via `WSLENV` without writing plaintext secrets to disk.
   - Never log full image Base64 in terminal or application logs.
   - Raw-bytes HMAC verification compares with `hmac.compare_digest` to prevent timing attacks.
3. **DoS & Resource Exhaustion Defense**:
   - 32MB maximum request body size guard in handlers and `function.yaml`.
   - Polygon vertex count limit (10,000 vertices/polygon) prevents CPU rasterization starvation.
   - Per-frame detection limit (500 objects/image) prevents memory exhaustion.
   - Decompression bomb protection (`Image.MAX_IMAGE_PIXELS = 89_478_485`).
   - Storage quotas on feedback crops: max 50MB, max 150 crops total, max 10 crops/label, max 512px per crop.
4. **Honest Confidence Score Policy**: No false 1.0 confidence score fallbacks.

---

## 🧪 Testing

The repository maintains a 100% test pass rate across **1,105 automated unit, integration, and hermetic regression tests**:

```bash
python -m pytest
```

Test breakdown across major sub-systems:
- **Buddha Multi-Limb Detection (42 tests)**:
  - `tests/test_buddha_contract.py`: 6 tests (data structures, CVAT skeleton format, 1-based SVG generation).
  - `tests/test_buddha_schema.py`: 5 tests (YAML config, 10-label specifications, sublabel topologies).
  - `tests/test_buddha_coordinate_reprojection.py`: 4 tests (bidirectional transforms, degenerate box handling).
  - `tests/test_buddha_arm_ordering.py`: 4 tests (clockwise polar sorting, tie-breakers, group ID assignment).
  - `tests/test_buddha_arm_dedup.py`: 4 tests (forearm vector cosine similarity $\ge 0.92$, joint distance, IoU).
  - `tests/test_buddha_hand21.py`: 6 tests (21-joint kinematic chains, geometry quality gates, mudra validation).
  - `tests/test_buddha_model_resolution.py`: 8 tests (strict Opus 5.5 resolution, gated fallback, env overrides).
  - `tests/test_buddha_model_handler.py`: 2 tests (end-to-end multi-pass pipeline, concurrency, failure isolation).
  - `tests/test_buddha_cvat_shapes.py`: 2 tests (CVAT 2.75.1 shape compliance, visibility normalization).
- **Week 2 Pose17 & VF50 Skeletons (310+ tests)**:
  - `tests/test_pose17_model_handler.py`: 19 tests (4-case occlusion merging, crop refinement, ROI reprojection).
  - `tests/test_vf50_face_landmark.py`: 53 tests (7-component VinFast topology, laterality invariants, SVG XML compliance).
  - `tests/test_vf50_model_handler.py`: 12 tests (face ROI refinement, multi-component merging).
  - `tests/test_week2_schema.py`: 30 tests (schema validation, visibility mapping, geometry contracts).
  - `tests/test_skeleton_contract.py`: 46 tests (CVAT skeleton format, sublabel types, points validation).
  - `tests/test_skeleton_geometry_quality.py`: 40 tests (kinematic edge angles, anatomical plausibility).
- **Phase 3B Full 31-Label Multi-Shape & Correction Memory (210+ tests)**:
  - `tests/test_harden_feedback_multimodal.py`: 12 tests (fingerprinting, multimodal few-shot payloads, storage bounds, webhook HMAC).
  - `tests/test_feedback_learning.py`: 25 tests (bipartite diff engine, 9-tier taxonomy, SQLite CRUD, rule derivation).
  - `tests/test_phase3b_taxonomy.py`: 43 tests (31-label master schema, exact grouping, ambiguity policies).
  - `tests/test_line_geometry.py`: 22 tests (2D spatial covariance PCA aspect ratio, medial pair resampling, DP simplification).
  - `tests/test_phase3b_service.py`: 10 tests (3-tier shape routing, instance pairing, lane centerlines).
  - `tests/test_phase3b_duplicate_suppression.py`: 10 tests (region rectangle suppression, lane bbox suppression).
  - `tests/test_phase3b_nuclio_handler.py`: 11 tests (31-label function definition, 32MB payload guard).
  - `tests/test_phase3b_geometry.py`: 12 tests (vector conversion, trailing bbox verification).
- **Phase 2 & Phase 3 Core Detectors (160+ tests)**:
  - `tests/test_geometry.py`: 34 tests (Pillow rasterization, CVAT 1D flat list, Shoelace area).
  - `tests/test_phase3_parser.py`: 8 tests (Box+mask parsing, strict schema validation).
  - `tests/test_phase3_service.py`: 10 tests (modes: `box`, `mask`, `box_and_mask`, `group_id` pairing).
  - `tests/test_phase3_nuclio_handlers.py`: 20 tests (Nuclio handlers, payload guards, error masking).
  - `tests/test_phase3_security_reliability.py`: 21 tests (DoS mitigation, memory limits, vertex boundaries, key sanitization).
- **Client, Config, Security & Hermetic Suites (360+ tests)**:
  - CLI, configuration loaders, image operations, regression benchmarks, and hermetic network isolation guards.

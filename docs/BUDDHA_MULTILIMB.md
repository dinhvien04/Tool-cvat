# 9Router Buddha Multi-Limb Pose & Hand Landmark Annotation Architecture

**Version:** 1.0  
**Date:** September 2026  
**Status:** Canonical & Production-Ready  
**Detector:** `ninerouter-buddha-multilimbs`  
**Trigger Port:** HTTP Pinned Host Port `5776`  
**CVAT Serverless Contract:** Native Skeleton Hierarchy (10 Labels)  

---

## 1. Executive Summary & Problem Formulation

Standard human pose estimation frameworks (COCO Keypoints, OpenPose, MediaPipe Pose, YOLOv8-pose) are predicated on anatomical priors of single-body or standard multi-person topologies with exactly two arms (left shoulder -> left elbow -> left wrist; right shoulder -> right elbow -> right wrist). When confronted with **complex Buddhist and Hindu sacred iconography**—such as the Thousand-Armed Avalokiteshvara (Quán Thế Âm Bồ Tát Thiên Thủ Thiên Nhãn), Sahasrabhuja, or multi-armed esoteric deities—conventional architectures suffer catastrophic failure modes:

1. **False Multi-Person Hallucination:** Radial arms projecting from a single central torso are misinterpreted by standard multi-person pose detectors as clusters of fragmented, overlapping people, generating dozens of phantom torsos, phantom legs, and conflicting keypoints.
2. **Kinematic Graph Truncation:** Standard 17-point skeletons can only bind two upper extremities, completely dropping or truncating all secondary, tertiary, and outer radial arms.
3. **Severe Hand Truncation & Occlusion Confusion:** In dense sacred iconography, dozens of hands appear in mudras (symbolic ritual gestures) or clasping sacred attributes (lotus, vajra, kalasha, dharmachakra). Standard models either fail to segment these hands or fail to link them to their corresponding parent arms.
4. **Local Resource Exhaustion:** Running local multi-stage neural models (e.g., PyTorch, CUDA, SAM2, MediaPipe) on annotator workstations introduces multi-gigabyte dependency sprawl, GPU memory exhaustion, and cross-platform instability.

The `ninerouter-buddha-multilimbs` detector resolves these challenges by introducing a **Zero Local Heavy ML, Hierarchical Multi-Pass Perception Pipeline** orchestrated through 9Router vision model endpoints (defaulting to Claude Opus 5.5 High Thinking).

```
                                      [Input Image]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │     Pass 1: Global Scene Discovery            │
                    │  - Central Torso Center (cx, cy)              │
                    │  - Face Bounding Box & Body Pose17            │
                    │  - Coarse Buddha Arm & Hand Candidates        │
                    └───────────────────────┬───────────────────────┘
                                            │
                  ┌─────────────────────────┼─────────────────────────┐
                  ▼                         ▼                         ▼
      ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
      │  Pass 2: Arm Refine   │ │  Pass 3: Hand Refine  │ │  Pass 4: Face VF50    │
      │  - 15% Padded Crop    │ │  - 20% Padded Crop    │ │  - Face ROI Crop      │
      │  - Root/Elbow/Wrist   │ │  - 21-Joint Mudra     │ │  - 7 VinFast Skeletons│
      │  - Reproject to Global│ │  - Reproject to Global│ │  - Reproject to Global│
      └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                  │                         │                         │
                  └─────────────────────────┼─────────────────────────┘
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │     Pass 6: Geometric Merging & Validation    │
                    │  - Arm Deduplication (Cosine + IoU)           │
                    │  - Deterministic Angular Arm Ordering         │
                    │  - Greedy Bipartite Arm-to-Hand Linking       │
                    │  - Group ID Assignment (1=Body, 2..N=Arms)    │
                    │  - CVAT Native Skeleton Adjacency & Packaging │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                            [CVAT 2.75.1 Native Skeletons]
```

---

## 2. Zero Local Heavy ML Directive

The detector strictly enforces the repository-wide **Zero Local Heavy ML Directive**:
- **Zero Local Weights:** Absolutely no local `.pt`, `.pth`, `.bin`, `.onnx`, or `.engine` neural network weight files.
- **Zero Heavy Frameworks:** No runtime dependency on `torch`, `torchvision`, `cuda`, `tensorflow`, `ultralytics`, `segment-anything`, or `mediapipe`.
- **Pure Python & Pillow Geometry:** All geometric operations—bounding box cropping, padding, affine coordinate transformations, angular sorting, vector cosine deduplication, and bipartite matching—run using standard library math and `PIL.Image`.
- **Serverless Delegation:** Vision perception is delegated entirely via OpenAI-compatible vision requests to 9Router (`http://127.0.0.1:20128` on host, `http://host.docker.internal:20128` inside Docker/Nuclio).
- **Featherweight Container:** Builds from lightweight `python:3.11-slim` in seconds, minimizing memory and CPU footprints on annotation workstations.

---

## 3. CVAT 10-Label Skeleton Schema Specification

The detector outputs exactly **10 distinct skeleton labels** into CVAT. Each skeleton represents an independent geometric structure with dedicated child `points` sublabels, human-editable keypoint visibility states, and valid SVG adjacency lines:

| ID | Label Name | Type | Keypoint Count | Adjacency Edges | Semantic Function |
|---|---|:---:|:---:|:---:|---|
| **1** | `person` | `skeleton` | 17 | 18 | Central torso, head, and primary legs (COCO Pose17 topology) |
| **2** | `longmaytrai` | `skeleton` | 5 | 4 | Left eyebrow (outer -> inner, points 0..4) |
| **3** | `longmayphai` | `skeleton` | 5 | 4 | Right eyebrow (inner -> outer, points 5..9) |
| **4** | `songmui` | `skeleton` | 4 | 3 | Nose bridge (top -> base, points 10..13) |
| **5** | `mattrai` | `skeleton` | 8 | 8 | Left eye contour (closed contour, points 14..21) |
| **6** | `matphai` | `skeleton` | 8 | 8 | Right eye contour (closed contour, points 22..29) |
| **7** | `moingoai` | `skeleton` | 12 | 12 | Outer lips (closed contour, points 30..41) |
| **8** | `moitrong` | `skeleton` | 8 | 8 | Inner lips (closed contour, points 42..49) |
| **9** | `buddha_arm` | `skeleton` | 3 | 2 | Dynamic radial arm (`root`, `elbow`, `wrist`) |
| **10** | `buddha_hand` | `skeleton` | 21 | 20 | Dynamic hand skeleton (`wrist` + 5 kinematic finger chains) |

### 3.1 Kinematic Topologies

#### `buddha_arm` (3 Keypoints, 2 Edges)
- **Keypoints:**
  1. `root` (Arm shoulder/torso root attachment point)
  2. `elbow` (Arm elbow joint)
  3. `wrist` (Arm wrist joint)
- **Edges:**
  - `(root, elbow)`
  - `(elbow, wrist)`
- **SVG Representation:** 1-based node IDs (`1`, `2`, `3`) with directed edges:
  ```xml
  <svg>
    <line x1="100" y1="200" x2="150" y2="150" stroke="#00ffff" stroke-width="2" data-node-id="1" />
    <line x1="150" y1="150" x2="200" y2="100" stroke="#00ffff" stroke-width="2" data-node-id="2" />
    <circle cx="100" cy="200" r="3" fill="#ff0000" data-node-id="1" />
    <circle cx="150" cy="150" r="3" fill="#00ff00" data-node-id="2" />
    <circle cx="200" cy="100" r="3" fill="#0000ff" data-node-id="3" />
  </svg>
  ```

#### `buddha_hand` (21 Keypoints, 20 Edges)
Comprising a wrist root joint and 5 sequential finger chains matching the canonical hand anatomy:
- **Joint 1 (`wrist`):** Hand wrist root anchor.
- **Thumb Chain (Joints 2..5):** `thumb_cmc` -> `thumb_mcp` -> `thumb_ip` -> `thumb_tip`.
- **Index Chain (Joints 6..9):** `index_mcp` -> `index_pip` -> `index_dip` -> `index_tip`.
- **Middle Chain (Joints 10..13):** `middle_mcp` -> `middle_pip` -> `middle_dip` -> `middle_tip`.
- **Ring Chain (Joints 14..17):** `ring_mcp` -> `ring_pip` -> `ring_dip` -> `ring_tip`.
- **Pinky Chain (Joints 18..21):** `pinky_mcp` -> `pinky_pip` -> `pinky_dip` -> `pinky_tip`.
- **Kinematic Tree:**
  $$\text{Edges} = \{(1, 2), (2, 3), (3, 4), (4, 5)\} \cup \bigcup_{k \in \{6, 10, 14, 18\}} \{(1, k), (k, k+1), (k+1, k+2), (k+2, k+3)\}$$
- **CVAT 1-Based Node Contract:** To satisfy CVAT's strict SVG XML validator, sublabel IDs and SVG `data-node-id` attributes are strictly 1-indexed (`1` through `21`).

---

## 4. Hierarchical Multi-Pass Orchestration

### Pass 1: Global Scene Discovery
- The full image is evaluated at a baseline resolution.
- Prompts instruct the vision model to detect:
  1. The central torso center $(c_x, c_y)$.
  2. The face bounding box `face_roi`.
  3. Central body 17-keypoint skeleton (filtering out radial arms).
  4. Candidate radial arms: root, elbow, wrist coordinates, and optional `hand_roi` bounding box.

### Pass 2: Arm Crop Refinement
- For each detected arm candidate from Pass 1, a bounding box is derived around `root`, `elbow`, and `wrist` with **15% contextual padding**:
  $$\text{box} = [y_{\min} - 0.15h, x_{\min} - 0.15w, y_{\max} + 0.15h, x_{\max} + 0.15w]$$
- The region of interest is cropped from the original high-resolution image using pure Pillow and passed to 9Router.
- The refined normalized coordinates $[0, 1000]$ are reprojected back into global normalized coordinates.

### Pass 3: Hand Crop Refinement
- For each arm with an identified wrist or `hand_roi`, a tight crop with **20% padding** is extracted around the wrist joint.
- The crop is queried at maximum resolution for the 21 hand joints.
- Reprojection maps crop coordinates back into the full canvas frame.
- **Failure Isolation:** Refinements are dispatched concurrently via Python's `ThreadPoolExecutor(max_workers=refine_workers)`. If any individual hand crop fails due to occlusions, extreme lighting, or network timeouts, the pipeline isolates the exception, logs a warning, and retains the parent arm without crashing the entire detection session.

### Pass 4: Face Landmark Refinement
- Crops the detected `face_roi` and extracts 50 landmarks partitioned across the 7 VinFast face components (`longmaytrai`, `longmayphai`, `songmui`, `mattrai`, `matphai`, `moingoai`, `moitrong`).
- Reprojects all 50 coordinates into global space.

### Pass 5: Central Body Refinement
- Performs localized refinement of the primary torso and legs, ensuring that radial arms do not distort shoulder or hip coordinates.

---

## 5. Mathematical & Geometric Algorithms

### 5.1 Coordinate Spaces and Bidirectional Reprojection

All vision prompts operate on a normalized $[0, 1000] \times [0, 1000]$ integer grid. Given an image of pixel width $W$ and height $H$:

1. **Denormalization to Canvas Pixels:**
   $$x_{\text{pixel}} = \frac{x_{\text{norm}}}{1000.0} \times W, \quad y_{\text{pixel}} = \frac{y_{\text{norm}}}{1000.0} \times H$$

2. **Forward Crop Normalization:**
   Given a crop bounding box $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ in normalized global units:
   $$x_{\text{crop}} = \frac{x_{\text{global}} - x_{\min}}{x_{\max} - x_{\min}} \times 1000.0, \quad y_{\text{crop}} = \frac{y_{\text{global}} - y_{\min}}{y_{\max} - y_{\min}} \times 1000.0$$

3. **Inverse Global Reprojection:**
   $$x_{\text{global}} = x_{\min} + \frac{x_{\text{crop}}}{1000.0} \times (x_{\max} - x_{\min}), \quad y_{\text{global}} = y_{\min} + \frac{y_{\text{crop}}}{1000.0} \times (y_{\max} - y_{\min})$$

4. **Degenerate Box Guard:**
   If $x_{\max} - x_{\min} \le 0$ or $y_{\max} - y_{\min} \le 0$, the transformation safely returns the original input coordinates without division-by-zero errors.

### 5.2 Deterministic Spatial / Angular Arm Ordering

Because vision models return candidate lists in non-deterministic order depending on token generation paths, arms are sorted deterministically based on their polar position relative to the central torso anchor $(c_x, c_y)$:

1. **Wrist Vector:**
   $$\Delta x = x_{\text{wrist}} - c_x, \quad \Delta y = y_{\text{wrist}} - c_y$$

2. **Clockwise Polar Angle (from 12 o'clock / North):**
   $$\theta = \text{atan2}(\Delta y, \Delta x)$$
   $$\theta_{\text{clockwise}} = \left(\theta + \frac{\pi}{2} + 2\pi\right) \pmod{2\pi}$$

3. **Stable Multi-Tier Tie-Breaking:**
   If two arms share identical angular bearings:
   $$\text{Key} = (\text{round}(\theta_{\text{clockwise}}, 4), \; -r_{\text{wrist}}, \; x_{\text{root}}, \; y_{\text{root}})$$
   where $r_{\text{wrist}} = \sqrt{\Delta x^2 + \Delta y^2}$ is radial distance.

4. **Sequential Group ID Assignment:**
   - Group `1`: Reserved for central body (`person`) and all 7 facial skeletons (`longmaytrai`..`moitrong`).
   - Groups `2, 3, 4, ...`: Assigned sequentially to sorted radial arms in clockwise order.

### 5.3 Vector Cosine & Joint Distance Arm Deduplication

To eliminate duplicate candidate detections while preserving valid, tightly packed parallel limbs, the deduplication engine enforces a composite geometric threshold:

1. **Forearm Orientation Vector:**
   $$\vec{u} = (x_{\text{wrist}} - x_{\text{elbow}}, \; y_{\text{wrist}} - y_{\text{elbow}})$$

2. **Cosine Similarity:**
   $$\cos(\theta) = \frac{\vec{u}_1 \cdot \vec{u}_2}{\|\vec{u}_1\|_2 \; \|\vec{u}_2\|_2}$$

3. **Composite Duplicate Rule:**
   Two arms $A_1$ and $A_2$ are classified as duplicates if and only if:
   $$\cos(\vec{u}_1, \vec{u}_2) \ge 0.92 \quad \land \quad \|J_1 - J_2\|_2 < 25.0 \quad \land \quad \text{IoU}(\text{Box}_1, \text{Box}_2) \ge 0.65$$
   When detected, the instance with higher prediction confidence is retained.

### 5.4 Greedy Minimum-Distance Bipartite Hand-Arm Matching

To maintain strict 1:1 structural pairing between radial arms and hands:

1. **Cost Matrix Formulation:**
   For $M$ candidate hands and $N$ confirmed arms, compute the Euclidean distance between hand landmark 0 (`wrist`) and arm keypoint 3 (`wrist`):
   $$D_{i,j} = \|\vec{x}_{\text{hand } i, \text{wrist}} - \vec{x}_{\text{arm } j, \text{wrist}}\|_2$$

2. **Greedy Matching:**
   Iteratively select $\min_{(i,j)} D_{i,j}$ subject to $D_{i,j} \le \tau_{\text{hand\_match}}$ (default $\tau = 80.0$ in normalized coordinates).
   - Once matched, hand $i$ inherits the exact `group_id` of arm $j$, establishing explicit 1:1 pairing in CVAT.
   - Any unmatched hands receive unique independent group IDs ($> \max(\text{arm\_groups})$) with `parent_arm_id = None`.

---

## 6. Model Resolution & Fallback Policy

The detector adheres to a strict model resolution hierarchy:

```
[Target: claude-opus-5-5 / ag/claude-opus-5-5]
                      │
        ┌─────────────┴─────────────┐
        ▼ Available                 ▼ Not Available
  [Deploy Opus 5.5]          BUDDHA_ALLOW_MODEL_FALLBACK == 1 ?
                                    │
                      ┌─────────────┴─────────────┐
                      ▼ Yes                       ▼ No
               [Fallback Hierarchy]         [Raise RuntimeError]
               1. ag/claude-opus-4-6-thinking
               2. claude-sonnet-4-6
               3. claude-sonnet-4-5
```

- **Strict Mode (Default):** In production environments, silent model downgrades are strictly forbidden. If Claude Opus 5.5 is unavailable, initialization raises an explicit `RuntimeError`.
- **Authorized Fallback:** Setting `BUDDHA_ALLOW_MODEL_FALLBACK=1` permits graceful downgrade to secondary vision models.
- **Explicit Override:** Operators can specify `BUDDHA_MODEL` or `BUDDHA_REFINE_MODEL` to target specific models during experimentation.

---

## 7. Operational Runbook & PowerShell Automation

### 7.1 Pre-Flight Diagnostics (`scripts/buddha_preflight.ps1`)

Validates 9Router connectivity, model availability, port 5776 status, Docker daemon readiness, `nuctl` CLI presence, function YAML specification validity, and zero module drift:

```powershell
# Run complete pre-flight check
powershell -ExecutionPolicy Bypass -File scripts/buddha_preflight.ps1

# Pre-flight with optional CVAT task label schema inspection
powershell -ExecutionPolicy Bypass -File scripts/buddha_preflight.ps1 -TaskId 20
```

### 7.2 Safe Deployment (`scripts/buddha_deploy.ps1`)

Deploys `ninerouter-buddha-multilimbs` to CVAT Serverless via Nuclio, binding the HTTP trigger to pinned port 5776:

```powershell
# Deploy with default settings (port 5776, fallback enabled)
powershell -ExecutionPolicy Bypass -File scripts/buddha_deploy.ps1

# Deploy in strict mode (Opus 5.5 strictly required)
powershell -ExecutionPolicy Bypass -File scripts/buddha_deploy.ps1 -Strict

# Deploy with explicit custom models
powershell -ExecutionPolicy Bypass -File scripts/buddha_deploy.ps1 -BuddhaModel "claude-opus-5-5" -BuddhaRefineModel "claude-sonnet-4-6"
```

### 7.3 Smoke Testing (`scripts/buddha_smoke_test.ps1`)

Verifies the deployed container endpoint or local ModelHandler execution:

```powershell
# Test deployed Nuclio container on port 5776
powershell -ExecutionPolicy Bypass -File scripts/buddha_smoke_test.ps1

# Test on a specific test image
powershell -ExecutionPolicy Bypass -File scripts/buddha_smoke_test.ps1 -ImagePath "tests/fixtures/buddha_sample.jpg"

# Test local ModelHandler in-process without network container
powershell -ExecutionPolicy Bypass -File scripts/buddha_smoke_test.ps1 -LocalOnly
```

### 7.4 Schema Generation & Validation (`scripts/buddha_schema.py`)

Inspects and outputs the canonical CVAT label specification:

```powershell
# Display visual summary table of all 10 skeleton labels and sublabels
python scripts/buddha_schema.py --summary

# Output CVAT-compliant JSON specification
python scripts/buddha_schema.py --json

# Rebuild serverless/ninerouter-buddha-multilimbs/nuclio/function.yaml
python scripts/build_buddha_yaml.py
```

---

## 8. CVAT Integration & User Workflow

1. Open CVAT at `http://localhost:18080`.
2. Navigate to your project or task containing Buddhist multi-limb iconography (e.g., Task #20).
3. Ensure the task has the 10 skeleton labels configured (run `python scripts/buddha_schema.py --summary` to inspect required labels).
4. Click **AI Tools** -> **Detectors** in the left sidebar.
5. Select **9Router Buddha Multi-Limb Pose & Hand Detector** (`ninerouter-buddha-multilimbs`).
6. Because `function.yaml` specifies `"type": "skeleton"` across all 10 labels, CVAT automatically maps the detector labels to the task labels.
7. Click **Annotate**. The system executes the hierarchical multi-pass pipeline:
   - Central body appears as a standard 17-point `person` skeleton (`group_id=1`).
   - Facial features appear as 7 distinct VinFast component skeletons (`group_id=1`).
   - Radial arms appear as 3-joint `buddha_arm` skeletons (`group_id=2, 3, ...` in clockwise order).
   - Hands appear as 21-joint `buddha_hand` skeletons sharing the exact `group_id` of their parent arm.

---

## 9. Security, Privacy & Reliability Guardrails

1. **Credential Hygiene:** API keys and tokens are never printed, logged, or serialized to disk. Masking via `mask_api_key()` (`sk-...xyz`) is strictly enforced across all handlers.
2. **Hermetic Test Isolation:** Tests run under `tests/conftest.py` hermetic guards, prohibiting unauthorized unmocked network egress to ports `20128`, `18080`, or `8070`.
3. **Payload Protection:** 32MB payload guards prevent buffer overflow attacks, and `Image.MAX_IMAGE_PIXELS = 89_478_485` guards against decompression bomb exploits.
4. **Volume Protection:** Operational scripts never invoke `docker compose down -v` or delete persistent volumes.
5. **Deterministic Port Binding:** Nuclio HTTP trigger is pinned strictly to host port `5776`, eliminating port collisions with existing Phase 2/3/3B detectors.

---

## 10. Verification & Test Metrics

The detector is validated by 42 dedicated unit and integration tests across 9 test suites:

| Test Suite | Tests | Scope |
|---|:---:|---|
| `tests/test_buddha_contract.py` | 6 | Data structures, serialization, CVAT skeleton format, SVG generation |
| `tests/test_buddha_schema.py` | 5 | YAML configuration, 10-label specifications, sublabel topologies |
| `tests/test_buddha_coordinate_reprojection.py` | 4 | Bidirectional coordinate transforms, degenerate box handling |
| `tests/test_buddha_arm_ordering.py` | 4 | Angular clockwise sorting, tie-breaking, group ID assignment |
| `tests/test_buddha_arm_dedup.py` | 4 | Vector cosine similarity, joint distance, IoU deduplication |
| `tests/test_buddha_hand21.py` | 6 | 21-joint kinematic chains, geometry quality gates, mudra validation |
| `tests/test_buddha_model_resolution.py` | 8 | Strict Opus 5.5 resolution, gated fallbacks, environment overrides |
| `tests/test_buddha_model_handler.py` | 2 | End-to-end multi-pass pipeline, concurrency, failure isolation |
| `tests/test_buddha_cvat_shapes.py` | 2 | CVAT 2.75.1 shape compliance, visibility normalization |
| **Total Dedicated Buddha Tests** | **42** | **100% Hermetic Pass Rate** |
| **Total Repository Tests** | **1,105** | **100% Pass Rate (0 Regressions)** |

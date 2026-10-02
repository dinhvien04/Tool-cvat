# Codebase Module Map & Architectural Catalog

**Scope**: File-by-File Catalog, Key Classes, Methods, Dependencies & Data Flow  
**Target Audience**: AI Agents maintaining, refactoring, or extending the repository  
**Last Updated**: October 2026

---

## 1. Directory Layout & Module Structure

```
D:/tool-cvat/
├── app/                  # Application runtime, 9Router client, feedback loop, pipeline
├── config/               # Authoritative YAML/JSON schemas, taxonomies, label specs
├── core/                 # Pure Python math, geometry, safety invariants, contracts
├── docs/                 # Documentation suite for humans and agents
│   └── agent/            # Deep-dive architectural docs for AI agents
├── scripts/              # PowerShell & Python deployment, synchronization, testing tools
├── serverless/           # Self-contained build contexts for Nuclio serverless containers
│   ├── ninerouter-box/
│   ├── ninerouter-polygon/
│   ├── ninerouter-mask/
│   ├── ninerouter-box-3d/
│   └── ninerouter-human-pose-17/
├── tests/                # Exhaustive pytest test suites (unit, safety, integration, drift)
├── main.py               # CLI entrypoint for pipeline execution and feedback commands
└── requirements.txt      # Root Python dependencies (Pillow, requests, pyyaml, pytest)
```

---

## 2. `core/` — Invariants, Geometry & Data Contracts

All code in `core/` strictly follows the **Zero Local Heavy ML Directive** (standard library + `Pillow` only).

### 2.1. `core/cvat_safety.py`
- **Purpose**: Enforces read-only safety for production CVAT tasks and jobs.
- **Key Constants**:
  * `DEFAULT_PROTECTED_JOBS = frozenset({12, 13, 15})`
  * `DEFAULT_PROTECTED_TASKS = frozenset({12, 13, 15})`
- **Key Functions**:
  * `assert_job_writable(job_id: int) -> None`: Raises `DataSafetyViolationError` if job is in protected set.
  * `assert_task_writable(task_id: int) -> None`: Raises `DataSafetyViolationError` if task is in protected set.
  * `is_job_protected(job_id: int) -> bool`
  * `is_task_protected(task_id: int) -> bool`
- **Usage Invariant**: Every test or script that creates, edits, or deletes annotations MUST call these guards before any write call.

### 2.2. `core/taxonomy.py`
- **Purpose**: Authoritative 31-label master schema validator and shape policy router.
- **Key Constants**:
  * `MASTER_31_LABELS`: Tuple of 31 canonical label names in exact indexed order.
  * `POLICY_BOX_MASK` (`"box_mask"`): 14 foreground countable instances.
  * `POLICY_POLYGON_MASK` (`"polygon_mask"`): 10 background semantic regions.
  * `POLICY_POLYLINE` (`"polyline"`): 7 lane markings and road geometry.
- **Key Classes / Functions**:
  * `get_label_policy(label: str) -> str`: Returns policy slug in $O(1)$.
  * `get_allowed_shapes_for_label(label: str) -> FrozenSet[str]`
  * `resolve_ambiguity(label: str) -> str`: Enforces zero-aliasing (e.g. `pedestrian` $\neq$ `person`).

### 2.3. `core/geometry.py`
- **Purpose**: 2D coordinate transformation, bounding box operations, polygon clipping, and mask encoding.
- **Key Functions**:
  * `calculate_polygon_area(points: Sequence[Tuple[float, float]]) -> float`: Pure Python Green's Theorem / Shoelace formula.
  * `denormalize_box(box_norm: Sequence[int], width: int, height: int) -> List[float]`: Converts $[ymin, xmin, ymax, xmax]$ in $[0, 1000]$ to $[xtl, ytl, xbr, ybr]$ in pixel space.
  * `denormalize_contour(contour: Sequence[Sequence[int]], width: int, height: int) -> List[Tuple[float, float]]`
  * `polygon_to_cvat_mask(polygon_px, width, height) -> Dict[str, Any]`: Uses Pillow `ImageDraw.polygon()` to rasterize binary mask, crops to tight bounding box, and serializes to CVAT 1D Flat List.
  * `calculate_box_iou(box1: Sequence[float], box2: Sequence[float]) -> float`

### 2.4. `core/line_geometry.py`
- **Purpose**: Centerline approximation and simplification for Policy C (Polyline).
- **Key Functions**:
  * `calculate_polygon_aspect_ratio(vertices) -> float`: Computes spatial elongation using 2D covariance matrix eigenvalues (PCA).
  * `extract_centerline_from_polygon(polygon_px, num_samples=15) -> Optional[List[Tuple[float, float]]]`: Medial pair reduction.
  * `simplify_polyline(points, epsilon=1.5) -> List[Tuple[float, float]]`: Ramer-Douglas-Peucker simplification.

### 2.5. `core/vision_contract.py`
- **Purpose**: Prompt templating and coordinate normalization for remote 9Router vision models.
- **Key Functions**:
  * `build_box_prompt(allowed_labels) -> str`: Structured prompt enforcing tight bounding boxes and no adjacent vehicle merging.
  * `build_polygon_prompt(allowed_labels) -> str`: Semantic polygon prompt enforcing contiguous regions.
  * `build_mask_prompt(allowed_labels) -> str`
  * `build_openai_vision_payload(prompt, base64_image, model, ...) -> Dict[str, Any]`

### 2.6. `core/week2_schema.py`
- **Purpose**: Canonical YAML schema loader for human pose and facial landmark datasets.
- **Key Classes**:
  * `Pose17Schema`: Immutable dataclass loaded from `config/week2_pose17.yaml` (17 keypoints, 18 edges).
  * `VF50Schema`: Immutable dataclass loaded from `config/week2_vf50.yaml` (50 landmarks, 7 components).
  * `load_pose17() -> Pose17Schema` (memoized via `@lru_cache`)
  * `load_vf50() -> VF50Schema` (memoized via `@lru_cache`)

### 2.7. `core/skeleton_contract.py`
- **Purpose**: Native CVAT 2.75.1 skeleton structure builder and visibility flag mapper.
- **Key Functions**:
  * `build_cvat_skeleton(parent_label, elements, confidence) -> Dict[str, Any]`
  * `map_visibility_to_cvat_flags(vis_code: int) -> Tuple[bool, bool]`: Maps `0/1/2` to `(outside, occluded)`.
  * `validate_skeleton_invariants(skeleton: Dict[str, Any]) -> None`: Asserts no point has both `outside=True` and `occluded=True`.

### 2.8. `core/quality_gate.py`
- **Purpose**: Heuristic validation of keypoint skeletons to trigger adaptive crop refinement.
- **Key Functions**:
  * `assess_pose17_quality(keypoints, width, height) -> PoseQualityAssessment`: Flags floating joints, reversed limbs, or anomalous bone ratios.

---

## 3. `app/` — Application Layer & Feedback Loop

### 3.1. `app/client.py`
- **`NineRouterClient`**:
  * `check_health() -> bool`: Verifies gateway connectivity.
  * `list_models() -> List[str]`: Fetches available vision models.
  * `send_vision_request(payload, timeout=60.0) -> Dict[str, Any]`: Executes `/v1/chat/completions` HTTP POST.
  * `mask_api_key(key: str) -> str`: Masks API keys for secure logging (`sk-...xyz`).

### 3.2. `app/parser.py`
- **Purpose**: Resilient parser for vision model JSON responses.
- **Key Functions**:
  * `extract_json_from_text(raw_text: str) -> str`: Strips markdown fences (` ```json `), extracts balanced JSON substring.
  * `parse_and_validate(raw_text: str, width: int, height: int, mode: str) -> List[Dict[str, Any]]`: Denormalizes coordinates and builds CVAT shape dictionaries.

### 3.3. `app/image_ops.py`
- **Purpose**: Image loading, resizing, encoding, and canonical fingerprinting.
- **Key Functions**:
  * `compute_image_fingerprint(image_bytes: bytes) -> str`: Generates canonical visual SHA-256 fingerprint `{w}x{h}_{sha256}`.
  * `resize_image_if_needed(image: Image.Image, max_dim: int) -> Tuple[Image.Image, float]`: Preserves aspect ratio.
  * `encode_image_to_base64(image: Image.Image, format="JPEG") -> str`

### 3.4. `app/feedback.py`
- **Purpose**: SQLite persistence and bipartite matching diff engine for human feedback.
- **Key Classes**:
  * `FeedbackStore`: Manages SQLite connection with `WAL` mode.
  * `GeometricDiffEngine`: Compares AI baseline shapes with annotator ground truth:
    - Bipartite matching across labels via IoU matrix.
    - Classifies diffs: `RELABEL`, `BOX_MOVE`, `BOX_RESIZE`, `MASK_EDIT`, `DELETE_FALSE_POSITIVE`, `ADD_MISSING`, `NO_CHANGE`.
  * `RuleGenerator`: Synthesizes corrections into textual prompt rules when count $\ge 3$.

### 3.5. `app/pipeline.py` & `app/service.py`
- **`run_pipeline(options: PipelineOptions) -> PipelineResult`**:
  Orchestrates the entire local inference flow: load image $\rightarrow$ compute fingerprint $\rightarrow$ retrieve feedback $\rightarrow$ invoke 9Router $\rightarrow$ parse $\rightarrow$ convert $\rightarrow$ save output artifacts.

---

## 4. `config/` — Declarative Schemas & Configurations

| File | Purpose | Authoritative For |
|---|---|---|
| `config/labels.yaml` | ID, name, and color palette for 31 labels | CVAT task label definitions |
| `config/cvat_labels.json` | JSON export ready for CVAT API task creation | Task setup via REST API |
| `config/label_geometry.yaml` | Strict mapping of labels to allowed shape types | Policy A/B/C routing |
| `config/label_semantics.yaml` | Hierarchical descriptions and attributes for labels | Prompt context enrichment |
| `config/learning.yaml` | Thresholds and storage limits for feedback engine | WAL pragma, crop size, quotas |
| `config/week2_pose17.yaml` | 17 keypoint IDs, semantic names, laterality, 18 edges | Human Pose 17 detector |
| `config/week2_vf50.yaml` | 50 facial landmark IDs, 7 components | Face VF-50 detector |

---

## 5. `scripts/` — Operations, Deployment & Sync

### 5.1. `scripts/sync_serverless_modules.py`
- **Role**: Master sync tool enforcing zero drift between root source code and `serverless/*/nuclio/` directories.
- **Commands**:
  * `python scripts/sync_serverless_modules.py`: Synchronizes all manifests.
  * `python scripts/sync_serverless_modules.py --check`: Validates without writing; exits with code `1` if drift exists.
  * `python scripts/sync_serverless_modules.py --target box`: Scoped sync for specific detector.

### 5.2. `scripts/deploy.ps1`
- **Role**: Unified PowerShell deployment orchestrator for CVAT Nuclio functions.
- **Parameters**:
  * `-Target <target>`: `box`, `polygon`, `mask`, `box-3d`, `human-pose-17`, `active-all`, `three`, etc.
  * `-NineRouterUrl <url>`: Override default gateway URL.
  * Automatically invokes `sync_serverless_modules.py --check` prior to build.

### 5.3. `scripts/remove.ps1`
- **Role**: Safe function un-deployer using `nuctl`.
- **Safety Invariant**: Explicitly refuses to remove non-9Router third-party models (`pth-mmpose-hrnet32`, `openvino-*`, etc.).

---

## 6. `serverless/` — Active Detector Functions

Every active directory under `serverless/` represents a self-contained Docker build context containing:
- `function.yaml`: Nuclio metadata, CVAT annotation spec, environment variables, mounts, and HTTP trigger.
- `main.py`: Nuclio entrypoint handler function (`handler(context, event)`).
- `model_handler.py`: Model loader, image pre-processor, 9Router caller, and shape converter.
- Synchronized local copies of `app/`, `core/`, and `config/`.

| Serverless Directory | Active Function Name | Pinned Port |
|---|---|:---:|
| `serverless/ninerouter-box/nuclio/` | `ninerouter-box` | Dynamic |
| `serverless/ninerouter-polygon/nuclio/` | `ninerouter-polygon` | Dynamic |
| `serverless/ninerouter-mask/nuclio/` | `ninerouter-mask` | Dynamic |
| `serverless/ninerouter-box-3d/nuclio/` | `ninerouter-box-3d` | Dynamic |
| `serverless/ninerouter-human-pose-17/nuclio/` | `ninerouter-human-pose-17` | `5997` |

---

## 7. `tests/` — Test Suite Architecture

| Test File | Primary Focus |
|---|---|
| `test_source_truth_drift.py` | Validates 100% byte-for-byte synchronization between root and serverless copies |
| `test_cvat_data_safety_harness.py` | Enforces read-only safety on tasks/jobs 12, 13, 15 and protects third-party models |
| `test_box_detector.py` | Validates `ninerouter-box` rectangle-only schema and NMS IoU overlap suppression |
| `test_box3d_detector.py` | Validates 16-coordinate 3D cuboid generation and front-face geometry |
| `test_pose17_model_handler.py` | Validates Pose17 skeleton formatting, viewer laterality, and visibility flags |
| `test_three_policies_strict.py` | Enforces Policy A/B/C partition rules across all 31 labels |
| `test_feedback_learning.py` | Tests SQLite WAL mode, bipartite IoU diff engine, and prompt rule derivation |
| `test_line_geometry.py` | Validates PCA aspect ratio, centerline sampling, and Douglas-Peucker simplification |
| `test_parser.py` | Tests JSON parsing resilience against markdown fences, numeric anomalies, and malformed arrays |

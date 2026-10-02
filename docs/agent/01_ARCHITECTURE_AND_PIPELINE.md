# System Architecture & Processing Pipeline

**Target Audience:** AI Agents & Core Maintainers  
**Context:** CVAT 2.75.1 (API v2) × 9Router Serverless AI Annotation Engine  
**Last Updated:** October 2026

---

## 1. High-Level Architectural Overview

The `Tool-cvat` system integrates **CVAT Community Edition (v2.75.1)** with remote Multimodal Large Language Models (Gemini 3.8/3.7 Flash, Claude Sonnet/Opus, etc.) through **9Router**, an OpenAI-compatible proxy gateway (`https://9router-production-e47a.up.railway.app`).

### Zero Local Heavy ML Directive (Non-Negotiable)

A core foundational principle of this codebase is the **Zero Local Heavy ML Directive**:
1. **No Local Weights or Heavy Frameworks**: Strictly NO PyTorch, TorchVision, CUDA, TensorFlow, Ultralytics, SAM/SAM-2, ONNX Runtime, or HuggingFace Transformers in the local environment or container images.
2. **Pure Python & Pillow Geometry**: All vector conversions, polygon rasterization, contour extraction, Shoelace area calculations, and CVAT 1D flat list encoding run exclusively through Python standard libraries (`math`, `typing`, `collections`, `sqlite3`) and `Pillow` (`PIL.Image`, `PIL.ImageDraw`).
3. **Featherweight Containers**: Serverless Nuclio containers build in seconds from `python:3.11-slim` with base images `< 200MB` (installed packages: `requests`, `pillow`, `pyyaml`).
4. **Remote Perception**: All cognitive perception and localization occur over HTTP requests to 9Router.

---

## 2. System Topology & Component Interactions

```
+-----------------------------------------------------------------------------------+
| HOST / WORKSTATION ENVIRONMENT                                                    |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | CVAT Web Application (http://localhost:18080)                            |   |
|   |  - Canvas 2D / 3D                                                         |   |
|   |  - AI Tools -> Detectors Dropdown                                         |   |
|   |  - Webhook Dispatcher (Event: update:job, completed)                      |   |
|   +---------------------------------------------------------------------------+   |
|                                     │                                             |
|                   Nuclio RPC        │ (POST / HTTP image payload)                 |
|                   Trigger Port      ▼                                             |
|   +---------------------------------------------------------------------------+   |
|   | CVAT Serverless Engine (Nuclio on Docker)                                 |   |
|   |                                                                           |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | Active Detector Containers (python:3.11-slim)                     |   |   |
|   |   |   - cvat.custom.ninerouter.box           (Rectangles only)        |   |   |
|   |   |   - cvat.custom.ninerouter.polygon       (Semantic polygons)      |   |   |
|   |   |   - cvat.custom.ninerouter.mask          (Binary bitmap masks)    |   |   |
|   |   |   - cvat.custom.ninerouter.box.3d        (16-coord 3D cuboids)    |   |   |
|   |   |   - cvat.custom.ninerouter.human.pose.17 (Port 5997, Skeletons)   |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |                                     │                                     |   |
|   |                     Host Bind Mount │ (Shared SQLite & Crops)             |   |
|   |                                     ▼                                     |   |
|   |   +-------------------------------------------------------------------+   |   |
|   |   | Local Persistence Directory: D:/tool-cvat/.tool-cvat/             |   |   |
|   |   |   - feedback.sqlite3 (WAL mode, concurrent reads/writes)          |   |   |
|   |   |   - examples/ (Privacy-preserving visual few-shot crops)          |   |   |
|   |   +-------------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------------+   |
|                                     │                                             |
|                                     │ HTTPS REST Calls                            |
|                                     │ (OpenAI-compatible Vision Payloads)         |
|                                     ▼                                             |
+-----------------------------------------------------------------------------------+
| 9ROUTER CLOUD / REMOTE GATEWAY (https://9router-production-e47a.up.railway.app)   |
|   - Multi-tenant model routing (ag/gemini-3.8-flash-low, medium, high, sonnet)   |
|   - Authentication via Bearer Token ($NINEROUTER_KEY)                             |
|   - Native multimodal vision completion (/v1/chat/completions)                    |
+-----------------------------------------------------------------------------------+
```

---

## 3. End-to-End Execution Flow (Detector Request)

When an annotator clicks **Annotate** inside CVAT UI or an automated test invokes a detector, the following sequence executes:

```
[1. CVAT Server]
       │  HTTP POST / with JSON: {"image": "<base64_string>", "threshold": 0.5}
       ▼
[2. Nuclio `main.py::handler()`]
       │  1. Check body size <= 32MB (MAX_REQUEST_BODY_SIZE = 33,554,432 bytes)
       │  2. Extract & decode Base64 image bytes
       │  3. Clamp threshold: max(0.0, min(1.0, threshold))
       ▼
[3. `model_handler.py::ModelHandler.infer()`]
       │  1. Load image via Pillow: `Image.open(io.BytesIO(image_bytes))`
       │  2. Compute canonical visual fingerprint: `{w}x{h}_` + SHA256(RGB_buffer)
       │  3. Resize image if max(w, h) > MAX_IMAGE_SIZE (preserving aspect ratio)
       │  4. Retrieve few-shot rules & crops from SQLite (if feedback enabled)
       │  5. Build structured prompt (JSON output schema + rules + allowed labels)
       ▼
[4. `app.client.py::NineRouterClient.send_vision_request()`]
       │  1. Encode resized image to Base64 data URL: `data:image/jpeg;base64,...`
       │  2. Assemble OpenAI-compatible chat messages:
       │     - System prompt (strict schema enforcement)
       │     - User prompt (text instructions + image_url)
       │  3. Post to 9Router `/v1/chat/completions` with streaming disabled
       │  4. Enforce timeout (default: 45.0s to 90.0s depending on detector)
       ▼
[5. `app.parser.py::parse_and_validate()`]
       │  1. Strip markdown code fences (` ```json ... ``` `)
       │  2. Parse JSON payload; handle trailing commas, numeric quirks
       │  3. Validate fields: `label`, `box_2d`, `mask`, `polygon`, `keypoints`
       │  4. Denormalize coordinates from [0, 1000] integer space to pixel space
       ▼
[6. Geometry Post-Processing & Conversion]
       │  - Bounding Box: [ymin, xmin, ymax, xmax] -> [xtl, ytl, xbr, ybr] + IoU suppression
       │  - Polygon: [[x, y], ...] -> flattened points list [x1, y1, x2, y2, ...]
       │  - Mask: Pillow polygon rasterization -> CVAT 1D Flat List [crop_pixels..., xmin, ymin, xmax, ymax]
       │  - 3D Box: front face + depth_ratio -> 16 float cuboid points via `cuboid_geometry.py`
       │  - Pose17: person bbox -> keypoints -> visibility normalization -> CVAT skeleton elements
       ▼
[7. Response Formatting]
       │  Return JSON array of CVAT shape dictionaries to CVAT Server
       ▼
[8. CVAT Canvas]
       Renders shapes directly on the annotation viewport.
```

---

## 4. Human-in-the-Loop Feedback & Active Learning Lifecycle

The system incorporates an automated feedback loop allowing annotator corrections in CVAT to continuously improve future AI detections **without retraining or local fine-tuning**:

```
                              [1. AI Inference Run]
                                        │
                                        ▼
                          [Save Baseline Prediction]
                       Saved to feedback.sqlite3 keyed by
                       canonical visual SHA-256 fingerprint
                                        │
                                        ▼
                         [2. Human Review in CVAT]
                     Annotator corrects boxes, edits masks,
                       swaps labels, or deletes false hits
                                        │
                                        ▼
                      [3. Webhook or CLI Sync Trigger]
                    - Webhook: CVAT emits update:job (completed)
                    - CLI: `python main.py feedback-sync --job <id>`
                                        │
                                        ▼
                        [4. Geometric Diff Engine]
                     Bipartite IoU matching between AI shapes
                       and Human shapes classifies changes:
                     * RELABEL (cross-label IoU >= 0.60)
                     * BOX_MOVE / BOX_RESIZE
                     * MASK_EDIT / REGION_EDIT / LANE_EDIT
                     * DELETE_FALSE_POSITIVE
                     * ADD_MISSING
                     * NO_CHANGE (IoU >= 0.92)
                                        │
                                        ▼
                       [5. Automatic Rule Derivation]
                      Patterns with >= 3 occurrences are
                    synthesized into actionable prompt rules:
                     "When detecting heavy vehicles, prefer
                               truck over car."
                                        │
                                        ▼
                     [6. Adaptive Prompt Injection (Next Run)]
                      Future requests retrieve matching rules
                     + 2-3 visual crops (<= 512px) dynamically
                       prepended to vision model input.
```

### Persistence & Storage Governance
- **Host Location**: `.tool-cvat/feedback.sqlite3` and `.tool-cvat/examples/`
- **Docker Mount**: Bound to `/opt/nuclio/feedback` in all Nuclio functions
- **SQLite Concurrency**: Configured with `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=10000`
- **Quota Safeguards**:
  * Maximum total storage: 50MB
  * Maximum total crops: 150
  * Maximum crops per label: 10
  * Maximum single crop dimension: 512px

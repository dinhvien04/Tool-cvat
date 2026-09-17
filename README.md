# CVAT × 9Router AI Annotation — Phase 1 MVP

Phase 1 implementation of the automated 2D object detection pipeline connecting **9Router** vision models (Gemini 3.8 Flash, Claude Sonnet, etc.) with the **CVAT** annotation schema.

This phase is completely decoupled from CVAT core, Docker, and database services. It takes an input image, queries the local 9Router vision model using a strict JSON contract, validates labels and normalized coordinates, denormalizes coordinates to pixel space on the original image, draws annotated bounding boxes, and generates structured artifacts.

---

## Multi-Agent Engineering Architecture

Phase 1 was built and verified using a coordinated 5-subagent workflow:

1. **9Router Integration Agent (`router-integration`)**:
   - Probed `http://127.0.0.1:20128` health (`GET /api/health`).
   - Discovered vision-capable models via `GET /v1/models/image-to-text` and `GET /v1/models` (`capabilities.vision == true`).
   - Tested OpenAI-compatible multimodal endpoint (`POST /v1/chat/completions`) with `stream: False` and Base64 Data URLs.
2. **Vision Contract / Prompt Agent (`vision-contract`)**:
   - Encoded the exact 31 CVAT labels in `config/labels.yaml`.
   - Preserved intentional label ambiguities without merging (`pedestrian` vs `person`, `traffic light` vs `traffic_light`, `traffic sign` vs `traffic_sign`).
   - Defined the default 13 rectangular bounding box candidate labels.
   - Designed the strict vision prompt enforcing raw JSON and `[ymin, xmin, ymax, xmax]` normalized coordinates in `[0, 1000]`.
3. **Python Implementation Agent (`python-impl`)**:
   - Implemented `app/client.py` (9Router HTTP client with streaming disabled, timing, auth support).
   - Implemented `app/image_ops.py` (Pillow image processing, aspect-ratio preserving resizing for transmission, bounding box rendering on original image).
   - Implemented `app/parser.py` (robust response parser handling markdown code blocks, JSON extraction, coordinate validation, clamping, and rejection of invalid boxes).
   - Implemented `app/models.py` (`python -m app.models` model inspection tool).
   - Implemented `app/pipeline.py` and `main.py` (end-to-end orchestration).
4. **Test / QA Agent (`qa-tester`)**:
   - Built 106 automated tests covering malformed LLM outputs, negative and out-of-bounds coordinates, inverted boxes, degenerate boxes, coordinate clamping, label validation, HTTP mock failures, and CLI options.
5. **Security / Reliability Reviewer (`security-reviewer`)**:
   - Audited credential masking (preventing API keys in logs, console, or reports).
   - Ensured raw Base64 image payloads are excluded from logs.
   - Verified strict timeout enforcement and exception resilience.
   - Verified `.gitignore` and `.env.example` compliance.

---

## Pipeline Overview

```
Input Image (test.jpg / test.png)
       ↓
Load & inspect original dimensions (W, H)
       ↓
Aspect-ratio preserving resize copy for transmission (max-size=1600px)
       ↓
Encode copy to Base64 Data URL (data:image/jpeg;base64,...)
       ↓
POST /v1/chat/completions (9Router, stream=False)
       ↓
Parse & clean raw LLM output (strip markdown ```json ... ```, extract JSON)
       ↓
Validate schema & check labels against CVAT schema (preserve distinctions)
       ↓
Validate & clamp coordinates [ymin, xmin, ymax, xmax] in [0, 1000]
       ↓
Denormalize coordinates to original image pixels:
    x1 = (xmin / 1000.0) * W
    y1 = (ymin / 1000.0) * H
    x2 = (xmax / 1000.0) * W
    y2 = (ymax / 1000.0) * H
       ↓
Reject invalid boxes (x2 <= x1 or y2 <= y1)
       ↓
Draw bounding boxes & label badges on ORIGINAL image
       ↓
Save Artifacts:
    output/result_bbox.jpg
    output/predictions.json
    output/raw_response.txt
    output/run_report.json
```

---

## Installation & Requirements

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Running local 9Router instance at `http://127.0.0.1:20128`

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` if custom configuration is desired:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `NINEROUTER_URL` | `http://127.0.0.1:20128` | Local 9Router endpoint |
| `NINEROUTER_KEY` | *(empty)* | Optional API key if 9Router requires auth |
| `VISION_MODEL` | `ag/gemini-3.8-flash-high` | Preferred vision model |
| `MAX_IMAGE_SIZE` | `1600` | Max dimension for API copy |
| `OUTPUT_DIR` | `output` | Directory for output artifacts |

---

## Usage

### 1. Discover Vision Models

Query 9Router and display all available vision models:

```bash
python -m app.models
```

Output format:
```text
9Router Status: Connected to http://127.0.0.1:20128 (Health: {'ok': True})
--------------------------------------------------------------------------------
#   | Model ID                       | Owner      | Context    | Vision  | Notes
--------------------------------------------------------------------------------
1   | ag/gemini-3.8-flash-high       | ag         | 1048576    | Yes     | [Recommended]
2   | ag/gemini-3.8-flash-medium     | ag         | 1048576    | Yes     | [Recommended]
...
Total vision models available: 19

Default auto-selection for detection pipeline: 'ag/gemini-3.8-flash-high'
```

To output raw JSON:
```bash
python -m app.models --json
```

### 2. Run Detection Pipeline

Run annotation on an image:
```bash
python main.py --image test.jpg
```

With specific model or parameters:
```bash
python main.py --image test.jpg --model ag/gemini-3.8-flash-high --output-dir output --max-image-size 1600
```

Strict mode (fails if model generates any label not in candidate list):
```bash
python main.py --image test.jpg --strict
```

---

## Output Artifacts

Running the pipeline on `test.jpg` creates the required artifacts in `output/`:

1. **`output/result_bbox.jpg`**:
   The original full-resolution image with bounding boxes drawn in distinct high-contrast colors and labeled badges.
2. **`output/predictions.json`**:
   Structured detection results containing original image metadata, model ID, normalized `box_2d`, denormalized `pixel_box`, and CVAT-compatible shape annotations.
3. **`output/raw_response.txt`**:
   The exact raw string returned by the vision model.
4. **`output/run_report.json`**:
   Execution metadata including timestamps, model used, latency breakdown (API duration vs total duration), original vs sent dimensions, detection counts, label distribution, token usage, and warning logs.

---

## CVAT Label Schema

`config/labels.yaml` contains all 31 exact CVAT labels:

```yaml
all_labels:
  - "pedestrian"
  - "rider"
  - "car"
  - "truck"
  - "bus"
  - "train"
  - "motorcycle"
  - "bicycle"
  - "traffic light"
  - "traffic sign"
  - "area/alternative"
  - "area/drivable"
  - "lane/crosswalk"
  - "lane/double white"
  - "lane/double yellow"
  - "lane/road curb"
  - "lane/single other"
  - "lane/single white"
  - "lane/single yellow"
  - "road"
  - "sidewalk"
  - "building"
  - "wall"
  - "fence"
  - "pole"
  - "vegetation"
  - "terrain"
  - "sky"
  - "person"
  - "traffic_light"
  - "traffic_sign"
```

### Ambiguities & Distinctions
The pipeline explicitly maintains and never merges or aliases:
- `pedestrian` vs `person`
- `traffic light` vs `traffic_light`
- `traffic sign` vs `traffic_sign`

### Default Bbox Candidates (13 labels)
Used for the Phase 1 rectangle detection MVP:
`pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`, `person`, `traffic_light`, `traffic_sign`.

---

## Test Suite

Run the full pytest suite:

```bash
python -m pytest -v
```

All 106 tests cover:
- Client health checks, timeouts, 401/403/500 HTTP failures, model capabilities filtering.
- Response parsing, markdown stripping, preamble/postamble removal, trailing commas, single-quote repairs.
- Coordinate validation, negative values, values > 1000, inverted coordinates, degenerate boxes.
- Coordinate clamping to original image dimensions.
- Exact label matching, ambiguity preservation, and non-bbox label rejection.
- Pipeline orchestration, error handling on missing/corrupted files, and model fallback.
- Security audits: credential masking, base64 data exclusion from reports, and `.gitignore` coverage.

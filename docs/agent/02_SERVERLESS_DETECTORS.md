# Serverless Detectors Reference

**Scope**: Specifications, Configurations, and Handlers for Nuclio Serverless Functions  
**Target Environment**: CVAT 2.75.1 (AI Tools -> Detectors)  
**Last Updated**: October 2026

---

## 1. Active Portfolio Overview (The 5 Core Detectors)

In the current production architecture, the multi-detector monoliths have been decomposed into **5 specialized, lightweight serverless detectors**:

| Function Name | CVAT Shape Type | Label Count | Default Model | Pinned Port | Key Specialty |
|---|:---:|:---:|---|:---:|---|
| **`ninerouter-box`** | `rectangle` | 14 | `ag/gemini-3.8-flash-medium` | Dynamic | Foreground vehicle & human bounding boxes with NMS suppression |
| **`ninerouter-polygon`** | `polygon` | 10 | `ag/gemini-3.8-flash-low` | Dynamic | Background semantic surface regions (road, building, sky, etc.) |
| **`ninerouter-mask`** | `mask` | 14 | `ag/gemini-3.8-flash-low` | Dynamic | Pixel-accurate binary bitmap instance masks |
| **`ninerouter-box-3d`** | `cuboid` | 8 | `ag/gemini-3.8-flash-low` | Dynamic | 8-vertex, 16-coordinate 3D bounding cuboids for road traffic |
| **`ninerouter-human-pose-17`**| `skeleton` | 17 keypoints | `ag/gemini-3.8-flash-low` | `5997` | Driver cabin human pose estimation with viewer laterality & crop refinement |

---

## 2. Detailed Detector Specifications

### 2.1. `ninerouter-box` (2D Bounding Box Detector)

- **Target Path**: `serverless/ninerouter-box/nuclio/`
- **Output Shape**: `type: "rectangle"`, `points: [xtl, ytl, xbr, ybr]`
- **Spec Type**: `rectangle`
- **Allowed Labels (14)**:
  `pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`, `pole`, `person`, `traffic_light`, `traffic_sign`.
- **Runtime Environment Variables**:
  * `BOX_MODEL`: `ag/gemini-3.8-flash-medium`
  * `DETECTION_MODE`: `box`
  * `MAX_IMAGE_SIZE`: `1600`
  * `MAX_TOKENS`: `4096`
  * `NINEROUTER_TIMEOUT`: `90.0`
- **NMS & Overlap Suppression**:
  Implemented in `model_handler.py::suppress_overlapping_rectangles()`:
  * When two detected boxes share the exact same label and their 2D IoU $\ge 0.75$, the box with the higher confidence score is retained and the redundant box is dropped.
- **Prompt Rules**:
  * One tight box per physical object touching outermost visible pixels.
  * No merging of adjacent vehicles (e.g. car next to bus).
  * Distant small vehicles are classified as `car`, not `truck`/`bus`.

---

### 2.2. `ninerouter-polygon` (Semantic Polygon Detector)

- **Target Path**: `serverless/ninerouter-polygon/nuclio/`
- **Output Shape**: `type: "polygon"`, `points: [x1, y1, x2, y2, ...]` (minimum 6 values, even length)
- **Spec Type**: `polygon`
- **Allowed Labels (10)**:
  `area/alternative`, `area/drivable`, `road`, `sidewalk`, `building`, `wall`, `fence`, `vegetation`, `terrain`, `sky`.
- **Runtime Environment Variables**:
  * `POLYGON_MODEL`: `ag/gemini-3.8-flash-low`
  * `DETECTION_MODE`: `polygon`
  * `MAX_IMAGE_SIZE`: `1280`
  * `MAX_TOKENS`: `2500`
  * `NINEROUTER_TIMEOUT`: `45.0`
- **Validation Constraints**:
  * Vertices are parsed as `[x, y]` pairs in image pixel space.
  * Minimum 3 vertices (6 coordinates) per polygon.
  * `group_id` is stripped/omitted (`ungrouped`) because background semantic regions do not represent countable paired instances.

---

### 2.3. `ninerouter-mask` (Bitmap Mask Detector)

- **Target Path**: `serverless/ninerouter-mask/nuclio/`
- **Output Shape**: `type: "mask"`, `points: [p0, p1, ..., xmin, ymin, xmax, ymax]`
- **Spec Type**: `mask`
- **Allowed Labels (14)**:
  Same 14 foreground classes as `ninerouter-box`.
- **Runtime Environment Variables**:
  * `MASK_MODEL`: `ag/gemini-3.8-flash-low`
  * `DETECTION_MODE`: `mask`
  * `MAX_IMAGE_SIZE`: `1280`
  * `MAX_TOKENS`: `2000`
  * `NINEROUTER_TIMEOUT`: `45.0`
- **CVAT Mask Encoding Mechanics**:
  * Pure Pillow rasterization: `ImageDraw.Draw(mask_img).polygon(pixel_points, fill=1)`
  * Cropped to tight bounding box `[xmin, ymin, xmax, ymax]`.
  * Flattened row-major binary mask + trailing 4 box bounds.

---

### 2.4. `ninerouter-box-3d` (3D Cuboid Detector)

- **Target Path**: `serverless/ninerouter-box-3d/nuclio/`
- **Output Shape**: `type: "cuboid"`, `points: [x1, y1, x2, y2, ..., x8, y8]` (16 float coordinates)
- **Spec Type**: `cuboid`
- **Allowed Labels (8)**:
  `car`, `truck`, `bus`, `trailer`, `construction_vehicle`, `pedestrian`, `motorcycle`, `bicycle`.
- **Runtime Environment Variables**:
  * `BOX3D_MODEL`: `ag/gemini-3.8-flash-low`
  * `DETECTION_MODE`: `box_3d`
  * `MAX_IMAGE_SIZE`: `1280`
  * `MAX_TOKENS`: `1600`
  * `NINEROUTER_TIMEOUT`: `45.0`
- **Geometry & Construction Engine (`cuboid_geometry.py`)**:
  * Vision model outputs front face `box_2d: [ymin, xmin, ymax, xmax]`, `depth_ratio: float` (0.08..0.45), and `side: "right" | "left"`.
  * `cuboid_from_front_face()` synthesizes 4 primary construction points and calculates vector offsets.
  * Generates 8 vertices conforming strictly to CVAT's `cuboidFrom4Points` ordering:
    - Front face: Points 1, 2, 3, 4
    - Rear face: Points 5, 6, 7, 8
  * Enables native, direct in-canvas manipulation and rotation in CVAT without parsing errors.

---

### 2.5. `ninerouter-human-pose-17` (Human Pose Skeleton Detector)

- **Target Path**: `serverless/ninerouter-human-pose-17/nuclio/`
- **Output Shape**: `type: "skeleton"`, `label: "person"`, `elements: [...]`
- **Spec Type**: `skeleton`
- **Sublabels (17 Points)**: `"1"` through `"17"`
- **Pinned Port**: `5997` (Trigger attribute `port: 5997` in `function.yaml`)
- **Authoritative Source**: `config/week2_pose17.yaml`
- **Topology (18 Edges)**:
  Includes 17 COCO standard edges + 2 ear-to-shoulder connections (4->6, 5->7).
- **Laterality Standard: Viewer Perspective**:
  * Even IDs (2, 4, 6, 8, 10, 12, 14, 16) = Viewer Right side of frame ($x$ larger).
  * Odd IDs (3, 5, 7, 9, 11, 13, 15, 17) = Viewer Left side of frame ($x$ smaller).
- **Visibility Contract**:
  * `0` (outside/untracked) $\rightarrow$ `outside=True, occluded=False`
  * `1` (occluded/covered) $\rightarrow$ `outside=False, occluded=True` (renders dashed lines)
  * `2` (visible) $\rightarrow$ `outside=False, occluded=False` (solid lines)
- **Adaptive Two-Pass Refinement**:
  * Pass 1: Global cabin context detects person bounding box and initial joints.
  * Quality Gate (`assess_pose17_quality`): Evaluates bone length ratios and flags floating joints.
  * Pass 2: If quality issues detected, crops person patch (+18% margin), refines coordinates at high resolution, and reprojects back to full image.

---

## 3. Specialized & Historical Detectors (For Reference)

The repository also contains implementations or historical provenance for specialized detectors:

1. **`ninerouter-face-vf50` (VinFast 50 Facial Landmarks)**:
   - Partitions 50 landmarks across **exactly 7 independent component skeletons**:
     `longmaytrai` (5 pts), `longmayphai` (5 pts), `songmui` (4 pts), `mattrai` (8 pts), `matphai` (8 pts), `moingoai` (12 pts), `moitrong` (8 pts).
   - Point 13 is the **nose bridge base** (not nose tip).
   - `moitrong` is strictly enclosed inside `moingoai`.
   - Multi-face scenes are grouped via integer `group_id`.

2. **`ninerouter-buddha-multilimbs` (Buddhist Sacred Iconography)**:
   - Targets thousand-armed Avalokiteshvara / Guanyin iconography.
   - Pinned to HTTP port `5776`.
   - Emits 10 native CVAT skeleton labels: central body (`person`), 7 facial skeletons, radial arms (`buddha_arm`), and 21-joint mudra hands (`buddha_hand`).
   - Forearm vector cosine deduplication ($\ge 0.92$) and clockwise polar sorting.

3. **`ninerouter-vision-31` (Full 31-Label Multi-Shape Detector)**:
   - Unified tri-policy detector covering autonomous driving datasets (BDD100K/Cityscapes).
   - Policy A (Instances): Paired `rectangle` + `mask` with shared `group_id`.
   - Policy B (Regions): Native `mask` with polygon boundary points; bboxes suppressed.
   - Policy C (Lanes): Centerline extraction via 2D spatial covariance PCA into `polyline`.

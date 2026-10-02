# Data Contracts, Coordinate Systems & Annotation Schemas

**Scope**: Formal Data Schemas, Mathematical Formulas, Normalization & Invariant Equations  
**Target Environments**: 9Router Models, Nuclio Handlers, CVAT 2.75.1 Canvas Engine  
**Last Updated**: October 2026

---

## 1. Master 31-Label Taxonomy & 3-Tier Policy Partition

The repository maintains an authoritative 31-label taxonomy for road scene and autonomous perception (`config/labels.yaml` & `config/label_geometry.yaml`). Every label belongs to **exactly one** strict shape policy:

```
                              Master 31 Labels
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
    POLICY A                    POLICY B                    POLICY C
   "box_mask"                "polygon_mask"                "polyline"
   (14 Labels)                 (10 Labels)                 (7 Labels)
   Countable Instances         Semantic Regions            Lane Markings
   Allowed: rectangle, mask    Allowed: polygon, mask      Allowed: polyline only
```

### 1.1. Partition Tables

#### Policy A: Countable Foreground Instances (14 labels)
| Label Name | Preferred Shape | Allowed Shapes | Group ID Assignment |
|---|:---:|:---:|:---:|
| `pedestrian` | `rectangle` | `rectangle`, `mask` | Paired (e.g. `1, 2, ...`) |
| `rider` | `rectangle` | `rectangle`, `mask` | Paired |
| `car` | `rectangle` | `rectangle`, `mask` | Paired |
| `truck` | `rectangle` | `rectangle`, `mask` | Paired |
| `bus` | `rectangle` | `rectangle`, `mask` | Paired |
| `train` | `rectangle` | `rectangle`, `mask` | Paired |
| `motorcycle` | `rectangle` | `rectangle`, `mask` | Paired |
| `bicycle` | `rectangle` | `rectangle`, `mask` | Paired |
| `traffic light` | `rectangle` | `rectangle`, `mask` | Paired |
| `traffic sign` | `rectangle` | `rectangle`, `mask` | Paired |
| `pole` | `rectangle` | `rectangle`, `mask` | Paired |
| `person` | `rectangle` | `rectangle`, `mask` | Paired |
| `traffic_light` | `rectangle` | `rectangle`, `mask` | Paired |
| `traffic_sign` | `rectangle` | `rectangle`, `mask` | Paired |

*Note*: `polygon` shape is strictly forbidden for Policy A instances.

#### Policy B: Amorphous Background Semantic Regions (10 labels)
| Label Name | Preferred Shape | Allowed Shapes | Group ID Assignment |
|---|:---:|:---:|:---:|
| `area/alternative` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `area/drivable` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `road` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `sidewalk` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `building` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `wall` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `fence` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `vegetation` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `terrain` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |
| `sky` | `mask` | `polygon`, `mask` | Ungrouped (`None`) |

*Note*: Bounding boxes (`rectangle`) are strictly suppressed for Policy B regions.

#### Policy C: Lane Markings & Linear Road Demarcations (7 labels)
| Label Name | Preferred Shape | Allowed Shapes | Centerline Extraction |
|---|:---:|:---:|:---:|
| `lane/crosswalk` | `polyline` | `polyline` only | 2D Spatial Covariance PCA |
| `lane/double white` | `polyline` | `polyline` only | Medial pair sample reduction |
| `lane/double yellow` | `polyline` | `polyline` only | Medial pair sample reduction |
| `lane/road curb` | `polyline` | `polyline` only | Medial pair sample reduction |
| `lane/single other` | `polyline` | `polyline` only | Medial pair sample reduction |
| `lane/single white` | `polyline` | `polyline` only | Medial pair sample reduction |
| `lane/single yellow` | `polyline` | `polyline` only | Medial pair sample reduction |

*Note*: Zero fallback to bounding box or mask. If polyline cannot be extracted, the shape is omitted.

---

### 1.2. Strict Anti-Aliasing Invariant (No Label Merging)

Under NO circumstances may an AI Agent merge or alias the following pairs:
- `"pedestrian"` $\neq$ `"person"` (BDD100K pedestrian vs generic person)
- `"traffic light"` $\neq$ `"traffic_light"` (spaced vs underscored schema variants)
- `"traffic sign"` $\neq$ `"traffic_sign"` (spaced vs underscored schema variants)

CVAT tasks validate labels strictly against database IDs. Renaming or aliasing will result in `400 Bad Request` or silent rejection.

---

## 2. Coordinate Systems & Transformations

```
             Remote Vision Model Space               Image Pixel Space
                 Normalized Integers                 Continuous / Clamped
                    [0, 1000]                     [0, W - 1] x [0, H - 1]
               ───────────────────────           ─────────────────────────
Bbox format:   [ymin, xmin, ymax, xmax]    -->    [xtl, ytl, xbr, ybr]
Contour/Poly:  [[x, y], [x, y], ...]       -->    [(x_px, y_px), ...]
```

### Critical Ordering Difference
- **Gemini / 9Router Bounding Boxes (`box_2d`)**:
  `[ymin, xmin, ymax, xmax]` — **Vertical $y$ comes FIRST**.
- **Gemini / 9Router Polygons & Contours (`mask`, `polygon`)**:
  `[[x, y], [x, y], ...]` — **Horizontal $x$ comes FIRST**.
- **CVAT Rectangle (`points`)**:
  `[xtl, ytl, xbr, ybr]` — $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ in pixel coordinates.

### Forward Denormalization Formulas

For an image of width $W$ and height $H$:
$$x_{\text{px}} = \text{clamp}\left(\frac{x_{\text{norm}}}{1000.0} \times (W - 1), 0.0, W - 1.0\right)$$
$$y_{\text{px}} = \text{clamp}\left(\frac{y_{\text{norm}}}{1000.0} \times (H - 1), 0.0, H - 1.0\right)$$

For bounding boxes:
$$\text{xtl} = \text{round}\left(\frac{\text{xmin}}{1000.0} \times (W - 1)\right), \quad \text{ytl} = \text{round}\left(\frac{\text{ymin}}{1000.0} \times (H - 1)\right)$$
$$\text{xbr} = \text{round}\left(\frac{\text{xmax}}{1000.0} \times (W - 1)\right), \quad \text{ybr} = \text{round}\left(\frac{\text{ymax}}{1000.0} \times (H - 1)\right)$$

---

## 3. CVAT Native Shape Serialization Contracts

### 3.1. Rectangle (`type: "rectangle"`)
```json
{
  "type": "rectangle",
  "label": "car",
  "points": [120.0, 350.0, 480.0, 620.0],
  "confidence": "0.92",
  "group_id": 1
}
```

### 3.2. Polygon (`type: "polygon"`)
```json
{
  "type": "polygon",
  "label": "road",
  "points": [0.0, 720.0, 300.0, 450.0, 980.0, 450.0, 1280.0, 720.0],
  "confidence": "0.88"
}
```
*Constraints*: Flattened 1D list of float pairs `[x1, y1, x2, y2, ...]`. Minimum 6 floats (3 points).

### 3.3. Mask (`type: "mask"`)
CVAT 2.75.1 encodes raster masks as a **1D flat list of run-length or binary bits followed by bounding box bounds**:
$$\text{points} = [p_0, p_1, p_2, \dots, p_{N-1}, x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$$

- Dimensions of the cropped bounding box:
  $$\text{width} = x_{\text{max}} - x_{\text{min}} + 1, \quad \text{height} = y_{\text{max}} - y_{\text{min}} + 1$$
  $$\text{total binary pixels } N = \text{width} \times \text{height}$$
- Pixels are flattened in **row-major order** ($0$ = background, $1$ = foreground).
- Total elements in array: $N + 4$.

```json
{
  "type": "mask",
  "label": "car",
  "points": [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 120, 350, 480, 620],
  "confidence": "0.92",
  "group_id": 1
}
```

### 3.4. 3D Cuboid (`type: "cuboid"`)
Encodes exactly **8 vertices (16 float coordinates)** in the CVAT `cuboidFrom4Points` format:
$$\text{points} = [x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8]$$

- Points $1 \dots 4$: Front face corners (clockwise)
- Points $5 \dots 8$: Rear face corners (projected by depth vector)

```json
{
  "type": "cuboid",
  "label": "truck",
  "points": [
    200.0, 300.0, 200.0, 500.0, 450.0, 500.0, 450.0, 300.0,
    490.0, 280.0, 490.0, 480.0, 240.0, 280.0, 240.0, 480.0
  ],
  "confidence": "0.85"
}
```

### 3.5. Skeleton (`type: "skeleton"`)
A parent container containing individual keypoints as child `points` elements:

```json
{
  "type": "skeleton",
  "label": "person",
  "confidence": 0.89,
  "elements": [
    {
      "type": "points",
      "label": "1",
      "points": [640.5, 210.2],
      "outside": false,
      "occluded": false
    },
    {
      "type": "points",
      "label": "2",
      "points": [655.0, 198.0],
      "outside": false,
      "occluded": false
    },
    {
      "type": "points",
      "label": "10",
      "points": [720.0, 450.0],
      "outside": false,
      "occluded": true
    }
  ]
}
```

---

## 4. Visibility & Native Occlusion Contract

CVAT 2.75.1 supports native occlusion rendering via the `.cvat_canvas_shape_occluded` CSS class (renders dashed connecting bone lines with `stroke-dasharray: 5`).

### 4.1. The 3-State Visibility Table

| 9Router Flag | Meaning | CVAT `outside` | CVAT `occluded` | Canvas Rendering Behavior |
|:---:|---|:---:|:---:|---|
| **`2`** | Clearly Visible | `False` | `False` | Solid point, solid incident bone edges |
| **`1`** | Occluded / Covered | `False` | `True` | Solid point, **dashed** incident edges (`stroke-dasharray: 5`) |
| **`0`** | Outside Frame | `True` | `False` | Hidden / bypassed point, incident bone edges omitted |

### 4.2. Invariant Rules
1. **Forbidden Combination**: `outside = True` AND `occluded = True` is **STRICTLY FORBIDDEN** (violates CVAT schema validation).
2. **Outside Coordinates**: Points marked `outside: true` should have coordinates set to `[0.0, 0.0]`.
3. **NO Custom Attributes**: Do NOT inject attributes like `{"name": "dashed", "value": "true"}` or `{"name": "occluded_edge", "value": "1"}`. CVAT rejects unregistered attributes with `400 Bad Request`.

---

## 5. Laterality Standard: Viewer Perspective

For Human Pose 17 and Facial Landmarks:
- **Standard Medical Convention (Subject Perspective)**: Left/Right refers to the anatomical side of the subject (screen inverted for front-facing person).
- **VinFast / Tool-cvat Standard (Viewer Perspective)**: Left/Right refers strictly to the **display screen as seen by the annotator**:
  * **Right (`R_*`, `*phai`)**: Image right ($x$ coordinate larger).
  * **Left (`L_*`, `*trai`)**: Image left ($x$ coordinate smaller).

### Human Pose 17 Keypoint Index Table
| ID | Semantic Label | Viewer Side | Laterality Invariant |
|:---:|---|:---:|---|
| **1** | `nose` | Center | Anchor point |
| **2** | `right_eye` | **Right** | $x_2 > x_3$ |
| **3** | `left_eye` | **Left** | $x_3 < x_2$ |
| **4** | `right_ear` | **Right** | $x_4 > x_5$ |
| **5** | `left_ear` | **Left** | $x_5 < x_4$ |
| **6** | `right_shoulder`| **Right** | $x_6 > x_7$ |
| **7** | `left_shoulder` | **Left** | $x_7 < x_6$ |
| **8** | `right_elbow` | **Right** | $x_8 > x_9$ |
| **9** | `left_elbow` | **Left** | $x_9 < x_8$ |
| **10**| `right_wrist` | **Right** | $x_{10} > x_{11}$ |
| **11**| `left_wrist` | **Left** | $x_{11} < x_{10}$ |
| **12**| `right_hip` | **Right** | $x_{12} > x_{13}$ |
| **13**| `left_hip` | **Left** | $x_{13} < x_{12}$ |
| **14**| `right_knee` | **Right** | $x_{14} > x_{15}$ |
| **15**| `left_knee` | **Left** | $x_{15} < x_{14}$ |
| **16**| `right_ankle` | **Right** | $x_{16} > x_{17}$ |
| **17**| `left_ankle` | **Left** | $x_{17} < x_{16}$ |

**Rule of Thumb**: Even IDs (2, 4, 6, 8, 10, 12, 14, 16) = Viewer Right; Odd IDs (3, 5, 7, 9, 11, 13, 15, 17) = Viewer Left.

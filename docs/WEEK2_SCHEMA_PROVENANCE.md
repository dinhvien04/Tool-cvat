# Week-2 AI Annotation Schema Provenance & Architectural Contract

**Version:** 1.0  
**Date:** September 2026  
**Status:** Approved & Canonical  
**Target Detectors:** `ninerouter-human-pose-17` & `ninerouter-face-vf50`  

---

## 1. Executive Summary

This document establishes the authoritative schema provenance, coordinate conventions, kinematic topologies, and CVAT 2.75.1 integration contracts for the two Week-2 AI annotation detectors:
1. **9Router Human Pose 17** (`ninerouter-human-pose-17`): Driver cabin human pose estimation with 17 keypoints.
2. **9Router Face Landmark VF-50** (`ninerouter-face-vf50`): VinFast 50 facial landmarks structured across exactly 7 component skeletons.

This specification resolves historical schema duplication, eliminates legacy monolithic `"face"` skeleton remnants, formalizes the VinFast viewer-perspective laterality standard, and documents the root cause and end-to-end fix for the CVAT skeleton mapping error observed in Task 13.

---

## 2. Canonical Sources of Truth

The single authoritative source of truth for each detector is maintained under `config/`:

| Detector | Schema File | Authoritative Source Document | Keypoint / Landmark Count | Topology Structure |
|---|---|---|---|---|
| **Human Pose 17** | `config/week2_pose17.yaml` | `Week2_Guideline_HumanPose17_HocVien_v1.1.docx` | 17 keypoints | 1 parent skeleton (`person`), 16 edges |
| **Face Landmark VF-50** | `config/week2_vf50.yaml` | `Week2_Guideline_Face_Landmark_VF50_HocVien_v1.3.docx` | 50 landmarks | 7 component skeletons, 47 edges |

Any changes to labels, keypoints, topologies, or bounding conditions must be authored in these YAML files. All downstream artifacts (Nuclio `function.yaml` files, test fixtures, CVAT specs, prompt builders) derive directly from these schemas.

---

## 3. VinFast Laterality Standard: Viewer Perspective

A primary source of historical confusion and quality-gate rejection was the laterality definition:
- **COCO / Standard Medical Convention (Subject Perspective):** Anatomical right is on the person's right side (screen left for a front-facing person, $x_{\text{right}} < x_{\text{left}}$).
- **VinFast Week-2 Standard (Viewer Perspective / Display Image View):** Keypoint labels are defined strictly from the perspective of the **annotator viewing the display screen**:
  * **Right (`R_*`, `*phai`)** designates the right side of the image frame ($x_{\text{right}} > x_{\text{left}}$).
  * **Left (`L_*`, `*trai`)** designates the left side of the image frame ($x_{\text{left}} < x_{\text{right}}$).

### 3.1 Human Pose 17 Laterality Table

In `config/week2_pose17.yaml` and `core/pose_face_schema.py`:

| ID | Name | Semantic Label | Viewer Side | Laterality Invariant |
|---|---|---|---|---|
| **1** | `1` | `nose` | Center | Anchor point |
| **2** | `2` | `right_eye` | **Right** | $x_2 > x_3$ (for frontal view) |
| **3** | `3` | `left_eye` | **Left** | $x_3 < x_2$ |
| **4** | `4` | `right_ear` | **Right** | $x_4 > x_5$ |
| **5** | `5` | `left_ear` | **Left** | $x_5 < x_4$ |
| **6** | `6` | `right_shoulder` | **Right** | $x_6 > x_7$ |
| **7** | `7` | `left_shoulder` | **Left** | $x_7 < x_6$ |
| **8** | `8` | `right_elbow` | **Right** | $x_8 > x_9$ |
| **9** | `9` | `left_elbow` | **Left** | $x_9 < x_8$ |
| **10** | `10` | `right_wrist` | **Right** | $x_{10} > x_{11}$ |
| **11** | `11` | `left_wrist` | **Left** | $x_{11} < x_{10}$ |
| **12** | `12` | `right_hip` | **Right** | $x_{12} > x_{13}$ |
| **13** | `13` | `left_hip` | **Left** | $x_{13} < x_{12}$ |
| **14** | `14` | `right_knee` | **Right** | $x_{14} > x_{15}$ |
| **15** | `15` | `left_knee` | **Left** | $x_{15} < x_{14}$ |
| **16** | `16` | `right_ankle` | **Right** | $x_{16} > x_{17}$ |
| **17** | `17` | `left_ankle` | **Left** | $x_{17} < x_{16}$ |

**Rule:** Even IDs (2, 4, 6, 8, 10, 12, 14, 16) represent Viewer Right ($x$ larger). Odd IDs (3, 5, 7, 9, 11, 13, 15, 17) represent Viewer Left ($x$ smaller).

---

## 4. VinFast VF-50: 7 Component Skeletons Topology

VinFast Facial Landmark VF-50 does **NOT** use a single monolithic `"face"` skeleton. Per Guideline Section 5.3 and Appendix 10, it consists of **exactly 7 independent component skeleton labels** totaling 50 landmarks and 47 edges:

```
VF-50 Topology Structure:
├── longmaytrai (5 points: 0..4, open contour, 4 edges: (0,1), (1,2), (2,3), (3,4)) [Display Left]
├── longmayphai (5 points: 5..9, open contour, 4 edges: (5,6), (6,7), (7,8), (8,9)) [Display Right]
├── songmui     (4 points: 10..13, open contour, 3 edges: (10,11), (11,12), (12,13)) [Nose Bridge]
├── mattrai     (8 points: 14..21, closed contour, 8 edges: (14,15)..(20,21), (21,14)) [Display Left Eye]
├── matphai     (8 points: 22..29, closed contour, 8 edges: (22,23)..(28,29), (29,22)) [Display Right Eye]
├── moingoai    (12 points: 30..41, closed contour, 12 edges: (30,31)..(40,41), (41,30)) [Outer Lips]
└── moitrong    (8 points: 42..49, closed contour, 8 edges: (42,43)..(48,49), (49,42)) [Inner Lips]
Total: 50 points (0..49 contiguous), 47 edges
```

### 4.1 Critical Anatomical Notes
1. **Point 13 (`songmui_13`):** Represents the **base of the nose bridge** (just above the nostrils), **NOT** the nose tip.
2. **`moitrong` Containment:** The inner lip contour (42..49) must remain strictly geometrically inside the outer lip contour (30..41).
3. **Multi-Face Grouping:** When multiple faces appear in a frame, all 7 component skeletons belonging to the same person are grouped using the integer property `group_id` / `group`:
   * Face 1: `group_id = 1, group = 1` across all 7 skeletons (`longmaytrai`, ..., `moitrong`).
   * Face 2: `group_id = 2, group = 2` across all 7 skeletons.
4. **No Unified "Face" Skeleton:** There is no parent bounding box or parent skeleton named `"face"` in VF-50.

---

## 5. CVAT 2.75.1 Skeleton Mapping Contract & Root Cause Analysis

### 5.1 CVAT Backend Validation Logic
In `/opt/cvat/cvat/apps/lambda_manager/views.py` (lines 446–450):
```python
if md_label["type"] == "skeleton" and db_label.type == "skeleton":
    if "sublabels" not in mapping_item:
        raise ValidationError(
            f'Mapping for elements was not specified in skeleton "{model_label_name}" '
        )
    validate_labels_mapping(
        mapping_item["sublabels"], md_label["sublabels"], db_label.sublabels.all()
    )
```

### 5.2 Root Cause of the Live Error on Task 13
In `/tasks/13/jobs/13`, invoking Face Landmark VF-50 threw:
`Detection error occurred: Mapping for elements was not specified in skeleton "face"`

**Root Cause Chain:**
1. The deployed container `nuclio-nuclio-ninerouter-face-vf50` was running a stale build that exposed a single skeleton `face` with 50 sublabels (`longmaytrai_00` .. `moitrong_49`).
2. Task 13 (Project 6 "Day4") defines a 5-point skeleton `face` (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`).
3. In the CVAT UI, when the user clicked "Annotate", CVAT's frontend (`labels-mapper.tsx`) auto-matched the parent label `face` to `face`.
4. However, because none of the 50 model sublabels matched the 5 task sublabels, `subMapping` was empty (`[]`).
5. CVAT's `convertMappingToServer()` omits the `"sublabels"` key when `subMapping` is empty:
   `...(subMapping.length ? { sublabels: convertMappingToServer(subMapping) } : {})`
6. The resulting wire mapping payload was `{"face": {"name": "face", "attributes": {}}}` without `"sublabels"`.
7. CVAT backend checked `if "sublabels" not in mapping_item:` and raised `ValidationError(f'Mapping for elements was not specified in skeleton "face" ')`.

### 5.3 Architectural Fix
1. **Canonical 7-Component Specification:** Redeploy `ninerouter-face-vf50` with the authoritative 7 component skeletons (`longmaytrai`, `longmayphai`, `songmui`, `mattrai`, `matphai`, `moingoai`, `moitrong`). The stale unified `"face"` skeleton is completely eradicated.
2. **Preflight Validator (`scripts/week2_mapping_preflight.py`):** Inspects task labels vs. model specs, checks sublabel compatibility, and validates mapping dictionaries before sending API calls.
3. **5-Point Face Compatibility Subset:** When annotating a task with a 5-point face skeleton, `scripts/week2_mapping_preflight.py` provides the canonical keypoint mapping:
   * `mattrai_18` (inner corner) $\rightarrow$ `left_eye`
   * `matphai_26` (inner corner) $\rightarrow$ `right_eye`
   * `songmui_13` (bridge base) $\rightarrow$ `nose`
   * `moingoai_30` (left corner) $\rightarrow$ `mouth_left`
   * `moingoai_36` (right corner) $\rightarrow$ `mouth_right`

---

## 6. Pose 17 Accuracy: Adaptive Two-Pass Refinement

### 6.1 Cabin Pose Inaccuracy in Task 13
In `/tasks/13/jobs/13`, Human Pose 17 produced severe geometric anomalies: `RIGHT_WRIST` and `RIGHT_ELBOW` floating high in the background above the person.

**Root Causes:**
1. **Resolution Dilution:** Full cabin images downscaled to 1280px leave arms and wrists spanning only 15–30px, confusing background cabin trim (B-pillars, grab handles) with limb joints.
2. **Missing Kinematic Continuity:** Prompt lacked strict kinematic chain constraints (`shoulder` $\rightarrow$ `elbow` $\rightarrow$ `wrist`).
3. **Quality Gate Bypassed:** `core/quality_gate.py` was not wired in `model_handler.py`.

### 6.2 Adaptive Refinement Solution
1. **Kinematic Prompting:** `build_pose17_prompt()` requests person bounding boxes (`box_2d`) and enforces anatomical bone length limits.
2. **Quality Gate Assessment:** `assess_pose17_quality()` checks bone lengths, arm ratios, and joint detachment with `laterality_convention=LATERALITY_VIEWER`.
3. **Two-Pass Refinement:** If quality gate flags floating joints or impossible bone lengths:
   * Expand the person bounding box by a 15–20% margin.
   * Crop high-resolution patch directly from the unscaled original image.
   * Run targeted refinement on the cropped person patch.
   * Project coordinates back to original frame:
     $$x_{\text{orig}} = \text{clamp}\left(x_{\text{min}} + \frac{x_{\text{crop}}}{1000.0} \times (x_{\text{max}} - x_{\text{min}}), 0, W\right)$$
     $$y_{\text{orig}} = \text{clamp}\left(y_{\text{min}} + \frac{y_{\text{crop}}}{1000.0} \times (y_{\text{max}} - y_{\text{min}}), 0, H\right)$$
   * Select candidate with highest quality score.

---

## 7. VF-50 Accuracy: Two-Pass Refinement & Zero Silent Drop Guarantee

1. **Two-Pass Face Crop:** For small faces ($< 100\text{px}$) or faces with anomalies, a high-resolution face crop with 20% margin is passed to 9Router.
2. **Zero Silent Drop Guarantee:** When quality assessment detects minor anomalies on difficult poses or low-resolution faces, `faces_to_cvat_skeletons()` activates a salvageable candidate fallback:
   * Pure point collapses ($< 15$ diagonal units) are still rejected.
   * Valid detected faces with minor discretizations are salvaged and emitted, ensuring annotators never receive an unexpected 0-shape result.

---

## 8. Data Safety & Disposable Test Harness Policy

Per strict Lead Engineer safety directives:
1. **Protected Production Tasks & Jobs:**
   * Jobs `12`, `13`, `15` and Tasks `12`, `13`, `15` are **STRICTLY PROTECTED**.
   * Any write or delete attempt against these IDs raises `DataSafetyViolationError`.
2. **Safe Annotation Harness (`core/cvat_safety.py`):**
   * Snapshots pre-existing annotations before tests.
   * Tracks test-created shape IDs explicitly.
   * Clean-up deletes **ONLY** test-created IDs, leaving all pre-existing human annotations intact.
   * Verified by `tests/test_cvat_data_safety_harness.py` with 100-annotation safety guarantee.
3. **No Volume Deletions:** `docker compose down -v` is strictly forbidden.

# CVAT 2.75.1 Native Occlusion Contract & Rendering Mechanics

**Scope**: VinFast Week-2 HumanPose-17 (`ninerouter-human-pose-17`) and VF-50 Facial Landmarks (`ninerouter-face-vf50`).  
**CVAT Target**: CVAT 2.75.1 (API v2).

---

## 1. Executive Summary

In CVAT 2.75.1, occlusion for skeleton elements is a first-class native concept. This document defines the formal contract governing visibility and occlusion states across the 9Router vision pipeline, the serverless detector handlers, and CVAT's native canvas rendering engine.

**Key Rule**: All occlusion states must be communicated through CVAT's standard boolean attributes (`occluded` and `outside`). Under no circumstances should custom attributes (such as `dashed=true`), custom inline CSS, or CVAT frontend modifications be introduced to achieve dashed rendering.

---

## 2. Canonical Visibility Mapping Contract

The 9Router vision model produces a 3-level discrete visibility flag for each keypoint or facial landmark, conforming to standard COCO and VinFast guidelines. The serverless detector normalizes inputs and translates these flags deterministically into CVAT's native boolean representation using `core.week2_schema.normalize_visibility` and `core.week2_schema.map_visibility_to_cvat`:

| 9Router Visibility Flag | Semantic Meaning | CVAT `outside` | CVAT `occluded` | Canvas Rendering Behavior |
| :---: | :--- | :---: | :---: | :--- |
| **`2`** | **Clearly Visible** | `False` | `False` | Solid point, solid incident skeleton bone edges |
| **`1`** | **Occluded / Covered** | `False` | `True` | Solid point with occluded marker, **dashed** incident edges (`stroke-dasharray: 5`) |
| **`0`** | **Outside Frame / Untracked** | `True` | `False` | Hidden / bypassed point, incident skeleton bone edges omitted |

### Visibility Normalization Specification (`normalize_visibility`)

To guarantee system resilience across diverse vision backends and manual CSV/JSON inputs:

- **Authoritative Canonical Values**: `0`, `1`, `2`.
- **Accepted String Aliases**: `"0"`, `"1"`, `"2"`, `"outside"`, `"occluded"`, `"visible"` (case-insensitive, trimmed).
- **Strict Mode (`strict=True`)**:
  - Validates strictly against canonical values and accepted string aliases.
  - Any out-of-range integer (`-1`, `3`, `100`), non-finite float (`NaN`, `Inf`), ambiguous string (`"banana"`, `"uncertain"`, `"partial"`), boolean (`True`, `False`), or `None` immediately raises `ValueError`.
- **Non-Strict Mode (`strict=False`)**:
  - Deterministic safe resolution with configurable fallback `default` (defaults to `2` for production continuity, or `0` for defensive fail-closed pipelines).
  - Tolerates boolean values (`True -> 2`, `False -> 0`).
  - Resolves legacy aliases (`"absent" -> 0`, `"false" -> 0`, `"partial" -> 1`, `"true" -> 2`).
  - Unparseable strings, invalid types, and out-of-range numerics safely resolve to `default`.

### Invariant Equations

```python
# Forward mapping (9Router / Model -> CVAT)
outside = (vis == 0)
occluded = (vis == 1)

# Reverse mapping (CVAT -> Model / Exporter)
# Invariant: outside=True and occluded=True is forbidden (raises ValueError in strict mode)
if outside:
    vis = 0
elif occluded:
    vis = 1
else:
    vis = 2
```

---

## 3. Native Canvas Rendering Mechanics

CVAT 2.75.1's frontend canvas SVG engine natively handles occluded points and skeleton lines without custom code:

1. **SVG Class Application**: When a child point element in a skeleton has `occluded: true`, CVAT adds the SVG class `.cvat_canvas_shape_occluded` to the rendered node element.
2. **Incident Edge Styling**: Incident connecting lines defined in the skeleton SVG topology inherit or calculate styling from incident vertices. In CVAT Canvas, occluded edges render with:
   ```css
   .cvat_canvas_shape_occluded {
       stroke-dasharray: 5;
   }
   ```
3. **Point Color and Fill**: Occluded points render with distinct visual feedback (e.g., striped or dimmed fill) in the standard CVAT theme.

---

## 4. Strict Prohibitions & Anti-Patterns

To ensure system reliability, backwards compatibility, and clean CVAT annotations:

1. **NO Custom Shape Attributes**:
   - **Forbidden**: `{"attributes": [{"name": "dashed", "value": "true"}]}`
   - **Forbidden**: `{"attributes": [{"name": "occluded_edge", "value": "1"}]}`
   - *Reason*: CVAT validates labels against task label specifications. Unregistered attributes will either fail CVAT validation (`400 Bad Request`) or pollute user exports.

2. **NO Custom SVG Injection in Detection Responses**:
   - The detector response must return standard CVAT skeleton shapes containing child points with `outside` and `occluded` booleans.
   - SVG topology is defined **only** once in `function.yaml` spec for the detector label; never in runtime detection shapes.

3. **NO CVAT Frontend Source Modifications**:
   - The CVAT React/TypeScript frontend must not be modified or patched to add dashed line hacks. The native `.cvat_canvas_shape_occluded` CSS handles all required visual cues.

4. **NO Inverted Outside / Occluded Semantics**:
   - An occluded keypoint (e.g. driver's hand behind the steering wheel) is **NOT** outside. Setting `outside: true` for occluded points drops them from the skeleton topology, destroying limb kinematic tracking.
   - Points marked `outside: true` MUST have coordinates set to `[0.0, 0.0]` or omitted, and must have `occluded: false`.

---

## 5. Human Annotator Workflow & Editability

When automated detections from 9Router are loaded into CVAT:

1. **Visual Inspection**: Annotators immediately see dashed lines connecting occluded keypoints.
2. **Native Toggling**: Annotators can press `Q` (or the configured CVAT shortcut) to toggle the `occluded` property on any selected keypoint or line.
3. **Coordinate Adjustment**: Even when occluded, points can be moved, corrected, and saved natively without breaking serverless detector re-runs or quality gate validation.

---

## 6. Implementation Verification

Hermetic verification of this contract is enforced in:
- `core/week2_schema.py::normalize_visibility()`
- `core/week2_schema.py::map_visibility_to_cvat()`
- `core/week2_schema.py::map_cvat_to_visibility()`
- `core/skeleton_contract.py::VF50Landmark.to_cvat_element()`
- `core/skeleton_contract.py::PoseKeypoint.to_cvat_element()`
- `tests/test_skeleton_contract.py::TestVisibilityMapping`
- `tests/test_week2_schema.py::TestVisibilityContract`

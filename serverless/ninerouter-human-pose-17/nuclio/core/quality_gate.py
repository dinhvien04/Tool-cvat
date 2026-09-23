"""Deterministic geometry and laterality quality gate for adaptive refinement and skeleton validation.

The gate does not call any model. It identifies structurally valid but suspicious geometry,
validates anatomical vs. viewer laterality invariants, checks coordinate bounds, ensures
topological nesting, and detects axis swaps (x vs y) for:
1. Master 31-label instance/region/line detections (rectangle_mask, polygon_mask, polyline).
2. Human Pose 17 keypoint skeletons (VinFast Week-2 HumanPose-17 topology, viewer-space laterality).
3. Face Landmark VF-50 (VinFast 7-component face topology, viewer laterality).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union


try:
    from core.week2_schema import (
        load_pose17,
        load_vf50,
        pose17_coco_keypoints,
        pose17_edges_as_coco_names,
        pose17_paired_keypoints,
        pose17_rigid_paired_keypoints,
        pose17_articulated_paired_keypoints,
        pose17_laterality_convention,
        vf50_component_point_counts,
        vf50_eyelid_opposing_pairs,
    )
    _HAS_WEEK2_SCHEMA = True
except ImportError:
    _HAS_WEEK2_SCHEMA = False


# ==============================================================================
# 1. CONSTANTS & CANONICAL SCHEMAS (Derived from canonical YAML loaders)
# ==============================================================================

LATERALITY_SUBJECT: str = "subject"  # Anatomical left/right of the person being viewed
LATERALITY_VIEWER: str = "viewer"    # Image/screen left/right (viewer's left/right)

if _HAS_WEEK2_SCHEMA:
    _pose17 = load_pose17()
    DEFAULT_POSE17_LATERALITY: str = pose17_laterality_convention()

    # Canonical 17 keypoints in indexed VinFast order derived from load_pose17()
    POSE17_KEYPOINTS: Tuple[str, ...] = pose17_coco_keypoints()
    POSE17_KEYPOINTS_SET: FrozenSet[str] = frozenset(POSE17_KEYPOINTS)

    # Symmetric left/right pairs for Pose 17 laterality validation
    POSE17_PAIRED_KEYPOINTS: Tuple[Tuple[str, str], ...] = pose17_paired_keypoints()

    # Rigid axial/head anchors that determine overall body laterality
    # Inversion of rigid anchors for a frontal subject indicates genuine laterality error.
    POSE17_RIGID_PAIRED_KEYPOINTS: Tuple[Tuple[str, str], ...] = pose17_rigid_paired_keypoints()

    # Articulated appendicular limb pairs that articulate in 3D and cross naturally (arms/legs crossing)
    # Crossing of distal limbs is an anatomical soft warning, NOT a fatal invalidation.
    POSE17_ARTICULATED_PAIRED_KEYPOINTS: Tuple[Tuple[str, str], ...] = pose17_articulated_paired_keypoints()

    # VinFast Guideline (output_pose17_guideline.txt) 1..17 index to canonical name mapping
    VINFAST_POSE17_INDEX_TO_NAME: Dict[int, str] = dict(_pose17.id_to_coco_name)

    # Canonical bone connectivity graph for Pose 17
    # Matches config/week2_pose17.yaml authoritative edge set (18 edges including ear-to-shoulder)
    POSE17_SKELETON_EDGES: Tuple[Tuple[str, str], ...] = pose17_edges_as_coco_names()

    _vf50 = load_vf50()

    # VF-50 Canonical Component Names & Point Counts derived from load_vf50()
    # Authoritative VinFast Week-2 specification
    VF50_COMPONENT_NAMES: Tuple[str, ...] = _vf50.component_names
    VF50_COMPONENT_COUNTS: Dict[str, int] = vf50_component_point_counts()
    VF50_EXPECTED_TOTAL_POINTS: int = _vf50.total_points

    # Index ranges for components in continuous 0..49 indexing: [start, end + 1)
    VF50_COMPONENT_RANGES: Dict[str, Tuple[int, int]] = {
        c.name: (c.start_id, c.end_id + 1) for c in _vf50.components
    }

    # Opposing eyelid pairs (upper eyelid index, lower eyelid index) for VF-50
    # Rule: y(upper) <= y(lower) (remember y increases downwards)
    VF50_EYELID_OPPOSING_PAIRS: Tuple[Tuple[int, int], ...] = vf50_eyelid_opposing_pairs()
else:
    # Graceful fallbacks for isolated 3-detector serverless targets where Week 2 schemas are omitted
    DEFAULT_POSE17_LATERALITY = LATERALITY_VIEWER
    POSE17_KEYPOINTS = ()
    POSE17_KEYPOINTS_SET = frozenset()
    POSE17_PAIRED_KEYPOINTS = ()
    POSE17_RIGID_PAIRED_KEYPOINTS = ()
    POSE17_ARTICULATED_PAIRED_KEYPOINTS = ()
    VINFAST_POSE17_INDEX_TO_NAME = {}
    POSE17_SKELETON_EDGES = ()
    VF50_COMPONENT_NAMES = ()
    VF50_COMPONENT_COUNTS = {}
    VF50_EXPECTED_TOTAL_POINTS = 50
    VF50_COMPONENT_RANGES = {}
    VF50_EYELID_OPPOSING_PAIRS = ()


# ==============================================================================
# 2. QUALITY REPORTS
# ==============================================================================

@dataclass
class QualityReport:
    """Master Quality Report compatible with existing 9Router pipeline."""
    score: float
    needs_refine: bool
    reasons: List[str] = field(default_factory=list)
    suspect_labels: List[str] = field(default_factory=list)
    item_count: int = 0
    suspect_count: int = 0
    hard_errors: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)


@dataclass
class Pose17QualityReport:
    """Rigorous report from deterministic geometry quality gate for Human Pose 17."""
    score: float
    is_valid: bool
    needs_refine: bool
    reasons: List[str] = field(default_factory=list)
    visible_count: int = 0
    active_count: int = 0
    suspect_bones: List[str] = field(default_factory=list)
    laterality_status: str = "ok"
    axis_swap_detected: bool = False
    hard_errors: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)


@dataclass
class VF50QualityReport:
    """Rigorous report from deterministic geometry quality gate for Face Landmark VF-50."""
    score: float
    is_valid: bool
    needs_refine: bool
    reasons: List[str] = field(default_factory=list)
    point_count: int = 0
    component_counts: Dict[str, int] = field(default_factory=dict)
    suspect_components: List[str] = field(default_factory=list)
    laterality_status: str = "ok"
    axis_swap_detected: bool = False
    hard_errors: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)


@dataclass
class LateralityVerificationResult:
    """Detailed verification outcome for laterality invariants."""
    is_valid: bool
    detected_convention: str
    expected_convention: str
    reasons: List[str] = field(default_factory=list)
    delta_x: float = 0.0
    hard_errors: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    crossed_limbs: List[str] = field(default_factory=list)


# ==============================================================================
# 3. BASIC GEOMETRY PRIMITIVES
# ==============================================================================

def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        total += float(p[0]) * float(q[1]) - float(q[0]) * float(p[1])
    return abs(total) * 0.5


def _bbox_from_points(points: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def derive_skeleton_bbox(
    data: Any,
    schema_type: str = "pose17",
    pad_ratio: float = 0.05,
) -> Optional[List[int]]:
    """Derive bounding box [ymin, xmin, ymax, xmax] in normalized [0..1000] from skeleton keypoints.

    Works across Pose 17 and VF-50 generic dicts, lists, or instance objects.
    """
    schema_clean = str(schema_type).strip().lower()
    pts: List[Tuple[float, float]] = []
    if "pose" in schema_clean:
        kp_map = _extract_pose17_points(data)
        pts = [(v[0], v[1]) for v in kp_map.values() if v[2] > 0]
    elif "vf50" in schema_clean or "face" in schema_clean:
        lm_map = _extract_vf50_points(data)
        pts = [(v[0], v[1]) for v in lm_map.values() if v[2] > 0]

    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    return [
        int(max(0.0, min(1000.0, round(ymin - pad_y)))),
        int(max(0.0, min(1000.0, round(xmin - pad_x)))),
        int(max(0.0, min(1000.0, round(ymax + pad_y)))),
        int(max(0.0, min(1000.0, round(xmax + pad_x)))),
    ]


def derive_skeleton_center(
    data: Any,
    schema_type: str = "pose17",
) -> Optional[Tuple[float, float]]:
    """Derive center coordinates (cx, cy) in normalized [0..1000] space from skeleton keypoints."""
    schema_clean = str(schema_type).strip().lower()
    pts: List[Tuple[float, float]] = []
    if "pose" in schema_clean:
        kp_map = _extract_pose17_points(data)
        pts = [(v[0], v[1]) for v in kp_map.values() if v[2] > 0]
    elif "vf50" in schema_clean or "face" in schema_clean:
        lm_map = _extract_vf50_points(data)
        pts = [(v[0], v[1]) for v in lm_map.values() if v[2] > 0]

    if not pts:
        return None

    return (
        round(sum(p[0] for p in pts) / len(pts), 2),
        round(sum(p[1] for p in pts) / len(pts), 2),
    )


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 1e-9 else 0.0


def _orient(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def _segments_intersect(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _self_intersects(points: Sequence[Sequence[float]], *, closed: bool = True) -> bool:
    n = len(points)
    if n < 4:
        return False
    if n > 200:
        points = points[:200]
        n = len(points)
    edge_count = n if closed else n - 1
    for i in range(edge_count):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, edge_count):
            if abs(i - j) <= 1:
                continue
            if closed and i == 0 and j == edge_count - 1:
                continue
            c, d = points[j], points[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _line_path_ratio(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 999.0
    path = sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(points, points[1:])
    )
    chord = math.hypot(
        float(points[-1][0]) - float(points[0][0]),
        float(points[-1][1]) - float(points[0][1]),
    )
    return path / max(chord, 1.0)


# ==============================================================================
# 4. SKELETON PARSING HELPERS (Extract generic dict representations)
# ==============================================================================

def _normalize_pose_name(raw_name: Any) -> str:
    name_str = str(raw_name or "").strip().lower()
    if name_str.isdigit():
        idx = int(name_str)
        if idx in VINFAST_POSE17_INDEX_TO_NAME:
            return VINFAST_POSE17_INDEX_TO_NAME[idx]
    return name_str


def _extract_pose17_points(
    item: Any,
) -> Dict[str, Tuple[float, float, int, float]]:
    """Extract canonical Pose 17 points mapping: name -> (x, y, visibility, confidence)."""
    res: Dict[str, Tuple[float, float, int, float]] = {}

    if hasattr(item, "keypoints"):
        kps = getattr(item, "keypoints")
        if isinstance(kps, dict):
            for k, v in kps.items():
                name = _normalize_pose_name(k)
                if hasattr(v, "x") and hasattr(v, "y"):
                    res[name] = (
                        float(v.x),
                        float(v.y),
                        int(getattr(v, "visibility", 2)),
                        float(getattr(v, "confidence", 1.0)),
                    )
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    vis = int(v[2]) if len(v) > 2 else 2
                    conf = float(v[3]) if len(v) > 3 else 1.0
                    res[name] = (float(v[0]), float(v[1]), vis, conf)
            return res
        elif isinstance(kps, (list, tuple)):
            item = kps

    if isinstance(item, dict):
        elements = item.get("elements") or item.get("keypoints") or item.get("points")
        if isinstance(elements, dict):
            for k, v in elements.items():
                name = _normalize_pose_name(k)
                if hasattr(v, "x") and hasattr(v, "y"):
                    res[name] = (
                        float(v.x),
                        float(v.y),
                        int(getattr(v, "visibility", 2)),
                        float(getattr(v, "confidence", 1.0)),
                    )
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    vis = int(v[2]) if len(v) > 2 else 2
                    conf = float(v[3]) if len(v) > 3 else 1.0
                    res[name] = (float(v[0]), float(v[1]), vis, conf)
                elif isinstance(v, dict):
                    pt = v.get("point") or [v.get("x", 0.0), v.get("y", 0.0)]
                    vis = int(v.get("visibility", 2))
                    conf = float(v.get("confidence", 1.0))
                    res[name] = (float(pt[0]), float(pt[1]), vis, conf)
            if res:
                return res
        elif isinstance(elements, list):
            for el in elements:
                if isinstance(el, dict):
                    name = _normalize_pose_name(el.get("label") or el.get("name"))
                    pt = el.get("points") or el.get("point") or [el.get("x", 0.0), el.get("y", 0.0)]
                    outside = bool(el.get("outside", False))
                    occluded = bool(el.get("occluded", False))
                    vis = 0 if outside else (1 if occluded else 2)
                    conf = float(el.get("confidence", 1.0))
                    if len(pt) >= 2 and name:
                        res[name] = (float(pt[0]), float(pt[1]), vis, conf)
            if res:
                return res
        for k, v in item.items():
            name = _normalize_pose_name(k)
            if name in POSE17_KEYPOINTS_SET:
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    vis = int(v[2]) if len(v) > 2 else 2
                    conf = float(v[3]) if len(v) > 3 else 1.0
                    res[name] = (float(v[0]), float(v[1]), vis, conf)
                elif isinstance(v, dict):
                    pt = v.get("point") or [v.get("x", 0.0), v.get("y", 0.0)]
                    vis = int(v.get("visibility", 2))
                    conf = float(v.get("confidence", 1.0))
                    res[name] = (float(pt[0]), float(pt[1]), vis, conf)
        return res

    if isinstance(item, (list, tuple)):
        for idx, el in enumerate(item):
            if isinstance(el, dict):
                raw_name = el.get("label") or el.get("name") or (POSE17_KEYPOINTS[idx] if idx < 17 else "")
                name = _normalize_pose_name(raw_name)
                pt = el.get("points") or el.get("point") or [el.get("x", 0.0), el.get("y", 0.0)]
                outside = bool(el.get("outside", False))
                occluded = bool(el.get("occluded", False))
                vis = 0 if outside else (1 if occluded else 2)
                conf = float(el.get("confidence", 1.0))
                if len(pt) >= 2 and name:
                    res[name] = (float(pt[0]), float(pt[1]), vis, conf)
            elif isinstance(el, (list, tuple)) and len(el) >= 2 and idx < len(POSE17_KEYPOINTS):
                name = POSE17_KEYPOINTS[idx]
                vis = int(el[2]) if len(el) > 2 else 2
                conf = float(el[3]) if len(el) > 3 else 1.0
                res[name] = (float(el[0]), float(el[1]), vis, conf)

    return res


def _extract_vf50_points(
    item: Any,
) -> Dict[int, Tuple[float, float, int, float]]:
    """Extract canonical VF-50 points mapping: point_id (0..49) -> (x, y, visibility, confidence)."""
    res: Dict[int, Tuple[float, float, int, float]] = {}

    if hasattr(item, "landmarks"):
        item = getattr(item, "landmarks")

    if isinstance(item, list):
        # Could be list of 7 skeletons, list of elements, or direct list of 50 tuples
        if item and all(isinstance(x, dict) and ("elements" in x or "label" in x) for x in item):
            for skel in item:
                skel_name = str(skel.get("label") or skel.get("name") or "").strip().lower()
                elements = skel.get("elements", [])
                base_idx = VF50_COMPONENT_RANGES.get(skel_name, (0, 0))[0]
                for idx, el in enumerate(elements):
                    pt = el.get("points") or el.get("point") or [el.get("x", 0.0), el.get("y", 0.0)]
                    raw_id = el.get("label") or el.get("name") or el.get("id")
                    target_id = None
                    try:
                        target_id = int(str(raw_id).strip())
                    except (ValueError, TypeError):
                        target_id = base_idx + idx
                    outside = bool(el.get("outside", False))
                    occluded = bool(el.get("occluded", False))
                    vis = 0 if outside else (1 if occluded else 2)
                    conf = float(el.get("confidence", 1.0))
                    if 0 <= target_id < VF50_EXPECTED_TOTAL_POINTS and len(pt) >= 2:
                        res[target_id] = (float(pt[0]), float(pt[1]), vis, conf)
            return res

        if len(item) == VF50_EXPECTED_TOTAL_POINTS and all(isinstance(x, (list, tuple)) for x in item):
            for idx, pt in enumerate(item):
                vis = int(pt[2]) if len(pt) > 2 else 2
                conf = float(pt[3]) if len(pt) > 3 else 1.0
                res[idx] = (float(pt[0]), float(pt[1]), vis, conf)
            return res

    if isinstance(item, dict):
        if "landmarks" in item or "points" in item:
            item = item.get("landmarks") or item.get("points")
            return _extract_vf50_points(item)

        # Check for 7 skeleton keys
        has_component_keys = any(k in VF50_COMPONENT_COUNTS for k in item.keys())
        if has_component_keys:
            for comp_name, pts in item.items():
                start_idx, end_idx = VF50_COMPONENT_RANGES.get(comp_name, (0, 0))
                if isinstance(pts, list):
                    for idx, pt_item in enumerate(pts):
                        p_id = start_idx + idx
                        if p_id >= end_idx:
                            break
                        if isinstance(pt_item, dict):
                            coords = pt_item.get("points") or pt_item.get("point") or [pt_item.get("x", 0), pt_item.get("y", 0)]
                            vis = int(pt_item.get("visibility", 2))
                            conf = float(pt_item.get("confidence", 1.0))
                            res[p_id] = (float(coords[0]), float(coords[1]), vis, conf)
                        elif isinstance(pt_item, (list, tuple)) and len(pt_item) >= 2:
                            vis = int(pt_item[2]) if len(pt_item) > 2 else 2
                            conf = float(pt_item[3]) if len(pt_item) > 3 else 1.0
                            res[p_id] = (float(pt_item[0]), float(pt_item[1]), vis, conf)
            return res

        # Dict mapping point IDs (e.g. 0..49 or "0".."49")
        for k, v in item.items():
            try:
                p_id = int(str(k).strip())
            except ValueError:
                continue
            if 0 <= p_id < VF50_EXPECTED_TOTAL_POINTS:
                if hasattr(v, "x") and hasattr(v, "y"):
                    vis = int(getattr(v, "visibility", 2))
                    conf = float(getattr(v, "confidence", 1.0))
                    res[p_id] = (float(v.x), float(v.y), vis, conf)
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    vis = int(v[2]) if len(v) > 2 else 2
                    conf = float(v[3]) if len(v) > 3 else 1.0
                    res[p_id] = (float(v[0]), float(v[1]), vis, conf)
                elif isinstance(v, dict):
                    coords = v.get("points") or v.get("point") or [v.get("x", 0), v.get("y", 0)]
                    vis = int(v.get("visibility", 2))
                    conf = float(v.get("confidence", 1.0))
                    res[p_id] = (float(coords[0]), float(coords[1]), vis, conf)

    return res


# ==============================================================================
# 5. COORDINATE AXIS SWAP (X VS Y) DETECTOR
# ==============================================================================

def detect_coordinate_axis_swap(
    data: Any,
    schema_type: str = "pose17",
) -> Tuple[bool, str]:
    """Deterministically detect if model emitted swapped (y, x) instead of (x, y).

    Mathematical invariant:
    - In upright Human Pose 17:
      * Body axis (ankles_mid - shoulders_mid) is predominantly vertical: dy >> dx.
      * Shoulder span (left_shoulder - right_shoulder) is predominantly horizontal: dx >> dy.
      * Under an axis swap, body axis becomes horizontal and shoulder span becomes vertical!
    - In upright Face Landmark VF-50:
      * Inter-ocular axis (right_eye_center - left_eye_center) is predominantly horizontal: dx >> dy.
      * Facial midline axis (mouth_center - nose_top) is predominantly vertical: dy >> dx.
      * Under an axis swap, eye span becomes vertical and midline becomes horizontal!
    """
    schema_clean = str(schema_type).strip().lower()

    if "pose" in schema_clean:
        pts = _extract_pose17_points(data)
        ls = pts.get("left_shoulder")
        rs = pts.get("right_shoulder")
        lh = pts.get("left_hip")
        rh = pts.get("right_hip")
        if not (ls and rs and lh and rh):
            return False, "insufficient_points_for_pose_axis_check"

        s_mid = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
        h_mid = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)

        torso_dx = abs(h_mid[0] - s_mid[0])
        torso_dy = abs(h_mid[1] - s_mid[1])
        shoulder_dx = abs(ls[0] - rs[0])
        shoulder_dy = abs(ls[1] - rs[1])

        # If torso is horizontal and shoulders are vertical, check secondary anchors
        if torso_dx > 1.35 * torso_dy and shoulder_dy > 1.35 * shoulder_dx:
            # Check secondary head/eye anchor to avoid false positives on reclining/lying drivers
            le = pts.get("left_eye")
            re = pts.get("right_eye")
            if le and re and le[2] > 0 and re[2] > 0:
                eye_dx = abs(le[0] - re[0])
                eye_dy = abs(le[1] - re[1])
                # If eyes are horizontal, the head is horizontal - this is a person reclining/lying down!
                if eye_dx > 1.25 * eye_dy:
                    return False, "reclining_or_lying_pose: eyes remain horizontal"
            return True, "coordinate_axis_swap_detected: torso is horizontal while shoulder line is vertical"
        return False, "pose_axes_normal"

    elif "vf50" in schema_clean or "face" in schema_clean:
        pts_map = _extract_vf50_points(data)
        left_eye_pts = [pts_map[i] for i in range(14, 22) if i in pts_map]
        right_eye_pts = [pts_map[i] for i in range(22, 30) if i in pts_map]
        nose_pts = [pts_map[i] for i in range(10, 14) if i in pts_map]
        mouth_pts = [pts_map[i] for i in range(30, 42) if i in pts_map]

        if not (left_eye_pts and right_eye_pts and nose_pts and mouth_pts):
            return False, "insufficient_landmarks_for_face_axis_check"

        c_left_eye = (
            sum(p[0] for p in left_eye_pts) / len(left_eye_pts),
            sum(p[1] for p in left_eye_pts) / len(left_eye_pts),
        )
        c_right_eye = (
            sum(p[0] for p in right_eye_pts) / len(right_eye_pts),
            sum(p[1] for p in right_eye_pts) / len(right_eye_pts),
        )
        c_nose_top = pts_map.get(10, (sum(p[0] for p in nose_pts)/len(nose_pts), sum(p[1] for p in nose_pts)/len(nose_pts)))
        c_mouth = (
            sum(p[0] for p in mouth_pts) / len(mouth_pts),
            sum(p[1] for p in mouth_pts) / len(mouth_pts),
        )

        eyes_dx = abs(c_right_eye[0] - c_left_eye[0])
        eyes_dy = abs(c_right_eye[1] - c_left_eye[1])
        midline_dx = abs(c_mouth[0] - c_nose_top[0])
        midline_dy = abs(c_mouth[1] - c_nose_top[1])

        # If eyes are vertical and midline is horizontal, axes are swapped
        if eyes_dy > 1.35 * eyes_dx and midline_dx > 1.35 * midline_dy:
            return True, "coordinate_axis_swap_detected: eyes vertically aligned and facial midline horizontal"
        return False, "face_axes_normal"

    return False, "unknown_schema_type"


# ==============================================================================
# 6. LATERALITY SPECIFICATION & PROOF SYSTEM
# ==============================================================================

def verify_laterality_convention(
    data: Any,
    schema_type: str = "pose17",
    expected_convention: Optional[str] = None,
) -> LateralityVerificationResult:
    """Analyze and prove laterality convention (subject vs. viewer).

    Conventions:
    - subject: anatomical perspective. For a frontal person, subject's left is on screen right (x_left > x_right).
    - viewer: screen/image perspective. Viewer's left is on screen left (x_left < x_right).

    Returns:
        LateralityVerificationResult with detected convention, validity, and delta_x.
    """
    schema_clean = str(schema_type).strip().lower()

    if "pose" in schema_clean:
        target_conv = expected_convention or DEFAULT_POSE17_LATERALITY
        pts = _extract_pose17_points(data)
        ls = pts.get("left_shoulder")
        rs = pts.get("right_shoulder")
        nose = pts.get("nose")

        if not (ls and rs):
            return LateralityVerificationResult(
                is_valid=False,
                detected_convention="unknown",
                expected_convention=target_conv,
                reasons=["missing_shoulders_for_laterality_check"],
            )

        # delta_x = x_left - x_right
        delta_x = ls[0] - rs[0]
        detected = "subject" if delta_x > 0 else "viewer"

        # Check front-facing subject: nose between shoulders horizontally
        is_frontal = False
        if nose:
            min_sx, max_sx = min(ls[0], rs[0]), max(ls[0], rs[0])
            if min_sx - 20.0 <= nose[0] <= max_sx + 20.0:
                is_frontal = True

        reasons: List[str] = []
        hard_errors: List[str] = []
        soft_warnings: List[str] = []
        crossed_limbs: List[str] = []
        is_valid = True
        if target_conv == LATERALITY_SUBJECT:
            if is_frontal and delta_x < 0:
                is_valid = False
                msg = "laterality_inversion: left_shoulder has smaller x than right_shoulder for frontal subject"
                reasons.append(msg)
                hard_errors.append(msg)
        elif target_conv == LATERALITY_VIEWER:
            if delta_x > 0:
                is_valid = False
                msg = "laterality_inversion: viewer left has larger x than viewer right"
                reasons.append(msg)
                hard_errors.append(msg)

        # 1. Verify rigid axial/head pairs (eyes, ears, shoulders, hips)
        rigid_inversions: List[str] = []
        for left_k, right_k in POSE17_RIGID_PAIRED_KEYPOINTS:
            lp = pts.get(left_k)
            rp = pts.get(right_k)
            if lp and rp and lp[2] > 0 and rp[2] > 0:
                if target_conv == LATERALITY_VIEWER:
                    # In viewer convention, left_k (odd) must have smaller x than right_k (even)
                    if lp[0] > rp[0] + 5.0:
                        rigid_inversions.append(f"{left_k} ({lp[0]:.1f}) > {right_k} ({rp[0]:.1f})")
                elif target_conv == LATERALITY_SUBJECT and is_frontal:
                    # In subject convention for frontal, left_k must have larger x than right_k
                    if lp[0] < rp[0] - 5.0:
                        rigid_inversions.append(f"{left_k} ({lp[0]:.1f}) < {right_k} ({rp[0]:.1f})")
        if rigid_inversions:
            is_valid = False
            reasons.append(f"paired_laterality_inversions: {rigid_inversions}")
            hard_errors.append(f"paired_laterality_inversions: {rigid_inversions}")

        # 2. Verify articulated limb pairs (elbows, wrists, knees, ankles) - crossing arms or legs
        # Natural human kinesiology allows limbs to cross; these are soft warnings, NOT hard failures!
        articulated_crossings: List[str] = []
        for left_k, right_k in POSE17_ARTICULATED_PAIRED_KEYPOINTS:
            lp = pts.get(left_k)
            rp = pts.get(right_k)
            if lp and rp and lp[2] > 0 and rp[2] > 0:
                if target_conv == LATERALITY_VIEWER:
                    if lp[0] > rp[0] + 5.0:
                        articulated_crossings.append(f"{left_k} ({lp[0]:.1f}) > {right_k} ({rp[0]:.1f})")
                elif target_conv == LATERALITY_SUBJECT and is_frontal:
                    if lp[0] < rp[0] - 5.0:
                        articulated_crossings.append(f"{left_k} ({lp[0]:.1f}) < {right_k} ({rp[0]:.1f})")
        if articulated_crossings:
            crossed_limbs.extend(articulated_crossings)
            soft_warnings.append(f"crossed_limbs_detected: {articulated_crossings}")
            reasons.append(f"crossed_limbs_detected: {articulated_crossings} (soft_warning)")

        return LateralityVerificationResult(
            is_valid=is_valid,
            detected_convention=detected,
            expected_convention=target_conv,
            reasons=reasons,
            delta_x=round(delta_x, 2),
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            crossed_limbs=crossed_limbs,
        )

    elif "vf50" in schema_clean or "face" in schema_clean:
        target_conv = expected_convention or LATERALITY_VIEWER  # VF-50 standard is viewer convention
        pts_map = _extract_vf50_points(data)
        left_eye_pts = [pts_map[i] for i in range(14, 22) if i in pts_map]
        right_eye_pts = [pts_map[i] for i in range(22, 30) if i in pts_map]

        if not (left_eye_pts and right_eye_pts):
            return LateralityVerificationResult(
                is_valid=False,
                detected_convention="unknown",
                expected_convention=target_conv,
                reasons=["missing_eyes_for_face_laterality_check"],
                hard_errors=["missing_eyes_for_face_laterality_check"],
            )

        c_left = sum(p[0] for p in left_eye_pts) / len(left_eye_pts)
        c_right = sum(p[0] for p in right_eye_pts) / len(right_eye_pts)

        # In viewer convention: mattrai is on screen left (smaller x)
        delta_x = c_right - c_left  # positive if mattrai is on screen left
        detected = "viewer" if delta_x > 0 else "subject"

        reasons: List[str] = []
        hard_errors: List[str] = []
        soft_warnings: List[str] = []
        is_valid = True
        if target_conv == LATERALITY_VIEWER:
            if delta_x < 0:
                is_valid = False
                msg = "laterality_inversion: mattrai has larger x than matphai (violates viewer convention)"
                reasons.append(msg)
                hard_errors.append(msg)
        elif target_conv == LATERALITY_SUBJECT:
            if delta_x > 0:
                is_valid = False
                msg = "laterality_inversion: subject right eye has larger x than left eye"
                reasons.append(msg)
                hard_errors.append(msg)

        # Check eyebrows (longmaytrai 0..4 vs longmayphai 5..9)
        left_eb_pts = [pts_map[i] for i in range(0, 5) if i in pts_map]
        right_eb_pts = [pts_map[i] for i in range(5, 10) if i in pts_map]
        if left_eb_pts and right_eb_pts:
            c_left_eb = sum(p[0] for p in left_eb_pts) / len(left_eb_pts)
            c_right_eb = sum(p[0] for p in right_eb_pts) / len(right_eb_pts)
            if target_conv == LATERALITY_VIEWER and c_left_eb > c_right_eb:
                is_valid = False
                msg = f"eyebrow_laterality_inversion: longmaytrai ({c_left_eb:.1f}) > longmayphai ({c_right_eb:.1f})"
                reasons.append(msg)
                hard_errors.append(msg)
            elif target_conv == LATERALITY_SUBJECT and c_left_eb < c_right_eb:
                is_valid = False
                msg = f"eyebrow_laterality_inversion: subject right eyebrow ({c_right_eb:.1f}) > left eyebrow ({c_left_eb:.1f})"
                reasons.append(msg)
                hard_errors.append(msg)

        return LateralityVerificationResult(
            is_valid=is_valid,
            detected_convention=detected,
            expected_convention=target_conv,
            reasons=reasons,
            delta_x=round(delta_x, 2),
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            crossed_limbs=[],
        )

    return LateralityVerificationResult(
        is_valid=False,
        detected_convention="unknown",
        expected_convention=expected_convention or "unknown",
        reasons=["unsupported_schema"],
    )


def verify_mirrored_laterality_invariance(
    data: Any,
    schema_type: str = "pose17",
    convention: str = "subject",
) -> Tuple[bool, str]:
    """Mathematically verify consistency under horizontal mirror reflection (x' = 1000 - x).

    Section 3 Proof:
    - Flipping an image horizontally maps point (x, y) to (1000 - x, y).
    - Under subject convention:
      Symmetric pairs must be swapped to preserve anatomical correctness:
      (left_shoulder <-> right_shoulder). After swap, the mirrored keypoints
      must satisfy the same anatomical frontal invariant.
    - Under viewer convention:
      Points remain defined by screen position. If coordinates are flipped without
      swapping labels, viewer laterality inverts (violating the convention).
      Swapping labels restores screen-left / screen-right alignment.
    """
    schema_clean = str(schema_type).strip().lower()

    if "pose" in schema_clean:
        pts = _extract_pose17_points(data)
        if not pts:
            return False, "no_pose_points_to_test"

        # Apply horizontal reflection: x' = 1000 - x
        mirrored_pts: Dict[str, Tuple[float, float, int, float]] = {
            k: (1000.0 - v[0], v[1], v[2], v[3]) for k, v in pts.items()
        }

        if convention == LATERALITY_SUBJECT:
            # Symmetrically swap left <-> right labels
            swapped_mirrored: Dict[str, Tuple[float, float, int, float]] = dict(mirrored_pts)
            for left_k, right_k in POSE17_PAIRED_KEYPOINTS:
                if left_k in mirrored_pts and right_k in mirrored_pts:
                    swapped_mirrored[left_k] = mirrored_pts[right_k]
                    swapped_mirrored[right_k] = mirrored_pts[left_k]

            res = verify_laterality_convention(swapped_mirrored, schema_type="pose17", expected_convention=LATERALITY_SUBJECT)
            if not res.is_valid:
                return False, f"mirrored_subject_invariance_failed: {res.reasons}"
            return True, "mirrored_subject_invariance_verified"
        else:
            res = verify_laterality_convention(mirrored_pts, schema_type="pose17", expected_convention=LATERALITY_VIEWER)
            return res.is_valid, "mirrored_viewer_check"

    elif "vf50" in schema_clean or "face" in schema_clean:
        pts_map = _extract_vf50_points(data)
        if not pts_map:
            return False, "no_vf50_points_to_test"

        # Apply horizontal reflection
        mirrored_map: Dict[int, Tuple[float, float, int, float]] = {
            k: (1000.0 - v[0], v[1], v[2], v[3]) for k, v in pts_map.items()
        }

        # In viewer convention, swapping the paired eye/eyebrow skeletons maintains viewer alignment
        swapped_vf50: Dict[int, Tuple[float, float, int, float]] = dict(mirrored_map)
        # Swap longmaytrai (0..4) and longmayphai (5..9)
        for i in range(5):
            swapped_vf50[i] = mirrored_map[5 + (4 - i)]
            swapped_vf50[5 + i] = mirrored_map[4 - i]
        # Swap mattrai (14..21) and matphai (22..29)
        for i in range(8):
            swapped_vf50[14 + i] = mirrored_map[22 + i]
            swapped_vf50[22 + i] = mirrored_map[14 + i]

        res = verify_laterality_convention(swapped_vf50, schema_type="vf50", expected_convention=LATERALITY_VIEWER)
        if not res.is_valid:
            return False, f"mirrored_vf50_invariance_failed: {res.reasons}"
        return True, "mirrored_vf50_invariance_verified"

    return False, "unknown_schema"


# Alias for backward compatibility (ensure pytest does not collect it as a test function)
test_mirrored_laterality_invariance = verify_mirrored_laterality_invariance
getattr(test_mirrored_laterality_invariance, "__dict__", {})["__test__"] = False


# ==============================================================================
# 7. QUALITY GATE: HUMAN POSE 17 (Section 16)
# ==============================================================================

def assess_pose17_quality(
    data: Any,
    *,
    laterality_convention: Optional[str] = None,
    min_visible_keypoints: int = 4,
    min_confidence: float = 0.30,
    refine_score_threshold: float = 0.76,
) -> Pose17QualityReport:
    """Rigorous deterministic quality gate for Human Pose 17 skeletons.

    Defaults to the canonical Week-2 schema laterality convention ('viewer').

    Validation checks:
    1. 17 exact expected keypoint names.
    2. No duplicate or coincident points.
    3. All coordinates within image bounds [0, 1000].
    4. Coordinate axis swap detection (x vs y) with head-orientation verification.
    5. Shoulder / Hip vertical ordering with forward-lean / recline tolerance.
    6. Left/Right rigid laterality vs. soft articulated crossing (crossed arms/legs).
    7. Adaptive torso-scale proportion check and 0-span degenerate collapse detection.
    """
    effective_laterality = laterality_convention if laterality_convention is not None else DEFAULT_POSE17_LATERALITY
    pts = _extract_pose17_points(data)
    reasons: List[str] = []
    hard_errors: List[str] = []
    soft_warnings: List[str] = []
    suspect_bones: List[str] = []
    score = 1.0

    # 1. Exact keypoint count & expected names
    present_names = set(pts.keys())
    missing = POSE17_KEYPOINTS_SET - present_names
    extra = present_names - POSE17_KEYPOINTS_SET

    if missing:
        score -= 0.15 * (len(missing) / 17.0)
        reasons.append(f"missing_keypoints: {sorted(missing)}")
        soft_warnings.append(f"missing_keypoints: {len(missing)}")
    if extra:
        score -= 0.10
        reasons.append(f"unexpected_keypoints: {sorted(extra)}")
        soft_warnings.append(f"unexpected_keypoints: {len(extra)}")

    # Filter active and visible points
    active_kps = {k: v for k, v in pts.items() if v[2] > 0 and v[3] >= min_confidence}
    visible_kps = {k: v for k, v in pts.items() if v[2] == 2 and v[3] >= min_confidence}
    active_count = len(active_kps)
    visible_count = len(visible_kps)

    if active_count < min_visible_keypoints:
        msg = f"too_few_active_keypoints: {active_count} < {min_visible_keypoints}"
        reasons.append(msg)
        hard_errors.append(msg)
        return Pose17QualityReport(
            score=0.0,
            is_valid=False,
            needs_refine=True,
            reasons=reasons,
            visible_count=visible_count,
            active_count=active_count,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
        )

    # 2. Coordinate bounds check [0, 1000] & NaN/Inf detection
    out_of_bounds: List[str] = []
    severe_oob: List[str] = []
    has_nan_or_inf = False
    for name, (x, y, vis, conf) in active_kps.items():
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            has_nan_or_inf = True
            severe_oob.append(f"{name}:(nan/inf)")
        elif not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            out_of_bounds.append(f"{name}:({x:.1f},{y:.1f})")
            if x < -50.0 or x > 1050.0 or y < -50.0 or y > 1050.0:
                severe_oob.append(f"{name}:({x:.1f},{y:.1f})")

    if has_nan_or_inf:
        msg = f"severe_out_of_bounds: NaN or Inf coordinates in {severe_oob}"
        reasons.append(msg)
        hard_errors.append(msg)
        return Pose17QualityReport(
            score=0.0,
            is_valid=False,
            needs_refine=True,
            reasons=reasons,
            visible_count=visible_count,
            active_count=active_count,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
        )

    if severe_oob:
        score -= 0.35
        hard_errors.append(f"severe_out_of_bounds: {severe_oob}")
        reasons.append(f"coordinates_out_of_bounds: {out_of_bounds or severe_oob}")
    elif out_of_bounds:
        score -= 0.25
        soft_warnings.append(f"coordinates_out_of_bounds: {out_of_bounds}")
        reasons.append(f"coordinates_out_of_bounds: {out_of_bounds}")

    # 3. Duplicate / coincident active points
    coord_seen: Dict[Tuple[int, int], str] = {}
    duplicates: List[str] = []
    for name, (x, y, vis, conf) in active_kps.items():
        key = (int(round(x)), int(round(y)))
        if key in coord_seen:
            duplicates.append(f"{name} co-located with {coord_seen[key]}")
        else:
            coord_seen[key] = name
    if duplicates:
        score -= 0.20
        soft_warnings.append(f"duplicate_coincident_joints: {duplicates}")
        reasons.append(f"duplicate_coincident_joints: {duplicates}")

    # Section 1.1 Item 2: Machine failure fallback cluster dumped at top-left [0..35, 0..35]
    corner_dump = [name for name, (x, y, vis, conf) in active_kps.items() if x <= 35.0 and y <= 35.0]
    if len(corner_dump) >= 2:
        score -= 0.35
        msg = f"corner_cluster_dump: keypoints clumped at top-left [0..35, 0..35]: {corner_dump}"
        hard_errors.append(msg)
        reasons.append(msg)

    # 4. Non-zero skeleton dimensions and collapse detection
    xs = [v[0] for v in active_kps.values()]
    ys = [v[1] for v in active_kps.values()]
    span_w = max(xs) - min(xs)
    span_h = max(ys) - min(ys)
    bbox_diag = math.hypot(span_w, span_h)

    if bbox_diag < 10.0:
        score -= 0.50
        msg = f"degenerate_skeleton_collapse: diagonal={bbox_diag:.1f} span=({span_w:.1f}x{span_h:.1f})"
        hard_errors.append(msg)
        reasons.append(msg)
    elif span_w < 3.0 or span_h < 3.0:
        score -= 0.15
        soft_warnings.append(f"extreme_aspect_ratio_span: span=({span_w:.1f}x{span_h:.1f})")
        reasons.append(f"extreme_aspect_ratio_span: span=({span_w:.1f}x{span_h:.1f})")

    # 5. Coordinate axis swap check
    axis_swapped, swap_reason = detect_coordinate_axis_swap(pts, schema_type="pose17")
    if axis_swapped:
        score -= 0.40
        hard_errors.append(swap_reason)
        reasons.append(swap_reason)

    # 6. Vertical orientation (Shoulders above hips in upright poses, with forward-lean allowance)
    ls = pts.get("left_shoulder")
    rs = pts.get("right_shoulder")
    lh = pts.get("left_hip")
    rh = pts.get("right_hip")
    la = pts.get("left_ankle")
    ra = pts.get("right_ankle")
    nose = pts.get("nose")

    if ls and rs and lh and rh:
        shoulder_y = (ls[1] + rs[1]) * 0.5
        hip_y = (lh[1] + rh[1]) * 0.5
        ankles_y = None
        if la and ra and la[2] > 0 and ra[2] > 0:
            ankles_y = (la[1] + ra[1]) * 0.5

        if ankles_y is not None and ankles_y > hip_y:
            if shoulder_y > hip_y + 40.0:  # 40 normalized px margin
                # If nose is also below hips, person is bending forward (e.g. driver reaching to footwell)
                if nose and nose[2] > 0 and nose[1] > hip_y:
                    score -= 0.15
                    soft_warnings.append(f"forward_lean_pose: shoulders (y={shoulder_y:.1f}) and nose (y={nose[1]:.1f}) below hips (y={hip_y:.1f})")
                    reasons.append(f"forward_lean_pose: shoulders below hips during forward bend")
                else:
                    score -= 0.35
                    soft_warnings.append(f"inverted_vertical_orientation: shoulders (y={shoulder_y:.1f}) below hips (y={hip_y:.1f})")
                    reasons.append(f"inverted_vertical_orientation: shoulders (y={shoulder_y:.1f}) below hips (y={hip_y:.1f})")

    # 7. Laterality verification (Rigid anchors vs Articulated crossed limbs)
    lat_result = verify_laterality_convention(pts, schema_type="pose17", expected_convention=effective_laterality)
    if not lat_result.is_valid:
        score -= 0.30
        hard_errors.extend(lat_result.hard_errors)
        reasons.extend(lat_result.reasons)
    elif lat_result.soft_warnings:
        score -= 0.08
        soft_warnings.extend(lat_result.soft_warnings)
        reasons.extend(lat_result.soft_warnings)

    # 8. Adaptive torso-scale limb length & proportion sanity (handles seated & profile poses)
    torso_scales: List[float] = []
    if ls and rs and ls[2] > 0 and rs[2] > 0:
        torso_scales.append(math.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    if lh and rh and lh[2] > 0 and rh[2] > 0:
        torso_scales.append(math.hypot(rh[0] - lh[0], rh[1] - lh[1]))
    if ls and lh and ls[2] > 0 and lh[2] > 0:
        torso_scales.append(math.hypot(lh[0] - ls[0], lh[1] - ls[1]) * 0.6)
    if rs and rh and rs[2] > 0 and rh[2] > 0:
        torso_scales.append(math.hypot(rh[0] - rs[0], rh[1] - rs[1]) * 0.6)

    torso_scale: Optional[float] = max(torso_scales) if torso_scales else None

    arm_edges = {
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    }

    for u_name, v_name in POSE17_SKELETON_EDGES:
        u_pt = pts.get(u_name)
        v_pt = pts.get(v_name)
        if not u_pt or not v_pt or u_pt[2] == 0 or v_pt[2] == 0:
            continue
        bone_len = math.hypot(v_pt[0] - u_pt[0], v_pt[1] - u_pt[1])
        bone_tag = f"{u_name}-{v_name}"
        is_arm_bone = (u_name, v_name) in arm_edges or (v_name, u_name) in arm_edges

        # Segment length exceeding 55% of image size (implausible spatial jump)
        if bone_len > 550.0:
            score -= 0.25
            if bone_tag not in suspect_bones:
                suspect_bones.append(bone_tag)
            if bone_len > 720.0:
                hard_errors.append(f"excessive_bone_length: {bone_tag} len={bone_len:.1f}")
            else:
                soft_warnings.append(f"excessive_bone_length: {bone_tag} len={bone_len:.1f}")
            reasons.append(f"excessive_bone_length: {bone_tag} len={bone_len:.1f}")
        elif is_arm_bone and bone_len > 450.0:
            # Forearm or upper arm spanning > 45% of image indicates floating/detached limb in cabin
            score -= 0.20
            if bone_tag not in suspect_bones:
                suspect_bones.append(bone_tag)
            soft_warnings.append(f"excessive_arm_length: {bone_tag} len={bone_len:.1f}")
            reasons.append(f"excessive_arm_length: {bone_tag} len={bone_len:.1f}")

        # Segment length exceeding adaptive torso scale
        if torso_scale is not None and torso_scale > 15.0:
            if is_arm_bone and bone_len > 2.2 * torso_scale:
                score -= 0.20
                if bone_tag not in suspect_bones:
                    suspect_bones.append(bone_tag)
                soft_warnings.append(f"disproportionate_arm_bone: {bone_tag} ratio={bone_len / torso_scale:.2f}")
                reasons.append(f"disproportionate_arm_bone: {bone_tag} ratio={bone_len / torso_scale:.2f}")
            elif bone_len > 4.5 * torso_scale:
                score -= 0.20
                if bone_tag not in suspect_bones:
                    suspect_bones.append(bone_tag)
                soft_warnings.append(f"disproportionate_bone: {bone_tag} ratio={bone_len / torso_scale:.2f}")
                reasons.append(f"disproportionate_bone: {bone_tag} ratio={bone_len / torso_scale:.2f}")

    # 9. Arm proportion consistency check (upper arm vs forearm) for detached/floating joints
    le = pts.get("left_elbow")
    lw = pts.get("left_wrist")
    if ls and le and lw and ls[2] > 0 and le[2] > 0 and lw[2] > 0:
        l_upper = math.hypot(le[0] - ls[0], le[1] - ls[1])
        l_fore = math.hypot(lw[0] - le[0], lw[1] - le[1])
        if l_upper > 15.0 and l_fore > 15.0:
            if l_fore > 2.4 * l_upper:
                tag = "left_elbow-left_wrist"
                if tag not in suspect_bones:
                    suspect_bones.append(tag)
                score -= 0.15
                soft_warnings.append(f"disproportionate_forearm: left forearm ({l_fore:.1f}) > 2.4x upper arm ({l_upper:.1f})")
                reasons.append(f"disproportionate_forearm: left forearm > 2.4x upper arm (floating wrist)")
            elif l_upper > 2.8 * l_fore:
                tag = "left_shoulder-left_elbow"
                if tag not in suspect_bones:
                    suspect_bones.append(tag)
                score -= 0.15
                soft_warnings.append(f"disproportionate_upper_arm: left upper arm ({l_upper:.1f}) > 2.8x forearm ({l_fore:.1f})")
                reasons.append(f"disproportionate_upper_arm: left upper arm > 2.8x forearm (floating elbow)")

    re = pts.get("right_elbow")
    rw = pts.get("right_wrist")
    if rs and re and rw and rs[2] > 0 and re[2] > 0 and rw[2] > 0:
        r_upper = math.hypot(re[0] - rs[0], re[1] - rs[1])
        r_fore = math.hypot(rw[0] - re[0], rw[1] - re[1])
        if r_upper > 15.0 and r_fore > 15.0:
            if r_fore > 2.4 * r_upper:
                tag = "right_elbow-right_wrist"
                if tag not in suspect_bones:
                    suspect_bones.append(tag)
                score -= 0.15
                soft_warnings.append(f"disproportionate_forearm: right forearm ({r_fore:.1f}) > 2.4x upper arm ({r_upper:.1f})")
                reasons.append(f"disproportionate_forearm: right forearm > 2.4x upper arm (floating wrist)")
            elif r_upper > 2.8 * r_fore:
                tag = "right_shoulder-right_elbow"
                if tag not in suspect_bones:
                    suspect_bones.append(tag)
                score -= 0.15
                soft_warnings.append(f"disproportionate_upper_arm: right upper arm ({r_upper:.1f}) > 2.8x forearm ({r_fore:.1f})")
                reasons.append(f"disproportionate_upper_arm: right upper arm > 2.8x forearm (floating elbow)")

    score = max(0.0, min(1.0, score))
    is_valid = len(hard_errors) == 0 and score >= 0.35 and not axis_swapped

    return Pose17QualityReport(
        score=round(score, 3),
        is_valid=is_valid,
        needs_refine=(score < refine_score_threshold or bool(hard_errors) or bool(soft_warnings) or bool(reasons)),
        reasons=reasons,
        visible_count=visible_count,
        active_count=active_count,
        suspect_bones=suspect_bones,
        laterality_status=lat_result.detected_convention,
        axis_swap_detected=axis_swapped,
        hard_errors=hard_errors,
        soft_warnings=soft_warnings,
    )


# ==============================================================================
# 8. QUALITY GATE: FACE LANDMARK VF-50 (Sections 17 & 18)
# ==============================================================================

def assess_vf50_quality(
    data: Any,
    *,
    laterality_convention: str = LATERALITY_VIEWER,
    refine_score_threshold: float = 0.76,
) -> VF50QualityReport:
    """Rigorous deterministic quality gate for Face Landmark VF-50 (50 points, 7 skeletons).

    Validation checks:
    1. Exact 50 points total, partitioned across 7 canonical components.
    2. Coordinates in bounds [0, 1000] and NaN/Inf detection.
    3. Coordinate axis swap detection with eye-line orientation check.
    4. Laterality (Viewer convention by VinFast guideline: mattrai on image left).
    5. Eyebrows positioned above corresponding eyes (with head roll compensation).
    6. Nose bridge (10..13) alignment and central positioning between eyes and mouth.
    7. Eyelid contour vs pupil/iris (eyelid boundaries, upper above lower, no figure-8 crossing).
    8. Outer lip vs inner lip topological constraints (inner lip enclosed inside outer lip).
    9. Head tilt/roll tolerance with conservative geometric checks.
    """
    pts_map = _extract_vf50_points(data)
    reasons: List[str] = []
    hard_errors: List[str] = []
    soft_warnings: List[str] = []
    suspect_components: List[str] = []
    score = 1.0

    # 1. Total point count & completeness
    total_pts = len(pts_map)
    component_counts: Dict[str, int] = {}
    for comp_name, (start, end) in VF50_COMPONENT_RANGES.items():
        cnt = sum(1 for i in range(start, end) if i in pts_map)
        component_counts[comp_name] = cnt
        expected_cnt = end - start
        if cnt < expected_cnt:
            score -= 0.12 * ((expected_cnt - cnt) / expected_cnt)
            suspect_components.append(comp_name)
            msg = f"incomplete_component_{comp_name}: {cnt}/{expected_cnt} points"
            reasons.append(msg)
            soft_warnings.append(msg)

    if total_pts < VF50_EXPECTED_TOTAL_POINTS:
        score -= 0.25 * ((VF50_EXPECTED_TOTAL_POINTS - total_pts) / 50.0)
        msg = f"invalid_total_points: {total_pts}/{VF50_EXPECTED_TOTAL_POINTS}"
        reasons.append(msg)
        if total_pts < 35:
            hard_errors.append(msg)
        else:
            soft_warnings.append(msg)

    # 2. Coordinates within image bounds [0, 1000] & NaN/Inf detection
    out_of_bounds: List[int] = []
    severe_oob: List[int] = []
    has_nan_or_inf = False
    for p_id, (x, y, vis, conf) in pts_map.items():
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            has_nan_or_inf = True
            severe_oob.append(p_id)
        elif not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            out_of_bounds.append(p_id)
            if x < -50.0 or x > 1050.0 or y < -50.0 or y > 1050.0:
                severe_oob.append(p_id)

    if has_nan_or_inf:
        msg = f"severe_out_of_bounds: NaN or Inf coordinates in IDs {severe_oob[:10]}"
        reasons.append(msg)
        hard_errors.append(msg)
        return VF50QualityReport(
            score=0.0,
            is_valid=False,
            needs_refine=True,
            reasons=reasons,
            point_count=total_pts,
            component_counts=component_counts,
            suspect_components=list(set(suspect_components)),
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
        )

    if severe_oob:
        score -= 0.35
        hard_errors.append(f"severe_out_of_bounds: IDs {severe_oob[:10]}")
        reasons.append(f"landmarks_out_of_bounds: IDs {(out_of_bounds or severe_oob)[:10]}")
    elif out_of_bounds:
        score -= 0.25
        soft_warnings.append(f"landmarks_out_of_bounds: IDs {out_of_bounds[:10]}")
        reasons.append(f"landmarks_out_of_bounds: IDs {out_of_bounds[:10]}")

    # 3. Coordinate axis swap check
    axis_swapped, swap_reason = detect_coordinate_axis_swap(pts_map, schema_type="vf50")
    if axis_swapped:
        score -= 0.45
        hard_errors.append(swap_reason)
        reasons.append(swap_reason)

    # 4. Laterality verification (Viewer convention: mattrai is on screen left)
    lat_result = verify_laterality_convention(pts_map, schema_type="vf50", expected_convention=laterality_convention)
    if not lat_result.is_valid:
        score -= 0.35
        suspect_components.extend(["mattrai", "matphai"])
        hard_errors.extend(lat_result.hard_errors)
        reasons.extend(lat_result.reasons)
    elif lat_result.soft_warnings:
        score -= 0.08
        soft_warnings.extend(lat_result.soft_warnings)
        reasons.extend(lat_result.soft_warnings)

    # Calculate face roll angle and inter-ocular distance (IOD)
    left_eye_pts = [pts_map[i] for i in range(14, 22) if i in pts_map]
    right_eye_pts = [pts_map[i] for i in range(22, 30) if i in pts_map]

    iod = 0.0
    roll_angle_rad = 0.0
    c_left_eye = (0.0, 0.0)
    c_right_eye = (0.0, 0.0)

    if left_eye_pts and right_eye_pts:
        c_left_eye = (
            sum(p[0] for p in left_eye_pts) / len(left_eye_pts),
            sum(p[1] for p in left_eye_pts) / len(left_eye_pts),
        )
        c_right_eye = (
            sum(p[0] for p in right_eye_pts) / len(right_eye_pts),
            sum(p[1] for p in right_eye_pts) / len(right_eye_pts),
        )
        iod = math.hypot(c_right_eye[0] - c_left_eye[0], c_right_eye[1] - c_left_eye[1])
        roll_angle_rad = math.atan2(c_right_eye[1] - c_left_eye[1], c_right_eye[0] - c_left_eye[0])

    # Un-rotated coordinates helper for robust tilt evaluation
    cos_r = math.cos(-roll_angle_rad)
    sin_r = math.sin(-roll_angle_rad)
    cx_face = (c_left_eye[0] + c_right_eye[0]) * 0.5
    cy_face = (c_left_eye[1] + c_right_eye[1]) * 0.5

    def to_face_frame(pt: Tuple[float, float, int, float]) -> Tuple[float, float]:
        dx = pt[0] - cx_face
        dy = pt[1] - cy_face
        return (cx_face + dx * cos_r - dy * sin_r, cy_face + dx * sin_r + dy * cos_r)

    # 5. Eyebrows above eyes
    left_eb_pts = [pts_map[i] for i in range(0, 5) if i in pts_map]
    right_eb_pts = [pts_map[i] for i in range(5, 10) if i in pts_map]

    if left_eb_pts and left_eye_pts:
        y_eb_left = sum(to_face_frame(p)[1] for p in left_eb_pts) / len(left_eb_pts)
        y_eye_left = sum(to_face_frame(p)[1] for p in left_eye_pts) / len(left_eye_pts)
        # Eyebrows must be above eyes (smaller y in un-rotated face frame)
        if y_eb_left > y_eye_left - 3.0:
            score -= 0.20
            suspect_components.append("longmaytrai")
            msg = f"eyebrow_below_eye: longmaytrai (y={y_eb_left:.1f}) below or touching mattrai (y={y_eye_left:.1f})"
            soft_warnings.append(msg)
            reasons.append(msg)

    if right_eb_pts and right_eye_pts:
        y_eb_right = sum(to_face_frame(p)[1] for p in right_eb_pts) / len(right_eb_pts)
        y_eye_right = sum(to_face_frame(p)[1] for p in right_eye_pts) / len(right_eye_pts)
        if y_eb_right > y_eye_right - 3.0:
            score -= 0.20
            suspect_components.append("longmayphai")
            msg = f"eyebrow_below_eye: longmayphai (y={y_eb_right:.1f}) below or touching matphai (y={y_eye_right:.1f})"
            soft_warnings.append(msg)
            reasons.append(msg)

    # 6. Eyelid contour vs pupil/iris & opposing eyelid checks
    # Upper eyelid points must be higher (or equal) to opposing lower eyelid points
    for upper_id, lower_id in VF50_EYELID_OPPOSING_PAIRS:
        if upper_id in pts_map and lower_id in pts_map:
            u_rot = to_face_frame(pts_map[upper_id])
            l_rot = to_face_frame(pts_map[lower_id])
            # Tolerance 4.0 normalized units for squinting/closed eyes
            if u_rot[1] > l_rot[1] + 4.0:
                score -= 0.15
                comp = "mattrai" if upper_id < 22 else "matphai"
                suspect_components.append(comp)
                msg = f"inverted_eyelid: upper pt {upper_id} below lower pt {lower_id}"
                soft_warnings.append(msg)
                reasons.append(msg)

    # Check for self-intersecting eye loops (figure-8 / X-shape) - Hard error
    if len(left_eye_pts) == 8:
        poly_left_eye = [(pts_map[i][0], pts_map[i][1]) for i in range(14, 22)]
        if _self_intersects(poly_left_eye, closed=True):
            score -= 0.25
            suspect_components.append("mattrai")
            msg = "self_intersecting_eye_contour: mattrai forms figure-8"
            hard_errors.append(msg)
            reasons.append(msg)

    if len(right_eye_pts) == 8:
        poly_right_eye = [(pts_map[i][0], pts_map[i][1]) for i in range(22, 30)]
        if _self_intersects(poly_right_eye, closed=True):
            score -= 0.25
            suspect_components.append("matphai")
            msg = "self_intersecting_eye_contour: matphai forms figure-8"
            hard_errors.append(msg)
            reasons.append(msg)

    # 7. Nose alignment & position between eyes and mouth
    nose_pts = [pts_map[i] for i in range(10, 14) if i in pts_map]
    if len(nose_pts) == 4:
        # Check monotonic downwards progression along nose bridge
        y_nose_rot = [to_face_frame(pts_map[i])[1] for i in range(10, 14)]
        if not (y_nose_rot[0] <= y_nose_rot[1] + 3.0 <= y_nose_rot[2] + 6.0 <= y_nose_rot[3] + 9.0):
            score -= 0.15
            suspect_components.append("songmui")
            msg = "disordered_nose_bridge: points 10..13 not monotonic down bridge"
            soft_warnings.append(msg)
            reasons.append(msg)

    mouth_outer_pts = [pts_map[i] for i in range(30, 42) if i in pts_map]
    if 13 in pts_map and left_eye_pts and right_eye_pts and mouth_outer_pts:
        y_nose_base = to_face_frame(pts_map[13])[1]
        y_eyes = (to_face_frame(c_left_eye + (0, 0))[1] + to_face_frame(c_right_eye + (0, 0))[1]) * 0.5
        y_mouth_top = min(to_face_frame(p)[1] for p in mouth_outer_pts)

        if not (y_eyes < y_nose_base < y_mouth_top):
            score -= 0.25
            suspect_components.append("songmui")
            msg = f"nose_vertical_position_invalid: base y={y_nose_base:.1f} not between eyes ({y_eyes:.1f}) and mouth ({y_mouth_top:.1f})"
            soft_warnings.append(msg)
            reasons.append(msg)

    # 8. Outer lip vs Inner lip topological constraints
    mouth_inner_pts = [pts_map[i] for i in range(42, 50) if i in pts_map]
    if len(mouth_outer_pts) == 12 and len(mouth_inner_pts) == 8:
        poly_outer = [(pts_map[i][0], pts_map[i][1]) for i in range(30, 42)]
        poly_inner = [(pts_map[i][0], pts_map[i][1]) for i in range(42, 50)]

        area_outer = _polygon_area(poly_outer)
        area_inner = _polygon_area(poly_inner)

        # Topological constraint: inner lip area cannot exceed outer lip area (Hard Error)
        if area_inner > area_outer + 5.0:
            score -= 0.35
            suspect_components.append("moitrong")
            msg = f"inner_lip_larger_than_outer: inner={area_inner:.1f} > outer={area_outer:.1f}"
            hard_errors.append(msg)
            reasons.append(msg)

        # Horizontal nesting constraint: inner corners inside outer corners
        x_outer_left = pts_map[30][0]
        x_outer_right = pts_map[36][0]
        x_inner_left = pts_map[42][0]
        x_inner_right = pts_map[46][0]

        if x_inner_left < x_outer_left - 4.0 or x_inner_right > x_outer_right + 4.0:
            score -= 0.25
            suspect_components.append("moitrong")
            msg = f"inner_lip_protrudes_horizontally: inner=[{x_inner_left:.1f},{x_inner_right:.1f}] outer=[{x_outer_left:.1f},{x_outer_right:.1f}]"
            soft_warnings.append(msg)
            reasons.append(msg)

        if _self_intersects(poly_outer, closed=True):
            score -= 0.20
            suspect_components.append("moingoai")
            msg = "self_intersecting_lip_contour: moingoai self-intersects"
            soft_warnings.append(msg)
            reasons.append(msg)
        if _self_intersects(poly_inner, closed=True):
            score -= 0.20
            suspect_components.append("moitrong")
            msg = "self_intersecting_lip_contour: moitrong self-intersects"
            soft_warnings.append(msg)
            reasons.append(msg)

    score = max(0.0, min(1.0, score))
    is_valid = len(hard_errors) == 0 and score >= 0.35 and not axis_swapped

    return VF50QualityReport(
        score=round(score, 3),
        is_valid=is_valid,
        needs_refine=(score < refine_score_threshold or bool(hard_errors) or bool(soft_warnings) or bool(reasons)),
        reasons=reasons,
        point_count=total_pts,
        component_counts=component_counts,
        suspect_components=list(set(suspect_components)),
        laterality_status=lat_result.detected_convention,
        axis_swap_detected=axis_swapped,
        hard_errors=hard_errors,
        soft_warnings=soft_warnings,
    )


# ==============================================================================
# 9. MASTER ASSESS QUALITY DISPATCHER
# ==============================================================================

def assess_quality(
    items: Iterable[Any],
    mode: str,
    *,
    min_confidence: float = 0.80,
    refine_score_threshold: float = 0.76,
    laterality_convention: Optional[str] = None,
) -> QualityReport:
    """Return a deterministic heuristic quality report for parsed 0..1000 geometry."""
    mode = (mode or "").strip().lower()
    items = list(items)

    # Dispatch to Human Pose 17 quality gate
    if mode in ("human_pose_17", "pose_17", "pose17", "skeleton_pose17"):
        if not items:
            return QualityReport(
                score=0.0,
                needs_refine=True,
                reasons=["empty_pose_detection"],
                suspect_labels=[],
                item_count=0,
                suspect_count=1,
                hard_errors=["empty_pose_detection"],
            )
        conv = laterality_convention or DEFAULT_POSE17_LATERALITY
        sub_reports = [assess_pose17_quality(it, laterality_convention=conv, refine_score_threshold=refine_score_threshold) for it in items]
        avg_score = sum(r.score for r in sub_reports) / max(1, len(sub_reports))
        reasons: List[str] = []
        suspects: List[str] = []
        hard_errors: List[str] = []
        soft_warnings: List[str] = []
        for idx, r in enumerate(sub_reports, start=1):
            hard_errors.extend(f"person_{idx}:{e}" for e in r.hard_errors)
            soft_warnings.extend(f"person_{idx}:{w}" for w in r.soft_warnings)
            if not r.is_valid or r.needs_refine:
                label_id = f"person_{idx}"
                suspects.append(label_id)
                reasons.extend(f"{label_id}:{reason}" for reason in r.reasons)
        return QualityReport(
            score=round(avg_score, 3),
            needs_refine=(bool(suspects) or avg_score < refine_score_threshold or bool(hard_errors) or bool(soft_warnings)),
            reasons=reasons,
            suspect_labels=suspects,
            item_count=len(items),
            suspect_count=len(suspects),
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
        )

    # Dispatch to Face Landmark VF-50 quality gate
    if mode in ("face_landmark_vf50", "face_vf50", "vf50", "face_landmark"):
        if not items:
            return QualityReport(
                score=0.0,
                needs_refine=True,
                reasons=["empty_face_detection"],
                suspect_labels=[],
                item_count=0,
                suspect_count=1,
                hard_errors=["empty_face_detection"],
            )
        conv = laterality_convention or LATERALITY_VIEWER
        face_payload = items if len(items) > 1 else items[0]
        vf_report = assess_vf50_quality(face_payload, laterality_convention=conv, refine_score_threshold=refine_score_threshold)
        reasons = [f"face:{r}" for r in vf_report.reasons]
        suspects = [f"face_{c}" for c in vf_report.suspect_components]
        hard_errors = [f"face:{e}" for e in vf_report.hard_errors]
        soft_warnings = [f"face:{w}" for w in vf_report.soft_warnings]
        return QualityReport(
            score=vf_report.score,
            needs_refine=vf_report.needs_refine,
            reasons=reasons,
            suspect_labels=suspects,
            item_count=len(items),
            suspect_count=len(suspects),
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
        )

    # Standard 31-label policies (rectangle_mask, polygon_mask, polyline)
    if not items:
        return QualityReport(
            score=0.0,
            needs_refine=True,
            reasons=["empty_detection"],
            suspect_labels=[],
            item_count=0,
            suspect_count=1,
            hard_errors=["empty_detection"],
        )

    scores: List[float] = []
    reasons_list: List[str] = []
    suspects_list: List[str] = []
    hard_errors_list: List[str] = []
    soft_warnings_list: List[str] = []

    for item in items:
        score = 1.0
        item_reasons: List[str] = []
        item_hard_errors: List[str] = []
        item_soft_warnings: List[str] = []
        lbl = str(getattr(item, "label", "unknown"))

        confidence = getattr(item, "confidence", None)
        if confidence is not None and float(confidence) < min_confidence:
            score -= 0.18
            item_reasons.append("low_confidence")
            item_soft_warnings.append("low_confidence")

        if mode in ("rectangle_mask", "box_mask"):
            box = getattr(item, "box_2d", None)
            mask = getattr(item, "instance_mask", None) or getattr(item, "mask", None)
            if not box or not mask or len(mask) < 3:
                score = 0.0
                item_reasons.append("missing_box_or_mask")
                item_hard_errors.append("missing_box_or_mask")
            else:
                ymin, xmin, ymax, xmax = map(float, box)
                box_xyxy = (xmin, ymin, xmax, ymax)
                box_area = max(1.0, (xmax - xmin) * (ymax - ymin))
                mask_bbox = _bbox_from_points(mask)
                mask_area = _polygon_area(mask)
                if len(mask) < 6:
                    score -= 0.18
                    item_reasons.append("coarse_mask")
                    item_soft_warnings.append("coarse_mask")
                if mask_bbox is not None and _bbox_iou_xyxy(box_xyxy, mask_bbox) < 0.62:
                    score -= 0.30
                    item_reasons.append("mask_box_mismatch")
                    item_soft_warnings.append("mask_box_mismatch")
                fill_ratio = mask_area / box_area
                if fill_ratio < 0.15 or fill_ratio > 1.25:
                    score -= 0.30
                    item_reasons.append("implausible_mask_fill")
                    item_soft_warnings.append("implausible_mask_fill")
                if _self_intersects(mask, closed=True):
                    score -= 0.30
                    item_reasons.append("self_intersecting_mask")
                    item_hard_errors.append("self_intersecting_mask")

        elif mode == "polygon_mask":
            poly = getattr(item, "region_polygon", None) or getattr(item, "mask", None)
            if not poly or len(poly) < 3:
                score = 0.0
                item_reasons.append("missing_polygon")
                item_hard_errors.append("missing_polygon")
            else:
                area = _polygon_area(poly)
                bbox = _bbox_from_points(poly)
                bbox_area = 0.0
                if bbox is not None:
                    bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                if area < 150.0:
                    score -= 0.28
                    item_reasons.append("tiny_region")
                    item_soft_warnings.append("tiny_region")
                if len(poly) < 7:
                    score -= 0.22
                    item_reasons.append("coarse_polygon")
                    item_soft_warnings.append("coarse_polygon")
                if bbox_area >= 180_000.0 and len(poly) < 16:
                    score -= 0.30
                    item_reasons.append("large_region_too_coarse")
                    item_soft_warnings.append("large_region_too_coarse")
                if _self_intersects(poly, closed=True):
                    score -= 0.35
                    item_reasons.append("self_intersecting_polygon")
                    item_hard_errors.append("self_intersecting_polygon")

        elif mode == "polyline":
            line = getattr(item, "lane_polyline", None) or getattr(item, "mask", None)
            if not line or len(line) < 2:
                score = 0.0
                item_reasons.append("missing_polyline")
                item_hard_errors.append("missing_polyline")
            else:
                if len(line) < 3:
                    score -= 0.18
                    item_reasons.append("undersampled_polyline")
                    item_soft_warnings.append("undersampled_polyline")
                if _line_path_ratio(line) > 3.2:
                    score -= 0.32
                    item_reasons.append("zigzag_polyline")
                    item_soft_warnings.append("zigzag_polyline")
                max_seg = max(
                    (
                        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
                        for a, b in zip(line, line[1:])
                    ),
                    default=0.0,
                )
                if max_seg > 700.0:
                    score -= 0.20
                    item_reasons.append("oversized_line_jump")
                    item_soft_warnings.append("oversized_line_jump")

        score = max(0.0, min(1.0, score))
        scores.append(score)
        hard_errors_list.extend(f"{lbl}:{e}" for e in item_hard_errors)
        soft_warnings_list.extend(f"{lbl}:{w}" for w in item_soft_warnings)
        if score < refine_score_threshold or item_hard_errors:
            suspects_list.append(lbl)
            reasons_list.extend(f"{lbl}:{r}" for r in item_reasons)

    aggregate = sum(scores) / max(1, len(scores))
    return QualityReport(
        score=round(aggregate, 3),
        needs_refine=(bool(suspects_list) or aggregate < refine_score_threshold or bool(hard_errors_list)),
        reasons=reasons_list,
        suspect_labels=suspects_list,
        item_count=len(items),
        suspect_count=len(suspects_list),
        hard_errors=hard_errors_list,
        soft_warnings=soft_warnings_list,
    )


def should_accept_refinement(
    before: QualityReport,
    after: QualityReport,
    *,
    min_gain: float = 0.03,
) -> bool:
    """Conservatively select a second-pass result without collapsing recall."""
    if after.item_count <= 0:
        return False
    if before.item_count > 1 and after.item_count < max(1, math.ceil(before.item_count * 0.65)):
        return False
    if after.suspect_count < before.suspect_count and after.score >= before.score - 0.03:
        return True
    if after.score >= before.score + min_gain:
        return True
    if before.needs_refine and after.item_count >= before.item_count and after.score >= before.score - 0.015:
        return True
    return False

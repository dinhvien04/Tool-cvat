"""Deterministic geometry and laterality quality gate for adaptive refinement and skeleton validation.

The gate does not call any model. It identifies structurally valid but suspicious geometry,
validates anatomical vs. viewer laterality invariants, checks coordinate bounds, ensures
topological nesting, and detects axis swaps (x vs y) for:
1. Master 31-label instance/region/line detections (rectangle_mask, polygon_mask, polyline).
2. Human Pose 17 keypoint skeletons (COCO-17 topology, subject/anatomical laterality).
3. Face Landmark VF-50 (VinFast 7-component face topology, viewer laterality).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union


# ==============================================================================
# 1. CONSTANTS & CANONICAL SCHEMAS
# ==============================================================================

LATERALITY_SUBJECT: str = "subject"  # Anatomical left/right of the person being viewed
LATERALITY_VIEWER: str = "viewer"    # Image/screen left/right (viewer's left/right)

# Canonical 17 COCO keypoints in indexed anatomical order
POSE17_KEYPOINTS: Tuple[str, ...] = (
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
)
POSE17_KEYPOINTS_SET: FrozenSet[str] = frozenset(POSE17_KEYPOINTS)

# Symmetric left/right pairs for Pose 17 laterality validation
POSE17_PAIRED_KEYPOINTS: Tuple[Tuple[str, str], ...] = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# VinFast Guideline (output_pose17_guideline.txt) 1..17 index to canonical name mapping
VINFAST_POSE17_INDEX_TO_NAME: Dict[int, str] = {
    1: "nose",
    2: "right_eye",
    3: "left_eye",
    4: "right_ear",
    5: "left_ear",
    6: "right_shoulder",
    7: "left_shoulder",
    8: "right_elbow",
    9: "left_elbow",
    10: "right_wrist",
    11: "left_wrist",
    12: "right_hip",
    13: "left_hip",
    14: "right_knee",
    15: "left_knee",
    16: "right_ankle",
    17: "left_ankle",
}

# Canonical bone connectivity graph for Pose 17
POSE17_SKELETON_EDGES: Tuple[Tuple[str, str], ...] = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

# VF-50 Canonical Component Names & Point Counts
# Authoritative VinFast Week-2 specification (output_vf50_guideline.txt)
VF50_COMPONENT_NAMES: Tuple[str, ...] = (
    "longmaytrai",  # 0..4 (5 pts, open)
    "longmayphai",  # 5..9 (5 pts, open)
    "songmui",      # 10..13 (4 pts, open)
    "mattrai",      # 14..21 (8 pts, closed loop)
    "matphai",      # 22..29 (8 pts, closed loop)
    "moingoai",     # 30..41 (12 pts, closed loop)
    "moitrong",     # 42..49 (8 pts, closed loop)
)

VF50_COMPONENT_COUNTS: Dict[str, int] = {
    "longmaytrai": 5,
    "longmayphai": 5,
    "songmui": 4,
    "mattrai": 8,
    "matphai": 8,
    "moingoai": 12,
    "moitrong": 8,
}
VF50_EXPECTED_TOTAL_POINTS: int = 50

# Index ranges for components in continuous 0..49 indexing
VF50_COMPONENT_RANGES: Dict[str, Tuple[int, int]] = {
    "longmaytrai": (0, 5),
    "longmayphai": (5, 10),
    "songmui": (10, 14),
    "mattrai": (14, 22),
    "matphai": (22, 30),
    "moingoai": (30, 42),
    "moitrong": (42, 50),
}

# Opposing eyelid pairs (upper eyelid index, lower eyelid index) for VF-50
# Rule: y(upper) <= y(lower) (remember y increases downwards)
VF50_EYELID_OPPOSING_PAIRS: Tuple[Tuple[int, int], ...] = (
    # mattrai: upper 15, 16, 17 vs lower 21, 20, 19
    (15, 21),
    (16, 20),
    (17, 19),
    # matphai: upper 23, 24, 25 vs lower 29, 28, 27
    (23, 29),
    (24, 28),
    (25, 27),
)


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


@dataclass
class LateralityVerificationResult:
    """Detailed verification outcome for laterality invariants."""
    is_valid: bool
    detected_convention: str
    expected_convention: str
    reasons: List[str] = field(default_factory=list)
    delta_x: float = 0.0


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
        if isinstance(elements, list):
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
                if isinstance(v, (list, tuple)) and len(v) >= 2:
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

        # If torso is horizontal and shoulders are vertical, axes are swapped
        if torso_dx > 1.35 * torso_dy and shoulder_dy > 1.35 * shoulder_dx:
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
        target_conv = expected_convention or LATERALITY_SUBJECT
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
        is_valid = True
        if target_conv == LATERALITY_SUBJECT:
            if is_frontal and delta_x < 0:
                is_valid = False
                reasons.append("laterality_inversion: left_shoulder has smaller x than right_shoulder for frontal subject")
        elif target_conv == LATERALITY_VIEWER:
            if delta_x > 0:
                is_valid = False
                reasons.append("laterality_inversion: viewer left has larger x than viewer right")

        return LateralityVerificationResult(
            is_valid=is_valid,
            detected_convention=detected,
            expected_convention=target_conv,
            reasons=reasons,
            delta_x=round(delta_x, 2),
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
            )

        c_left = sum(p[0] for p in left_eye_pts) / len(left_eye_pts)
        c_right = sum(p[0] for p in right_eye_pts) / len(right_eye_pts)

        # In viewer convention: mattrai is on screen left (smaller x)
        delta_x = c_right - c_left  # positive if mattrai is on screen left
        detected = "viewer" if delta_x > 0 else "subject"

        reasons: List[str] = []
        is_valid = True
        if target_conv == LATERALITY_VIEWER:
            if delta_x < 0:
                is_valid = False
                reasons.append("laterality_inversion: mattrai has larger x than matphai (violates viewer convention)")
        elif target_conv == LATERALITY_SUBJECT:
            if delta_x > 0:
                is_valid = False
                reasons.append("laterality_inversion: subject right eye has larger x than left eye")

        return LateralityVerificationResult(
            is_valid=is_valid,
            detected_convention=detected,
            expected_convention=target_conv,
            reasons=reasons,
            delta_x=round(delta_x, 2),
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
    laterality_convention: str = LATERALITY_SUBJECT,
    min_visible_keypoints: int = 4,
    min_confidence: float = 0.30,
    refine_score_threshold: float = 0.76,
) -> Pose17QualityReport:
    """Rigorous deterministic quality gate for Human Pose 17 skeletons.

    Validation checks:
    1. 17 exact expected keypoint names.
    2. No duplicate or coincident points.
    3. All coordinates within image bounds [0, 1000].
    4. Coordinate axis swap detection (x vs y).
    5. Shoulder / Hip vertical ordering (shoulders above hips in upright poses).
    6. Left/Right pair sanity and laterality correctness.
    7. Non-zero skeleton area and limb length ratio sanity.
    """
    pts = _extract_pose17_points(data)
    reasons: List[str] = []
    suspect_bones: List[str] = []
    score = 1.0

    # 1. Exact keypoint count & expected names
    present_names = set(pts.keys())
    missing = POSE17_KEYPOINTS_SET - present_names
    extra = present_names - POSE17_KEYPOINTS_SET

    if missing:
        score -= 0.15 * (len(missing) / 17.0)
        reasons.append(f"missing_keypoints: {sorted(missing)}")
    if extra:
        score -= 0.10
        reasons.append(f"unexpected_keypoints: {sorted(extra)}")

    # Filter active and visible points
    active_kps = {k: v for k, v in pts.items() if v[2] > 0 and v[3] >= min_confidence}
    visible_kps = {k: v for k, v in pts.items() if v[2] == 2 and v[3] >= min_confidence}
    active_count = len(active_kps)
    visible_count = len(visible_kps)

    if active_count < min_visible_keypoints:
        reasons.append(f"too_few_active_keypoints: {active_count} < {min_visible_keypoints}")
        return Pose17QualityReport(
            score=0.0,
            is_valid=False,
            needs_refine=True,
            reasons=reasons,
            visible_count=visible_count,
            active_count=active_count,
        )

    # 2. Coordinate bounds check [0, 1000]
    out_of_bounds: List[str] = []
    for name, (x, y, vis, conf) in active_kps.items():
        if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            out_of_bounds.append(f"{name}:({x:.1f},{y:.1f})")
    if out_of_bounds:
        score -= 0.35
        reasons.append(f"coordinates_out_of_bounds: {out_of_bounds}")

    # 3. Duplicate / coincident active points
    coord_seen: Dict[Tuple[int, int], str] = {}
    duplicates: List[str] = []
    for name, (x, y, vis, conf) in active_kps.items():
        # Round to 1 decimal place to detect suspicious snapping
        key = (int(round(x)), int(round(y)))
        if key in coord_seen:
            duplicates.append(f"{name} co-located with {coord_seen[key]}")
        else:
            coord_seen[key] = name
    if duplicates:
        score -= 0.25
        reasons.append(f"duplicate_coincident_joints: {duplicates}")

    # Section 1.1 Item 2: Machine failure fallback cluster dumped at top-left [0..35, 0..35]
    corner_dump = [name for name, (x, y, vis, conf) in active_kps.items() if x <= 35.0 and y <= 35.0]
    if len(corner_dump) >= 2:
        score -= 0.35
        reasons.append(f"corner_cluster_dump: keypoints clumped at top-left [0..35, 0..35]: {corner_dump}")

    # 4. Non-zero skeleton dimensions and collapse detection
    xs = [v[0] for v in active_kps.values()]
    ys = [v[1] for v in active_kps.values()]
    span_w = max(xs) - min(xs)
    span_h = max(ys) - min(ys)
    bbox_diag = math.hypot(span_w, span_h)

    if span_w < 3.0 or span_h < 3.0 or bbox_diag < 10.0:
        score -= 0.50
        reasons.append(f"degenerate_skeleton_collapse: diagonal={bbox_diag:.1f} span=({span_w:.1f}x{span_h:.1f})")

    # 5. Coordinate axis swap check
    axis_swapped, swap_reason = detect_coordinate_axis_swap(pts, schema_type="pose17")
    if axis_swapped:
        score -= 0.40
        reasons.append(swap_reason)

    # 6. Vertical orientation (Shoulders above hips in upright poses)
    ls = pts.get("left_shoulder")
    rs = pts.get("right_shoulder")
    lh = pts.get("left_hip")
    rh = pts.get("right_hip")
    la = pts.get("left_ankle")
    ra = pts.get("right_ankle")

    if ls and rs and lh and rh:
        shoulder_y = (ls[1] + rs[1]) * 0.5
        hip_y = (lh[1] + rh[1]) * 0.5
        # If ankles are below hips, person is upright; shoulders must be above hips
        ankles_y = None
        if la and ra and la[2] > 0 and ra[2] > 0:
            ankles_y = (la[1] + ra[1]) * 0.5

        if ankles_y is not None and ankles_y > hip_y:
            if shoulder_y > hip_y + 40.0:  # 40 normalized px margin for extreme forward leans
                score -= 0.35
                reasons.append(f"inverted_vertical_orientation: shoulders (y={shoulder_y:.1f}) below hips (y={hip_y:.1f})")

    # 7. Laterality verification
    lat_result = verify_laterality_convention(pts, schema_type="pose17", expected_convention=laterality_convention)
    if not lat_result.is_valid:
        score -= 0.30
        reasons.extend(lat_result.reasons)

    # 8. Limb length & proportion sanity
    torso_scale: Optional[float] = None
    if ls and rs and ls[2] > 0 and rs[2] > 0:
        s_width = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
        if s_width > 5.0:
            torso_scale = s_width

    for u_name, v_name in POSE17_SKELETON_EDGES:
        u_pt = pts.get(u_name)
        v_pt = pts.get(v_name)
        if not u_pt or not v_pt or u_pt[2] == 0 or v_pt[2] == 0:
            continue
        bone_len = math.hypot(v_pt[0] - u_pt[0], v_pt[1] - u_pt[1])
        bone_tag = f"{u_name}-{v_name}"

        # Segment length exceeding 55% of image size (impossible for single limb segment)
        if bone_len > 550.0:
            score -= 0.25
            suspect_bones.append(bone_tag)
            reasons.append(f"excessive_bone_length: {bone_tag} len={bone_len:.1f}")

        # Segment length exceeding 3.2x shoulder width
        if torso_scale is not None and torso_scale > 15.0 and bone_len > 3.2 * torso_scale:
            score -= 0.20
            suspect_bones.append(bone_tag)
            reasons.append(f"disproportionate_bone: {bone_tag} ratio={bone_len / torso_scale:.2f}")

    score = max(0.0, min(1.0, score))
    is_valid = score >= 0.40 and len(suspect_bones) < 3 and not axis_swapped

    return Pose17QualityReport(
        score=round(score, 3),
        is_valid=is_valid,
        needs_refine=(score < refine_score_threshold or bool(reasons)),
        reasons=reasons,
        visible_count=visible_count,
        active_count=active_count,
        suspect_bones=suspect_bones,
        laterality_status=lat_result.detected_convention,
        axis_swap_detected=axis_swapped,
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
    2. Coordinates in bounds [0, 1000].
    3. Coordinate axis swap detection.
    4. Laterality (Viewer convention by VinFast guideline: mattrai on image left).
    5. Eyebrows positioned above corresponding eyes.
    6. Nose bridge (10..13) alignment and central positioning between eyes and mouth.
    7. Eyelid contour vs pupil/iris (eyelid boundaries, upper above lower, no figure-8 crossing).
    8. Outer lip vs inner lip topological constraints (inner lip enclosed inside outer lip).
    9. Head tilt/roll tolerance with conservative geometric checks.
    """
    pts_map = _extract_vf50_points(data)
    reasons: List[str] = []
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
            reasons.append(f"incomplete_component_{comp_name}: {cnt}/{expected_cnt} points")

    if total_pts < VF50_EXPECTED_TOTAL_POINTS:
        score -= 0.25 * ((VF50_EXPECTED_TOTAL_POINTS - total_pts) / 50.0)
        reasons.append(f"invalid_total_points: {total_pts}/{VF50_EXPECTED_TOTAL_POINTS}")

    # 2. Coordinates within image bounds [0, 1000]
    out_of_bounds: List[int] = []
    for p_id, (x, y, vis, conf) in pts_map.items():
        if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            out_of_bounds.append(p_id)
    if out_of_bounds:
        score -= 0.35
        reasons.append(f"landmarks_out_of_bounds: IDs {out_of_bounds[:10]}")

    # 3. Coordinate axis swap check
    axis_swapped, swap_reason = detect_coordinate_axis_swap(pts_map, schema_type="vf50")
    if axis_swapped:
        score -= 0.45
        reasons.append(swap_reason)

    # 4. Laterality verification (Viewer convention: mattrai is on screen left)
    lat_result = verify_laterality_convention(pts_map, schema_type="vf50", expected_convention=laterality_convention)
    if not lat_result.is_valid:
        score -= 0.35
        suspect_components.extend(["mattrai", "matphai"])
        reasons.extend(lat_result.reasons)

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
            reasons.append(f"eyebrow_below_eye: longmaytrai (y={y_eb_left:.1f}) below or touching mattrai (y={y_eye_left:.1f})")

    if right_eb_pts and right_eye_pts:
        y_eb_right = sum(to_face_frame(p)[1] for p in right_eb_pts) / len(right_eb_pts)
        y_eye_right = sum(to_face_frame(p)[1] for p in right_eye_pts) / len(right_eye_pts)
        if y_eb_right > y_eye_right - 3.0:
            score -= 0.20
            suspect_components.append("longmayphai")
            reasons.append(f"eyebrow_below_eye: longmayphai (y={y_eb_right:.1f}) below or touching matphai (y={y_eye_right:.1f})")

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
                reasons.append(f"inverted_eyelid: upper pt {upper_id} below lower pt {lower_id}")

    # Check for self-intersecting eye loops (figure-8 / X-shape)
    if len(left_eye_pts) == 8:
        poly_left_eye = [(pts_map[i][0], pts_map[i][1]) for i in range(14, 22)]
        if _self_intersects(poly_left_eye, closed=True):
            score -= 0.25
            suspect_components.append("mattrai")
            reasons.append("self_intersecting_eye_contour: mattrai forms figure-8")

    if len(right_eye_pts) == 8:
        poly_right_eye = [(pts_map[i][0], pts_map[i][1]) for i in range(22, 30)]
        if _self_intersects(poly_right_eye, closed=True):
            score -= 0.25
            suspect_components.append("matphai")
            reasons.append("self_intersecting_eye_contour: matphai forms figure-8")

    # 7. Nose alignment & position between eyes and mouth
    nose_pts = [pts_map[i] for i in range(10, 14) if i in pts_map]
    if len(nose_pts) == 4:
        # Check monotonic downwards progression along nose bridge
        y_nose_rot = [to_face_frame(pts_map[i])[1] for i in range(10, 14)]
        if not (y_nose_rot[0] <= y_nose_rot[1] + 3.0 <= y_nose_rot[2] + 6.0 <= y_nose_rot[3] + 9.0):
            score -= 0.15
            suspect_components.append("songmui")
            reasons.append("disordered_nose_bridge: points 10..13 not monotonic down bridge")

    mouth_outer_pts = [pts_map[i] for i in range(30, 42) if i in pts_map]
    if 13 in pts_map and left_eye_pts and right_eye_pts and mouth_outer_pts:
        y_nose_base = to_face_frame(pts_map[13])[1]
        y_eyes = (to_face_frame(c_left_eye + (0, 0))[1] + to_face_frame(c_right_eye + (0, 0))[1]) * 0.5
        y_mouth_top = min(to_face_frame(p)[1] for p in mouth_outer_pts)

        if not (y_eyes < y_nose_base < y_mouth_top):
            score -= 0.25
            suspect_components.append("songmui")
            reasons.append(f"nose_vertical_position_invalid: base y={y_nose_base:.1f} not between eyes ({y_eyes:.1f}) and mouth ({y_mouth_top:.1f})")

    # 8. Outer lip vs Inner lip topological constraints
    mouth_inner_pts = [pts_map[i] for i in range(42, 50) if i in pts_map]
    if len(mouth_outer_pts) == 12 and len(mouth_inner_pts) == 8:
        poly_outer = [(pts_map[i][0], pts_map[i][1]) for i in range(30, 42)]
        poly_inner = [(pts_map[i][0], pts_map[i][1]) for i in range(42, 50)]

        area_outer = _polygon_area(poly_outer)
        area_inner = _polygon_area(poly_inner)

        # Topological constraint: inner lip area cannot exceed outer lip area
        if area_inner > area_outer + 5.0:
            score -= 0.35
            suspect_components.append("moitrong")
            reasons.append(f"inner_lip_larger_than_outer: inner={area_inner:.1f} > outer={area_outer:.1f}")

        # Horizontal nesting constraint: inner corners inside outer corners
        x_outer_left = pts_map[30][0]
        x_outer_right = pts_map[36][0]
        x_inner_left = pts_map[42][0]
        x_inner_right = pts_map[46][0]

        if x_inner_left < x_outer_left - 4.0 or x_inner_right > x_outer_right + 4.0:
            score -= 0.25
            suspect_components.append("moitrong")
            reasons.append(f"inner_lip_protrudes_horizontally: inner=[{x_inner_left:.1f},{x_inner_right:.1f}] outer=[{x_outer_left:.1f},{x_outer_right:.1f}]")

        if _self_intersects(poly_outer, closed=True):
            score -= 0.20
            suspect_components.append("moingoai")
            reasons.append("self_intersecting_lip_contour: moingoai self-intersects")
        if _self_intersects(poly_inner, closed=True):
            score -= 0.20
            suspect_components.append("moitrong")
            reasons.append("self_intersecting_lip_contour: moitrong self-intersects")

    score = max(0.0, min(1.0, score))
    is_valid = score >= 0.40 and not axis_swapped and len(suspect_components) < 4

    return VF50QualityReport(
        score=round(score, 3),
        is_valid=is_valid,
        needs_refine=(score < refine_score_threshold or bool(reasons)),
        reasons=reasons,
        point_count=total_pts,
        component_counts=component_counts,
        suspect_components=list(set(suspect_components)),
        laterality_status=lat_result.detected_convention,
        axis_swap_detected=axis_swapped,
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
            )
        conv = laterality_convention or LATERALITY_SUBJECT
        sub_reports = [assess_pose17_quality(it, laterality_convention=conv, refine_score_threshold=refine_score_threshold) for it in items]
        avg_score = sum(r.score for r in sub_reports) / max(1, len(sub_reports))
        reasons: List[str] = []
        suspects: List[str] = []
        for idx, r in enumerate(sub_reports, start=1):
            if not r.is_valid or r.needs_refine:
                label_id = f"person_{idx}"
                suspects.append(label_id)
                reasons.extend(f"{label_id}:{reason}" for reason in r.reasons)
        return QualityReport(
            score=round(avg_score, 3),
            needs_refine=(bool(suspects) or avg_score < refine_score_threshold),
            reasons=reasons,
            suspect_labels=suspects,
            item_count=len(items),
            suspect_count=len(suspects),
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
            )
        conv = laterality_convention or LATERALITY_VIEWER
        face_payload = items if len(items) > 1 else items[0]
        vf_report = assess_vf50_quality(face_payload, laterality_convention=conv, refine_score_threshold=refine_score_threshold)
        reasons = [f"face:{r}" for r in vf_report.reasons]
        suspects = [f"face_{c}" for c in vf_report.suspect_components]
        return QualityReport(
            score=vf_report.score,
            needs_refine=vf_report.needs_refine,
            reasons=reasons,
            suspect_labels=suspects,
            item_count=len(items),
            suspect_count=len(suspects),
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
        )

    scores: List[float] = []
    reasons_list: List[str] = []
    suspects_list: List[str] = []

    for item in items:
        score = 1.0
        item_reasons: List[str] = []
        confidence = getattr(item, "confidence", None)
        if confidence is not None and float(confidence) < min_confidence:
            score -= 0.18
            item_reasons.append("low_confidence")

        if mode in ("rectangle_mask", "box_mask"):
            box = getattr(item, "box_2d", None)
            mask = getattr(item, "instance_mask", None) or getattr(item, "mask", None)
            if not box or not mask or len(mask) < 3:
                score = 0.0
                item_reasons.append("missing_box_or_mask")
            else:
                ymin, xmin, ymax, xmax = map(float, box)
                box_xyxy = (xmin, ymin, xmax, ymax)
                box_area = max(1.0, (xmax - xmin) * (ymax - ymin))
                mask_bbox = _bbox_from_points(mask)
                mask_area = _polygon_area(mask)
                if len(mask) < 6:
                    score -= 0.18
                    item_reasons.append("coarse_mask")
                if mask_bbox is not None and _bbox_iou_xyxy(box_xyxy, mask_bbox) < 0.62:
                    score -= 0.30
                    item_reasons.append("mask_box_mismatch")
                fill_ratio = mask_area / box_area
                if fill_ratio < 0.18 or fill_ratio > 1.20:
                    score -= 0.30
                    item_reasons.append("implausible_mask_fill")
                if _self_intersects(mask, closed=True):
                    score -= 0.30
                    item_reasons.append("self_intersecting_mask")

        elif mode == "polygon_mask":
            poly = getattr(item, "region_polygon", None) or getattr(item, "mask", None)
            if not poly or len(poly) < 3:
                score = 0.0
                item_reasons.append("missing_polygon")
            else:
                area = _polygon_area(poly)
                bbox = _bbox_from_points(poly)
                bbox_area = 0.0
                if bbox is not None:
                    bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                if area < 150.0:
                    score -= 0.28
                    item_reasons.append("tiny_region")
                if len(poly) < 7:
                    score -= 0.22
                    item_reasons.append("coarse_polygon")
                if bbox_area >= 180_000.0 and len(poly) < 16:
                    score -= 0.30
                    item_reasons.append("large_region_too_coarse")
                if _self_intersects(poly, closed=True):
                    score -= 0.35
                    item_reasons.append("self_intersecting_polygon")

        elif mode == "polyline":
            line = getattr(item, "lane_polyline", None) or getattr(item, "mask", None)
            if not line or len(line) < 2:
                score = 0.0
                item_reasons.append("missing_polyline")
            else:
                if len(line) < 3:
                    score -= 0.18
                    item_reasons.append("undersampled_polyline")
                if _line_path_ratio(line) > 3.2:
                    score -= 0.32
                    item_reasons.append("zigzag_polyline")
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

        score = max(0.0, min(1.0, score))
        scores.append(score)
        if score < refine_score_threshold:
            suspects_list.append(str(getattr(item, "label", "unknown")))
            reasons_list.extend(f"{getattr(item, 'label', 'unknown')}:{r}" for r in item_reasons)

    aggregate = sum(scores) / max(1, len(scores))
    return QualityReport(
        score=round(aggregate, 3),
        needs_refine=(bool(suspects_list) or aggregate < refine_score_threshold),
        reasons=reasons_list,
        suspect_labels=suspects_list,
        item_count=len(items),
        suspect_count=len(suspects_list),
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

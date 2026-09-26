"""Human Pose 17 Contract, CVAT Native Skeleton Generation, and Deterministic Quality Gate.

Week 2 Architectural Specification:
1. Canonical COCO 17-Keypoint Topology & Sublabel Schema.
2. Remote Model Contract (Section 9):
   - Normalized coordinates [0..1000]
   - Confidence [0.0..1.0]
   - Visibility flag (0=outside/unlabeled, 1=occluded, 2=visible)
   - Clean JSON structure for multi-person and keypoints.
3. Coordinate Transforms:
   - Normalized [0..1000] -> clamped image pixels [x, y].
4. Native CVAT Skeleton Formatting (Section 6):
   - Parent shape: type="skeleton", label="<parent>", confidence=float, elements=[...]
   - Child elements: type="points", label="<sublabel>", points=[x, y], outside=bool, occluded=bool
5. Visibility and Occlusion Mapping (Section 8):
   - 0 -> outside=True, occluded=False
   - 1 -> outside=False, occluded=True
   - 2 -> outside=False, occluded=False
6. Multi-Person Isolation (Section 12):
   - 0 people -> []
   - 1 person -> 1 skeleton
   - N people -> N distinct skeletons without keypoint cross-contamination.
7. Deterministic Quality Gate (Section 16):
   - Coordinate clamping, minimum visible keypoints, bone length ratio sanity,
     collapse detection, and corrupt pose rejection.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

from core.week2_schema import (
    SchemaValidationError,
    load_pose17,
    load_vf50,
    pose17_coco_keypoints,
    pose17_edges_as_coco_names,
    pose17_keypoint_descriptions,
    vf50_component_configs,
    vf50_edges_by_component,
    vf50_all_edges,
    vf50_point_to_component,
    map_visibility_to_cvat,
    map_cvat_to_visibility,
    normalize_visibility,
    VISIBILITY_OUTSIDE,
    VISIBILITY_OCCLUDED,
    VISIBILITY_VISIBLE,
)

_pose17 = load_pose17()

# VinFast Week-2 HumanPose-17 canonical keypoint sequence derived from authoritative YAML
POSE17_KEYPOINTS: Tuple[str, ...] = pose17_coco_keypoints()
POSE17_KEYPOINTS_SET: FrozenSet[str] = frozenset(POSE17_KEYPOINTS)
KEYPOINT_COUNT: int = len(POSE17_KEYPOINTS)  # exactly 17
KEYPOINT_INDEX_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(POSE17_KEYPOINTS)}

# Canonical 18 bone pairs connecting keypoints in VinFast Week-2 HumanPose-17 topology
# Matches config/week2_pose17.yaml authoritative edge set (including ear-to-shoulder 4->6, 5->7)
POSE17_SKELETON_EDGES: Tuple[Tuple[str, str], ...] = pose17_edges_as_coco_names()

DEFAULT_PARENT_LABEL: str = _pose17.parent_label

# ==============================================================================
# VINFAST HUMANPOSE-17 CANONICAL SPECIFICATION (output_pose17_guideline.txt)
# ==============================================================================
# Sublabels 1..17 strictly mapped to anatomical joints in viewer space:
# Even indices (2, 4, 6, 8, 10, 12, 14, 16) = RIGHT in frame (viewer perspective, larger X)
# Odd indices (3, 5, 7, 9, 11, 13, 15, 17) = LEFT in frame (viewer perspective, smaller X)
VINFAST_POSE17_KEYPOINTS: Tuple[str, ...] = tuple(kp.numeric_name for kp in _pose17.keypoints)
VINFAST_POSE17_INDEX_TO_NAME: Dict[int, str] = {kp.id: kp.semantic_name for kp in _pose17.keypoints}

COCO_NAME_TO_VINFAST_ID: Dict[str, str] = {name: str(kp_id) for name, kp_id in _pose17.coco_name_to_id.items()}
VINFAST_ID_TO_COCO_NAME: Dict[str, str] = {str(kp_id): name for kp_id, name in _pose17.id_to_coco_name.items()}


def denormalize_keypoint(
    x_norm: float,
    y_norm: float,
    width: int,
    height: int,
    *,
    clamp: bool = True,
) -> Tuple[float, float]:
    """Convert normalized [0..1000] coordinates to image pixel space.

    Formula:
        pixel_x = x_norm * width / 1000.0
        pixel_y = y_norm * height / 1000.0

    Args:
        x_norm: Normalized horizontal coordinate in [0, 1000].
        y_norm: Normalized vertical coordinate in [0, 1000].
        width: Image pixel width (> 0).
        height: Image pixel height (> 0).
        clamp: If True, clamps resulting pixels to [0.0, width] and [0.0, height].

    Returns:
        Tuple of (pixel_x, pixel_y) rounded to 2 decimal places.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

    px = (float(x_norm) / 1000.0) * float(width)
    py = (float(y_norm) / 1000.0) * float(height)

    if clamp:
        px = max(0.0, min(float(width), px))
        py = max(0.0, min(float(height), py))

    return round(px, 2), round(py, 2)


def normalize_keypoint(
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    *,
    clamp: bool = True,
) -> Tuple[float, float]:
    """Convert image pixel coordinates back to normalized [0..1000] range."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

    norm_x = (float(pixel_x) / float(width)) * 1000.0
    norm_y = (float(pixel_y) / float(height)) * 1000.0

    if clamp:
        norm_x = max(0.0, min(1000.0, norm_x))
        norm_y = max(0.0, min(1000.0, norm_y))

    return round(norm_x, 1), round(norm_y, 1)


@dataclass
class PoseKeypoint:
    """Represents a single keypoint in normalized [0..1000] coordinates."""
    name: str
    x: float  # Normalized 0..1000
    y: float  # Normalized 0..1000
    visibility: int = VISIBILITY_VISIBLE  # 0, 1, 2
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        self.x = float(max(0.0, min(1000.0, self.x)))
        self.y = float(max(0.0, min(1000.0, self.y)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        if self.visibility not in (VISIBILITY_OUTSIDE, VISIBILITY_OCCLUDED, VISIBILITY_VISIBLE):
            out, occ = map_visibility_to_cvat(self.visibility)
            self.visibility = map_cvat_to_visibility(out, occ)

    @property
    def is_visible(self) -> bool:
        """True if keypoint is labeled and visible."""
        return self.visibility == VISIBILITY_VISIBLE

    @property
    def is_outside(self) -> bool:
        """True if keypoint is outside the image frame or untracked."""
        return self.visibility == VISIBILITY_OUTSIDE

    @property
    def is_occluded(self) -> bool:
        """True if keypoint is present but occluded."""
        return self.visibility == VISIBILITY_OCCLUDED

    def to_cvat_element(self, width: int, height: int) -> Dict[str, Any]:
        """Format as a native CVAT skeleton element shape dictionary.

        Contract (Section 6):
        {
            "label": "<sublabel>",
            "type": "points",
            "points": [pixel_x, pixel_y],
            "outside": bool,
            "occluded": bool,
            "confidence": float
        }
        """
        outside, occluded = map_visibility_to_cvat(self.visibility)
        px, py = denormalize_keypoint(self.x, self.y, width=width, height=height)
        return {
            "label": self.name,
            "type": "points",
            "points": [px, py],
            "outside": outside,
            "occluded": occluded,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class PersonPose17:
    """Represents a single person detection with 17 COCO-topology keypoints."""
    id: Optional[int] = None
    label: str = DEFAULT_PARENT_LABEL
    box_2d: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] in [0..1000]
    confidence: float = 1.0
    keypoints: Dict[str, PoseKeypoint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = str(self.label).strip() or DEFAULT_PARENT_LABEL
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        # Ensure all 17 keypoints are populated
        for name in POSE17_KEYPOINTS:
            if name not in self.keypoints:
                self.keypoints[name] = PoseKeypoint(
                    name=name,
                    x=0.0,
                    y=0.0,
                    visibility=VISIBILITY_OUTSIDE,
                    confidence=0.0,
                )

    @property
    def visible_keypoints_count(self) -> int:
        """Number of keypoints marked visible (visibility=2)."""
        return sum(1 for kp in self.keypoints.values() if kp.is_visible)

    @property
    def active_keypoints_count(self) -> int:
        """Number of keypoints inside image frame (visibility in [1, 2])."""
        return sum(1 for kp in self.keypoints.values() if not kp.is_outside)

    def get_keypoint(self, name: str) -> Optional[PoseKeypoint]:
        """Safely fetch keypoint by name."""
        return self.keypoints.get(name.strip().lower())

    def derive_bbox(self, pad_ratio: float = 0.05) -> Optional[List[int]]:
        """Derive bounding box [ymin, xmin, ymax, xmax] in normalized [0..1000] from active keypoints."""
        active = [kp for kp in self.keypoints.values() if not kp.is_outside]
        if not active:
            return self.box_2d
        xs = [kp.x for kp in active]
        ys = [kp.y for kp in active]
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

    def get_center(self) -> Optional[Tuple[float, float]]:
        """Derive center coordinates (cx, cy) in normalized [0..1000] space."""
        active = [kp for kp in self.keypoints.values() if not kp.is_outside]
        if not active:
            if self.box_2d:
                return ((self.box_2d[1] + self.box_2d[3]) * 0.5, (self.box_2d[0] + self.box_2d[2]) * 0.5)
            return None
        return (
            round(sum(kp.x for kp in active) / len(active), 2),
            round(sum(kp.y for kp in active) / len(active), 2),
        )

    def to_cvat_skeleton(
        self,
        width: int,
        height: int,
        sublabel_names: Optional[Sequence[str]] = None,
        *,
        convention: str = "coco",
    ) -> Dict[str, Any]:
        """Convert to native CVAT skeleton shape dictionary.

        Specification (Section 6 & 12):
        - Parent label: label
        - type: "skeleton"
        - confidence: float
        - elements: exactly 17 points elements ordered canonically.
        - convention: "coco" (default, named keypoints) or "vinfast" (numeric 1..17).
        """
        elements: List[Dict[str, Any]] = []
        if convention == "vinfast":
            # VinFast 1..17 ordering matching output_pose17_guideline.txt Sec 2.2
            for vf_idx in range(1, 18):
                coco_name = VINFAST_ID_TO_COCO_NAME[str(vf_idx)]
                kp = self.keypoints.get(coco_name)
                if kp is None:
                    kp = PoseKeypoint(
                        name=coco_name,
                        x=0.0,
                        y=0.0,
                        visibility=VISIBILITY_OUTSIDE,
                        confidence=0.0,
                    )
                elem = kp.to_cvat_element(width=width, height=height)
                elem["label"] = str(sublabel_names[vf_idx - 1]) if sublabel_names and len(sublabel_names) == KEYPOINT_COUNT else str(vf_idx)
                elements.append(elem)
        else:
            for idx, name in enumerate(POSE17_KEYPOINTS):
                kp = self.keypoints.get(name)
                if kp is None:
                    kp = PoseKeypoint(
                        name=name,
                        x=0.0,
                        y=0.0,
                        visibility=VISIBILITY_OUTSIDE,
                        confidence=0.0,
                    )
                elem = kp.to_cvat_element(width=width, height=height)
                if sublabel_names and len(sublabel_names) == KEYPOINT_COUNT:
                    elem["label"] = str(sublabel_names[idx])
                elements.append(elem)

        res: Dict[str, Any] = {
            "label": self.label,
            "type": "skeleton",
            "confidence": round(self.confidence, 3),
            "elements": elements,
        }
        return res


def sanitize_corner_dump_keypoints(
    person: PersonPose17,
    *,
    corner_box: Tuple[float, float, float, float] = (0.0, 0.0, 35.0, 35.0),
    min_cluster_size: int = 2,
) -> List[str]:
    """Detect and sanitize machine prediction corner dumps (output_pose17_guideline.txt Sec 1.1).

    When vision models fail on joints (observed on ankles/ears in 68% of images), they dump
    untracked joints at the top-left corner [0..35, 0..35].
    These coordinates are sanitized to outside=True, occluded=False, confidence=0.0.

    Returns:
        List of keypoint names that were sanitized.
    """
    xmin, ymin, xmax, ymax = corner_box
    dumped_names: List[str] = []

    for name, kp in person.keypoints.items():
        if not kp.is_outside and (xmin <= kp.x <= xmax) and (ymin <= kp.y <= ymax):
            dumped_names.append(name)

    if len(dumped_names) >= min_cluster_size:
        for name in dumped_names:
            kp = person.keypoints[name]
            kp.visibility = VISIBILITY_OUTSIDE
            kp.confidence = 0.0
            kp.x = 0.0
            kp.y = 0.0

    return dumped_names



@dataclass
class Pose17QualityReport:
    """Report from deterministic geometry quality gate."""
    is_valid: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    visible_count: int = 0
    active_count: int = 0
    suspect_bones: List[str] = field(default_factory=list)


def assess_pose17_quality(
    person: PersonPose17,
    width: int = 1000,
    height: int = 1000,
    *,
    min_visible_keypoints: int = 4,
    min_confidence: float = 0.30,
) -> Pose17QualityReport:
    """Deterministic quality gate for Pose 17 skeletons (Section 16).

    Checks:
    1. Minimum visible keypoints threshold.
    2. Coordinate clamping & bounds verification.
    3. Degenerate collapse (all points collapsing to identical location).
    4. Head / Torso anchor presence.
    5. Bone length ratios and implausible spatial jumps.
    """
    reasons: List[str] = []
    suspect_bones: List[str] = []
    score = 1.0

    kps = person.keypoints
    active_kps = [kp for kp in kps.values() if not kp.is_outside and kp.confidence >= min_confidence]
    visible_kps = [kp for kp in kps.values() if kp.is_visible and kp.confidence >= min_confidence]

    visible_count = len(visible_kps)
    active_count = len(active_kps)

    # 1. Minimum keypoint count check
    if active_count < min_visible_keypoints:
        reasons.append(f"too_few_keypoints: only {active_count} active (required >= {min_visible_keypoints})")
        score = 0.0
        return Pose17QualityReport(
            is_valid=False,
            score=0.0,
            reasons=reasons,
            visible_count=visible_count,
            active_count=active_count,
        )

    # 2. Degenerate collapse check (points collapsed into a single pixel or tiny dot)
    coords = [(kp.x, kp.y) for kp in active_kps]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    bbox_diag = math.hypot(span_x, span_y)

    if bbox_diag < 10.0:  # In normalized 0..1000 space (< 1% image span)
        reasons.append(f"degenerate_collapse: keypoint bounding box diagonal is {bbox_diag:.1f} < 10.0")
        return Pose17QualityReport(
            is_valid=False,
            score=0.0,
            reasons=reasons,
            visible_count=visible_count,
            active_count=active_count,
        )

    # 2b. Corner dump cluster detection (output_pose17_guideline.txt Sec 1.1)
    corner_dumped = [kp for kp in active_kps if kp.x <= 35.0 and kp.y <= 35.0]
    if len(corner_dumped) >= 2:
        reasons.append(
            f"corner_dump_cluster: {len(corner_dumped)} keypoints dumped at top-left corner [0..35, 0..35]"
        )
        score -= 0.30

    # 3. Torso scale estimation
    ls = kps.get("left_shoulder")
    rs = kps.get("right_shoulder")
    lh = kps.get("left_hip")
    rh = kps.get("right_hip")

    torso_scale: Optional[float] = None
    if ls and rs and not ls.is_outside and not rs.is_outside:
        shoulder_width = math.hypot(rs.x - ls.x, rs.y - ls.y)
        if shoulder_width > 5.0:
            torso_scale = shoulder_width

    # 4. Check limb connectivity & bone lengths
    for p1_name, p2_name in POSE17_SKELETON_EDGES:
        p1 = kps.get(p1_name)
        p2 = kps.get(p2_name)
        if not p1 or not p2 or p1.is_outside or p2.is_outside:
            continue

        bone_len = math.hypot(p2.x - p1.x, p2.y - p1.y)

        # Huge jump exceeding 700 normalized units (70% of image) is physically implausible
        if bone_len > 700.0:
            bone_name = f"{p1_name}-{p2_name}"
            suspect_bones.append(bone_name)
            reasons.append(f"excessive_bone_length: {bone_name} length={bone_len:.1f}")
            score -= 0.25

        # Check relative to torso scale if torso is established
        if torso_scale is not None and torso_scale > 15.0:
            # A single limb segment (e.g. wrist to elbow) cannot be 6x the shoulder width
            if bone_len > 6.0 * torso_scale:
                bone_name = f"{p1_name}-{p2_name}"
                suspect_bones.append(bone_name)
                reasons.append(f"disproportionate_bone: {bone_name} ratio={bone_len / torso_scale:.2f}")
                score -= 0.20

    # 5. Arm / Leg proportion sanity checks
    # Left arm: shoulder->elbow vs elbow->wrist
    le = kps.get("left_elbow")
    lw = kps.get("left_wrist")
    if ls and le and lw and not ls.is_outside and not le.is_outside and not lw.is_outside:
        upper = math.hypot(le.x - ls.x, le.y - ls.y)
        fore = math.hypot(lw.x - le.x, lw.y - le.y)
        if upper > 15.0 and fore > 15.0:
            ratio = upper / fore
            if ratio < 0.15 or ratio > 6.5:
                suspect_bones.append("left_arm_proportion")
                reasons.append(f"implausible_left_arm_ratio: {ratio:.2f}")
                score -= 0.15

    # Right arm: shoulder->elbow vs elbow->wrist
    re_pt = kps.get("right_elbow")
    rw = kps.get("right_wrist")
    if rs and re_pt and rw and not rs.is_outside and not re_pt.is_outside and not rw.is_outside:
        upper = math.hypot(re_pt.x - rs.x, re_pt.y - rs.y)
        fore = math.hypot(rw.x - re_pt.x, rw.y - re_pt.y)
        if upper > 15.0 and fore > 15.0:
            ratio = upper / fore
            if ratio < 0.15 or ratio > 6.5:
                suspect_bones.append("right_arm_proportion")
                reasons.append(f"implausible_right_arm_ratio: {ratio:.2f}")
                score -= 0.15

    score = max(0.0, min(1.0, score))
    is_valid = score >= 0.40 and len(suspect_bones) < 3

    return Pose17QualityReport(
        is_valid=is_valid,
        score=round(score, 3),
        reasons=reasons,
        visible_count=visible_count,
        active_count=active_count,
        suspect_bones=suspect_bones,
    )


def parse_pose17_response(
    raw_response: Union[str, Dict[str, Any], List[Any]],
    *,
    default_label: str = DEFAULT_PARENT_LABEL,
) -> List[PersonPose17]:
    """Parse remote vision model output into validated PersonPose17 instances.

    Handles (Section 9 & 12):
    - Multi-person support: 0 -> [], 1 -> [p1], N -> [p1..pN].
    - JSON extraction from markdown fences or text blocks.
    - Multiple input schemas:
        Format A (canonical dictionary of people):
            {"people": [{"id": 1, "keypoints": [...]}, ...]}
        Format B (direct list):
            [{"id": 1, "keypoints": [...]}, ...]
        Format C (single person dictionary):
            {"keypoints": [...]}
    - Keypoint formats:
        List of objects: [{"name": "nose", "point": [x, y], "visibility": 2}, ...]
        List of objects: [{"name": "nose", "x": x, "y": y, "visibility": 2}, ...]
        Dict mapping: {"nose": [x, y, v], ...} or {"nose": {"point": [x, y], ...}}
        Ordered tuple list: [[x, y, v], ...] matching POSE17_KEYPOINTS order.
    """
    if not raw_response:
        return []

    data: Any = raw_response

    # If raw response is string, parse JSON
    if isinstance(data, str):
        cleaned = data.strip()
        # Strip markdown fences
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()

        # Extract outer JSON structure if embedded in commentary
        if not (cleaned.startswith("{") or cleaned.startswith("[")):
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
            if match:
                cleaned = match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response for Pose 17: {e}")
            return []

    people_raw: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        if "people" in data and isinstance(data["people"], list):
            people_raw = [p for p in data["people"] if isinstance(p, dict)]
        elif "persons" in data and isinstance(data["persons"], list):
            people_raw = [p for p in data["persons"] if isinstance(p, dict)]
        elif "keypoints" in data:
            # Single person payload
            people_raw = [data]
        elif "predictions" in data and isinstance(data["predictions"], list):
            people_raw = [p for p in data["predictions"] if isinstance(p, dict)]
    elif isinstance(data, list):
        people_raw = [p for p in data if isinstance(p, dict)]

    parsed_people: List[PersonPose17] = []

    for idx, p_item in enumerate(people_raw, start=1):
        person_id = p_item.get("id") or idx
        label = str(p_item.get("label") or default_label).strip()
        confidence = float(p_item.get("confidence", 1.0))
        box_2d = p_item.get("box_2d")
        if isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
            try:
                box_2d = [int(v) for v in box_2d]
            except (ValueError, TypeError):
                box_2d = None
        else:
            box_2d = None

        raw_kps = p_item.get("keypoints") or p_item.get("elements") or p_item.get("points")
        kps_dict: Dict[str, PoseKeypoint] = {}

        if isinstance(raw_kps, dict):
            # Dict mapping name -> details
            for name, details in raw_kps.items():
                clean_name = str(name).strip().lower()
                if clean_name not in POSE17_KEYPOINTS_SET:
                    continue
                kp = _parse_single_keypoint(clean_name, details)
                if kp:
                    kps_dict[clean_name] = kp

        elif isinstance(raw_kps, list):
            # List of keypoint elements
            if raw_kps and isinstance(raw_kps[0], dict):
                for item in raw_kps:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("label") or "").strip().lower()
                    if name in POSE17_KEYPOINTS_SET:
                        kp = _parse_single_keypoint(name, item)
                        if kp:
                            kps_dict[name] = kp
            elif raw_kps and isinstance(raw_kps[0], (list, tuple)):
                # List of points in canonical COCO order: [[x, y, v], ...]
                for i, pt_val in enumerate(raw_kps):
                    if i >= len(POSE17_KEYPOINTS):
                        break
                    name = POSE17_KEYPOINTS[i]
                    kp = _parse_sequence_keypoint(name, pt_val)
                    if kp:
                        kps_dict[name] = kp

        # Multi-person isolation guarantee (Section 12):
        # Create a fresh PersonPose17 with completely independent keypoint objects
        person = PersonPose17(
            id=person_id,
            label=label,
            box_2d=box_2d,
            confidence=confidence,
            keypoints=kps_dict,
        )
        parsed_people.append(person)

    return parsed_people


def _parse_single_keypoint(name: str, data: Any) -> Optional[PoseKeypoint]:
    """Helper to parse a single keypoint from dictionary or sequence."""
    if isinstance(data, dict):
        # Extract x, y
        pt = data.get("point") or data.get("points")
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x, y = float(pt[0]), float(pt[1])
        else:
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))

        vis_val = data.get("visibility")
        if vis_val is None:
            # Check outside / occluded flags directly
            outside = bool(data.get("outside", False))
            occluded = bool(data.get("occluded", False))
            vis_val = map_cvat_to_visibility(outside, occluded)
        else:
            vis_val = normalize_visibility(vis_val, strict=False, default=VISIBILITY_VISIBLE)

        conf = float(data.get("confidence", 1.0))
        return PoseKeypoint(
            name=name,
            x=x,
            y=y,
            visibility=vis_val,
            confidence=conf,
        )

    if isinstance(data, (list, tuple)):
        return _parse_sequence_keypoint(name, data)

    return None


def _parse_sequence_keypoint(name: str, seq: Sequence[Any]) -> Optional[PoseKeypoint]:
    """Helper to parse keypoint from sequence [x, y, (vis), (conf)]."""
    if len(seq) < 2:
        return None
    try:
        x = float(seq[0])
        y = float(seq[1])
        raw_vis = seq[2] if len(seq) >= 3 else VISIBILITY_VISIBLE
        vis = normalize_visibility(raw_vis, strict=False, default=VISIBILITY_VISIBLE)
        conf = float(seq[3]) if len(seq) >= 4 else 1.0
        return PoseKeypoint(
            name=name,
            x=x,
            y=y,
            visibility=vis,
            confidence=conf,
        )
    except (ValueError, TypeError):
        return None


def poses_to_cvat_skeletons(
    people: Sequence[PersonPose17],
    width: int,
    height: int,
    *,
    filter_corrupt: bool = True,
    min_visible_keypoints: int = 4,
    convention: str = "coco",
    sanitize_corner_dumps: bool = False,
) -> List[Dict[str, Any]]:
    """Convert a sequence of PersonPose17 to native CVAT skeleton shape dictionaries.

    Ensures:
    - 0 people -> []
    - 1 person -> 1 skeleton
    - N people -> N isolated skeletons without cross-contamination.
    - Each skeleton element list contains all 17 canonical keypoints in order.
    - Deterministic quality gate filters corrupt poses if filter_corrupt is True.

    Args:
        people: Sequence of PersonPose17 instances.
        width: Image width in pixels.
        height: Image height in pixels.
        filter_corrupt: If True, filters out anatomically corrupt or degenerate poses.
        min_visible_keypoints: Minimum keypoint threshold for quality gate.
        convention: "coco" (default, named keypoints) or "vinfast" (numeric 1..17).
        sanitize_corner_dumps: If True, sanitizes top-left [0..35, 0..35] cluster glitches.

    Returns:
        List of CVAT skeleton shape dictionaries.
    """
    if not people:
        return []

    skeletons: List[Dict[str, Any]] = []

    for person in people:
        if sanitize_corner_dumps:
            sanitize_corner_dump_keypoints(person)

        if filter_corrupt:
            report = assess_pose17_quality(
                person,
                width=width,
                height=height,
                min_visible_keypoints=min_visible_keypoints,
            )
            if not report.is_valid:
                logger.warning(
                    f"Dropped corrupt pose (person id={person.id}): reasons={report.reasons}"
                )
                continue

        skeleton_dict = person.to_cvat_skeleton(width=width, height=height, convention=convention)
        skeletons.append(skeleton_dict)

    return skeletons


POSE17_OCCLUSION_INSTRUCTIONS: str = (
    "Physical Occlusion vs Optical Blur (Confidence != Occlusion!):\n"
    "- Physical Occlusion (visibility = 1): A keypoint is physically occluded / obstructed by an opaque physical object "
    "(e.g. steering wheel, seat back, headrest, dashboard, center console, door trim, passenger body, or own torso). "
    "Predict its estimated true anatomical position BEHIND the occluder (do NOT place the occluded joint on the occluder's border/rim) "
    "and set visibility = 1 (occluded).\n"
    "  * RIGHT_WRIST hidden behind steering wheel -> estimate true wrist center behind wheel (NOT on wheel rim) -> visibility = 1\n"
    "  * RIGHT_ELBOW behind seat back / door -> estimate anatomical elbow location -> visibility = 1\n"
    "  * HIP hidden by car seat cushion -> estimate anatomical hip center -> visibility = 1\n"
    "  * KNEE behind dashboard / lower steering column -> estimate anatomical knee -> visibility = 1\n"
    "- Optical Degradation / Low Confidence (visibility = 2): If a keypoint is within direct line of sight but degraded by "
    "motion blur, camera defocus, dark cabin lighting, deep shadows, or sensor noise, it is STILL VISIBLE: set visibility = 2. "
    "Express visual uncertainty through a lower 'confidence' score (e.g. 0.35 - 0.70), NOT by setting visibility = 1. "
    "Optical blur is NOT occlusion; Confidence != Occlusion!\n"
    "- Outside Frame / Unlabelable (visibility = 0): Keypoints completely outside the camera's field of view, cropped out, or beyond image boundary.\n"
    "  * ANKLE or FOOT cut off below image border -> visibility = 0\n"
    "  * Rule: Physical occlusion within frame & inferable -> visibility = 1. visibility = 0 is ONLY for points outside image frame / unlabelable.\n"
)


def build_pose17_prompt(
    parent_label: Union[str, Sequence[str]] = DEFAULT_PARENT_LABEL,
    keypoints: Optional[Sequence[str]] = None,
) -> str:
    """Generate strict, deterministic prompt for 9Router vision inference for Pose 17.

    Derives keypoints, laterality semantics, and edge connectivity directly from canonical load_pose17().
    """
    if isinstance(parent_label, (list, tuple)):
        keypoints = parent_label
        parent_label = DEFAULT_PARENT_LABEL

    schema = load_pose17()
    effective_keypoints = tuple(keypoints) if keypoints is not None else tuple(schema.id_to_coco_name[kp.id] for kp in schema.keypoints)
    kps_str = ", ".join(f'"{k}"' for k in effective_keypoints)

    right_kps = [schema.id_to_coco_name[i] for i in sorted(schema.right_keypoint_ids) if schema.id_to_coco_name[i] in effective_keypoints]
    left_kps = [schema.id_to_coco_name[i] for i in sorted(schema.left_keypoint_ids) if schema.id_to_coco_name[i] in effective_keypoints]
    right_str = ", ".join(right_kps)
    left_str = ", ".join(left_kps)

    kp_desc_lines = []
    for kp in schema.keypoints:
        coco_name = schema.id_to_coco_name[kp.id]
        if coco_name in effective_keypoints:
            extra = f" - {kp.anatomical}" if getattr(kp, "anatomical", None) else ""
            kp_desc_lines.append(f"  * {coco_name}: {kp.semantic_name}{extra} [{kp.side}]")
    kp_desc_block = "\n".join(kp_desc_lines)

    laterality_explanation = (
        "VinFast Viewer-Perspective Convention (MANDATORY):\n"
        "VinFast Week-2 HumanPose-17 topology uses explicit VinFast viewer-space point naming below (RIGHT/LEFT refer to displayed image sides):\n"
        f"- 'right_*' ({right_str}) refer to the VIEWER'S RIGHT side of the displayed image/frame (larger X coordinate).\n"
        "  For example, 'right_eye' means the eye appearing on the RIGHT side of the image, NOT the anatomical right of the person.\n"
        "  Similarly, right_ear, right_shoulder, right_elbow, right_wrist, right_hip, right_knee, right_ankle are on the image's right.\n"
        f"- 'left_*' ({left_str}) refer to the VIEWER'S LEFT side of the displayed image/frame (smaller X coordinate).\n"
        "  For example, 'left_eye' means the eye appearing on the LEFT side of the image.\n"
        "- 'nose': center line of the face.\n"
    ) if schema.laterality_convention == "viewer" else ""

    return (
        f"Perform multi-person 2D Human Pose Estimation (VinFast Week-2 HumanPose-17 topology) on this image.\n"
        f"Parent label is '{parent_label}'.\n"
        f"Detect all persons and estimate their 17 keypoints: [{kps_str}].\n"
        f"Canonical Keypoint Metadata:\n{kp_desc_block}\n"
        f"{laterality_explanation}"
        "Kinematic Chain & Cabin Constraints:\n"
        "- Enforce strict anatomical connectivity: shoulder -> elbow -> wrist, and hip -> knee -> ankle.\n"
        "- Wrists and elbows MUST attach to their corresponding limb; NEVER predict floating wrists or elbows in the cabin background, roof lining, or car seats.\n"
        "- If a person is seated (e.g. driver or passenger) and lower limbs or hands are physically occluded behind steering wheel, seat back, or dashboard within the frame, set visibility flag = 1 (occluded) at their estimated true anatomical location. Do NOT hallucinate floating joints. Set visibility flag = 0 (outside) ONLY if the keypoint is completely outside the image frame or beyond the field of view.\n"
        f"{POSE17_OCCLUSION_INSTRUCTIONS}"
        "Coordinate & Visibility Convention:\n"
        "- Normalize all keypoint (x, y) coordinates to integers in [0, 1000] range relative to image width and height.\n"
        "- Bounding box 'box_2d': [ymin, xmin, ymax, xmax] in [0, 1000].\n"
        "- Visibility flag: 0 = outside image frame, 1 = present but occluded, 2 = clearly visible.\n"
        "- Output MUST be strict, valid JSON with no markdown and no extraneous text:\n"
        "{\n"
        '  "people": [\n'
        "    {\n"
        '      "id": 1,\n'
        f'      "label": "{parent_label}",\n'
        '      "confidence": 0.95,\n'
        '      "box_2d": [ymin, xmin, ymax, xmax],\n'
        '      "keypoints": {\n'
        '        "nose": [x, y, 2],\n'
        '        "right_eye": [x, y, 2],\n'
        '        "left_eye": [x, y, 2],\n'
        "        ...\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        'If no persons are present, return {"people": []}.'
    )


def build_pose17_crop_prompt(keypoints: Optional[Sequence[str]] = None) -> str:
    """Build targeted prompt for high-resolution person crop refinement (Pass 2).

    Keypoints, perspective conventions, and occlusion rules are derived from canonical schema metadata.
    """
    schema = load_pose17()
    effective_keypoints = tuple(keypoints) if keypoints is not None else tuple(schema.id_to_coco_name[kp.id] for kp in schema.keypoints)
    kps_str = ", ".join(f'"{k}"' for k in effective_keypoints)

    right_kps = [schema.id_to_coco_name[i] for i in sorted(schema.right_keypoint_ids) if schema.id_to_coco_name[i] in effective_keypoints]
    left_kps = [schema.id_to_coco_name[i] for i in sorted(schema.left_keypoint_ids) if schema.id_to_coco_name[i] in effective_keypoints]
    right_str = ", ".join(right_kps)
    left_str = ", ".join(left_kps)

    laterality_explanation = (
        "VinFast Viewer-Perspective Convention (MANDATORY):\n"
        "VinFast Week-2 HumanPose-17 topology uses explicit VinFast viewer-space point naming (RIGHT/LEFT refer to displayed image sides):\n"
        f"- 'right_*' ({right_str}) refer to the VIEWER'S RIGHT side of the displayed image/crop (larger X coordinate).\n"
        "  For example, 'right_eye' means the eye appearing on the RIGHT side of the image/crop, NOT the anatomical right of the person.\n"
        "  Similarly, right_ear, right_shoulder, right_elbow, right_wrist, right_hip, right_knee, right_ankle are on the crop's right.\n"
        f"- 'left_*' ({left_str}) refer to the VIEWER'S LEFT side of the displayed image/crop (smaller X coordinate).\n"
        "  For example, 'left_eye' means the eye appearing on the LEFT side of the image/crop.\n"
        "- 'nose': center line of the face.\n"
    ) if schema.laterality_convention == "viewer" else ""

    return (
        "High-resolution close-up person crop refinement.\n"
        f"Detect exactly the 17 keypoints according to VinFast Week-2 HumanPose-17 topology for the single person in this cropped image: [{kps_str}].\n"
        f"{laterality_explanation}"
        "Kinematic Chain & Anti-Floating Constraints:\n"
        "- Strict limb connectivity: shoulder -> elbow -> wrist, and hip -> knee -> ankle. No floating joints!\n"
        "- Coordinates: normalized integers [x, y] in [0, 1000] relative to THIS CROP.\n"
        f"{POSE17_OCCLUSION_INSTRUCTIONS}"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "id": 1,\n'
        '  "confidence": 0.98,\n'
        '  "keypoints": {\n'
        '    "nose": [x, y, 2],\n'
        "    ...\n"
        "  }\n"
        "}\n"
    )


def reproject_crop_point(
    x_crop: float,
    y_crop: float,
    crop_bbox: Sequence[Union[int, float]],
    *,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
    crop_window_px: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[float, float]:
    """Reproject normalized point [0..1000] from crop coordinates to full image [0..1000] space.

    Args:
        x_crop: Horizontal coordinate in crop [0..1000].
        y_crop: Vertical coordinate in crop [0..1000].
        crop_bbox: Full-image normalized crop box [ymin, xmin, ymax, xmax] in [0..1000].
        orig_w: Optional image width in pixels.
        orig_h: Optional image height in pixels.
        crop_window_px: Optional pixel crop window [px1, py1, px2, py2].

    Returns:
        Tuple of (norm_x, norm_y) clamped to [0.0, 1000.0] and rounded to 1 decimal place.
    """
    c_ymin, c_xmin, c_ymax, c_xmax = [float(v) for v in crop_bbox]
    if (
        crop_window_px is not None
        and orig_w is not None
        and orig_h is not None
        and orig_w > 0
        and orig_h > 0
    ):
        px1, py1, px2, py2 = crop_window_px
        crop_w = max(1, px2 - px1)
        crop_h = max(1, py2 - py1)
        x_px = px1 + (float(x_crop) / 1000.0) * crop_w
        y_px = py1 + (float(y_crop) / 1000.0) * crop_h
        norm_x = max(0.0, min(1000.0, (x_px / float(orig_w)) * 1000.0))
        norm_y = max(0.0, min(1000.0, (y_px / float(orig_h)) * 1000.0))
    else:
        crop_w_norm = max(0.0, c_xmax - c_xmin)
        crop_h_norm = max(0.0, c_ymax - c_ymin)
        norm_x = max(0.0, min(1000.0, c_xmin + (float(x_crop) / 1000.0) * crop_w_norm))
        norm_y = max(0.0, min(1000.0, c_ymin + (float(y_crop) / 1000.0) * crop_h_norm))

    return round(norm_x, 1), round(norm_y, 1)


def merge_refined_keypoint(
    name: str,
    pass1_pt: Optional[Sequence[Any]],
    pass2_pt: Optional[Sequence[Any]],
    crop_bbox: Optional[Sequence[Union[int, float]]] = None,
    *,
    crop_box_norm: Optional[Sequence[Union[int, float]]] = None,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
    crop_window_px: Optional[Tuple[int, int, int, int]] = None,
    pass2_is_reprojected: bool = False,
    strict_vis: bool = False,
) -> Optional[List[Any]]:
    """Merge a single keypoint between Pass 1 global detection and Pass 2 crop refinement.

    Implements the canonical 4-case visibility merge policy:
      - CASE A: Pass 2 point is outside crop window (vis == 0), and Pass 1 coordinate
        lies OUTSIDE the crop bounds (crop did not cover the full limb).
        -> Retain Pass 1 detection if geometrically plausible.
      - CASE B: Pass 2 says point is physically occluded but inferable (vis == 1).
        -> Use Pass 2 vis=1 and refined coordinate.
      - CASE C: Pass 2 says point cannot be labeled / outside full image (vis == 0),
        and Pass 1 coordinate lies INSIDE the crop bounds (Pass 2 actually inspected this region).
        -> Do NOT restore Pass 1! Respect Pass 2's determination (vis = 0).
      - CASE D: Pass 2 response omitted the keypoint entirely (None, missing, invalid).
        -> Fallback to Pass 1.
      - Happy path: Pass 2 says point is clearly visible (vis == 2).
        -> Use Pass 2 vis=2 and refined coordinate.

    Args:
        name: Canonical keypoint name (e.g. 'right_wrist').
        pass1_pt: Pass 1 keypoint [x, y, (vis), (conf)] in full image normalized [0..1000] space.
        pass2_pt: Pass 2 keypoint [x, y, (vis), (conf)]. In crop-relative [0..1000] space
            (if pass2_is_reprojected=False) or full image [0..1000] space (if pass2_is_reprojected=True).
        crop_bbox: Crop bounding box [ymin, xmin, ymax, xmax] in full-image normalized [0..1000] space.
        crop_box_norm: Alias for crop_bbox.
        orig_w: Optional full image width in pixels.
        orig_h: Optional full image height in pixels.
        crop_window_px: Optional crop window in full image pixels [px1, py1, px2, py2].
        pass2_is_reprojected: If True, pass2_pt is already in full image [0..1000] space.
        strict_vis: If True, enforce strict visibility normalization.

    Returns:
        Merged keypoint [norm_x, norm_y, vis] in full image [0..1000] space, or None.
    """
    effective_box = crop_bbox if crop_bbox is not None else crop_box_norm
    if effective_box is None or len(effective_box) < 4:
        raise ValueError("crop_bbox or crop_box_norm [ymin, xmin, ymax, xmax] must be provided")

    c_ymin, c_xmin, c_ymax, c_xmax = [float(v) for v in effective_box[:4]]

    # Parse Pass 1
    has_p1 = False
    p1_x, p1_y = 0.0, 0.0
    p1_vis = VISIBILITY_OUTSIDE
    if pass1_pt is not None and isinstance(pass1_pt, (list, tuple)) and len(pass1_pt) >= 2:
        try:
            p1_x = float(pass1_pt[0])
            p1_y = float(pass1_pt[1])
            raw_vis1 = pass1_pt[2] if len(pass1_pt) > 2 else VISIBILITY_VISIBLE
            p1_vis = normalize_visibility(raw_vis1, strict=strict_vis, default=VISIBILITY_VISIBLE)
            has_p1 = True
        except (ValueError, TypeError):
            if strict_vis:
                raise
            has_p1 = False

    # Parse Pass 2
    has_p2 = False
    p2_x, p2_y = 0.0, 0.0
    p2_vis = VISIBILITY_OUTSIDE
    if pass2_pt is not None and isinstance(pass2_pt, (list, tuple)) and len(pass2_pt) >= 2:
        try:
            p2_x = float(pass2_pt[0])
            p2_y = float(pass2_pt[1])
            raw_vis2 = pass2_pt[2] if len(pass2_pt) > 2 else VISIBILITY_VISIBLE
            if isinstance(raw_vis2, (int, float)) and raw_vis2 < 0:
                p2_vis = VISIBILITY_OUTSIDE
            elif isinstance(raw_vis2, str) and raw_vis2.strip() in ("-1", "-1.0"):
                p2_vis = VISIBILITY_OUTSIDE
            else:
                p2_vis = normalize_visibility(raw_vis2, strict=strict_vis, default=VISIBILITY_VISIBLE)
            has_p2 = True
        except (ValueError, TypeError):
            if strict_vis:
                raise
            has_p2 = False

    # CASE D: Pass 2 response omitted the keypoint entirely
    if not has_p2:
        if has_p1:
            return [round(p1_x, 1), round(p1_y, 1), p1_vis]
        return None

    # Reproject Pass 2 coordinates to full image if needed
    if pass2_is_reprojected:
        norm_x = max(0.0, min(1000.0, p2_x))
        norm_y = max(0.0, min(1000.0, p2_y))
    else:
        norm_x, norm_y = reproject_crop_point(
            p2_x,
            p2_y,
            effective_box,
            orig_w=orig_w,
            orig_h=orig_h,
            crop_window_px=crop_window_px,
        )

    # Happy Path: Pass 2 says clearly visible (vis == 2)
    if p2_vis == VISIBILITY_VISIBLE:
        return [round(norm_x, 1), round(norm_y, 1), VISIBILITY_VISIBLE]

    # CASE B: Pass 2 says physically occluded but inferable (vis == 1)
    if p2_vis == VISIBILITY_OCCLUDED:
        return [round(norm_x, 1), round(norm_y, 1), VISIBILITY_OCCLUDED]

    # Pass 2 says point is outside / cannot be labeled (vis == 0)
    # Check if Pass 1 had an active detection
    if not has_p1 or p1_vis == VISIBILITY_OUTSIDE:
        return [round(norm_x, 1), round(norm_y, 1), VISIBILITY_OUTSIDE]

    # Pass 1 was visible or occluded (p1_vis > 0)
    p1_inside_crop = (
        (c_xmin - 5.0) <= p1_x <= (c_xmax + 5.0)
        and (c_ymin - 5.0) <= p1_y <= (c_ymax + 5.0)
    )
    p1_plausible = (
        0.0 <= p1_x <= 1000.0
        and 0.0 <= p1_y <= 1000.0
        and not math.isnan(p1_x)
        and not math.isnan(p1_y)
    )

    if not p1_inside_crop and p1_plausible:
        # CASE A: Pass 2 point is outside person crop because crop did not cover the full limb
        # (Pass 1 coordinate lies OUTSIDE the crop bounds).
        # -> Retain Pass 1 if geometrically plausible.
        return [round(p1_x, 1), round(p1_y, 1), p1_vis]
    else:
        # CASE C: Pass 2 says point cannot be labeled / outside full image (vis == 0)
        # and Pass 1 coordinate lies INSIDE the crop bounds (Pass 2 actually inspected this region).
        # -> Do NOT restore Pass 1!
        return [round(norm_x, 1), round(norm_y, 1), VISIBILITY_OUTSIDE]


def generate_cvat_pose17_svg(
    sublabel_names: Optional[Sequence[str]] = None,
    numeric_sublabels: bool = True,
) -> str:
    """Generate the canonical SVG topology string required by CVAT function.yaml spec.

    Matches CVAT lambda_manager/views.py expectations (lines and circles with data-node-id).
    Derives keypoint labels directly from canonical load_pose17() schema.
    """
    from core.week2_schema import load_pose17
    schema = load_pose17()
    if sublabel_names and len(sublabel_names) == KEYPOINT_COUNT:
        labels = list(sublabel_names)
    elif numeric_sublabels:
        labels = [kp.numeric_name for kp in schema.keypoints]
    else:
        labels = [schema.id_to_coco_name[kp.id] for kp in schema.keypoints]

    nodes_info = [
        (1, 48.87604904174805, 9.485294342041016),    # 1: nose
        (2, 51.2289924621582, 7.636554718017578),     # 2: right_eye (viewer right / cx > 50)
        (3, 47.195377349853516, 7.636554718017578),    # 3: left_eye (viewer left / cx < 50)
        (4, 54.25419998168945, 7.804621696472168),     # 4: right_ear (viewer right / cx > 50)
        (5, 44.170169830322266, 7.804621696472168),    # 5: left_ear (viewer left / cx < 50)
        (6, 60.80882263183594, 19.90546226501465),     # 6: right_shoulder (viewer right / cx > 50)
        (7, 37.78361511230469, 20.409664154052734),    # 7: left_shoulder (viewer left / cx < 50)
        (8, 63.83403396606445, 34.023109436035156),    # 8: right_elbow (viewer right / cx > 50)
        (9, 35.93487548828125, 34.35924530029297),     # 9: left_elbow (viewer left / cx < 50)
        (10, 66.85924530029297, 47.132354736328125),   # 10: right_wrist (viewer right / cx > 50)
        (11, 33.918067932128906, 47.46848678588867),   # 11: left_wrist (viewer left / cx < 50)
        (12, 57.11134338378906, 49.65336227416992),    # 12: right_hip (viewer right / cx > 50)
        (13, 44.00210189819336, 50.157562255859375),   # 13: left_hip (viewer left / cx < 50)
        (14, 58.119747161865234, 71.16596984863281),   # 14: right_knee (viewer right / cx > 50)
        (15, 44.338233947753906, 70.6617660522461),    # 15: left_knee (viewer left / cx < 50)
        (16, 57.78361511230469, 87.97268676757812),    # 16: right_ankle (viewer right / cx > 50)
        (17, 46.186973571777344, 92.51050567626953),   # 17: left_ankle (viewer left / cx < 50)
    ]

    # Edges matching config/week2_pose17.yaml authoritative 18-edge topology derived from schema
    edges_pairs = [list(e) for e in schema.edges]
    node_coords = {nid: (cx, cy) for nid, cx, cy in nodes_info}

    lines = [
        f'<line x1="{node_coords[nf][0]}" y1="{node_coords[nf][1]}" x2="{node_coords[nt][0]}" y2="{node_coords[nt][1]}" data-type="edge" data-node-from="{nf}" data-node-to="{nt}"></line>'
        for nf, nt in edges_pairs
    ]
    circles = [
        f'<circle r="0.75" cx="{cx}" cy="{cy}" data-type="element node" data-element-id="{nid}" data-node-id="{nid}" data-label-name="{labels[nid-1]}"></circle>'
        for nid, cx, cy in nodes_info
    ]
    inner_svg = "".join(lines + circles)
    return f'<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">{inner_svg}</svg>'


def build_cvat_pose17_spec(
    parent_label: str = DEFAULT_PARENT_LABEL,
    *,
    label_id: int = 1,
    sublabel_names: Optional[Sequence[str]] = None,
    include_svg: bool = True,
    numeric_sublabels: bool = True,
) -> Dict[str, Any]:
    """Build CVAT function.yaml annotations spec item for Pose 17 skeleton.

    Derives directly from canonical load_pose17() schema.

    Args:
        parent_label: Name of parent skeleton label (defaults to schema parent_label 'person').
        label_id: Unique integer ID for the parent skeleton spec entry.
        sublabel_names: Optional custom sublabel names (defaults to canonical VinFast sequence from schema).
        include_svg: If True, populates mandatory 'svg' attribute for CVAT Lambda Manager.
        numeric_sublabels: If True, defaults sublabel names to canonical numeric strings '1'..'17'.
    """
    from core.week2_schema import load_pose17
    schema = load_pose17()
    effective_parent = parent_label if parent_label != DEFAULT_PARENT_LABEL else schema.parent_label
    if sublabel_names and len(sublabel_names) == len(schema.keypoints):
        names = list(sublabel_names)
    elif numeric_sublabels:
        names = [kp.numeric_name for kp in schema.keypoints]
    else:
        names = [schema.id_to_coco_name[kp.id] for kp in schema.keypoints]

    sublabels = [
        {"id": kp.id, "name": name, "type": "points", "attributes": []}
        for kp, name in zip(schema.keypoints, names)
    ]
    spec_dict: Dict[str, Any] = {
        "id": label_id,
        "name": effective_parent,
        "type": "skeleton",
        "attributes": [],
        "sublabels": sublabels,
    }
    if include_svg:
        spec_dict["svg"] = generate_cvat_pose17_svg(sublabel_names=names, numeric_sublabels=numeric_sublabels)
    return spec_dict


# ==============================================================================
# VF-50 FACE LANDMARK CANONICAL SPECIFICATION & SKELETON CONTRACT
# ==============================================================================

_vf50 = load_vf50()

# Exact 7 component skeletons in canonical topological order from config/week2_vf50.yaml
VF50_COMPONENT_NAMES: Tuple[str, ...] = _vf50.component_names
VF50_POINTS_COUNT: int = _vf50.total_points

# Global point IDs as string numbers "0".."49" conforming to CVAT sublabels
VF50_POINT_NAMES: Tuple[str, ...] = tuple(kp.numeric_name for kp in _vf50.all_keypoints)

# Component boundaries and properties derived from load_vf50()
VF50_COMPONENT_CONFIG: Dict[str, Dict[str, Any]] = vf50_component_configs()

# Reverse mapping: point ID (0..49) -> component name
VF50_POINT_TO_COMPONENT: Dict[int, str] = vf50_point_to_component()

# Canonical edges for each component (conforming to SVG skeleton graph)
VF50_EDGES_BY_COMPONENT: Dict[str, Tuple[Tuple[int, int], ...]] = vf50_edges_by_component()

# All 47 edges across all 7 skeletons
VF50_ALL_EDGES: Tuple[Tuple[int, int], ...] = vf50_all_edges()
if len(VF50_ALL_EDGES) != 47:
    raise SchemaValidationError(f"VF-50 must have exactly 47 edges, got {len(VF50_ALL_EDGES)}")

# The 12 canonical anchor points defined in VinFast Guideline Section 5.5
VF50_ANCHOR_POINTS: Tuple[int, ...] = (0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36)

VF50_LATERALITY: str = "image_perspective"  # Left on image is mattrai/longmaytrai (x_left < x_right)

# Anatomically-positioned VF50 SVG coordinates (100x100 viewport)
VF50_SVG_COORDS: Dict[int, Tuple[int, int]] = {
    # longmaytrai (left eyebrow, 5 points: 0-4, open, left->right)
    0: (18, 28), 1: (23, 24), 2: (30, 22), 3: (37, 23), 4: (43, 27),
    # longmayphai (right eyebrow, 5 points: 5-9, left->right)
    5: (57, 27), 6: (63, 23), 7: (70, 22), 8: (77, 24), 9: (82, 28),
    # songmui (nose bridge, 4 points: 10-13, top->bottom)
    10: (50, 28), 11: (50, 38), 12: (50, 48), 13: (50, 55),
    # mattrai (left eye, 8 points: 14-21, closed contour)
    14: (24, 34), 15: (28, 31), 16: (33, 30), 17: (38, 31),
    18: (42, 34), 19: (38, 37), 20: (33, 38), 21: (28, 37),
    # matphai (right eye, 8 points: 22-29, closed contour)
    22: (58, 34), 23: (62, 31), 24: (67, 30), 25: (72, 31),
    26: (76, 34), 27: (72, 37), 28: (67, 38), 29: (62, 37),
    # moingoai (outer lip, 12 points: 30-41, closed contour)
    30: (34, 68), 31: (38, 64), 32: (43, 62), 33: (50, 61),
    34: (57, 62), 35: (62, 64), 36: (66, 68), 37: (62, 73),
    38: (57, 76), 39: (50, 77), 40: (43, 76), 41: (38, 73),
    # moitrong (inner lip, 8 points: 42-49, closed contour)
    42: (38, 68), 43: (43, 66), 44: (50, 65), 45: (57, 66),
    46: (62, 68), 47: (57, 72), 48: (50, 73), 49: (43, 72),
}


# ==============================================================================
# SECTION 10 & 11: CROP STRATEGY AND COORDINATE TRANSFORMS
# ==============================================================================

def expand_face_bbox(
    box_2d: Sequence[Union[int, float]],
    margin: float = 0.15,
) -> List[int]:
    """Expand face bounding box by a fractional margin clamped to [0, 1000].

    Args:
        box_2d: [ymin, xmin, ymax, xmax] in normalized [0, 1000].
        margin: Expansion fraction on all sides (0.15 = 15%).

    Returns:
        Padded bounding box [ymin, xmin, ymax, xmax] as clamped integers in [0, 1000].
    """
    if len(box_2d) != 4:
        raise ValueError(f"box_2d must have 4 coordinates, got {box_2d!r}")

    ymin, xmin, ymax, xmax = float(box_2d[0]), float(box_2d[1]), float(box_2d[2]), float(box_2d[3])
    bw = xmax - xmin
    bh = ymax - ymin

    pad_x = bw * float(margin)
    pad_y = bh * float(margin)

    new_xmin = int(round(max(0.0, xmin - pad_x)))
    new_ymin = int(round(max(0.0, ymin - pad_y)))
    new_xmax = int(round(min(1000.0, xmax + pad_x)))
    new_ymax = int(round(min(1000.0, ymax + pad_y)))

    return [new_ymin, new_xmin, new_ymax, new_xmax]


def crop_to_image_coords(
    x_crop_norm: float,
    y_crop_norm: float,
    crop_box_norm: Sequence[Union[int, float]],
    orig_w: int,
    orig_h: int,
    *,
    clamp: bool = True,
) -> Tuple[float, float]:
    """Map normalized coordinates [0..1000] from a crop box to full image coordinates.

    Zero off-by-one, zero axis inversion guarantee:
    - Horizontal normalized coordinate strictly interpolates between crop_xmin and crop_xmax.
    - Vertical normalized coordinate strictly interpolates between crop_ymin and crop_ymax.
    - Resulting coordinates scaled by original image dimensions (orig_w, orig_h).

    Formula:
        x_orig_norm = crop_xmin + (x_crop_norm / 1000.0) * (crop_xmax - crop_xmin)
        y_orig_norm = crop_ymin + (y_crop_norm / 1000.0) * (crop_ymax - crop_ymin)
        pixel_x = x_orig_norm * orig_w / 1000.0
        pixel_y = y_orig_norm * orig_h / 1000.0

    Args:
        x_crop_norm: Horizontal coordinate normalized in crop [0, 1000].
        y_crop_norm: Vertical coordinate normalized in crop [0, 1000].
        crop_box_norm: [ymin, xmin, ymax, xmax] of the crop in original normalized [0, 1000].
        orig_w: Original image width in pixels (> 0).
        orig_h: Original image height in pixels (> 0).
        clamp: If True, clamps resulting pixels to [0, orig_w] and [0, orig_h].

    Returns:
        Tuple of (pixel_x, pixel_y) in full image pixel space.
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid original image dimensions: {orig_w}x{orig_h}")

    c_ymin, c_xmin, c_ymax, c_xmax = (
        float(crop_box_norm[0]),
        float(crop_box_norm[1]),
        float(crop_box_norm[2]),
        float(crop_box_norm[3]),
    )

    norm_x = c_xmin + (float(x_crop_norm) / 1000.0) * (c_xmax - c_xmin)
    norm_y = c_ymin + (float(y_crop_norm) / 1000.0) * (c_ymax - c_ymin)

    if clamp:
        norm_x = max(0.0, min(1000.0, norm_x))
        norm_y = max(0.0, min(1000.0, norm_y))

    px = (norm_x / 1000.0) * float(orig_w)
    py = (norm_y / 1000.0) * float(orig_h)

    if clamp:
        px = max(0.0, min(float(orig_w), px))
        py = max(0.0, min(float(orig_h), py))

    return round(px, 2), round(py, 2)


def image_to_crop_coords(
    x_orig_norm: float,
    y_orig_norm: float,
    crop_box_norm: Sequence[Union[int, float]],
    *,
    clamp: bool = True,
) -> Tuple[float, float]:
    """Inverse mapping: map normalized coordinates [0..1000] from full image into crop [0..1000]."""
    c_ymin, c_xmin, c_ymax, c_xmax = (
        float(crop_box_norm[0]),
        float(crop_box_norm[1]),
        float(crop_box_norm[2]),
        float(crop_box_norm[3]),
    )

    span_x = c_xmax - c_xmin
    span_y = c_ymax - c_ymin

    if span_x <= 0.0 or span_y <= 0.0:
        raise ValueError(f"Degenerate crop box: span_x={span_x}, span_y={span_y}")

    crop_x = ((float(x_orig_norm) - c_xmin) / span_x) * 1000.0
    crop_y = ((float(y_orig_norm) - c_ymin) / span_y) * 1000.0

    if clamp:
        crop_x = max(0.0, min(1000.0, crop_x))
        crop_y = max(0.0, min(1000.0, crop_y))

    return round(crop_x, 1), round(crop_y, 1)


# ==============================================================================
# VF-50 DATA STRUCTURES & PARENT VS COMPONENT SKELETON EMISSION
# ==============================================================================

@dataclass
class VF50Landmark:
    """Represents a single VF-50 facial landmark in normalized [0..1000] coordinates."""
    id: int                       # 0..49
    name: str                     # Global string ID "0".."49"
    x: float                      # Normalized horizontal coordinate in [0, 1000]
    y: float                      # Normalized vertical coordinate in [0, 1000]
    visibility: int = VISIBILITY_VISIBLE  # 0=outside, 1=occluded, 2=visible
    confidence: float = 1.0
    component: str = ""           # e.g. "longmaytrai", "mattrai", etc.

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.name = str(self.name or self.id)
        self.x = float(max(0.0, min(1000.0, self.x)))
        self.y = float(max(0.0, min(1000.0, self.y)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        if not self.component:
            self.component = VF50_POINT_TO_COMPONENT.get(self.id, "")
        self.visibility = normalize_visibility(self.visibility, strict=False, default=VISIBILITY_VISIBLE)

    @property
    def is_visible(self) -> bool:
        return self.visibility == VISIBILITY_VISIBLE

    @property
    def is_outside(self) -> bool:
        return self.visibility == VISIBILITY_OUTSIDE

    @property
    def is_occluded(self) -> bool:
        return self.visibility == VISIBILITY_OCCLUDED

    def to_cvat_element(self, width: int, height: int) -> Dict[str, Any]:
        """Convert to CVAT skeleton child element shape."""
        outside, occluded = map_visibility_to_cvat(self.visibility)
        px, py = denormalize_keypoint(self.x, self.y, width=width, height=height)
        return {
            "label": self.name,
            "type": "points",
            "points": [px, py],
            "outside": outside,
            "occluded": occluded,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class VF50Face:
    """Represents a complete detected face containing exactly 50 VF-50 facial landmarks."""
    face_id: Optional[int] = None
    box_2d: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] in [0..1000]
    confidence: float = 1.0
    landmarks: Dict[int, VF50Landmark] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        # Ensure all 50 landmark slots are created
        for pt_id in range(VF50_POINTS_COUNT):
            if pt_id not in self.landmarks:
                self.landmarks[pt_id] = VF50Landmark(
                    id=pt_id,
                    name=str(pt_id),
                    x=0.0,
                    y=0.0,
                    visibility=VISIBILITY_OUTSIDE,
                    confidence=0.0,
                    component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
                )

    @property
    def visible_landmarks_count(self) -> int:
        return sum(1 for lm in self.landmarks.values() if lm.is_visible)

    @property
    def active_landmarks_count(self) -> int:
        return sum(1 for lm in self.landmarks.values() if not lm.is_outside)

    def get_landmark(self, pt_id: int) -> Optional[VF50Landmark]:
        return self.landmarks.get(pt_id)

    def get_component_landmarks(self, component_name: str) -> List[VF50Landmark]:
        """Fetch all landmarks belonging to a named component in index order."""
        cfg = VF50_COMPONENT_CONFIG.get(component_name)
        if not cfg:
            return []
        res = []
        for pt_id in range(cfg["start"], cfg["end"] + 1):
            lm = self.landmarks.get(pt_id)
            if lm:
                res.append(lm)
        return res

    def derive_bbox(self, pad_ratio: float = 0.05) -> Optional[List[int]]:
        """Derive bounding box [ymin, xmin, ymax, xmax] in normalized [0..1000] from active landmarks."""
        active = [lm for lm in self.landmarks.values() if not lm.is_outside]
        if not active:
            return self.box_2d
        xs = [lm.x for lm in active]
        ys = [lm.y for lm in active]
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

    def get_center(self) -> Optional[Tuple[float, float]]:
        """Derive center coordinates (cx, cy) in normalized [0..1000] space."""
        active = [lm for lm in self.landmarks.values() if not lm.is_outside]
        if not active:
            if self.box_2d:
                return ((self.box_2d[1] + self.box_2d[3]) * 0.5, (self.box_2d[0] + self.box_2d[2]) * 0.5)
            return None
        return (
            round(sum(lm.x for lm in active) / len(active), 2),
            round(sum(lm.y for lm in active) / len(active), 2),
        )

    def to_cvat_component_skeletons(
        self,
        width: int,
        height: int,
        group_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convert into the authoritative 7 CVAT component skeletons (Guideline Section 1.1 & 2.2).

        Emits exactly 7 skeleton shape dictionaries:
        - longmaytrai (5 points: "0".."4")
        - longmayphai (5 points: "5".."9")
        - songmui (4 points: "10".."13")
        - mattrai (8 points: "14".."21")
        - matphai (8 points: "22".."29")
        - moingoai (12 points: "30".."41")
        - moitrong (8 points: "42".."49")

        All 7 skeletons share the exact same group_id representing this face instance.
        """
        skeletons: List[Dict[str, Any]] = []

        for comp_name in VF50_COMPONENT_NAMES:
            lms = self.get_component_landmarks(comp_name)
            elements = [lm.to_cvat_element(width=width, height=height) for lm in lms]

            skel_dict: Dict[str, Any] = {
                "label": comp_name,
                "type": "skeleton",
                "confidence": round(self.confidence, 3),
                "elements": elements,
            }
            if group_id is not None:
                skel_dict["group_id"] = int(group_id)
                skel_dict["group"] = int(group_id)

            skeletons.append(skel_dict)

        return skeletons

    # NOTE: Legacy to_cvat_single_skeleton() removed in Phase 4 cleanup.
    # VF50 always emits 7-component skeletons via to_cvat_component_skeletons().


# ==============================================================================
# GEOMETRY HELPERS FOR QUALITY GATE
# ==============================================================================

def _shoelace_area(points: Sequence[Tuple[float, float]]) -> float:
    """Compute unsigned 2D polygon area using Shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Check if line segment p1-p2 strictly intersects line segment p3-p4."""
    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _has_self_intersection(points: Sequence[Tuple[float, float]]) -> bool:
    """Detect if closed polygon has crossing edges (X-crossing)."""
    n = len(points)
    if n < 4:
        return False
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        # Check against non-adjacent segments
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent segments at wrap-around
            p3 = points[j]
            p4 = points[(j + 1) % n]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    return False


def merge_vf50_landmarks(
    pass1_face: Optional[VF50Face] = None,
    pass2_face: Optional[VF50Face] = None,
    crop_box_norm: Optional[Sequence[Union[int, float]]] = None,
    *,
    initial_face: Optional[VF50Face] = None,
    refined_face: Optional[VF50Face] = None,
    crop_bbox: Optional[Sequence[Union[int, float]]] = None,
    min_confidence: float = 0.30,
    edge_margin: float = 5.0,
    pass2_is_reprojected: bool = True,
) -> VF50Face:
    """Deterministically merge Pass 1 global detection and Pass 2 crop refinement facial landmarks.

    Adheres to the 5 canonical merge policies:
      1. Missing / Malformed Pass 2 landmark: If Pass 2 response omitted a point or returned
         a dummy placeholder (0, 0, vis=0, conf=0), fallback to Pass 1 landmark.
      2. Low Confidence Pass 2: If Pass 2 point has confidence < min_confidence while Pass 1 has
         confidence >= min_confidence and is active, retain Pass 1.
      3. Pass 2 Visibility 0 (outside): If Pass 1 was active (vis > 0), check whether Pass 1 coordinate
         lies outside or near the border of crop window [ymin, xmin, ymax, xmax]. If outside crop coverage,
         Pass 2 never saw this area; retain Pass 1. If inside crop coverage and Pass 2 inspected the area,
         accept Pass 2 vis=0.
      4. Pass 2 Visibility 1 (occluded): High-resolution physical occlusion (fine hair, opaque glasses frame,
         mask, hand) takes precedence over coarse Pass 1 visibility=2 if Pass 2 confidence >= min_confidence.
         A coarse Pass 1 detection must never override legitimate Pass 2 physical occlusion.
      5. Pass 2 Visibility 2 (visible): High-resolution refined coordinates and confidence are adopted.

    Args:
        pass1_face: Initial VF50Face from global image inference (alias: initial_face).
        pass2_face: Refined VF50Face from face crop inference (alias: refined_face).
        crop_box_norm: Crop bounding box [ymin, xmin, ymax, xmax] in normalized [0..1000] space (alias: crop_bbox).
        min_confidence: Minimum confidence threshold to accept refined points (default: 0.30).
        edge_margin: Margin in normalized units to detect crop window boundaries (default: 5.0).
        pass2_is_reprojected: Whether pass2_face coordinates are already reprojected to full image space.

    Returns:
        Merged VF50Face instance.
    """
    p1_target = pass1_face if pass1_face is not None else initial_face
    p2_target = pass2_face if pass2_face is not None else refined_face
    effective_crop_box = crop_box_norm if crop_box_norm is not None else crop_bbox

    if p1_target is None or p2_target is None:
        raise ValueError("Both pass1_face (initial_face) and pass2_face (refined_face) must be provided")
    if effective_crop_box is None or len(effective_crop_box) < 4:
        raise ValueError("crop_box_norm or crop_bbox must contain [ymin, xmin, ymax, xmax]")

    c_ymin, c_xmin, c_ymax, c_xmax = [float(v) for v in effective_crop_box[:4]]
    merged_lms: Dict[int, VF50Landmark] = {}

    for pt_id in range(VF50_POINTS_COUNT):
        p1_lm = p1_target.landmarks.get(pt_id)
        p2_lm = p2_target.landmarks.get(pt_id)

        # Normalize visibilities
        p1_vis = normalize_visibility(p1_lm.visibility, strict=False, default=VISIBILITY_VISIBLE) if p1_lm else VISIBILITY_OUTSIDE
        p2_vis = normalize_visibility(p2_lm.visibility, strict=False, default=VISIBILITY_VISIBLE) if p2_lm else VISIBILITY_OUTSIDE

        # Detect dummy placeholder from VF50Face.__post_init__ or missing in Pass 2
        p2_is_dummy = (
            p2_lm is None
            or (p2_lm.x == 0.0 and p2_lm.y == 0.0 and p2_lm.confidence == 0.0 and p2_vis == VISIBILITY_OUTSIDE)
            or math.isnan(p2_lm.x)
            or math.isnan(p2_lm.y)
        )

        p1_is_valid = (
            p1_lm is not None
            and not (p1_lm.x == 0.0 and p1_lm.y == 0.0 and p1_lm.confidence == 0.0 and p1_vis == VISIBILITY_OUTSIDE)
            and not (math.isnan(p1_lm.x) or math.isnan(p1_lm.y))
        )

        # CASE 1: Pass 2 omitted or malformed landmark -> fallback to Pass 1
        if p2_is_dummy:
            if p1_is_valid and p1_lm is not None:
                merged_lms[pt_id] = VF50Landmark(
                    id=pt_id,
                    name=str(pt_id),
                    x=p1_lm.x,
                    y=p1_lm.y,
                    visibility=p1_vis,
                    confidence=p1_lm.confidence,
                    component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
                )
            continue

        assert p2_lm is not None

        # Determine coordinates in full image [0..1000]
        if pass2_is_reprojected:
            p2_x = float(max(0.0, min(1000.0, p2_lm.x)))
            p2_y = float(max(0.0, min(1000.0, p2_lm.y)))
        else:
            p2_x = float(max(0.0, min(1000.0, c_xmin + (p2_lm.x / 1000.0) * (c_xmax - c_xmin))))
            p2_y = float(max(0.0, min(1000.0, c_ymin + (p2_lm.y / 1000.0) * (c_ymax - c_ymin))))

        # CASE 2: Low confidence in Pass 2 while Pass 1 has reliable detection
        if (
            p2_lm.confidence < min_confidence
            and p1_is_valid
            and p1_lm is not None
            and p1_lm.confidence >= min_confidence
            and p1_vis != VISIBILITY_OUTSIDE
        ):
            merged_lms[pt_id] = VF50Landmark(
                id=pt_id,
                name=str(pt_id),
                x=p1_lm.x,
                y=p1_lm.y,
                visibility=p1_vis,
                confidence=p1_lm.confidence,
                component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
            )
            continue

        # CASE 3: Pass 2 says visibility=0 (outside)
        if p2_vis == VISIBILITY_OUTSIDE:
            if p1_is_valid and p1_lm is not None and p1_vis != VISIBILITY_OUTSIDE:
                # Check if Pass 1 point is inside crop coverage
                inside_crop = (
                    (c_xmin + edge_margin <= p1_lm.x <= c_xmax - edge_margin)
                    and (c_ymin + edge_margin <= p1_lm.y <= c_ymax - edge_margin)
                )
                if not inside_crop:
                    # Pass 2 crop cut off this landmark; retain Pass 1
                    merged_lms[pt_id] = VF50Landmark(
                        id=pt_id,
                        name=str(pt_id),
                        x=p1_lm.x,
                        y=p1_lm.y,
                        visibility=p1_vis,
                        confidence=p1_lm.confidence,
                        component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
                    )
                    continue

            # Pass 2 had crop access and verified point is outside
            merged_lms[pt_id] = VF50Landmark(
                id=pt_id,
                name=str(pt_id),
                x=p2_x,
                y=p2_y,
                visibility=VISIBILITY_OUTSIDE,
                confidence=p2_lm.confidence,
                component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
            )
            continue

        # CASE 4: Pass 2 says visibility=1 (occluded)
        # Legitimate physical occlusion observed at high resolution overrides coarse Pass 1
        if p2_vis == VISIBILITY_OCCLUDED:
            merged_lms[pt_id] = VF50Landmark(
                id=pt_id,
                name=str(pt_id),
                x=p2_x,
                y=p2_y,
                visibility=VISIBILITY_OCCLUDED,
                confidence=p2_lm.confidence,
                component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
            )
            continue

        # CASE 5: Pass 2 says visibility=2 (visible) -> use refined point
        merged_lms[pt_id] = VF50Landmark(
            id=pt_id,
            name=str(pt_id),
            x=p2_x,
            y=p2_y,
            visibility=VISIBILITY_VISIBLE,
            confidence=max(p2_lm.confidence, p1_lm.confidence if p1_lm else 0.0),
            component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
        )

    # Preserve bounding box and face metadata
    box_2d = p1_target.box_2d or p2_target.box_2d
    conf = max(p1_target.confidence, p2_target.confidence)

    return VF50Face(
        face_id=p1_target.face_id if p1_target.face_id is not None else p2_target.face_id,
        box_2d=list(box_2d) if box_2d else None,
        confidence=conf,
        landmarks=merged_lms,
    )


# ==============================================================================
# SECTION 17 & 18: DETERMINISTIC QUALITY GATE & SEMANTIC DISTINCTIONS
# ==============================================================================

@dataclass
class VF50QualityReport:
    """Quality gate validation report for VF-50 Face Landmarks."""
    is_valid: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    visible_count: int = 0
    active_count: int = 0
    iod_px: float = 0.0
    anomalies: List[str] = field(default_factory=list)


def assess_vf50_quality(
    face: VF50Face,
    width: int = 1000,
    height: int = 1000,
    *,
    min_active_landmarks: int = 20,
    min_confidence: float = 0.30,
) -> VF50QualityReport:
    """Deterministic Quality Gate for VF-50 Face Landmark annotations (Section 17 & 18).

    Validates:
    1. Minimum active landmark count.
    2. Degenerate collapse check (bounding box diagonal >= 15 normalized units).
    3. Inter-Ocular Distance (IOD) bounds check.
    4. Left / Right laterality and eyebrow monotonic order for frontal/near-frontal face.
    5. Eye contour vs eyelid boundaries:
       - Upper eyelid points above or equal to lower eyelid points (y_upper <= y_lower + delta).
       - Closed eyes / squinting handled gracefully without false rejection.
       - No self-intersecting X-crossing contours.
    6. Nose bridge centrality:
       - Points 10..13 ordered downwards vertically.
       - Segments roughly equal.
       - Point 13 is base of nasal bridge (above nostrils and upper lip).
    7. Outer lip vs Inner lip topological containment:
       - Inner lip strictly contained inside outer lip (x and y bounding limits).
       - Closed mouth condition: allows coinciding seams without error.
       - Area containment: Area(moitrong) <= Area(moingoai).
    """
    reasons: List[str] = []
    anomalies: List[str] = []
    score = 1.0

    lms = face.landmarks
    active_lms = [lm for lm in lms.values() if not lm.is_outside and lm.confidence >= min_confidence]
    visible_lms = [lm for lm in lms.values() if lm.is_visible and lm.confidence >= min_confidence]

    visible_count = len(visible_lms)
    active_count = len(active_lms)

    # 1. Minimum count check
    if active_count < min_active_landmarks:
        reasons.append(f"too_few_landmarks: only {active_count} active (required >= {min_active_landmarks})")
        return VF50QualityReport(
            is_valid=False,
            score=0.0,
            reasons=reasons,
            visible_count=visible_count,
            active_count=active_count,
        )

    # 2. Degenerate collapse check
    xs = [lm.x for lm in active_lms]
    ys = [lm.y for lm in active_lms]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    bbox_diag = math.hypot(span_x, span_y)

    if bbox_diag < 15.0:  # < 1.5% of normalized frame
        reasons.append(f"degenerate_collapse: landmark bounding box diagonal is {bbox_diag:.1f} < 15.0")
        return VF50QualityReport(
            is_valid=False,
            score=0.0,
            reasons=reasons,
            visible_count=visible_count,
            active_count=active_count,
        )

    # 3. Inter-Ocular Distance (IOD) calculation
    # Left eye center (points 14 and 18), Right eye center (points 22 and 26)
    p14, p18 = lms.get(14), lms.get(18)
    p22, p26 = lms.get(22), lms.get(26)

    iod = 0.0
    has_valid_eyes = False
    if p14 and p18 and p22 and p26 and not (p14.is_outside or p18.is_outside or p22.is_outside or p26.is_outside):
        c_left_x = (p14.x + p18.x) / 2.0
        c_left_y = (p14.y + p18.y) / 2.0
        c_right_x = (p22.x + p26.x) / 2.0
        c_right_y = (p22.y + p26.y) / 2.0
        iod = math.hypot(c_right_x - c_left_x, c_right_y - c_left_y)
        has_valid_eyes = True

    iod_px = (iod / 1000.0) * float(width)

    # Tolerance calibrated to IOD (fallback to 30 normalized units if IOD not measurable)
    tol = max(5.0, iod * 0.08) if iod > 0 else 25.0

    # 4. Laterality & Eyebrows Monotonicity (Frontal / Near-Frontal check)
    if has_valid_eyes:
        # Check roll angle
        eye_dx = c_right_x - c_left_x
        eye_dy = c_right_y - c_left_y
        roll_deg = math.degrees(math.atan2(eye_dy, eye_dx))

        # In viewer perspective, left eye must be on the left (smaller x)
        if c_left_x > c_right_x + 5.0:
            anomalies.append("inverted_laterality: left eye is to the right of right eye")
            reasons.append("inverted_laterality")
            score -= 0.35

        # If head roll is moderate (|roll| <= 25 deg), check horizontal eyebrow monotonicity
        if abs(roll_deg) <= 25.0:
            # longmaytrai (0..4): x_0 <= x_1 <= x_2 <= x_3 <= x_4
            for i in range(4):
                lm1, lm2 = lms.get(i), lms.get(i + 1)
                if lm1 and lm2 and not lm1.is_outside and not lm2.is_outside:
                    if lm1.x > lm2.x + tol:
                        anomalies.append(f"eyebrow_left_order: pt {i} (x={lm1.x:.1f}) > pt {i+1} (x={lm2.x:.1f})")
                        score -= 0.10
                        break

            # longmayphai (5..9): x_5 <= x_6 <= x_7 <= x_8 <= x_9
            for i in range(5, 9):
                lm1, lm2 = lms.get(i), lms.get(i + 1)
                if lm1 and lm2 and not lm1.is_outside and not lm2.is_outside:
                    if lm1.x > lm2.x + tol:
                        anomalies.append(f"eyebrow_right_order: pt {i} (x={lm1.x:.1f}) > pt {i+1} (x={lm2.x:.1f})")
                        score -= 0.10
                        break

    # 5. Eye Contour vs Eyelid Boundaries
    # Left eye: upper eyelid points (15, 16, 17) vs lower eyelid points (21, 20, 19)
    left_eye_pts = [(lms[i].x, lms[i].y) for i in range(14, 22) if i in lms and not lms[i].is_outside]
    if len(left_eye_pts) == 8:
        # Check vertical eyelid boundary (upper eyelid y <= lower eyelid y + tol)
        eyelid_pairs = [(15, 21), (16, 20), (17, 19)]
        for up_id, low_id in eyelid_pairs:
            up_lm, low_lm = lms.get(up_id), lms.get(low_id)
            if up_lm and low_lm:
                if up_lm.y > low_lm.y + tol:
                    anomalies.append(f"left_eyelid_inversion: pt {up_id} below pt {low_id}")
                    score -= 0.15

        # Check self-intersection
        if _has_self_intersection(left_eye_pts):
            anomalies.append("left_eye_self_intersection: contour forms X-shape")
            reasons.append("left_eye_contour_self_intersection")
            score -= 0.25

    # Right eye: upper eyelid points (23, 24, 25) vs lower eyelid points (29, 28, 27)
    right_eye_pts = [(lms[i].x, lms[i].y) for i in range(22, 30) if i in lms and not lms[i].is_outside]
    if len(right_eye_pts) == 8:
        eyelid_pairs_r = [(23, 29), (24, 28), (25, 27)]
        for up_id, low_id in eyelid_pairs_r:
            up_lm, low_lm = lms.get(up_id), lms.get(low_id)
            if up_lm and low_lm:
                if up_lm.y > low_lm.y + tol:
                    anomalies.append(f"right_eyelid_inversion: pt {up_id} below pt {low_id}")
                    score -= 0.15

        if _has_self_intersection(right_eye_pts):
            anomalies.append("right_eye_self_intersection: contour forms X-shape")
            reasons.append("right_eye_contour_self_intersection")
            score -= 0.25

    # 6. Nose Bridge Centrality
    p10, p11, p12, p13 = lms.get(10), lms.get(11), lms.get(12), lms.get(13)
    if p10 and p11 and p12 and p13 and not (p10.is_outside or p11.is_outside or p12.is_outside or p13.is_outside):
        # Vertical descent: y_10 < y_11 < y_12 < y_13
        if not (p10.y <= p11.y + tol and p11.y <= p12.y + tol and p12.y <= p13.y + tol):
            anomalies.append("nose_bridge_non_monotonic: nose points do not descend vertically")
            score -= 0.15

        # Three segments length ratio check
        d0 = math.hypot(p11.x - p10.x, p11.y - p10.y)
        d1 = math.hypot(p12.x - p11.x, p12.y - p11.y)
        d2 = math.hypot(p13.x - p12.x, p13.y - p12.y)
        lens = [d0, d1, d2]
        if min(lens) > 1.0:
            ratio = max(lens) / min(lens)
            if ratio > 3.0:
                anomalies.append(f"nose_segments_unequal: max/min segment ratio is {ratio:.2f}")
                score -= 0.10

        # Point 13 must be above upper lip (pt 30 / pt 33)
        p33 = lms.get(33)
        if p33 and not p33.is_outside:
            if p13.y > p33.y:
                anomalies.append("nose_base_below_lip: point 13 is below upper lip center")
                score -= 0.20

    # 7. Outer Lip vs Inner Lip Topological Containment
    p30, p36 = lms.get(30), lms.get(36)
    p42, p46 = lms.get(42), lms.get(46)
    if p30 and p36 and p42 and p46 and not (p30.is_outside or p36.is_outside or p42.is_outside or p46.is_outside):
        # Left corner: inner pt 42 must be >= outer pt 30 (to the right)
        if p42.x < p30.x - tol:
            anomalies.append("inner_lip_protrudes_left: pt 42 is left of pt 30")
            score -= 0.15

        # Right corner: inner pt 46 must be <= outer pt 36 (to the left)
        if p46.x > p36.x + tol:
            anomalies.append("inner_lip_protrudes_right: pt 46 is right of pt 36")
            score -= 0.15

    # Area containment check
    outer_lip_pts = [(lms[i].x, lms[i].y) for i in range(30, 42) if i in lms and not lms[i].is_outside]
    inner_lip_pts = [(lms[i].x, lms[i].y) for i in range(42, 50) if i in lms and not lms[i].is_outside]
    if len(outer_lip_pts) >= 6 and len(inner_lip_pts) >= 4:
        area_outer = _shoelace_area(outer_lip_pts)
        area_inner = _shoelace_area(inner_lip_pts)
        if area_inner > area_outer + 20.0:  # Inner lip area cannot exceed outer lip area
            anomalies.append(f"inner_lip_exceeds_outer_area: {area_inner:.1f} > {area_outer:.1f}")
            score -= 0.25

    score = max(0.0, min(1.0, score))
    is_valid = score >= 0.40 and len(anomalies) <= 3

    if not is_valid:
        reasons.extend(anomalies)

    return VF50QualityReport(
        is_valid=is_valid,
        score=round(score, 3),
        reasons=reasons,
        visible_count=visible_count,
        active_count=active_count,
        iod_px=round(iod_px, 1),
        anomalies=anomalies,
    )


# ==============================================================================
# SECTION 12: MULTI-FACE SUPPORT & RESPONSE PARSER
# ==============================================================================

def is_vf50_face_degenerate(report: VF50QualityReport) -> bool:
    """Check if a rejected face is truly degenerate (unusable collapse or insufficient points).

    A face is truly degenerate if:
    1. Bounding box diagonal < 15 normalized units (degenerate_collapse), OR
    2. Active landmarks count < 15 (insufficient active landmarks to reconstruct a face).
    """
    if report.active_count < 15:
        return True
    if any("degenerate_collapse" in reas for reas in report.reasons):
        return True
    return False


def is_vf50_face_salvageable(report: VF50QualityReport) -> bool:
    """Check if an invalid face is salvageable as an editable draft.

    A face is salvageable if it failed strict quality validation (e.g. minor
    eyelid inversion, eyebrow monotonicity, lip protrusion, or nose non-monotonicity),
    but has sufficient active landmarks (>= 15) and is not geometrically collapsed.
    """
    return not is_vf50_face_degenerate(report)


def faces_to_cvat_skeletons(
    faces: Sequence[VF50Face],
    width: int,
    height: int,
    *,
    base_group_id: int = 1,
    filter_corrupt: bool = True,
    as_components: bool = True,  # Deprecated: always True. Kept for backward compat.
    fallback_on_corrupt: bool = True,
    max_salvage: int = 5,
) -> List[Dict[str, Any]]:
    """Convert multiple detected faces to native CVAT skeleton shape dictionaries.

    Multi-Face Isolation Guarantees (Section 12):
    - 0 faces -> returns []
    - 1 face -> returns exactly 7 component skeletons sharing group_id.
    - N faces -> returns N * 7 component skeletons. Skeletons for Face k share group_id.
    - Skeletons belonging to distinct faces never share group_id.
    - Completely separate landmark instances preventing any cross-contamination.
    - Per-face quality gate & salvage decisions:
      * Valid face -> emitted (7 skeletons).
      * Truly degenerate face (collapse / < 15 points) -> dropped.
      * Salvageable invalid face -> if fallback_on_corrupt, emitted as editable draft (up to max_salvage).
      * Salvaging decisions are evaluated per face, so a valid face in the frame does not cause
        a salvageable face in the same frame to be dropped.

    Args:
        faces: Sequence of VF50Face instances.
        width: Image pixel width.
        height: Image pixel height.
        base_group_id: Starting group_id for instance grouping.
        filter_corrupt: If True, filters out anatomically collapsed or corrupted faces.
        as_components: Deprecated, always True. Kept for backward compatibility.
        fallback_on_corrupt: If True, salvages candidate faces failing strict quality gate
                             (up to max_salvage) as editable drafts to prevent silent drop in CVAT.
        max_salvage: Maximum number of salvageable faces to emit as editable drafts (default: 5).

    Returns:
        List of CVAT skeleton shape dictionaries.
    """
    if not faces:
        return []

    # Step 1: Evaluate each face individually
    evaluated_faces: List[Tuple[VF50Face, Optional[VF50QualityReport], str]] = []
    salvageable_indices: List[int] = []

    for idx, face in enumerate(faces):
        if not filter_corrupt:
            evaluated_faces.append((face, None, "VALID"))
            continue

        report = assess_vf50_quality(face, width=width, height=height)
        if report.is_valid:
            evaluated_faces.append((face, report, "VALID"))
        elif is_vf50_face_degenerate(report):
            logger.warning(
                f"Dropped truly degenerate face (face id={face.face_id}): reasons={report.reasons}"
            )
            evaluated_faces.append((face, report, "DEGENERATE"))
        else:
            # Face failed strict quality check but is not degenerate
            if fallback_on_corrupt:
                evaluated_faces.append((face, report, "SALVAGEABLE"))
                salvageable_indices.append(idx)
            else:
                logger.warning(
                    f"Dropped corrupt face under strict mode (face id={face.face_id}): reasons={report.reasons}"
                )
                evaluated_faces.append((face, report, "CORRUPT_DROPPED"))

    # Step 2: If salvageable faces exceed max_salvage cap, select highest-scoring candidates
    allowed_salvage_set: Set[int] = set()
    if salvageable_indices:
        if len(salvageable_indices) > max_salvage:
            sorted_indices = sorted(
                salvageable_indices,
                key=lambda i: (
                    evaluated_faces[i][1].score if evaluated_faces[i][1] else 0.0,
                    evaluated_faces[i][0].confidence,
                ),
                reverse=True,
            )
            allowed_salvage_set = set(sorted_indices[:max_salvage])
            for i in sorted_indices[max_salvage:]:
                f_drop, r_drop, _ = evaluated_faces[i]
                logger.info(
                    f"Dropped salvageable face (id={f_drop.face_id}, "
                    f"score={r_drop.score if r_drop else 0.0:.2f}) exceeding max_salvage cap ({max_salvage})."
                )
        else:
            allowed_salvage_set = set(salvageable_indices)

    # Step 3: Emit component skeletons with unique, sequential group_id per face
    skeletons: List[Dict[str, Any]] = []
    curr_group_id = int(base_group_id)

    for idx, (face, report, status) in enumerate(evaluated_faces):
        if status == "VALID":
            face_skels = face.to_cvat_component_skeletons(
                width=width,
                height=height,
                group_id=curr_group_id,
            )
            skeletons.extend(face_skels)
            curr_group_id += 1
        elif status == "SALVAGEABLE" and idx in allowed_salvage_set:
            score_str = f"{report.score:.2f}" if report else "N/A"
            act_str = f"{report.active_count}" if report else "N/A"
            logger.info(
                f"Quality gate fallback: salvaging face (id={face.face_id}, "
                f"score={score_str}, active={act_str}/50) as editable draft."
            )
            face_skels = face.to_cvat_component_skeletons(
                width=width,
                height=height,
                group_id=curr_group_id,
            )
            skeletons.extend(face_skels)
            curr_group_id += 1

    return skeletons


def parse_vf50_response(
    raw_response: Union[str, Dict[str, Any], List[Any]],
    *,
    default_group_id: int = 1,
    crop_box: Optional[Sequence[Union[int, float]]] = None,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
) -> List[VF50Face]:
    """Parse remote 9Router vision response into validated VF50Face instances.

    Multi-Schema and Coordinate Crop Handling (Section 10, 11 & 12):
    - Strips markdown code blocks.
    - Handles multiple formats:
        Format A: {"faces": [{"id": 1, "box_2d": [...], "landmarks": [...]}]}
        Format B: {"faces": [{"id": 1, "components": {"longmaytrai": [...]}}]}
        Format C: Direct list of face dictionaries [{"id": 1, ...}]
        Format D: Single face dictionary {"landmarks": [...]}
    - Landmark coordinate mapping:
        If crop_box is provided, maps crop-relative normalized coordinates [0..1000]
        back to original image normalized [0..1000] using crop_to_image_coords.
    """
    if not raw_response:
        return []

    data: Any = raw_response

    if isinstance(data, str):
        cleaned = data.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
        if not (cleaned.startswith("{") or cleaned.startswith("[")):
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
            if match:
                cleaned = match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response for VF-50: {e}")
            return []

    faces_raw: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if "faces" in data and isinstance(data["faces"], list):
            faces_raw = [f for f in data["faces"] if isinstance(f, dict)]
        elif "landmarks" in data or "components" in data:
            faces_raw = [data]
        elif "predictions" in data and isinstance(data["predictions"], list):
            faces_raw = [f for f in data["predictions"] if isinstance(f, dict)]
    elif isinstance(data, list):
        faces_raw = [f for f in data if isinstance(f, dict)]

    parsed_faces: List[VF50Face] = []

    for idx, f_item in enumerate(faces_raw, start=default_group_id):
        face_id = f_item.get("id") or idx
        confidence = float(f_item.get("confidence", 1.0))
        box_2d = f_item.get("box_2d")
        if isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
            try:
                box_2d = [int(v) for v in box_2d]
                if crop_box is not None and len(crop_box) == 4:
                    c_ymin, c_xmin, c_ymax, c_xmax = float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3])
                    box_2d = [
                        int(round(c_ymin + (box_2d[0] / 1000.0) * (c_ymax - c_ymin))),
                        int(round(c_xmin + (box_2d[1] / 1000.0) * (c_xmax - c_xmin))),
                        int(round(c_ymin + (box_2d[2] / 1000.0) * (c_ymax - c_ymin))),
                        int(round(c_xmin + (box_2d[3] / 1000.0) * (c_xmax - c_xmin))),
                    ]
            except (ValueError, TypeError):
                box_2d = None
        else:
            box_2d = None

        lms_dict: Dict[int, VF50Landmark] = {}

        # Check if landmarks are nested in components
        components_dict = f_item.get("components")
        if isinstance(components_dict, dict):
            for comp_name, pts_list in components_dict.items():
                clean_comp = comp_name.strip().lower()
                cfg = VF50_COMPONENT_CONFIG.get(clean_comp)
                if not cfg or not isinstance(pts_list, (list, tuple)):
                    continue
                start_id = cfg["start"]
                for pt_offset, pt_data in enumerate(pts_list):
                    pt_id = start_id + pt_offset
                    if pt_id > cfg["end"]:
                        break
                    lm = _parse_single_vf50_landmark(pt_id, pt_data, crop_box=crop_box, orig_w=orig_w, orig_h=orig_h)
                    if lm:
                        lms_dict[pt_id] = lm

        raw_lms = f_item.get("landmarks") or f_item.get("points") or f_item.get("elements")
        if isinstance(raw_lms, dict):
            for key, val in raw_lms.items():
                try:
                    pt_id = int(key)
                except ValueError:
                    if "_" in str(key):
                        suffix = str(key).split("_")[-1]
                        try:
                            pt_id = int(suffix)
                        except ValueError:
                            continue
                    else:
                        continue
                if 0 <= pt_id < VF50_POINTS_COUNT:
                    lm = _parse_single_vf50_landmark(pt_id, val, crop_box=crop_box, orig_w=orig_w, orig_h=orig_h)
                    if lm:
                        lms_dict[pt_id] = lm
        elif isinstance(raw_lms, (list, tuple)):
            if raw_lms and isinstance(raw_lms[0], dict):
                for item in raw_lms:
                    if not isinstance(item, dict):
                        continue
                    pt_id_raw = item.get("id") if item.get("id") is not None else item.get("name")
                    try:
                        pt_id = int(pt_id_raw)
                    except (ValueError, TypeError):
                        continue
                    if 0 <= pt_id < VF50_POINTS_COUNT:
                        lm = _parse_single_vf50_landmark(pt_id, item, crop_box=crop_box, orig_w=orig_w, orig_h=orig_h)
                        if lm:
                            lms_dict[pt_id] = lm
            elif raw_lms and isinstance(raw_lms[0], (list, tuple)):
                for pt_id, seq in enumerate(raw_lms):
                    if pt_id >= VF50_POINTS_COUNT:
                        break
                    lm = _parse_sequence_vf50_landmark(pt_id, seq, crop_box=crop_box, orig_w=orig_w, orig_h=orig_h)
                    if lm:
                        lms_dict[pt_id] = lm

        face = VF50Face(
            face_id=face_id,
            box_2d=box_2d,
            confidence=confidence,
            landmarks=lms_dict,
        )
        parsed_faces.append(face)

    return parsed_faces


def _parse_single_vf50_landmark(
    pt_id: int,
    data: Any,
    crop_box: Optional[Sequence[Union[int, float]]] = None,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
) -> Optional[VF50Landmark]:
    """Helper to parse a single VF-50 landmark with optional crop coordinate remapping."""
    if isinstance(data, dict):
        pt = data.get("point") or data.get("points")
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x, y = float(pt[0]), float(pt[1])
        else:
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))

        vis_val = data.get("visibility")
        if vis_val is None:
            outside = bool(data.get("outside", False))
            occluded = bool(data.get("occluded", False))
            vis_val = map_cvat_to_visibility(outside, occluded)
        else:
            vis_val = normalize_visibility(vis_val, strict=False, default=VISIBILITY_VISIBLE)

        conf = float(data.get("confidence", 1.0))

        # If coordinates are given in crop space, map to original full image space
        if crop_box is not None and len(crop_box) == 4:
            # Map normalized crop [0..1000] -> original normalized [0..1000]
            c_ymin, c_xmin, c_ymax, c_xmax = float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3])
            x = c_xmin + (x / 1000.0) * (c_xmax - c_xmin)
            y = c_ymin + (y / 1000.0) * (c_ymax - c_ymin)

        return VF50Landmark(
            id=pt_id,
            name=str(pt_id),
            x=x,
            y=y,
            visibility=vis_val,
            confidence=conf,
            component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
        )

    if isinstance(data, (list, tuple)):
        return _parse_sequence_vf50_landmark(pt_id, data, crop_box=crop_box, orig_w=orig_w, orig_h=orig_h)

    return None


def _parse_sequence_vf50_landmark(
    pt_id: int,
    seq: Sequence[Any],
    crop_box: Optional[Sequence[Union[int, float]]] = None,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
) -> Optional[VF50Landmark]:
    """Helper to parse landmark from sequence [x, y, (vis), (conf)]."""
    if len(seq) < 2:
        return None
    try:
        x = float(seq[0])
        y = float(seq[1])
        raw_vis = seq[2] if len(seq) >= 3 else VISIBILITY_VISIBLE
        vis = normalize_visibility(raw_vis, strict=False, default=VISIBILITY_VISIBLE)
        conf = float(seq[3]) if len(seq) >= 4 else 1.0

        if crop_box is not None and len(crop_box) == 4:
            c_ymin, c_xmin, c_ymax, c_xmax = float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3])
            x = c_xmin + (x / 1000.0) * (c_xmax - c_xmin)
            y = c_ymin + (y / 1000.0) * (c_ymax - c_ymin)

        return VF50Landmark(
            id=pt_id,
            name=str(pt_id),
            x=x,
            y=y,
            visibility=vis,
            confidence=conf,
            component=VF50_POINT_TO_COMPONENT.get(pt_id, ""),
        )
    except (ValueError, TypeError):
        return None


# ==============================================================================
# SECTION 10 & PROMPT GENERATION
# ==============================================================================

VF50_OCCLUSION_INSTRUCTIONS: str = (
    "Physical Occlusion vs Optical Blur & Shadows (Confidence != Occlusion!):\n"
    "- Physical Occlusion (visibility = 1):\n"
    "  * A facial landmark is physically obstructed by an opaque physical object:\n"
    "    - Hair / bangs covering eyebrow (longmaytrai / longmayphai) -> visibility = 1 (occluded).\n"
    "    - Hand, fingers, or arm covering mouth or chin (moingoai / moitrong) -> visibility = 1 (occluded).\n"
    "    - Medical mask, cloth mask, or scarf covering mouth / nose -> visibility = 1 (occluded).\n"
    "    - Opaque eyeglasses frame physically covering landmark, dark opaque sunglasses lens preventing landmark observation, or temple arm covering eyebrow/eye contour -> visibility = 1 (occluded).\n"
    "    - Clear / transparent corrective eyeglass lenses: Eye contour and landmark points viewed directly through clear transparent glass are VISIBLE (visibility = 2), NOT occluded. Express any slight glare or refraction via lower confidence, NOT visibility = 1.\n"
    "    - Microphone, cup, or telephone held against face -> visibility = 1 (occluded).\n"
    "  * For physically occluded landmarks, estimate the anatomically true location and set visibility = 1.\n"
    "- Optical Degradation / Blur, Shadows & Low Lighting (visibility = 2):\n"
    "  * If a facial feature is within direct optical line of sight but degraded by motion blur, camera defocus, "
    "shadows (e.g. from vehicle cabin, cap visor, nose shadow), sensor noise, or low illumination, it is STILL VISIBLE: set visibility = 2.\n"
    "  * Express visual uncertainty through a lower 'confidence' score (e.g. 0.35 - 0.70), NOT by setting visibility = 1. "
    "Optical blur, shadow, and low light are NOT physical occlusion; Confidence != Occlusion!\n"
    "- Outside Frame / Unlabelable (visibility = 0):\n"
    "  * Landmark is outside the image boundary or cropped out -> set visibility = 0.\n"
    "  * Rule: Physical occlusion within frame & inferable -> visibility = 1. visibility = 0 is ONLY for points outside image frame / unlabelable."
)


def generate_vf50_canonical_descriptions() -> str:
    """Generate authoritative 7-component landmark descriptions derived directly from week2_vf50 schema.

    Single Source of Truth guarantee: Both global prompt and crop refinement prompt
    share the exact same canonical landmark descriptions generated from schema data.
    """
    from core.week2_schema import load_vf50
    schema = load_vf50()
    lines = []
    for idx, comp in enumerate(schema.components, start=1):
        desc = comp.description if getattr(comp, "description", None) else ""
        lines.append(f"  {idx}. {comp.name} (points {comp.start_id}..{comp.end_id}): {desc}")
    return "\n".join(lines)


def build_vf50_prompt() -> str:
    """Generate strict, deterministic prompt for 9Router vision inference for VF-50 Face Landmarks."""
    desc_block = generate_vf50_canonical_descriptions()
    return (
        "Perform facial landmark estimation conforming to the VinFast VF-50 schema on this image.\n"
        "Guidelines:\n"
        "- Exactly 50 landmarks, partitioned into 7 components:\n"
        f"{desc_block}\n"
        "\n"
        "Critical Anatomical Constraints:\n"
        "- Viewer perspective: features on the left side of the image are 'trai' (x_left < x_right).\n"
        "- Eyebrow ordering: longmaytrai proceeds lateral-to-medial (0->4); longmayphai proceeds medial-to-lateral (5->9).\n"
        "- Eyelid vertical order: upper eyelid points MUST have y <= lower eyelid points (upper eyelid is above lower eyelid).\n"
        "- Eye corners: mattrai starts at outer corner (14) with inner corner at (18); matphai starts at inner corner (22) with outer corner at (26).\n"
        "- Nasal bridge point 13 is the base of the nose bridge above nostrils, NOT the nasal tip.\n"
        "- Mouth topology: inner lip (moitrong) MUST be completely enclosed within outer lip (moingoai). If mouth is closed, inner lip seam coincides with mouth line.\n"
        "- Inner lip ordering: moitrong starts at inner left corner (42) -> upper inner lip (43..45, center 44) -> inner right corner (46) -> lower inner lip (47..49, center 48) -> closes back to 42.\n"
        "- Coordinate convention:\n"
        "  * Normalize all coordinates to integers in [0, 1000] relative to image width and height.\n"
        "  * Visibility flag: 0=outside image frame, 1=occluded, 2=visible.\n"
        f"{VF50_OCCLUSION_INSTRUCTIONS}\n"
        "- Output MUST be strict, valid JSON with no extraneous text:\n"
        "{\n"
        '  "faces": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "confidence": 0.98,\n'
        '      "box_2d": [ymin, xmin, ymax, xmax],\n'
        '      "landmarks": [\n'
        '        {"id": 0, "point": [x, y], "visibility": 2, "confidence": 0.99},\n'
        '        {"id": 1, "point": [x, y], "visibility": 2, "confidence": 0.99},\n'
        "        ...\n"
        '        {"id": 49, "point": [x, y], "visibility": 2, "confidence": 0.95}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        'If no face is present, return {"faces": []}.'
    )


def build_vf50_crop_prompt() -> str:
    """Generate targeted prompt for 9Router vision inference on cropped face patches.

    Optimized for high-resolution single-face crops with strict anatomical topological guidance:
    - Exactly 50 landmarks conforming to VinFast VF-50 schema across 7 components.
    - Explicit ordering constraints (lateral-to-medial left eyebrow 0->4, medial-to-lateral right eyebrow 5->9,
      outer-to-inner left eye 14->18, inner-to-outer right eye 22->26, non-inverted eyelids, contained inner lips).
    - Normalized to [0, 1000] relative to the cropped patch.
    """
    desc_block = generate_vf50_canonical_descriptions()
    return (
        "Perform fine-grained facial landmark estimation on this cropped human face adhering to the VinFast VF-50 schema.\n"
        "This image is a close-up crop containing a single face. Localize all 50 landmarks with high precision across all 7 components:\n"
        f"{desc_block}\n"
        "\n"
        "Critical Anatomical Constraints:\n"
        "- Viewer perspective: features on the image left side are 'trai' (x_left < x_right).\n"
        "- Eyebrow ordering: longmaytrai proceeds lateral-to-medial (0->4); longmayphai proceeds medial-to-lateral (5->9).\n"
        "- Eyelid vertical order: upper eyelid points MUST have y <= lower eyelid points (upper eyelid is above lower eyelid).\n"
        "- Eye corners: mattrai starts at outer corner (14) with inner corner at (18); matphai starts at inner corner (22) with outer corner at (26).\n"
        "- Nasal bridge point 13 is the base of the nose bridge above nostrils, NOT the nasal tip.\n"
        "- Mouth topology: inner lip (moitrong) MUST be completely enclosed within outer lip (moingoai). If mouth is closed, inner lip seam coincides with mouth line.\n"
        "- Inner lip ordering: moitrong starts at inner left corner (42) -> upper inner lip (43..45, center 44) -> inner right corner (46) -> lower inner lip (47..49, center 48) -> closes back to 42.\n"
        "- Coordinates MUST be normalized to integers in [0, 1000] relative to this cropped image patch [width=1000, height=1000].\n"
        "- Visibility flag: 0=outside frame, 1=occluded, 2=visible.\n"
        f"{VF50_OCCLUSION_INSTRUCTIONS}\n"
        "Output MUST be strict, valid JSON with no extra commentary:\n"
        "{\n"
        '  "faces": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "confidence": 0.99,\n'
        '      "landmarks": [\n'
        '        {"id": 0, "point": [x, y], "visibility": 2, "confidence": 0.99},\n'
        "        ...\n"
        '        {"id": 49, "point": [x, y], "visibility": 2, "confidence": 0.99}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )


def build_cvat_vf50_spec() -> List[Dict[str, Any]]:
    """Build CVAT function.yaml annotations spec items for the 7 VF-50 component skeletons.

    Derives directly from canonical load_vf50() schema.
    Conforms to VinFast Guideline Section 5.3 & Appendix 10 (Raw JSON schema):
    - Exactly 7 skeleton labels: longmaytrai, longmayphai, songmui, mattrai, matphai, moingoai, moitrong.
    - Each label has type "skeleton", attributes [].
    - Each sublabel has name matching point ID ("0".."49"), type "points", attributes [].
    - SVG templates with circles (nodes) and lines (edges).
    """
    from core.week2_schema import load_vf50
    schema = load_vf50()
    specs: List[Dict[str, Any]] = []

    for idx, comp in enumerate(schema.components, start=1):
        node_offset = comp.start_id

        sublabels = [
            {"id": kp.id - node_offset + 1, "name": kp.numeric_name, "type": "points", "attributes": []}
            for kp in comp.keypoints
        ]

        # Generate SVG template with anatomically-positioned coordinates
        svg_circles = []
        for kp in comp.keypoints:
            n_id = kp.id - node_offset + 1
            cx, cy = VF50_SVG_COORDS[kp.id]
            svg_circles.append(
                f'<circle cx="{cx}" cy="{cy}" r="3" data-type="element node" '
                f'data-element-id="{n_id}" data-node-id="{n_id}" data-label-name="{kp.id}"/>'
            )

        svg_lines = []
        for p1, p2 in comp.edges:
            n1 = p1 - node_offset + 1
            n2 = p2 - node_offset + 1
            x1, y1 = VF50_SVG_COORDS[p1]
            x2, y2 = VF50_SVG_COORDS[p2]
            svg_lines.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'data-type="edge" data-node-from="{n1}" data-node-to="{n2}"/>'
            )

        svg_content = f'<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">{"".join(svg_circles + svg_lines)}</svg>'

        spec_item = {
            "id": idx,
            "name": comp.name,
            "type": "skeleton",
            "attributes": [],
            "sublabels": sublabels,
            "svg": svg_content,
        }
        specs.append(spec_item)

    return specs


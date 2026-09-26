"""Canonical Buddha Multi-Limb Contract, Schemas, Geometry, and Quality Gates.

Supports:
1. Canonical buddha_arm (3 keypoints: root, elbow, wrist).
2. Canonical buddha_hand (21 keypoints: wrist + 5 finger chains).
3. Central Body (Pose17) & Central Face (VF50 7-component) integration.
4. Deterministic angular arm ordering around torso reference center.
5. Deterministic deduplication of radial arm candidates.
6. 1:1 deterministic hand-to-arm linking with synchronized group_id.
7. Coordinate reprojection (crop [0..1000] <-> global [0..1000] and pixels).
8. Strict visibility normalization [0=outside, 1=occluded, 2=visible].
9. Claude Opus 5.5 model resolution with explicit fallback gating.
10. Native CVAT skeleton specifications with SVG graphs.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, Union

import yaml

from core.week2_schema import (
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUTSIDE,
    VISIBILITY_VISIBLE,
    map_cvat_to_visibility,
    map_visibility_to_cvat,
    normalize_visibility,
)
from core.pose_face_schema import (
    build_cvat_skeleton_spec,
    generate_svg_representation,
    validate_svg_node_ids,
)

logger = logging.getLogger(__name__)

# ─── Configuration Loader ──────────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_BUDDHA_YAML = _CONFIG_DIR / "buddha_multilimbs.yaml"

BUDDHA_MULTILIMB_LABELS: Tuple[str, ...] = (
    "person",
    "longmaytrai",
    "longmayphai",
    "songmui",
    "mattrai",
    "matphai",
    "moingoai",
    "moitrong",
    "buddha_arm",
    "buddha_hand",
)


@dataclass(frozen=True)
class BuddhaArmPointDef:
    id: int
    name: str
    semantic: str


@dataclass(frozen=True)
class BuddhaHandPointDef:
    id: int
    name: str
    finger: str
    semantic: str


@dataclass(frozen=True)
class BuddhaMultilimbsSchema:
    name: str
    version: str
    description: str
    arm_keypoints: Tuple[BuddhaArmPointDef, ...]
    arm_edges: Tuple[Tuple[int, int], ...]
    hand_keypoints: Tuple[BuddhaHandPointDef, ...]
    hand_edges: Tuple[Tuple[int, int], ...]
    arm_point_names: Tuple[str, ...]
    hand_point_names: Tuple[str, ...]
    arm_crop_padding: float
    hand_crop_padding: float
    max_arms: int
    max_hand_refinements: int
    min_confidence: float
    two_pass: bool
    hand_refine: bool
    refine_workers: int
    global_max_tokens: int
    arm_max_tokens: int
    hand_max_tokens: int
    wrist_dist_thresh: float
    elbow_dist_thresh: float
    root_dist_thresh: float
    dir_sim_thresh: float
    hand_iou_thresh: float

    @property
    def labels(self) -> List[str]:
        return list(BUDDHA_MULTILIMB_LABELS)



_CACHED_SCHEMA: Optional[BuddhaMultilimbsSchema] = None


def load_buddha_schema(yaml_path: Optional[Path] = None) -> BuddhaMultilimbsSchema:
    """Load and cache canonical Buddha multi-limbs schema from YAML."""
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is not None and yaml_path is None:
        return _CACHED_SCHEMA

    target_path = yaml_path or _BUDDHA_YAML
    if not target_path.exists():
        raise FileNotFoundError(f"Buddha schema YAML not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    labels_cfg = data.get("labels", {})
    arm_cfg = labels_cfg.get("arm", {})
    hand_cfg = labels_cfg.get("hand", {})

    arm_kps = tuple(
        BuddhaArmPointDef(id=int(kp["id"]), name=str(kp["name"]), semantic=str(kp.get("semantic", "")))
        for kp in arm_cfg.get("keypoints", [])
    )
    arm_edges = tuple((int(e[0]), int(e[1])) for e in arm_cfg.get("edges", []))

    hand_kps = tuple(
        BuddhaHandPointDef(
            id=int(kp["id"]),
            name=str(kp["name"]),
            finger=str(kp.get("finger", "")),
            semantic=str(kp.get("semantic", "")),
        )
        for kp in hand_cfg.get("keypoints", [])
    )
    hand_edges = tuple((int(e[0]), int(e[1])) for e in hand_cfg.get("edges", []))

    pipe_cfg = data.get("pipeline", {})
    dedup_cfg = data.get("deduplication", {})

    schema = BuddhaMultilimbsSchema(
        name=str(data.get("schema", {}).get("name", "buddha_multilimbs")),
        version=str(data.get("schema", {}).get("version", "1.0")),
        description=str(data.get("schema", {}).get("description", "")),
        arm_keypoints=arm_kps,
        arm_edges=arm_edges,
        hand_keypoints=hand_kps,
        hand_edges=hand_edges,
        arm_point_names=tuple(kp.name for kp in arm_kps),
        hand_point_names=tuple(kp.name for kp in hand_kps),
        arm_crop_padding=float(pipe_cfg.get("arm_crop_padding", 0.25)),
        hand_crop_padding=float(pipe_cfg.get("hand_crop_padding", 0.35)),
        max_arms=int(pipe_cfg.get("max_arms", 100)),
        max_hand_refinements=int(pipe_cfg.get("max_hand_refinements", 60)),
        min_confidence=float(pipe_cfg.get("min_confidence", 0.30)),
        two_pass=bool(pipe_cfg.get("two_pass", True)),
        hand_refine=bool(pipe_cfg.get("hand_refine", True)),
        refine_workers=int(pipe_cfg.get("refine_workers", 2)),
        global_max_tokens=int(pipe_cfg.get("global_max_tokens", 4000)),
        arm_max_tokens=int(pipe_cfg.get("arm_max_tokens", 1500)),
        hand_max_tokens=int(pipe_cfg.get("hand_max_tokens", 2000)),
        wrist_dist_thresh=float(dedup_cfg.get("wrist_distance_threshold", 35.0)),
        elbow_dist_thresh=float(dedup_cfg.get("elbow_distance_threshold", 45.0)),
        root_dist_thresh=float(dedup_cfg.get("root_distance_threshold", 60.0)),
        dir_sim_thresh=float(dedup_cfg.get("direction_similarity_threshold", 0.92)),
        hand_iou_thresh=float(dedup_cfg.get("hand_iou_threshold", 0.65)),
    )

    if yaml_path is None:
        _CACHED_SCHEMA = schema
    return schema


# ─── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BuddhaArmKeypoint:
    """A single arm joint in normalized [0..1000] coordinates."""
    name: str  # 'root', 'elbow', 'wrist'
    x: float
    y: float
    visibility: int = VISIBILITY_VISIBLE
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        self.x = float(max(0.0, min(1000.0, self.x)))
        self.y = float(max(0.0, min(1000.0, self.y)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.visibility = normalize_visibility(self.visibility, strict=False, default=VISIBILITY_VISIBLE)

    @property
    def is_outside(self) -> bool:
        return self.visibility == VISIBILITY_OUTSIDE

    @property
    def is_occluded(self) -> bool:
        return self.visibility == VISIBILITY_OCCLUDED

    @property
    def is_visible(self) -> bool:
        return self.visibility == VISIBILITY_VISIBLE

    def to_cvat_element(self, width: int, height: int) -> Dict[str, Any]:
        outside, occluded = map_visibility_to_cvat(self.visibility)
        px = round(self.x * float(width) / 1000.0, 2)
        py = round(self.y * float(height) / 1000.0, 2)
        px = max(0.0, min(float(width), px))
        py = max(0.0, min(float(height), py))
        return {
            "label": self.name,
            "type": "points",
            "points": [px, py],
            "outside": outside,
            "occluded": occluded,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class BuddhaArmInstance:
    """A single extra Buddha arm instance with 3 canonical joints."""
    arm_id: str  # e.g. 'arm_001'
    root: BuddhaArmKeypoint
    elbow: BuddhaArmKeypoint
    wrist: BuddhaArmKeypoint
    confidence: float = 1.0
    hand_roi: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] in [0..1000]
    group_id: Optional[int] = None
    side: Optional[str] = None  # 'left' or 'right' in viewer space

    def derive_hand_roi(self, size_norm: float = 70.0) -> List[int]:
        """Derive hand crop ROI centered at wrist if hand_roi is not explicitly set."""
        if self.hand_roi and len(self.hand_roi) == 4:
            return self.hand_roi
        # Heuristic: square crop centered around wrist with extent based on elbow->wrist distance
        dx = self.wrist.x - self.elbow.x
        dy = self.wrist.y - self.elbow.y
        dist = math.hypot(dx, dy)
        half_size = max(size_norm * 0.5, dist * 0.4)
        ymin = int(round(max(0.0, self.wrist.y - half_size)))
        xmin = int(round(max(0.0, self.wrist.x - half_size)))
        ymax = int(round(min(1000.0, self.wrist.y + half_size)))
        xmax = int(round(min(1000.0, self.wrist.x + half_size)))
        return [ymin, xmin, ymax, xmax]

    def derive_arm_roi(self, pad_ratio: float = 0.25) -> List[int]:
        """Derive bounding box [ymin, xmin, ymax, xmax] covering all 3 arm joints with padding."""
        xs = [self.root.x, self.elbow.x, self.wrist.x]
        ys = [self.root.y, self.elbow.y, self.wrist.y]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w, h = max(20.0, xmax - xmin), max(20.0, ymax - ymin)
        px = w * pad_ratio
        py = h * pad_ratio
        return [
            int(round(max(0.0, ymin - py))),
            int(round(max(0.0, xmin - px))),
            int(round(min(1000.0, ymax + py))),
            int(round(min(1000.0, xmax + px))),
        ]

    def to_cvat_skeleton(
        self,
        width: int = 1000,
        height: int = 1000,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Format as native CVAT skeleton shape dictionary."""
        w = int(img_w if img_w is not None else width)
        h = int(img_h if img_h is not None else height)
        elements = [
            self.root.to_cvat_element(w, h),
            self.elbow.to_cvat_element(w, h),
            self.wrist.to_cvat_element(w, h),
        ]
        res: Dict[str, Any] = {
            "label": "buddha_arm",
            "type": "skeleton",
            "confidence": round(self.confidence, 3),
            "elements": elements,
        }
        if self.group_id is not None:
            res["group_id"] = int(self.group_id)
            res["group"] = int(self.group_id)
        return res


@dataclass
class BuddhaHand21Keypoint:
    """A single hand joint (out of 21) in normalized [0..1000] coordinates."""
    id: int
    name: str
    x: float
    y: float
    visibility: int = VISIBILITY_VISIBLE
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        self.x = float(max(0.0, min(1000.0, self.x)))
        self.y = float(max(0.0, min(1000.0, self.y)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.visibility = normalize_visibility(self.visibility, strict=False, default=VISIBILITY_VISIBLE)

    @property
    def is_outside(self) -> bool:
        return self.visibility == VISIBILITY_OUTSIDE

    @property
    def is_occluded(self) -> bool:
        return self.visibility == VISIBILITY_OCCLUDED

    @property
    def is_visible(self) -> bool:
        return self.visibility == VISIBILITY_VISIBLE

    def to_cvat_element(self, width: int, height: int) -> Dict[str, Any]:
        outside, occluded = map_visibility_to_cvat(self.visibility)
        px = round(self.x * float(width) / 1000.0, 2)
        py = round(self.y * float(height) / 1000.0, 2)
        px = max(0.0, min(float(width), px))
        py = max(0.0, min(float(height), py))
        return {
            "label": self.name,
            "type": "points",
            "points": [px, py],
            "outside": outside,
            "occluded": occluded,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class BuddhaHand21Instance:
    """A 21-keypoint Buddha hand instance linked deterministically to an arm."""
    hand_id: str  # e.g. 'hand_001'
    landmarks: Dict[int, BuddhaHand21Keypoint]  # 0..20
    parent_arm_id: Optional[str] = None  # e.g. 'arm_001'
    confidence: float = 1.0
    group_id: Optional[int] = None

    def to_cvat_skeleton(
        self,
        width: int = 1000,
        height: int = 1000,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Format as native CVAT skeleton shape dictionary."""
        w = int(img_w if img_w is not None else width)
        h = int(img_h if img_h is not None else height)
        # Canonical order: 0 to 20
        elements = [
            self.landmarks[i].to_cvat_element(w, h)
            for i in range(21)
            if i in self.landmarks
        ]
        res: Dict[str, Any] = {
            "label": "buddha_hand",
            "type": "skeleton",
            "confidence": round(self.confidence, 3),
            "elements": elements,
        }
        if self.group_id is not None:
            res["group_id"] = int(self.group_id)
            res["group"] = int(self.group_id)
        return res


# ─── Coordinate Reprojection ───────────────────────────────────────────────────

def crop_norm_to_global_norm(
    crop_coords: Tuple[float, float],
    crop_box_norm: Sequence[Union[int, float]],
) -> Tuple[float, float]:
    """Reproject (x, y) from crop [0..1000] normalized space back to full image [0..1000] normalized space.

    Args:
        crop_coords: (x, y) in [0..1000] inside crop.
        crop_box_norm: [ymin, xmin, ymax, xmax] in [0..1000] full image space.

    Returns:
        (x_full, y_full) in [0..1000] rounded to 2 decimal places.
    """
    ymin, xmin, ymax, xmax = [float(v) for v in crop_box_norm]
    if (xmax - xmin) <= 0.0 or (ymax - ymin) <= 0.0:
        return (round(float(crop_coords[0]), 2), round(float(crop_coords[1]), 2))

    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)

    cx, cy = crop_coords
    gx = xmin + (cx / 1000.0) * w
    gy = ymin + (cy / 1000.0) * h

    gx_clamped = max(0.0, min(1000.0, round(gx, 2)))
    gy_clamped = max(0.0, min(1000.0, round(gy, 2)))
    return gx_clamped, gy_clamped


def global_norm_to_crop_norm(
    global_coords: Tuple[float, float],
    crop_box_norm: Sequence[Union[int, float]],
) -> Tuple[float, float]:
    """Project (x, y) from full image [0..1000] space into crop [0..1000] space."""
    ymin, xmin, ymax, xmax = [float(v) for v in crop_box_norm]
    if (xmax - xmin) <= 0.0 or (ymax - ymin) <= 0.0:
        return (round(float(global_coords[0]), 2), round(float(global_coords[1]), 2))

    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)

    gx, gy = global_coords
    cx = ((gx - xmin) / w) * 1000.0
    cy = ((gy - ymin) / h) * 1000.0

    return round(cx, 2), round(cy, 2)


# ─── Deterministic Arm Ordering ────────────────────────────────────────────────

def sort_arms_deterministically(
    arms: Sequence[BuddhaArmInstance],
    torso_center: Optional[Tuple[float, float]] = None,
) -> List[BuddhaArmInstance]:
    """Sort Buddha arms into a stable, deterministic sequence invariant to model output order.

    Strategy:
    1. Compute angle theta = atan2(wrist.y - cy, wrist.x - cx) around torso center.
       Angle is mapped to [0, 2*pi) starting clockwise from top (-pi/2, 12 o'clock).
    2. Primary key: angle theta (angular sweep around torso).
    3. Secondary key: distance of wrist from torso center (radial depth).
    4. Tertiary key: root.x then root.y (tie breakers).
    5. Re-assign arm_id sequentially ('arm_001', 'arm_002', ...) and group_id.

    Returns:
        New list of BuddhaArmInstance sorted deterministically.
    """
    if not arms:
        return []

    cx, cy = torso_center if torso_center is not None else (500.0, 500.0)

    def _arm_sort_key(arm: BuddhaArmInstance) -> Tuple[float, float, float, float]:
        dx = arm.wrist.x - cx
        dy = arm.wrist.y - cy
        # atan2 gives [-pi, pi]. Offset by +pi/2 so top is 0.0 radians
        angle = math.atan2(dy, dx) + (math.pi / 2.0)
        if angle < 0.0:
            angle += 2.0 * math.pi
        dist = math.hypot(dx, dy)
        return (round(angle, 4), round(dist, 2), round(arm.root.x, 2), round(arm.root.y, 2))

    sorted_arms = sorted(arms, key=_arm_sort_key)

    result: List[BuddhaArmInstance] = []
    for idx, raw_arm in enumerate(sorted_arms, start=1):
        arm_id = raw_arm.arm_id if raw_arm.arm_id else f"arm_{idx:03d}"
        group_id = 1 + idx
        side = "right" if raw_arm.wrist.x >= cx else "left"
        new_arm = BuddhaArmInstance(
            arm_id=arm_id,
            root=copy.deepcopy(raw_arm.root),
            elbow=copy.deepcopy(raw_arm.elbow),
            wrist=copy.deepcopy(raw_arm.wrist),
            confidence=raw_arm.confidence,
            hand_roi=list(raw_arm.hand_roi) if raw_arm.hand_roi else None,
            group_id=group_id,
            side=side,
        )
        result.append(new_arm)

    return result


# ─── Arm Deduplication ─────────────────────────────────────────────────────────

def _compute_vector_cosine_sim(
    p1: BuddhaArmKeypoint,
    p2: BuddhaArmKeypoint,
    q1: BuddhaArmKeypoint,
    q2: BuddhaArmKeypoint,
) -> float:
    """Compute cosine similarity between 2 vectors: (p2 - p1) and (q2 - q1)."""
    v1x, v1y = p2.x - p1.x, p2.y - p1.y
    v2x, v2y = q2.x - q1.x, q2.y - q1.y
    m1 = math.hypot(v1x, v1y)
    m2 = math.hypot(v2x, v2y)
    if m1 < 1e-6 or m2 < 1e-6:
        return 1.0
    dot = (v1x * v2x) + (v1y * v2y)
    return max(-1.0, min(1.0, dot / (m1 * m2)))


def _compute_box_iou(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Compute IoU between two boxes [ymin, xmin, ymax, xmax]."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def deduplicate_arms(
    arms: Sequence[BuddhaArmInstance],
    wrist_dist_thresh: float = 35.0,
    elbow_dist_thresh: float = 45.0,
    root_dist_thresh: float = 60.0,
    dir_sim_thresh: float = 0.92,
    hand_iou_thresh: float = 0.65,
) -> List[BuddhaArmInstance]:
    """Suppress duplicate arm detections using combined joint distances, limb directions, and IoU.

    Rules:
    - Never merge distinct parallel arms merely because they are close:
      Two arms are duplicates ONLY if:
      (a) Wrists are very close (< wrist_dist_thresh) AND elbows are close (< elbow_dist_thresh)
          AND overall limb direction is nearly identical (cosine similarity > dir_sim_thresh).
      OR
      (b) Wrists are close AND roots are close AND hand crops have high IoU (> hand_iou_thresh).
    - When duplicate is found, retain the one with higher confidence score.
    """
    if len(arms) <= 1:
        return list(arms)

    # Sort descending by confidence so highest confidence is evaluated first
    candidates = sorted(arms, key=lambda a: a.confidence, reverse=True)
    kept: List[BuddhaArmInstance] = []

    for cand in candidates:
        is_dup = False
        cand_hand_box = cand.derive_hand_roi()

        for accepted in kept:
            d_wrist = math.hypot(cand.wrist.x - accepted.wrist.x, cand.wrist.y - accepted.wrist.y)
            d_elbow = math.hypot(cand.elbow.x - accepted.elbow.x, cand.elbow.y - accepted.elbow.y)
            d_root = math.hypot(cand.root.x - accepted.root.x, cand.root.y - accepted.root.y)

            sim_forearm = _compute_vector_cosine_sim(cand.elbow, cand.wrist, accepted.elbow, accepted.wrist)
            acc_hand_box = accepted.derive_hand_roi()
            iou_hand = _compute_box_iou(cand_hand_box, acc_hand_box)

            # Criteria 1: Close wrists + close elbows + parallel forearm vector
            if d_wrist < wrist_dist_thresh and d_elbow < elbow_dist_thresh and sim_forearm >= dir_sim_thresh:
                is_dup = True
                break

            # Criteria 2: Close wrists + close roots + overlapping hand crop
            if d_wrist < wrist_dist_thresh and d_root < root_dist_thresh and iou_hand >= hand_iou_thresh:
                is_dup = True
                break

        if not is_dup:
            kept.append(cand)

    return kept


# ─── Hand-to-Arm Association ───────────────────────────────────────────────────

def link_hands_to_arms(
    arms: Sequence[BuddhaArmInstance],
    hands: Sequence[BuddhaHand21Instance],
    max_wrist_link_dist: float = 75.0,
) -> Tuple[List[BuddhaArmInstance], List[BuddhaHand21Instance]]:
    """Deterministically associate each hand with exactly one parent arm.

    Guarantees:
    - 1:1 association: Each arm has at most one hand, each hand has at most one arm.
    - Synchronizes group_id: hand.group_id = arm.group_id.
    - Matches hand root (wrist, point 0) to arm wrist joint in normalized space.
    - If a hand cannot find a valid arm within max_wrist_link_dist, it is dropped or unlinked.
    """
    if not arms or not hands:
        return list(arms), []

    # Map arm_id to arm
    arm_by_id = {arm.arm_id: arm for arm in arms}

    # Compute pairwise distance matrix between arm wrist and hand wrist (point 0)
    cost_pairs: List[Tuple[float, str, int]] = []
    for h_idx, hand in enumerate(hands):
        hand_wrist = hand.landmarks.get(0)
        if not hand_wrist or hand_wrist.is_outside:
            continue
        for arm in arms:
            dist = math.hypot(arm.wrist.x - hand_wrist.x, arm.wrist.y - hand_wrist.y)
            if dist <= max_wrist_link_dist:
                cost_pairs.append((dist, arm.arm_id, h_idx))

    # Sort pairs by distance (greedy minimum bipartite matching)
    cost_pairs.sort(key=lambda item: item[0])

    assigned_arms: Set[str] = set()
    assigned_hands: Set[int] = set()
    linked_hands: List[BuddhaHand21Instance] = []

    for dist, arm_id, h_idx in cost_pairs:
        if arm_id in assigned_arms or h_idx in assigned_hands:
            continue
        arm = arm_by_id[arm_id]
        raw_hand = hands[h_idx]

        assigned_arms.add(arm_id)
        assigned_hands.add(h_idx)

        new_hand = BuddhaHand21Instance(
            hand_id=raw_hand.hand_id if raw_hand.hand_id else f"hand_{len(linked_hands) + 1:03d}",
            parent_arm_id=arm.arm_id,
            landmarks=copy.deepcopy(raw_hand.landmarks),
            confidence=min(arm.confidence, raw_hand.confidence),
            group_id=arm.group_id,
        )
        linked_hands.append(new_hand)

    # Sort linked hands to match the arm order
    arm_order = {arm.arm_id: idx for idx, arm in enumerate(arms)}
    linked_hands.sort(key=lambda h: arm_order.get(h.parent_arm_id or "", 9999))

    # Any unlinked hands get their own unique group_id and parent_arm_id=None
    arm_group_ids = [a.group_id for a in arms if a.group_id is not None]
    max_arm_group = max(arm_group_ids) if arm_group_ids else 1
    next_group_id = max_arm_group + 1

    for h_idx, raw_hand in enumerate(hands):
        if h_idx not in assigned_hands:
            unlinked_hand = BuddhaHand21Instance(
                hand_id=raw_hand.hand_id if raw_hand.hand_id else f"hand_{len(linked_hands) + 1:03d}",
                parent_arm_id=None,
                landmarks=copy.deepcopy(raw_hand.landmarks),
                confidence=raw_hand.confidence,
                group_id=next_group_id,
            )
            next_group_id += 1
            linked_hands.append(unlinked_hand)

    return list(arms), linked_hands


# ─── Geometry Quality Gates ────────────────────────────────────────────────────

def assess_buddha_arm_geometry(
    arm: BuddhaArmInstance,
    min_length_pixels: float = 5.0,
    img_w: int = 1000,
    img_h: int = 1000,
) -> Tuple[bool, List[str]]:
    """Assess whether a buddha_arm candidate represents valid, non-collapsed geometry."""
    errors: List[str] = []

    for pt in (arm.root, arm.elbow, arm.wrist):
        if math.isnan(pt.x) or math.isnan(pt.y) or math.isinf(pt.x) or math.isinf(pt.y):
            errors.append(f"Joint {pt.name} contains NaN or Inf coordinates")
        if not (0.0 <= pt.x <= 1000.0 and 0.0 <= pt.y <= 1000.0):
            errors.append(f"Joint {pt.name} out of bounds: ({pt.x}, {pt.y})")

    # Check zero-length or collapsed arm
    d1 = math.hypot((arm.elbow.x - arm.root.x) * img_w / 1000.0, (arm.elbow.y - arm.root.y) * img_h / 1000.0)
    d2 = math.hypot((arm.wrist.x - arm.elbow.x) * img_w / 1000.0, (arm.wrist.y - arm.elbow.y) * img_h / 1000.0)
    total_len = d1 + d2

    if total_len < min_length_pixels:
        errors.append(f"Arm total length {total_len:.1f}px is under minimum threshold {min_length_pixels}px")

    if d1 < 1.0 and d2 < 1.0:
        errors.append("Arm joints completely collapsed to a single coordinate")

    return len(errors) == 0, errors


def assess_buddha_hand21_geometry(
    hand: BuddhaHand21Instance,
    min_visible_points: int = 5,
    max_collapse_dist: float = 2.0,
    img_w: int = 1000,
    img_h: int = 1000,
) -> Tuple[bool, List[str]]:
    """Assess whether a buddha_hand candidate represents valid 21-point topology."""
    errors: List[str] = []

    if len(hand.landmarks) != 21:
        errors.append(f"Hand must have exactly 21 landmarks, found {len(hand.landmarks)}")

    active_pts = [lm for lm in hand.landmarks.values() if not lm.is_outside]
    if len(active_pts) < min_visible_points:
        errors.append(f"Hand has only {len(active_pts)} visible points (min {min_visible_points})")

    for lm in hand.landmarks.values():
        if math.isnan(lm.x) or math.isnan(lm.y) or math.isinf(lm.x) or math.isinf(lm.y):
            errors.append(f"Hand landmark {lm.name} ({lm.id}) contains NaN or Inf")
            break

    # Check for all-points collapse to a single point
    if len(active_pts) >= 5:
        xs = [lm.x * img_w / 1000.0 for lm in active_pts]
        ys = [lm.y * img_h / 1000.0 for lm in active_pts]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x < max_collapse_dist and span_y < max_collapse_dist:
            errors.append(f"Hand collapsed to degenerate cluster (span: {span_x:.1f}x{span_y:.1f}px)")

    return len(errors) == 0, errors


# ─── Model Resolution (Opus 5.5 Policy) ────────────────────────────────────────

OPUS_5_5_PATTERNS: Tuple[str, ...] = (
    "claude-opus-5-5",
    "claude-opus-5.5",
    "claude-opus-5",
    "opus-5-5",
    "opus-5.5",
    "opus-5",
)

OPUS_FALLBACK_PATTERNS: Tuple[str, ...] = (
    "ag/claude-opus-4-6-thinking",
    "claude-opus-4-6-thinking",
    "ag/claude-opus-4-6",
    "claude-opus-4-6",
    "ag/claude-sonnet-4-6",
    "claude-sonnet-4-6",
    "ag/gemini-3.8-flash-high",
    "ag/gemini-3.8-flash-low",
)


def resolve_buddha_model(
    client: Any,
    preferred_model: Optional[str] = None,
    allow_fallback: Optional[bool] = None,
) -> str:
    """Resolve vision inference model for Buddha Multi-Limb detector.

    Policy:
    1. If BUDDHA_MODEL (or preferred_model) is explicitly configured:
       - Must exist in 9Router /v1/models.
       - Otherwise raises NineRouterError showing available vision models.
    2. Otherwise locate Claude Opus 5.5 by capability and name matching.
    3. Strict mode: Do NOT silently fall back to Gemini or older Claude by default.
    4. Optional fallback exists ONLY when explicitly enabled via:
       BUDDHA_ALLOW_MODEL_FALLBACK=1 (or allow_fallback=True).
    5. Returns resolved model ID string.
    """
    from app.client import NineRouterError

    env_model = os.getenv("BUDDHA_MODEL")
    target_override = preferred_model or env_model

    if hasattr(client, "get_vision_model_ids"):
        available_ids = client.get_vision_model_ids()
    elif hasattr(client, "list_models"):
        available_ids = client.list_models()
    else:
        available_ids = []
    base_url = getattr(client, "base_url", "http://127.0.0.1:20128")
    if not available_ids:
        raise NineRouterError(f"No vision-capable models found in 9Router at {base_url}.")

    # Policy 1: Explicit model configured
    if target_override and target_override.strip():
        target = target_override.strip()
        if target in available_ids:
            logger.info(f"Using explicitly configured Buddha vision model: '{target}'")
            return target
        available_str = ", ".join(available_ids)
        raise NineRouterError(
            f"Preferred vision model '{target}' not found in 9Router. "
            f"Available vision models: {available_str}"
        )

    # Policy 2: Search for Claude Opus 5.5
    for m_id in available_ids:
        m_lower = m_id.lower()
        if any(pat in m_lower for pat in OPUS_5_5_PATTERNS):
            logger.info(f"Discovered and resolved Claude Opus 5.5 model: '{m_id}'")
            return m_id

    # Policy 3 & 4: Gated Fallback
    fb_env = os.getenv("BUDDHA_ALLOW_MODEL_FALLBACK", "0").strip().lower()
    is_fallback_allowed = allow_fallback if allow_fallback is not None else (fb_env in ("1", "true", "yes"))

    if not is_fallback_allowed:
        available_str = ", ".join(available_ids)
        raise NineRouterError(
            f"Claude Opus 5.5 vision model was not found in 9Router at {base_url}, and "
            f"BUDDHA_ALLOW_MODEL_FALLBACK is disabled (strict policy). "
            f"Available vision models: {available_str}. "
            f"Set BUDDHA_MODEL=<exact-id> or BUDDHA_ALLOW_MODEL_FALLBACK=1 to permit fallback."
        )

    # Fallback to best available Opus/Claude/Flash model
    for fb_pat in OPUS_FALLBACK_PATTERNS:
        for m_id in available_ids:
            if fb_pat in m_id.lower():
                logger.warning(
                    f"Claude Opus 5.5 not found. BUDDHA_ALLOW_MODEL_FALLBACK=1 active: "
                    f"Falling back to '{m_id}'"
                )
                return m_id

    # Ultimate fallback to first available vision model
    first_model = available_ids[0]
    logger.warning(
        f"Claude Opus 5.5 not found. BUDDHA_ALLOW_MODEL_FALLBACK=1 active: "
        f"Falling back to first available model '{first_model}'"
    )
    return first_model


# ─── Prompt Engineering ────────────────────────────────────────────────────────

def build_buddha_global_prompt() -> str:
    """Build Pass 1 Global Scene Discovery prompt.

    Enforces:
    - Single main/central body (Pose17).
    - Single main central face ROI.
    - Exhaustive recall of ALL visible extra arms.
    - Torso center reference point.
    - Strict prohibition of decorative artifacts (crowns, lotus petals, halos, jewelry) as limbs.
    - Never assume a fixed number of arms (dynamic count).
    """
    return (
        "You are an expert AI vision perception system specializing in Buddhist iconography, "
        "specifically multi-armed and thousand-armed Buddha/Bodhisattva deities (e.g. Avalokiteshvara/Guanyin).\n\n"
        "TASK:\n"
        "Detect the CENTRAL BODY, CENTRAL FACE, and EVERY VISIBLE EXTRA ARM independently.\n\n"
        "CRITICAL RULES & PROHIBITIONS:\n"
        "1. SINGLE CENTRAL BODY:\n"
        "   - Identify the single main/central Buddha body.\n"
        "   - Do NOT interpret the additional radial arms as multiple people. There is only ONE person.\n"
        "   - Estimate the central Pose17 keypoints for the main torso and legs.\n"
        "2. CENTRAL FACE ONLY:\n"
        "   - Identify the bounding box of the single primary central face.\n"
        "   - Do NOT detect faces in crown ornaments, background statues, or decorative halos.\n"
        "3. DYNAMIC EXTRA ARMS (HIGH RECALL):\n"
        "   - Detect EVERY visible extra arm radiating from the torso/shoulders.\n"
        "   - The figure may have 4, 8, 18, 42, or 100+ arms. NEVER assume a fixed arm count.\n"
        "   - For each arm, predict 3 joints: [root, elbow, wrist] and an approximate hand bounding box 'hand_roi'.\n"
        "4. ANTI-HALLUCINATION / DECORATIVE SEPARATION:\n"
        "   - CROWNS and headpieces are NOT limbs or heads.\n"
        "   - LOTUS PETALS on the throne are NOT hands or feet.\n"
        "   - JEWELRY, necklaces, and ribbons are NOT fingers or bones.\n"
        "   - ROBE FOLDS and drapery are NOT arms.\n"
        "   - HALO RAYS and background flame ornaments are NOT hands.\n"
        "   - Only label physical anatomical limb structures connected to the deity.\n\n"
        "COORDINATE & VISIBILITY CONVENTIONS:\n"
        "- All coordinates MUST be normalized integers in [0, 1000] relative to image width and height.\n"
        "- Joint format: [x, y, visibility] where:\n"
        "  0 = outside image frame\n"
        "  1 = present on body but occluded behind another limb/object\n"
        "  2 = clearly visible\n"
        "- Bounding box format: [ymin, xmin, ymax, xmax] in [0, 1000].\n\n"
        "REQUIRED OUTPUT JSON FORMAT (STRICT JSON ONLY, NO MARKDOWN, NO EXPLANATION):\n"
        "{\n"
        '  "torso_center": [x, y],\n'
        '  "face_roi": [ymin, xmin, ymax, xmax],\n'
        '  "central_body": {\n'
        '    "label": "person",\n'
        '    "confidence": 0.95,\n'
        '    "box_2d": [ymin, xmin, ymax, xmax],\n'
        '    "keypoints": {\n'
        '      "nose": [x, y, 2],\n'
        '      "right_eye": [x, y, 2],\n'
        '      "left_eye": [x, y, 2],\n'
        '      "right_shoulder": [x, y, 2],\n'
        '      "left_shoulder": [x, y, 2],\n'
        '      "right_elbow": [x, y, 2],\n'
        '      "left_elbow": [x, y, 2],\n'
        '      "right_wrist": [x, y, 2],\n'
        '      "left_wrist": [x, y, 2],\n'
        '      "right_hip": [x, y, 2],\n'
        '      "left_hip": [x, y, 2],\n'
        '      "right_knee": [x, y, 2],\n'
        '      "left_knee": [x, y, 2],\n'
        '      "right_ankle": [x, y, 2],\n'
        '      "left_ankle": [x, y, 2]\n'
        "    }\n"
        "  },\n"
        '  "arms": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "confidence": 0.92,\n'
        '      "root": [x, y, 2],\n'
        '      "elbow": [x, y, 2],\n'
        '      "wrist": [x, y, 2],\n'
        '      "hand_roi": [ymin, xmin, ymax, xmax]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def build_buddha_arm_refine_prompt() -> str:
    """Build Pass 2 Arm Crop Refinement prompt."""
    return (
        "High-resolution close-up refinement of a single Buddhist deity arm.\n\n"
        "TASK:\n"
        "Precisely locate the 3 kinematic arm joints in this cropped patch: ROOT, ELBOW, and WRIST.\n\n"
        "RULES:\n"
        "1. 'root': Joint where arm connects to shoulder or torso cluster.\n"
        "2. 'elbow': Articulated elbow joint.\n"
        "3. 'wrist': Base of the hand where palm begins.\n"
        "4. Enforce strict anatomical connectivity: root -> elbow -> wrist.\n"
        "5. Do NOT confuse robe folds, weapons, or mudra scepters with the arm limb.\n"
        "6. Coordinates: Normalized integers [x, y, visibility] in [0, 1000] relative to THIS CROPPED PATCH.\n"
        "7. Visibility: 0=outside patch, 1=occluded, 2=visible.\n\n"
        "RETURN STRICT JSON ONLY:\n"
        "{\n"
        '  "confidence": 0.95,\n'
        '  "root": [x, y, 2],\n'
        '  "elbow": [x, y, 2],\n'
        '  "wrist": [x, y, 2],\n'
        '  "hand_roi": [ymin, xmin, ymax, xmax]\n'
        "}\n"
    )


def build_buddha_hand21_prompt() -> str:
    """Build Pass 3 Hand21 Landmark prompt."""
    return (
        "High-resolution hand landmark estimation for a Buddhist deity hand (Mudra / Ritual Gesture).\n\n"
        "TASK:\n"
        "Estimate exactly 21 canonical hand keypoints (21 2D hand landmark coordinates, 0..20) for the primary hand in this cropped patch.\n\n"
        "TOPOLOGY & JOINT INDEXING:\n"
        "  0: wrist (root of hand)\n"
        "  Thumb:  1 (thumb_cmc) -> 2 (thumb_mcp) -> 3 (thumb_ip) -> 4 (thumb_tip)\n"
        "  Index:  5 (index_mcp) -> 6 (index_pip) -> 7 (index_dip) -> 8 (index_tip)\n"
        "  Middle: 9 (middle_mcp) -> 10 (middle_pip) -> 11 (middle_dip) -> 12 (middle_tip)\n"
        "  Ring:   13 (ring_mcp) -> 14 (ring_pip) -> 15 (ring_dip) -> 16 (ring_tip)\n"
        "  Pinky:  17 (pinky_mcp) -> 18 (pinky_pip) -> 19 (pinky_dip) -> 20 (pinky_tip)\n\n"
        "MUDRA & RITUAL GESTURE RULES:\n"
        "- Buddha hands often hold ritual items (lotus, wheel, vase, sword, rosary bead) or form mudras.\n"
        "- Scepters, lotus stems, or beads are NOT fingers. Only mark the actual anatomical digits.\n"
        "- If fingers curl inward or overlap behind another finger, mark visibility = 1 (occluded).\n"
        "- Coordinates: Normalized integers [x, y] in [0, 1000] relative to THIS CROPPED PATCH.\n"
        "- Visibility: 0=outside patch, 1=occluded, 2=visible.\n\n"
        "RETURN STRICT JSON ONLY:\n"
        "{\n"
        '  "confidence": 0.95,\n'
        '  "landmarks": {\n'
        '    "0": [x, y, 2],\n'
        '    "1": [x, y, 2],\n'
        '    "2": [x, y, 2],\n'
        '    "3": [x, y, 2],\n'
        '    "4": [x, y, 2],\n'
        '    "5": [x, y, 2],\n'
        '    "6": [x, y, 2],\n'
        '    "7": [x, y, 2],\n'
        '    "8": [x, y, 2],\n'
        '    "9": [x, y, 2],\n'
        '    "10": [x, y, 2],\n'
        '    "11": [x, y, 2],\n'
        '    "12": [x, y, 2],\n'
        '    "13": [x, y, 2],\n'
        '    "14": [x, y, 2],\n'
        '    "15": [x, y, 2],\n'
        '    "16": [x, y, 2],\n'
        '    "17": [x, y, 2],\n'
        '    "18": [x, y, 2],\n'
        '    "19": [x, y, 2],\n'
        '    "20": [x, y, 2]\n'
        "  }\n"
        "}\n"
    )


# ─── CVAT Skeleton Spec Builders ───────────────────────────────────────────────

def generate_buddha_arm_svg() -> str:
    """Generate SVG representation for 3-joint buddha_arm skeleton."""
    sublabels = ["root", "elbow", "wrist"]
    edges = [(1, 2), (2, 3)]
    node_coords = {
        1: (20, 50),
        2: (50, 30),
        3: (80, 50),
    }
    return generate_svg_representation(sublabels, edges, id_offset=1, width=100, height=100, node_coords=node_coords)


def generate_buddha_hand_svg() -> str:
    """Generate SVG representation for 21-joint buddha_hand skeleton."""
    schema = load_buddha_schema()
    sublabels = [kp.name for kp in schema.hand_keypoints]
    edges_1based = [(u + 1, v + 1) for u, v in schema.hand_edges]

    # Node positions on 100x100 canvas (1-based node IDs 1..21)
    node_coords: Dict[int, Tuple[int, int]] = {
        1: (50, 90),   # wrist
        # Thumb
        2: (30, 80), 3: (20, 65), 4: (15, 50), 5: (10, 35),
        # Index
        6: (35, 55), 7: (32, 40), 8: (30, 25), 9: (28, 10),
        # Middle
        10: (50, 50), 11: (50, 35), 12: (50, 20), 13: (50, 5),
        # Ring
        14: (65, 55), 15: (68, 40), 16: (70, 25), 17: (72, 10),
        # Pinky
        18: (78, 65), 19: (82, 52), 20: (86, 40), 21: (90, 28),
    }

    return generate_svg_representation(sublabels, edges_1based, id_offset=1, width=100, height=100, node_coords=node_coords)


def build_cvat_buddha_arm_spec(label_id: int = 9) -> Dict[str, Any]:
    """Build CVAT spec item for buddha_arm skeleton."""
    sublabels = [
        {"id": 1, "name": "root", "type": "points", "attributes": []},
        {"id": 2, "name": "elbow", "type": "points", "attributes": []},
        {"id": 3, "name": "wrist", "type": "points", "attributes": []},
    ]
    return {
        "id": label_id,
        "name": "buddha_arm",
        "type": "skeleton",
        "attributes": [],
        "sublabels": sublabels,
        "svg": generate_buddha_arm_svg(),
    }


def build_cvat_buddha_hand_spec(label_id: int = 10) -> Dict[str, Any]:
    """Build CVAT spec item for buddha_hand skeleton."""
    schema = load_buddha_schema()
    sublabels = [
        {"id": kp.id + 1, "name": kp.name, "type": "points", "attributes": []}
        for kp in schema.hand_keypoints
    ]
    return {
        "id": label_id,
        "name": "buddha_hand",
        "type": "skeleton",
        "attributes": [],
        "sublabels": sublabels,
        "svg": generate_buddha_hand_svg(),
    }


def build_cvat_buddha_multilimbs_full_spec() -> List[Dict[str, Any]]:
    """Build the complete 10-label CVAT annotations spec for ninerouter-buddha-multilimbs.

    Structure:
    1. person (Pose17 skeleton)
    2..8. VF50 7 component skeletons (longmaytrai..moitrong)
    9. buddha_arm (3-joint skeleton)
    10. buddha_hand (21-joint skeleton)
    """
    from core.skeleton_contract import build_cvat_pose17_spec, build_cvat_vf50_spec

    specs: List[Dict[str, Any]] = []

    # 1. Central Body Pose17
    specs.append(build_cvat_pose17_spec(parent_label="person", label_id=1, numeric_sublabels=True))

    # 2..8. Central Face VF-50 (7 components)
    vf50_specs = build_cvat_vf50_spec()
    for idx, s in enumerate(vf50_specs, start=2):
        s_copy = copy.deepcopy(s)
        s_copy["id"] = idx
        specs.append(s_copy)

    # 9. Extra Buddha Arm
    specs.append(build_cvat_buddha_arm_spec(label_id=9))

    # 10. Buddha Hand21
    specs.append(build_cvat_buddha_hand_spec(label_id=10))

    return specs

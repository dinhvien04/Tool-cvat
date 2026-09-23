"""Canonical Schema, Taxonomy, and Geometry Contract for Week-2: Pose17 and VF50.

Specifications:
1. POSE17: Standard COCO 17-keypoint Human Pose estimation schema for driver cabin.
   - Exact 17 keypoints.
   - VinFast Frame-based laterality convention:
     * R_* (even index: 2, 4, 6, 8, 10, 12, 14, 16) lies on right side of display frame.
     * L_* (odd index: 3, 5, 7, 9, 11, 13, 15, 17) lies on left side of display frame.
     * 1: Nose (central).
   - Limb graph topology with 16-19 edges.

2. VF50: VinFast 50-landmark Facial Landmark schema.
   - Exact 50 landmarks partitioned into 7 official VinFast skeletons:
     * longmaytrai (0 - 4): 5 points (open contour, outer to inner)
     * longmayphai (5 - 9): 5 points (open contour, inner to outer)
     * songmui     (10 - 13): 4 points (open contour, bridge top to base)
     * mattrai     (14 - 21): 8 points (closed contour, outer corner -> upper lid -> inner corner -> lower lid)
     * matphai     (22 - 29): 8 points (closed contour, inner corner -> upper lid -> outer corner -> lower lid)
     * moingoai    (30 - 41): 12 points (closed contour, left corner -> upper lip -> right corner -> lower lip)
     * moitrong    (42 - 49): 8 points (closed contour, left inner corner -> upper inner -> right inner -> lower inner)
     Total: 5 + 5 + 4 + 8 + 8 + 12 + 8 = 50 points (IDs 0..49).
   - Laterality convention: "matphai" / "mattrai" defined strictly by display image view (frame-based):
     * *trai lies on left side of face in frame (smaller x coordinates).
     * *phai lies on right side of face in frame (larger x coordinates).

3. CVAT Serverless Skeleton Contract:
   - Parent label: type "skeleton"
   - Child sublabels: type "points"
   - SVG adjacency lines with data-node-id attributes
   - Human editability compliance (outside, occluded, points).
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# ==============================================================================
# 1. POSE17 CANONICAL SPECIFICATION
# ==============================================================================

POSE17_KEYPOINTS: Tuple[str, ...] = (
    "nose",             # 1
    "right_eye",        # 2 (R Eye - VinFast order: R is even)
    "left_eye",         # 3 (L Eye - VinFast order: L is odd)
    "right_ear",        # 4
    "left_ear",         # 5
    "right_shoulder",   # 6
    "left_shoulder",    # 7
    "right_elbow",      # 8
    "left_elbow",       # 9
    "right_wrist",      # 10
    "left_wrist",       # 11
    "right_hip",        # 12
    "left_hip",         # 13
    "right_knee",       # 14
    "left_knee",        # 15
    "right_ankle",      # 16
    "left_ankle",       # 17
)

POSE17_KEYPOINT_TO_ID: Dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(POSE17_KEYPOINTS)
}
POSE17_ID_TO_KEYPOINT: Dict[int, str] = {
    idx + 1: name for idx, name in enumerate(POSE17_KEYPOINTS)
}

# Standard 17-keypoint skeleton edges (1-indexed node IDs)
# Matches config/week2_pose17.yaml authoritative edge set (18 edges including ear-to-shoulder)
POSE17_EDGES: Tuple[Tuple[int, int], ...] = (
    (1, 2),    # nose -> right_eye
    (1, 3),    # nose -> left_eye
    (2, 4),    # right_eye -> right_ear
    (3, 5),    # left_eye -> left_ear
    (4, 6),    # right_ear -> right_shoulder (ear-to-shoulder per YAML)
    (5, 7),    # left_ear -> left_shoulder (ear-to-shoulder per YAML)
    (6, 7),    # right_shoulder -> left_shoulder
    (6, 8),    # right_shoulder -> right_elbow
    (8, 10),   # right_elbow -> right_wrist
    (7, 9),    # left_shoulder -> left_elbow
    (9, 11),   # left_elbow -> left_wrist
    (6, 12),   # right_shoulder -> right_hip
    (7, 13),   # left_shoulder -> left_hip
    (12, 13),  # right_hip -> left_hip
    (12, 14),  # right_hip -> right_knee
    (14, 16),  # right_knee -> right_ankle
    (13, 15),  # left_hip -> left_knee
    (15, 17),  # left_knee -> left_ankle
)

POSE17_LATERALITY = "frame_based_vf"  # R on right of frame, L on left of frame


# ==============================================================================
# 2. VF50 CANONICAL SPECIFICATION (Official VinFast Schema)
# ==============================================================================

VIN_VF50_COMPONENTS: Dict[str, Tuple[int, int, str]] = {
    # name: (start_id, end_id, topology)
    "longmaytrai": (0, 4, "open"),     # 5 points
    "longmayphai": (5, 9, "open"),     # 5 points
    "songmui":     (10, 13, "open"),   # 4 points
    "mattrai":     (14, 21, "closed"), # 8 points
    "matphai":     (22, 29, "closed"), # 8 points
    "moingoai":    (30, 41, "closed"), # 12 points
    "moitrong":    (42, 49, "closed"), # 8 points
}

VF50_COMPONENT_COUNTS: Dict[str, int] = {
    name: (end_id - start_id + 1)
    for name, (start_id, end_id, _) in VIN_VF50_COMPONENTS.items()
}

def _generate_vf50_landmarks() -> Tuple[str, ...]:
    names: List[str] = []
    for comp, (start_id, end_id, _) in VIN_VF50_COMPONENTS.items():
        for i in range(start_id, end_id + 1):
            names.append(f"{comp}_{i:02d}")
    return tuple(names)

VF50_LANDMARKS: Tuple[str, ...] = _generate_vf50_landmarks()
assert len(VF50_LANDMARKS) == 50, f"VF50 must have exactly 50 landmarks, got {len(VF50_LANDMARKS)}"

# Point IDs are continuous 0..49
VF50_LANDMARK_TO_ID: Dict[str, int] = {
    name: int(name.split("_")[-1]) for name in VF50_LANDMARKS
}
VF50_ID_TO_LANDMARK: Dict[int, str] = {
    nid: name for name, nid in VF50_LANDMARK_TO_ID.items()
}

def _generate_vf50_edges() -> Tuple[Tuple[int, int], ...]:
    edges: List[Tuple[int, int]] = []
    for comp, (start_id, end_id, topology) in VIN_VF50_COMPONENTS.items():
        for i in range(start_id, end_id):
            edges.append((i, i + 1))
        if topology == "closed":
            edges.append((end_id, start_id))
    return tuple(edges)

VF50_EDGES: Tuple[Tuple[int, int], ...] = _generate_vf50_edges()

def _build_vf50_mirror_map() -> Dict[str, str]:
    """Mirror map for horizontal reflection.
    Swaps *trai <-> *phai components and reverses symmetry.
    """
    mirror: Dict[str, str] = {}
    # longmaytrai (0..4) <-> longmayphai (5..9)
    # 0 (outer left) mirrors to 9 (outer right)
    for i in range(5):
        l_name = f"longmaytrai_{i:02d}"
        r_name = f"longmayphai_{9 - i:02d}"
        mirror[l_name] = r_name
        mirror[r_name] = l_name

    # songmui (10..13): bridge is vertical center line -> self-mirrors
    for i in range(10, 14):
        name = f"songmui_{i:02d}"
        mirror[name] = name

    # mattrai (14..21) <-> matphai (22..29)
    # mattrai: 14 khoe ngoai, 18 khoe trong
    # matphai: 22 khoe trong, 26 khoe ngoai
    # 14 (khoe ngoai trai) mirrors to 26 (khoe ngoai phai)
    # 18 (khoe trong trai) mirrors to 22 (khoe trong phai)
    eye_l_to_r = {
        14: 26, 15: 25, 16: 24, 17: 23,
        18: 22, 19: 29, 20: 28, 21: 27,
    }
    for l_id, r_id in eye_l_to_r.items():
        l_name = f"mattrai_{l_id:02d}"
        r_name = f"matphai_{r_id:02d}"
        mirror[l_name] = r_name
        mirror[r_name] = l_name

    # moingoai (30..41): 30 khoe trai <-> 36 khoe phai
    lip_outer_mirror = {
        30: 36, 31: 35, 32: 34, 33: 33, 34: 32, 35: 31, 36: 30,
        37: 41, 38: 40, 39: 39, 40: 38, 41: 37,
    }
    for a, b in lip_outer_mirror.items():
        mirror[f"moingoai_{a:02d}"] = f"moingoai_{b:02d}"

    # moitrong (42..49): 42 khoe trai <-> 46 khoe phai
    lip_inner_mirror = {
        42: 46, 43: 45, 44: 44, 45: 43, 46: 42,
        47: 49, 48: 48, 49: 47,
    }
    for a, b in lip_inner_mirror.items():
        mirror[f"moitrong_{a:02d}"] = f"moitrong_{b:02d}"

    return mirror

VF50_MIRROR_MAP: Dict[str, str] = _build_vf50_mirror_map()
VF50_LATERALITY = "matphai_mattrai"


# ==============================================================================
# 3. CVAT SKELETON SPEC BUILDER & VALIDATOR
# ==============================================================================

def generate_svg_representation(
    sublabels: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    id_offset: int = 1,
    width: int = 100,
    height: int = 100,
    node_coords: Optional[Dict[int, Tuple[int, int]]] = None,
) -> str:
    """Build an SVG string connecting node IDs for CVAT skeleton visualization."""
    lines: List[str] = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    coord_map: Dict[int, Tuple[int, int]] = {}
    for idx, name in enumerate(sublabels):
        nid = idx + id_offset
        if node_coords and nid in node_coords:
            cx, cy = node_coords[nid]
        else:
            cx, cy = width // 2, height // 2
        coord_map[nid] = (cx, cy)
        lines.append(
            f'  <circle id="node_{nid}" data-type="element node" data-element-id="{nid}" '
            f'data-node-id="{nid}" data-label-name="{name}" cx="{cx}" cy="{cy}" r="3" />'
        )
    for u, v in edges:
        x1, y1 = coord_map.get(u, (0, 0))
        x2, y2 = coord_map.get(v, (0, 0))
        lines.append(
            f'  <line id="edge_{u}_{v}" data-type="edge" data-node-from="{u}" data-node-to="{v}" '
            f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="red" stroke-width="1" />'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def build_cvat_skeleton_spec(
    parent_label: str,
    sublabels: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    label_id: int = 1,
    id_offset: int = 1,
    node_coords: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Construct CVAT function.yaml spec entry for a skeleton detector."""
    sublabel_items: List[Dict[str, Any]] = []
    for idx, name in enumerate(sublabels):
        nid = idx + id_offset
        sublabel_items.append({
            "id": nid,
            "name": name,
            "type": "points",
            "attributes": [],
        })

    svg_str = generate_svg_representation(sublabels, edges, id_offset=id_offset, node_coords=node_coords)

    return {
        "id": label_id,
        "name": parent_label,
        "type": "skeleton",
        "attributes": [],
        "sublabels": sublabel_items,
        "svg": svg_str,
    }


def validate_svg_node_ids(svg_content: str, expected_node_ids: Set[int]) -> Tuple[bool, List[str]]:
    """Validate SVG XML string for well-formedness and proper node ID references."""
    errors: List[str] = []
    try:
        root = ET.fromstring(svg_content)
    except Exception as e:
        return False, [f"Invalid SVG XML: {e}"]

    found_nodes: Set[int] = set()
    for elem in root.iter():
        node_id_attr = elem.attrib.get("data-node-id")
        if node_id_attr is not None:
            try:
                nid = int(node_id_attr)
                found_nodes.add(nid)
            except ValueError:
                errors.append(f"Non-integer data-node-id attribute: {node_id_attr!r}")

        from_attr = elem.attrib.get("data-node-from")
        to_attr = elem.attrib.get("data-node-to")
        if from_attr is not None and to_attr is not None:
            try:
                u, v = int(from_attr), int(to_attr)
                if u not in expected_node_ids:
                    errors.append(f"Edge references unknown source node: {u}")
                if v not in expected_node_ids:
                    errors.append(f"Edge references unknown target node: {v}")
                if u == v:
                    errors.append(f"Self-loop edge detected on node: {u}")
            except ValueError:
                errors.append(f"Non-integer node endpoints: from={from_attr!r}, to={to_attr!r}")

    missing_nodes = expected_node_ids - found_nodes
    if missing_nodes:
        errors.append(f"SVG missing node declarations for IDs: {sorted(missing_nodes)}")

    return len(errors) == 0, errors


# ==============================================================================
# 4. GEOMETRY & COORDINATE DENORMALIZATION
# ==============================================================================

def denormalize_point(
    x: float,
    y: float,
    img_width: int,
    img_height: int,
    coord_range: float = 1000.0,
    clamp: bool = True,
) -> Tuple[float, float]:
    """Convert normalized coordinates (e.g. 0..1000 or 0..1) to absolute pixels."""
    if coord_range <= 1.0:
        px = x * img_width
        py = y * img_height
    else:
        px = (x / coord_range) * img_width
        py = (y / coord_range) * img_height

    if clamp:
        px = max(0.0, min(float(img_width), px))
        py = max(0.0, min(float(img_height), py))

    return round(px, 2), round(py, 2)


def denormalize_keypoints_list(
    raw_points: Sequence[Sequence[float]],
    img_width: int,
    img_height: int,
    coord_range: float = 1000.0,
    clamp: bool = True,
) -> List[List[float]]:
    """Denormalize a sequence of [x, y] points to [x, y] pixel coordinates."""
    out: List[List[float]] = []
    for pt in raw_points:
        if len(pt) >= 2:
            x, y = float(pt[0]), float(pt[1])
            px, py = denormalize_point(x, y, img_width, img_height, coord_range=coord_range, clamp=clamp)
            out.append([px, py])
    return out


# ==============================================================================
# 5. SKELETON PARSER AND SANITIZER
# ==============================================================================

@dataclass
class KeypointElement:
    label: str
    points: List[float]
    type: str = "points"
    occluded: bool = False
    outside: bool = False
    attributes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SkeletonInstance:
    label: str
    elements: List[KeypointElement]
    type: str = "skeleton"
    confidence: float = 1.0
    group: int = 0
    attributes: List[Dict[str, Any]] = field(default_factory=list)

    def to_cvat_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "label": self.label,
            "confidence": self.confidence,
            "group": self.group,
            "attributes": self.attributes,
            "elements": [
                {
                    "type": elem.type,
                    "label": elem.label,
                    "points": elem.points,
                    "occluded": elem.occluded,
                    "outside": elem.outside,
                    "attributes": elem.attributes,
                }
                for elem in self.elements
            ],
        }


def parse_and_sanitize_pose17_instance(
    raw_data: Dict[str, Any],
    img_width: int,
    img_height: int,
    coord_range: float = 1000.0,
) -> Optional[SkeletonInstance]:
    """Parse and sanitize a raw Pose17 prediction into a valid SkeletonInstance."""
    raw_keypoints = raw_data.get("keypoints")
    if not raw_keypoints or not isinstance(raw_keypoints, (dict, list)):
        return None

    elements: List[KeypointElement] = []
    seen_labels: Set[str] = set()

    if isinstance(raw_keypoints, dict):
        for name in POSE17_KEYPOINTS:
            pt_data = raw_keypoints.get(name)
            if pt_data is None:
                elements.append(KeypointElement(
                    label=name,
                    points=[0.0, 0.0],
                    occluded=False,
                    outside=True,
                ))
                continue

            if name in seen_labels:
                continue
            seen_labels.add(name)

            if isinstance(pt_data, (list, tuple)) and len(pt_data) >= 2:
                x, y = float(pt_data[0]), float(pt_data[1])
                vis = pt_data[2] if len(pt_data) > 2 else 2

                # Official VinFast CVAT mapping:
                # vis=0 -> outside=True, occluded=False
                # vis=1 -> outside=False, occluded=True
                # vis=2 -> outside=False, occluded=False
                occluded = (vis == 1) if vis in (0, 1, 2) else False
                outside = (vis == 0) if vis in (0, 1, 2) else False

                px, py = denormalize_point(x, y, img_width, img_height, coord_range=coord_range, clamp=True)
                elements.append(KeypointElement(
                    label=name,
                    points=[px, py],
                    occluded=occluded,
                    outside=outside,
                ))

    elif isinstance(raw_keypoints, list):
        for item in raw_keypoints:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("label")
            if not name or name not in POSE17_KEYPOINT_TO_ID:
                continue
            if name in seen_labels:
                continue
            seen_labels.add(name)

            pts = item.get("point") or item.get("points") or [0, 0]
            x, y = float(pts[0]), float(pts[1])
            vis = item.get("visibility", 2)
            occluded = bool(item.get("occluded", vis == 1))
            outside = bool(item.get("outside", vis == 0))

            px, py = denormalize_point(x, y, img_width, img_height, coord_range=coord_range, clamp=True)
            elements.append(KeypointElement(
                label=name,
                points=[px, py],
                occluded=occluded,
                outside=outside,
            ))

    if not elements:
        return None

    return SkeletonInstance(
        label="person",
        elements=elements,
        confidence=float(raw_data.get("confidence", 1.0)),
        group=int(raw_data.get("group", 0)),
    )


def parse_and_sanitize_vf50_instance(
    raw_data: Dict[str, Any],
    img_width: int,
    img_height: int,
    coord_range: float = 1000.0,
) -> Optional[SkeletonInstance]:
    """Parse and sanitize a raw VF50 facial landmark prediction into a SkeletonInstance."""
    raw_landmarks = raw_data.get("landmarks", raw_data.get("keypoints"))
    if not raw_landmarks or not isinstance(raw_landmarks, (dict, list)):
        return None

    elements: List[KeypointElement] = []
    seen_labels: Set[str] = set()

    if isinstance(raw_landmarks, dict):
        for name in VF50_LANDMARKS:
            pt_data = raw_landmarks.get(name)
            if pt_data is None:
                elements.append(KeypointElement(
                    label=name,
                    points=[0.0, 0.0],
                    occluded=False,
                    outside=True,
                ))
                continue

            if name in seen_labels:
                continue
            seen_labels.add(name)

            if isinstance(pt_data, (list, tuple)) and len(pt_data) >= 2:
                x, y = float(pt_data[0]), float(pt_data[1])
                vis = pt_data[2] if len(pt_data) > 2 else 2
                occluded = (vis == 1) if vis in (0, 1, 2) else False
                outside = (vis == 0) if vis in (0, 1, 2) else False

                px, py = denormalize_point(x, y, img_width, img_height, coord_range=coord_range, clamp=True)
                elements.append(KeypointElement(
                    label=name,
                    points=[px, py],
                    occluded=occluded,
                    outside=outside,
                ))

    elif isinstance(raw_landmarks, list):
        for item in raw_landmarks:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("label")
            if not name or name not in VF50_LANDMARK_TO_ID:
                continue
            if name in seen_labels:
                continue
            seen_labels.add(name)

            pts = item.get("point") or item.get("points") or [0, 0]
            x, y = float(pts[0]), float(pts[1])
            vis = item.get("visibility", 2)
            occluded = bool(item.get("occluded", vis == 1))
            outside = bool(item.get("outside", vis == 0))

            px, py = denormalize_point(x, y, img_width, img_height, coord_range=coord_range, clamp=True)
            elements.append(KeypointElement(
                label=name,
                points=[px, py],
                occluded=occluded,
                outside=outside,
            ))

    if not elements:
        return None

    return SkeletonInstance(
        label="face",
        elements=elements,
        confidence=float(raw_data.get("confidence", 1.0)),
        group=int(raw_data.get("group", 0)),
    )

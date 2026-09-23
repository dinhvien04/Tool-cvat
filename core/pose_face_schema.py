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

from core.week2_schema import (
    load_pose17,
    load_vf50,
    pose17_coco_keypoints,
    vf50_all_edges,
    vf50_landmarks,
    vf50_mirror_map,
)

# ==============================================================================
# 1. POSE17 CANONICAL SPECIFICATION (Derived from canonical load_pose17())
# ==============================================================================

_pose17 = load_pose17()
POSE17_KEYPOINTS: Tuple[str, ...] = pose17_coco_keypoints()
POSE17_KEYPOINT_TO_ID: Dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(POSE17_KEYPOINTS)
}
POSE17_ID_TO_KEYPOINT: Dict[int, str] = {
    idx + 1: name for idx, name in enumerate(POSE17_KEYPOINTS)
}
POSE17_EDGES: Tuple[Tuple[int, int], ...] = _pose17.edges
POSE17_LATERALITY = "frame_based_vf"  # R on right of frame, L on left of frame


# ==============================================================================
# 2. VF50 CANONICAL SPECIFICATION (Derived from canonical load_vf50())
# ==============================================================================

_vf50 = load_vf50()
VIN_VF50_COMPONENTS: Dict[str, Tuple[int, int, str]] = {
    c.name: (c.start_id, c.end_id, c.topology) for c in _vf50.components
}
VF50_COMPONENT_COUNTS: Dict[str, int] = {
    c.name: c.point_count for c in _vf50.components
}
VF50_LANDMARKS: Tuple[str, ...] = vf50_landmarks()
assert len(VF50_LANDMARKS) == 50, f"VF50 must have exactly 50 landmarks, got {len(VF50_LANDMARKS)}"

# Point IDs are continuous 0..49
VF50_LANDMARK_TO_ID: Dict[str, int] = {
    name: int(name.split("_")[-1]) for name in VF50_LANDMARKS
}
VF50_ID_TO_LANDMARK: Dict[int, str] = {
    nid: name for name, nid in VF50_LANDMARK_TO_ID.items()
}
VF50_EDGES: Tuple[Tuple[int, int], ...] = vf50_all_edges()
VF50_MIRROR_MAP: Dict[str, str] = vf50_mirror_map()
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

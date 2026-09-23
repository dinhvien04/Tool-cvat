"""Canonical Week-2 Schema Loader — Single Source of Truth.

Reads config/week2_pose17.yaml and config/week2_vf50.yaml at first access,
validates schema integrity, and exposes immutable, typed, cached schema objects.
All downstream modules MUST import schema definitions from here instead of
defining their own redundant constants.

Usage:
    from core.week2_schema import load_pose17, load_vf50

    pose = load_pose17()
    assert len(pose.keypoints) == 17
    assert len(pose.edges) == 18  # YAML authoritative count

    vf = load_vf50()
    assert vf.total_points == 50
    assert len(vf.components) == 7
"""

from __future__ import annotations

import hashlib
import json
import os
import yaml
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

# ─── Path Resolution ───────────────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_POSE17_YAML = _CONFIG_DIR / "week2_pose17.yaml"
_VF50_YAML = _CONFIG_DIR / "week2_vf50.yaml"

# ─── Laterality Constants ──────────────────────────────────────────────────────
LATERALITY_VIEWER = "viewer"
LATERALITY_SUBJECT = "subject"


# ─── Immutable Schema Dataclasses ──────────────────────────────────────────────

@dataclass(frozen=True)
class Keypoint:
    """A single keypoint/landmark definition."""
    id: int              # 1-based for Pose17, 0-based for VF50
    numeric_name: str    # "1", "2", ... or "0", "1", ...
    semantic_name: str   # "Nose", "R Eye", ... or "longmaytrai_00", ...
    side: str            # "center", "right", "left"


@dataclass(frozen=True)
class Pose17Schema:
    """Immutable Pose17 schema loaded from config/week2_pose17.yaml."""
    version: str
    laterality_convention: str   # "viewer"
    parent_label: str            # "person"
    keypoints: Tuple[Keypoint, ...]           # ordered tuple of 17
    edges: Tuple[Tuple[int, int], ...]        # 1-based ID pairs (18 edges)
    id_to_keypoint: Mapping[int, Keypoint]    # 1→Keypoint
    name_to_keypoint: Mapping[str, Keypoint]  # "1"→Keypoint (numeric name)
    semantic_to_id: Mapping[str, int]         # "Nose"→1, "R Eye"→2
    id_to_semantic: Mapping[int, str]         # 1→"Nose"
    keypoint_names: FrozenSet[str]            # {"1","2",...,"17"}

    # COCO-compatible name mappings (lowercase, underscore-joined)
    coco_name_to_id: Mapping[str, int]        # "nose"→1, "right_eye"→2
    id_to_coco_name: Mapping[int, str]        # 1→"nose"

    # Pre-computed convenience
    right_keypoint_ids: FrozenSet[int]        # {2,4,6,8,10,12,14,16}
    left_keypoint_ids: FrozenSet[int]         # {3,5,7,9,11,13,15,17}

    # Named edge pairs (semantic)
    edges_as_coco_names: Tuple[Tuple[str, str], ...]  # ("nose","right_eye"), ...

    # SHA-256 fingerprint of canonical spec
    spec_fingerprint: str


@dataclass(frozen=True)
class VF50Component:
    """A single VF50 facial component definition."""
    name: str                            # "longmaytrai"
    start_id: int                        # 0
    end_id: int                          # 4
    point_count: int                     # 5
    topology: str                        # "open" or "closed"
    keypoints: Tuple[Keypoint, ...]      # ordered points
    edges: Tuple[Tuple[int, int], ...]   # 0-based ID pairs


@dataclass(frozen=True)
class VF50Schema:
    """Immutable VF50 schema loaded from config/week2_vf50.yaml."""
    version: str
    laterality_convention: str
    total_points: int                    # 50
    total_edges: int                     # 47
    components: Tuple[VF50Component, ...] # 7 components in order
    component_names: Tuple[str, ...]     # ("longmaytrai", ..., "moitrong")
    all_edges: Tuple[Tuple[int, int], ...]
    all_keypoints: Tuple[Keypoint, ...]  # 50 keypoints in order
    id_to_keypoint: Mapping[int, Keypoint]
    point_to_component: Mapping[int, str]
    component_by_name: Mapping[str, VF50Component]

    # SHA-256 fingerprint of canonical spec
    spec_fingerprint: str


# ─── Visibility Contract ──────────────────────────────────────────────────────

VISIBILITY_OUTSIDE = 0    # Not labeled / outside image frame
VISIBILITY_OCCLUDED = 1   # Present but occluded
VISIBILITY_VISIBLE = 2    # Clearly visible

def map_visibility_to_cvat(vis: int) -> Tuple[bool, bool]:
    """Map visibility flag to CVAT (outside, occluded) booleans.

    Returns:
        (outside, occluded) tuple.
    """
    if vis == 0:
        return (True, False)
    elif vis == 1:
        return (False, True)
    else:  # vis == 2
        return (False, False)


def map_cvat_to_visibility(outside: bool, occluded: bool) -> int:
    """Map CVAT (outside, occluded) booleans to visibility flag."""
    if outside:
        return 0
    elif occluded:
        return 1
    else:
        return 2


# ─── Loader Functions (cached, validated) ──────────────────────────────────────

def _semantic_to_coco_name(semantic: str) -> str:
    """Convert VinFast semantic name to COCO-style lowercase name.

    "Nose" → "nose", "R Eye" → "right_eye", "L Shoulder" → "left_shoulder"
    """
    s = semantic.strip()
    s = s.replace("R ", "right_").replace("L ", "left_")
    return s.lower().replace(" ", "_")


def _compute_fingerprint(data: object) -> str:
    """Compute SHA-256 fingerprint over deterministically-serialized JSON."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_pose17() -> Pose17Schema:
    """Load, validate, and cache Pose17 schema from YAML.

    Raises:
        FileNotFoundError: If config/week2_pose17.yaml is missing.
        AssertionError: If schema validation fails.
    """
    with open(_POSE17_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Validate structure
    sublabels = raw["sublabels"]
    expected = raw["expected_point_count"]
    assert len(sublabels) == expected == 17, (
        f"Expected 17 sublabels, got {len(sublabels)}"
    )

    keypoints: List[Keypoint] = []
    id_to_kp: Dict[int, Keypoint] = {}
    name_to_kp: Dict[str, Keypoint] = {}
    semantic_to_id: Dict[str, int] = {}
    id_to_semantic: Dict[int, str] = {}
    coco_name_to_id: Dict[str, int] = {}
    id_to_coco_name: Dict[int, str] = {}
    right_ids: set = set()
    left_ids: set = set()

    for sl in sublabels:
        kp = Keypoint(
            id=sl["id"],
            numeric_name=str(sl["name"]),
            semantic_name=sl["semantic_name"],
            side=sl.get("side", "center"),
        )
        keypoints.append(kp)
        id_to_kp[kp.id] = kp
        name_to_kp[kp.numeric_name] = kp
        semantic_to_id[kp.semantic_name] = kp.id
        id_to_semantic[kp.id] = kp.semantic_name

        coco_name = _semantic_to_coco_name(kp.semantic_name)
        coco_name_to_id[coco_name] = kp.id
        id_to_coco_name[kp.id] = coco_name

        if kp.side == "right":
            right_ids.add(kp.id)
        elif kp.side == "left":
            left_ids.add(kp.id)

    edges = tuple((e["from"], e["to"]) for e in raw["edges"])

    # Validate all edge endpoints reference valid IDs
    valid_ids = {kp.id for kp in keypoints}
    for u, v in edges:
        assert u in valid_ids and v in valid_ids, (
            f"Edge ({u},{v}) references invalid ID. Valid: {valid_ids}"
        )

    # Build named edge pairs
    edges_as_coco = tuple(
        (id_to_coco_name[u], id_to_coco_name[v]) for u, v in edges
    )

    # Compute fingerprint from canonical spec data
    spec_data = {
        "keypoints": [{"id": kp.id, "name": kp.numeric_name, "semantic": kp.semantic_name, "side": kp.side} for kp in keypoints],
        "edges": [list(e) for e in edges],
        "parent_label": raw["parent_labels"][0]["name"],
        "laterality": raw["laterality_convention"],
    }
    fingerprint = _compute_fingerprint(spec_data)

    return Pose17Schema(
        version=raw["schema_version"],
        laterality_convention=raw["laterality_convention"],
        parent_label=raw["parent_labels"][0]["name"],
        keypoints=tuple(keypoints),
        edges=edges,
        id_to_keypoint=id_to_kp,
        name_to_keypoint=name_to_kp,
        semantic_to_id=semantic_to_id,
        id_to_semantic=id_to_semantic,
        keypoint_names=frozenset(kp.numeric_name for kp in keypoints),
        coco_name_to_id=coco_name_to_id,
        id_to_coco_name=id_to_coco_name,
        right_keypoint_ids=frozenset(right_ids),
        left_keypoint_ids=frozenset(left_ids),
        edges_as_coco_names=edges_as_coco,
        spec_fingerprint=fingerprint,
    )


@lru_cache(maxsize=1)
def load_vf50() -> VF50Schema:
    """Load, validate, and cache VF50 schema from YAML.

    Raises:
        FileNotFoundError: If config/week2_vf50.yaml is missing.
        AssertionError: If schema validation fails.
    """
    with open(_VF50_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    assert raw["expected_point_count"] == 50, (
        f"Expected 50 points, got {raw['expected_point_count']}"
    )

    components: List[VF50Component] = []
    all_keypoints: List[Keypoint] = []
    all_edges: List[Tuple[int, int]] = []
    id_to_kp: Dict[int, Keypoint] = {}
    point_to_comp: Dict[int, str] = {}
    comp_by_name: Dict[str, VF50Component] = {}

    for comp_raw in raw["components"]:
        comp_kps: List[Keypoint] = []
        for sl in comp_raw["sublabels"]:
            kp = Keypoint(
                id=sl["id"],
                numeric_name=str(sl["name"]),
                semantic_name=sl["semantic_name"],
                side="center",
            )
            comp_kps.append(kp)
            all_keypoints.append(kp)
            id_to_kp[kp.id] = kp
            point_to_comp[kp.id] = comp_raw["name"]

        comp_edges = tuple(tuple(e) for e in comp_raw["edges"])
        all_edges.extend(comp_edges)

        pr = comp_raw["point_range"]
        comp = VF50Component(
            name=comp_raw["name"],
            start_id=pr[0],
            end_id=pr[1],
            point_count=comp_raw["point_count"],
            topology=comp_raw["topology"],
            keypoints=tuple(comp_kps),
            edges=comp_edges,
        )
        components.append(comp)
        comp_by_name[comp.name] = comp

    assert len(all_keypoints) == 50, f"Expected 50 keypoints, got {len(all_keypoints)}"
    assert len(all_edges) == 47, f"Expected 47 edges, got {len(all_edges)}"

    # Compute fingerprint
    spec_data = {
        "components": [
            {
                "name": c.name,
                "start_id": c.start_id,
                "end_id": c.end_id,
                "point_count": c.point_count,
                "topology": c.topology,
                "edges": [list(e) for e in c.edges],
            }
            for c in components
        ],
        "total_points": 50,
        "total_edges": 47,
        "laterality": raw["laterality_convention"],
    }
    fingerprint = _compute_fingerprint(spec_data)

    comp_names = tuple(c.name for c in components)

    return VF50Schema(
        version=raw["schema_version"],
        laterality_convention=raw["laterality_convention"],
        total_points=50,
        total_edges=47,
        components=tuple(components),
        component_names=comp_names,
        all_edges=tuple(all_edges),
        all_keypoints=tuple(all_keypoints),
        id_to_keypoint=id_to_kp,
        point_to_component=point_to_comp,
        component_by_name=comp_by_name,
        spec_fingerprint=fingerprint,
    )


# ─── Convenience Accessors ────────────────────────────────────────────────────

def pose17_coco_keypoints() -> Tuple[str, ...]:
    """Return Pose17 keypoint names in COCO-compatible lowercase format, VinFast order.

    Returns:
        ('nose', 'right_eye', 'left_eye', 'right_ear', ..., 'left_ankle')
    """
    schema = load_pose17()
    return tuple(schema.id_to_coco_name[kp.id] for kp in schema.keypoints)


def pose17_edges_as_coco_names() -> Tuple[Tuple[str, str], ...]:
    """Return Pose17 edges as tuples of COCO-compatible lowercase name pairs."""
    return load_pose17().edges_as_coco_names


def pose17_edges_0indexed() -> Tuple[Tuple[int, int], ...]:
    """Return Pose17 edges as 0-indexed pairs (for COCO convention compatibility).

    The YAML uses 1-indexed IDs. This converts (1,2) to (0,1), etc.
    """
    return tuple((u - 1, v - 1) for u, v in load_pose17().edges)


def vf50_component_names() -> Tuple[str, ...]:
    """Return the 7 canonical VF50 component names in order."""
    return load_vf50().component_names


def vf50_component_point_ranges() -> Dict[str, Tuple[int, int]]:
    """Return {component_name: (start_id, end_id)} for all 7 VF50 components."""
    return {c.name: (c.start_id, c.end_id) for c in load_vf50().components}


def vf50_component_point_counts() -> Dict[str, int]:
    """Return {component_name: point_count} for all 7 VF50 components."""
    return {c.name: c.point_count for c in load_vf50().components}


def compute_spec_fingerprint(spec_items: object) -> str:
    """Compute SHA-256 fingerprint over any JSON-serializable spec data.

    This normalizes the data via sorted keys and compact separators,
    ensuring deterministic fingerprints regardless of key ordering or
    formatting in the source.

    Args:
        spec_items: Any JSON-serializable Python object (list, dict, etc.)

    Returns:
        64-character hex SHA-256 digest.
    """
    return _compute_fingerprint(spec_items)


# ─── Build SHA ────────────────────────────────────────────────────────────────

def get_build_sha() -> Optional[str]:
    """Return the TOOL_CVAT_BUILD_SHA from environment, if set."""
    return os.getenv("TOOL_CVAT_BUILD_SHA")

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
import math
import os
import yaml
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

# ─── Path Resolution ───────────────────────────────────────────────────────────
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_POSE17_YAML = _CONFIG_DIR / "week2_pose17.yaml"
_VF50_YAML = _CONFIG_DIR / "week2_vf50.yaml"
_POSE17_PATH = _POSE17_YAML
_VF50_PATH = _VF50_YAML

# ─── Laterality Constants ──────────────────────────────────────────────────────
LATERALITY_VIEWER = "viewer"
LATERALITY_SUBJECT = "subject"


class SchemaValidationError(ValueError):
    """Raised when a canonical Week-2 schema fails structural or semantic validation."""


# ─── Immutable Schema Dataclasses ──────────────────────────────────────────────

@dataclass(frozen=True)
class Keypoint:
    """A single keypoint/landmark definition."""
    id: int              # 1-based for Pose17, 0-based for VF50
    numeric_name: str    # "1", "2", ... or "0", "1", ...
    semantic_name: str   # "Nose", "R Eye", ... or "longmaytrai_00", ...
    side: str            # "center", "right", "left"
    anatomical: Optional[str] = None


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
    description: str = ""                # component description from schema


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


def normalize_visibility(
    vis: Union[int, float, str, bool, None],
    strict: bool = False,
    default: int = VISIBILITY_VISIBLE,
) -> int:
    """Normalize input visibility value to canonical integer (0, 1, or 2).

    Canonical Contract:
      0 = outside image frame / untracked (CVAT: outside=True, occluded=False)
      1 = present but physically occluded (CVAT: outside=False, occluded=True)
      2 = clearly visible in direct line of sight (CVAT: outside=False, occluded=False)

    Accepted explicit inputs (strict=True):
      - 0, 0.0, "0", "outside" -> 0 (VISIBILITY_OUTSIDE)
      - 1, 1.0, "1", "occluded" -> 1 (VISIBILITY_OCCLUDED)
      - 2, 2.0, "2", "visible"  -> 2 (VISIBILITY_VISIBLE)

    Non-strict mode (strict=False) safe fallbacks:
      - None defaults to default (default: VISIBILITY_VISIBLE = 2)
      - Booleans: True -> 2 (visible), False -> 0 (outside)
      - Legacy string aliases: 'absent', 'false' -> 0; 'partial' -> 1; 'true' -> 2
      - Unrecognized values, non-integer floats, out-of-range numbers, and invalid strings return default.

    Strict mode (strict=True) rejections (raise ValueError):
      - Negative numbers (-1, -1.0, -5, etc.)
      - Numbers > 2 (3, 3.0, 100, etc.)
      - Non-integer floats (0.1, 0.4, 0.49, 0.5, 0.6, 0.999, 1.01, 1.5, 1.99, 2.001, etc.)
      - Non-finite floats (NaN, Inf, -Inf)
      - Ambiguous / ungrounded strings ('partial', 'uncertain', 'banana', '0.0', etc.)
      - Booleans and None

    Args:
        vis: Input visibility value.
        strict: If True, raise ValueError on any unrecognised or ambiguous input.
        default: Fallback value when strict=False (default: VISIBILITY_VISIBLE = 2).

    Returns:
        Exact integer 0, 1, or 2.

    Raises:
        ValueError: If vis is invalid and strict=True.
    """
    if vis is None:
        if strict:
            raise ValueError("Visibility value cannot be None in strict mode")
        return default

    if isinstance(vis, bool):
        if strict:
            raise ValueError(
                f"Boolean {vis} is not a valid strict visibility flag. "
                f"Expected 0, 1, 2, 'outside', 'occluded', or 'visible'."
            )
        return VISIBILITY_VISIBLE if vis else VISIBILITY_OUTSIDE

    if isinstance(vis, (int, float)):
        if isinstance(vis, float) and (math.isnan(vis) or math.isinf(vis)):
            if strict:
                raise ValueError(f"Invalid non-finite visibility float: {vis}")
            return default
        try:
            f = float(vis)
            if not f.is_integer():
                if strict:
                    raise ValueError(
                        f"Non-integer numeric visibility {vis} is not allowed in strict mode. "
                        f"Expected exact integer 0, 1, or 2."
                    )
                return default
            iv = int(f)
        except (ValueError, TypeError, OverflowError) as e:
            if strict:
                raise ValueError(f"Cannot parse visibility numeric: {vis}") from e
            return default
        if iv in (VISIBILITY_OUTSIDE, VISIBILITY_OCCLUDED, VISIBILITY_VISIBLE):
            return iv
        if strict:
            raise ValueError(f"Invalid numeric visibility: {vis} (must be 0, 1, or 2)")
        return default

    if isinstance(vis, str):
        v = vis.strip().lower()
        if v in ("0", "outside"):
            return VISIBILITY_OUTSIDE
        elif v in ("1", "occluded"):
            return VISIBILITY_OCCLUDED
        elif v in ("2", "visible"):
            return VISIBILITY_VISIBLE
        elif not strict:
            if v in ("absent", "false"):
                return VISIBILITY_OUTSIDE
            elif v in ("partial",):
                return VISIBILITY_OCCLUDED
            elif v in ("true",):
                return VISIBILITY_VISIBLE
            return default
        else:
            raise ValueError(
                f"Invalid visibility string: {vis!r}. "
                f"Allowed: '0', '1', '2', 'outside', 'occluded', 'visible'."
            )

    if strict:
        raise ValueError(f"Unsupported visibility type: {type(vis).__name__} ({vis!r})")
    return default


def map_visibility_to_cvat(
    vis: Union[int, float, str, bool, None],
    strict: bool = False,
    default: int = VISIBILITY_VISIBLE,
) -> Tuple[bool, bool]:
    """Map visibility flag to CVAT (outside, occluded) booleans.

    Args:
        vis: Numeric (0, 1, 2), string ("outside", "occluded", "visible"), bool, or None.
        strict: If True, invalid values raise ValueError. If False, invalid values
                that cannot be normalized default safely to default.
        default: Fallback integer (0, 1, or 2) when strict=False.

    Returns:
        (outside, occluded) tuple.
    """
    norm_v = normalize_visibility(vis, strict=strict, default=default)
    if norm_v == VISIBILITY_OUTSIDE:
        return (True, False)
    elif norm_v == VISIBILITY_OCCLUDED:
        return (False, True)
    else:
        return (False, False)


def map_cvat_to_visibility(outside: bool, occluded: bool, strict: bool = False) -> int:
    """Map CVAT (outside, occluded) booleans to visibility flag.

    Contract:
        - outside=True,  occluded=False -> 0 (VISIBILITY_OUTSIDE)
        - outside=False, occluded=True  -> 1 (VISIBILITY_OCCLUDED)
        - outside=False, occluded=False -> 2 (VISIBILITY_VISIBLE)

    Forbidden Invariant:
        - outside=True,  occluded=True  -> Invalid state.
          If strict=True: raises ValueError.
          If strict=False: outside takes precedence, returns 0.

    Args:
        outside: CVAT outside attribute boolean.
        occluded: CVAT occluded attribute boolean.
        strict: If True, raises ValueError if outside and occluded are both True.

    Returns:
        0 (outside), 1 (occluded), or 2 (visible).
    """
    b_out = bool(outside)
    b_occ = bool(occluded)
    if b_out and b_occ:
        if strict:
            raise ValueError(
                "Invalid CVAT state: a keypoint cannot be simultaneously outside and occluded."
            )
        return VISIBILITY_OUTSIDE
    if b_out:
        return VISIBILITY_OUTSIDE
    elif b_occ:
        return VISIBILITY_OCCLUDED
    else:
        return VISIBILITY_VISIBLE


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
        SchemaValidationError: If config/week2_pose17.yaml is missing or schema validation fails.
    """
    target_path = Path(_POSE17_YAML)
    if not target_path.exists():
        raise SchemaValidationError(f"Pose17 YAML file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise SchemaValidationError("Pose17 YAML content must be a dictionary")

    if "schema_version" not in raw:
        raise SchemaValidationError("Pose17 schema missing required 'schema_version'")

    if raw.get("laterality_convention") != LATERALITY_VIEWER:
        raise SchemaValidationError(
            f"Pose17 schema specifies unknown laterality: {raw.get('laterality_convention')!r}"
        )

    parent_labels = raw.get("parent_labels")
    if not isinstance(parent_labels, list) or not parent_labels or parent_labels[0].get("name") != "person":
        raise SchemaValidationError("Pose17 schema parent label must be 'person'")

    # Validate structure
    sublabels = raw.get("sublabels", raw.get("keypoints", []))
    expected = raw.get("expected_point_count", len(sublabels))
    if len(sublabels) != 17 or expected != 17:
        raise SchemaValidationError(
            f"Expected exactly 17 sublabels, got {len(sublabels)} (expected_point_count={expected})"
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

    seen_ids = set()
    seen_names = set()
    seen_semantics = set()

    for sl in sublabels:
        kp_id = sl.get("id")
        if not isinstance(kp_id, int):
            raise SchemaValidationError(f"Pose17 keypoint ID must be an integer, got {kp_id!r}")
        if kp_id in seen_ids:
            raise SchemaValidationError(f"Duplicate keypoint ID {kp_id} in Pose17 schema")
        seen_ids.add(kp_id)

        numeric_name = str(sl.get("name", ""))
        if numeric_name in seen_names:
            raise SchemaValidationError(f"Duplicate numeric name {numeric_name!r} in Pose17 schema")
        seen_names.add(numeric_name)

        semantic_name = sl.get("semantic_name", "")
        if semantic_name in seen_semantics:
            raise SchemaValidationError(f"Duplicate semantic name {semantic_name!r} in Pose17 schema")
        seen_semantics.add(semantic_name)

        side = sl.get("side", "center")
        if side not in ("center", "right", "left"):
            raise SchemaValidationError(f"Invalid side {side!r} for keypoint {kp_id}")

        kp = Keypoint(
            id=kp_id,
            numeric_name=numeric_name,
            semantic_name=semantic_name,
            side=side,
            anatomical=sl.get("anatomical"),
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

    if seen_ids != set(range(1, 18)):
        raise SchemaValidationError(f"Pose17 keypoint IDs must be contiguous 1..17, got {sorted(seen_ids)}")

    raw_edges = raw.get("edges", [])
    if len(raw_edges) != 18:
        raise SchemaValidationError(f"Expected exactly 18 edges for Pose17, got {len(raw_edges)}")

    edges = tuple((e["from"], e["to"]) for e in raw_edges)

    # Validate all edge endpoints reference valid IDs
    valid_ids = {kp.id for kp in keypoints}
    seen_edges = set()
    for u, v in edges:
        if u not in valid_ids or v not in valid_ids:
            raise SchemaValidationError(
                f"Edge ({u},{v}) references invalid ID. Valid: {valid_ids}"
            )
        if u == v:
            raise SchemaValidationError(f"Self-loop edge ({u},{v}) is forbidden")
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
            raise SchemaValidationError(f"Duplicate edge ({u},{v}) in Pose17 schema")
        seen_edges.add(edge_key)

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
        SchemaValidationError: If config/week2_vf50.yaml is missing or schema validation fails.
    """
    target_path = Path(_VF50_YAML)
    if not target_path.exists():
        raise SchemaValidationError(f"VF50 YAML file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise SchemaValidationError("VF50 YAML content must be a dictionary")

    if raw.get("expected_point_count") != 50:
        raise SchemaValidationError(
            f"Expected 50 points, got {raw.get('expected_point_count')}"
        )

    comps_raw = raw.get("components", [])
    if len(comps_raw) != 7:
        raise SchemaValidationError(f"Expected exactly 7 components in VF50 schema, got {len(comps_raw)}")

    expected_comp_names = (
        "longmaytrai",
        "longmayphai",
        "songmui",
        "mattrai",
        "matphai",
        "moingoai",
        "moitrong",
    )
    actual_names = tuple(c.get("name") for c in comps_raw)
    if actual_names != expected_comp_names:
        raise SchemaValidationError(
            f"VF50 component names or order mismatch: expected {expected_comp_names}, got {actual_names}"
        )

    components: List[VF50Component] = []
    all_keypoints: List[Keypoint] = []
    all_edges: List[Tuple[int, int]] = []
    id_to_kp: Dict[int, Keypoint] = {}
    point_to_comp: Dict[int, str] = {}
    comp_by_name: Dict[str, VF50Component] = {}
    seen_pt_ids = set()

    for comp_raw in comps_raw:
        c_name = comp_raw.get("name")
        pr = comp_raw.get("point_range")
        if not pr or len(pr) != 2:
            raise SchemaValidationError(f"Component {c_name} missing valid point_range [start, end]")
        start_id, end_id = pr[0], pr[1]
        c_count = comp_raw.get("point_count")
        if end_id - start_id + 1 != c_count:
            raise SchemaValidationError(
                f"Component {c_name} point_count {c_count} does not match point_range [{start_id}, {end_id}]"
            )

        sublabels = comp_raw.get("sublabels", [])
        if len(sublabels) != c_count:
            raise SchemaValidationError(
                f"Component {c_name} sublabels length {len(sublabels)} does not match point_count {c_count}"
            )

        comp_kps: List[Keypoint] = []
        for sl in sublabels:
            pt_id = sl.get("id")
            if not isinstance(pt_id, int) or not (start_id <= pt_id <= end_id):
                raise SchemaValidationError(
                    f"Sublabel ID {pt_id} outside component {c_name} range [{start_id}, {end_id}]"
                )
            if pt_id in seen_pt_ids:
                raise SchemaValidationError(f"Duplicate keypoint ID {pt_id} in VF50 schema")
            seen_pt_ids.add(pt_id)

            kp = Keypoint(
                id=pt_id,
                numeric_name=str(sl["name"]),
                semantic_name=sl["semantic_name"],
                side="center",
                anatomical=sl.get("anatomical"),
            )
            comp_kps.append(kp)
            all_keypoints.append(kp)
            id_to_kp[kp.id] = kp
            point_to_comp[kp.id] = c_name

        raw_comp_edges = comp_raw.get("edges", [])
        c_seen_edges = set()
        comp_edges_list = []
        for e in raw_comp_edges:
            if len(e) != 2:
                raise SchemaValidationError(f"Component {c_name} has invalid edge tuple: {e}")
            u, v = e[0], e[1]
            if not (start_id <= u <= end_id and start_id <= v <= end_id):
                raise SchemaValidationError(
                    f"Component {c_name} edge ({u}, {v}) endpoint outside range [{start_id}, {end_id}]"
                )
            if u == v:
                raise SchemaValidationError(f"Self-loop edge ({u}, {v}) forbidden in component {c_name}")
            edge_key = (min(u, v), max(u, v))
            if edge_key in c_seen_edges:
                raise SchemaValidationError(f"Duplicate edge ({u}, {v}) in component {c_name}")
            c_seen_edges.add(edge_key)
            comp_edges_list.append((u, v))

        comp_edges = tuple(comp_edges_list)
        all_edges.extend(comp_edges)

        comp = VF50Component(
            name=c_name,
            start_id=start_id,
            end_id=end_id,
            point_count=c_count,
            topology=comp_raw["topology"],
            keypoints=tuple(comp_kps),
            edges=comp_edges,
            description=comp_raw.get("description", ""),
        )
        components.append(comp)
        comp_by_name[comp.name] = comp

    if len(all_keypoints) != 50 or seen_pt_ids != set(range(50)):
        raise SchemaValidationError(f"Expected 50 contiguous keypoints (0..49), got {len(all_keypoints)}")
    if len(all_edges) != 47:
        raise SchemaValidationError(f"Expected exactly 47 edges across VF50 components, got {len(all_edges)}")

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


def pose17_semantic_keypoints() -> Tuple[str, ...]:
    """Return Pose17 keypoint names in canonical VinFast 1..17 order.

    Canonical accessor for VinFast Week-2 HumanPose-17 viewer-space keypoints.
    Equivalent to pose17_coco_keypoints(), retained as the authoritative alias.
    """
    return pose17_coco_keypoints()


def pose17_laterality_convention() -> str:
    """Return canonical laterality convention for Pose17 ('viewer') from authoritative YAML."""
    return load_pose17().laterality_convention


def pose17_keypoint_descriptions() -> Dict[str, str]:
    """Return mapping of COCO keypoint name to its canonical schema metadata description.

    Example:
        {'nose': 'Nose (Chop mui) [center]', 'right_eye': 'R Eye (Tam dong tu mat phia PHAI...) [right]', ...}
    """
    schema = load_pose17()
    res: Dict[str, str] = {}
    for kp in schema.keypoints:
        coco_name = schema.id_to_coco_name[kp.id]
        desc = kp.semantic_name
        if kp.anatomical:
            desc += f" ({kp.anatomical})"
        desc += f" [{kp.side}]"
        res[coco_name] = desc
    return res


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


def pose17_paired_keypoints() -> Tuple[Tuple[str, str], ...]:
    """Return symmetric (left, right) pairs for Pose17 laterality validation."""
    return (
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    )


def pose17_rigid_paired_keypoints() -> Tuple[Tuple[str, str], ...]:
    """Return rigid axial/head (left, right) pairs for Pose17 laterality validation."""
    return (
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    )


def pose17_articulated_paired_keypoints() -> Tuple[Tuple[str, str], ...]:
    """Return articulated appendicular limb (left, right) pairs for Pose17."""
    return (
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    )


def vf50_edges_by_component() -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Return {component_name: edges_tuple} for all 7 VF50 components."""
    return {c.name: c.edges for c in load_vf50().components}


def vf50_all_edges() -> Tuple[Tuple[int, int], ...]:
    """Return all 47 canonical VF50 edges as 0-based integer pairs."""
    return load_vf50().all_edges


def vf50_point_to_component() -> Dict[int, str]:
    """Return mapping from point ID (0..49) to its component name."""
    return dict(load_vf50().point_to_component)


def vf50_component_configs() -> Dict[str, Dict[str, Any]]:
    """Return component configurations matching VF50_COMPONENT_CONFIG."""
    from typing import Any
    return {
        c.name: {
            "start": c.start_id,
            "end": c.end_id,
            "count": c.point_count,
            "topology": c.topology,
            "desc": c.name,
        }
        for c in load_vf50().components
    }


def vf50_landmarks() -> Tuple[str, ...]:
    """Return canonical VF50 landmark names formatted as '<component>_<id:02d>'."""
    vf = load_vf50()
    return tuple(f"{vf.point_to_component[kp.id]}_{kp.id:02d}" for kp in vf.all_keypoints)


def vf50_eyelid_opposing_pairs() -> Tuple[Tuple[int, int], ...]:
    """Return upper/lower opposing eyelid landmark ID pairs for vertical inversion checks.

    mattrai (left eye, viewer space):
      upper (15, 16, 17) opposing lower (21, 20, 19):
      (15, 21), (16, 20), (17, 19)
    matphai (right eye, viewer space):
      upper (23, 24, 25) opposing lower (29, 28, 27):
      (23, 29), (24, 28), (25, 27)
    """
    return (
        (15, 21), (16, 20), (17, 19),
        (23, 29), (24, 28), (25, 27),
    )


def vf50_mirror_map() -> Dict[str, str]:
    """Return horizontal reflection mirror map for VF50 landmark strings."""
    mirror: Dict[str, str] = {}
    for i in range(5):
        l_name = f"longmaytrai_{i:02d}"
        r_name = f"longmayphai_{9 - i:02d}"
        mirror[l_name] = r_name
        mirror[r_name] = l_name

    for i in range(10, 14):
        name = f"songmui_{i:02d}"
        mirror[name] = name

    eye_l_to_r = {
        14: 26, 15: 25, 16: 24, 17: 23,
        18: 22, 19: 29, 20: 28, 21: 27,
    }
    for l_id, r_id in eye_l_to_r.items():
        l_name = f"mattrai_{l_id:02d}"
        r_name = f"matphai_{r_id:02d}"
        mirror[l_name] = r_name
        mirror[r_name] = l_name

    lip_outer_mirror = {
        30: 36, 31: 35, 32: 34, 33: 33, 34: 32, 35: 31, 36: 30,
        37: 41, 38: 40, 39: 39, 40: 38, 41: 37,
    }
    for a, b in lip_outer_mirror.items():
        mirror[f"moingoai_{a:02d}"] = f"moingoai_{b:02d}"

    lip_inner_mirror = {
        42: 46, 43: 45, 44: 44, 45: 43, 46: 42,
        47: 49, 48: 48, 49: 47,
    }
    for a, b in lip_inner_mirror.items():
        mirror[f"moitrong_{a:02d}"] = f"moitrong_{b:02d}"

    return mirror


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


# ─── Authoritative Schema & Spec Hashes ───────────────────────────────────────

# 1. Semantic Schema Hashes (derived from YAML contracts)
POSE_SCHEMA_HASH: str = load_pose17().spec_fingerprint
VF50_SCHEMA_HASH: str = load_vf50().spec_fingerprint


def get_canonical_cvat_spec(key: str) -> List[Dict[str, Any]]:
    """Return authoritative CVAT spec dictionary list for detector key.

    Derives directly from canonical load_pose17() and load_vf50() schemas.
    """
    from core.skeleton_contract import build_cvat_pose17_spec, build_cvat_vf50_spec
    if key == "pose17":
        return [build_cvat_pose17_spec(parent_label=load_pose17().parent_label)]
    elif key == "vf50":
        return build_cvat_vf50_spec()
    raise ValueError(f"Unknown detector key: {key}")


# 2. Canonical CVAT Spec Hashes (derived from canonical CVAT annotation spec dictionary list, computed lazily to break import cycles)
_POSE_CVAT_SPEC_HASH: Optional[str] = None
_VF50_CVAT_SPEC_HASH: Optional[str] = None


def get_canonical_schema_hash(key: str) -> str:
    """Return authoritative YAML schema SHA-256 fingerprint for detector key."""
    if key == "pose17":
        return POSE_SCHEMA_HASH
    elif key == "vf50":
        return VF50_SCHEMA_HASH
    raise ValueError(f"Unknown detector key: {key}")


def get_canonical_cvat_spec_hash(key: str) -> str:
    """Return authoritative CVAT spec SHA-256 fingerprint for detector key."""
    global _POSE_CVAT_SPEC_HASH, _VF50_CVAT_SPEC_HASH
    if key == "pose17":
        if _POSE_CVAT_SPEC_HASH is None:
            _POSE_CVAT_SPEC_HASH = compute_spec_fingerprint(get_canonical_cvat_spec("pose17"))
        return _POSE_CVAT_SPEC_HASH
    elif key == "vf50":
        if _VF50_CVAT_SPEC_HASH is None:
            _VF50_CVAT_SPEC_HASH = compute_spec_fingerprint(get_canonical_cvat_spec("vf50"))
        return _VF50_CVAT_SPEC_HASH
    raise ValueError(f"Unknown detector key: {key}")


def __getattr__(name: str) -> Any:
    if name == "POSE_CVAT_SPEC_HASH":
        return get_canonical_cvat_spec_hash("pose17")
    elif name == "VF50_CVAT_SPEC_HASH":
        return get_canonical_cvat_spec_hash("vf50")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ─── Build SHA ────────────────────────────────────────────────────────────────

def get_build_sha() -> Optional[str]:
    """Return the TOOL_CVAT_BUILD_SHA from environment, if set."""
    return os.getenv("TOOL_CVAT_BUILD_SHA")

"""Master 31-Label Taxonomy, Geometry Routing, and Semantics Module.

Phase 3B Architectural Specification:
- Validates the exact 31-label master schema partitioned into 3 strict shape policies:
    1. POLICY_BOX_MASK ("box_mask"): 14 countable foreground instances. Allowed shapes: rectangle, mask.
    2. POLICY_POLYGON_MASK ("polygon_mask"): 10 background semantic regions. Allowed shapes: polygon, mask.
    3. POLICY_POLYLINE ("polyline"): 7 lane markings and road geometry. Allowed shape: polyline only.
- Enforces strict rules:
    - Every label belongs to exactly one policy.
    - Polygon shape is strictly removed/forbidden from instance / box_mask label permissions.
    - No label renaming, no merging of ambiguous labels ('pedestrian' != 'person', etc.).
- Provides O(1) lookups for policies, groups, shapes, ambiguity resolution, and routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

# Canonical paths to configuration files
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
LABEL_GEOMETRY_YAML = CONFIG_DIR / "label_geometry.yaml"
LABEL_SEMANTICS_YAML = CONFIG_DIR / "label_semantics.yaml"

# The EXACT 31 master labels in official indexed order
MASTER_31_LABELS: Tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
    "area/alternative",
    "area/drivable",
    "lane/crosswalk",
    "lane/double white",
    "lane/double yellow",
    "lane/road curb",
    "lane/single other",
    "lane/single white",
    "lane/single yellow",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "traffic_light",
    "traffic_sign",
)

EXPECTED_LABEL_COUNT: int = 31

# Strict Shape Policies
POLICY_BOX_MASK: str = "box_mask"
POLICY_POLYGON_MASK: str = "polygon_mask"
POLICY_POLYLINE: str = "polyline"
VALID_POLICIES: FrozenSet[str] = frozenset(
    {POLICY_BOX_MASK, POLICY_POLYGON_MASK, POLICY_POLYLINE}
)

# Canonical 31-Label Partitions across the 3 Strict Policies
BOX_MASK_LABELS: Tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
    "pole",
    "person",
    "traffic_light",
    "traffic_sign",
)

POLYGON_MASK_LABELS: Tuple[str, ...] = (
    "area/alternative",
    "area/drivable",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "vegetation",
    "terrain",
    "sky",
)

POLYLINE_LABELS: Tuple[str, ...] = (
    "lane/crosswalk",
    "lane/double white",
    "lane/double yellow",
    "lane/road curb",
    "lane/single other",
    "lane/single white",
    "lane/single yellow",
)

# Recognized legacy taxonomy groups (backward compatibility)
GROUP_INSTANCE: str = "instance"
GROUP_REGION: str = "region"
GROUP_LANE: str = "lane"
VALID_GROUPS: FrozenSet[str] = frozenset({GROUP_INSTANCE, GROUP_REGION, GROUP_LANE})

# Group <-> Policy Mappings
GROUP_TO_POLICY: Dict[str, str] = {
    GROUP_INSTANCE: POLICY_BOX_MASK,
    GROUP_REGION: POLICY_POLYGON_MASK,
    GROUP_LANE: POLICY_POLYLINE,
}

POLICY_TO_GROUP: Dict[str, str] = {
    POLICY_BOX_MASK: GROUP_INSTANCE,
    POLICY_POLYGON_MASK: GROUP_REGION,
    POLICY_POLYLINE: GROUP_LANE,
}

# Direct Label -> Policy Mapping
LABEL_TO_POLICY: Dict[str, str] = {
    **{lbl: POLICY_BOX_MASK for lbl in BOX_MASK_LABELS},
    **{lbl: POLICY_POLYGON_MASK for lbl in POLYGON_MASK_LABELS},
    **{lbl: POLICY_POLYLINE for lbl in POLYLINE_LABELS},
}

# Direct Label -> Group Mapping
LABEL_TO_GROUP: Dict[str, str] = {
    lbl: POLICY_TO_GROUP[pol] for lbl, pol in LABEL_TO_POLICY.items()
}

# Recognized CVAT shape types
SHAPE_RECTANGLE = "rectangle"
SHAPE_MASK = "mask"
SHAPE_POLYGON = "polygon"
SHAPE_POLYLINE = "polyline"
VALID_SHAPES: FrozenSet[str] = frozenset(
    {SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON, SHAPE_POLYLINE}
)

# Ambiguity pairs requiring strict isolation and explicit fallback mapping
AMBIGUITY_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    (
        "pedestrian",
        "person",
        "Pedestrian (traffic on foot / BDD100K) vs Person (generic human / COCO).",
    ),
    (
        "traffic light",
        "traffic_light",
        "Space-separated (BDD100K/Cityscapes) vs underscore-separated (COCO/YOLO) syntax.",
    ),
    (
        "traffic sign",
        "traffic_sign",
        "Space-separated (human annotation) vs underscore-separated (programmatic) syntax.",
    ),
)


class TaxonomyValidationError(ValueError):
    """Raised when taxonomy schema or label validation fails."""
    pass


@dataclass(frozen=True)
class LabelMetadata:
    """Immutable metadata and geometric rules for a single CVAT label."""
    name: str
    group: str
    allowed_shapes: Tuple[str, ...]
    preferred_shape: str
    policy: str = ""
    special_handling: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.policy:
            object.__setattr__(
                self,
                "policy",
                LABEL_TO_POLICY.get(self.name, GROUP_TO_POLICY.get(self.group, POLICY_BOX_MASK)),
            )

    @property
    def is_box_mask(self) -> bool:
        """Return True if label belongs to box_mask policy."""
        return self.policy == POLICY_BOX_MASK

    @property
    def is_polygon_mask(self) -> bool:
        """Return True if label belongs to polygon_mask policy."""
        return self.policy == POLICY_POLYGON_MASK

    @property
    def is_polyline(self) -> bool:
        """Return True if label belongs to polyline policy."""
        return self.policy == POLICY_POLYLINE

    @property
    def is_instance(self) -> bool:
        """Return True if label represents countable foreground instance."""
        return self.group == GROUP_INSTANCE or self.policy == POLICY_BOX_MASK

    @property
    def is_region(self) -> bool:
        """Return True if label represents background region/stuff."""
        return self.group == GROUP_REGION or self.policy == POLICY_POLYGON_MASK

    @property
    def is_lane(self) -> bool:
        """Return True if label represents lane boundary or road marking."""
        return self.group == GROUP_LANE or self.policy == POLICY_POLYLINE

    @property
    def supports_bounding_box(self) -> bool:
        """Return True if rectangle/box output is permitted."""
        return SHAPE_RECTANGLE in self.allowed_shapes

    @property
    def supports_mask(self) -> bool:
        """Return True if raster mask output is permitted."""
        return SHAPE_MASK in self.allowed_shapes

    @property
    def supports_polygon(self) -> bool:
        """Return True if vector polygon output is permitted."""
        return SHAPE_POLYGON in self.allowed_shapes

    @property
    def supports_polyline(self) -> bool:
        """Return True if vector polyline output is permitted."""
        return SHAPE_POLYLINE in self.allowed_shapes

    @property
    def is_thin_structure(self) -> bool:
        """Return True if object is a thin structure (e.g. pole or lane line)."""
        return bool(
            self.special_handling.get("thin_object")
            or self.special_handling.get("thin_line")
            or self.special_handling.get("is_thin_structure")
        )

    @property
    def is_area_like(self) -> bool:
        """Return True if feature represents 2D surface area (e.g. crosswalk)."""
        return bool(self.special_handling.get("area_like"))


# Hardcoded fallback specification guarantee (used when YAML files are unavailable).
# Strictly removes polygon from all instance/box_mask labels.
_BUILTIN_LABEL_GEOMETRY: Dict[str, Dict[str, Any]] = {
    "pedestrian": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Moving/standing human on foot in traffic scene (BDD100K).",
    },
    "rider": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Human mounted on bicycle, motorcycle, horse, or mobility device.",
    },
    "car": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Passenger automobile, sedan, SUV, hatchback, taxi.",
    },
    "truck": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Freight truck, pickup truck, semi-trailer.",
    },
    "bus": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Public transit bus, coach, school bus.",
    },
    "train": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Rail transit train, locomotive, carriage, tram on tracks.",
    },
    "motorcycle": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Motorized two-wheeler or scooter.",
    },
    "bicycle": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Non-motorized pedal bicycle.",
    },
    "traffic light": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Traffic control signal housing (space-separated syntax).",
    },
    "traffic sign": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Road traffic sign face or board (space-separated syntax).",
    },
    "area/alternative": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Alternative drivable roadway space or auxiliary lane.",
    },
    "area/drivable": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Directly drivable lane surface in ego travel corridor.",
    },
    "lane/crosswalk": {
        "group": GROUP_LANE,
        "policy": POLICY_POLYLINE,
        "allowed_shapes": [SHAPE_POLYLINE],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {
            "area_like": False,
            "thin_line": True,
            "supports_bounding_box": False,
            "supports_mask": False,
            "supports_polygon": False,
            "supports_polyline": True,
        },
        "notes": "Pedestrian zebra crosswalk. Emitted strictly as polyline traversal centerline per Policy C.",
    },
    "lane/double white": {
        "group": GROUP_LANE,
        "policy": POLICY_POLYLINE,
        "allowed_shapes": [SHAPE_POLYLINE],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {
            "area_like": False,
            "thin_line": True,
            "supports_bounding_box": False,
            "supports_mask": False,
            "supports_polygon": False,
            "supports_polyline": True,
        },
        "notes": "Linear lane boundary marking: double solid white divider line.",
    },
    "lane/double yellow": {
        "group": GROUP_LANE,
        "policy": POLICY_POLYLINE,
        "allowed_shapes": [SHAPE_POLYLINE],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {
            "area_like": False,
            "thin_line": True,
            "supports_bounding_box": False,
            "supports_mask": False,
            "supports_polygon": False,
            "supports_polyline": True,
        },
        "notes": "Linear lane boundary marking: double solid yellow opposing traffic line.",
    },
    "lane/road curb": {
        "group": GROUP_LANE,
        "policy": POLICY_POLYLINE,
        "allowed_shapes": [SHAPE_POLYLINE],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {
            "area_like": False,
            "thin_line": True,
            "supports_bounding_box": False,
            "supports_mask": False,
            "supports_polygon": False,
            "supports_polyline": True,
        },
        "notes": "Linear boundary: physical raised curb interface separating road from sidewalk.",
    },
    "lane/single other": {
        "group": GROUP_LANE,
        "policy": POLICY_POLYLINE,
        "allowed_shapes": [SHAPE_POLYLINE],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {
            "area_like": False,
            "thin_line": True,
            "supports_bounding_box": False,
            "supports_mask": False,
            "supports_polygon": False,
            "supports_polyline": True,
        },
        "notes": "Linear lane marking: single dashed or other non-standard line.",
    },
    "lane/single white": {
        "group": GROUP_LANE,
        "policy": POLICY_POLYLINE,
        "allowed_shapes": [SHAPE_POLYLINE],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {
            "area_like": False,
            "thin_line": True,
            "supports_bounding_box": False,
            "supports_mask": False,
            "supports_polygon": False,
            "supports_polyline": True,
        },
        "notes": "Linear lane marking: single white lane divider in same travel direction.",
    },
    "lane/single yellow": {
        "group": GROUP_LANE,
        "policy": POLICY_POLYLINE,
        "allowed_shapes": [SHAPE_POLYLINE],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {
            "area_like": False,
            "thin_line": True,
            "supports_bounding_box": False,
            "supports_mask": False,
            "supports_polygon": False,
            "supports_polyline": True,
        },
        "notes": "Linear lane marking: single yellow demarcation line.",
    },
    "road": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "General paved roadway surface excluding curbs and sidewalks.",
    },
    "sidewalk": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Paved footway or pedestrian pavement adjacent to roadway.",
    },
    "building": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Permanent architectural structures, building facades, houses.",
    },
    "wall": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Freestanding brick, masonry, or concrete perimeter wall boundary.",
    },
    "fence": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Wire mesh, wood palisade, or metal railing fence enclosure.",
    },
    "pole": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {
            "thin_object": True,
            "optional_rectangle": True,
            "supports_bounding_box": True,
            "supports_mask": True,
        },
        "notes": "Vertical utility pole/post. Mask preferred due to slender aspect ratio; rectangle required under paired Policy A.",
    },
    "vegetation": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Trees, shrubs, hedges, grass, foliage cover.",
    },
    "terrain": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Unpaved natural terrain, dirt, gravel, rocks, sand.",
    },
    "sky": {
        "group": GROUP_REGION,
        "policy": POLICY_POLYGON_MASK,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Open sky hemisphere, clouds, horizon background.",
    },
    "person": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "General human being regardless of pose or activity (COCO).",
    },
    "traffic_light": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Traffic control signal housing (underscore-separated syntax).",
    },
    "traffic_sign": {
        "group": GROUP_INSTANCE,
        "policy": POLICY_BOX_MASK,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Road traffic sign face or board (underscore-separated syntax).",
    },
}


class Taxonomy:
    """Master Taxonomy Manager for the CVAT 31-Label Schema and 3 Shape Policies.

    Features:
    - O(1) dictionary lookups for metadata, policies, groups, shapes, and special rules.
    - Schema validation ensuring all 31 labels exist exactly once without renaming or merging.
    - Strict partitioning into 3 policies:
        - POLICY_BOX_MASK (14 labels)
        - POLICY_POLYGON_MASK (10 labels)
        - POLICY_POLYLINE (7 labels)
    - Strict non-permission of polygon shape on instance / box_mask labels.
    - Ambiguity isolation and target-task fallback mapping.
    - Helper methods for parser, prompt builder, and shape router.
    """

    def __init__(
        self,
        geometry_config_path: Optional[Union[str, Path]] = None,
        semantics_config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._labels: Dict[str, LabelMetadata] = {}
        self._policy_to_labels: Dict[str, List[str]] = {
            POLICY_BOX_MASK: [],
            POLICY_POLYGON_MASK: [],
            POLICY_POLYLINE: [],
        }
        self._group_to_labels: Dict[str, List[str]] = {
            GROUP_INSTANCE: [],
            GROUP_REGION: [],
            GROUP_LANE: [],
        }
        self._shape_to_labels: Dict[str, List[str]] = {
            SHAPE_RECTANGLE: [],
            SHAPE_MASK: [],
            SHAPE_POLYGON: [],
            SHAPE_POLYLINE: [],
        }
        self._ambiguity_map: Dict[str, str] = {}
        self._ambiguity_pairs: List[Tuple[str, str, str]] = list(AMBIGUITY_PAIRS)

        # Initialize ambiguity bidirectional mappings
        for label_a, label_b, desc in self._ambiguity_pairs:
            self._ambiguity_map[label_a] = label_b
            self._ambiguity_map[label_b] = label_a

        # Load configurations
        self._load_geometry(geometry_config_path)
        self._load_semantics(semantics_config_path)

        # Strictly validate integrity
        self.validate()

    def _load_geometry(self, path: Optional[Union[str, Path]]) -> None:
        """Load geometry configuration from YAML or built-in defaults."""
        data: Dict[str, Any] = {}
        target_path = Path(path) if path else LABEL_GEOMETRY_YAML

        if target_path.exists() and target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict) and "labels" in loaded:
                        data = loaded["labels"]
            except Exception as e:
                logger.warning(f"Failed to read {target_path}: {e}. Using built-in geometry.")

        # If data could not be loaded from file, use built-in geometry
        if not data:
            data = _BUILTIN_LABEL_GEOMETRY

        # Populate labels
        for name, spec in data.items():
            grp = str(spec.get("group", LABEL_TO_GROUP.get(name, GROUP_INSTANCE)))
            pol = str(spec.get("policy", ""))
            if not pol or pol not in VALID_POLICIES:
                pol = LABEL_TO_POLICY.get(name, GROUP_TO_POLICY.get(grp, POLICY_BOX_MASK))

            raw_shapes = list(spec.get("allowed_shapes", [SHAPE_RECTANGLE]))
            # Enforce exact allowed shapes per policy:
            # Policy A: rectangle + mask (no polygon, no polyline)
            # Policy B: polygon + mask (no rectangle, no polyline)
            # Policy C: polyline only (no polygon, no mask, no rectangle)
            if pol == POLICY_BOX_MASK or grp == GROUP_INSTANCE:
                shapes = tuple(s for s in raw_shapes if s in (SHAPE_RECTANGLE, SHAPE_MASK))
                if not shapes:
                    shapes = (SHAPE_RECTANGLE, SHAPE_MASK)
            elif pol == POLICY_POLYLINE or grp == GROUP_LANE:
                shapes = (SHAPE_POLYLINE,)
            elif pol == POLICY_POLYGON_MASK or grp == GROUP_REGION:
                shapes = tuple(s for s in raw_shapes if s in (SHAPE_POLYGON, SHAPE_MASK))
                if not shapes:
                    shapes = (SHAPE_POLYGON, SHAPE_MASK)
            else:
                shapes = tuple(raw_shapes)

            pref_shape = str(spec.get("preferred_shape", shapes[0] if shapes else SHAPE_RECTANGLE))
            if pref_shape not in shapes and shapes:
                pref_shape = shapes[0]

            special = dict(spec.get("special_handling", {}))
            notes = str(spec.get("notes", ""))

            meta = LabelMetadata(
                name=name,
                group=grp,
                allowed_shapes=shapes,
                preferred_shape=pref_shape,
                policy=pol,
                special_handling=special,
                notes=notes,
            )
            self._labels[name] = meta

            # Populate policy index
            if pol in self._policy_to_labels:
                self._policy_to_labels[pol].append(name)
            else:
                self._policy_to_labels[pol] = [name]

            # Populate group index
            if grp in self._group_to_labels:
                self._group_to_labels[grp].append(name)
            else:
                self._group_to_labels[grp] = [name]

            # Populate shape index
            for s in shapes:
                if s in self._shape_to_labels:
                    self._shape_to_labels[s].append(name)
                else:
                    self._shape_to_labels[s] = [name]

    def _load_semantics(self, path: Optional[Union[str, Path]]) -> None:
        """Load semantics and ambiguity rules from YAML if present."""
        target_path = Path(path) if path else LABEL_SEMANTICS_YAML
        if target_path.exists() and target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict) and "ambiguity_pairs" in loaded:
                        pairs_dict = loaded["ambiguity_pairs"]
                        for pair_id, pair_info in pairs_dict.items():
                            labels = pair_info.get("labels", [])
                            if len(labels) == 2:
                                la, lb = labels[0], labels[1]
                                self._ambiguity_map[la] = lb
                                self._ambiguity_map[lb] = la
            except Exception as e:
                logger.warning(f"Failed to read {target_path}: {e}")

    def validate(self) -> None:
        """Validate that all 31 labels exist exactly once and conform to taxonomy constraints.

        Raises:
            TaxonomyValidationError: If label count != 31, any master label is missing,
                extra labels exist, or group/policy/shape configurations are invalid.
        """
        loaded_names = set(self._labels.keys())
        expected_names = set(MASTER_31_LABELS)

        if len(loaded_names) != EXPECTED_LABEL_COUNT:
            raise TaxonomyValidationError(
                f"Taxonomy validation failed: Expected exactly {EXPECTED_LABEL_COUNT} labels, "
                f"found {len(loaded_names)}."
            )

        missing_labels = expected_names - loaded_names
        if missing_labels:
            raise TaxonomyValidationError(
                f"Taxonomy validation failed: Missing master labels: {sorted(missing_labels)}"
            )

        unexpected_labels = loaded_names - expected_names
        if unexpected_labels:
            raise TaxonomyValidationError(
                f"Taxonomy validation failed: Unexpected labels found: {sorted(unexpected_labels)}"
            )

        # Verify strict non-merging of ambiguous labels
        for la, lb, _ in self._ambiguity_pairs:
            if la not in self._labels or lb not in self._labels:
                raise TaxonomyValidationError(
                    f"Ambiguity pair ({la!r}, {lb!r}) missing from taxonomy."
                )
            if la == lb:
                raise TaxonomyValidationError(
                    f"Ambiguity pair ({la!r}, {lb!r}) is merged or identical."
                )

        # Verify policy, group, and shape integrity
        for name, meta in self._labels.items():
            if meta.policy not in VALID_POLICIES:
                raise TaxonomyValidationError(
                    f"Label {name!r} has invalid policy {meta.policy!r}. Allowed: {sorted(VALID_POLICIES)}"
                )

            if meta.group not in VALID_GROUPS:
                raise TaxonomyValidationError(
                    f"Label {name!r} has invalid group {meta.group!r}. Allowed: {sorted(VALID_GROUPS)}"
                )

            for s in meta.allowed_shapes:
                if s not in VALID_SHAPES:
                    raise TaxonomyValidationError(
                        f"Label {name!r} has invalid shape {s!r}. Allowed: {sorted(VALID_SHAPES)}"
                    )

            if meta.preferred_shape not in meta.allowed_shapes:
                raise TaxonomyValidationError(
                    f"Label {name!r} preferred_shape {meta.preferred_shape!r} is not in allowed_shapes: {meta.allowed_shapes}"
                )

        # Enforce exact allowed shapes per policy:
        # Policy A (14 labels): strictly rectangle and mask (NO polygon, NO polyline)
        for name in self.get_box_mask_labels():
            meta = self._labels[name]
            if SHAPE_POLYGON in meta.allowed_shapes:
                raise TaxonomyValidationError(
                    f"Policy A (box_mask) label {name!r} must not permit polygon shape. Allowed: {meta.allowed_shapes}"
                )
            if SHAPE_POLYLINE in meta.allowed_shapes:
                raise TaxonomyValidationError(
                    f"Policy A (box_mask) label {name!r} must not permit polyline shape. Allowed: {meta.allowed_shapes}"
                )

        # Policy B (10 labels): strictly polygon and mask (NO rectangle, NO polyline)
        for name in self.get_polygon_mask_labels():
            meta = self._labels[name]
            if SHAPE_RECTANGLE in meta.allowed_shapes:
                raise TaxonomyValidationError(
                    f"Policy B (polygon_mask) label {name!r} must not permit rectangle shape. Allowed: {meta.allowed_shapes}"
                )
            if SHAPE_POLYLINE in meta.allowed_shapes:
                raise TaxonomyValidationError(
                    f"Policy B (polygon_mask) label {name!r} must not permit polyline shape. Allowed: {meta.allowed_shapes}"
                )

        # Policy C (7 labels): strictly polyline ONLY (NO rectangle, NO mask, NO polygon)
        for name in self.get_polyline_labels():
            meta = self._labels[name]
            if meta.allowed_shapes != (SHAPE_POLYLINE,):
                raise TaxonomyValidationError(
                    f"Policy C (polyline) label {name!r} must permit polyline ONLY. Allowed: {meta.allowed_shapes}"
                )

        # Validate partition counts and mutual exclusivity for the 3 policies
        box_masks = self.get_box_mask_labels()
        polygon_masks = self.get_polygon_mask_labels()
        polylines = self.get_polyline_labels()

        if len(box_masks) != 14:
            raise TaxonomyValidationError(
                f"Expected 14 box_mask labels, got {len(box_masks)}: {box_masks}"
            )
        if len(polygon_masks) != 10:
            raise TaxonomyValidationError(
                f"Expected 10 polygon_mask labels, got {len(polygon_masks)}: {polygon_masks}"
            )
        if len(polylines) != 7:
            raise TaxonomyValidationError(
                f"Expected 7 polyline labels, got {len(polylines)}: {polylines}"
            )

        bm_set = set(box_masks)
        pm_set = set(polygon_masks)
        pl_set = set(polylines)

        if not bm_set.isdisjoint(pm_set):
            raise TaxonomyValidationError(
                f"Policy overlap between box_mask and polygon_mask: {bm_set & pm_set}"
            )
        if not bm_set.isdisjoint(pl_set):
            raise TaxonomyValidationError(
                f"Policy overlap between box_mask and polyline: {bm_set & pl_set}"
            )
        if not pm_set.isdisjoint(pl_set):
            raise TaxonomyValidationError(
                f"Policy overlap between polygon_mask and polyline: {pm_set & pl_set}"
            )

        if (bm_set | pm_set | pl_set) != expected_names:
            raise TaxonomyValidationError(
                f"Union of 3 policies does not equal 31 master labels: {(bm_set | pm_set | pl_set) ^ expected_names}"
            )

        # Validate count per legacy group
        instances = self.get_instance_labels()
        regions = self.get_region_labels()
        lanes = self.get_lane_labels()

        if len(instances) != 14:
            raise TaxonomyValidationError(
                f"Expected 14 instance labels, got {len(instances)}: {instances}"
            )
        if len(regions) != 10:
            raise TaxonomyValidationError(
                f"Expected 10 region labels, got {len(regions)}: {regions}"
            )
        if len(lanes) != 7:
            raise TaxonomyValidationError(
                f"Expected 7 lane labels, got {len(lanes)}: {lanes}"
            )

    # --------------------------------------------------------------------------
    # Policy Lookups & Partition Methods
    # --------------------------------------------------------------------------

    def get_policy(self, label: str) -> str:
        """Retrieve policy ('box_mask', 'polygon_mask', 'polyline') for a label in O(1) time.

        Raises:
            KeyError: If label is not present in taxonomy.
        """
        return self.get_label_info(label).policy

    def is_box_mask(self, label: str) -> bool:
        """Check if label belongs to POLICY_BOX_MASK in O(1) time."""
        if label not in self._labels:
            return False
        return self._labels[label].policy == POLICY_BOX_MASK

    def is_polygon_mask(self, label: str) -> bool:
        """Check if label belongs to POLICY_POLYGON_MASK in O(1) time."""
        if label not in self._labels:
            return False
        return self._labels[label].policy == POLICY_POLYGON_MASK

    def is_polyline(self, label: str) -> bool:
        """Check if label belongs to POLICY_POLYLINE in O(1) time."""
        if label not in self._labels:
            return False
        return self._labels[label].policy == POLICY_POLYLINE

    def get_labels_by_policy(self, policy: str) -> Tuple[str, ...]:
        """Return all labels belonging to the specified policy."""
        return tuple(self._policy_to_labels.get(policy, []))

    def get_box_mask_labels(self) -> Tuple[str, ...]:
        """Return all 14 box_mask labels."""
        return self.get_labels_by_policy(POLICY_BOX_MASK)

    def get_polygon_mask_labels(self) -> Tuple[str, ...]:
        """Return all 10 polygon_mask labels."""
        return self.get_labels_by_policy(POLICY_POLYGON_MASK)

    def get_polyline_labels(self) -> Tuple[str, ...]:
        """Return all 7 polyline labels."""
        return self.get_labels_by_policy(POLICY_POLYLINE)

    # --------------------------------------------------------------------------
    # O(1) Lookups & Inspection Methods
    # --------------------------------------------------------------------------

    def is_valid_label(self, label: str) -> bool:
        """Check if label exists in master 31-label taxonomy in O(1) time."""
        return label in self._labels

    def get_label_info(self, label: str) -> LabelMetadata:
        """Retrieve complete metadata for a label in O(1) time.

        Raises:
            KeyError: If label is not present in taxonomy.
        """
        if label not in self._labels:
            raise KeyError(f"Label {label!r} is not in the 31-label taxonomy.")
        return self._labels[label]

    def get_metadata(self, label: str) -> Optional[LabelMetadata]:
        """Retrieve metadata for label, or None if unknown."""
        return self._labels.get(label)

    def get_group(self, label: str) -> str:
        """Retrieve group ('instance', 'region', 'lane') for a label in O(1) time."""
        return self.get_label_info(label).group

    def get_allowed_shapes(self, label: str) -> Tuple[str, ...]:
        """Retrieve allowed CVAT shapes for a label in O(1) time."""
        return self.get_label_info(label).allowed_shapes

    def get_preferred_shape(self, label: str) -> str:
        """Retrieve preferred CVAT shape for a label in O(1) time."""
        return self.get_label_info(label).preferred_shape

    def get_special_handling(self, label: str) -> Dict[str, Any]:
        """Retrieve special geometric handling parameters in O(1) time."""
        return self.get_label_info(label).special_handling

    # --------------------------------------------------------------------------
    # Collection Accessors
    # --------------------------------------------------------------------------

    def get_all_labels(self) -> Tuple[str, ...]:
        """Return all 31 master labels in canonical schema order."""
        return MASTER_31_LABELS

    def get_all_label_names(self) -> Tuple[str, ...]:
        """Return all 31 master labels in canonical schema order (alias for get_all_labels)."""
        return self.get_all_labels()

    def get_labels_by_group(self, group: str) -> Tuple[str, ...]:
        """Return all labels belonging to the specified group."""
        return tuple(self._group_to_labels.get(group, []))

    def get_instance_labels(self) -> Tuple[str, ...]:
        """Return all 14 instance labels."""
        return self.get_labels_by_group(GROUP_INSTANCE)

    def get_region_labels(self) -> Tuple[str, ...]:
        """Return all 10 region labels."""
        return self.get_labels_by_group(GROUP_REGION)

    def get_lane_labels(self) -> Tuple[str, ...]:
        """Return all 7 lane labels."""
        return self.get_labels_by_group(GROUP_LANE)

    def get_labels_by_shape(self, shape: str) -> Tuple[str, ...]:
        """Return all labels that support a specific CVAT shape type."""
        return tuple(self._shape_to_labels.get(shape, []))

    def get_policy_a_labels(self) -> Tuple[str, ...]:
        """Return all 14 Policy A (box_mask) labels."""
        return self.get_box_mask_labels()

    def get_policy_b_labels(self) -> Tuple[str, ...]:
        """Return all 10 Policy B (polygon_mask) labels."""
        return self.get_polygon_mask_labels()

    def get_policy_c_labels(self) -> Tuple[str, ...]:
        """Return all 7 Policy C (polyline) labels."""
        return self.get_polyline_labels()

    def get_bbox_compatible_labels(self) -> Tuple[str, ...]:
        """Return all labels supporting rectangular bounding box output (strictly 14 Policy A instance labels)."""
        return self.get_box_mask_labels()

    # --------------------------------------------------------------------------
    # Ambiguity & Task-Adaptive Fallback Policy
    # --------------------------------------------------------------------------

    def is_ambiguous_label(self, label: str) -> bool:
        """Check if label has an ambiguous counterpart in the taxonomy."""
        return label in self._ambiguity_map

    def get_ambiguity_counterpart(self, label: str) -> Optional[str]:
        """Get the counterpart label for an ambiguous label, or None."""
        return self._ambiguity_map.get(label)

    def resolve_label_for_task(
        self,
        source_label: str,
        target_task_labels: Sequence[str],
    ) -> Tuple[str, Optional[str]]:
        """Resolve a predicted or incoming label against a target CVAT task schema.

        Rules:
        1. If source_label is in target_task_labels, preserve it verbatim (exact match).
        2. If source_label is ambiguous (e.g. 'pedestrian' vs 'person', 'traffic light' vs 'traffic_light'):
           - If target task contains BOTH variations, NO cross-mapping is performed.
           - If target task contains ONLY the counterpart variation, map source_label
             to the counterpart and return an audit warning.
        3. If neither or no counterpart exists, return source_label unchanged.

        Args:
            source_label: Predicted or input label name.
            target_task_labels: Sequence of labels defined in the target CVAT task.

        Returns:
            Tuple of (resolved_label, audit_warning_or_none).
        """
        target_set = set(target_task_labels)

        # Rule 1: Exact match in target task
        if source_label in target_set:
            return source_label, None

        # Rule 2: Ambiguity adaptation
        counterpart = self.get_ambiguity_counterpart(source_label)
        if counterpart is not None:
            # Check if task defines counterpart
            if counterpart in target_set:
                # If task defines both, strict isolation prohibits cross-mapping
                if source_label in target_set:
                    return source_label, None

                # Task has ONLY the counterpart: apply adaptive fallback
                warning = (
                    f"Label '{source_label}' dynamically mapped to task schema '{counterpart}' "
                    f"(target task defines only '{counterpart}')."
                )
                return counterpart, warning

        return source_label, None

    # --------------------------------------------------------------------------
    # Parser & Validator Helpers
    # --------------------------------------------------------------------------

    def validate_shape_for_label(self, label: str, shape_type: str) -> bool:
        """Validate whether shape_type is permitted for the given label."""
        if label not in self._labels:
            return False
        return shape_type in self._labels[label].allowed_shapes

    # --------------------------------------------------------------------------
    # Shape Router Helpers
    # --------------------------------------------------------------------------

    def route_shapes(
        self,
        label: str,
        requested_mode: str = "box_and_mask",
        has_box: bool = True,
        has_mask: bool = False,
    ) -> List[str]:
        """Determine which CVAT shape types should be output for an object detection.

        Strictly enforces the 3 annotation policies:
        - Policy A (14 instance labels): strictly rectangle + mask. Must receive both box
          and mask to emit both [rectangle, mask]; otherwise dropped (returns []).
          Pole is treated strictly under Policy A with zero legacy divergence.
        - Policy B (10 region labels): strictly polygon + mask. Must receive contour/mask
          (has_mask=True) to emit both [polygon, mask]. Never emits rectangle.
        - Policy C (7 lane labels): strictly polyline only. Emits [polyline] if has_mask=True.
          Never emits rectangle, polygon, or mask.

        Args:
            label: Target label name.
            requested_mode: Requested mode (full_31 / box_and_mask).
            has_box: Whether a bounding box is available.
            has_mask: Whether a segmentation mask/polygon/polyline is available.

        Returns:
            List of CVAT shape types ('rectangle', 'mask', 'polygon', 'polyline') to produce.
        """
        if label not in self._labels:
            return []

        meta = self._labels[label]

        # Policy A: Foreground Instances (14 classes) -> RECTANGLE + MASK
        if meta.is_box_mask or meta.is_instance:
            if has_box and has_mask:
                return [SHAPE_RECTANGLE, SHAPE_MASK]
            return []

        # Policy B: Semantic Regions (10 classes) -> POLYGON + MASK (never rectangle)
        if meta.is_polygon_mask or meta.is_region:
            if has_mask:
                return [SHAPE_POLYGON, SHAPE_MASK]
            return []

        # Policy C: Lane Markings & Demarcations (7 classes) -> POLYLINE ONLY
        if meta.is_polyline or meta.is_lane:
            if has_mask:
                return [SHAPE_POLYLINE]
            return []

        return []

    # --------------------------------------------------------------------------
    # Prompt Builder Helpers
    # --------------------------------------------------------------------------

    def filter_labels_by_mode(
        self,
        mode: str = "box",
        requested_labels: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Filter candidate labels compatible with the requested inference mode.

        Args:
            mode: 'box', 'mask', 'polyline', or 'box_and_mask'.
            requested_labels: Optional candidate subset. If None, considers all 31 labels.

        Returns:
            List of label strings compatible with the mode.
        """
        candidates = list(requested_labels) if requested_labels is not None else list(MASTER_31_LABELS)

        if mode == "box":
            return [c for c in candidates if c in self._labels and self._labels[c].supports_bounding_box]
        elif mode in ("mask", "polygon"):
            return [
                c
                for c in candidates
                if c in self._labels and (self._labels[c].supports_mask or self._labels[c].supports_polygon)
            ]
        elif mode == "polyline":
            return [c for c in candidates if c in self._labels and self._labels[c].supports_polyline]
        elif mode in ("box_and_mask", "full_31"):
            return [
                c
                for c in candidates
                if c in self._labels
                and (
                    self._labels[c].supports_bounding_box
                    or self._labels[c].supports_mask
                    or self._labels[c].supports_polygon
                    or self._labels[c].supports_polyline
                )
            ]
        return candidates

    def build_prompt_manifest(
        self,
        active_labels: Sequence[str],
        include_groups: bool = True,
    ) -> str:
        """Format an active label list for LLM vision prompts with group grouping."""
        valid_active = [lbl for lbl in active_labels if lbl in self._labels]

        if not include_groups:
            return "\n".join(f"- {lbl}" for lbl in valid_active)

        instances = [lbl for lbl in valid_active if self._labels[lbl].is_instance]
        regions = [lbl for lbl in valid_active if self._labels[lbl].is_region]
        lanes = [lbl for lbl in valid_active if self._labels[lbl].is_lane]

        sections: List[str] = []
        if instances:
            sections.append("Foreground Instances - Policy A (Rectangle + Mask):\n" + "\n".join(f"- {lbl}" for lbl in instances))
        if regions:
            sections.append("Background Regions & Semantic Surfaces - Policy B (Polygon):\n" + "\n".join(f"- {lbl}" for lbl in regions))
        if lanes:
            sections.append("Lane Markings - Policy C (Polyline Only):\n" + "\n".join(f"- {lbl}" for lbl in lanes))

        return "\n\n".join(sections)

    def generate_ambiguity_warning_prompt(self, active_labels: Sequence[str]) -> str:
        """Generate targeted prompt directives for ambiguous labels in active_labels."""
        active_set = set(active_labels)
        warnings: List[str] = []

        for la, lb, desc in self._ambiguity_pairs:
            has_a = la in active_set
            has_b = lb in active_set

            if has_a and has_b:
                warnings.append(
                    f"CRITICAL: Both '{la}' and '{lb}' are in the allowed label list. "
                    f"Do NOT confuse them or use them interchangeably. Output the exact intended class."
                )
            elif has_a and not has_b:
                warnings.append(
                    f"STRICT: Use ONLY '{la}'. The variation '{lb}' is NOT permitted. Never substitute '{lb}'."
                )
            elif has_b and not has_a:
                warnings.append(
                    f"STRICT: Use ONLY '{lb}'. The variation '{la}' is NOT permitted. Never substitute '{la}'."
                )

        return "\n".join(warnings)


# Singleton / cached instance
_TAXONOMY_INSTANCE: Optional[Taxonomy] = None


def get_taxonomy() -> Taxonomy:
    """Return the global cached Taxonomy instance."""
    global _TAXONOMY_INSTANCE
    if _TAXONOMY_INSTANCE is None:
        _TAXONOMY_INSTANCE = Taxonomy()
    return _TAXONOMY_INSTANCE


def get_policy(label: str) -> str:
    """Return the policy ('box_mask', 'polygon_mask', 'polyline') for a label."""
    return get_taxonomy().get_policy(label)


def is_box_mask(label: str) -> bool:
    """Check if label belongs to POLICY_BOX_MASK using the global taxonomy instance."""
    return get_taxonomy().is_box_mask(label)


def is_polygon_mask(label: str) -> bool:
    """Check if label belongs to POLICY_POLYGON_MASK using the global taxonomy instance."""
    return get_taxonomy().is_polygon_mask(label)


def is_polyline(label: str) -> bool:
    """Check if label belongs to POLICY_POLYLINE using the global taxonomy instance."""
    return get_taxonomy().is_polyline(label)


def validate_cvat_output_shapes(
    shapes: List[Dict[str, Any]],
    taxonomy: Optional[Taxonomy] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Strict runtime output validator for CVAT shapes before returning to CVAT.

    Guarantees:
    - In 31-label mode:
      * Policy A (14 instance labels): Every accepted object has BOTH a rectangle and mask
        sharing the exact same integer group_id. Single shapes (box-only or mask-only) are rejected.
      * Policy B (10 region labels): Every accepted region has BOTH a polygon and mask
        sharing the exact same integer group_id. Never bounding box, never single shape.
      * Policy C (7 lane labels): Strictly polyline ONLY. Never rectangle, polygon, or mask.
    - All coordinates and points conform to CVAT data format specs.

    Args:
        shapes: Candidate list of CVAT shape dictionaries.
        taxonomy: Optional Taxonomy instance. If omitted, uses global taxonomy.

    Returns:
        Tuple of (validated_shapes, validation_warnings).
    """
    tax = taxonomy or get_taxonomy()
    warnings: List[str] = []

    if not isinstance(shapes, list):
        return [], ["Invalid shapes container: expected list"]

    # Step 1: Basic schema & geometry validation
    structurally_valid: List[Tuple[int, Dict[str, Any]]] = []
    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            warnings.append(f"shape_schema_error: Shape at index {idx} is not a dictionary; dropped")
            continue

        label = shape.get("label")
        shape_type = shape.get("type")

        if not isinstance(label, str) or not label.strip():
            warnings.append(f"shape_schema_error: Shape at index {idx} missing valid label; dropped")
            continue

        label = label.strip()
        if label not in tax.get_all_label_names():
            warnings.append(f"unknown_label: Label {label!r} not in 31-label taxonomy; dropped")
            continue

        if shape_type not in (SHAPE_RECTANGLE, SHAPE_POLYGON, SHAPE_POLYLINE, SHAPE_MASK):
            warnings.append(f"invalid_shape_type: Unknown shape type {shape_type!r} for {label}; dropped")
            continue

        # Geometric points / mask structure check
        if shape_type == SHAPE_RECTANGLE:
            pts = shape.get("points")
            if not isinstance(pts, (list, tuple)) or len(pts) != 4:
                warnings.append(f"invalid_box_points: Rectangle for {label} must have 4 coordinates; dropped")
                continue
            if float(pts[2]) <= float(pts[0]) or float(pts[3]) <= float(pts[1]):
                warnings.append(f"invalid_box_dims: Rectangle for {label} has non-positive area; dropped")
                continue

        elif shape_type == SHAPE_POLYGON:
            pts = shape.get("points")
            if not isinstance(pts, (list, tuple)) or len(pts) < 6 or len(pts) % 2 != 0:
                warnings.append(f"invalid_polygon_points: Polygon for {label} must have >= 3 points (6 coords); dropped")
                continue

        elif shape_type == SHAPE_POLYLINE:
            pts = shape.get("points")
            if not isinstance(pts, (list, tuple)) or len(pts) < 4 or len(pts) % 2 != 0:
                warnings.append(f"invalid_polyline_points: Polyline for {label} must have >= 2 points (4 coords); dropped")
                continue

        elif shape_type == SHAPE_MASK:
            mask_data = shape.get("mask")
            pts = shape.get("points")
            if not isinstance(mask_data, (list, tuple)) or len(mask_data) == 0:
                warnings.append(f"invalid_mask_data: Mask for {label} missing raster mask array; dropped")
                continue
            if pts is not None and (not isinstance(pts, (list, tuple)) or len(pts) < 4):
                warnings.append(f"invalid_mask_points: Mask for {label} has invalid points array; dropped")
                continue

        structurally_valid.append((idx, shape))

    # Step 2: Policy Enforcement
    # Policy A: group_id -> must contain exactly 1 rectangle and 1 mask
    # Policy B: group_id -> must contain exactly 1 polygon and 1 mask
    # Policy C: shape -> must be polyline ONLY

    policy_a_groups: Dict[Tuple[Any, str], List[Tuple[int, Dict[str, Any]]]] = {}
    policy_b_groups: Dict[Tuple[Any, str], List[Tuple[int, Dict[str, Any]]]] = {}
    approved_indices: set = set()

    for orig_idx, shape in structurally_valid:
        lbl = shape["label"]
        stype = shape["type"]
        policy = tax.get_policy(lbl)
        gid = shape.get("group_id")

        if policy == POLICY_BOX_MASK:
            if gid is None:
                warnings.append(f"policy_a_violation: Instance shape '{lbl}' ({stype}) missing group_id; dropped")
                continue
            if stype not in (SHAPE_RECTANGLE, SHAPE_MASK):
                warnings.append(f"policy_a_violation: Instance shape '{lbl}' cannot have type {stype!r}; dropped")
                continue
            key = (gid, lbl)
            policy_a_groups.setdefault(key, []).append((orig_idx, shape))

        elif policy == POLICY_POLYGON_MASK:
            if gid is None:
                warnings.append(f"policy_b_violation: Region shape '{lbl}' ({stype}) missing group_id; dropped")
                continue
            if stype not in (SHAPE_POLYGON, SHAPE_MASK):
                warnings.append(f"policy_b_violation: Region shape '{lbl}' cannot have type {stype!r}; dropped")
                continue
            key = (gid, lbl)
            policy_b_groups.setdefault(key, []).append((orig_idx, shape))

        elif policy == POLICY_POLYLINE:
            if stype != SHAPE_POLYLINE:
                warnings.append(f"policy_c_violation: Lane label '{lbl}' must be polyline ONLY, got {stype!r}; dropped")
                continue
            approved_indices.add(orig_idx)

    # Validate Policy A groups (must have exactly 1 rectangle and 1 mask)
    for (gid, lbl), group_entries in policy_a_groups.items():
        types = [s["type"] for _, s in group_entries]
        has_rect = types.count(SHAPE_RECTANGLE) == 1
        has_mask = types.count(SHAPE_MASK) == 1
        if has_rect and has_mask and len(group_entries) == 2:
            for o_idx, _ in group_entries:
                approved_indices.add(o_idx)
        else:
            warnings.append(
                f"policy_a_violation: Instance group {gid} for '{lbl}' must have exactly 1 rectangle and 1 mask. Found {types}; annotation dropped"
            )

    # Validate Policy B groups (must have exactly 1 polygon and 1 mask)
    for (gid, lbl), group_entries in policy_b_groups.items():
        types = [s["type"] for _, s in group_entries]
        has_poly = types.count(SHAPE_POLYGON) == 1
        has_mask = types.count(SHAPE_MASK) == 1
        if has_poly and has_mask and len(group_entries) == 2:
            for o_idx, _ in group_entries:
                approved_indices.add(o_idx)
        else:
            warnings.append(
                f"policy_b_violation: Region group {gid} for '{lbl}' must have exactly 1 polygon and 1 mask. Found {types}; annotation dropped"
            )

    # Construct validated shapes preserving original sequence order
    validated_shapes: List[Dict[str, Any]] = [
        shape for orig_idx, shape in structurally_valid if orig_idx in approved_indices
    ]

    return validated_shapes, warnings


def get_group(label: str) -> str:
    """Return the legacy group ('instance', 'region', 'lane') for a label."""
    return get_taxonomy().get_group(label)


class TaxonomyRegistry:
    """Compatibility registry wrapper pointing to global Taxonomy singleton."""

    @classmethod
    def get_instance(cls) -> Taxonomy:
        return get_taxonomy()

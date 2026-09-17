"""Master 31-Label Taxonomy, Geometry Routing, and Semantics Module.

Phase 3B Architectural Specification:
- Validates the exact 31-label master schema.
- Enforces strict rules: no label renaming, no merging of ambiguous labels.
  ('pedestrian' != 'person', 'traffic light' != 'traffic_light', 'traffic sign' != 'traffic_sign').
- Maps each label to its category/group ('instance', 'region', 'lane').
- Specifies allowed CVAT shape types ('rectangle', 'mask', 'polygon', 'polyline')
  and special geometric handling (e.g. pole thin structure, lane/crosswalk area-like surface).
- Provides O(1) label lookups, parser adaptation, prompt construction, and shape routing.
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

# Recognized taxonomy groups
GROUP_INSTANCE = "instance"
GROUP_REGION = "region"
GROUP_LANE = "lane"
VALID_GROUPS: FrozenSet[str] = frozenset({GROUP_INSTANCE, GROUP_REGION, GROUP_LANE})

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
    special_handling: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_instance(self) -> bool:
        """Return True if label represents countable foreground instance."""
        return self.group == GROUP_INSTANCE

    @property
    def is_region(self) -> bool:
        """Return True if label represents background region/stuff."""
        return self.group == GROUP_REGION

    @property
    def is_lane(self) -> bool:
        """Return True if label represents lane boundary or road marking."""
        return self.group == GROUP_LANE

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


# Hardcoded fallback specification guarantee (used when YAML files are unavailable)
_BUILTIN_LABEL_GEOMETRY: Dict[str, Dict[str, Any]] = {
    "pedestrian": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Moving/standing human on foot in traffic scene (BDD100K).",
    },
    "rider": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Human mounted on bicycle, motorcycle, horse, or mobility device.",
    },
    "car": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Passenger automobile, sedan, SUV, hatchback, taxi.",
    },
    "truck": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Freight truck, pickup truck, semi-trailer.",
    },
    "bus": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Public transit bus, coach, school bus.",
    },
    "train": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Rail transit train, locomotive, carriage, tram on tracks.",
    },
    "motorcycle": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Motorized two-wheeler or scooter.",
    },
    "bicycle": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Non-motorized pedal bicycle.",
    },
    "traffic light": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Traffic control signal housing (space-separated syntax).",
    },
    "traffic sign": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Road traffic sign face or board (space-separated syntax).",
    },
    "area/alternative": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Alternative drivable roadway space or auxiliary lane.",
    },
    "area/drivable": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Directly drivable lane surface in ego travel corridor.",
    },
    "lane/crosswalk": {
        "group": GROUP_LANE,
        "allowed_shapes": [SHAPE_POLYGON, SHAPE_MASK, SHAPE_RECTANGLE],
        "preferred_shape": SHAPE_POLYGON,
        "special_handling": {
            "area_like": True,
            "thin_line": False,
            "supports_bounding_box": True,
            "supports_mask": True,
            "supports_polyline": False,
        },
        "notes": "Pedestrian zebra crosswalk. 2D surface area (area-like vs thin line).",
    },
    "lane/double white": {
        "group": GROUP_LANE,
        "allowed_shapes": [SHAPE_POLYLINE, SHAPE_POLYGON, SHAPE_MASK],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {"area_like": False, "thin_line": True, "supports_polyline": True},
        "notes": "Linear lane boundary marking: double solid white divider line.",
    },
    "lane/double yellow": {
        "group": GROUP_LANE,
        "allowed_shapes": [SHAPE_POLYLINE, SHAPE_POLYGON, SHAPE_MASK],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {"area_like": False, "thin_line": True, "supports_polyline": True},
        "notes": "Linear lane boundary marking: double solid yellow opposing traffic line.",
    },
    "lane/road curb": {
        "group": GROUP_LANE,
        "allowed_shapes": [SHAPE_POLYLINE, SHAPE_POLYGON, SHAPE_MASK],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {"area_like": False, "thin_line": True, "supports_polyline": True},
        "notes": "Linear boundary: physical raised curb interface separating road from sidewalk.",
    },
    "lane/single other": {
        "group": GROUP_LANE,
        "allowed_shapes": [SHAPE_POLYLINE, SHAPE_POLYGON, SHAPE_MASK],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {"area_like": False, "thin_line": True, "supports_polyline": True},
        "notes": "Linear lane marking: single dashed or other non-standard line.",
    },
    "lane/single white": {
        "group": GROUP_LANE,
        "allowed_shapes": [SHAPE_POLYLINE, SHAPE_POLYGON, SHAPE_MASK],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {"area_like": False, "thin_line": True, "supports_polyline": True},
        "notes": "Linear lane marking: single white lane divider in same travel direction.",
    },
    "lane/single yellow": {
        "group": GROUP_LANE,
        "allowed_shapes": [SHAPE_POLYLINE, SHAPE_POLYGON, SHAPE_MASK],
        "preferred_shape": SHAPE_POLYLINE,
        "special_handling": {"area_like": False, "thin_line": True, "supports_polyline": True},
        "notes": "Linear lane marking: single yellow demarcation line.",
    },
    "road": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "General paved roadway surface excluding curbs and sidewalks.",
    },
    "sidewalk": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Paved footway or pedestrian pavement adjacent to roadway.",
    },
    "building": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Permanent architectural structures, building facades, houses.",
    },
    "wall": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Freestanding brick, masonry, or concrete perimeter wall boundary.",
    },
    "fence": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Wire mesh, wood palisade, or metal railing fence enclosure.",
    },
    "pole": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON, SHAPE_RECTANGLE],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {
            "thin_object": True,
            "mask_preferred": True,
            "optional_rectangle": True,
            "supports_bounding_box": True,
            "supports_mask": True,
        },
        "notes": "Vertical utility pole/post. Mask preferred due to slender aspect ratio; rectangle optional.",
    },
    "vegetation": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Trees, shrubs, hedges, grass, foliage cover.",
    },
    "terrain": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Unpaved natural terrain, dirt, gravel, rocks, sand.",
    },
    "sky": {
        "group": GROUP_REGION,
        "allowed_shapes": [SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_MASK,
        "special_handling": {"supports_bounding_box": False, "supports_mask": True},
        "notes": "Open sky hemisphere, clouds, horizon background.",
    },
    "person": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "General human being regardless of pose or activity (COCO).",
    },
    "traffic_light": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Traffic control signal housing (underscore-separated syntax).",
    },
    "traffic_sign": {
        "group": GROUP_INSTANCE,
        "allowed_shapes": [SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON],
        "preferred_shape": SHAPE_RECTANGLE,
        "special_handling": {"supports_bounding_box": True, "supports_mask": True},
        "notes": "Road traffic sign face or board (underscore-separated syntax).",
    },
}


class Taxonomy:
    """Master Taxonomy Manager for the CVAT 31-Label Schema.

    Features:
    - O(1) dictionary lookups for metadata, groups, shapes, and special rules.
    - Schema validation ensuring all 31 labels exist exactly once without renaming or merging.
    - Strict ambiguity isolation and target-task fallback mapping.
    - Helper methods for parser, prompt builder, and shape router.
    """

    def __init__(
        self,
        geometry_config_path: Optional[Union[str, Path]] = None,
        semantics_config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._labels: Dict[str, LabelMetadata] = {}
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
            grp = str(spec.get("group", GROUP_INSTANCE))
            shapes = tuple(spec.get("allowed_shapes", [SHAPE_RECTANGLE]))
            pref_shape = str(spec.get("preferred_shape", shapes[0] if shapes else SHAPE_RECTANGLE))
            special = dict(spec.get("special_handling", {}))
            notes = str(spec.get("notes", ""))

            meta = LabelMetadata(
                name=name,
                group=grp,
                allowed_shapes=shapes,
                preferred_shape=pref_shape,
                special_handling=special,
                notes=notes,
            )
            self._labels[name] = meta

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
                extra labels exist, or group/shape configurations are invalid.
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

        # Verify group and shape integrity
        for name, meta in self._labels.items():
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

        # Validate count per group
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
        return self._labels[label]

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

    def get_bbox_compatible_labels(self) -> Tuple[str, ...]:
        """Return all labels supporting rectangular bounding box output (14 instance + crosswalk)."""
        return tuple(
            name
            for name in MASTER_31_LABELS
            if SHAPE_RECTANGLE in self._labels[name].allowed_shapes
        )

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

        Args:
            label: Target label name.
            requested_mode: Requested mode ('box', 'mask', 'box_and_mask').
            has_box: Whether a bounding box is available.
            has_mask: Whether a segmentation mask/polygon is available.

        Returns:
            List of CVAT shape types ('rectangle', 'mask', 'polygon', 'polyline') to produce.
        """
        if label not in self._labels:
            return [SHAPE_RECTANGLE] if has_box else []

        meta = self._labels[label]
        shapes: List[str] = []

        # Region objects: Never output rectangles, only mask or polygon
        if meta.is_region:
            if has_mask and SHAPE_MASK in meta.allowed_shapes:
                return [SHAPE_MASK]
            elif has_mask and SHAPE_POLYGON in meta.allowed_shapes:
                return [SHAPE_POLYGON]
            return []

        # Lane markings:
        if meta.is_lane:
            if meta.name == "lane/crosswalk":
                # Area-like crosswalk
                if requested_mode == "box" and has_box:
                    return [SHAPE_RECTANGLE]
                if has_mask:
                    return [SHAPE_POLYGON if SHAPE_POLYGON in meta.allowed_shapes else SHAPE_MASK]
                return [SHAPE_RECTANGLE] if has_box else []
            else:
                # Linear lane markings (lines / curbs)
                if has_mask and SHAPE_POLYLINE in meta.allowed_shapes:
                    return [SHAPE_POLYLINE]
                elif has_mask and SHAPE_MASK in meta.allowed_shapes:
                    return [SHAPE_MASK]
                return []

        # Instance objects:
        if meta.is_instance:
            # Special handling for pole: mask preferred
            if meta.name == "pole":
                if requested_mode == "box" and has_box:
                    return [SHAPE_RECTANGLE]
                if has_mask:
                    if requested_mode == "box_and_mask" and has_box:
                        return [SHAPE_MASK, SHAPE_RECTANGLE]
                    return [SHAPE_MASK]
                return [SHAPE_RECTANGLE] if has_box else []

            # Standard instances
            if requested_mode == "box":
                if has_box and SHAPE_RECTANGLE in meta.allowed_shapes:
                    return [SHAPE_RECTANGLE]
            elif requested_mode == "mask":
                if has_mask and SHAPE_MASK in meta.allowed_shapes:
                    return [SHAPE_MASK]
                elif has_mask and SHAPE_POLYGON in meta.allowed_shapes:
                    return [SHAPE_POLYGON]
            elif requested_mode == "box_and_mask":
                if has_box and SHAPE_RECTANGLE in meta.allowed_shapes:
                    shapes.append(SHAPE_RECTANGLE)
                if has_mask and SHAPE_MASK in meta.allowed_shapes:
                    shapes.append(SHAPE_MASK)
                return shapes

        return [SHAPE_RECTANGLE] if (has_box and SHAPE_RECTANGLE in meta.allowed_shapes) else []

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
            mode: 'box', 'mask', or 'box_and_mask'.
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
        elif mode == "box_and_mask":
            return [
                c
                for c in candidates
                if c in self._labels
                and (
                    self._labels[c].supports_bounding_box
                    or self._labels[c].supports_mask
                    or self._labels[c].supports_polygon
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
            sections.append("Foreground Instances (Bounding Box & Segment):\n" + "\n".join(f"- {lbl}" for lbl in instances))
        if regions:
            sections.append("Background Regions & Structures (Segment/Mask only):\n" + "\n".join(f"- {lbl}" for lbl in regions))
        if lanes:
            sections.append("Lane Markings & Road Geometry (Polyline/Polygon):\n" + "\n".join(f"- {lbl}" for lbl in lanes))

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

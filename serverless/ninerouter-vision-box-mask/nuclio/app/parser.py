"""Robust JSON parser and coordinate converter for 9Router Vision responses.

Handles:
- Stripping markdown code fences (```json ... ```).
- Extracting outermost JSON objects or arrays.
- Cleaning trailing commas and formatting quirks.
- Validating against schema across 3 strict annotation policies:
  * Policy A: Object instances (14 labels) with box_2d and mask contour (>= 3 vertices).
  * Policy B: Semantic regions (10 labels) with polygon contour (>= 3 vertices) without box.
  * Policy C: Lane demarcations (7 labels) with polyline centerline (>= 2 points) without box/mask.
- Strict category and label verification against allowed labels without unwanted merging/aliasing.
- First-class typed geometry fields on ParsedObject (instance_mask, region_polygon, lane_polyline).
- Explicit legacy migration helper for Phase 3B responses (migrate_legacy_full31_item).
- Normalized [0, 1000] integer checks and coordinate denormalization to clamped pixel boxes.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.taxonomy import (
    BOX_MASK_LABELS,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    MASTER_31_LABELS,
)

logger = logging.getLogger(__name__)


class VisionParseError(ValueError):
    """Raised when parsing or validating vision model responses fails."""
    pass


MAX_CONTOUR_VERTICES: int = 10_000

BOX_MASK_SET: Set[str] = set(BOX_MASK_LABELS)
POLYGON_MASK_SET: Set[str] = set(POLYGON_MASK_LABELS)
POLYLINE_SET: Set[str] = set(POLYLINE_LABELS)


@dataclass
class ParsedObject:
    """Represents a validated detected object, semantic region, or lane demarcation."""
    label: str
    box_2d: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] in [0, 1000] (Policy A only)
    pixel_box: Optional[List[float]] = None  # [x1, y1, x2, y2] clamped to image dimensions (Policy A only)
    confidence: Optional[float] = None

    # First-class distinct geometry fields (Section 7)
    instance_mask: Optional[List[List[Union[int, float]]]] = None  # [[x, y], ...] in [0, 1000] (Policy A)
    region_polygon: Optional[List[List[Union[int, float]]]] = None  # [[x, y], ...] in [0, 1000] (Policy B)
    lane_polyline: Optional[List[List[Union[int, float]]]] = None  # [[x, y], ...] in [0, 1000] (Policy C)

    pixel_polygon: Optional[List[Tuple[float, float]]] = None  # [(x, y), ...] in image pixel space
    pixel_polyline: Optional[List[Tuple[float, float]]] = None  # [(x, y), ...] in image pixel space for lanes
    cvat_mask: Optional[List[int]] = None  # [crop_pixels..., xmin, ymin, xmax, ymax]
    group_id: Optional[int] = None  # Optional instance grouping ID (Policy A only)
    _legacy_mask: Optional[List[List[Union[int, float]]]] = field(default=None, repr=False)

    def __init__(
        self,
        label: str,
        box_2d: Optional[List[int]] = None,
        pixel_box: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        mask: Optional[List[List[Union[int, float]]]] = None,
        pixel_polygon: Optional[List[Tuple[float, float]]] = None,
        cvat_mask: Optional[List[int]] = None,
        group_id: Optional[int] = None,
        *,
        instance_mask: Optional[List[List[Union[int, float]]]] = None,
        region_polygon: Optional[List[List[Union[int, float]]]] = None,
        lane_polyline: Optional[List[List[Union[int, float]]]] = None,
        pixel_polyline: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.label = label
        self.box_2d = box_2d
        self.pixel_box = pixel_box
        self.confidence = confidence
        self.pixel_polygon = pixel_polygon
        self.pixel_polyline = pixel_polyline
        self.cvat_mask = cvat_mask
        self.group_id = group_id

        # Route explicit first-class attributes if provided
        self.instance_mask = instance_mask
        self.region_polygon = region_polygon
        self.lane_polyline = lane_polyline
        self._legacy_mask = mask

        # Backward compatibility routing if generic `mask` was supplied
        if mask is not None:
            if self.instance_mask is None and self.region_polygon is None and self.lane_polyline is None:
                if label in POLYGON_MASK_SET:
                    self.region_polygon = mask
                elif label in POLYLINE_SET:
                    self.lane_polyline = mask
                else:
                    self.instance_mask = mask

    @property
    def mask(self) -> Optional[List[List[Union[int, float]]]]:
        """Backward compatibility accessor for raw contour / coordinates."""
        if self.instance_mask is not None:
            return self.instance_mask
        if self.region_polygon is not None:
            return self.region_polygon
        if self.lane_polyline is not None:
            return self.lane_polyline
        return self._legacy_mask

    @mask.setter
    def mask(self, val: Optional[List[List[Union[int, float]]]]) -> None:
        self._legacy_mask = val
        if self.label in POLYGON_MASK_SET:
            self.region_polygon = val
        elif self.label in POLYLINE_SET:
            self.lane_polyline = val
        else:
            self.instance_mask = val

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation matching required category schema."""
        data: Dict[str, Any] = {
            "label": self.label,
        }
        if self.box_2d is not None:
            data["box_2d"] = [int(c) for c in self.box_2d]
        if self.confidence is not None:
            data["confidence"] = round(float(self.confidence), 3)
        if self.pixel_box is not None:
            data["pixel_box"] = [round(float(c), 2) for c in self.pixel_box]

        # First-class category geometry mapping
        if self.instance_mask is not None:
            data["mask"] = [[int(round(float(c))) for c in pt] for pt in self.instance_mask]
        elif self.region_polygon is not None:
            data["polygon"] = [[int(round(float(c))) for c in pt] for pt in self.region_polygon]
        elif self.lane_polyline is not None:
            data["polyline"] = [[int(round(float(c))) for c in pt] for pt in self.lane_polyline]
        elif self._legacy_mask is not None:
            data["mask"] = [[int(round(float(c))) for c in pt] for pt in self._legacy_mask]

        if self.pixel_polygon is not None:
            data["pixel_polygon"] = [[round(float(c), 2) for c in pt] for pt in self.pixel_polygon]
        if self.pixel_polyline is not None:
            data["pixel_polyline"] = [[round(float(c), 2) for c in pt] for pt in self.pixel_polyline]
        if self.group_id is not None:
            data["group_id"] = self.group_id
        return data

    def to_cvat_shape(self, occluded: bool = False) -> Dict[str, Any]:
        """Convert to CVAT rectangle shape specification."""
        points = self.pixel_box if self.pixel_box is not None else [0.0, 0.0, 0.0, 0.0]
        res: Dict[str, Any] = {
            "type": "rectangle",
            "label": self.label,
            "points": [round(float(p), 2) for p in points],
            "occluded": occluded,
            "attributes": [],
        }
        if self.confidence is not None:
            res["confidence"] = round(float(self.confidence), 3)
        if self.group_id is not None:
            res["group_id"] = self.group_id
        return res

    def to_cvat_mask_shape(self) -> Optional[Dict[str, Any]]:
        """Convert to CVAT mask shape specification."""
        if not self.cvat_mask:
            return None
        res: Dict[str, Any] = {
            "type": "mask",
            "label": self.label,
            "mask": self.cvat_mask,
        }
        if self.pixel_polygon:
            res["points"] = [round(float(c), 2) for pt in self.pixel_polygon for c in pt]
        if self.confidence is not None:
            res["confidence"] = str(round(float(self.confidence), 2))
        if self.group_id is not None:
            res["group_id"] = self.group_id
        return res

    def to_cvat_polygon_shape(self) -> Optional[Dict[str, Any]]:
        """Convert to CVAT polygon shape specification (Policy B)."""
        pts = self.pixel_polygon
        if not pts and self.region_polygon:
            pts = [(float(p[0]), float(p[1])) for p in self.region_polygon]
        if not pts:
            return None
        res: Dict[str, Any] = {
            "type": "polygon",
            "label": self.label,
            "points": [round(float(c), 2) for pt in pts for c in pt],
        }
        if self.confidence is not None:
            res["confidence"] = str(round(float(self.confidence), 2))
        return res

    def to_cvat_polyline_shape(self) -> Optional[Dict[str, Any]]:
        """Convert to CVAT polyline shape specification (Policy C)."""
        pts = self.pixel_polyline
        if not pts and self.lane_polyline:
            pts = [(float(p[0]), float(p[1])) for p in self.lane_polyline]
        if not pts:
            return None
        res: Dict[str, Any] = {
            "type": "polyline",
            "label": self.label,
            "points": [round(float(c), 2) for pt in pts for c in pt],
        }
        if self.confidence is not None:
            res["confidence"] = str(round(float(self.confidence), 2))
        return res


@dataclass
class ParseResult:
    """Result of parsing and validating model response."""
    objects: List[ParsedObject] = field(default_factory=list)
    regions: List[ParsedObject] = field(default_factory=list)
    lanes: List[ParsedObject] = field(default_factory=list)
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def all_items(self) -> List[ParsedObject]:
        """Return all parsed objects, regions, and lanes combined."""
        return self.objects + self.regions + self.lanes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to prediction dictionary matching required output schema."""
        res: Dict[str, Any] = {
            "objects": [obj.to_dict() for obj in self.objects]
        }
        if self.regions:
            res["regions"] = [obj.to_dict() for obj in self.regions]
        if self.lanes:
            res["lanes"] = [obj.to_dict() for obj in self.lanes]
        return res

    def to_cvat_annotations(self) -> List[Dict[str, Any]]:
        """Convert all objects to CVAT rectangle shapes."""
        return [obj.to_cvat_shape() for obj in self.all_items]


def clean_json_string(text: str) -> str:
    """Strip markdown formatting and extract outermost JSON."""
    cleaned = text.strip()

    # Step 1: Check for markdown code fences (```json ... ``` or ``` ... ```)
    markdown_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = markdown_pattern.search(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Step 2: Extract outermost JSON object { ... } or array [ ... ]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")

    # Prefer object { ... } if present and valid
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        # Check if array starts earlier than object
        if first_bracket != -1 and first_bracket < first_brace and last_bracket > last_brace:
            cleaned = cleaned[first_bracket : last_bracket + 1]
        else:
            cleaned = cleaned[first_brace : last_brace + 1]
    elif first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        cleaned = cleaned[first_bracket : last_bracket + 1]

    # Step 3: Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    return cleaned


def parse_confidence_score(val: Any) -> float:
    """Parse and normalize confidence score into [0.0, 1.0]."""
    if isinstance(val, str):
        val = val.strip().rstrip("%")
    c = float(val)
    if 1.0 < c <= 100.0:
        c = c / 100.0
    return round(max(0.0, min(1.0, c)), 3)


def migrate_legacy_full31_item(
    item: Dict[str, Any],
    category: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Explicit migration helper for legacy Phase 3B item geometry (Section 5).

    Converts:
    - region: 'mask' -> 'polygon'
    - lane: 'mask' -> 'polyline'
    - lane: 'points' -> 'polyline'

    Only performed when explicitly enabled. Emits 'legacy_geometry_migrated' warning.

    Args:
        item: Raw detection dictionary.
        category: 'objects', 'regions', or 'lanes'.

    Returns:
        Tuple of (migrated_item_dict, list_of_migration_warnings).
    """
    item_copy = dict(item)
    warnings: List[str] = []
    label = item_copy.get("label", "unknown")

    if category == "regions":
        if "polygon" not in item_copy and "mask" in item_copy:
            item_copy["polygon"] = item_copy.pop("mask")
            warnings.append(
                f"legacy_geometry_migrated: Region detection '{label}' legacy 'mask' field migrated to 'polygon'"
            )
    elif category == "lanes":
        if "polyline" not in item_copy:
            if "mask" in item_copy:
                item_copy["polyline"] = item_copy.pop("mask")
                warnings.append(
                    f"legacy_geometry_migrated: Lane detection '{label}' legacy 'mask' field migrated to 'polyline'"
                )
            elif "points" in item_copy:
                item_copy["polyline"] = item_copy.pop("points")
                warnings.append(
                    f"legacy_geometry_migrated: Lane detection '{label}' legacy 'points' field migrated to 'polyline'"
                )

    return item_copy, warnings


def parse_legacy_full31_geometry(
    item: Dict[str, Any],
    category: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Alias for migrate_legacy_full31_item."""
    return migrate_legacy_full31_item(item, category)


def _validate_points_sequence(
    points_raw: Any,
    min_vertices: int,
    field_name: str,
    label: str,
    idx: int,
    strict: bool,
) -> Tuple[Optional[List[List[Union[int, float]]]], List[str]]:
    """Helper to validate and normalize coordinate points [[x, y], ...]."""
    warnings: List[str] = []
    if not isinstance(points_raw, (list, tuple)):
        msg = f"Detection at index {idx} ('{label}') has non-list {field_name}: {type(points_raw).__name__}"
        if strict:
            raise VisionParseError(msg)
        warnings.append(msg)
        return None, warnings

    if len(points_raw) > MAX_CONTOUR_VERTICES:
        msg = f"Detection at index {idx} ('{label}') {field_name} has {len(points_raw)} vertices, exceeding maximum allowed limit {MAX_CONTOUR_VERTICES}"
        if strict:
            raise VisionParseError(msg)
        warnings.append(f"Detection at index {idx} ('{label}') {field_name} has {len(points_raw)} vertices, exceeding limit {MAX_CONTOUR_VERTICES}; clamped to first {MAX_CONTOUR_VERTICES}")
        points_raw = points_raw[:MAX_CONTOUR_VERTICES]

    if len(points_raw) < min_vertices:
        msg = f"Detection at index {idx} ('{label}') has degenerate {field_name} with fewer than {min_vertices} vertices: {points_raw!r}"
        if strict:
            raise VisionParseError(msg)
        warnings.append(f"Detection at index {idx} ('{label}') has degenerate mask: {points_raw!r}")
        return None, warnings

    norm_points: List[List[Union[int, float]]] = []
    for pt_idx, pt in enumerate(points_raw):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            msg = f"Detection at index {idx} ('{label}') {field_name} vertex at {pt_idx} is not [x, y]: {pt!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            return None, warnings
        try:
            px = float(pt[0])
            py = float(pt[1])
            if math.isnan(px) or math.isnan(py) or math.isinf(px) or math.isinf(py):
                raise ValueError("NaN or Inf coordinate")
            # Range clamp [0, 1000]
            px = max(0.0, min(1000.0, px))
            py = max(0.0, min(1000.0, py))
            norm_points.append([px, py])
        except (ValueError, TypeError) as e:
            msg = f"Detection at index {idx} ('{label}') {field_name} vertex {pt_idx} non-numeric: {e}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            return None, warnings

    return norm_points, warnings


def parse_and_validate(
    raw_response: Union[str, Dict[str, Any]],
    allowed_labels: Optional[List[str]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    strict: bool = True,
    fallback_confidence: Optional[float] = None,
    mode: Optional[str] = None,
    require_object_mask: Optional[bool] = None,
    enable_legacy_migration: bool = False,
) -> ParseResult:
    """Parse raw response, validate category-aware schema and coordinates, and compute pixel geometries.

    Args:
        raw_response: Model output string or dictionary.
        allowed_labels: List of valid labels. If provided, labels not in this list
            cause an error (strict=True) or are skipped (strict=False).
        image_width: Image width for calculating clamped pixel coordinates.
        image_height: Image height for calculating clamped pixel coordinates.
        strict: If True, raises VisionParseError on invalid boxes or labels;
            if False, skips invalid entries with a warning.
        fallback_confidence: Default confidence to assign when confidence is omitted.
        mode: Detection mode ('full_31', 'box', 'mask', 'box_and_mask').
        require_object_mask: If True, Policy A objects strictly require mask.
            Defaults to True in 'full_31' mode or when regions/lanes are present.
        enable_legacy_migration: If True, allows legacy Phase 3B geometry migration
            (region.mask -> polygon, lane.mask -> polyline) with warnings.
            Default production parsing is strictly False.

    Returns:
        ParseResult containing list of validated ParsedObjects.
    """
    raw_text = ""
    warnings: List[str] = []

    # Extract text from response dict if needed
    if isinstance(raw_response, dict):
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            choice = raw_response["choices"][0]
            message = choice.get("message", {})
            raw_text = message.get("content", "")
        elif "objects" in raw_response or "regions" in raw_response or "lanes" in raw_response:
            data = raw_response
            raw_text = json.dumps(raw_response)
        else:
            raw_text = json.dumps(raw_response)
    else:
        raw_text = str(raw_response)

    if not isinstance(raw_response, dict) or ("objects" not in raw_response and "regions" not in raw_response and "lanes" not in raw_response):
        cleaned = clean_json_string(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            try:
                evaluated = ast.literal_eval(cleaned)
                if isinstance(evaluated, (dict, list)):
                    data = evaluated
                else:
                    raise VisionParseError(
                        f"Failed to parse vision model response as valid JSON: {e}. Cleaned text was: {cleaned[:300]!r}"
                    )
            except Exception:
                raise VisionParseError(
                    f"Failed to parse vision model response as valid JSON: {e}. Cleaned text was: {cleaned[:300]!r}"
                )

    # Allow top-level list [ {...}, {...} ] or {"objects": [ ... ], "regions": [ ... ], "lanes": [ ... ]}
    raw_objects: List[Any] = []
    raw_regions: List[Any] = []
    raw_lanes: List[Any] = []

    if isinstance(data, list):
        raw_objects = data
    elif isinstance(data, dict):
        has_any = False
        if "objects" in data:
            raw_objects = data["objects"]
            has_any = True
        if "regions" in data:
            raw_regions = data["regions"]
            has_any = True
        if "lanes" in data:
            raw_lanes = data["lanes"]
            has_any = True
        if not has_any:
            raise VisionParseError("Missing 'objects' key in vision model response")
    else:
        raise VisionParseError(f"Expected top-level JSON object or array, got {type(data).__name__}")

    if not isinstance(raw_objects, list):
        raise VisionParseError(f"Expected 'objects' to be a list, got {type(raw_objects).__name__}")
    if not isinstance(raw_regions, list):
        raise VisionParseError(f"Expected 'regions' to be a list, got {type(raw_regions).__name__}")
    if not isinstance(raw_lanes, list):
        raise VisionParseError(f"Expected 'lanes' to be a list, got {type(raw_lanes).__name__}")

    # Determine 3-policy strict mode vs legacy box-only mode
    is_3_policy_scene = (
        mode == "full_31"
        or (mode is None and (len(raw_regions) > 0 or len(raw_lanes) > 0 or (isinstance(data, dict) and ("regions" in data or "lanes" in data))))
    )

    if require_object_mask is None:
        require_object_mask = is_3_policy_scene

    allowed_set = set(allowed_labels) if allowed_labels is not None else None
    validated_objects: List[ParsedObject] = []
    validated_regions: List[ParsedObject] = []
    validated_lanes: List[ParsedObject] = []

    current_group_counter = 1

    # =========================================================================
    # Category 1: OBJECTS (Policy A - 14 labels)
    # Required fields: label, box_2d, mask (when require_object_mask=True).
    # Allowed geometry: mask ([[x, y], ...] contour with >= 3 vertices).
    # Reject: object polygon, object polyline, object points-only, missing box_2d, missing mask.
    # =========================================================================
    for idx, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            msg = f"Object detection at index {idx} is not a dictionary: {item!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            msg = f"Object detection at index {idx} has missing or empty label: {label!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        label = label.strip()

        if allowed_set is not None and label not in allowed_set:
            msg = f"Object detection at index {idx} has label {label!r} not in allowed labels: {sorted(allowed_set)}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        # Strictly Category-Aware: reject Policy B or Policy C labels in Policy A objects
        if is_3_policy_scene:
            if label in POLYGON_MASK_SET:
                msg = f"Detection at index {idx} has label {label!r} which belongs to semantic regions (Policy B), not objects (Policy A)"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue
            if label in POLYLINE_SET:
                msg = f"Detection at index {idx} has label {label!r} which belongs to lane markings (Policy C), not objects (Policy A)"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue

        # Reject prohibited geometries for objects
        if "polygon" in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited geometry 'polygon' for Policy A object; allowed geometry is 'mask'"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        if "polyline" in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited geometry 'polyline' for Policy A object; allowed geometry is 'mask'"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        if "points" in item and "mask" not in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited points-only geometry for Policy A object; allowed geometry is 'mask'"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        # Required box_2d validation
        box_2d = item.get("box_2d")
        if box_2d is None:
            msg = f"Detection at index {idx} ('{label}') has invalid box_2d: None. Expected 4 coordinates [ymin, xmin, ymax, xmax]."
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
            msg = f"Detection at index {idx} ('{label}') has invalid box_2d: {box_2d!r}. Expected 4 coordinates [ymin, xmin, ymax, xmax]."
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        try:
            ymin = int(round(float(box_2d[0])))
            xmin = int(round(float(box_2d[1])))
            ymax = int(round(float(box_2d[2])))
            xmax = int(round(float(box_2d[3])))
        except (ValueError, TypeError) as e:
            msg = f"Detection at index {idx} coordinates cannot be converted to int: {e}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        # Inversion checks
        if ymin > ymax:
            if strict:
                raise VisionParseError(f"Detection at index {idx} has inverted y coordinates: ymin ({ymin}) > ymax ({ymax})")
            warnings.append(f"Fixed inverted y coordinates for {label}: ymin={ymin}, ymax={ymax}")
            ymin, ymax = ymax, ymin

        if xmin > xmax:
            if strict:
                raise VisionParseError(f"Detection at index {idx} has inverted x coordinates: xmin ({xmin}) > xmax ({xmax})")
            warnings.append(f"Fixed inverted x coordinates for {label}: xmin={xmin}, xmax={xmax}")
            xmin, xmax = xmax, xmin

        # Degenerate 0-area box
        if ymin == ymax or xmin == xmax:
            msg = f"Detection at index {idx} has zero area: {[ymin, xmin, ymax, xmax]}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        # Range bounds [0, 1000]
        if not (0 <= ymin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmin <= 1000 and 0 <= xmax <= 1000):
            if strict:
                raise VisionParseError(f"Detection at index {idx} coordinates out of [0, 1000] bounds: {[ymin, xmin, ymax, xmax]}")
            warnings.append(f"Clamped out of bounds coordinates for {label}: {[ymin, xmin, ymax, xmax]}")

        ymin = max(0, min(1000, ymin))
        xmin = max(0, min(1000, xmin))
        ymax = max(0, min(1000, ymax))
        xmax = max(0, min(1000, xmax))

        # Calculate pixel box
        pixel_box: Optional[List[float]] = None
        if image_width is not None and image_height is not None:
            if image_width <= 0 or image_height <= 0:
                raise VisionParseError(f"Invalid image dimensions: {image_width}x{image_height}")
            x1 = (xmin / 1000.0) * float(image_width)
            y1 = (ymin / 1000.0) * float(image_height)
            x2 = (xmax / 1000.0) * float(image_width)
            y2 = (ymax / 1000.0) * float(image_height)
            x1 = max(0.0, min(float(image_width), x1))
            y1 = max(0.0, min(float(image_height), y1))
            x2 = max(0.0, min(float(image_width), x2))
            y2 = max(0.0, min(float(image_height), y2))

            if x2 <= x1 or y2 <= y1:
                msg = f"Detection at index {idx} produced invalid pixel box: {[x1, y1, x2, y2]}"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue
            pixel_box = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

        box_2d_valid = [ymin, xmin, ymax, xmax]

        # Parse confidence
        raw_conf = item.get("confidence")
        conf_val: Optional[float] = None
        if raw_conf is not None:
            try:
                conf_val = parse_confidence_score(raw_conf)
            except Exception:
                if strict:
                    raise VisionParseError(f"Detection at index {idx} has invalid confidence: {raw_conf!r}")
                warnings.append(f"Detection at index {idx} has invalid confidence {raw_conf!r}; using fallback {fallback_confidence}")
                conf_val = fallback_confidence
        else:
            conf_val = fallback_confidence

        # Parse required or optional mask contour
        raw_mask = item.get("mask")
        parsed_mask: Optional[List[List[Union[int, float]]]] = None
        pixel_polygon: Optional[List[Tuple[float, float]]] = None
        cvat_mask: Optional[List[int]] = None

        if raw_mask is None:
            if require_object_mask:
                msg = f"Detection at index {idx} ('{label}') is missing required 'mask' contour per Policy A"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue
        else:
            norm_mask, mask_warnings = _validate_points_sequence(
                points_raw=raw_mask,
                min_vertices=3,
                field_name="mask",
                label=label,
                idx=idx,
                strict=strict,
            )
            warnings.extend(mask_warnings)
            if norm_mask is None:
                if strict:
                    raise VisionParseError(f"Detection at index {idx} has invalid mask format: {raw_mask!r}")
                parsed_mask = None
            else:
                parsed_mask = norm_mask
                if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
                    from core.geometry import polygon_to_cvat_mask
                    geo_res = polygon_to_cvat_mask(parsed_mask, width=image_width, height=image_height)
                    if geo_res:
                        pixel_polygon = geo_res.get("pixel_polygon")
                        cvat_mask = geo_res.get("mask")

        # Group ID handling for instance objects
        raw_gid = item.get("group_id")
        if raw_gid is not None:
            try:
                group_id = int(raw_gid)
            except (ValueError, TypeError):
                group_id = current_group_counter
        else:
            group_id = current_group_counter
        current_group_counter = max(current_group_counter + 1, (group_id + 1) if isinstance(group_id, int) else current_group_counter + 1)

        obj = ParsedObject(
            label=label,
            box_2d=box_2d_valid,
            pixel_box=pixel_box,
            confidence=conf_val,
            instance_mask=parsed_mask,
            pixel_polygon=pixel_polygon,
            cvat_mask=cvat_mask,
            group_id=group_id,
        )
        validated_objects.append(obj)

    # =========================================================================
    # Category 2: REGIONS (Policy B - 10 labels)
    # Required fields: label, polygon.
    # Allowed geometry: polygon ([[x, y], ...] contour with >= 3 vertices).
    # Reject: region mask, region polyline, region box_2d as required geometry.
    # Note: Tool-cvat derives CVAT native mask from the polygon contour.
    # =========================================================================
    for idx, raw_item in enumerate(raw_regions):
        if not isinstance(raw_item, dict):
            msg = f"Region detection at index {idx} is not a dictionary: {raw_item!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        item = raw_item
        if enable_legacy_migration:
            item, mig_warns = migrate_legacy_full31_item(item, "regions")
            warnings.extend(mig_warns)

        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            msg = f"Region detection at index {idx} has missing or empty label: {label!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        label = label.strip()

        if allowed_set is not None and label not in allowed_set:
            msg = f"Region detection at index {idx} has label {label!r} not in allowed labels: {sorted(allowed_set)}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        # Strictly Category-Aware
        if is_3_policy_scene:
            if label in BOX_MASK_SET:
                msg = f"Detection at index {idx} has label {label!r} which belongs to object instances (Policy A), not regions (Policy B)"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue
            if label in POLYLINE_SET:
                msg = f"Detection at index {idx} has label {label!r} which belongs to lane markings (Policy C), not regions (Policy B)"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue

        # Reject prohibited geometries for regions
        if "mask" in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited geometry 'mask' for Policy B region. Allowed geometry is 'polygon' only."
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        if "polyline" in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited geometry 'polyline' for Policy B region. Allowed geometry is 'polygon' only."
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        raw_poly = item.get("polygon")
        if raw_poly is None:
            if "box_2d" in item:
                msg = f"Detection at index {idx} ('{label}') has prohibited box_2d as required geometry for Policy B region. Allowed geometry is 'polygon'."
            else:
                msg = f"Detection at index {idx} ('{label}') is missing required 'polygon' field per Policy B"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        if "box_2d" in item:
            warnings.append(f"region_box_suppressed: Suppressed box_2d for semantic region '{label}'")

        norm_poly, poly_warnings = _validate_points_sequence(
            points_raw=raw_poly,
            min_vertices=3,
            field_name="polygon",
            label=label,
            idx=idx,
            strict=strict,
        )
        warnings.extend(poly_warnings)
        if norm_poly is None:
            if strict:
                raise VisionParseError(f"Detection at index {idx} has invalid polygon format: {raw_poly!r}")
            continue

        # Derive CVAT native mask and pixel polygon from the polygon contour
        pixel_polygon = None
        cvat_mask = None
        if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
            from core.geometry import polygon_to_cvat_mask
            geo_res = polygon_to_cvat_mask(norm_poly, width=image_width, height=image_height)
            if geo_res:
                pixel_polygon = geo_res.get("pixel_polygon")
                cvat_mask = geo_res.get("mask")

        raw_conf = item.get("confidence")
        conf_val = None
        if raw_conf is not None:
            try:
                conf_val = parse_confidence_score(raw_conf)
            except Exception:
                if strict:
                    raise VisionParseError(f"Detection at index {idx} has invalid confidence: {raw_conf!r}")
                conf_val = fallback_confidence
        else:
            conf_val = fallback_confidence

        obj = ParsedObject(
            label=label,
            box_2d=None,
            pixel_box=None,
            confidence=conf_val,
            region_polygon=norm_poly,
            pixel_polygon=pixel_polygon,
            cvat_mask=cvat_mask,
            group_id=None,
        )
        validated_regions.append(obj)

    # =========================================================================
    # Category 3: LANES (Policy C - 7 labels)
    # Required fields: label, polyline (>= 2 ordered points).
    # Allowed geometry: polyline.
    # Reject: lane mask, lane polygon, lane box_2d.
    # CRITICAL (Section 6): DO NOT rasterize polyline as polygon mask! Never call polygon_to_cvat_mask() on lane items.
    # =========================================================================
    for idx, raw_item in enumerate(raw_lanes):
        if not isinstance(raw_item, dict):
            msg = f"Lane detection at index {idx} is not a dictionary: {raw_item!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        item = raw_item
        if enable_legacy_migration:
            item, mig_warns = migrate_legacy_full31_item(item, "lanes")
            warnings.extend(mig_warns)

        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            msg = f"Lane detection at index {idx} has missing or empty label: {label!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        label = label.strip()

        if allowed_set is not None and label not in allowed_set:
            msg = f"Lane detection at index {idx} has label {label!r} not in allowed labels: {sorted(allowed_set)}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        # Strictly Category-Aware
        if is_3_policy_scene:
            if label in BOX_MASK_SET:
                msg = f"Detection at index {idx} has label {label!r} which belongs to object instances (Policy A), not lanes (Policy C)"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue
            if label in POLYGON_MASK_SET:
                msg = f"Detection at index {idx} has label {label!r} which belongs to semantic regions (Policy B), not lanes (Policy C)"
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                continue

        # Reject prohibited geometries for lanes
        if "mask" in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited geometry 'mask' for Policy C lane. Allowed geometry is 'polyline' only."
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        if "polygon" in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited geometry 'polygon' for Policy C lane. Allowed geometry is 'polyline' only."
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue
        if "box_2d" in item:
            msg = f"Detection at index {idx} ('{label}') has prohibited 'box_2d' for Policy C lane. Allowed geometry is 'polyline' only."
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        raw_line = item.get("polyline")
        if raw_line is None:
            msg = f"Detection at index {idx} ('{label}') is missing required 'polyline' field per Policy C"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        norm_line, line_warnings = _validate_points_sequence(
            points_raw=raw_line,
            min_vertices=2,
            field_name="polyline",
            label=label,
            idx=idx,
            strict=strict,
        )
        warnings.extend(line_warnings)
        if norm_line is None:
            if strict:
                raise VisionParseError(f"Detection at index {idx} has invalid polyline format: {raw_line!r}")
            continue

        # Compute pixel polyline (CRITICAL Section 6: DO NOT rasterize polyline as polygon mask!)
        pixel_polyline: Optional[List[Tuple[float, float]]] = None
        if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
            pixel_polyline = [
                (
                    round(max(0.0, min(float(image_width), (float(pt[0]) / 1000.0) * float(image_width))), 2),
                    round(max(0.0, min(float(image_height), (float(pt[1]) / 1000.0) * float(image_height))), 2),
                )
                for pt in norm_line
            ]

        raw_conf = item.get("confidence")
        conf_val = None
        if raw_conf is not None:
            try:
                conf_val = parse_confidence_score(raw_conf)
            except Exception:
                if strict:
                    raise VisionParseError(f"Detection at index {idx} has invalid confidence: {raw_conf!r}")
                conf_val = fallback_confidence
        else:
            conf_val = fallback_confidence

        obj = ParsedObject(
            label=label,
            box_2d=None,
            pixel_box=None,
            confidence=conf_val,
            lane_polyline=norm_line,
            pixel_polyline=pixel_polyline,
            pixel_polygon=None,
            cvat_mask=None,  # Strictly None! Never rasterize polyline as polygon mask!
            group_id=None,
        )
        validated_lanes.append(obj)

    return ParseResult(
        objects=validated_objects,
        regions=validated_regions,
        lanes=validated_lanes,
        raw_text=raw_text,
        warnings=warnings,
    )

"""Robust JSON parser and coordinate converter for 9Router Vision responses.

Handles:
- Stripping markdown code fences (```json ... ```).
- Extracting outermost JSON objects or arrays.
- Cleaning trailing commas and formatting quirks.
- Validating against schema: {"objects": [{"label": ..., "box_2d": [ymin, xmin, ymax, xmax]}]}.
- Strict label verification against allowed labels without unwanted merging/aliasing.
- Normalized [0, 1000] integer checks and coordinate denormalization to clamped pixel boxes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class VisionParseError(ValueError):
    """Raised when parsing or validating vision model responses fails."""
    pass


MAX_CONTOUR_VERTICES: int = 10_000


@dataclass
class ParsedObject:
    """Represents a validated detected object."""
    label: str
    box_2d: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] in [0, 1000]
    pixel_box: Optional[List[float]] = None  # [x1, y1, x2, y2] clamped to image dimensions
    confidence: Optional[float] = None
    mask: Optional[List[List[Union[int, float]]]] = None  # [[x, y], ...] normalized in [0, 1000]
    pixel_polygon: Optional[List[Tuple[float, float]]] = None  # [(x, y), ...] in image pixel space
    cvat_mask: Optional[List[int]] = None  # [crop_pixels..., xmin, ymin, xmax, ymax]
    group_id: Optional[int] = None  # Optional instance grouping ID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data: Dict[str, Any] = {
            "label": self.label,
        }
        if self.box_2d is not None:
            data["box_2d"] = [int(c) for c in self.box_2d]
        if self.confidence is not None:
            data["confidence"] = round(float(self.confidence), 3)
        if self.pixel_box is not None:
            data["pixel_box"] = [round(float(c), 2) for c in self.pixel_box]
        if self.mask is not None:
            data["mask"] = [[int(round(float(c))) for c in pt] for pt in self.mask]
        if self.pixel_polygon is not None:
            data["pixel_polygon"] = [[round(float(c), 2) for c in pt] for pt in self.pixel_polygon]
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


def parse_and_validate(
    raw_response: Union[str, Dict[str, Any]],
    allowed_labels: Optional[List[str]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    strict: bool = True,
    fallback_confidence: Optional[float] = None,
) -> ParseResult:
    """Parse raw response, validate schema and coordinates, and compute pixel boxes.

    Args:
        raw_response: Model output string or dictionary.
        allowed_labels: List of valid labels. If provided, labels not in this list
            cause an error (strict=True) or are skipped (strict=False).
        image_width: Image width for calculating clamped pixel coordinates.
        image_height: Image height for calculating clamped pixel coordinates.
        strict: If True, raises VisionParseError on invalid boxes or labels;
            if False, skips invalid entries with a warning.
        fallback_confidence: Default confidence to assign when confidence is omitted.

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
        elif "objects" in raw_response:
            data = raw_response
            raw_text = json.dumps(raw_response)
        else:
            raw_text = json.dumps(raw_response)
    else:
        raw_text = str(raw_response)

    if not isinstance(raw_response, dict) or "objects" not in raw_response:
        cleaned = clean_json_string(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Fallback: attempt ast.literal_eval for single-quoted Python dict/list literals
            try:
                import ast
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

    allowed_set = set(allowed_labels) if allowed_labels is not None else None
    validated_objects: List[ParsedObject] = []
    validated_regions: List[ParsedObject] = []
    validated_lanes: List[ParsedObject] = []

    region_and_lane_labels = {
        "area/alternative", "area/drivable", "road", "sidewalk",
        "building", "wall", "fence", "vegetation", "terrain", "sky",
        "lane/crosswalk", "lane/double white", "lane/double yellow",
        "lane/road curb", "lane/single other", "lane/single white",
        "lane/single yellow"
    }

    current_group_counter = 1

    def _parse_item(item: Any, idx: int, category: str, group_counter: int) -> Tuple[Optional[ParsedObject], int]:
        if not isinstance(item, dict):
            msg = f"Detection at index {idx} is not a dictionary: {item!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            return None, group_counter

        label = item.get("label")
        box_2d = item.get("box_2d")

        # Label validation
        if not isinstance(label, str) or not label.strip():
            msg = f"Detection at index {idx} has missing or empty label: {label!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            return None, group_counter

        label = label.strip()

        # Strict check against allowed labels
        if allowed_set is not None and label not in allowed_set:
            msg = f"Detection at index {idx} has label {label!r} not in allowed labels: {sorted(allowed_set)}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            return None, group_counter

        is_region_or_lane = (category in ("regions", "lanes")) or (label in region_and_lane_labels)
        require_box = not is_region_or_lane

        box_2d_valid: Optional[List[int]] = None
        pixel_box: Optional[List[float]] = None

        if box_2d is None:
            if require_box:
                msg = f"Detection at index {idx} has invalid box_2d: {box_2d!r}. Expected 4 coordinates [ymin, xmin, ymax, xmax]."
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                return None, group_counter
        else:
            # Coordinate format validation: [ymin, xmin, ymax, xmax]
            if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
                msg = f"Detection at index {idx} has invalid box_2d: {box_2d!r}. Expected 4 coordinates [ymin, xmin, ymax, xmax]."
                if strict:
                    raise VisionParseError(msg)
                warnings.append(msg)
                return None, group_counter

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
                return None, group_counter

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
                return None, group_counter

            # Range bounds [0, 1000]
            if not (0 <= ymin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmin <= 1000 and 0 <= xmax <= 1000):
                if strict:
                    raise VisionParseError(f"Detection at index {idx} coordinates out of [0, 1000] bounds: {[ymin, xmin, ymax, xmax]}")
                warnings.append(f"Clamped out of bounds coordinates for {label}: {[ymin, xmin, ymax, xmax]}")

            ymin = max(0, min(1000, ymin))
            xmin = max(0, min(1000, xmin))
            ymax = max(0, min(1000, ymax))
            xmax = max(0, min(1000, xmax))

            # Calculate pixel coordinates if dimensions provided
            if image_width is not None and image_height is not None:
                if image_width <= 0 or image_height <= 0:
                    raise VisionParseError(f"Invalid image dimensions: {image_width}x{image_height}")

                x1 = (xmin / 1000.0) * float(image_width)
                y1 = (ymin / 1000.0) * float(image_height)
                x2 = (xmax / 1000.0) * float(image_width)
                y2 = (ymax / 1000.0) * float(image_height)

                # Clamp to image boundaries
                x1 = max(0.0, min(float(image_width), x1))
                y1 = max(0.0, min(float(image_height), y1))
                x2 = max(0.0, min(float(image_width), x2))
                y2 = max(0.0, min(float(image_height), y2))

                # Reject invalid pixel boxes (x2 <= x1 or y2 <= y1)
                if x2 <= x1 or y2 <= y1:
                    msg = f"Detection at index {idx} produced invalid pixel box: {[x1, y1, x2, y2]}"
                    if strict:
                        raise VisionParseError(msg)
                    warnings.append(msg)
                    return None, group_counter

                pixel_box = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

            box_2d_valid = [ymin, xmin, ymax, xmax]

        # Parse confidence with fallback policy
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

        # Parse optional mask / polygon contour
        raw_mask = item.get("mask") or item.get("polygon") or item.get("polyline") or item.get("points")
        parsed_mask: Optional[List[List[Union[int, float]]]] = None
        pixel_polygon: Optional[List[Tuple[float, float]]] = None
        cvat_mask: Optional[List[int]] = None

        if raw_mask is not None:
            if isinstance(raw_mask, list) and len(raw_mask) > MAX_CONTOUR_VERTICES:
                if strict:
                    raise VisionParseError(
                        f"Detection at index {idx} has {len(raw_mask)} vertices, exceeding maximum allowed limit ({MAX_CONTOUR_VERTICES})"
                    )
                warnings.append(
                    f"Detection at index {idx} has {len(raw_mask)} vertices exceeding limit {MAX_CONTOUR_VERTICES}; clamped to first {MAX_CONTOUR_VERTICES}"
                )
                raw_mask = raw_mask[:MAX_CONTOUR_VERTICES]

            is_lane_item = (category == "lanes") or (label.startswith("lane/"))
            min_vertices = 2 if is_lane_item else 3

            if isinstance(raw_mask, list) and len(raw_mask) >= min_vertices:
                valid_contour = True
                norm_contour: List[List[Union[int, float]]] = []
                for pt_idx, pt in enumerate(raw_mask):
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        valid_contour = False
                        break
                    try:
                        px = float(pt[0])
                        py = float(pt[1])
                        norm_contour.append([px, py])
                    except (ValueError, TypeError):
                        valid_contour = False
                        break
                if valid_contour and len(norm_contour) >= min_vertices:
                    parsed_mask = norm_contour
                    if len(norm_contour) >= 3 and image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
                        from core.geometry import polygon_to_cvat_mask
                        geo_res = polygon_to_cvat_mask(parsed_mask, width=image_width, height=image_height)
                        if geo_res:
                            pixel_polygon = geo_res.get("pixel_polygon")
                            cvat_mask = geo_res["mask"]
                elif strict:
                    raise VisionParseError(f"Detection at index {idx} has invalid mask format: {raw_mask!r}")
                else:
                    warnings.append(f"Detection at index {idx} has degenerate mask ignored: {raw_mask!r}")
            elif strict and raw_mask is not None:
                raise VisionParseError(f"Detection at index {idx} has invalid mask format: {raw_mask!r}")
            elif raw_mask is not None:
                warnings.append(f"Detection at index {idx} has degenerate mask ignored: {raw_mask!r}")

        # Group ID handling
        if is_region_or_lane:
            group_id = None
            next_counter = group_counter
        else:
            raw_gid = item.get("group_id")
            if raw_gid is not None:
                try:
                    group_id = int(raw_gid)
                except (ValueError, TypeError):
                    group_id = group_counter
            else:
                group_id = group_counter
            next_counter = max(group_counter + 1, (group_id + 1) if isinstance(group_id, int) else group_counter + 1)

        obj = ParsedObject(
            label=label,
            box_2d=box_2d_valid,
            pixel_box=pixel_box,
            confidence=conf_val,
            mask=parsed_mask,
            pixel_polygon=pixel_polygon,
            cvat_mask=cvat_mask,
            group_id=group_id,
        )
        return obj, next_counter

    for idx, item in enumerate(raw_objects):
        obj, current_group_counter = _parse_item(item, idx, "objects", current_group_counter)
        if obj is not None:
            validated_objects.append(obj)

    for idx, item in enumerate(raw_regions):
        obj, current_group_counter = _parse_item(item, idx, "regions", current_group_counter)
        if obj is not None:
            validated_regions.append(obj)

    for idx, item in enumerate(raw_lanes):
        obj, current_group_counter = _parse_item(item, idx, "lanes", current_group_counter)
        if obj is not None:
            validated_lanes.append(obj)

    return ParseResult(
        objects=validated_objects,
        regions=validated_regions,
        lanes=validated_lanes,
        raw_text=raw_text,
        warnings=warnings,
    )

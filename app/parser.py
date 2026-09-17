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


@dataclass
class ParsedObject:
    """Represents a validated detected object."""
    label: str
    box_2d: List[int]  # [ymin, xmin, ymax, xmax] in [0, 1000]
    pixel_box: Optional[List[float]] = None  # [x1, y1, x2, y2] clamped to image dimensions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data: Dict[str, Any] = {
            "label": self.label,
            "box_2d": [int(c) for c in self.box_2d],
        }
        if self.pixel_box is not None:
            data["pixel_box"] = [round(float(c), 2) for c in self.pixel_box]
        return data

    def to_cvat_shape(self, occluded: bool = False) -> Dict[str, Any]:
        """Convert to CVAT rectangle shape specification."""
        points = self.pixel_box if self.pixel_box is not None else [0.0, 0.0, 0.0, 0.0]
        return {
            "type": "rectangle",
            "label": self.label,
            "points": [round(float(p), 2) for p in points],
            "occluded": occluded,
            "attributes": [],
        }


@dataclass
class ParseResult:
    """Result of parsing and validating model response."""
    objects: List[ParsedObject] = field(default_factory=list)
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to prediction dictionary matching required output schema."""
        return {
            "objects": [obj.to_dict() for obj in self.objects]
        }

    def to_cvat_annotations(self) -> List[Dict[str, Any]]:
        """Convert all objects to CVAT rectangle shapes."""
        return [obj.to_cvat_shape() for obj in self.objects]


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


def parse_and_validate(
    raw_response: Union[str, Dict[str, Any]],
    allowed_labels: Optional[List[str]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    strict: bool = True,
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

    # Allow top-level list [ {...}, {...} ] or {"objects": [ ... ]}
    if isinstance(data, list):
        raw_objects = data
    elif isinstance(data, dict):
        if "objects" not in data:
            raise VisionParseError("Missing 'objects' key in vision model response")
        raw_objects = data["objects"]
    else:
        raise VisionParseError(f"Expected top-level JSON object or array, got {type(data).__name__}")

    if not isinstance(raw_objects, list):
        raise VisionParseError(f"Expected 'objects' to be a list, got {type(raw_objects).__name__}")

    allowed_set = set(allowed_labels) if allowed_labels is not None else None
    validated_objects: List[ParsedObject] = []

    for idx, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            msg = f"Detection at index {idx} is not a dictionary: {item!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        label = item.get("label")
        box_2d = item.get("box_2d")

        # Label validation
        if not isinstance(label, str) or not label.strip():
            msg = f"Detection at index {idx} has missing or empty label: {label!r}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        label = label.strip()

        # Strict check against allowed labels
        if allowed_set is not None and label not in allowed_set:
            msg = f"Detection at index {idx} has label {label!r} not in allowed labels: {sorted(allowed_set)}"
            if strict:
                raise VisionParseError(msg)
            warnings.append(msg)
            continue

        # Coordinate format validation: [ymin, xmin, ymax, xmax]
        if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
            msg = f"Detection at index {idx} has invalid box_2d: {box_2d!r}. Expected 4 coordinates [ymin, xmin, ymax, xmax]."
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

        # Calculate pixel coordinates if dimensions provided
        pixel_box: Optional[List[float]] = None
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
                continue

            pixel_box = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

        obj = ParsedObject(
            label=label,
            box_2d=[ymin, xmin, ymax, xmax],
            pixel_box=pixel_box,
        )
        validated_objects.append(obj)

    return ParseResult(
        objects=validated_objects,
        raw_text=raw_text,
        warnings=warnings,
    )

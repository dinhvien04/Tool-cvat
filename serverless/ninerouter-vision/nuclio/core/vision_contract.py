"""Vision Contract and Prompt Specification for CVAT x 9Router AI Annotation.

This module defines:
1. Exact CVAT 31-label schema and the 13 default rectangular bounding box labels.
2. Coordinate normalization / denormalization between 9Router [0, 1000] [ymin, xmin, ymax, xmax]
   and CVAT pixel coordinates [xtl, ytl, xbr, ybr].
3. Strict Vision Prompt generator and OpenAI-compatible chat completion payload builder.
4. Robust response parser and contract validator.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# Path to configuration
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
LABELS_YAML_PATH = CONFIG_DIR / "labels.yaml"
CVAT_LABELS_JSON_PATH = CONFIG_DIR / "cvat_labels.json"

# All 31 CVAT labels in exact order
ALL_31_LABELS: Tuple[str, ...] = (
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

# Default rectangular bounding box label candidates (13 labels)
DEFAULT_BBOX_LABELS: Tuple[str, ...] = (
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
    "person",
    "traffic_light",
    "traffic_sign",
)

# Intentional label distinctions to guard against merging
INTENTIONAL_DISTINCTIONS = {
    ("pedestrian", "person"): "Intentional distinction between general person and pedestrian annotations.",
    ("traffic light", "traffic_light"): "Intentional distinction between spaced and underscored traffic light labels.",
    ("traffic sign", "traffic_sign"): "Intentional distinction between spaced and underscored traffic sign labels.",
}

# Detection and Segmentation Modes
MODE_BOX = "box"
MODE_MASK = "mask"
MODE_BOX_AND_MASK = "box_and_mask"
MODE_FULL_31 = "full_31"
SUPPORTED_MODES = (MODE_BOX, MODE_MASK, MODE_BOX_AND_MASK, MODE_FULL_31)


@dataclass
class DetectedObject:
    """Represents a single detected object in 9Router normalized coordinates [0, 1000]."""
    label: str
    box_2d: List[int]  # [ymin, xmin, ymax, xmax] in [0, 1000]
    confidence: Optional[float] = None
    mask: Optional[List[List[Union[int, float]]]] = None  # [[x1, y1], [x2, y2], ...] normalized in [0, 1000]

    def validate(self, allowed_labels: Optional[List[str]] = None) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"Invalid label: {self.label!r}")

        if allowed_labels is not None and self.label not in allowed_labels:
            raise ValueError(f"Label {self.label!r} is not in allowed labels: {allowed_labels}")

        if not isinstance(self.box_2d, (list, tuple)) or len(self.box_2d) != 4:
            raise ValueError(f"box_2d must be a list/tuple of 4 integers [ymin, xmin, ymax, xmax], got {self.box_2d!r}")

        ymin, xmin, ymax, xmax = self.box_2d
        for val, name in [(ymin, "ymin"), (xmin, "xmin"), (ymax, "ymax"), (xmax, "xmax")]:
            if not isinstance(val, int) and not (isinstance(val, float) and val.is_integer()):
                raise ValueError(f"Coordinate {name}={val!r} must be an integer in [0, 1000]")

        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)

        if not (0 <= ymin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmin <= 1000 and 0 <= xmax <= 1000):
            raise ValueError(f"Coordinates out of bounds [0, 1000]: {[ymin, xmin, ymax, xmax]}")

        if ymin >= ymax:
            raise ValueError(f"Invalid vertical coordinates: ymin ({ymin}) >= ymax ({ymax})")

        if xmin >= xmax:
            raise ValueError(f"Invalid horizontal coordinates: xmin ({xmin}) >= xmax ({xmax})")

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise ValueError(f"Confidence must be numeric, got {self.confidence!r}")
            if not (0.0 <= float(self.confidence) <= 1.0):
                raise ValueError(f"Confidence out of bounds [0.0, 1.0]: {self.confidence!r}")

        if self.mask is not None:
            if not isinstance(self.mask, (list, tuple)):
                raise ValueError(f"Mask must be a list or tuple of points, got {type(self.mask).__name__}")
            if len(self.mask) < 3:
                raise ValueError(f"Mask contour must contain at least 3 vertices, got {len(self.mask)}")
            if len(self.mask) > 10_000:
                raise ValueError(f"Mask contour vertex count ({len(self.mask)}) exceeds maximum limit of 10000")
            for pt_idx, pt in enumerate(self.mask):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise ValueError(f"Mask vertex at index {pt_idx} must be a 2-element sequence [x, y], got {pt!r}")
                px, py = pt
                if isinstance(px, bool) or isinstance(py, bool) or not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
                    raise ValueError(f"Mask vertex at index {pt_idx} coordinates must be numeric, got {[px, py]}")
                px_f, py_f = float(px), float(py)
                if math.isnan(px_f) or math.isnan(py_f) or math.isinf(px_f) or math.isinf(py_f):
                    raise ValueError(f"Mask vertex at index {pt_idx} coordinates cannot be NaN or Inf: {[px, py]}")

    def to_cvat_rect(self, width: int, height: int) -> Dict[str, Any]:
        """Convert to CVAT rectangle format [xtl, ytl, xbr, ybr] in image pixel space."""
        xtl, ytl, xbr, ybr = box_2d_to_cvat_rect(self.box_2d, width=width, height=height)
        res: Dict[str, Any] = {
            "type": "rectangle",
            "label": self.label,
            "points": [xtl, ytl, xbr, ybr],
            "occluded": False,
        }
        if self.confidence is not None:
            res["confidence"] = round(float(self.confidence), 3)
        return res

    def to_cvat_mask(self, width: int, height: int) -> Optional[Dict[str, Any]]:
        """Convert to CVAT native mask format with 1D flattened crop and [xmin, ymin, xmax, ymax]."""
        if not self.mask:
            return None
        from core.geometry import polygon_to_cvat_mask
        geo = polygon_to_cvat_mask(self.mask, width=width, height=height)
        if not geo:
            return None
        res: Dict[str, Any] = {
            "type": "mask",
            "label": self.label,
            "mask": geo["mask"],
        }
        if geo.get("pixel_polygon"):
            res["points"] = [round(float(c), 2) for pt in geo["pixel_polygon"] for c in pt]
        if self.confidence is not None:
            res["confidence"] = str(round(float(self.confidence), 2))
        return res


@dataclass
class DetectionResult:
    """Represents the complete detection result conforming to the 9Router vision contract."""
    objects: List[DetectedObject]

    def to_dict(self) -> Dict[str, Any]:
        objects_out = []
        for obj in self.objects:
            d: Dict[str, Any] = {
                "label": obj.label,
                "box_2d": [int(c) for c in obj.box_2d],
            }
            if obj.confidence is not None:
                d["confidence"] = round(float(obj.confidence), 3)
            objects_out.append(d)
        return {"objects": objects_out}

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_cvat_annotations(self, width: int, height: int) -> List[Dict[str, Any]]:
        """Convert all objects into CVAT shape objects."""
        return [obj.to_cvat_rect(width=width, height=height) for obj in self.objects]


def load_label_config(yaml_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load labels.yaml configuration."""
    path = Path(yaml_path) if yaml_path else LABELS_YAML_PATH
    if not path.exists():
        return {
            "all_labels": list(ALL_31_LABELS),
            "bbox_labels": list(DEFAULT_BBOX_LABELS),
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def box_2d_to_cvat_rect(
    box_2d: List[int],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert 9Router normalized coordinates [ymin, xmin, ymax, xmax] in [0, 1000]
    to CVAT pixel coordinates [xtl, ytl, xbr, ybr].
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

    ymin, xmin, ymax, xmax = box_2d
    xtl = max(0.0, min(float(width), (float(xmin) / 1000.0) * float(width)))
    ytl = max(0.0, min(float(height), (float(ymin) / 1000.0) * float(height)))
    xbr = max(0.0, min(float(width), (float(xmax) / 1000.0) * float(width)))
    ybr = max(0.0, min(float(height), (float(ymax) / 1000.0) * float(height)))

    return round(xtl, 2), round(ytl, 2), round(xbr, 2), round(ybr, 2)


def cvat_rect_to_box_2d(
    xtl: float,
    ytl: float,
    xbr: float,
    ybr: float,
    width: int,
    height: int,
) -> List[int]:
    """
    Convert CVAT pixel coordinates [xtl, ytl, xbr, ybr]
    to 9Router normalized coordinates [ymin, xmin, ymax, xmax] in [0, 1000].
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: width={width}, height={height}")

    # Clamp coordinates to image boundaries
    xtl_clamped = max(0.0, min(float(width), float(xtl)))
    ytl_clamped = max(0.0, min(float(height), float(ytl)))
    xbr_clamped = max(0.0, min(float(width), float(xbr)))
    ybr_clamped = max(0.0, min(float(height), float(ybr)))

    xmin = int(round((xtl_clamped / float(width)) * 1000.0))
    ymin = int(round((ytl_clamped / float(height)) * 1000.0))
    xmax = int(round((xbr_clamped / float(width)) * 1000.0))
    ymax = int(round((ybr_clamped / float(height)) * 1000.0))

    # Ensure bounds [0, 1000]
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))

    return [ymin, xmin, ymax, xmax]


# Vision Prompt Templates

SYSTEM_PROMPT = """You are an expert autonomous driving computer vision system.
Your task is 2D object detection on the provided image.
You must output ONLY a valid JSON object. Do not include markdown formatting (no ```json code blocks), no explanations, and no conversational text.

Return your response adhering strictly to the following JSON schema:
{
  "objects": [
    {
      "label": "<label_name>",
      "box_2d": [ymin, xmin, ymax, xmax],
      "confidence": 0.95
    }
  ]
}

Rules:
1. Coordinate System:
   - box_2d is [ymin, xmin, ymax, xmax].
   - All coordinates are normalized integers in the range [0, 1000].
   - ymin is the top boundary (0 <= ymin < ymax <= 1000).
   - xmin is the left boundary (0 <= xmin < xmax <= 1000).
   - ymax is the bottom boundary (0 <= ymin < ymax <= 1000).
   - xmax is the right boundary (0 <= xmin < xmax <= 1000).
2. Allowed Labels:
   - You MUST ONLY select labels from the allowed list provided in the user prompt.
   - Match the label string EXACTLY as provided, preserving exact casing, spaces, and underscores.
   - Do NOT rename, alter, substitute, or invent labels.
   - If "traffic light" is requested, do NOT output "traffic_light".
   - If "traffic_light" is requested, do NOT output "traffic light".
   - If "pedestrian" is requested, do NOT output "person".
   - If "person" is requested, do NOT output "pedestrian".
   - If "traffic sign" is requested, do NOT output "traffic_sign".
   - If "traffic_sign" is requested, do NOT output "traffic sign".
3. Confidence Score:
   - For each detected object, include an estimated heuristic confidence score between 0.0 and 1.0 (float) reflecting detection certainty and visual clarity.
4. Detection Precision:
   - Provide a tight bounding box around the visible boundaries of each object instance.
   - Detect every clearly visible object instance of the requested classes.
   - Do not hallucinate invisible, fully occluded, or non-existent objects.
   - Separate distinct overlapping or adjacent instances with individual bounding boxes.
5. Output:
   - Output pure JSON only.
   - If no target objects from the allowed labels are visible, return {"objects": []}.
"""


def build_full_31_prompt(
    allowed_labels: Optional[List[str]] = None,
    include_confidence: bool = True,
) -> str:
    """Build the unified Phase 3B vision prompt for full 31-label multi-shape annotation.

    Categorizes labels into 3 strict annotation policies:
    1. Policy A: Object Instances (14 classes: box_2d + mask contour).
    2. Policy B: Semantic Regions (10 classes: boundary contour mask for polygon/mask emission).
    3. Policy C: Lane Markings & Crosswalks (7 classes: ribbon contour for polyline centerline emission).
    """
    labels = set(allowed_labels) if allowed_labels is not None else set(ALL_31_LABELS)

    # 14 Instance Labels (Policy A)
    instance_labels = [
        l for l in (
            "pedestrian", "rider", "car", "truck", "bus", "train",
            "motorcycle", "bicycle", "traffic light", "traffic sign",
            "pole", "person", "traffic_light", "traffic_sign"
        ) if l in labels
    ]

    # 10 Semantic Region Labels (Policy B)
    region_labels = [
        l for l in (
            "area/alternative", "area/drivable", "road", "sidewalk",
            "building", "wall", "fence", "vegetation", "terrain", "sky"
        ) if l in labels
    ]

    # 7 Lane / Marking Labels (Policy C)
    lane_labels = [
        l for l in (
            "lane/crosswalk", "lane/double white", "lane/double yellow",
            "lane/road curb", "lane/single other", "lane/single white",
            "lane/single yellow"
        ) if l in labels
    ]

    inst_str = "\n".join(f"  - {l}" for l in instance_labels) or "  (none)"
    reg_str = "\n".join(f"  - {l}" for l in region_labels) or "  (none)"
    lane_str = "\n".join(f"  - {l}" for l in lane_labels) or "  (none)"

    conf_field = ',\n      "confidence": 0.95' if include_confidence else ""

    return f"""Perform comprehensive road-scene multi-shape annotation on this image for autonomous driving perception across 3 strict annotation policies:

1. Policy A — Foreground Instances (14 classes):
   For every countable object instance, provide BOTH bounding box (box_2d) AND boundary contour (mask).
2. Policy B — Semantic Regions (10 classes):
   For every surface/infrastructure region, provide the precise boundary contour (mask) without bounding box.
3. Policy C — Lane Demarcations & Crosswalks (7 classes):
   For every lane boundary, marking, and pedestrian crosswalk, provide the ribbon contour (mask) along its traversal path without bounding box.

Allowed Labels by Category:
- Policy A — Object Instances (14 classes):
{inst_str}

- Policy B — Semantic Regions (10 classes):
{reg_str}

- Policy C — Lane Markings & Crosswalks (7 classes):
{lane_str}

Output schema:
{{
  "objects": [
    {{
      "label": "<instance_label>",
      "box_2d": [ymin, xmin, ymax, xmax],
      "mask": [[x1, y1], [x2, y2], [x3, y3], ...]{conf_field}
    }}
  ],
  "regions": [
    {{
      "label": "<region_label>",
      "mask": [[x1, y1], [x2, y2], [x3, y3], ...]{conf_field}
    }}
  ],
  "lanes": [
    {{
      "label": "<lane_label>",
      "mask": [[x1, y1], [x2, y2], [x3, y3], ...]{conf_field}
    }}
  ]
}}

Rules:
1. Coordinates: All coordinates are normalized integers in [0, 1000].
2. box_2d: [ymin, xmin, ymax, xmax] required ONLY for Policy A 'objects'. Do NOT include box_2d for Policy B 'regions' or Policy C 'lanes'.
3. mask: Closed polygon contour boundary [[x1, y1], [x2, y2], ...] tracing the outer edge of each instance, region, or lane. Points are [x, y] in [0, 1000].
4. Labels: Choose ONLY from the allowed lists. Never rename or alter labels.
5. If no items are found for a category, return an empty array [].
6. Output raw JSON only (no markdown fences, no explanatory text)."""


def build_user_prompt(
    allowed_labels: Optional[List[str]] = None,
    include_confidence: bool = True,
    mode: str = MODE_BOX,
) -> str:
    """Build the user text prompt with the active candidate labels and requested mode.

    Args:
        allowed_labels: List of candidate labels.
        include_confidence: Whether to request confidence score.
        mode: Detection mode ('box', 'mask', 'box_and_mask', or 'full_31').
    """
    if mode == MODE_FULL_31:
        return build_full_31_prompt(allowed_labels=allowed_labels, include_confidence=include_confidence)

    labels = allowed_labels if allowed_labels is not None else list(DEFAULT_BBOX_LABELS)
    labels_formatted = "\n".join(f"- {label}" for label in labels)

    conf_field = ',\n      "confidence": 0.95' if include_confidence else ""

    if mode in (MODE_MASK, MODE_BOX_AND_MASK):
        return f"""Perform 2D object detection and instance segmentation on this image.
Detect all visible instances of the allowed object classes.

Allowed labels (choose ONLY from this list):
{labels_formatted}

Output schema:
{{
  "objects": [
    {{
      "label": "<exact_allowed_label>",
      "box_2d": [ymin, xmin, ymax, xmax],
      "mask": [[x1, y1], [x2, y2], [x3, y3], ...]{conf_field}
    }}
  ]
}}

Rules:
1. box_2d: Normalized integer coordinates [ymin, xmin, ymax, xmax] in [0, 1000].
2. mask: A closed polygon contour boundary tracing the visible object perimeter.
   - Points MUST be [x, y] coordinates (x horizontal, y vertical) normalized in [0, 1000].
   - Provide at least 4 perimeter vertices per object. Do NOT omit mask.
3. Strictly output raw JSON only (no markdown fences, no explanatory text)."""

    return f"""Detect all visible instances of the allowed object classes in this image.

Allowed labels (choose ONLY from this list):
{labels_formatted}

Output schema:
{{
  "objects": [
    {{
      "label": "<exact_allowed_label>",
      "box_2d": [ymin, xmin, ymax, xmax]{conf_field}
    }}
  ]
}}

Remember:
- Normalized integer coordinates in [0, 1000].
- Format is [ymin, xmin, ymax, xmax].
- Strictly output raw JSON only (no markdown fences, no explanatory text)."""


def build_openai_vision_payload(
    image_base64: str,
    allowed_labels: Optional[List[str]] = None,
    model: str = "gpt-4o",
    image_format: str = "jpeg",
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    mode: str = MODE_BOX,
) -> Dict[str, Any]:
    """
    Build an OpenAI-compatible multimodal chat completions request payload.
    """
    sys_prompt = system_prompt or SYSTEM_PROMPT
    user_prompt = build_user_prompt(allowed_labels=allowed_labels, mode=mode)

    # Clean base64 string if data url prefix is already present
    if image_base64.startswith("data:image/"):
        image_url = image_base64
    else:
        image_url = f"data:image/{image_format};base64,{image_base64}"

    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }


def parse_vision_response(
    raw_response: Union[str, Dict[str, Any]],
    allowed_labels: Optional[List[str]] = None,
    strict: bool = True,
) -> DetectionResult:
    """
    Parse and validate the raw model output into a DetectionResult.
    Handles raw JSON, markdown-wrapped JSON (```json ... ```), and repairs minor format glitches.
    """
    text = ""
    if isinstance(raw_response, dict):
        # OpenAI chat completion structure
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            choice = raw_response["choices"][0]
            message = choice.get("message", {})
            text = message.get("content", "")
        elif "objects" in raw_response:
            data = raw_response
        else:
            text = json.dumps(raw_response)
    else:
        text = str(raw_response)

    if not isinstance(raw_response, dict) or "objects" not in raw_response:
        # Strip markdown fences if present
        cleaned = text.strip()
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if markdown_match:
            cleaned = markdown_match.group(1).strip()
        else:
            # Look for outermost curly braces
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        # Clean trailing commas before closing braces/brackets
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Fallback: attempt ast.literal_eval for single-quoted Python dict/list literals
            try:
                import ast
                evaluated = ast.literal_eval(cleaned)
                if isinstance(evaluated, dict):
                    data = evaluated
                else:
                    raise ValueError(f"Failed to parse vision model response as JSON: {e}. Raw: {text[:200]!r}")
            except Exception:
                raise ValueError(f"Failed to parse vision model response as JSON: {e}. Raw: {text[:200]!r}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object, got {type(data).__name__}")

    if "objects" not in data:
        raise ValueError("Missing 'objects' key in vision model response")

    raw_objects = data["objects"]
    if not isinstance(raw_objects, list):
        raise ValueError(f"Expected 'objects' to be a list, got {type(raw_objects).__name__}")

    allowed_set = set(allowed_labels) if allowed_labels is not None else None

    validated_objects: List[DetectedObject] = []
    for idx, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            if strict:
                raise ValueError(f"Object at index {idx} is not a dictionary: {item!r}")
            continue

        label = item.get("label")
        box_2d = item.get("box_2d")

        if not isinstance(label, str):
            if strict:
                raise ValueError(f"Object at index {idx} has missing or non-string label: {label!r}")
            continue

        # Check allowed labels
        if allowed_set is not None and label not in allowed_set:
            if strict:
                raise ValueError(f"Object at index {idx} has label {label!r} not in allowed list: {allowed_set}")
            continue

        # Check and normalize box_2d
        if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
            if strict:
                raise ValueError(f"Object at index {idx} has invalid box_2d: {box_2d!r}")
            continue

        try:
            ymin = int(round(float(box_2d[0])))
            xmin = int(round(float(box_2d[1])))
            ymax = int(round(float(box_2d[2])))
            xmax = int(round(float(box_2d[3])))
        except (ValueError, TypeError) as e:
            if strict:
                raise ValueError(f"Object at index {idx} coordinates cannot be converted to int: {e}")
            continue

        # In case coordinates were reversed (ymin > ymax or xmin > xmax)
        if ymin > ymax:
            if strict:
                raise ValueError(f"Object at index {idx} has inverted y coordinates: ymin={ymin}, ymax={ymax}")
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            if strict:
                raise ValueError(f"Object at index {idx} has inverted x coordinates: xmin={xmin}, xmax={xmax}")
            xmin, xmax = xmax, xmin

        # Degenerate boxes (width or height == 0)
        if ymin == ymax or xmin == xmax:
            if strict:
                raise ValueError(f"Object at index {idx} has zero area: {[ymin, xmin, ymax, xmax]}")
            continue

        # Check range bounds [0, 1000]
        if not (0 <= ymin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmin <= 1000 and 0 <= xmax <= 1000):
            if strict:
                raise ValueError(f"Object at index {idx} coordinates out of bounds [0, 1000]: {[ymin, xmin, ymax, xmax]}")

        # Clamp bounds
        ymin = max(0, min(1000, ymin))
        xmin = max(0, min(1000, xmin))
        ymax = max(0, min(1000, ymax))
        xmax = max(0, min(1000, xmax))

        # Parse confidence if present
        raw_conf = item.get("confidence")
        conf_val = None
        if raw_conf is not None:
            try:
                if isinstance(raw_conf, str):
                    raw_conf = raw_conf.strip().rstrip("%")
                num_conf = float(raw_conf)
                if 1.0 < num_conf <= 100.0:
                    num_conf = num_conf / 100.0
                conf_val = round(max(0.0, min(1.0, num_conf)), 3)
            except (ValueError, TypeError):
                if strict:
                    raise ValueError(f"Object at index {idx} has invalid confidence: {raw_conf!r}")
                conf_val = None

        obj = DetectedObject(label=label, box_2d=[ymin, xmin, ymax, xmax], confidence=conf_val)
        validated_objects.append(obj)

    return DetectionResult(objects=validated_objects)

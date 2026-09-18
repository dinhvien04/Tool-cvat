"""Correction Learning & Human-in-the-Loop Feedback Engine for Tool-cvat.

This module provides:
1. Baseline prediction persistence (keyed by SHA-256 image fingerprint).
2. Correction Diff Engine: bipartite matching reconciling AI baseline predictions
   against final human annotations to detect RELABEL, BOX_MOVE, BOX_RESIZE, MASK_EDIT,
   REGION_EDIT, LANE_EDIT, DELETE_FALSE_POSITIVE, ADD_MISSING, and NO_CHANGE.
3. Local SQLite database (.tool-cvat/feedback.sqlite3) tracking all annotations,
   corrections, derived rules, and per-label statistics across all 31 labels.
4. Automatic rule derivation (minimum sample threshold >= 3).
5. Crop generation and bounded storage pruning.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from PIL import Image, ImageChops, ImageDraw

from core.geometry import calculate_box_iou, cvat_mask_to_binary_image, extract_shape_bbox
from core.line_geometry import (
    DEFAULT_LANE_COMPARE_WIDTH_PX,
    DEFAULT_LANE_MAX_MATCH_DIST_PX,
    DEFAULT_LANE_TOLERANCE_PX,
    compute_polyline_distance,
    extract_polyline_points,
)
from core.taxonomy import (
    GROUP_INSTANCE,
    GROUP_LANE,
    GROUP_REGION,
    MASTER_31_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    Taxonomy,
)

ALL_31_LABELS = list(MASTER_31_LABELS)

# Configurable Lane Comparison Constants
LANE_COMPARE_WIDTH_PX: int = DEFAULT_LANE_COMPARE_WIDTH_PX
LANE_TOLERANCE_PX: float = DEFAULT_LANE_TOLERANCE_PX
LANE_MAX_MATCH_DIST_PX: float = DEFAULT_LANE_MAX_MATCH_DIST_PX

# Correction Classification Types
CORRECTION_RELABEL = "RELABEL"
CORRECTION_BOX_MOVE = "BOX_MOVE"
CORRECTION_BOX_RESIZE = "BOX_RESIZE"
CORRECTION_MASK_EDIT = "MASK_EDIT"
CORRECTION_REGION_EDIT = "REGION_EDIT"
CORRECTION_LANE_EDIT = "LANE_EDIT"
CORRECTION_DELETE_FALSE_POSITIVE = "DELETE_FALSE_POSITIVE"
CORRECTION_ADD_MISSING = "ADD_MISSING"
CORRECTION_NO_CHANGE = "NO_CHANGE"

ALL_CORRECTION_TYPES = [
    CORRECTION_RELABEL,
    CORRECTION_BOX_MOVE,
    CORRECTION_BOX_RESIZE,
    CORRECTION_MASK_EDIT,
    CORRECTION_REGION_EDIT,
    CORRECTION_LANE_EDIT,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_ADD_MISSING,
    CORRECTION_NO_CHANGE,
]


def compute_raw_file_hash(source: Union[bytes, bytearray, str, Path]) -> str:
    """Compute SHA-256 hex digest of raw encoded file bytes for diagnostics."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {source}")
        return hashlib.sha256(p.read_bytes()).hexdigest()
    if isinstance(source, (bytes, bytearray)):
        return hashlib.sha256(source).hexdigest()
    raise TypeError(f"Unsupported source type for raw file hash: {type(source).__name__}")


def compute_image_hash(image_source: Union[bytes, bytearray, str, Path, Image.Image]) -> str:
    """Compute ONE canonical visual fingerprint of an image.

    Decodes any decodable image representation (JPEG bytes, PNG bytes, PIL Image,
    local file, CVAT downloaded frame) to uncompressed RGB pixel bytes, prepends
    width/height header, and returns a deterministic SHA-256 hexadecimal digest.

    Guarantees:
    - Same pixel content produces the exact same fingerprint whether supplied as:
      * JPEG bytes
      * PNG bytes
      * PIL Image
      * CVAT downloaded frame
      * Locally loaded image path
    - Does NOT hash encoded container bytes as the primary visual identity.
    - Gracefully falls back to raw byte hash if non-image binary data is passed.

    Args:
        image_source: Raw image bytes, filesystem path, or PIL.Image.Image.

    Returns:
        64-character lowercase SHA-256 hex string.
    """
    if isinstance(image_source, Image.Image):
        im = image_source.convert("RGB")
    elif isinstance(image_source, (bytes, bytearray)):
        try:
            im = Image.open(io.BytesIO(image_source)).convert("RGB")
        except Exception:
            # Fallback for arbitrary non-image binary data (e.g. mock test buffers)
            return hashlib.sha256(image_source).hexdigest()
    elif isinstance(image_source, (str, Path)):
        p = Path(image_source)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_source}")
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            return hashlib.sha256(p.read_bytes()).hexdigest()
    else:
        raise TypeError(f"Unsupported image source type for hashing: {type(image_source).__name__}")

    w, h = im.size
    header = f"{w}x{h}_".encode("ascii")
    pixel_bytes = im.tobytes()
    return hashlib.sha256(header + pixel_bytes).hexdigest()


# Alias for explicit visual identity contract
compute_visual_fingerprint = compute_image_hash


def compute_perceptual_hash(image_source: Union[bytes, bytearray, str, Path, Image.Image]) -> str:
    """Compute 64-bit perceptual difference hash (dHash) using pure Pillow.

    Secondary image identity fallback for lossily re-encoded or compressed frames.
    Returns 16-character lowercase hexadecimal hash.
    """
    if isinstance(image_source, Image.Image):
        im = image_source.convert("RGB")
    elif isinstance(image_source, (bytes, bytearray)):
        try:
            im = Image.open(io.BytesIO(image_source)).convert("RGB")
        except Exception:
            return ""
    elif isinstance(image_source, (str, Path)):
        p = Path(image_source)
        if not p.exists() or not p.is_file():
            return ""
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            return ""
    else:
        return ""

    gray = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    raw = gray.tobytes()
    diff = []
    for r in range(8):
        row_offset = r * 9
        for c in range(8):
            diff.append(1 if raw[row_offset + c] > raw[row_offset + c + 1] else 0)

    hash_int = 0
    for bit in diff:
        hash_int = (hash_int << 1) | bit
    return f"{hash_int:016x}"


def hamming_distance(hex1: str, hex2: str) -> int:
    """Compute Hamming distance between two 16-character hex hash strings."""
    if not hex1 or not hex2 or len(hex1) != 16 or len(hex2) != 16:
        return 64
    try:
        v1 = int(hex1, 16)
        v2 = int(hex2, 16)
        return bin(v1 ^ v2).count("1")
    except ValueError:
        return 64


def _rasterize_shape(
    shape: Dict[str, Any],
    width: int,
    height: int,
    line_width: int = 4,
    fill_value: int = 1,
) -> Optional[Image.Image]:
    """Render a CVAT shape (mask, polygon, polyline, or rectangle) to a binary PIL image."""
    if width <= 0 or height <= 0:
        return None

    # 1. Native CVAT mask with flat list
    if "mask" in shape and isinstance(shape["mask"], (list, tuple)) and len(shape["mask"]) >= 5:
        try:
            mask_img = cvat_mask_to_binary_image(shape["mask"], width=width, height=height)
            if fill_value != 1:
                mask_img = mask_img.point(lambda p: fill_value if p else 0)
            return mask_img
        except Exception:
            pass

    stype = shape.get("type")
    pts = shape.get("points") or []

    # 2. Polyline / line geometry (strokes rasterized with line_width)
    if stype in ("polyline", "line") and len(pts) >= 4:
        try:
            pts_tuples = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts) - 1, 2)]
            img = Image.new("L", (width, height), 0)
            ImageDraw.Draw(img).line(pts_tuples, fill=fill_value, width=line_width)
            return img
        except Exception:
            pass

    # 3. Polygon geometry
    if (stype in ("polygon", "mask") or len(pts) >= 6) and len(pts) >= 6:
        try:
            pts_tuples = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts) - 1, 2)]
            if len(pts_tuples) >= 3:
                img = Image.new("L", (width, height), 0)
                ImageDraw.Draw(img).polygon(pts_tuples, fill=fill_value)
                return img
        except Exception:
            pass

    # 4. Rectangle
    bbox = extract_shape_bbox(shape)
    if bbox is not None:
        try:
            img = Image.new("L", (width, height), 0)
            bx1, by1, bx2, by2 = [int(round(b)) for b in bbox]
            ImageDraw.Draw(img).rectangle([bx1, by1, bx2, by2], fill=fill_value)
            return img
        except Exception:
            pass

    return None


def _compute_shape_iou(
    shape_a: Dict[str, Any],
    shape_b: Dict[str, Any],
    width: int,
    height: int,
    line_width: int = 4,
) -> float:
    """Compute true raster IoU (Intersection over Union) using pure Pillow geometry.

    Renders masks, polygons, polylines, or rectangles onto binary canvases and computes
    pixel overlap via Pillow ImageChops and histogram. Falls back to bounding-box IoU
    only if rasterization fails.
    """
    bbox_a = extract_shape_bbox(shape_a)
    bbox_b = extract_shape_bbox(shape_b)
    if bbox_a is None or bbox_b is None:
        return 0.0

    mask_a = _rasterize_shape(shape_a, width, height, line_width=line_width, fill_value=1)
    mask_b = _rasterize_shape(shape_b, width, height, line_width=line_width, fill_value=2)
    if mask_a is None or mask_b is None:
        return calculate_box_iou(bbox_a, bbox_b)

    pad = max(int(line_width * 2), 4)
    ux1 = max(0, int(math.floor(min(bbox_a[0], bbox_b[0]))) - pad)
    uy1 = max(0, int(math.floor(min(bbox_a[1], bbox_b[1]))) - pad)
    ux2 = min(width - 1, int(math.ceil(max(bbox_a[2], bbox_b[2]))) + pad)
    uy2 = min(height - 1, int(math.ceil(max(bbox_a[3], bbox_b[3]))) + pad)

    if ux1 <= ux2 and uy1 <= uy2:
        crop_a = mask_a.crop((ux1, uy1, ux2 + 1, uy2 + 1))
        crop_b = mask_b.crop((ux1, uy1, ux2 + 1, uy2 + 1))
    else:
        crop_a = mask_a
        crop_b = mask_b

    combo = ImageChops.add(crop_a, crop_b)
    hist = combo.histogram()
    inter = hist[3]
    union = hist[1] + hist[2] + hist[3]
    return float(inter / union) if union > 0 else 0.0


@dataclass
class CorrectionDiffItem:
    """Individual shape-level difference between AI prediction and human annotation."""
    correction_type: str
    ai_label: Optional[str] = None
    human_label: Optional[str] = None
    ai_shape: Optional[Dict[str, Any]] = None
    human_shape: Optional[Dict[str, Any]] = None
    iou: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


LANE_COMPARE_WIDTH_PX: int = 5


class CorrectionDiffEngine:
    """Bipartite matching engine comparing AI baseline shapes vs human shapes.

    Distinguishes:
    - Cross-label shifts (IoU >= min_iou_relabel) -> RELABEL
    - Positional translation -> BOX_MOVE
    - Dimension adjustment -> BOX_RESIZE
    - Polygon/mask contour changes -> MASK_EDIT / REGION_EDIT
    - Lane line modifications -> LANE_EDIT
    - Unmatched AI prediction -> DELETE_FALSE_POSITIVE
    - Unmatched human shape -> ADD_MISSING
    - Identical or near-identical shape -> NO_CHANGE (Accepted)
    """

    def __init__(
        self,
        min_iou_relabel: float = 0.60,
        min_iou_match: float = 0.30,
        taxonomy: Optional[Taxonomy] = None,
        lane_compare_width_px: int = LANE_COMPARE_WIDTH_PX,
        lane_tolerance_px: float = LANE_TOLERANCE_PX,
        lane_match_max_dist_px: float = LANE_MAX_MATCH_DIST_PX,
    ):
        self.min_iou_relabel = float(min_iou_relabel)
        self.min_iou_match = float(min_iou_match)
        self.taxonomy = taxonomy or Taxonomy()
        self.lane_compare_width_px = int(lane_compare_width_px)
        self.lane_tolerance_px = float(lane_tolerance_px)
        self.lane_match_max_dist_px = float(lane_match_max_dist_px)

    def _group_shapes(self, shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group paired CVAT shapes sharing the same group_id into composite semantic annotations.

        In 31-label mode:
        - Policy A: Rectangle + Mask share group_id -> 1 composite annotation.
        - Policy B: Polygon + Mask share group_id -> 1 composite annotation.
        - Policy C: Polyline has no group_id -> 1 annotation.
        - Un-grouped shapes (e.g. from legacy or custom tasks) remain individual annotations.
        """
        grouped: Dict[Tuple[Any, str], List[Dict[str, Any]]] = {}
        standalone: List[Dict[str, Any]] = []

        for s in shapes:
            gid = s.get("group_id")
            lbl = s.get("label", "")
            if gid is not None:
                key = (gid, lbl)
                grouped.setdefault(key, []).append(s)
            else:
                standalone.append(s)

        annotations: List[Dict[str, Any]] = []

        # Process grouped composite shapes
        for (gid, lbl), sub_shapes in grouped.items():
            # Pick primary shape: prefer rectangle or polygon for clear geometry, then mask
            rect_shape = next((s for s in sub_shapes if s.get("type") == "rectangle"), None)
            poly_shape = next((s for s in sub_shapes if s.get("type") == "polygon"), None)
            mask_shape = next((s for s in sub_shapes if s.get("type") == "mask"), None)
            polyline_shape = next((s for s in sub_shapes if s.get("type") == "polyline"), None)

            primary = rect_shape or poly_shape or mask_shape or polyline_shape or sub_shapes[0]

            # Compute union bbox across all sub-shapes
            bboxes = [extract_shape_bbox(s) for s in sub_shapes]
            valid_bboxes = [b for b in bboxes if b is not None]
            if valid_bboxes:
                union_bbox = [
                    min(b[0] for b in valid_bboxes),
                    min(b[1] for b in valid_bboxes),
                    max(b[2] for b in valid_bboxes),
                    max(b[3] for b in valid_bboxes),
                ]
            else:
                union_bbox = None

            annotations.append({
                "group_id": gid,
                "label": lbl,
                "shape": primary,
                "sub_shapes": sub_shapes,
                "rect_shape": rect_shape,
                "poly_shape": poly_shape,
                "mask_shape": mask_shape,
                "polyline_shape": polyline_shape,
                "bbox": union_bbox,
                "type": primary.get("type", "rectangle"),
                "group": self.taxonomy.get_group(lbl) if self.taxonomy.is_valid_label(lbl) else None,
                "policy": self.taxonomy.get_policy(lbl) if self.taxonomy.is_valid_label(lbl) else None,
            })

        # Process standalone shapes
        for s in standalone:
            lbl = s.get("label", "")
            bbox = extract_shape_bbox(s)
            stype = s.get("type", "rectangle")
            annotations.append({
                "group_id": None,
                "label": lbl,
                "shape": s,
                "sub_shapes": [s],
                "rect_shape": s if stype == "rectangle" else None,
                "poly_shape": s if stype == "polygon" else None,
                "mask_shape": s if stype == "mask" else None,
                "polyline_shape": s if stype == "polyline" else None,
                "bbox": bbox,
                "type": stype,
                "group": self.taxonomy.get_group(lbl) if self.taxonomy.is_valid_label(lbl) else None,
                "policy": self.taxonomy.get_policy(lbl) if self.taxonomy.is_valid_label(lbl) else None,
            })

        return annotations

    def _populate_shape_details(
        self,
        entry: Optional[Dict[str, Any]],
        prefix: str,
        details: Dict[str, Any],
    ) -> None:
        """Populate details dictionary with paired sub-shapes for an annotation entry."""
        if not entry:
            return
        if entry.get("rect_shape"):
            details[f"{prefix}_rect"] = entry["rect_shape"]
        if entry.get("mask_shape"):
            details[f"{prefix}_mask"] = entry["mask_shape"]
        if entry.get("poly_shape"):
            details[f"{prefix}_poly"] = entry["poly_shape"]
        if entry.get("polyline_shape"):
            details[f"{prefix}_polyline"] = entry["polyline_shape"]
        if entry.get("sub_shapes"):
            details[f"{prefix}_sub_shapes"] = entry["sub_shapes"]

    def _compute_candidate_similarity(
        self,
        a: Dict[str, Any],
        h: Dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute policy-specific candidate similarity and diagnostic metrics.

        Policy-Specific Candidate Matching:
        - POLICY A (Instance / box_mask): uses bounding-box IoU prefilter, returning box IoU.
        - POLICY B (Region / polygon_mask): prioritizes true rasterized shape IoU (polygon/mask overlap).
        - POLICY C (Lane / polyline): uses polyline-aware similarity (symmetric average distance
          and rasterized stroke IoU with configurable lane_compare_width_px), NOT high bbox IoU.
        """
        policy_a = a.get("policy")
        policy_h = h.get("policy")
        group_a = a.get("group")
        group_h = h.get("group")
        lbl_a = a.get("label", "")
        lbl_h = h.get("label", "")

        is_lane_a = (
            policy_a == POLICY_POLYLINE
            or group_a == GROUP_LANE
            or lbl_a.startswith("lane/")
            or a.get("type") == "polyline"
        )
        is_lane_h = (
            policy_h == POLICY_POLYLINE
            or group_h == GROUP_LANE
            or lbl_h.startswith("lane/")
            or h.get("type") == "polyline"
        )

        is_region_a = (
            policy_a == POLICY_POLYGON_MASK
            or group_a == GROUP_REGION
            or "area/" in lbl_a
            or a.get("type") in ("polygon", "mask")
        )
        is_region_h = (
            policy_h == POLICY_POLYGON_MASK
            or group_h == GROUP_REGION
            or "area/" in lbl_h
            or h.get("type") in ("polygon", "mask")
        )

        # ----------------------------------------------------------------------
        # 1. POLICY C: Polyline Lane Matching (distance & stroke aware)
        # ----------------------------------------------------------------------
        if is_lane_a and is_lane_h:
            a_line = a.get("polyline_shape") or a["shape"]
            h_line = h.get("polyline_shape") or h["shape"]
            pts_a = extract_polyline_points(a_line)
            pts_h = extract_polyline_points(h_line)
            if len(pts_a) < 2 or len(pts_h) < 2:
                return 0.0, {}

            # Symmetric average distance in pixels (pure Python, O(N))
            avg_dist = compute_polyline_distance(pts_a, pts_h)
            # Dilated line stroke IoU with configurable stroke width
            stroke_iou = _compute_shape_iou(
                a_line,
                h_line,
                width=image_width,
                height=image_height,
                line_width=self.lane_compare_width_px,
            )
            bbox_iou = calculate_box_iou(a["bbox"], h["bbox"]) if (a.get("bbox") and h.get("bbox")) else 0.0

            metrics = {
                "avg_dist": avg_dist,
                "stroke_iou": stroke_iou,
                "bbox_iou": bbox_iou,
                "policy": POLICY_POLYLINE,
            }

            # Distance-based normalized similarity
            dist_sim = max(0.0, 1.0 - avg_dist / self.lane_match_max_dist_px)
            if avg_dist <= self.lane_match_max_dist_px:
                sim = max(dist_sim, stroke_iou)
            else:
                sim = stroke_iou

            # Special case: crossing / realigned lines sharing same corridor
            if bbox_iou >= 0.50 and stroke_iou > 0.0:
                sim = max(sim, bbox_iou * 0.5)

            return sim, metrics

        # Cross-policy isolation: polyline cannot match box or region
        if is_lane_a != is_lane_h:
            return 0.0, {}

        # ----------------------------------------------------------------------
        # 2. POLICY B: Semantic Region Matching (true rasterized shape IoU)
        # ----------------------------------------------------------------------
        if is_region_a and is_region_h:
            bbox_a = a.get("bbox")
            bbox_h = h.get("bbox")
            # Coarse bbox prefilter: zero intersection implies zero raster IoU
            if bbox_a and bbox_h:
                if (
                    bbox_a[2] < bbox_h[0]
                    or bbox_h[2] < bbox_a[0]
                    or bbox_a[3] < bbox_h[1]
                    or bbox_h[3] < bbox_a[1]
                ):
                    return 0.0, {}

            a_region = a.get("poly_shape") or a.get("mask_shape") or a["shape"]
            h_region = h.get("poly_shape") or h.get("mask_shape") or h["shape"]
            shape_iou = _compute_shape_iou(
                a_region,
                h_region,
                width=image_width,
                height=image_height,
            )
            bbox_iou = calculate_box_iou(bbox_a, bbox_h) if (bbox_a and bbox_h) else 0.0

            metrics = {
                "shape_iou": shape_iou,
                "bbox_iou": bbox_iou,
                "policy": POLICY_POLYGON_MASK,
            }
            return shape_iou, metrics

        # Cross-policy isolation: region cannot match box
        if is_region_a != is_region_h:
            return 0.0, {}

        # ----------------------------------------------------------------------
        # 3. POLICY A: Instance Matching (bounding-box IoU prefilter)
        # ----------------------------------------------------------------------
        bbox_a = a.get("bbox")
        bbox_h = h.get("bbox")
        if not bbox_a or not bbox_h:
            return 0.0, {}

        if (
            bbox_a[2] < bbox_h[0]
            or bbox_h[2] < bbox_a[0]
            or bbox_a[3] < bbox_h[1]
            or bbox_h[3] < bbox_a[1]
        ):
            return 0.0, {}

        box_iou = calculate_box_iou(bbox_a, bbox_h)
        metrics = {
            "box_iou": box_iou,
            "policy": POLICY_BOX_MASK,
        }
        return box_iou, metrics

    def _pair_annotations(
        self,
        ai_entries: List[Dict[str, Any]],
        human_entries: List[Dict[str, Any]],
        image_width: int = 1280,
        image_height: int = 720,
    ) -> Tuple[List[CorrectionDiffItem], Set[int], Set[int]]:
        """Match AI predicted annotations against human annotations using policy-specific metrics.

        Enforces:
        - Policy A: bbox IoU prefilter, box + mask comparison
        - Policy B: true rasterized shape IoU (polygon/mask overlap)
        - Policy C: polyline-aware similarity (symmetric distance and line stroke IoU)

        Returns:
            (results, matched_ai_indices, matched_human_indices)
        """
        results: List[CorrectionDiffItem] = []
        matched_ai: Set[int] = set()
        matched_human: Set[int] = set()

        candidates = []
        for a in ai_entries:
            for h in human_entries:
                sim, metrics = self._compute_candidate_similarity(a, h, image_width, image_height)
                if sim > 0.0:
                    candidates.append((sim, a["idx"], h["idx"], metrics))

        # Sort candidates descending by similarity score
        candidates.sort(key=lambda x: x[0], reverse=True)

        # ----------------------------------------------------------------------
        # Pass 1: Same-label candidate matching
        # ----------------------------------------------------------------------
        for sim, a_idx, h_idx, metrics in candidates:
            if a_idx in matched_ai or h_idx in matched_human:
                continue

            a = ai_entries[a_idx]
            h = human_entries[h_idx]
            if a["label"] == h["label"]:
                policy = metrics.get("policy")
                matched = False
                if policy == POLICY_POLYLINE:
                    avg_dist = metrics.get("avg_dist", 999.0)
                    stroke_iou = metrics.get("stroke_iou", 0.0)
                    bbox_iou = metrics.get("bbox_iou", 0.0)
                    if (
                        avg_dist <= self.lane_match_max_dist_px
                        or sim >= self.min_iou_match
                        or stroke_iou > 0.05
                        or (bbox_iou >= 0.5 and stroke_iou > 0.0)
                    ):
                        matched = True
                elif policy == POLICY_POLYGON_MASK:
                    shape_iou = metrics.get("shape_iou", 0.0)
                    if shape_iou >= self.min_iou_match:
                        matched = True
                else:
                    box_iou = metrics.get("box_iou", 0.0)
                    if box_iou >= self.min_iou_match:
                        matched = True

                if matched:
                    item = self._classify_same_label(
                        a, h, sim, image_width, image_height, metrics=metrics
                    )
                    results.append(item)
                    matched_ai.add(a_idx)
                    matched_human.add(h_idx)

        # ----------------------------------------------------------------------
        # Pass 2: Cross-label relabel matching
        # ----------------------------------------------------------------------
        for sim, a_idx, h_idx, metrics in candidates:
            if a_idx in matched_ai or h_idx in matched_human:
                continue

            a = ai_entries[a_idx]
            h = human_entries[h_idx]
            if a["label"] != h["label"]:
                ai_lbl = a["label"]
                human_lbl = h["label"]
                policy = metrics.get("policy")
                is_relabel = False
                relabel_iou = sim
                reason = ""

                if policy == POLICY_POLYLINE:
                    # Policy C relabel: line stroke IoU or near-zero shift
                    stroke_iou = metrics.get("stroke_iou", 0.0)
                    avg_dist = metrics.get("avg_dist", 999.0)
                    if stroke_iou >= self.min_iou_relabel or (
                        avg_dist <= self.lane_tolerance_px and sim >= self.min_iou_relabel
                    ):
                        is_relabel = True
                        relabel_iou = stroke_iou if stroke_iou > 0 else sim
                        reason = f"Annotator relabeled '{ai_lbl}' to '{human_lbl}' (lane stroke IoU {relabel_iou:.2f})"
                elif policy == POLICY_POLYGON_MASK:
                    # Policy B relabel: true rasterized shape IoU
                    shape_iou = metrics.get("shape_iou", 0.0)
                    if shape_iou >= self.min_iou_relabel:
                        is_relabel = True
                        relabel_iou = shape_iou
                        reason = f"Annotator relabeled '{ai_lbl}' to '{human_lbl}' (shape IoU {shape_iou:.2f})"
                else:
                    # Policy A relabel: bounding-box IoU
                    box_iou = metrics.get("box_iou", 0.0)
                    if box_iou >= self.min_iou_relabel:
                        is_relabel = True
                        relabel_iou = box_iou
                        reason = f"Annotator relabeled '{ai_lbl}' to '{human_lbl}' (IoU {box_iou:.2f})"

                if is_relabel:
                    relabel_details = {
                        "reason": reason,
                        "ai_bbox": a["bbox"],
                        "human_bbox": h["bbox"],
                        "ai_group_id": a.get("group_id"),
                        "human_group_id": h.get("group_id"),
                    }
                    if policy == POLICY_POLYLINE:
                        relabel_details["lane_iou"] = round(relabel_iou, 4)
                        relabel_details["stroke_iou"] = round(relabel_iou, 4)
                        relabel_details["lane_shift_px"] = round(metrics.get("avg_dist", 0.0), 2)
                    elif policy == POLICY_POLYGON_MASK:
                        relabel_details["shape_iou"] = round(relabel_iou, 4)
                        relabel_details["mask_iou"] = round(relabel_iou, 4)
                        relabel_details["polygon_mask_iou"] = round(relabel_iou, 4)
                    else:
                        relabel_details["box_iou"] = round(relabel_iou, 4)

                    self._populate_shape_details(h, "human", relabel_details)
                    self._populate_shape_details(a, "ai", relabel_details)
                    item = CorrectionDiffItem(
                        correction_type=CORRECTION_RELABEL,
                        ai_label=ai_lbl,
                        human_label=human_lbl,
                        ai_shape=a["shape"],
                        human_shape=h["shape"],
                        iou=round(relabel_iou, 4),
                        details=relabel_details,
                    )
                    results.append(item)
                    matched_ai.add(a_idx)
                    matched_human.add(h_idx)

        return results, matched_ai, matched_human

    # Alias for audit and backward compatibility
    _match_shapes = _pair_annotations

    def diff(
        self,
        ai_shapes: List[Dict[str, Any]],
        human_shapes: List[Dict[str, Any]],
        image_width: int = 1280,
        image_height: int = 720,
    ) -> List[CorrectionDiffItem]:
        """Compute bipartite matching diff between AI predictions and human annotations.

        Treats paired shapes sharing the same group_id (e.g. box+mask or polygon+mask)
        as single semantic annotations to prevent duplicate or mismatched diff events.

        Args:
            ai_shapes: List of AI predicted shapes (e.g. from AnnotationResult or CVAT API).
            human_shapes: List of final human edited shapes from CVAT job.
            image_width: Width of image in pixels.
            image_height: Height of image in pixels.

        Returns:
            List of CorrectionDiffItem entries classifying every shape difference.
        """
        results: List[CorrectionDiffItem] = []

        # 1. Group paired shapes (sharing group_id) into single semantic annotations
        ai_entries = self._group_shapes(ai_shapes)
        human_entries = self._group_shapes(human_shapes)

        for idx, a in enumerate(ai_entries):
            a["idx"] = idx
        for idx, h in enumerate(human_entries):
            h["idx"] = idx

        # 2. Match annotations using policy-specific candidate pairing
        results, matched_ai, matched_human = self._pair_annotations(
            ai_entries, human_entries, image_width=image_width, image_height=image_height
        )

        # 3. Remaining unmatched AI annotations -> False Positives (Annotator deleted AI prediction)
        for a in ai_entries:
            if a["idx"] not in matched_ai:
                fp_details = {
                    "reason": f"AI predicted '{a['label']}' which was deleted or rejected by annotator",
                    "ai_bbox": a["bbox"],
                    "group_id": a.get("group_id"),
                    "sub_shapes_count": len(a.get("sub_shapes", [])),
                }
                self._populate_shape_details(a, "ai", fp_details)
                results.append(
                    CorrectionDiffItem(
                        correction_type=CORRECTION_DELETE_FALSE_POSITIVE,
                        ai_label=a["label"],
                        human_label=None,
                        ai_shape=a["shape"],
                        human_shape=None,
                        iou=0.0,
                        details=fp_details,
                    )
                )

        # 4. Remaining unmatched Human annotations -> False Negatives (Annotator added missing object)
        for h in human_entries:
            if h["idx"] not in matched_human:
                fn_details = {
                    "reason": f"Annotator manually added missing '{h['label']}'",
                    "human_bbox": h["bbox"],
                    "group_id": h.get("group_id"),
                    "sub_shapes_count": len(h.get("sub_shapes", [])),
                }
                self._populate_shape_details(h, "human", fn_details)
                results.append(
                    CorrectionDiffItem(
                        correction_type=CORRECTION_ADD_MISSING,
                        ai_label=None,
                        human_label=h["label"],
                        ai_shape=None,
                        human_shape=h["shape"],
                        iou=0.0,
                        details=fn_details,
                    )
                )

        return results

    def _classify_same_label(
        self,
        a: Dict[str, Any],
        h: Dict[str, Any],
        iou: float,
        img_w: int,
        img_h: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> CorrectionDiffItem:
        """Classify shape difference when AI and human share the same label.

        Strictly enforces the three policies:
        - POLICY_BOX_MASK: Compares BOTH rectangle geometry (box_iou, shift, resize)
          and mask geometry (mask_iou). Detects:
            * box unchanged + mask changed -> MASK_EDIT
            * box changed + mask unchanged -> BOX_MOVE / BOX_RESIZE
            * box changed + mask changed -> structured details with primary type
            * box unchanged + mask unchanged -> NO_CHANGE
        - POLICY_POLYGON_MASK: Compares true shape overlap of polygon/mask.
          Stores polygon/mask IoU with bbox IoU as diagnostic.
        - POLICY_POLYLINE: Compares true line stroke overlap.
        """
        lbl = a["label"]
        group = a.get("group")
        policy = a.get("policy")
        a_bbox = a["bbox"]
        h_bbox = h["bbox"]

        details: Dict[str, Any] = {
            "ai_group_id": a.get("group_id"),
            "human_group_id": h.get("group_id"),
            "ai_bbox": a_bbox,
            "human_bbox": h_bbox,
        }
        self._populate_shape_details(h, "human", details)
        self._populate_shape_details(a, "ai", details)

        # ----------------------------------------------------------------------
        # 1. POLICY_BOX_MASK (14 Instance Labels): Compare BOTH box & mask
        # ----------------------------------------------------------------------
        if policy == POLICY_BOX_MASK or group == GROUP_INSTANCE:
            a_rect = a.get("rect_shape") or (a["shape"] if a["shape"].get("type") == "rectangle" else None)
            h_rect = h.get("rect_shape") or (h["shape"] if h["shape"].get("type") == "rectangle" else None)
            a_box = extract_shape_bbox(a_rect) if a_rect else a_bbox
            h_box = extract_shape_bbox(h_rect) if h_rect else h_bbox

            box_iou = calculate_box_iou(a_box, h_box) if (a_box and h_box) else iou
            box_changed = False
            box_moved = False
            box_resized = False
            box_change_type: Optional[str] = None
            shift_x = 0.0
            shift_y = 0.0
            dw = 0.0
            dh = 0.0

            if a_box and h_box:
                ax1, ay1, ax2, ay2 = a_box
                hx1, hy1, hx2, hy2 = h_box
                aw, ah = ax2 - ax1, ay2 - ay1
                hw, hh = hx2 - hx1, hy2 - hy1
                acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
                hcx, hcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
                shift_x = abs(acx - hcx)
                shift_y = abs(acy - hcy)
                dw = abs(aw - hw) / max(aw, hw, 1.0)
                dh = abs(ah - hh) / max(ah, hh, 1.0)

                if dw > 0.15 or dh > 0.15:
                    box_resized = True
                if shift_x > 5.0 or shift_y > 5.0 or box_iou < 0.90:
                    box_moved = True

                if box_resized:
                    box_changed = True
                    box_change_type = CORRECTION_BOX_RESIZE
                elif box_moved:
                    box_changed = True
                    box_change_type = CORRECTION_BOX_MOVE
            elif (a_rect is not None) ^ (h_rect is not None):
                box_changed = True
                box_moved = True
                box_change_type = CORRECTION_BOX_MOVE
                box_iou = 0.0

            # Mask geometry comparison
            a_mask = a.get("mask_shape") or (a["shape"] if a["shape"].get("type") == "mask" else None)
            h_mask = h.get("mask_shape") or (h["shape"] if h["shape"].get("type") == "mask" else None)

            if a_mask and h_mask:
                mask_iou = _compute_shape_iou(a_mask, h_mask, img_w, img_h)
                mask_changed = (mask_iou < 0.92)
            elif a_mask is not None or h_mask is not None:
                mask_changed = True
                mask_iou = 0.0
            else:
                mask_changed = False
                mask_iou = box_iou

            details["box_iou"] = round(box_iou, 4)
            details["mask_iou"] = round(mask_iou, 4)
            details["shift"] = {"dx": round(shift_x, 2), "dy": round(shift_y, 2)}
            details["size_diff"] = {"dw": round(dw, 3), "dh": round(dh, 3)}
            details["box_changed"] = box_changed
            details["box_moved"] = box_moved
            details["box_resized"] = box_resized
            details["mask_changed"] = mask_changed
            if box_change_type:
                details["box_change_type"] = box_change_type

            if box_changed and not mask_changed:
                corr_type = box_change_type or CORRECTION_BOX_MOVE
                details["reason"] = f"Instance '{lbl}' bounding box adjusted ({corr_type})"
            elif not box_changed and mask_changed:
                corr_type = CORRECTION_MASK_EDIT
                details["reason"] = f"Instance '{lbl}' mask boundary adjusted by annotator (mask IoU {mask_iou:.2f})"
            elif box_changed and mask_changed:
                corr_type = box_change_type or CORRECTION_BOX_MOVE
                details["box_change_type"] = box_change_type
                details["reason"] = f"Instance '{lbl}' both box ({box_change_type}) and mask (mask IoU {mask_iou:.2f}) adjusted"
            else:
                corr_type = CORRECTION_NO_CHANGE
                details["status"] = "Instance accepted without edit"

            return CorrectionDiffItem(
                correction_type=corr_type,
                ai_label=lbl,
                human_label=lbl,
                ai_shape=a["shape"],
                human_shape=h["shape"],
                iou=round(min(box_iou, mask_iou), 4) if (box_changed or mask_changed) else round(box_iou, 4),
                details=details,
            )

        # ----------------------------------------------------------------------
        # 2. POLICY_POLYGON_MASK (10 Region Labels): Compare BOTH polygon & mask
        # ----------------------------------------------------------------------
        if policy == POLICY_POLYGON_MASK or group == GROUP_REGION or "area/" in lbl:
            a_poly = a.get("poly_shape") or (a["shape"] if a["shape"].get("type") == "polygon" else None)
            h_poly = h.get("poly_shape") or (h["shape"] if h["shape"].get("type") == "polygon" else None)
            a_mask = a.get("mask_shape") or (a["shape"] if a["shape"].get("type") == "mask" else None)
            h_mask = h.get("mask_shape") or (h["shape"] if h["shape"].get("type") == "mask" else None)

            # Fallback for standalone/unpaired shapes
            if not a_poly and not a_mask:
                if a["shape"].get("type") == "polygon" or len(a["shape"].get("points") or []) >= 6:
                    a_poly = a["shape"]
                else:
                    a_mask = a["shape"]
            if not h_poly and not h_mask:
                if h["shape"].get("type") == "polygon" or len(h["shape"].get("points") or []) >= 6:
                    h_poly = h["shape"]
                else:
                    h_mask = h["shape"]

            # Independent polygon similarity / shape IoU
            if a_poly and h_poly:
                poly_iou = _compute_shape_iou(a_poly, h_poly, img_w, img_h)
                polygon_changed = (poly_iou < 0.92)
            elif (a_poly is not None) ^ (h_poly is not None):
                polygon_changed = True
                poly_iou = 0.0
            else:
                polygon_changed = False
                poly_iou = 1.0

            # Independent mask IoU
            if a_mask and h_mask:
                mask_iou = _compute_shape_iou(a_mask, h_mask, img_w, img_h)
                mask_changed = (mask_iou < 0.92)
            elif (a_mask is not None) ^ (h_mask is not None):
                mask_changed = True
                mask_iou = 0.0
            else:
                mask_changed = False
                mask_iou = 1.0

            # Diagnostic BBox IoU
            bbox_iou = calculate_box_iou(a_bbox, h_bbox) if (a_bbox and h_bbox) else 0.0

            # Composite IoU across active geometries
            active_ious = []
            if a_poly and h_poly:
                active_ious.append(poly_iou)
            if a_mask and h_mask:
                active_ious.append(mask_iou)
            composite_iou = min(active_ious) if active_ious else min(poly_iou, mask_iou)

            details["polygon_changed"] = polygon_changed
            details["mask_changed"] = mask_changed
            details["polygon_iou"] = round(poly_iou, 4)
            details["shape_iou"] = round(poly_iou, 4)
            details["polygon_similarity"] = round(poly_iou, 4)
            details["mask_iou"] = round(mask_iou, 4)
            details["polygon_mask_iou"] = round(composite_iou, 4)
            details["bbox_iou_diagnostic"] = round(bbox_iou, 4)
            details["bbox_iou"] = round(bbox_iou, 4)

            # Expected behavior:
            # polygon unchanged, mask unchanged -> NO_CHANGE
            # polygon changed, mask unchanged -> REGION_EDIT
            # polygon unchanged, mask changed -> REGION_EDIT
            # both changed -> REGION_EDIT
            if not polygon_changed and not mask_changed:
                details["status"] = "Region accepted with negligible change"
                return CorrectionDiffItem(
                    correction_type=CORRECTION_NO_CHANGE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(composite_iou, 4),
                    details=details,
                )

            if polygon_changed and not mask_changed:
                details["reason"] = f"Semantic region '{lbl}' polygon contour edited by annotator (polygon IoU {poly_iou:.2f})"
            elif not polygon_changed and mask_changed:
                details["reason"] = f"Semantic region '{lbl}' mask boundary edited by annotator (mask IoU {mask_iou:.2f})"
            else:
                details["reason"] = f"Semantic region '{lbl}' both polygon (IoU {poly_iou:.2f}) and mask (IoU {mask_iou:.2f}) edited by annotator"

            return CorrectionDiffItem(
                correction_type=CORRECTION_REGION_EDIT,
                ai_label=lbl,
                human_label=lbl,
                ai_shape=a["shape"],
                human_shape=h["shape"],
                iou=round(composite_iou, 4),
                details=details,
            )

        # ----------------------------------------------------------------------
        # 3. POLICY_POLYLINE (7 Lane Labels): Compare true line stroke overlap
        # ----------------------------------------------------------------------
        if policy == POLICY_POLYLINE or group == GROUP_LANE or lbl.startswith("lane/"):
            a_line = a.get("polyline_shape") or a["shape"]
            h_line = h.get("polyline_shape") or h["shape"]
            pts_a = extract_polyline_points(a_line)
            pts_h = extract_polyline_points(h_line)

            if metrics and "avg_dist" in metrics:
                avg_dist = metrics["avg_dist"]
            elif pts_a and pts_h:
                avg_dist = compute_polyline_distance(pts_a, pts_h)
            else:
                avg_dist = 0.0

            if metrics and "stroke_iou" in metrics:
                true_iou = metrics["stroke_iou"]
            else:
                true_iou = _compute_shape_iou(
                    a_line, h_line, img_w, img_h, line_width=self.lane_compare_width_px
                )

            if metrics and "bbox_iou" in metrics:
                bbox_iou = metrics["bbox_iou"]
            else:
                bbox_iou = calculate_box_iou(a_bbox, h_bbox) if (a_bbox and h_bbox) else 0.0

            details["lane_iou"] = round(true_iou, 4)
            details["stroke_iou"] = round(true_iou, 4)
            details["lane_shift_px"] = round(avg_dist, 2)
            details["avg_dist"] = round(avg_dist, 2)
            details["bbox_iou_diagnostic"] = round(bbox_iou, 4)
            details["bbox_iou"] = round(bbox_iou, 4)

            # Tolerance check: negligible jitter / tiny change (<= lane_tolerance_px) or very high stroke IoU
            if avg_dist <= self.lane_tolerance_px or true_iou >= 0.90:
                details["status"] = "Lane marking accepted with negligible change"
                return CorrectionDiffItem(
                    correction_type=CORRECTION_NO_CHANGE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(true_iou, 4),
                    details=details,
                )
            details["reason"] = f"Lane marking '{lbl}' geometry edited by annotator (lane stroke IoU {true_iou:.2f}, avg shift {avg_dist:.1f}px)"
            return CorrectionDiffItem(
                correction_type=CORRECTION_LANE_EDIT,
                ai_label=lbl,
                human_label=lbl,
                ai_shape=a["shape"],
                human_shape=h["shape"],
                iou=round(true_iou, 4),
                details=details,
            )

        # Default fallback if matching
        return CorrectionDiffItem(
            correction_type=CORRECTION_NO_CHANGE,
            ai_label=lbl,
            human_label=lbl,
            ai_shape=a["shape"],
            human_shape=h["shape"],
            iou=round(iou, 4),
            details=details,
        )


class FeedbackDatabase:
    """Local SQLite repository managing correction memory, predictions, and rules."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        examples_dir: Optional[Union[str, Path]] = None,
        max_examples_total: int = 150,
        max_storage_mb: int = 50,
        max_crop_dimension: int = 512,
    ):
        env_data_dir = os.getenv("FEEDBACK_DATA_DIR")
        env_db_path = os.getenv("FEEDBACK_DB_PATH")

        if db_path is not None:
            resolved_db = Path(db_path)
        elif env_db_path:
            resolved_db = Path(env_db_path)
        elif env_data_dir:
            resolved_db = Path(env_data_dir) / "feedback.sqlite3"
        else:
            resolved_db = Path(".tool-cvat") / "feedback.sqlite3"

        if examples_dir is not None:
            resolved_examples = Path(examples_dir)
        elif env_data_dir:
            resolved_examples = Path(env_data_dir) / "examples"
        else:
            resolved_examples = resolved_db.parent / "examples"

        self.db_path = resolved_db
        self.examples_dir = resolved_examples
        self.max_examples_total = max_examples_total
        self.max_storage_mb = max_storage_mb
        self.max_crop_dimension = max_crop_dimension

        # Ensure parent directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def resolve_crop_path(self, crop_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve a stored relative or absolute crop path to an existing local file."""
        if not crop_path:
            return None
        p = Path(crop_path)
        if p.is_absolute() and p.exists() and p.is_file():
            return p
        # Check relative to examples_dir.parent (e.g. .tool-cvat/examples/crop.jpg)
        cand1 = self.examples_dir.parent / p
        if cand1.exists() and cand1.is_file():
            return cand1
        # Check relative to examples_dir directly (e.g. filename only)
        cand2 = self.examples_dir / p.name
        if cand2.exists() and cand2.is_file():
            return cand2
        return None

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with robust file-locking journal mode and row factory.

        Uses TRUNCATE journal mode and busy_timeout=30000. TRUNCATE is universally
        supported across Windows NTFS, Docker bind mounts (9P/VirtioFS), WSL2, and Linux,
        completely avoiding POSIX shared-memory (-shm) failures seen with WAL mode on bind mounts.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=TRUNCATE")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError as exc:
            logger.warning("Could not set SQLite pragmas (%s); proceeding with defaults", exc)
        return conn

    def _init_db(self) -> None:
        """Initialize database schema with proper indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Predictions baseline table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT NOT NULL,
                    perceptual_hash TEXT,
                    task_id INTEGER,
                    job_id INTEGER,
                    frame_index INTEGER,
                    model TEXT,
                    mode TEXT,
                    shapes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_hash ON predictions(image_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_job_frame ON predictions(job_id, frame_index)")

            # Auto-migrate existing predictions table if perceptual_hash column is missing
            try:
                col_info = cursor.execute("PRAGMA table_info(predictions)").fetchall()
                col_names = {row["name"] for row in col_info}
                if "perceptual_hash" not in col_names:
                    cursor.execute("ALTER TABLE predictions ADD COLUMN perceptual_hash TEXT")
            except Exception:
                pass

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_phash ON predictions(perceptual_hash)")

            # 2. Human annotations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS human_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT NOT NULL,
                    task_id INTEGER,
                    job_id INTEGER,
                    frame_index INTEGER,
                    shapes_json TEXT NOT NULL,
                    quality_tier TEXT DEFAULT 'completed',
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_human_hash ON human_annotations(image_hash)")

            # 3. Corrections diff table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT NOT NULL,
                    task_id INTEGER,
                    job_id INTEGER,
                    frame_index INTEGER,
                    correction_type TEXT NOT NULL,
                    ai_label TEXT,
                    human_label TEXT,
                    ai_shape_json TEXT,
                    human_shape_json TEXT,
                    iou REAL,
                    crop_path TEXT,
                    details_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corr_hash ON corrections(image_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corr_type ON corrections(correction_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corr_ai_lbl ON corrections(ai_label)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corr_human_lbl ON corrections(human_label)")

            # 4. Derived rules table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_label TEXT,
                    target_label TEXT,
                    rule_type TEXT NOT NULL,
                    rule_text TEXT NOT NULL UNIQUE,
                    sample_count INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    is_active INTEGER DEFAULT 1,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_labels ON rules(source_label, target_label)")

            # 5. Metadata / Settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.commit()

    # -------------------------------------------------------------------------
    # Baseline Prediction Persistence
    # -------------------------------------------------------------------------

    def save_prediction_baseline(
        self,
        image_hash: str,
        shapes: List[Dict[str, Any]],
        model: str,
        mode: str,
        task_id: Optional[int] = None,
        job_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        perceptual_hash: Optional[str] = None,
    ) -> int:
        """Store original AI prediction as baseline (sanitized, zero secrets/base64).

        Args:
            image_hash: SHA-256 fingerprint of the image.
            shapes: List of predicted CVAT shapes.
            model: Name of vision model used.
            mode: Pipeline mode ('full_31', 'box_and_mask', etc.).
            task_id: Optional CVAT task ID.
            job_id: Optional CVAT job ID.
            frame_index: Optional frame index.
            perceptual_hash: Optional 64-bit dHash string for secondary fuzzy matching.

        Returns:
            Inserted record row id.
        """
        # Sanitize shapes to avoid unbounded memory
        clean_shapes = []
        for s in shapes:
            item = {
                "type": s.get("type"),
                "label": s.get("label"),
                "confidence": s.get("confidence"),
            }
            if "points" in s:
                item["points"] = s["points"]
            if "mask" in s:
                # Store mask points or tight crop
                item["mask"] = s["mask"]
            if "group_id" in s:
                item["group_id"] = s["group_id"]
            clean_shapes.append(item)

        now_iso = datetime.now(timezone.utc).isoformat()
        shapes_str = json.dumps(clean_shapes)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions
                    (image_hash, perceptual_hash, task_id, job_id, frame_index, model, mode, shapes_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (image_hash, perceptual_hash, task_id, job_id, frame_index, model, mode, shapes_str, now_iso),
            )
            conn.commit()
            return cursor.lastrowid

    def get_prediction_baseline(
        self,
        image_hash: str,
        perceptual_hash: Optional[str] = None,
        job_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        max_hamming_distance: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve most recent AI prediction baseline with two-level matching strategy.

        Priority:
        1. Primary: Exact decoded-RGB SHA-256 fingerprint (image_hash).
        2. Secondary: Lightweight perceptual hash fallback (dHash with Hamming distance <= 5).
        3. Tertiary: Exact job_id + frame_index fallback.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Level 1: Exact decoded-RGB SHA-256 fingerprint
            cursor.execute(
                """
                SELECT id, image_hash, perceptual_hash, task_id, job_id, frame_index, model, mode, shapes_json, created_at
                FROM predictions
                WHERE image_hash = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (image_hash,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "image_hash": row["image_hash"],
                    "perceptual_hash": row["perceptual_hash"],
                    "task_id": row["task_id"],
                    "job_id": row["job_id"],
                    "frame_index": row["frame_index"],
                    "model": row["model"],
                    "mode": row["mode"],
                    "shapes": json.loads(row["shapes_json"]),
                    "created_at": row["created_at"],
                    "match_type": "exact_fingerprint",
                }

            # Level 2: Secondary perceptual difference hash fallback
            if perceptual_hash:
                cursor.execute(
                    """
                    SELECT id, image_hash, perceptual_hash, task_id, job_id, frame_index, model, mode, shapes_json, created_at
                    FROM predictions
                    WHERE perceptual_hash IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 100
                    """
                )
                candidates = cursor.fetchall()
                best_row = None
                min_dist = 65
                for cand in candidates:
                    c_phash = cand["perceptual_hash"]
                    if c_phash:
                        dist = hamming_distance(perceptual_hash, c_phash)
                        if dist <= max_hamming_distance and dist < min_dist:
                            min_dist = dist
                            best_row = cand
                if best_row:
                    return {
                        "id": best_row["id"],
                        "image_hash": best_row["image_hash"],
                        "perceptual_hash": best_row["perceptual_hash"],
                        "task_id": best_row["task_id"],
                        "job_id": best_row["job_id"],
                        "frame_index": best_row["frame_index"],
                        "model": best_row["model"],
                        "mode": best_row["mode"],
                        "shapes": json.loads(best_row["shapes_json"]),
                        "created_at": best_row["created_at"],
                        "match_type": "perceptual_hash",
                        "hamming_distance": min_dist,
                    }

            # Level 3: Tertiary job_id + frame_index fallback
            if job_id is not None and frame_index is not None:
                cursor.execute(
                    """
                    SELECT id, image_hash, perceptual_hash, task_id, job_id, frame_index, model, mode, shapes_json, created_at
                    FROM predictions
                    WHERE job_id = ? AND frame_index = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (job_id, frame_index),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "image_hash": row["image_hash"],
                        "perceptual_hash": row["perceptual_hash"],
                        "task_id": row["task_id"],
                        "job_id": row["job_id"],
                        "frame_index": row["frame_index"],
                        "model": row["model"],
                        "mode": row["mode"],
                        "shapes": json.loads(row["shapes_json"]),
                        "created_at": row["created_at"],
                        "match_type": "job_frame_fallback",
                    }

            return None

    # -------------------------------------------------------------------------
    # Human Annotations Persistence
    # -------------------------------------------------------------------------

    def save_human_annotations(
        self,
        image_hash: str,
        shapes: List[Dict[str, Any]],
        task_id: Optional[int] = None,
        job_id: Optional[int] = None,
        frame_index: Optional[int] = None,
        quality_tier: str = "completed",
    ) -> int:
        """Save human annotations imported from CVAT."""
        now_iso = datetime.now(timezone.utc).isoformat()
        shapes_str = json.dumps(shapes)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO human_annotations
                    (image_hash, task_id, job_id, frame_index, shapes_json, quality_tier, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (image_hash, task_id, job_id, frame_index, shapes_str, quality_tier, now_iso),
            )
            conn.commit()
            return cursor.lastrowid

    # -------------------------------------------------------------------------
    # Recording Corrections & Generating Crops
    # -------------------------------------------------------------------------

    def record_corrections(
        self,
        corrections: List[CorrectionDiffItem],
        image_hash: str,
        image: Optional[Image.Image] = None,
        task_id: Optional[int] = None,
        job_id: Optional[int] = None,
        frame_index: Optional[int] = None,
    ) -> List[int]:
        """Record correction items into database and generate visual crops when applicable.

        Args:
            corrections: List of CorrectionDiffItem instances.
            image_hash: SHA-256 fingerprint of the image.
            image: Optional PIL Image to extract small local crops from.
            task_id: Optional CVAT task ID.
            job_id: Optional CVAT job ID.
            frame_index: Optional frame index.

        Returns:
            List of created correction record IDs.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        created_ids: List[int] = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for item in corrections:
                crop_rel_path = None
                # Create visual crop for high-value correction signals if image is present
                if image is not None and item.correction_type in (
                    CORRECTION_RELABEL,
                    CORRECTION_BOX_MOVE,
                    CORRECTION_BOX_RESIZE,
                    CORRECTION_MASK_EDIT,
                    CORRECTION_REGION_EDIT,
                    CORRECTION_LANE_EDIT,
                    CORRECTION_ADD_MISSING,
                    CORRECTION_DELETE_FALSE_POSITIVE,
                ):
                    crop_rel_path = self._save_correction_crop(image, item, image_hash)

                cursor.execute(
                    """
                    INSERT INTO corrections
                        (image_hash, task_id, job_id, frame_index, correction_type,
                         ai_label, human_label, ai_shape_json, human_shape_json,
                         iou, crop_path, details_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        image_hash,
                        task_id,
                        job_id,
                        frame_index,
                        item.correction_type,
                        item.ai_label,
                        item.human_label,
                        json.dumps(item.ai_shape) if item.ai_shape else None,
                        json.dumps(item.human_shape) if item.human_shape else None,
                        float(item.iou),
                        crop_rel_path,
                        json.dumps(item.details),
                        now_iso,
                    ),
                )
                created_ids.append(cursor.lastrowid)

            conn.commit()

        # Prune storage if crop limits exceeded
        self._prune_crops_if_needed()
        return created_ids

    def _save_correction_crop(
        self,
        image: Image.Image,
        item: CorrectionDiffItem,
        image_hash: str,
    ) -> Optional[str]:
        """Save a bounded privacy-preserving crop around corrected object."""
        try:
            target_shape = item.human_shape or item.ai_shape
            bbox = extract_shape_bbox(target_shape) if target_shape else None
            if not bbox and item.details:
                bbox = item.details.get("human_bbox") or item.details.get("ai_bbox")
            if not bbox:
                return None

            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                return None

            img_w, img_h = image.size
            # Add padding around bounding box for context (minimum 16px to protect thin lane polylines)
            pad_x = max(w * 0.12, 16.0)
            pad_y = max(h * 0.12, 16.0)
            crop_x1 = max(0, int(math.floor(x1 - pad_x)))
            crop_y1 = max(0, int(math.floor(y1 - pad_y)))
            crop_x2 = min(img_w, int(math.ceil(x2 + pad_x)))
            crop_y2 = min(img_h, int(math.ceil(y2 + pad_y)))

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                return None

            crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cw, ch = crop_img.size

            # Store crop bounding coordinates in details for exact few-shot geometry normalization
            item.details["crop_coords"] = [crop_x1, crop_y1, crop_x2, crop_y2]
            item.details["crop_size"] = [cw, ch]

            # Resize if exceeding max allowed dimension (e.g. 512px)
            if max(cw, ch) > self.max_crop_dimension:
                scale = float(self.max_crop_dimension) / float(max(cw, ch))
                new_w = max(1, int(round(cw * scale)))
                new_h = max(1, int(round(ch * scale)))
                crop_img = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            unique_suffix = hashlib.md5(f"{x1}_{y1}_{item.correction_type}_{time.time()}".encode()).hexdigest()[:8]
            fname = f"crop_{image_hash[:12]}_{unique_suffix}.jpg"
            out_file = self.examples_dir / fname
            crop_img.convert("RGB").save(out_file, format="JPEG", quality=85)

            return str(out_file.relative_to(self.examples_dir.parent))
        except Exception:
            return None

    def _prune_crops_if_needed(self) -> None:
        """Enforce max_examples_total and max_storage_mb boundaries on local crops."""
        try:
            if not self.examples_dir.exists():
                return
            crops = list(self.examples_dir.glob("*.jpg"))
            if not crops:
                return

            # Check total count
            if len(crops) > self.max_examples_total:
                # Sort oldest first
                crops.sort(key=lambda p: p.stat().st_mtime)
                to_delete = crops[: len(crops) - self.max_examples_total]
                for p in to_delete:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass

            # Check storage size
            total_bytes = sum(p.stat().st_size for p in self.examples_dir.glob("*.jpg"))
            max_bytes = self.max_storage_mb * 1024 * 1024
            if total_bytes > max_bytes:
                crops = sorted(self.examples_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
                for p in crops:
                    if total_bytes <= max_bytes:
                        break
                    sz = p.stat().st_size
                    try:
                        p.unlink(missing_ok=True)
                        total_bytes -= sz
                    except OSError:
                        pass
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Correction Statistics across 31 Labels
    # -------------------------------------------------------------------------

    def get_label_statistics(self) -> Dict[str, Dict[str, int]]:
        """Compute comprehensive correction statistics across all 31 master labels.

        Guarantees that every label in ALL_31_LABELS has an entry in the returned dict.
        """
        stats: Dict[str, Dict[str, int]] = {
            lbl: {
                "accepted": 0,
                "relabeled_from": 0,
                "relabeled_to": 0,
                "box_moved": 0,
                "box_resized": 0,
                "mask_edited": 0,
                "region_edited": 0,
                "lane_edited": 0,
                "false_positives": 0,
                "missed_adds": 0,
                "total_corrections": 0,
            }
            for lbl in ALL_31_LABELS
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT correction_type, ai_label, human_label, COUNT(*) as cnt
                FROM corrections
                GROUP BY correction_type, ai_label, human_label
                """
            )
            for row in cursor.fetchall():
                ctype = row["correction_type"]
                ai_lbl = row["ai_label"]
                human_lbl = row["human_label"]
                cnt = int(row["cnt"])

                if ctype == CORRECTION_NO_CHANGE and ai_lbl in stats:
                    stats[ai_lbl]["accepted"] += cnt

                elif ctype == CORRECTION_RELABEL:
                    if ai_lbl in stats:
                        stats[ai_lbl]["relabeled_from"] += cnt
                        stats[ai_lbl]["total_corrections"] += cnt
                    if human_lbl in stats:
                        stats[human_lbl]["relabeled_to"] += cnt
                        stats[human_lbl]["total_corrections"] += cnt

                elif ctype == CORRECTION_BOX_MOVE and ai_lbl in stats:
                    stats[ai_lbl]["box_moved"] += cnt
                    stats[ai_lbl]["total_corrections"] += cnt

                elif ctype == CORRECTION_BOX_RESIZE and ai_lbl in stats:
                    stats[ai_lbl]["box_resized"] += cnt
                    stats[ai_lbl]["total_corrections"] += cnt

                elif ctype == CORRECTION_MASK_EDIT and ai_lbl in stats:
                    stats[ai_lbl]["mask_edited"] += cnt
                    stats[ai_lbl]["total_corrections"] += cnt

                elif ctype == CORRECTION_REGION_EDIT and ai_lbl in stats:
                    stats[ai_lbl]["region_edited"] += cnt
                    stats[ai_lbl]["total_corrections"] += cnt

                elif ctype == CORRECTION_LANE_EDIT and ai_lbl in stats:
                    stats[ai_lbl]["lane_edited"] += cnt
                    stats[ai_lbl]["total_corrections"] += cnt

                elif ctype == CORRECTION_DELETE_FALSE_POSITIVE and ai_lbl in stats:
                    stats[ai_lbl]["false_positives"] += cnt
                    stats[ai_lbl]["total_corrections"] += cnt

                elif ctype == CORRECTION_ADD_MISSING and human_lbl in stats:
                    stats[human_lbl]["missed_adds"] += cnt
                    stats[human_lbl]["total_corrections"] += cnt

        return stats

    def get_corrections(
        self,
        limit: int = 50,
        label: Optional[str] = None,
        correction_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent correction records."""
        query = "SELECT * FROM corrections WHERE 1=1"
        params: List[Any] = []

        if label:
            query += " AND (ai_label = ? OR human_label = ?)"
            params.extend([label, label])

        if correction_type:
            query += " AND correction_type = ?"
            params.append(correction_type)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Derived Rule Generation
    # -------------------------------------------------------------------------

    def derive_rules(self, min_samples: int = 3) -> List[Dict[str, Any]]:
        """Synthesize actionable rules from correction patterns meeting sample threshold.

        Rules cover:
        1. Confusion pairs (e.g. car -> truck corrected >= min_samples times).
        2. High false-positive rates (label deleted >= min_samples times).
        3. Frequently missed objects (label added >= min_samples times).

        Returns:
            List of active derived rules.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        rules_added: List[Dict[str, Any]] = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Confusion pairs: RELABEL
            cursor.execute(
                """
                SELECT ai_label, human_label, COUNT(*) as cnt
                FROM corrections
                WHERE correction_type = 'RELABEL' AND ai_label IS NOT NULL AND human_label IS NOT NULL
                GROUP BY ai_label, human_label
                HAVING cnt >= ?
                """,
                (min_samples,),
            )
            for row in cursor.fetchall():
                src = row["ai_label"]
                tgt = row["human_label"]
                cnt = int(row["cnt"])
                rule_text = (
                    f"Frequent confusion pattern: Objects predicted as '{src}' were corrected to '{tgt}' "
                    f"in {cnt} human reviews. Carefully distinguish '{src}' from '{tgt}' by inspecting vehicle bed, "
                    f"size, or visual context before outputting."
                )
                cursor.execute(
                    """
                    INSERT INTO rules
                        (source_label, target_label, rule_type, rule_text, sample_count, confidence, is_active, updated_at)
                    VALUES (?, ?, 'confusion_pair', ?, ?, 1.0, 1, ?)
                    ON CONFLICT(rule_text) DO UPDATE SET
                        sample_count = excluded.sample_count,
                        updated_at = excluded.updated_at
                    """,
                    (src, tgt, rule_text, cnt, now_iso),
                )
                rules_added.append({
                    "source_label": src,
                    "target_label": tgt,
                    "rule_type": "confusion_pair",
                    "rule_text": rule_text,
                    "sample_count": cnt,
                })

            # 2. Frequent False Positives: DELETE_FALSE_POSITIVE
            cursor.execute(
                """
                SELECT ai_label, COUNT(*) as cnt
                FROM corrections
                WHERE correction_type = 'DELETE_FALSE_POSITIVE' AND ai_label IS NOT NULL
                GROUP BY ai_label
                HAVING cnt >= ?
                """,
                (min_samples,),
            )
            for row in cursor.fetchall():
                src = row["ai_label"]
                cnt = int(row["cnt"])
                rule_text = (
                    f"High false-positive rate: Model frequently hallucinated '{src}' ({cnt} deleted by annotators). "
                    f"Require high visual certainty before predicting '{src}'."
                )
                cursor.execute(
                    """
                    INSERT INTO rules
                        (source_label, target_label, rule_type, rule_text, sample_count, confidence, is_active, updated_at)
                    VALUES (?, NULL, 'false_positive', ?, ?, 1.0, 1, ?)
                    ON CONFLICT(rule_text) DO UPDATE SET
                        sample_count = excluded.sample_count,
                        updated_at = excluded.updated_at
                    """,
                    (src, rule_text, cnt, now_iso),
                )
                rules_added.append({
                    "source_label": src,
                    "target_label": None,
                    "rule_type": "false_positive",
                    "rule_text": rule_text,
                    "sample_count": cnt,
                })

            # 3. Frequently Missed Additions: ADD_MISSING
            cursor.execute(
                """
                SELECT human_label, COUNT(*) as cnt
                FROM corrections
                WHERE correction_type = 'ADD_MISSING' AND human_label IS NOT NULL
                GROUP BY human_label
                HAVING cnt >= ?
                """,
                (min_samples,),
            )
            for row in cursor.fetchall():
                tgt = row["human_label"]
                cnt = int(row["cnt"])
                rule_text = (
                    f"Frequently missed class: Annotators manually added '{tgt}' ({cnt} times). "
                    f"Thoroughly inspect the image foreground and background for overlooked '{tgt}' instances."
                )
                cursor.execute(
                    """
                    INSERT INTO rules
                        (source_label, target_label, rule_type, rule_text, sample_count, confidence, is_active, updated_at)
                    VALUES (NULL, ?, 'missed_addition', ?, ?, 1.0, 1, ?)
                    ON CONFLICT(rule_text) DO UPDATE SET
                        sample_count = excluded.sample_count,
                        updated_at = excluded.updated_at
                    """,
                    (tgt, rule_text, cnt, now_iso),
                )
                rules_added.append({
                    "source_label": None,
                    "target_label": tgt,
                    "rule_type": "missed_addition",
                    "rule_text": rule_text,
                    "sample_count": cnt,
                })

            conn.commit()

        return rules_added

    def get_active_rules(self, labels: Optional[Sequence[str]] = None) -> List[str]:
        """Retrieve active rule strings filtered by candidate labels."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if labels:
                placeholders = ",".join("?" for _ in labels)
                query = f"""
                    SELECT rule_text FROM rules
                    WHERE is_active = 1
                      AND (source_label IN ({placeholders}) OR target_label IN ({placeholders}) OR (source_label IS NULL AND target_label IS NULL))
                    ORDER BY sample_count DESC, id DESC
                """
                cursor.execute(query, list(labels) + list(labels))
            else:
                cursor.execute("SELECT rule_text FROM rules WHERE is_active = 1 ORDER BY sample_count DESC, id DESC")

            rows = cursor.fetchall()
            return [str(r["rule_text"]) for r in rows]

    # -------------------------------------------------------------------------
    # Maintenance & Settings
    # -------------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        """Toggle feedback learning globally."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO feedback_meta (key, value) VALUES ('enabled', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("true" if enabled else "false",),
            )
            conn.commit()

    def is_enabled(self) -> bool:
        """Check if feedback learning is enabled."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM feedback_meta WHERE key = 'enabled'")
                row = cursor.fetchone()
                if not row:
                    return True  # Enabled by default
                return row["value"].lower() == "true"
        except Exception as e:
            logger.warning("Error reading feedback_meta: %s; defaulting is_enabled to True", e)
            return True

    def clear(
        self,
        all_data: bool = True,
        crops: bool = False,
        rules: bool = False,
    ) -> None:
        """Clear database records and crops."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if all_data:
                cursor.execute("DELETE FROM corrections")
                cursor.execute("DELETE FROM predictions")
                cursor.execute("DELETE FROM human_annotations")
                cursor.execute("DELETE FROM rules")
            elif rules:
                cursor.execute("DELETE FROM rules")
            conn.commit()

        if (all_data or crops) and self.examples_dir.exists():
            for p in self.examples_dir.glob("*.jpg"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def reconstruct_shapes(
        correction_record: Dict[str, Any],
        target: str = "human",
    ) -> List[Dict[str, Any]]:
        """Reconstruct both member shapes from a stored correction record.

        Supports Policy A (human_rect + human_mask), Policy B (human_poly + human_mask),
        and Policy C (human_polyline), with fallback to human_shape_json.

        Args:
            correction_record: Row dictionary from corrections table or CorrectionDiffItem dict.
            target: 'human' or 'ai'.

        Returns:
            List of CVAT shape dictionaries representing the full composite annotation.
        """
        details: Dict[str, Any] = {}
        raw_details = correction_record.get("details_json") or correction_record.get("details")
        if isinstance(raw_details, str):
            try:
                details = json.loads(raw_details)
            except Exception:
                details = {}
        elif isinstance(raw_details, dict):
            details = raw_details

        # 1. Exact sub_shapes stored in details
        sub_shapes_key = f"{target}_sub_shapes"
        if details.get(sub_shapes_key) and isinstance(details[sub_shapes_key], list):
            return details[sub_shapes_key]

        # 2. Structured paired member shapes from details
        shapes: List[Dict[str, Any]] = []
        rect_shape = details.get(f"{target}_rect")
        poly_shape = details.get(f"{target}_poly")
        mask_shape = details.get(f"{target}_mask")
        polyline_shape = details.get(f"{target}_polyline")

        if rect_shape:
            shapes.append(rect_shape)
        if poly_shape:
            shapes.append(poly_shape)
        if mask_shape:
            shapes.append(mask_shape)
        if polyline_shape:
            shapes.append(polyline_shape)

        if shapes:
            return shapes

        # 3. Fallback to single primary shape in human_shape_json / ai_shape_json
        shape_key = f"{target}_shape_json" if f"{target}_shape_json" in correction_record else f"{target}_shape"
        raw_shape = correction_record.get(shape_key)
        if isinstance(raw_shape, str):
            try:
                parsed = json.loads(raw_shape)
                if isinstance(parsed, dict):
                    return [parsed]
                elif isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        elif isinstance(raw_shape, dict):
            return [raw_shape]
        elif isinstance(raw_shape, list):
            return raw_shape

        return []


# Aliases for feedback correction storage repository and shape reconstruction
CorrectionDatabase = FeedbackDatabase
reconstruct_shapes = FeedbackDatabase.reconstruct_shapes


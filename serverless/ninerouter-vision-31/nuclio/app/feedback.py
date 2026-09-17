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
from core.taxonomy import (
    GROUP_INSTANCE,
    GROUP_LANE,
    GROUP_REGION,
    MASTER_31_LABELS,
    Taxonomy,
)

ALL_31_LABELS = list(MASTER_31_LABELS)

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

    ux1 = max(0, int(math.floor(min(bbox_a[0], bbox_b[0]))))
    uy1 = max(0, int(math.floor(min(bbox_a[1], bbox_b[1]))))
    ux2 = min(width - 1, int(math.ceil(max(bbox_a[2], bbox_b[2]))))
    uy2 = min(height - 1, int(math.ceil(max(bbox_a[3], bbox_b[3]))))

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
    ):
        self.min_iou_relabel = float(min_iou_relabel)
        self.min_iou_match = float(min_iou_match)
        self.taxonomy = taxonomy or Taxonomy()

    def diff(
        self,
        ai_shapes: List[Dict[str, Any]],
        human_shapes: List[Dict[str, Any]],
        image_width: int = 1280,
        image_height: int = 720,
    ) -> List[CorrectionDiffItem]:
        """Compute bipartite matching diff between AI predictions and human annotations.

        Args:
            ai_shapes: List of AI predicted shapes (e.g. from AnnotationResult or CVAT API).
            human_shapes: List of final human edited shapes from CVAT job.
            image_width: Width of image in pixels.
            image_height: Height of image in pixels.

        Returns:
            List of CorrectionDiffItem entries classifying every shape difference.
        """
        results: List[CorrectionDiffItem] = []

        # 1. Filter and compute bounding boxes for all AI shapes
        ai_entries = []
        for idx, s in enumerate(ai_shapes):
            bbox = extract_shape_bbox(s)
            ai_entries.append({
                "idx": idx,
                "shape": s,
                "label": s.get("label", ""),
                "bbox": bbox,
                "type": s.get("type", "rectangle"),
                "group": self.taxonomy.get_group(s.get("label", "")),
            })

        # 2. Filter and compute bounding boxes for all Human shapes
        human_entries = []
        for idx, s in enumerate(human_shapes):
            bbox = extract_shape_bbox(s)
            human_entries.append({
                "idx": idx,
                "shape": s,
                "label": s.get("label", ""),
                "bbox": bbox,
                "type": s.get("type", "rectangle"),
                "group": self.taxonomy.get_group(s.get("label", "")),
            })

        matched_ai: Set[int] = set()
        matched_human: Set[int] = set()

        # 3. Compute pairwise IoU candidates
        candidates = []
        for a in ai_entries:
            if a["bbox"] is None:
                continue
            for h in human_entries:
                if h["bbox"] is None:
                    continue
                iou = calculate_box_iou(a["bbox"], h["bbox"])
                if iou > 0.0:
                    candidates.append((iou, a["idx"], h["idx"]))

        # Sort candidate pairs by IoU descending (greedy bipartite matching)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 4. First pass: Match same-label or cross-label pairs
        for iou, a_idx, h_idx in candidates:
            if a_idx in matched_ai or h_idx in matched_human:
                continue

            a = ai_entries[a_idx]
            h = human_entries[h_idx]
            ai_lbl = a["label"]
            human_lbl = h["label"]

            if ai_lbl == human_lbl:
                if iou >= self.min_iou_match:
                    item = self._classify_same_label(a, h, iou, image_width, image_height)
                    results.append(item)
                    matched_ai.add(a_idx)
                    matched_human.add(h_idx)
            else:
                # Cross-label candidate: requires higher IoU threshold to avoid false match
                if iou >= self.min_iou_relabel:
                    item = CorrectionDiffItem(
                        correction_type=CORRECTION_RELABEL,
                        ai_label=ai_lbl,
                        human_label=human_lbl,
                        ai_shape=a["shape"],
                        human_shape=h["shape"],
                        iou=round(iou, 4),
                        details={
                            "reason": f"Annotator relabeled '{ai_lbl}' to '{human_lbl}' (IoU {iou:.2f})",
                            "ai_bbox": a["bbox"],
                            "human_bbox": h["bbox"],
                        },
                    )
                    results.append(item)
                    matched_ai.add(a_idx)
                    matched_human.add(h_idx)

        # 5. Remaining unmatched AI shapes -> False Positives (Annotator deleted AI prediction)
        for a in ai_entries:
            if a["idx"] not in matched_ai:
                results.append(
                    CorrectionDiffItem(
                        correction_type=CORRECTION_DELETE_FALSE_POSITIVE,
                        ai_label=a["label"],
                        human_label=None,
                        ai_shape=a["shape"],
                        human_shape=None,
                        iou=0.0,
                        details={
                            "reason": f"AI predicted '{a['label']}' which was deleted or rejected by annotator",
                            "ai_bbox": a["bbox"],
                        },
                    )
                )

        # 6. Remaining unmatched Human shapes -> False Negatives (Annotator added missing object)
        for h in human_entries:
            if h["idx"] not in matched_human:
                results.append(
                    CorrectionDiffItem(
                        correction_type=CORRECTION_ADD_MISSING,
                        ai_label=None,
                        human_label=h["label"],
                        ai_shape=None,
                        human_shape=h["shape"],
                        iou=0.0,
                        details={
                            "reason": f"Annotator manually added missing '{h['label']}'",
                            "human_bbox": h["bbox"],
                        },
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
    ) -> CorrectionDiffItem:
        """Classify shape difference when AI and human share the same label."""
        lbl = a["label"]
        group = a["group"]
        stype = h.get("type", a.get("type", "rectangle"))
        a_bbox = a["bbox"]
        h_bbox = h["bbox"]

        # 1. Semantic Regions (True Mask IoU)
        if group == GROUP_REGION or "area/" in lbl:
            true_iou = _compute_shape_iou(a["shape"], h["shape"], img_w, img_h)
            if true_iou >= 0.92:
                return CorrectionDiffItem(
                    correction_type=CORRECTION_NO_CHANGE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(true_iou, 4),
                    details={
                        "status": "Region accepted with negligible change",
                        "mask_iou": round(true_iou, 4),
                        "bbox_iou": round(iou, 4),
                    },
                )
            return CorrectionDiffItem(
                correction_type=CORRECTION_REGION_EDIT,
                ai_label=lbl,
                human_label=lbl,
                ai_shape=a["shape"],
                human_shape=h["shape"],
                iou=round(true_iou, 4),
                details={
                    "reason": f"Semantic region '{lbl}' contour edited by annotator (true mask IoU {true_iou:.2f})",
                    "mask_iou": round(true_iou, 4),
                    "bbox_iou": round(iou, 4),
                },
            )

        # 2. Lane Markings (True Line Stroke Overlap)
        if group == GROUP_LANE or lbl.startswith("lane/"):
            true_iou = _compute_shape_iou(a["shape"], h["shape"], img_w, img_h, line_width=4)
            if true_iou >= 0.90:
                return CorrectionDiffItem(
                    correction_type=CORRECTION_NO_CHANGE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(true_iou, 4),
                    details={
                        "status": "Lane marking accepted",
                        "lane_iou": round(true_iou, 4),
                        "bbox_iou": round(iou, 4),
                    },
                )
            return CorrectionDiffItem(
                correction_type=CORRECTION_LANE_EDIT,
                ai_label=lbl,
                human_label=lbl,
                ai_shape=a["shape"],
                human_shape=h["shape"],
                iou=round(true_iou, 4),
                details={
                    "reason": f"Lane marking '{lbl}' geometry edited by annotator (true lane IoU {true_iou:.2f})",
                    "lane_iou": round(true_iou, 4),
                    "bbox_iou": round(iou, 4),
                },
            )

        # 3. Instance Masks (True Mask IoU)
        if stype == "mask" or a.get("type") == "mask":
            true_iou = _compute_shape_iou(a["shape"], h["shape"], img_w, img_h)
            if true_iou >= 0.92:
                return CorrectionDiffItem(
                    correction_type=CORRECTION_NO_CHANGE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(true_iou, 4),
                    details={
                        "status": "Mask accepted",
                        "mask_iou": round(true_iou, 4),
                        "bbox_iou": round(iou, 4),
                    },
                )
            return CorrectionDiffItem(
                correction_type=CORRECTION_MASK_EDIT,
                ai_label=lbl,
                human_label=lbl,
                ai_shape=a["shape"],
                human_shape=h["shape"],
                iou=round(true_iou, 4),
                details={
                    "reason": f"Instance mask '{lbl}' boundary adjusted (true mask IoU {true_iou:.2f})",
                    "mask_iou": round(true_iou, 4),
                    "bbox_iou": round(iou, 4),
                },
            )

        # 4. Rectangles / Bounding Boxes
        if a_bbox and h_bbox:
            ax1, ay1, ax2, ay2 = a_bbox
            hx1, hy1, hx2, hy2 = h_bbox

            aw, ah = ax2 - ax1, ay2 - ay1
            hw, hh = hx2 - hx1, hy2 - hy1

            acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
            hcx, hcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0

            shift_x = abs(acx - hcx)
            shift_y = abs(acy - hcy)
            dw = abs(aw - hw) / max(aw, hw, 1.0)
            dh = abs(ah - hh) / max(ah, hh, 1.0)

            # High IoU and minimal translation/size change -> Accepted
            if iou >= 0.92 and shift_x <= 4.0 and shift_y <= 4.0 and dw <= 0.08 and dh <= 0.08:
                return CorrectionDiffItem(
                    correction_type=CORRECTION_NO_CHANGE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(iou, 4),
                    details={"status": "Bounding box accepted without edit"},
                )

            # Significant dimension change
            if dw > 0.15 or dh > 0.15:
                return CorrectionDiffItem(
                    correction_type=CORRECTION_BOX_RESIZE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(iou, 4),
                    details={
                        "reason": f"Bounding box for '{lbl}' resized (dw={dw:.2f}, dh={dh:.2f})",
                        "size_diff": {"dw": round(dw, 3), "dh": round(dh, 3)},
                    },
                )

            # Center position translation
            if shift_x > 5.0 or shift_y > 5.0:
                return CorrectionDiffItem(
                    correction_type=CORRECTION_BOX_MOVE,
                    ai_label=lbl,
                    human_label=lbl,
                    ai_shape=a["shape"],
                    human_shape=h["shape"],
                    iou=round(iou, 4),
                    details={
                        "reason": f"Bounding box for '{lbl}' shifted position (dx={shift_x:.1f}, dy={shift_y:.1f})",
                        "shift": {"dx": round(shift_x, 2), "dy": round(shift_y, 2)},
                    },
                )

        # Default fallback if matching
        return CorrectionDiffItem(
            correction_type=CORRECTION_NO_CHANGE,
            ai_label=lbl,
            human_label=lbl,
            ai_shape=a["shape"],
            human_shape=h["shape"],
            iou=round(iou, 4),
            details={"status": "Accepted"},
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
        """Create a connection with WAL mode (or safe fallback journal mode) and row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("PRAGMA journal_mode=WAL")
            row = cur.fetchone()
            mode = (row[0] if row else "").upper()
            if mode != "WAL":
                logger.warning(
                    "SQLite WAL mode not supported on filesystem (mode=%s); falling back to TRUNCATE",
                    mode,
                )
                conn.execute("PRAGMA journal_mode=TRUNCATE")
        except sqlite3.OperationalError as exc:
            logger.warning(
                "SQLite WAL mode failed (%s); falling back to TRUNCATE safe journal mode",
                exc,
            )
            conn.execute("PRAGMA journal_mode=TRUNCATE")
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass
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
            if not target_shape:
                return None
            bbox = extract_shape_bbox(target_shape)
            if not bbox:
                return None

            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                return None

            img_w, img_h = image.size
            # Add 12% padding around bounding box for context
            pad_x = w * 0.12
            pad_y = h * 0.12
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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM feedback_meta WHERE key = 'enabled'")
            row = cursor.fetchone()
            if not row:
                return True  # Enabled by default
            return row["value"].lower() == "true"

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

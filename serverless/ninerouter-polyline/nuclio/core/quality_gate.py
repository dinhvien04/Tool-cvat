"""Cheap deterministic geometry quality gate for adaptive refinement.

The gate does not call any model. It only identifies structurally valid but
suspicious geometry so the annotation service can make at most ONE optional
higher-quality second pass. Human CVAT edits remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple


@dataclass
class QualityReport:
    score: float
    needs_refine: bool
    reasons: List[str] = field(default_factory=list)
    suspect_labels: List[str] = field(default_factory=list)
    item_count: int = 0
    suspect_count: int = 0


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        total += float(p[0]) * float(q[1]) - float(q[0]) * float(p[1])
    return abs(total) * 0.5


def _bbox_from_points(points: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 1e-9 else 0.0


def _orient(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def _segments_intersect(a, b, c, d) -> bool:
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _self_intersects(points: Sequence[Sequence[float]], *, closed: bool = True) -> bool:
    n = len(points)
    if n < 4:
        return False
    if n > 200:
        points = points[:200]
        n = len(points)
    edge_count = n if closed else n - 1
    for i in range(edge_count):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, edge_count):
            if abs(i - j) <= 1:
                continue
            if closed and i == 0 and j == edge_count - 1:
                continue
            c, d = points[j], points[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _line_path_ratio(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 999.0
    path = sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(points, points[1:])
    )
    chord = math.hypot(
        float(points[-1][0]) - float(points[0][0]),
        float(points[-1][1]) - float(points[0][1]),
    )
    return path / max(chord, 1.0)


def assess_quality(
    items: Iterable[Any],
    mode: str,
    *,
    min_confidence: float = 0.80,
    refine_score_threshold: float = 0.76,
) -> QualityReport:
    """Return a cheap heuristic quality report for parsed 0..1000 geometry."""
    mode = (mode or "").strip().lower()
    items = list(items)
    if not items:
        # Empty output is not "perfect quality" for an annotation request. Treat it
        # as suspicious so the bounded stronger pass gets one chance to recover
        # a missed road/drivable/object/lane result.
        return QualityReport(
            score=0.0,
            needs_refine=True,
            reasons=["empty_detection"],
            suspect_labels=[],
            item_count=0,
            suspect_count=1,
        )

    scores: List[float] = []
    reasons: List[str] = []
    suspects: List[str] = []

    for item in items:
        score = 1.0
        item_reasons: List[str] = []
        confidence = getattr(item, "confidence", None)
        if confidence is not None and float(confidence) < min_confidence:
            score -= 0.18
            item_reasons.append("low_confidence")

        if mode in ("rectangle_mask", "box_mask"):
            box = getattr(item, "box_2d", None)
            mask = getattr(item, "instance_mask", None) or getattr(item, "mask", None)
            if not box or not mask or len(mask) < 3:
                score = 0.0
                item_reasons.append("missing_box_or_mask")
            else:
                ymin, xmin, ymax, xmax = map(float, box)
                box_xyxy = (xmin, ymin, xmax, ymax)
                box_area = max(1.0, (xmax - xmin) * (ymax - ymin))
                mask_bbox = _bbox_from_points(mask)
                mask_area = _polygon_area(mask)
                if len(mask) < 6:
                    score -= 0.18
                    item_reasons.append("coarse_mask")
                if mask_bbox is not None and _bbox_iou_xyxy(box_xyxy, mask_bbox) < 0.62:
                    score -= 0.30
                    item_reasons.append("mask_box_mismatch")
                fill_ratio = mask_area / box_area
                if fill_ratio < 0.18 or fill_ratio > 1.20:
                    score -= 0.30
                    item_reasons.append("implausible_mask_fill")
                if _self_intersects(mask, closed=True):
                    score -= 0.30
                    item_reasons.append("self_intersecting_mask")

        elif mode == "polygon_mask":
            poly = getattr(item, "region_polygon", None) or getattr(item, "mask", None)
            if not poly or len(poly) < 3:
                score = 0.0
                item_reasons.append("missing_polygon")
            else:
                area = _polygon_area(poly)
                bbox = _bbox_from_points(poly)
                bbox_area = 0.0
                if bbox is not None:
                    bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                if area < 150.0:
                    score -= 0.28
                    item_reasons.append("tiny_region")
                if len(poly) < 7:
                    score -= 0.22
                    item_reasons.append("coarse_polygon")
                # Large road/sky/drivable regions with only a few corners are the
                # most visible failure mode of the fast first pass.
                if bbox_area >= 180_000.0 and len(poly) < 16:
                    score -= 0.30
                    item_reasons.append("large_region_too_coarse")
                if _self_intersects(poly, closed=True):
                    score -= 0.35
                    item_reasons.append("self_intersecting_polygon")

        elif mode == "polyline":
            line = getattr(item, "lane_polyline", None) or getattr(item, "mask", None)
            if not line or len(line) < 2:
                score = 0.0
                item_reasons.append("missing_polyline")
            else:
                if len(line) < 3:
                    score -= 0.18
                    item_reasons.append("undersampled_polyline")
                if _line_path_ratio(line) > 3.2:
                    score -= 0.32
                    item_reasons.append("zigzag_polyline")
                max_seg = max(
                    (
                        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
                        for a, b in zip(line, line[1:])
                    ),
                    default=0.0,
                )
                if max_seg > 700.0:
                    score -= 0.20
                    item_reasons.append("oversized_line_jump")

        score = max(0.0, min(1.0, score))
        scores.append(score)
        if score < refine_score_threshold:
            suspects.append(str(getattr(item, "label", "unknown")))
            reasons.extend(f"{getattr(item, 'label', 'unknown')}:{r}" for r in item_reasons)

    aggregate = sum(scores) / max(1, len(scores))
    return QualityReport(
        score=round(aggregate, 3),
        needs_refine=(bool(suspects) or aggregate < refine_score_threshold),
        reasons=reasons,
        suspect_labels=suspects,
        item_count=len(items),
        suspect_count=len(suspects),
    )


def should_accept_refinement(
    before: QualityReport,
    after: QualityReport,
    *,
    min_gain: float = 0.03,
) -> bool:
    """Conservatively select a second-pass result without collapsing recall."""
    if after.item_count <= 0:
        return False
    if before.item_count > 1 and after.item_count < max(1, math.ceil(before.item_count * 0.65)):
        return False
    if after.suspect_count < before.suspect_count and after.score >= before.score - 0.03:
        return True
    if after.score >= before.score + min_gain:
        return True
    # Heuristics cannot directly judge semantic boundary alignment. If a stronger
    # pass preserves recall and is not measurably worse, let it replace the draft.
    if before.needs_refine and after.item_count >= before.item_count and after.score >= before.score - 0.015:
        return True
    return False

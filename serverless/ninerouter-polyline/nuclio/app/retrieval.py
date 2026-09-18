"""Lightweight Few-Shot Retrieval and Adaptive Prompt Injection Engine.

Zero-Heavy-ML Directive:
- Strictly NO local embedding models (no PyTorch, Transformers, Sentence-Transformers).
- Strictly NO local vector databases (no Chroma, FAISS, Milvus).
- Fast, deterministic metadata-driven retrieval based on label taxonomy, confusion pairs,
  quality tiers, and recent correction recency.
- Strictly bounded: 2-4 examples maximum per pass to conserve API tokens.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

from app.feedback import (
    CORRECTION_ADD_MISSING,
    CORRECTION_BOX_MOVE,
    CORRECTION_BOX_RESIZE,
    CORRECTION_DELETE_FALSE_POSITIVE,
    CORRECTION_LANE_EDIT,
    CORRECTION_MASK_EDIT,
    CORRECTION_NO_CHANGE,
    CORRECTION_REGION_EDIT,
    CORRECTION_RELABEL,
    FeedbackDatabase,
)
from core.taxonomy import (
    BOX_MASK_LABELS,
    GROUP_INSTANCE,
    GROUP_LANE,
    GROUP_REGION,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    Taxonomy,
)

logger = logging.getLogger(__name__)


def resolve_policy_from_labels(
    candidate_labels: Optional[Sequence[str]] = None,
    policy: Optional[str] = None,
) -> Optional[str]:
    """Resolve and normalize explicit policy string ('box_mask', 'polygon_mask', 'polyline', or None for full_31)."""
    if policy:
        p = policy.strip().lower()
        if p in (POLICY_BOX_MASK, "rectangle_mask", "box_mask", "instance", "detector_a"):
            return POLICY_BOX_MASK
        if p in (POLICY_POLYGON_MASK, "polygon_mask", "region", "detector_b"):
            return POLICY_POLYGON_MASK
        if p in (POLICY_POLYLINE, "polyline", "lane", "detector_c"):
            return POLICY_POLYLINE
        if p in ("full_31", "all", "none"):
            return None

    return None


@dataclass
class RetrievalResult:
    """Packaged correction rules and few-shot examples for adaptive inference."""
    rules: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    prompt_extension: str = ""
    visual_examples: List[Dict[str, Any]] = field(default_factory=list)

    def has_content(self) -> bool:
        return bool(self.rules or self.examples or self.prompt_extension or self.visual_examples)


class CorrectionRetrievalEngine:
    """Selects high-value correction rules and few-shot guidance for current inference."""

    def __init__(
        self,
        db: Optional[FeedbackDatabase] = None,
        max_examples_per_pass: int = 3,
        use_rules: bool = True,
        use_examples: bool = True,
        use_visual_examples: bool = True,
    ):
        self.db = db or FeedbackDatabase()
        self.max_examples_per_pass = max_examples_per_pass
        self.use_rules = use_rules
        self.use_examples = use_examples
        self.use_visual_examples = use_visual_examples

    def retrieve(
        self,
        candidate_labels: Optional[Sequence[str]] = None,
        policy: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve active derived rules and relevant few-shot correction examples.

        Args:
            candidate_labels: List of active class labels for the current detection task.
            policy: Optional policy ('box_mask', 'polygon_mask', 'polyline', 'full_31').
                If omitted, automatically inferred from candidate_labels.

        Returns:
            RetrievalResult containing rule strings, example records, visual examples, and formatted prompt text.
        """
        if not self.db.is_enabled():
            return RetrievalResult()

        canon_policy = resolve_policy_from_labels(candidate_labels=candidate_labels, policy=policy)

        # Restrict candidate labels to the active policy if active
        effective_labels = candidate_labels
        if canon_policy == POLICY_BOX_MASK:
            allowed = set(BOX_MASK_LABELS)
            effective_labels = [l for l in candidate_labels if l in allowed] if candidate_labels else list(BOX_MASK_LABELS)
        elif canon_policy == POLICY_POLYGON_MASK:
            allowed = set(POLYGON_MASK_LABELS)
            effective_labels = [l for l in candidate_labels if l in allowed] if candidate_labels else list(POLYGON_MASK_LABELS)
        elif canon_policy == POLICY_POLYLINE:
            allowed = set(POLYLINE_LABELS)
            effective_labels = [l for l in candidate_labels if l in allowed] if candidate_labels else list(POLYLINE_LABELS)

        rules: List[str] = []
        if self.use_rules:
            rules = self.db.get_active_rules(labels=effective_labels)

        examples: List[Dict[str, Any]] = []
        visual_examples: List[Dict[str, Any]] = []
        if self.use_examples:
            examples = self._select_few_shot_examples(candidate_labels=effective_labels, policy=canon_policy)
            if self.use_visual_examples:
                visual_examples = self._build_visual_examples(examples, policy=canon_policy)

        prompt_ext = self._format_prompt_extension(rules, examples)

        return RetrievalResult(
            rules=rules,
            examples=examples,
            prompt_extension=prompt_ext,
            visual_examples=visual_examples,
        )

    def _select_few_shot_examples(
        self,
        candidate_labels: Optional[Sequence[str]] = None,
        policy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Select top relevant few-shot correction records matching candidate labels and policy."""
        # Query recent corrections prioritizing RELABEL, ADD_MISSING, DELETE_FALSE_POSITIVE
        conn = self.db._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                SELECT id, image_hash, correction_type, ai_label, human_label,
                       ai_shape_json, human_shape_json, iou, crop_path, details_json, created_at
                FROM corrections
                WHERE correction_type != 'NO_CHANGE'
            """
            params: List[Any] = []

            if candidate_labels:
                placeholders = ",".join("?" for _ in candidate_labels)
                query += f" AND (ai_label IN ({placeholders}) OR human_label IN ({placeholders}))"
                params.extend(list(candidate_labels) + list(candidate_labels))

            # Zero cross-policy leakage: exclude records involving classes from other policies
            foreign_labels: List[str] = []
            if policy == POLICY_BOX_MASK:
                foreign_labels = list(POLYGON_MASK_LABELS) + list(POLYLINE_LABELS)
            elif policy == POLICY_POLYGON_MASK:
                foreign_labels = list(BOX_MASK_LABELS) + list(POLYLINE_LABELS)
            elif policy == POLICY_POLYLINE:
                foreign_labels = list(BOX_MASK_LABELS) + list(POLYGON_MASK_LABELS)

            if foreign_labels:
                f_ph = ",".join("?" for _ in foreign_labels)
                query += f" AND (ai_label IS NULL OR ai_label NOT IN ({f_ph}))"
                query += f" AND (human_label IS NULL OR human_label NOT IN ({f_ph}))"
                params.extend(foreign_labels + foreign_labels)

            # Prioritize RELABEL first, then ADD_MISSING, then DELETE_FALSE_POSITIVE
            query += """
                ORDER BY
                    CASE correction_type
                        WHEN 'RELABEL' THEN 1
                        WHEN 'ADD_MISSING' THEN 2
                        WHEN 'DELETE_FALSE_POSITIVE' THEN 3
                        ELSE 4
                    END ASC,
                    id DESC
                LIMIT ?
            """
            params.append(self.max_examples_per_pass * 2 if policy else self.max_examples_per_pass)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            taxonomy = Taxonomy()
            selected = []
            for r in rows:
                ai_lbl = r["ai_label"]
                human_lbl = r["human_label"]
                tgt_lbl = human_lbl or ai_lbl
                if policy and tgt_lbl:
                    if taxonomy.get_policy(tgt_lbl) != policy:
                        continue

                item = {
                    "id": r["id"],
                    "image_hash": r["image_hash"],
                    "correction_type": r["correction_type"],
                    "ai_label": ai_lbl,
                    "human_label": human_lbl,
                    "ai_shape_json": r["ai_shape_json"],
                    "human_shape_json": r["human_shape_json"],
                    "iou": r["iou"],
                    "crop_path": r["crop_path"],
                    "details_json": r["details_json"],
                    "created_at": r["created_at"],
                }
                selected.append(item)
                if len(selected) >= self.max_examples_per_pass:
                    break
            return selected
        finally:
            conn.close()

    def _build_visual_examples(
        self,
        examples: List[Dict[str, Any]],
        policy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build real multimodal few-shot visual examples with crop data URLs and expected annotations.

        Converts human-corrected geometry and labels into structured ground-truth annotations
        matching the 9Router vision contract. Safely skips unreadable, corrupt, or missing crops.
        Enforces policy-aware expected_output schemas (Section 14):
        - Policy A: single-key {"objects": [{"label": ..., "box_2d": ..., "mask": ...}]}
        - Policy B: single-key {"regions": [{"label": ..., "polygon": ...}]}
        - Policy C: single-key {"lanes": [{"label": ..., "polyline": ...}]}
        - Full 31: multi-key {"objects": [...], "regions": [...], "lanes": [...]}
        """
        visual_examples: List[Dict[str, Any]] = []

        for item in examples:
            if len(visual_examples) >= self.max_examples_per_pass:
                break

            crop_rel = item.get("crop_path")
            if not crop_rel:
                continue

            resolved_path = self.db.resolve_crop_path(crop_rel)
            if not resolved_path or not resolved_path.exists() or not resolved_path.is_file():
                continue

            # Load and encode crop safely
            try:
                with Image.open(resolved_path) as im:
                    rgb_im = im.convert("RGB")
                    if max(rgb_im.size) > 512:
                        rgb_im.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    rgb_im.save(buf, format="JPEG", quality=85)
                    crop_data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            except Exception as e:
                logger.debug(f"Skipping corrupt or unreadable crop file {resolved_path}: {e}")
                continue

            ctype = item.get("correction_type")
            ai_lbl = item.get("ai_label")
            human_lbl = item.get("human_label")

            # Parse details and shape JSONs
            details = {}
            if item.get("details_json"):
                try:
                    details = json.loads(item["details_json"])
                except Exception:
                    details = {}

            human_shape = {}
            if item.get("human_shape_json"):
                try:
                    human_shape = json.loads(item["human_shape_json"])
                except Exception:
                    human_shape = {}

            # Construct expected structured annotation aligned with 3-tier vision contract
            target_label = human_lbl or ai_lbl
            if not target_label:
                continue
            crop_coords = details.get("crop_coords")  # [cx1, cy1, cx2, cy2] in original image pixels

            taxonomy = Taxonomy()
            deleted_label = ai_lbl or target_label
            deleted_group = taxonomy.get_group(deleted_label)
            target_group = taxonomy.get_group(target_label)

            # Strict policy isolation: skip examples from other policies
            if policy == POLICY_BOX_MASK and target_group != GROUP_INSTANCE:
                continue
            if policy == POLICY_POLYGON_MASK and target_group != GROUP_REGION:
                continue
            if policy == POLICY_POLYLINE and target_group != GROUP_LANE:
                continue

            if ctype == CORRECTION_DELETE_FALSE_POSITIVE:
                # Reviewer deleted false positive; emit negative example matching target policy schema
                if policy == POLICY_BOX_MASK:
                    expected_output = {"objects": []}
                elif policy == POLICY_POLYGON_MASK:
                    expected_output = {"regions": []}
                elif policy == POLICY_POLYLINE:
                    expected_output = {"lanes": []}
                else:
                    expected_output = {
                        "objects": [],
                        "regions": [],
                        "lanes": [],
                    }
                if deleted_group == GROUP_REGION:
                    description = f"False positive semantic region '{deleted_label}' deleted by human reviewer (negative example)"
                elif deleted_group == GROUP_LANE:
                    description = f"False positive lane marking '{deleted_label}' deleted by human reviewer (negative example)"
                else:
                    description = f"False positive instance '{deleted_label}' deleted by human reviewer (negative example)"
            else:
                # Strictly require valid crop bounding coordinates; NEVER fabricate geometry
                if not crop_coords or len(crop_coords) != 4:
                    continue
                cx1, cy1, cx2, cy2 = [float(c) for c in crop_coords]
                cw = max(cx2 - cx1, 1.0)
                ch = max(cy2 - cy1, 1.0)

                # Route into objects[], regions[], or lanes[] according to 31-label taxonomy
                if target_group == GROUP_REGION or "area/" in target_label:
                    if policy in (POLICY_BOX_MASK, POLICY_POLYLINE):
                        continue

                    # Semantic Region: POLICY B -> "polygon" key (NEVER mask-only, NEVER box)
                    poly_pts = None
                    if details.get("human_poly") and details["human_poly"].get("points"):
                        poly_pts = details["human_poly"]["points"]
                    elif human_shape.get("type") == "polygon" and human_shape.get("points"):
                        poly_pts = human_shape["points"]
                    elif human_shape.get("points") and len(human_shape["points"]) >= 6:
                        poly_pts = human_shape["points"]

                    if not poly_pts or len(poly_pts) < 6:
                        # Cannot safely reconstruct region polygon without fabricating; skip example
                        continue

                    norm_poly = []
                    for i in range(0, len(poly_pts) - 1, 2):
                        px = int(round(max(0.0, min(1000.0, (poly_pts[i] - cx1) / cw * 1000.0))))
                        py = int(round(max(0.0, min(1000.0, (poly_pts[i + 1] - cy1) / ch * 1000.0))))
                        norm_poly.append([px, py])

                    if len(norm_poly) < 3:
                        continue

                    reg_item = {
                        "label": target_label,
                        "polygon": norm_poly,
                    }
                    if policy == POLICY_POLYGON_MASK:
                        expected_output = {"regions": [reg_item]}
                    else:
                        expected_output = {
                            "objects": [],
                            "regions": [reg_item],
                            "lanes": [],
                        }

                elif target_group == GROUP_LANE or target_label.startswith("lane/"):
                    if policy in (POLICY_BOX_MASK, POLICY_POLYGON_MASK):
                        continue

                    # Lane Marking: POLICY C -> "polyline" key (NEVER mask, NEVER polygon)
                    line_pts = None
                    if details.get("human_polyline") and details["human_polyline"].get("points"):
                        line_pts = details["human_polyline"]["points"]
                    elif details.get("human_line") and details["human_line"].get("points"):
                        line_pts = details["human_line"]["points"]
                    elif human_shape.get("type") in ("polyline", "line") and human_shape.get("points"):
                        line_pts = human_shape["points"]
                    elif human_shape.get("points") and len(human_shape["points"]) >= 4:
                        line_pts = human_shape["points"]

                    if not line_pts or len(line_pts) < 4:
                        # Cannot safely reconstruct lane polyline without fabricating; skip example
                        continue

                    norm_line = []
                    for i in range(0, len(line_pts) - 1, 2):
                        px = int(round(max(0.0, min(1000.0, (line_pts[i] - cx1) / cw * 1000.0))))
                        py = int(round(max(0.0, min(1000.0, (line_pts[i + 1] - cy1) / ch * 1000.0))))
                        norm_line.append([px, py])

                    if len(norm_line) < 2:
                        continue

                    lane_item = {
                        "label": target_label,
                        "polyline": norm_line,
                    }
                    if policy == POLICY_POLYLINE:
                        expected_output = {"lanes": [lane_item]}
                    else:
                        expected_output = {
                            "objects": [],
                            "regions": [],
                            "lanes": [lane_item],
                        }

                else:
                    if policy in (POLICY_POLYGON_MASK, POLICY_POLYLINE):
                        continue

                    # Instance Object: POLICY A -> BOTH box_2d AND mask (NEVER box-only, NEVER fabricate)
                    rect_pts = None
                    if details.get("human_rect") and details["human_rect"].get("points"):
                        rect_pts = details["human_rect"]["points"]
                    elif human_shape.get("type") == "rectangle" and human_shape.get("points"):
                        rect_pts = human_shape["points"]
                    elif details.get("human_bbox"):
                        rect_pts = details["human_bbox"]
                    elif human_shape.get("points") and len(human_shape["points"]) == 4:
                        rect_pts = human_shape["points"]

                    if not rect_pts or len(rect_pts) < 4:
                        continue

                    xtl, ytl, xbr, ybr = rect_pts[:4]
                    rx1 = int(round(max(0.0, min(1000.0, (xtl - cx1) / cw * 1000.0))))
                    ry1 = int(round(max(0.0, min(1000.0, (ytl - cy1) / ch * 1000.0))))
                    rx2 = int(round(max(0.0, min(1000.0, (xbr - cx1) / cw * 1000.0))))
                    ry2 = int(round(max(0.0, min(1000.0, (ybr - cy1) / ch * 1000.0))))
                    ymin, ymax = min(ry1, ry2), max(ry1, ry2)
                    xmin, xmax = min(rx1, rx2), max(rx1, rx2)
                    if ymax <= ymin or xmax <= xmin:
                        continue
                    box_2d = [ymin, xmin, ymax, xmax]

                    # Extract mask contour points
                    mask_pts = None
                    if details.get("human_mask") and details["human_mask"].get("points"):
                        mask_pts = details["human_mask"]["points"]
                    elif details.get("human_poly") and details["human_poly"].get("points"):
                        mask_pts = details["human_poly"]["points"]
                    elif human_shape.get("type") in ("mask", "polygon") and human_shape.get("points"):
                        mask_pts = human_shape["points"]
                    elif human_shape.get("points") and len(human_shape["points"]) >= 6:
                        mask_pts = human_shape["points"]

                    norm_mask = []
                    if mask_pts and len(mask_pts) >= 6:
                        for i in range(0, len(mask_pts) - 1, 2):
                            px = int(round(max(0.0, min(1000.0, (mask_pts[i] - cx1) / cw * 1000.0))))
                            py = int(round(max(0.0, min(1000.0, (mask_pts[i + 1] - cy1) / ch * 1000.0))))
                            norm_mask.append([px, py])

                    # Policy A ATOMICITY: BOTH box_2d AND mask MUST be present.
                    # If mask is missing (< 3 vertices), skip the visual example completely!
                    # Never send a box-only (or mask-only) instance example.
                    if len(norm_mask) < 3:
                        continue

                    inst_obj: Dict[str, Any] = {
                        "label": target_label,
                        "box_2d": box_2d,
                        "mask": norm_mask,
                    }

                    if policy == POLICY_BOX_MASK:
                        expected_output = {"objects": [inst_obj]}
                    else:
                        expected_output = {
                            "objects": [inst_obj],
                            "regions": [],
                            "lanes": [],
                        }

                if ctype == CORRECTION_RELABEL:
                    description = f"Relabeled: '{ai_lbl}' corrected to human label '{human_lbl}'"
                elif ctype == CORRECTION_ADD_MISSING:
                    description = f"Missed {target_group}: human added omitted '{human_lbl}'"
                elif ctype in (CORRECTION_BOX_MOVE, CORRECTION_BOX_RESIZE):
                    description = f"Box alignment: human adjusted bounding box for '{target_label}'"
                elif ctype in (CORRECTION_MASK_EDIT, CORRECTION_REGION_EDIT):
                    description = f"Contour refinement: human refined segmentation contour for '{target_label}'"
                elif ctype == CORRECTION_LANE_EDIT:
                    description = f"Lane alignment: human refined lane marking for '{target_label}'"
                else:
                    description = f"Human correction: '{target_label}'"

            visual_examples.append({
                "id": item["id"],
                "correction_type": ctype,
                "crop_data_url": crop_data_url,
                "expected_output": expected_output,
                "expected_output_json": json.dumps(expected_output, indent=2),
                "description": description,
                "ai_label": ai_lbl,
                "human_label": human_lbl,
            })

        return visual_examples

    def _format_prompt_extension(
        self,
        rules: List[str],
        examples: List[Dict[str, Any]],
    ) -> str:
        """Format rules and few-shot examples into concise prompt guidance."""
        sections: List[str] = []

        if rules:
            rule_lines = [f"{idx + 1}. {r}" for idx, r in enumerate(rules[:5])]
            sections.append(
                "### Human Annotator Correction Rules (Apply Strictly):\n"
                + "\n".join(rule_lines)
            )

        if examples:
            example_lines: List[str] = []
            for ex in examples:
                ctype = ex["correction_type"]
                ai_lbl = ex.get("ai_label")
                human_lbl = ex.get("human_label")
                raw_iou = ex.get("iou")
                iou = float(raw_iou) if raw_iou is not None else 0.0

                if ctype == CORRECTION_RELABEL:
                    example_lines.append(
                        f"- Correction Example: Object previously predicted as '{ai_lbl}' was relabeled to '{human_lbl}' by human reviewer (IoU={iou:.2f})."
                    )
                elif ctype == CORRECTION_DELETE_FALSE_POSITIVE:
                    example_lines.append(
                        f"- False Positive Example: Object predicted as '{ai_lbl}' was deleted by human reviewer as a false detection."
                    )
                elif ctype == CORRECTION_ADD_MISSING:
                    example_lines.append(
                        f"- Missed Object Example: Annotator manually added missed '{human_lbl}' that model failed to detect."
                    )
                elif ctype in (CORRECTION_BOX_MOVE, CORRECTION_BOX_RESIZE):
                    example_lines.append(
                        f"- Box Adjustment Example: Bounding box for '{ai_lbl}' was adjusted by reviewer for tighter boundary alignment."
                    )
                elif ctype in (CORRECTION_MASK_EDIT, CORRECTION_REGION_EDIT):
                    example_lines.append(
                        f"- Contour Adjustment Example: Segmentation boundary for '{ai_lbl}' was refined by annotator."
                    )
                elif ctype == CORRECTION_LANE_EDIT:
                    example_lines.append(
                        f"- Lane Adjustment Example: Lane marking line for '{ai_lbl}' was realigned by annotator."
                    )

            if example_lines:
                sections.append(
                    "### Recent Human Correction Guidance:\n"
                    + "\n".join(example_lines)
                )

        if not sections:
            return ""

        return (
            "--- Human-in-the-Loop Correction Memory ---\n"
            + "\n\n".join(sections)
            + "\n--------------------------------------------"
        )

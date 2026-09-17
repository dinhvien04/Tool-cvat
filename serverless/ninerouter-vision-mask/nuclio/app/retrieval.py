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
from core.taxonomy import GROUP_INSTANCE, GROUP_LANE, GROUP_REGION, Taxonomy

logger = logging.getLogger(__name__)


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
    ) -> RetrievalResult:
        """Retrieve active derived rules and relevant few-shot correction examples.

        Args:
            candidate_labels: List of active class labels for the current detection task.

        Returns:
            RetrievalResult containing rule strings, example records, visual examples, and formatted prompt text.
        """
        if not self.db.is_enabled():
            return RetrievalResult()

        rules: List[str] = []
        if self.use_rules:
            rules = self.db.get_active_rules(labels=candidate_labels)

        examples: List[Dict[str, Any]] = []
        visual_examples: List[Dict[str, Any]] = []
        if self.use_examples:
            examples = self._select_few_shot_examples(candidate_labels=candidate_labels)
            if self.use_visual_examples:
                visual_examples = self._build_visual_examples(examples)

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
    ) -> List[Dict[str, Any]]:
        """Select top relevant few-shot correction records matching candidate labels."""
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
            params.append(self.max_examples_per_pass)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            selected = []
            for r in rows:
                item = {
                    "id": r["id"],
                    "image_hash": r["image_hash"],
                    "correction_type": r["correction_type"],
                    "ai_label": r["ai_label"],
                    "human_label": r["human_label"],
                    "ai_shape_json": r["ai_shape_json"],
                    "human_shape_json": r["human_shape_json"],
                    "iou": r["iou"],
                    "crop_path": r["crop_path"],
                    "details_json": r["details_json"],
                    "created_at": r["created_at"],
                }
                selected.append(item)
            return selected
        finally:
            conn.close()

    def _build_visual_examples(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build real multimodal few-shot visual examples with crop data URLs and expected annotations.

        Converts human-corrected geometry and labels into structured ground-truth annotations
        matching the 9Router vision contract. Safely skips unreadable, corrupt, or missing crops.
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
            target_label = human_lbl or ai_lbl or "object"
            crop_coords = details.get("crop_coords")  # [cx1, cy1, cx2, cy2] in original image pixels

            taxonomy = Taxonomy()
            deleted_label = ai_lbl or target_label
            deleted_group = taxonomy.get_group(deleted_label)
            target_group = taxonomy.get_group(target_label)

            if ctype == CORRECTION_DELETE_FALSE_POSITIVE:
                # Reviewer deleted false positive; emit the full 3-array contract with empty arrays
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
                # Compute normalized polygon contour if present
                norm_poly = []
                if crop_coords and len(crop_coords) == 4:
                    cx1, cy1, cx2, cy2 = [float(c) for c in crop_coords]
                    cw = max(cx2 - cx1, 1.0)
                    ch = max(cy2 - cy1, 1.0)
                    poly_pts = human_shape.get("points") or []
                    if len(poly_pts) >= 6:
                        for i in range(0, len(poly_pts) - 1, 2):
                            px = int(round(max(0.0, min(1000.0, (poly_pts[i] - cx1) / cw * 1000.0))))
                            py = int(round(max(0.0, min(1000.0, (poly_pts[i + 1] - cy1) / ch * 1000.0))))
                            norm_poly.append([px, py])

                # Route into objects[], regions[], or lanes[] according to 31-label taxonomy
                if target_group == GROUP_REGION or "area/" in target_label:
                    # Semantic Region: mask only, NO box_2d
                    reg_mask = norm_poly if len(norm_poly) >= 3 else [[100, 100], [900, 100], [900, 900], [100, 900]]
                    expected_output = {
                        "objects": [],
                        "regions": [
                            {
                                "label": target_label,
                                "mask": reg_mask,
                            }
                        ],
                        "lanes": [],
                    }
                elif target_group == GROUP_LANE or target_label.startswith("lane/"):
                    # Lane Marking: mask only, NO box_2d
                    lane_mask = norm_poly if len(norm_poly) >= 3 else [[100, 900], [450, 500], [550, 500], [900, 900]]
                    expected_output = {
                        "objects": [],
                        "regions": [],
                        "lanes": [
                            {
                                "label": target_label,
                                "mask": lane_mask,
                            }
                        ],
                    }
                else:
                    # Instance Object: box_2d + optional mask
                    box_2d = [100, 100, 900, 900]  # Centered default fallback
                    if crop_coords and len(crop_coords) == 4:
                        cx1, cy1, cx2, cy2 = [float(c) for c in crop_coords]
                        cw = max(cx2 - cx1, 1.0)
                        ch = max(cy2 - cy1, 1.0)

                        pts = human_shape.get("points") or []
                        if human_shape.get("type") == "rectangle" and len(pts) >= 4:
                            xtl, ytl, xbr, ybr = pts[:4]
                            rx1 = int(round(max(0.0, min(1000.0, (xtl - cx1) / cw * 1000.0))))
                            ry1 = int(round(max(0.0, min(1000.0, (ytl - cy1) / ch * 1000.0))))
                            rx2 = int(round(max(0.0, min(1000.0, (xbr - cx1) / cw * 1000.0))))
                            ry2 = int(round(max(0.0, min(1000.0, (ybr - cy1) / ch * 1000.0))))
                            ymin, ymax = min(ry1, ry2), max(ry1, ry2)
                            xmin, xmax = min(rx1, rx2), max(rx1, rx2)
                            if ymax - ymin < 10:
                                ymax = min(1000, ymin + 10)
                                ymin = max(0, ymax - 10)
                            if xmax - xmin < 10:
                                xmax = min(1000, xmin + 10)
                                xmin = max(0, xmax - 10)
                            box_2d = [ymin, xmin, ymax, xmax]

                    inst_obj: Dict[str, Any] = {
                        "label": target_label,
                        "box_2d": box_2d,
                    }
                    if len(norm_poly) >= 3:
                        inst_obj["mask"] = norm_poly

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

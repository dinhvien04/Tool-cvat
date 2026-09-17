"""Lightweight Few-Shot Retrieval and Adaptive Prompt Injection Engine.

Zero-Heavy-ML Directive:
- Strictly NO local embedding models (no PyTorch, Transformers, Sentence-Transformers).
- Strictly NO local vector databases (no Chroma, FAISS, Milvus).
- Fast, deterministic metadata-driven retrieval based on label taxonomy, confusion pairs,
  quality tiers, and recent correction recency.
- Strictly bounded: 2-4 examples maximum per pass to conserve API tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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


@dataclass
class RetrievalResult:
    """Packaged correction rules and few-shot examples for adaptive inference."""
    rules: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    prompt_extension: str = ""

    def has_content(self) -> bool:
        return bool(self.rules or self.examples or self.prompt_extension)


class CorrectionRetrievalEngine:
    """Selects high-value correction rules and few-shot guidance for current inference."""

    def __init__(
        self,
        db: Optional[FeedbackDatabase] = None,
        max_examples_per_pass: int = 3,
        use_rules: bool = True,
        use_examples: bool = True,
    ):
        self.db = db or FeedbackDatabase()
        self.max_examples_per_pass = max_examples_per_pass
        self.use_rules = use_rules
        self.use_examples = use_examples

    def retrieve(
        self,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        """Retrieve active derived rules and relevant few-shot correction examples.

        Args:
            candidate_labels: List of active class labels for the current detection task.

        Returns:
            RetrievalResult containing rule strings, example records, and formatted prompt text.
        """
        if not self.db.is_enabled():
            return RetrievalResult()

        rules: List[str] = []
        if self.use_rules:
            rules = self.db.get_active_rules(labels=candidate_labels)

        examples: List[Dict[str, Any]] = []
        if self.use_examples:
            examples = self._select_few_shot_examples(candidate_labels=candidate_labels)

        prompt_ext = self._format_prompt_extension(rules, examples)

        return RetrievalResult(
            rules=rules,
            examples=examples,
            prompt_extension=prompt_ext,
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
                    "iou": r["iou"],
                    "crop_path": r["crop_path"],
                    "created_at": r["created_at"],
                }
                selected.append(item)
            return selected
        finally:
            conn.close()

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
                iou = ex.get("iou", 0.0)

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

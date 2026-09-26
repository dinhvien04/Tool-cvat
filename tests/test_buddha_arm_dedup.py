"""Tests for Vector Cosine and Joint Distance Arm Deduplication (core/buddha_contract.py).

Validates:
- True duplicate arm detections are merged / deduplicated.
- Higher confidence instance is retained.
- Parallel adjacent arms with distinct wrists or roots are preserved.
- Hand crop IoU overlap triggers deduplication when joint distances are close.
"""

from __future__ import annotations

import pytest
from core.buddha_contract import (
    BuddhaArmInstance,
    BuddhaArmKeypoint,
    deduplicate_arms,
)


def make_arm(
    arm_id: str,
    root: tuple[float, float],
    elbow: tuple[float, float],
    wrist: tuple[float, float],
    conf: float = 0.90,
    hand_roi: list[int] | None = None,
) -> BuddhaArmInstance:
    return BuddhaArmInstance(
        arm_id=arm_id,
        root=BuddhaArmKeypoint(name="root", x=root[0], y=root[1], visibility=2, confidence=conf),
        elbow=BuddhaArmKeypoint(name="elbow", x=elbow[0], y=elbow[1], visibility=2, confidence=conf),
        wrist=BuddhaArmKeypoint(name="wrist", x=wrist[0], y=wrist[1], visibility=2, confidence=conf),
        confidence=conf,
        hand_roi=hand_roi,
    )


class TestBuddhaArmDeduplication:
    """Validate arm deduplication logic."""

    def test_duplicate_arms_merged_keeping_higher_confidence(self):
        # Two arms with nearly identical coordinates (within 5 pixels)
        arm_low_conf = make_arm("arm_1", (500, 500), (600, 400), (700, 300), conf=0.65)
        arm_high_conf = make_arm("arm_2", (502, 501), (598, 402), (701, 299), conf=0.95)

        deduped = deduplicate_arms([arm_low_conf, arm_high_conf])
        assert len(deduped) == 1
        assert deduped[0].arm_id == "arm_2"
        assert deduped[0].confidence == 0.95

    def test_parallel_adjacent_arms_preserved(self):
        """Two arms radiating outward in parallel with separated wrists must not be merged."""
        # Arm A and Arm B are parallel but separated by 70 pixels at wrist and 50 pixels at root
        arm_a = make_arm("arm_a", (450, 500), (550, 400), (650, 300), conf=0.90)
        arm_b = make_arm("arm_b", (500, 520), (600, 420), (710, 320), conf=0.90)

        deduped = deduplicate_arms([arm_a, arm_b], wrist_dist_thresh=35.0, root_dist_thresh=60.0)
        assert len(deduped) == 2
        ids = {a.arm_id for a in deduped}
        assert ids == {"arm_a", "arm_b"}

    def test_distinct_direction_arms_preserved(self):
        """Two arms sharing a shoulder root but pointing in different directions must be preserved."""
        arm_up = make_arm("arm_up", (500, 500), (500, 400), (500, 300), conf=0.90)
        arm_right = make_arm("arm_right", (500, 500), (600, 500), (700, 500), conf=0.90)

        deduped = deduplicate_arms([arm_up, arm_right])
        assert len(deduped) == 2

    def test_hand_iou_overlap_deduplication(self):
        """Close arms with overlapping hand ROIs (IoU > 0.65) are deduplicated."""
        # Identical hand_roi: [ymin, xmin, ymax, xmax]
        roi = [200, 650, 350, 750]
        arm_1 = make_arm("arm_1", (500, 500), (590, 410), (690, 310), conf=0.80, hand_roi=roi)
        arm_2 = make_arm("arm_2", (510, 510), (600, 400), (700, 300), conf=0.92, hand_roi=roi)

        deduped = deduplicate_arms([arm_1, arm_2], hand_iou_thresh=0.65)
        assert len(deduped) == 1
        assert deduped[0].arm_id == "arm_2"

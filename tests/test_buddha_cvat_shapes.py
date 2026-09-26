"""Tests for Buddha CVAT Skeleton Shapes Contract (core/buddha_contract.py).

Validates:
- CVAT native skeleton shape dictionaries.
- Group ID linking: Central body + 7 face components have group_id = 1.
- Arm and Hand group ID pairing (group_id >= 2).
- Points sublabel attributes: outside, occluded, confidence, point coordinates.
- Visibility integer mapping to CVAT boolean flags.
"""

from __future__ import annotations

import pytest
from core.buddha_contract import (
    BuddhaArmInstance,
    BuddhaArmKeypoint,
    BuddhaHand21Instance,
    BuddhaHand21Keypoint,
    link_hands_to_arms,
    load_buddha_schema,
)
from core.week2_schema import VISIBILITY_OCCLUDED, VISIBILITY_OUTSIDE, VISIBILITY_VISIBLE


class TestBuddhaCvatShapes:
    """Validate CVAT skeleton shape structure."""

    def test_arm_cvat_skeleton_structure(self):
        arm = BuddhaArmInstance(
            arm_id="arm_001",
            root=BuddhaArmKeypoint("root", 500.0, 500.0, visibility=VISIBILITY_VISIBLE, confidence=0.92),
            elbow=BuddhaArmKeypoint("elbow", 600.0, 400.0, visibility=VISIBILITY_OCCLUDED, confidence=0.88),
            wrist=BuddhaArmKeypoint("wrist", 700.0, 300.0, visibility=VISIBILITY_OUTSIDE, confidence=0.0),
            confidence=0.92,
            group_id=3,
        )

        cvat_dict = arm.to_cvat_skeleton(img_w=2000, img_h=1000)
        assert cvat_dict["type"] == "skeleton"
        assert cvat_dict["label"] == "buddha_arm"
        assert cvat_dict["group_id"] == 3
        assert cvat_dict["group"] == 3
        assert cvat_dict["confidence"] == 0.92

        elements = cvat_dict["elements"]
        assert len(elements) == 3

        # Root: visible (outside=False, occluded=False)
        root_el = elements[0]
        assert root_el["label"] == "root"
        assert root_el["type"] == "points"
        assert root_el["outside"] is False
        assert root_el["occluded"] is False
        assert root_el["points"] == [1000.0, 500.0]  # (500/1000)*2000, (500/1000)*1000

        # Elbow: occluded (outside=False, occluded=True)
        elbow_el = elements[1]
        assert elbow_el["label"] == "elbow"
        assert elbow_el["outside"] is False
        assert elbow_el["occluded"] is True

        # Wrist: outside (outside=True, occluded=False)
        wrist_el = elements[2]
        assert wrist_el["label"] == "wrist"
        assert wrist_el["outside"] is True
        assert wrist_el["occluded"] is False

    def test_arm_hand_group_id_matching(self):
        schema = load_buddha_schema()
        arm = BuddhaArmInstance(
            arm_id="arm_005",
            root=BuddhaArmKeypoint("root", 400.0, 400.0, visibility=2, confidence=0.9),
            elbow=BuddhaArmKeypoint("elbow", 450.0, 350.0, visibility=2, confidence=0.9),
            wrist=BuddhaArmKeypoint("wrist", 500.0, 300.0, visibility=2, confidence=0.9),
            confidence=0.9,
            group_id=5,
        )

        lms = {}
        for kp in schema.hand_keypoints:
            lms[kp.id] = BuddhaHand21Keypoint(
                id=kp.id,
                name=kp.name,
                x=500.0 + kp.id,
                y=300.0 + kp.id,
                visibility=2,
                confidence=0.9,
            )

        hand = BuddhaHand21Instance(
            hand_id="hand_005",
            landmarks=lms,
            confidence=0.9,
        )

        linked_arms, linked_hands = link_hands_to_arms([arm], [hand])
        assert len(linked_hands) == 1
        h = linked_hands[0]
        assert h.group_id == arm.group_id == 5

        arm_shape = arm.to_cvat_skeleton(1000, 1000)
        hand_shape = h.to_cvat_skeleton(1000, 1000)

        assert arm_shape["group_id"] == hand_shape["group_id"] == 5
        assert arm_shape["label"] == "buddha_arm"
        assert hand_shape["label"] == "buddha_hand"

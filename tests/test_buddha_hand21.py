"""Tests for Buddha 21-Joint Hand Skeleton and Arm Linking (core/buddha_contract.py).

Validates:
- 21-point hand instance representation and CVAT skeleton serialization.
- Hand geometry assessment (collapse detection, minimum visible points).
- Greedy minimum-distance bipartite hand-arm matching (1:1 parent linking, matching group_id).
- Unlinked hands and arms handling.
"""

from __future__ import annotations

import pytest
from core.buddha_contract import (
    BuddhaArmInstance,
    BuddhaArmKeypoint,
    BuddhaHand21Instance,
    BuddhaHand21Keypoint,
    assess_buddha_hand21_geometry,
    link_hands_to_arms,
    load_buddha_schema,
)
from core.week2_schema import VISIBILITY_OCCLUDED, VISIBILITY_OUTSIDE, VISIBILITY_VISIBLE


def make_hand(
    hand_id: str,
    wrist_x: float,
    wrist_y: float,
    conf: float = 0.90,
    collapse: bool = False,
    visible_count: int = 21,
) -> BuddhaHand21Instance:
    schema = load_buddha_schema()
    landmarks = {}
    for idx, kp in enumerate(schema.hand_keypoints):
        if collapse:
            # All points identical
            x, y = wrist_x, wrist_y
        else:
            # Spread out points
            x = wrist_x + (idx % 5) * 8.0
            y = wrist_y + (idx // 5) * 8.0

        vis = VISIBILITY_VISIBLE if idx < visible_count else VISIBILITY_OUTSIDE
        landmarks[kp.id] = BuddhaHand21Keypoint(
            id=kp.id,
            name=kp.name,
            x=x,
            y=y,
            visibility=vis,
            confidence=conf,
        )
    return BuddhaHand21Instance(
        hand_id=hand_id,
        landmarks=landmarks,
        confidence=conf,
    )


def make_arm(arm_id: str, wx: float, wy: float, group_id: int) -> BuddhaArmInstance:
    return BuddhaArmInstance(
        arm_id=arm_id,
        root=BuddhaArmKeypoint(name="root", x=wx - 100, y=wy - 100, visibility=2, confidence=0.9),
        elbow=BuddhaArmKeypoint(name="elbow", x=wx - 50, y=wy - 50, visibility=2, confidence=0.9),
        wrist=BuddhaArmKeypoint(name="wrist", x=wx, y=wy, visibility=2, confidence=0.9),
        confidence=0.9,
        group_id=group_id,
    )


class TestBuddhaHand21:
    """Validate 21-point hand instances and hand-arm association."""

    def test_hand_cvat_skeleton_serialization(self):
        hand = make_hand("hand_001", 600.0, 400.0)
        hand.group_id = 4
        hand.parent_arm_id = "arm_003"

        cvat_dict = hand.to_cvat_skeleton(img_w=1920, img_h=1080)
        assert cvat_dict["type"] == "skeleton"
        assert cvat_dict["label"] == "buddha_hand"
        assert cvat_dict["group_id"] == 4
        assert cvat_dict["group"] == 4

        elements = cvat_dict["elements"]
        assert len(elements) == 21

        # Check sublabels
        for elem in elements:
            assert elem["type"] == "points"
            assert "outside" in elem
            assert "occluded" in elem
            assert "confidence" in elem
            assert len(elem["points"]) == 2
            # Scaled to pixel space
            px, py = elem["points"]
            assert 0 <= px <= 1920
            assert 0 <= py <= 1080

    def test_hand_geometry_assessment_valid(self):
        hand = make_hand("valid_hand", 500.0, 500.0)
        is_valid, errs = assess_buddha_hand21_geometry(hand)
        assert is_valid
        assert len(errs) == 0

    def test_hand_geometry_assessment_collapsed(self):
        hand = make_hand("collapsed_hand", 500.0, 500.0, collapse=True)
        is_valid, errs = assess_buddha_hand21_geometry(hand)
        assert not is_valid
        assert any("collapsed" in e.lower() for e in errs)

    def test_hand_geometry_too_few_visible(self):
        hand = make_hand("sparse_hand", 500.0, 500.0, visible_count=3)
        is_valid, errs = assess_buddha_hand21_geometry(hand, min_visible_points=5)
        assert not is_valid
        assert any("visible" in e.lower() for e in errs)

    def test_greedy_hand_arm_bipartite_linking(self):
        # 3 arms with wrists at known locations and group_ids
        arm1 = make_arm("arm_1", 300.0, 300.0, group_id=2)
        arm2 = make_arm("arm_2", 600.0, 300.0, group_id=3)
        arm3 = make_arm("arm_3", 900.0, 300.0, group_id=4)

        # 3 hands close to arm wrists (within 15px)
        hand1 = make_hand("hand_near_arm1", 305.0, 308.0)
        hand2 = make_hand("hand_near_arm2", 595.0, 302.0)
        hand3 = make_hand("hand_near_arm3", 902.0, 298.0)

        linked_arms, linked_hands = link_hands_to_arms([arm1, arm2, arm3], [hand1, hand2, hand3])

        assert len(linked_hands) == 3
        # Match each hand by ID and verify matching group_id and parent_arm_id
        h_map = {h.hand_id: h for h in linked_hands}
        assert h_map["hand_near_arm1"].parent_arm_id == "arm_1"
        assert h_map["hand_near_arm1"].group_id == 2

        assert h_map["hand_near_arm2"].parent_arm_id == "arm_2"
        assert h_map["hand_near_arm2"].group_id == 3

        assert h_map["hand_near_arm3"].parent_arm_id == "arm_3"
        assert h_map["hand_near_arm3"].group_id == 4

    def test_unlinked_distant_hand_gets_unique_group_id(self):
        """A hand located far away from any arm wrist should not link to the arm."""
        arm = make_arm("arm_1", 200.0, 200.0, group_id=2)
        # Distant hand at (800, 800) > max_wrist_link_dist (75.0)
        distant_hand = make_hand("hand_far", 800.0, 800.0)

        linked_arms, linked_hands = link_hands_to_arms([arm], [distant_hand], max_wrist_link_dist=75.0)
        assert len(linked_hands) == 1
        h = linked_hands[0]
        assert h.parent_arm_id is None
        assert h.group_id > 2, "Unlinked hand gets an independent group_id"

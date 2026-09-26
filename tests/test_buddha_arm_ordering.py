"""Tests for Deterministic Spatial/Angular Arm Ordering (core/buddha_contract.py).

Validates:
- Clockwise angular sorting from 12 o'clock relative to torso center.
- Invariance to permutation / candidate input order from model JSON.
- Deterministic tie-breaking using radial distance and root coordinates.
- Proper group_id assignment starting at 2.
"""

from __future__ import annotations

import random
import pytest
from core.buddha_contract import (
    BuddhaArmInstance,
    BuddhaArmKeypoint,
    sort_arms_deterministically,
)


def make_arm(
    arm_id: str,
    root_x: float,
    root_y: float,
    wrist_x: float,
    wrist_y: float,
    conf: float = 0.95,
) -> BuddhaArmInstance:
    """Helper to instantiate an arm instance."""
    elbow_x = (root_x + wrist_x) / 2.0
    elbow_y = (root_y + wrist_y) / 2.0
    return BuddhaArmInstance(
        arm_id=arm_id,
        root=BuddhaArmKeypoint(name="root", x=root_x, y=root_y, visibility=2, confidence=conf),
        elbow=BuddhaArmKeypoint(name="elbow", x=elbow_x, y=elbow_y, visibility=2, confidence=conf),
        wrist=BuddhaArmKeypoint(name="wrist", x=wrist_x, y=wrist_y, visibility=2, confidence=conf),
        confidence=conf,
    )


class TestBuddhaArmOrdering:
    """Validate deterministic angular arm ordering."""

    def test_clockwise_angular_order(self):
        torso_center = (500.0, 500.0)

        # 4 arms at cardinal directions relative to (500, 500)
        arm_top = make_arm("arm_top", 500, 400, 500, 200)      # 12 o'clock
        arm_right = make_arm("arm_right", 600, 500, 800, 500)  # 3 o'clock
        arm_bottom = make_arm("arm_bottom", 500, 600, 500, 800) # 6 o'clock
        arm_left = make_arm("arm_left", 400, 500, 200, 500)    # 9 o'clock

        # Input in arbitrary shuffled order
        shuffled = [arm_bottom, arm_left, arm_top, arm_right]
        sorted_arms = sort_arms_deterministically(shuffled, torso_center=torso_center)

        sorted_ids = [a.arm_id for a in sorted_arms]
        assert sorted_ids == ["arm_top", "arm_right", "arm_bottom", "arm_left"]

    def test_permutation_invariance(self):
        """Order must be identical regardless of random shuffle permutations."""
        torso_center = (500.0, 500.0)

        arms = [
            make_arm(f"arm_{i}", 500 + i * 2, 500 + i * 3, 300 + (i * 47) % 400, 200 + (i * 83) % 600)
            for i in range(12)
        ]

        canonical_sorted = sort_arms_deterministically(arms, torso_center=torso_center)
        canonical_ids = [a.arm_id for a in canonical_sorted]

        # Shuffle 10 times and verify identical sort result
        rng = random.Random(42)
        for _ in range(10):
            shuffled = list(arms)
            rng.shuffle(shuffled)
            sorted_res = sort_arms_deterministically(shuffled, torso_center=torso_center)
            assert [a.arm_id for a in sorted_res] == canonical_ids

    def test_tie_breaking_radial_and_root(self):
        """Arms with identical wrist angles must tie-break deterministically."""
        torso_center = (500.0, 500.0)

        # Two arms directly above center (same angle 0), one with wrist further out
        arm_far = make_arm("arm_far", 500, 420, 500, 100)
        arm_near = make_arm("arm_near", 500, 450, 500, 250)

        sorted_arms = sort_arms_deterministically([arm_far, arm_near], torso_center=torso_center)
        # Nearer arm (smaller radial dist) comes first
        assert [a.arm_id for a in sorted_arms] == ["arm_near", "arm_far"]

    def test_group_id_assignment(self):
        """Sorted arms must receive unique incremental group_ids starting from 2."""
        arms = [
            make_arm("a1", 500, 400, 500, 200),
            make_arm("a2", 600, 500, 800, 500),
            make_arm("a3", 400, 500, 200, 500),
        ]
        sorted_arms = sort_arms_deterministically(arms)
        group_ids = [a.group_id for a in sorted_arms]
        assert group_ids == [2, 3, 4]

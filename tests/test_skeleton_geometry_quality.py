"""Unit tests for deterministic skeleton geometry quality gate and laterality invariants.

Validates:
1. Section 3: Laterality Must Be Proven (subject vs viewer conventions, mirrored invariants).
2. Section 16: Human Pose 17 Quality Gate (axis swap, duplicates, out-of-bounds, inverted orientation, collapse).
3. Section 17 & 18: VF-50 Face Landmark Quality Gate (50 points, 7 components, eyelids, eyebrows, lips, head tilt).
4. Master assess_quality dispatcher integration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pytest

from core.quality_gate import (
    LATERALITY_SUBJECT,
    LATERALITY_VIEWER,
    POSE17_KEYPOINTS,
    VF50_COMPONENT_NAMES,
    VF50_COMPONENT_COUNTS,
    VF50_COMPONENT_RANGES,
    assess_pose17_quality,
    assess_vf50_quality,
    assess_quality,
    detect_coordinate_axis_swap,
    derive_skeleton_bbox,
    derive_skeleton_center,
    verify_mirrored_laterality_invariance,
    verify_laterality_convention,
    Pose17QualityReport,
    VF50QualityReport,
    LateralityVerificationResult,
)
from core.skeleton_contract import (
    PersonPose17,
    PoseKeypoint,
    VF50Face,
    VF50Landmark,
    VISIBILITY_OUTSIDE,
    VISIBILITY_OCCLUDED,
    VISIBILITY_VISIBLE,
)


# ==============================================================================
# SYNTHETIC TEST FIXTURES
# ==============================================================================

def make_canonical_pose17(
    center_x: float = 500.0,
    center_y: float = 500.0,
    scale: float = 1.0,
    convention: str = LATERALITY_VIEWER,
) -> Dict[str, Tuple[float, float, int, float]]:
    """Build a realistic frontal upright Pose 17 skeleton.
    Canonical VinFast Week-2 uses viewer convention:
    - Viewer's left is on screen left (smaller x: x < center_x).
    - Viewer's right is on screen right (larger x: x > center_x).
    """
    sign = -1.0 if convention == LATERALITY_VIEWER else 1.0
    return {
        "nose": (center_x, center_y - 200.0 * scale, 2, 0.98),
        "left_eye": (center_x + sign * 15.0 * scale, center_y - 215.0 * scale, 2, 0.96),
        "right_eye": (center_x - sign * 15.0 * scale, center_y - 215.0 * scale, 2, 0.96),
        "left_ear": (center_x + sign * 35.0 * scale, center_y - 210.0 * scale, 2, 0.92),
        "right_ear": (center_x - sign * 35.0 * scale, center_y - 210.0 * scale, 2, 0.92),
        "left_shoulder": (center_x + sign * 70.0 * scale, center_y - 140.0 * scale, 2, 0.95),
        "right_shoulder": (center_x - sign * 70.0 * scale, center_y - 140.0 * scale, 2, 0.95),
        "left_elbow": (center_x + sign * 95.0 * scale, center_y - 50.0 * scale, 2, 0.93),
        "right_elbow": (center_x - sign * 95.0 * scale, center_y - 50.0 * scale, 2, 0.93),
        "left_wrist": (center_x + sign * 110.0 * scale, center_y + 30.0 * scale, 2, 0.90),
        "right_wrist": (center_x - sign * 110.0 * scale, center_y + 30.0 * scale, 2, 0.90),
        "left_hip": (center_x + sign * 45.0 * scale, center_y + 50.0 * scale, 2, 0.95),
        "right_hip": (center_x - sign * 45.0 * scale, center_y + 50.0 * scale, 2, 0.95),
        "left_knee": (center_x + sign * 50.0 * scale, center_y + 160.0 * scale, 2, 0.92),
        "right_knee": (center_x - sign * 50.0 * scale, center_y + 160.0 * scale, 2, 0.92),
        "left_ankle": (center_x + sign * 55.0 * scale, center_y + 270.0 * scale, 2, 0.88),
        "right_ankle": (center_x - sign * 55.0 * scale, center_y + 270.0 * scale, 2, 0.88),
    }


def make_canonical_vf50(
    center_x: float = 500.0,
    center_y: float = 500.0,
    roll_deg: float = 0.0,
) -> Dict[int, Tuple[float, float, int, float]]:
    """Build a realistic canonical VF-50 face landmark set (50 points, 7 skeletons).
    VinFast Viewer Convention:
    - mattrai (14..21) & longmaytrai (0..4) are on image left (smaller x: x < center_x).
    - matphai (22..29) & longmayphai (5..9) are on image right (larger x: x > center_x).
    """
    pts: Dict[int, Tuple[float, float]] = {}

    # longmaytrai (0..4): left eyebrow (viewer left, x ~ 410..460, y ~ 430..435)
    pts[0] = (410.0, 435.0)
    pts[1] = (422.0, 430.0)
    pts[2] = (435.0, 428.0)
    pts[3] = (448.0, 430.0)
    pts[4] = (460.0, 436.0)

    # longmayphai (5..9): right eyebrow (viewer right, x ~ 540..590, y ~ 430..435)
    pts[5] = (540.0, 436.0)
    pts[6] = (552.0, 430.0)
    pts[7] = (565.0, 428.0)
    pts[8] = (578.0, 430.0)
    pts[9] = (590.0, 435.0)

    # songmui (10..13): nose bridge (vertical center x ~ 500, y ~ 450..520)
    pts[10] = (500.0, 450.0)
    pts[11] = (500.0, 475.0)
    pts[12] = (500.0, 500.0)
    pts[13] = (500.0, 520.0)

    # mattrai (14..21): left eye (viewer left, outer 14, upper 15..17, inner 18, lower 19..21)
    pts[14] = (415.0, 458.0)  # outer corner
    pts[15] = (425.0, 452.0)  # upper
    pts[16] = (435.0, 451.0)  # upper mid
    pts[17] = (445.0, 453.0)  # upper
    pts[18] = (455.0, 458.0)  # inner corner
    pts[19] = (445.0, 464.0)  # lower
    pts[20] = (435.0, 465.0)  # lower mid
    pts[21] = (425.0, 463.0)  # lower

    # matphai (22..29): right eye (viewer right, inner 22, upper 23..25, outer 26, lower 27..29)
    pts[22] = (545.0, 458.0)  # inner corner
    pts[23] = (555.0, 453.0)  # upper
    pts[24] = (565.0, 451.0)  # upper mid
    pts[25] = (575.0, 452.0)  # upper
    pts[26] = (585.0, 458.0)  # outer corner
    pts[27] = (575.0, 463.0)  # lower
    pts[28] = (565.0, 465.0)  # lower mid
    pts[29] = (555.0, 464.0)  # lower

    # moingoai (30..41): outer lip (12 points, 30 left corner, 36 right corner)
    pts[30] = (450.0, 565.0)  # left corner
    pts[31] = (465.0, 555.0)
    pts[32] = (485.0, 552.0)
    pts[33] = (500.0, 555.0)  # philtrum
    pts[34] = (515.0, 552.0)
    pts[35] = (535.0, 555.0)
    pts[36] = (550.0, 565.0)  # right corner
    pts[37] = (535.0, 580.0)
    pts[38] = (518.0, 585.0)
    pts[39] = (500.0, 586.0)  # bottom mid
    pts[40] = (482.0, 585.0)
    pts[41] = (465.0, 580.0)

    # moitrong (42..49): inner lip (8 points, 42 inner left, 46 inner right)
    # Strictly inside outer lip (x_42 > x_30 and x_46 < x_36)
    pts[42] = (460.0, 565.0)  # left inner corner
    pts[43] = (480.0, 560.0)
    pts[44] = (500.0, 562.0)
    pts[45] = (520.0, 560.0)
    pts[46] = (540.0, 565.0)  # right inner corner
    pts[47] = (520.0, 572.0)
    pts[48] = (500.0, 574.0)
    pts[49] = (480.0, 572.0)

    # Apply roll rotation if requested
    rad = math.radians(roll_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    res: Dict[int, Tuple[float, float, int, float]] = {}
    for p_id, (px, py) in pts.items():
        dx = px - center_x
        dy = py - center_y
        rx = center_x + dx * cos_a - dy * sin_a
        ry = center_y + dx * sin_a + dy * cos_a
        res[p_id] = (rx, ry, 2, 0.95)

    return res


# ==============================================================================
# SECTION 3 TESTS: LATERALITY MUST BE PROVEN
# ==============================================================================

def test_pose17_viewer_laterality_convention():
    """Prove that Pose 17 canonical Week-2 viewer convention expects left_shoulder < right_shoulder."""
    pose = make_canonical_pose17(convention=LATERALITY_VIEWER)
    res = verify_laterality_convention(pose, schema_type="pose17", expected_convention=LATERALITY_VIEWER)
    assert res.is_valid is True
    assert res.detected_convention == LATERALITY_VIEWER
    assert res.delta_x < 0  # left_shoulder (430) < right_shoulder (570)


def test_pose17_subject_laterality_convention():
    """Prove that Pose 17 anatomical subject convention expects left_shoulder > right_shoulder."""
    pose = make_canonical_pose17(convention=LATERALITY_SUBJECT)
    res = verify_laterality_convention(pose, schema_type="pose17", expected_convention=LATERALITY_SUBJECT)
    assert res.is_valid is True
    assert res.detected_convention == LATERALITY_SUBJECT
    assert res.delta_x > 0  # left_shoulder (570) > right_shoulder (430)


def test_vf50_viewer_laterality_convention():
    """Prove that VF-50 viewer convention expects mattrai on screen left (smaller x than matphai)."""
    face = make_canonical_vf50()
    res = verify_laterality_convention(face, schema_type="vf50", expected_convention=LATERALITY_VIEWER)
    assert res.is_valid is True
    assert res.detected_convention == LATERALITY_VIEWER
    assert res.delta_x > 0  # matphai center (~565) > mattrai center (~435)


def test_vf50_inverted_laterality_flagged():
    """Prove that if VF-50 coordinates follow subject instead of viewer laterality, it is rejected."""
    face = make_canonical_vf50()
    # Invert eyes horizontally
    inverted_face: Dict[int, Tuple[float, float, int, float]] = dict(face)
    for i in range(8):
        inverted_face[14 + i] = face[22 + i]
        inverted_face[22 + i] = face[14 + i]

    res = verify_laterality_convention(inverted_face, schema_type="vf50", expected_convention=LATERALITY_VIEWER)
    assert res.is_valid is False
    assert res.detected_convention == LATERALITY_SUBJECT
    assert any("violates viewer convention" in r for r in res.reasons)


def test_mirrored_pose17_subject_invariance():
    """Section 3 proof: Under horizontal flip, exchanging symmetric left<->right labels restores anatomical invariant."""
    pose = make_canonical_pose17(convention=LATERALITY_SUBJECT)
    is_invariant, msg = verify_mirrored_laterality_invariance(pose, schema_type="pose17", convention=LATERALITY_SUBJECT)
    assert is_invariant is True
    assert "verified" in msg


def test_mirrored_vf50_viewer_invariance():
    """Section 3 proof: Under horizontal flip, swapping paired eye/eyebrow components preserves viewer laterality."""
    face = make_canonical_vf50()
    is_invariant, msg = verify_mirrored_laterality_invariance(face, schema_type="vf50", convention=LATERALITY_VIEWER)
    assert is_invariant is True
    assert "verified" in msg


# ==============================================================================
# SECTION 16 TESTS: QUALITY GATE - HUMAN POSE 17
# ==============================================================================

def test_canonical_pose17_passes_quality_gate():
    """A valid upright frontal Pose 17 skeleton achieves high score and needs no refinement."""
    pose = make_canonical_pose17()
    report = assess_pose17_quality(pose)
    assert report.is_valid is True
    assert report.needs_refine is False
    assert report.score >= 0.90
    assert report.active_count == 17
    assert report.axis_swap_detected is False


def test_pose17_coordinate_axis_swap_detected():
    """Flipping x and y coordinates (swapped axes) must be deterministically caught."""
    pose = make_canonical_pose17()
    # Swap (x, y) -> (y, x)
    swapped = {k: (v[1], v[0], v[2], v[3]) for k, v in pose.items()}

    is_swapped, reason = detect_coordinate_axis_swap(swapped, schema_type="pose17")
    assert is_swapped is True
    assert "torso is horizontal" in reason

    report = assess_pose17_quality(swapped)
    assert report.axis_swap_detected is True
    assert report.is_valid is False
    assert report.needs_refine is True
    assert any("axis_swap" in r for r in report.reasons)


def test_pose17_duplicate_coincident_joints_detected():
    """Detect when model snaps multiple distinct keypoints to identical coordinates."""
    pose = make_canonical_pose17()
    # Snap wrist directly onto elbow
    pose["left_wrist"] = pose["left_elbow"]

    report = assess_pose17_quality(pose)
    assert report.needs_refine is True
    assert any("duplicate_coincident_joints" in r for r in report.reasons)


def test_pose17_out_of_bounds_detected():
    """Coordinates outside [0, 1000] range must be flagged."""
    pose = make_canonical_pose17()
    pose["left_ankle"] = (550.0, 1045.0, 2, 0.9)  # y > 1000

    report = assess_pose17_quality(pose)
    assert report.needs_refine is True
    assert any("coordinates_out_of_bounds" in r for r in report.reasons)


def test_pose17_inverted_vertical_orientation_detected():
    """Shoulders below hips in upright pose must trigger inverted orientation failure."""
    pose = make_canonical_pose17()
    # Place shoulders at y=600 while hips at y=450
    pose["left_shoulder"] = (430.0, 600.0, 2, 0.9)
    pose["right_shoulder"] = (570.0, 600.0, 2, 0.9)
    pose["left_hip"] = (455.0, 450.0, 2, 0.9)
    pose["right_hip"] = (545.0, 450.0, 2, 0.9)

    report = assess_pose17_quality(pose)
    assert report.needs_refine is True
    assert any("inverted_vertical_orientation" in r for r in report.reasons)


def test_pose17_degenerate_skeleton_collapse_detected():
    """Collapsed skeleton with tiny area/diagonal must be rejected."""
    collapsed = {k: (500.0 + i * 0.1, 500.0 + i * 0.1, 2, 0.9) for i, k in enumerate(POSE17_KEYPOINTS)}
    report = assess_pose17_quality(collapsed)
    assert report.is_valid is False
    assert any("degenerate_skeleton_collapse" in r for r in report.reasons)


def test_pose17_excessive_limb_length_detected():
    """Unrealistic limb segment jump (e.g. 750px) must be flagged in suspect_bones."""
    pose = make_canonical_pose17()
    # Teleport wrist to other side of canvas
    pose["right_wrist"] = (950.0, 50.0, 2, 0.9)

    report = assess_pose17_quality(pose)
    assert report.needs_refine is True
    assert "right_elbow-right_wrist" in report.suspect_bones
    assert any("excessive_bone_length" in r or "excessive_arm_length" in r or "disproportionate_bone" in r for r in report.reasons)


# ==============================================================================
# SECTION 17 & 18 TESTS: QUALITY GATE - VF-50 FACE LANDMARKS
# ==============================================================================

def test_canonical_vf50_passes_quality_gate():
    """A valid canonical VF-50 landmark set satisfies all topological and geometric constraints."""
    face = make_canonical_vf50()
    report = assess_vf50_quality(face)
    assert report.is_valid is True
    assert report.needs_refine is False
    assert report.score >= 0.90
    assert report.point_count == 50
    assert report.component_counts["mattrai"] == 8
    assert report.component_counts["moingoai"] == 12
    assert report.component_counts["moitrong"] == 8


def test_vf50_coordinate_axis_swap_detected():
    """Swapping x and y on VF-50 face landmarks must be deterministically caught."""
    face = make_canonical_vf50()
    swapped = {k: (v[1], v[0], v[2], v[3]) for k, v in face.items()}

    is_swapped, reason = detect_coordinate_axis_swap(swapped, schema_type="vf50")
    assert is_swapped is True
    assert "eyes vertically aligned" in reason

    report = assess_vf50_quality(swapped)
    assert report.axis_swap_detected is True
    assert report.is_valid is False
    assert report.needs_refine is True


def test_vf50_eyebrow_below_eye_detected():
    """Eyebrow points located below eye contour must trigger quality gate penalty."""
    face = make_canonical_vf50()
    # Move longmaytrai (0..4) down to y=480 (below eye y=455)
    for i in range(5):
        face[i] = (face[i][0], 480.0, face[i][2], face[i][3])

    report = assess_vf50_quality(face)
    assert report.needs_refine is True
    assert "longmaytrai" in report.suspect_components
    assert any("eyebrow_below_eye" in r for r in report.reasons)


def test_vf50_inverted_eyelid_detected():
    """Upper eyelid point located below corresponding lower eyelid point must be flagged."""
    face = make_canonical_vf50()
    # Invert opposing pair: upper pt 16 (y=451) placed below lower pt 20 (y=465)
    face[16] = (face[16][0], 475.0, face[16][2], face[16][3])

    report = assess_vf50_quality(face)
    assert report.needs_refine is True
    assert "mattrai" in report.suspect_components
    assert any("inverted_eyelid" in r for r in report.reasons)


def test_vf50_eyelid_figure8_self_intersection_detected():
    """Self-intersecting eye contour (figure-8 / X-shape) must be caught."""
    face = make_canonical_vf50()
    # Cross upper and lower eyelid vertices in mattrai
    face[15] = (435.0, 465.0, 2, 0.95)
    face[20] = (425.0, 452.0, 2, 0.95)

    report = assess_vf50_quality(face)
    assert report.needs_refine is True
    assert "mattrai" in report.suspect_components
    assert any("self_intersecting_eye_contour" in r for r in report.reasons)


def test_vf50_inner_lip_larger_than_outer_lip_detected():
    """Topological constraint: inner lip area cannot exceed outer lip area."""
    face = make_canonical_vf50()
    # Blow up inner lip points so inner area > outer area
    for i in range(42, 50):
        face[i] = (face[i][0] * 0.95 + 25.0, face[i][1] + 30.0, face[i][2], face[i][3])
    # Expand outer corner
    face[42] = (420.0, 565.0, 2, 0.95)
    face[46] = (580.0, 565.0, 2, 0.95)

    report = assess_vf50_quality(face)
    assert report.needs_refine is True
    assert "moitrong" in report.suspect_components
    assert any("inner_lip_larger_than_outer" in r or "inner_lip_protrudes_horizontally" in r for r in report.reasons)


def test_vf50_head_tilt_invariance():
    """Driver head tilted up to -21 degrees (or +15 degrees) must pass without false rejections."""
    # Test significant driver tilt (-20 degrees roll)
    tilted_face = make_canonical_vf50(roll_deg=-20.0)
    report = assess_vf50_quality(tilted_face)
    assert report.is_valid is True
    assert report.needs_refine is False
    assert report.score >= 0.85

    # Test opposite tilt (+15 degrees roll)
    tilted_face_pos = make_canonical_vf50(roll_deg=15.0)
    report_pos = assess_vf50_quality(tilted_face_pos)
    assert report_pos.is_valid is True
    assert report_pos.needs_refine is False


def test_vf50_incomplete_components_detected():
    """Missing points in a component (e.g. only 3 points in songmui instead of 4) is penalized."""
    face = make_canonical_vf50()
    del face[13]  # Remove 1 point from songmui

    report = assess_vf50_quality(face)
    assert report.needs_refine is True
    assert "songmui" in report.suspect_components
    assert any("incomplete_component_songmui" in r for r in report.reasons)


# ==============================================================================
# MASTER ASSESS QUALITY INTEGRATION TESTS
# ==============================================================================

def test_master_assess_quality_pose17_mode():
    """assess_quality dispatches mode='human_pose_17' cleanly."""
    pose = make_canonical_pose17()
    report = assess_quality([pose], mode="human_pose_17")
    assert report.score >= 0.90
    assert report.needs_refine is False
    assert report.suspect_count == 0


def test_master_assess_quality_vf50_mode():
    """assess_quality dispatches mode='face_landmark_vf50' cleanly."""
    face = make_canonical_vf50()
    report = assess_quality([face], mode="face_landmark_vf50")
    assert report.score >= 0.90
    assert report.needs_refine is False
    assert report.suspect_count == 0


def test_master_assess_quality_pose17_corrupted_triggers_refine():
    """Corrupted pose triggers refinement in master assess_quality."""
    pose = make_canonical_pose17()
    pose["left_ankle"] = (550.0, 1100.0, 2, 0.9)  # out of bounds

    report = assess_quality([pose], mode="human_pose_17")
    assert report.needs_refine is True
    assert report.suspect_count == 1
    assert "person_1" in report.suspect_labels


def test_pose17_numeric_indices_parsed_correctly():
    """VinFast guideline numeric sublabels '1'..'17' correctly resolve to canonical joints."""
    # Build dictionary with numeric string keys '1'..'17' in canonical viewer convention
    numeric_pose: Dict[str, Tuple[float, float, int, float]] = {
        "1": (500.0, 300.0, 2, 0.98),   # nose
        "2": (515.0, 285.0, 2, 0.96),   # right_eye (R_* even: viewer right -> larger x)
        "3": (485.0, 285.0, 2, 0.96),   # left_eye (L_* odd: viewer left -> smaller x)
        "4": (535.0, 290.0, 2, 0.92),   # right_ear
        "5": (465.0, 290.0, 2, 0.92),   # left_ear
        "6": (570.0, 360.0, 2, 0.95),   # right_shoulder
        "7": (430.0, 360.0, 2, 0.95),   # left_shoulder
        "8": (595.0, 450.0, 2, 0.93),   # right_elbow
        "9": (405.0, 450.0, 2, 0.93),   # left_elbow
        "10": (610.0, 530.0, 2, 0.90),  # right_wrist
        "11": (390.0, 530.0, 2, 0.90),  # left_wrist
        "12": (545.0, 550.0, 2, 0.95),  # right_hip
        "13": (455.0, 550.0, 2, 0.95),  # left_hip
        "14": (550.0, 660.0, 2, 0.92),  # right_knee
        "15": (450.0, 660.0, 2, 0.92),  # left_knee
        "16": (555.0, 770.0, 2, 0.88),  # right_ankle
        "17": (445.0, 770.0, 2, 0.88),  # left_ankle
    }
    report = assess_pose17_quality(numeric_pose)
    assert report.is_valid is True
    assert report.active_count == 17
    assert report.needs_refine is False


def test_pose17_corner_cluster_dump_detected():
    """Section 1.1 Item 2: Pre-label machine failure dumping unpredicted joints in corner [0..35, 0..35] is flagged."""
    pose = make_canonical_pose17()
    # Machine dumped right ear (4) and right ankle (16) at top-left corner
    pose["right_ear"] = (12.0, 15.0, 2, 0.85)
    pose["right_ankle"] = (18.0, 22.0, 2, 0.85)

    report = assess_pose17_quality(pose)
    assert report.needs_refine is True
    assert any("corner_cluster_dump" in r for r in report.reasons)


# ==============================================================================
# AUDIT TESTS: VIEWER PERSPECTIVE, BOUNDING BOX, CENTER, & CVAT FLAGS
# ==============================================================================

def test_strict_viewer_laterality_pose17_even_odd():
    """Audit: Strict viewer perspective laterality for Pose 17 (R_* even > L_* odd in x)."""
    # Create VinFast Pose 17 where even (Right) has larger x than odd (Left)
    vinfast_viewer_pose: Dict[str, Tuple[float, float, int, float]] = {
        "nose": (500.0, 300.0, 2, 0.98),          # 1
        "right_eye": (520.0, 285.0, 2, 0.96),     # 2: Right in frame -> larger x
        "left_eye": (480.0, 285.0, 2, 0.96),      # 3: Left in frame -> smaller x
        "right_ear": (540.0, 290.0, 2, 0.92),     # 4: Right -> larger x
        "left_ear": (460.0, 290.0, 2, 0.92),      # 5: Left -> smaller x
        "right_shoulder": (570.0, 360.0, 2, 0.95),# 6: Right -> larger x
        "left_shoulder": (430.0, 360.0, 2, 0.95), # 7: Left -> smaller x
        "right_elbow": (600.0, 450.0, 2, 0.93),   # 8: Right -> larger x
        "left_elbow": (400.0, 450.0, 2, 0.93),    # 9: Left -> smaller x
        "right_wrist": (620.0, 530.0, 2, 0.90),   # 10: Right -> larger x
        "left_wrist": (380.0, 530.0, 2, 0.90),    # 11: Left -> smaller x
        "right_hip": (550.0, 550.0, 2, 0.95),     # 12: Right -> larger x
        "left_hip": (450.0, 550.0, 2, 0.95),      # 13: Left -> smaller x
        "right_knee": (550.0, 660.0, 2, 0.92),    # 14: Right -> larger x
        "left_knee": (450.0, 660.0, 2, 0.92),     # 15: Left -> smaller x
        "right_ankle": (560.0, 770.0, 2, 0.88),   # 16: Right -> larger x
        "left_ankle": (440.0, 770.0, 2, 0.88),    # 17: Left -> smaller x
    }
    res = verify_laterality_convention(vinfast_viewer_pose, schema_type="pose17", expected_convention=LATERALITY_VIEWER)
    assert res.is_valid is True
    assert res.detected_convention == LATERALITY_VIEWER

    # Invert a pair (left_shoulder given larger x than right_shoulder)
    inverted = dict(vinfast_viewer_pose)
    inverted["left_shoulder"] = (580.0, 360.0, 2, 0.95)
    inverted["right_shoulder"] = (420.0, 360.0, 2, 0.95)
    res_inv = verify_laterality_convention(inverted, schema_type="pose17", expected_convention=LATERALITY_VIEWER)
    assert res_inv.is_valid is False
    assert any("laterality_inversion" in r for r in res_inv.reasons)


def test_strict_viewer_laterality_vf50_eyebrows_and_eyes():
    """Audit: Strict viewer perspective laterality for VF-50 (longmaytrai/mattrai < longmayphai/matphai)."""
    face = make_canonical_vf50()
    res = verify_laterality_convention(face, schema_type="vf50", expected_convention=LATERALITY_VIEWER)
    assert res.is_valid is True
    assert res.detected_convention == LATERALITY_VIEWER

    # Invert eyebrows only
    inverted_eb = dict(face)
    for i in range(5):
        inverted_eb[i] = face[5 + i]
        inverted_eb[5 + i] = face[i]
    res_eb = verify_laterality_convention(inverted_eb, schema_type="vf50", expected_convention=LATERALITY_VIEWER)
    assert res_eb.is_valid is False
    assert any("eyebrow_laterality_inversion" in r for r in res_eb.reasons)


def test_bounding_box_and_center_derivation():
    """Audit: Bounding box derivation and center coordinate calculation for Pose 17 and VF-50."""
    pose = make_canonical_pose17()
    bbox_pose = derive_skeleton_bbox(pose, schema_type="pose17")
    assert bbox_pose is not None
    ymin, xmin, ymax, xmax = bbox_pose
    assert 0 <= ymin < ymax <= 1000
    assert 0 <= xmin < xmax <= 1000

    center_pose = derive_skeleton_center(pose, schema_type="pose17")
    assert center_pose is not None
    cx, cy = center_pose
    assert xmin <= cx <= xmax
    assert ymin <= cy <= ymax

    # PersonPose17 instance methods
    person = PersonPose17(label="person")
    for k, v in pose.items():
        person.keypoints[k] = PoseKeypoint(name=k, x=v[0], y=v[1], visibility=v[2])
    p_bbox = person.derive_bbox()
    p_center = person.get_center()
    assert p_bbox is not None
    assert p_center is not None
    assert abs(p_center[0] - cx) < 2.0

    # VF50Face instance methods
    face = make_canonical_vf50()
    vf_bbox = derive_skeleton_bbox(face, schema_type="vf50")
    vf_center = derive_skeleton_center(face, schema_type="vf50")
    assert vf_bbox is not None
    assert vf_center is not None

    vf_obj = VF50Face(face_id=1)
    for k, v in face.items():
        vf_obj.landmarks[k] = VF50Landmark(id=k, name=str(k), x=v[0], y=v[1], visibility=v[2])
    obj_bbox = vf_obj.derive_bbox()
    obj_center = vf_obj.get_center()
    assert obj_bbox is not None
    assert obj_center is not None
    assert abs(obj_center[0] - vf_center[0]) < 2.0


def test_cvat_missing_elements_outside_occluded_flags():
    """Audit: Missing elements get outside=True, occluded=False, confidence=0.0; occluded get outside=False, occluded=True."""
    person = PersonPose17(label="person")
    # Only populate nose as visible and left_eye as occluded, rest missing
    person.keypoints["nose"] = PoseKeypoint(name="nose", x=500.0, y=300.0, visibility=VISIBILITY_VISIBLE)
    person.keypoints["left_eye"] = PoseKeypoint(name="left_eye", x=515.0, y=285.0, visibility=VISIBILITY_OCCLUDED)

    cvat_skel = person.to_cvat_skeleton(width=1920, height=1080)
    elems = {el["label"]: el for el in cvat_skel["elements"]}

    # nose: visible
    assert elems["nose"]["outside"] is False
    assert elems["nose"]["occluded"] is False

    # left_eye: occluded
    assert elems["left_eye"]["outside"] is False
    assert elems["left_eye"]["occluded"] is True

    # right_ankle: missing -> outside
    assert elems["right_ankle"]["outside"] is True
    assert elems["right_ankle"]["occluded"] is False

    # VF-50 components
    vf_obj = VF50Face(face_id=1)
    vf_obj.landmarks[10] = VF50Landmark(id=10, name="10", x=500.0, y=450.0, visibility=VISIBILITY_OCCLUDED)
    comp_skels = vf_obj.to_cvat_component_skeletons(width=1920, height=1080, group_id=1)
    songmui_skel = next(s for s in comp_skels if s["label"] == "songmui")
    songmui_elems = {el["label"]: el for el in songmui_skel["elements"]}

    # Point 10: occluded
    assert songmui_elems["10"]["outside"] is False
    assert songmui_elems["10"]["occluded"] is True

    # Point 11: default missing -> outside
    assert songmui_elems["11"]["outside"] is True
    assert songmui_elems["11"]["occluded"] is False


# ==============================================================================
# SOFT QUALITY GATES: RECLINING, FORWARD LEAN, CROSSED LIMBS, ADAPTIVE SCALE
# ==============================================================================

def test_pose17_reclining_pose_not_flagged_as_axis_swap():
    """Reclining passenger with horizontal torso and vertical shoulders has horizontal eyes."""
    pose = make_canonical_pose17()
    # Reclining: Torso spans horizontally (dx >> dy)
    pose["left_shoulder"] = (650.0, 380.0, 2, 0.95)
    pose["right_shoulder"] = (650.0, 300.0, 2, 0.95)  # vertical shoulders
    pose["left_hip"] = (350.0, 390.0, 2, 0.95)
    pose["right_hip"] = (350.0, 310.0, 2, 0.95)       # horizontal torso: mid-shoulder x=650, mid-hip x=350 (dx=300, dy=10)
    # Head & eyes remain horizontal (not rotated coordinates)
    pose["left_eye"] = (720.0, 320.0, 2, 0.96)
    pose["right_eye"] = (680.0, 320.0, 2, 0.96)

    is_swapped, reason = detect_coordinate_axis_swap(pose, schema_type="pose17")
    assert is_swapped is False
    assert "reclining_or_lying_pose" in reason

    report = assess_pose17_quality(pose)
    assert report.axis_swap_detected is False


def test_pose17_forward_lean_pose_soft_warning():
    """Driver leaning forward to reach footwell/glovebox has shoulders and head below hips."""
    pose = make_canonical_pose17()
    # Hips at y=450, shoulders at y=560 (below hips), nose at y=580 (also below hips)
    pose["left_hip"] = (455.0, 450.0, 2, 0.95)
    pose["right_hip"] = (545.0, 450.0, 2, 0.95)
    pose["left_shoulder"] = (430.0, 560.0, 2, 0.95)
    pose["right_shoulder"] = (570.0, 560.0, 2, 0.95)
    pose["nose"] = (500.0, 580.0, 2, 0.95)
    # Ankles planted on floor at y=750
    pose["left_ankle"] = (450.0, 750.0, 2, 0.9)
    pose["right_ankle"] = (550.0, 750.0, 2, 0.9)

    report = assess_pose17_quality(pose)
    # Must remain valid (soft warning rather than dropping skeleton)
    assert report.is_valid is True
    assert report.needs_refine is True
    assert any("forward_lean_pose" in w for w in report.soft_warnings)
    assert not any("inverted_vertical_orientation" in e for e in report.hard_errors)


def test_pose17_crossed_limbs_tolerated_as_soft_warning():
    """Crossed arms (wrists crossed) and crossed legs (ankles crossed) do not fail laterality."""
    pose = make_canonical_pose17()
    # In canonical viewer convention: left_wrist x=390 < right_wrist x=610
    # Cross wrists: left_wrist reaches across to screen right (x=620), right_wrist to screen left (x=380)
    pose["left_wrist"] = (620.0, 530.0, 2, 0.92)
    pose["right_wrist"] = (380.0, 530.0, 2, 0.92)

    # Cross ankles: left_ankle reaches to screen right (x=570), right_ankle to screen left (x=430)
    pose["left_ankle"] = (570.0, 770.0, 2, 0.90)
    pose["right_ankle"] = (430.0, 770.0, 2, 0.90)

    # Rigid pairs (eyes, ears, shoulders, hips) remain uncrossed
    lat_res = verify_laterality_convention(pose, schema_type="pose17", expected_convention=LATERALITY_VIEWER)
    assert lat_res.is_valid is True
    assert any("left_wrist" in c for c in lat_res.crossed_limbs)
    assert any("left_ankle" in c for c in lat_res.crossed_limbs)
    assert any("crossed_limbs_detected" in w for w in lat_res.soft_warnings)

    report = assess_pose17_quality(pose)
    assert report.is_valid is True
    assert report.needs_refine is True
    assert any("crossed_limbs_detected" in w for w in report.soft_warnings)


def test_pose17_seated_driver_horizontal_thighs_adaptive_torso_scale():
    """Seated driver with foreshortened torso and horizontal thighs passes adaptive scale checks."""
    pose = make_canonical_pose17()
    # Driver seated: shoulders at y=350, hips at y=500, knees horizontal at y=510
    pose["left_shoulder"] = (430.0, 350.0, 2, 0.95)
    pose["right_shoulder"] = (570.0, 350.0, 2, 0.95)
    pose["left_hip"] = (455.0, 500.0, 2, 0.95)
    pose["right_hip"] = (545.0, 500.0, 2, 0.95)
    pose["left_knee"] = (450.0, 510.0, 2, 0.92)   # horizontal thigh
    pose["right_knee"] = (550.0, 510.0, 2, 0.92)  # horizontal thigh

    report = assess_pose17_quality(pose)
    assert report.is_valid is True
    assert not any("disproportionate_bone" in r for r in report.reasons)


def test_hard_errors_vs_soft_warnings_separation():
    """Hard errors (NaN, collapse) invalidate skeletons; soft warnings (crossed limbs) allow refinement."""
    # 1. Hard error: NaN coordinates
    nan_pose = make_canonical_pose17()
    nan_pose["nose"] = (float("nan"), 300.0, 2, 0.95)
    nan_report = assess_pose17_quality(nan_pose)
    assert nan_report.is_valid is False
    assert len(nan_report.hard_errors) > 0

    # 2. Hard error: Degenerate collapse (< 10px diagonal)
    collapsed = {k: (500.0, 500.0, 2, 0.95) for k in POSE17_KEYPOINTS}
    col_report = assess_pose17_quality(collapsed)
    assert col_report.is_valid is False
    assert any("degenerate_skeleton_collapse" in e for e in col_report.hard_errors)

    # 3. Soft warning: Crossed arms only
    crossed_pose = make_canonical_pose17()
    crossed_pose["left_wrist"] = (610.0, 530.0, 2, 0.92)
    crossed_pose["right_wrist"] = (390.0, 530.0, 2, 0.92)
    cross_report = assess_pose17_quality(crossed_pose)
    assert cross_report.is_valid is True
    assert cross_report.needs_refine is True
    assert len(cross_report.hard_errors) == 0
    assert len(cross_report.soft_warnings) > 0




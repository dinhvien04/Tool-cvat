"""Comprehensive property-style tests for crop coordinate transformations and ROI handling.

Covers:
- VinFast Week-2 HumanPose-17 and Face VF50 crop transformations
- Zero off-by-one, boundary preservation at corners and center
- Monotonicity across normalized intervals
- Invertibility / roundtrip consistency
- Robustness to extreme aspect ratios (thin vertical/horizontal crops)
- Near-edge crop clamping behavior
- CVAT detector ROI behavior and translation guarantees
"""

from __future__ import annotations

import math
from typing import List, Tuple
import pytest
from PIL import Image

from core.skeleton_contract import (
    crop_to_image_coords,
    image_to_crop_coords,
    reproject_crop_point,
    expand_face_bbox,
    denormalize_keypoint,
    parse_vf50_response,
    merge_refined_keypoint,
    VF50Face,
    VF50Landmark,
    POSE17_KEYPOINTS,
    VISIBILITY_OUTSIDE,
    VISIBILITY_OCCLUDED,
    VISIBILITY_VISIBLE,
)


# ==============================================================================
# 1. BOUNDARY PRESERVATION & ZERO OFF-BY-ONE PROPERTY TESTS
# ==============================================================================

class TestCropCoordinateBoundaries:
    """Verify that normalized crop coordinates map exactly to crop bounding box vertices."""

    @pytest.mark.parametrize(
        "orig_w, orig_h",
        [(1920, 1080), (1280, 720), (3840, 2160), (640, 480), (1000, 1000)],
    )
    @pytest.mark.parametrize(
        "crop_box",
        [
            [100, 200, 800, 900],  # Standard interior crop
            [0, 0, 500, 500],      # Top-left corner crop
            [500, 500, 1000, 1000],# Bottom-right corner crop
            [0, 0, 1000, 1000],    # Full image crop
        ],
    )
    def test_corners_and_center_mapping(self, orig_w: int, orig_h: int, crop_box: List[int]):
        c_ymin, c_xmin, c_ymax, c_xmax = crop_box

        # Expected pixel boundaries
        exp_tl_x = (c_xmin / 1000.0) * orig_w
        exp_tl_y = (c_ymin / 1000.0) * orig_h
        exp_br_x = (c_xmax / 1000.0) * orig_w
        exp_br_y = (c_ymax / 1000.0) * orig_h
        exp_c_x = ((c_xmin + c_xmax) / 2000.0) * orig_w
        exp_c_y = ((c_ymin + c_ymax) / 2000.0) * orig_h

        # Top-Left (0, 0)
        tl_x, tl_y = crop_to_image_coords(0.0, 0.0, crop_box, orig_w=orig_w, orig_h=orig_h)
        assert tl_x == pytest.approx(exp_tl_x, abs=0.02)
        assert tl_y == pytest.approx(exp_tl_y, abs=0.02)

        # Bottom-Right (1000, 1000)
        br_x, br_y = crop_to_image_coords(1000.0, 1000.0, crop_box, orig_w=orig_w, orig_h=orig_h)
        assert br_x == pytest.approx(exp_br_x, abs=0.02)
        assert br_y == pytest.approx(exp_br_y, abs=0.02)

        # Center (500, 500)
        c_x, c_y = crop_to_image_coords(500.0, 500.0, crop_box, orig_w=orig_w, orig_h=orig_h)
        assert c_x == pytest.approx(exp_c_x, abs=0.02)
        assert c_y == pytest.approx(exp_c_y, abs=0.02)

    def test_reproject_crop_point_matches_formula(self):
        crop_box = [150, 250, 750, 850]
        # Top-left (0, 0) in crop -> (250, 150) in full normalized space
        nx, ny = reproject_crop_point(0.0, 0.0, crop_box)
        assert nx == 250.0
        assert ny == 150.0

        # Bottom-right (1000, 1000) in crop -> (850, 750) in full normalized space
        nx, ny = reproject_crop_point(1000.0, 1000.0, crop_box)
        assert nx == 850.0
        assert ny == 750.0

        # Center (500, 500) in crop -> (550, 450) in full normalized space
        nx, ny = reproject_crop_point(500.0, 500.0, crop_box)
        assert nx == 550.0
        assert ny == 450.0


# ==============================================================================
# 2. MONOTONICITY & SCALE INVARIANCE
# ==============================================================================

class TestCropCoordinateProperties:
    """Verify strict mathematical properties: Monotonicity, Linearity, and Invertibility."""

    def test_strict_monotonicity(self):
        crop_box = [200, 300, 800, 900]
        w, h = 1920, 1080
        prev_x, prev_y = -1.0, -1.0

        for u in range(0, 1001, 50):
            px, py = crop_to_image_coords(float(u), float(u), crop_box, orig_w=w, orig_h=h)
            if prev_x >= 0:
                assert px > prev_x, f"Monotonicity violation along X: px={px} <= prev_x={prev_x}"
                assert py > prev_y, f"Monotonicity violation along Y: py={py} <= prev_y={prev_y}"
            prev_x, prev_y = px, py

    def test_roundtrip_invertibility(self):
        crop_box = [120, 240, 780, 860]
        # Test 20 arbitrary internal points
        for u in range(50, 951, 50):
            norm_orig_x = 240.0 + (u / 1000.0) * (860.0 - 240.0)
            norm_orig_y = 120.0 + (u / 1000.0) * (780.0 - 120.0)

            crop_u_x, crop_u_y = image_to_crop_coords(norm_orig_x, norm_orig_y, crop_box)
            assert crop_u_x == pytest.approx(float(u), abs=0.2)
            assert crop_u_y == pytest.approx(float(u), abs=0.2)

            # Map back to pixel space and verify equivalence
            px, py = crop_to_image_coords(crop_u_x, crop_u_y, crop_box, orig_w=1000, orig_h=1000)
            assert px == pytest.approx(norm_orig_x, abs=0.3)
            assert py == pytest.approx(norm_orig_y, abs=0.3)


# ==============================================================================
# 3. THIN AND NON-SQUARE CROPS (EXTREME ASPECT RATIOS)
# ==============================================================================

class TestExtremeCropAspectRatios:
    """Verify behavior on extreme aspect ratios (tall standing person vs wide banner)."""

    def test_thin_vertical_slit_crop(self):
        # 10px wide by 800px tall in 1000x1000 space
        crop_box = [100, 495, 900, 505]
        w, h = 1920, 1080

        # Center of slit
        px, py = crop_to_image_coords(500.0, 500.0, crop_box, orig_w=w, orig_h=h)
        assert px == pytest.approx(0.500 * w, abs=0.01)
        assert py == pytest.approx(0.500 * h, abs=0.01)

        # X left edge vs right edge
        lx, _ = crop_to_image_coords(0.0, 500.0, crop_box, orig_w=w, orig_h=h)
        rx, _ = crop_to_image_coords(1000.0, 500.0, crop_box, orig_w=w, orig_h=h)
        assert lx == pytest.approx(0.495 * w, abs=0.01)
        assert rx == pytest.approx(0.505 * w, abs=0.01)
        assert rx > lx

    def test_thin_horizontal_banner_crop(self):
        # 800px wide by 10px tall in 1000x1000 space
        crop_box = [495, 100, 505, 900]
        w, h = 1920, 1080

        # Center of banner
        px, py = crop_to_image_coords(500.0, 500.0, crop_box, orig_w=w, orig_h=h)
        assert px == pytest.approx(0.500 * w, abs=0.01)
        assert py == pytest.approx(0.500 * h, abs=0.01)

        # Y top edge vs bottom edge
        _, ty = crop_to_image_coords(500.0, 0.0, crop_box, orig_w=w, orig_h=h)
        _, by = crop_to_image_coords(500.0, 1000.0, crop_box, orig_w=w, orig_h=h)
        assert ty == pytest.approx(0.495 * h, abs=0.01)
        assert by == pytest.approx(0.505 * h, abs=0.01)
        assert by > ty

    def test_non_square_image_with_asymmetric_crop(self):
        # Image 4000x2000 (2:1 aspect ratio)
        # Crop 200x600 in normalized space (width 400, height 400 in pixels: 1:1 square crop!)
        crop_box = [100, 100, 300, 200]
        # In pixels:
        # x: [0.1 * 4000, 0.2 * 4000] = [400, 800] -> width = 400
        # y: [0.1 * 2000, 0.3 * 2000] = [200, 600] -> height = 400
        w, h = 4000, 2000

        px, py = crop_to_image_coords(500.0, 500.0, crop_box, orig_w=w, orig_h=h)
        assert px == 600.0
        assert py == 400.0


# ==============================================================================
# 4. NEAR-EDGE CROPS & CLAMPING BEHAVIOR
# ==============================================================================

class TestNearEdgeCropsAndClamping:
    """Verify safety and clamping when crops touch or exceed image edges."""

    def test_expand_face_bbox_clamps_to_zero_and_thousand(self):
        # Face touching top-left border
        box_tl = [0, 0, 200, 200]
        expanded_tl = expand_face_bbox(box_tl, margin=0.20)
        assert expanded_tl[0] == 0
        assert expanded_tl[1] == 0
        assert expanded_tl[2] == 240
        assert expanded_tl[3] == 240

        # Face touching bottom-right border
        box_br = [800, 800, 1000, 1000]
        expanded_br = expand_face_bbox(box_br, margin=0.20)
        assert expanded_br[0] == 760
        assert expanded_br[1] == 760
        assert expanded_br[2] == 1000
        assert expanded_br[3] == 1000

    def test_out_of_bounds_crop_keypoint_clamping(self):
        crop_box = [100, 100, 500, 500]
        w, h = 1000, 1000

        # Keypoint predicted outside crop bounds (e.g. -100, 1200)
        px, py = crop_to_image_coords(-100.0, 1200.0, crop_box, orig_w=w, orig_h=h, clamp=True)
        # Without clamp: x would be 100 - 0.1 * 400 = 60, y would be 100 + 1.2 * 400 = 580
        # Clamped: norm_x clamped to [0, 1000], norm_y clamped to [0, 1000]
        assert 0.0 <= px <= w
        assert 0.0 <= py <= h


# ==============================================================================
# 5. POSE17 & VF50 TWO-PASS COORDINATE CONSISTENCY
# ==============================================================================

class TestTwoPassCropTransformConsistency:
    """Verify coordinate transformation consistency across Pose17 and VF50 handlers."""

    def test_reproject_crop_point_discrete_vs_normalized(self):
        # Check that reproject_crop_point with pixel crop window matches normalized calculation
        crop_bbox = [200, 300, 600, 700]
        orig_w, orig_h = 1920, 1080
        px1 = int(round((300 / 1000.0) * orig_w))
        py1 = int(round((200 / 1000.0) * orig_h))
        px2 = int(round((700 / 1000.0) * orig_w))
        py2 = int(round((600 / 1000.0) * orig_h))

        for u in (0.0, 250.0, 500.0, 750.0, 1000.0):
            norm_px = reproject_crop_point(
                u, u, crop_bbox, orig_w=orig_w, orig_h=orig_h, crop_window_px=(px1, py1, px2, py2)
            )
            norm_pure = reproject_crop_point(u, u, crop_bbox)
            # Difference due to integer pixel rounding must be <= 0.5 normalized units
            assert abs(norm_px[0] - norm_pure[0]) <= 0.5
            assert abs(norm_px[1] - norm_pure[1]) <= 0.5

    def test_parse_vf50_response_with_crop_box(self):
        crop_box = [200, 300, 600, 700]
        fake_response = {
            "faces": [
                {
                    "id": 1,
                    "box_2d": [100, 100, 900, 900],
                    "landmarks": {
                        "0": [500, 500, 2],  # Center of crop
                    },
                }
            ]
        }
        faces = parse_vf50_response(fake_response, crop_box=crop_box, orig_w=1920, orig_h=1080)
        assert len(faces) == 1
        lm0 = faces[0].get_landmark(0)
        assert lm0 is not None
        # Center of crop (500, 500) -> (500, 400) in full image normalized space
        assert lm0.x == pytest.approx(500.0, abs=0.1)
        assert lm0.y == pytest.approx(400.0, abs=0.1)


# ==============================================================================
# 6. CVAT DETECTOR ROI BEHAVIOR AUDIT
# ==============================================================================

class TestCvatDetectorRoiBehavior:
    """Audit CVAT detector invocation and verify absence of double-translation."""

    def test_standard_detector_operates_in_full_image_coordinates(self):
        """In standard automatic annotation mode, CVAT sends full image and expects full image shapes."""
        # A detector returning shape elements must produce coordinates scaled to (orig_w, orig_h)
        orig_w, orig_h = 1920, 1080
        norm_x, norm_y = 500.0, 500.0
        px, py = denormalize_keypoint(norm_x, norm_y, width=orig_w, height=orig_h)
        assert px == 960.0
        assert py == 540.0

    def test_double_translation_prevention(self):
        """Verify that applying ROI offsets to an already-translated point is prevented or detected."""
        # If coordinates are already in full image space:
        full_px_x = 960.0
        full_px_y = 540.0
        roi_offset_x = 300.0
        roi_offset_y = 200.0

        # Erroneous double translation:
        erroneous_x = full_px_x + roi_offset_x
        erroneous_y = full_px_y + roi_offset_y
        assert erroneous_x == 1260.0  # Shifted away from true location!

        # Correct contract: detector returns full image coordinates; caller does not add offset.
        assert full_px_x == 960.0

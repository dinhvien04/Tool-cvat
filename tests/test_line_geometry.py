"""Unit tests for the lane and linear feature geometry module (core/line_geometry.py).

Tests:
1. PCA-based aspect ratio and elongation metric.
2. Douglas-Peucker line simplification.
3. Centerline extraction on straight, diagonal, curved, and perspective-tapered lane ribbons.
4. Edge case handling: degenerate polygons, isotropic squares, low aspect ratio, invalid coordinates.
5. End-to-end CVAT polyline generation and lane pipeline dispatch with graceful fallback.
"""

import math
import pytest
from typing import List, Tuple

from core.line_geometry import (
    ALL_LANE_LABELS,
    LANE_CROSSWALK,
    LANE_DOUBLE_WHITE,
    LANE_DOUBLE_YELLOW,
    LANE_ROAD_CURB,
    LANE_SINGLE_OTHER,
    LANE_SINGLE_WHITE,
    LANE_SINGLE_YELLOW,
    calculate_polygon_aspect_ratio,
    douglas_peucker,
    extract_centerline_from_polygon,
    extract_lane_centerline,
    denormalize_polyline,
    polyline_to_cvat_polyline,
    is_thin_ribbon,
    lane_shape_pipeline,
    polygon_to_cvat_polyline,
)


class TestPolygonAspectRatio:
    """Tests for PCA-based polygon elongation / aspect ratio."""

    def test_square_patch_has_unit_aspect_ratio(self):
        """An isotropic square patch (e.g. crosswalk tile) should have elongation ~ 1.0."""
        square = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
        ratio = calculate_polygon_aspect_ratio(square)
        assert abs(ratio - 1.0) < 0.05
        is_ribbon, _ = is_thin_ribbon(square, min_aspect_ratio=2.5)
        assert is_ribbon is False

    def test_thin_vertical_lane_has_high_aspect_ratio(self):
        """A thin vertical lane marking (e.g. 500px long, 10px wide) should have high elongation."""
        lane = [(100.0, 200.0), (110.0, 200.0), (110.0, 700.0), (100.0, 700.0)]
        ratio = calculate_polygon_aspect_ratio(lane)
        assert ratio > 20.0
        is_ribbon, _ = is_thin_ribbon(lane, min_aspect_ratio=2.5)
        assert is_ribbon is True

    def test_thin_horizontal_lane_has_high_aspect_ratio(self):
        """A thin horizontal stop line (e.g. 10px tall, 600px wide) should have high elongation."""
        stop_line = [(100.0, 500.0), (700.0, 500.0), (700.0, 510.0), (100.0, 510.0)]
        ratio = calculate_polygon_aspect_ratio(stop_line)
        assert ratio > 20.0
        is_ribbon, _ = is_thin_ribbon(stop_line, min_aspect_ratio=2.5)
        assert is_ribbon is True

    def test_thin_diagonal_lane_has_high_aspect_ratio(self):
        """A 45-degree diagonal lane marking should have high elongation regardless of rotation."""
        # Ribbon along y = x
        diagonal = [(100.0, 105.0), (105.0, 100.0), (505.0, 500.0), (500.0, 505.0)]
        ratio = calculate_polygon_aspect_ratio(diagonal)
        assert ratio > 20.0
        is_ribbon, _ = is_thin_ribbon(diagonal, min_aspect_ratio=2.5)
        assert is_ribbon is True

    def test_few_points_returns_unit_ratio(self):
        assert calculate_polygon_aspect_ratio([(0, 0), (1, 1)]) == 1.0
        is_ribbon, _ = is_thin_ribbon([(0, 0), (1, 1)])
        assert is_ribbon is False


class TestDouglasPeucker:
    """Tests for pure-Python Ramer-Douglas-Peucker line simplification."""

    def test_strictly_collinear_points_reduced_to_endpoints(self):
        """Points along a perfectly straight line should simplify to exactly 2 endpoints."""
        straight_pts = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)]
        simplified = douglas_peucker(straight_pts, epsilon=1.0)
        assert len(simplified) == 2
        assert simplified[0] == (0.0, 0.0)
        assert simplified[-1] == (40.0, 40.0)

    def test_sharp_bend_is_preserved(self):
        """A distinct bend exceeding epsilon must be retained in the simplified polyline."""
        bent_pts = [(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)]
        simplified = douglas_peucker(bent_pts, epsilon=2.0)
        assert len(simplified) == 3
        assert simplified[1] == (50.0, 50.0)

    def test_small_jitter_is_removed(self):
        """Micro-jitter smaller than epsilon should be smoothed away."""
        jitter_pts = [(0.0, 0.0), (25.0, 0.5), (50.0, -0.4), (75.0, 0.3), (100.0, 0.0)]
        simplified = douglas_peucker(jitter_pts, epsilon=1.0)
        assert len(simplified) == 2
        assert simplified[0] == (0.0, 0.0)
        assert simplified[1] == (100.0, 0.0)

    def test_short_sequences(self):
        assert douglas_peucker([(1.0, 1.0), (2.0, 2.0)]) == [(1.0, 1.0), (2.0, 2.0)]
        assert douglas_peucker([]) == []


class TestCenterlineExtraction:
    """Tests for extract_centerline_from_polygon."""

    def test_perspective_tapered_highway_lane(self):
        """A realistic highway lane mark that is 20px wide at bottom and 4px wide at top."""
        # Bottom: x from 100 to 120 (midpoint 110, y=800)
        # Top: x from 448 to 452 (midpoint 450, y=400)
        tapered_lane = [
            (100.0, 800.0),
            (120.0, 800.0),
            (452.0, 400.0),
            (448.0, 400.0),
        ]
        centerline = extract_centerline_from_polygon(tapered_lane, num_samples=5)
        assert centerline is not None
        assert len(centerline) >= 2

        # Verify bottom and top endpoints approximate the true midpoints
        pt_start = centerline[0]
        pt_end = centerline[-1]
        # One end should be near (110, 800) and the other near (450, 400)
        near_bottom = any(abs(p[0] - 110.0) < 5.0 and abs(p[1] - 800.0) < 5.0 for p in (pt_start, pt_end))
        near_top = any(abs(p[0] - 450.0) < 5.0 and abs(p[1] - 400.0) < 5.0 for p in (pt_start, pt_end))
        assert near_bottom
        assert near_top

    def test_curved_lane_ribbon(self):
        """A curved arc ribbon following a parabola y = 0.001*(x-300)^2."""
        pts = []
        # Outer curve (left edge)
        for y in range(400, 801, 50):
            x = 300.0 + 0.001 * ((y - 400.0) ** 2)
            pts.append((x, float(y)))
        # Inner curve (right edge, 15px offset)
        for y in range(800, 399, -50):
            x = 315.0 + 0.001 * ((y - 400.0) ** 2)
            pts.append((x, float(y)))

        centerline = extract_centerline_from_polygon(pts, num_samples=10, simplify_epsilon=0.5)
        assert centerline is not None
        assert len(centerline) >= 3  # Must retain curvature vertices

    def test_square_crosswalk_rejected_from_centerline(self):
        """A 2D square crosswalk patch should return None because it is not a thin ribbon."""
        crosswalk_box = [(200.0, 200.0), (350.0, 200.0), (350.0, 350.0), (200.0, 350.0)]
        centerline = extract_centerline_from_polygon(crosswalk_box, min_aspect_ratio=2.5)
        assert centerline is None

    def test_degenerate_zero_area_polygon_returns_none(self):
        degenerate = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)]
        assert extract_centerline_from_polygon(degenerate) is None

    def test_too_few_vertices_returns_none(self):
        assert extract_centerline_from_polygon([(0, 0), (1, 1), (2, 2)]) is None


class TestPolygonToCvatPolyline:
    """Tests for normalized contour conversion to CVAT polyline shape."""

    def test_valid_lane_ribbon_to_cvat_polyline(self):
        # Normalized coordinates [0, 1000] for a lane ribbon
        contour = [
            [100, 800],
            [120, 800],
            [452, 400],
            [448, 400],
        ]
        shape = polygon_to_cvat_polyline(
            contour=contour,
            width=1000,
            height=1000,
            label=LANE_SINGLE_WHITE,
            confidence=0.92,
        )
        assert shape is not None
        assert shape["type"] == "polyline"
        assert shape["label"] == LANE_SINGLE_WHITE
        assert shape["confidence"] == "0.92"
        assert isinstance(shape["points"], list)
        assert len(shape["points"]) >= 4  # at least [x1, y1, x2, y2]
        assert len(shape["points"]) % 2 == 0

    def test_non_ribbon_contour_returns_none(self):
        square_contour = [[200, 200], [400, 200], [400, 400], [200, 400]]
        shape = polygon_to_cvat_polyline(
            contour=square_contour,
            width=1000,
            height=1000,
            label=LANE_CROSSWALK,
        )
        assert shape is None

    def test_invalid_dimensions_returns_none(self):
        contour = [[100, 800], [120, 800], [452, 400], [448, 400]]
        assert polygon_to_cvat_polyline(contour, width=0, height=1000) is None
        assert polygon_to_cvat_polyline(contour, width=1000, height=-10) is None


class TestLaneShapePipeline:
    """Tests for the end-to-end lane shape dispatcher and fallback behavior."""

    def test_crosswalk_emits_polyline_per_policy_c(self):
        """Crosswalks under Policy C strictly emit polyline traversal centerline."""
        crosswalk_contour = [[200, 200], [600, 200], [600, 400], [200, 400]]
        shape = lane_shape_pipeline(
            label=LANE_CROSSWALK,
            contour=crosswalk_contour,
            width=1000,
            height=1000,
            confidence=0.88,
            preferred_geometry="polyline",
        )
        assert shape is not None
        assert shape["type"] == "polyline"
        assert shape["label"] == LANE_CROSSWALK
        assert shape["confidence"] == "0.88"
        assert len(shape["points"]) >= 4  # >= 2 points * 2 coordinates

    def test_crosswalk_strictly_polyline_ignores_mask_preference(self):
        """Policy C enforces polyline only; even if mask was requested, emits polyline."""
        crosswalk_contour = [[200, 200], [600, 200], [600, 400], [200, 400]]
        shape = lane_shape_pipeline(
            label=LANE_CROSSWALK,
            contour=crosswalk_contour,
            width=1000,
            height=1000,
            confidence=0.88,
            preferred_geometry="mask",
        )
        assert shape is not None
        assert shape["type"] == "polyline"

    def test_thin_lane_emits_polyline_in_auto_mode(self):
        lane_contour = [[100, 800], [115, 800], [455, 300], [445, 300]]
        shape = lane_shape_pipeline(
            label=LANE_SINGLE_YELLOW,
            contour=lane_contour,
            width=1000,
            height=1000,
            confidence=0.95,
            preferred_geometry="auto",
        )
        assert shape is not None
        assert shape["type"] == "polyline"
        assert shape["label"] == LANE_SINGLE_YELLOW

    def test_thin_lane_no_auto_fallback_returns_none_per_policy_c(self):
        """Under Policy C, auto fallback is removed. If polyline fails, returns None."""
        fat_patch = [[100, 100], [250, 100], [250, 200], [100, 200]]  # aspect ratio ~ 1.5 < 2.5
        shape = lane_shape_pipeline(
            label=LANE_ROAD_CURB,
            contour=fat_patch,
            width=1000,
            height=1000,
            confidence=0.75,
            preferred_geometry="polyline",
            allow_fallback=False,
        )
        assert shape is None  # Dropped per Policy C; caller logs lane_polyline_failed

    def test_no_polygon_fallback_under_policy_c(self):
        """Under strict Policy C, lane_shape_pipeline returns None and emits no polygon/mask fallback."""
        fat_patch = [[100, 100], [250, 100], [250, 200], [100, 200]]
        shape = lane_shape_pipeline(
            label=LANE_ROAD_CURB,
            contour=fat_patch,
            width=1000,
            height=1000,
            confidence=0.75,
            allow_fallback=True,
        )
        assert shape is None  # Strictly returns None; no polygon or mask fallback permitted under Policy C

    def test_open_polyline_2_points_emits_polyline(self):
        """Verify 2-point line segments from models emit valid polylines directly."""
        line_2pts = [[100, 200], [500, 800]]
        shape = lane_shape_pipeline(
            label=LANE_SINGLE_WHITE,
            contour=line_2pts,
            width=1000,
            height=1000,
            confidence=0.88,
        )
        assert shape is not None
        assert shape["type"] == "polyline"
        assert shape["label"] == LANE_SINGLE_WHITE
        assert len(shape["points"]) == 4
        # Due to denormalize_contour using (width - 1), coordinates are ~[99.9, 199.8, 499.5, 799.2]
        assert pytest.approx(shape["points"], abs=1.0) == [100.0, 200.0, 500.0, 800.0]

    def test_open_polyline_3_points_curved_emits_polyline(self):
        """Verify 3-point curves emit simplified polylines."""
        curve_3pts = [[100, 200], [300, 450], [500, 800]]
        shape = lane_shape_pipeline(
            label=LANE_DOUBLE_YELLOW,
            contour=curve_3pts,
            width=1000,
            height=1000,
            confidence=0.91,
        )
        assert shape is not None
        assert shape["type"] == "polyline"
        assert shape["label"] == LANE_DOUBLE_YELLOW
        assert len(shape["points"]) >= 4

    def test_open_polyline_degenerate_2_points_returns_none(self):
        """Degenerate 2-point segment (length < 1 px) returns None without fallback."""
        degenerate_2pts = [[100, 200], [100.2, 200.3]]
        shape = lane_shape_pipeline(
            label=LANE_SINGLE_WHITE,
            contour=degenerate_2pts,
            width=1000,
            height=1000,
        )
        assert shape is None

    def test_all_7_lane_labels_recognized(self):
        assert len(ALL_LANE_LABELS) == 7
        for label in (
            LANE_CROSSWALK,
            LANE_DOUBLE_WHITE,
            LANE_DOUBLE_YELLOW,
            LANE_ROAD_CURB,
            LANE_SINGLE_OTHER,
            LANE_SINGLE_WHITE,
            LANE_SINGLE_YELLOW,
        ):
            assert label in ALL_LANE_LABELS


class TestDirectPolylinePipeline:
    """Tests for direct Polyline -> CVAT Polyline (Production Policy C)."""

    def test_direct_polyline_preserves_exact_point_order_and_count(self):
        """Verify direct polyline denormalization strictly preserves point order and count."""
        # Arbitrary multi-point polyline from model
        pts = [
            [100, 200],
            [200, 300],
            [350, 450],
            [500, 600],
            [700, 900],
        ]
        shape = polyline_to_cvat_polyline(
            polyline=pts,
            width=1000,
            height=1000,
            label=LANE_SINGLE_WHITE,
            confidence=0.95,
        )
        assert shape is not None
        assert shape["type"] == "polyline"
        assert shape["label"] == LANE_SINGLE_WHITE
        assert shape["confidence"] == "0.95"
        # Exactly 5 points (10 coordinates) in identical sequence
        assert shape["points"] == [
            100.0, 200.0,
            200.0, 300.0,
            350.0, 450.0,
            500.0, 600.0,
            700.0, 900.0,
        ]

    def test_crosswalk_direct_polyline_traversal(self):
        """Crosswalks predicted as direct traversal polylines must preserve point sequence."""
        traversal_pts = [
            [150, 500],
            [400, 520],
            [650, 480],
            [900, 500],
        ]
        shape = polyline_to_cvat_polyline(
            polyline=traversal_pts,
            width=1280,
            height=720,
            label=LANE_CROSSWALK,
            confidence=0.89,
        )
        assert shape is not None
        assert shape["type"] == "polyline"
        assert shape["label"] == LANE_CROSSWALK
        # Expected coordinates: (pt[0]/1000 * 1280), (pt[1]/1000 * 720)
        assert shape["points"] == [
            192.0, 360.0,
            512.0, 374.4,
            832.0, 345.6,
            1152.0, 360.0,
        ]

    def test_denormalize_polyline_validation(self):
        """Verify denormalize_polyline enforces min points, numeric coordinates, and bounds."""
        # Too few points raises ValueError
        with pytest.raises(ValueError, match="at least 2 points"):
            denormalize_polyline([[100, 200]], width=1000, height=1000)

        # Non-numeric coordinate raises TypeError
        with pytest.raises(TypeError, match="numeric"):
            denormalize_polyline([[100, "abc"], [200, 300]], width=1000, height=1000)

        # Invalid dimensions
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            denormalize_polyline([[100, 200], [200, 300]], width=0, height=1000)

        # Grossly out-of-bounds raises ValueError
        with pytest.raises(ValueError, match="grossly out of bounds"):
            denormalize_polyline([[100, 200], [2000, 300]], width=1000, height=1000)

        # Moderate overshoot within [-100, 1100] is clamped
        clamped = denormalize_polyline([[-10, 50], [1050, 950]], width=1000, height=1000, clamp=True)
        assert clamped[0] == (0.0, 50.0)
        assert clamped[1] == (1000.0, 950.0)

    def test_clean_separation_between_direct_polyline_and_contour_centerline(self):
        """Ensure clean separation between direct polyline denormalization and legacy contour centerline extraction."""
        # 1. Direct polyline: 4 collinear points stay exactly 4 points without PCA thinning
        collinear_pts = [[100, 100], [200, 200], [300, 300], [400, 400]]
        direct_shape = polyline_to_cvat_polyline(collinear_pts, width=1000, height=1000, label=LANE_DOUBLE_WHITE)
        assert len(direct_shape["points"]) == 8  # all 4 points retained verbatim

        # 2. Legacy ribbon contour centerline: runs PCA, medial resampling, and Douglas-Peucker
        ribbon_contour = [(100.0, 800.0), (120.0, 800.0), (452.0, 400.0), (448.0, 400.0)]
        centerline = extract_lane_centerline(ribbon_contour, num_samples=5)
        assert centerline is not None
        assert extract_centerline_from_polygon(ribbon_contour, num_samples=5) == centerline

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

    def test_crosswalk_always_emits_polygon_never_polyline(self):
        """Crosswalks are 2D surfaces and should always be emitted as polygons, even in auto mode."""
        crosswalk_contour = [[200, 200], [600, 200], [600, 400], [200, 400]]
        shape = lane_shape_pipeline(
            label=LANE_CROSSWALK,
            contour=crosswalk_contour,
            width=1000,
            height=1000,
            confidence=0.88,
            preferred_geometry="auto",
        )
        assert shape is not None
        assert shape["type"] == "polygon"
        assert shape["label"] == LANE_CROSSWALK
        assert shape["confidence"] == "0.88"
        assert len(shape["points"]) == 8  # 4 vertices * 2 coordinates

    def test_crosswalk_emits_mask_when_requested(self):
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
        assert shape["type"] == "mask"
        assert "mask" in shape

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

    def test_thin_lane_fallback_to_polygon_when_polyline_fails(self):
        """When a lane marking is not sufficiently elongated, it falls back to polygon."""
        fat_patch = [[100, 100], [250, 100], [250, 200], [100, 200]]  # aspect ratio ~ 1.5 < 2.5
        shape = lane_shape_pipeline(
            label=LANE_ROAD_CURB,
            contour=fat_patch,
            width=1000,
            height=1000,
            confidence=0.75,
            preferred_geometry="auto",
        )
        assert shape is not None
        assert shape["type"] == "polygon"  # Graceful fallback!
        assert shape["label"] == LANE_ROAD_CURB

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

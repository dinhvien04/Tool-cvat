"""Tests for Buddha Coordinate Reprojection (core/buddha_contract.py).

Validates:
- crop_norm_to_global_norm and global_norm_to_crop_norm mathematical precision.
- Exact invertibility for arbitrary crop bounding boxes.
- Handling of edge cases: boundary coordinates (0, 1000), zero-area crops.
"""

from __future__ import annotations

import pytest
from core.buddha_contract import (
    crop_norm_to_global_norm,
    global_norm_to_crop_norm,
)


class TestBuddhaCoordinateReprojection:
    """Validate normalized crop <-> global coordinate mappings."""

    def test_full_image_crop_identity(self):
        """A crop box covering the full image [0, 0, 1000, 1000] should act as an identity transform."""
        full_box = [0, 0, 1000, 1000]
        pts = [(0.0, 0.0), (500.0, 500.0), (1000.0, 1000.0), (250.5, 750.2)]
        for pt in pts:
            g = crop_norm_to_global_norm(pt, full_box)
            assert pytest.approx(g[0], abs=1e-4) == pt[0]
            assert pytest.approx(g[1], abs=1e-4) == pt[1]

            c = global_norm_to_crop_norm(pt, full_box)
            assert pytest.approx(c[0], abs=1e-4) == pt[0]
            assert pytest.approx(c[1], abs=1e-4) == pt[1]

    def test_sub_crop_coordinate_mapping(self):
        """Crop box [ymin=200, xmin=300, ymax=600, xmax=700] (w=400, h=400)."""
        box = [200, 300, 600, 700]

        # Center of crop (500, 500) -> Global center of crop (500, 400)
        # gx = 300 + 500/1000 * 400 = 500
        # gy = 200 + 500/1000 * 400 = 400
        gx, gy = crop_norm_to_global_norm((500.0, 500.0), box)
        assert pytest.approx(gx, abs=1e-4) == 500.0
        assert pytest.approx(gy, abs=1e-4) == 400.0

        # Top-left of crop (0, 0) -> Global (xmin=300, ymin=200)
        gx, gy = crop_norm_to_global_norm((0.0, 0.0), box)
        assert pytest.approx(gx, abs=1e-4) == 300.0
        assert pytest.approx(gy, abs=1e-4) == 200.0

        # Bottom-right of crop (1000, 1000) -> Global (xmax=700, ymax=600)
        gx, gy = crop_norm_to_global_norm((1000.0, 1000.0), box)
        assert pytest.approx(gx, abs=1e-4) == 700.0
        assert pytest.approx(gy, abs=1e-4) == 600.0

    def test_invertibility_roundtrip(self):
        """Coordinate roundtripping from global -> crop -> global must be lossless."""
        boxes = [
            [100, 150, 450, 600],
            [350, 200, 800, 900],
            [0, 500, 1000, 1000],
            [700, 0, 950, 400],
        ]

        test_points = [
            (250.0, 300.0),
            (500.0, 500.0),
            (150.0, 100.0),
            (900.0, 850.0),
        ]

        for box in boxes:
            for g_orig in test_points:
                # 1. Global -> Crop
                c = global_norm_to_crop_norm(g_orig, box)
                # 2. Crop -> Global
                g_roundtrip = crop_norm_to_global_norm(c, box)

                assert pytest.approx(g_roundtrip[0], abs=1e-4) == g_orig[0]
                assert pytest.approx(g_roundtrip[1], abs=1e-4) == g_orig[1]

    def test_degenerate_box_handling(self):
        """Degenerate zero-size box should safely return original point."""
        zero_box = [200, 200, 200, 200]
        res = crop_norm_to_global_norm((500.0, 500.0), zero_box)
        assert res == (500.0, 500.0)

        res2 = global_norm_to_crop_norm((500.0, 500.0), zero_box)
        assert res2 == (500.0, 500.0)

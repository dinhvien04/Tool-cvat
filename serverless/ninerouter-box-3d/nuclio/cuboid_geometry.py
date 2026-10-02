"""Convert a front face into the 8-point cuboid CVAT draws on an image.

Point order follows cvat-canvas cuboidFrom4Points so the shape stays editable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Point = Dict[str, float]


def _sort_points_clockwise(points: List[Point]) -> List[Point]:
    ordered = [dict(point) for point in points]
    ordered.sort(key=lambda point: point["y"])
    cy = (ordered[0]["y"] + ordered[-1]["y"]) / 2
    ordered.sort(key=lambda point: -point["x"])
    cx = (ordered[0]["x"] + ordered[-1]["x"]) / 2
    start_ang = None
    for point in ordered:
        ang = math.atan2(point["y"] - cy, point["x"] - cx)
        if start_ang is None:
            start_ang = ang
        elif ang < start_ang:
            ang += math.pi * 2
        point["angle"] = ang
    ordered.sort(key=lambda point: point["angle"])
    ordered.reverse()
    return ordered


def _rotate(points: List[Point], times: int) -> None:
    remaining = times
    while remaining:
        points.append(points.pop(0))
        remaining -= 1


def _setup_cuboid_points(points: Sequence[Point]) -> List[Point]:
    height = (
        abs(points[1]["y"] - points[0]["y"])
        if abs(points[0]["x"] - points[1]["x"]) < abs(points[1]["x"] - points[2]["x"])
        else abs(points[1]["y"] - points[2]["y"])
    )
    if points[0]["x"] < points[2]["x"]:
        left = points[0]
        right = points[2]
    else:
        right = points[0]
        left = points[2]

    if left["y"] < right["y"]:
        left2 = {"x": left["x"], "y": left["y"] + height}
        right2 = {"x": right["x"], "y": right["y"] - height}
    else:
        left2 = {"x": left["x"], "y": left["y"] - height}
        right2 = {"x": right["x"], "y": right["y"] + height}

    vec = {"x": points[3]["x"] - points[2]["x"], "y": points[3]["y"] - points[2]["y"]}
    if left["y"] < left2["y"]:
        p1, p2 = left, left2
    else:
        p1, p2 = left2, left
    if right["y"] < right2["y"]:
        p3, p4 = right, right2
    else:
        p3, p4 = right2, right

    p5 = {"x": p3["x"] + vec["x"], "y": p3["y"] + vec["y"] + 0.1}
    p6 = {"x": p4["x"] + vec["x"], "y": p4["y"] + vec["y"] - 0.1}
    p7 = {"x": p1["x"] + vec["x"], "y": p1["y"] + vec["y"] + 0.1}
    p8 = {"x": p2["x"] + vec["x"], "y": p2["y"] + vec["y"] - 0.1}
    p1["y"] += 0.1
    return [p1, p2, p3, p4, p5, p6, p7, p8]


def cuboid_from_4_points(flattened: Sequence[float]) -> List[float]:
    """Return 16 CVAT cuboid coordinates from four construction points."""
    points = [
        {"x": float(flattened[i]), "y": float(flattened[i + 1])}
        for i in range(0, 8, 2)
    ]
    unsorted = points[:3]
    vector = {
        "x": points[2]["x"] - points[1]["x"],
        "y": points[2]["y"] - points[1]["y"],
    }
    unsorted.append({"x": points[0]["x"] + vector["x"], "y": points[0]["y"] + vector["y"]})
    sorted_plane = _sort_points_clockwise(unsorted)
    left_index = 0
    for index in range(4):
        if sorted_plane[index]["x"] < sorted_plane[left_index]["x"]:
            left_index = index
    _rotate(sorted_plane, left_index)
    plane1 = {
        "p1": sorted_plane[0],
        "p2": sorted_plane[1],
        "p3": sorted_plane[2],
        "p4": sorted_plane[3],
    }
    vec = {
        "x": points[3]["x"] - points[2]["x"],
        "y": points[3]["y"] - points[2]["y"],
    }
    angle = math.atan2(vec["y"], vec["x"])
    plane2 = {
        key: {"x": plane1[key]["x"] + vec["x"], "y": plane1[key]["y"] + vec["y"]}
        for key in ("p1", "p2", "p3", "p4")
    }
    if abs(angle) < math.pi / 2 - 0.1 or abs(angle) > math.pi / 2 + 0.1:
        cuboid_points = _setup_cuboid_points(points)
    elif angle > 0:
        cuboid_points = [
            plane1["p1"], plane2["p1"], plane1["p2"], plane2["p2"],
            plane1["p3"], plane2["p3"], plane1["p4"], plane2["p4"],
        ]
        cuboid_points[0]["y"] += 0.1
        cuboid_points[4]["y"] += 0.1
    else:
        cuboid_points = [
            plane2["p1"], plane1["p1"], plane2["p2"], plane1["p2"],
            plane2["p3"], plane1["p3"], plane2["p4"], plane1["p4"],
        ]
        cuboid_points[0]["y"] += 0.1
        cuboid_points[4]["y"] += 0.1

    flat: List[float] = []
    for point in cuboid_points:
        flat.append(float(point["x"]))
        flat.append(float(point["y"]))
    return flat


def cuboid_from_front_face(
    xtl: float,
    ytl: float,
    xbr: float,
    ybr: float,
    depth_ratio: float = 0.18,
    side: str = "right",
) -> List[float]:
    """Build a CVAT cuboid from the visible front rectangle.

    side='right' shifts the back face up and to the right, matching CVAT's
    default drag. side='left' shifts it up and to the left.
    """
    width = max(1.0, xbr - xtl)
    height = max(1.0, ybr - ytl)
    ratio = min(0.45, max(0.08, float(depth_ratio)))
    dx = width * ratio
    dy = height * ratio
    if str(side).strip().lower() == "left":
        corners = [xbr, ybr, xtl, ybr, xtl, ytl, xtl - dx, ytl - dy]
    else:
        corners = [xtl, ybr, xbr, ybr, xbr, ytl, xbr + dx, ytl - dy]
    return cuboid_from_4_points(corners)


def normalized_box_to_pixels(
    box_2d: Sequence[float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Convert [ymin, xmin, ymax, xmax] in 0..1000 into pixel edges."""
    ymin, xmin, ymax, xmax = [float(value) for value in box_2d]
    xtl = xmin / 1000.0 * width
    ytl = ymin / 1000.0 * height
    xbr = xmax / 1000.0 * width
    ybr = ymax / 1000.0 * height
    if xbr < xtl:
        xtl, xbr = xbr, xtl
    if ybr < ytl:
        ytl, ybr = ybr, ytl
    return xtl, ytl, xbr, ybr

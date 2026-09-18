"""Micro-benchmarks for Detector #1 (Rectangle + Mask) scaling.

Measures how each local stage scales with:
1. Image resolution: 720p (1280x720), 1080p (1920x1080), 4K (3840x2160)
2. Object count: 1, 5, 15, 30, 50, 100 instances
3. Raster mask generation and CVAT mask serialization overhead
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.image_ops import load_image, resize_image_if_needed, image_to_data_url
from app.feedback import compute_image_hash, compute_perceptual_hash, FeedbackDatabase
from core.geometry import polygon_to_cvat_mask, rasterize_polygon_to_mask, mask_to_cvat_flat_list
from core.taxonomy import validate_detector_a_shapes, Taxonomy


def benchmark_resolution_scaling():
    print("\n--- Image Resolution Scaling Benchmark ---")
    resolutions = [
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
        ("4K", 3840, 2160),
    ]

    for name, w, h in resolutions:
        # Generate dummy JPEG
        im = Image.new("RGB", (w, h), color=(120, 140, 160))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # 1. Load image
        t0 = time.perf_counter()
        loaded = load_image(raw_bytes)
        t_load = (time.perf_counter() - t0) * 1000

        # 2. Resize to max 1600
        t0 = time.perf_counter()
        resized, was_resized, new_dim = resize_image_if_needed(loaded, max_size=1600)
        t_resize = (time.perf_counter() - t0) * 1000

        # 3. JPEG encode (with optimize=True)
        t0 = time.perf_counter()
        data_url = image_to_data_url(resized, format="JPEG", quality=90)
        t_encode = (time.perf_counter() - t0) * 1000

        # 4. SHA-256 raw RGB hash
        t0 = time.perf_counter()
        h_sha = compute_image_hash(loaded)
        t_sha = (time.perf_counter() - t0) * 1000

        # 5. Perceptual dHash
        t0 = time.perf_counter()
        h_dhash = compute_perceptual_hash(loaded)
        t_dhash = (time.perf_counter() - t0) * 1000

        print(f"[{name}: {w}x{h} ({w*h/1e6:.1f} MP)]")
        print(f"  - load_image: {t_load:.2f} ms")
        print(f"  - resize (-> {new_dim[0]}x{new_dim[1]}): {t_resize:.2f} ms (was_resized={was_resized})")
        print(f"  - JPEG encode (optimize=True): {t_encode:.2f} ms (b64 len: {len(data_url)})")
        print(f"  - compute_image_hash (tobytes + sha256): {t_sha:.2f} ms (raw: {w*h*3/1e6:.1f} MB)")
        print(f"  - compute_perceptual_hash: {t_dhash:.2f} ms")
        print(f"  - Total image prep & hash: {t_load + t_resize + t_encode + t_sha + t_dhash:.2f} ms")


def benchmark_object_count_scaling():
    print("\n--- Object Count Scaling Benchmark (1920x1080 canvas) ---")
    w, h = 1920, 1080
    counts = [1, 5, 15, 30, 50, 100]

    # Sample normalized polygon contour for a vehicle
    base_contour = [
        [200, 300], [450, 300], [500, 400], [520, 600],
        [480, 650], [220, 650], [180, 550], [190, 400]
    ]

    f_db = FeedbackDatabase(db_path=".tool-cvat/test_bench_feedback.sqlite3")

    for cnt in counts:
        # Generate N contours
        contours = []
        for i in range(cnt):
            shift = (i * 7) % 300
            c = [[(pt[0] + shift) % 950, (pt[1] + shift) % 950] for pt in base_contour]
            contours.append(c)

        # 1. Mask rasterization + flattening
        t0 = time.perf_counter()
        masks = []
        for c in contours:
            m = polygon_to_cvat_mask(c, width=w, height=h)
            masks.append(m)
        t_mask_gen = (time.perf_counter() - t0) * 1000

        # 2. Assembling shapes (paired rectangle + mask)
        t0 = time.perf_counter()
        shapes = []
        for i, m in enumerate(masks):
            if not m:
                continue
            gid = i + 1
            bbox = m["bbox"]
            shapes.append({
                "type": "rectangle",
                "label": "car",
                "points": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "group_id": gid,
                "confidence": "0.95",
            })
            shapes.append({
                "type": "mask",
                "label": "car",
                "points": m["points"],
                "mask": m["mask"],
                "group_id": gid,
                "confidence": "0.95",
            })
        t_assemble = (time.perf_counter() - t0) * 1000

        # 3. CVAT Output validation
        t0 = time.perf_counter()
        taxonomy = Taxonomy()
        val_shapes, warnings = validate_detector_a_shapes(shapes, taxonomy=taxonomy)
        t_validate = (time.perf_counter() - t0) * 1000

        # 4. JSON Serialization & Baseline DB storage
        t0 = time.perf_counter()
        row_id = f_db.save_prediction_baseline(
            image_hash=f"bench_hash_{cnt}",
            shapes=val_shapes,
            model="ag/gemini-3.8-flash-high",
            mode="rectangle_mask",
        )
        t_store = (time.perf_counter() - t0) * 1000

        total_mask_bytes = sum(len(m["mask"]) for m in masks if m)
        print(f"[{cnt} instances -> {len(shapes)} shapes]")
        print(f"  - mask rasterization ({cnt} masks): {t_mask_gen:.2f} ms ({t_mask_gen/cnt:.2f} ms/mask)")
        print(f"  - shape assembly & pairing: {t_assemble:.2f} ms")
        print(f"  - CVAT validation ({len(val_shapes)} shapes): {t_validate:.2f} ms")
        print(f"  - SQLite baseline store ({total_mask_bytes} mask ints): {t_store:.2f} ms")
        print(f"  - Total geometry + validation + storage: {t_mask_gen + t_assemble + t_validate + t_store:.2f} ms")

    # Clean up bench db
    try:
        Path(".tool-cvat/test_bench_feedback.sqlite3").unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    benchmark_resolution_scaling()
    benchmark_object_count_scaling()

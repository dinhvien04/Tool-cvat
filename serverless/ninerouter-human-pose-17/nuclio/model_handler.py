"""ModelHandler for 9Router Human Pose 17 Nuclio Detector.

Bridges CVAT's serverless detector invocation with the local 9Router vision model.
Enforces COCO 17-Keypoint Human Pose Topology:
- Exactly 17 canonical keypoints
- Emits native CVAT skeleton shapes
- Clamps coordinates to image bounds and maps visibility/occlusion states
- Deterministic quality gate assessment (kinematic chain & seated driver tolerance)
- Adaptive person-crop two-pass refinement
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml
from PIL import Image

# Ensure project modules (app, core, config) are importable
current_dir = Path(__file__).resolve().parent
if (current_dir / "app").exists():
    sys.path.insert(0, str(current_dir))
else:
    project_root = current_dir.parent.parent.parent
    if (project_root / "app").exists():
        sys.path.insert(0, str(project_root))

from app.client import NineRouterClient, NineRouterConnectionError, NineRouterError
from app.config import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_NINEROUTER_TIMEOUT,
    DEFAULT_NINEROUTER_URL_CONTAINER,
    DEFAULT_VISION_MODEL,
    mask_api_key,
)
from app.parser import clean_json_string
from core.pose_face_schema import (
    POSE17_KEYPOINTS,
    POSE17_EDGES,
    parse_and_sanitize_pose17_instance,
)
from core.quality_gate import (
    assess_pose17_quality,
    LATERALITY_VIEWER,
    Pose17QualityReport,
)
from core.week2_schema import get_build_sha, load_pose17

logger = logging.getLogger("cvat.nuclio.ninerouter.human_pose_17")

DEFAULT_POSE17_MAX_TOKENS = 2000
DEFAULT_POSE17_MAX_IMAGE_SIZE = 1280


def build_pose17_prompt(keypoints: Sequence[str]) -> str:
    """Build strict, deterministic prompt for 9Router Pose 17 vision model."""
    kps_str = ", ".join(f'"{k}"' for k in keypoints)
    return (
        "Detect all persons and estimate their 17 COCO keypoints in this image.\n"
        f"Keypoints to detect: [{kps_str}].\n"
        "VinFast Viewer-Perspective Convention:\n"
        "- 'right_*' (right_eye, right_ear, right_shoulder, right_elbow, right_wrist, right_hip, right_knee, right_ankle) "
        "refer to the VIEWER'S RIGHT side of the image frame (larger X coordinate).\n"
        "- 'left_*' (left_eye, left_ear, left_shoulder, left_elbow, left_wrist, left_hip, left_knee, left_ankle) "
        "refer to the VIEWER'S LEFT side of the image frame (smaller X coordinate).\n"
        "Kinematic Chain & Cabin Constraints:\n"
        "- Enforce strict anatomical connectivity: shoulder -> elbow -> wrist, and hip -> knee -> ankle.\n"
        "- Wrists and elbows MUST attach to their corresponding limb; NEVER predict floating wrists or elbows in the cabin background, roof lining, or car seats.\n"
        "- If a person is seated (e.g. driver or passenger) and lower limbs or hands are occluded behind steering wheel or seats, set visibility flag = 1 (occluded) or 0 (outside), do NOT hallucinate floating joints.\n"
        "Coordinate & Visibility Convention:\n"
        "- Coordinates must be normalized integers [x, y] in range [0, 1000] relative to image width and height.\n"
        "- Visibility flag: 0 = outside image frame, 1 = present but occluded, 2 = clearly visible.\n"
        "- Bounding box 'box_2d': [ymin, xmin, ymax, xmax] in [0, 1000].\n"
        "Return STRICT JSON only, matching this structure:\n"
        "{\n"
        '  "people": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "label": "person",\n'
        '      "confidence": 0.95,\n'
        '      "box_2d": [ymin, xmin, ymax, xmax],\n'
        '      "keypoints": {\n'
        '        "nose": [x, y, 2],\n'
        '        "right_eye": [x, y, 2],\n'
        '        "left_eye": [x, y, 2],\n'
        "        ...\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        'If no persons are found, return {"people": []}.'
    )


def build_pose17_crop_prompt(keypoints: Sequence[str]) -> str:
    """Build targeted prompt for high-resolution person crop refinement (Pass 2)."""
    kps_str = ", ".join(f'"{k}"' for k in keypoints)
    return (
        "High-resolution close-up person crop refinement.\n"
        f"Detect exactly the 17 COCO keypoints for the single person in this cropped image: [{kps_str}].\n"
        "Viewer Perspective:\n"
        "- right_* = viewer's right (larger X)\n"
        "- left_* = viewer's left (smaller X)\n"
        "Kinematic Chain:\n"
        "- Strict limb connectivity: shoulder -> elbow -> wrist. No floating joints!\n"
        "- Coordinates: normalized integers [x, y] in [0, 1000] relative to THIS CROP.\n"
        "- Visibility: 0 = outside, 1 = occluded, 2 = visible.\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "id": 1,\n'
        '  "confidence": 0.98,\n'
        '  "keypoints": {\n'
        '    "nose": [x, y, 2],\n'
        "    ...\n"
        "  }\n"
        "}\n"
    )


def derive_person_bbox(person_dict: Dict[str, Any], pad_ratio: float = 0.18) -> Optional[List[int]]:
    """Derive bounding box [ymin, xmin, ymax, xmax] in [0..1000] space from person dict."""
    raw_box = person_dict.get("box_2d") or person_dict.get("bbox")
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        try:
            ymin, xmin, ymax, xmax = [float(v) for v in raw_box]
            if 0.0 <= xmin < xmax <= 1000.0 and 0.0 <= ymin < ymax <= 1000.0:
                return [int(round(ymin)), int(round(xmin)), int(round(ymax)), int(round(xmax))]
        except (ValueError, TypeError):
            pass

    kps = person_dict.get("keypoints", {})
    coords: List[Tuple[float, float]] = []
    if isinstance(kps, dict):
        for pt in kps.values():
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                vis = pt[2] if len(pt) > 2 else 2
                if vis > 0:
                    coords.append((float(pt[0]), float(pt[1])))
    elif isinstance(kps, list):
        for item in kps:
            if isinstance(item, dict):
                pt = item.get("point") or item.get("points")
                if pt and len(pt) >= 2:
                    vis = item.get("visibility", 2)
                    if vis > 0:
                        coords.append((float(pt[0]), float(pt[1])))

    if not coords:
        return None

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(10.0, max_x - min_x)
    h = max(10.0, max_y - min_y)
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio

    return [
        int(max(0.0, min(1000.0, round(min_y - pad_y)))),
        int(max(0.0, min(1000.0, round(min_x - pad_x)))),
        int(max(0.0, min(1000.0, round(max_y + pad_y)))),
        int(max(0.0, min(1000.0, round(max_x + pad_x)))),
    ]


class ModelHandler:
    """Manages the 9Router vision client and runs Human Pose 17 inference for CVAT."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("POSE17_MODEL") or os.getenv("VISION_MODEL")
        self.refine_model = os.getenv("POSE17_REFINE_MODEL") or os.getenv("QUALITY_REFINE_MODEL")

        timeout_env = os.getenv("NINEROUTER_TIMEOUT")
        try:
            val_timeout = timeout if timeout is not None else (timeout_env or DEFAULT_NINEROUTER_TIMEOUT)
            self.timeout = float(val_timeout)
            if self.timeout <= 0:
                self.timeout = DEFAULT_NINEROUTER_TIMEOUT
        except (ValueError, TypeError):
            self.timeout = DEFAULT_NINEROUTER_TIMEOUT

        max_size_env = os.getenv("MAX_IMAGE_SIZE")
        try:
            val_size = max_image_size if max_image_size is not None else (max_size_env or DEFAULT_POSE17_MAX_IMAGE_SIZE)
            self.max_image_size = int(val_size)
            if self.max_image_size <= 0:
                self.max_image_size = DEFAULT_POSE17_MAX_IMAGE_SIZE
        except (ValueError, TypeError):
            self.max_image_size = DEFAULT_POSE17_MAX_IMAGE_SIZE

        max_tokens_env = os.getenv("MAX_TOKENS")
        try:
            val_tokens = max_tokens if max_tokens is not None else (max_tokens_env or DEFAULT_POSE17_MAX_TOKENS)
            self.max_tokens = int(val_tokens)
            if self.max_tokens <= 0:
                self.max_tokens = DEFAULT_POSE17_MAX_TOKENS
        except (ValueError, TypeError):
            self.max_tokens = DEFAULT_POSE17_MAX_TOKENS

        # Initialize client
        self.client = NineRouterClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Dynamic vision model resolution
        try:
            if hasattr(self.client, "resolve_pose17_model"):
                self.active_model: str = self.client.resolve_pose17_model(self.requested_model)
            else:
                self.active_model = self.client.resolve_vision_model(self.requested_model)
        except Exception as e:
            logger.error(f"Failed to resolve vision model {self.requested_model!r} from 9Router: {e}")
            raise

        # Resolve refinement model (falls back to active_model if not configured)
        self.active_refine_model = self.active_model
        if self.refine_model:
            try:
                self.active_refine_model = self.client.resolve_vision_model(self.refine_model)
                logger.info(f"Resolved refine model: {self.active_refine_model!r}")
            except Exception as e:
                logger.warning(f"Failed to resolve refine model {self.refine_model!r}, using primary: {e}")
                self.active_refine_model = self.active_model

        logger.info(
            f"Initialized ModelHandler (Human Pose 17): base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)}, model={self.active_model!r}, "
            f"refine_model={self.active_refine_model!r}, timeout={self.timeout}s"
        )
        logger.info(f"Pose17 spec fingerprint: {load_pose17().spec_fingerprint}")
        build_sha = get_build_sha()
        if build_sha:
            logger.info(f"TOOL_CVAT_BUILD_SHA: {build_sha}")

    def __repr__(self) -> str:
        """Safe string representation masking API keys."""
        return (
            f"ModelHandler(human_pose_17, base_url={self.base_url!r}, "
            f"key={mask_api_key(self.api_key)!r}, "
            f"model={self.active_model!r}, "
            f"timeout={self.timeout})"
        )

    def infer(
        self,
        image_bytes: bytes,
        threshold: float = 0.5,
        mode: Optional[str] = None,
        roi: Optional[Sequence[Union[int, float]]] = None,
        refine_crops: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Process incoming image bytes and return CVAT native skeleton shapes.

        Args:
            image_bytes: Raw JPEG/PNG image binary data.
            threshold: Minimum confidence score [0.0, 1.0].
            mode: Optional detection mode override.
            roi: Optional sub-region of interest [x1, y1, x2, y2].
            refine_crops: Whether to run two-pass crop refinement (defaults to POSE17_TWO_PASS_REFINE env var or True).

        Returns:
            List of CVAT skeleton shape dictionaries.
        """
        if not image_bytes:
            raise ValueError("Empty image payload received")

        t0 = time.perf_counter()

        # Load image and obtain dimensions
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size
            base_pil_img = img.convert("RGB")
            working_img = base_pil_img.copy()

            # Apply ROI if specified
            if roi is not None and len(roi) == 4:
                rx1 = max(0, min(orig_w - 1, int(round(float(roi[0])))))
                ry1 = max(0, min(orig_h - 1, int(round(float(roi[1])))))
                rx2 = max(rx1 + 1, min(orig_w, int(round(float(roi[2])))))
                ry2 = max(ry1 + 1, min(orig_h, int(round(float(roi[3])))))
                working_img = working_img.crop((rx1, ry1, rx2, ry2))

            # Resize if exceeding max size
            curr_w, curr_h = working_img.size
            if max(curr_w, curr_h) > self.max_image_size:
                ratio = self.max_image_size / float(max(curr_w, curr_h))
                new_w = max(1, int(round(curr_w * ratio)))
                new_h = max(1, int(round(curr_h * ratio)))
                working_img = working_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            working_img.save(buf, format="JPEG", quality=90)
            send_bytes = buf.getvalue()

        # Build prompt
        prompt = build_pose17_prompt(POSE17_KEYPOINTS)

        # Execute 9Router vision inference (Pass 1)
        resp = self.client.send_vision_request(
            model=self.active_model,
            image_bytes_or_b64=send_bytes,
            prompt=prompt,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Parse JSON
        cleaned = clean_json_string(resp.content)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse JSON from 9Router response: {e}. Content: {resp.content[:200]!r}")
            return []

        # Parse candidate person instances
        raw_people: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            if "people" in parsed and isinstance(parsed["people"], list):
                raw_people = [p for p in parsed["people"] if isinstance(p, dict)]
            elif "persons" in parsed and isinstance(parsed["persons"], list):
                raw_people = [p for p in parsed["persons"] if isinstance(p, dict)]
            elif "keypoints" in parsed:
                raw_people = [parsed]
        elif isinstance(parsed, list):
            raw_people = [p for p in parsed if isinstance(p, dict)]

        enable_two_pass = (
            refine_crops if refine_crops is not None
            else os.getenv("POSE17_TWO_PASS_REFINE", "1").lower() in ("1", "true", "yes")
        )

        shapes: List[Dict[str, Any]] = []
        for person_dict in raw_people:
            conf = float(person_dict.get("confidence", 1.0))
            if conf < threshold:
                continue

            # Quality gate initial assessment
            q_report = assess_pose17_quality(person_dict, laterality_convention=LATERALITY_VIEWER)

            best_person = person_dict
            best_report = q_report

            needs_refine = (
                q_report.needs_refine
                or not q_report.is_valid
                or q_report.score < 0.76
                or len(q_report.suspect_bones) > 0
            )

            if enable_two_pass and needs_refine:
                bbox = derive_person_bbox(person_dict, pad_ratio=0.18)
                if bbox:
                    best_person, best_report = self._refine_person_crop(
                        base_pil_img, person_dict, bbox, orig_w, orig_h, q_report
                    )

            # Filter degenerate collapse (<10px diagonal) or clumped corner dumps
            if not best_report.is_valid:
                has_fatal_collapse = any("degenerate_skeleton_collapse" in r for r in best_report.hard_errors)
                has_corner_dump = any("corner_cluster_dump" in r for r in best_report.hard_errors)
                if has_fatal_collapse or has_corner_dump:
                    logger.warning(f"Discarding corrupt/degenerate pose: {best_report.hard_errors}")
                    continue

            instance = parse_and_sanitize_pose17_instance(
                best_person,
                img_width=orig_w,
                img_height=orig_h,
                coord_range=1000.0,
            )
            if instance and instance.elements:
                shapes.append(instance.to_cvat_dict())

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Human Pose 17 inference completed in {elapsed:.2f}s: "
            f"detected {len(shapes)} person skeleton(s), "
            f"spec_fingerprint={load_pose17().spec_fingerprint[:12]}"
        )
        return shapes

    def _refine_person_crop(
        self,
        full_img: Image.Image,
        person_dict: Dict[str, Any],
        bbox: Sequence[Union[int, float]],
        orig_w: int,
        orig_h: int,
        initial_report: Pose17QualityReport,
    ) -> Tuple[Dict[str, Any], Pose17QualityReport]:
        """Execute second-pass person crop refinement with targeted 17-keypoint prompt.

        1. Expands person bounding box by 15-20% margin.
        2. Crops the high-resolution patch directly from unscaled original image.
        3. Invokes 9Router with targeted crop prompt enforcing kinematic chain.
        4. Transforms coordinates from crop-normalized space back to full image [0..1000] space.
        5. Compares quality score against initial candidate and returns the superior candidate.
        """
        try:
            c_ymin, c_xmin, c_ymax, c_xmax = [float(v) for v in bbox]
            px1 = max(0, int(round((c_xmin / 1000.0) * orig_w)))
            py1 = max(0, int(round((c_ymin / 1000.0) * orig_h)))
            px2 = min(orig_w, int(round((c_xmax / 1000.0) * orig_w)))
            py2 = min(orig_h, int(round((c_ymax / 1000.0) * orig_h)))

            crop_w = px2 - px1
            crop_h = py2 - py1
            if crop_w < 20 or crop_h < 20:
                return person_dict, initial_report

            crop_patch = full_img.crop((px1, py1, px2, py2))
            if max(crop_patch.size) > self.max_image_size:
                ratio = self.max_image_size / float(max(crop_patch.size))
                crop_patch = crop_patch.resize(
                    (max(1, int(round(crop_patch.width * ratio))), max(1, int(round(crop_patch.height * ratio)))),
                    Image.Resampling.LANCZOS,
                )

            buf = io.BytesIO()
            crop_patch.save(buf, format="JPEG", quality=92)
            crop_bytes = buf.getvalue()

            prompt = build_pose17_crop_prompt(POSE17_KEYPOINTS)
            resp = self.client.send_vision_request(
                model=self.active_refine_model,
                image_bytes_or_b64=crop_bytes,
                prompt=prompt,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )

            cleaned = clean_json_string(resp.content)
            parsed = json.loads(cleaned)

            crop_kps: Dict[str, Any] = {}
            if isinstance(parsed, dict):
                if "keypoints" in parsed and isinstance(parsed["keypoints"], dict):
                    crop_kps = parsed["keypoints"]
                elif "people" in parsed and isinstance(parsed["people"], list) and parsed["people"]:
                    p0 = parsed["people"][0]
                    if isinstance(p0, dict) and "keypoints" in p0:
                        crop_kps = p0["keypoints"]
            elif isinstance(parsed, list) and parsed:
                p0 = parsed[0]
                if isinstance(p0, dict) and "keypoints" in p0:
                    crop_kps = p0["keypoints"]

            if not crop_kps:
                return person_dict, initial_report

            # Reproject coordinates back to full image [0..1000]
            reprojected: Dict[str, List[Any]] = {}
            for name in POSE17_KEYPOINTS:
                pt_val = crop_kps.get(name)
                if pt_val is None or not isinstance(pt_val, (list, tuple)) or len(pt_val) < 2:
                    # Fall back to initial keypoint if missing in crop
                    orig_pt = person_dict.get("keypoints", {}).get(name)
                    if orig_pt:
                        reprojected[name] = list(orig_pt)
                    continue

                x_crop = float(pt_val[0])
                y_crop = float(pt_val[1])
                vis = int(pt_val[2]) if len(pt_val) > 2 else 2

                x_px = px1 + (x_crop / 1000.0) * crop_w
                y_px = py1 + (y_crop / 1000.0) * crop_h

                norm_x = max(0.0, min(1000.0, (x_px / float(orig_w)) * 1000.0))
                norm_y = max(0.0, min(1000.0, (y_px / float(orig_h)) * 1000.0))
                reprojected[name] = [round(norm_x, 1), round(norm_y, 1), vis]

            refined_person = dict(person_dict)
            refined_person["keypoints"] = reprojected
            refined_report = assess_pose17_quality(refined_person, laterality_convention=LATERALITY_VIEWER)

            # Accept refinement if it improves score or fixes hard errors
            if refined_report.is_valid and (not initial_report.is_valid or refined_report.score >= initial_report.score):
                logger.info(
                    f"Pose17 two-pass refinement accepted: score {initial_report.score:.2f} -> {refined_report.score:.2f}"
                )
                return refined_person, refined_report
            elif refined_report.score > initial_report.score + 0.05:
                logger.info(
                    f"Pose17 two-pass refinement accepted (score boost): {initial_report.score:.2f} -> {refined_report.score:.2f}"
                )
                return refined_person, refined_report
            else:
                logger.debug(
                    f"Pose17 two-pass refinement discarded: initial score {initial_report.score:.2f} vs refined {refined_report.score:.2f}"
                )
                return person_dict, initial_report

        except Exception as e:
            logger.warning(f"Pose17 two-pass refinement failed: {e}")
            return person_dict, initial_report

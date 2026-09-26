"""ModelHandler for 9Router Buddha Multi-Limb CVAT AI Detector.

Hierarchical Multi-Pass Pipeline:
1. Pass 1: Global Discovery (single central body, single face ROI, torso center, dynamic extra arms).
2. Pass 2: Arm Refinement (padded crop, root/elbow/wrist estimation, reprojection, dedup, angular sorting).
3. Pass 3: Hand Refinement (padded crop, 21-point hand landmarks, reprojection, 1:1 parent-arm linking).
4. Pass 4: Central Face (VF50 7-component facial landmark pipeline).
5. Pass 5: Central Body (Pose17 17-keypoint single person pipeline).
6. Pass 6: Merge into native CVAT skeleton shapes.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

# Ensure project modules (app, core, config) are importable
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
if (project_root / "app").exists():
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
elif (current_dir / "app").exists():
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

from app.client import NineRouterClient, NineRouterConnectionError, NineRouterError
from app.config import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_NINEROUTER_TIMEOUT,
    DEFAULT_NINEROUTER_URL_CONTAINER,
    DEFAULT_VISION_MODEL,
    mask_api_key,
)
from app.parser import clean_json_string
from core.buddha_contract import (
    BuddhaArmInstance,
    BuddhaArmKeypoint,
    BuddhaHand21Instance,
    BuddhaHand21Keypoint,
    assess_buddha_arm_geometry,
    assess_buddha_hand21_geometry,
    build_buddha_arm_refine_prompt,
    build_buddha_global_prompt,
    build_buddha_hand21_prompt,
    crop_norm_to_global_norm,
    deduplicate_arms,
    link_hands_to_arms,
    load_buddha_schema,
    resolve_buddha_model,
    sort_arms_deterministically,
)
from core.pose_face_schema import (
    POSE17_KEYPOINTS,
    parse_and_sanitize_pose17_instance,
)
from core.skeleton_contract import (
    VF50_COMPONENT_NAMES,
    VF50Face,
    build_vf50_crop_prompt,
    expand_face_bbox,
    parse_vf50_response,
)
from core.week2_schema import (
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUTSIDE,
    VISIBILITY_VISIBLE,
    get_build_sha,
    normalize_visibility,
)

logger = logging.getLogger("cvat.nuclio.ninerouter.buddha_multilimbs")


class ModelHandler:
    """Manages 9Router client and runs hierarchical Buddha multi-limb perception."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        refine_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_arms: Optional[int] = None,
        max_hand_refinements: Optional[int] = None,
        two_pass: Optional[bool] = None,
        hand_refine: Optional[bool] = None,
        refine_workers: Optional[int] = None,
        min_confidence: Optional[float] = None,
        allow_fallback: Optional[bool] = None,
    ):
        self.schema = load_buddha_schema()
        self.base_url = base_url or os.getenv("NINEROUTER_URL", DEFAULT_NINEROUTER_URL_CONTAINER)
        self.api_key = api_key or os.getenv("NINEROUTER_KEY")
        self.requested_model = model or os.getenv("BUDDHA_MODEL") or os.getenv("VISION_MODEL")
        self.requested_refine_model = (
            refine_model
            or os.getenv("BUDDHA_REFINE_MODEL")
            or os.getenv("BUDDHA_MODEL")
            or os.getenv("VISION_MODEL")
        )

        timeout_env = os.getenv("NINEROUTER_TIMEOUT")
        self.timeout = float(timeout if timeout is not None else (timeout_env or 120.0))

        img_size_env = os.getenv("MAX_IMAGE_SIZE")
        self.max_image_size = int(max_image_size if max_image_size is not None else (img_size_env or 1280))

        tokens_env = os.getenv("MAX_TOKENS")
        self.max_tokens = int(max_tokens if max_tokens is not None else (tokens_env or self.schema.global_max_tokens))

        arms_env = os.getenv("BUDDHA_MAX_ARMS")
        self.max_arms = int(max_arms if max_arms is not None else (arms_env or self.schema.max_arms))

        hands_env = os.getenv("BUDDHA_MAX_HAND_REFINEMENTS")
        self.max_hand_refinements = int(
            max_hand_refinements if max_hand_refinements is not None else (hands_env or self.schema.max_hand_refinements)
        )

        tp_env = os.getenv("BUDDHA_TWO_PASS", "1").strip().lower()
        self.two_pass = two_pass if two_pass is not None else (tp_env in ("1", "true", "yes"))

        hr_env = os.getenv("BUDDHA_HAND_REFINE", "1").strip().lower()
        self.hand_refine = hand_refine if hand_refine is not None else (hr_env in ("1", "true", "yes"))

        workers_env = os.getenv("BUDDHA_REFINE_WORKERS")
        self.refine_workers = int(refine_workers if refine_workers is not None else (workers_env or self.schema.refine_workers))
        self.refine_workers = max(1, min(8, self.refine_workers))

        conf_env = os.getenv("BUDDHA_MIN_CONFIDENCE")
        self.min_confidence = float(min_confidence if min_confidence is not None else (conf_env or self.schema.min_confidence))

        fb_env = os.getenv("BUDDHA_ALLOW_MODEL_FALLBACK", "1").strip().lower()
        self.allow_fallback = allow_fallback if allow_fallback is not None else (fb_env in ("1", "true", "yes"))

        # Initialize client
        self.client = NineRouterClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Resolve models prioritizing Claude Opus 5.5
        try:
            self.active_model = self.client.resolve_buddha_model(
                preferred_model=self.requested_model,
                allow_fallback=self.allow_fallback,
            )
        except Exception as e:
            logger.warning(f"Buddha primary model resolution error: {e}. Falling back to default.")
            self.active_model = self.requested_model or "ag/claude-opus-4-6-thinking"

        try:
            self.active_refine_model = self.client.resolve_buddha_model(
                preferred_model=self.requested_refine_model or self.active_model,
                allow_fallback=self.allow_fallback,
            )
        except Exception as e:
            self.active_refine_model = self.active_model

        masked_key = mask_api_key(self.api_key)
        build_sha = get_build_sha()
        sha_str = build_sha[:8] if build_sha else "unknown"
        logger.info(
            f"Initialized 9Router Buddha Multi-Limb ModelHandler: "
            f"base_url={self.base_url}, active_model={self.active_model}, "
            f"active_refine_model={self.active_refine_model}, "
            f"timeout={self.timeout}s, max_arms={self.max_arms}, "
            f"max_hand_refinements={self.max_hand_refinements}, "
            f"refine_workers={self.refine_workers}, "
            f"two_pass={self.two_pass}, hand_refine={self.hand_refine}, "
            f"build_sha={sha_str}, api_key={masked_key}"
        )

    def handle_image(
        self,
        image_bytes_or_pil: Union[bytes, Image.Image],
        roi: Optional[Sequence[Union[int, float]]] = None,
        context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Run hierarchical Buddha multi-limb perception on input image.

        Args:
            image_bytes_or_pil: Raw JPEG/PNG image bytes or PIL Image.
            roi: Optional bounding box [x1, y1, x2, y2] in pixel space.
            context: Optional Nuclio execution context for logging.

        Returns:
            List of native CVAT skeleton shape dictionaries.
        """
        t0 = time.perf_counter()

        # 1. Load full resolution image
        if isinstance(image_bytes_or_pil, Image.Image):
            base_pil_img = image_bytes_or_pil.convert("RGB")
        else:
            base_pil_img = Image.open(io.BytesIO(image_bytes_or_pil)).convert("RGB")

        orig_w, orig_h = base_pil_img.size
        working_img = base_pil_img.copy()

        # Apply ROI if specified
        roi_box_norm: Optional[List[int]] = None
        if roi is not None and len(roi) == 4:
            rx1 = max(0, min(orig_w - 1, int(round(float(roi[0])))))
            ry1 = max(0, min(orig_h - 1, int(round(float(roi[1])))))
            rx2 = max(rx1 + 1, min(orig_w, int(round(float(roi[2])))))
            ry2 = max(ry1 + 1, min(orig_h, int(round(float(roi[3])))))
            working_img = working_img.crop((rx1, ry1, rx2, ry2))
            roi_box_norm = [
                int(round(ry1 / float(orig_h) * 1000.0)),
                int(round(rx1 / float(orig_w) * 1000.0)),
                int(round(ry2 / float(orig_h) * 1000.0)),
                int(round(rx2 / float(orig_w) * 1000.0)),
            ]

        # Resize for Pass 1 Global Inference
        curr_w, curr_h = working_img.size
        if max(curr_w, curr_h) > self.max_image_size:
            ratio = self.max_image_size / float(max(curr_w, curr_h))
            new_w = max(1, int(round(curr_w * ratio)))
            new_h = max(1, int(round(curr_h * ratio)))
            working_img = working_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        working_img.save(buf, format="JPEG", quality=90)
        global_img_bytes = buf.getvalue()

        # =====================================================================
        # PASS 1: GLOBAL SCENE DISCOVERY
        # =====================================================================
        global_prompt = build_buddha_global_prompt()
        logger.info(f"Executing Pass 1 Global Discovery using model '{self.active_model}'...")

        try:
            resp = self.client.send_vision_request(
                model=self.active_model,
                image_bytes_or_b64=global_img_bytes,
                prompt=global_prompt,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            cleaned = clean_json_string(resp.content)
            parsed_global = json.loads(cleaned)
        except Exception as e:
            logger.error(f"Pass 1 Global Discovery failed: {e}")
            return []

        if not isinstance(parsed_global, dict):
            logger.warning("Pass 1 response did not return a valid dictionary.")
            return []

        # Extract elements from Pass 1
        raw_torso = parsed_global.get("torso_center")
        torso_center: Tuple[float, float] = (500.0, 500.0)
        if isinstance(raw_torso, (list, tuple)) and len(raw_torso) >= 2:
            torso_center = (float(raw_torso[0]), float(raw_torso[1]))

        raw_face_roi = parsed_global.get("face_roi")
        raw_central_body = parsed_global.get("central_body")
        raw_arms = parsed_global.get("arms", [])
        if not isinstance(raw_arms, list):
            raw_arms = []

        logger.info(
            f"Pass 1 Discovered: torso_center={torso_center}, "
            f"has_face_roi={raw_face_roi is not None}, "
            f"has_central_body={raw_central_body is not None}, "
            f"candidate_arms={len(raw_arms)}"
        )

        # Map ROI coordinates back to full image space if ROI was used
        if roi_box_norm is not None and len(roi_box_norm) == 4:
            torso_center = crop_norm_to_global_norm(torso_center, roi_box_norm)
            if raw_face_roi and len(raw_face_roi) == 4:
                ymin, xmin, ymax, xmax = [float(v) for v in raw_face_roi]
                pt1 = crop_norm_to_global_norm((xmin, ymin), roi_box_norm)
                pt2 = crop_norm_to_global_norm((xmax, ymax), roi_box_norm)
                raw_face_roi = [int(round(pt1[1])), int(round(pt1[0])), int(round(pt2[1])), int(round(pt2[0]))]

        # Parse candidate arm instances
        arm_candidates: List[BuddhaArmInstance] = []
        for idx, a_dict in enumerate(raw_arms, start=1):
            if not isinstance(a_dict, dict):
                continue
            root_pt = a_dict.get("root")
            elbow_pt = a_dict.get("elbow")
            wrist_pt = a_dict.get("wrist")

            if not (root_pt and elbow_pt and wrist_pt):
                continue

            rx = float(root_pt[0])
            ry = float(root_pt[1])
            rv = int(root_pt[2]) if len(root_pt) > 2 else VISIBILITY_VISIBLE

            ex = float(elbow_pt[0])
            ey = float(elbow_pt[1])
            ev = int(elbow_pt[2]) if len(elbow_pt) > 2 else VISIBILITY_VISIBLE

            wx = float(wrist_pt[0])
            wy = float(wrist_pt[1])
            wv = int(wrist_pt[2]) if len(wrist_pt) > 2 else VISIBILITY_VISIBLE

            if roi_box_norm is not None:
                rx, ry = crop_norm_to_global_norm((rx, ry), roi_box_norm)
                ex, ey = crop_norm_to_global_norm((ex, ey), roi_box_norm)
                wx, wy = crop_norm_to_global_norm((wx, wy), roi_box_norm)

            conf = float(a_dict.get("confidence", 0.90))
            if conf < self.min_confidence:
                continue

            hand_roi = a_dict.get("hand_roi")
            if hand_roi and len(hand_roi) == 4 and roi_box_norm is not None:
                h_ymin, h_xmin, h_ymax, h_xmax = [float(v) for v in hand_roi]
                h_pt1 = crop_norm_to_global_norm((h_xmin, h_ymin), roi_box_norm)
                h_pt2 = crop_norm_to_global_norm((h_xmax, h_ymax), roi_box_norm)
                hand_roi = [int(round(h_pt1[1])), int(round(h_pt1[0])), int(round(h_pt2[1])), int(round(h_pt2[0]))]

            arm_cand = BuddhaArmInstance(
                arm_id=f"arm_{idx:03d}",
                root=BuddhaArmKeypoint(name="root", x=rx, y=ry, visibility=rv, confidence=conf),
                elbow=BuddhaArmKeypoint(name="elbow", x=ex, y=ey, visibility=ev, confidence=conf),
                wrist=BuddhaArmKeypoint(name="wrist", x=wx, y=wy, visibility=wv, confidence=conf),
                confidence=conf,
                hand_roi=hand_roi,
            )
            is_valid, errs = assess_buddha_arm_geometry(arm_cand, img_w=orig_w, img_h=orig_h)
            if is_valid:
                arm_candidates.append(arm_cand)
            else:
                logger.debug(f"Discarded invalid candidate arm #{idx}: {errs}")

        # Deduplicate Pass 1 arms
        arm_candidates = deduplicate_arms(
            arm_candidates,
            wrist_dist_thresh=self.schema.wrist_dist_thresh,
            elbow_dist_thresh=self.schema.elbow_dist_thresh,
            root_dist_thresh=self.schema.root_dist_thresh,
            dir_sim_thresh=self.schema.dir_sim_thresh,
            hand_iou_thresh=self.schema.hand_iou_thresh,
        )

        # =====================================================================
        # PASS 2: ARM REFINEMENT (Optional Two-Pass)
        # =====================================================================
        refined_arms: List[BuddhaArmInstance] = []
        if self.two_pass and arm_candidates:
            logger.info(f"Executing Pass 2 Arm Refinement on {len(arm_candidates)} candidates...")
            for cand in arm_candidates:
                try:
                    ref_arm = self._refine_single_arm_crop(base_pil_img, cand, orig_w, orig_h)
                    refined_arms.append(ref_arm)
                except Exception as e:
                    logger.warning(f"Refinement failed for arm {cand.arm_id}: {e}. Retaining Pass 1 candidate.")
                    refined_arms.append(cand)
        else:
            refined_arms = arm_candidates

        # Deduplicate refined arms
        refined_arms = deduplicate_arms(
            refined_arms,
            wrist_dist_thresh=self.schema.wrist_dist_thresh,
            elbow_dist_thresh=self.schema.elbow_dist_thresh,
            root_dist_thresh=self.schema.root_dist_thresh,
            dir_sim_thresh=self.schema.dir_sim_thresh,
            hand_iou_thresh=self.schema.hand_iou_thresh,
        )

        # Sort arms deterministically around torso center
        sorted_arms = sort_arms_deterministically(refined_arms, torso_center=torso_center)

        # Enforce maximum arms limit safely
        if len(sorted_arms) > self.max_arms:
            logger.warning(
                f"Arm count {len(sorted_arms)} exceeded BUDDHA_MAX_ARMS ({self.max_arms}). "
                f"Clamping to {self.max_arms} highest confidence arms."
            )
            # Re-sort by confidence, take top max_arms, then re-sort deterministically
            top_arms = sorted(sorted_arms, key=lambda a: a.confidence, reverse=True)[: self.max_arms]
            sorted_arms = sort_arms_deterministically(top_arms, torso_center=torso_center)

        # =====================================================================
        # PASS 3: HAND REFINEMENT (21-Point Landmarks)
        # =====================================================================
        hand_instances: List[BuddhaHand21Instance] = []
        if self.hand_refine and sorted_arms:
            hands_to_refine = sorted_arms[: self.max_hand_refinements]
            logger.info(
                f"Executing Pass 3 Hand Refinement on {len(hands_to_refine)} arms "
                f"(concurrency workers: {self.refine_workers})..."
            )

            if self.refine_workers > 1 and len(hands_to_refine) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.refine_workers) as executor:
                    future_to_arm = {
                        executor.submit(self._refine_single_hand_crop, base_pil_img, arm, orig_w, orig_h): arm
                        for arm in hands_to_refine
                    }
                    for future in concurrent.futures.as_completed(future_to_arm):
                        arm_ref = future_to_arm[future]
                        try:
                            hand_inst = future.result()
                            if hand_inst:
                                hand_instances.append(hand_inst)
                        except Exception as e:
                            logger.warning(f"Hand refinement failed for arm {arm_ref.arm_id}: {e}")
            else:
                for arm in hands_to_refine:
                    try:
                        hand_inst = self._refine_single_hand_crop(base_pil_img, arm, orig_w, orig_h)
                        if hand_inst:
                            hand_instances.append(hand_inst)
                    except Exception as e:
                        logger.warning(f"Hand refinement failed for arm {arm.arm_id}: {e}")

        # Deterministically link hands to arms (matching group_id)
        sorted_arms, linked_hands = link_hands_to_arms(sorted_arms, hand_instances)

        # =====================================================================
        # PASS 4: CENTRAL FACE (VF50 7-Component Facial Landmarks)
        # =====================================================================
        face_skeletons: List[Dict[str, Any]] = []
        try:
            face_skeletons = self._process_central_face(
                base_pil_img,
                face_roi=raw_face_roi,
                orig_w=orig_w,
                orig_h=orig_h,
            )
        except Exception as e:
            logger.warning(f"Central face VF-50 processing failed: {e}")

        # =====================================================================
        # PASS 5: CENTRAL BODY (Pose17)
        # =====================================================================
        body_skeletons: List[Dict[str, Any]] = []
        try:
            body_skeletons = self._process_central_body(
                raw_central_body,
                roi_box_norm=roi_box_norm,
                orig_w=orig_w,
                orig_h=orig_h,
            )
        except Exception as e:
            logger.warning(f"Central body Pose-17 processing failed: {e}")

        # =====================================================================
        # PASS 6: MERGE ALL DETECTIONS INTO NATIVE CVAT SKELETON SHAPES
        # =====================================================================
        all_shapes: List[Dict[str, Any]] = []

        # 1. Central Body
        all_shapes.extend(body_skeletons)

        # 2. Central Face (7 component skeletons)
        all_shapes.extend(face_skeletons)

        # 3. Extra Buddha Arms
        for arm in sorted_arms:
            all_shapes.append(arm.to_cvat_skeleton(orig_w, orig_h))

        # 4. Buddha Hands
        for hand in linked_hands:
            all_shapes.append(hand.to_cvat_skeleton(orig_w, orig_h))

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Buddha Multi-Limb inference complete in {elapsed:.2f}s: "
            f"Body={len(body_skeletons)}, FaceComps={len(face_skeletons)}, "
            f"Arms={len(sorted_arms)}, Hands={len(linked_hands)}, "
            f"TotalShapes={len(all_shapes)}"
        )
        return all_shapes

    # ─── Internal Refinement Helpers ───────────────────────────────────────────

    def _refine_single_arm_crop(
        self,
        full_img: Image.Image,
        arm: BuddhaArmInstance,
        orig_w: int,
        orig_h: int,
    ) -> BuddhaArmInstance:
        """Refine a single arm in an isolated high-resolution crop."""
        arm_roi_norm = arm.derive_arm_roi(pad_ratio=self.schema.arm_crop_padding)
        ymin, xmin, ymax, xmax = arm_roi_norm

        px1 = max(0, int(round((xmin / 1000.0) * orig_w)))
        py1 = max(0, int(round((ymin / 1000.0) * orig_h)))
        px2 = min(orig_w, int(round((xmax / 1000.0) * orig_w)))
        py2 = min(orig_h, int(round((ymax / 1000.0) * orig_h)))

        if (px2 - px1) < 20 or (py2 - py1) < 20:
            return arm

        crop_patch = full_img.crop((px1, py1, px2, py2))
        buf = io.BytesIO()
        crop_patch.save(buf, format="JPEG", quality=92)
        crop_bytes = buf.getvalue()

        prompt = build_buddha_arm_refine_prompt()
        resp = self.client.send_vision_request(
            model=self.active_refine_model,
            image_bytes_or_b64=crop_bytes,
            prompt=prompt,
            max_tokens=self.schema.arm_max_tokens,
            timeout=self.timeout,
        )

        cleaned = clean_json_string(resp.content)
        parsed = json.loads(cleaned)

        root_pt = parsed.get("root")
        elbow_pt = parsed.get("elbow")
        wrist_pt = parsed.get("wrist")

        if not (root_pt and elbow_pt and wrist_pt):
            return arm

        rx, ry = crop_norm_to_global_norm((float(root_pt[0]), float(root_pt[1])), arm_roi_norm)
        ex, ey = crop_norm_to_global_norm((float(elbow_pt[0]), float(elbow_pt[1])), arm_roi_norm)
        wx, wy = crop_norm_to_global_norm((float(wrist_pt[0]), float(wrist_pt[1])), arm_roi_norm)

        rv = int(root_pt[2]) if len(root_pt) > 2 else arm.root.visibility
        ev = int(elbow_pt[2]) if len(elbow_pt) > 2 else arm.elbow.visibility
        wv = int(wrist_pt[2]) if len(wrist_pt) > 2 else arm.wrist.visibility

        conf = float(parsed.get("confidence", arm.confidence))

        refined = BuddhaArmInstance(
            arm_id=arm.arm_id,
            root=BuddhaArmKeypoint(name="root", x=rx, y=ry, visibility=rv, confidence=conf),
            elbow=BuddhaArmKeypoint(name="elbow", x=ex, y=ey, visibility=ev, confidence=conf),
            wrist=BuddhaArmKeypoint(name="wrist", x=wx, y=wy, visibility=wv, confidence=conf),
            confidence=max(arm.confidence, conf),
            hand_roi=arm.hand_roi,
            group_id=arm.group_id,
            side=arm.side,
        )
        is_valid, _ = assess_buddha_arm_geometry(refined, img_w=orig_w, img_h=orig_h)
        return refined if is_valid else arm

    def _refine_single_hand_crop(
        self,
        full_img: Image.Image,
        arm: BuddhaArmInstance,
        orig_w: int,
        orig_h: int,
    ) -> Optional[BuddhaHand21Instance]:
        """Refine a 21-point hand skeleton in an isolated high-resolution crop."""
        hand_roi_norm = arm.derive_hand_roi()
        ymin, xmin, ymax, xmax = hand_roi_norm

        px1 = max(0, int(round((xmin / 1000.0) * orig_w)))
        py1 = max(0, int(round((ymin / 1000.0) * orig_h)))
        px2 = min(orig_w, int(round((xmax / 1000.0) * orig_w)))
        py2 = min(orig_h, int(round((ymax / 1000.0) * orig_h)))

        if (px2 - px1) < 15 or (py2 - py1) < 15:
            return None

        crop_patch = full_img.crop((px1, py1, px2, py2))
        buf = io.BytesIO()
        crop_patch.save(buf, format="JPEG", quality=92)
        crop_bytes = buf.getvalue()

        prompt = build_buddha_hand21_prompt()
        resp = self.client.send_vision_request(
            model=self.active_refine_model,
            image_bytes_or_b64=crop_bytes,
            prompt=prompt,
            max_tokens=self.schema.hand_max_tokens,
            timeout=self.timeout,
        )

        cleaned = clean_json_string(resp.content)
        parsed = json.loads(cleaned)

        raw_lms = parsed.get("landmarks", {})
        if not isinstance(raw_lms, dict) or len(raw_lms) < 10:
            return None

        landmarks_map: Dict[int, BuddhaHand21Keypoint] = {}
        for kp_def in self.schema.hand_keypoints:
            pt_id = kp_def.id
            pt_val = raw_lms.get(str(pt_id)) or raw_lms.get(pt_id)
            if pt_val and len(pt_val) >= 2:
                cx, cy = float(pt_val[0]), float(pt_val[1])
                vis = int(pt_val[2]) if len(pt_val) > 2 else VISIBILITY_VISIBLE
                gx, gy = crop_norm_to_global_norm((cx, cy), hand_roi_norm)
                landmarks_map[pt_id] = BuddhaHand21Keypoint(
                    id=pt_id,
                    name=kp_def.name,
                    x=gx,
                    y=gy,
                    visibility=vis,
                    confidence=float(parsed.get("confidence", 0.90)),
                )
            else:
                landmarks_map[pt_id] = BuddhaHand21Keypoint(
                    id=pt_id,
                    name=kp_def.name,
                    x=arm.wrist.x,
                    y=arm.wrist.y,
                    visibility=VISIBILITY_OUTSIDE,
                    confidence=0.0,
                )

        hand_inst = BuddhaHand21Instance(
            hand_id=f"hand_for_{arm.arm_id}",
            parent_arm_id=arm.arm_id,
            landmarks=landmarks_map,
            confidence=float(parsed.get("confidence", 0.90)),
            group_id=arm.group_id,
        )
        is_valid, _ = assess_buddha_hand21_geometry(hand_inst, img_w=orig_w, img_h=orig_h)
        return hand_inst if is_valid else None

    def _process_central_face(
        self,
        full_img: Image.Image,
        face_roi: Optional[Sequence[Union[int, float]]],
        orig_w: int,
        orig_h: int,
    ) -> List[Dict[str, Any]]:
        """Process central face and return 7 VF50 component skeletons."""
        if not face_roi or len(face_roi) != 4:
            return []

        ymin, xmin, ymax, xmax = [float(v) for v in face_roi]
        expanded = expand_face_bbox([ymin, xmin, ymax, xmax], pad_ratio=0.20)
        c_ymin, c_xmin, c_ymax, c_xmax = expanded

        px1 = max(0, int(round((c_xmin / 1000.0) * orig_w)))
        py1 = max(0, int(round((c_ymin / 1000.0) * orig_h)))
        px2 = min(orig_w, int(round((c_xmax / 1000.0) * orig_w)))
        py2 = min(orig_h, int(round((c_ymax / 1000.0) * orig_h)))

        if (px2 - px1) < 25 or (py2 - py1) < 25:
            return []

        crop_patch = full_img.crop((px1, py1, px2, py2))
        buf = io.BytesIO()
        crop_patch.save(buf, format="JPEG", quality=92)
        crop_bytes = buf.getvalue()

        prompt = build_vf50_crop_prompt()
        resp = self.client.send_vision_request(
            model=self.active_refine_model,
            image_bytes_or_b64=crop_bytes,
            prompt=prompt,
            max_tokens=2500,
            timeout=self.timeout,
        )

        faces = parse_vf50_response(
            resp.content,
            img_width=orig_w,
            img_height=orig_h,
            crop_box=expanded,
        )
        if not faces:
            return []

        # Return 7 component skeletons with group_id = 1 (central figure)
        primary_face = faces[0]
        return primary_face.to_cvat_component_skeletons(width=orig_w, height=orig_h, group_id=1)

    def _process_central_body(
        self,
        raw_body: Optional[Dict[str, Any]],
        roi_box_norm: Optional[List[int]],
        orig_w: int,
        orig_h: int,
    ) -> List[Dict[str, Any]]:
        """Process central body Pose17 instance and return CVAT skeleton shape."""
        if not raw_body or not isinstance(raw_body, dict):
            return []

        if roi_box_norm is not None and "keypoints" in raw_body:
            kps = raw_body["keypoints"]
            if isinstance(kps, dict):
                for k_name, pt_val in kps.items():
                    if isinstance(pt_val, (list, tuple)) and len(pt_val) >= 2:
                        gx, gy = crop_norm_to_global_norm((float(pt_val[0]), float(pt_val[1])), roi_box_norm)
                        kps[k_name] = [gx, gy] + list(pt_val[2:])

        instance = parse_and_sanitize_pose17_instance(
            raw_body,
            img_width=orig_w,
            img_height=orig_h,
            coord_range=1000.0,
        )
        if not instance or not instance.elements:
            return []

        body_dict = instance.to_cvat_dict(numeric_sublabels=True)
        body_dict["group_id"] = 1
        body_dict["group"] = 1
        return [body_dict]

    def infer(
        self,
        image_bytes: Union[bytes, Image.Image],
        threshold: float = 0.5,
        mode: Optional[str] = None,
        roi: Optional[Sequence[Union[int, float]]] = None,
        context: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Inference entrypoint compatible with Nuclio detector handler."""
        return self.handle_image(image_bytes_or_pil=image_bytes, roi=roi, context=context)


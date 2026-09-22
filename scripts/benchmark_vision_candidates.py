"""Benchmark candidate 9Router vision models for Week-2 Pose17 and VF50 tasks.

Measures:
- Remote latency (seconds)
- Token consumption (prompt, completion, total)
- JSON formatting reliability
- Point completeness (17/17 for Pose17, 50/50 for VF50)
- Coordinate validity in [0, 1000]
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = os.getenv("NINEROUTER_BASE_URL", "http://127.0.0.1:20128")

# Candidate models to evaluate
CANDIDATE_MODELS = [
    "ag/gemini-3.8-flash-low",
    "ag/gemini-3.8-flash-medium",
    "ag/gemini-3.8-flash-high",
    "ag/gemini-3.7-flash-low",
    "ag/claude-sonnet-4-6",
]

# 1. Pose17 Prompt
SYSTEM_PROMPT_POSE17 = """You are an expert human pose estimation vision model.
Your task is to detect all human persons in the image and estimate their 17 body keypoints.
You must output ONLY a valid JSON object. Do not include markdown (no ```json code blocks), no explanations, and no conversational text.

Return your response adhering strictly to this schema:
{
  "persons": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "confidence": 0.95,
      "keypoints": {
        "nose": [x, y, visibility],
        "right_eye": [x, y, visibility],
        "left_eye": [x, y, visibility],
        "right_ear": [x, y, visibility],
        "left_ear": [x, y, visibility],
        "right_shoulder": [x, y, visibility],
        "left_shoulder": [x, y, visibility],
        "right_elbow": [x, y, visibility],
        "left_elbow": [x, y, visibility],
        "right_wrist": [x, y, visibility],
        "left_wrist": [x, y, visibility],
        "right_hip": [x, y, visibility],
        "left_hip": [x, y, visibility],
        "right_knee": [x, y, visibility],
        "left_knee": [x, y, visibility],
        "right_ankle": [x, y, visibility],
        "left_ankle": [x, y, visibility]
      }
    }
  ]
}

Rules:
1. Coordinate System:
   - box_2d is [ymin, xmin, ymax, xmax] in normalized integers [0, 1000].
   - Each keypoint is [x, y, visibility] with x and y normalized integers [0, 1000] (x horizontal, y vertical).
2. Laterality Convention (VinFast frame convention):
   - Right (right_*) is on the RIGHT side of the image as viewed by observer.
   - Left (left_*) is on the LEFT side of the image as viewed by observer.
3. Visibility:
   - 2: Clearly visible in the image.
   - 1: Occluded or hidden behind object/body, but position reliably inferrable.
   - 0: Outside image boundaries or completely unobservable.
4. Exact Keys:
   - Exactly the 17 specified keypoint names must be present for every person.
   - Do NOT invent keypoints or change naming.
5. If no person is visible, return {"persons": []}.
"""

# 2. VF50 Prompt
SYSTEM_PROMPT_VF50 = """You are an expert facial landmark analysis vision model.
Your task is to detect faces in the image and estimate the exact 50 VinFast VF-50 facial landmarks.
You must output ONLY a valid JSON object. Do not include markdown (no ```json code blocks), no explanations, and no conversational text.

Return your response adhering strictly to this schema:
{
  "faces": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "confidence": 0.98,
      "landmarks": {
        "0": [x, y, visibility],
        "1": [x, y, visibility],
        ...
        "49": [x, y, visibility]
      }
    }
  ]
}

VF-50 Landmark Partition (IDs 0 to 49):
- longmaytrai (0-4): Left eyebrow (observer's left), 5 points from outer tail (0) to inner head (4).
- longmayphai (5-9): Right eyebrow (observer's right), 5 points from inner head (5) to outer tail (9).
- songmui (10-13): Nose bridge, 4 points along bridge from top between eyes (10) down to above nostrils (13).
- mattrai (14-21): Left eye (observer's left), 8 points along eyelid contour starting at outer corner (14) -> upper lid (15, 16, 17) -> inner corner (18) -> lower lid (19, 20, 21).
- matphai (22-29): Right eye (observer's right), 8 points starting at inner corner (22) -> upper lid (23, 24, 25) -> outer corner (26) -> lower lid (27, 28, 29).
- moingoai (30-41): Outer lip contour, 12 points from left corner (30) -> upper lip (31-35) -> right corner (36) -> lower lip (37-41).
- moitrong (42-49): Inner lip contour, 8 points from left corner (42) -> upper inner (43-45) -> right corner (46) -> lower inner (47-49).

Rules:
1. Coordinates are normalized integers in [0, 1000] (x horizontal, y vertical).
2. Laterality: Observer perspective (left on image is *trai, right on image is *phai).
3. Visibility: 2 = Visible, 1 = Occluded, 0 = Outside.
4. Landmarks dict MUST contain string keys "0" through "49" (all 50 points).
5. Output pure JSON only. If no face is detected, return {"faces": []}.
"""


def execute_request(model: str, system_prompt: str, user_prompt: str, img_b64: str, timeout: float = 30.0) -> Dict[str, Any]:
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]
    }

    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    client_sec = time.perf_counter() - t0

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "client_sec": client_sec}

    ct = resp.headers.get("content-type", "")
    content = ""
    usage = {}
    if "event-stream" in ct:
        parts = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    c = json.loads(line[6:])
                    delta = c.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        parts.append(delta["content"])
                    if "usage" in c:
                        usage = c["usage"]
                except Exception:
                    pass
        content = "".join(parts)
    else:
        try:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
        except Exception as e:
            return {"error": f"JSON decode failed: {e}", "client_sec": client_sec}

    return {
        "content": content,
        "client_sec": client_sec,
        "usage": usage,
    }


def evaluate_pose17(content: str) -> Dict[str, Any]:
    # Clean markdown if present
    clean = content.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(clean)
    except Exception as e:
        return {"valid_json": False, "error": str(e)}

    persons = data.get("persons", [])
    if not isinstance(persons, list) or len(persons) == 0:
        return {"valid_json": True, "num_persons": 0, "pts_count": 0, "valid_keys": False}

    p0 = persons[0]
    kps = p0.get("keypoints", {})
    expected = [
        "nose", "right_eye", "left_eye", "right_ear", "left_ear",
        "right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
        "right_wrist", "left_wrist", "right_hip", "left_hip",
        "right_knee", "left_knee", "right_ankle", "left_ankle"
    ]
    present_keys = [k for k in expected if k in kps]
    valid_coords = True
    for k in present_keys:
        pt = kps[k]
        if not isinstance(pt, list) or len(pt) < 2:
            valid_coords = False
            break
        if not (0 <= pt[0] <= 1000 and 0 <= pt[1] <= 1000):
            valid_coords = False
            break

    return {
        "valid_json": True,
        "num_persons": len(persons),
        "pts_count": len(present_keys),
        "total_expected": len(expected),
        "valid_keys": len(present_keys) == len(expected),
        "valid_coords": valid_coords,
        "sample": {k: kps[k] for k in present_keys[:3]}
    }


def evaluate_vf50(content: str) -> Dict[str, Any]:
    clean = content.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(clean)
    except Exception as e:
        return {"valid_json": False, "error": str(e)}

    faces = data.get("faces", [])
    if not isinstance(faces, list) or len(faces) == 0:
        return {"valid_json": True, "num_faces": 0, "pts_count": 0, "valid_keys": False}

    f0 = faces[0]
    lms = f0.get("landmarks", {})
    expected_ids = [str(i) for i in range(50)]
    present_ids = [k for k in expected_ids if k in lms]
    valid_coords = True
    for k in present_ids:
        pt = lms[k]
        if not isinstance(pt, list) or len(pt) < 2:
            valid_coords = False
            break
        if not (0 <= pt[0] <= 1000 and 0 <= pt[1] <= 1000):
            valid_coords = False
            break

    return {
        "valid_json": True,
        "num_faces": len(faces),
        "pts_count": len(present_ids),
        "total_expected": 50,
        "valid_keys": len(present_ids) == 50,
        "valid_coords": valid_coords,
        "sample": {k: lms[k] for k in present_ids[:4]}
    }


def main():
    print("=================================================================")
    print(" 9Router Vision Candidate Benchmark - Pose17 & VF50")
    print("=================================================================")

    # Test images
    img_driver = Path("test_driver.jpg")
    img_road = Path("test.jpg")

    if not img_driver.exists():
        print(f"Error: {img_driver} not found")
        return

    with open(img_driver, "rb") as f:
        b64_driver = base64.b64encode(f.read()).decode("ascii")

    with open(img_road, "rb") as f:
        b64_road = base64.b64encode(f.read()).decode("ascii")

    results = []

    # 1. Benchmark Pose17
    print("\n[Phase 1] Benchmarking POSE17 on Driver Image...")
    for model in CANDIDATE_MODELS:
        print(f"--> Testing POSE17 on {model}...")
        try:
            res = execute_request(
                model=model,
                system_prompt=SYSTEM_PROMPT_POSE17,
                user_prompt="Detect human pose keypoints for the driver person in this image.",
                img_b64=b64_driver,
                timeout=25.0,
            )
            if "error" in res:
                print(f"    FAILED: {res['error']}")
                results.append({"task": "pose17", "model": model, "status": "FAIL", "error": res["error"]})
                continue

            sec = res["client_sec"]
            eval_res = evaluate_pose17(res["content"])
            pts = eval_res.get("pts_count", 0)
            valid_keys = eval_res.get("valid_keys", False)
            valid_coords = eval_res.get("valid_coords", False)
            usage = res.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)

            print(f"    Latency: {sec:.2f}s | Points: {pts}/17 | Keys OK: {valid_keys} | Coords OK: {valid_coords} | Tokens: P={p_tok} C={c_tok}")
            results.append({
                "task": "pose17",
                "model": model,
                "status": "PASS" if (valid_keys and valid_coords) else "PARTIAL",
                "latency_sec": sec,
                "pts_count": pts,
                "valid_keys": valid_keys,
                "valid_coords": valid_coords,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "eval": eval_res,
            })
        except Exception as e:
            print(f"    EXCEPTION: {e}")
            results.append({"task": "pose17", "model": model, "status": "ERROR", "error": str(e)})

    # 2. Benchmark VF50
    print("\n[Phase 2] Benchmarking VF50 on Driver Image...")
    for model in CANDIDATE_MODELS:
        print(f"--> Testing VF50 on {model}...")
        try:
            res = execute_request(
                model=model,
                system_prompt=SYSTEM_PROMPT_VF50,
                user_prompt="Detect the driver's face and all 50 facial landmarks (IDs 0 to 49) according to VinFast VF-50 schema.",
                img_b64=b64_driver,
                timeout=30.0,
            )
            if "error" in res:
                print(f"    FAILED: {res['error']}")
                results.append({"task": "vf50", "model": model, "status": "FAIL", "error": res["error"]})
                continue

            sec = res["client_sec"]
            eval_res = evaluate_vf50(res["content"])
            pts = eval_res.get("pts_count", 0)
            valid_keys = eval_res.get("valid_keys", False)
            valid_coords = eval_res.get("valid_coords", False)
            usage = res.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)

            print(f"    Latency: {sec:.2f}s | Landmarks: {pts}/50 | Keys OK: {valid_keys} | Coords OK: {valid_coords} | Tokens: P={p_tok} C={c_tok}")
            results.append({
                "task": "vf50",
                "model": model,
                "status": "PASS" if (valid_keys and valid_coords) else "PARTIAL",
                "latency_sec": sec,
                "pts_count": pts,
                "valid_keys": valid_keys,
                "valid_coords": valid_coords,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "eval": eval_res,
            })
        except Exception as e:
            print(f"    EXCEPTION: {e}")
            results.append({"task": "vf50", "model": model, "status": "ERROR", "error": str(e)})

    print("\n=================================================================")
    print(" BENCHMARK SUMMARY")
    print("=================================================================")
    for r in results:
        t = r["task"]
        m = r["model"]
        st = r["status"]
        sec = r.get("latency_sec", 0.0)
        pts = r.get("pts_count", 0)
        c_tok = r.get("completion_tokens", 0)
        print(f"[{t.upper():6s}] {m:28s} | Status: {st:7s} | Latency: {sec:5.2f}s | Points: {pts:2d} | Tokens: {c_tok:4d}")


if __name__ == "__main__":
    main()

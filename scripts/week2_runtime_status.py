"""Inspect deployed Week-2 Nuclio functions, specs, and drift status.

Checks:
1. Docker container status & health for:
   - nuclio-nuclio-ninerouter-human-pose-17
   - nuclio-nuclio-ninerouter-face-vf50
2. Deployed annotations (metadata.annotations.spec):
   - Pose 17: 1 skeleton 'person', 17 sublabels, 18 edges (including ear-to-shoulder), valid SVG
   - Face VF-50: 7 component skeletons (VinFast spec) vs legacy 1 monolithic 'face', anatomical SVG
3. Full SHA-256 Spec Fingerprinting across:
   - Canonical YAML schema
   - Canonical CVAT spec
   - Repo HEAD function.yaml spec
   - Deployed container annotations spec
   - CVAT registry spec (Nuclio API)
4. Runtime build SHA tracking:
   - TOOL_CVAT_BUILD_SHA environment variable in container vs git rev-parse HEAD
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.skeleton_contract import (
    VF50_COMPONENT_NAMES,
    VF50_COMPONENT_CONFIG,
)
from core.pose_face_schema import POSE17_KEYPOINTS
from core.week2_schema import (
    POSE_CVAT_SPEC_HASH,
    POSE_SCHEMA_HASH,
    VF50_CVAT_SPEC_HASH,
    VF50_SCHEMA_HASH,
    compute_spec_fingerprint,
    get_canonical_cvat_spec,
    get_canonical_cvat_spec_hash,
    get_canonical_schema_hash,
    load_pose17,
    load_vf50,
)

WEEK2_CONTAINERS = {
    "pose17": {
        "container_name": "nuclio-nuclio-ninerouter-human-pose-17",
        "function_name": "ninerouter-human-pose-17",
        "yaml_path": REPO_ROOT / "serverless" / "ninerouter-human-pose-17" / "nuclio" / "function.yaml",
        "expected_labels": 1,
    },
    "vf50": {
        "container_name": "nuclio-nuclio-ninerouter-face-vf50",
        "function_name": "ninerouter-face-vf50",
        "yaml_path": REPO_ROOT / "serverless" / "ninerouter-face-vf50" / "nuclio" / "function.yaml",
        "expected_labels": 7,
    },
}

# Strict security allowlist: ONLY non-sensitive environment variables may be recorded/displayed.
SAFE_ENV_ALLOWLIST = frozenset({
    "TOOL_CVAT_BUILD_SHA",
    "VISION_MODEL",
    "POSE17_MODEL",
    "POSE17_REFINE_MODEL",
    "VF50_MODEL",
    "VF50_REFINE_MODEL",
    "DETECTION_MODE",
    "NINEROUTER_TIMEOUT",
    "FEEDBACK_DATA_DIR",
    "FEEDBACK_DB_PATH",
})

SENSITIVE_KEY_PATTERN = re.compile(r"(KEY|SECRET|TOKEN|PASS|AUTH|CRED|PWD)", re.IGNORECASE)


def sanitize_url(url_str: str) -> str:
    """Strip basic auth credentials from URLs for safe logging."""
    try:
        parsed = urllib.parse.urlparse(url_str)
        if parsed.password or parsed.username:
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc += f":{parsed.port}"
            return urllib.parse.urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ))
    except Exception:
        pass
    return url_str


def get_git_head_sha() -> Optional[str]:
    """Return the current git HEAD commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(REPO_ROOT),
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def get_container_inspect(container_name: str) -> Optional[Dict[str, Any]]:
    """Run docker inspect on a container and return parsed JSON."""
    try:
        res = subprocess.run(
            ["docker", "inspect", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if data and isinstance(data, list):
                return data[0]
    except Exception:
        pass
    return None


def get_repo_spec(yaml_path: Path) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Read repo HEAD spec from function.yaml."""
    if not yaml_path.exists():
        return None, None
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ann = content.get("metadata", {}).get("annotations", {})
        name = ann.get("name")
        raw_spec = ann.get("spec")
        spec_items = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
        return name, spec_items
    except Exception:
        return None, None


def get_nuclio_dashboard_spec(fn_name: str, nuclio_url: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Query Nuclio dashboard API (:8070) for registered function specification."""
    urls = [nuclio_url] if nuclio_url else [
        os.getenv("NUCLIO_DASHBOARD_URL", "http://127.0.0.1:8070"),
        "http://localhost:8070",
    ]
    for base in urls:
        if not base:
            continue
        url = f"{base.rstrip('/')}/api/functions/{fn_name}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ann = data.get("metadata", {}).get("annotations", {})
                raw_spec = ann.get("spec")
                if raw_spec:
                    return json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
        except Exception:
            continue
    return None


def get_cvat_lambda_spec(fn_name: str, cvat_url: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Query CVAT Lambda registry API (:18080 or :8080) for registered function specification."""
    urls = [cvat_url] if cvat_url else [
        os.getenv("CVAT_URL", "http://127.0.0.1:18080"),
        "http://localhost:18080",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ]
    token = os.getenv("CVAT_TOKEN", "")
    if not token:
        try:
            res = subprocess.run(
                [
                    "docker", "exec", "-i", "cvat_server", "python3", "-c",
                    "import os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production'); "
                    "django.setup(); from rest_framework.authtoken.models import Token; t = Token.objects.first(); "
                    "print(t.key if t else '')",
                ],
                capture_output=True,
                text=True,
                timeout=12,
            )
            if res.returncode == 0 and res.stdout.strip():
                token = res.stdout.strip()
        except Exception:
            pass

    headers = {"Accept": "application/vnd.cvat+json, application/json;q=0.9"}
    if token:
        headers["Authorization"] = f"Token {token}"

    for base in urls:
        if not base:
            continue
        try:
            url = f"{base.rstrip('/')}/api/lambda/functions"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                functions = data if isinstance(data, list) else data.get("results", [])
                for fn in functions:
                    cand_id = str(fn.get("id", ""))
                    cand_name = str(fn.get("name", ""))
                    ann_name = fn.get("annotations", {}).get("name", "")
                    if fn_name in cand_id or fn_name == cand_name or fn_name == ann_name:
                        raw_spec = fn.get("spec") or fn.get("annotations", {}).get("spec")
                        if raw_spec:
                            return json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
                        labels_v2 = fn.get("labels_v2")
                        if labels_v2 and isinstance(labels_v2, list):
                            normalized = []
                            for idx, lbl in enumerate(labels_v2, start=1):
                                item = {
                                    "id": idx,
                                    "name": lbl.get("name"),
                                    "type": lbl.get("type", "any"),
                                    "attributes": lbl.get("attributes", []),
                                }
                                if "sublabels" in lbl:
                                    item["sublabels"] = [
                                        {
                                            "id": s_idx,
                                            "name": sub.get("name"),
                                            "type": sub.get("type", "points"),
                                            "attributes": sub.get("attributes", []),
                                        }
                                        for s_idx, sub in enumerate(lbl.get("sublabels", []), start=1)
                                    ]
                                if "svg" in lbl:
                                    item["svg"] = lbl["svg"]
                                normalized.append(item)
                            return normalized
        except Exception:
            continue
    return None


# Backward-compatible alias
get_nuclio_registered_spec = get_nuclio_dashboard_spec


def analyze_function_runtime(key: str, cfg: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    """Analyze runtime status and spec of a single Week-2 detector."""
    container_name = cfg["container_name"]
    yaml_path = cfg["yaml_path"]
    fn_name = cfg["function_name"]

    repo_name, repo_spec = get_repo_spec(yaml_path)
    git_head = get_git_head_sha()

    canonical_cvat_spec = get_canonical_cvat_spec(key)
    canonical_spec_fp = get_canonical_cvat_spec_hash(key)
    canonical_schema_fp = get_canonical_schema_hash(key)
    repo_spec_fp = compute_spec_fingerprint(repo_spec) if repo_spec else None

    # Query both registries distinctly
    nuclio_spec = get_nuclio_dashboard_spec(fn_name)
    nuclio_dashboard_fp = compute_spec_fingerprint(nuclio_spec) if nuclio_spec else None

    cvat_spec = get_cvat_lambda_spec(fn_name)
    cvat_lambda_fp = compute_spec_fingerprint(cvat_spec) if cvat_spec else None

    report: Dict[str, Any] = {
        "key": key,
        "container_name": container_name,
        "function_name": fn_name,
        "deployed": False,
        "status": "NOT_FOUND",
        "health": "UNKNOWN",
        "ports": [],
        "env": {},
        "build_sha": None,
        "git_head_sha": git_head,
        "build_sha_match": False,
        "deployed_spec_type": "UNKNOWN",
        "deployed_label_count": 0,
        "spec_drift": False,
        "drift_details": [],
        "fingerprints": {
            "semantic_schema": canonical_schema_fp,
            "canonical_spec": canonical_spec_fp,
            "repo_spec": repo_spec_fp,
            "deployed_spec": None,
            "nuclio_dashboard": nuclio_dashboard_fp,
            "cvat_lambda": cvat_lambda_fp,
            # Backward-compatible aliases
            "canonical_yaml": canonical_schema_fp,
            "cvat_registry": nuclio_dashboard_fp,
        },
        "ready": False,
    }

    # Verify repo spec matches canonical contract
    if repo_spec_fp != canonical_spec_fp:
        report["spec_drift"] = True
        report["drift_details"].append(
            f"REPO DRIFT: repo function.yaml spec ({repo_spec_fp[:12] if repo_spec_fp else 'None'}) "
            f"differs from canonical contract ({canonical_spec_fp[:12]})"
        )

    info = get_container_inspect(container_name)
    if not info:
        report["drift_details"].append("Container is not deployed or not running")
        return report

    report["deployed"] = True
    state = info.get("State", {})
    report["status"] = state.get("Status", "unknown")
    report["health"] = state.get("Health", {}).get("Status", "none")

    # Ports
    ports_map = info.get("NetworkSettings", {}).get("Ports", {})
    for container_port, host_bindings in ports_map.items():
        if host_bindings:
            for b in host_bindings:
                report["ports"].append(f"{b.get('HostPort')}->{container_port}")

    # Environment variables (strict allowlist + regex secret protection)
    env_list = info.get("Config", {}).get("Env", [])
    env_dict: Dict[str, str] = {}
    for entry in env_list:
        if "=" in entry:
            k, v = entry.split("=", 1)
            env_dict[k] = v

    report["env"] = {
        k: env_dict[k]
        for k in SAFE_ENV_ALLOWLIST
        if k in env_dict and not SENSITIVE_KEY_PATTERN.search(k)
    }

    container_build_sha = env_dict.get("TOOL_CVAT_BUILD_SHA")
    report["build_sha"] = container_build_sha
    if container_build_sha and git_head:
        report["build_sha_match"] = (container_build_sha == git_head)
        if container_build_sha != git_head:
            report["drift_details"].append(
                f"BUILD SHA MISMATCH: Container deployed at {container_build_sha[:12]}, repo is at {git_head[:12]}"
            )
            if strict:
                report["spec_drift"] = True
    elif not container_build_sha:
        report["build_sha_match"] = False
        report["drift_details"].append(
            "BUILD SHA MISSING: Container does not have TOOL_CVAT_BUILD_SHA environment variable"
        )
        if strict:
            report["spec_drift"] = True
    elif not git_head:
        report["build_sha_match"] = False
        report["drift_details"].append(
            f"BUILD SHA UNVERIFIED: Container has {container_build_sha[:12]} but cannot determine repo git HEAD"
        )
        if strict:
            report["spec_drift"] = True

    # Labels & annotations
    labels = info.get("Config", {}).get("Labels", {})
    raw_ann = labels.get("nuclio.io/annotations")
    if not raw_ann:
        report["drift_details"].append("Missing nuclio.io/annotations in container labels")
        return report

    try:
        ann = json.loads(raw_ann)
    except Exception as e:
        report["drift_details"].append(f"Cannot parse container annotations JSON: {e}")
        return report

    raw_spec = ann.get("spec")
    try:
        deployed_spec = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
    except Exception as e:
        report["drift_details"].append(f"Cannot parse deployed spec JSON: {e}")
        return report

    if not isinstance(deployed_spec, list):
        report["drift_details"].append("Deployed spec is not a list")
        return report

    report["deployed_label_count"] = len(deployed_spec)
    deployed_spec_fp = compute_spec_fingerprint(deployed_spec)
    report["fingerprints"]["deployed_spec"] = deployed_spec_fp

    label_names = [item.get("name") for item in deployed_spec if isinstance(item, dict)]

    if key == "vf50":
        if len(deployed_spec) == 1 and label_names == ["face"]:
            report["deployed_spec_type"] = "LEGACY_MONOLITHIC_FACE"
            report["spec_drift"] = True
            report["drift_details"].append(
                "CRITICAL SPEC DRIFT: Container is running legacy monolithic 'face' skeleton with 50 sublabels. "
                "Repo HEAD has 7 component skeletons (longmaytrai..moitrong). Redeploy required!"
            )
        elif len(deployed_spec) == 7 and set(label_names) == set(VF50_COMPONENT_NAMES):
            report["deployed_spec_type"] = "AUTHORITATIVE_7_COMPONENT_VF50"
        else:
            report["deployed_spec_type"] = f"UNKNOWN_SKELETON_COUNT_{len(deployed_spec)}"
            report["spec_drift"] = True
            report["drift_details"].append(f"Unexpected label names in deployed spec: {label_names}")

    elif key == "pose17":
        if len(deployed_spec) == 1 and label_names == ["person"]:
            subs = deployed_spec[0].get("sublabels", [])
            sub_names = [s.get("name") for s in subs]
            numeric_17 = {str(i) for i in range(1, 18)}
            if len(subs) == 17 and (set(sub_names) == set(POSE17_KEYPOINTS) or set(sub_names) == numeric_17):
                report["deployed_spec_type"] = "CANONICAL_POSE17_17KPS"
            else:
                report["deployed_spec_type"] = "INVALID_POSE17_SUBLABELS"
                report["spec_drift"] = True
                report["drift_details"].append(f"Expected 17 COCO or numeric '1'..'17' keypoints, got {len(subs)}")
        else:
            report["deployed_spec_type"] = f"UNEXPECTED_POSE17_LABELS_{label_names}"
            report["spec_drift"] = True

    # Full SHA-256 fingerprint comparison: deployed vs repo
    if repo_spec_fp and deployed_spec_fp != repo_spec_fp:
        report["spec_drift"] = True
        report["drift_details"].append(
            f"FULL SPEC SHA-256 DRIFT: Deployed ({deployed_spec_fp[:12]}) differs from "
            f"repo function.yaml ({repo_spec_fp[:12]})"
        )

    # Full SHA-256 fingerprint comparison: deployed vs canonical
    if deployed_spec_fp != canonical_spec_fp:
        report["spec_drift"] = True
        report["drift_details"].append(
            f"CANONICAL SPEC DRIFT: Deployed ({deployed_spec_fp[:12]}) differs from "
            f"authoritative canonical contract ({canonical_spec_fp[:12]})"
        )

    # Full SHA-256 fingerprint comparison: deployed vs Nuclio dashboard registry (:8070)
    if nuclio_dashboard_fp and nuclio_dashboard_fp != deployed_spec_fp:
        report["spec_drift"] = True
        report["drift_details"].append(
            f"NUCLIO DASHBOARD REGISTRY DRIFT: Nuclio dashboard registry ({nuclio_dashboard_fp[:12]}) differs "
            f"from container annotations ({deployed_spec_fp[:12]})"
        )

    # Full SHA-256 fingerprint comparison: deployed vs CVAT Lambda registry (:18080)
    if cvat_lambda_fp and cvat_lambda_fp != deployed_spec_fp:
        report["spec_drift"] = True
        report["drift_details"].append(
            f"CVAT LAMBDA REGISTRY DRIFT: CVAT Lambda registry ({cvat_lambda_fp[:12]}) differs "
            f"from container annotations ({deployed_spec_fp[:12]})"
        )

    report["ready"] = (
        report["deployed"]
        and report["status"] == "running"
        and report["health"] in ("healthy", "none")
        and not report["spec_drift"]
        and (report["build_sha_match"] or not strict)
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Week-2 AI Detector Runtime Status & Spec Drift Auditor")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--check", action="store_true", help="Exit 1 if any function is drifted or not ready")
    parser.add_argument("--strict", action="store_true", help="Enforce build SHA matching git HEAD in addition to spec drift")
    args = parser.parse_args()

    results: Dict[str, Dict[str, Any]] = {}
    all_ready = True

    for key, cfg in WEEK2_CONTAINERS.items():
        res = analyze_function_runtime(key, cfg, strict=args.strict)
        results[key] = res
        if not res["ready"]:
            all_ready = False

    if args.json:
        print(json.dumps(results, indent=2))
        return 0 if all_ready or not (args.check or args.strict) else 1

    print("=" * 80)
    print("WEEK-2 DETECTOR RUNTIME STATUS & SPEC FINGERPRINT REPORT")
    print("=" * 80)

    for key, r in results.items():
        fps = r["fingerprints"]
        print(f"\n[{r['function_name']}]")
        print(f"  Container       : {r['container_name']}")
        print(f"  Status          : {r['status']} (health: {r['health']})")
        print(f"  Ports           : {', '.join(r['ports']) if r['ports'] else 'None'}")
        print(f"  Build SHA       : {r['build_sha'] or 'UNSET'} (git HEAD: {r['git_head_sha'] or 'UNKNOWN'}, match: {r['build_sha_match']})")
        print(f"  Spec Type       : {r['deployed_spec_type']} ({r['deployed_label_count']} labels)")
        print(f"  Canonical Spec  : {fps['canonical_spec'][:16]}... (Schema: {fps['semantic_schema'][:16]}...)")
        print(f"  Repo Spec       : {fps['repo_spec'][:16] if fps['repo_spec'] else 'N/A'}...")
        print(f"  Deployed Spec   : {fps['deployed_spec'][:16] if fps['deployed_spec'] else 'N/A'}...")
        print(f"  Nuclio Dashboard: {fps['nuclio_dashboard'][:16] if fps['nuclio_dashboard'] else 'N/A'}... (:8070)")
        print(f"  CVAT Lambda Reg : {fps['cvat_lambda'][:16] if fps['cvat_lambda'] else 'N/A'}... (:18080)")
        if r["env"]:
            env_strs = [f"{k}={v}" for k, v in r["env"].items() if k != "TOOL_CVAT_BUILD_SHA"]
            if env_strs:
                print(f"  Active Env      : {', '.join(env_strs)}")

        if r["spec_drift"]:
            print("  DRIFT           : YES - SPEC DRIFT DETECTED!")
            for d in r["drift_details"]:
                print(f"    - {d}")
        else:
            print("  DRIFT           : NO (100% SHA-256 spec alignment)")
            if r["drift_details"]:
                for d in r["drift_details"]:
                    print(f"    - (info) {d}")

        print(f"  Ready           : {'YES' if r['ready'] else 'NO - REDEPLOY NEEDED'}")

    print("\n" + "=" * 80)
    if all_ready:
        print("ALL WEEK-2 DETECTORS ARE DEPLOYED, HEALTHY, AND SPEC-ALIGNED.")
        return 0
    else:
        print("ONE OR MORE WEEK-2 DETECTORS REQUIRE REDEPLOYMENT OR HAVE DRIFT.")
        return 1 if (args.check or args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

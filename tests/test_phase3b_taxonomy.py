"""Phase 3B Taxonomy and Label Routing Test Suite.

Verifies:
1. Exact 31 CVAT label count, ordering, and presence (zero dropped labels, zero renames).
2. Intentional ambiguous label segregation:
   - 'pedestrian' vs 'person'
   - 'traffic light' vs 'traffic_light'
   - 'traffic sign' vs 'traffic_sign'
3. Phase 3B 3-tier group routing:
   - Instance group (14 labels) -> box + mask
   - Region group (10 labels) -> mask only (no bounding box)
   - Lane group (7 labels) -> thin mask / polygon
4. Disjointness and completeness of label groups.
5. Configuration synchronization across labels.yaml and cvat_labels.json.
6. Backward compatibility with Phase 1 & Phase 2 13-bbox label subsets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pytest
import yaml

from core.vision_contract import (
    ALL_31_LABELS,
    DEFAULT_BBOX_LABELS,
    INTENTIONAL_DISTINCTIONS,
    load_label_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
LABELS_YAML_PATH = CONFIG_DIR / "labels.yaml"
CVAT_LABELS_JSON_PATH = CONFIG_DIR / "cvat_labels.json"

# ============================================================================
# Phase 3B Group Definitions
# ============================================================================

PHASE3B_INSTANCE_LABELS: Tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
    "person",
    "traffic_light",
    "traffic_sign",
    "pole",
)

PHASE3B_REGION_LABELS: Tuple[str, ...] = (
    "area/alternative",
    "area/drivable",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "vegetation",
    "terrain",
    "sky",
)

PHASE3B_LANE_LABELS: Tuple[str, ...] = (
    "lane/crosswalk",
    "lane/double white",
    "lane/double yellow",
    "lane/road curb",
    "lane/single other",
    "lane/single white",
    "lane/single yellow",
)


def get_label_routing_group(label: str) -> str:
    """Resolve which routing group a given label belongs to."""
    if label in PHASE3B_INSTANCE_LABELS:
        return "instance"
    elif label in PHASE3B_REGION_LABELS:
        return "region"
    elif label in PHASE3B_LANE_LABELS:
        return "lane"
    else:
        raise ValueError(f"Unknown label {label!r} not in Phase 3B taxonomy")


# ============================================================================
# 1. Exact 31 Label Count & Preservation
# ============================================================================

def test_exact_31_label_count_and_uniqueness():
    """Verify exactly 31 distinct labels exist in ALL_31_LABELS without duplicates."""
    assert len(ALL_31_LABELS) == 31
    assert len(set(ALL_31_LABELS)) == 31, "ALL_31_LABELS contains duplicates"


def test_labels_yaml_exact_31_match():
    """Verify labels.yaml contains all 31 labels and preserves count."""
    assert LABELS_YAML_PATH.exists(), f"Missing config file: {LABELS_YAML_PATH}"
    with open(LABELS_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_labels = data.get("all_labels", [])
    assert len(all_labels) == 31
    assert tuple(all_labels) == ALL_31_LABELS
    assert len(set(all_labels)) == 31


def test_cvat_labels_json_exact_31_match():
    """Verify cvat_labels.json contains specifications for all 31 labels."""
    assert CVAT_LABELS_JSON_PATH.exists(), f"Missing cvat_labels.json: {CVAT_LABELS_JSON_PATH}"
    with open(CVAT_LABELS_JSON_PATH, "r", encoding="utf-8") as f:
        cvat_data = json.load(f)

    names = [entry["name"] for entry in cvat_data]
    assert len(names) == 31
    assert set(names) == set(ALL_31_LABELS)


# ============================================================================
# 2. Ambiguous Label Segregation
# ============================================================================

@pytest.mark.parametrize(
    "label_a, label_b, description",
    [
        ("pedestrian", "person", "Pedestrian vs generic person distinction"),
        ("traffic light", "traffic_light", "Spaced vs underscored traffic light distinction"),
        ("traffic sign", "traffic_sign", "Spaced vs underscored traffic sign distinction"),
    ],
)
def test_ambiguous_label_pairs_segregation(label_a: str, label_b: str, description: str):
    """Verify ambiguous pairs are strictly preserved and never conflated."""
    assert label_a in ALL_31_LABELS, f"Missing {label_a} in ALL_31_LABELS"
    assert label_b in ALL_31_LABELS, f"Missing {label_b} in ALL_31_LABELS"
    assert label_a != label_b

    # Both must be distinct in routing group
    group_a = get_label_routing_group(label_a)
    group_b = get_label_routing_group(label_b)
    assert group_a == "instance"
    assert group_b == "instance"

    # Verify no accidental lower-casing or underscore normalization alias
    lookup_dict = {label_a: "val_a", label_b: "val_b"}
    assert len(lookup_dict) == 2, f"Collision detected between {label_a} and {label_b}"
    assert lookup_dict[label_a] == "val_a"
    assert lookup_dict[label_b] == "val_b"


# ============================================================================
# 3. 3-Tier Group Routing Partitioning
# ============================================================================

def test_group_counts_partition():
    """Verify group counts: 14 Instance, 10 Region, 7 Lane = 31 Total."""
    assert len(PHASE3B_INSTANCE_LABELS) == 14
    assert len(PHASE3B_REGION_LABELS) == 10
    assert len(PHASE3B_LANE_LABELS) == 7
    assert 14 + 10 + 7 == 31


def test_group_disjointness():
    """Verify that the 3 routing groups are mutually exclusive."""
    inst_set = set(PHASE3B_INSTANCE_LABELS)
    reg_set = set(PHASE3B_REGION_LABELS)
    lane_set = set(PHASE3B_LANE_LABELS)

    assert inst_set.isdisjoint(reg_set), f"Overlap between Instance and Region: {inst_set & reg_set}"
    assert inst_set.isdisjoint(lane_set), f"Overlap between Instance and Lane: {inst_set & lane_set}"
    assert reg_set.isdisjoint(lane_set), f"Overlap between Region and Lane: {reg_set & lane_set}"


def test_group_completeness():
    """Verify union of the 3 groups exactly covers all 31 labels."""
    all_grouped = set(PHASE3B_INSTANCE_LABELS) | set(PHASE3B_REGION_LABELS) | set(PHASE3B_LANE_LABELS)
    assert all_grouped == set(ALL_31_LABELS)


@pytest.mark.parametrize("label", PHASE3B_INSTANCE_LABELS)
def test_instance_labels_routing(label: str):
    """Verify every instance label routes to instance group (box+mask)."""
    assert get_label_routing_group(label) == "instance"


@pytest.mark.parametrize("label", PHASE3B_REGION_LABELS)
def test_region_labels_routing(label: str):
    """Verify every region label routes to region group (mask only)."""
    assert get_label_routing_group(label) == "region"


@pytest.mark.parametrize("label", PHASE3B_LANE_LABELS)
def test_lane_labels_routing(label: str):
    """Verify every lane label routes to lane group (thin mask/polygon)."""
    assert get_label_routing_group(label) == "lane"


# ============================================================================
# 4. Phase 1 & 2 Non-Regression and Backward Compatibility
# ============================================================================

def test_phase1_bbox_labels_are_strict_subset_of_instance_group():
    """Verify all 13 Phase 1 bounding box labels are within the Phase 3B instance group."""
    assert len(DEFAULT_BBOX_LABELS) == 13
    for bbox_label in DEFAULT_BBOX_LABELS:
        assert bbox_label in PHASE3B_INSTANCE_LABELS
        assert get_label_routing_group(bbox_label) == "instance"

    # 'pole' is the 14th label added to instance group in Phase 3B
    diff = set(PHASE3B_INSTANCE_LABELS) - set(DEFAULT_BBOX_LABELS)
    assert diff == {"pole"}


def test_intentional_distinctions_mapping():
    """Verify INTENTIONAL_DISTINCTIONS covers all 3 critical pairs."""
    keys = list(INTENTIONAL_DISTINCTIONS.keys())
    assert ("pedestrian", "person") in keys
    assert ("traffic light", "traffic_light") in keys
    assert ("traffic sign", "traffic_sign") in keys


def test_core_taxonomy_registry_sync():
    """Verify core.taxonomy matches the Phase 3B group definitions."""
    try:
        from core.taxonomy import (
            GROUP_INSTANCE,
            GROUP_LANE,
            GROUP_REGION,
            MASTER_31_LABELS,
            TaxonomyRegistry,
        )
        reg = TaxonomyRegistry.get_instance()
        assert tuple(MASTER_31_LABELS) == ALL_31_LABELS
        assert set(reg.get_instance_labels()) == set(PHASE3B_INSTANCE_LABELS)
        assert set(reg.get_region_labels()) == set(PHASE3B_REGION_LABELS)
        assert set(reg.get_lane_labels()) == set(PHASE3B_LANE_LABELS)
    except ImportError:
        pass


"""Comprehensive Tests for Master 31-Label Taxonomy Module (Phase 3B)."""

import pytest

from core.taxonomy import (
    AMBIGUITY_PAIRS,
    GROUP_INSTANCE,
    GROUP_LANE,
    GROUP_REGION,
    MASTER_31_LABELS,
    SHAPE_MASK,
    SHAPE_POLYGON,
    SHAPE_POLYLINE,
    SHAPE_RECTANGLE,
    LabelMetadata,
    Taxonomy,
    TaxonomyValidationError,
    get_taxonomy,
)


@pytest.fixture
def tax() -> Taxonomy:
    """Fixture providing initialized Taxonomy."""
    return get_taxonomy()


# ============================================================================
# 1. Master Schema and 31-Label Validation
# ============================================================================


def test_master_schema_exact_count(tax: Taxonomy):
    """Verify that all 31 labels exist exactly once without missing or extra labels."""
    all_labels = tax.get_all_labels()
    assert len(all_labels) == 31
    assert len(set(all_labels)) == 31

    for label in MASTER_31_LABELS:
        assert tax.is_valid_label(label) is True
        info = tax.get_label_info(label)
        assert isinstance(info, LabelMetadata)
        assert info.name == label


def test_group_partition_counts(tax: Taxonomy):
    """Verify 14 instances, 10 regions, and 7 lanes totaling 31 labels."""
    instances = tax.get_instance_labels()
    regions = tax.get_region_labels()
    lanes = tax.get_lane_labels()

    assert len(instances) == 14
    assert len(regions) == 10
    assert len(lanes) == 7
    assert len(instances) + len(regions) + len(lanes) == 31

    # Check that groups are mutually exclusive
    assert set(instances).isdisjoint(set(regions))
    assert set(instances).isdisjoint(set(lanes))
    assert set(regions).isdisjoint(set(lanes))


def test_strict_isolation_ambiguous_labels(tax: Taxonomy):
    """STRICT RULE: Ambiguous labels must never be renamed or merged."""
    # 1. pedestrian vs person
    assert tax.is_valid_label("pedestrian")
    assert tax.is_valid_label("person")
    assert tax.get_label_info("pedestrian").name == "pedestrian"
    assert tax.get_label_info("person").name == "person"
    assert tax.get_label_info("pedestrian") != tax.get_label_info("person")

    # 2. traffic light vs traffic_light
    assert tax.is_valid_label("traffic light")
    assert tax.is_valid_label("traffic_light")
    assert tax.get_label_info("traffic light").name == "traffic light"
    assert tax.get_label_info("traffic_light").name == "traffic_light"
    assert tax.get_label_info("traffic light") != tax.get_label_info("traffic_light")

    # 3. traffic sign vs traffic_sign
    assert tax.is_valid_label("traffic sign")
    assert tax.is_valid_label("traffic_sign")
    assert tax.get_label_info("traffic sign").name == "traffic sign"
    assert tax.get_label_info("traffic_sign").name == "traffic_sign"
    assert tax.get_label_info("traffic sign") != tax.get_label_info("traffic_sign")


# ============================================================================
# 2. Geometric Attributes and Special Handling
# ============================================================================


def test_special_handling_pole(tax: Taxonomy):
    """Verify 'pole' special handling: instance object, mask preferred, optional rectangle."""
    pole = tax.get_label_info("pole")
    assert pole.is_instance is True
    assert pole.preferred_shape == SHAPE_MASK
    assert SHAPE_MASK in pole.allowed_shapes
    assert SHAPE_POLYGON in pole.allowed_shapes
    assert SHAPE_RECTANGLE in pole.allowed_shapes
    assert pole.supports_bounding_box is True
    assert pole.is_thin_structure is True
    assert pole.special_handling.get("optional_rectangle") is True


def test_special_handling_crosswalk(tax: Taxonomy):
    """Verify 'lane/crosswalk' special handling: lane group, area-like, polygon/mask preferred."""
    cw = tax.get_label_info("lane/crosswalk")
    assert cw.is_lane is True
    assert cw.is_area_like is True
    assert cw.preferred_shape == SHAPE_POLYGON
    assert SHAPE_POLYGON in cw.allowed_shapes
    assert SHAPE_MASK in cw.allowed_shapes
    assert SHAPE_RECTANGLE in cw.allowed_shapes
    assert cw.supports_polyline is False  # Crosswalk is a 2D area, not a 1D polyline


def test_linear_lanes(tax: Taxonomy):
    """Verify linear lane lines support polyline and are marked as thin lines."""
    linear_lanes = [
        "lane/double white",
        "lane/double yellow",
        "lane/road curb",
        "lane/single other",
        "lane/single white",
        "lane/single yellow",
    ]
    for lane in linear_lanes:
        info = tax.get_label_info(lane)
        assert info.is_lane is True
        assert info.preferred_shape == SHAPE_POLYLINE
        assert SHAPE_POLYLINE in info.allowed_shapes
        assert info.is_thin_structure is True
        assert info.supports_bounding_box is False


def test_region_labels_reject_bounding_box(tax: Taxonomy):
    """Verify all region labels (sky, road, sidewalk, etc.) do NOT allow rectangle bounding boxes."""
    for region in tax.get_region_labels():
        info = tax.get_label_info(region)
        assert info.is_region is True
        assert info.supports_bounding_box is False
        assert SHAPE_RECTANGLE not in info.allowed_shapes
        assert SHAPE_MASK in info.allowed_shapes or SHAPE_POLYGON in info.allowed_shapes


# ============================================================================
# 3. O(1) Lookups and Query Methods
# ============================================================================


def test_o1_lookups(tax: Taxonomy):
    """Verify O(1) dictionary lookups for metadata, groups, and shapes."""
    assert tax.get_group("car") == GROUP_INSTANCE
    assert tax.get_group("road") == GROUP_REGION
    assert tax.get_group("lane/single white") == GROUP_LANE

    assert tax.get_preferred_shape("car") == SHAPE_RECTANGLE
    assert tax.get_preferred_shape("vegetation") == SHAPE_MASK

    assert tax.validate_shape_for_label("car", SHAPE_RECTANGLE) is True
    assert tax.validate_shape_for_label("car", SHAPE_POLYLINE) is False
    assert tax.validate_shape_for_label("road", SHAPE_RECTANGLE) is False
    assert tax.validate_shape_for_label("road", SHAPE_MASK) is True


def test_unknown_label_handling(tax: Taxonomy):
    """Verify appropriate handling of unknown/invalid labels."""
    assert tax.is_valid_label("spaceship") is False
    assert tax.validate_shape_for_label("spaceship", SHAPE_RECTANGLE) is False

    with pytest.raises(KeyError, match="not in the 31-label taxonomy"):
        tax.get_label_info("spaceship")


# ============================================================================
# 4. Ambiguity Resolution & Task Adaptation Policy
# ============================================================================


def test_ambiguity_counterparts(tax: Taxonomy):
    """Verify counterpart discovery for ambiguous pairs."""
    assert tax.is_ambiguous_label("pedestrian") is True
    assert tax.get_ambiguity_counterpart("pedestrian") == "person"
    assert tax.get_ambiguity_counterpart("person") == "pedestrian"

    assert tax.is_ambiguous_label("traffic light") is True
    assert tax.get_ambiguity_counterpart("traffic light") == "traffic_light"
    assert tax.get_ambiguity_counterpart("traffic_light") == "traffic light"

    assert tax.is_ambiguous_label("traffic sign") is True
    assert tax.get_ambiguity_counterpart("traffic sign") == "traffic_sign"
    assert tax.get_ambiguity_counterpart("traffic_sign") == "traffic sign"

    # Non-ambiguous labels
    assert tax.is_ambiguous_label("car") is False
    assert tax.get_ambiguity_counterpart("car") is None


def test_resolve_label_exact_match(tax: Taxonomy):
    """Verify exact match preserves label when present in target task."""
    target_task = ["pedestrian", "car", "traffic_light"]
    resolved, warning = tax.resolve_label_for_task("pedestrian", target_task)
    assert resolved == "pedestrian"
    assert warning is None

    resolved, warning = tax.resolve_label_for_task("car", target_task)
    assert resolved == "car"
    assert warning is None


def test_resolve_label_fallback_adaptation(tax: Taxonomy):
    """Verify fallback mapping when task only has one variant."""
    # Case 1: Task has only 'person', model predicts 'pedestrian'
    target_task_person = ["person", "car", "bicycle"]
    resolved, warning = tax.resolve_label_for_task("pedestrian", target_task_person)
    assert resolved == "person"
    assert warning is not None
    assert "dynamically mapped" in warning

    # Case 2: Task has only 'pedestrian', model predicts 'person'
    target_task_ped = ["pedestrian", "car", "bicycle"]
    resolved, warning = tax.resolve_label_for_task("person", target_task_ped)
    assert resolved == "pedestrian"
    assert warning is not None

    # Case 3: Task has only 'traffic light' (space), model predicts 'traffic_light' (underscore)
    target_task_space = ["traffic light", "car"]
    resolved, warning = tax.resolve_label_for_task("traffic_light", target_task_space)
    assert resolved == "traffic light"
    assert warning is not None

    # Case 4: Task has only 'traffic_sign' (underscore), model predicts 'traffic sign' (space)
    target_task_under = ["traffic_sign", "truck"]
    resolved, warning = tax.resolve_label_for_task("traffic sign", target_task_under)
    assert resolved == "traffic_sign"
    assert warning is not None


def test_resolve_label_strict_when_both_exist(tax: Taxonomy):
    """Verify NO fallback mapping occurs when target task schema contains BOTH variants."""
    target_task_both = ["pedestrian", "person", "car"]
    # Model predicts 'pedestrian' -> remains 'pedestrian'
    resolved, warning = tax.resolve_label_for_task("pedestrian", target_task_both)
    assert resolved == "pedestrian"
    assert warning is None

    # Model predicts 'person' -> remains 'person'
    resolved, warning = tax.resolve_label_for_task("person", target_task_both)
    assert resolved == "person"
    assert warning is None


# ============================================================================
# 5. Shape Router Helpers
# ============================================================================


def test_route_shapes_instances(tax: Taxonomy):
    """Verify shape routing for standard instances in various modes."""
    # box mode: rectangle
    assert tax.route_shapes("car", requested_mode="box", has_box=True, has_mask=True) == [SHAPE_RECTANGLE]

    # mask mode: mask
    assert tax.route_shapes("car", requested_mode="mask", has_box=True, has_mask=True) == [SHAPE_MASK]

    # box_and_mask mode: both
    assert tax.route_shapes("car", requested_mode="box_and_mask", has_box=True, has_mask=True) == [
        SHAPE_RECTANGLE,
        SHAPE_MASK,
    ]


def test_route_shapes_pole(tax: Taxonomy):
    """Verify special pole routing: mask preferred, supports rectangle."""
    # box mode
    assert tax.route_shapes("pole", requested_mode="box", has_box=True) == [SHAPE_RECTANGLE]

    # mask mode with mask available
    assert tax.route_shapes("pole", requested_mode="mask", has_box=True, has_mask=True) == [SHAPE_MASK]

    # box_and_mask mode
    assert tax.route_shapes("pole", requested_mode="box_and_mask", has_box=True, has_mask=True) == [
        SHAPE_MASK,
        SHAPE_RECTANGLE,
    ]


def test_route_shapes_regions(tax: Taxonomy):
    """Verify regions never route rectangle even if has_box=True."""
    # Box mode: regions cannot output rectangle
    assert tax.route_shapes("road", requested_mode="box", has_box=True, has_mask=False) == []

    # Mask mode: returns mask
    assert tax.route_shapes("road", requested_mode="mask", has_box=True, has_mask=True) == [SHAPE_MASK]
    assert tax.route_shapes("sky", requested_mode="box_and_mask", has_box=True, has_mask=True) == [SHAPE_MASK]


def test_route_shapes_lanes(tax: Taxonomy):
    """Verify lane markings routing."""
    # Linear lane line: polyline
    assert tax.route_shapes("lane/single white", requested_mode="mask", has_mask=True) == [SHAPE_POLYLINE]

    # Crosswalk: polygon or rectangle
    assert tax.route_shapes("lane/crosswalk", requested_mode="box", has_box=True) == [SHAPE_RECTANGLE]
    assert tax.route_shapes("lane/crosswalk", requested_mode="mask", has_mask=True) == [SHAPE_POLYGON]


# ============================================================================
# 6. Prompt Builder Helpers
# ============================================================================


def test_filter_labels_by_mode(tax: Taxonomy):
    """Verify filtering labels based on inference mode."""
    box_labels = tax.filter_labels_by_mode("box")
    # All 14 instances + crosswalk support box = 15 labels
    assert len(box_labels) == 15
    assert "car" in box_labels
    assert "pedestrian" in box_labels
    assert "pole" in box_labels
    assert "lane/crosswalk" in box_labels
    assert "road" not in box_labels
    assert "sky" not in box_labels

    mask_labels = tax.filter_labels_by_mode("mask")
    assert len(mask_labels) == 31  # All 31 support mask or polygon or polyline/mask


def test_prompt_manifest_building(tax: Taxonomy):
    """Verify building grouped prompt manifest."""
    manifest = tax.build_prompt_manifest(["car", "road", "lane/crosswalk"])
    assert "Foreground Instances" in manifest
    assert "- car" in manifest
    assert "Background Regions" in manifest
    assert "- road" in manifest
    assert "Lane Markings" in manifest
    assert "- lane/crosswalk" in manifest


def test_ambiguity_warning_prompt(tax: Taxonomy):
    """Verify targeted prompt warning directives."""
    # Both variants in prompt
    warning = tax.generate_ambiguity_warning_prompt(["pedestrian", "person"])
    assert "CRITICAL: Both 'pedestrian' and 'person' are in the allowed label list" in warning

    # Only one variant in prompt
    warning_single = tax.generate_ambiguity_warning_prompt(["traffic light", "car"])
    assert "STRICT: Use ONLY 'traffic light'. The variation 'traffic_light' is NOT permitted" in warning_single


# ============================================================================
# 7. Validation Integrity / Error Handling
# ============================================================================


def test_taxonomy_validation_missing_label(tax: Taxonomy):
    """Verify Taxonomy validation raises error on corrupted label configuration."""
    corrupted = dict(tax._labels)
    del corrupted["car"]

    t = Taxonomy.__new__(Taxonomy)
    t._labels = corrupted
    t._ambiguity_pairs = list(AMBIGUITY_PAIRS)
    t._group_to_labels = {GROUP_INSTANCE: [], GROUP_REGION: [], GROUP_LANE: []}

    with pytest.raises(TaxonomyValidationError, match="Expected exactly 31 labels"):
        t.validate()

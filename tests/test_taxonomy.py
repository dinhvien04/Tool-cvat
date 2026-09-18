"""Comprehensive Tests for Master 31-Label Taxonomy Module (Phase 3B)."""

import pytest

from core.taxonomy import (
    AMBIGUITY_PAIRS,
    BOX_MASK_LABELS,
    GROUP_INSTANCE,
    GROUP_LANE,
    GROUP_REGION,
    MASTER_31_LABELS,
    POLICY_BOX_MASK,
    POLICY_POLYGON_MASK,
    POLICY_POLYLINE,
    POLYGON_MASK_LABELS,
    POLYLINE_LABELS,
    SHAPE_MASK,
    SHAPE_POLYGON,
    SHAPE_POLYLINE,
    SHAPE_RECTANGLE,
    VALID_POLICIES,
    LabelMetadata,
    Taxonomy,
    TaxonomyValidationError,
    get_policy,
    get_taxonomy,
    is_box_mask,
    is_polygon_mask,
    is_polyline,
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
    """Verify 'pole' special handling: instance object, mask preferred, optional rectangle.

    Polygon is strictly removed from instance label permissions.
    """
    pole = tax.get_label_info("pole")
    assert pole.is_instance is True
    assert pole.is_box_mask is True
    assert pole.policy == POLICY_BOX_MASK
    assert pole.preferred_shape == SHAPE_MASK
    assert SHAPE_MASK in pole.allowed_shapes
    assert SHAPE_POLYGON not in pole.allowed_shapes  # Polygon removed from instance permissions!
    assert pole.supports_polygon is False
    assert SHAPE_RECTANGLE in pole.allowed_shapes
    assert pole.supports_bounding_box is True
    assert pole.is_thin_structure is True
    assert pole.special_handling.get("optional_rectangle") is True


def test_special_handling_crosswalk(tax: Taxonomy):
    """Verify 'lane/crosswalk' Policy C handling: lane group, strictly polyline."""
    cw = tax.get_label_info("lane/crosswalk")
    assert cw.is_lane is True
    assert cw.is_polyline is True
    assert cw.policy == POLICY_POLYLINE
    assert cw.preferred_shape == SHAPE_POLYLINE
    assert cw.allowed_shapes == (SHAPE_POLYLINE,)
    assert cw.supports_polyline is True
    assert cw.supports_bounding_box is False
    assert cw.supports_polygon is False
    assert cw.supports_mask is False


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

    # Crosswalk (Policy C): strictly polyline, never rectangle or polygon
    assert tax.route_shapes("lane/crosswalk", requested_mode="box", has_box=True) == []
    assert tax.route_shapes("lane/crosswalk", requested_mode="mask", has_mask=True) == [SHAPE_POLYLINE]


# ============================================================================
# 6. Prompt Builder Helpers
# ============================================================================


def test_filter_labels_by_mode(tax: Taxonomy):
    """Verify filtering labels based on inference mode."""
    box_labels = tax.filter_labels_by_mode("box")
    # Exactly the 14 Policy A instances support box (crosswalk is Policy C polyline)
    assert len(box_labels) == 14
    assert "car" in box_labels
    assert "pedestrian" in box_labels
    assert "pole" in box_labels
    assert "lane/crosswalk" not in box_labels
    assert "road" not in box_labels
    assert "sky" not in box_labels

    mask_labels = tax.filter_labels_by_mode("mask")
    # 14 instances + 10 regions support mask = 24 labels
    assert len(mask_labels) == 24
    assert "car" in mask_labels
    assert "road" in mask_labels
    assert "lane/crosswalk" not in mask_labels

    polyline_labels = tax.filter_labels_by_mode("polyline")
    # Exactly 7 lane markings support polyline
    assert len(polyline_labels) == 7
    assert "lane/crosswalk" in polyline_labels
    assert "lane/single white" in polyline_labels

    all_labels = tax.filter_labels_by_mode("box_and_mask")
    # All 31 labels covered across policies
    assert len(all_labels) == 31


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
    t._policy_to_labels = {POLICY_BOX_MASK: [], POLICY_POLYGON_MASK: [], POLICY_POLYLINE: []}

    with pytest.raises(TaxonomyValidationError, match="Expected exactly 31 labels"):
        t.validate()


# ============================================================================
# 8. Strict 3-Policy Taxonomy and Shape Enforcement Tests
# ============================================================================


def test_exact_three_policies_partition(tax: Taxonomy):
    """Verify exact 31 labels partitioned into: 14 box_mask, 10 polygon_mask, 7 polyline."""
    box_masks = tax.get_box_mask_labels()
    polygon_masks = tax.get_polygon_mask_labels()
    polylines = tax.get_polyline_labels()

    assert len(box_masks) == 14
    assert len(polygon_masks) == 10
    assert len(polylines) == 7
    assert len(box_masks) + len(polygon_masks) + len(polylines) == 31

    # Exact membership matches
    assert set(box_masks) == set(BOX_MASK_LABELS)
    assert set(polygon_masks) == set(POLYGON_MASK_LABELS)
    assert set(polylines) == set(POLYLINE_LABELS)


def test_every_label_belongs_to_exactly_one_policy(tax: Taxonomy):
    """Ensure every label belongs to exactly one policy (mutually exclusive and exhaustive)."""
    box_masks = set(tax.get_box_mask_labels())
    polygon_masks = set(tax.get_polygon_mask_labels())
    polylines = set(tax.get_polyline_labels())

    # Pairwise disjoint
    assert box_masks.isdisjoint(polygon_masks)
    assert box_masks.isdisjoint(polylines)
    assert polygon_masks.isdisjoint(polylines)

    # Exhaustive union
    assert (box_masks | polygon_masks | polylines) == set(MASTER_31_LABELS)

    # Each individual label checks out
    for label in MASTER_31_LABELS:
        policy = tax.get_policy(label)
        assert policy in VALID_POLICIES
        if policy == POLICY_BOX_MASK:
            assert tax.is_box_mask(label) is True
            assert tax.is_polygon_mask(label) is False
            assert tax.is_polyline(label) is False
        elif policy == POLICY_POLYGON_MASK:
            assert tax.is_box_mask(label) is False
            assert tax.is_polygon_mask(label) is True
            assert tax.is_polyline(label) is False
        elif policy == POLICY_POLYLINE:
            assert tax.is_box_mask(label) is False
            assert tax.is_polygon_mask(label) is False
            assert tax.is_polyline(label) is True


def test_instance_labels_exclude_polygon_completely(tax: Taxonomy):
    """STRICT REQUIREMENT: All 14 instance/box_mask labels must NOT permit polygon shape."""
    for label in tax.get_box_mask_labels():
        meta = tax.get_label_info(label)
        assert SHAPE_POLYGON not in meta.allowed_shapes, f"Label {label} incorrectly allows polygon!"
        assert meta.supports_polygon is False, f"Label {label} supports_polygon is True!"
        assert tax.validate_shape_for_label(label, SHAPE_POLYGON) is False


def test_policy_lookups_and_helpers(tax: Taxonomy):
    """Verify O(1) policy helper methods: get_policy, is_box_mask, is_polygon_mask, is_polyline."""
    assert tax.get_policy("car") == POLICY_BOX_MASK
    assert tax.is_box_mask("car") is True
    assert tax.is_polygon_mask("car") is False
    assert tax.is_polyline("car") is False

    assert tax.get_policy("road") == POLICY_POLYGON_MASK
    assert tax.is_box_mask("road") is False
    assert tax.is_polygon_mask("road") is True
    assert tax.is_polyline("road") is False

    assert tax.get_policy("lane/single white") == POLICY_POLYLINE
    assert tax.is_box_mask("lane/single white") is False
    assert tax.is_polygon_mask("lane/single white") is False
    assert tax.is_polyline("lane/single white") is True

    # lane/crosswalk belongs to POLICY_POLYLINE
    assert tax.get_policy("lane/crosswalk") == POLICY_POLYLINE
    assert tax.is_polyline("lane/crosswalk") is True

    # Unknown label
    with pytest.raises(KeyError):
        tax.get_policy("nonexistent")
    assert tax.is_box_mask("nonexistent") is False
    assert tax.is_polygon_mask("nonexistent") is False
    assert tax.is_polyline("nonexistent") is False


def test_module_level_helpers():
    """Verify module-level helper functions delegating to global taxonomy."""
    assert get_policy("car") == POLICY_BOX_MASK
    assert is_box_mask("car") is True
    assert is_polygon_mask("car") is False
    assert is_polyline("car") is False

    assert get_policy("vegetation") == POLICY_POLYGON_MASK
    assert is_polygon_mask("vegetation") is True

    assert get_policy("lane/road curb") == POLICY_POLYLINE
    assert is_polyline("lane/road curb") is True


def test_taxonomy_validation_catches_polygon_in_box_mask(tax: Taxonomy):
    """Verify validation raises error if an instance/box_mask label contains polygon."""
    corrupted = dict(tax._labels)
    old_car = corrupted["car"]
    corrupted["car"] = LabelMetadata(
        name=old_car.name,
        group=old_car.group,
        allowed_shapes=(SHAPE_RECTANGLE, SHAPE_MASK, SHAPE_POLYGON),
        preferred_shape=old_car.preferred_shape,
        policy=old_car.policy,
    )

    t = Taxonomy.__new__(Taxonomy)
    t._labels = corrupted
    t._ambiguity_pairs = list(AMBIGUITY_PAIRS)
    t._group_to_labels = tax._group_to_labels
    t._policy_to_labels = tax._policy_to_labels

    with pytest.raises(TaxonomyValidationError, match="must not permit polygon shape"):
        t.validate()


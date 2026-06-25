"""Unit tests for narration.layout.block_matcher (geometry + assignment)."""

import pytest

from narration.layout.block_matcher import (
    _bbox_area,
    _bbox_intersection,
    compute_iou,
    compute_overlap_ratio,
    match_regions_to_blocks,
    regions_to_pdf_coords,
)
from narration.layout.models import LayoutLabel


class TestGeometry:
    def test_identical_boxes_iou_is_one(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_is_zero(self):
        assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap_iou(self):
        # Two 10x10 boxes overlapping in a 5x10 strip.
        a = (0, 0, 10, 10)
        b = (5, 0, 15, 10)
        # inter = 50, union = 100 + 100 - 50 = 150
        assert compute_iou(a, b) == pytest.approx(50 / 150)

    def test_intersection_area(self):
        assert _bbox_intersection((0, 0, 10, 10), (5, 5, 15, 15)) == 25.0

    def test_intersection_zero_when_touching_edge(self):
        # Shared edge has zero area.
        assert _bbox_intersection((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_bbox_area(self):
        assert _bbox_area((0, 0, 4, 3)) == 12.0

    def test_bbox_area_never_negative(self):
        # Degenerate / inverted box clamps to zero, not negative.
        assert _bbox_area((10, 10, 0, 0)) == 0.0

    def test_overlap_ratio_full_containment(self):
        block = (2, 2, 4, 4)  # area 4, fully inside region
        region = (0, 0, 10, 10)
        assert compute_overlap_ratio(block, region) == pytest.approx(1.0)

    def test_overlap_ratio_partial(self):
        block = (0, 0, 10, 10)  # area 100
        region = (5, 0, 15, 10)  # covers 50
        assert compute_overlap_ratio(block, region) == pytest.approx(0.5)

    def test_overlap_ratio_zero_area_block(self):
        assert compute_overlap_ratio((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestCoordConversion:
    def test_regions_to_pdf_coords_divides_by_scale(self, make_region):
        regions = [make_region(LayoutLabel.TEXT, (30.0, 60.0, 90.0, 120.0))]
        converted = regions_to_pdf_coords(regions, scale=3.0)
        assert converted[0].bbox == (10.0, 20.0, 30.0, 40.0)
        # Label and confidence are preserved.
        assert converted[0].label == LayoutLabel.TEXT
        assert converted[0].confidence == regions[0].confidence


class TestMatchRegionsToBlocks:
    def test_block_matched_by_iou(self, make_block, make_region):
        # Region in pixel space; scale=2 -> pdf bbox (0,0,50,50) overlapping block.
        block = make_block("hello", bbox=(0, 0, 50, 50))
        region = make_region(LayoutLabel.TITLE, (0, 0, 100, 100), confidence=0.8)
        result = match_regions_to_blocks([region], [block], scale=2.0)
        assert len(result) == 1
        assert result[0].label == LayoutLabel.TITLE
        assert result[0].confidence == pytest.approx(0.8)

    def test_unmatched_block_is_unknown(self, make_block, make_region):
        block = make_block("orphan", bbox=(0, 0, 10, 10))
        # Region far away in pdf space (scale 1) -> no overlap.
        region = make_region(LayoutLabel.TEXT, (500, 500, 600, 600))
        result = match_regions_to_blocks([region], [block], scale=1.0)
        assert result[0].label == LayoutLabel.UNKNOWN
        assert result[0].confidence == 0.0

    def test_overlap_pass_applies_confidence_penalty(self, make_block, make_region):
        # Block almost fully covered by region but with low IoU because the
        # region is much larger -> falls through to the overlap pass (x0.9).
        block = make_block("body", bbox=(0, 0, 10, 10))
        region = make_region(LayoutLabel.TEXT, (0, 0, 1000, 10), confidence=1.0)
        # IoU here = 100 / (100 + 10000 - 100) = 0.01 < 0.1 threshold,
        # overlap ratio = 1.0 >= 0.5 threshold.
        result = match_regions_to_blocks([region], [block], scale=1.0)
        assert result[0].label == LayoutLabel.TEXT
        assert result[0].confidence == pytest.approx(0.9)

    def test_output_order_matches_input(self, make_block, make_region):
        blocks = [make_block("a", bbox=(0, 0, 5, 5)),
                  make_block("b", bbox=(100, 100, 105, 105))]
        result = match_regions_to_blocks([], blocks, scale=1.0)
        assert len(result) == 2
        assert result[0].text == "a"
        assert result[1].text == "b"

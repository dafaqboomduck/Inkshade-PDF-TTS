"""Unit tests for narration.layout.models data structures."""

import pytest

from narration.layout.models import (
    DOCLAYNET_INDEX_TO_LABEL,
    ClassifiedBlock,
    LayoutLabel,
    LayoutRegion,
)


class TestDocLayNetMapping:
    def test_contains_all_eleven_positive_classes(self):
        # DocLayNet has 11 classes (indices 0-10); UNKNOWN (-1) is excluded.
        assert len(DOCLAYNET_INDEX_TO_LABEL) == 11
        assert -1 not in DOCLAYNET_INDEX_TO_LABEL

    def test_reverse_lookup_round_trips(self):
        for label in LayoutLabel:
            if label.value >= 0:
                assert DOCLAYNET_INDEX_TO_LABEL[label.value] is label

    def test_known_index(self):
        assert DOCLAYNET_INDEX_TO_LABEL[10] is LayoutLabel.TITLE


class TestLayoutRegion:
    def test_area(self, make_region):
        r = make_region(LayoutLabel.TEXT, (0, 0, 4, 5))
        assert r.area == 20.0

    def test_area_clamps_inverted_box(self, make_region):
        r = make_region(LayoutLabel.TEXT, (10, 10, 0, 0))
        assert r.area == 0.0

    def test_center(self, make_region):
        r = make_region(LayoutLabel.TEXT, (0, 0, 10, 20))
        assert r.center == (5.0, 10.0)

    def test_to_pdf_coords(self, make_region):
        r = make_region(LayoutLabel.PICTURE, (30, 60, 90, 120), confidence=0.7)
        pdf = r.to_pdf_coords(scale=3.0)
        assert pdf.bbox == (10.0, 20.0, 30.0, 40.0)
        assert pdf.label is LayoutLabel.PICTURE
        assert pdf.confidence == pytest.approx(0.7)

    def test_repr_includes_label_name(self, make_region):
        r = make_region(LayoutLabel.SECTION_HEADER, (1, 2, 3, 4), confidence=0.5)
        assert "SECTION_HEADER" in repr(r)


class TestClassifiedBlock:
    def test_text_and_bbox_proxy_to_block(self, make_block):
        block = make_block("proxy text", bbox=(1, 2, 3, 4))
        cb = ClassifiedBlock(block=block, label=LayoutLabel.TEXT, confidence=0.5)
        assert cb.text == "proxy text"
        assert cb.bbox == (1, 2, 3, 4)

    def test_defaults(self, make_block):
        cb = ClassifiedBlock(block=make_block("x"), label=LayoutLabel.TEXT,
                             confidence=0.5)
        assert cb.source_region is None
        assert cb.refined is False

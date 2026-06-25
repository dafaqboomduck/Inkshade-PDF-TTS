"""Unit tests for narration.layout.feature_refiner heuristics."""

import pytest

from narration.layout.feature_refiner import (
    PageStats,
    _classify_by_features,
    detect_running_headers_footers,
    refine_classifications,
)
from narration.layout.models import ClassifiedBlock, LayoutLabel

PAGE_W = 612.0
PAGE_H = 792.0


def _stats(make_block):
    """A page whose median font size is 12pt (two body blocks)."""
    body = make_block(
        "This is ordinary body text with quite a few words on the page.",
        font_size=12.0,
    )
    return PageStats([body, body], PAGE_W, PAGE_H)


class TestPageStats:
    def test_median_odd(self, make_block):
        blocks = [make_block("a", font_size=s) for s in (10.0, 12.0, 20.0)]
        stats = PageStats(blocks, PAGE_W, PAGE_H)
        assert stats.median_font_size == 12.0
        assert stats.max_font_size == 20.0
        assert stats.min_font_size == 10.0

    def test_median_even(self, make_block):
        blocks = [make_block("a", font_size=s) for s in (10.0, 14.0)]
        stats = PageStats(blocks, PAGE_W, PAGE_H)
        assert stats.median_font_size == pytest.approx(12.0)

    def test_empty_page_uses_defaults(self):
        stats = PageStats([], PAGE_W, PAGE_H)
        assert stats.median_font_size == 12.0


class TestClassifyByFeatures:
    def test_large_bold_top_block_is_title(self, make_block):
        title = make_block("Big Title Here", font_size=24.0, bold=True,
                           bbox=(50, 100, 400, 140))  # y-center ~0.15
        label, conf = _classify_by_features(title, _stats(make_block))
        assert label is LayoutLabel.TITLE
        assert conf > 0.5

    def test_page_number_at_bottom_is_footer(self, make_block):
        pn = make_block("12", font_size=10.0, bbox=(300, 770, 320, 785))
        label, _ = _classify_by_features(pn, _stats(make_block))
        assert label is LayoutLabel.PAGE_FOOTER

    def test_larger_short_block_is_section_header(self, make_block):
        sh = make_block("Section Header Title", font_size=16.0,
                        bbox=(50, 300, 400, 330))
        label, _ = _classify_by_features(sh, _stats(make_block))
        assert label is LayoutLabel.SECTION_HEADER

    def test_bulleted_block_is_list_item(self, make_block):
        li = make_block("• first item\n• second item", font_size=12.0,
                        bbox=(50, 300, 400, 360))
        label, _ = _classify_by_features(li, _stats(make_block))
        assert label is LayoutLabel.LIST_ITEM

    def test_plain_paragraph_is_text(self, make_block):
        body = make_block(
            "An ordinary paragraph of body text running to a fair length here.",
            font_size=12.0, bbox=(50, 400, 560, 460),
        )
        label, _ = _classify_by_features(body, _stats(make_block))
        assert label is LayoutLabel.TEXT


class TestRefineClassifications:
    def test_unknown_block_gets_reclassified_and_flagged(self, make_block):
        unknown = ClassifiedBlock(
            block=make_block("12", font_size=10.0, bbox=(300, 770, 320, 785)),
            label=LayoutLabel.UNKNOWN,
            confidence=0.0,
        )
        out = refine_classifications([unknown], PAGE_W, PAGE_H)
        assert out[0].label != LayoutLabel.UNKNOWN
        assert out[0].refined is True

    def test_high_confidence_detection_not_overridden(self, make_block):
        cb = ClassifiedBlock(
            block=make_block("Body text.", font_size=12.0),
            label=LayoutLabel.TEXT,
            confidence=0.95,
        )
        out = refine_classifications([cb], PAGE_W, PAGE_H)
        assert out[0].label is LayoutLabel.TEXT
        assert out[0].refined is False

    def test_low_confidence_section_header_at_top_becomes_page_header(self, make_block):
        cb = ClassifiedBlock(
            block=make_block("Chapter 1", font_size=12.0, bbox=(50, 10, 400, 30)),
            label=LayoutLabel.SECTION_HEADER,
            confidence=0.4,
        )
        out = refine_classifications([cb], PAGE_W, PAGE_H)
        assert out[0].label is LayoutLabel.PAGE_HEADER
        assert out[0].refined is True


class TestRunningHeaderFooter:
    def _header_block(self, make_block, text):
        return ClassifiedBlock(
            block=make_block(text, font_size=10.0, bbox=(50, 10, 400, 30)),
            label=LayoutLabel.TEXT,
            confidence=0.5,
        )

    def test_repeated_top_text_on_consecutive_pages_is_header(self, make_block):
        pages = {
            i: [self._header_block(make_block, "My Book Title")] for i in range(3)
        }
        heights = {i: PAGE_H for i in range(3)}
        detect_running_headers_footers(pages, heights)
        for i in range(3):
            assert pages[i][0].label is LayoutLabel.PAGE_HEADER
            assert pages[i][0].refined is True

    def test_text_below_threshold_is_not_reclassified(self, make_block):
        # Appears on only two pages -> below default min_occurrences=3.
        pages = {
            i: [self._header_block(make_block, "Rare Heading")] for i in range(2)
        }
        heights = {i: PAGE_H for i in range(2)}
        detect_running_headers_footers(pages, heights)
        for i in range(2):
            assert pages[i][0].label is LayoutLabel.TEXT

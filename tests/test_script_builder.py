"""Unit tests for narration.script.reading_script (script assembly)."""

from narration.layout.models import LayoutLabel
from narration.script.reading_script import (
    _block_sort_key,
    _build_text_to_char_map,
    build_document_script,
    build_page_script,
    preview_script,
)
from narration.script.models import TextRole


class TestBlockSortKey:
    def test_sorts_top_to_bottom_then_left_to_right(self, make_classified):
        a = make_classified("top", LayoutLabel.TEXT, bbox=(0, 0, 10, 10))
        b = make_classified("bottom", LayoutLabel.TEXT, bbox=(0, 100, 10, 110))
        c = make_classified("topright", LayoutLabel.TEXT, bbox=(50, 0, 60, 10))
        ordered = sorted([b, c, a], key=_block_sort_key)
        assert [cb.text for cb in ordered] == ["top", "topright", "bottom"]


class TestBuildTextToCharMap:
    def test_newlines_map_to_minus_one(self):
        # "ab\ncd" has 4 real chars; the newline maps to -1.
        m = _build_text_to_char_map("ab\ncd", num_chars=4)
        assert m == [0, 1, -1, 2, 3]

    def test_out_of_bounds_chars_map_to_minus_one(self):
        m = _build_text_to_char_map("abc", num_chars=1)
        assert m == [0, -1, -1]


class TestBuildPageScript:
    def test_skips_skip_roles(self, make_classified):
        header = make_classified("Running head", LayoutLabel.PAGE_HEADER)
        body = make_classified("Real content here.", LayoutLabel.TEXT)
        script = build_page_script([header, body], page_index=0)
        # PAGE_HEADER is dropped entirely; only body survives.
        assert all(inst.role != TextRole.SKIP for inst in script)
        assert any("Real content" in inst.text for inst in script)

    def test_body_block_is_split_into_sentences(self, make_classified):
        body = make_classified("First sentence. Second sentence.", LayoutLabel.TEXT)
        script = build_page_script([body], page_index=0)
        texts = [inst.text for inst in script]
        assert texts == ["First sentence.", "Second sentence."]

    def test_title_stays_single_instruction(self, make_classified):
        title = make_classified("A Grand Title. With Stops.", LayoutLabel.TITLE)
        script = build_page_script([title], page_index=2)
        assert len(script) == 1
        assert script[0].role == TextRole.TITLE
        assert script[0].page_index == 2

    def test_footnotes_skipped_by_default(self, make_classified):
        fn = make_classified("A footnote.", LayoutLabel.FOOTNOTE)
        assert build_page_script([fn], page_index=0) == []

    def test_footnotes_kept_when_flag_disabled(self, make_classified):
        fn = make_classified("A footnote.", LayoutLabel.FOOTNOTE)
        script = build_page_script([fn], page_index=0, skip_footnotes=False)
        assert len(script) == 1
        assert script[0].role == TextRole.FOOTNOTE

    def test_inter_sentence_pauses_on_multi_sentence_body(self, make_classified):
        body = make_classified("One. Two. Three.", LayoutLabel.TEXT)
        script = build_page_script([body], page_index=0)
        assert len(script) == 3
        # Middle sentence carries the small inter-sentence pause on both sides.
        assert script[1].prosody.pause_before > 0
        assert script[1].prosody.pause_after > 0


class TestBuildDocumentScript:
    def test_inserts_page_transition_between_pages_not_after_last(self, make_classified):
        pages = {
            0: [make_classified("Page one body.", LayoutLabel.TEXT)],
            1: [make_classified("Page two body.", LayoutLabel.TEXT)],
        }
        script = build_document_script(pages)
        transitions = [s for s in script if s.role == TextRole.PAGE_TRANSITION]
        assert len(transitions) == 1  # only between the two pages

    def test_pages_processed_in_sorted_order(self, make_classified):
        pages = {
            1: [make_classified("Second page.", LayoutLabel.TEXT)],
            0: [make_classified("First page.", LayoutLabel.TEXT)],
        }
        script = build_document_script(pages)
        spoken = [s.text for s in script if s.text]
        assert spoken[0] == "First page."

    def test_announce_pages_inserts_header(self, make_classified):
        pages = {0: [make_classified("Body.", LayoutLabel.TEXT)]}
        script = build_document_script(pages, announce_pages=True)
        assert script[0].text == "Page 1."
        assert script[0].role == TextRole.SECTION_HEADER

    def test_empty_document(self):
        assert build_document_script({}) == []


class TestPreviewScript:
    def test_includes_page_markers_and_transition(self, make_classified):
        pages = {
            0: [make_classified("Hello world here.", LayoutLabel.TEXT)],
            1: [make_classified("Second page here.", LayoutLabel.TEXT)],
        }
        out = preview_script(build_document_script(pages))
        assert "[PAGE 1]" in out
        assert "[PAGE 2]" in out
        assert "PAGE_TRANSITION" in out

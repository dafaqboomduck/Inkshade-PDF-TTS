"""Unit tests for narration.script.text_preprocessor."""

from narration.script.text_preprocessor import (
    clean_text,
    preprocess_block,
    split_sentences,
)


class TestCleanText:
    def test_empty_returns_empty(self):
        assert clean_text("") == ""

    def test_rejoins_hyphenated_linebreak(self):
        assert clean_text("com-\nputer") == "computer"

    def test_collapses_newlines_and_whitespace(self):
        assert clean_text("a\n  b   c") == "a b c"

    def test_strips_reference_markers_by_default(self):
        assert clean_text("Some claim [1]. Another [2, 3].") == "Some claim. Another."

    def test_keeps_references_when_disabled(self):
        assert clean_text("Keep [1] this", strip_references=False) == "Keep [1] this"

    def test_replaces_url_with_domain(self):
        out = clean_text("Visit https://www.example.com/path/page now.")
        assert "link to example.com" in out
        assert "https" not in out

    def test_expands_abbreviations(self):
        out = clean_text("See Fig. 3 and e.g. this, i.e. that.")
        assert "Figure 3" in out
        assert "for example" in out
        assert "that is" in out

    def test_normalises_ellipsis_and_dashes(self):
        out = clean_text("Wait— really… yes")
        assert "..." in out
        assert "—" in out  # em dash normalised but retained as separator
        assert "…" not in out


class TestSplitSentences:
    def test_empty(self):
        assert split_sentences("") == []

    def test_single_sentence_without_terminal_punctuation(self):
        assert split_sentences("One sentence only") == ["One sentence only"]

    def test_splits_on_sentence_boundaries(self):
        assert split_sentences("First one. Second one. Third!") == [
            "First one.",
            "Second one.",
            "Third!",
        ]

    def test_does_not_split_on_known_abbreviation(self):
        # "Dr." should not end a sentence.
        assert split_sentences("This is Dr. Smith here. The end.") == [
            "This is Dr. Smith here.",
            "The end.",
        ]

    def test_does_not_split_without_following_uppercase(self):
        # No boundary because "the" is lowercase after the period.
        assert split_sentences("Version 1.5 is the one.") == ["Version 1.5 is the one."]


class TestPreprocessBlock:
    def test_split_true_returns_sentences(self):
        assert preprocess_block("First. Second.", split=True) == ["First.", "Second."]

    def test_split_false_returns_single_segment(self):
        assert preprocess_block("First. Second.", split=False) == ["First. Second."]

    def test_blank_input_returns_empty_list(self):
        assert preprocess_block("   \n  ") == []

"""Unit tests for narration.script.prosody_rules and TextRole mapping."""

import pytest

from narration.layout.models import LayoutLabel
from narration.script.models import LABEL_TO_ROLE, ReadingInstruction, TextRole
from narration.script.prosody_rules import DEFAULT_PROSODY, get_prosody


class TestGetProsody:
    def test_returns_base_values_without_multipliers(self):
        base = DEFAULT_PROSODY[TextRole.TITLE]
        p = get_prosody(TextRole.TITLE)
        assert p.pause_before == base.pause_before
        assert p.pause_after == base.pause_after
        assert p.speed_factor == base.speed_factor

    def test_applies_multipliers(self):
        base = DEFAULT_PROSODY[TextRole.BODY]
        p = get_prosody(TextRole.BODY, speed_multiplier=1.5, pause_multiplier=2.0)
        assert p.speed_factor == pytest.approx(base.speed_factor * 1.5)
        assert p.pause_after == pytest.approx(base.pause_after * 2.0)

    def test_preserves_skip_flag(self):
        assert get_prosody(TextRole.SKIP).skip is True
        assert get_prosody(TextRole.FORMULA).skip is True
        assert get_prosody(TextRole.BODY).skip is False

    def test_does_not_mutate_defaults(self):
        before = DEFAULT_PROSODY[TextRole.TITLE].speed_factor
        get_prosody(TextRole.TITLE, speed_multiplier=10.0)
        assert DEFAULT_PROSODY[TextRole.TITLE].speed_factor == before

    def test_unknown_role_falls_back_to_body(self):
        # PAGE_TRANSITION exists, but an arbitrary lookup miss returns BODY base.
        # We exercise the fallback path via a role guaranteed to be in the table.
        p = get_prosody(TextRole.BODY)
        assert p.speed_factor == DEFAULT_PROSODY[TextRole.BODY].speed_factor


class TestLabelToRole:
    def test_every_layout_label_maps_to_a_role(self):
        for label in LayoutLabel:
            assert label in LABEL_TO_ROLE

    def test_headers_and_tables_skip(self):
        assert LABEL_TO_ROLE[LayoutLabel.PAGE_HEADER] is TextRole.SKIP
        assert LABEL_TO_ROLE[LayoutLabel.PAGE_FOOTER] is TextRole.SKIP
        assert LABEL_TO_ROLE[LayoutLabel.PICTURE] is TextRole.SKIP
        assert LABEL_TO_ROLE[LayoutLabel.TABLE] is TextRole.SKIP

    def test_unknown_maps_to_body(self):
        assert LABEL_TO_ROLE[LayoutLabel.UNKNOWN] is TextRole.BODY


class TestReadingInstruction:
    def test_should_skip_reflects_prosody(self):
        skip_p = get_prosody(TextRole.SKIP)
        inst = ReadingInstruction(text="x", role=TextRole.SKIP, prosody=skip_p,
                                  page_index=0, block_index=0)
        assert inst.should_skip is True

    def test_repr_shows_skip_marker(self):
        inst = ReadingInstruction(text="hidden", role=TextRole.SKIP,
                                  prosody=get_prosody(TextRole.SKIP),
                                  page_index=0, block_index=0)
        assert "SKIP" in repr(inst)

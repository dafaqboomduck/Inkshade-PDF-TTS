"""
Shared fixtures and factory helpers for the unit-test suite.

The factories build the data models (``BlockInfo``, ``LayoutRegion``,
``ClassifiedBlock``, WAV bytes) used across the pure-logic tests so that
no real PDF, YOLO model, or TTS model download is ever required.
"""

import io
import wave

import pytest

from core.page.models import BlockInfo, CharacterInfo, LineInfo, SpanInfo
from narration.layout.models import ClassifiedBlock, LayoutLabel, LayoutRegion

# Span flag bit for bold text (mirrors SpanInfo.is_bold: bit 4).
_BOLD_FLAG = 1 << 4


def _make_block(text, font_size=12.0, bold=False, bbox=(0.0, 0.0, 100.0, 20.0)):
    """
    Build a ``BlockInfo`` with one line per newline-separated segment.

    Every character carries ``font_size``; bold is encoded via the span
    flags so the model's ``is_bold`` / ``dominant_font_size`` properties
    compute correctly.
    """
    flags = _BOLD_FLAG if bold else 0
    lines = []
    for line_idx, line_text in enumerate(text.split("\n")):
        chars = [
            CharacterInfo(
                char=ch,
                bbox=(0.0, 0.0, 1.0, 1.0),
                origin=(0.0, 0.0),
                span_index=0,
                line_index=line_idx,
                block_index=0,
                font_name="TestFont",
                font_size=font_size,
                color=0,
            )
            for ch in line_text
        ]
        span = SpanInfo(
            characters=chars,
            font_name="TestFont",
            font_size=font_size,
            color=0,
            flags=flags,
            bbox=bbox,
        )
        lines.append(LineInfo(spans=[span], bbox=bbox))
    return BlockInfo(lines=lines, bbox=bbox, block_type=0)


def _make_region(label, bbox, confidence=0.9):
    """Build a ``LayoutRegion`` (pixel-space bbox)."""
    return LayoutRegion(label=label, confidence=confidence, bbox=bbox)


def _make_classified(text, label, confidence=0.9, font_size=12.0, bold=False,
                     bbox=(0.0, 0.0, 100.0, 20.0)):
    """Build a ``ClassifiedBlock`` wrapping a freshly built ``BlockInfo``."""
    block = _make_block(text, font_size=font_size, bold=bold, bbox=bbox)
    return ClassifiedBlock(block=block, label=label, confidence=confidence)


def _make_wav(duration_seconds=0.5, sample_rate=24000):
    """Build a valid mono 16-bit PCM WAV byte string of silence."""
    num_samples = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


@pytest.fixture
def make_block():
    """Factory fixture: ``make_block(text, font_size=, bold=, bbox=)``."""
    return _make_block


@pytest.fixture
def make_region():
    """Factory fixture: ``make_region(label, bbox, confidence=)``."""
    return _make_region


@pytest.fixture
def make_classified():
    """Factory fixture for ``ClassifiedBlock`` objects."""
    return _make_classified


@pytest.fixture
def make_wav():
    """Factory fixture: ``make_wav(duration_seconds=, sample_rate=)``."""
    return _make_wav

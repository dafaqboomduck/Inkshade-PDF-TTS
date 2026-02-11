"""
Text structure data models for PDF pages.
Stripped for narration POC — no link, interaction, or Qt dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CharacterInfo:
    """Represents a single character with its position and metadata."""

    char: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    origin: Tuple[float, float]  # baseline origin point
    span_index: int
    line_index: int
    block_index: int
    font_name: str
    font_size: float
    color: int  # Color as integer

    # Computed index in the character list (set after extraction)
    global_index: int = -1

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within character bounds."""
        return self.bbox[0] <= x <= self.bbox[2] and self.bbox[1] <= y <= self.bbox[3]


@dataclass
class SpanInfo:
    """A span of text with consistent formatting."""

    characters: List[CharacterInfo] = field(default_factory=list)
    font_name: str = ""
    font_size: float = 12.0
    color: int = 0
    flags: int = 0  # Font flags (bold, italic, etc.)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)

    @property
    def text(self) -> str:
        return "".join(c.char for c in self.characters)

    @property
    def is_bold(self) -> bool:
        """Check if this span is bold (bit 4 of flags)."""
        return bool(self.flags & (1 << 4))

    @property
    def is_italic(self) -> bool:
        """Check if this span is italic (bit 1 of flags)."""
        return bool(self.flags & (1 << 1))

    @property
    def is_monospaced(self) -> bool:
        """Check if this span uses a monospaced font (bit 3 of flags)."""
        return bool(self.flags & (1 << 3))


@dataclass
class LineInfo:
    """A line of text containing multiple spans."""

    spans: List[SpanInfo] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    wmode: int = 0  # Writing mode (0=horizontal, 1=vertical)
    dir_vector: Tuple[float, float] = (1, 0)  # Text direction

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)

    @property
    def all_characters(self) -> List[CharacterInfo]:
        chars = []
        for span in self.spans:
            chars.extend(span.characters)
        return chars

    @property
    def dominant_font_size(self) -> float:
        """Get the most common font size in this line."""
        if not self.spans:
            return 12.0
        sizes = {}
        for span in self.spans:
            for char in span.characters:
                sizes[char.font_size] = sizes.get(char.font_size, 0) + 1
        return max(sizes, key=sizes.get) if sizes else 12.0

    @property
    def is_bold(self) -> bool:
        """Check if the majority of text in this line is bold."""
        if not self.spans:
            return False
        bold_chars = sum(len(s.characters) for s in self.spans if s.is_bold)
        total_chars = sum(len(s.characters) for s in self.spans)
        return bold_chars > total_chars / 2 if total_chars > 0 else False


@dataclass
class BlockInfo:
    """A block of text (paragraph or text region)."""

    lines: List[LineInfo] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    block_type: int = 0  # 0 = text, 1 = image

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def word_count(self) -> int:
        """Approximate word count in this block."""
        return len(self.text.split())

    @property
    def dominant_font_size(self) -> float:
        """Get the most common font size across all lines."""
        sizes = {}
        for line in self.lines:
            for span in line.spans:
                for char in span.characters:
                    sizes[char.font_size] = sizes.get(char.font_size, 0) + 1
        return max(sizes, key=sizes.get) if sizes else 12.0

    @property
    def is_bold(self) -> bool:
        """Check if the majority of text in this block is bold."""
        bold_chars = 0
        total_chars = 0
        for line in self.lines:
            for span in line.spans:
                count = len(span.characters)
                total_chars += count
                if span.is_bold:
                    bold_chars += count
        return bold_chars > total_chars / 2 if total_chars > 0 else False

    @property
    def all_characters(self) -> List[CharacterInfo]:
        """Get all characters in this block in order."""
        chars = []
        for line in self.lines:
            chars.extend(line.all_characters)
        return chars

    @property
    def all_font_sizes(self) -> List[float]:
        """Get all font sizes used in this block."""
        return [c.font_size for c in self.all_characters]

    @property
    def all_font_names(self) -> List[str]:
        """Get all unique font names used in this block."""
        return list({c.font_name for c in self.all_characters})

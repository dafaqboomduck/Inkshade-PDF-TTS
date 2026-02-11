"""
Character-level text extraction for PDF pages.
Stripped for narration POC — no Qt dependencies.
"""

from typing import Dict, List, Optional, Tuple

import fitz

from .models import BlockInfo, CharacterInfo, LineInfo, SpanInfo


class PageTextLayer:
    """
    Extracts and manages character-level text data for a PDF page.

    Provides structured access to blocks, lines, spans, and individual
    characters with full typographic metadata (font name, size, bold/italic
    flags, color, position). This metadata drives the narration pipeline's
    document structure analysis.
    """

    def __init__(self, page: fitz.Page):
        self.page = page
        self.blocks: List[BlockInfo] = []
        self.characters: List[CharacterInfo] = []
        self._char_grid: Dict[Tuple[int, int], List[CharacterInfo]] = {}
        self._grid_size = 50  # Grid cell size for spatial lookup

        self._extract_text_structure()
        self._build_spatial_index()

    def _extract_text_structure(self):
        """Extract character-level text structure from the page."""
        flags = (
            fitz.TEXT_PRESERVE_WHITESPACE
            | fitz.TEXT_PRESERVE_LIGATURES
            | fitz.TEXT_PRESERVE_IMAGES
        )

        try:
            text_dict = self.page.get_text("rawdict", flags=flags)
        except Exception as e:
            print(f"Failed to extract text: {e}")
            return

        char_index = 0

        for block_idx, block_data in enumerate(text_dict.get("blocks", [])):
            # Skip image blocks
            if block_data.get("type") != 0:
                continue

            block = BlockInfo(
                bbox=tuple(block_data.get("bbox", (0, 0, 0, 0))), block_type=0
            )

            for line_idx, line_data in enumerate(block_data.get("lines", [])):
                line = LineInfo(
                    bbox=tuple(line_data.get("bbox", (0, 0, 0, 0))),
                    wmode=line_data.get("wmode", 0),
                    dir_vector=tuple(line_data.get("dir", (1, 0))),
                )

                for span_idx, span_data in enumerate(line_data.get("spans", [])):
                    span = SpanInfo(
                        font_name=span_data.get("font", ""),
                        font_size=span_data.get("size", 12.0),
                        color=span_data.get("color", 0),
                        flags=span_data.get("flags", 0),
                        bbox=tuple(span_data.get("bbox", (0, 0, 0, 0))),
                    )

                    for char_data in span_data.get("chars", []):
                        char = CharacterInfo(
                            char=char_data.get("c", ""),
                            bbox=tuple(char_data.get("bbox", (0, 0, 0, 0))),
                            origin=tuple(char_data.get("origin", (0, 0))),
                            span_index=span_idx,
                            line_index=line_idx,
                            block_index=block_idx,
                            font_name=span.font_name,
                            font_size=span.font_size,
                            color=span.color,
                            global_index=char_index,
                        )

                        span.characters.append(char)
                        self.characters.append(char)
                        char_index += 1

                    if span.characters:
                        line.spans.append(span)

                if line.spans:
                    block.lines.append(line)

            if block.lines:
                self.blocks.append(block)

    def _build_spatial_index(self):
        """Build a grid-based spatial index for fast character lookup."""
        self._char_grid.clear()

        for char in self.characters:
            min_col = int(char.bbox[0] / self._grid_size)
            max_col = int(char.bbox[2] / self._grid_size)
            min_row = int(char.bbox[1] / self._grid_size)
            max_row = int(char.bbox[3] / self._grid_size)

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    key = (row, col)
                    if key not in self._char_grid:
                        self._char_grid[key] = []
                    self._char_grid[key].append(char)

    def get_char_at_point(self, x: float, y: float) -> Optional[CharacterInfo]:
        """Find the character at the given PDF coordinates."""
        row = int(y / self._grid_size)
        col = int(x / self._grid_size)

        candidates = self._char_grid.get((row, col), [])

        for char in candidates:
            if char.contains_point(x, y):
                return char

        return None

    def get_chars_in_range(
        self, start: CharacterInfo, end: CharacterInfo
    ) -> List[CharacterInfo]:
        """Get all characters between start and end (inclusive)."""
        start_idx = start.global_index
        end_idx = end.global_index

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        return self.characters[start_idx : end_idx + 1]

    def get_chars_in_rect(
        self, rect: Tuple[float, float, float, float]
    ) -> List[CharacterInfo]:
        """Get all characters that intersect with a rectangle."""
        x0, y0, x1, y1 = rect
        result = []

        min_row = int(y0 / self._grid_size)
        max_row = int(y1 / self._grid_size)
        min_col = int(x0 / self._grid_size)
        max_col = int(x1 / self._grid_size)

        seen = set()

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                for char in self._char_grid.get((row, col), []):
                    if char.global_index in seen:
                        continue

                    if (
                        char.bbox[0] <= x1
                        and char.bbox[2] >= x0
                        and char.bbox[1] <= y1
                        and char.bbox[3] >= y0
                    ):
                        result.append(char)
                        seen.add(char.global_index)

        result.sort(key=lambda c: c.global_index)
        return result

    def get_text_from_chars(self, chars: List[CharacterInfo]) -> str:
        """Extract text string from a list of characters, preserving line breaks."""
        if not chars:
            return ""

        sorted_chars = sorted(chars, key=lambda c: c.global_index)

        lines = []
        current_line = []
        last_line_key = None

        for char in sorted_chars:
            line_key = (char.block_index, char.line_index)

            if last_line_key is not None and line_key != last_line_key:
                lines.append("".join(c.char for c in current_line))
                current_line = []

            current_line.append(char)
            last_line_key = line_key

        if current_line:
            lines.append("".join(c.char for c in current_line))

        return "\n".join(lines)

    @property
    def full_text(self) -> str:
        """Get all text on the page."""
        return self.get_text_from_chars(self.characters)

    @property
    def page_font_sizes(self) -> List[float]:
        """Get all font sizes used on this page."""
        return [c.font_size for c in self.characters]

    @property
    def median_font_size(self) -> float:
        """Get the median font size on this page."""
        sizes = self.page_font_sizes
        if not sizes:
            return 12.0
        sorted_sizes = sorted(sizes)
        mid = len(sorted_sizes) // 2
        if len(sorted_sizes) % 2 == 0:
            return (sorted_sizes[mid - 1] + sorted_sizes[mid]) / 2
        return sorted_sizes[mid]

    def __len__(self) -> int:
        return len(self.characters)

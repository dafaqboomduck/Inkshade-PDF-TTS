"""
Refines layout classifications using typographic and positional features.

Handles two cases:
1. UNKNOWN blocks (no YOLO match) — classified from scratch using font
   size, weight, position, and word count.
2. Low-confidence detections — cross-checked against text features and
   overridden when the evidence disagrees.
"""

import re
from typing import Dict, List, Optional, Tuple

from core.page.models import BlockInfo

from .models import ClassifiedBlock, LayoutLabel

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_RE_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_RE_BULLET = re.compile(
    r"^\s*(?:[•●○◦▪▸►–—-]|\d{1,3}[.)]\s|[a-zA-Z][.)]\s|[ivxIVX]+[.)]\s)"
)
_RE_REFERENCE_MARKER = re.compile(r"^\s*\[\d+\]")


# ---------------------------------------------------------------------------
# Page-level statistics
# ---------------------------------------------------------------------------


class PageStats:
    """Aggregate typographic statistics for one page."""

    def __init__(
        self,
        blocks: List[BlockInfo],
        page_width: float,
        page_height: float,
    ):
        self.page_width = page_width
        self.page_height = page_height

        # Collect all character font sizes across the page
        all_sizes: List[float] = []
        for b in blocks:
            all_sizes.extend(b.all_font_sizes)

        self.median_font_size = self._median(all_sizes) if all_sizes else 12.0
        self.max_font_size = max(all_sizes) if all_sizes else 12.0
        self.min_font_size = min(all_sizes) if all_sizes else 12.0

    @staticmethod
    def _median(values: List[float]) -> float:
        s = sorted(values)
        mid = len(s) // 2
        if len(s) % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2
        return s[mid]


# ---------------------------------------------------------------------------
# Feature-based classifier (for UNKNOWN blocks)
# ---------------------------------------------------------------------------


def _classify_by_features(
    block: BlockInfo,
    stats: PageStats,
) -> Tuple[LayoutLabel, float]:
    """
    Classify a block using typographic and positional heuristics.

    Returns (label, confidence) where confidence is in [0, 1].
    """
    text = block.text.strip()
    font_size = block.dominant_font_size
    ratio = font_size / stats.median_font_size if stats.median_font_size else 1.0
    bold = block.is_bold
    word_count = block.word_count
    y_center = (block.bbox[1] + block.bbox[3]) / 2
    y_ratio = y_center / stats.page_height if stats.page_height else 0.5

    # ---- Page number (small, near top or bottom, very short) ----
    if _RE_PAGE_NUMBER.match(text):
        if y_ratio < 0.08 or y_ratio > 0.92:
            return LayoutLabel.PAGE_FOOTER, 0.85
        # Numbers mid-page are more likely labels/formulas
        return LayoutLabel.TEXT, 0.3

    # ---- Page header / footer by position ----
    if y_ratio < 0.06 and word_count <= 15:
        return LayoutLabel.PAGE_HEADER, 0.7
    if y_ratio > 0.94 and word_count <= 15:
        return LayoutLabel.PAGE_FOOTER, 0.7

    # ---- Title: very large font, bold, short, near top ----
    if ratio >= 1.6 and bold and word_count <= 20 and y_ratio < 0.4:
        return LayoutLabel.TITLE, 0.75

    # ---- Section header: larger font or bold, short text ----
    if ratio >= 1.2 and word_count <= 20:
        return LayoutLabel.SECTION_HEADER, 0.7
    if bold and word_count <= 15 and ratio >= 1.0:
        return LayoutLabel.SECTION_HEADER, 0.6

    # ---- List item: starts with bullet / number pattern ----
    lines = text.split("\n")
    bullet_lines = sum(1 for l in lines if _RE_BULLET.match(l))
    if bullet_lines > 0 and bullet_lines >= len(lines) * 0.5:
        return LayoutLabel.LIST_ITEM, 0.65

    # ---- Footnote: small font, near bottom ----
    if ratio <= 0.85 and y_ratio > 0.75:
        return LayoutLabel.FOOTNOTE, 0.6

    # ---- Caption: small font, short, mid-page ----
    if ratio <= 0.9 and word_count <= 30 and 0.1 < y_ratio < 0.9:
        return LayoutLabel.CAPTION, 0.4

    # ---- Default: body text ----
    return LayoutLabel.TEXT, 0.5


# ---------------------------------------------------------------------------
# Cross-check / override logic
# ---------------------------------------------------------------------------


def _should_override(
    cb: ClassifiedBlock,
    stats: PageStats,
) -> Optional[Tuple[LayoutLabel, float]]:
    """
    Cross-check a YOLO classification against text features.

    Returns a (new_label, new_confidence) if the override is warranted,
    or None to keep the original label.
    """
    block = cb.block
    text = block.text.strip()
    font_size = block.dominant_font_size
    ratio = font_size / stats.median_font_size if stats.median_font_size else 1.0
    bold = block.is_bold
    word_count = block.word_count
    y_center = (block.bbox[1] + block.bbox[3]) / 2
    y_ratio = y_center / stats.page_height if stats.page_height else 0.5

    # Only override low-confidence detections
    if cb.confidence > 0.75:
        return None

    # YOLO says SECTION_HEADER but it's in the top/bottom margin →
    # almost certainly a running page header (e.g. "10  CHAPTER 1. INTRO")
    if cb.label == LayoutLabel.SECTION_HEADER:
        if y_ratio < 0.08:
            return LayoutLabel.PAGE_HEADER, 0.85
        if y_ratio > 0.92:
            return LayoutLabel.PAGE_FOOTER, 0.85

    # YOLO says TEXT but it looks like a header
    if cb.label == LayoutLabel.TEXT:
        if ratio >= 1.5 and bold and word_count <= 15:
            return LayoutLabel.SECTION_HEADER, 0.65
        if ratio >= 1.8 and word_count <= 20 and y_ratio < 0.35:
            return LayoutLabel.TITLE, 0.6

    # YOLO says TITLE but it's in the margin → running header
    if cb.label == LayoutLabel.TITLE:
        if y_ratio < 0.08 and word_count <= 15:
            return LayoutLabel.PAGE_HEADER, 0.85
        if y_ratio > 0.92 and word_count <= 15:
            return LayoutLabel.PAGE_FOOTER, 0.85

    # YOLO says SECTION_HEADER but it's a full paragraph
    if cb.label == LayoutLabel.SECTION_HEADER:
        if word_count > 40 and ratio < 1.15:
            return LayoutLabel.TEXT, 0.6

    # YOLO says TEXT but it's a page number
    if cb.label == LayoutLabel.TEXT and _RE_PAGE_NUMBER.match(text):
        if y_ratio < 0.08 or y_ratio > 0.92:
            return LayoutLabel.PAGE_FOOTER, 0.8

    # YOLO says TEXT but it has bullet patterns
    if cb.label == LayoutLabel.TEXT:
        lines = text.split("\n")
        bullet_lines = sum(1 for l in lines if _RE_BULLET.match(l))
        if bullet_lines >= len(lines) * 0.6 and len(lines) >= 2:
            return LayoutLabel.LIST_ITEM, 0.6

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def refine_classifications(
    classified_blocks: List[ClassifiedBlock],
    page_width: float,
    page_height: float,
) -> List[ClassifiedBlock]:
    """
    Refine layout classifications using typographic features.

    - UNKNOWN blocks are classified from scratch.
    - Low-confidence detections are cross-checked and potentially
      overridden.

    Modifies the list in-place and also returns it for convenience.
    """
    raw_blocks = [cb.block for cb in classified_blocks]
    stats = PageStats(raw_blocks, page_width, page_height)

    for cb in classified_blocks:
        # Classify UNKNOWN blocks
        if cb.label == LayoutLabel.UNKNOWN:
            label, conf = _classify_by_features(cb.block, stats)
            cb.label = label
            cb.confidence = conf
            cb.refined = True
            continue

        # Cross-check existing labels
        override = _should_override(cb, stats)
        if override is not None:
            cb.label, cb.confidence = override
            cb.refined = True

    return classified_blocks


# ---------------------------------------------------------------------------
# Running header/footer detection (cross-page)
# ---------------------------------------------------------------------------


def detect_running_headers_footers(
    pages_classified: Dict[int, List[ClassifiedBlock]],
    page_heights: Dict[int, float],
    min_occurrences: int = 3,
) -> Dict[int, List[ClassifiedBlock]]:
    """
    Detect repeated text at consistent positions across pages and
    reclassify as PAGE_HEADER / PAGE_FOOTER.

    A text string that appears in the top 8% or bottom 8% of the page
    on *min_occurrences* or more consecutive pages is considered a
    running header or footer.

    Modifies the blocks in-place and returns the same dict.
    """
    # Collect candidates: (normalised text, top/bottom) → list of (page, block)
    candidates: Dict[Tuple[str, str], List[Tuple[int, ClassifiedBlock]]] = {}

    for page_idx, blocks in sorted(pages_classified.items()):
        h = page_heights.get(page_idx, 792.0)  # default letter height
        for cb in blocks:
            y_center = (cb.bbox[1] + cb.bbox[3]) / 2
            y_ratio = y_center / h if h else 0.5

            if y_ratio < 0.08:
                pos = "top"
            elif y_ratio > 0.92:
                pos = "bottom"
            else:
                continue

            norm = cb.text.strip().lower()
            # Skip very short text (likely just page numbers already handled)
            if len(norm) < 3:
                continue

            key = (norm, pos)
            if key not in candidates:
                candidates[key] = []
            candidates[key].append((page_idx, cb))

    # Find runs of min_occurrences consecutive pages
    for (norm_text, pos), occurrences in candidates.items():
        if len(occurrences) < min_occurrences:
            continue

        pages_with = sorted(set(p for p, _ in occurrences))

        # Check for a consecutive run
        run_len = 1
        max_run = 1
        for i in range(1, len(pages_with)):
            if pages_with[i] == pages_with[i - 1] + 1:
                run_len += 1
                max_run = max(max_run, run_len)
            else:
                run_len = 1

        if max_run >= min_occurrences:
            new_label = (
                LayoutLabel.PAGE_HEADER if pos == "top" else LayoutLabel.PAGE_FOOTER
            )
            for _, cb in occurrences:
                cb.label = new_label
                cb.confidence = 0.9
                cb.refined = True

    return pages_classified

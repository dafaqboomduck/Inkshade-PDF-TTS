"""
Data models for document layout detection and block classification.

LayoutLabel corresponds to the 11 DocLayNet classes produced by the
YOLO detector.  ClassifiedBlock pairs a text BlockInfo with its
resolved layout label and confidence score.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from core.page.models import BlockInfo

# ---------------------------------------------------------------------------
# DocLayNet class index → label mapping (standard across community models)
# ---------------------------------------------------------------------------


class LayoutLabel(Enum):
    """Document layout element types from DocLayNet."""

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2
    LIST_ITEM = 3
    PAGE_FOOTER = 4
    PAGE_HEADER = 5
    PICTURE = 6
    SECTION_HEADER = 7
    TABLE = 8
    TEXT = 9
    TITLE = 10

    # Fallback for blocks not matched by YOLO
    UNKNOWN = -1


# Reverse lookup: int → LayoutLabel
DOCLAYNET_INDEX_TO_LABEL = {
    label.value: label for label in LayoutLabel if label.value >= 0
}


@dataclass
class LayoutRegion:
    """
    A single region detected by the YOLO layout model.

    Coordinates are in **pixel space** of the rendered page image
    (i.e. they depend on the render scale).
    """

    label: LayoutLabel
    confidence: float
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def to_pdf_coords(
        self, scale: float, page_width: float, page_height: float
    ) -> "LayoutRegion":
        """
        Return a copy with bbox converted from pixel coords to PDF
        point coords by dividing by *scale*.
        """
        x0, y0, x1, y1 = self.bbox
        return LayoutRegion(
            label=self.label,
            confidence=self.confidence,
            bbox=(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
        )

    def __repr__(self) -> str:
        x0, y0, x1, y1 = self.bbox
        return (
            f"LayoutRegion({self.label.name}, "
            f"conf={self.confidence:.2f}, "
            f"bbox=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}])"
        )


@dataclass
class ClassifiedBlock:
    """
    A text block paired with its resolved layout classification.

    This is the primary data structure handed to the reading script
    builder (Task 4).
    """

    block: BlockInfo
    label: LayoutLabel
    confidence: float

    # The YOLO region that matched this block (None if classified
    # purely by typographic features).
    source_region: Optional[LayoutRegion] = None

    # Set by the feature refiner when it overrides the YOLO label
    refined: bool = False

    @property
    def text(self) -> str:
        return self.block.text

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.block.bbox

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        tag = " [refined]" if self.refined else ""
        return (
            f"ClassifiedBlock({self.label.name}, "
            f"conf={self.confidence:.2f}{tag}, "
            f"'{preview}')"
        )

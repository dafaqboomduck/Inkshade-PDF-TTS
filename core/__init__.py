"""
Core backend for Inkshade PDF Narration.
Text extraction and page model — no annotations, selection, search, or Qt.
"""

from .document import PDFDocumentReader
from .page import BlockInfo, CharacterInfo, LineInfo, PageModel, PageTextLayer, SpanInfo

__all__ = [
    "PDFDocumentReader",
    "PageModel",
    "PageTextLayer",
    "CharacterInfo",
    "SpanInfo",
    "LineInfo",
    "BlockInfo",
]

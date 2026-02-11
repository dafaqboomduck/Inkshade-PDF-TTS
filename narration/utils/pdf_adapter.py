"""
Qt-free PDF adapter for the narration pipeline.

Provides clean functions to extract text structure and render page images
using fitz (PyMuPDF) directly, avoiding any Qt dependency.
"""

from typing import Dict, List, Optional, Tuple

import fitz
from PIL import Image

from core.page.models import BlockInfo
from core.page.text_layer import PageTextLayer


def open_pdf(pdf_path: str) -> fitz.Document:
    """
    Open a PDF document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A fitz.Document instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If fitz cannot open the file.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}") from e
    return doc


def extract_page_text_structure(pdf_path: str, page_index: int) -> List[BlockInfo]:
    """
    Extract structured text blocks for a single page.

    Opens the PDF, builds a PageTextLayer for the requested page, and
    returns the list of BlockInfo objects with full character / span / line
    metadata.

    Args:
        pdf_path:   Path to the PDF file.
        page_index: 0-based page number.

    Returns:
        List of BlockInfo for the page (may be empty for image-only pages).
    """
    doc = open_pdf(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(
                f"Page index {page_index} out of range "
                f"(document has {doc.page_count} pages)"
            )
        page = doc.load_page(page_index)
        text_layer = PageTextLayer(page)
        return text_layer.blocks
    finally:
        doc.close()


def extract_all_pages(
    pdf_path: str,
) -> Dict[int, List[BlockInfo]]:
    """
    Extract structured text blocks for every page in the document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping page index → list of BlockInfo.
    """
    doc = open_pdf(pdf_path)
    try:
        result: Dict[int, List[BlockInfo]] = {}
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            text_layer = PageTextLayer(page)
            result[idx] = text_layer.blocks
        return result
    finally:
        doc.close()


def render_page_to_pil(
    pdf_path: str, page_index: int, scale: float = 1.5
) -> Image.Image:
    """
    Render a single PDF page to a PIL Image (RGB).

    Args:
        pdf_path:   Path to the PDF file.
        page_index: 0-based page number.
        scale:      Resolution multiplier (1.5 ≈ 108 dpi, good for YOLO).

    Returns:
        PIL.Image.Image in RGB mode.
    """
    doc = open_pdf(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(
                f"Page index {page_index} out of range "
                f"(document has {doc.page_count} pages)"
            )
        page = doc.load_page(page_index)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    finally:
        doc.close()


def get_page_dimensions(pdf_path: str, page_index: int) -> Tuple[float, float]:
    """
    Get page width and height in PDF points (1 point = 1/72 inch).

    Args:
        pdf_path:   Path to the PDF file.
        page_index: 0-based page number.

    Returns:
        (width, height) in points.
    """
    doc = open_pdf(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(
                f"Page index {page_index} out of range "
                f"(document has {doc.page_count} pages)"
            )
        page = doc.load_page(page_index)
        rect = page.rect
        return rect.width, rect.height
    finally:
        doc.close()


def get_page_count(pdf_path: str) -> int:
    """Return the total number of pages in the PDF."""
    doc = open_pdf(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Convenience helpers for the pipeline (operate on an already-open document
# so we don't re-open the file for every page).
# ---------------------------------------------------------------------------


class PDFAdapter:
    """
    Stateful adapter that keeps the document open across multiple
    page operations.  Preferred over the standalone functions when
    processing an entire document sequentially.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = open_pdf(pdf_path)
        self.page_count = self.doc.page_count

    # -- text extraction ----------------------------------------------------

    def text_structure(self, page_index: int) -> List[BlockInfo]:
        """Return BlockInfo list for *page_index*."""
        page = self.doc.load_page(page_index)
        return PageTextLayer(page).blocks

    def text_layer(self, page_index: int) -> PageTextLayer:
        """Return the full PageTextLayer for *page_index*."""
        page = self.doc.load_page(page_index)
        return PageTextLayer(page)

    # -- rendering ----------------------------------------------------------

    def render(self, page_index: int, scale: float = 1.5) -> Image.Image:
        """Render *page_index* to a PIL RGB image."""
        page = self.doc.load_page(page_index)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # -- geometry -----------------------------------------------------------

    def dimensions(self, page_index: int) -> Tuple[float, float]:
        """Return (width, height) in PDF points."""
        rect = self.doc.load_page(page_index).rect
        return rect.width, rect.height

    # -- lifecycle ----------------------------------------------------------

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return f"PDFAdapter('{self.pdf_path}', pages={self.page_count})"

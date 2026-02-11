"""
PDF document reading for the narration pipeline.
Stripped for narration POC — no Qt dependencies, no rendering.
"""

from typing import List, Optional, Tuple

import fitz  # PyMuPDF


class PDFDocumentReader:
    """Handles PDF document loading and text extraction for narration."""

    def __init__(self):
        self.doc: Optional[fitz.Document] = None
        self.total_pages: int = 0
        self.current_file_path: Optional[str] = None

    def load_pdf(self, file_path: str) -> Tuple[bool, int]:
        """
        Load a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (success flag, number of pages)
        """
        try:
            if self.doc:
                self.close_document()

            self.doc = fitz.open(file_path)
            self.total_pages = self.doc.page_count
            self.current_file_path = file_path

            return True, self.total_pages

        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False, 0

    def close_document(self) -> None:
        """Close the current PDF document and clear all state."""
        if self.doc:
            self.doc.close()
            self.doc = None

        self.total_pages = 0
        self.current_file_path = None

    def get_page(self, page_index: int) -> Optional[fitz.Page]:
        """
        Get a page object for direct operations.

        Args:
            page_index: 0-based index of the page

        Returns:
            PyMuPDF page object, or None if invalid
        """
        if not self.doc or page_index < 0 or page_index >= self.total_pages:
            return None

        try:
            return self.doc.load_page(page_index)
        except Exception:
            return None

    def get_page_size(self, page_index: int) -> Tuple[float, float]:
        """
        Get the size of a page in points.

        Args:
            page_index: 0-based index of the page

        Returns:
            Tuple of (width, height) in points
        """
        page = self.get_page(page_index)
        if page:
            rect = page.rect
            return rect.width, rect.height
        return 0.0, 0.0

    def render_page_to_image(self, page_index: int, scale: float = 1.5):
        """
        Render a page to a PIL Image for layout detection.

        Args:
            page_index: 0-based index of the page
            scale: Resolution scale factor (1.5 is good for YOLO)

        Returns:
            PIL.Image.Image or None if rendering fails
        """
        page = self.get_page(page_index)
        if not page:
            return None

        try:
            from PIL import Image

            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img

        except ImportError:
            print("PIL/Pillow is required for page image rendering.")
            return None
        except Exception as e:
            print(f"Error rendering page {page_index} to image: {e}")
            return None

    def extract_text(self, page_index: int) -> str:
        """
        Extract plain text from a page.

        Args:
            page_index: 0-based index of the page

        Returns:
            Plain text content of the page
        """
        page = self.get_page(page_index)
        if page:
            return page.get_text()
        return ""

    def is_loaded(self) -> bool:
        """Check if a document is currently loaded."""
        return self.doc is not None

    def get_file_path(self) -> Optional[str]:
        """Get the path of the currently loaded file."""
        return self.current_file_path

    def get_page_count(self) -> int:
        """Get the total number of pages."""
        return self.total_pages

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close document on context exit."""
        self.close_document()
        return False

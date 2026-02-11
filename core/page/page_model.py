"""
Page model for narration pipeline.
Stripped for narration POC — no Qt, no rendering, no links.
Provides text layer access and page geometry only.
"""

from typing import Optional

import fitz

from .text_layer import PageTextLayer


class PageModel:
    """
    Lightweight page model providing structured text access.

    Uses lazy loading for the text layer to optimize memory when
    processing large documents page by page.
    """

    def __init__(self, doc: fitz.Document, page_index: int):
        self._doc = doc
        self.page_index = page_index
        self._page: Optional[fitz.Page] = None

        # Lazy-loaded text layer
        self._text_layer: Optional[PageTextLayer] = None

        # Cached page info
        self._rect: Optional[fitz.Rect] = None

    @property
    def page(self) -> fitz.Page:
        """Get the underlying fitz page, loading if necessary."""
        if self._page is None:
            self._page = self._doc.load_page(self.page_index)
            self._rect = self._page.rect
        return self._page

    @property
    def rect(self) -> fitz.Rect:
        """Get page rectangle (dimensions)."""
        if self._rect is None:
            _ = self.page
        return self._rect

    @property
    def width(self) -> float:
        """Page width in points."""
        return self.rect.width

    @property
    def height(self) -> float:
        """Page height in points."""
        return self.rect.height

    @property
    def text_layer(self) -> PageTextLayer:
        """Get text layer, creating if necessary."""
        if self._text_layer is None:
            try:
                self._text_layer = PageTextLayer(self.page)
            except Exception as e:
                print(f"Failed to extract text for page {self.page_index}: {e}")
                # Return empty text layer on failure
                self._text_layer = PageTextLayer.__new__(PageTextLayer)
                self._text_layer.characters = []
                self._text_layer.blocks = []
                self._text_layer._char_grid = {}
        return self._text_layer

    @property
    def has_text(self) -> bool:
        """Check if page has extractable text."""
        return len(self.text_layer) > 0

    def render_to_image(self, scale: float = 1.5):
        """
        Render page to a PIL Image for layout detection.

        Args:
            scale: Resolution scale factor

        Returns:
            PIL.Image.Image or None
        """
        try:
            from PIL import Image

            mat = fitz.Matrix(scale, scale)
            pix = self.page.get_pixmap(matrix=mat, alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        except ImportError:
            print("PIL/Pillow is required for page image rendering.")
            return None
        except Exception as e:
            print(f"Error rendering page {self.page_index}: {e}")
            return None

    def unload(self):
        """Unload page data to free memory."""
        self._text_layer = None
        self._page = None

    def __repr__(self) -> str:
        return (
            f"PageModel(page={self.page_index}, "
            f"size={self.width:.0f}x{self.height:.0f})"
        )

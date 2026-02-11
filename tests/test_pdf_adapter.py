"""
Quick validation script for the PDF adapter (Task 1.5).

Usage:
    python -m tests.test_pdf_adapter path/to/test.pdf [page_index]

Runs all four adapter functions and prints diagnostics.
"""

import sys
import time
from pathlib import Path

from narration.utils.pdf_adapter import (
    PDFAdapter,
    extract_all_pages,
    extract_page_text_structure,
    get_page_count,
    get_page_dimensions,
    render_page_to_pil,
)


def test_single_functions(pdf_path: str, page_idx: int = 0):
    """Test the standalone (stateless) helper functions."""
    sep = "-" * 60

    # -- page count ---------------------------------------------------------
    count = get_page_count(pdf_path)
    print(f"Page count: {count}")
    assert count > 0, "Document has no pages"
    print(sep)

    # -- page dimensions ----------------------------------------------------
    w, h = get_page_dimensions(pdf_path, page_idx)
    print(f"Page {page_idx} dimensions: {w:.1f} × {h:.1f} pt")
    print(sep)

    # -- text structure -----------------------------------------------------
    t0 = time.perf_counter()
    blocks = extract_page_text_structure(pdf_path, page_idx)
    elapsed = time.perf_counter() - t0
    print(f"Page {page_idx}: {len(blocks)} text blocks  ({elapsed:.3f}s)")

    for i, b in enumerate(blocks):
        n_chars = len(b.all_characters)
        font = b.dominant_font_size
        bold = "BOLD" if b.is_bold else ""
        preview = b.text[:80].replace("\n", " ")
        print(f"  Block {i}: {n_chars:4d} chars, {font:.1f}pt {bold:>5s} | {preview}")
    print(sep)

    # -- render to PIL ------------------------------------------------------
    t0 = time.perf_counter()
    img = render_page_to_pil(pdf_path, page_idx, scale=1.5)
    elapsed = time.perf_counter() - t0
    print(f"Rendered page {page_idx}: {img.size[0]}×{img.size[1]} px  ({elapsed:.3f}s)")

    out_path = Path("debug") / f"page_{page_idx}_render.png"
    out_path.parent.mkdir(exist_ok=True)
    img.save(str(out_path))
    print(f"  Saved to {out_path}")
    print(sep)


def test_adapter_class(pdf_path: str):
    """Test the stateful PDFAdapter (keeps document open)."""
    sep = "-" * 60

    with PDFAdapter(pdf_path) as adapter:
        print(f"Adapter: {adapter}")
        print(sep)

        t0 = time.perf_counter()
        for idx in range(min(adapter.page_count, 5)):
            blocks = adapter.text_structure(idx)
            total_chars = sum(len(b.all_characters) for b in blocks)
            print(f"  Page {idx}: {len(blocks)} blocks, {total_chars} chars")
        elapsed = time.perf_counter() - t0
        pages_done = min(adapter.page_count, 5)
        print(f"  Processed {pages_done} pages in {elapsed:.3f}s")
    print(sep)


def test_extract_all(pdf_path: str):
    """Test extracting every page (can be slow for large docs)."""
    sep = "-" * 60
    count = get_page_count(pdf_path)
    if count > 20:
        print(f"Skipping extract_all_pages (document has {count} pages)")
        return

    t0 = time.perf_counter()
    all_blocks = extract_all_pages(pdf_path)
    elapsed = time.perf_counter() - t0

    total_blocks = sum(len(v) for v in all_blocks.values())
    total_chars = sum(len(b.all_characters) for bs in all_blocks.values() for b in bs)
    print(
        f"extract_all_pages: {count} pages, {total_blocks} blocks, "
        f"{total_chars} chars  ({elapsed:.3f}s)"
    )
    print(sep)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tests.test_pdf_adapter <pdf_path> [page_index]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Testing PDF adapter with: {pdf_file}")
    print(f"Target page: {page}")
    print("=" * 60)

    test_single_functions(pdf_file, page)
    test_adapter_class(pdf_file)
    test_extract_all(pdf_file)

    print("All tests passed.")

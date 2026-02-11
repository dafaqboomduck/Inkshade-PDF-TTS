"""
Smoke test for the narration POC core backend.
Run this first to confirm all imports work and text extraction
produces usable data without any Qt dependency.

Usage:
    python verify_core.py path/to/test.pdf
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_core.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # === Test 1: Imports work without Qt ===
    print("=" * 60)
    print("TEST 1: Import core modules (no Qt)")
    print("=" * 60)

    from core import PageModel, PageTextLayer, PDFDocumentReader
    from core.page.models import BlockInfo, CharacterInfo, LineInfo, SpanInfo

    print("  [PASS] All core modules imported successfully.\n")

    # === Test 2: Load PDF ===
    print("=" * 60)
    print("TEST 2: Load PDF document")
    print("=" * 60)

    reader = PDFDocumentReader()
    success, total_pages = reader.load_pdf(pdf_path)

    if not success:
        print("  [FAIL] Could not load PDF.")
        sys.exit(1)

    print(f"  [PASS] Loaded: {pdf_path}")
    print(f"         Pages: {total_pages}\n")

    # === Test 3: Extract text structure from first page ===
    print("=" * 60)
    print("TEST 3: Extract text structure (page 1)")
    print("=" * 60)

    page_model = PageModel(reader.doc, 0)
    text_layer = page_model.text_layer

    print(f"  Blocks:     {len(text_layer.blocks)}")
    print(f"  Characters: {len(text_layer.characters)}")
    print(f"  Median font size: {text_layer.median_font_size:.1f}")
    print()

    if not text_layer.blocks:
        print("  [WARN] No text blocks found. PDF may be scanned/image-only.\n")
    else:
        print("  [PASS] Text structure extracted.\n")

    # === Test 4: Inspect block metadata (what the narration pipeline uses) ===
    print("=" * 60)
    print("TEST 4: Block metadata inspection (first 5 blocks)")
    print("=" * 60)

    median = text_layer.median_font_size

    for i, block in enumerate(text_layer.blocks[:5]):
        size_ratio = block.dominant_font_size / median if median > 0 else 1.0
        text_preview = block.text.replace("\n", " ")[:60]

        print(f"  Block {i}:")
        print(
            f'    Text:       "{text_preview}{"..." if len(block.text) > 60 else ""}"'
        )
        print(
            f"    Font size:  {block.dominant_font_size:.1f} ({size_ratio:.2f}x median)"
        )
        print(f"    Bold:       {block.is_bold}")
        print(f"    Words:      {block.word_count}")
        print(f"    Lines:      {len(block.lines)}")
        print(
            f"    Bbox:       ({block.bbox[0]:.0f}, {block.bbox[1]:.0f}, "
            f"{block.bbox[2]:.0f}, {block.bbox[3]:.0f})"
        )
        print(f"    Fonts:      {block.all_font_names}")
        print()

    print("  [PASS] Block metadata accessible.\n")

    # === Test 5: Span-level flag inspection ===
    print("=" * 60)
    print("TEST 5: Span flags (first 10 spans)")
    print("=" * 60)

    span_count = 0
    for block in text_layer.blocks:
        for line in block.lines:
            for span in line.spans:
                if span_count >= 10:
                    break
                text_preview = span.text[:40]
                print(f'  Span: "{text_preview}{"..." if len(span.text) > 40 else ""}"')
                print(
                    f"    Font: {span.font_name}, Size: {span.font_size:.1f}, "
                    f"Bold: {span.is_bold}, Italic: {span.is_italic}, "
                    f"Mono: {span.is_monospaced}"
                )
                span_count += 1
            if span_count >= 10:
                break
        if span_count >= 10:
            break

    print(f"\n  [PASS] Span-level flags accessible.\n")

    # === Test 6: Page image rendering (for YOLO) ===
    print("=" * 60)
    print("TEST 6: Render page to PIL Image")
    print("=" * 60)

    img = page_model.render_to_image(scale=1.5)
    if img is not None:
        print(f"  [PASS] Rendered to PIL Image: {img.size[0]}x{img.size[1]} px")
    else:
        print("  [WARN] PIL rendering failed. Install Pillow: pip install Pillow")

    print()

    # === Test 7: Page dimensions ===
    print("=" * 60)
    print("TEST 7: Page dimensions")
    print("=" * 60)

    w, h = reader.get_page_size(0)
    print(f"  Page size (points): {w:.1f} x {h:.1f}")
    print(f"  Page size (inches): {w / 72:.2f} x {h / 72:.2f}")
    print(f"  [PASS]\n")

    # === Test 8: Multi-page iteration ===
    print("=" * 60)
    print(f"TEST 8: Iterate all {total_pages} pages")
    print("=" * 60)

    total_blocks = 0
    total_chars = 0
    empty_pages = 0

    for page_idx in range(total_pages):
        pm = PageModel(reader.doc, page_idx)
        tl = pm.text_layer
        total_blocks += len(tl.blocks)
        total_chars += len(tl.characters)
        if len(tl.blocks) == 0:
            empty_pages += 1
        pm.unload()  # Free memory

    print(f"  Total blocks:     {total_blocks}")
    print(f"  Total characters: {total_chars}")
    print(f"  Empty pages:      {empty_pages}")
    print(f"  [PASS]\n")

    # === Cleanup ===
    reader.close_document()

    print("=" * 60)
    print("ALL TESTS PASSED — Core backend is ready for narration pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()

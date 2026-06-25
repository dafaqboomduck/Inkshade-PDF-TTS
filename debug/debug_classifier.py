"""
Visual debug for the full classification pipeline (Task 3.4/3.5).

Runs YOLO detection + block matching + feature refinement on PDF pages
and saves colour-coded overlays showing the final label per text block.

Usage:
    python -m debug.debug_classifier path/to/test.pdf [--pages 0-9] [--confidence 0.35]
"""

import argparse
import time
from pathlib import Path

from narration.layout.classifier import classify_page
from narration.layout.detector import LayoutDetector
from narration.layout.feature_refiner import detect_running_headers_footers
from narration.layout.visualize import draw_classified_blocks
from narration.utils.pdf_adapter import PDFAdapter


def run(pdf_path: str, page_start: int, page_end: int, confidence: float):
    out_dir = Path("debug") / "classified"
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = 1.5
    detector = LayoutDetector()

    with PDFAdapter(pdf_path) as pdf:
        page_end = min(page_end, pdf.page_count - 1)
        print(f"Classifying pages {page_start}–{page_end} of {pdf.page_count}\n")

        all_classified = {}
        page_heights = {}

        for idx in range(page_start, page_end + 1):
            t0 = time.perf_counter()
            img = pdf.render(idx, scale=scale)
            blocks = pdf.text_structure(idx)
            w, h = pdf.dimensions(idx)
            page_heights[idx] = h

            classified = classify_page(
                detector,
                img,
                blocks,
                w,
                h,
                scale=scale,
                confidence=confidence,
            )
            all_classified[idx] = classified
            elapsed = time.perf_counter() - t0

            # Summary
            print(f"--- Page {idx} ({elapsed:.2f}s) ---")
            label_counts = {}
            for cb in classified:
                label_counts[cb.label.name] = label_counts.get(cb.label.name, 0) + 1
            print(f"    Blocks: {len(classified)}  |  {label_counts}")
            for cb in classified:
                preview = cb.text[:70].replace("\n", " ")
                ref = " [refined]" if cb.refined else ""
                print(
                    f"    {cb.label.name:16s} {cb.confidence:.0%}{ref:10s} | {preview}"
                )
            print()

        # Cross-page header/footer detection
        if len(all_classified) >= 3:
            detect_running_headers_footers(all_classified, page_heights)
            print("[Running header/footer detection applied]\n")

        # Save debug images
        for idx in range(page_start, page_end + 1):
            img = pdf.render(idx, scale=scale)
            annotated = draw_classified_blocks(img, all_classified[idx], scale)
            out_path = out_dir / f"page_{idx:03d}.png"
            annotated.save(str(out_path))
            print(f"Saved {out_path}")

    print(f"\nDone. Check {out_dir}/ for annotated images.")


def main():
    parser = argparse.ArgumentParser(
        description="Test full block classification pipeline"
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--pages", default="0-4", help="Page range, e.g. 0-9 (default: 0-4)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="YOLO confidence threshold (default: 0.35)",
    )
    args = parser.parse_args()

    parts = args.pages.split("-")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start

    run(args.pdf, start, end, args.confidence)


if __name__ == "__main__":
    main()

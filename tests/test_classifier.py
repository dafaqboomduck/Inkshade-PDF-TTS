"""
Visual debug for the full classification pipeline (Task 3.4/3.5).

Runs YOLO detection + block matching + feature refinement on PDF pages
and saves colour-coded overlays showing the final label per text block.

Usage:
    python -m tests.test_classifier path/to/test.pdf [--pages 0-9] [--confidence 0.35]
"""

import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from narration.layout.classifier import classify_page
from narration.layout.detector import LayoutDetector
from narration.layout.feature_refiner import detect_running_headers_footers
from narration.layout.models import ClassifiedBlock, LayoutLabel
from narration.utils.pdf_adapter import PDFAdapter

LABEL_COLORS = {
    LayoutLabel.TITLE: (220, 40, 40),
    LayoutLabel.SECTION_HEADER: (230, 130, 20),
    LayoutLabel.TEXT: (50, 160, 50),
    LayoutLabel.LIST_ITEM: (50, 180, 130),
    LayoutLabel.CAPTION: (100, 100, 220),
    LayoutLabel.FOOTNOTE: (160, 100, 200),
    LayoutLabel.FORMULA: (200, 180, 50),
    LayoutLabel.TABLE: (50, 130, 200),
    LayoutLabel.PICTURE: (180, 180, 180),
    LayoutLabel.PAGE_HEADER: (180, 120, 100),
    LayoutLabel.PAGE_FOOTER: (140, 140, 140),
    LayoutLabel.UNKNOWN: (255, 255, 255),
}

# Labels that the narration will skip
SKIP_LABELS = {
    LayoutLabel.PAGE_HEADER,
    LayoutLabel.PAGE_FOOTER,
    LayoutLabel.PICTURE,
    LayoutLabel.TABLE,
    LayoutLabel.FORMULA,
}


def draw_classified_blocks(
    image: Image.Image,
    classified: list[ClassifiedBlock],
    scale: float,
    line_width: int = 2,
) -> Image.Image:
    """
    Draw colour-coded boxes around each text block using the
    *classified* label.  Skipped blocks get a dashed outline + ×.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for cb in classified:
        color = LABEL_COLORS.get(cb.label, (255, 255, 255))
        # Convert block bbox (PDF pts) → pixel coords
        x0 = cb.bbox[0] * scale
        y0 = cb.bbox[1] * scale
        x1 = cb.bbox[2] * scale
        y1 = cb.bbox[3] * scale

        is_skip = cb.label in SKIP_LABELS

        # Semi-transparent fill
        alpha = 35 if not is_skip else 20
        draw.rectangle([x0, y0, x1, y1], fill=(*color, alpha))

        # Border
        for i in range(line_width):
            draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)

        # Label tag
        ref = " ®" if cb.refined else ""
        skip = " SKIP" if is_skip else ""
        tag = f"{cb.label.name} {cb.confidence:.0%}{ref}{skip}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        draw.rectangle([x0, y0 - th - 4, x0 + tw + 6, y0], fill=(*color, 220))
        draw.text((x0 + 3, y0 - th - 2), tag, fill=(255, 255, 255), font=font)

    return img.convert("RGB")


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

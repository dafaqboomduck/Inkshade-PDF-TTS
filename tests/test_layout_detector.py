"""
Visual test for the layout detector (Task 2.4).

Runs YOLO detection on one or more PDF pages, prints results, and
saves annotated debug images with color-coded bounding boxes.

Usage:
    python -m tests.test_layout_detector path/to/test.pdf [--pages 0-4] [--confidence 0.35]
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from narration.layout.detector import LayoutDetector
from narration.layout.models import LayoutLabel, LayoutRegion
from narration.utils.pdf_adapter import PDFAdapter

# Distinct colours per label for the debug overlay
LABEL_COLORS = {
    LayoutLabel.TITLE: (220, 40, 40),  # red
    LayoutLabel.SECTION_HEADER: (230, 130, 20),  # orange
    LayoutLabel.TEXT: (50, 160, 50),  # green
    LayoutLabel.LIST_ITEM: (50, 180, 130),  # teal
    LayoutLabel.CAPTION: (100, 100, 220),  # blue
    LayoutLabel.FOOTNOTE: (160, 100, 200),  # purple
    LayoutLabel.FORMULA: (200, 180, 50),  # yellow
    LayoutLabel.TABLE: (50, 130, 200),  # sky blue
    LayoutLabel.PICTURE: (180, 180, 180),  # grey
    LayoutLabel.PAGE_HEADER: (180, 120, 100),  # brown
    LayoutLabel.PAGE_FOOTER: (140, 140, 140),  # dark grey
    LayoutLabel.UNKNOWN: (255, 255, 255),  # white
}


def draw_regions(
    image: Image.Image,
    regions: list[LayoutRegion],
    line_width: int = 3,
) -> Image.Image:
    """Draw colour-coded bounding boxes + labels on a copy of *image*."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for r in regions:
        color = LABEL_COLORS.get(r.label, (255, 255, 255))
        x0, y0, x1, y1 = r.bbox

        # Box
        for i in range(line_width):
            draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)

        # Label tag
        tag = f"{r.label.name} {r.confidence:.0%}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        draw.rectangle([x0, y0 - th - 4, x0 + tw + 6, y0], fill=color)
        draw.text((x0 + 3, y0 - th - 2), tag, fill=(255, 255, 255), font=font)

    return img


def run(pdf_path: str, page_start: int, page_end: int, confidence: float):
    out_dir = Path("debug") / "layout"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = LayoutDetector()
    print(f"Detector: {detector}")

    scale = 1.5

    with PDFAdapter(pdf_path) as pdf:
        page_end = min(page_end, pdf.page_count - 1)
        print(f"Processing pages {page_start}–{page_end} of {pdf.page_count}\n")

        for idx in range(page_start, page_end + 1):
            t0 = time.perf_counter()
            img = pdf.render(idx, scale=scale)
            t_render = time.perf_counter() - t0

            t0 = time.perf_counter()
            regions = detector.detect(img, confidence=confidence)
            t_detect = time.perf_counter() - t0

            # Print results
            print(f"--- Page {idx} ({img.size[0]}×{img.size[1]} px) ---")
            print(
                f"    render {t_render:.3f}s | detect {t_detect:.3f}s | "
                f"{len(regions)} regions"
            )
            for r in regions:
                print(f"    {r}")

            # Save annotated image
            annotated = draw_regions(img, regions)
            out_path = out_dir / f"page_{idx:03d}.png"
            annotated.save(str(out_path))
            print(f"    → {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Test layout detection on a PDF")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--pages", default="0-4", help="Page range, e.g. 0-4 (default: 0-4)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence (default: 0.35)",
    )
    args = parser.parse_args()

    parts = args.pages.split("-")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start

    run(args.pdf, start, end, args.confidence)
    print("Done. Check debug/layout/ for annotated images.")


if __name__ == "__main__":
    main()

"""
Test the reading script builder (Task 4.5 / 4.6).

Runs the full pipeline up to script generation and prints a
human-readable preview — no audio synthesis.

Usage:
    python -m tests.test_reading_script path/to/test.pdf [--pages 0-4]
"""

import argparse
import time

from narration.layout.classifier import classify_document
from narration.layout.detector import LayoutDetector
from narration.script.models import TextRole
from narration.script.reading_script import (
    build_document_script,
    preview_script,
)
from narration.utils.pdf_adapter import PDFAdapter


def run(pdf_path: str, page_start: int, page_end: int):
    detector = LayoutDetector()

    with PDFAdapter(pdf_path) as pdf:
        page_end = min(page_end, pdf.page_count - 1)
        print(f"Processing pages {page_start}–{page_end} of {pdf.page_count}\n")

        # Classify
        t0 = time.perf_counter()
        all_classified = classify_document(
            detector,
            pdf,
            page_range=(page_start, page_end),
        )
        t_classify = time.perf_counter() - t0

        # Build script
        t0 = time.perf_counter()
        script = build_document_script(all_classified)
        t_script = time.perf_counter() - t0

    # Preview
    print(preview_script(script))

    # Summary
    print("\n" + "=" * 60)
    total = len(script)
    spoken = sum(1 for s in script if not s.should_skip and s.text)
    skipped_transitions = sum(1 for s in script if s.role == TextRole.PAGE_TRANSITION)
    print(f"Classification: {t_classify:.2f}s")
    print(f"Script build:   {t_script:.4f}s")
    print(
        f"Instructions:   {total} total, {spoken} spoken, "
        f"{skipped_transitions} page transitions"
    )

    # Word count estimate
    words = sum(len(s.text.split()) for s in script if s.text)
    est_minutes = words / 150  # ~150 wpm average narration
    print(f"Word count:     ~{words} words")
    print(f"Est. duration:  ~{est_minutes:.1f} minutes (at 150 wpm)")


def main():
    parser = argparse.ArgumentParser(description="Preview the reading script for a PDF")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--pages", default="0-4", help="Page range, e.g. 0-9 (default: 0-4)"
    )
    args = parser.parse_args()

    parts = args.pages.split("-")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start

    run(args.pdf, start, end)


if __name__ == "__main__":
    main()

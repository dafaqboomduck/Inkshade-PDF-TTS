#!/usr/bin/env python3
"""
Inkshade PDF Narration — CLI entry point.

Converts a PDF document into a narrated MP3 (or WAV) audio file using
ML-based layout detection and prosody-aware TTS synthesis.

Usage:
    python narrate.py input.pdf output.mp3 [options]

Examples:
    python narrate.py paper.pdf paper.mp3
    python narrate.py book.pdf chapter1.mp3 --pages 1-12 --speed 1.1
    python narrate.py report.pdf --debug-script --pages 1-5
    python narrate.py paper.pdf --debug-layout debug/paper/
"""

import argparse
import sys
from pathlib import Path

from narration.pipeline import NarrationConfig, NarrationPipeline


def parse_page_range(s: str):
    """Parse a page range string like '1-10' into a 0-based (start, end) tuple."""
    parts = s.split("-")
    try:
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid page range '{s}'. Use N or N-M (1-based, e.g. 1-10)."
        )
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(
            f"Invalid page range '{s}'. Start must be >= 1 and end >= start."
        )
    # Convert to 0-based
    return (start - 1, end - 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a PDF into narrated audio using layout-aware TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python narrate.py paper.pdf paper.mp3\n"
            "  python narrate.py book.pdf ch1.mp3 --pages 1-12 --speed 1.1\n"
            "  python narrate.py report.pdf --debug-script --pages 1-5\n"
            "  python narrate.py paper.pdf --debug-layout debug/paper/\n"
        ),
    )

    # Positional
    p.add_argument("input", help="Path to the input PDF file")
    p.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output audio file path (.mp3 or .wav). "
        "Optional when using --debug-script or --debug-layout.",
    )

    # Voice
    voice = p.add_argument_group("voice")
    voice.add_argument(
        "--voice",
        default="en_US-lessac-medium",
        help="Piper voice name (default: en_US-lessac-medium)",
    )
    voice.add_argument(
        "--list-voices",
        action="store_true",
        help="List available and downloadable voices, then exit",
    )

    # Page range
    p.add_argument(
        "--pages",
        type=parse_page_range,
        default=None,
        metavar="N-M",
        help="Page range (1-based, inclusive). E.g. 1-10. Default: all pages.",
    )

    # Prosody
    prosody = p.add_argument_group("prosody")
    prosody.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Global speed multiplier (default: 1.0)",
    )
    prosody.add_argument(
        "--pause-scale",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Pause duration multiplier (default: 1.0)",
    )

    # Content
    content = p.add_argument_group("content")
    content.add_argument(
        "--skip-footnotes",
        action="store_true",
        default=True,
        help="Skip footnotes (default: on)",
    )
    content.add_argument(
        "--no-skip-footnotes",
        action="store_true",
        help="Read footnotes aloud",
    )
    content.add_argument(
        "--skip-captions",
        action="store_true",
        default=False,
        help="Skip figure/table captions",
    )
    content.add_argument(
        "--keep-references",
        action="store_true",
        default=False,
        help="Keep [1]-style citation markers in narrated text",
    )
    content.add_argument(
        "--announce-pages",
        action="store_true",
        default=False,
        help='Insert "Page N" announcements between pages',
    )

    # Audio
    audio = p.add_argument_group("audio")
    audio.add_argument(
        "--output-wav",
        action="store_true",
        help="Export as WAV instead of MP3",
    )
    audio.add_argument(
        "--bitrate",
        default="192k",
        help="MP3 bitrate (default: 192k)",
    )

    # Layout model
    model = p.add_argument_group("model")
    model.add_argument(
        "--yolo-model",
        default=None,
        metavar="PATH",
        help="Path to YOLO .pt weights (default: models/yolov8x_doclaynet.pt)",
    )
    model.add_argument(
        "--yolo-device",
        default=None,
        metavar="DEVICE",
        help='Force device for YOLO (e.g. "cpu", "cuda:0")',
    )
    model.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        metavar="FLOAT",
        help="YOLO detection confidence threshold (default: 0.35)",
    )

    # Debug
    debug = p.add_argument_group("debug")
    debug.add_argument(
        "--debug-layout",
        default=None,
        metavar="DIR",
        help="Save colour-coded layout debug images to DIR (no audio produced)",
    )
    debug.add_argument(
        "--debug-script",
        action="store_true",
        help="Print the reading script preview without generating audio",
    )

    return p


def cmd_list_voices(voice_name: str) -> None:
    """Print available and known voices."""
    from narration.tts.model_manager import ModelManager

    mgr = ModelManager()
    cached = mgr.list_available_voices()
    known = mgr.list_known_voices()

    print("Cached voices (ready to use):")
    if cached:
        for v in cached:
            print(f"  {v}")
    else:
        print("  (none)")

    print("\nDownloadable voices:")
    for v in known:
        tag = " [cached]" if v in cached else ""
        print(f"  {v}{tag}")

    print(f"\nUse --voice NAME to select. Default: {voice_name}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --list-voices
    if args.list_voices:
        cmd_list_voices(args.voice)
        return

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    if not input_path.suffix.lower() == ".pdf":
        parser.error(f"Input must be a PDF file: {input_path}")

    # Determine mode
    debug_only = args.debug_script or (args.debug_layout and args.output is None)

    # Resolve output path
    if args.output:
        output_path = args.output
    elif debug_only:
        # No output needed for debug-only modes
        output_path = None
    else:
        # Default: same name as input, with .mp3/.wav extension
        ext = ".wav" if args.output_wav else ".mp3"
        output_path = str(input_path.with_suffix(ext))

    # Force .wav extension if --output-wav
    if args.output_wav and output_path and not output_path.lower().endswith(".wav"):
        output_path = str(Path(output_path).with_suffix(".wav"))

    # For non-debug modes we need an output path
    if not debug_only and not output_path:
        parser.error(
            "Output path is required unless using --debug-script or --debug-layout."
        )

    # Handle --no-skip-footnotes overriding --skip-footnotes
    skip_fn = args.skip_footnotes and not args.no_skip_footnotes

    # Build config
    config = NarrationConfig(
        voice_name=args.voice,
        yolo_model_path=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_confidence=args.confidence,
        speed_multiplier=args.speed,
        pause_multiplier=args.pause_scale,
        skip_footnotes=skip_fn,
        skip_captions=args.skip_captions,
        strip_references=not args.keep_references,
        announce_pages=args.announce_pages,
        page_range=args.pages,
        mp3_bitrate=args.bitrate,
        debug_layout_dir=args.debug_layout,
        debug_script=args.debug_script,
    )

    # Print header
    print(f"Inkshade PDF Narration")
    print(f"  Input:  {input_path}")
    if output_path:
        print(f"  Output: {output_path}")
    if config.page_range:
        s, e = config.page_range
        print(f"  Pages:  {s + 1}–{e + 1}")
    print(f"  Voice:  {config.voice_name}")
    if config.speed_multiplier != 1.0:
        print(f"  Speed:  {config.speed_multiplier}x")
    if config.pause_multiplier != 1.0:
        print(f"  Pauses: {config.pause_multiplier}x")
    mode_tags = []
    if args.debug_script:
        mode_tags.append("script-preview")
    if args.debug_layout:
        mode_tags.append(f"layout-debug → {args.debug_layout}")
    if mode_tags:
        print(f"  Mode:   {', '.join(mode_tags)}")
    print()

    # Run
    pipeline = NarrationPipeline(config)

    if debug_only and not output_path:
        # For debug-only: use a throwaway path since we won't export
        output_path = "/dev/null"

    result = pipeline.narrate(str(input_path), output_path)

    # Exit code
    if result.spoken_instructions == 0 and not debug_only:
        print("\n[WARN] No spoken content was produced.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

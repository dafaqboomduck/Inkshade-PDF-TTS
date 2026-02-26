#!/usr/bin/env python3
"""
Inkshade PDF Narration — CLI entry point.

Converts a PDF document into a narrated MP3 (or WAV) audio file using
ML-based layout detection and Kokoro neural TTS synthesis.

Usage::

    python narrate.py input.pdf output.mp3
    python narrate.py book.pdf ch1.mp3 --pages 1-12 --speed 1.1
    python narrate.py input.pdf out.mp3 --voice af_heart --lang a
    python narrate.py report.pdf --debug-script --pages 1-5
    python narrate.py paper.pdf --debug-layout debug/paper/ -v 2

Verbosity levels::

    -v 0   Quiet — warnings and errors only.
    -v 1   Normal — phase summaries and progress bars (default).
    -v 2   Debug — per-segment synthesis detail, all internal decisions.
"""

import argparse
import logging
import sys
from pathlib import Path

from narration.pipeline import NarrationConfig, NarrationPipeline

logger = logging.getLogger("narration")

# Mapping from --verbose integer to logging level
_VERBOSITY_MAP = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------


def _parse_page_range(value: str):
    """
    Parse a 1-based page range string (e.g. ``"3-10"``) into a
    0-based ``(start, end)`` tuple.

    Raises:
        argparse.ArgumentTypeError: On malformed input.
    """
    parts = value.split("-")
    try:
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid page range '{value}'. Use N or N-M (1-based)."
        )
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(
            f"Invalid page range '{value}'. Start must be >= 1 and end >= start."
        )
    return (start - 1, end - 1)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all pipeline options."""
    p = argparse.ArgumentParser(
        description="Convert a PDF into narrated audio using layout-aware TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python narrate.py paper.pdf paper.mp3\n"
            "  python narrate.py book.pdf ch1.mp3 --pages 1-12 --speed 1.1\n"
            "  python narrate.py report.pdf --debug-script --pages 1-5\n"
            "  python narrate.py paper.pdf --debug-layout debug/ -v 2\n"
        ),
    )

    # -- Positional --------------------------------------------------------
    p.add_argument("input", help="Path to the input PDF file")
    p.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output audio file (.mp3 or .wav). "
        "Optional when using --debug-script or --debug-layout.",
    )

    # -- Voice -------------------------------------------------------------
    voice = p.add_argument_group("voice")
    voice.add_argument(
        "--voice",
        default="af_heart",
        help="Kokoro voice ID (default: af_heart). Use --list-voices to see all.",
    )
    voice.add_argument(
        "--lang",
        default="a",
        choices=["a", "b"],
        help="Language code: 'a' American English, 'b' British English (default: a)",
    )
    voice.add_argument(
        "--list-voices",
        action="store_true",
        help="List available Kokoro voices, then exit",
    )

    # -- Pages -------------------------------------------------------------
    p.add_argument(
        "--pages",
        type=_parse_page_range,
        default=None,
        metavar="N-M",
        help="Page range, 1-based inclusive (e.g. 1-10). Default: all.",
    )

    # -- Prosody -----------------------------------------------------------
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

    # -- Content -----------------------------------------------------------
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
        help="Keep [N]-style citation markers in narrated text",
    )
    content.add_argument(
        "--announce-pages",
        action="store_true",
        default=False,
        help='Insert "Page N" announcements between pages',
    )

    # -- Audio -------------------------------------------------------------
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

    # -- Layout model ------------------------------------------------------
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
        help='Force YOLO device (e.g. "cpu", "cuda:0")',
    )
    model.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        metavar="FLOAT",
        help="YOLO detection confidence threshold (default: 0.35)",
    )

    # -- Debug / output control --------------------------------------------
    debug = p.add_argument_group("debug & output")
    debug.add_argument(
        "--debug-layout",
        default=None,
        metavar="DIR",
        help="Save colour-coded layout images to DIR (no audio produced)",
    )
    debug.add_argument(
        "--debug-script",
        action="store_true",
        help="Print the reading script without generating audio",
    )
    debug.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity: 0=quiet, 1=normal (default), 2=debug",
    )
    debug.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )

    return p


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------


def _configure_logging(verbosity: int, disable_tqdm: bool) -> None:
    """
    Set up the root ``narration`` logger.

    At verbosity 0 (WARNING), uses a minimal format. At 1+ (INFO /
    DEBUG), includes the module name for traceability.  When tqdm is
    active, logs are routed through :class:`tqdm.contrib.logging`
    context or a compatible handler so progress bars aren't corrupted.
    """
    level = _VERBOSITY_MAP.get(verbosity, logging.INFO)

    if level <= logging.DEBUG:
        fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
        datefmt = "%H:%M:%S"
    elif level <= logging.INFO:
        fmt = "%(message)s"
        datefmt = None
    else:
        fmt = "%(levelname)s: %(message)s"
        datefmt = None

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger("narration")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers regardless of verbosity
    for name in ("ultralytics", "kokoro", "PIL", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------


def _cmd_list_voices() -> None:
    """Print available Kokoro voices, then exit."""
    from narration.tts.model_manager import KOKORO_VOICES, ModelManager

    logger.info("Available Kokoro voices (auto-download on first use):")
    logger.info("")
    logger.info("  %-15s %-10s %-8s %s", "ID", "ACCENT", "GENDER", "NAME")
    logger.info("  %-15s %-10s %-8s %s", "-" * 15, "-" * 10, "-" * 8, "-" * 10)
    for voice_id, info in KOKORO_VOICES.items():
        logger.info(
            "  %-15s %-10s %-8s %s",
            voice_id,
            info["accent"],
            info["gender"],
            info["name"],
        )
    logger.info("")
    logger.info("Use --voice ID to select. Default: af_heart")
    logger.info("Use --lang a/b for American/British accent.")


# ------------------------------------------------------------------
# Output path resolution
# ------------------------------------------------------------------


def _resolve_output_path(args: argparse.Namespace, debug_only: bool) -> str | None:
    """
    Determine the output file path from CLI arguments.

    Returns ``None`` for debug-only runs that don't produce audio.
    """
    if args.output:
        path = args.output
    elif debug_only:
        return None
    else:
        ext = ".wav" if args.output_wav else ".mp3"
        path = str(Path(args.input).with_suffix(ext))

    if args.output_wav and not path.lower().endswith(".wav"):
        path = str(Path(path).with_suffix(".wav"))

    return path


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    """Parse arguments, configure logging, and run the pipeline."""
    parser = _build_parser()
    args = parser.parse_args()

    # Logging must be configured before any logger calls
    disable_tqdm = args.no_progress or args.verbose == 0
    _configure_logging(args.verbose, disable_tqdm)

    # --list-voices exits early
    if args.list_voices:
        _cmd_list_voices()
        return

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        parser.error(f"Input must be a PDF file: {input_path}")

    # Determine run mode
    debug_only = args.debug_script or (args.debug_layout and args.output is None)

    output_path = _resolve_output_path(args, debug_only)
    if not debug_only and not output_path:
        parser.error(
            "Output path is required unless using --debug-script or --debug-layout."
        )

    skip_footnotes = args.skip_footnotes and not args.no_skip_footnotes

    # Build pipeline config
    config = NarrationConfig(
        kokoro_voice=args.voice,
        kokoro_lang_code=args.lang,
        yolo_model_path=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_confidence=args.confidence,
        speed_multiplier=args.speed,
        pause_multiplier=args.pause_scale,
        skip_footnotes=skip_footnotes,
        skip_captions=args.skip_captions,
        strip_references=not args.keep_references,
        announce_pages=args.announce_pages,
        page_range=args.pages,
        mp3_bitrate=args.bitrate,
        debug_layout_dir=args.debug_layout,
        debug_script=args.debug_script,
        disable_tqdm=disable_tqdm,
    )

    # Log run header
    logger.info("Inkshade PDF Narration")
    logger.info("  Input:  %s", input_path)
    if output_path:
        logger.info("  Output: %s", output_path)
    if config.page_range:
        s, e = config.page_range
        logger.info("  Pages:  %d–%d", s + 1, e + 1)
    logger.info("  Voice:  %s (lang=%s)", config.kokoro_voice, config.kokoro_lang_code)
    if config.speed_multiplier != 1.0:
        logger.info("  Speed:  %.2fx", config.speed_multiplier)
    if config.pause_multiplier != 1.0:
        logger.info("  Pauses: %.2fx", config.pause_multiplier)

    mode_tags = []
    if args.debug_script:
        mode_tags.append("script-preview")
    if args.debug_layout:
        mode_tags.append(f"layout-debug → {args.debug_layout}")
    if mode_tags:
        logger.info("  Mode:   %s", ", ".join(mode_tags))

    # Run pipeline
    pipeline = NarrationPipeline(config)

    # Debug-only runs still need a path for the pipeline signature;
    # use /dev/null as a no-op destination.
    effective_output = output_path or "/dev/null"
    result = pipeline.narrate(str(input_path), effective_output)

    if result.spoken_instructions == 0 and not debug_only:
        logger.warning("No spoken content was produced")
        sys.exit(1)


if __name__ == "__main__":
    main()

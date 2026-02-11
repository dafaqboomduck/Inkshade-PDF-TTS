"""
Narration pipeline orchestrator: PDF → analysis → script → audio → MP3.

Wires together layout detection, block classification, reading script
generation, TTS synthesis, and audio assembly into a single entry point.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from narration.layout.classifier import classify_document, classify_page
from narration.layout.detector import LayoutDetector
from narration.layout.feature_refiner import detect_running_headers_footers
from narration.layout.models import ClassifiedBlock
from narration.script.models import ReadingInstruction, TextRole
from narration.script.reading_script import (
    build_document_script,
    build_page_script,
    preview_script,
)
from narration.tts.audio_builder import AudioBuilder
from narration.tts.engine import TTSEngine
from narration.tts.model_manager import ModelManager
from narration.utils.pdf_adapter import PDFAdapter

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class NarrationConfig:
    """All tuneable knobs for the narration pipeline."""

    # Voice
    voice_name: str = "en_US-lessac-medium"
    voice_dir: Optional[Path] = None

    # Layout detection
    yolo_model_path: Optional[str] = None
    yolo_device: Optional[str] = None
    yolo_confidence: float = 0.35
    render_scale: float = 1.5

    # Script options
    speed_multiplier: float = 1.0
    pause_multiplier: float = 1.0
    skip_footnotes: bool = True
    skip_captions: bool = False
    strip_references: bool = True
    announce_pages: bool = False
    page_transition_pause: float = 1.0

    # Audio export
    mp3_bitrate: str = "192k"
    normalize_dBFS: float = -20.0
    crossfade_ms: int = 30

    # Page range (None = all pages)
    page_range: Optional[Tuple[int, int]] = None

    # Debug
    debug_layout_dir: Optional[str] = None
    debug_script: bool = False


# ------------------------------------------------------------------
# Pipeline result
# ------------------------------------------------------------------


@dataclass
class NarrationResult:
    """Summary returned after narration completes."""

    output_path: str = ""
    total_pages: int = 0
    pages_processed: int = 0
    total_instructions: int = 0
    spoken_instructions: int = 0
    skipped_blocks: int = 0
    word_count: int = 0
    audio_duration: float = 0.0
    file_size_mb: float = 0.0
    elapsed_seconds: float = 0.0

    # Per-phase timing
    time_layout: float = 0.0
    time_script: float = 0.0
    time_synthesis: float = 0.0
    time_export: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "NARRATION COMPLETE",
            "=" * 60,
            f"  Output:       {self.output_path}",
            f"  Pages:        {self.pages_processed} / {self.total_pages}",
            f"  Instructions: {self.spoken_instructions} spoken, "
            f"{self.skipped_blocks} skipped",
            f"  Words:        ~{self.word_count}",
            f"  Duration:     {self.audio_duration:.1f}s "
            f"({self.audio_duration / 60:.1f} min)",
            f"  File size:    {self.file_size_mb:.1f} MB",
            "",
            f"  Layout detection: {self.time_layout:.1f}s",
            f"  Script building:  {self.time_script:.2f}s",
            f"  TTS synthesis:    {self.time_synthesis:.1f}s",
            f"  Audio export:     {self.time_export:.2f}s",
            f"  Total wall time:  {self.elapsed_seconds:.1f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------


class NarrationPipeline:
    """
    End-to-end PDF narration pipeline.

    Usage::

        pipeline = NarrationPipeline(NarrationConfig(voice_name="en_US-lessac-medium"))
        result = pipeline.narrate("input.pdf", "output.mp3")
        print(result.summary())
    """

    def __init__(self, config: Optional[NarrationConfig] = None):
        self.config = config or NarrationConfig()
        self._detector: Optional[LayoutDetector] = None
        self._engine: Optional[TTSEngine] = None
        self._model_mgr: Optional[ModelManager] = None

    # ------------------------------------------------------------------
    # Lazy component initialisation
    # ------------------------------------------------------------------

    def _ensure_detector(self) -> LayoutDetector:
        if self._detector is None:
            print("Loading layout detector...")
            self._detector = LayoutDetector(
                model_path=self.config.yolo_model_path,
                device=self.config.yolo_device,
            )
            print(f"  {self._detector}")
        return self._detector

    def _ensure_tts(self) -> TTSEngine:
        if self._engine is None:
            self._model_mgr = ModelManager(
                voice_dir=self.config.voice_dir,
            )
            voice_path = self._model_mgr.ensure_voice_available(
                self.config.voice_name,
            )
            print(f"Loading TTS engine: {self.config.voice_name}")
            self._engine = TTSEngine(voice_path)
            print(f"  {self._engine}")
        return self._engine

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def narrate(
        self,
        pdf_path: str,
        output_path: str,
    ) -> NarrationResult:
        """
        Narrate a PDF document and export as audio.

        Args:
            pdf_path:    Path to the input PDF.
            output_path: Destination audio file (.mp3 or .wav).

        Returns:
            NarrationResult with timing and size info.
        """
        t_total_start = time.perf_counter()
        cfg = self.config
        result = NarrationResult(output_path=output_path)
        is_wav = output_path.lower().endswith(".wav")

        # --- Phase 0: Initialise components ---
        detector = self._ensure_detector()
        engine = self._ensure_tts()
        builder = AudioBuilder(sample_rate=engine.sample_rate)

        # --- Phase 1: Layout detection & classification ---
        print("\n--- Phase 1: Layout detection & classification ---")
        t0 = time.perf_counter()

        with PDFAdapter(pdf_path) as pdf:
            result.total_pages = pdf.page_count
            start = cfg.page_range[0] if cfg.page_range else 0
            end = cfg.page_range[1] if cfg.page_range else pdf.page_count - 1
            end = min(end, pdf.page_count - 1)

            all_classified: Dict[int, List[ClassifiedBlock]] = {}
            page_heights: Dict[int, float] = {}

            for idx in range(start, end + 1):
                t_page = time.perf_counter()
                print(
                    f"  Processing page {idx + 1}/{pdf.page_count}...",
                    end=" ",
                    flush=True,
                )

                try:
                    img = pdf.render(idx, scale=cfg.render_scale)
                    blocks = pdf.text_structure(idx)
                    w, h = pdf.dimensions(idx)
                    page_heights[idx] = h

                    if not blocks:
                        print("(no text — skipped)")
                        all_classified[idx] = []
                        continue

                    classified = classify_page(
                        detector,
                        img,
                        blocks,
                        w,
                        h,
                        scale=cfg.render_scale,
                        confidence=cfg.yolo_confidence,
                    )
                    all_classified[idx] = classified

                    elapsed_page = time.perf_counter() - t_page
                    print(f"{len(classified)} blocks, {elapsed_page:.2f}s")

                except Exception as e:
                    print(f"FAILED: {e}")
                    # Fallback: use feature-only classification
                    try:
                        blocks = pdf.text_structure(idx)
                        w, h = pdf.dimensions(idx)
                        page_heights[idx] = h
                        all_classified[idx] = _fallback_classify(
                            blocks,
                            w,
                            h,
                        )
                        print(f"  (fallback classification for page {idx + 1})")
                    except Exception:
                        all_classified[idx] = []

            # Cross-page header/footer detection
            if len(all_classified) >= 3:
                detect_running_headers_footers(all_classified, page_heights)
                print("  [Running header/footer detection applied]")

            # Save debug layout images if requested
            if cfg.debug_layout_dir:
                _save_debug_layouts(
                    pdf,
                    all_classified,
                    cfg.render_scale,
                    cfg.debug_layout_dir,
                )

        result.pages_processed = end - start + 1
        result.time_layout = time.perf_counter() - t0
        print(f"  Layout done in {result.time_layout:.1f}s")

        # --- Phase 2: Build reading script ---
        print("\n--- Phase 2: Building reading script ---")
        t0 = time.perf_counter()

        script = build_document_script(
            all_classified,
            speed_multiplier=cfg.speed_multiplier,
            pause_multiplier=cfg.pause_multiplier,
            skip_footnotes=cfg.skip_footnotes,
            skip_captions=cfg.skip_captions,
            strip_references=cfg.strip_references,
            announce_pages=cfg.announce_pages,
            page_transition_pause=cfg.page_transition_pause,
        )

        result.time_script = time.perf_counter() - t0
        result.total_instructions = len(script)
        result.spoken_instructions = sum(
            1 for s in script if not s.should_skip and s.text
        )
        result.word_count = sum(len(s.text.split()) for s in script if s.text)

        print(
            f"  {result.total_instructions} instructions, "
            f"{result.spoken_instructions} spoken, "
            f"~{result.word_count} words"
        )
        print(f"  Script built in {result.time_script:.3f}s")

        # Debug: print script and exit early
        if cfg.debug_script:
            print("\n" + preview_script(script))
            result.elapsed_seconds = time.perf_counter() - t_total_start
            return result

        # --- Phase 3: TTS synthesis & audio assembly ---
        print("\n--- Phase 3: TTS synthesis ---")
        t0 = time.perf_counter()

        spoken_count = 0
        skipped_count = 0
        current_page = -1

        for i, inst in enumerate(script):
            # Page change announcement
            if inst.page_index != current_page:
                current_page = inst.page_index
                print(f"\n  [Page {current_page + 1}]")

            # Skip instructions with no output
            if inst.should_skip:
                skipped_count += 1
                continue

            p = inst.prosody

            # Pre-pause
            if p.pause_before > 0:
                builder.add_silence(p.pause_before)

            # Speech (page transitions have empty text — silence only)
            if inst.text:
                try:
                    wav = engine.synthesize(inst.text, speed_factor=p.speed_factor)
                    builder.add_speech(wav)
                    spoken_count += 1

                    preview = inst.text[:65].replace("\n", " ")
                    dur = engine.get_audio_duration(wav)
                    print(
                        f"    [{inst.role.name:16s} {p.speed_factor:.2f}x] "
                        f'{dur:.1f}s | "{preview}"'
                    )

                except Exception as e:
                    print(f"    [TTS ERROR] Skipping: {e}")
                    skipped_count += 1
                    continue

            # Post-pause
            if p.pause_after > 0:
                builder.add_silence(p.pause_after)

        result.time_synthesis = time.perf_counter() - t0
        result.skipped_blocks = skipped_count
        print(f"\n  Synthesis done in {result.time_synthesis:.1f}s")
        print(f"  Spoken: {spoken_count}, Skipped: {skipped_count}")

        # --- Phase 4: Post-process & export ---
        print("\n--- Phase 4: Post-processing & export ---")
        t0 = time.perf_counter()

        if builder.is_empty:
            print("  [WARN] No audio was produced. Nothing to export.")
            result.elapsed_seconds = time.perf_counter() - t_total_start
            return result

        builder.normalize(target_dBFS=cfg.normalize_dBFS)
        builder.apply_crossfade(ms=cfg.crossfade_ms)

        if is_wav:
            builder.export_wav(output_path)
        else:
            builder.export_mp3(output_path, bitrate=cfg.mp3_bitrate)

        result.time_export = time.perf_counter() - t0
        result.audio_duration = builder.get_duration()

        out = Path(output_path)
        if out.exists():
            result.file_size_mb = out.stat().st_size / (1024 * 1024)

        result.elapsed_seconds = time.perf_counter() - t_total_start

        print(f"\n{result.summary()}")
        return result


# ------------------------------------------------------------------
# Fallback classification (no YOLO)
# ------------------------------------------------------------------


def _fallback_classify(
    blocks,
    page_width: float,
    page_height: float,
) -> List[ClassifiedBlock]:
    """
    Classify blocks using only typographic features when YOLO fails.

    Returns ClassifiedBlock list with LayoutLabel.UNKNOWN so the
    feature refiner handles everything.
    """
    from narration.layout.feature_refiner import refine_classifications
    from narration.layout.models import ClassifiedBlock, LayoutLabel

    classified = [
        ClassifiedBlock(
            block=b,
            label=LayoutLabel.UNKNOWN,
            confidence=0.0,
        )
        for b in blocks
    ]
    refine_classifications(classified, page_width, page_height)
    return classified


# ------------------------------------------------------------------
# Debug layout image saver
# ------------------------------------------------------------------


def _save_debug_layouts(
    pdf: PDFAdapter,
    all_classified: Dict[int, List[ClassifiedBlock]],
    scale: float,
    output_dir: str,
) -> None:
    """Save colour-coded classification overlay images."""
    try:
        from tests.test_classifier import draw_classified_blocks
    except ImportError:
        print("  [WARN] Could not import debug drawing — skipping layout images.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for idx, classified in sorted(all_classified.items()):
        img = pdf.render(idx, scale=scale)
        annotated = draw_classified_blocks(img, classified, scale)
        path = out / f"page_{idx:03d}.png"
        annotated.save(str(path))

    print(f"  Debug layout images saved to {out}/")

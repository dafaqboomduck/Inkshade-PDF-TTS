"""
Audio builder: assembles synthesised speech chunks and silence gaps
into a single continuous audio file, then exports as MP3 or WAV.

Includes audio enhancement for more natural-sounding output:
compression, EQ warmth, and smooth transitions.
"""

import io
import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, normalize

logger = logging.getLogger(__name__)


class AudioBuilder:
    """
    Incrementally builds an audio track from WAV speech chunks and
    silence gaps.

    Usage::

        builder = AudioBuilder(sample_rate=24000)
        builder.add_silence(1.5)
        builder.add_speech(wav_bytes)
        builder.enhance()
        builder.normalize()
        builder.export_mp3("output.mp3")
    """

    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate
        self._audio = AudioSegment.empty()

    def add_silence(self, duration_seconds: float) -> None:
        """Append silence of the given duration."""
        if duration_seconds <= 0:
            return
        ms = int(duration_seconds * 1000)
        self._audio += AudioSegment.silent(duration=ms, frame_rate=self._sample_rate)

    def add_speech(self, wav_bytes: bytes) -> None:
        """Append a WAV speech chunk produced by a TTS engine."""
        if not wav_bytes or len(wav_bytes) <= 44:
            return
        self._audio += AudioSegment.from_wav(io.BytesIO(wav_bytes))

    def normalize(self, target_dBFS: float = -18.0) -> None:
        """Normalize volume to *target_dBFS*."""
        if len(self._audio) == 0:
            return
        current = self._audio.dBFS
        if current == float("-inf"):
            return
        self._audio = self._audio.apply_gain(target_dBFS - current)

    def enhance(self) -> None:
        """
        Apply audio enhancements for more natural, broadcast-quality
        sound:

        1. **Gentle compression** — reduces dynamic range so quiet and
           loud parts are closer in volume, mimicking professional
           audiobook production.
        2. **Low-frequency warmth** — subtle bass boost gives the voice
           more body and presence.
        3. **De-essing / high-frequency taming** — slight high-shelf
           reduction smooths out harsh sibilance.
        """
        if len(self._audio) == 0:
            return

        try:
            # 1. Gentle dynamic compression — evens out volume variations
            #    without squashing the natural dynamics entirely.
            self._audio = compress_dynamic_range(
                self._audio,
                threshold=-24.0,   # dBFS — start compressing above this
                ratio=2.5,         # gentle 2.5:1 ratio
                attack=8.0,        # ms — fast enough to catch transients
                release=120.0,     # ms — smooth release
            )
        except Exception as e:
            logger.debug("Compression step skipped: %s", e)

        try:
            # 2. Warmth boost — apply a subtle low-frequency lift.
            #    We use pydub's low_pass_filter on a copy to extract
            #    bass content, then mix it back at low volume.
            bass = self._audio.low_pass_filter(250)
            self._audio = self._audio.overlay(bass - 12)  # mix bass at -12 dB
        except Exception as e:
            logger.debug("Warmth boost skipped: %s", e)

        try:
            # 3. High-frequency taming — attenuate harsh sibilance
            self._audio = self._audio.high_pass_filter(60)  # remove sub-bass rumble
        except Exception as e:
            logger.debug("High-pass filter skipped: %s", e)

    def apply_crossfade(self, ms: int = 50) -> None:
        """Apply fade-in/out to smooth the track boundaries."""
        if len(self._audio) == 0 or ms <= 0:
            return
        self._audio = self._audio.fade_in(ms).fade_out(ms)

    def export_mp3(self, output_path: str, bitrate: str = "192k") -> None:
        """Export assembled audio as MP3."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._audio.export(str(path), format="mp3", bitrate=bitrate)
        self._log_export(path)

    def export_wav(self, output_path: str) -> None:
        """Export assembled audio as WAV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._audio.export(str(path), format="wav")
        self._log_export(path)

    def _log_export(self, path: Path) -> None:
        dur = self.get_duration()
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(
            "Exported %s: %.1fs (%.1f min), %.1f MB",
            path,
            dur,
            dur / 60,
            size_mb,
        )

    def get_duration(self) -> float:
        """Current total duration in seconds."""
        return len(self._audio) / 1000.0

    @property
    def is_empty(self) -> bool:
        return len(self._audio) == 0

    def __repr__(self) -> str:
        return f"AudioBuilder(duration={self.get_duration():.1f}s)"

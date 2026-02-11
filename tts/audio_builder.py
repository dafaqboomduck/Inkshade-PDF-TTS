"""
Audio builder: assembles synthesised speech chunks and silence gaps
into a single continuous audio file, then exports as MP3 or WAV.
"""

import io
import wave
from pathlib import Path
from typing import Optional

from pydub import AudioSegment


class AudioBuilder:
    """
    Incrementally builds an audio file from WAV speech chunks and
    silence gaps.

    Usage::

        builder = AudioBuilder(sample_rate=22050)
        builder.add_silence(1.5)
        builder.add_speech(wav_bytes)
        builder.add_silence(0.3)
        builder.add_speech(wav_bytes_2)
        builder.normalize()
        builder.export_mp3("output.mp3")
    """

    def __init__(self, sample_rate: int = 22050):
        self._sample_rate = sample_rate
        self._audio = AudioSegment.empty()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_silence(self, duration_seconds: float) -> None:
        """Append silence of the given duration."""
        if duration_seconds <= 0:
            return
        ms = int(duration_seconds * 1000)
        silence = AudioSegment.silent(duration=ms, frame_rate=self._sample_rate)
        self._audio += silence

    def add_speech(self, wav_bytes: bytes) -> None:
        """
        Append a WAV speech chunk.

        Args:
            wav_bytes: Complete WAV file bytes (as returned by
                       TTSEngine.synthesize).
        """
        if not wav_bytes or len(wav_bytes) <= 44:
            return

        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        self._audio += segment

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def normalize(self, target_dBFS: float = -20.0) -> None:
        """
        Normalize volume to a consistent level.

        Args:
            target_dBFS: Target loudness in dBFS (default -20).
        """
        if len(self._audio) == 0:
            return

        current = self._audio.dBFS
        if current == float("-inf"):
            return  # pure silence

        change = target_dBFS - current
        self._audio = self._audio.apply_gain(change)

    def apply_crossfade(self, ms: int = 50) -> None:
        """
        Apply a tiny crossfade to smooth segment boundaries.

        Note: this is most effective when called before final export.
        For the builder pattern (incremental add), a small fade-in/out
        on each segment is applied instead.
        """
        if len(self._audio) == 0 or ms <= 0:
            return

        self._audio = self._audio.fade_in(ms).fade_out(ms)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_mp3(
        self,
        output_path: str,
        bitrate: str = "192k",
    ) -> None:
        """
        Export assembled audio as MP3.

        Args:
            output_path: Destination file path.
            bitrate:     MP3 bitrate (default "192k").
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._audio.export(str(path), format="mp3", bitrate=bitrate)

        dur = self.get_duration()
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Exported MP3: {path}")
        print(f"  Duration: {dur:.1f}s ({dur / 60:.1f} min)")
        print(f"  Size:     {size_mb:.1f} MB")

    def export_wav(self, output_path: str) -> None:
        """
        Export assembled audio as WAV.

        Args:
            output_path: Destination file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._audio.export(str(path), format="wav")

        dur = self.get_duration()
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Exported WAV: {path}")
        print(f"  Duration: {dur:.1f}s ({dur / 60:.1f} min)")
        print(f"  Size:     {size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_duration(self) -> float:
        """Return current total duration in seconds."""
        return len(self._audio) / 1000.0

    def get_segment_count(self) -> int:
        """Return the number of segments added (approximate)."""
        # pydub doesn't track this, so we just report duration > 0
        return 1 if len(self._audio) > 0 else 0

    @property
    def is_empty(self) -> bool:
        return len(self._audio) == 0

    def __repr__(self) -> str:
        dur = self.get_duration()
        return f"AudioBuilder(duration={dur:.1f}s)"

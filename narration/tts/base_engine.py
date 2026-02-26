"""
Abstract base class for TTS engines.

Provides a unified interface so the pipeline can swap between
different TTS backends in the future.
"""

import io
import wave
from abc import ABC, abstractmethod


class BaseTTSEngine(ABC):
    """
    Common interface for all TTS engines used by the narration pipeline.

    Subclasses must implement :meth:`synthesize` and expose
    ``sample_rate``, ``sample_width``, and ``channels`` properties.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""

    @property
    @abstractmethod
    def sample_width(self) -> int:
        """Sample width in bytes (e.g. 2 for 16-bit PCM)."""

    @property
    @abstractmethod
    def channels(self) -> int:
        """Number of audio channels (1 = mono)."""

    @abstractmethod
    def synthesize(self, text: str, speed_factor: float = 1.0) -> bytes:
        """
        Synthesise *text* to WAV bytes.

        Args:
            text:         Text to speak.
            speed_factor: Speed multiplier (>1 = faster, <1 = slower).

        Returns:
            Complete WAV file as bytes (16-bit mono PCM).
        """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine identifier."""

    def generate_silence(self, duration_seconds: float) -> bytes:
        """Generate a WAV file containing *duration_seconds* of silence."""
        num_samples = int(self.sample_rate * max(0, duration_seconds))
        pcm_data = b"\x00\x00" * num_samples * self.channels
        return self._wrap_wav(pcm_data)

    def get_audio_duration(self, wav_bytes: bytes) -> float:
        """Return the duration in seconds of a WAV byte string."""
        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate if rate > 0 else 0.0
        except Exception:
            return 0.0

    def _wrap_wav(self, pcm_data: bytes) -> bytes:
        """Wrap raw PCM in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

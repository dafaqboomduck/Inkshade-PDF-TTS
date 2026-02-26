"""
Kokoro TTS engine wrapper.

Synthesises text using the Kokoro neural TTS model, which produces
highly natural, expressive speech.  Runs locally on CPU or GPU via
PyTorch — no API keys required.

The model (~82 MB) auto-downloads from HuggingFace on first use
and is cached locally.
"""

import logging

import numpy as np

from .base_engine import BaseTTSEngine

logger = logging.getLogger(__name__)

# Kokoro native sample rate
_KOKORO_SAMPLE_RATE = 24000

# Available Kokoro voices grouped by accent and gender
KOKORO_VOICES = {
    # American Female
    "af_heart": {"accent": "American", "gender": "Female", "name": "Heart"},
    "af_bella": {"accent": "American", "gender": "Female", "name": "Bella"},
    "af_nicole": {"accent": "American", "gender": "Female", "name": "Nicole"},
    "af_sarah": {"accent": "American", "gender": "Female", "name": "Sarah"},
    "af_sky": {"accent": "American", "gender": "Female", "name": "Sky"},
    # American Male
    "am_adam": {"accent": "American", "gender": "Male", "name": "Adam"},
    "am_michael": {"accent": "American", "gender": "Male", "name": "Michael"},
    # British Female
    "bf_emma": {"accent": "British", "gender": "Female", "name": "Emma"},
    "bf_isabella": {"accent": "British", "gender": "Female", "name": "Isabella"},
    # British Male
    "bm_george": {"accent": "British", "gender": "Male", "name": "George"},
    "bm_lewis": {"accent": "British", "gender": "Male", "name": "Lewis"},
}

DEFAULT_KOKORO_VOICE = "af_heart"


class KokoroEngine(BaseTTSEngine):
    """
    Expressive neural TTS via the Kokoro model.

    Usage::

        engine = KokoroEngine(voice="af_heart")
        wav_bytes = engine.synthesize("Hello world", speed_factor=1.0)

    The underlying model is loaded lazily on the first call to
    :meth:`synthesize`.
    """

    def __init__(self, voice: str = DEFAULT_KOKORO_VOICE, lang_code: str = "a"):
        """
        Initialise the Kokoro engine.

        Args:
            voice:     Voice identifier (see :data:`KOKORO_VOICES`).
            lang_code: Language code: ``'a'`` American English,
                       ``'b'`` British English.

        Raises:
            ImportError: If ``kokoro`` is not installed.
        """
        self._voice = voice
        self._lang_code = lang_code
        self._pipeline = None  # lazy init
        self._sample_rate = _KOKORO_SAMPLE_RATE
        self._sample_width = 2  # 16-bit PCM
        self._channels = 1

        # Validate the kokoro package is available
        try:
            import kokoro  # noqa: F401
        except ImportError:
            raise ImportError(
                "kokoro is required for the Kokoro TTS engine. "
                "Install with: pip install kokoro soundfile"
            )

    def _ensure_pipeline(self):
        """Lazy-load the Kokoro pipeline (downloads model on first use)."""
        if self._pipeline is not None:
            return

        logger.info(
            "Loading Kokoro TTS pipeline (voice=%s, lang=%s)...",
            self._voice,
            self._lang_code,
        )

        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code=self._lang_code)
        logger.info("Kokoro pipeline ready")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def sample_width(self) -> int:
        return self._sample_width

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def engine_name(self) -> str:
        return f"Kokoro ({self._voice})"

    @property
    def voice(self) -> str:
        return self._voice

    @voice.setter
    def voice(self, value: str):
        self._voice = value

    def synthesize(self, text: str, speed_factor: float = 1.0) -> bytes:
        """
        Synthesise *text* to WAV bytes using Kokoro.

        Args:
            text:         Text to speak.
            speed_factor: Speed multiplier (>1 = faster, <1 = slower).

        Returns:
            Complete WAV file as bytes (16-bit mono PCM at 24 kHz).
        """
        if not text or not text.strip():
            return self.generate_silence(0.0)

        self._ensure_pipeline()

        speed = max(0.1, speed_factor)

        # Kokoro yields chunks for long text; collect all audio
        audio_chunks = []
        try:
            generator = self._pipeline(
                text,
                voice=self._voice,
                speed=speed,
            )
            for _graphemes, _phonemes, audio_chunk in generator:
                if audio_chunk is not None:
                    audio_chunks.append(audio_chunk)
        except Exception as e:
            logger.error("Kokoro synthesis failed: %s", e)
            return self.generate_silence(0.0)

        if not audio_chunks:
            return self.generate_silence(0.0)

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks)

        # Convert float32 [-1, 1] → int16 PCM
        audio = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio * 32767).astype(np.int16).tobytes()

        return self._wrap_wav(pcm_data)

    def __repr__(self) -> str:
        return (
            f"KokoroEngine(voice={self._voice}, "
            f"rate={self._sample_rate}Hz, {self._sample_width * 8}bit, mono)"
        )

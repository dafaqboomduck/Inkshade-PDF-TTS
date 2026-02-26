"""
Voice model manager.

Provides metadata about Kokoro voices (which auto-download via
HuggingFace on first use and don't need manual management).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Kokoro voices (managed by the kokoro package, auto-downloaded)
KOKORO_VOICES: Dict[str, Dict[str, str]] = {
    "af_heart": {"accent": "American", "gender": "Female", "name": "Heart"},
    "af_bella": {"accent": "American", "gender": "Female", "name": "Bella"},
    "af_nicole": {"accent": "American", "gender": "Female", "name": "Nicole"},
    "af_sarah": {"accent": "American", "gender": "Female", "name": "Sarah"},
    "af_sky": {"accent": "American", "gender": "Female", "name": "Sky"},
    "am_adam": {"accent": "American", "gender": "Male", "name": "Adam"},
    "am_michael": {"accent": "American", "gender": "Male", "name": "Michael"},
    "bf_emma": {"accent": "British", "gender": "Female", "name": "Emma"},
    "bf_isabella": {"accent": "British", "gender": "Female", "name": "Isabella"},
    "bm_george": {"accent": "British", "gender": "Male", "name": "George"},
    "bm_lewis": {"accent": "British", "gender": "Male", "name": "Lewis"},
}
"""Kokoro voices — auto-managed by the kokoro package."""


class ModelManager:
    """
    Manages voice model metadata.

    For Kokoro, models auto-download via the ``kokoro`` package on
    first use.  This class provides a consistent interface for
    listing available voices.

    Usage::

        mgr = ModelManager()
        voices = mgr.list_kokoro_voices()
    """

    def __init__(self, voice_dir: Optional[Path] = None):
        self.voice_dir = voice_dir or (
            Path.home() / ".local" / "share" / "InkshadePDF" / "voices"
        )
        self.voice_dir.mkdir(parents=True, exist_ok=True)

    def list_kokoro_voices(self) -> List[str]:
        """Names of all available Kokoro voices."""
        return list(KOKORO_VOICES.keys())

    def get_kokoro_voice_info(self, voice_id: str) -> Optional[Dict[str, str]]:
        """Return metadata for a Kokoro voice, or ``None`` if unknown."""
        return KOKORO_VOICES.get(voice_id)

    def __repr__(self) -> str:
        return f"ModelManager(kokoro_voices={len(KOKORO_VOICES)})"

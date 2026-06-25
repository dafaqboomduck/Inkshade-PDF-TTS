"""Unit tests for TTS WAV helpers and Kokoro voice metadata.

The Kokoro model itself is never loaded — we test the WAV container
helpers via a tiny concrete BaseTTSEngine and the voice table as data.
"""

import io
import wave

from narration.tts.base_engine import BaseTTSEngine
from narration.tts.kokoro_engine import DEFAULT_KOKORO_VOICE, KOKORO_VOICES


class _DummyEngine(BaseTTSEngine):
    """Minimal concrete engine exposing the WAV helper machinery."""

    @property
    def sample_rate(self):
        return 24000

    @property
    def sample_width(self):
        return 2

    @property
    def channels(self):
        return 1

    @property
    def engine_name(self):
        return "Dummy"

    def synthesize(self, text, speed_factor=1.0):
        return self.generate_silence(1.0)


class TestWavHelpers:
    def test_generate_silence_has_correct_duration(self):
        eng = _DummyEngine()
        wav = eng.generate_silence(1.5)
        assert eng.get_audio_duration(wav) == 1.5

    def test_generated_wav_has_expected_format(self):
        eng = _DummyEngine()
        wav = eng.generate_silence(0.5)
        with wave.open(io.BytesIO(wav), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() == 12000  # 0.5s * 24000

    def test_zero_duration_silence(self):
        eng = _DummyEngine()
        assert eng.get_audio_duration(eng.generate_silence(0.0)) == 0.0

    def test_negative_duration_clamped_to_zero(self):
        eng = _DummyEngine()
        assert eng.get_audio_duration(eng.generate_silence(-1.0)) == 0.0

    def test_get_audio_duration_on_garbage_returns_zero(self):
        eng = _DummyEngine()
        assert eng.get_audio_duration(b"not a wav file") == 0.0

    def test_synthesize_via_base_helpers(self):
        eng = _DummyEngine()
        assert eng.get_audio_duration(eng.synthesize("hello")) == 1.0


class TestKokoroVoiceTable:
    def test_default_voice_is_present(self):
        assert DEFAULT_KOKORO_VOICE in KOKORO_VOICES

    def test_every_voice_has_required_metadata(self):
        for vid, meta in KOKORO_VOICES.items():
            assert {"accent", "gender", "name"} <= set(meta)
            assert meta["accent"] in {"American", "British"}
            assert meta["gender"] in {"Male", "Female"}

    def test_voice_id_prefix_matches_accent_and_gender(self):
        # Kokoro IDs encode accent (a/b) and gender (f/m) in the first two chars.
        accent_map = {"a": "American", "b": "British"}
        gender_map = {"f": "Female", "m": "Male"}
        for vid, meta in KOKORO_VOICES.items():
            assert accent_map[vid[0]] == meta["accent"]
            assert gender_map[vid[1]] == meta["gender"]

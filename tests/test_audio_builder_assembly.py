"""Unit tests for narration.tts.audio_builder (assembly + post-processing).

These exercise the AudioBuilder math and pydub pipeline without any TTS
model — speech chunks are plain silence WAV bytes from the make_wav factory.
"""

from narration.tts.audio_builder import AudioBuilder


class TestAssembly:
    def test_starts_empty(self):
        b = AudioBuilder(sample_rate=24000)
        assert b.is_empty
        assert b.get_duration() == 0.0

    def test_add_silence_extends_duration(self):
        b = AudioBuilder()
        b.add_silence(1.5)
        assert b.get_duration() == 1.5
        assert not b.is_empty

    def test_non_positive_silence_is_ignored(self):
        b = AudioBuilder()
        b.add_silence(0.0)
        b.add_silence(-2.0)
        assert b.is_empty

    def test_add_speech_appends_wav(self, make_wav):
        b = AudioBuilder(sample_rate=24000)
        b.add_speech(make_wav(0.5, sample_rate=24000))
        assert b.get_duration() == 0.5

    def test_add_speech_ignores_empty_or_header_only(self):
        b = AudioBuilder()
        b.add_speech(b"")
        b.add_speech(b"\x00" * 40)  # shorter than a 44-byte WAV header
        assert b.is_empty

    def test_durations_accumulate(self, make_wav):
        b = AudioBuilder(sample_rate=24000)
        b.add_silence(1.0)
        b.add_speech(make_wav(0.5, sample_rate=24000))
        b.add_silence(0.5)
        assert b.get_duration() == 2.0


class TestPostProcessing:
    def test_normalize_on_empty_is_noop(self):
        b = AudioBuilder()
        b.normalize()  # must not raise
        assert b.is_empty

    def test_enhance_on_empty_is_noop(self):
        b = AudioBuilder()
        b.enhance()
        assert b.is_empty

    def test_crossfade_on_empty_is_noop(self):
        b = AudioBuilder()
        b.apply_crossfade(50)
        assert b.is_empty

    def test_pipeline_preserves_duration(self, make_wav):
        b = AudioBuilder(sample_rate=24000)
        b.add_silence(0.5)
        b.add_speech(make_wav(1.0, sample_rate=24000))
        before = b.get_duration()
        b.enhance()
        b.normalize(target_dBFS=-18.0)
        b.apply_crossfade(ms=30)
        # Enhancement/normalisation/crossfade must not change track length.
        assert b.get_duration() == before

    def test_export_wav_writes_file(self, tmp_path, make_wav):
        b = AudioBuilder(sample_rate=24000)
        b.add_speech(make_wav(0.25, sample_rate=24000))
        out = tmp_path / "out.wav"
        b.export_wav(str(out))
        assert out.exists()
        assert out.stat().st_size > 44  # more than a bare WAV header

    def test_repr(self):
        b = AudioBuilder()
        b.add_silence(2.0)
        assert "2.0s" in repr(b)

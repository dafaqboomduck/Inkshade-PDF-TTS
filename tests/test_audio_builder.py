"""
Test the audio builder (Task 6.5).

Synthesises a few phrases, assembles them with silence gaps,
normalizes, and exports as both WAV and MP3.

Usage:
    python -m tests.test_audio_builder [--voice en_US-lessac-medium]
"""

import argparse
from pathlib import Path

from narration.tts.audio_builder import AudioBuilder
from narration.tts.engine import TTSEngine
from narration.tts.model_manager import ModelManager

SEGMENTS = [
    (1.5, 0.85, "Introduction to Machine Learning."),
    (1.2, 0.88, "Chapter One: Supervised Learning."),
    (
        0.3,
        1.00,
        "Machine learning is a subfield of artificial intelligence "
        "that focuses on building systems that learn from data.",
    ),
    (
        0.3,
        1.00,
        "This chapter introduces the fundamental concepts and "
        "provides a broad overview of the field.",
    ),
    (1.0, None, None),  # page transition (silence only)
    (1.2, 0.88, "Key Terminology."),
    (
        0.3,
        1.00,
        "A training set is a collection of labeled examples used to fit a model.",
    ),
    (0.3, 1.05, "See appendix B for a detailed derivation."),
]


def run(voice_name: str):
    out_dir = Path("debug") / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    mgr = ModelManager()
    voice_path = mgr.ensure_voice_available(voice_name)
    engine = TTSEngine(voice_path)

    print(f"Engine: {engine}\n")
    print("Assembling audio...\n")

    builder = AudioBuilder(sample_rate=engine.sample_rate)

    for pause, speed, text in SEGMENTS:
        # Pre-pause
        if pause > 0:
            builder.add_silence(pause)

        # Speech (or silence-only for page transitions)
        if text and speed:
            wav = engine.synthesize(text, speed_factor=speed)
            dur = engine.get_audio_duration(wav)
            builder.add_speech(wav)
            preview = text[:60]
            print(f'  [{speed}x] {dur:.2f}s | "{preview}"')
        else:
            print(f"  [pause] {pause:.1f}s")

    print(f"\nRaw duration: {builder.get_duration():.1f}s")

    # Post-process
    builder.normalize(target_dBFS=-20.0)
    builder.apply_crossfade(ms=30)

    print(f"After processing: {builder.get_duration():.1f}s\n")

    # Export both formats
    builder.export_wav(str(out_dir / "test_output.wav"))
    print()
    builder.export_mp3(str(out_dir / "test_output.mp3"))

    print("\nDone! Listen to the files in debug/audio/")


def main():
    parser = argparse.ArgumentParser(description="Test audio builder")
    parser.add_argument(
        "--voice",
        default="en_US-lessac-medium",
        help="Piper voice name (default: en_US-lessac-medium)",
    )
    args = parser.parse_args()
    run(args.voice)


if __name__ == "__main__":
    main()

"""
Test the Kokoro TTS engine (synthesise test phrases at different speeds).

Usage:
    python -m tests.test_tts_engine [--voice af_heart] [--lang a]
"""

import argparse
import time
from pathlib import Path

from narration.tts.kokoro_engine import KokoroEngine, KOKORO_VOICES

TEST_PHRASES = [
    ("title", 0.85, "Introduction to Machine Learning"),
    ("heading", 0.88, "Chapter 3: Supervised Learning"),
    (
        "body",
        1.00,
        "Machine learning is a subfield of artificial intelligence "
        "that focuses on building systems that learn from data.",
    ),
    ("footnote", 1.05, "See appendix B for a detailed derivation of this formula."),
    ("fast", 1.15, "This is a quick aside that should be read a bit faster."),
]


def run(voice: str, lang_code: str):
    out_dir = Path("debug") / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load engine
    engine = KokoroEngine(voice=voice, lang_code=lang_code)
    print(f"Engine: {engine}\n")

    # Synthesise test phrases
    total_synth = 0.0
    total_audio = 0.0

    for label, speed, text in TEST_PHRASES:
        t0 = time.perf_counter()
        wav = engine.synthesize(text, speed_factor=speed)
        elapsed = time.perf_counter() - t0
        duration = engine.get_audio_duration(wav)

        total_synth += elapsed
        total_audio += duration

        out_path = out_dir / f"{label}_{speed}x.wav"
        out_path.write_bytes(wav)

        rtf = elapsed / duration if duration > 0 else 0
        print(
            f"  [{label:8s}] speed={speed}x | "
            f"synth={elapsed:.3f}s | audio={duration:.2f}s | "
            f"RTF={rtf:.2f}x | {out_path}"
        )

    # Silence test
    silence = engine.generate_silence(1.5)
    sil_dur = engine.get_audio_duration(silence)
    sil_path = out_dir / "silence_1.5s.wav"
    sil_path.write_bytes(silence)
    print(f"\n  [silence ] duration={sil_dur:.2f}s | {sil_path}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Total synthesis time: {total_synth:.2f}s")
    print(f"Total audio duration: {total_audio:.2f}s")
    avg_rtf = total_synth / total_audio if total_audio > 0 else 0
    print(f"Average RTF: {avg_rtf:.2f}x (< 1.0 = faster than real-time)")
    print(f"\nWAV files saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Test Kokoro TTS engine")
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Kokoro voice ID (default: af_heart)",
    )
    parser.add_argument(
        "--lang",
        default="a",
        choices=["a", "b"],
        help="Language code: 'a' American, 'b' British (default: a)",
    )
    args = parser.parse_args()
    run(args.voice, args.lang)


if __name__ == "__main__":
    main()

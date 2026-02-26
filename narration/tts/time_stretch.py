"""
Pitch-preserving time stretching for speech audio.

Uses a phase-vocoder algorithm to change playback speed without
affecting pitch.  This avoids chipmunk / deep-voice artefacts that
naive resampling causes.

Only depends on ``numpy`` (already required by ultralytics / kokoro).
"""

import io
import logging
import wave

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def time_stretch_wav(wav_bytes: bytes, speed_factor: float) -> bytes:
    """
    Pitch-preserving time stretch of WAV audio.

    Args:
        wav_bytes:    Complete WAV file as bytes.
        speed_factor: Speed multiplier (>1 = faster, <1 = slower).
                      Clamped to [0.25, 4.0].

    Returns:
        New WAV file bytes at the same pitch but different duration.
        Returns the input unchanged if *speed_factor* ≈ 1.0.
    """
    speed_factor = max(0.25, min(4.0, speed_factor))

    if abs(speed_factor - 1.0) < 0.02:
        return wav_bytes

    # Decode WAV
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as e:
        logger.warning("time_stretch_wav: failed to decode WAV: %s", e)
        return wav_bytes

    if n_frames == 0:
        return wav_bytes

    # Convert to float64 samples
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
    else:
        logger.warning("time_stretch_wav: unsupported sample width %d", sample_width)
        return wav_bytes

    # Handle multi-channel: process first channel, replicate later
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
        mono = samples[:, 0]
    else:
        mono = samples

    # Apply phase vocoder time stretch
    stretched = _phase_vocoder_stretch(mono, speed_factor, sample_rate)

    # Convert back to original format
    if n_channels > 1:
        # Stretch all channels
        all_channels = []
        all_channels.append(stretched)
        for ch in range(1, n_channels):
            all_channels.append(
                _phase_vocoder_stretch(samples[:, ch], speed_factor, sample_rate)
            )
        # Interleave
        min_len = min(len(c) for c in all_channels)
        interleaved = np.column_stack([c[:min_len] for c in all_channels]).ravel()
    else:
        interleaved = stretched

    # Clip and convert to int
    if sample_width == 2:
        pcm = np.clip(interleaved, -32768, 32767).astype(np.int16)
    else:
        pcm = np.clip(interleaved, -2147483648, 2147483647).astype(np.int32)

    # Re-encode as WAV
    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out_buf.getvalue()


# ------------------------------------------------------------------
# Phase vocoder internals
# ------------------------------------------------------------------


def _phase_vocoder_stretch(
    audio: np.ndarray,
    rate: float,
    sample_rate: int,
) -> np.ndarray:
    """
    Pitch-preserving time stretch using a phase vocoder.

    The phase vocoder works in three steps:
    1. STFT the signal into overlapping spectral frames
    2. Resample the magnitude/phase trajectory at the desired rate,
       carefully unwrapping and accumulating phase to avoid artefacts
    3. ISTFT back to the time domain

    Args:
        audio:       Mono float64 audio samples.
        rate:        Speed factor (>1 = shorter/faster).
        sample_rate: Audio sample rate (used to pick FFT size).

    Returns:
        Time-stretched float64 audio.
    """
    if len(audio) < 256:
        return audio

    # Choose FFT parameters appropriate for the sample rate.
    # Larger windows give better frequency resolution (less phasing),
    # but speech benefits from moderate sizes.
    if sample_rate >= 32000:
        n_fft = 2048
    elif sample_rate >= 16000:
        n_fft = 1024
    else:
        n_fft = 512

    hop = n_fft // 4
    window = np.hanning(n_fft)

    # --- STFT ---
    # Pad signal so we don't lose the tail
    pad_len = n_fft + hop
    padded = np.concatenate([audio, np.zeros(pad_len)])
    n_frames = (len(padded) - n_fft) // hop + 1

    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = padded[start: start + n_fft] * window
        stft[:, i] = np.fft.rfft(frame)

    # --- Phase vocoder resample ---
    n_bins = stft.shape[0]
    time_steps = np.arange(0, n_frames - 1, rate)
    n_out = len(time_steps)

    out_stft = np.zeros((n_bins, n_out), dtype=np.complex128)

    # Expected phase advance per hop for each frequency bin
    omega = 2.0 * np.pi * np.arange(n_bins) * hop / n_fft

    # Accumulate phase starting from the first frame
    phase_acc = np.angle(stft[:, 0])
    out_stft[:, 0] = np.abs(stft[:, 0]) * np.exp(1j * phase_acc)

    for t in range(1, n_out):
        # Interpolated position in the original STFT
        pos = time_steps[t]
        i = int(pos)
        frac = pos - i

        # Interpolate magnitude
        i0 = min(i, n_frames - 1)
        i1 = min(i + 1, n_frames - 1)
        mag = (1.0 - frac) * np.abs(stft[:, i0]) + frac * np.abs(stft[:, i1])

        # Phase advance: unwrap the instantaneous frequency
        dphi = np.angle(stft[:, i1]) - np.angle(stft[:, i0])

        # Remove expected phase advance, wrap to [-π, π], re-add
        dphi_wrapped = dphi - omega
        dphi_wrapped -= 2.0 * np.pi * np.round(dphi_wrapped / (2.0 * np.pi))
        true_freq = omega + dphi_wrapped

        phase_acc += true_freq
        out_stft[:, t] = mag * np.exp(1j * phase_acc)

    # --- ISTFT (overlap-add) ---
    out_len = n_fft + (n_out - 1) * hop
    output = np.zeros(out_len)
    win_sum = np.zeros(out_len)

    for t in range(n_out):
        frame = np.fft.irfft(out_stft[:, t]) * window
        start = t * hop
        end = start + n_fft
        output[start:end] += frame
        win_sum[start:end] += window * window

    # Normalise by window overlap
    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum

    # Trim to expected duration
    target_len = int(len(audio) / rate)
    output = output[:target_len]

    return output

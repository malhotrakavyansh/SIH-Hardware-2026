"""
generate_synthetic_background.py -- one-off generator for synthetic
data/background/ content, complementing the real ambient recording.

Produces 18 files (background_synthetic_01.wav .. _18.wav), 30s each,
16kHz mono 16-bit PCM:
    01-05  pure silence + tiny dither (-60 dBFS uniform noise)
    06-10  white noise, -40 to -30 dBFS
    11-15  pink noise (1/f), -40 to -30 dBFS
    16-18  mostly silence with sparse short impulse taps/clicks

Not part of the Step 1-7 pipeline -- run manually, once:
    python generate_synthetic_background.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent / "kws"))
import config

OUT_DIR = config.BACKGROUND_DIR
DURATION_S = 30.0
SEED = config.SEED


def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def pure_silence(rng: np.random.Generator, n: int) -> np.ndarray:
    amp = db_to_amplitude(-60.0)
    return rng.uniform(-amp, amp, size=n).astype(np.float32)


def white_noise(rng: np.random.Generator, n: int, db: float) -> np.ndarray:
    amp = db_to_amplitude(db)
    noise = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    noise /= np.max(np.abs(noise)) + 1e-9
    return (noise * amp).astype(np.float32)


def pink_noise(rng: np.random.Generator, n: int, db: float) -> np.ndarray:
    """1/f noise via frequency-domain shaping of white noise."""
    white = rng.normal(0.0, 1.0, size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
    pink /= np.max(np.abs(pink)) + 1e-9
    amp = db_to_amplitude(db)
    return (pink * amp).astype(np.float32)


def impulses(rng: np.random.Generator, n: int, sample_rate: int) -> np.ndarray:
    """Mostly silence (tiny dither) with sparse short taps/clicks."""
    audio = pure_silence(rng, n)
    duration_s = n / sample_rate
    num_impulses = int(rng.integers(5, 16))
    for _ in range(num_impulses):
        click_len = int(rng.uniform(0.01, 0.05) * sample_rate)  # 10-50ms
        start = int(rng.uniform(0, max(1, n - click_len)))
        peak_db = rng.uniform(-20.0, -10.0)
        peak_amp = db_to_amplitude(peak_db)
        click = rng.normal(0.0, 1.0, size=click_len).astype(np.float32)
        envelope = np.exp(-np.linspace(0, 6, click_len))  # fast decay
        click = click * envelope
        click /= np.max(np.abs(click)) + 1e-9
        audio[start:start + click_len] += click * peak_amp
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = int(config.SAMPLE_RATE * DURATION_S)
    rng = np.random.default_rng(SEED)

    index = 1

    for _ in range(5):
        audio = pure_silence(rng, n)
        path = OUT_DIR / f"background_synthetic_{index:02d}.wav"
        sf.write(path, audio, config.SAMPLE_RATE, subtype="PCM_16")
        print(f"wrote {path.name}  (pure silence, -60 dBFS dither)")
        index += 1

    white_dbs = np.linspace(-40.0, -30.0, 5)
    for db in white_dbs:
        audio = white_noise(rng, n, float(db))
        path = OUT_DIR / f"background_synthetic_{index:02d}.wav"
        sf.write(path, audio, config.SAMPLE_RATE, subtype="PCM_16")
        print(f"wrote {path.name}  (white noise, {db:.1f} dBFS)")
        index += 1

    pink_dbs = np.linspace(-40.0, -30.0, 5)
    for db in pink_dbs:
        audio = pink_noise(rng, n, float(db))
        path = OUT_DIR / f"background_synthetic_{index:02d}.wav"
        sf.write(path, audio, config.SAMPLE_RATE, subtype="PCM_16")
        print(f"wrote {path.name}  (pink noise, {db:.1f} dBFS)")
        index += 1

    for _ in range(3):
        audio = impulses(rng, n, config.SAMPLE_RATE)
        path = OUT_DIR / f"background_synthetic_{index:02d}.wav"
        sf.write(path, audio, config.SAMPLE_RATE, subtype="PCM_16")
        print(f"wrote {path.name}  (sparse impulses/taps)")
        index += 1

    print(f"\n{index - 1} synthetic background files written to {OUT_DIR}")


if __name__ == "__main__":
    main()

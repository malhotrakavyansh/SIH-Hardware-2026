"""
features.py -- MFCC front end.

This is the module the C firmware has to match bit-for-bit (Step 2). Keep it
dependency-light and free of framework magic: plain numpy, explicit loops over
frames, no librosa defaults leaking in. Anything that is a number comes from
config.

Pipeline, in order:
    int16 pcm -> float32 [-1, 1)
    -> pre-emphasis (PREEMPHASIS)
    -> frame (FRAME_LEN, hop STRIDE_LEN)  -> (NUM_FRAMES, FRAME_LEN)
    -> window (WINDOW)
    -> zero-pad to NFFT, rfft                -> (NUM_FRAMES, NUM_FREQ_BINS)
    -> power spectrum
    -> mel filterbank (NUM_MEL, FMIN..FMAX)  -> (NUM_FRAMES, NUM_MEL)
    -> log(x + LOG_MEL_FLOOR)
    -> DCT-II, keep NUM_MFCC                 -> (NUM_FRAMES, NUM_MFCC)

Implemented in Step 2.
"""

from __future__ import annotations

import numpy as np

import config


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """HTK mel scale. Must match the C implementation exactly."""
    raise NotImplementedError("Step 2")


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Inverse of hz_to_mel."""
    raise NotImplementedError("Step 2")


def mel_filterbank() -> np.ndarray:
    """Triangular mel filterbank.

    Returns float32 of shape (config.NUM_MEL, config.NUM_FREQ_BINS).
    Deterministic: this array is what export_c.py dumps as a C constant, so it
    must not depend on anything outside config.
    """
    raise NotImplementedError("Step 2")


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    """int16 samples -> float32 in [-1, 1). No resampling, no dc removal."""
    raise NotImplementedError("Step 2")


def preemphasis(samples: np.ndarray) -> np.ndarray:
    """y[n] = x[n] - PREEMPHASIS * x[n-1], with y[0] = x[0]."""
    raise NotImplementedError("Step 2")


def frame_signal(samples: np.ndarray) -> np.ndarray:
    """Split a clip into overlapping analysis frames.

    Args:
        samples: float32, exactly config.CLIP_SAMPLES long.
    Returns:
        float32 (config.NUM_FRAMES, config.FRAME_LEN).
    """
    raise NotImplementedError("Step 2")


def window_fn() -> np.ndarray:
    """Analysis window of length config.FRAME_LEN, per config.WINDOW."""
    raise NotImplementedError("Step 2")


def power_spectrum(frames: np.ndarray) -> np.ndarray:
    """Windowed frames -> power spectra, zero-padding FRAME_LEN to NFFT.

    Returns float32 (config.NUM_FRAMES, config.NUM_FREQ_BINS).
    """
    raise NotImplementedError("Step 2")


def log_mel(frames: np.ndarray) -> np.ndarray:
    """Power spectra -> log mel energies (config.NUM_FRAMES, config.NUM_MEL)."""
    raise NotImplementedError("Step 2")


def dct(log_mel_energies: np.ndarray) -> np.ndarray:
    """Orthonormal DCT-II, truncated to config.NUM_MFCC coefficients."""
    raise NotImplementedError("Step 2")


def mfcc(samples: np.ndarray) -> np.ndarray:
    """Full front end for one clip.

    Args:
        samples: float32 in [-1, 1), exactly config.CLIP_SAMPLES long.
    Returns:
        float32 of shape config.FEATURE_SHAPE.
    """
    raise NotImplementedError("Step 2")


def mfcc_batch(clips: np.ndarray) -> np.ndarray:
    """Vectorised mfcc() over a batch.

    Args:
        clips: float32 (batch, config.CLIP_SAMPLES).
    Returns:
        float32 (batch, config.NUM_FRAMES, config.NUM_MFCC).
    """
    raise NotImplementedError("Step 2")


if __name__ == "__main__":
    print(config.summary())
    print(f"\nfeature tensor per clip: {config.FEATURE_SHAPE}")

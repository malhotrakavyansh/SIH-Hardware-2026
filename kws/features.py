"""
features.py -- MFCC front end.

This is the module the C firmware has to match bit-for-bit (Step 2, the
gating step). Kept dependency-light and free of framework magic: plain numpy,
every stage spelled out as an explicit array op, no `librosa.mfcc()` call
whose padding/normalization defaults you can't inspect or reproduce in C.
Every number comes from config -- nothing here is a tunable.

Pipeline, in order:
    int16 pcm                                (CLIP_SAMPLES,)
    -> float32 [-1, 1)                       pcm16_to_float
    -> pre-emphasis                          preemphasis
    -> frame (FRAME_LEN, hop STRIDE_LEN)     frame_signal   -> (NUM_FRAMES, FRAME_LEN)
    -> Hann window                           window_fn
    -> zero-pad FRAME_LEN -> NFFT, rfft, |.|^2  power_spectrum -> (NUM_FRAMES, NUM_FREQ_BINS)
    -> mel filterbank                        log_mel        -> (NUM_FRAMES, NUM_MEL)
    -> log(x + LOG_MEL_FLOOR)
    -> DCT-II, keep NUM_MFCC                 dct            -> (NUM_FRAMES, NUM_MFCC)

The mel filterbank and DCT matrix are fixed lookup tables that depend only on
config, never on audio. That's deliberate: export_c.py dumps them as C
arrays, so the firmware never re-derives a mel scale or a cosine table on the
MCU -- it does a matrix multiply against Python's own numbers. Whatever's
left to get right on the C side (FFT, windowing, the two matmuls) is a small,
auditable surface.

Two things a firmware port must match exactly, spelled out because they're
invisible if you only read the shapes:
  * Hann window is the SYMMETRIC form, w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1))
    (matches CMSIS-DSP's arm_hanning_f32 -- confirm esp-dsp's
    dsps_wind_hann_f32 uses the same N-1 denominator, not a periodic N one).
  * power_spectrum is raw |FFT|^2, no 1/N normalization. A missing/extra
    normalization shows up as a uniform per-bin scale factor in the parity
    diff, which is the easiest failure mode to recognize and fix.
"""

from __future__ import annotations

import numpy as np

import config


# =============================================================================
# Mel scale (HTK formula -- must match the C implementation exactly)
# =============================================================================

def hz_to_mel(hz):
    """HTK mel scale: mel = 2595 * log10(1 + hz/700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    """Inverse of hz_to_mel."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


# =============================================================================
# Fixed lookup tables -- what export_c.py dumps as C arrays
# =============================================================================

def build_mel_filterbank() -> np.ndarray:
    """Triangular mel filterbank, standard HTK/Slaney construction.

    NUM_MEL+2 points equally spaced on the mel scale between FMIN and FMAX,
    mapped back to Hz, then to FFT bin indices. Filter m is a triangle rising
    from bin[m-1] to bin[m] and falling from bin[m] to bin[m+1].

    Returns float32 (config.NUM_MEL, config.NUM_FREQ_BINS). Depends only on
    config -- must be exactly reproducible from a fresh interpreter.
    """
    low_mel = hz_to_mel(config.FMIN)
    high_mel = hz_to_mel(config.FMAX)
    mel_points = np.linspace(low_mel, high_mel, config.NUM_MEL + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor(
        (config.NFFT + 1) * hz_points / config.SAMPLE_RATE
    ).astype(np.int64)

    filterbank = np.zeros((config.NUM_MEL, config.NUM_FREQ_BINS), dtype=np.float64)
    for m in range(1, config.NUM_MEL + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]

        if f_center > f_left:
            for k in range(max(f_left, 0), min(f_center, config.NUM_FREQ_BINS)):
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)

        if f_right > f_center:
            for k in range(max(f_center, 0), min(f_right, config.NUM_FREQ_BINS)):
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return filterbank.astype(np.float32)


def build_dct_matrix() -> np.ndarray:
    """Orthonormal DCT-II, truncated to the first config.NUM_MFCC rows.

    y_k = w_k * sum_n x_n * cos(pi/N * (n + 0.5) * k),
    w_0 = sqrt(1/N), w_k = sqrt(2/N) for k > 0, N = config.NUM_MEL.

    Same normalization as scipy.fft.dct(type=2, norm='ortho') / librosa's
    default -- spelled out here as a matrix so the firmware does a plain
    matmul instead of needing its own DCT primitive.

    Returns float32 (config.NUM_MFCC, config.NUM_MEL).
    """
    n = np.arange(config.NUM_MEL, dtype=np.float64)
    k = np.arange(config.NUM_MFCC, dtype=np.float64)[:, None]

    matrix = np.cos(np.pi / config.NUM_MEL * (n + 0.5) * k)
    matrix *= np.sqrt(2.0 / config.NUM_MEL)
    matrix[0, :] = np.sqrt(1.0 / config.NUM_MEL)

    return matrix.astype(np.float32)


def window_fn() -> np.ndarray:
    """Symmetric Hann window of length config.FRAME_LEN.

    w[n] = 0.5 - 0.5*cos(2*pi*n / (FRAME_LEN - 1))  -- see module docstring
    for why the symmetric (N-1) form, not the periodic (N) one, was chosen.
    """
    if config.WINDOW != "hann":
        raise ValueError(f"unsupported window: {config.WINDOW!r}")
    n = np.arange(config.FRAME_LEN, dtype=np.float64)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (config.FRAME_LEN - 1))
    return w.astype(np.float32)


# Computed once at import -- audio-independent, so there is no reason to
# rebuild them per clip. build_mel_filterbank()/build_dct_matrix()/window_fn()
# stay callable on their own for export_c.py and the parity/self-tests.
_MEL_FILTERBANK = build_mel_filterbank()
_DCT_MATRIX = build_dct_matrix()
_HANN_WINDOW = window_fn()


# =============================================================================
# Per-clip stages
# =============================================================================

def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    """int16 samples -> float32 in [-1, 1). No resampling, no dc removal."""
    return (pcm.astype(np.float32)) / np.float32(32768.0)


def preemphasis(samples: np.ndarray) -> np.ndarray:
    """y[n] = x[n] - PREEMPHASIS * x[n-1], with y[0] = x[0]."""
    coeff = np.float32(config.PREEMPHASIS)
    out = np.empty_like(samples)
    out[0] = samples[0]
    out[1:] = samples[1:] - coeff * samples[:-1]
    return out


def frame_signal(samples: np.ndarray) -> np.ndarray:
    """Split a clip into overlapping analysis frames.

    Args:
        samples: float32, exactly config.CLIP_SAMPLES long.
    Returns:
        float32 (config.NUM_FRAMES, config.FRAME_LEN).
    """
    assert samples.shape[-1] == config.CLIP_SAMPLES, (
        f"expected {config.CLIP_SAMPLES} samples, got {samples.shape[-1]}"
    )
    frame_starts = config.STRIDE_LEN * np.arange(config.NUM_FRAMES)
    within_frame = np.arange(config.FRAME_LEN)
    indices = frame_starts[:, None] + within_frame[None, :]
    return samples[indices].astype(np.float32)


def power_spectrum(frames: np.ndarray) -> np.ndarray:
    """Windowed frames -> power spectra, zero-padding FRAME_LEN to NFFT.

    Raw |FFT|^2 -- no 1/N normalization (see module docstring).
    Returns float32 (config.NUM_FRAMES, config.NUM_FREQ_BINS).
    """
    spectrum = np.fft.rfft(frames, n=config.NFFT, axis=-1)
    return (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)


def log_mel(power_frames: np.ndarray) -> np.ndarray:
    """Power spectra -> log mel energies (config.NUM_FRAMES, config.NUM_MEL)."""
    mel_energy = power_frames.astype(np.float64) @ _MEL_FILTERBANK.T
    return np.log(mel_energy + config.LOG_MEL_FLOOR).astype(np.float32)


def dct(log_mel_energies: np.ndarray) -> np.ndarray:
    """Orthonormal DCT-II, truncated to config.NUM_MFCC coefficients."""
    return (log_mel_energies.astype(np.float64) @ _DCT_MATRIX.T).astype(np.float32)


def extract_mfcc(audio_int16: np.ndarray) -> np.ndarray:
    """Full front end for one clip.

    Args:
        audio_int16: int16, exactly config.CLIP_SAMPLES long.
    Returns:
        float32 of shape config.FEATURE_SHAPE, i.e. (NUM_FRAMES, NUM_MFCC).
    """
    assert audio_int16.shape[-1] == config.CLIP_SAMPLES, (
        f"expected {config.CLIP_SAMPLES} int16 samples, "
        f"got {audio_int16.shape[-1]}"
    )
    samples = pcm16_to_float(audio_int16)
    samples = preemphasis(samples)
    frames = frame_signal(samples)
    frames = frames * _HANN_WINDOW[None, :]
    power = power_spectrum(frames)
    logmel = log_mel(power)
    return dct(logmel)


def mfcc_batch(clips: np.ndarray) -> np.ndarray:
    """extract_mfcc() over a batch. Not vectorised across clips -- a plain
    loop keeps this parity-critical stage easy to read; revisit only if
    training throughput actually becomes the bottleneck.

    Args:
        clips: int16 (batch, config.CLIP_SAMPLES).
    Returns:
        float32 (batch, config.NUM_FRAMES, config.NUM_MFCC).
    """
    return np.stack([extract_mfcc(clip) for clip in clips]).astype(np.float32)


# =============================================================================
# Self-test -- shape and sanity checks only. Real parity against the C
# implementation is make_test_vectors.py + check_parity.py, run against
# whatever the firmware side prints.
# =============================================================================

def _self_test() -> None:
    fb = build_mel_filterbank()
    assert fb.shape == (config.NUM_MEL, config.NUM_FREQ_BINS), fb.shape
    assert np.all(fb >= 0.0), "filterbank must be non-negative"
    row_max = fb.max(axis=1)
    assert np.all(row_max > 0.0), "every mel filter must have some passband"

    dct_mat = build_dct_matrix()
    assert dct_mat.shape == (config.NUM_MFCC, config.NUM_MEL), dct_mat.shape
    # rows are orthonormal vectors in R^NUM_MEL -> dct @ dct.T ~= identity
    gram = dct_mat.astype(np.float64) @ dct_mat.astype(np.float64).T
    assert np.allclose(gram, np.eye(config.NUM_MFCC), atol=1e-4), (
        "DCT rows are not orthonormal -- check the formula"
    )

    win = window_fn()
    assert win.shape == (config.FRAME_LEN,), win.shape
    assert win[0] == 0.0 and abs(win[-1] - 0.0) < 1e-6, (
        "symmetric Hann window should be ~0 at both endpoints"
    )

    rng = np.random.default_rng(config.SEED)
    tone = (
        0.5 * np.sin(2 * np.pi * 440.0 * np.arange(config.CLIP_SAMPLES) / config.SAMPLE_RATE)
    )
    tone_i16 = (tone * 32767.0).astype(np.int16)
    feat = extract_mfcc(tone_i16)
    assert feat.shape == config.FEATURE_SHAPE, feat.shape
    assert np.all(np.isfinite(feat)), "MFCC output contains NaN/Inf"

    silence_i16 = np.zeros(config.CLIP_SAMPLES, dtype=np.int16)
    feat_silence = extract_mfcc(silence_i16)
    assert np.all(np.isfinite(feat_silence)), "silence must not produce NaN/Inf"

    noise_i16 = (rng.uniform(-1.0, 1.0, config.CLIP_SAMPLES) * 3000).astype(np.int16)
    batch = mfcc_batch(np.stack([tone_i16, silence_i16, noise_i16]))
    assert batch.shape == (3, *config.FEATURE_SHAPE), batch.shape

    print("PASS  mel filterbank shape + non-negativity")
    print("PASS  DCT rows orthonormal")
    print("PASS  Hann window shape + endpoints")
    print("PASS  extract_mfcc shape + finite on tone/silence/noise")
    print("PASS  mfcc_batch shape")


if __name__ == "__main__":
    print(config.summary())
    print(f"\nfeature tensor per clip: {config.FEATURE_SHAPE}\n")
    _self_test()

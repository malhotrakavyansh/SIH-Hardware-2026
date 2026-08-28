"""
config.py -- SINGLE source of truth for the whole KWS pipeline.

Every other module imports from here. There are NO numeric constants
anywhere else in the repo; verify_step1.py enforces that.

Two sections:

  FROZEN   -- the feature/model I/O contract. Training, quantization, the
              streaming evaluator and the exported C firmware must all agree
              on these to the last bit. Once Step 2 (feature parity between
              Python and C) passes, changing anything here invalidates every
              trained checkpoint and every exported array. Don't.

  TUNABLE  -- training recipe, augmentation, detection policy. Safe to sweep;
              changing these does not break the Python<->C contract.
"""

import hashlib
from pathlib import Path

# =============================================================================
# FROZEN -- audio front end
# =============================================================================

SAMPLE_RATE = 16000       # Hz, mono
CLIP_MS     = 1000        # analysis window fed to the model
FRAME_MS    = 40          # STFT window length
STRIDE_MS   = 20          # STFT hop (50% overlap)
NUM_MFCC    = 10          # cepstral coefficients kept per frame
NUM_MEL     = 40          # mel filterbank channels
FMIN, FMAX  = 20, 4000    # Hz, mel filterbank edges
NFFT        = 1024        # FFT size -- see note below
NUM_FRAMES  = 49          # (CLIP_MS - FRAME_MS) / STRIDE_MS + 1

# NFFT note: a 40 ms frame at 16 kHz is 640 samples. The FFT size must be
# >= the frame length or every frame gets truncated and ~20% of each window is
# silently thrown away. 1024 is the next power of two above 640, which is also
# what ARM's reference DS-CNN MFCC front end (mfcc.cpp, m_frameLenPadded) uses.
# Frames are zero-padded 640 -> 1024. If you need 512 instead (cheaper on the
# MCU), drop FRAME_MS to 32 so the frame is exactly 512 samples -- do NOT pair
# FRAME_MS=40 with NFFT=512.

PREEMPHASIS   = 0.97      # first-order pre-emphasis coefficient
WINDOW        = "hann"    # analysis window; must match the C implementation
LOG_MEL_FLOOR = 1e-6      # added before log() to avoid log(0)
DITHER        = 0.0       # no dither: must be bit-reproducible against C

# =============================================================================
# FROZEN -- label space
# =============================================================================

LABELS         = ["silence", "unknown", "keyword"]
SILENCE_INDEX  = 0
UNKNOWN_INDEX  = 1
KEYWORD_INDEX  = 2
NUM_CLASSES    = len(LABELS)
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

# The wake word being spotted. Routes data/ and is part of the frozen
# contract, so setting it changes FEATURE_CONTRACT_HASH -- pick it before you
# record positives, not after.
KEYWORD = "wakeword"      # TODO: set the actual wake word (Step 3 blocker)

# =============================================================================
# FROZEN -- derived sample counts (never hand-type these anywhere else)
# =============================================================================

CLIP_SAMPLES   = SAMPLE_RATE * CLIP_MS   // 1000   # 16000
FRAME_LEN      = SAMPLE_RATE * FRAME_MS  // 1000   # 640
STRIDE_LEN     = SAMPLE_RATE * STRIDE_MS // 1000   # 320
NUM_FREQ_BINS  = NFFT // 2 + 1                     # 513
FEATURE_SHAPE  = (NUM_FRAMES, NUM_MFCC)            # (49, 10)
FEATURE_SIZE   = NUM_FRAMES * NUM_MFCC             # 490

# =============================================================================
# FROZEN -- streaming inference contract
# =============================================================================

HOP_MS      = 100                                   # slide the window this far
HOP_SAMPLES = SAMPLE_RATE * HOP_MS // 1000          # 1600
HOP_FRAMES  = HOP_MS // STRIDE_MS                   # 5 new MFCC frames per hop

# =============================================================================
# FROZEN -- model topology (DS-CNN-S, ARM ML-KWS)
# =============================================================================

DSCNN_CHANNELS     = 64
DSCNN_NUM_BLOCKS   = 4
DSCNN_CONV_KERNEL  = (10, 4)    # (time, freq) for the first full conv
DSCNN_CONV_STRIDE  = (2, 2)
DSCNN_DW_KERNEL    = (3, 3)     # depthwise separable blocks
DSCNN_DW_STRIDE    = (1, 1)
DSCNN_DROPOUT      = 0.2
BN_MOMENTUM        = 0.99
BN_EPSILON         = 1e-3

# =============================================================================
# FROZEN -- feature contract hash
# =============================================================================
#
# A digest over every value above that the Python front end and the C firmware
# must agree on. train.py stamps it into each checkpoint and export_c.py bakes
# it into the generated headers, so an edit to anything frozen cannot silently
# reach a trained model or the MCU -- the mismatch shows up as a failed hash
# compare instead of a silently wrong detector. Tunables are deliberately left
# out: sweeping the training recipe must not invalidate the contract.

_FROZEN_CONTRACT = (
    ("SAMPLE_RATE", SAMPLE_RATE),
    ("CLIP_MS", CLIP_MS),
    ("FRAME_MS", FRAME_MS),
    ("STRIDE_MS", STRIDE_MS),
    ("NUM_MFCC", NUM_MFCC),
    ("NUM_MEL", NUM_MEL),
    ("FMIN", FMIN),
    ("FMAX", FMAX),
    ("NFFT", NFFT),
    ("NUM_FRAMES", NUM_FRAMES),
    ("PREEMPHASIS", PREEMPHASIS),
    ("WINDOW", WINDOW),
    ("LOG_MEL_FLOOR", LOG_MEL_FLOOR),
    ("DITHER", DITHER),
    ("LABELS", tuple(LABELS)),
    ("KEYWORD", KEYWORD),
    ("HOP_MS", HOP_MS),
    ("DSCNN_CHANNELS", DSCNN_CHANNELS),
    ("DSCNN_NUM_BLOCKS", DSCNN_NUM_BLOCKS),
    ("DSCNN_CONV_KERNEL", DSCNN_CONV_KERNEL),
    ("DSCNN_CONV_STRIDE", DSCNN_CONV_STRIDE),
    ("DSCNN_DW_KERNEL", DSCNN_DW_KERNEL),
    ("DSCNN_DW_STRIDE", DSCNN_DW_STRIDE),
    ("BN_MOMENTUM", BN_MOMENTUM),
    ("BN_EPSILON", BN_EPSILON),
)

FEATURE_CONTRACT_HASH = hashlib.sha256(
    repr(_FROZEN_CONTRACT).encode("utf-8")
).hexdigest()[:16]

# =============================================================================
# TUNABLE -- dataset split
# =============================================================================

# Hash-based split on the speaker id, so the same speaker never straddles
# train/val/test. Constant is from the Google Speech Commands reference code.
MAX_WAVS_PER_CLASS  = 2 ** 27 - 1
VALIDATION_PERCENT  = 10.0
TESTING_PERCENT     = 10.0

# Class mix per training epoch, as a fraction of the number of keyword clips.
SILENCE_PERCENT = 10.0
UNKNOWN_PERCENT = 10.0

# =============================================================================
# TUNABLE -- augmentation
# =============================================================================

TIME_SHIFT_MS         = 100     # random +/- shift before padding to CLIP_MS
BACKGROUND_PROB       = 0.8     # fraction of clips that get noise mixed in
BACKGROUND_VOL_MAX    = 0.1     # linear gain ceiling on the noise
SNR_DB_RANGE          = (0.0, 20.0)
SPEED_PERTURB_RANGE   = (0.9, 1.1)
SPEED_PERTURB_PROB    = 0.3
GAIN_DB_RANGE         = (-6.0, 6.0)
REVERB_PROB           = 0.0     # off until an RIR set is in data/background/

# =============================================================================
# TUNABLE -- training
# =============================================================================

SEED             = 1337
BATCH_SIZE       = 64
EPOCHS           = 60
LEARNING_RATE    = 1e-3
LR_DECAY_EPOCHS  = (30, 45)     # step LR down at these epoch boundaries
LR_DECAY_FACTOR  = 0.1
WEIGHT_DECAY     = 1e-5
LABEL_SMOOTHING  = 0.05
EARLY_STOP_PATIENCE = 12

# =============================================================================
# TUNABLE -- quantization
# =============================================================================

QUANT_DTYPE            = "int8"
REPRESENTATIVE_SAMPLES = 500    # clips drawn from train for the calibrator

# =============================================================================
# TUNABLE -- streaming detection policy
# =============================================================================

SMOOTH_WINDOW_MS   = 300        # posterior smoothing window (Chen et al.)
DETECT_THRESHOLD   = 0.5        # operating point; sweep this to draw the DET
REFRACTORY_MS      = 1000       # suppress re-fires inside this window
DET_THRESHOLD_GRID = 101        # points on the DET curve

SMOOTH_WINDOW_HOPS = SMOOTH_WINDOW_MS // HOP_MS
REFRACTORY_HOPS    = REFRACTORY_MS // HOP_MS

# =============================================================================
# Paths
# =============================================================================

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

POSITIVES_DIR  = DATA_DIR / "positives"
HARD_NEG_DIR   = DATA_DIR / "hard_neg"
GSC_DIR        = DATA_DIR / "gsc"
BACKGROUND_DIR = DATA_DIR / "background"

ARTIFACTS_DIR  = ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
EXPORT_DIR     = ARTIFACTS_DIR / "export"

# =============================================================================
# C export
# =============================================================================

C_PREFIX          = "kws_"
C_HEADER_NAME     = "kws_model_data.h"
C_MEL_HEADER_NAME = "kws_mel_filterbank.h"

# =============================================================================
# Self-check -- runs on import. Catches a bad edit immediately.
# =============================================================================


def _validate() -> None:
    assert CLIP_SAMPLES == 16000, CLIP_SAMPLES
    assert FRAME_LEN == 640, FRAME_LEN
    assert STRIDE_LEN == 320, STRIDE_LEN

    expected_frames = (CLIP_SAMPLES - FRAME_LEN) // STRIDE_LEN + 1
    assert NUM_FRAMES == expected_frames, (
        f"NUM_FRAMES={NUM_FRAMES} contradicts the timing constants "
        f"(expected {expected_frames})"
    )

    assert NFFT >= FRAME_LEN, (
        f"NFFT={NFFT} < FRAME_LEN={FRAME_LEN}: frames would be truncated. "
        f"Raise NFFT to {1 << (FRAME_LEN - 1).bit_length()} or lower FRAME_MS."
    )
    assert NFFT & (NFFT - 1) == 0, "NFFT must be a power of two"

    assert 0 < NUM_MFCC <= NUM_MEL, (NUM_MFCC, NUM_MEL)
    assert 0 < FMIN < FMAX <= SAMPLE_RATE // 2, (FMIN, FMAX)

    assert LABELS[SILENCE_INDEX] == "silence"
    assert LABELS[UNKNOWN_INDEX] == "unknown"
    assert LABELS[KEYWORD_INDEX] == "keyword"

    assert HOP_MS % STRIDE_MS == 0, (
        "HOP_MS must be a whole number of MFCC strides so the streaming "
        "feature ring buffer advances by whole frames"
    )
    assert HOP_MS <= CLIP_MS, (HOP_MS, CLIP_MS)
    assert SMOOTH_WINDOW_MS % HOP_MS == 0, (SMOOTH_WINDOW_MS, HOP_MS)
    assert REFRACTORY_MS % HOP_MS == 0, (REFRACTORY_MS, HOP_MS)

    assert 0.0 <= DETECT_THRESHOLD <= 1.0, DETECT_THRESHOLD
    assert VALIDATION_PERCENT + TESTING_PERCENT < 100.0


_validate()


def summary() -> str:
    """Human-readable dump of the frozen contract. Print this in every log."""
    return "\n".join([
        "KWS frozen configuration",
        "------------------------",
        f"  contract   : {FEATURE_CONTRACT_HASH}",
        f"  keyword    : {KEYWORD!r}",
        f"  audio      : {SAMPLE_RATE} Hz mono, {CLIP_MS} ms clips"
        f" ({CLIP_SAMPLES} samples)",
        f"  framing    : {FRAME_MS} ms / {STRIDE_MS} ms hop ->"
        f" {FRAME_LEN} / {STRIDE_LEN} samples, {NUM_FRAMES} frames",
        f"  fft        : {NFFT} pt ({NUM_FREQ_BINS} bins),"
        f" frames zero-padded {FRAME_LEN} -> {NFFT}",
        f"  mel        : {NUM_MEL} bands, {FMIN}-{FMAX} Hz",
        f"  features   : {FEATURE_SHAPE} = {FEATURE_SIZE} values"
        f" ({FEATURE_SIZE * 4 / 1024:.2f} KiB f32,"
        f" {FEATURE_SIZE / 1024:.2f} KiB int8)",
        f"  labels     : {LABELS}",
        f"  streaming  : {HOP_MS} ms hop = {HOP_SAMPLES} samples"
        f" = {HOP_FRAMES} new frames/hop",
        f"  detection  : smooth {SMOOTH_WINDOW_MS} ms"
        f" ({SMOOTH_WINDOW_HOPS} hops), refractory {REFRACTORY_MS} ms,"
        f" thr {DETECT_THRESHOLD}",
    ])


if __name__ == "__main__":
    print(summary())

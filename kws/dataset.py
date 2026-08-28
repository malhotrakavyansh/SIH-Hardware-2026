"""
dataset.py -- loading, splitting, augmentation.

Sources under data/:
    positives/   your own recordings of config.KEYWORD  -> label "keyword"
    hard_neg/    confusable words, near-misses          -> label "unknown"
    gsc/         Google Speech Commands                 -> label "unknown"
    background/  long continuous noise / speech         -> label "silence"
                 (also the mixing source for augmentation)

Splitting is hash-based on the speaker id so one speaker never appears in two
splits -- a random per-file split leaks speaker identity and inflates val
accuracy by several points.

Implemented in Step 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config


@dataclass(frozen=True)
class Sample:
    """One training item, before any audio is read."""

    path: Path
    label_index: int      # index into config.LABELS
    speaker: str          # used by which_set(); "" for synthetic silence
    source: str           # "positives" | "hard_neg" | "gsc" | "background"


def speaker_id(path: Path) -> str:
    """Extract the speaker id from a filename.

    GSC uses '<speaker_hash>_nohash_<n>.wav'. Own recordings should follow the
    same convention so the split logic is uniform.
    """
    raise NotImplementedError("Step 3")


def which_set(speaker: str) -> str:
    """Deterministic 'train' | 'val' | 'test' assignment for a speaker.

    Stable across dataset growth: adding files never moves an existing speaker
    between splits. Uses config.MAX_WAVS_PER_CLASS,
    config.VALIDATION_PERCENT, config.TESTING_PERCENT.
    """
    raise NotImplementedError("Step 3")


def load_wav(path: Path) -> np.ndarray:
    """Read a wav as float32 mono at config.SAMPLE_RATE.

    Raises if the file is not already at the target rate -- resampling belongs
    in a preprocessing pass, not silently in the training loop.
    """
    raise NotImplementedError("Step 3")


def fit_to_clip(samples: np.ndarray, offset: int = 0) -> np.ndarray:
    """Pad or crop to exactly config.CLIP_SAMPLES, honouring a shift offset."""
    raise NotImplementedError("Step 3")


def build_manifest() -> list[Sample]:
    """Walk data/ and produce every Sample, with labels and speakers resolved."""
    raise NotImplementedError("Step 3")


def make_splits(manifest: list[Sample]) -> dict[str, list[Sample]]:
    """Group a manifest into {'train': [...], 'val': [...], 'test': [...]}.

    Also enforces the class mix from config.SILENCE_PERCENT and
    config.UNKNOWN_PERCENT relative to the keyword count.
    """
    raise NotImplementedError("Step 3")


def random_background(rng: np.random.Generator) -> np.ndarray:
    """Random config.CLIP_SAMPLES-long excerpt from data/background/."""
    raise NotImplementedError("Step 3")


def augment(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train-time augmentation chain.

    time shift (config.TIME_SHIFT_MS) -> speed perturb
    (config.SPEED_PERTURB_PROB / _RANGE) -> gain (config.GAIN_DB_RANGE)
    -> background mix (config.BACKGROUND_PROB, config.SNR_DB_RANGE,
    config.BACKGROUND_VOL_MAX).

    Never applied to val/test.
    """
    raise NotImplementedError("Step 3")


def make_dataset(split: str, training: bool):
    """Build the batched, feature-extracted pipeline for one split.

    Yields (features, label) with features float32 config.FEATURE_SHAPE and
    batches of config.BATCH_SIZE.
    """
    raise NotImplementedError("Step 3")


if __name__ == "__main__":
    print(config.summary())

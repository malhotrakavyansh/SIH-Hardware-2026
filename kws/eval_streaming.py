"""
eval_streaming.py -- DET curve on continuous audio.

Clip-level accuracy is not the metric that matters. The metric is: how many
false accepts per hour on continuous background, at a recall high enough to be
usable. This module runs the model the way the firmware will -- a
config.CLIP_MS window sliding by config.HOP_MS -- over long recordings, and
reports false-accepts/hour against miss rate.

Implemented in Step 7.
"""

from __future__ import annotations

import numpy as np

import config


def stream_posteriors(samples: np.ndarray, model) -> np.ndarray:
    """Slide the model over a long recording.

    Args:
        samples: float32, arbitrary length, config.SAMPLE_RATE.
    Returns:
        float32 (num_hops, config.NUM_CLASSES) softmax posteriors, one row per
        config.HOP_MS.
    """
    raise NotImplementedError("Step 7")


def smooth(posteriors: np.ndarray) -> np.ndarray:
    """Moving average over config.SMOOTH_WINDOW_HOPS, causal (no lookahead --
    the firmware cannot see the future)."""
    raise NotImplementedError("Step 7")


def detect(smoothed: np.ndarray, threshold: float) -> np.ndarray:
    """Fire when the keyword posterior crosses `threshold`, then stay quiet for
    config.REFRACTORY_HOPS.

    Returns hop indices of the firings.
    """
    raise NotImplementedError("Step 7")


def det_curve(recordings, model) -> dict:
    """Sweep config.DET_THRESHOLD_GRID thresholds over labelled continuous
    audio.

    Returns {'thresholds', 'miss_rate', 'false_accepts_per_hour'}.
    """
    raise NotImplementedError("Step 7")


def main() -> None:
    print(config.summary())
    raise NotImplementedError("Step 7")


if __name__ == "__main__":
    main()

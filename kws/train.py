"""
train.py -- float training loop.

Every run prints config.summary() first and writes it next to the checkpoint,
so a checkpoint can never be separated from the feature contract it was
trained under.

Implemented in Step 5.
"""

from __future__ import annotations

import config


def set_seeds(seed: int = config.SEED) -> None:
    """Seed python / numpy / the framework RNGs."""
    raise NotImplementedError("Step 5")


def lr_schedule(epoch: int) -> float:
    """Step decay: config.LEARNING_RATE, dropped by config.LR_DECAY_FACTOR at
    each boundary in config.LR_DECAY_EPOCHS."""
    raise NotImplementedError("Step 5")


def main() -> None:
    """Train DS-CNN-S and write the best checkpoint to config.CHECKPOINT_DIR."""
    print(config.summary())
    raise NotImplementedError("Step 5")


if __name__ == "__main__":
    main()

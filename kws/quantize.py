"""
quantize.py -- full-integer post-training quantization.

config.QUANT_DTYPE everywhere: int8 weights, int8 activations, int8 input and
output tensors. A float input tensor would force the firmware to carry a
float->int8 conversion for every one of the config.FEATURE_SIZE MFCC values on
every hop, which is exactly the cost we are trying to avoid.

Implemented in Step 6.
"""

from __future__ import annotations

import config


def representative_dataset():
    """Yield config.REPRESENTATIVE_SAMPLES real training clips as features.

    Must come from the *train* split with augmentation ON -- calibrating on
    clean audio gives activation ranges that clip the moment there is noise.
    """
    raise NotImplementedError("Step 6")


def convert(checkpoint_path):
    """Float checkpoint -> fully-quantized tflite bytes."""
    raise NotImplementedError("Step 6")


def compare_float_vs_int8(checkpoint_path, tflite_bytes) -> dict:
    """Per-class accuracy delta and worst-case logit drift.

    Anything past ~1% absolute accuracy loss means the calibration set is
    wrong, not that quantization is 'just lossy'.
    """
    raise NotImplementedError("Step 6")


def main() -> None:
    print(config.summary())
    raise NotImplementedError("Step 6")


if __name__ == "__main__":
    main()

"""
model.py -- DS-CNN-S.

Depthwise-separable CNN from ARM's ML-KWS-for-MCU (Zhang et al., "Hello Edge").
The 'S' variant is the one that fits comfortably in an MCU's SRAM with room
left for the audio ring buffer.

Shape flow, all from config:
    input                 (NUM_FRAMES, NUM_MFCC, 1)      = (49, 10, 1)
    Conv2D  DSCNN_CHANNELS, DSCNN_CONV_KERNEL, stride DSCNN_CONV_STRIDE
    -> BN -> ReLU
    DSCNN_NUM_BLOCKS x [ DepthwiseConv2D DSCNN_DW_KERNEL -> BN -> ReLU
                         Conv2D 1x1 DSCNN_CHANNELS       -> BN -> ReLU ]
    GlobalAveragePooling2D
    Dropout DSCNN_DROPOUT
    Dense NUM_CLASSES (logits -- softmax lives in the loss and in
                       eval_streaming, never baked into the graph, so the
                       int8 export keeps a clean output tensor)

Implemented in Step 4.
"""

from __future__ import annotations

import config


def build_dscnn_s(input_shape: tuple[int, int] = config.FEATURE_SHAPE,
                  num_classes: int = config.NUM_CLASSES):
    """Construct the DS-CNN-S keras model. Returns an uncompiled model."""
    raise NotImplementedError("Step 4")


def param_count(model) -> int:
    """Trainable parameter count -- log this next to the flash budget."""
    raise NotImplementedError("Step 4")


def macs(model) -> int:
    """Multiply-accumulates per inference. The number that decides whether
    this runs in real time at a config.HOP_MS cadence."""
    raise NotImplementedError("Step 4")


if __name__ == "__main__":
    print(config.summary())

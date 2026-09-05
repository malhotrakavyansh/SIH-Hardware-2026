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

import tensorflow as tf
from tensorflow.keras import layers

import config


def build_dscnn_s(input_shape: tuple[int, int] = config.FEATURE_SHAPE,
                  num_classes: int = config.NUM_CLASSES):
    """Construct the DS-CNN-S keras model. Returns an uncompiled model.

    Output is raw logits -- no softmax. Softmax belongs in the training loss
    (from_logits=True) and in eval_streaming.py's posterior smoothing, not
    baked into the graph: an int8 export of a softmax output would saturate
    at the quantized dynamic range's extremes instead of preserving the
    relative confidence the streaming smoother needs.
    """
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        filters=config.DSCNN_CHANNELS,
        kernel_size=config.DSCNN_CONV_KERNEL,
        strides=config.DSCNN_CONV_STRIDE,
        padding="same",
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization(
        momentum=config.BN_MOMENTUM, epsilon=config.BN_EPSILON
    )(x)
    x = layers.ReLU()(x)

    for _ in range(config.DSCNN_NUM_BLOCKS):
        x = layers.DepthwiseConv2D(
            kernel_size=config.DSCNN_DW_KERNEL,
            strides=config.DSCNN_DW_STRIDE,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization(
            momentum=config.BN_MOMENTUM, epsilon=config.BN_EPSILON
        )(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters=config.DSCNN_CHANNELS,
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization(
            momentum=config.BN_MOMENTUM, epsilon=config.BN_EPSILON
        )(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.DSCNN_DROPOUT)(x)
    outputs = layers.Dense(num_classes)(x)  # logits

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dscnn_s")


def param_count(model) -> int:
    """Trainable parameter count -- log this next to the flash budget.

    Excludes BatchNorm's non-trainable moving mean/variance: those live in
    the graph but aren't weights the optimizer touches, and after folding
    (standard for inference export) they don't cost separate flash the way
    trainable weights do.
    """
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))

    print(f"trainable params : {trainable:,}")
    print(f"flash (float32)  : {trainable * 4 / 1024:.1f} KiB")
    print(f"flash (int8)     : {trainable * 1 / 1024:.1f} KiB")

    return trainable


def macs(model) -> int:
    """Multiply-accumulates per inference. The number that decides whether
    this runs in real time at a config.HOP_MS cadence.

    Walks the layers directly rather than using a library: only Conv2D,
    DepthwiseConv2D (depth_multiplier=1, matching the architecture above),
    and Dense actually spend MACs -- BatchNorm/ReLU/Dropout/pooling are
    elementwise or reduction ops and are ignored here.
    """
    total = 0

    for layer in model.layers:
        if isinstance(layer, layers.DepthwiseConv2D):
            _, out_h, out_w, channels = layer.output.shape
            kh, kw = layer.kernel_size
            total += out_h * out_w * kh * kw * channels

        elif isinstance(layer, layers.Conv2D):
            _, out_h, out_w, out_channels = layer.output.shape
            kh, kw = layer.kernel_size
            in_channels = layer.input.shape[-1]
            total += out_h * out_w * kh * kw * in_channels * out_channels

        elif isinstance(layer, layers.Dense):
            in_dim = layer.input.shape[-1]
            out_dim = layer.units
            total += in_dim * out_dim

    total = int(total)

    if total >= 1_000_000:
        print(f"MACs/inference   : {total:,} ({total / 1e6:.2f} M MACs/inference)")
    else:
        print(f"MACs/inference   : {total:,} ({total / 1e3:.2f} K MACs/inference)")

    return total


if __name__ == "__main__":
    print(config.summary())

    print("\n--- Step 4 smoke test ---")

    model = build_dscnn_s()
    model.summary()

    print()
    param_count(model)

    print()
    macs(model)

    print()
    dummy = tf.zeros((2,) + config.FEATURE_SHAPE + (1,), dtype=tf.float32)
    output = model(dummy, training=False)
    print(f"forward pass output shape = {output.shape}, dtype = {output.dtype}")
    assert tuple(output.shape) == (2, config.NUM_CLASSES)

    print("\nAll Step 4 checks passed.")

"""
export_c.py -- dump the mel filterbank and the quantized model as C arrays.

Two headers land in config.EXPORT_DIR:

    config.C_MEL_HEADER_NAME    the filterbank from features.mel_filterbank(),
                                so the firmware never recomputes it
    config.C_HEADER_NAME        the tflite flatbuffer as a byte array

Both carry a generated banner with the config values they were built from.
The firmware asserts those against its own constants at build time -- that
assertion is the whole reason config.py exists.

Implemented in Step 8.
"""

from __future__ import annotations

import numpy as np

import config


def c_identifier(name: str) -> str:
    """config.C_PREFIX + a sanitised name."""
    raise NotImplementedError("Step 8")


def format_banner() -> str:
    """Comment block naming every frozen config value baked into this header."""
    raise NotImplementedError("Step 8")


def dump_float_array(name: str, array: np.ndarray) -> str:
    """Flatten to a `static const float name[] = {...};` definition."""
    raise NotImplementedError("Step 8")


def dump_byte_array(name: str, data: bytes) -> str:
    """`static const unsigned char name[] = {...};`, aligned for the tflite
    interpreter."""
    raise NotImplementedError("Step 8")


def export_mel_filterbank() -> None:
    """Write config.C_MEL_HEADER_NAME."""
    raise NotImplementedError("Step 8")


def export_model(tflite_path) -> None:
    """Write config.C_HEADER_NAME."""
    raise NotImplementedError("Step 8")


def main() -> None:
    print(config.summary())
    raise NotImplementedError("Step 8")


if __name__ == "__main__":
    main()

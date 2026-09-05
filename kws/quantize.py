"""
quantize.py -- full-integer post-training quantization.

config.QUANT_DTYPE everywhere: int8 weights, int8 activations, int8 input and
output tensors. A float input tensor would force the firmware to carry a
float->int8 conversion for every one of the config.FEATURE_SIZE MFCC values on
every hop, which is exactly the cost we are trying to avoid.

Implemented in Step 6.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

import config
import dataset
import features

RUN_NAME = "nakshatra_mvp_v1"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _clip_to_features(path: Path) -> np.ndarray:
    """Same preprocessing as training, minus augmentation: load -> fit_to_clip
    -> extract_mfcc -> (NUM_FRAMES, NUM_MFCC, 1) float32."""
    clip = dataset.fit_to_clip(dataset.load_wav(path))
    pcm16 = np.clip(np.round(clip * 32768.0), -32768, 32767).astype(np.int16)
    return features.extract_mfcc(pcm16)[..., np.newaxis].astype(np.float32)


def build_representative_dataset(
    num_samples: int = config.REPRESENTATIVE_SAMPLES,
) -> Iterable:
    """Representative dataset for calibrating INT8 quantization scales.

    Deliberately drawn only from data/positives/ and data/hard_neg/ (our own
    recordings), never GSC -- the quantization scales are set from whatever
    activation distribution this dataset produces, and calibrating against a
    distribution the deployed model won't actually see costs accuracy in a
    way that looks exactly like a model bug.

    Returns a generator; TFLiteConverter calls this function itself (not the
    generator) each time it needs a fresh pass, so re-invoking reproduces the
    same config.SEED-seeded sample selection.
    """
    files = (
        sorted(config.POSITIVES_DIR.glob("*.wav"))
        + sorted(config.HARD_NEG_DIR.glob("*.wav"))
    )
    rng = np.random.default_rng(config.SEED)
    n = min(num_samples, len(files))
    chosen = [files[i] for i in sorted(rng.choice(len(files), size=n, replace=False))]

    def _generator():
        for path in chosen:
            feats = _clip_to_features(path)
            yield [feats[np.newaxis, ...]]  # (1, NUM_FRAMES, NUM_MFCC, 1)

    return _generator()


def quantize_model(model_path: Path, out_dir: Path) -> Path:
    """Float keras checkpoint -> full-integer int8 .tflite.

    Writes model_int8.tflite, quant_params.json (the input/output scale and
    zero_point the firmware needs to quantize its MFCC input and dequantize
    the output logits), and copies the checkpoint's config_summary.txt
    alongside -- a quantized model must carry its feature contract with it
    exactly like the float checkpoint does.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)

    # TFLiteConverter.from_keras_model() crashes on this TF 2.16 / Keras 3
    # combination (LLVM "missing attribute 'value'" while tracing
    # BatchNormalization's moving-variance ReadVariableOp) -- routing through
    # a SavedModel export first sidesteps the broken codepath and converts
    # cleanly.
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"
        model.export(str(saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = build_representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_bytes = converter.convert()

    tflite_path = out_dir / "model_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    quant_params = {
        "input": {
            "scale": float(in_scale),
            "zero_point": int(in_zero),
            "dtype": str(input_details["dtype"].__name__),
            "shape": [int(d) for d in input_details["shape"]],
        },
        "output": {
            "scale": float(out_scale),
            "zero_point": int(out_zero),
            "dtype": str(output_details["dtype"].__name__),
            "shape": [int(d) for d in output_details["shape"]],
        },
        "labels": config.LABELS,
        "feature_contract_hash": config.FEATURE_CONTRACT_HASH,
    }
    quant_params_path = out_dir / "quant_params.json"
    quant_params_path.write_text(json.dumps(quant_params, indent=2), encoding="utf-8")

    src_summary = model_path.parent / "config_summary.txt"
    dst_summary = out_dir / "config_summary.txt"
    if src_summary.exists() and src_summary.resolve() != dst_summary.resolve():
        shutil.copy(src_summary, dst_summary)

    print(f"wrote {tflite_path} ({tflite_path.stat().st_size:,} bytes)")
    print(f"wrote {quant_params_path}")
    print(f"  input  scale={in_scale:.6f} zero_point={in_zero}")
    print(f"  output scale={out_scale:.6f} zero_point={out_zero}")

    return tflite_path


def _predict_keyword_prob_tflite(interpreter, input_details, output_details,
                                  path: Path) -> float:
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    feats = _clip_to_features(path)
    q = np.round(feats / in_scale + in_zero)
    q = np.clip(q, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details["index"], q[np.newaxis, ...])
    interpreter.invoke()
    out_q = interpreter.get_tensor(output_details["index"])[0]
    logits = (out_q.astype(np.float32) - out_zero) * out_scale

    return float(_softmax(logits)[config.KEYWORD_INDEX])


def evaluate_tflite(tflite_path: Path, positives_dir: Path, hardneg_dir: Path) -> dict:
    """Same file-level probe as sanity_check.py, but running the quantized
    .tflite through tf.lite.Interpreter (manual int8 quantize-in /
    dequantize-out) instead of the float Keras model.

    A drop of more than ~1-2% absolute vs. the float sanity check means the
    representative dataset is miscalibrated, not that "quantization is just
    lossy".
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    def predict(path: Path) -> float:
        return _predict_keyword_prob_tflite(interpreter, input_details, output_details, path)

    positive_files = sorted(Path(positives_dir).glob("*.wav"))
    hardneg_files = sorted(Path(hardneg_dir).glob("*.wav"))

    pos_probs = {p.name: predict(p) for p in positive_files}
    neg_probs = {p.name: predict(p) for p in hardneg_files}

    true_positives = [n for n, v in pos_probs.items() if v > 0.5]
    true_negatives = [n for n, v in neg_probs.items() if v <= 0.5]
    false_positives = [(n, v) for n, v in neg_probs.items() if v > 0.5]

    pos_arr = np.array(list(pos_probs.values())) if pos_probs else np.array([0.0])
    neg_arr = np.array(list(neg_probs.values())) if neg_probs else np.array([0.0])

    result = {
        "positive_count": len(pos_probs),
        "hardneg_count": len(neg_probs),
        "true_positive_rate": len(true_positives) / max(len(pos_probs), 1),
        "true_negative_rate": len(true_negatives) / max(len(neg_probs), 1),
        "positive_prob": {
            "min": float(pos_arr.min()), "max": float(pos_arr.max()),
            "mean": float(pos_arr.mean()),
        },
        "hardneg_prob": {
            "min": float(neg_arr.min()), "max": float(neg_arr.max()),
            "mean": float(neg_arr.mean()),
        },
        "false_positives": false_positives,
    }

    print(f"[int8] TPR (positives -> keyword): "
          f"{len(true_positives)}/{len(pos_probs)} "
          f"({100 * result['true_positive_rate']:.1f}%)")
    print(f"[int8] TNR (hard_neg -> unknown)  : "
          f"{len(true_negatives)}/{len(neg_probs)} "
          f"({100 * result['true_negative_rate']:.1f}%)")
    print(f"[int8] positive prob : min={pos_arr.min():.3f} "
          f"max={pos_arr.max():.3f} mean={pos_arr.mean():.3f}")
    print(f"[int8] hard_neg prob : min={neg_arr.min():.3f} "
          f"max={neg_arr.max():.3f} mean={neg_arr.mean():.3f}")

    if false_positives:
        print(f"[int8] FALSE POSITIVES ({len(false_positives)}):")
        for name, prob in sorted(false_positives, key=lambda r: -r[1]):
            print(f"    {name:35s} keyword_prob={prob:.3f}")
    else:
        print("[int8] no false positives")

    return result


def export_c_array(tflite_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """.tflite flatbuffer -> a C header + source pair the firmware includes
    directly, xxd -i style but with an explicit alignas(8) for the MCU's
    alignment requirements (xxd's own output has none)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = tflite_path.read_bytes()

    header_path = out_dir / "model_data.h"
    source_path = out_dir / "model_data.cc"
    guard = header_path.stem.upper() + "_H_"

    header = "\n".join([
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "extern const unsigned char g_model[];",
        "extern const unsigned int g_model_len;",
        "",
        f"#endif  /* {guard} */",
        "",
    ])
    header_path.write_text(header, encoding="utf-8")

    hex_bytes = [f"0x{b:02x}" for b in data]
    lines = [", ".join(hex_bytes[i:i + 12]) for i in range(0, len(hex_bytes), 12)]
    body = ",\n".join(lines)

    source = "\n".join([
        f'#include "{header_path.name}"',
        "",
        "alignas(8) const unsigned char g_model[] = {",
        body + ",",
        "};",
        "",
        f"const unsigned int g_model_len = {len(data)};",
        "",
    ])
    source_path.write_text(source, encoding="utf-8")

    print(f"wrote {header_path} ({header_path.stat().st_size:,} bytes)")
    print(f"wrote {source_path} ({source_path.stat().st_size:,} bytes, "
          f"{len(data):,} model bytes)")

    return header_path, source_path


def _evaluate_keras(model, positives_dir: Path, hardneg_dir: Path) -> dict:
    """Float-model counterpart of evaluate_tflite(), used only for the
    float-vs-int8 comparison printed in main()."""
    def predict(path: Path) -> float:
        feats = _clip_to_features(path)
        logits = model(feats[np.newaxis, ...], training=False).numpy()[0]
        return float(_softmax(logits)[config.KEYWORD_INDEX])

    positive_files = sorted(Path(positives_dir).glob("*.wav"))
    hardneg_files = sorted(Path(hardneg_dir).glob("*.wav"))

    pos_probs = [predict(p) for p in positive_files]
    neg_probs = [predict(p) for p in hardneg_files]

    tp = sum(1 for v in pos_probs if v > 0.5)
    tn = sum(1 for v in neg_probs if v <= 0.5)

    return {
        "true_positive_rate": tp / max(len(pos_probs), 1),
        "true_negative_rate": tn / max(len(neg_probs), 1),
    }


def main() -> None:
    print(config.summary())

    checkpoint_dir = config.CHECKPOINT_DIR / RUN_NAME
    float_path = checkpoint_dir / "float.keras"
    out_dir = checkpoint_dir

    print(f"\nQuantizing {float_path} ...\n")
    tflite_path = quantize_model(float_path, out_dir)

    float_size = float_path.stat().st_size
    int8_size = tflite_path.stat().st_size

    print("\n--- Evaluating float model (for comparison) ---")
    float_model = tf.keras.models.load_model(float_path, compile=False)
    float_result = _evaluate_keras(float_model, config.POSITIVES_DIR, config.HARD_NEG_DIR)
    print(f"[float] TPR: {100 * float_result['true_positive_rate']:.1f}%   "
          f"TNR: {100 * float_result['true_negative_rate']:.1f}%")

    print("\n--- Evaluating int8 model ---")
    int8_result = evaluate_tflite(tflite_path, config.POSITIVES_DIR, config.HARD_NEG_DIR)

    print("\n--- Exporting C arrays ---")
    header_path, source_path = export_c_array(tflite_path, out_dir)

    print("\n=== Summary ===")
    print(f"float model size : {float_size / 1024:.1f} KiB")
    print(f"int8 model size  : {int8_size / 1024:.1f} KiB "
          f"({float_size / int8_size:.1f}x smaller)")
    print(
        f"positive TPR     : float={100 * float_result['true_positive_rate']:.1f}%  "
        f"int8={100 * int8_result['true_positive_rate']:.1f}%  "
        f"(delta={100 * (float_result['true_positive_rate'] - int8_result['true_positive_rate']):+.1f}pp)"
    )
    print(
        f"hard_neg TNR     : float={100 * float_result['true_negative_rate']:.1f}%  "
        f"int8={100 * int8_result['true_negative_rate']:.1f}%  "
        f"(delta={100 * (float_result['true_negative_rate'] - int8_result['true_negative_rate']):+.1f}pp)"
    )
    print("\nFiles created:")
    for p in (
        tflite_path,
        out_dir / "quant_params.json",
        header_path,
        source_path,
        out_dir / "config_summary.txt",
    ):
        print(f"  {p}")


if __name__ == "__main__":
    main()

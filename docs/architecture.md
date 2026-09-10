# System Architecture

## High-level flow

```text
Microphone
  |
  v
MFCC feature extraction (src/kws/features.py)
  |
  v
DS-CNN-S model (src/kws/model.py, trained via src/kws/train.py)
  |
  v
INT8 post-training quantization (src/kws/quantize.py)
  |
  v
C export -- model + mel filterbank as C arrays (src/kws/export_c.py)
  |
  v
Firmware / MCU inference
  |
  v
Streaming detection: posterior smoothing (k-of-window) + refractory period
(src/kws/eval_streaming.py contract; src/live_demo.py runs the same logic
against a live mic for real-time demos)
  |
  v
Wake event
```

## Components

### `src/kws/config.py` — single source of truth
Every frozen (feature/model I/O) and tunable (training/detection) constant
lives here. `config.FEATURE_CONTRACT_HASH` stamps each checkpoint and
exported header so a model can never silently run against a front end it
wasn't trained on.

### `src/kws/features.py` — MFCC front end
Explicit array ops (framing, symmetric Hann window, FFT, mel filterbank,
log, DCT-II) instead of an opaque library call, so the firmware MFCC port
can be checked bit-for-bit against Python's output
(`src/kws/check_parity.py`, `src/kws/make_test_vectors.py`).

### `src/kws/dataset.py`, `src/kws/train.py` — data + training
Loads recorded wake-word positives, hard negatives, and background audio
(`src/kws/data/`), builds train/val/test splits, and trains the DS-CNN-S
model (`src/kws/model.py`).

### `src/kws/quantize.py` — INT8 post-training quantization
Converts the trained float model to INT8 using representative samples from
the training set, for MCU-sized inference.

### `src/kws/export_c.py` — firmware export
Emits the quantized model and mel filterbank as C arrays
(`model_data.cc`/`.h`) for direct inclusion in firmware.

### `src/kws/eval_streaming.py` — streaming evaluation
Runs the full detection policy (posterior smoothing window, k-of-window
threshold, refractory period) over continuous audio and produces a DET
curve to pick the operating threshold.

### `src/live_demo.py` — real-time demo
Runs the same feature extraction + streaming detection pipeline against a
live microphone feed, with configurable threshold/k/window, for live
demonstrations.

## Firmware / hardware

[TODO: fill — MCU/board used, sensor/mic hardware, how the exported C model
is integrated into firmware, any additional hardware components]

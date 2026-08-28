# kws — on-device keyword spotting

Wake-word detector for the SIH Hardware 2026 build. DS-CNN-S trained in
Python, quantized to int8, exported as C arrays for the MCU.

The whole repo is organised around one rule: **every number lives in
[`config.py`](config.py)**. The training script, the quantizer, the streaming
evaluator and the firmware all read the same constants, so a model can never
be run against a front end it wasn't trained on.

## Layout

```
kws/
├── config.py          # SINGLE source of truth
├── features.py        # MFCC extraction              (Step 2)
├── dataset.py         # loading, splitting, augmentation (Step 3)
├── model.py           # DS-CNN-S                     (Step 4)
├── train.py           #                              (Step 5)
├── quantize.py        # int8 PTQ                     (Step 6)
├── eval_streaming.py  # DET curve on continuous audio (Step 7)
├── export_c.py        # mel filterbank + model as C arrays (Step 8)
├── verify_step1.py    # enforces the Step 1 criterion
└── data/
    ├── positives/     # your recordings of the wake word
    ├── hard_neg/      # confusable words
    ├── gsc/           # Google Speech Commands
    └── background/    # long continuous negatives
```

Each `data/` subfolder has its own README describing what belongs in it and
in what format.

## Status

| Step | What | State |
|------|------|-------|
| 1 | Repo + frozen config | **done** |
| 2 | MFCC front end, Python↔C parity | not started |
| 3 | Dataset, splits, augmentation | not started |
| 4 | DS-CNN-S | not started |
| 5 | Float training | not started |
| 6 | int8 quantization | not started |
| 7 | Streaming DET evaluation | not started |
| 8 | C export | not started |

Steps 2–8 exist as modules with fixed signatures, docstrings that pin down
every tensor shape, and `NotImplementedError("Step N")` bodies. Nothing to
redesign later — just fill them in.

## Running

Step 1 is stdlib-only, no install needed:

```bash
python kws/verify_step1.py
```

That prints the frozen contract and checks the Step 1 done-criterion: config
exists, every module imports it, no frozen value is hardcoded anywhere else.

Print the contract on its own:

```bash
python kws/config.py
```

## The frozen contract

`config.py` splits into **FROZEN** (feature/model I/O — must match the
firmware bit for bit) and **TUNABLE** (training recipe, detection policy —
sweep freely). Once Step 2 passes, changing anything in FROZEN invalidates
every checkpoint and every exported header.

`config.FEATURE_CONTRACT_HASH` is a short SHA-256 digest over just the FROZEN
values. `config.summary()` prints it, and later steps stamp it into each
checkpoint and each exported header so a model can't be loaded or flashed
against a front end it wasn't built for — the mismatch surfaces as a hash
compare, not a silently wrong detector.

`config._validate()` runs on import and asserts the invariants — derived frame
count, `NFFT >= FRAME_LEN`, `HOP_MS` being a whole number of strides,
label-index agreement. A bad edit fails at import, not 40 minutes into a
training run.

### One change from the original spec: `NFFT` is 1024, not 512

A 40 ms frame at 16 kHz is 640 samples. With `NFFT = 512` every frame gets
truncated and ~20% of each window is silently discarded — and the Python and C
front ends would have to agree on *how* they truncate, which is exactly the
class of bug Step 2 exists to catch. 1024 is the next power of two above 640,
and matches ARM's reference DS-CNN front end (`mfcc.cpp`, `m_frameLenPadded`),
which pads 640 → 1024.

If you want 512 for the MCU's FFT cost, set `FRAME_MS = 32` so the frame is
exactly 512 samples. Do not pair `FRAME_MS = 40` with `NFFT = 512`; the
assertion in `_validate()` blocks it.

## Environment

TensorFlow has no wheels for Python 3.14 (the interpreter currently on this
machine). Build the venv on 3.11:

```bash
py -3.11 -m venv .venv
```

Then `.venv\Scripts\activate` and `pip install -r kws/requirements.txt`.

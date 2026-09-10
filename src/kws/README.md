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
├── features.py        # MFCC extraction                     (Step 2, done)
├── export_c.py        # feature tables + model as C arrays  (Step 2 half done)
├── make_test_vectors.py  # 3 synthetic clips -> wav + C input + ref MFCC
├── check_parity.py       # diffs a firmware MFCC dump against the reference
├── dataset.py          # loading, splitting, augmentation   (Step 3)
├── model.py            # DS-CNN-S                           (Step 4)
├── train.py            #                                    (Step 5)
├── quantize.py         # int8 PTQ                           (Step 6)
├── eval_streaming.py   # DET curve on continuous audio      (Step 7)
├── verify_step1.py     # enforces the Step 1 criterion
├── test_vectors/       # generated -- see make_test_vectors.py
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
| 2 | MFCC front end, Python↔C parity | **Python side done** — waiting on firmware dump to run the actual parity check |
| 3 | Dataset, splits, augmentation | not started |
| 4 | DS-CNN-S | not started |
| 5 | Float training | not started |
| 6 | int8 quantization | not started |
| 7 | Streaming DET evaluation | not started |
| 8 | C export (model) | not started |

Steps 3–8 exist as modules with fixed signatures, docstrings that pin down
every tensor shape, and `NotImplementedError("Step N")` bodies. Nothing to
redesign later — just fill them in.

## Step 2 — feature extractor + C parity

`features.py` implements the MFCC front end as explicit array ops (framing,
Hann window, FFT, mel filterbank, log, DCT-II) — no `librosa.mfcc()` call
whose padding/normalization defaults you can't inspect. Everything after the
FFT is two fixed matrices (`build_mel_filterbank()`, `build_dct_matrix()`)
that depend only on `config`, never on audio — so the firmware never
re-derives a mel scale on the MCU, it just does a matmul against Python's own
numbers.

```bash
python kws/features.py          # self-test: shapes, DCT orthonormality, finite output
python kws/export_c.py          # writes artifacts/export/kws_mel_filterbank.h
python kws/make_test_vectors.py # writes test_vectors/: 3 clips, wav + C input + ref MFCC
```

Two things a firmware port has to match exactly (see `features.py`'s module
docstring for why): the **symmetric** Hann window (`N-1` denominator, not
periodic `N`), and **unnormalized** `|FFT|^2` (no `1/N` scaling).

**Running the actual parity test** (once the firmware side has an MFCC
extractor to test): feed `test_vectors/<clip>_input.h` in as a hardcoded C
array — never through a mic/speaker for this check, so a failure can only be
a math bug — print the firmware's `(49, 10)` output over serial, then:

```bash
python kws/check_parity.py tone_440hz  path/to/esp32_dump.csv
python kws/check_parity.py silence     path/to/esp32_dump.csv
python kws/check_parity.py white_noise path/to/esp32_dump.csv
```

Done when all 3 print `PASS` (max abs diff < 0.05). This unblocks Steps 3–8.

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

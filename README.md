# SIH Hardware 2026 — Wake-Word Keyword Spotting

## 1. Project Information

- **Project Title:** [TODO: fill — official project/product name]
- **PS ID:** [TODO: fill]
- **PS Title:** [TODO: fill]
- **Category:** Hardware
- **Theme:** [TODO: fill]
- **Team Members & Roles:** [TODO: fill — name, role, for each team member]

## 2. Problem Statement

[TODO: fill — the problem this project addresses]

## 3. Proposed Solution

On-device wake-word (keyword spotting) detector: a DS-CNN-S model trained in
Python, quantized to INT8, and exported as C arrays for deployment on an
MCU/firmware target. See [docs/architecture.md](docs/architecture.md) for
the full pipeline.

## 4. Key Features

- Custom wake-word detection trained on real recorded sessions
- MFCC feature extraction with Python↔C numerical parity checks
- INT8 quantization for on-device (MCU) inference
- Streaming evaluation (DET curve, k-of-window smoothing, refractory logic)
- `src/live_demo.py` — real-time microphone demo with configurable
  threshold/k/window
- [TODO: fill in any hardware-specific features — sensors, enclosure, etc.]

## 5. Technology Stack

- Python (TensorFlow, NumPy, SciPy, scikit-learn) for model training/quantization
- C (exported model + MFCC front end) for firmware/MCU deployment
- [TODO: fill in MCU/board, sensors, and any other hardware/firmware stack details]

## 6. Architecture

See [docs/architecture.md](docs/architecture.md).

```text
Microphone
  |
  v
MFCC feature extraction (src/kws/features.py)
  |
  v
DS-CNN-S model (src/kws/model.py) -- trained (src/kws/train.py)
  |
  v
INT8 quantization (src/kws/quantize.py)
  |
  v
C export (src/kws/export_c.py) -> firmware / MCU
  |
  v
Streaming detection (smoothing + refractory) -> wake event
```

## 7. Repository Structure

```text
SIH-Hardware-2026/
├── README.md
├── SUBMISSION_GUIDE.md
├── submission/
│   ├── PRESENTATION.md
│   └── DEMO.md
├── src/
│   ├── live_demo.py
│   ├── sanity_check.py
│   ├── split_positives.py
│   ├── split_hardneg.py
│   ├── generate_synthetic_background.py
│   └── kws/                # ML pipeline: config, features, model, train,
│                            # quantize, eval_streaming, export_c, data/,
│                            # artifacts/, test_vectors/ — see src/kws/README.md
├── docs/
│   └── architecture.md
├── assets/
│   └── screenshots/
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 8. Final Presentation

See [submission/PRESENTATION.md](submission/PRESENTATION.md).

## 9. Demo Video

See [submission/DEMO.md](submission/DEMO.md).

## 10. Screenshots / Prototype Photos

See [assets/screenshots/](assets/screenshots/).

## 11. Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd SIH-Hardware-2026
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12. Run

```bash
# Step 1 sanity check (stdlib-only)
python src/kws/verify_step1.py

# Feature-extractor self-test
python src/kws/features.py

# Real-time microphone demo
python src/live_demo.py
```

See [src/kws/README.md](src/kws/README.md) for the full ML pipeline
walkthrough (steps 1–8).

## 13. Future Scope

[TODO: fill — realistic next steps / extensions]

## Important

Before submission, make sure the repository is accessible to reviewers. Do
**not** upload passwords, API keys, access tokens, `.env` files containing
secrets, or other confidential credentials.

"""
sanity_check.py -- standalone probe of keyword-vs-confusable discrimination.

The random_per_file test split (train.py) ended up with zero hard-negative
samples, so it never actually validated the thing that matters for a wake
word: does the model reject confusable words? This script runs the trained
float checkpoint over every file in data/positives/ and data/hard_neg/
directly (no held-out split, no augmentation) to answer that.

Not part of the Step 1-7 pipeline -- a diagnostic, run manually:
    python sanity_check.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "kws"))

import config
import dataset
import features

CHECKPOINT = config.CHECKPOINT_DIR / "nakshatra_mvp_v1" / "float.keras"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def hard_neg_word(path: Path) -> str:
    """'<speaker>_<word>_<n>.wav' -> '<word>'."""
    stem = path.stem
    match = re.match(r"^[^_]+_(.+)_\d+$", stem)
    return match.group(1) if match else "unknown"


def predict_keyword_prob(model, path: Path) -> float:
    clip = dataset.fit_to_clip(dataset.load_wav(path))
    pcm16 = np.clip(np.round(clip * 32768.0), -32768, 32767).astype(np.int16)
    feats = features.extract_mfcc(pcm16)[..., np.newaxis]  # (49, 10, 1)
    logits = model(feats[np.newaxis, ...], training=False).numpy()[0]  # (3,)
    probs = softmax(logits)
    return float(probs[config.KEYWORD_INDEX])


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT} not found -- run kws/train.py first."
        )

    # compile=False: this is inference-only, and the checkpoint's compile
    # config references train.py's custom metric classes (_KeywordPrecision/
    # _KeywordRecall), which aren't registered for deserialization here.
    model = tf.keras.models.load_model(CHECKPOINT, compile=False)

    positive_files = sorted(config.POSITIVES_DIR.glob("*.wav"))
    hard_neg_files = sorted(config.HARD_NEG_DIR.glob("*.wav"))

    results = []  # (filename, true_label, pred_label, keyword_prob)

    for path in positive_files:
        prob = predict_keyword_prob(model, path)
        pred_label = "keyword" if prob > 0.5 else "unknown"
        results.append((path.name, "keyword", pred_label, prob))

    for path in hard_neg_files:
        prob = predict_keyword_prob(model, path)
        pred_label = "keyword" if prob > 0.5 else "unknown"
        results.append((path.name, "unknown", pred_label, prob))

    positives = [r for r in results if r[1] == "keyword"]
    hard_negs = [r for r in results if r[1] == "unknown"]

    true_positives = [r for r in positives if r[2] == "keyword"]
    false_negatives = [r for r in positives if r[2] == "unknown"]
    true_negatives = [r for r in hard_negs if r[2] == "unknown"]
    false_positives = [r for r in hard_negs if r[2] == "keyword"]

    print(f"positives  : {len(positives)} files")
    print(f"hard_neg   : {len(hard_negs)} files")

    print(
        f"\nTrue positive rate (positives -> keyword) : "
        f"{len(true_positives)}/{len(positives)} "
        f"({100 * len(true_positives) / max(len(positives), 1):.1f}%)"
    )
    print(
        f"True negative rate (hard_neg -> unknown)  : "
        f"{len(true_negatives)}/{len(hard_negs)} "
        f"({100 * len(true_negatives) / max(len(hard_negs), 1):.1f}%)"
    )

    print(f"\nFALSE POSITIVES (hard_neg mistaken for keyword, prob > 0.5): "
          f"{len(false_positives)}")
    for name, _, _, prob in sorted(false_positives, key=lambda r: -r[3]):
        print(f"  {name:35s} keyword_prob={prob:.3f}")

    print(f"\nFALSE NEGATIVES (positives mistaken for unknown, prob < 0.5): "
          f"{len(false_negatives)}")
    for name, _, _, prob in sorted(false_negatives, key=lambda r: r[3]):
        print(f"  {name:35s} keyword_prob={prob:.3f}")

    pos_probs = np.array([r[3] for r in positives])
    print(f"\nKeyword probability distribution -- positives:")
    print(
        f"  min={pos_probs.min():.3f}  max={pos_probs.max():.3f}  "
        f"mean={pos_probs.mean():.3f}"
    )

    print(f"\nKeyword probability distribution -- hard_neg, by word:")
    words = sorted(set(hard_neg_word(Path(r[0])) for r in hard_negs))
    for word in words:
        word_probs = np.array([
            r[3] for r in hard_negs if hard_neg_word(Path(r[0])) == word
        ])
        if word_probs.size == 0:
            continue
        print(
            f"  {word:12s} n={word_probs.size:3d}  "
            f"min={word_probs.min():.3f}  max={word_probs.max():.3f}  "
            f"mean={word_probs.mean():.3f}"
        )

    # -- silence-class checks (added after the Step 7 streaming eval exposed
    # that the model had never seen a silence-labeled training example) --
    print("\n--- Silence-class checks ---")

    silence_clip = np.zeros(config.CLIP_SAMPLES, dtype=np.int16)
    silence_feats = features.extract_mfcc(silence_clip)[..., np.newaxis]
    silence_logits = model(silence_feats[np.newaxis, ...], training=False).numpy()[0]
    silence_prob = float(softmax(silence_logits)[config.KEYWORD_INDEX])
    print(f"pure silence (all zeros)     keyword_prob={silence_prob:.3f}  "
          f"(expect <0.1)")

    rng = np.random.default_rng(config.SEED)
    background_files = sorted(config.BACKGROUND_DIR.glob("*.wav"))
    sample_bg = [background_files[i] for i in
                 sorted(rng.choice(len(background_files), size=min(3, len(background_files)), replace=False))]
    print("real background samples:")
    for path in sample_bg:
        prob = predict_keyword_prob(model, path)
        print(f"  {path.name:35s} keyword_prob={prob:.3f}  (expect <0.1)")

    sample_pos = [positive_files[i] for i in
                  sorted(rng.choice(len(positive_files), size=min(3, len(positive_files)), replace=False))]
    print("random positives:")
    for path in sample_pos:
        prob = predict_keyword_prob(model, path)
        print(f"  {path.name:35s} keyword_prob={prob:.3f}  (expect >0.8)")

    sample_neg = [hard_neg_files[i] for i in
                  sorted(rng.choice(len(hard_neg_files), size=min(3, len(hard_neg_files)), replace=False))]
    print("random hard negatives:")
    for path in sample_neg:
        prob = predict_keyword_prob(model, path)
        print(f"  {path.name:35s} keyword_prob={prob:.3f}  (expect ~0.3 mean)")


if __name__ == "__main__":
    main()

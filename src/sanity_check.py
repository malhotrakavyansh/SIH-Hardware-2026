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

import argparse
import re
from pathlib import Path

import numpy as np
import scipy.signal
import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "kws"))

import config
import dataset
import features

DEFAULT_RUN_NAME = "nakshatra_mvp_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyword-vs-confusable discrimination probe.")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME,
                         help="checkpoint subfolder under config.CHECKPOINT_DIR (default: %(default)s)")
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def hard_neg_word(path: Path) -> str:
    """'<speaker>_<word>_<n>.wav' -> '<word>'."""
    stem = path.stem
    match = re.match(r"^[^_]+_(.+)_\d+$", stem)
    return match.group(1) if match else "unknown"


def predict_clip_probs(model, clip: np.ndarray) -> np.ndarray:
    """Full [silence, unknown, keyword] posterior for one CLIP_SAMPLES float clip."""
    pcm16 = np.clip(np.round(clip * 32768.0), -32768, 32767).astype(np.int16)
    feats = features.extract_mfcc(pcm16)[..., np.newaxis]  # (49, 10, 1)
    logits = model(feats[np.newaxis, ...], training=False).numpy()[0]  # (3,)
    return softmax(logits)


def predict_keyword_prob(model, path: Path) -> float:
    clip = dataset.fit_to_clip(dataset.load_wav(path))
    return float(predict_clip_probs(model, clip)[config.KEYWORD_INDEX])


# -- live-realism probe -------------------------------------------------------
# Clean positives are the clips the model trained on (or siblings from the
# same session), so 0.99 on them says nothing about the live mic. These
# conditions push positives toward what a live mic actually delivers: a
# different level, a different microphone's spectral tilt (the one that
# actually broke nakshatra_v3 live -- see _low_shelf), room noise at low SNR,
# and an off-centre word. Collapse here
# on a model that aces the clean clips = memorised, predicts live failure.
REALISM_NUM_POSITIVES = 12
REALISM_GAINS_DB = (-30.0, -20.0, -12.0, 0.0, 6.0, 12.0)
REALISM_SNRS_DB = (10.0, 5.0, 0.0)
REALISM_SHIFT_MS = 250


def _mix_at_snr(clip: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    """Uncapped SNR mix -- unlike dataset.augment(), which caps noise at
    BACKGROUND_VOL_MAX and so never actually reaches the low SNRs a live
    room can produce."""
    bg_rms = np.sqrt(np.mean(background ** 2))
    if bg_rms == 0:
        return clip
    scale = np.sqrt(np.mean(clip ** 2)) / (bg_rms * 10.0 ** (snr_db / 20.0))
    return np.clip(clip + scale * background, -1.0, 1.0).astype(np.float32)


def _low_shelf(clip: np.ndarray, gain_db: float, cutoff_hz: float = 500.0) -> np.ndarray:
    """RBJ low-shelf biquad. The live Jabra headset chain carries ~+10 dB
    more energy below 500 Hz (relative to 500-1000 Hz) than any training
    session -- close-talk proximity effect / headset DSP. That tilt plus the
    ~15 dB hotter level is what took nakshatra_v3 from 0.998 to 0.16 on
    held-out positives, reproducing its live failure offline."""
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * cutoff_hz / config.SAMPLE_RATE
    cos_w0 = np.cos(w0)
    two_sqrt_a_alpha = 2.0 * np.sqrt(a_gain) * np.sin(w0) / 2.0 * np.sqrt(2.0)
    b = [a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 + two_sqrt_a_alpha),
         2 * a_gain * ((a_gain - 1) - (a_gain + 1) * cos_w0),
         a_gain * ((a_gain + 1) - (a_gain - 1) * cos_w0 - two_sqrt_a_alpha)]
    a = [(a_gain + 1) + (a_gain - 1) * cos_w0 + two_sqrt_a_alpha,
         -2 * ((a_gain - 1) + (a_gain + 1) * cos_w0),
         (a_gain + 1) + (a_gain - 1) * cos_w0 - two_sqrt_a_alpha]
    return np.clip(scipy.signal.lfilter(b, a, clip), -1.0, 1.0).astype(np.float32)


def _to_level(clip: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(clip.astype(np.float64) ** 2))
    if rms == 0:
        return clip
    return np.clip(clip * (10.0 ** (target_dbfs / 20.0) / rms), -1.0, 1.0).astype(np.float32)


def _shift(clip: np.ndarray, offset: int) -> np.ndarray:
    padded = np.pad(clip, (abs(offset), abs(offset)), mode="constant")
    return dataset.fit_to_clip(padded, offset=offset)


def live_realism_check(model, positive_files: list[Path], rng: np.random.Generator) -> None:
    print("\n--- Live-realism check (augmented positives) ---")
    picks = sorted(rng.choice(len(positive_files),
                              size=min(REALISM_NUM_POSITIVES, len(positive_files)),
                              replace=False))
    clips = [dataset.fit_to_clip(dataset.load_wav(positive_files[i])) for i in picks]

    def report(name: str, variants: list[np.ndarray]) -> None:
        probs = np.array([predict_clip_probs(model, v) for v in variants])
        kw = probs[:, config.KEYWORD_INDEX]
        print(f"  {name:32s} kw mean={kw.mean():.3f} min={kw.min():.3f}  "
              f"detected(>0.5)={int((kw > 0.5).sum())}/{len(kw)}  "
              f"unknown mean={probs[:, config.UNKNOWN_INDEX].mean():.3f}")

    report("clean", clips)
    for gain_db in REALISM_GAINS_DB:
        g = 10.0 ** (gain_db / 20.0)
        report(f"gain {gain_db:+.0f} dB", [np.clip(c * g, -1.0, 1.0) for c in clips])

    shift = config.SAMPLE_RATE * REALISM_SHIFT_MS // 1000
    report(f"time shift +/-{REALISM_SHIFT_MS} ms",
           [_shift(c, int(rng.choice([-shift, shift]))) for c in clips])

    for snr_db in REALISM_SNRS_DB:
        report(f"background noise SNR {snr_db:.0f} dB",
               [_mix_at_snr(c, dataset.random_background(rng), snr_db) for c in clips])

    # Live Jabra headset chain, measured from live_probe.wav: speech at
    # ~-11 dBFS RMS with a +10 dB low shelf below 500 Hz.
    report("bass tilt +10 dB @500 Hz", [_low_shelf(c, 10.0) for c in clips])
    report("live chain (bass +10, -11 dBFS)",
           [_to_level(_low_shelf(c, 10.0), -11.0) for c in clips])

    # Everything at once -- closest single proxy for a live mic.
    combined = []
    for c in clips:
        c = _shift(c, int(rng.integers(-shift, shift + 1)))
        c = _mix_at_snr(c, dataset.random_background(rng), float(rng.choice(REALISM_SNRS_DB)))
        c = np.clip(c * 10.0 ** (rng.uniform(-24.0, 6.0) / 20.0), -1.0, 1.0)
        combined.append(c)
    report("combined (shift+noise+gain -24..+6)", combined)


def main() -> None:
    args = parse_args()
    checkpoint = config.CHECKPOINT_DIR / args.run_name / "float.keras"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found -- run kws/train.py first."
        )

    # compile=False: this is inference-only, and the checkpoint's compile
    # config references train.py's custom metric classes (_KeywordPrecision/
    # _KeywordRecall), which aren't registered for deserialization here.
    model = tf.keras.models.load_model(checkpoint, compile=False)

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

    live_realism_check(model, positive_files, rng)


if __name__ == "__main__":
    main()

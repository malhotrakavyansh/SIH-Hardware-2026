"""
eval_heldout.py -- score checkpoints on the held-out session (config.HELDOUT_SESSION) ONLY.

    python eval_heldout.py --runs nakshatra_mvp_v1 nakshatra_v3 nakshatra_v5

Reads the RAW held-out recordings (never the trimmed copies):
  * positives / hard negatives: each take is embedded in 1 s of silence either
    side and run through the INT8 model hop-by-hop, like the firmware. Per-take
    score = max keyword probability over all hops.
  * talking ambient (5 min, no wake word): run through the same detector; every
    fire is a false trigger -> false triggers per hour at each operating point.
Nothing here feeds back into training or into checkpoint selection.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf

import config
import dataset
import eval_streaming as ev

THETAS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98)
POLICIES = ((1, 1), (2, 3), (3, 5))     # (k, window)
REFRACTORY = config.REFRACTORY_HOPS         # hops (1.0 s), same as the live agent


def held_files(sub: str):
    d = config.RECORDINGS_DIR / sub
    return sorted(p for p in d.glob("*.wav") if dataset.speaker_id(p) == config.HELDOUT_SESSION)


def take_stream(x: np.ndarray) -> np.ndarray:
    pad = np.zeros(config.CLIP_SAMPLES, dtype=np.float32)
    return np.concatenate([pad, x, pad])


def float_take_max(model, x: np.ndarray) -> float:
    s = take_stream(x)
    n = ev._num_windows(len(s))
    w = np.stack([features_of(s[i * config.HOP_SAMPLES:i * config.HOP_SAMPLES + config.CLIP_SAMPLES]) for i in range(n)])
    logits = model.predict(w, verbose=0)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return float((e / e.sum(1, keepdims=True))[:, config.KEYWORD_INDEX].max())


def features_of(win: np.ndarray) -> np.ndarray:
    import features
    pcm = np.clip(np.round(win * 32768.0), -32768, 32767).astype(np.int16)
    return features.extract_mfcc(pcm)[..., np.newaxis].astype(np.float32)


def stats(v):
    v = np.asarray(v)
    return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "median": float(np.median(v))}


def evaluate(run: str) -> dict:
    ck = config.CHECKPOINT_DIR / run
    tfl = ck / "model_int8.tflite"
    pos = held_files("positive")
    neg = held_files("hardneg")
    amb = [p for p in held_files("ambient")]
    print(f"\n===== {run}: {len(pos)} held-out positives, {len(neg)} hard negatives, {len(amb)} ambient =====")

    pos_probs = [ev.simulate_streaming(tfl, take_stream(dataset.load_wav(p)), config.SAMPLE_RATE) for p in pos]
    neg_probs = [ev.simulate_streaming(tfl, take_stream(dataset.load_wav(p)), config.SAMPLE_RATE) for p in neg]
    pmax = np.array([p.max() for p in pos_probs])
    nmax = np.array([p.max() for p in neg_probs])

    out = {"run": run, "n_pos": len(pos), "n_neg": len(neg),
           "pos_max_prob": stats(pmax), "neg_max_prob": stats(nmax),
           "separation_gap": float(pmax.min() - nmax.max()),
           "separation_gap_p10_p90": float(np.percentile(pmax, 10) - np.percentile(nmax, 90)),
           "pos_below_0.5": [pos[i].name for i in np.where(pmax <= 0.5)[0]],
           "neg_above_0.5": [neg[i].name for i in np.where(nmax > 0.5)[0]]}
    print(f"positives  max-prob min/max/mean = {pmax.min():.3f} / {pmax.max():.3f} / {pmax.mean():.3f}  (median {np.median(pmax):.3f})")
    print(f"hard negs  max-prob min/max/mean = {nmax.min():.3f} / {nmax.max():.3f} / {nmax.mean():.3f}  (median {np.median(nmax):.3f})")
    print(f"separation gap = min(pos) - max(neg) = {pmax.min():.3f} - {nmax.max():.3f} = {pmax.min() - nmax.max():+.3f}")
    print(f"percentile gap = p10(pos) - p90(neg) = {np.percentile(pmax, 10):.3f} - {np.percentile(nmax, 90):.3f} = {out['separation_gap_p10_p90']:+.3f}")

    out["clip_level_int8"] = {"TPR@0.5": float((pmax > 0.5).mean()), "TNR@0.5": float((nmax <= 0.5).mean())}

    # float vs int8 on the same takes
    fm = tf.keras.models.load_model(ck / "float.keras", compile=False)
    fpmax = np.array([float_take_max(fm, dataset.load_wav(p)) for p in pos])
    fnmax = np.array([float_take_max(fm, dataset.load_wav(p)) for p in neg])
    out["float_vs_int8"] = {
        "float_TPR@0.5": float((fpmax > 0.5).mean()), "int8_TPR@0.5": float((pmax > 0.5).mean()),
        "float_TNR@0.5": float((fnmax <= 0.5).mean()), "int8_TNR@0.5": float((nmax <= 0.5).mean()),
    }
    d = out["float_vs_int8"]
    print(f"float vs int8 @0.5: TPR {100*d['float_TPR@0.5']:.1f}% -> {100*d['int8_TPR@0.5']:.1f}% "
          f"({100*(d['float_TPR@0.5']-d['int8_TPR@0.5']):+.1f}pp),  TNR {100*d['float_TNR@0.5']:.1f}% -> "
          f"{100*d['int8_TNR@0.5']:.1f}% ({100*(d['float_TNR@0.5']-d['int8_TNR@0.5']):+.1f}pp)")

    # talking ambient
    amb_probs = None
    hours = 0.0
    if amb:
        x = dataset.load_wav(amb[0])
        amb_probs = ev.simulate_streaming(tfl, x, config.SAMPLE_RATE)
        hours = len(x) / config.SAMPLE_RATE / 3600.0
        out["ambient_seconds"] = len(x) / config.SAMPLE_RATE
        out["ambient_prob_pctl_50_90_99_max"] = [float(np.percentile(amb_probs, q)) for q in (50, 90, 99, 100)]
        print(f"talking ambient {len(x)/config.SAMPLE_RATE:.0f}s: keyword prob p50/p90/p99/max = "
              + " / ".join(f"{v:.3f}" for v in out["ambient_prob_pctl_50_90_99_max"]))

    print(f"\n{'theta':>6} {'k/win':>6} | {'TPR':>6} {'TNR':>6} | {'FT/300s':>7} {'FT/hour':>8}")
    grid = []
    for k, w in POLICIES:
        for th in THETAS:
            tp = np.mean([len(ev.post_process(p, th, k, w, REFRACTORY)) > 0 for p in pos_probs])
            fp_neg = np.mean([len(ev.post_process(p, th, k, w, REFRACTORY)) == 0 for p in neg_probs])
            n_ft = len(ev.post_process(amb_probs, th, k, w, REFRACTORY)) if amb_probs is not None else None
            row = {"theta": th, "k": k, "window": w, "refractory_hops": REFRACTORY,
                   "TPR": float(tp), "TNR": float(fp_neg), "false_triggers": n_ft,
                   "false_triggers_per_hour": (n_ft / hours) if n_ft is not None else None}
            grid.append(row)
            print(f"{th:6.2f} {k}/{w:<4} | {100*tp:5.1f}% {100*fp_neg:5.1f}% | {n_ft!s:>7} {row['false_triggers_per_hour']:8.1f}")
    out["grid"] = grid
    (ck / "heldout_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    for r in ap.parse_args().runs:
        evaluate(r)

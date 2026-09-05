"""
eval_streaming.py -- DET curve on continuous audio.

Clip-level accuracy is not the metric that matters. The metric is: how many
false accepts per hour on continuous background, at a recall high enough to be
usable. This module runs the model the way the firmware will -- a
config.CLIP_MS window sliding by config.HOP_MS, INT8 end to end -- over long
recordings, and reports false-accepts/hour against miss rate.

CONTRACT: simulate_streaming() must match the firmware's windowing hop-for-hop
(same CLIP_MS window, same HOP_MS advance, same INT8 quant params). If the
firmware ever uses a different hop or window size, the numbers in this file
stop meaning anything for it.

MVP DATA NOTE: "background_audio" concatenates the one real ~10-minute
ambient recording (data/background/background_ambient_01.wav -- typing,
chair movement, room noise, non-speech vocalizations) with synthetic
low-amplitude white noise and sparse silence patches. FAR_ambient_per_hour is
measured on this combined stream. FAR_gauntlet_per_utterance, separately, is
measured on a stream that packs positives + hard_neg clips back-to-back with
only 200ms gaps -- an adversarial per-utterance rate, not a per-hour one,
since that stream's utterance spacing is nothing like real deployment audio.

Implemented in Step 7.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _num_windows(num_samples: int) -> int:
    """How many config.HOP_MS-spaced config.CLIP_MS windows fit in a
    recording of this length -- the same formula simulate_streaming() uses,
    exposed so callers can compute hop offsets without re-running inference."""
    if num_samples < config.CLIP_SAMPLES:
        return 0
    return (num_samples - config.CLIP_SAMPLES) // config.HOP_SAMPLES + 1


def simulate_streaming(model_path: Path, audio: np.ndarray, sr: int) -> np.ndarray:
    """Slide a config.CLIP_MS window across `audio`, advancing config.HOP_MS
    each step, and return the keyword posterior at every hop.

    Runs the actual INT8 .tflite model via tf.lite.Interpreter -- not the
    float Keras model -- with the same quantize-in/dequantize-out arithmetic
    the firmware performs. This is the ESP32 firmware's inner loop in Python:
    same window, same hop, same INT8 math. Getting any of those three out of
    sync with the firmware invalidates this file's numbers for that firmware.

    Returns:
        float32 (num_windows,) -- keyword probability at each hop.
    """
    if sr != config.SAMPLE_RATE:
        raise ValueError(
            f"audio sample rate {sr} != config.SAMPLE_RATE "
            f"({config.SAMPLE_RATE}) -- this evaluator does not resample."
        )

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    num_windows = _num_windows(audio.shape[0])
    probs = np.zeros(num_windows, dtype=np.float32)

    for i in range(num_windows):
        start = i * config.HOP_SAMPLES
        window = audio[start:start + config.CLIP_SAMPLES]

        pcm16 = np.clip(np.round(window * 32768.0), -32768, 32767).astype(np.int16)
        feats = features.extract_mfcc(pcm16)[..., np.newaxis].astype(np.float32)

        q = np.clip(np.round(feats / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], q[np.newaxis, ...])
        interpreter.invoke()
        out_q = interpreter.get_tensor(output_details["index"])[0]
        logits = (out_q.astype(np.float32) - out_zero) * out_scale

        probs[i] = _softmax(logits)[config.KEYWORD_INDEX]

    return probs


def post_process(
    probs: np.ndarray, threshold: float, k: int, window: int, refractory_hops: int,
) -> list[int]:
    """Per-hop probabilities -> discrete detection events.

    Slides a length-`window` box over `probs`; a "fire" is registered at the
    box's center hop whenever at least `k` of the `window` probabilities
    inside it exceed `threshold`. After a fire, no new fire is allowed for
    `refractory_hops` hops -- this is what stops one utterance from
    registering as several detections while its posterior stays high.
    """
    fires: list[int] = []
    last_fire_hop = -refractory_hops - 1  # so the very first candidate can always fire

    n = len(probs)
    for i in range(0, n - window + 1):
        count = int(np.sum(probs[i:i + window] > threshold))
        if count < k:
            continue
        center = i + window // 2
        if center - last_fire_hop >= refractory_hops:
            fires.append(center)
            last_fire_hop = center

    return fires


def compute_metrics(
    fires: list[int], keyword_intervals: list[tuple[int, int]], total_hours: float,
) -> dict:
    """Score fires against ground-truth keyword intervals (start_hop, end_hop).

    A fire landing inside any interval is a true positive; an interval with
    no fire inside it is a missed detection (drives FRR); a fire outside
    every interval is a false activation (drives FAR/hour).
    """
    def _in_any_interval(hop: int) -> bool:
        return any(s <= hop < e for s, e in keyword_intervals)

    false_positive_fires = [f for f in fires if not _in_any_interval(f)]
    missed_intervals = [
        iv for iv in keyword_intervals
        if not any(iv[0] <= f < iv[1] for f in fires)
    ]

    num_keywords = len(keyword_intervals)
    num_fn = len(missed_intervals)
    frr = num_fn / num_keywords if num_keywords else 0.0

    return {
        "FAR_per_hour": len(false_positive_fires) / max(total_hours, 1e-9),
        "FRR": frr,
        "TPR": 1.0 - frr,
        "num_fires": len(fires),
        "num_keywords": num_keywords,
        "num_false_positive_fires": len(false_positive_fires),
        "num_missed": num_fn,
    }


def _score_streams(
    bg_probs: np.ndarray,
    pos_probs: np.ndarray,
    keyword_intervals: list[tuple[int, int]],
    background_hours: float,
    num_hardneg_utterances: int,
    theta: float,
    k: int,
    window: int,
    refractory_hops: int,
) -> dict:
    """Score the two streams SEPARATELY rather than concatenating them into
    one FAR/hour denominator -- combining them was misleading. The two
    streams answer different questions:

    FAR_ambient_per_hour: false fires on the background-only stream, per
    hour. This is what "near-zero false activations" (continuous idle
    listening) actually means, and is what the DET curve/operating point
    should be chosen against.

    FAR_gauntlet_per_utterance: false fires on the positives+hard_neg stream
    that land outside any keyword interval, divided by the number of
    hard-neg utterances (not by stream duration/hours) -- a per-event rate
    for "given a confusable word is spoken, how often does it wrongly fire",
    deliberately NOT extrapolated to per-hour since the gauntlet stream packs
    confusable words back-to-back with only 200ms gaps and is not
    representative of real-world utterance spacing.
    """
    fires_bg = post_process(bg_probs, theta, k, window, refractory_hops)
    far_ambient_per_hour = len(fires_bg) / max(background_hours, 1e-9)

    fires_pos = post_process(pos_probs, theta, k, window, refractory_hops)
    pos_metrics = compute_metrics(fires_pos, keyword_intervals, total_hours=1.0)
    far_gauntlet_per_utterance = (
        pos_metrics["num_false_positive_fires"] / max(num_hardneg_utterances, 1)
    )

    return {
        "FAR_ambient_per_hour": far_ambient_per_hour,
        "FAR_gauntlet_per_utterance": far_gauntlet_per_utterance,
        "FRR": pos_metrics["FRR"],
        "TPR": pos_metrics["TPR"],
    }


def sweep_threshold(
    model_path: Path,
    background_audio: np.ndarray,
    positives_audio: np.ndarray,
    keyword_intervals: list[tuple[int, int]],
    num_hardneg_utterances: int,
) -> dict:
    """Sweep threshold theta in [0.30, 0.95] step 0.05 at fixed k=2, window=3,
    refractory=15 hops (1.5s). Saves the DET curve (FRR vs FAR_ambient_per_hour)
    to kws/artifacts/checkpoints/<run>/det_curve.png and picks the operating
    point: the highest theta with FAR_ambient_per_hour <= 1.0.

    FAR_ambient_per_hour is the metric the ISRO PS's "near-zero false
    activations" requirement actually means (continuous idle listening) --
    it's what the operating point is chosen against. FAR_gauntlet_per_utterance
    is reported alongside for visibility but does not drive the choice.

    Runs inference (simulate_streaming) exactly once per audio stream; every
    threshold in the sweep is scored by re-running the cheap post_process /
    compute_metrics pass over the same posteriors.
    """
    k, window, refractory_hops = 2, 3, 15
    background_hours = background_audio.shape[0] / config.SAMPLE_RATE / 3600.0

    bg_probs = simulate_streaming(model_path, background_audio, config.SAMPLE_RATE)
    pos_probs = simulate_streaming(model_path, positives_audio, config.SAMPLE_RATE)

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.951, 0.05)]
    results = {
        "threshold": [], "FAR_ambient_per_hour": [],
        "FAR_gauntlet_per_utterance": [], "FRR": [], "TPR": [],
    }

    for theta in thresholds:
        m = _score_streams(
            bg_probs, pos_probs, keyword_intervals, background_hours,
            num_hardneg_utterances, theta, k, window, refractory_hops,
        )
        results["threshold"].append(theta)
        results["FAR_ambient_per_hour"].append(m["FAR_ambient_per_hour"])
        results["FAR_gauntlet_per_utterance"].append(m["FAR_gauntlet_per_utterance"])
        results["FRR"].append(m["FRR"])
        results["TPR"].append(m["TPR"])

    out_path = config.CHECKPOINT_DIR / RUN_NAME / "det_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(results["FAR_ambient_per_hour"], results["FRR"], marker="o")
    for th, far, frr in zip(results["threshold"], results["FAR_ambient_per_hour"], results["FRR"]):
        ax.annotate(f"{th:.2f}", (far, frr), fontsize=7,
                     textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("False accepts / hour on ambient background (real + synthetic)")
    ax.set_ylabel("False rejection rate")
    ax.set_title(f"DET curve -- {RUN_NAME} (ambient FAR only)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    candidates = [
        (th, far, frr, gaunt) for th, far, frr, gaunt in
        zip(results["threshold"], results["FAR_ambient_per_hour"],
            results["FRR"], results["FAR_gauntlet_per_utterance"])
        if far <= 1.0
    ]
    if candidates:
        chosen_theta, chosen_far, chosen_frr, chosen_gauntlet = max(
            candidates, key=lambda c: c[0]
        )
    else:
        print("[!] No threshold in the sweep reached FAR_ambient_per_hour "
              "<= 1.0 -- falling back to the lowest-ambient-FAR threshold "
              "available.")
        chosen_theta, chosen_far, chosen_frr, chosen_gauntlet = min(
            zip(results["threshold"], results["FAR_ambient_per_hour"],
                results["FRR"], results["FAR_gauntlet_per_utterance"]),
            key=lambda c: c[1],
        )

    print(f"Operating point: theta={chosen_theta:.2f}  "
          f"FAR_ambient={chosen_far:.3f}/hr  FRR={chosen_frr:.3f}  "
          f"FAR_gauntlet={chosen_gauntlet:.3f}/utterance")

    results["operating_point"] = {
        "threshold": chosen_theta,
        "FAR_ambient_per_hour": chosen_far,
        "FAR_gauntlet_per_utterance": chosen_gauntlet,
        "FRR": chosen_frr,
    }
    return results


def sweep_k(
    model_path: Path,
    background_audio: np.ndarray,
    positives_audio: np.ndarray,
    keyword_intervals: list[tuple[int, int]],
    num_hardneg_utterances: int,
    theta: float,
) -> dict:
    """At the fixed operating threshold, sweep k in {1,2,3} x window in
    {1,3,5} (k<=window only) and report the added-latency trade-off.

    Added latency = (window-1)/2 * HOP_MS ms -- the center-of-window fire
    lags the actual end of the keyword by half the smoothing window.
    """
    refractory_hops = 15
    background_hours = background_audio.shape[0] / config.SAMPLE_RATE / 3600.0

    bg_probs = simulate_streaming(model_path, background_audio, config.SAMPLE_RATE)
    pos_probs = simulate_streaming(model_path, positives_audio, config.SAMPLE_RATE)

    rows = []
    for k in (1, 2, 3):
        for window in (1, 3, 5):
            if k > window:
                continue
            m = _score_streams(
                bg_probs, pos_probs, keyword_intervals, background_hours,
                num_hardneg_utterances, theta, k, window, refractory_hops,
            )
            added_latency_ms = (window - 1) / 2 * config.HOP_MS
            rows.append({
                "k": k, "window": window,
                "FAR_ambient_per_hour": m["FAR_ambient_per_hour"],
                "FAR_gauntlet_per_utterance": m["FAR_gauntlet_per_utterance"],
                "FRR": m["FRR"],
                "added_latency_ms": added_latency_ms,
            })

    print(f"\n{'k':>3} {'window':>7} {'FAR_amb/hr':>11} "
          f"{'FAR_gaunt/utt':>14} {'FRR':>7} {'latency_ms':>11}")
    for r in rows:
        print(f"{r['k']:>3} {r['window']:>7} {r['FAR_ambient_per_hour']:>11.3f} "
              f"{r['FAR_gauntlet_per_utterance']:>14.3f} "
              f"{r['FRR']:>7.3f} {r['added_latency_ms']:>11.0f}")

    return {"rows": rows}


# =============================================================================
# MVP synthetic evaluation audio (no real continuous recordings yet)
# =============================================================================

def _build_positives_stream(gap_ms: int = 200) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Concatenate every positives/*.wav and hard_neg/*.wav, each separated
    by `gap_ms` of silence, into one continuous recording. Returns the audio
    plus (start_sample, end_sample) spans for the keyword-bearing (positives)
    segments only -- hard_neg segments are filler that can trigger false
    activations, but they are not the keyword, so they get no ground-truth
    interval."""
    files = (
        [(p, True) for p in sorted(config.POSITIVES_DIR.glob("*.wav"))]
        + [(p, False) for p in sorted(config.HARD_NEG_DIR.glob("*.wav"))]
    )
    gap_samples = config.SAMPLE_RATE * gap_ms // 1000  # allow-literal

    chunks = []
    keyword_spans = []
    cursor = 0
    gap = np.zeros(gap_samples, dtype=np.float32)

    for path, is_keyword in files:
        clip = dataset.load_wav(path)
        chunks.append(clip)
        if is_keyword:
            keyword_spans.append((cursor, cursor + clip.shape[0]))
        cursor += clip.shape[0]
        chunks.append(gap)
        cursor += gap_samples

    audio = np.concatenate(chunks).astype(np.float32)
    return audio, keyword_spans


def _build_background_stream(
    synthetic_duration_s: float = 300.0, seed: int = config.SEED,
) -> np.ndarray:
    """'No keyword' audio: the real ambient recording(s) in data/background/
    (typing, chair movement, room noise, non-speech vocalizations) followed
    by `synthetic_duration_s` of synthetic low-amplitude white noise with
    sparse 1-second silence patches. The synthetic portion deliberately
    excludes speech -- we have no negative speech recordings beyond the one
    real file."""
    real_files = sorted(config.BACKGROUND_DIR.glob("background_ambient*.wav"))
    real_chunks = [dataset.load_wav(p) for p in real_files]

    rng = np.random.default_rng(seed)
    n = int(config.SAMPLE_RATE * synthetic_duration_s)
    synthetic = rng.normal(loc=0.0, scale=0.01, size=n).astype(np.float32)

    silence_span = config.SAMPLE_RATE  # 1 second
    approx_gap_s = 20.0  # allow-literal
    num_silences = max(1, int(synthetic_duration_s / approx_gap_s))
    for _ in range(num_silences):
        start = int(rng.uniform(0, max(1, n - silence_span)))
        synthetic[start:start + silence_span] = 0.0

    return np.concatenate(real_chunks + [synthetic]).astype(np.float32)


def _spans_to_hop_intervals(
    spans: list[tuple[int, int]], offset_hops: int = 0,
) -> list[tuple[int, int]]:
    """Sample-domain (start, end) spans -> hop-index (start_hop, end_hop)
    intervals, shifted by `offset_hops`. A hop is "inside" a span if its
    window start falls anywhere within the clip's sample range -- generous
    on purpose, since our recordings are already close to CLIP_SAMPLES long."""
    intervals = []
    for start, end in spans:
        start_hop = start // config.HOP_SAMPLES
        end_hop = max(start_hop + 1, end // config.HOP_SAMPLES)
        intervals.append((start_hop + offset_hops, end_hop + offset_hops))
    return intervals


def main() -> None:
    print(config.summary())
    print(
        "\n[!] Background stream for this run mixes the real ~10-minute "
        "ambient recording with synthetic noise/silence -- FAR_ambient_per_hour "
        "below is the meaningful 'idle listening' number. FAR_gauntlet_per_"
        "utterance is measured on a stream that packs confusable words "
        "back-to-back with only 200ms gaps -- an adversarial rate per "
        "utterance, deliberately NOT extrapolated to per-hour since that "
        "stream doesn't represent real utterance spacing.\n"
    )

    model_path = config.CHECKPOINT_DIR / RUN_NAME / "model_int8.tflite"

    positives_audio, keyword_spans = _build_positives_stream(gap_ms=200)
    background_audio = _build_background_stream(synthetic_duration_s=300.0)
    num_hardneg_utterances = len(sorted(config.HARD_NEG_DIR.glob("*.wav")))

    keyword_intervals = _spans_to_hop_intervals(keyword_spans, offset_hops=0)
    background_hours = background_audio.shape[0] / config.SAMPLE_RATE / 3600.0

    print(f"background audio : {background_audio.shape[0] / config.SAMPLE_RATE:.1f}s "
          f"(real ambient + synthetic)")
    print(f"positives audio  : {positives_audio.shape[0] / config.SAMPLE_RATE:.1f}s "
          f"({len(keyword_spans)} keyword occurrences, "
          f"{num_hardneg_utterances} hard-neg utterances)")

    print("\n--- Sweeping threshold ---")
    sweep_results = sweep_threshold(
        model_path, background_audio, positives_audio, keyword_intervals,
        num_hardneg_utterances,
    )
    op = sweep_results["operating_point"]

    print("\n--- Sweeping k / window at the chosen threshold ---")
    k_results = sweep_k(
        model_path, background_audio, positives_audio, keyword_intervals,
        num_hardneg_utterances, op["threshold"],
    )

    rows = k_results["rows"]
    acceptable = [r for r in rows if r["FAR_ambient_per_hour"] <= 1.0]
    if acceptable:
        recommended = min(acceptable, key=lambda r: r["added_latency_ms"])
        reason = "lowest added latency among (k, window) combos with FAR_ambient <= 1.0/hr"
    else:
        recommended = min(rows, key=lambda r: r["FAR_ambient_per_hour"])
        reason = (
            "no (k, window) combo reached FAR_ambient <= 1.0/hr -- picked "
            "the lowest-ambient-FAR combo"
        )

    print("\n=== Recommended firmware parameters ===")
    print(f"threshold (theta) : {op['threshold']:.2f}")
    print(f"k                 : {recommended['k']}")
    print(f"window            : {recommended['window']}")
    print(f"refractory_hops   : 15  (1.5s)")
    print(
        f"reasoning         : chose k={recommended['k']}, "
        f"window={recommended['window']} because {reason} "
        f"(FAR_ambient={recommended['FAR_ambient_per_hour']:.3f}/hr, "
        f"FAR_gauntlet={recommended['FAR_gauntlet_per_utterance']:.3f}/utterance, "
        f"FRR={recommended['FRR']:.3f}, "
        f"added latency={recommended['added_latency_ms']:.0f}ms)"
    )

    metrics_summary = {
        "run_name": RUN_NAME,
        "feature_contract_hash": config.FEATURE_CONTRACT_HASH,
        "eval_audio": {
            "background_seconds": round(background_audio.shape[0] / config.SAMPLE_RATE, 1),
            "background_composition": "real ambient recording + synthetic noise/silence",
            "positives_seconds": round(positives_audio.shape[0] / config.SAMPLE_RATE, 1),
            "num_keyword_occurrences": len(keyword_spans),
            "num_hardneg_utterances": num_hardneg_utterances,
        },
        "operating_point": {
            "threshold": op["threshold"],
            "k": recommended["k"],
            "window": recommended["window"],
            "refractory_hops": 15,
            "FAR_ambient_per_hour": recommended["FAR_ambient_per_hour"],
            "FAR_gauntlet_per_utterance": recommended["FAR_gauntlet_per_utterance"],
            "FRR": recommended["FRR"],
            "TPR": 1.0 - recommended["FRR"],
            "added_latency_ms": recommended["added_latency_ms"],
        },
        "k_window_sweep": rows,
        "det_curve_path": str(config.CHECKPOINT_DIR / RUN_NAME / "det_curve.png"),
    }
    metrics_path = config.CHECKPOINT_DIR / RUN_NAME / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    print(f"\nwrote {metrics_path}")


if __name__ == "__main__":
    main()

"""
trim_takes.py -- energy-based re-trim of every recorded take.

record_session.py holds a take open until END_SILENCE_S of quiet, so noisy /
long takes (esp. the Jabra) carry seconds of non-speech around the word and the
centre-crop to CLIP_MS can clip it. This locates the speech region in each take
and writes it, with a little padding, to data/recordings_trimmed/<label>/ so
dataset.fit_to_clip() centres the word. Raw recordings are never modified.
Takes with no detectable speech are dropped (listed in the report).

    python trim_takes.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import config
import dataset

FRAME_MS_TRIM = 20            # energy frame
HOP_MS_TRIM = 10
MERGE_GAP_MS = 300            # bridge dips inside a word / between syllables
PAD_MS = 150                  # kept either side of the detected speech
REL_DB = 28.0                 # threshold: this far below the loudest frame ...
FLOOR_MARGIN_DB = 12.0        # ... but at least this far above the noise floor
MIN_SPEECH_DYNAMIC_DB = 15.0  # loudest frame must beat the floor by this
MIN_SPEECH_MS = 120
OUT_DIR = config.DATA_DIR / "recordings_trimmed"


def _frame_db(x: np.ndarray) -> np.ndarray:
    fl = config.SAMPLE_RATE * FRAME_MS_TRIM // 1000
    hp = config.SAMPLE_RATE * HOP_MS_TRIM // 1000
    n = max(1 + (len(x) - fl) // hp, 0)
    if n == 0:
        return np.zeros(0)
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms)


def find_speech(x: np.ndarray) -> tuple[int, int] | None:
    """(start, end) sample indices of the dominant speech region, or None."""
    db = _frame_db(x)
    if len(db) == 0:
        return None
    floor = float(np.percentile(db, 10))
    peak = float(db.max())
    if peak - floor < MIN_SPEECH_DYNAMIC_DB:
        return None
    thr = max(floor + FLOOR_MARGIN_DB, peak - REL_DB)
    voiced = db > thr
    if not voiced.any():
        return None
    hp = config.SAMPLE_RATE * HOP_MS_TRIM // 1000
    fl = config.SAMPLE_RATE * FRAME_MS_TRIM // 1000
    gap = MERGE_GAP_MS // HOP_MS_TRIM
    # contiguous runs, merged across short gaps
    runs, start, last = [], None, None
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            elif i - last > gap:
                runs.append((start, last))
                start = i
            last = i
    runs.append((start, last))
    power = 10.0 ** (db / 10.0)
    best = max(runs, key=lambda r: power[r[0]:r[1] + 1].sum())
    s, e = best[0] * hp, best[1] * hp + fl
    if (e - s) * 1000 < MIN_SPEECH_MS * config.SAMPLE_RATE:
        return None
    return s, e


def main() -> None:
    manifest = {r["file"]: r for r in csv.DictReader(
        open(config.RECORDINGS_DIR / "manifest.csv", encoding="utf-8"))}
    pad = config.SAMPLE_RATE * PAD_MS // 1000
    stats = {}
    dropped = []
    for label in ("positive", "hardneg"):
        (OUT_DIR / label).mkdir(parents=True, exist_ok=True)
        for src in sorted((config.RECORDINGS_DIR / label).glob("*.wav")):
            x = dataset.load_wav(src)
            span = find_speech(x)
            mic = manifest.get(src.name, {}).get("mic", "?")
            sess = dataset.speaker_id(src)
            if span is None:
                dropped.append(src.name)
                continue
            s, e = span
            out = x[max(0, s - pad):min(len(x), e + pad)]
            sf.write(OUT_DIR / label / src.name, out, config.SAMPLE_RATE, subtype="PCM_16")
            stats.setdefault((label, mic), []).append(
                (len(x) / config.SAMPLE_RATE, (e - s) / config.SAMPLE_RATE, len(out) / config.SAMPLE_RATE))

    print("\nduration distribution (seconds): raw take | detected speech | trimmed clip")
    for (label, mic), v in sorted(stats.items()):
        a = np.array(v)
        def q(c):
            return "/".join(f"{np.percentile(a[:, c], p):.2f}" for p in (10, 50, 90))
        print(f"  {label:8s} {mic:6s} n={len(a):3d}  raw p10/50/90={q(0)}  speech={q(1)}  "
              f"trimmed={q(2)}  raw>2s: {(a[:,0] > 2).sum()}  trimmed>1s: {(a[:,2] > 1.0).sum()}")
    print(f"\ndropped (no detectable speech): {len(dropped)}")
    for d in dropped:
        print("  ", d)
    (OUT_DIR / "trim_report.json").write_text(json.dumps(
        {"dropped": dropped,
         "stats": {f"{l}/{m}": v for (l, m), v in stats.items()}}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

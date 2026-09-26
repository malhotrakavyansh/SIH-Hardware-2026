"""
make_talking_windows.py -- cut the "talking no keyword" ambient recordings into
1 s windows -> data/recordings_derived/talking/, labelled "unknown" by dataset.py.

Continuous speech without the wake word is the condition earlier models
false-triggered on. Held-out session (config.HELDOUT_SESSION) is never cut.
Windows quieter than TALKING_MIN_DBFS are pauses (silence class, not unknown)
and are skipped.
"""
import csv

import numpy as np
import soundfile as sf

import config
import dataset

OUT = config.DATA_DIR / "recordings_derived" / "talking"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.wav"):
        old.unlink()
    rows = {r["file"]: r for r in csv.DictReader(
        open(config.RECORDINGS_DIR / "manifest.csv", encoding="utf-8"))}
    total = kept = 0
    for r in rows.values():
        if r["label"] != "ambient" or "talking" not in r["cond"]:
            continue
        if r["session"] == config.HELDOUT_SESSION:
            print(f"skip held-out {r['file']}")
            continue
        x = dataset.load_wav(config.RECORDINGS_DIR / "ambient" / r["file"])
        hop = config.SAMPLE_RATE * config.TALKING_HOP_MS // 1000
        n_win = (len(x) - config.CLIP_SAMPLES) // hop + 1
        k = 0
        for i in range(n_win):
            w = x[i * hop:i * hop + config.CLIP_SAMPLES]
            db = 20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-12)
            total += 1
            if db < config.TALKING_MIN_DBFS:
                continue
            sf.write(OUT / f"talking_{r['mic']}_{r['session']}_{i:03d}.wav", w,
                     config.SAMPLE_RATE, subtype="PCM_16")
            k += 1
            kept += 1
        print(f"{r['file']}: {n_win} windows, kept {k}")
    print(f"total windows {total}, kept {kept}, skipped as pauses {total - kept}")


if __name__ == "__main__":
    main()

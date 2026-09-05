"""
split_hardneg.py -- split the hard-negative raw recording into per-word clips.

Input: kws/data/raw_sessions/harshit_session5_hardneg_raw.wav
    5 confusable words, 4 utterances each, spoken in this fixed order:
        nakli, natak, kshatriya, raksha, lakshan

Output: kws/data/hard_neg/harshit_<word>_<n>.wav for n in 1..4 per word.

NOTE on group boundaries: the original plan was to detect word-group
transitions via a long (~5s) silence gap between groups of 4. In practice
this recording does not contain any such gap -- the largest silence gap
found anywhere in the file is comparable to ordinary inter-word gaps
(see verify_groups() below). Group assignment is therefore done by fixed
sequential position (words spoken in a known, fixed order), not by gap
detection. This script prints the gap before each detected clip so you can
manually confirm the grouping still looks right by ear/eye.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from split_positives import RAW_DIR as _RAW_DIR_UNUSED  # noqa: F401 (keep modules aligned)
from split_positives import find_speech_regions

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "kws" / "data" / "raw_sessions"
OUT_DIR = REPO_ROOT / "kws" / "data" / "hard_neg"

IN_FILE = RAW_DIR / "harshit_session5_hardneg_raw.wav"
WORDS = ["nakli", "natak", "kshatriya", "raksha", "lakshan"]
PER_WORD = 4
TARGET_COUNT = len(WORDS) * PER_WORD  # 20

MIN_DURATION_S = 0.25  # these words are shorter than "Nakshatra"; loosen the floor
PAD_MS = 110
GROUP_GAP_WARN_S = 3.0  # gap we'd expect at a group boundary if present


def split(threshold_db: float, merge_ms: float, min_duration_s: float = MIN_DURATION_S):
    audio, sr = sf.read(IN_FILE, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    regions = find_speech_regions(audio, sr, frame_ms=10.0, threshold_db=threshold_db, merge_ms=merge_ms)
    regions = [(s, e) for s, e in regions if (e - s) / sr >= min_duration_s]

    pad = int(sr * PAD_MS / 1000)
    clips = []
    gaps = []
    prev_end = None
    for s, e in regions:
        gap = (s - prev_end) / sr if prev_end is not None else None
        gaps.append(gap)
        prev_end = e
        s_pad = max(0, s - pad)
        e_pad = min(len(audio), e + pad)
        clips.append(audio[s_pad:e_pad])

    return clips, sr, gaps


def auto_split():
    """Sweep for exactly TARGET_COUNT regions; these words have their own
    internal stops (e.g. kshatriya, raksha) so the merge window matters here too."""
    merge_candidates = [350, 400, 300, 250, 200, 150]
    threshold_candidates = [-34, -33, -36, -32, -30, -28, -25, -22, -20]

    best = None
    for merge_ms in merge_candidates:
        for threshold_db in threshold_candidates:
            clips, sr, gaps = split(threshold_db, merge_ms)
            n = len(clips)
            if n == TARGET_COUNT:
                return clips, sr, gaps, threshold_db, merge_ms
            if best is None or abs(n - TARGET_COUNT) < abs(best[0] - TARGET_COUNT):
                best = (n, clips, sr, gaps, threshold_db, merge_ms)

    n, clips, sr, gaps, threshold_db, merge_ms = best
    print(f"WARNING: could not hit exactly {TARGET_COUNT} clips; closest was {n} "
          f"(threshold={threshold_db} dB, merge={merge_ms} ms)")
    return clips, sr, gaps, threshold_db, merge_ms


def main() -> None:
    clips, sr, gaps, threshold_db, merge_ms = auto_split()
    print(f"detected {len(clips)} clips (threshold={threshold_db} dB, merge_ms={merge_ms})\n")

    max_gap = max((g for g in gaps if g is not None), default=0.0)
    print(f"largest inter-clip gap found: {max_gap:.2f}s "
          f"(expected ~5s at group boundaries per recording plan)")
    if max_gap < GROUP_GAP_WARN_S:
        print("GROUP BOUNDARY WARNING: no gap in the recording exceeds "
              f"{GROUP_GAP_WARN_S:.0f}s -- the ~5s group-separator pauses described "
              "for this session are not present in the audio. Falling back to "
              "fixed sequential-position grouping (words were spoken in a known, "
              "fixed order): first 4 clips = nakli, next 4 = natak, etc. "
              "Please sanity-check the grouping below against the actual recording.\n")

    if len(clips) != TARGET_COUNT:
        print(f"NOT {TARGET_COUNT} clips -- cannot safely assign fixed groups of {PER_WORD}. Aborting write.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = 0
    for word in WORDS:
        durs = []
        for n in range(1, PER_WORD + 1):
            clip = clips[idx]
            gap = gaps[idx]
            out_path = OUT_DIR / f"harshit_{word}_{n}.wav"
            sf.write(out_path, clip, sr, subtype="PCM_16")
            dur = len(clip) / sr
            durs.append(dur)
            gap_str = f"{gap:.2f}s" if gap is not None else "n/a (first clip)"
            flag = ""
            if dur < 0.5:
                flag = "  <-- SHORT (<0.5s)"
            elif dur > 2.0:
                flag = "  <-- LONG (>2.0s)"
            print(f"  {out_path.name}: {dur:.3f}s  (gap before: {gap_str}){flag}")
            idx += 1
        print(f"  -- {word}: {min(durs):.3f}s - {max(durs):.3f}s, mean {sum(durs)/len(durs):.3f}s\n")

    print(f"wrote {len(clips)} clips to {OUT_DIR}")


if __name__ == "__main__":
    main()

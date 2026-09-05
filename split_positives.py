"""
split_positives.py -- split raw multi-utterance recordings into individual
per-word clips.

Usage:
    python split_positives.py --sessions 1 --start-index 1
    python split_positives.py --sessions 2,3,4,6 --start-index 26

Reads kws/data/raw_sessions/<speaker>_session<N>_raw*.wav (25 repetitions of
the wake word in one take) and writes kws/data/positives/<speaker>_nohash_<n>.wav
with sequential numbering starting at --start-index across all sessions given.

Silence-based split with a merge window so that the internal /t/ stop in
"Nakshatra" ("-tra") does not get cut into two clips: energy regions closer
together than --merge-ms are treated as one utterance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "kws" / "data" / "raw_sessions"
OUT_DIR = REPO_ROOT / "kws" / "data" / "positives"

TARGET_COUNT = 25
MIN_DURATION_S = 0.3
PAD_MS = 110


def find_speech_regions(
    audio: np.ndarray, sr: int, frame_ms: float, threshold_db: float, merge_ms: float
) -> list[tuple[int, int]]:
    """Return (start_sample, end_sample) for each merged speech region."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return []

    trimmed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    is_speech = rms_db > threshold_db

    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0
    for i, speech in enumerate(is_speech):
        if speech and not in_region:
            in_region = True
            start = i
        elif not speech and in_region:
            in_region = False
            regions.append((start * frame_len, i * frame_len))
    if in_region:
        regions.append((start * frame_len, n_frames * frame_len))

    if not regions:
        return []

    merge_samples = int(sr * merge_ms / 1000)
    merged = [regions[0]]
    for s, e in regions[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= merge_samples:
            merged[-1] = (last_s, e)
        else:
            merged.append((s, e))

    return merged


def find_session_file(speaker: str, session: int) -> Path:
    """Session raw files may carry a suffix, e.g. '..._session6_headphones_raw.wav'."""
    matches = sorted(RAW_DIR.glob(f"{speaker}_session{session}_*raw.wav"))
    if not matches:
        raise FileNotFoundError(f"no raw file found for session {session} in {RAW_DIR}")
    if len(matches) > 1:
        raise FileNotFoundError(f"ambiguous raw files for session {session}: {matches}")
    return matches[0]


def split(
    speaker: str,
    session: int,
    threshold_db: float,
    merge_ms: float,
    min_duration_s: float = MIN_DURATION_S,
) -> tuple[list[np.ndarray], int]:
    in_path = find_session_file(speaker, session)
    audio, sr = sf.read(in_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    regions = find_speech_regions(audio, sr, frame_ms=10.0, threshold_db=threshold_db, merge_ms=merge_ms)

    pad = int(sr * PAD_MS / 1000)
    clips = []
    for s, e in regions:
        if (e - s) / sr < min_duration_s:
            continue
        s_pad = max(0, s - pad)
        e_pad = min(len(audio), e + pad)
        clips.append(audio[s_pad:e_pad])

    return clips, sr


def auto_split(speaker: str, session: int) -> tuple[list[np.ndarray], int, float, float]:
    """Sweep threshold/merge window until we land on TARGET_COUNT clips."""
    merge_candidates = [200, 250, 150, 300, 100, 350, 400]
    threshold_candidates = [-30, -25, -35, -23, -32, -28, -26, -33, -27, -24,
                             -29, -31, -22, -21, -34, -40, -45, -20, -50]

    best = None
    for merge_ms in merge_candidates:
        for threshold_db in threshold_candidates:
            clips, sr = split(speaker, session, threshold_db, merge_ms)
            n = len(clips)
            if n == TARGET_COUNT:
                return clips, sr, threshold_db, merge_ms
            if best is None or abs(n - TARGET_COUNT) < abs(best[0] - TARGET_COUNT):
                best = (n, clips, sr, threshold_db, merge_ms)

    n, clips, sr, threshold_db, merge_ms = best
    print(f"WARNING: could not hit exactly {TARGET_COUNT} clips; closest was {n} "
          f"(threshold={threshold_db} dB, merge={merge_ms} ms)")
    return clips, sr, threshold_db, merge_ms


def diagnose(speaker: str, session: int) -> None:
    """Print a full threshold x merge grid with clip count and duration stats."""
    thresholds = [-15, -20, -25, -30, -35, -40, -45, -50]
    merges = [150, 200, 250, 300, 400]
    print(f"\n--- diagnostic grid for session {session} ---")
    print(f"{'thr(dB)':>8} {'merge(ms)':>10} {'count':>6} {'min_s':>7} {'max_s':>7} {'mean_s':>7}")
    for threshold_db in thresholds:
        for merge_ms in merges:
            clips, sr = split(speaker, session, threshold_db, merge_ms)
            if clips:
                durs = [len(c) / sr for c in clips]
                print(f"{threshold_db:>8} {merge_ms:>10} {len(clips):>6} "
                      f"{min(durs):>7.3f} {max(durs):>7.3f} {sum(durs)/len(durs):>7.3f}")
            else:
                print(f"{threshold_db:>8} {merge_ms:>10} {0:>6}")


def parse_sessions(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default="harshit")
    parser.add_argument("--sessions", default="1", help="comma-separated session numbers, e.g. 2,3,4,6")
    parser.add_argument("--start-index", type=int, default=1, help="first output index (harshit_nohash_<n>.wav)")
    parser.add_argument("--diagnose", action="store_true", help="print threshold/merge grid, write nothing")
    args = parser.parse_args()

    sessions = parse_sessions(args.sessions)

    if args.diagnose:
        for session in sessions:
            diagnose(args.speaker, session)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    next_index = args.start_index
    all_durations = []

    for session in sessions:
        clips, sr, threshold_db, merge_ms = auto_split(args.speaker, session)
        print(f"\n=== session {session}: detected {len(clips)} clips "
              f"(threshold={threshold_db} dB, merge_ms={merge_ms}) ===")

        if len(clips) != TARGET_COUNT:
            print(f"NOT {TARGET_COUNT} -- inspect kws/data/raw_sessions/ recording or widen sweep ranges.")

        session_durations = []
        for clip in clips:
            out_path = OUT_DIR / f"{args.speaker}_nohash_{next_index}.wav"
            sf.write(out_path, clip, sr, subtype="PCM_16")
            dur = len(clip) / sr
            session_durations.append(dur)
            all_durations.append(dur)
            flag = ""
            if dur < 0.5:
                flag = "  <-- SHORT (<0.5s)"
            elif dur > 2.0:
                flag = "  <-- LONG (>2.0s)"
            print(f"  {out_path.name}: {dur:.3f}s{flag}")
            next_index += 1

        if session_durations:
            print(f"  session {session} duration range: {min(session_durations):.3f}s - "
                  f"{max(session_durations):.3f}s, mean {sum(session_durations)/len(session_durations):.3f}s")

    total = next_index - args.start_index
    print(f"\nwrote {total} clips total to {OUT_DIR} (indices {args.start_index}-{next_index - 1})")
    if all_durations:
        print(f"overall duration range: {min(all_durations):.3f}s - {max(all_durations):.3f}s, "
              f"mean {sum(all_durations)/len(all_durations):.3f}s")


if __name__ == "__main__":
    main()

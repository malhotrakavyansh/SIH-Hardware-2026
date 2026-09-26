"""
record_session.py -- guided multi-utterance recorder for the wake-word dataset.

Records through sounddevice at 16 kHz mono, the exact capture path
edge_agent.py uses -- never Audacity, whose capture chain is what the old
dataset was stuck on. Each utterance is cut automatically (recording stops
after ~1.2 s of silence) and saved as its own PCM16 file, so there is no
manual splitting afterwards.

Files land in kws/data/recordings/<label>/ as

    <label>_<mic>_<session>_<n>.wav      e.g. positive_laptop_s03_017.wav

and every take is also logged to kws/data/recordings/manifest.csv with the
full device name, host API, prompt, condition and levels, so the capture
chain of any clip is recoverable later.

Run:
    python record_session.py --list-devices
    python record_session.py --device 1 --mic laptop --label positive --prefix s01 --count 70 \\
        --cond "30cm facing" --prompts "normal,quiet,loud,slow,fast"
    python record_session.py --device 1 --mic laptop --label hardneg  --prefix s01 --count 30
    python record_session.py --device 1 --mic laptop --label ambient  --prefix s01 --seconds 120 --cond "quiet room"
    python record_session.py --device 1 --mic laptop --label positive --prefix s01 --redo

During a session, press r in the pause after a take to redo it, q to stop.
"""

from __future__ import annotations

import argparse
import collections
import csv
import queue
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent / "kws"))

import config  # noqa: E402  (kws/config.py)
from audio_devices import list_input_devices, resolve_input_device  # noqa: E402

try:
    import msvcrt  # Windows: non-blocking key check for redo/quit between takes
except ImportError:
    msvcrt = None

OUT_ROOT = config.DATA_DIR / "recordings"
MANIFEST = OUT_ROOT / "manifest.csv"
MANIFEST_FIELDS = ["file", "label", "mic", "session", "n", "prompt", "cond", "device_index",
                   "device", "duration_s", "rms_dbfs", "peak_dbfs", "recorded_at", "redo"]

BLOCK = config.SAMPLE_RATE // 10          # 100 ms blocks
BLOCK_S = BLOCK / config.SAMPLE_RATE
END_SILENCE_S = 1.2                       # stop an utterance after this much silence
PRE_ROLL_BLOCKS = 3                       # keep 300 ms before the detected onset
TAIL_BLOCKS = 2                           # keep 200 ms after the last voiced block
MAX_UTTERANCE_S = 4.0
ONSET_TIMEOUT_S = 8.0                     # re-prompt if nothing is said
PAUSE_S = 1.5                             # gap between takes (the redo window)
COUNTDOWN_S = 3

# Cycled on screen for hard negatives unless --prompts is given: the words
# already in data/hard_neg/ plus other near-rhymes of "nakshatra".
DEFAULT_HARDNEG_PROMPTS = ["kshatriya", "lakshan", "nakli", "natak", "raksha",
                           "naksha", "rashtra", "shastra", "chhatra", "akshar"]


def dbfs(x: np.ndarray) -> float:
    return 20.0 * float(np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def peak_dbfs(x: np.ndarray) -> float:
    return 20.0 * float(np.log10(np.max(np.abs(x)) + 1e-12))


def slug(text: str) -> str:
    """Filename-safe tag: underscores separate fields, so none inside one."""
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Guided wake-word dataset recorder.")
    p.add_argument("--device", type=int, help="input device index (see --list-devices)")
    p.add_argument("--label", choices=("positive", "hardneg", "ambient"))
    p.add_argument("--prefix", help="session id, e.g. s01 (goes in the filename)")
    p.add_argument("--count", type=int, default=25, help="utterances to record this run (default: %(default)s)")
    p.add_argument("--seconds", type=float, default=120.0, help="ambient: recording length (default: %(default)s)")
    p.add_argument("--mic", help="short mic tag for the filename, e.g. laptop / jabra "
                                 "(default: derived from the device name)")
    p.add_argument("--prompts", help="comma list shown in turn with each utterance -- speaking "
                                     "styles for positives, words for hard negatives")
    p.add_argument("--cond", default="", help="free-text condition logged to the manifest, e.g. '1m off-axis'")
    p.add_argument("--threshold", type=float, default=None,
                   help="speech-onset level in dBFS (default: room floor + 10 dB, clamped to -55..-30)")
    p.add_argument("--redo", action="store_true", help="re-record the last saved utterance of this "
                                                       "label/mic/session, then exit")
    p.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    args = p.parse_args()
    if not args.list_devices:
        missing = [f"--{k}" for k in ("device", "label", "prefix") if getattr(args, k) is None]
        if missing:
            p.error("required: " + ", ".join(missing))
    return args


class Mic:
    """One InputStream held open for the whole session; 100 ms blocks queued."""

    def __init__(self, device: int):
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.stream = sd.InputStream(device=device, samplerate=config.SAMPLE_RATE, channels=1,
                                     dtype="float32", blocksize=BLOCK, callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        self.q.put(indata[:, 0].copy())

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def drain(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()

    def read(self) -> np.ndarray:
        return self.q.get(timeout=2.0)


def capture_utterance(mic: Mic, start_thr: float) -> tuple[np.ndarray | None, bool]:
    """Wait for speech, record until END_SILENCE_S of silence. Returns
    (audio or None on timeout, hit_max_length)."""
    end_thr = start_thr - 3.0
    mic.drain()
    pre: collections.deque = collections.deque(maxlen=PRE_ROLL_BLOCKS)
    waited = 0.0
    while True:
        b = mic.read()
        waited += BLOCK_S
        if dbfs(b) > start_thr:
            break
        pre.append(b)
        if waited >= ONSET_TIMEOUT_S:
            return None, False

    blocks = list(pre) + [b]
    last_voiced = len(blocks) - 1
    silent = 0
    hit_max = False
    while True:
        b = mic.read()
        blocks.append(b)
        if dbfs(b) >= end_thr:
            silent = 0
            last_voiced = len(blocks) - 1
        else:
            silent += 1
            if silent * BLOCK_S >= END_SILENCE_S:
                break
        if len(blocks) * BLOCK_S >= MAX_UTTERANCE_S:
            hit_max = True
            break
    return np.concatenate(blocks[: last_voiced + 1 + TAIL_BLOCKS]), hit_max


def save(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
    sf.write(path, pcm16, config.SAMPLE_RATE, subtype="PCM_16")


def log_manifest(row: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def take_report(audio: np.ndarray, label: str, hit_max: bool) -> str:
    dur = len(audio) / config.SAMPLE_RATE
    rms, pk = dbfs(audio), peak_dbfs(audio)
    flags = []
    if pk >= -0.5:
        flags.append("CLIPPED")
    if rms < -45.0:
        flags.append("very quiet")
    if hit_max:
        flags.append(f"hit {MAX_UTTERANCE_S:.0f}s cap")
    elif label != "ambient" and dur > 2.0:
        flags.append("long -- two words or noise?")
    if label != "ambient" and dur < 0.35:
        flags.append("very short -- clipped word?")
    warn = ("   <-- " + ", ".join(flags) + " (press r to redo)") if flags else ""
    return f"{dur:5.2f}s  RMS {rms:6.1f} dBFS  peak {pk:6.1f} dBFS{warn}"


def existing_numbers(label: str, mic: str, session: str) -> list[int]:
    pat = re.compile(rf"^{label}_{re.escape(mic)}_{re.escape(session)}_(\d+)\.wav$")
    d = OUT_ROOT / label
    return sorted(int(m.group(1)) for p in d.glob("*.wav") if (m := pat.match(p.name))) if d.is_dir() else []


def key_pressed() -> str | None:
    if msvcrt and msvcrt.kbhit():
        return msvcrt.getwch().lower()
    return None


def pause_for_keys(seconds: float) -> str | None:
    """The gap between takes; returns 'r' / 'q' if pressed, else None."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        k = key_pressed()
        if k in ("r", "q"):
            return k
        time.sleep(0.05)
    return None


def countdown_and_calibrate(mic: Mic) -> float:
    """3 s countdown; the last second doubles as a room-floor measurement."""
    print("  stay quiet for the countdown...")
    floor_blocks = []
    for s in range(COUNTDOWN_S, 0, -1):
        print(f"  {s}...", flush=True)
        mic.drain()
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            b = mic.read()
            if s == 1:
                floor_blocks.append(dbfs(b))
    return float(np.percentile(floor_blocks, 20)) if floor_blocks else -90.0


def record_ambient(mic: Mic, args, mic_tag: str, dev_index: int, dev_name: str) -> None:
    n = (existing_numbers("ambient", mic_tag, args.prefix) or [0])[-1] + 1
    name = f"ambient_{mic_tag}_{args.prefix}_{n:03d}.wav"
    print(f"\nAMBIENT: recording {args.seconds:.0f} s continuously -> {name}")
    for s in range(COUNTDOWN_S, 0, -1):
        print(f"  {s}...", flush=True)
        time.sleep(1.0)
    mic.drain()
    blocks, total = [], int(round(args.seconds / BLOCK_S))
    for i in range(total):
        blocks.append(mic.read())
        if (i + 1) % 100 == 0:
            print(f"  {((i + 1) * BLOCK_S):5.0f} / {args.seconds:.0f} s", flush=True)
    audio = np.concatenate(blocks)
    save(audio, OUT_ROOT / "ambient" / name)
    levels = np.array([dbfs(b) for b in blocks])
    print(f"  saved {name}  {take_report(audio, 'ambient', False)}")
    print(f"  floor p5 {np.percentile(levels, 5):.1f} dBFS, exact-zero samples {(audio == 0).mean() * 100:.1f}%")
    log_manifest(dict(file=name, label="ambient", mic=mic_tag, session=args.prefix, n=n, prompt="",
                      cond=args.cond, device_index=dev_index, device=dev_name,
                      duration_s=f"{len(audio) / config.SAMPLE_RATE:.2f}", rms_dbfs=f"{dbfs(audio):.1f}",
                      peak_dbfs=f"{peak_dbfs(audio):.1f}", recorded_at=datetime.now().isoformat(timespec="seconds"),
                      redo=0))


def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return

    # Refuses (SystemExit with a clear message) if the device can't do 16 kHz mono.
    dev_index, dev_name = resolve_input_device(args.device, config.SAMPLE_RATE)
    mic_tag = slug(args.mic) if args.mic else slug(dev_name.split("[")[0])[:12].strip("-")
    session = slug(args.prefix)
    args.prefix = session
    print(f"device  : {dev_index} -- {dev_name}")
    print(f"files   : {OUT_ROOT / args.label}  as  {args.label}_{mic_tag}_{session}_<n>.wav")

    with Mic(dev_index) as mic:
        if args.label == "ambient":
            record_ambient(mic, args, mic_tag, dev_index, dev_name)
            return

        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()] if args.prompts else (
            DEFAULT_HARDNEG_PROMPTS if args.label == "hardneg" else ["normal"])
        numbers = existing_numbers(args.label, mic_tag, session)

        if args.redo:
            if not numbers:
                raise SystemExit(f"--redo: nothing recorded yet for {args.label}/{mic_tag}/{session}")
            plan = [numbers[-1]]
            print(f"REDO of utterance #{numbers[-1]}")
        else:
            start = (numbers[-1] if numbers else 0) + 1
            plan = list(range(start, start + args.count))
            if numbers:
                print(f"resuming: {len(numbers)} already in this session, new files start at #{start}")

        floor = countdown_and_calibrate(mic)
        start_thr = args.threshold if args.threshold is not None else min(max(floor + 10.0, -55.0), -30.0)
        print(f"  room floor {floor:.1f} dBFS -> speech onset above {start_thr:.1f} dBFS\n")

        saved, i, redoing = 0, 0, bool(args.redo)
        while i < len(plan):
            n = plan[i]
            prompt = prompts[(n - 1) % len(prompts)]
            say = f"say: NAKSHATRA   ({prompt})" if args.label == "positive" else f"say: {prompt.upper()}"
            print(f"utterance {i + 1} of {len(plan)}  [#{n}]   {say}", flush=True)

            audio, hit_max = capture_utterance(mic, start_thr)
            if audio is None:
                print("  no speech detected -- try again (or check --threshold)")
                continue

            name = f"{args.label}_{mic_tag}_{session}_{n:03d}.wav"
            save(audio, OUT_ROOT / args.label / name)
            saved += 1
            print(f"  saved {name}  {take_report(audio, args.label, hit_max)}")
            log_manifest(dict(file=name, label=args.label, mic=mic_tag, session=session, n=n, prompt=prompt,
                              cond=args.cond, device_index=dev_index, device=dev_name,
                              duration_s=f"{len(audio) / config.SAMPLE_RATE:.2f}",
                              rms_dbfs=f"{dbfs(audio):.1f}", peak_dbfs=f"{peak_dbfs(audio):.1f}",
                              recorded_at=datetime.now().isoformat(timespec="seconds"),
                              redo=int(redoing)))

            if args.redo:
                break
            key = pause_for_keys(PAUSE_S)
            if key == "q":
                print("stopped.")
                break
            if key == "r":
                print(f"  redo #{n}")
                redoing = True
                continue  # same n, overwritten
            redoing = False
            i += 1

    print(f"\ndone: {saved} file(s) saved to {OUT_ROOT / args.label}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted -- files saved so far are kept.")

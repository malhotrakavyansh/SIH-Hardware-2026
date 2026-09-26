"""
record_probe.py -- capture a short probe clip through the exact capture chain
edge_agent.py uses (sounddevice, 16 kHz mono float32 -> PCM16), so offline
analysis sees what the live wake-word model sees.

Run:
    python record_probe.py --list-devices
    python record_probe.py --device 1 laptop_probe.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent / "kws"))

import config  # noqa: E402  (kws/config.py)
from audio_devices import list_input_devices, resolve_input_device  # noqa: E402

COUNTDOWN_S = 3
DEFAULT_DURATION_S = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a probe clip through edge_agent's capture chain.")
    parser.add_argument("output", type=Path, nargs="?", help="output .wav path")
    parser.add_argument("--device", type=int, default=None,
                        help="input device index (see --list-devices); default: Windows default input")
    parser.add_argument("--seconds", type=float, default=DEFAULT_DURATION_S,
                        help="recording length (default: %(default)s)")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    args = parser.parse_args()
    if not args.list_devices and args.output is None:
        parser.error("output path is required unless --list-devices is given")
    return args


def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return

    device_index, device_name = resolve_input_device(args.device, config.SAMPLE_RATE)
    print(f"input device: {device_index} -- {device_name}")

    for remaining in range(COUNTDOWN_S, 0, -1):
        print(f"  recording in {remaining}...", flush=True)
        time.sleep(1.0)
    print(f"RECORDING {args.seconds:.0f} s -- speak now", flush=True)

    audio = sd.rec(int(round(args.seconds * config.SAMPLE_RATE)), samplerate=config.SAMPLE_RATE,
                   channels=1, dtype="float32", device=device_index, blocking=True)[:, 0]
    print("done.")

    # Same float -> PCM16 conversion edge_agent applies before feature extraction.
    pcm16 = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, pcm16, config.SAMPLE_RATE, subtype="PCM_16")

    rms_db = 20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-12)
    print(f"saved {args.output}  ({args.seconds:.0f} s, RMS {rms_db:.1f} dBFS, peak {peak_db:.1f} dBFS)")


if __name__ == "__main__":
    main()

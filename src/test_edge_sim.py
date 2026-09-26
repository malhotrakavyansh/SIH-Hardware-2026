"""
test_edge_sim.py -- replays a WAV file at the edge_agent.py wire protocol
(see PROTOCOL.md) so cloud_server.py + the UI can be exercised without a
microphone or the real edge agent running.

Sends: wake -> PCM frames at real-time pace (1600 samples / 100ms, matching
HOP_SAMPLES) -> utterance_end.

Run:
    python test_edge_sim.py path/to/clip.wav
    python test_edge_sim.py path/to/clip.wav --server ws://localhost:8000/ws/edge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("[setup] soundfile not found -- installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile as sf

try:
    import websockets
except ImportError:
    print("[setup] websockets not found -- installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

SAMPLE_RATE = 16000
HOP_SAMPLES = 1600  # 100 ms, matches kws/config.py HOP_MS


def load_pcm16_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sr} Hz, expected {SAMPLE_RATE} Hz. "
            f"Resample it first (e.g. `sox in.wav -r 16000 -c 1 out.wav`)."
        )
    return audio


async def replay(wav_path: Path, server_url: str, prob: float) -> None:
    pcm = load_pcm16_mono(wav_path)
    n_chunks = int(np.ceil(len(pcm) / HOP_SAMPLES))
    duration_s = len(pcm) / SAMPLE_RATE
    print(f"loaded {wav_path.name}: {duration_s:.2f}s, {n_chunks} chunks of {HOP_SAMPLES} samples")

    async with websockets.connect(server_url) as ws:
        t1 = time.time() * 1000.0
        await ws.send(json.dumps({"event": "wake", "t1": t1, "prob": prob}))
        print(f"sent wake (t1={t1:.1f})")

        for i in range(n_chunks):
            start = i * HOP_SAMPLES
            end = start + HOP_SAMPLES
            chunk = pcm[start:end]
            if len(chunk) < HOP_SAMPLES:
                chunk = np.pad(chunk, (0, HOP_SAMPLES - len(chunk)))
            await ws.send(chunk.astype("<i2").tobytes())
            await asyncio.sleep(HOP_SAMPLES / SAMPLE_RATE)  # real-time pace

        await ws.send(json.dumps({"event": "utterance_end"}))
        print(f"sent utterance_end ({n_chunks} chunks, {duration_s:.2f}s)")

        # Keep the connection open briefly so any late server-side logging
        # (not visible to us -- broadcasts only go to /ws/ui) has time to run.
        await asyncio.sleep(0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a WAV file as a simulated edge device.")
    parser.add_argument("wav", type=Path, help="16kHz mono WAV file to replay")
    parser.add_argument("--server", type=str, default="ws://localhost:8000/ws/edge", help="server WebSocket URL")
    parser.add_argument("--prob", type=float, default=0.9, help="fake keyword probability to report in the wake event")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wav.exists():
        print(f"error: {args.wav} does not exist", file=sys.stderr)
        sys.exit(1)
    asyncio.run(replay(args.wav, args.server, args.prob))


if __name__ == "__main__":
    main()

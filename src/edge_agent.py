"""
edge_agent.py -- headless edge-side agent for the "nakshatra" wake-word model.

Runs the same quantize -> tflite -> dequantize -> softmax -> k-of-window /
refractory pipeline as live_demo.py, but with no UI: this process's only job
is to watch the mic, detect the wake word, and stream the utterance that
follows to a server over WebSocket.

It is deliberately written as a spec for the ESP32 firmware port: an
explicit IDLE -> STREAMING -> IDLE state machine, nothing in the hot loop
that couldn't exist on an MCU (no pandas/sklearn, plain arrays and a small
JSON/binary wire protocol), and every message it emits is documented in
PROTOCOL.md. When the firmware exists, it should be swappable for this
process without the server noticing.

Run:
    python edge_agent.py --list-devices
    python edge_agent.py --device <idx> --server ws://localhost:8000/ws/edge

Ctrl+C to stop; prints a session summary on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional deps, pulled in on first run so this works on a fresh machine.
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
except ImportError:
    print("[setup] sounddevice not found -- installing (pip install sounddevice)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
    import sounddevice as sd

try:
    import websockets
except ImportError:
    print("[setup] websockets not found -- installing (pip install websockets)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

ROOT = Path(__file__).resolve().parent
KWS_DIR = ROOT / "kws"
sys.path.insert(0, str(KWS_DIR))

import config  # noqa: E402  (kws/config.py)
import features  # noqa: E402  (kws/features.py)
from audio_devices import list_input_devices, resolve_input_device  # noqa: E402

import tensorflow as tf  # noqa: E402

RUN_NAME = "nakshatra_v5"  # held-out: TPR 89.8%, hard-neg TNR 96.7%, 0 FT/300s ambient at theta 0.75, k 3/5 (v1: 74.6%/86.7%, 204 FT/h)
CHECKPOINT_DIR = KWS_DIR / "artifacts" / "checkpoints" / RUN_NAME
MODEL_PATH = CHECKPOINT_DIR / "model_int8.tflite"
QUANT_PARAMS_PATH = CHECKPOINT_DIR / "quant_params.json"
FINAL_METRICS_PATH = CHECKPOINT_DIR / "final_metrics.json"

SAMPLE_RATE = config.SAMPLE_RATE      # 16000 Hz
CLIP_SAMPLES = config.CLIP_SAMPLES    # 16000 (1 s analysis window)
HOP_MS = config.HOP_MS                # 100 ms inference cadence
HOP_SAMPLES = SAMPLE_RATE * HOP_MS // 1000  # 1600 samples/chunk

# --- streaming policy (not part of the frozen feature contract) -----------
LEVEL_EVERY_N_CHUNKS = 2          # ~200 ms between "level" events in IDLE
DEBUG_PROBS_EVERY_N_CHUNKS = 5    # ~500 ms between --debug-probs print lines
PREROLL_CHUNKS = 3                # ~300 ms of pre-roll sent ahead of live audio
SILENCE_RMS_THRESHOLD = 0.02      # float32 RMS below this counts as silence
STREAMING_SILENCE_S = 1.5         # stop after this much continuous silence
STREAMING_MAX_S = 8.0             # hard cap on utterance length
OUTBOUND_QUEUE_MAXLEN = 100       # ~10 s of buffered messages while disconnected
RECONNECT_BASE_S = 2.0
RECONNECT_MAX_S = 10.0


def ts() -> str:
    """[HH:MM:SS.mmm] wall-clock timestamp for log lines."""
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


# =============================================================================
# Model: quant params, tflite interpreter, k-of-window/refractory policy.
# Mirrors live_demo.KeywordSpotter exactly (same contract, no UI concerns).
# =============================================================================

def load_quant_params() -> dict:
    if not QUANT_PARAMS_PATH.exists():
        raise FileNotFoundError(f"missing {QUANT_PARAMS_PATH}")
    with open(QUANT_PARAMS_PATH, "r") as f:
        return json.load(f)


def load_operating_point(cli_overrides: dict) -> tuple[dict, str]:
    """Precedence: CLI args > final_metrics.json > hardcoded defaults."""
    defaults = {"threshold": config.DETECT_THRESHOLD, "k": config.DETECT_K,
                "window": config.SMOOTH_WINDOW_HOPS, "refractory_hops": config.REFRACTORY_HOPS}

    if any(v is not None for v in cli_overrides.values()):
        base = {}
        if FINAL_METRICS_PATH.exists():
            with open(FINAL_METRICS_PATH, "r") as f:
                base = json.load(f).get("operating_point", {})
        op = {
            "threshold": cli_overrides.get("threshold") if cli_overrides.get("threshold") is not None else base.get("threshold", defaults["threshold"]),
            "k": cli_overrides.get("k") if cli_overrides.get("k") is not None else base.get("k", defaults["k"]),
            "window": cli_overrides.get("window") if cli_overrides.get("window") is not None else base.get("window", defaults["window"]),
            "refractory_hops": cli_overrides.get("refractory_hops") if cli_overrides.get("refractory_hops") is not None else base.get("refractory_hops", defaults["refractory_hops"]),
        }
        return op, "CLI"

    if FINAL_METRICS_PATH.exists():
        with open(FINAL_METRICS_PATH, "r") as f:
            op = json.load(f).get("operating_point", {})
        return {
            "threshold": op.get("threshold", defaults["threshold"]),
            "k": op.get("k", defaults["k"]),
            "window": op.get("window", defaults["window"]),
            "refractory_hops": op.get("refractory_hops", defaults["refractory_hops"]),
        }, "final_metrics.json"

    return defaults, "hardcoded defaults"


class KeywordSpotter:
    """tflite interpreter + quant params + k-of-window/refractory state."""

    def __init__(self, cli_overrides: dict):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"missing {MODEL_PATH}")

        self.quant = load_quant_params()
        self.op, self.op_source = load_operating_point(cli_overrides)

        self.interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        in_scale, in_zero = self.input_details["quantization"]
        out_scale, out_zero = self.output_details["quantization"]
        self.in_scale = in_scale or self.quant["input"]["scale"]
        self.in_zero = in_zero or self.quant["input"]["zero_point"]
        self.out_scale = out_scale or self.quant["output"]["scale"]
        self.out_zero = out_zero or self.quant["output"]["zero_point"]

        self.labels = self.quant["labels"]
        self.keyword_index = self.labels.index("keyword")
        self.contract_hash = self.quant["feature_contract_hash"]

        self.prob_window: collections.deque[float] = collections.deque(maxlen=self.op["window"])
        self._last_fire_hop = -self.op["refractory_hops"] - 1
        self._hop_index = 0

    def infer_full(self, window_f32: np.ndarray) -> np.ndarray:
        """Full 3-class posterior [silence, unknown, keyword] for one 1s
        window. infer() (below) is just this indexed down to the keyword
        scalar -- kept as its own method so --debug-probs can see the other
        two classes too, not just the one number the detector acts on."""
        pcm16 = np.clip(np.round(window_f32 * 32768.0), -32768, 32767).astype(np.int16)
        feats = features.extract_mfcc(pcm16)[..., np.newaxis].astype(np.float32)

        q = np.clip(np.round(feats / self.in_scale + self.in_zero), -128, 127).astype(np.int8)
        self.interpreter.set_tensor(self.input_details["index"], q[np.newaxis, ...])
        self.interpreter.invoke()
        out_q = self.interpreter.get_tensor(self.output_details["index"])[0]
        logits = (out_q.astype(np.float32) - self.out_zero) * self.out_scale

        return softmax(logits)

    def infer(self, window_f32: np.ndarray) -> float:
        return float(self.infer_full(window_f32)[self.keyword_index])

    def step(self, window_f32: np.ndarray) -> tuple[float, bool]:
        """One hop of inference + k-of-window/refractory. Returns (prob, fired)."""
        prob = self.infer(window_f32)
        self.prob_window.append(prob)
        self._hop_index += 1

        fired = False
        window, theta, k = self.op["window"], self.op["threshold"], self.op["k"]
        refractory_hops = self.op["refractory_hops"]

        if len(self.prob_window) == window:
            count = sum(1 for p in self.prob_window if p > theta)
            if count >= k and self._hop_index - self._last_fire_hop >= refractory_hops:
                fired = True
                self._last_fire_hop = self._hop_index

        return prob, fired

    def warmup(self) -> None:
        self.infer(np.zeros(CLIP_SAMPLES, dtype=np.float32))


# =============================================================================
# Edge agent: state machine + mic capture + WebSocket link
# =============================================================================

class EdgeAgent:
    """IDLE -> STREAMING -> IDLE. No other states.

    IDLE:       run wake-word inference every HOP_MS, send periodic level
                events, watch for a k-of-window/refractory detection.
    STREAMING:  no inference; stream raw PCM to the server until silence or
                the hard cap, then send utterance_end and go back to IDLE.
    """

    def __init__(self, spotter: KeywordSpotter, server_url: str, debug_probs: bool = False):
        self.spotter = spotter
        self.server_url = server_url
        self.debug_probs = debug_probs

        self.state = "IDLE"
        self.ws: "websockets.WebSocketClientProtocol | None" = None
        self.outbound: collections.deque = collections.deque(maxlen=OUTBOUND_QUEUE_MAXLEN)

        # Fixed-size 1 s window for inference, updated one HOP_SAMPLES chunk
        # at a time (same shift-register behaviour as live_demo's ring buffer).
        self._window = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        self._preroll: collections.deque = collections.deque(maxlen=PREROLL_CHUNKS)

        self._chunk_counter = 0
        self._streaming_start = 0.0
        self._silence_run = 0.0
        self._streaming_chunks_sent = 0

        self.stop_requested = False
        self.start_time = time.monotonic()
        self.detections = 0
        self.reconnects = 0
        self._connected_once = False

    # -- audio path ----------------------------------------------------

    def _push_window(self, chunk_f32: np.ndarray) -> None:
        self._window[:-HOP_SAMPLES] = self._window[HOP_SAMPLES:]
        self._window[-HOP_SAMPLES:] = chunk_f32

    def handle_chunk(self, chunk_f32: np.ndarray) -> None:
        """One HOP_MS chunk of mono float32 mic audio, in [-1, 1)."""
        pcm16_bytes = np.clip(np.round(chunk_f32 * 32768.0), -32768, 32767).astype("<i2").tobytes()
        self._preroll.append(pcm16_bytes)

        if self.state == "IDLE":
            self._handle_idle_chunk(chunk_f32, pcm16_bytes)
        else:
            self._handle_streaming_chunk(pcm16_bytes, chunk_f32)

    def _handle_idle_chunk(self, chunk_f32: np.ndarray, pcm16_bytes: bytes) -> None:
        self._push_window(chunk_f32)
        prob, fired = self.spotter.step(self._window)

        self._chunk_counter += 1
        if self._chunk_counter % LEVEL_EVERY_N_CHUNKS == 0:
            rms = float(np.sqrt(np.mean(np.square(chunk_f32))))
            self._enqueue({"event": "level", "rms": rms})

        if self.debug_probs and self._chunk_counter % DEBUG_PROBS_EVERY_N_CHUNKS == 0:
            dist = self.spotter.infer_full(self._window)
            print(f"[{ts()}] probs  silence={dist[config.SILENCE_INDEX]:.3f}  "
                  f"unknown={dist[config.UNKNOWN_INDEX]:.3f}  "
                  f"keyword={dist[config.KEYWORD_INDEX]:.3f}")

        if fired:
            t1 = time.time() * 1000.0
            print(f"[{ts()}] WAKE prob={prob:.2f} -> streaming")
            self._enqueue({"event": "wake", "t1": t1, "prob": prob})
            self.detections += 1

            self.state = "STREAMING"
            self._streaming_start = time.monotonic()
            self._silence_run = 0.0
            self._streaming_chunks_sent = 0

            # Pre-roll first (oldest to newest), so the word right after the
            # wake word doesn't get clipped by detection latency.
            for pb in self._preroll:
                self._enqueue(pb)
                self._streaming_chunks_sent += 1

    def _handle_streaming_chunk(self, pcm16_bytes: bytes, chunk_f32: np.ndarray) -> None:
        self._enqueue(pcm16_bytes)
        self._streaming_chunks_sent += 1

        rms = float(np.sqrt(np.mean(np.square(chunk_f32))))
        if rms < SILENCE_RMS_THRESHOLD:
            self._silence_run += HOP_MS / 1000.0
        else:
            self._silence_run = 0.0

        elapsed = time.monotonic() - self._streaming_start
        if self._silence_run >= STREAMING_SILENCE_S or elapsed >= STREAMING_MAX_S:
            self._enqueue({"event": "utterance_end"})
            print(f"[{ts()}] utterance end ({elapsed:.1f}s, {self._streaming_chunks_sent} chunks sent)")
            self.state = "IDLE"
            # Re-arm: don't run inference on stale audio captured mid-utterance.
            self._window[:] = 0.0

    # -- outbound queue --------------------------------------------------

    def _enqueue(self, item) -> None:
        # deque(maxlen=...) silently drops the oldest entry on overflow --
        # exactly the FIFO-drop-oldest behaviour wanted while disconnected.
        self.outbound.append(item)

    # -- mic callback (runs on PortAudio's thread) ------------------------

    def make_audio_callback(self, loop: asyncio.AbstractEventLoop, audio_queue: asyncio.Queue):
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[mic] {status}", file=sys.stderr)
            chunk = indata[:, 0].copy()
            loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)

        return callback

    # -- async tasks -------------------------------------------------------

    async def audio_processor(self, audio_queue: asyncio.Queue) -> None:
        while not self.stop_requested:
            chunk = await audio_queue.get()
            self.handle_chunk(chunk)

    async def sender(self) -> None:
        while not self.stop_requested:
            if self.ws is not None and self.outbound:
                item = self.outbound.popleft()
                try:
                    if isinstance(item, (bytes, bytearray)):
                        await self.ws.send(item)
                    else:
                        await self.ws.send(json.dumps(item))
                except websockets.exceptions.ConnectionClosed:
                    self.outbound.appendleft(item)
                    self.ws = None
                    await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.01)

    async def connection_manager(self) -> None:
        backoff = RECONNECT_BASE_S
        while not self.stop_requested:
            try:
                async with websockets.connect(self.server_url) as ws:
                    self.ws = ws
                    backoff = RECONNECT_BASE_S
                    if self._connected_once:
                        self.reconnects += 1
                        print(f"[{ts()}] reconnected to {self.server_url}")
                    else:
                        self._connected_once = True
                        print(f"[{ts()}] connected to {self.server_url}")
                    await ws.wait_closed()
            except Exception as exc:  # noqa: BLE001 -- any connect/recv failure
                if self._connected_once or backoff == RECONNECT_BASE_S:
                    print(f"[{ts()}] connection error ({exc.__class__.__name__}); retrying in {backoff:.0f}s")
            self.ws = None
            if self.stop_requested:
                break
            print(f"[{ts()}] disconnected -- retrying in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)


# =============================================================================
# main
# =============================================================================

def print_startup_banner(spotter: KeywordSpotter, server_url: str,
                         device_index: int, device_name: str, device_pinned: bool) -> None:
    op = spotter.op
    print("=" * 70)
    print("NAKSHATRA edge agent")
    print("=" * 70)
    print(f"model              : {MODEL_PATH.relative_to(ROOT)}")
    print(f"feature contract   : {spotter.contract_hash}")
    print(f"labels             : {spotter.labels}")
    print(f"operating point    : theta={op['threshold']} k={op['k']} "
          f"window={op['window']} refractory_hops={op['refractory_hops']} "
          f"(source: {spotter.op_source})")
    print(f"server             : {server_url}")
    print(f"input device       : {device_index} -- {device_name}"
          + ("" if device_pinned else "  (Windows default -- pass --device to pin it)"))
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless edge agent for the nakshatra wake-word model.")
    parser.add_argument("--theta", type=float, default=None, help="detection threshold (overrides final_metrics.json)")
    parser.add_argument("--k", type=int, default=None, help="k-of-window count (overrides final_metrics.json)")
    parser.add_argument("--window", type=int, default=None, help="k-of-window size (overrides final_metrics.json)")
    parser.add_argument("--refractory", type=int, default=None, help="refractory period in hops (overrides final_metrics.json)")
    parser.add_argument("--server", type=str, default="ws://localhost:8000/ws/edge", help="server WebSocket URL")
    parser.add_argument("--run-name", type=str, default=RUN_NAME,
                         help="checkpoint subfolder under kws/artifacts/checkpoints/ to load "
                              "(default: %(default)s) -- e.g. --run-name nakshatra_mvp_v1 to roll back to v1")
    parser.add_argument("--device", type=int, default=None,
                        help="input device index (see --list-devices); default: the Windows "
                             "default input, which can change when a headset is plugged in")
    parser.add_argument("--list-devices", action="store_true",
                        help="list input devices and exit")
    parser.add_argument("--debug-probs", action="store_true",
                         help="print raw [silence, unknown, keyword] probabilities every ~500ms, "
                              "regardless of detection -- diagnostic only")
    return parser.parse_args()


async def run(agent: EdgeAgent, device_index: int) -> None:
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue = asyncio.Queue()

    stream = sd.InputStream(
        device=device_index,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=HOP_SAMPLES,
        callback=agent.make_audio_callback(loop, audio_queue),
    )

    stream.start()
    tasks = [
        asyncio.create_task(agent.audio_processor(audio_queue)),
        asyncio.create_task(agent.sender()),
        asyncio.create_task(agent.connection_manager()),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        agent.stop_requested = True
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if agent.ws is not None:
            try:
                await agent.ws.close()
            except Exception:
                pass


def main() -> None:
    global RUN_NAME, CHECKPOINT_DIR, MODEL_PATH, QUANT_PARAMS_PATH, FINAL_METRICS_PATH
    args = parse_args()

    if args.list_devices:
        list_input_devices()
        return
    device_index, device_name = resolve_input_device(args.device, SAMPLE_RATE)

    RUN_NAME = args.run_name
    CHECKPOINT_DIR = KWS_DIR / "artifacts" / "checkpoints" / RUN_NAME
    MODEL_PATH = CHECKPOINT_DIR / "model_int8.tflite"
    QUANT_PARAMS_PATH = CHECKPOINT_DIR / "quant_params.json"
    FINAL_METRICS_PATH = CHECKPOINT_DIR / "final_metrics.json"

    cli_overrides = {
        "threshold": args.theta,
        "k": args.k,
        "window": args.window,
        "refractory_hops": args.refractory,
    }

    spotter = KeywordSpotter(cli_overrides)
    spotter.warmup()
    print_startup_banner(spotter, args.server, device_index, device_name,
                         device_pinned=args.device is not None)

    agent = EdgeAgent(spotter, args.server, debug_probs=args.debug_probs)

    try:
        asyncio.run(run(agent, device_index))
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - agent.start_time
        print("\n" + "=" * 70)
        print("Session summary")
        print("=" * 70)
        print(f"uptime          : {elapsed:.1f} s")
        print(f"detections      : {agent.detections}")
        print(f"reconnects      : {agent.reconnects}")
        print("=" * 70)


if __name__ == "__main__":
    main()

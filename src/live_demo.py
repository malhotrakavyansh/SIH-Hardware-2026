"""
live_demo.py -- live microphone demo of the INT8 "nakshatra" wake-word model.

Streams mic audio, runs the exact same quantize -> tflite -> dequantize ->
softmax -> k-of-window/refractory pipeline as kws/eval_streaming.py, and
shows a large, presentable matplotlib UI (waveform / probability trace /
status light) meant to be recorded on video for the internal round.

Run:
    python live_demo.py

Ctrl+C to stop; prints a detection summary on exit.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sounddevice is optional at install time -- pull it in on first run so the
# demo works on a fresh machine without a manual pip step.
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
except ImportError:
    print("[setup] sounddevice not found -- installing (pip install sounddevice)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
    import sounddevice as sd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent
KWS_DIR = ROOT / "kws"
sys.path.insert(0, str(KWS_DIR))

import config  # noqa: E402  (kws/config.py)
import features  # noqa: E402  (kws/features.py)

import tensorflow as tf  # noqa: E402

RUN_NAME = "nakshatra_mvp_v1"
CHECKPOINT_DIR = KWS_DIR / "artifacts" / "checkpoints" / RUN_NAME
MODEL_PATH = CHECKPOINT_DIR / "model_int8.tflite"
QUANT_PARAMS_PATH = CHECKPOINT_DIR / "quant_params.json"
FINAL_METRICS_PATH = CHECKPOINT_DIR / "final_metrics.json"

SAMPLE_RATE = config.SAMPLE_RATE  # 16000
CLIP_SAMPLES = config.CLIP_SAMPLES  # 16000 (1 sec)
HOP_MS = config.HOP_MS  # 100 ms inference cadence
HOP_SAMPLES_UI = SAMPLE_RATE * HOP_MS // 1000

PROB_HISTORY_SECONDS = 10
FLASH_SECONDS = 1.5


# =============================================================================
# Load model + quant params
# =============================================================================

def load_quant_params() -> dict:
    if not QUANT_PARAMS_PATH.exists():
        raise FileNotFoundError(f"missing {QUANT_PARAMS_PATH}")
    with open(QUANT_PARAMS_PATH, "r") as f:
        return json.load(f)


def load_operating_point(cli_overrides: dict | None = None) -> tuple[dict, str]:
    """Resolve the operating point (theta/k/window/refractory_hops).

    Precedence: CLI args > final_metrics.json > hardcoded defaults. Returns
    (op, source) where source is one of "CLI", "final_metrics.json", or
    "hardcoded defaults", for display in the startup banner.
    """
    defaults = {"threshold": 0.75, "k": 2, "window": 3, "refractory_hops": 15}
    cli_overrides = cli_overrides or {}

    if any(v is not None for v in cli_overrides.values()):
        if FINAL_METRICS_PATH.exists():
            with open(FINAL_METRICS_PATH, "r") as f:
                base = json.load(f).get("operating_point", {})
        else:
            base = {}
        op = {
            "threshold": cli_overrides.get("threshold") if cli_overrides.get("threshold") is not None else base.get("threshold", defaults["threshold"]),
            "k": cli_overrides.get("k") if cli_overrides.get("k") is not None else base.get("k", defaults["k"]),
            "window": cli_overrides.get("window") if cli_overrides.get("window") is not None else base.get("window", defaults["window"]),
            "refractory_hops": cli_overrides.get("refractory_hops") if cli_overrides.get("refractory_hops") is not None else base.get("refractory_hops", defaults["refractory_hops"]),
        }
        return op, "CLI"

    if FINAL_METRICS_PATH.exists():
        with open(FINAL_METRICS_PATH, "r") as f:
            metrics = json.load(f)
        op = metrics.get("operating_point", {})
        return {
            "threshold": op.get("threshold", defaults["threshold"]),
            "k": op.get("k", defaults["k"]),
            "window": op.get("window", defaults["window"]),
            "refractory_hops": op.get("refractory_hops", defaults["refractory_hops"]),
        }, "final_metrics.json"

    return defaults, "hardcoded defaults"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class KeywordSpotter:
    """Wraps the tflite interpreter + quant params + k-of-window/refractory
    post-processing, matching kws/eval_streaming.py's contract exactly."""

    def __init__(self, cli_overrides: dict | None = None):
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
        # Fall back to quant_params.json if the tflite metadata is empty.
        self.in_scale = in_scale or self.quant["input"]["scale"]
        self.in_zero = in_zero or self.quant["input"]["zero_point"]
        self.out_scale = out_scale or self.quant["output"]["scale"]
        self.out_zero = out_zero or self.quant["output"]["zero_point"]

        self.labels = self.quant["labels"]
        self.keyword_index = self.labels.index("keyword")
        self.contract_hash = self.quant["feature_contract_hash"]

        # k-of-window/refractory state
        self.prob_window: collections.deque[float] = collections.deque(
            maxlen=self.op["window"]
        )
        self._last_fire_hop = -self.op["refractory_hops"] - 1
        self._hop_index = 0

    def infer(self, window_f32: np.ndarray) -> float:
        """window_f32: float32 samples in [-1, 1), exactly CLIP_SAMPLES long.
        Returns the keyword posterior for this window."""
        pcm16 = np.clip(np.round(window_f32 * 32768.0), -32768, 32767).astype(np.int16)
        feats = features.extract_mfcc(pcm16)[..., np.newaxis].astype(np.float32)

        q = np.clip(
            np.round(feats / self.in_scale + self.in_zero), -128, 127
        ).astype(np.int8)
        self.interpreter.set_tensor(self.input_details["index"], q[np.newaxis, ...])
        self.interpreter.invoke()
        out_q = self.interpreter.get_tensor(self.output_details["index"])[0]
        logits = (out_q.astype(np.float32) - self.out_zero) * self.out_scale

        return float(softmax(logits)[self.keyword_index])

    def step(self, window_f32: np.ndarray) -> tuple[float, bool]:
        """Run inference on one hop's window and update k-of-window/refractory
        state. Returns (probability, fired)."""
        prob = self.infer(window_f32)
        self.prob_window.append(prob)
        self._hop_index += 1

        fired = False
        window = self.op["window"]
        theta = self.op["threshold"]
        k = self.op["k"]
        refractory_hops = self.op["refractory_hops"]

        if len(self.prob_window) == window:
            count = sum(1 for p in self.prob_window if p > theta)
            if count >= k:
                if self._hop_index - self._last_fire_hop >= refractory_hops:
                    fired = True
                    self._last_fire_hop = self._hop_index

        return prob, fired

    def warmup(self) -> None:
        """Run one throwaway inference on silence so tflite's first-call
        overhead (kernel selection, buffer allocation) doesn't land on the
        first real detection during the demo."""
        silence = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        self.infer(silence)


# =============================================================================
# Mic capture -- callback-based ring buffer of the most recent 1 second
# =============================================================================

class MicRingBuffer:
    def __init__(self, sample_rate: int, seconds: float):
        self.sample_rate = sample_rate
        self.buf = np.zeros(int(sample_rate * seconds), dtype=np.float32)
        self.lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        mono = indata[:, 0]
        with self.lock:
            n = len(mono)
            if n >= len(self.buf):
                self.buf[:] = mono[-len(self.buf):]
            else:
                self.buf[:-n] = self.buf[n:]
                self.buf[-n:] = mono

    def snapshot(self) -> np.ndarray:
        with self.lock:
            return self.buf.copy()


# =============================================================================
# UI
# =============================================================================

class LiveDemoUI:
    def __init__(self, spotter: KeywordSpotter, ring: MicRingBuffer):
        self.spotter = spotter
        self.ring = ring

        self.prob_history_len = int(PROB_HISTORY_SECONDS * 1000 / HOP_MS)
        self.prob_times = collections.deque(maxlen=self.prob_history_len)
        self.prob_values = collections.deque(maxlen=self.prob_history_len)
        self.t0 = time.monotonic()

        self.detections: list[tuple[str, float]] = []
        self._flash_until = 0.0

        matplotlib.rcParams["font.size"] = 14
        self.fig = plt.figure(figsize=(11, 8.5), facecolor="#101418")
        gs = self.fig.add_gridspec(
            3, 1, height_ratios=[2, 2, 1.4], hspace=0.45, top=0.90, bottom=0.10
        )

        op = spotter.op
        self.fig.suptitle(
            "NAKSHATRA -- Live Wake-Word Demo",
            color="white", fontsize=20, fontweight="bold",
        )
        self.fig.text(
            0.5, 0.935,
            f"contract: {spotter.contract_hash}   |   "
            f"theta={op['threshold']}   k={op['k']}/{op['window']}   "
            f"refractory={op['refractory_hops']} hops",
            color="#9aa4ad", fontsize=12, ha="center",
        )

        # --- waveform panel ---
        self.ax_wave = self.fig.add_subplot(gs[0])
        self.ax_wave.set_facecolor("#161b21")
        self.wave_x = np.linspace(-1.0, 0.0, CLIP_SAMPLES)
        (self.wave_line,) = self.ax_wave.plot(self.wave_x, np.zeros(CLIP_SAMPLES), color="#4fd1c5", linewidth=0.8)
        self.ax_wave.set_ylim(-1.05, 1.05)
        self.ax_wave.set_xlim(-1.0, 0.0)
        self.ax_wave.set_title("Microphone (last 1 s)", color="white", fontsize=14)
        self.ax_wave.tick_params(colors="#9aa4ad")
        for spine in self.ax_wave.spines.values():
            spine.set_color("#333c44")

        # --- probability panel ---
        self.ax_prob = self.fig.add_subplot(gs[1])
        self.ax_prob.set_facecolor("#161b21")
        (self.prob_line,) = self.ax_prob.plot([], [], color="#63b3ed", linewidth=2.0)
        self.ax_prob.axhline(op["threshold"], color="#f56565", linestyle="--", linewidth=1.5, label=f"theta={op['threshold']}")
        self.ax_prob.set_ylim(0.0, 1.0)
        self.ax_prob.set_xlim(-PROB_HISTORY_SECONDS, 0.0)
        self.ax_prob.set_title("Keyword probability (last 10 s)", color="white", fontsize=14)
        self.ax_prob.legend(loc="upper left", fontsize=11, facecolor="#161b21", edgecolor="#333c44", labelcolor="white")
        self.ax_prob.tick_params(colors="#9aa4ad")
        for spine in self.ax_prob.spines.values():
            spine.set_color("#333c44")

        # --- status panel ---
        self.ax_status = self.fig.add_subplot(gs[2])
        self.ax_status.set_facecolor("#161b21")
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])
        for spine in self.ax_status.spines.values():
            spine.set_color("#333c44")
        self.status_text = self.ax_status.text(
            0.5, 0.5, "LISTENING...", ha="center", va="center",
            fontsize=34, fontweight="bold", color="#2f855a",
            transform=self.ax_status.transAxes,
        )

        self.anim = FuncAnimation(
            self.fig, self._update, interval=HOP_MS, blit=False, cache_frame_data=False,
        )

    def _update(self, _frame):
        window_f32 = self.ring.snapshot()
        prob, fired = self.spotter.step(window_f32)

        now = time.monotonic() - self.t0
        self.prob_times.append(now)
        self.prob_values.append(prob)

        self.wave_line.set_ydata(window_f32)

        if len(self.prob_times) >= 2:
            rel = np.array(self.prob_times) - self.prob_times[-1]
            self.prob_line.set_data(rel, self.prob_values)

        if fired:
            ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
            print(f"[{ts}] DETECTED - prob={prob:.3f}")
            self.detections.append((ts, prob))
            self._flash_until = time.monotonic() + FLASH_SECONDS

        if time.monotonic() < self._flash_until:
            self.status_text.set_text("NAKSHATRA DETECTED")
            self.status_text.set_color("#c53030")
        else:
            self.status_text.set_text("LISTENING...")
            self.status_text.set_color("#2f855a")

        return self.wave_line, self.prob_line, self.status_text

    def show(self):
        plt.show()


# =============================================================================
# main
# =============================================================================

def print_startup_banner(spotter: KeywordSpotter):
    op = spotter.op
    print("=" * 70)
    print("NAKSHATRA live wake-word demo")
    print("=" * 70)
    print(f"model              : {MODEL_PATH.relative_to(ROOT)}")
    print(f"feature contract   : {spotter.contract_hash}")
    print(f"labels             : {spotter.labels}")
    print(f"operating point    : theta={op['threshold']} k={op['k']} "
          f"window={op['window']} refractory_hops={op['refractory_hops']} "
          f"(source: {spotter.op_source})")
    print("-" * 70)
    print("HOW TO USE")
    print("  1. Make sure your laptop mic is selected as the default input.")
    print("  2. A window will open with a waveform, probability trace, and")
    print("     a big LISTENING / DETECTED status panel.")
    print(f"  3. Say the wake word ('{config.KEYWORD}') -- a detection prints")
    print("     to this terminal and flashes red on screen.")
    print("  4. Press Ctrl+C in this terminal to stop and see a summary.")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live mic demo for the nakshatra INT8 wake-word model.")
    parser.add_argument("--theta", type=float, default=None, help="detection threshold (overrides final_metrics.json)")
    parser.add_argument("--k", type=int, default=None, help="k-of-window count (overrides final_metrics.json)")
    parser.add_argument("--window", type=int, default=None, help="k-of-window size (overrides final_metrics.json)")
    parser.add_argument("--refractory", type=int, default=None, help="refractory period in hops (overrides final_metrics.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    cli_overrides = {
        "threshold": args.theta,
        "k": args.k,
        "window": args.window,
        "refractory_hops": args.refractory,
    }

    spotter = KeywordSpotter(cli_overrides)
    spotter.warmup()
    print_startup_banner(spotter)

    ring = MicRingBuffer(SAMPLE_RATE, seconds=CLIP_SAMPLES / SAMPLE_RATE)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=HOP_SAMPLES_UI,
        callback=ring.callback,
    )

    ui = LiveDemoUI(spotter, ring)

    start_time = time.time()
    stream.start()
    try:
        ui.show()
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown order matters: stop capturing audio first (so the
        # callback thread stops touching the ring buffer / animation
        # state), then tear down the Tk/matplotlib window, then print the
        # summary. Doing this out of order is what produces the
        # "_tkinter.TclError: pyimage3" noise on Ctrl+C.
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

        try:
            plt.close("all")
        except Exception:
            pass

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("Session summary")
        print("=" * 70)
        print(f"duration        : {elapsed:.1f} s")
        print(f"detections      : {len(ui.detections)}")
        for ts, prob in ui.detections:
            print(f"  [{ts}] prob={prob:.3f}")
        print("=" * 70)


if __name__ == "__main__":
    main()

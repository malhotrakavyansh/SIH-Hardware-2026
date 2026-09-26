"""
cloud_server.py -- receives audio from edge_agent.py, transcribes it with
Vosk, matches the transcript to ISRO content, and broadcasts everything to
connected browser UIs in real time.

Endpoints:
    /ws/edge    edge_agent.py (or firmware) connects here -- see PROTOCOL.md
                for the exact wire format this endpoint implements.
    /ws/ui      browsers connect here; receive a broadcast of every event
                the server produces (forwarded edge events, ASR partials/
                finals, ISRO matches, latency breakdowns, connection state).
    /           minimal placeholder UI (src/web/index.html) that dumps raw
                events -- replaced by the real UI in a later task.
    /assets/*   static files (ISRO images) from src/web/assets/

Run:
    python cloud_server.py
    (or: uvicorn cloud_server:app --host 0.0.0.0 --port 8000)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional deps, pulled in on first run.
# ---------------------------------------------------------------------------
def _ensure(pkg_import: str, pip_name: str | None = None):
    try:
        return __import__(pkg_import)
    except ImportError:
        pip_name = pip_name or pkg_import
        print(f"[setup] {pkg_import} not found -- installing (pip install {pip_name})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return __import__(pkg_import)


_ensure("fastapi")
_ensure("uvicorn")
vosk = _ensure("vosk")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from isro_matcher import get_matcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("cloud_server")
vosk.SetLogLevel(-1)  # silence Vosk's own C++ logging

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
VOSK_MODEL_DIR = ROOT / "models" / "vosk" / "vosk-model-small-en-in-0.4"
SAMPLE_RATE = 16000

app = FastAPI(title="Nakshatra cloud server")


# =============================================================================
# Global state
# =============================================================================

class UIHub:
    """Broadcast hub for every /ws/ui client."""

    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def register(self, ws: WebSocket) -> None:
        self.clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        if not self.clients:
            return
        payload = json.dumps(message)
        dead = []
        for client in list(self.clients):
            try:
                await client.send_text(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)


ui_hub = UIHub()
isro_matcher = get_matcher()

vosk_model = None


def resolve_vosk_model(name_or_path: str) -> Path:
    """A bare name is looked up under models/vosk/; anything else is a path."""
    p = Path(name_or_path)
    return p if p.exists() else ROOT / "models" / "vosk" / name_or_path


def load_vosk_model(path: Path) -> None:
    global vosk_model, VOSK_MODEL_DIR
    try:
        if not path.exists():
            raise FileNotFoundError(f"missing Vosk model at {path}")
        t0 = time.perf_counter()
        vosk_model = vosk.Model(str(path))
        VOSK_MODEL_DIR = path
        logger.info("Vosk model loaded from %s in %.2fs", path, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 -- ASR must not prevent server startup
        vosk_model = None
        logger.warning("ASR disabled: could not load Vosk model (%s)", exc)


# When run as a script, main() loads the model after parsing --vosk-model;
# when imported (tests), load the default now.
if __name__ != "__main__":
    load_vosk_model(VOSK_MODEL_DIR)


def now_ms() -> float:
    return time.time() * 1000.0


# =============================================================================
# ASR domain vocabulary -- biases Vosk toward ISRO proper nouns
# =============================================================================
# vosk-model-small-en-in-0.4 is speed-optimized and these proper nouns are
# poorly represented in its LM, so without help "Chandrayaan" routinely comes
# out as unrelated words. KaldiRecognizer's optional third argument (a JSON
# word list) constrains/biases the decoder toward exactly those words --
# since every query here is about an ISRO mission, that trade-off is right.

_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve",
}

_QUERY_FRAMING_WORDS = [
    "tell", "me", "about", "what", "is", "show", "the", "mission", "launch",
    "when", "was", "how", "of", "a", "and", "rocket", "satellite", "moon",
    "mars", "sun", "india", "isro",
]

_VOCAB_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+")


def build_asr_vocabulary(entities: list[dict]) -> list[str]:
    """Every entity name + alias, lowercased and split into individual
    words (letters and digit runs split apart, digits spelled out --
    "Chandrayaan-3" contributes "chandrayaan" and "three"), plus the fixed
    query-framing word list so ordinary sentences still transcribe."""
    words = set(_QUERY_FRAMING_WORDS)
    for ent in entities:
        for text in [ent["name"], *ent.get("aliases", [])]:
            for tok in _VOCAB_TOKEN_RE.findall(text.lower()):
                words.add(_NUM_WORDS.get(tok, tok))
    return sorted(words)


ASR_VOCAB_WORDS = build_asr_vocabulary(isro_matcher.entities)
ASR_VOCAB_GRAMMAR = json.dumps(ASR_VOCAB_WORDS + ["[unk]"])

# Set from main()'s --no-vocab-bias before uvicorn starts serving; read per-
# utterance in UtteranceASR.__init__, which only runs once requests are being
# handled (i.e. always after main() has had a chance to flip this).
VOCAB_BIAS_ENABLED = True


# =============================================================================
# Static files + placeholder UI
# =============================================================================

(WEB_DIR / "assets" / "isro").mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

# NOTE: the catch-all mount that serves "/" (index.html), "/app.js", and
# "/vendor/three.min.js" is registered at the BOTTOM of this file, after the
# /ws/edge and /ws/ui routes -- Starlette matches routes in registration
# order, and a Mount at "/" matches every path prefix, so mounting it here
# would shadow the websocket routes below before they ever get a chance to
# match.


# =============================================================================
# /ws/ui -- browser clients
# =============================================================================

@app.websocket("/ws/ui")
async def ws_ui(websocket: WebSocket) -> None:
    await websocket.accept()
    await ui_hub.register(websocket)
    await websocket.send_text(json.dumps({
        "event": "edge_status",
        "connected": edge_state.connected,
    }))
    try:
        while True:
            # UI clients don't send us anything meaningful; just block until
            # they disconnect so we notice and clean up.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ui_hub.unregister(websocket)


# =============================================================================
# /ws/edge -- the edge device (edge_agent.py today, ESP32 firmware later)
# =============================================================================

class EdgeState:
    """Tracks whether an edge device is currently connected, so a fresh
    /ws/ui client can be told the current status instead of guessing."""

    def __init__(self):
        self.connected = False


edge_state = EdgeState()


class UtteranceASR:
    """One Vosk recognizer + latency timestamps for a single wake -> streaming
    -> utterance_end cycle. Recreated fresh on every `wake` event.
    """

    def __init__(self, t1: float):
        self.t1 = t1        # wake detected (from edge)
        self.t2: float | None = None   # first PCM frame received at server
        self.t3: float | None = None   # first ASR partial produced
        self.last_partial = ""
        # Vosk's endpointer finalises a segment whenever it hears trailing silence
        # (which every real utterance has: the edge only sends utterance_end after
        # 1.5 s of it), so the words arrive as "final" results DURING streaming.
        # FinalResult() at utterance_end then returns only what is left after the
        # last endpoint -- usually "". Keep every finalised segment so
        # final_result() can return the whole utterance.
        self.segments: list[str] = []

        if vosk_model is None:
            self.recognizer = None
        elif VOCAB_BIAS_ENABLED:
            self.recognizer = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE, ASR_VOCAB_GRAMMAR)
        else:
            self.recognizer = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)

    def process_frame(self, pcm: bytes) -> tuple[str, str]:
        """Runs on a worker thread (via run_in_executor) so the receive loop
        never blocks on native Vosk calls. Returns (kind, text) where kind is
        "final", "partial", or "none"."""
        if self.recognizer is None:
            return ("none", "")
        if self.recognizer.AcceptWaveform(pcm):
            text = json.loads(self.recognizer.Result()).get("text", "")
            if text:
                self.segments.append(text)
            return ("final", text)
        partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
        return ("partial", partial)

    def final_result(self) -> str:
        """The whole utterance: every segment Vosk finalised while streaming,
        plus whatever FinalResult() still holds. Called once, after the last
        frame, so no worker thread is appending any more."""
        if self.recognizer is None:
            return ""
        tail = json.loads(self.recognizer.FinalResult()).get("text", "")
        return " ".join([*self.segments, *([tail] if tail else [])])


@app.websocket("/ws/edge")
async def ws_edge(websocket: WebSocket) -> None:
    await websocket.accept()
    edge_state.connected = True
    logger.info("edge connected")
    await ui_hub.broadcast({"event": "edge_status", "connected": True})

    loop = asyncio.get_running_loop()
    utterance: UtteranceASR | None = None

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            data = message.get("bytes")

            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("dropping malformed JSON frame from edge: %r", text[:200])
                    continue
                await _handle_edge_event(payload, loop)
                event = payload.get("event")
                if event == "wake":
                    utterance = UtteranceASR(t1=payload.get("t1", now_ms()))
                elif event == "utterance_end":
                    if utterance is not None:
                        await _finish_utterance(utterance, loop)
                    utterance = None

            elif data is not None:
                if utterance is None:
                    # Defensive: PCM with no active utterance (e.g. we missed
                    # the wake event). Ignore -- never crash the connection.
                    continue
                if utterance.t2 is None:
                    utterance.t2 = now_ms()
                await _handle_pcm_frame(utterance, data, loop)

    except WebSocketDisconnect:
        pass
    finally:
        edge_state.connected = False
        logger.info("edge disconnected")
        await ui_hub.broadcast({"event": "edge_status", "connected": False})


async def _handle_edge_event(payload: dict, loop: asyncio.AbstractEventLoop) -> None:
    event = payload.get("event")
    if event == "level":
        await ui_hub.broadcast({"event": "level", "rms": payload.get("rms")})
    elif event == "wake":
        await ui_hub.broadcast({
            "event": "wake",
            "t1": payload.get("t1"),
            "prob": payload.get("prob"),
        })
    elif event == "utterance_end":
        pass  # handled by _finish_utterance, called by the caller
    else:
        logger.warning("unknown edge event type: %r", event)


async def _handle_pcm_frame(utterance: UtteranceASR, pcm: bytes, loop: asyncio.AbstractEventLoop) -> None:
    kind, value = await loop.run_in_executor(None, utterance.process_frame, pcm)
    if kind == "final" and value:
        await ui_hub.broadcast({"event": "final_segment", "text": value})
    elif kind == "partial" and value and value != utterance.last_partial:
        if utterance.t3 is None:
            utterance.t3 = now_ms()
        utterance.last_partial = value
        await ui_hub.broadcast({"event": "partial", "text": value})


async def _finish_utterance(utterance: UtteranceASR, loop: asyncio.AbstractEventLoop) -> None:
    final_text = await loop.run_in_executor(None, utterance.final_result)
    t4 = now_ms()
    logger.info("utterance transcript %r (%d segment(s) finalised while streaming)",
                final_text, len(utterance.segments))

    # NOTE: T1 (edge) and T2 (server) currently share the same laptop clock,
    # so T2 - T1 is a clean wall-clock latency measurement today. Once the
    # edge is a real ESP32 on the network, T1 comes from a different clock
    # than T2-T5 and this subtraction will need explicit clock-offset
    # correction (e.g. an NTP-style handshake at connect time) or the
    # latency numbers will silently include clock skew.
    latency_ms = (utterance.t2 - utterance.t1) if (utterance.t2 is not None) else None

    await ui_hub.broadcast({
        "event": "transcript",
        "text": final_text,
        "t1": utterance.t1,
        "t2": utterance.t2,
        "t3": utterance.t3,
        "t4": t4,
        "latency_ms": latency_ms,
    })

    match = isro_matcher.match(final_text)
    t5 = now_ms()

    match_payload = None
    if match is not None:
        match_payload = {
            **match["entity"],
            "confidence": round(match["confidence"], 3),
            "matched_by": match["layer"],
        }

    await ui_hub.broadcast({
        "event": "content",
        "transcript": final_text,
        "match": match_payload,
        "t5": t5,
        "match_latency_ms": t5 - utterance.t1,
    })


# Catch-all static mount for everything else under web/ (index.html at "/",
# app.js, vendor/three.min.js, ...). Registered last -- see the NOTE above
# the /assets mount for why this must come after every other route.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# =============================================================================
# main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nakshatra cloud server.")
    parser.add_argument("--no-vocab-bias", action="store_true",
                         help="disable the ISRO domain word list passed to Vosk's KaldiRecognizer, "
                              "for A/B testing against unconstrained transcription")
    parser.add_argument("--vosk-model", default=VOSK_MODEL_DIR.name,
                         help="Vosk model directory name under models/vosk/ (or a full path), "
                              "e.g. vosk-model-en-in-0.5 for A/B against the small model "
                              "(default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    global VOCAB_BIAS_ENABLED
    args = parse_args()
    VOCAB_BIAS_ENABLED = not args.no_vocab_bias
    load_vosk_model(resolve_vosk_model(args.vosk_model))
    logger.info(
        "ASR vocabulary bias: %s (%d words + [unk])",
        "ON" if VOCAB_BIAS_ENABLED else "OFF", len(ASR_VOCAB_WORDS),
    )

    import uvicorn
    # Pass the app object directly, not the "module:attr" string form -- the
    # string form makes uvicorn re-import this file under the module name
    # "cloud_server" even though it's already loaded as "__main__", which
    # silently double-runs every module-level startup step (Vosk model
    # loaded twice, ~2s wasted, double the memory) and serves the second
    # import's app instance while the first import's globals go unused.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()

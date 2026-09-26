# Edge <-> server wire protocol

This document is the contract between the edge device (currently
`edge_agent.py` running on a laptop; eventually ESP32 firmware) and the
server. The firmware implementation must produce byte-identical messages to
`edge_agent.py` so the server never needs to know which one it's talking to.

Transport: a single WebSocket connection, opened by the edge device.

```
ws://<host>:<port>/ws/edge
```

The edge device is always the client; the server never initiates the
connection. There is currently no message the server sends back to the edge
device -- the link is edge -> server only.

## State machine

The edge device is always in exactly one of two states:

```
IDLE -----[wake-word detected]-----> STREAMING
STREAMING --[silence or 8s cap]----> IDLE
```

- **IDLE**: mic audio is scored against the wake-word model every 100 ms
  (`HOP_MS`). No raw audio is sent, only periodic level readings.
- **STREAMING**: no wake-word inference runs. Raw PCM is streamed to the
  server, starting with a pre-roll buffer captured before the wake event.

There are no other states. A firmware port must not add hidden states
(e.g. a "connecting" state that changes what it sends) -- if the WebSocket
is down, the edge device keeps running the same IDLE/STREAMING logic and
buffers what it would have sent (see "Disconnection behaviour" below).

## Message types

Two kinds of frames are sent, both edge -> server:

- **Text frames**: a single JSON object per frame, UTF-8 encoded. One event
  per frame -- never batched.
- **Binary frames**: raw PCM audio, sent only while in STREAMING.

### `level` (text, IDLE only)

Sent roughly every 200 ms (every 2nd 100 ms hop) so the server/UI can show a
live input-level meter without transmitting audio.

```json
{"event": "level", "rms": 0.0173}
```

| field | type  | meaning                                             |
|-------|-------|------------------------------------------------------|
| rms   | float | root-mean-square of the last 100 ms chunk, samples in [-1, 1) |

### `wake` (text, IDLE -> STREAMING transition)

Sent the instant the k-of-window/refractory detector fires, before any
audio frames for this utterance.

```json
{"event": "wake", "t1": 1732364821123.4, "prob": 0.87}
```

| field | type  | meaning                                                        |
|-------|-------|-----------------------------------------------------------------|
| t1    | float | detection timestamp, milliseconds since Unix epoch (`time.time() * 1000`) |
| prob  | float | keyword posterior probability that triggered detection (0-1)   |

`t1` is the reference point the server uses to align this device's clock
against its own; capture it as close to the detection instant as the
platform allows (on the ESP32 side, immediately after the model inference
that crosses threshold, before any network call).

### Binary PCM frames (STREAMING only)

Raw audio, no envelope/header -- the frame body *is* the sample data.

- Encoding: **signed 16-bit PCM, little-endian**
- Channels: **1 (mono)**
- Sample rate: **16000 Hz**
- Frame size: **1600 samples = 3200 bytes = 100 ms** per frame (`HOP_SAMPLES`
  in `kws/config.py`)

The first frames of an utterance are the **pre-roll**: the ~300 ms
(`PREROLL_CHUNKS = 3` frames) of audio already buffered at the moment `wake`
fired, sent in chronological order, immediately followed by live audio
frames as they're captured. Without the pre-roll the start of the word
right after the wake word gets clipped by detection latency.

### `utterance_end` (text, STREAMING -> IDLE transition)

Sent once, after the last PCM frame of an utterance, when either stop
condition is met:

```json
{"event": "utterance_end"}
```

Stop conditions (either one ends the utterance):

- **Silence**: RMS of the last 100 ms chunk stays below `0.02` for a
  continuous `1.5` s (`STREAMING_SILENCE_S`).
- **Hard cap**: `8.0` s (`STREAMING_MAX_S`) of streaming elapsed, regardless
  of silence.

No fields. The next message after this is either another `wake`/PCM
sequence or a `level` event, once the device is back in IDLE.

## Full message order for one detection

```
level                    (IDLE, repeating ~every 200ms)
level
...
wake            {t1, prob}
<binary PCM>    pre-roll frame 1  (~100ms, oldest)
<binary PCM>    pre-roll frame 2
<binary PCM>    pre-roll frame 3  (~newest of the pre-roll)
<binary PCM>    live frame 1
<binary PCM>    live frame 2
...
<binary PCM>    live frame N
utterance_end
level                    (back in IDLE)
...
```

## Disconnection behaviour

If the WebSocket connection drops, the edge device does **not** stop
detecting or stop running its state machine -- it keeps producing the same
events and PCM frames, and queues them locally:

- Buffer capacity: ~10 s of messages (`OUTBOUND_QUEUE_MAXLEN = 100` items,
  matched to one message roughly every 100 ms in the worst case).
- Overflow policy: **FIFO, drop oldest** -- once full, the oldest buffered
  message is discarded to make room for the newest one. The server should
  therefore expect gaps in `level`/PCM sequences after a long outage, not
  reordering.
- Reconnection: retried with exponential backoff starting at 2 s, doubling
  up to a 10 s ceiling, resetting to 2 s after a successful connect.
- On reconnect, the entire buffer is flushed to the server in original
  order before any newly generated message is sent.

## Ambiguity resolutions (added when the server was implemented)

- **One edge connection at a time.** The protocol above assumes a single
  active edge device. `cloud_server.py` accepts only one `/ws/edge`
  connection as "current" at a time; if a second one connects, it replaces
  the first (the server logs this). Firmware should not open a second
  connection while the first is still alive -- reconnect logic (see
  "Disconnection behaviour") already handles the single-device case.
- **`t1`/`t2` latency validity across a reconnect.** If the WebSocket drops
  mid-utterance and PCM frames are replayed from the edge's local buffer on
  reconnect, the server's arrival timestamp for those frames no longer
  reflects real-time capture -- the resulting `T2 - T1` measurement for that
  utterance is not a true network-latency sample and should be treated as an
  outlier, not trusted data. This only affects the rare disconnect-during-
  utterance case; the common path (connected throughout) is unaffected.

## Server -> UI broadcast events

Separate from the edge<->server contract above, the server also broadcasts
JSON text events to every browser connected at `/ws/ui` (fan-out, no binary
frames, no reply channel). This section documents that protocol too since
it's also a wire format implementers may need. It is **not** something
firmware needs to produce -- only `cloud_server.py` emits these.

| event            | fields                                                          | when |
|------------------|------------------------------------------------------------------|------|
| `edge_status`    | `connected` (bool)                                                | edge device connects/disconnects, so the UI can show a disconnected state instead of freezing on stale data |
| `level`          | `rms` (float)                                                     | forwarded verbatim from the edge's `level` events |
| `wake`           | `t1` (float), `prob` (float)                                      | forwarded verbatim from the edge's `wake` event |
| `partial`        | `text` (string)                                                   | Vosk partial-result text changed since the last partial |
| `final_segment`  | `text` (string)                                                   | Vosk `AcceptWaveform` returned a completed segment mid-utterance |
| `transcript`     | `text`, `t1`, `t2`, `t3`, `t4`, `latency_ms`                       | full transcript assembled after `utterance_end`; `t3`/`t4` may be `null` if no speech was detected. `latency_ms` is `t2 - t1`, the headline judged metric. `t5` is **not** included here -- see `content` below. |
| `content`        | `transcript`, `match` (entity object + `confidence`/`matched_by`, or `null`), `t5`, `match_latency_ms` | broadcast right after `transcript`, once the ISRO matcher has run; `match_latency_ms` is `t5 - t1` |

The full T1-T5 latency waterfall for one utterance is therefore split across
two consecutive broadcasts (`transcript` then `content`), not one -- T5
literally cannot be known until after the matcher runs, which happens after
the transcript is assembled.

## Things intentionally left out of the protocol

- No ack/ping frames from the server are required for the edge device to
  operate; it is a fire-and-forget stream keyed by the `wake` /
  `utterance_end` boundaries.
- No audio is sent during IDLE beyond the `level.rms` scalar -- this keeps
  idle bandwidth and MCU workload minimal.
- No compression/codec: raw int16 PCM only, to keep the firmware port
  trivial (no codec library needed on the MCU).

## Implementation notes (ambiguities resolved during server implementation)

These weren't fully specified above; this is the behaviour `cloud_server.py`
actually implements, and what a firmware port should assume.

- **One edge connection at a time.** The server does not multiplex multiple
  simultaneous edge devices on `/ws/edge` -- it tracks a single
  connected/disconnected boolean. If a second edge device connects while
  another is active, the server does not reject it, but downstream state
  (the in-flight `UtteranceASR`) is shared, so two edges streaming
  concurrently will interleave PCM into the same recognizer and produce
  garbage transcripts. Treat `/ws/edge` as single-tenant for now.
- **`utterance_end` must precede the next `wake`.** The server relies on the
  state-machine invariant in this document (STREAMING excludes new `wake`
  events) and does not defend against a `wake` arriving before the previous
  utterance's `utterance_end`. If that invariant is ever violated (e.g. a
  buggy firmware port), the server silently discards the unfinished
  utterance's ASR state and starts fresh on the new `wake` -- no error is
  raised. Keep the invariant intact rather than relying on this behaviour.
- **PCM frames without a preceding `wake` are dropped.** If the server
  receives a binary frame while it has no in-flight utterance (e.g. it
  missed the `wake` message, or connected mid-utterance), it silently
  ignores the frame rather than crashing the connection. This should not
  happen in normal operation since `wake` is always sent before any PCM.
- **T1 vs. T2 clock assumption.** `t1` (wake time, from the edge) and the
  server's own `t2`/`t3`/`t4`/`t5` timestamps are only directly comparable
  when edge and server share a clock, which is true today (`edge_agent.py`
  and `cloud_server.py` run on the same machine). Once the edge is a real
  network-attached device, `t2 - t1` conflates true network/processing
  latency with clock skew between the two devices, and will need an
  explicit clock-sync step (e.g. an NTP-style handshake at connect time) to
  stay meaningful.

## Server -> UI broadcast events (for reference, not part of the edge contract)

`cloud_server.py` forwards/derives a second, one-way protocol from
`/ws/ui` to connected browsers. This is not a contract the firmware needs to
implement -- it exists purely between the server and the browser UI -- but
is documented here since it shares the same JSON-text-frame convention.

| event | fields | when |
|---|---|---|
| `edge_status` | `connected: bool` | on every `/ws/ui` connect (current state), and whenever the edge connects/disconnects |
| `level` | `rms: float` | forwarded from the edge's `level` event |
| `wake` | `t1: float, prob: float` | forwarded from the edge's `wake` event |
| `partial` | `text: str` | a Vosk partial hypothesis changed |
| `final_segment` | `text: str` | Vosk finalized one segment mid-utterance |
| `transcript` | `text: str, t1, t2, t3, t4, latency_ms` | full utterance transcript, once `utterance_end` is processed |
| `content` | `transcript: str, match: object\|null, confidence: float\|null, layer: "fuzzy"\|"embedding"\|null, t5: float` | ISRO knowledge-base match result for the transcript |

`match`, when not `null`, is the full entity object from `isro_kb.json`
(`id`, `name`, `aliases`, `one_line`, `key_facts`, `image`, `launch_date`,
`status`, `org_unit`).

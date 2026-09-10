# data/positives/

Your own recordings of the wake word (`config.KEYWORD`). Label: `keyword`.

- 16 kHz, mono, 16-bit PCM `.wav`, ~1 s each.
- Filename convention: `<speaker>_nohash_<n>.wav` — `dataset.speaker_id()`
  parses this, and `dataset.which_set()` splits on the speaker so one voice
  never lands in two splits.
- Target for a working demo: **≥ 20 speakers**, ~30 utterances each. Fewer
  speakers is the single most common reason a KWS model tests at 98% and then
  fails on a stranger.
- Vary deliberately: distance from mic, room, loudness, speed, and the
  phonetic run-in/run-out (say it mid-sentence, not just in isolation).

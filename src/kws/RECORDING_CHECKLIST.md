# Recording checklist -- new dataset on the demo capture chain

Single speaker, two mics, everything through `record_session.py` (sounddevice,
16 kHz mono). Never Audacity. Files go to `kws/data/recordings/<label>/`, every
take is logged in `kws/data/recordings/manifest.csv`.

**Targets (recorded, real):**

| | laptop | Jabra | total |
|---|---|---|---|
| positives, training | 600 | 300 | **900** |
| hard negatives, training | 200 | 100 | **300** |
| ambient, training | 14 min | 6 min | **20 min** |
| held-out test (laptop, recorded LAST) | 60 pos + 30 hard neg + 5 min ambient | -- | -- |

Speaking time is about 3 hours, spread over **three days**. Don't try to do it
in one sitting (see "Fatigue rules").

All commands run from `src/`. `L` = laptop mic index (currently **1**, MME).
`J` = Jabra index -- **indices change when devices are plugged in/out**, so run
`python record_session.py --list-devices` at the start of every sitting and use
the MME entry for each mic.

---

## Fatigue rules (read first)

- **Sittings of at most ~20 minutes / ~150 takes**, then a 10-minute break with
  water. At most three sittings per day.
- Stop the sitting early if your voice gets hoarse, or if you notice yourself
  saying the word the same way every time -- repeating one word hundreds of
  times flattens the delivery before real vocal fatigue sets in, and those
  near-identical takes add little.
- Stick to the on-screen style prompt (normal / quiet / loud / slow / fast);
  it's what keeps late takes as varied as early ones.
- If the printed RMS drifts steadily down or durations steadily shorten across
  a session, that's fatigue -- end the sitting.

---

## 0. Capture chain

**Rule: do not change any Windows audio setting until the demo is recorded.**
No enhancement toggles, no input-volume changes, no default-device switches,
no driver/Windows-update-driven tinkering. The mic runs through the same chain
`edge_agent.py` uses, so training and inference see identical processing, and
that only holds if the chain stays frozen from the first recording through the
demo. (Windows enhancements can't be turned off on this machine -- that's fine.)

- [ ] Close apps that grab the mic (Teams, Zoom, Discord).
- [ ] Run ambient session a01 first. It is a **baseline fingerprint** of the
      chain, not a gate: note its `exact-zero samples X%` (a high value, e.g.
      ~15%, just means Windows is gating the mic) and keep the file. Re-run the
      same a01 command later; a big change in that number means the chain
      moved.
- [ ] The demo runs on this exact chain (same mic, same settings,
      `edge_agent.py --device L`).

---

## Day 1 -- laptop

### Ambient chain check (first thing)

| done | session | what | command |
|---|---|---|---|
| [ ] | a01 | quiet room, 180 s (baseline fingerprint) | `python record_session.py --device L --mic laptop --label ambient --prefix a01 --seconds 180 --cond "quiet room"` |

### Positives -- laptop (sessions of 80, about 11 min each)

Each session cycles speaking style on screen: normal / quiet / loud / slow /
fast. Press **r** in the pause after a bad take to redo it, **q** to stop. An
interrupted session resumes with the same command (numbering continues).

| done | session | distance | position | count | command |
|---|---|---|---|---|---|
| [ ] | s01 | 30 cm | facing   | 80 | `python record_session.py --device L --mic laptop --label positive --prefix s01 --count 80 --prompts "normal,quiet,loud,slow,fast" --cond "30cm facing"` |
| [ ] | s02 | 1 m   | facing   | 80 | same, `--prefix s02 --cond "1m facing"` |
| [ ] | s03 | 2 m   | facing   | 80 | same, `--prefix s03 --cond "2m facing"` |
| [ ] | s04 | 30 cm | off-axis (60-90 deg to the side) | 80 | same, `--prefix s04 --cond "30cm off-axis"` |

### Hard negatives -- laptop

Default prompts cycle: kshatriya, lakshan, nakli, natak, raksha, naksha,
rashtra, shastra, chhatra, akshar. Say the word shown, naturally.

| done | session | condition | count | command |
|---|---|---|---|---|
| [ ] | s11 | 30 cm facing | 50 | `python record_session.py --device L --mic laptop --label hardneg --prefix s11 --count 50 --cond "30cm facing"` |
| [ ] | s12 | 1 m facing   | 50 | same, `--prefix s12 --cond "1m facing"` |

---

## Day 2 -- laptop (continued)

| done | session | what | count | command |
|---|---|---|---|---|
| [ ] | s05 | positives, 1 m off-axis | 80 | `python record_session.py --device L --mic laptop --label positive --prefix s05 --count 80 --prompts "normal,quiet,loud,slow,fast" --cond "1m off-axis"` |
| [ ] | s06 | positives, 2 m off-axis | 80 | same, `--prefix s06 --cond "2m off-axis"` |
| [ ] | s07 | positives, demo seat/posture exactly as in the video | 120 | same, `--prefix s07 --count 120 --cond "demo position"` (take a break at 60) |
| [ ] | s13 | hard negatives, 2 m facing   | 50 | `python record_session.py --device L --mic laptop --label hardneg --prefix s13 --count 50 --cond "2m facing"` |
| [ ] | s14 | hard negatives, 1 m off-axis | 50 | same, `--prefix s14 --cond "1m off-axis"` |

Laptop ambient (the talking session matters most -- continuous speech without
the wake word is exactly where v1 false-triggers):

| done | session | what | seconds | command |
|---|---|---|---|---|
| [ ] | a02 | fan/AC on + typing on the laptop | 180 | `python record_session.py --device L --mic laptop --label ambient --prefix a02 --seconds 180 --cond "fan + typing"` |
| [ ] | a03 | you talking continuously, no wake word (read ISRO text aloud, describe the project, ask demo questions) | 300 | same, `--prefix a03 --seconds 300 --cond "talking no keyword"` |
| [ ] | a04 | the actual demo room, if you can | 180 | same, `--prefix a04 --seconds 180 --cond "demo room"` |

Laptop totals: 600 positives, 200 hard negatives, 14 min ambient.

---

## Day 3 -- Jabra, then the held-out test set

Jabra boom distance is fixed, so vary boom placement and room instead.

| done | session | what | count | command |
|---|---|---|---|---|
| [ ] | s08 | positives, boom normal | 120 | `python record_session.py --device J --mic jabra --label positive --prefix s08 --count 120 --prompts "normal,quiet,loud,slow,fast" --cond "boom normal"` (break at 60) |
| [ ] | s09 | positives, boom pushed down/away from mouth | 90 | same, `--prefix s09 --count 90 --cond "boom away"` |
| [ ] | s10 | positives, fan/AC on, boom normal | 90 | same, `--prefix s10 --count 90 --cond "fan on"` |
| [ ] | s15 | hard negatives, boom normal | 50 | `python record_session.py --device J --mic jabra --label hardneg --prefix s15 --count 50 --cond "boom normal"` |
| [ ] | s16 | hard negatives, boom away   | 50 | same, `--prefix s16 --cond "boom away"` |
| [ ] | a05 | ambient, quiet room, 120 s | -- | `python record_session.py --device J --mic jabra --label ambient --prefix a05 --seconds 120 --cond "quiet room"` |
| [ ] | a06 | ambient, talking no wake word, 240 s | -- | same, `--prefix a06 --seconds 240 --cond "talking no keyword"` |

Jabra totals: 300 positives, 100 hard negatives, 6 min ambient.

---

## HELD-OUT TEST SET -- record LAST, never train on it

Record after every training session is done -- after a proper break, or on a
fourth day. Session id **test** on every file. These files are the honest
number: never copy them into training, never tune thresholds by looking at
them, never re-record them to "fix" a bad result.

Laptop mic, demo chain, mixed conditions:

| done | what | count | command |
|---|---|---|---|
| [ ] | positives, 30 cm | 20 | `python record_session.py --device L --mic laptop --label positive --prefix test --count 20 --prompts "normal,quiet,loud,slow,fast" --cond "test 30cm"` |
| [ ] | positives, 1 m   | 20 | same command again with `--cond "test 1m"` (numbering continues) |
| [ ] | positives, 2 m   | 20 | same with `--cond "test 2m"` |
| [ ] | hard negatives, 1 m | 30 | `python record_session.py --device L --mic laptop --label hardneg --prefix test --count 30 --cond "test 1m"` |
| [ ] | talking, no wake word | 180 s | `python record_session.py --device L --mic laptop --label ambient --prefix test --seconds 180 --cond "test talking"` |
| [ ] | quiet room | 120 s | same, `--seconds 120 --cond "test quiet"` |

The talking + quiet ambient gives a false-triggers-per-hour estimate on the
demo chain (5 min with 0 false triggers still only bounds it to about 36/hour
at 95% -- record longer if you want a tighter number). Note this test set is
still your voice: it measures the demo condition, not other speakers.

---

## Watch for during recording

- Each take prints `duration  RMS  peak`. Redo (press **r**) if flagged
  `CLIPPED`, `very quiet`, `long -- two words or noise?`, or `very short`.
- Positive/hard-negative takes should be ~0.9-1.6 s (includes 0.5 s of
  padding). Several `hit 4s cap` in a row means background noise is keeping the
  recording open: rerun that session with a higher (less sensitive)
  threshold, e.g. `--threshold -25`.
- "no speech detected" repeating means the onset threshold is too high for
  that distance/loudness: add `--threshold -50`.
- Keep speaking style honest -- "quiet" at 2 m is supposed to be hard.

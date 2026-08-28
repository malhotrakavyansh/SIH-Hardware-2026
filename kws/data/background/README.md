# data/background/

Long continuous recordings. Two jobs:

1. **Source for the `silence` class** — random `config.CLIP_MS` excerpts.
2. **Augmentation noise** mixed into every other class
   (`config.BACKGROUND_PROB`, `config.SNR_DB_RANGE`).
3. **The `eval_streaming.py` denominator** — false accepts per *hour* only
   means something if there are hours of audio here.

What to put in it:

- GSC's `_background_noise_/`.
- **Hours of the actual deployment environment** with the wake word never
  spoken: room tone, fans, traffic, crowd noise, a TV, background
  conversation. Long files (minutes, not seconds) — `dataset.random_background`
  crops from them.

16 kHz mono `.wav`. No speaker convention needed; these are not split by
speaker.

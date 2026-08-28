# data/hard_neg/

Confusable words — the near-misses that a naive model fires on. Label:
`unknown`.

These matter more than the bulk negatives. Pick words sharing the wake word's
syllable structure, stress pattern and leading consonant, plus partial
utterances (the first syllable alone, the word cut short, the word said inside
a longer phrase).

Same 16 kHz mono `.wav` and `<speaker>_nohash_<n>.wav` convention as
`positives/`.

Every false accept you see in `eval_streaming.py` should come back here as a
new recording. That loop is what drives false-accepts/hour down.

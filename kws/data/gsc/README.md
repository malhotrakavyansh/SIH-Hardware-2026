# data/gsc/

Google Speech Commands v2 — bulk negatives. Label: `unknown`.

    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Extract so the per-word folders sit directly here:

    data/gsc/yes/  no/  up/  down/  ...  _background_noise_/

Already 16 kHz mono with the `<speaker>_nohash_<n>.wav` convention, which is
where that convention comes from.

Move `_background_noise_/` into `data/background/` — this pipeline treats it
as a noise source and a `silence` source, not as an `unknown` word.

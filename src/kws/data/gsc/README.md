# data/gsc/

Google Speech Commands v2 — bulk negatives. Label: `unknown`.

    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Extract so the per-word folders sit directly here:

    data/gsc/yes/  no/  up/  down/  ...  _background_noise_/

Already 16 kHz mono with the `<speaker>_nohash_<n>.wav` convention, which is
where that convention comes from.

Move `_background_noise_/` into `data/background/` — this pipeline treats it
as a noise source and a `silence` source, not as an `unknown` word.

## Not in git

This folder's contents (~105,800 wav files) are **never committed**; only this README and `.gitkeep` are.
Download it yourself:

    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

(Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv:1804.03209 --
CC BY 4.0. Version 0.02 = 105,829 one-second clips; `find data/gsc -name "*.wav" | wc -l` should
report about that many plus the `_background_noise_` files.)

**Extract into a temporary folder and copy the per-word directories in.** Extracting the tarball
straight into `data/gsc/` overwrites this README with the dataset's own README.md.

    mkdir gsc_tmp && tar -xzf speech_commands_v0.02.tar.gz -C gsc_tmp
    cp -r gsc_tmp/*/ src/kws/data/gsc/          # yes/ no/ up/ ... (and _background_noise_/)
    mv src/kws/data/gsc/_background_noise_/* src/kws/data/background/

Training expects the word folders directly under `data/gsc/`, and `dataset.py` holds out speakers by
hashing the `<speaker>_nohash_<n>.wav` prefix, so do not rename or re-split the files.

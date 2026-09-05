"""
dataset.py -- loading, splitting, augmentation.

Sources under data/:
    positives/   your own recordings of config.KEYWORD  -> label "keyword"
    hard_neg/    confusable words, near-misses          -> label "unknown"
    gsc/         Google Speech Commands                 -> label "unknown"
    background/  long continuous noise / speech         -> label "silence"
                 (also the mixing source for augmentation)

Splitting is hash-based on the speaker id so one speaker never appears in two
splits -- a random per-file split leaks speaker identity and inflates val
accuracy by several points.

Implemented in Step 3.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf

import config
import features


@dataclass(frozen=True)
class Sample:
    """One training item, before any audio is read."""

    path: Path
    label_index: int      # index into config.LABELS
    speaker: str          # used by which_set(); "" for synthetic silence
    source: str           # "positives" | "hard_neg" | "gsc" | "background"


def speaker_id(path: Path) -> str:
    """Extract the speaker id from a filename.

    Convention (shared by our own recordings and GSC): the speaker is
    everything before the first underscore --
        '<speaker>_nohash_<n>.wav'   (positives, GSC)
        '<speaker>_<word>_<n>.wav'   (hard negatives, named after the
                                      confusable word they contain)

    Files with no underscore (synthetic silence, or anything without a clear
    speaker) return "" so which_set() routes them straight to 'train'.
    """
    stem = path.stem
    if "_" not in stem:
        return ""
    return stem.split("_", 1)[0]


def which_set(speaker: str) -> str:
    """Deterministic 'train' | 'val' | 'test' assignment for a speaker.

    Same trick as Google Speech Commands' which_set(): hash the speaker id,
    take the hash mod 100 as a percentage, and bucket it against
    config.VALIDATION_PERCENT / config.TESTING_PERCENT. Hashing (rather than
    e.g. a running counter) is what makes this stable -- a speaker's bucket
    depends only on their own id, so adding more speakers later never moves
    an existing one across the train/val/test boundary.

    Empty speaker (synthetic silence, no clear speaker) always goes to
    'train' -- there's no identity to leak, so there's no reason to hold it
    out.
    """
    if speaker == "":
        return "train"

    digest = hashlib.sha1(speaker.encode("utf-8")).hexdigest()
    # First 8 hex chars (32 bits) is plenty of entropy for a mod-100 bucket;
    # cap at MAX_WAVS_PER_CLASS the way the GSC reference does, purely to
    # keep the percentage computation in the same integer range they used.
    hash_int = int(digest[:8], 16)
    percentage = (hash_int % (config.MAX_WAVS_PER_CLASS + 1)) * (100.0 / config.MAX_WAVS_PER_CLASS)

    if percentage < config.TESTING_PERCENT:
        return "test"
    elif percentage < (config.TESTING_PERCENT + config.VALIDATION_PERCENT):
        return "val"
    else:
        return "train"


def load_wav(path: Path) -> np.ndarray:
    """Read a wav as float32 mono at config.SAMPLE_RATE.

    Raises if the file is not already at the target rate -- resampling
    belongs in a preprocessing pass, not silently in the training loop, since
    a silent resample would mean train and eval could each be seeing subtly
    different filtering without anyone noticing.
    """
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != config.SAMPLE_RATE:
        raise ValueError(
            f"{path}: sample rate {sample_rate} Hz != config.SAMPLE_RATE "
            f"({config.SAMPLE_RATE} Hz). Resample it offline before adding "
            f"it to data/ -- this loader deliberately does not resample."
        )
    if samples.shape[1] > 1:
        samples = samples.mean(axis=1, keepdims=True)
    return samples[:, 0].astype(np.float32)


def fit_to_clip(samples: np.ndarray, offset: int = 0) -> np.ndarray:
    """Pad or crop to exactly config.CLIP_SAMPLES, honouring a shift offset.

    Baseline placement is centered. `offset` then slides that centered
    window: positive shifts it later in the clip (skip more from the front
    when cropping / pad more at the front when padding), negative shifts it
    earlier. This is what augment() uses for +/-100ms position jitter -- a
    keyword should be recognized wherever it lands in the window, not just
    dead center.
    """
    target = config.CLIP_SAMPLES
    n = samples.shape[0]

    if n >= target:
        base_start = (n - target) // 2
        start = base_start + offset
        start = max(0, min(start, n - target))
        cropped = samples[start:start + target]
        return cropped.astype(np.float32)
    else:
        pad_total = target - n
        base_left = pad_total // 2
        left = base_left + offset
        left = max(0, min(left, pad_total))
        right = pad_total - left
        padded = np.pad(samples, (left, right), mode="constant")
        return padded.astype(np.float32)


def _wav_files(directory: Path) -> list[Path]:
    """Every non-hidden *.wav directly under `directory`, sorted for determinism.

    Sorting matters here: os/Path directory iteration order is not guaranteed,
    and an unstable manifest order would make the seeded shuffle in
    make_splits() non-reproducible across machines.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix == ".wav" and not p.name.startswith(".")
    )


def build_manifest() -> list[Sample]:
    """Walk data/ and produce every Sample, with labels and speakers resolved.

    data/background/ is deliberately excluded -- those clips are the mixing
    source for augmentation (random_background(), Batch 3), never training
    samples in their own right.
    """
    manifest: list[Sample] = []

    for path in _wav_files(config.POSITIVES_DIR):
        manifest.append(Sample(
            path=path,
            label_index=config.KEYWORD_INDEX,
            speaker=speaker_id(path),
            source="positives",
        ))

    for path in _wav_files(config.HARD_NEG_DIR):
        manifest.append(Sample(
            path=path,
            label_index=config.UNKNOWN_INDEX,
            speaker=speaker_id(path),
            source="hard_neg",
        ))

    # GSC ships as <word>/<speaker>_nohash_<n>.wav -- one subdirectory per
    # word. May not be downloaded yet, so a missing/empty gsc/ is fine.
    if config.GSC_DIR.is_dir():
        for word_dir in sorted(p for p in config.GSC_DIR.iterdir() if p.is_dir()):
            for path in _wav_files(word_dir):
                manifest.append(Sample(
                    path=path,
                    label_index=config.UNKNOWN_INDEX,
                    speaker=speaker_id(path),
                    source="gsc",
                ))

    print(f"build_manifest: {len(manifest)} samples")
    by_source: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for s in manifest:
        by_source[s.source] = by_source.get(s.source, 0) + 1
        label = config.LABELS[s.label_index]
        by_label[label] = by_label.get(label, 0) + 1
    print(f"  by source : {by_source}")
    print(f"  by label  : {by_label}")

    return manifest


def make_splits(
    manifest: list[Sample],
    split_mode: str = "speaker_disjoint",
) -> dict[str, list[Sample]]:
    """Group a manifest into {'train': [...], 'val': [...], 'test': [...]}.

    split_mode:
      'speaker_disjoint' (default, correct for production) -- groups by
          which_set() on each sample's speaker, so no speaker straddles two
          splits. Needs enough distinct speakers that val/test aren't empty
          or dominated by a single voice; with ~15-20+ speakers the hash
          bucketing in which_set() evens out.

      'random_per_file' -- ⚠️  MVP-ONLY escape hatch for the single-speaker
          bring-up phase. Shuffles the manifest with a config.SEED-seeded RNG
          and slices by config.VALIDATION_PERCENT / config.TESTING_PERCENT,
          ignoring speaker identity entirely. This LEAKS speaker identity
          across splits (val/test see the same voice as train) and will
          inflate accuracy relative to real-world, multi-speaker performance.
          It exists only so the pipeline can be exercised end-to-end before
          more speakers are recorded -- switch back to 'speaker_disjoint' the
          moment there are enough speakers for val/test to be meaningful.

    Also enforces the unknown/keyword class mix from config.UNKNOWN_PERCENT
    within each split (subsampling excess unknowns with the same seeded RNG).
    Silence is not enforced here -- there are no real silence samples in the
    manifest; they're synthesized from data/background/ at training time by
    augment() (Batch 3).
    """
    if split_mode not in ("speaker_disjoint", "random_per_file"):
        raise ValueError(f"unknown split_mode: {split_mode!r}")

    rng = np.random.default_rng(config.SEED)

    if split_mode == "random_per_file":
        print(
            "\n[!] USING RANDOM PER-FILE SPLIT -- MVP MODE ONLY. "
            "Switch to speaker_disjoint before multi-speaker eval.\n"
        )
        shuffled = list(manifest)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = round(n * config.TESTING_PERCENT / 100.0)
        n_val = round(n * config.VALIDATION_PERCENT / 100.0)

        splits = {
            "test": shuffled[:n_test],
            "val": shuffled[n_test:n_test + n_val],
            "train": shuffled[n_test + n_val:],
        }
    else:
        splits = {"train": [], "val": [], "test": []}
        for sample in manifest:
            splits[which_set(sample.speaker)].append(sample)

    # Enforce unknown quota relative to keyword count, per split.
    for name, samples in splits.items():
        keyword_count = sum(1 for s in samples if s.label_index == config.KEYWORD_INDEX)
        unknown_quota = round(keyword_count * config.UNKNOWN_PERCENT / 100.0)

        unknowns = [s for s in samples if s.label_index == config.UNKNOWN_INDEX]
        others = [s for s in samples if s.label_index != config.UNKNOWN_INDEX]

        if len(unknowns) > unknown_quota:
            keep_idx = rng.choice(len(unknowns), size=unknown_quota, replace=False)
            unknowns = [unknowns[i] for i in sorted(keep_idx)]

        splits[name] = others + unknowns

    print("silence samples are injected per split in make_dataset() from data/background/")

    print("\nmake_splits summary:")
    for name in ("train", "val", "test"):
        samples = splits[name]
        by_label: dict[str, int] = {}
        for s in samples:
            label = config.LABELS[s.label_index]
            by_label[label] = by_label.get(label, 0) + 1
        print(f"  {name:5s}: {len(samples)} total, by label = {by_label}")

    return splits


_BACKGROUND_FILES: list[Path] | None = None


def _background_files() -> list[Path]:
    """Cached listing of data/background/*.wav.

    Populated once on first use rather than per-call -- random_background()
    runs once per augmented sample, i.e. thousands of times per epoch, and
    re-scanning the directory every time would be pure waste.
    """
    global _BACKGROUND_FILES
    if _BACKGROUND_FILES is None:
        _BACKGROUND_FILES = _wav_files(config.BACKGROUND_DIR)
    return _BACKGROUND_FILES


def random_background(rng: np.random.Generator) -> np.ndarray:
    """Random config.CLIP_SAMPLES-long excerpt from data/background/.

    We don't have real background recordings for the single-speaker MVP yet,
    so an empty/missing data/background/ returns silence (all zeros) instead
    of raising -- training must still run, just without noise mixing.
    """
    files = _background_files()
    if not files:
        return np.zeros(config.CLIP_SAMPLES, dtype=np.float32)

    path = files[rng.integers(len(files))]
    audio = load_wav(path)

    if audio.shape[0] <= config.CLIP_SAMPLES:
        return fit_to_clip(audio)  # short file: zero-pad, centered

    max_start = audio.shape[0] - config.CLIP_SAMPLES
    start = int(rng.integers(0, max_start + 1))
    return audio[start:start + config.CLIP_SAMPLES].astype(np.float32)


def augment(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train-time augmentation chain, applied to an already config.CLIP_SAMPLES
    -long clip.

    time shift (config.TIME_SHIFT_MS) -> speed perturb
    (config.SPEED_PERTURB_PROB, factor in {0.9, 1.1}) -> background mix
    (config.BACKGROUND_PROB, SNR in {0, 5, 10, 20} dB, capped by
    config.BACKGROUND_VOL_MAX) -> gain (config.GAIN_DB_RANGE) -> clip to
    [-1, 1].

    Never applied to val/test -- make_dataset() only calls this when
    training=True.
    """
    # -- time shift --
    # samples is already exactly CLIP_SAMPLES long, so fit_to_clip's crop
    # branch has no slack to shift within (start clamps to 0). Pad both ends
    # by the max shift first so there IS slack, then let fit_to_clip's
    # offset-aware crop do the actual windowing -- reuses its logic instead
    # of re-deriving the pad/crop arithmetic here.
    shift_max = config.SAMPLE_RATE * config.TIME_SHIFT_MS // 1000  # ms -> samples, allow-literal
    offset = int(rng.integers(-shift_max, shift_max + 1))
    padded = np.pad(samples, (shift_max, shift_max), mode="constant")
    samples = fit_to_clip(padded, offset=offset)

    # -- speed perturbation --
    if rng.random() < config.SPEED_PERTURB_PROB:
        factor = float(rng.choice([0.9, 1.1]))
        new_len = int(round(samples.shape[0] / factor))
        resampled = scipy.signal.resample(samples, new_len).astype(np.float32)
        samples = fit_to_clip(resampled)

    # -- background noise mix --
    if rng.random() < config.BACKGROUND_PROB:
        snr_db = float(rng.choice([0, 5, 10, 20]))
        bg = random_background(rng)
        bg_rms = np.sqrt(np.mean(bg ** 2))
        if bg_rms > 0:
            signal_rms = np.sqrt(np.mean(samples ** 2))
            desired_bg_rms = signal_rms / (10.0 ** (snr_db / 20.0))
            scale = desired_bg_rms / bg_rms
            # Cap peak background amplitude so a quiet signal (small
            # signal_rms) can't back the noise up to full volume.
            peak = np.max(np.abs(bg))
            if peak > 0:
                scale = min(scale, config.BACKGROUND_VOL_MAX / peak)
            samples = samples + scale * bg

    # -- gain --
    gain_db = rng.uniform(*config.GAIN_DB_RANGE)
    samples = samples * (10.0 ** (gain_db / 20.0))

    # -- prevent overflow before quantization to int16 in make_dataset --
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def _samples_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16, the input dtype features.extract_mfcc()
    expects (it mirrors what the ESP32 ADC hands the firmware front end)."""
    return np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int16)


_SILENCE_SENTINEL = "__silence__"  # Sample.path placeholder for injected silence


def _load_features_and_label(
    path: bytes, label_index: int, index: int, training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs inside tf.numpy_function -- everything here is plain numpy."""
    path_str = path.decode("utf-8")

    # A fresh Generator per call: cheap, reproducible given (SEED, index), and
    # safe under tf.data's parallel map (no shared mutable RNG state that
    # concurrent calls could corrupt). Used for the background pick below
    # (silence samples) and/or augmentation -- same seed for both, so a given
    # dataset "slot" is fully deterministic run to run.
    rng = np.random.default_rng([config.SEED, int(index)])

    if path_str == _SILENCE_SENTINEL:
        # Deterministic even when training=False: val/test must inject the
        # same silence clip every evaluation, not a fresh random one.
        clip = random_background(rng)
    else:
        clip = fit_to_clip(load_wav(Path(path_str)))

    if training:
        clip = augment(clip, rng)

    feats = features.extract_mfcc(_samples_to_pcm16(clip))
    feats = feats[..., np.newaxis].astype(np.float32)  # (NUM_FRAMES, NUM_MFCC, 1)

    label = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    label[label_index] = 1.0

    return feats, label


def make_dataset(split: list[Sample], training: bool) -> tf.data.Dataset:
    """Build the batched, feature-extracted pipeline for one split.

    Yields (features, label) with features float32 (NUM_FRAMES, NUM_MFCC, 1)
    and one-hot labels float32 (NUM_CLASSES,), batched to config.BATCH_SIZE.

    MFCC extraction (features.extract_mfcc -- the exact front end the ESP32
    firmware runs, not tf.signal or librosa) and augmentation are plain numpy,
    so they're wrapped in tf.numpy_function rather than reimplemented in
    tf ops. That keeps this pipeline CPU-bound and non-GPU-parallel, but MFCC
    extraction is the bottleneck regardless of the wrapping at this dataset
    size.

    Silence (label "silence") has no real on-disk samples -- data/background/
    is mixing-source audio, not a labeled class. This injects
    config.SILENCE_PERCENT% (relative to this split's keyword count) of
    synthetic silence samples, each a random_background() clip. Injected for
    both training and eval splits: training=False still adds them (so val/
    test actually measure silence rejection), but deterministically -- the
    same clips every evaluation, via the index-seeded RNG in
    _load_features_and_label.
    """
    keyword_count = sum(1 for s in split if s.label_index == config.KEYWORD_INDEX)
    num_silence = round(keyword_count * config.SILENCE_PERCENT / 100.0)
    silence_samples = [
        Sample(path=Path(_SILENCE_SENTINEL), label_index=config.SILENCE_INDEX,
               speaker="", source="background")
        for _ in range(num_silence)
    ]
    full_split = list(split) + silence_samples
    print(f"make_dataset(training={training}): {len(split)} samples "
          f"+ {num_silence} synthetic silence")

    paths = [str(s.path) for s in full_split]
    label_indices = [s.label_index for s in full_split]
    indices = list(range(len(full_split)))

    ds = tf.data.Dataset.from_tensor_slices((paths, label_indices, indices))

    def _map_fn(path, label_index, index):
        feats, label = tf.numpy_function(
            func=lambda p, l, i: _load_features_and_label(p, l, i, training),
            inp=[path, label_index, index],
            Tout=(tf.float32, tf.float32),
        )
        feats.set_shape(config.FEATURE_SHAPE + (1,))
        label.set_shape((config.NUM_CLASSES,))
        return feats, label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        buffer_size = max(1, min(len(full_split), 1000))  # shuffle-buffer memory cap, allow-literal
        ds = ds.shuffle(
            buffer_size=buffer_size, seed=config.SEED, reshuffle_each_iteration=True
        )

    # drop_remainder=False even for training: with this dataset's size, a
    # tiny partial last batch is a far smaller cost than starving BatchNorm's
    # per-epoch update count by discarding it every epoch.
    ds = ds.batch(config.BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    print(config.summary())

    print("\n--- Batch 1 smoke test ---")

    sample_path = config.POSITIVES_DIR / "harshit_nohash_1.wav"
    sid = speaker_id(sample_path)
    print(f"speaker_id({sample_path.name!r}) = {sid!r}")

    split = which_set(sid)
    print(f"which_set({sid!r}) = {split!r}")
    # Deterministic: calling again must give the identical answer.
    assert which_set(sid) == split, "which_set is not deterministic!"

    wav = load_wav(sample_path)
    print(f"load_wav shape = {wav.shape}, dtype = {wav.dtype}")

    clipped = fit_to_clip(wav)
    print(f"fit_to_clip shape = {clipped.shape}, dtype = {clipped.dtype}")
    assert clipped.shape[0] == config.CLIP_SAMPLES

    print("\n--- fit_to_clip padding test (synthetic zeros) ---")
    short = np.zeros(8000, dtype=np.float32)
    short[3990:4010] = 1.0  # a little "blip" so we can see where it lands

    centered = fit_to_clip(short, offset=0)
    print(f"centered: shape={centered.shape}, "
          f"blip at [{np.argmax(centered)}]")
    assert centered.shape[0] == config.CLIP_SAMPLES

    test_offset = 480  # arbitrary nonzero shift, in samples
    shifted_later = fit_to_clip(short, offset=test_offset)
    print(f"offset=+{test_offset}: blip at [{np.argmax(shifted_later)}] "
          f"(should be ~{test_offset} samples later than centered)")

    shifted_earlier = fit_to_clip(short, offset=-test_offset)
    print(f"offset=-{test_offset}: blip at [{np.argmax(shifted_earlier)}] "
          f"(should be ~{test_offset} samples earlier than centered)")

    print("\nAll Batch 1 checks passed.")

    print("\n--- Batch 2 smoke test ---")

    manifest = build_manifest()

    # Only harshit recorded so far -- speaker_disjoint would starve val/test.
    splits = make_splits(manifest, split_mode="random_per_file")
    assert set(splits.keys()) == {"train", "val", "test"}
    assert sum(len(v) for v in splits.values()) <= len(manifest)

    print("\nAll Batch 2 checks passed.")

    print("\n--- Batch 3 smoke test ---")

    train_ds = make_dataset(splits["train"], training=True)
    train_feats, train_labels = next(iter(train_ds))
    print(f"train batch: features shape={train_feats.shape}, "
          f"dtype={train_feats.dtype}")
    print(f"train batch: labels   shape={train_labels.shape}, "
          f"dtype={train_labels.dtype}")
    feats_np = train_feats.numpy()
    print(f"train features range: min={feats_np.min():.3f}, "
          f"max={feats_np.max():.3f}, mean={feats_np.mean():.3f}")
    label_counts = train_labels.numpy().sum(axis=0)
    print(f"train label distribution (silence/unknown/keyword) = "
          f"{label_counts.tolist()}")

    expected_shape = (config.BATCH_SIZE,) + config.FEATURE_SHAPE + (1,)
    assert tuple(train_feats.shape) == expected_shape, train_feats.shape
    assert tuple(train_labels.shape) == (config.BATCH_SIZE, config.NUM_CLASSES)

    val_ds = make_dataset(splits["val"], training=False)
    val_feats, val_labels = next(iter(val_ds))
    print(f"val batch:   features shape={val_feats.shape}, "
          f"dtype={val_feats.dtype}")
    print(f"val batch:   labels   shape={val_labels.shape}, "
          f"dtype={val_labels.dtype}")
    assert val_feats.shape[1:] == train_feats.shape[1:]
    assert val_labels.shape[1:] == train_labels.shape[1:]

    print("\nAll Batch 3 checks passed.")

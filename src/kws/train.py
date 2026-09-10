"""
train.py -- float training loop.

Every run prints config.summary() first and writes it next to the checkpoint,
so a checkpoint can never be separated from the feature contract it was
trained under.

Implemented in Step 5.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
import dataset
from model import build_dscnn_s, macs, param_count

RUN_NAME = "nakshatra_mvp_v1"


def set_seeds(seed: int = config.SEED) -> None:
    """Seed python / numpy / the framework RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def lr_schedule(epoch: int) -> float:
    """Step decay: config.LEARNING_RATE, dropped by config.LR_DECAY_FACTOR at
    each boundary in config.LR_DECAY_EPOCHS."""
    lr = config.LEARNING_RATE
    for boundary in config.LR_DECAY_EPOCHS:
        if epoch >= boundary:
            lr *= config.LR_DECAY_FACTOR
    return lr


class _KeywordPrecision(tf.keras.metrics.Metric):
    """Precision for the keyword class only -- argmax-based, so it's
    unaffected by whether y_pred is logits or a softmax (both share the
    same argmax)."""

    def __init__(self, name: str = "keyword_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_idx = tf.argmax(y_true, axis=-1)
        pred_idx = tf.argmax(y_pred, axis=-1)
        is_pred_kw = tf.equal(pred_idx, config.KEYWORD_INDEX)
        is_true_kw = tf.equal(true_idx, config.KEYWORD_INDEX)
        self.tp.assign_add(tf.reduce_sum(
            tf.cast(is_pred_kw & is_true_kw, tf.float32)))
        self.fp.assign_add(tf.reduce_sum(
            tf.cast(is_pred_kw & ~is_true_kw, tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fp + 1e-7)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)


class _KeywordRecall(tf.keras.metrics.Metric):
    """Recall for the keyword class only -- the number that actually matters
    for a wake word: missed activations are worse than a false unknown."""

    def __init__(self, name: str = "keyword_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_idx = tf.argmax(y_true, axis=-1)
        pred_idx = tf.argmax(y_pred, axis=-1)
        is_pred_kw = tf.equal(pred_idx, config.KEYWORD_INDEX)
        is_true_kw = tf.equal(true_idx, config.KEYWORD_INDEX)
        self.tp.assign_add(tf.reduce_sum(
            tf.cast(is_pred_kw & is_true_kw, tf.float32)))
        self.fn.assign_add(tf.reduce_sum(
            tf.cast(~is_pred_kw & is_true_kw, tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fn + 1e-7)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


def _add_class_sample_weight(features, label):
    """Per-class sample_weight (config.KEYWORD_CLASS_WEIGHT,
    config.UNKNOWN_CLASS_WEIGHT, silence=1.0 baseline) via a per-example
    sample_weight, rather than Keras's class_weight= kwarg -- that path is
    finicky with one-hot targets coming from a tf.data.Dataset, while
    sample_weight is unambiguous.

    Unknown got its own weight after adding the silence class diluted the
    keyword/unknown boundary (confusable-word rejection regressed once
    unknown had to compete with silence for training signal too)."""
    class_idx = tf.argmax(label, axis=-1)
    class_weights = [1.0] * config.NUM_CLASSES
    class_weights[config.UNKNOWN_INDEX] = config.UNKNOWN_CLASS_WEIGHT
    class_weights[config.KEYWORD_INDEX] = config.KEYWORD_CLASS_WEIGHT
    weight = tf.gather(tf.constant(class_weights, dtype=tf.float32), class_idx)
    return features, label, weight


def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    width = max(len(l) for l in labels) + 2
    header = " " * width + "".join(f"{l:>{width}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{width}}" + "".join(f"{v:>{width}}" for v in cm[i])
        print(row)


def main() -> None:
    """Train DS-CNN-S and write the best checkpoint to config.CHECKPOINT_DIR."""
    print(config.summary())

    set_seeds()

    run_dir = config.CHECKPOINT_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- data --
    manifest = dataset.build_manifest()
    # Single-speaker MVP: speaker_disjoint would starve val/test entirely.
    splits = dataset.make_splits(manifest, split_mode="random_per_file")
    print(
        f"\nsplit sizes: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    train_ds = dataset.make_dataset(splits["train"], training=True)
    train_ds = train_ds.map(_add_class_sample_weight)
    val_ds = dataset.make_dataset(splits["val"], training=False)
    test_ds = dataset.make_dataset(splits["test"], training=False)

    # -- model --
    model = build_dscnn_s()
    print()
    param_count(model)
    print()
    macs(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", _KeywordPrecision(), _KeywordRecall()],
    )

    checkpoint_path = run_dir / "float.keras"
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        # val_loss, not val_accuracy: with val split heavily skewed toward
        # keyword (10/11), accuracy stays ~flat/uninformative for many
        # epochs while loss keeps moving -- val_loss is the signal that
        # actually reflects whether the model is still improving.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]
    # No EarlyStopping: config.EPOCHS is a ~30s run on this dataset size, and
    # the tiny val split's early plateaus previously triggered a premature
    # stop (6 real gradient steps) well before BatchNorm had converged.
    # ModelCheckpoint(save_best_only=True) still protects against overfitting
    # in the later epochs by keeping the best val_loss checkpoint, not
    # necessarily the last one.

    epochs = config.EPOCHS

    print(f"\nTraining for {epochs} epochs (no early stopping)...\n")

    start = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed:.1f}s")

    # -- test evaluation --
    print("\n--- Test evaluation ---")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print(f"test loss     : {test_metrics['loss']:.4f}")
    print(f"test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"test keyword precision : {test_metrics['keyword_precision']:.4f}")
    print(f"test keyword recall    : {test_metrics['keyword_recall']:.4f}")

    y_true_batches = []
    for _, y in test_ds:
        y_true_batches.append(np.argmax(y.numpy(), axis=-1))
    y_true = np.concatenate(y_true_batches)

    y_pred_logits = model.predict(test_ds)
    y_pred = np.argmax(y_pred_logits, axis=-1)

    print("\nPer-class precision/recall/F1:")
    print(classification_report(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
        target_names=config.LABELS, zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    print("Confusion matrix (rows=true, cols=pred):")
    _print_confusion_matrix(cm, config.LABELS)

    # -- checkpoint must carry its feature contract with it --
    (run_dir / "config_summary.txt").write_text(config.summary(), encoding="utf-8")

    print(f"\nSaved checkpoint : {checkpoint_path}")
    print(f"Saved log        : {run_dir / 'training_log.csv'}")
    print(f"Saved contract   : {run_dir / 'config_summary.txt'}")


if __name__ == "__main__":
    main()

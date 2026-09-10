"""
check_parity.py -- Step 2 gating check.

Compares Python's reference MFCC (from make_test_vectors.py) against what
the firmware printed over serial for the *same hardcoded int16 input*.

Usage:
    python check_parity.py <clip_name> <esp32_dump.csv>

    python check_parity.py tone_440hz   dumps/tone_440hz.csv
    python check_parity.py silence      dumps/silence.csv
    python check_parity.py white_noise  dumps/white_noise.csv

<clip_name> is one of the names make_test_vectors.py produced (tone_440hz,
silence, white_noise). <esp32_dump.csv> is whatever the firmware printed,
reshaped to (config.NUM_FRAMES, config.NUM_MFCC) -- i.e. the same layout as
test_vectors/<clip_name>_mfcc_reference.csv. Adjust load_dump() if the
firmware's actual serial format differs (e.g. one flat line instead of 49).

Done-criterion from the spec: max abs diff < 0.05 on all 3 test clips. This
unblocks everything downstream -- don't move to Step 3 until all 3 pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import config

TOLERANCE = 0.05  # from the Step 2 spec -- do not loosen without discussion

VECTORS_DIR = config.ROOT / "test_vectors"


def load_dump(path: Path) -> np.ndarray:
    """Load a firmware MFCC dump and reshape it to config.FEATURE_SHAPE.

    Accepts either 49 lines of 10 comma-separated values, or one flat line
    of 490 values -- whichever the firmware's serial print happens to use.
    """
    flat = np.loadtxt(path, delimiter=",").astype(np.float32).ravel()
    if flat.size != config.FEATURE_SIZE:
        raise ValueError(
            f"{path}: expected {config.FEATURE_SIZE} values "
            f"({config.FEATURE_SHAPE}), got {flat.size}"
        )
    return flat.reshape(config.FEATURE_SHAPE)


def check_one(clip_name: str, dump_path: Path) -> bool:
    ref_path = VECTORS_DIR / f"{clip_name}_mfcc_reference.npy"
    if not ref_path.exists():
        print(
            f"FAIL  {clip_name}: no reference at {ref_path} "
            f"-- run make_test_vectors.py first"
        )
        return False

    reference = np.load(ref_path)
    dumped = load_dump(dump_path)

    if dumped.shape != reference.shape:
        print(
            f"FAIL  {clip_name}: shape mismatch, "
            f"reference {reference.shape} vs dump {dumped.shape}"
        )
        return False

    diff = np.abs(reference - dumped)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    frame, coeff = np.unravel_index(np.argmax(diff), diff.shape)

    passed = max_diff < TOLERANCE
    status = "PASS" if passed else "FAIL"
    print(
        f"{status}  {clip_name}: max diff {max_diff:.6f} "
        f"(tolerance {TOLERANCE}), mean diff {mean_diff:.6f}, "
        f"worst at frame {frame} coeff {coeff} "
        f"(py={reference[frame, coeff]:.6f}, esp={dumped[frame, coeff]:.6f})"
    )
    return passed


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    clip_name, dump_arg = sys.argv[1], sys.argv[2]
    dump_path = Path(dump_arg)
    if not dump_path.exists():
        print(f"FAIL  {dump_arg} does not exist")
        return 2

    ok = check_one(clip_name, dump_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

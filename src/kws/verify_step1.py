"""
verify_step1.py -- enforces the Step 1 done-criterion.

  "config.py exists and is imported everywhere. No magic numbers elsewhere."

Three checks:
  1. every module in the pipeline exists and imports config
  2. config's own invariants hold (config._validate on import)
  3. no frozen config value appears as a bare numeric literal in any other file

Check 3 is a duplicate-constant lint, not a general magic-number linter: it
only complains about numbers that are already named in config. Escape hatch is
a trailing `# allow-literal` on the offending line.

Run:  python verify_step1.py
"""

from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path

import config

MODULES = [
    "config.py",
    "features.py",
    "dataset.py",
    "model.py",
    "train.py",
    "quantize.py",
    "eval_streaming.py",
    "export_c.py",
]

DATA_DIRS = [
    config.POSITIVES_DIR,
    config.HARD_NEG_DIR,
    config.GSC_DIR,
    config.BACKGROUND_DIR,
]

# Only integers this large are worth linting; below it, literals like 0/1/2 are
# ordinary indexing and flagging them would be noise.
MIN_LINTED_VALUE = 32

ESCAPE = "allow-literal"

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def frozen_values() -> dict[int, list[str]]:
    """Map every sufficiently large int constant in config to its name(s)."""
    values: dict[int, list[str]] = {}
    for name in dir(config):
        if name.startswith("_") or name.islower():
            continue
        value = getattr(config, name)
        items = value if isinstance(value, tuple) else [value]
        for item in items:
            if isinstance(item, bool) or not isinstance(item, int):
                continue
            if abs(item) < MIN_LINTED_VALUE:
                continue
            values.setdefault(item, []).append(name)
    return values


def escaped_lines(source: str) -> set[int]:
    """Line numbers carrying an `# allow-literal` comment."""
    lines = set()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok in tokens:
        if tok.type == tokenize.COMMENT and ESCAPE in tok.string:
            lines.add(tok.start[0])
    return lines


def lint_literals(path: Path, values: dict[int, list[str]]) -> list[str]:
    """Report frozen config values hardcoded in `path`."""
    source = path.read_text(encoding="utf-8")
    allowed = escaped_lines(source)
    problems = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok in tokens:
        if tok.type != tokenize.NUMBER or tok.start[0] in allowed:
            continue
        try:
            value = int(tok.string, 0)
        except ValueError:
            continue
        names = values.get(value)
        if names:
            problems.append(
                f"{path.name}:{tok.start[0]}: literal {tok.string} duplicates "
                f"config.{' / config.'.join(names)}"
            )
    return problems


def check_modules() -> list[str]:
    problems = []
    for name in MODULES:
        path = config.ROOT / name
        if not path.exists():
            problems.append(f"missing module: {name}")
            continue
        if name == "config.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "import config" not in source:
            problems.append(f"{name}: does not import config")
    return problems


def check_data_dirs() -> list[str]:
    return [f"missing data dir: {d.relative_to(config.ROOT)}"
            for d in DATA_DIRS if not d.is_dir()]


def main() -> int:
    values = frozen_values()
    problems: list[str] = []

    problems += check_modules()
    problems += check_data_dirs()

    for name in MODULES:
        path = config.ROOT / name
        if name == "config.py" or not path.exists():
            continue
        problems += lint_literals(path, values)

    print(config.summary())
    print()
    print(f"{DIM}linting {len(values)} frozen values across "
          f"{len(MODULES) - 1} modules{RESET}")
    print()

    if problems:
        for problem in problems:
            print(f"{RED}FAIL{RESET}  {problem}")
        print(f"\n{RED}Step 1 not done: {len(problems)} problem(s).{RESET}")
        return 1

    print(f"{GREEN}PASS{RESET}  config.py is the single source of truth")
    print(f"{GREEN}PASS{RESET}  all {len(MODULES)} modules present, "
          f"all import config")
    print(f"{GREEN}PASS{RESET}  no frozen value hardcoded outside config.py")
    print(f"{GREEN}PASS{RESET}  data/ tree in place")
    print(f"\n{GREEN}Step 1 done.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

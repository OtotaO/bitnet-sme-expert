"""Gold-dataset loader with an explicit, committed train/holdout split.

Each domain has two version-controlled files under ``datasets/``:

    {domain}.train.jsonl     — used to compile/optimize programs
    {domain}.holdout.jsonl   — used to *report* scores; never trained on

Keeping the partition in git (rather than a runtime random shuffle) means every
eval run and every optimizer compile sees the same split, so a reported
baseline-vs-compiled delta is reproducible and can't silently overfit to the
data the optimizer already saw. ``scripts/optimize.py`` trains on ``train`` and
measures on ``holdout``; ``scripts/run_eval.py`` and the pytest baseline tests
score on ``holdout``.

This module is the single source of truth for both the file-naming convention
and the per-domain ``dspy.Example`` input fields, so the scripts and the test
fixtures can't drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy

DATA_DIR = Path(__file__).parent / "datasets"

# Which fields are *inputs* to each domain's program (the remaining fields —
# answer / must_contain — are the labels the metric grades against).
DOMAIN_INPUTS: dict[str, tuple[str, ...]] = {
    "math": ("question",),
    "code": ("request", "language"),
    "general": ("question",),
}

SPLITS = ("train", "holdout")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_split(domain: str, split: str) -> list[dspy.Example]:
    """Return the ``dspy.Example`` list for ``domain``/``split``.

    ``split`` is one of :data:`SPLITS`. Raises ``ValueError`` for an unknown
    domain or split, and ``FileNotFoundError`` if the dataset file is missing.
    """
    if domain not in DOMAIN_INPUTS:
        raise ValueError(f"unknown domain: {domain!r} (expected one of {sorted(DOMAIN_INPUTS)})")
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r} (expected one of {list(SPLITS)})")
    inputs = DOMAIN_INPUTS[domain]
    rows = _load_jsonl(DATA_DIR / f"{domain}.{split}.jsonl")
    return [dspy.Example(**row).with_inputs(*inputs) for row in rows]

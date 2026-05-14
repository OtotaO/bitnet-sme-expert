"""Eval-specific fixtures: gold dataset loaders and shared metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import dspy
import pytest

DATA_DIR = Path(__file__).parent / "datasets"


def _load_jsonl(name: str) -> list[dict]:
    path = DATA_DIR / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def math_examples() -> list[dspy.Example]:
    return [dspy.Example(**row).with_inputs("question") for row in _load_jsonl("math.jsonl")]


@pytest.fixture(scope="session")
def code_examples() -> list[dspy.Example]:
    return [
        dspy.Example(**row).with_inputs("request", "language") for row in _load_jsonl("code.jsonl")
    ]


@pytest.fixture(scope="session")
def general_examples() -> list[dspy.Example]:
    return [dspy.Example(**row).with_inputs("question") for row in _load_jsonl("general.jsonl")]


def pytest_collection_modifyitems(config, items) -> None:
    """Skip eval tests unless ``RUN_EVAL=1`` is set (they require an LM)."""
    if os.environ.get("RUN_EVAL") == "1":
        return
    skip = pytest.mark.skip(reason="Eval tests require an LM; set RUN_EVAL=1 to run.")
    for item in items:
        if "tests/eval/" in str(item.fspath):
            item.add_marker(skip)

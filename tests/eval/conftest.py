"""Eval-specific fixtures: gold dataset loaders and shared metrics.

The baseline ``*_examples`` fixtures resolve to the **holdout** split — the
honest measure of generalization, since the optimizer only ever sees ``train``.
``*_trainset`` fixtures are provided for tests that need the compile set.
"""

from __future__ import annotations

import os

import dspy
import pytest

from tests.eval.loader import load_split


@pytest.fixture(scope="session")
def math_examples() -> list[dspy.Example]:
    return load_split("math", "holdout")


@pytest.fixture(scope="session")
def code_examples() -> list[dspy.Example]:
    return load_split("code", "holdout")


@pytest.fixture(scope="session")
def general_examples() -> list[dspy.Example]:
    return load_split("general", "holdout")


@pytest.fixture(scope="session")
def math_trainset() -> list[dspy.Example]:
    return load_split("math", "train")


@pytest.fixture(scope="session")
def code_trainset() -> list[dspy.Example]:
    return load_split("code", "train")


@pytest.fixture(scope="session")
def general_trainset() -> list[dspy.Example]:
    return load_split("general", "train")


def pytest_collection_modifyitems(config, items) -> None:
    """Skip eval tests unless ``RUN_EVAL=1`` is set (they require an LM)."""
    if os.environ.get("RUN_EVAL") == "1":
        return
    skip = pytest.mark.skip(reason="Eval tests require an LM; set RUN_EVAL=1 to run.")
    for item in items:
        if "tests/eval/" in str(item.fspath):
            item.add_marker(skip)

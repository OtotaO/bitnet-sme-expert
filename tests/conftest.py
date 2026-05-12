"""Pytest fixtures for dspy-sme-expert.

The key fixture is ``stub_lm`` — a ``dspy.utils.DummyLM`` configured with the
canned outputs each test needs. This lets us exercise the full DSPy pipeline
without hitting real provider APIs.
"""

from __future__ import annotations

import os
from collections.abc import Generator, Iterable

import dspy
import pytest

# Make sure config is loaded against a known-safe env before app modules import.
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("ENABLE_RATE_LIMITING", "false")
os.environ.setdefault("ENABLE_CACHING", "false")


@pytest.fixture
def stub_responses() -> dict[str, object]:
    """Canned predictions keyed by signature/module class name."""
    return {}


@pytest.fixture
def stub_lm(stub_responses: dict[str, object]) -> Generator[dspy.LM, None, None]:
    """Install a DummyLM as the global DSPy LM for the duration of the test."""
    answers: Iterable = stub_responses.get("answers") or [
        {"reasoning": "stub", "answer": "stub-answer", "explanation": "stub", "code": "pass",
         "domain": "general", "confidence": 0.9}
    ]
    lm = dspy.utils.DummyLM(answers)
    with dspy.context(lm=lm, async_max_workers=1):
        yield lm


@pytest.fixture
def sample_math_query() -> dict[str, object]:
    return {"question": "What is the derivative of x^2 + 3x + 2?", "domain": "math"}


@pytest.fixture
def sample_code_query() -> dict[str, object]:
    return {
        "question": "Write a Python function to calculate fibonacci numbers",
        "domain": "code",
        "context": {"language": "python"},
    }


@pytest.fixture
def sample_general_query() -> dict[str, object]:
    return {"question": "What is the capital of France?", "domain": "general"}

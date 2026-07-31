"""Unit tests for the MathProgram deterministic fast-path (no LM required).

Locks in two recent fixes: the fast-path returns *exact* values (not lossy
floats), and the sympy guard refuses oversized / pathological input instead of
hanging.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.dspy_modules import math_module
from app.dspy_modules.math_module import (
    _MAX_EXPR_LEN,
    _bounded,
    _eval_exact,
    _reject_reason,
    _try_fast_path,
)
from app.schemas.request import MAX_QUESTION_LEN, CollaborateRequest, QueryRequest


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is 2 + 2?", "4"),
        ("Calculate 15 * 24", "360"),
        ("1/2 + 1/3", "5/6"),  # exact rational, not 0.8333…
        ("144 / 12", "12"),  # exact integer, not 12.000…
        ("3 + 4 * 2", "11"),
        ("2^10", "1024"),
    ],
)
def test_fast_path_is_exact(question: str, expected: str) -> None:
    assert _try_fast_path(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is the derivative of x^2?",  # free symbol -> ReAct, not fast-path
        "Who discovered calculus?",  # not arithmetic
        "",  # empty
    ],
)
def test_fast_path_falls_through(question: str) -> None:
    assert _try_fast_path(question) is None


def test_fast_path_refuses_oversized_input() -> None:
    # A long arithmetic string matches the regex but must not reach sympy.
    assert _try_fast_path("1+" * _MAX_EXPR_LEN + "1") is None


def test_bounded_length_guard() -> None:
    assert _bounded(_eval_exact, "9" * (_MAX_EXPR_LEN + 1)).startswith("[refused")


# ---------------------------------------------------------------------------
# Regression: unbounded exponentiation must be refused BEFORE sympy runs.
#
# `9^9^9^9` is seven characters, so it passed the length cap, was rewritten to
# `9**9**9**9`, and wedged a worker in the shared sympy thread pool while
# allocating until the process OOMed. The timeout bounded latency, not work.
# These tests assert the refusal happens without evaluating anything — if the
# guard regresses, the test itself hangs the suite rather than passing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "9^9^9^9",  # the reported payload
        "9**9**9**9",
        "2**3**4**5",
        "2^999999999",  # single huge literal exponent
        "10**10000000",
        "(1+2)^(9^9^9)",  # tower nested inside another expression
    ],
)
def test_rejects_unbounded_exponentiation(expression: str) -> None:
    assert _reject_reason(expression) is not None
    assert _bounded(_eval_exact, expression).startswith("[refused")


def test_rejection_is_immediate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must refuse without ever submitting work to the executor."""
    submitted: list[object] = []

    def _explode(*args: object, **kwargs: object) -> None:
        submitted.append(args)
        raise AssertionError("executor must not be reached for a refused expression")

    monkeypatch.setattr(math_module._executor, "submit", _explode)
    assert _bounded(_eval_exact, "9^9^9^9").startswith("[refused")
    assert submitted == []


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("2^10", "1024"),  # ordinary powers still evaluate
        ("2^100", str(2**100)),
        ("(2+3)^3", "125"),
    ],
)
def test_guard_does_not_refuse_legitimate_powers(question: str, expected: str) -> None:
    assert _reject_reason(question) is None
    assert _try_fast_path(question) == expected


def test_symbolic_exponent_is_not_refused() -> None:
    # x**n stays symbolic in sympy — no integer blow-up, so it must be allowed
    # through to the ReAct tools (differentiate, integrate, ...).
    assert _reject_reason("x**n") is None
    assert _reject_reason("x^2 + 3*x") is None


def test_fast_path_refuses_power_tower() -> None:
    assert _try_fast_path("What is 9^9^9^9?") is None


def test_saturated_executor_stops_accepting_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once every worker is wedged, further calls refuse instead of piling up."""
    monkeypatch.setattr(math_module._stuck, "count", math_module._EXECUTOR_WORKERS)
    assert _bounded(_eval_exact, "2 + 2") == (
        "[refused: evaluator saturated by a previous runaway expression]"
    )
    # ...and recovers once the wedged workers are accounted for.
    monkeypatch.setattr(math_module._stuck, "count", 0)
    assert _bounded(_eval_exact, "2 + 2") == "4"


def test_question_length_is_capped_at_the_api_boundary() -> None:
    """An unauthenticated endpoint must not accept an unbounded question body."""
    for model in (QueryRequest, CollaborateRequest):
        with pytest.raises(ValidationError):
            model(question="x" * (MAX_QUESTION_LEN + 1))
        assert model(question="x" * MAX_QUESTION_LEN).question

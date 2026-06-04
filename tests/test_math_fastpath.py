"""Unit tests for the MathProgram deterministic fast-path (no LM required).

Locks in two recent fixes: the fast-path returns *exact* values (not lossy
floats), and the sympy guard refuses oversized / pathological input instead of
hanging.
"""

from __future__ import annotations

import pytest

from app.dspy_modules.math_module import _MAX_EXPR_LEN, _bounded, _eval_exact, _try_fast_path


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

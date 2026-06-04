"""MathProgram — ``dspy.ReAct`` over ``SolveMathProblem`` with sympy tools.

Why ReAct and not CodeAct: the DSPy docs explicitly forbid CodeAct tools from
calling external libraries (sympy, numpy, etc.) — the tool bodies have to be
self-contained. ReAct has no such restriction; the model chooses which tool to
call and we own the implementation, so sympy is fair game.

A deterministic sympy fast-path runs before the ReAct loop kicks in: for
questions that are pure arithmetic expressions (digits and ``+ - * / ( ) ^``
only, e.g. ``"2 + 2"``, ``"144 / 12"``, ``"1/2 + 1/3"``), we short-circuit to a
one-shot exact evaluation. This keeps simple queries cheap and exact.
"""

from __future__ import annotations

import concurrent.futures
import functools
import re
from collections.abc import Callable
from typing import Any

import dspy
import sympy

from .signatures import SolveMathProblem

# ---------------------------------------------------------------------------
# Guard: sympy.sympify on untrusted (LLM-relayed user) input can hang or blow up
# memory on adversarial expressions — e.g. a nested power tower like 9**9**9.
# We cap input length and bound wall-clock time. NB: CPython can't kill a running
# thread, so a timed-out evaluation keeps burning a background CPU until it
# finishes on its own; this bounds *latency*, not total work. The length cap is
# the cheaper first line of defense.
# ---------------------------------------------------------------------------

_MAX_EXPR_LEN = 256
_EVAL_TIMEOUT_S = 5.0
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="sympy")


def _bounded(fn: Callable[..., str], *args: Any, **kwargs: Any) -> str:
    """Run a sympy tool with an input-length cap and a wall-clock timeout."""
    for a in args:
        if isinstance(a, str) and len(a) > _MAX_EXPR_LEN:
            return f"[refused: expression exceeds {_MAX_EXPR_LEN} characters]"
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=_EVAL_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return f"[refused: evaluation exceeded {_EVAL_TIMEOUT_S:.0f}s]"
    except Exception as exc:  # surface sympy errors as a string, don't raise into ReAct
        return f"[error: {type(exc).__name__}: {exc}]"


def _guard(fn: Callable[..., str]) -> Callable[..., str]:
    """Wrap a tool so every call goes through :func:`_bounded`.

    ``functools.wraps`` (with ``__wrapped__``) preserves the name, docstring, and
    signature DSPy introspects to build the tool-call spec.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        return _bounded(fn, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Tools exposed to ReAct. Each is a plain function with a typed signature and
# a clear docstring — DSPy turns those into tool-call specs for the LM.
# ---------------------------------------------------------------------------


def evaluate_expression(expression: str) -> str:
    """Evaluate a numeric or symbolic mathematical expression. Returns the result as a string."""
    return str(sympy.sympify(expression.replace("^", "**")).evalf())


def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve an equation for the given variable. ``equation`` may use ``=`` or be already moved to one side."""
    expr = equation
    if "=" in equation:
        lhs, rhs = equation.split("=", 1)
        expr = f"({lhs}) - ({rhs})"
    var = sympy.Symbol(variable)
    solutions = sympy.solve(sympy.sympify(expr.replace("^", "**")), var)
    return str(solutions)


def differentiate(expression: str, variable: str = "x", order: int = 1) -> str:
    """Compute the derivative of an expression with respect to ``variable`` to the given order."""
    var = sympy.Symbol(variable)
    return str(sympy.diff(sympy.sympify(expression.replace("^", "**")), var, order))


def integrate(expression: str, variable: str = "x") -> str:
    """Compute the indefinite integral of an expression with respect to ``variable``."""
    var = sympy.Symbol(variable)
    return str(sympy.integrate(sympy.sympify(expression.replace("^", "**")), var))


def simplify(expression: str) -> str:
    """Simplify a symbolic expression."""
    return str(sympy.simplify(sympy.sympify(expression.replace("^", "**"))))


def factor(expression: str) -> str:
    """Factor a polynomial expression."""
    return str(sympy.factor(sympy.sympify(expression.replace("^", "**"))))


def series_expansion(expression: str, variable: str = "x", point: str = "0", order: int = 5) -> str:
    """Compute the Taylor series of an expression around ``point`` to the given ``order``."""
    var = sympy.Symbol(variable)
    expr = sympy.sympify(expression.replace("^", "**"))
    return str(sympy.series(expr, var, sympy.sympify(point), order + 1).removeO())


MATH_TOOLS = [
    _guard(fn)
    for fn in (
        evaluate_expression,
        solve_equation,
        differentiate,
        integrate,
        simplify,
        factor,
        series_expansion,
    )
]


# ---------------------------------------------------------------------------
# Fast-path: try pure sympy evaluation before going to the LM.
# ---------------------------------------------------------------------------

_ARITHMETIC_ONLY = re.compile(r"^[\d\s+\-*/().^]+$")


def _eval_exact(candidate: str) -> str:
    """Evaluate a numeric expression and render it *exactly* (no lossy float).

    ``sympify`` already evaluates arithmetic, so ``"1/2 + 1/3"`` -> ``5/6`` and
    ``"2 + 2"`` -> ``4`` without calling ``.evalf()`` (which would have returned
    ``0.8333…`` / ``4.0`` and broken the "exact / simplest form" contract). For
    decimal input, ``nsimplify`` recovers the exact rational where possible.
    """
    result = sympy.sympify(candidate)
    if not result.is_number:
        return ""  # has free symbols — not a fast-path case
    if result.is_Float:
        result = sympy.nsimplify(result, rational=True)
    return str(result)


def _try_fast_path(question: str) -> str | None:
    """Return an answer if the question is a clean expression solvable by sympy alone."""
    stripped = question.strip().rstrip(".?!").replace("^", "**")
    if not stripped:
        return None
    # Only attempt the fast path on questions that look like pure expressions.
    candidate = stripped
    for prefix in ("what is ", "calculate ", "compute ", "evaluate "):
        if candidate.lower().startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
            break
    if not _ARITHMETIC_ONLY.match(candidate) or len(candidate) > _MAX_EXPR_LEN:
        return None
    # _bounded guards against pathological arithmetic (e.g. 2**9**9**9) that
    # matches the regex; on guard-trip ("[refused…]") or non-number (""), fall
    # through to the ReAct loop instead of returning a bogus answer.
    out = _bounded(_eval_exact, candidate)
    if not out or out.startswith("["):
        return None
    return out


class MathProgram(dspy.Module):
    """ReAct-based math solver with a deterministic sympy fast-path."""

    def __init__(self, max_iters: int = 6) -> None:
        super().__init__()
        self.react = dspy.ReAct(SolveMathProblem, tools=MATH_TOOLS, max_iters=max_iters)

    def forward(self, question: str, **_: Any) -> dspy.Prediction:
        fast = _try_fast_path(question)
        if fast is not None:
            return dspy.Prediction(
                reasoning=f"Direct evaluation of '{question.strip()}'.",
                answer=fast,
            )
        return self.react(question=question)

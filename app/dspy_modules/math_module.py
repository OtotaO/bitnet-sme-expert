"""MathProgram — ``dspy.ReAct`` over ``SolveMathProblem`` with sympy tools.

Why ReAct and not CodeAct: the DSPy docs explicitly forbid CodeAct tools from
calling external libraries (sympy, numpy, etc.) — the tool bodies have to be
self-contained. ReAct has no such restriction; the model chooses which tool to
call and we own the implementation, so sympy is fair game.

A deterministic sympy fast-path runs before the ReAct loop kicks in: for
expressions that parse cleanly as sympy and contain no free variables (e.g.
``"2 + 2"``, ``"sqrt(144)"``), we short-circuit to a one-shot evaluation. This
keeps simple queries cheap and exact.
"""

from __future__ import annotations

import re
from typing import Any

import dspy
import sympy

from .signatures import SolveMathProblem

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
    evaluate_expression,
    solve_equation,
    differentiate,
    integrate,
    simplify,
    factor,
    series_expansion,
]


# ---------------------------------------------------------------------------
# Fast-path: try pure sympy evaluation before going to the LM.
# ---------------------------------------------------------------------------

_ARITHMETIC_ONLY = re.compile(r"^[\d\s+\-*/().^]+$")


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
    if not _ARITHMETIC_ONLY.match(candidate):
        return None
    try:
        result = sympy.sympify(candidate).evalf()
        return str(result)
    except (sympy.SympifyError, TypeError, ValueError):
        return None


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

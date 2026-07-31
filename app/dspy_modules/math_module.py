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

import ast
import concurrent.futures
import functools
import math
import operator
import re
import threading
from collections.abc import Callable
from typing import Any

import dspy
import sympy

from .signatures import SolveMathProblem

# ---------------------------------------------------------------------------
# Guard: sympy.sympify on untrusted (LLM-relayed user) input can hang or blow up
# memory on adversarial expressions.
#
# The length + timeout caps below bound *latency* only. They do not bound
# *work*: CPython cannot kill a running thread, so a timed-out evaluation keeps
# burning a CPU and allocating until it finishes on its own. `9^9^9^9` is seven
# characters — it sails past the length cap, is rewritten to `9**9**9**9`, and
# then allocates until the process dies. Verified locally against sympy 1.14.0:
# still running and growing after two minutes.
#
# So the real defense has to refuse the expression *before* any evaluation
# starts. `_reject_reason` parses the expression with `ast` and rejects
# unbounded exponentiation; `_bounded` then applies the length cap, a
# concurrency circuit-breaker, and the timeout as backstops.
#
# Every guard here only ever *refuses*. None of them makes the evaluator
# attempt more work than before.
# ---------------------------------------------------------------------------

_MAX_EXPR_LEN = 256
_EVAL_TIMEOUT_S = 5.0
_EXECUTOR_WORKERS = 4
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_EXECUTOR_WORKERS, thread_name_prefix="sympy"
)

# A literal exponent above this is refused outright. 9**5000 is ~4,800 digits —
# computed instantly, harmless — so this is far above any legitimate query.
_MAX_LITERAL_EXPONENT = 5_000
# ...and, when both operands are literals, the result must fit in this many
# bits (~64 KiB of integer). Catches 2**500000 as well as 999999**99999.
_MAX_POW_RESULT_BITS = 1 << 19


class _StuckCounter:
    """Circuit breaker for evaluations that timed out but cannot be killed.

    A thread wedged inside sympy can never be reclaimed, so once every worker is
    stuck we stop submitting instead of letting stuck threads (and their
    allocations) pile up without limit. Bounds the leak at ``_EXECUTOR_WORKERS``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count = 0

    def saturated(self, limit: int) -> bool:
        with self._lock:
            return self.count >= limit

    def acquire(self) -> None:
        with self._lock:
            self.count += 1

    def release(self) -> None:
        with self._lock:
            self.count -= 1


_stuck = _StuckCounter()


# An exponent has to be evaluated in order to bound the power, so the exponent
# itself must be cheap. Anything whose own magnitude needs more bits than this
# is refused without being evaluated. 64 bits is far above any real query and
# keeps the evaluation of ordinary big literals (e.g. `2^999999999`) instant, so
# they still get the precise "exponent exceeds ..." message.
_MAX_EXPONENT_BITS = 64


class _Refused(Exception):
    """Raised by the magnitude analysis when an expression must not be evaluated."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_UNARY_OPS: dict[type[ast.AST], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_BINARY_OPS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
# Operators whose result size is bounded by the sum of the operand sizes.
_SUM_SIZED_OPS = (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)


def _const_value(node: ast.AST) -> int | float | None:
    """Exactly evaluate a constant-only arithmetic subtree, or return ``None``.

    Only ever called on subtrees whose magnitude has already been bounded by
    :func:`_magnitude_bits`, so the arithmetic here is cheap by construction.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        return node.value

    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        operand = _const_value(node.operand)
        return None if unary is None or operand is None else unary(operand)

    if isinstance(node, ast.BinOp):
        binary = _BINARY_OPS.get(type(node.op))
        left = _const_value(node.left)
        right = _const_value(node.right)
        if binary is None or left is None or right is None:
            return None
        # Refuse rather than materialise an exponent we have not sized.
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_LITERAL_EXPONENT:
            return None
        try:
            return binary(left, right)
        except (ZeroDivisionError, OverflowError, ValueError, TypeError):
            return None

    return None


def _magnitude_bits(node: ast.AST) -> float | None:
    """Upper bound on ``log2(|value|)`` for a constant-only subtree.

    Returns ``None`` when the subtree contains a free symbol — sympy keeps those
    symbolic, so they cannot produce an integer blow-up. Raises :class:`_Refused`
    when the subtree provably yields a value too large to compute.

    This has to be recursive: bounding only a literal ``base ** literal``
    lets `(10^5000)^5000` and `2^(99999*99999)` through, and both of those burn
    GBs of RSS inside a thread that CPython cannot kill.
    """

    if isinstance(node, ast.Expression):
        return _magnitude_bits(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        magnitude = abs(node.value)
        return 0.0 if magnitude <= 1 else math.log2(magnitude)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _magnitude_bits(node.operand)

    if isinstance(node, ast.BinOp):
        return _binop_magnitude_bits(node)

    # Anything else (names, calls, comparisons) is symbolic as far as we know,
    # but still has to be walked so a refusal nested inside it fires.
    for child in ast.iter_child_nodes(node):
        _magnitude_bits(child)
    return None


def _cap(bits: float) -> float:
    if bits > _MAX_POW_RESULT_BITS:
        raise _Refused("result too large to compute")
    return bits


def _pow_magnitude_bits(node: ast.BinOp) -> float | None:
    """Bound ``base ** exponent``, refusing anything unbounded."""
    # Recurse into both sides first: a refusal deeper in the tree (e.g. the
    # inner (10^5000)^5000 of a three-level nest) still fires.
    base_bits = _magnitude_bits(node.left)
    exponent_bits = _magnitude_bits(node.right)

    # A power tower (a**b**c). Kept as an explicit case so the refusal reason
    # stays specific; the size bound below would also catch it.
    stripped = node.right
    while isinstance(stripped, ast.UnaryOp) and type(stripped.op) in _UNARY_OPS:
        stripped = stripped.operand
    if isinstance(stripped, ast.BinOp) and isinstance(stripped.op, ast.Pow):
        raise _Refused("chained exponentiation")

    if exponent_bits is None:
        return None  # symbolic exponent (x**n) — sympy keeps it symbolic
    if exponent_bits > _MAX_EXPONENT_BITS:
        raise _Refused(f"exponent exceeds {_MAX_LITERAL_EXPONENT}")

    exponent = _const_value(node.right)
    if exponent is None:
        return None
    if abs(exponent) > _MAX_LITERAL_EXPONENT:
        raise _Refused(f"exponent exceeds {_MAX_LITERAL_EXPONENT}")
    if base_bits is None:
        return None  # symbolic base with a small literal exponent
    return _cap(base_bits * abs(exponent))


def _binop_magnitude_bits(node: ast.BinOp) -> float | None:
    if isinstance(node.op, ast.Pow):
        return _pow_magnitude_bits(node)

    left = _magnitude_bits(node.left)
    right = _magnitude_bits(node.right)
    if left is None or right is None:
        return None
    if isinstance(node.op, (ast.Add, ast.Sub)):
        return _cap(max(left, right) + 1.0)
    if isinstance(node.op, _SUM_SIZED_OPS):
        # |a*b|, and the numerator+denominator of a/b, are both bounded by the
        # sum of the operand sizes.
        return _cap(left + right)
    return None


def _pow_is_unbounded(node: ast.AST) -> str | None:
    """Return a refusal reason if ``node`` contains unbounded exponentiation."""
    try:
        _magnitude_bits(node)
    except _Refused as refused:
        return refused.reason
    except RecursionError:
        return "expression nested too deeply"
    return None


def _reject_reason(expression: str) -> str | None:
    """Return a refusal reason for an expression we must not hand to sympy.

    Returns ``None`` when the expression is safe to evaluate *or* when it cannot
    be parsed as Python — in the latter case sympy's own parser raises a normal
    error, which is already handled.
    """
    if len(expression) > _MAX_EXPR_LEN:
        return f"expression exceeds {_MAX_EXPR_LEN} characters"
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    return _pow_is_unbounded(tree)


def _bounded(fn: Callable[..., str], *args: Any, **kwargs: Any) -> str:
    """Run a sympy tool behind the refusal guards described above."""
    for a in args:
        if isinstance(a, str):
            if len(a) > _MAX_EXPR_LEN:
                return f"[refused: expression exceeds {_MAX_EXPR_LEN} characters]"
            reason = _reject_reason(a)
            if reason is not None:
                return f"[refused: {reason}]"

    if _stuck.saturated(_EXECUTOR_WORKERS):
        return "[refused: evaluator saturated by a previous runaway expression]"

    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=_EVAL_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        # The thread keeps running; we can only count it and stop feeding more.
        _stuck.acquire()
        future.add_done_callback(lambda _f: _stuck.release())
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

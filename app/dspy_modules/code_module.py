"""CodeProgram — DSPy code expert with optional sandboxed execution.

By default the module is a plain ``dspy.ChainOfThought`` over the
:class:`GenerateCode` signature. When ``CODE_SANDBOX_ENABLED=1`` is set *and*
the request language is Python, it switches to ``dspy.ReAct`` with a
Deno+Pyodide ``PythonInterpreter`` tool, letting the model iterate
(write → run → fix) before returning. This mirrors the tool-augmented pattern
``MathProgram`` already uses with sympy.

Why opt-in:

* Pyodide doesn't run arbitrary pip packages — only sandbox-friendly Python is
  useful here.
* The interpreter requires Deno on PATH at runtime.
* Each ReAct iteration is an extra LLM call; for one-shot codegen the latency
  isn't worth it.

Env knobs:

* ``CODE_SANDBOX_ENABLED`` — ``1`` to enable, anything else (default) to keep
  the ChainOfThought path.
* ``CODE_SANDBOX_MAX_ITERS`` — ReAct loop budget; defaults to ``3``.
"""

from __future__ import annotations

import logging
import os

import dspy

from .signatures import GenerateCode

logger = logging.getLogger(__name__)


_LANG_HINTS = {
    "python": ("python", "py ", "pandas", "numpy", "fastapi", "django", "flask"),
    "javascript": ("javascript", "node", "react", "vue", "npm "),
    "typescript": ("typescript", " ts ", "tsx", "interface "),
    "rust": ("rust", "cargo", " rs "),
    "go": (" golang", " go "),
    "sql": ("sql", "select ", "postgres", "mysql", "sqlite"),
}


def _infer_language(request: str, default: str = "python") -> str:
    """Best-effort language detection from the request text."""
    lowered = f" {request.lower()} "
    for lang, hints in _LANG_HINTS.items():
        if any(hint in lowered for hint in hints):
            return lang
    return default


def _python_exec(code: str) -> str:
    """Execute Python in a Deno+Pyodide WASM sandbox; return stdout or error.

    Use this to verify candidate code before returning it. The sandbox has no
    host filesystem, network, or environment access. Only Python stdlib plus
    Pyodide's bundled packages (numpy, pandas, etc.) are available — pip
    installs are not supported.

    Args:
        code: A complete, runnable Python snippet. Use ``print(...)`` to
            surface values you want to inspect.

    Returns:
        The combined stdout/stderr from the snippet, or an ``[execution error]``
        prefixed message if the sandbox raised.
    """
    try:
        from dspy import PythonInterpreter
    except ImportError as exc:
        return f"[execution error] PythonInterpreter unavailable: {exc}"
    try:
        with PythonInterpreter() as interp:
            return interp(code)
    except Exception as exc:
        return f"[execution error] {type(exc).__name__}: {exc}"


def _sandbox_enabled() -> bool:
    return os.environ.get("CODE_SANDBOX_ENABLED", "0").lower() in {"1", "true", "yes"}


def _max_iters() -> int:
    raw = os.environ.get("CODE_SANDBOX_MAX_ITERS", "3")
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


class CodeProgram(dspy.Module):
    """Generate / explain / debug / refactor code, optionally with execution."""

    def __init__(self) -> None:
        super().__init__()
        self.cot = dspy.ChainOfThought(GenerateCode)
        if _sandbox_enabled():
            self.react = dspy.ReAct(
                GenerateCode,
                tools=[_python_exec],
                max_iters=_max_iters(),
            )
            logger.info("code_module.sandbox.enabled", extra={"max_iters": _max_iters()})
        else:
            self.react = None

    def forward(self, request: str, language: str | None = None) -> dspy.Prediction:
        lang = language or _infer_language(request)
        if self.react is not None and lang == "python":
            return self.react(request=request, language=lang)
        return self.cot(request=request, language=lang)

"""CodeProgram — ``dspy.ChainOfThought`` over ``GenerateCode``.

Starts simple. Upgrade path: swap to ``dspy.ReAct`` with file-read / code-exec
tools when sandboxing is wired in.
"""

from __future__ import annotations

import dspy

from .signatures import GenerateCode

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


class CodeProgram(dspy.Module):
    """Generate / explain / debug / refactor code."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(GenerateCode)

    def forward(self, request: str, language: str | None = None) -> dspy.Prediction:
        lang = language or _infer_language(request)
        return self.predict(request=request, language=lang)

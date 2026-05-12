"""Math expert — backed by ``MathProgram`` (ReAct + sympy tools)."""

from __future__ import annotations

from typing import Any

import dspy

from ..dspy_modules import MathProgram
from ..schemas.base import ExpertDomain
from .base_expert import DSPyExpert


class MathExpert(DSPyExpert):
    """Solve mathematics problems with a deterministic sympy fast-path + ReAct fallback."""

    DOMAIN = ExpertDomain.MATH
    LM_ROLE = "math"

    def _build_program(self) -> dspy.Module:
        return MathProgram()

    def _format_prediction(self, prediction: dspy.Prediction) -> dict[str, Any]:
        reasoning = getattr(prediction, "reasoning", "")
        answer = getattr(prediction, "answer", "")
        response_text = f"{answer}\n\nReasoning:\n{reasoning}".strip() if reasoning else answer
        return {
            "response": response_text,
            "metadata": {"answer": answer, "reasoning": reasoning},
        }

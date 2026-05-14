"""General expert — backed by ``GeneralProgram`` (ChainOfThought)."""

from __future__ import annotations

from typing import Any

import dspy

from ..dspy_modules import GeneralProgram
from ..schemas.base import ExpertDomain
from .base_expert import DSPyExpert


class GeneralExpert(DSPyExpert):
    """Answer general-knowledge questions with a self-rated confidence."""

    DOMAIN = ExpertDomain.GENERAL
    LM_ROLE = "general"

    def _build_program(self) -> dspy.Module:
        return GeneralProgram()

    def _format_prediction(self, prediction: dspy.Prediction) -> dict[str, Any]:
        answer = getattr(prediction, "answer", "")
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        return {
            "response": answer,
            "metadata": {"confidence": confidence},
        }

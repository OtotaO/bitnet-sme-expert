"""Code expert — backed by ``CodeProgram`` (ChainOfThought)."""

from __future__ import annotations

from typing import Any

import dspy

from ..dspy_modules import CodeProgram
from ..schemas.base import ExpertDomain
from .base_expert import DSPyExpert


class CodeExpert(DSPyExpert):
    """Generate, explain, debug, and refactor code."""

    DOMAIN = ExpertDomain.CODE
    LM_ROLE = "code"

    def _build_program(self) -> dspy.Module:
        return CodeProgram()

    async def _invoke(
        self, input_text: str, context: dict[str, Any], **_: Any
    ) -> dspy.Prediction:
        assert self.program is not None
        language = (context or {}).get("language")
        return await self.program(request=input_text, language=language)

    def _format_prediction(self, prediction: dspy.Prediction) -> dict[str, Any]:
        explanation = getattr(prediction, "explanation", "")
        code = getattr(prediction, "code", "")
        language = getattr(prediction, "language", "python")
        parts: list[str] = []
        if explanation:
            parts.append(explanation)
        if code:
            parts.append(f"```{language}\n{code}\n```")
        return {
            "response": "\n\n".join(parts) or "(empty response)",
            "metadata": {"language": language, "code": code, "explanation": explanation},
        }

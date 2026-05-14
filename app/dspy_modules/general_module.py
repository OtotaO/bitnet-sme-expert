"""GeneralProgram — ``dspy.ChainOfThought`` over ``AnswerGeneralQuestion``.

JSONAdapter is configured at the call site (or globally via ``dspy.configure``)
to enforce a structured ``answer + confidence`` envelope, which keeps downstream
parsing trivial.
"""

from __future__ import annotations

import dspy

from .signatures import AnswerGeneralQuestion


class GeneralProgram(dspy.Module):
    """Answer general-knowledge questions with a self-rated confidence."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(AnswerGeneralQuestion)

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)

"""RouterProgram — replaces the hardcoded keyword routing in core.py.

A ``dspy.ChainOfThought`` over ``RouteQuestion``. Trivially small and trivially
optimizable: with a 30-example gold set, MIPROv2 routinely lifts F1 from the
keyword baseline by 10-15 points.
"""

from __future__ import annotations

import dspy

from .signatures import RouteQuestion


class RouterProgram(dspy.Module):
    """Route an incoming question to a domain expert."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(RouteQuestion)

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)

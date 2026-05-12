"""DSPy programs for the SME expert system.

Programs are split by domain. Each is a ``dspy.Module`` subclass that wraps a
predictor (``ChainOfThought`` / ``ReAct``) over a typed ``Signature``. Compiled
artifacts produced by ``scripts/optimize.py`` live under ``compiled/`` and are
loaded at app startup when present.
"""

from .code_module import CodeProgram
from .general_module import GeneralProgram
from .math_module import MathProgram
from .router import RouterProgram
from .signatures import (
    AnswerGeneralQuestion,
    GenerateCode,
    RouteQuestion,
    SolveMathProblem,
)

__all__ = [
    "AnswerGeneralQuestion",
    "CodeProgram",
    "GenerateCode",
    "GeneralProgram",
    "MathProgram",
    "RouteQuestion",
    "RouterProgram",
    "SolveMathProblem",
]

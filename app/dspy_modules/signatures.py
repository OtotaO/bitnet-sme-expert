"""DSPy signatures — the typed contracts for each task.

Signatures are intentionally minimal: a short docstring (which DSPy uses as the
high-level instruction), one or more ``InputField`` declarations, and one or
more ``OutputField`` declarations. Optimizers tune the instruction and few-shot
demonstrations attached to these signatures.
"""

from __future__ import annotations

from typing import Literal

import dspy

Domain = Literal["math", "code", "general"]


class RouteQuestion(dspy.Signature):
    """Classify the user's question into the domain best suited to answer it.

    Choose ``math`` for calculation, algebra, calculus, statistics, or symbolic
    manipulation. Choose ``code`` for programming questions, code review, or
    algorithm design. Choose ``general`` for everything else.
    """

    question: str = dspy.InputField(desc="The user's question")
    domain: Domain = dspy.OutputField(desc="One of: math, code, general")
    confidence: float = dspy.OutputField(desc="Routing confidence between 0.0 and 1.0")


class SolveMathProblem(dspy.Signature):
    """Solve a mathematics problem with explicit step-by-step reasoning.

    Prefer exact symbolic results when possible. Use the provided tools for
    arithmetic, algebra, calculus, and series operations rather than estimating
    in your head.
    """

    question: str = dspy.InputField(desc="The math problem to solve")
    reasoning: str = dspy.OutputField(desc="Step-by-step derivation")
    answer: str = dspy.OutputField(desc="The final answer in simplest form")


class GenerateCode(dspy.Signature):
    """Generate, explain, debug, or refactor code based on the user's request.

    Default to Python when no language is specified. Always include runnable,
    fully-formed snippets — no placeholders, no pseudocode unless explicitly
    requested.
    """

    request: str = dspy.InputField(desc="The user's code-related request")
    language: str = dspy.InputField(desc="Target language (e.g. 'python', 'rust')")
    explanation: str = dspy.OutputField(desc="Short prose explaining the approach")
    code: str = dspy.OutputField(desc="Complete, runnable code")


class AnswerGeneralQuestion(dspy.Signature):
    """Answer a general-knowledge question with a clear, accurate response.

    If you are uncertain about a factual detail, say so explicitly. Do not
    fabricate sources, dates, or quotations.
    """

    question: str = dspy.InputField(desc="The user's question")
    answer: str = dspy.OutputField(desc="The answer")
    confidence: float = dspy.OutputField(desc="Self-rated confidence between 0.0 and 1.0")

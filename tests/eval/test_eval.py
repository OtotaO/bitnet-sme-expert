"""End-to-end DSPy evaluation against gold sets.

Run with ``RUN_EVAL=1 pytest tests/eval``. Requires provider API keys. The
metrics are deliberately simple — exact match for math, substring containment
for code/general. The point of this file is to give optimizers (MIPROv2, GEPA)
a callable that grades a prediction, not to be a strict benchmark.
"""

from __future__ import annotations

import dspy
import pytest

from app.dspy_modules import CodeProgram, GeneralProgram, MathProgram
from app.llm import configure_dspy


@pytest.fixture(scope="module", autouse=True)
def _dspy_configured() -> None:
    configure_dspy(enable_mlflow=False)


# ---------------------------------------------------------------------------
# Score scaling
# ---------------------------------------------------------------------------


def to_fraction(raw_score: float) -> float:
    """Normalize a ``dspy.Evaluate`` result to a 0-1 fraction.

    ``dspy.Evaluate`` reports the average metric as a 0-100 *percentage* (e.g.
    ``73.33`` for 11/15). Thresholds in this repo are expressed as fractions
    (math/code = 0.6, general = 0.7), so the raw score must be divided by 100
    before any ``score >= threshold`` comparison — otherwise the gate compares
    a percentage against a fraction and passes for essentially any score.
    """
    return float(raw_score) / 100.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def math_metric(example: dspy.Example, prediction: dspy.Prediction, *_, **__) -> float:
    """1.0 if the predicted answer sympy-equals the gold answer."""
    import sympy

    try:
        pred = sympy.sympify(str(getattr(prediction, "answer", "")).replace("^", "**"))
        gold = sympy.sympify(str(example.answer).replace("^", "**"))
        return 1.0 if sympy.simplify(pred - gold) == 0 else 0.0
    except (sympy.SympifyError, TypeError, ValueError):
        return float(str(getattr(prediction, "answer", "")).strip() == str(example.answer).strip())


def code_metric(example: dspy.Example, prediction: dspy.Prediction, *_, **__) -> float:
    """1.0 if the generated code contains every required substring."""
    code = str(getattr(prediction, "code", ""))
    needles = example.must_contain  # type: ignore[attr-defined]
    return float(all(needle in code for needle in needles))


def general_metric(example: dspy.Example, prediction: dspy.Prediction, *_, **__) -> float:
    """1.0 if any required substring appears in the answer."""
    answer = str(getattr(prediction, "answer", "")).lower()
    needles = example.must_contain  # type: ignore[attr-defined]
    return float(any(needle.lower() in answer for needle in needles))


# ---------------------------------------------------------------------------
# Tests — guarded by RUN_EVAL=1 in conftest.
# ---------------------------------------------------------------------------


def test_math_baseline(math_examples) -> None:
    program = MathProgram()
    evaluator = dspy.Evaluate(devset=math_examples, num_threads=4, display_progress=True)
    score = to_fraction(evaluator(program, metric=math_metric))
    print(f"\nMathProgram baseline (holdout): {score:.2f}")
    assert score >= 0.6, f"baseline math score too low: {score}"


def test_code_baseline(code_examples) -> None:
    program = CodeProgram()
    evaluator = dspy.Evaluate(devset=code_examples, num_threads=4, display_progress=True)
    score = to_fraction(evaluator(program, metric=code_metric))
    print(f"\nCodeProgram baseline (holdout): {score:.2f}")
    assert score >= 0.6, f"baseline code score too low: {score}"


def test_general_baseline(general_examples) -> None:
    program = GeneralProgram()
    evaluator = dspy.Evaluate(devset=general_examples, num_threads=4, display_progress=True)
    score = to_fraction(evaluator(program, metric=general_metric))
    print(f"\nGeneralProgram baseline (holdout): {score:.2f}")
    assert score >= 0.7, f"baseline general score too low: {score}"

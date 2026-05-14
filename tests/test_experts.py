"""Smoke tests for DSPy-backed experts.

Real LM calls are out of scope here — those live under ``tests/eval/``. These
tests verify wiring: experts initialize, dispatch through DSPy with a stub LM,
and return the expected response envelope.
"""

from __future__ import annotations

import pytest

from app.dspy_modules.math_module import _try_fast_path
from app.experts.code_expert import CodeExpert
from app.experts.general_expert import GeneralExpert
from app.experts.math_expert import MathExpert
from app.schemas.base import ExpertDomain


@pytest.mark.parametrize(
    ("expression", "expected_prefix"),
    [
        ("2 + 2", "4"),
        ("What is 5 * 8?", "40"),
        ("Calculate (3 + 4) * 2", "14"),
    ],
)
def test_math_fast_path_handles_arithmetic(expression: str, expected_prefix: str) -> None:
    result = _try_fast_path(expression)
    assert result is not None
    assert result.startswith(expected_prefix)


def test_math_fast_path_returns_none_for_symbolic() -> None:
    assert _try_fast_path("Solve for x: 2x + 5 = 15") is None
    assert _try_fast_path("What is the derivative of x^2?") is None


@pytest.mark.asyncio
async def test_math_expert_initialization() -> None:
    expert = MathExpert({"name": "Math Expert", "domain": ExpertDomain.MATH})
    assert not expert.initialized
    await expert.initialize()
    assert expert.initialized
    assert expert.domain == ExpertDomain.MATH


@pytest.mark.asyncio
async def test_math_expert_uses_fast_path_without_lm() -> None:
    """The arithmetic fast-path should answer without ever touching the LM."""
    expert = MathExpert({"name": "Math Expert", "domain": ExpertDomain.MATH})
    await expert.initialize()
    result = await expert.generate("What is 5 * 8?", context={})
    assert "40" in result["response"]
    assert result["metadata"]["domain"] == "math"


@pytest.mark.asyncio
async def test_code_expert_initialization() -> None:
    expert = CodeExpert({"name": "Code Expert", "domain": ExpertDomain.CODE})
    await expert.initialize()
    assert expert.initialized
    assert expert.domain == ExpertDomain.CODE


@pytest.mark.asyncio
async def test_general_expert_initialization() -> None:
    expert = GeneralExpert({"name": "General Expert", "domain": ExpertDomain.GENERAL})
    await expert.initialize()
    assert expert.initialized
    assert expert.domain == ExpertDomain.GENERAL


@pytest.mark.asyncio
async def test_code_expert_infers_language() -> None:
    """The CodeProgram should infer 'python' as a sensible default."""
    from app.dspy_modules.code_module import _infer_language

    assert _infer_language("write a python function") == "python"
    assert _infer_language("how do I write a select statement in postgres") == "sql"
    assert _infer_language("explain this rust trait") == "rust"
    assert _infer_language("explain garbage collection") == "python"  # default


def test_code_program_default_uses_chain_of_thought(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no CODE_SANDBOX_ENABLED env var, the program uses ChainOfThought."""
    monkeypatch.delenv("CODE_SANDBOX_ENABLED", raising=False)
    from app.dspy_modules.code_module import CodeProgram

    program = CodeProgram()
    assert program.cot is not None
    assert program.react is None


def test_code_program_sandbox_mode_activates_react(monkeypatch: pytest.MonkeyPatch) -> None:
    """CODE_SANDBOX_ENABLED=1 wires up the dspy.ReAct predictor."""
    monkeypatch.setenv("CODE_SANDBOX_ENABLED", "1")
    from app.dspy_modules.code_module import CodeProgram

    program = CodeProgram()
    assert program.react is not None
    assert program.cot is not None


def test_python_exec_returns_error_string_on_failure() -> None:
    """The sandbox tool must surface errors as text — never raise into ReAct."""
    from app.dspy_modules.code_module import _python_exec

    out = _python_exec("raise ValueError('boom')")
    assert out.startswith("[execution error]") or "ValueError" in out

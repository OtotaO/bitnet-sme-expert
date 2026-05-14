"""Smoke tests for ``scripts/run_eval.py``.

We exercise the argparser and the env-var-side-effect of ``_set_per_role_lm``
without actually invoking ``dspy.Evaluate`` (which needs an LM). The full
end-to-end eval lives in ``tests/eval/`` and is gated by ``RUN_EVAL=1``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/run_eval.py as a module without making `scripts` a package.
SPEC = importlib.util.spec_from_file_location(
    "run_eval",
    Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py",
)
assert SPEC is not None and SPEC.loader is not None
run_eval = importlib.util.module_from_spec(SPEC)
sys.modules["run_eval"] = run_eval
SPEC.loader.exec_module(run_eval)


def test_set_per_role_lm_sets_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSPY_LM_MATH", raising=False)
    run_eval._set_per_role_lm("math", "openai/gpt-4o-mini")
    import os

    assert os.environ["DSPY_LM_MATH"] == "openai/gpt-4o-mini"


def test_set_per_role_lm_noop_on_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSPY_LM_CODE", raising=False)
    run_eval._set_per_role_lm("code", None)
    import os

    assert "DSPY_LM_CODE" not in os.environ


def test_default_thresholds_cover_every_domain() -> None:
    assert set(run_eval.DEFAULT_THRESHOLDS) == {"math", "code", "general"}
    assert all(0.0 <= v <= 1.0 for v in run_eval.DEFAULT_THRESHOLDS.values())


def test_build_unknown_domain_raises() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        run_eval._build("nonsense")

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


def test_wilson_interval_small_sample() -> None:
    # 31/50 = 0.62 -> Wilson 95% ~ [0.48, 0.74]; much wider than the naive 0.62.
    lo, hi = run_eval.wilson_interval(31, 50)
    assert 0.47 < lo < 0.50
    assert 0.73 < hi < 0.75
    assert lo < 0.62 < hi


def test_wilson_interval_saturated_and_empty() -> None:
    lo, hi = run_eval.wilson_interval(50, 50)
    assert lo > 0.92  # even a perfect score isn't certainty at N=50
    assert hi == 1.0
    assert run_eval.wilson_interval(0, 0) == (0.0, 0.0)


def test_pin_temperature_sets_all_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for role in run_eval._ALL_ROLES:
        monkeypatch.delenv(f"DSPY_LM_{role.upper()}_TEMPERATURE", raising=False)
    run_eval._pin_temperature(0.0)
    assert all(os.environ[f"DSPY_LM_{r.upper()}_TEMPERATURE"] == "0.0" for r in run_eval._ALL_ROLES)

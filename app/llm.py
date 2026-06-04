"""DSPy LM configuration with per-role LiteLLM routing and optional MLflow tracing.

Single source of truth for which model serves which task. Providers are addressed by
LiteLLM strings (``"openai/gpt-5"``, ``"anthropic/claude-4.5-sonnet"``,
``"gemini/gemini-2.5-pro"``, ``"openai/bitnet"`` against a local bitnet.cpp server).

DSPy 3.2 has been decoupled from litellm at the import level, but its ``dspy.LM``
wrapper still uses litellm internally for chat completions — so model strings here
follow the litellm format.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import dspy

from .config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LMSpec:
    """A single per-role LM specification.

    ``fallbacks`` is reserved for a future litellm fallback-chain wiring; it is
    not consumed by :meth:`build` today.
    """

    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    fallbacks: tuple[str, ...] = ()  # reserved; not yet wired into build()

    def build(self) -> dspy.LM:
        kwargs: dict[str, object] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                kwargs["api_key"] = key
        return dspy.LM(self.model, **kwargs)


_DEFAULT_SPECS: dict[str, LMSpec] = {
    "router": LMSpec(model="openai/gpt-4o-mini", max_tokens=128, temperature=0.0),
    "math": LMSpec(model="openai/gpt-4o-mini", max_tokens=2048, temperature=0.1),
    "code": LMSpec(model="anthropic/claude-haiku-4-5-20251001", max_tokens=4096, temperature=0.2),
    "general": LMSpec(model="openai/gpt-4o-mini", max_tokens=2048, temperature=0.5),
}


def _resolve_spec(role: str) -> LMSpec:
    """Resolve an LM spec from environment overrides or defaults.

    Env overrides allow swapping a per-role model without touching code:
    ``DSPY_LM_MATH=openai/bitnet`` plus ``DSPY_LM_MATH_API_BASE=http://localhost:8080/v1``.
    """
    override = os.environ.get(f"DSPY_LM_{role.upper()}")
    base = _DEFAULT_SPECS[role]
    if not override:
        return base
    return LMSpec(
        model=override,
        api_base=os.environ.get(f"DSPY_LM_{role.upper()}_API_BASE"),
        api_key_env=os.environ.get(f"DSPY_LM_{role.upper()}_API_KEY_ENV"),
        max_tokens=int(os.environ.get(f"DSPY_LM_{role.upper()}_MAX_TOKENS", base.max_tokens)),
        temperature=float(os.environ.get(f"DSPY_LM_{role.upper()}_TEMPERATURE", base.temperature)),
    )


@lru_cache(maxsize=8)
def get_lm(role: str = "general") -> dspy.LM:
    """Return a cached LM for the given role.

    Roles: ``"router"``, ``"math"``, ``"code"``, ``"general"``.
    """
    if role not in _DEFAULT_SPECS:
        raise ValueError(f"Unknown LM role: {role!r}. Valid roles: {sorted(_DEFAULT_SPECS)}")
    spec = _resolve_spec(role)
    logger.info(
        "llm.build",
        extra={"role": role, "model": spec.model, "api_base": spec.api_base},
    )
    return spec.build()


def configure_dspy(enable_mlflow: bool | None = None) -> None:
    """Configure global DSPy defaults and optional MLflow autolog.

    Called once at app startup. MLflow autolog gives free OpenTelemetry-based
    tracing of every module call. Toggled by ``MLFLOW_TRACKING_URI`` being set,
    or the explicit arg. (``scripts/optimize.py`` runs with autolog disabled and
    records optimizer results as committed receipts under ``eval/receipts/``
    instead; sending optimizer runs to MLflow is a future enhancement.)
    """
    dspy.configure(lm=get_lm("general"), async_max_workers=settings.MAX_CONCURRENT_REQUESTS)

    enable = (
        enable_mlflow if enable_mlflow is not None else bool(os.environ.get("MLFLOW_TRACKING_URI"))
    )
    if not enable:
        logger.info("llm.mlflow_disabled")
        return

    try:
        import mlflow

        mlflow.dspy.autolog()
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "dspy-sme-expert")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        logger.info(
            "llm.mlflow_enabled", extra={"tracking_uri": tracking_uri, "experiment": experiment}
        )
    except Exception:
        logger.exception("llm.mlflow_setup_failed")


def reset_caches() -> None:
    """Drop cached LMs. Useful for tests."""
    get_lm.cache_clear()

"""Base expert — thin wrapper around a DSPy module.

The expert holds a ``dspy.Module``, configures it with a domain-specific LM via
``dspy.context(...)`` on every call, optionally loads a compiled / optimized
version from disk, and converts the resulting ``dspy.Prediction`` into the dict
shape the API contract still uses.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy

from ..llm import get_lm
from ..models.expert import BaseExpert as AbstractExpert
from ..schemas.base import ExpertDomain

logger = logging.getLogger(__name__)

COMPILED_DIR = Path(__file__).resolve().parent.parent.parent / "compiled"


class DSPyExpert(AbstractExpert):
    """Base for experts that delegate to a ``dspy.Module``."""

    DOMAIN: ExpertDomain
    LM_ROLE: str = "general"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        config.setdefault("name", self.__class__.__name__)
        config.setdefault("description", f"{self.__class__.__name__} expert")
        config.setdefault("domain", self.DOMAIN.value)
        super().__init__(config)
        self.id: str = str(uuid.uuid4())
        self.program: dspy.Module | None = None

    # -- subclass contract ---------------------------------------------------

    def _build_program(self) -> dspy.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def _format_prediction(self, prediction: dspy.Prediction) -> dict[str, Any]:
        """Convert a Prediction to ``{response, metadata}``. Override for richer shapes."""
        return {"response": str(prediction.toDict() if hasattr(prediction, "toDict") else prediction)}

    # -- lifecycle -----------------------------------------------------------

    async def _initialize(self) -> None:
        program = self._build_program()
        compiled = COMPILED_DIR / f"{self.DOMAIN.value}.json"
        if compiled.exists():
            try:
                program.load(str(compiled))
                logger.info("expert.compiled.loaded", extra={"path": str(compiled)})
            except Exception:
                logger.exception("expert.compiled.load_failed", extra={"path": str(compiled)})
        self.program = dspy.asyncify(program)

    async def _generate_impl(
        self, input_text: str, context: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        if self.program is None:
            await self.initialize()
        assert self.program is not None

        start = time.perf_counter()
        lm = get_lm(self.LM_ROLE)
        with dspy.context(lm=lm):
            prediction = await self._invoke(input_text, context, **kwargs)
        elapsed = time.perf_counter() - start

        response = self._format_prediction(prediction)
        response.setdefault("sources", [])
        meta = response.setdefault("metadata", {})
        meta.update(
            {
                "expert": self.config.name,
                "domain": self.DOMAIN.value,
                "model": getattr(lm, "model", "unknown"),
                "processing_time": elapsed,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        return response

    async def _invoke(self, input_text: str, _context: dict[str, Any], **_: Any) -> dspy.Prediction:
        """Default invocation passes ``question=input_text``. Override if needed."""
        assert self.program is not None
        return await self.program(question=input_text)

    async def _cleanup(self) -> None:
        self.program = None

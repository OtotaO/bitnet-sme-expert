"""Expert base class and configuration model.

Pydantic v2 throughout. The abstract ``BaseExpert`` lives here so that both the
DSPy-backed experts (``app/experts/``) and the service layer can depend on it
without circular imports.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..schemas.base import ExpertDomain

logger = logging.getLogger(__name__)


class ExpertConfig(BaseModel):
    """Configuration for an expert."""

    model_config = ConfigDict(use_enum_values=True, extra="ignore")

    name: str = Field(..., description="Name of the expert")
    description: str = Field(..., description="Description of the expert's capabilities")
    domain: ExpertDomain = Field(..., description="Domain of expertise")
    version: str = Field("3.0.0", description="Expert version")
    model_name: str = Field("dspy-managed", description="Display name for the underlying model")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    stop_sequences: list[str] = Field(default_factory=list)
    is_custom: bool = Field(False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExpertContext(BaseModel):
    """Optional context carried through a request."""

    session_id: str | None = None
    user_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class BaseExpert(ABC):
    """Abstract base. Concrete subclasses live in ``app/experts/``."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = ExpertConfig(**(config or {}))
        self.initialized = False
        self.logger = logger.getChild(f"expert.{self.config.domain}.{self.config.name}")
        self.id: str = str(uuid.uuid4())

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def domain(self) -> ExpertDomain:
        # ``use_enum_values=True`` stores the enum's value, not the enum itself.
        # Re-wrap so callers can rely on ``expert.domain.value``.
        value = self.config.domain
        return value if isinstance(value, ExpertDomain) else ExpertDomain(value)

    async def initialize(self) -> None:
        if self.initialized:
            return
        start = time.perf_counter()
        self.logger.info("expert.init.started", extra={"expert": self.config.name})
        await self._initialize()
        self.initialized = True
        self.logger.info(
            "expert.init.completed",
            extra={"expert": self.config.name, "elapsed_s": time.perf_counter() - start},
        )

    async def _initialize(self) -> None:  # pragma: no cover - default no-op
        return None

    async def generate(
        self,
        input_text: str,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.initialized:
            await self.initialize()
        start = time.perf_counter()
        try:
            response = await self._generate_impl(input_text, context or {}, **kwargs)
        except Exception:
            self.logger.exception("expert.generate.failed", extra={"expert": self.config.name})
            raise
        if not isinstance(response, dict):
            response = {"response": str(response)}
        meta = response.setdefault("metadata", {})
        meta.setdefault("expert_id", self.id)
        meta.setdefault("expert_name", self.config.name)
        meta.setdefault("expert_domain", self.domain.value)
        meta.setdefault("timestamp", datetime.now(UTC).isoformat())
        meta.setdefault("processing_time", time.perf_counter() - start)
        return response

    @abstractmethod
    async def _generate_impl(
        self, input_text: str, context: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]: ...

    async def cleanup(self) -> None:
        if not self.initialized:
            return
        try:
            await self._cleanup()
        finally:
            self.initialized = False

    async def _cleanup(self) -> None:  # pragma: no cover - default no-op
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.config.name,
            "description": self.config.description,
            "domain": self.domain.value,
            "version": self.config.version,
            "model": self.config.model_name,
            "is_custom": self.config.is_custom,
            "initialized": self.initialized,
            "metadata": self.config.metadata,
        }

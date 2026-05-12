"""Expert service — manages a pool of DSPy-backed experts and routes queries."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import dspy
from fastapi import HTTPException, status

from ..dspy_modules import RouterProgram
from ..llm import get_lm
from ..models.expert import BaseExpert
from ..observability import get_request_id, record_domain_outcome
from ..schemas.base import ExpertDomain
from ..schemas.request import CollaborateRequest, QueryRequest

logger = logging.getLogger(__name__)


class ExpertService:
    """Manages experts and routes queries to them."""

    def __init__(self) -> None:
        self._experts: dict[str, BaseExpert] = {}
        self._expert_classes: dict[ExpertDomain, type[BaseExpert]] = {}
        self._expert_configs: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._router: dspy.Module | None = None
        self.logger = logger.getChild("ExpertService")

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.logger.info("ExpertService.initializing")
        start = datetime.now(UTC)

        init_tasks = [e.initialize() for e in self._experts.values() if not e.initialized]
        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)

        # Lazy router so unit tests can bypass it.
        self._router = dspy.asyncify(RouterProgram())

        self._initialized = True
        elapsed = (datetime.now(UTC) - start).total_seconds()
        self.logger.info(
            "ExpertService.initialized",
            extra={"experts": len(self._experts), "elapsed_s": elapsed},
        )

    def register_expert_class(
        self,
        domain: ExpertDomain | str,
        expert_class: type[BaseExpert],
        config: dict[str, Any] | None = None,
    ) -> None:
        if not issubclass(expert_class, BaseExpert):
            raise ValueError(f"{expert_class!r} must subclass BaseExpert")
        if isinstance(domain, str):
            domain = ExpertDomain(domain.lower())
        self._expert_classes[domain] = expert_class
        self._expert_configs[domain.value] = config or {
            "name": f"{domain.value}_expert",
            "description": f"{domain.value.capitalize()} expert",
            "domain": domain,
        }
        self.logger.info("expert.registered", extra={"class": expert_class.__name__, "domain": domain.value})

    async def create_expert(
        self,
        domain: ExpertDomain | str,
        config: dict[str, Any] | None = None,
        expert_id: str | None = None,
    ) -> BaseExpert:
        if isinstance(domain, str):
            domain = ExpertDomain(domain.lower())
        if domain not in self._expert_classes:
            raise ValueError(f"No expert class registered for domain: {domain}")

        merged = dict(self._expert_configs.get(domain.value, {}))
        if config:
            merged.update(config)

        expert = self._expert_classes[domain](merged)
        if expert_id:
            expert.id = expert_id
        if self._initialized:
            await expert.initialize()
        self._experts[expert.id] = expert
        return expert

    async def get_expert(
        self,
        expert_id: str,
        domain: ExpertDomain | str | None = None,
    ) -> BaseExpert:
        expert = self._experts.get(expert_id)
        if not expert:
            raise ValueError(f"Expert not found: {expert_id}")
        if domain is not None:
            if isinstance(domain, str):
                domain = ExpertDomain(domain.lower())
            if expert.domain != domain:
                raise ValueError(
                    f"Expert {expert_id} is {expert.domain.value}, not {domain.value}"
                )
        return expert

    async def get_experts_by_domain(
        self, domain: ExpertDomain | str, only_initialized: bool = True
    ) -> list[BaseExpert]:
        if isinstance(domain, str):
            domain = ExpertDomain(domain.lower())
        return [
            e
            for e in self._experts.values()
            if e.domain == domain and (not only_initialized or e.initialized)
        ]

    async def get_all_experts(self, only_initialized: bool = True) -> dict[str, BaseExpert]:
        if only_initialized:
            return {eid: e for eid, e in self._experts.items() if e.initialized}
        return dict(self._experts)

    async def route(self, question: str) -> tuple[ExpertDomain, float]:
        """Use the DSPy RouterProgram to pick a domain for the question."""
        if self._router is None:  # pragma: no cover - initialize() always sets this
            raise RuntimeError("ExpertService not initialized")
        with dspy.context(lm=get_lm("router")):
            prediction = await self._router(question=question)
        domain_str = str(getattr(prediction, "domain", "general")).strip().lower()
        try:
            domain = ExpertDomain(domain_str)
        except ValueError:
            domain = ExpertDomain.GENERAL
        try:
            confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return domain, confidence

    async def query_expert(
        self,
        expert_id: str,
        request: QueryRequest,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        expert = await self.get_expert(expert_id)
        self.logger.info(
            "expert.query.started",
            extra={
                "request_id": get_request_id(),
                "expert_id": expert_id,
                "domain": expert.domain.value,
            },
        )
        try:
            response = await expert.generate(
                input_text=request.question,
                context={"context": context or {}, **(request.context or {})},
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            record_domain_outcome(expert.domain.value, success=True)
            return response
        except Exception:
            record_domain_outcome(expert.domain.value, success=False)
            self.logger.exception(
                "expert.query.failed",
                extra={"request_id": get_request_id(), "expert_id": expert_id},
            )
            raise

    async def collaborate(
        self,
        request: CollaborateRequest,
        context: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not request.domains:
            experts = await self.get_all_experts()
        else:
            experts = {}
            for d in request.domains:
                for e in await self.get_experts_by_domain(d):
                    experts[e.id] = e

        if not experts:
            raise ValueError("No experts available for the specified domains")

        async def _safe(expert: BaseExpert) -> dict[str, Any]:
            try:
                resp = await expert.generate(
                    input_text=request.question,
                    context={"context": context or {}, **(request.context or {})},
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )
                resp["success"] = True
                record_domain_outcome(expert.domain.value, success=True)
                return resp
            except Exception as exc:  # noqa: BLE001 - collaborate isolates failures
                record_domain_outcome(expert.domain.value, success=False)
                self.logger.exception(
                    "collaborate.expert_failed",
                    extra={"expert_id": expert.id, "domain": expert.domain.value},
                )
                return {
                    "error": str(exc),
                    "success": False,
                    "expert_id": expert.id,
                    "expert_domain": expert.domain.value,
                }

        results = await asyncio.gather(*[_safe(e) for e in experts.values()])
        return {expert.id: result for expert, result in zip(experts.values(), results, strict=True)}

    async def list_experts(self) -> list[dict[str, Any]]:
        all_experts = await self.get_all_experts()
        return [e.to_dict() for e in all_experts.values()]

    async def cleanup(self) -> None:
        self.logger.info("ExpertService.cleanup")
        await asyncio.gather(
            *[e.cleanup() for e in self._experts.values()], return_exceptions=True
        )
        self._experts.clear()
        self._router = None
        self._initialized = False


async def get_expert_service() -> ExpertService:
    """FastAPI dependency for the singleton ExpertService."""
    from ..main import expert_service

    if expert_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Expert service not initialized",
        )
    return expert_service

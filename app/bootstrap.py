"""Shared construction of an initialized ``ExpertService``.

Used by both the FastAPI app (``main.py`` lifespan) and the MCP server
(``app/mcp_server.py``) so expert registration lives in exactly one place.
"""

from __future__ import annotations

import logging

from app.llm import configure_dspy
from app.schemas.base import ExpertDomain
from app.services.expert_service import ExpertService

logger = logging.getLogger(__name__)


async def register_experts(service: ExpertService) -> None:
    """Register the bundled experts and materialize one instance per domain."""
    from app.experts.code_expert import CodeExpert
    from app.experts.general_expert import GeneralExpert
    from app.experts.math_expert import MathExpert

    service.register_expert_class(
        domain=ExpertDomain.MATH,
        expert_class=MathExpert,
        config={
            "name": "Math Expert",
            "description": "ReAct over sympy tools, with a deterministic fast-path for trivial expressions.",
            "domain": ExpertDomain.MATH,
        },
    )
    service.register_expert_class(
        domain=ExpertDomain.CODE,
        expert_class=CodeExpert,
        config={
            "name": "Code Expert",
            "description": "ChainOfThought for code generation, debugging, refactoring, and review.",
            "domain": ExpertDomain.CODE,
        },
    )
    service.register_expert_class(
        domain=ExpertDomain.GENERAL,
        expert_class=GeneralExpert,
        config={
            "name": "General Expert",
            "description": "ChainOfThought for open-ended general knowledge questions.",
            "domain": ExpertDomain.GENERAL,
        },
    )

    for domain in (ExpertDomain.MATH, ExpertDomain.CODE, ExpertDomain.GENERAL):
        await service.create_expert(domain)

    logger.info("experts.registered", extra={"count": len(service._experts)})


async def build_expert_service(*, configure: bool = True) -> ExpertService:
    """Return a fully-initialized ``ExpertService`` (configures DSPy by default)."""
    if configure:
        configure_dspy()
    service = ExpertService()
    await register_experts(service)
    await service.initialize()
    return service

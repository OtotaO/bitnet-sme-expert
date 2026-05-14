"""Core API endpoints: query, collaborate, list experts, feedback."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.base import BaseResponse, ExpertDomain
from app.schemas.request import CollaborateRequest, FeedbackRequest, QueryRequest
from app.schemas.response import (
    CollaborateResponse,
    ExpertInfo,
    ExpertResponse,
    ListExpertsResponse,
    QueryResponse,
)
from app.services.expert_service import ExpertService, get_expert_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _now() -> datetime:
    return datetime.now(UTC)


@router.get(
    "/health",
    response_model=BaseResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> BaseResponse:
    return BaseResponse(success=True, message="API is running")


@router.get(
    "/experts",
    response_model=ListExpertsResponse,
    summary="List available experts",
    tags=["Experts"],
)
async def list_experts(
    expert_service: ExpertService = Depends(get_expert_service),
) -> ListExpertsResponse:
    experts = await expert_service.list_experts()
    return ListExpertsResponse(
        success=True,
        message=f"Found {len(experts)} experts",
        count=len(experts),
        data=[
            ExpertInfo(
                id=e["id"],
                name=e["name"],
                domain=ExpertDomain(e["domain"]),
                description=e["description"],
                model=e["model"],
                version="3.0.0",
                is_custom=e.get("is_custom", False),
                metadata=e.get("metadata", {}),
            )
            for e in experts
        ],
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query an expert",
    tags=["Query"],
)
async def query_expert(
    request: QueryRequest,
    expert_service: ExpertService = Depends(get_expert_service),
) -> QueryResponse:
    query_id = f"qry_{uuid.uuid4().hex[:12]}"
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty",
        )

    start = _now()

    # Resolve the domain: explicit > DSPy router classifier.
    if request.domain is not None:
        domain = request.domain
        confidence = 1.0
    else:
        domain, confidence = await expert_service.route(request.question)

    experts = await expert_service.get_experts_by_domain(domain)
    if not experts:
        # Fall back to whatever's available.
        all_experts = await expert_service.get_all_experts()
        if not all_experts:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No experts available",
            )
        expert = next(iter(all_experts.values()))
    else:
        expert = experts[0]

    expert_response = await expert_service.query_expert(
        expert_id=expert.id, request=request, context=request.context
    )

    processing_time = (_now() - start).total_seconds()
    metadata = expert_response.get("metadata", {})

    return QueryResponse(
        success=True,
        message="Query processed successfully",
        query_id=query_id,
        data=ExpertResponse(
            expert_id=expert.id,
            expert_name=expert.name,
            domain=expert.domain,
            response=expert_response.get("response", ""),
            confidence=float(metadata.get("confidence", confidence) or 0.0),
            model=metadata.get("model", "unknown"),
            tokens_used=int(metadata.get("tokens_used", 0) or 0),
            processing_time=processing_time,
            metadata={**metadata, "routed_domain": domain.value, "routing_confidence": confidence},
            sources=expert_response.get("sources", []),
        ),
    )


@router.post(
    "/collaborate",
    response_model=CollaborateResponse,
    summary="Collaborate with multiple experts",
    tags=["Collaboration"],
)
async def collaborate(
    request: CollaborateRequest,
    expert_service: ExpertService = Depends(get_expert_service),
) -> CollaborateResponse:
    query_id = f"col_{uuid.uuid4().hex[:12]}"
    start = _now()

    raw = await expert_service.collaborate(request, context=request.context)

    successful: dict[str, ExpertResponse] = {}
    for expert_id, payload in raw.items():
        if not payload.get("success"):
            continue
        expert = await expert_service.get_expert(expert_id)
        metadata = payload.get("metadata", {})
        successful[expert_id] = ExpertResponse(
            expert_id=expert_id,
            expert_name=expert.name,
            domain=expert.domain,
            response=payload.get("response", ""),
            confidence=float(metadata.get("confidence", 1.0) or 1.0),
            model=metadata.get("model", "unknown"),
            tokens_used=int(metadata.get("tokens_used", 0) or 0),
            processing_time=float(metadata.get("processing_time", 0.0) or 0.0),
            metadata=metadata,
            sources=payload.get("sources", []),
        )

    processing_time = (_now() - start).total_seconds()
    failed = {k: v.get("error") for k, v in raw.items() if not v.get("success")}
    summary = (
        f"{len(successful)}/{len(raw)} experts responded in {processing_time:.2f}s"
    )
    return CollaborateResponse(
        success=bool(successful),
        message=f"Collaboration completed with {len(successful)} experts",
        query_id=query_id,
        data=successful,
        summary=summary,
        error={"failed": failed} if failed else None,
    )


@router.post(
    "/feedback",
    response_model=BaseResponse,
    summary="Provide feedback",
    tags=["Feedback"],
)
async def submit_feedback(
    request: FeedbackRequest,
    _: ExpertService = Depends(get_expert_service),
) -> BaseResponse:
    logger.info(
        "feedback.received",
        extra={
            "query_id": request.query_id,
            "rating": request.rating,
            "feedback_len": len(request.feedback or ""),
        },
    )
    return BaseResponse(success=True, message="Feedback received, thank you!")

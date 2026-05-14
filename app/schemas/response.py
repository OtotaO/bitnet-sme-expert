from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .base import BaseResponse, ExpertDomain


def _now() -> datetime:
    return datetime.now(UTC)


class ExpertOutput(BaseModel):
    """Canonical envelope every ``DSPyExpert`` must produce.

    Enforced at the ``DSPyExpert._generate_impl`` boundary so downstream API
    contracts (``ExpertResponse``, ``QueryResponse``) have a stable shape
    regardless of which expert ran. Experts may add domain-specific keys to
    ``metadata`` freely.
    """

    response: str = Field(..., description="The expert's textual answer")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExpertResponse(BaseModel):
    """Response model for a single expert's output."""

    expert_id: str = Field(..., description="Unique identifier for the expert")
    expert_name: str = Field(..., description="Human-readable name of the expert")
    domain: ExpertDomain = Field(..., description="Domain of expertise")
    response: str = Field(..., description="The expert's response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    model: str = Field(..., description="Model used to generate the response")
    tokens_used: int = Field(..., ge=0, description="Number of tokens used")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=_now, description="Response timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, Any]] | None = Field(default=None)


class QueryResponse(BaseResponse[ExpertResponse]):
    """Response model for a query to a single expert."""

    query_id: str = Field(..., description="Unique identifier for the query")
    session_id: str | None = None
    data: ExpertResponse = Field(..., description="The expert's response")


class CollaborateResponse(BaseResponse[dict[str, ExpertResponse]]):
    """Response model for collaboration between multiple experts."""

    query_id: str = Field(..., description="Unique identifier for the query")
    session_id: str | None = None
    data: dict[str, ExpertResponse] = Field(..., description="Mapping of expert ids to responses")
    summary: str | None = None


class TrainingStatus(StrEnum):
    """Status of a training job."""

    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingMetrics(BaseModel):
    """Metrics from model training."""

    epoch: int
    loss: float
    learning_rate: float
    step: int
    samples_per_second: float | None = None
    epoch_time: float | None = None
    eval_loss: float | None = None
    eval_accuracy: float | None = None
    eval_f1: float | None = None
    eval_precision: float | None = None
    eval_recall: float | None = None


class TrainingJobResponse(BaseResponse[dict[str, Any]]):
    """Response model for a training job."""

    job_id: str
    name: str
    status: TrainingStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: TrainingMetrics | None = None
    model_path: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ExpertInfo(BaseModel):
    """Information about an available expert."""

    id: str
    name: str
    domain: ExpertDomain
    description: str
    model: str
    version: str
    is_custom: bool = False
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ListExpertsResponse(BaseResponse[list[ExpertInfo]]):
    """Response model for listing available experts."""

    count: int
    data: list[ExpertInfo]


# ---------------------------------------------------------------------------
# Ops endpoints — health, readiness, cache
# ---------------------------------------------------------------------------


class RootInfoResponse(BaseModel):
    """Payload returned by ``GET /``."""

    name: str
    version: str
    environment: str
    docs: str | None = None


class HealthCheckResponse(BaseModel):
    """Payload returned by ``GET /health``."""

    status: Literal["ok"] = "ok"
    timestamp: datetime = Field(default_factory=_now)
    version: str


class LivenessResponse(BaseModel):
    """Payload returned by ``GET /health/live`` (k8s livenessProbe)."""

    status: Literal["ok"] = "ok"
    service: str


class ReadinessResponse(BaseModel):
    """Payload returned by ``GET /health/ready`` (k8s readinessProbe).

    ``checks`` is intentionally permissive because each dependency may surface
    its own structured detail (e.g. redis ping latency, db pool stats).
    """

    status: Literal["ok", "degraded"]
    timestamp: datetime = Field(default_factory=_now)
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CacheClearResponse(BaseModel):
    """Payload returned by ``POST /cache/clear``."""

    status: Literal["ok"] = "ok"
    timestamp: datetime = Field(default_factory=_now)


class CacheStatsResponse(BaseModel):
    """Payload returned by ``GET /cache/stats``."""

    total_entries: int = Field(ge=0)
    expired_entries: int = Field(ge=0)
    max_size: int = Field(ge=0)
    ttl_seconds: int = Field(ge=0)

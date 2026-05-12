from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .base import BaseResponse, ExpertDomain


def _now() -> datetime:
    return datetime.now(UTC)


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

class TrainingStatus(str, Enum):
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

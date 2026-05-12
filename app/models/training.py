"""Training job models — Pydantic v2."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(UTC)


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(BaseModel):
    """Represents a training job."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    status: TrainingStatus = TrainingStatus.PENDING
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] | None = None
    progress: float | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    start_time: datetime | None = None
    end_time: datetime | None = None


class TrainingJobCreate(BaseModel):
    """Schema for creating a new training job."""

    model_name: str
    dataset_path: str
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingJobResponse(BaseModel):
    """Response schema for training job operations."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    status: TrainingStatus
    created_at: datetime
    config: dict[str, Any]
    metrics: dict[str, Any] | None = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    @classmethod
    def from_job(cls, job: TrainingJob) -> "TrainingJobResponse":
        return cls.model_validate(job, from_attributes=True)

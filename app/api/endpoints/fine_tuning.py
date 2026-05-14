"""API endpoints for SFT jobs (Unsloth + TRL).

These are guarded behind authentication and the ``[finetune]`` extra. The
service itself imports heavy deps lazily, so this endpoint only fails when an
SFT job is actually started without the extra installed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from app.models.training import TrainingJob, TrainingStatus

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory store; swap for Redis or DB in production.
_active_jobs: dict[str, TrainingJob] = {}


def _now() -> datetime:
    return datetime.now(UTC)


class FineTuningRequest(BaseModel):
    """Request model for starting a fine-tuning job."""

    model_name: str = Field(
        default="unsloth/llama-3.1-8b-bnb-4bit",
        description="Base model to fine-tune",
    )
    dataset_path: str = Field(..., description="Path to the training dataset")
    output_dir: str | None = Field(None, description="Output directory")
    max_seq_length: int = Field(2048)
    batch_size: int = Field(2)
    num_train_epochs: int = Field(1)
    learning_rate: float = Field(2e-4)
    lora_rank: int = Field(16)
    lora_alpha: int = Field(16)


class FineTuningResponse(BaseModel):
    job_id: str
    status: str
    message: str
    details: dict[str, Any] | None = None


class TrainingJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    metrics: dict[str, Any] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None


def _run_training_job(job_id: str, request: FineTuningRequest) -> None:
    """Background task — synchronous because the trainer holds the GPU."""
    job = _active_jobs[job_id]
    job.status = TrainingStatus.RUNNING.value
    job.start_time = _now()
    try:
        from app.services.fine_tuning import FineTuningService, TrainingConfig

        cfg = TrainingConfig(
            model_name=request.model_name,
            dataset_path=request.dataset_path,
            output_dir=request.output_dir or f"models/finetuned/{job_id}",
            max_seq_length=request.max_seq_length,
            per_device_train_batch_size=request.batch_size,
            num_train_epochs=request.num_train_epochs,
            learning_rate=request.learning_rate,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_alpha,
        )
        FineTuningService(cfg).train()
        job.status = TrainingStatus.COMPLETED.value
        job.metrics = {"output_dir": cfg.output_dir, "model_name": request.model_name}
    except Exception as exc:
        logger.exception("finetune.failed", extra={"job_id": job_id})
        job.status = TrainingStatus.FAILED.value
        job.error = str(exc)
    finally:
        job.end_time = _now()


@router.post("/fine-tune", response_model=FineTuningResponse)
async def start_fine_tuning(
    request: FineTuningRequest,
    background_tasks: BackgroundTasks,
) -> FineTuningResponse:
    job_id = str(uuid.uuid4())
    job = TrainingJob(id=job_id, status=TrainingStatus.PENDING, config=request.model_dump())
    _active_jobs[job_id] = job
    background_tasks.add_task(_run_training_job, job_id, request)
    return FineTuningResponse(
        job_id=job_id,
        status="started",
        message=f"Training job {job_id} queued",
        details={
            "model": request.model_name,
            "dataset": request.dataset_path,
            "status_url": f"/api/v1/training/status/{job_id}",
        },
    )


@router.get("/training/status/{job_id}", response_model=TrainingJobStatusResponse)
async def get_training_status(job_id: str) -> TrainingJobStatusResponse:
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found"
        )
    return TrainingJobStatusResponse(
        job_id=job.id,
        status=str(job.status),
        progress=job.progress,
        metrics=job.metrics,
        start_time=job.start_time,
        end_time=job.end_time,
        error=job.error,
    )


@router.get("/training/jobs", response_model=dict[str, str])
async def list_training_jobs() -> dict[str, str]:
    return {job_id: str(job.status) for job_id, job in _active_jobs.items()}

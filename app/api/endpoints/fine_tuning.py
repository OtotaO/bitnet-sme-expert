"""API endpoints for fine-tuning models with Unsloth."""
import os
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ....services.fine_tuning import FineTuningService, TrainingConfig
from ....models.training import TrainingJob, TrainingStatus
from ....database import get_db, Session

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory store for active training jobs (replace with Redis in production)
active_jobs: Dict[str, TrainingJob] = {}

class FineTuningRequest(BaseModel):
    """Request model for starting a fine-tuning job."""
    model_name: str = Field(
        default="unsloth/llama-3-8b-bnb-4bit",
        description="Base model to fine-tune"
    )
    dataset_path: str = Field(
        ...,
        description="Path to the training dataset (local file or Hugging Face dataset name)"
    )
    output_dir: Optional[str] = Field(
        None,
        description="Directory to save the fine-tuned model (default: models/finetuned/{job_id})"
    )
    max_seq_length: int = Field(
        2048,
        description="Maximum sequence length"
    )
    batch_size: int = Field(
        4,
        description="Batch size per device"
    )
    num_train_epochs: int = Field(
        1,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        2e-4,
        description="Learning rate"
    )
    lora_rank: int = Field(
        16,
        description="LoRA rank"
    )
    lora_alpha: int = Field(
        32,
        description="LoRA alpha"
    )

class FineTuningResponse(BaseModel):
    """Response model for fine-tuning job creation."""
    job_id: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class TrainingJobStatus(BaseModel):
    """Training job status response."""
    job_id: str
    status: str
    progress: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None

async def run_training_job(
    job_id: str,
    request: FineTuningRequest,
    db: Session
):
    """Run the training job in the background."""
    try:
        # Update job status
        job = active_jobs[job_id]
        job.status = TrainingStatus.RUNNING
        job.start_time = datetime.utcnow()
        
        # Configure training
        config = TrainingConfig(
            model_name=request.model_name,
            dataset_path=request.dataset_path,
            output_dir=request.output_dir or f"models/finetuned/{job_id}",
            max_seq_length=request.max_seq_length,
            batch_size=request.batch_size,
            num_train_epochs=request.num_train_epochs,
            learning_rate=request.learning_rate,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_alpha,
        )
        
        # Initialize and run training
        trainer = FineTuningService(config)
        trainer.train()
        
        # Update job status
        job.status = TrainingStatus.COMPLETED
        job.end_time = datetime.utcnow()
        job.metrics = {
            "output_dir": config.output_dir,
            "model_name": request.model_name,
        }
        
    except Exception as e:
        logger.exception(f"Training job {job_id} failed")
        job.status = TrainingStatus.FAILED
        job.error = str(e)
        job.end_time = datetime.utcnow()
    finally:
        # Clean up
        job.end_time = job.end_time or datetime.utcnow()
        # Here you would typically save the job to a database
        # For now, we're just keeping it in memory

@router.post("/fine-tune", response_model=FineTuningResponse)
async def start_fine_tuning(
    request: FineTuningRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a new fine-tuning job."""
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = TrainingJob(
        id=job_id,
        status=TrainingStatus.PENDING,
        config=request.dict(),
        created_at=datetime.utcnow()
    )
    
    # Store job
    active_jobs[job_id] = job
    
    # Start training in background
    background_tasks.add_task(run_training_job, job_id, request, db)
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": f"Training job {job_id} has been queued",
        "details": {
            "model": request.model_name,
            "dataset": request.dataset_path,
            "status_url": f"/api/v1/training/status/{job_id}"
        }
    }

@router.get("/training/status/{job_id}", response_model=TrainingJobStatus)
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    job = active_jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    return {
        "job_id": job_id,
        "status": job.status.value,
        "progress": job.progress,
        "metrics": job.metrics,
        "start_time": job.start_time,
        "end_time": job.end_time,
        "error": job.error
    }

@router.get("/training/jobs", response_model=Dict[str, str])
async def list_training_jobs():
    """List all training jobs."""
    return {
        job_id: job.status.value
        for job_id, job in active_jobs.items()
    }

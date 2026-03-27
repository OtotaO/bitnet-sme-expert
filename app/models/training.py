"""Training job models and database schema."""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class TrainingStatus(str, Enum):
    """Status of a training job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingJob(BaseModel):
    """Represents a training job."""
    id: str
    status: TrainingStatus = TrainingStatus.PENDING
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class TrainingJobCreate(BaseModel):
    """Schema for creating a new training job."""
    model_name: str
    dataset_path: str
    config: Dict[str, Any] = Field(default_factory=dict)

class TrainingJobResponse(BaseModel):
    """Response schema for training job operations."""
    id: str
    status: TrainingStatus
    created_at: datetime
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @classmethod
    def from_orm(cls, job: 'TrainingJob') -> 'TrainingJobResponse':
        """Convert a TrainingJob to a TrainingJobResponse."""
        return cls(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            config=job.config,
            metrics=job.metrics,
            error=job.error,
            start_time=job.start_time,
            end_time=job.end_time,
        )

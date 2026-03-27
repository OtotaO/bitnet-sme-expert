from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from .base import ExpertDomain, BaseResponse

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
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    sources: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Sources or references used to generate the response"
    )

class QueryResponse(BaseResponse[ExpertResponse]):
    """Response model for a query to a single expert."""
    query_id: str = Field(..., description="Unique identifier for the query")
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for multi-turn conversations"
    )
    data: ExpertResponse = Field(..., description="The expert's response")

class CollaborateResponse(BaseResponse[Dict[str, ExpertResponse]]):
    """Response model for collaboration between multiple experts."""
    query_id: str = Field(..., description="Unique identifier for the query")
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for multi-turn conversations"
    )
    data: Dict[str, ExpertResponse] = Field(
        ...,
        description="Mapping of expert domains to their responses"
    )
    summary: Optional[str] = Field(
        None,
        description="Optional summary of the experts' responses"
    )

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
    samples_per_second: Optional[float] = None
    epoch_time: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    eval_f1: Optional[float] = None
    eval_precision: Optional[float] = None
    eval_recall: Optional[float] = None

class TrainingJobResponse(BaseResponse[Dict[str, Any]]):
    """Response model for a training job."""
    job_id: str = Field(..., description="Unique identifier for the training job")
    name: str = Field(..., description="Name of the expert being trained")
    status: TrainingStatus = Field(..., description="Current status of the training job")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Training progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="When the job was created")
    started_at: Optional[datetime] = Field(None, description="When training started")
    completed_at: Optional[datetime] = Field(None, description="When training completed")
    metrics: Optional[TrainingMetrics] = Field(None, description="Training metrics")
    model_path: Optional[str] = Field(None, description="Path to the trained model")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional job data")

class ExpertInfo(BaseModel):
    """Information about an available expert."""
    id: str = Field(..., description="Unique identifier for the expert")
    name: str = Field(..., description="Human-readable name")
    domain: ExpertDomain = Field(..., description="Domain of expertise")
    description: str = Field(..., description="Description of the expert's capabilities")
    model: str = Field(..., description="Base model used by the expert")
    version: str = Field(..., description="Expert version")
    is_custom: bool = Field(False, description="Whether this is a custom expert")
    created_at: datetime = Field(..., description="When the expert was created")
    updated_at: datetime = Field(..., description="When the expert was last updated")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the expert"
    )

class ListExpertsResponse(BaseResponse[List[ExpertInfo]]):
    """Response model for listing available experts."""
    count: int = Field(..., description="Total number of experts")
    data: List[ExpertInfo] = Field(..., description="List of available experts")

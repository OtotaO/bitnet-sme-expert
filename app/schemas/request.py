from typing import Any

from pydantic import BaseModel, Field

from .base import ExpertDomain


class QueryRequest(BaseModel):
    """Request model for querying an expert."""

    question: str = Field(..., description="The question to ask the expert")
    domain: ExpertDomain | None = Field(
        None, description="Specific expert domain to query (auto-detected if not provided)"
    )
    max_tokens: int = Field(
        150, ge=10, le=2048, description="Maximum number of tokens in the response"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    context: dict[str, Any] | None = Field(
        None, description="Additional context for the expert to consider"
    )


class CollaborateRequest(BaseModel):
    """Request model for collaborating with multiple experts."""

    question: str = Field(..., description="The question to ask the experts")
    domains: list[ExpertDomain] = Field(
        default_factory=list,
        description="List of expert domains to consult (all available experts if empty)",
    )
    max_tokens: int = Field(
        100, ge=10, le=1024, description="Maximum number of tokens per expert response"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    context: dict[str, Any] | None = Field(
        None, description="Additional context for the experts to consider"
    )


class TrainingRequest(BaseModel):
    """Request model for training a custom expert."""

    name: str = Field(..., description="Name of the expert")
    description: str = Field(..., description="Description of the expert's capabilities")
    domain: ExpertDomain = Field(..., description="Domain of expertise")
    base_model: str = Field(..., description="Base model to fine-tune")
    training_data: list[dict[str, str]] = Field(
        ..., description="List of training examples with 'input' and 'output' fields"
    )
    epochs: int = Field(3, ge=1, le=10, description="Number of training epochs")
    learning_rate: float = Field(5e-5, ge=1e-6, le=1e-3, description="Learning rate")
    batch_size: int = Field(8, ge=1, le=32, description="Batch size")
    max_length: int = Field(512, ge=128, le=4096, description="Maximum sequence length")
    use_quantization: bool = Field(True, description="Whether to use 1-bit quantization")
    parameters: dict[str, Any] | None = Field(None, description="Additional training parameters")


class FeedbackRequest(BaseModel):
    """Request model for providing feedback on expert responses."""

    query_id: str = Field(..., description="ID of the query being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback: str | None = Field(None, description="Detailed feedback")
    corrections: dict[str, Any] | None = Field(
        None, description="Corrections to the expert's response"
    )


class SearchRequest(BaseModel):
    """Request model for searching across experts."""

    query: str = Field(..., description="Search query")
    domains: list[ExpertDomain] | None = Field(
        None, description="Expert domains to search within (all domains if not specified)"
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_sources: bool = Field(True, description="Whether to include source information")

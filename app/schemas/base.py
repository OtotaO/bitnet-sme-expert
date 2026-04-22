from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime

T = TypeVar('T')

class ExpertDomain(str, Enum):
    """Available expert domains."""
    MATH = "math"
    CODE = "code"
    GENERAL = "general"
    FINANCE = "finance"
    HEALTH = "health"
    LEGAL = "legal"

class BaseResponse(BaseModel, Generic[T]):
    """Base response model for all API responses."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    data: Optional[T] = None
    error: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {},
                "data": None,
                "error": None
            }
        }

class ErrorResponse(BaseResponse[None]):
    """Error response model."""
    success: bool = False
    error: Dict[str, Any] = Field(
        default_factory=lambda: {"code": "error_code", "detail": "Error details"}
    )

class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: str
    uptime: float
    environment: str

"""Common schemas: domain enum, generic envelopes."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ExpertDomain(StrEnum):
    """Supported expert domains."""

    MATH = "math"
    CODE = "code"
    GENERAL = "general"
    FINANCE = "finance"
    HEALTH = "health"
    LEGAL = "legal"


class BaseResponse[T](BaseModel):
    """Standard envelope for API responses."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": None,
                "error": None,
            }
        }
    )

    success: bool = True
    message: str = "Operation completed successfully"
    data: T | None = None
    error: dict[str, Any] | None = None


class ErrorResponse(BaseResponse[None]):
    """Error envelope."""

    success: bool = False
    error: dict[str, Any] = Field(
        default_factory=lambda: {"code": "error_code", "detail": "Error details"}
    )


class HealthCheck(BaseModel):
    """Health-check payload."""

    status: str
    version: str
    timestamp: str
    uptime: float
    environment: str

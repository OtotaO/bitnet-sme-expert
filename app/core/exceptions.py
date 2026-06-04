"""Custom exceptions and HTTP error handlers for dspy-sme-expert."""

import logging
import traceback
import uuid
from enum import StrEnum
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    """Standard error codes for the application."""

    # General errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"

    # Expert system errors
    EXPERT_NOT_FOUND = "EXPERT_NOT_FOUND"
    EXPERT_INITIALIZATION_ERROR = "EXPERT_INITIALIZATION_ERROR"
    EXPERT_GENERATION_ERROR = "EXPERT_GENERATION_ERROR"
    EXPERT_TIMEOUT = "EXPERT_TIMEOUT"

    # Model errors
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    MODEL_QUOTA_EXCEEDED = "MODEL_QUOTA_EXCEEDED"
    MODEL_INVALID_REQUEST = "MODEL_INVALID_REQUEST"
    MODEL_AUTHENTICATION_ERROR = "MODEL_AUTHENTICATION_ERROR"

    # Database errors
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_OPERATION_ERROR = "DATABASE_OPERATION_ERROR"

    # Cache errors
    CACHE_ERROR = "CACHE_ERROR"

    # Fine-tuning errors
    FINE_TUNING_ERROR = "FINE_TUNING_ERROR"
    FINE_TUNING_DATA_ERROR = "FINE_TUNING_DATA_ERROR"


class AppException(Exception):
    """Base exception class for BitNet SME Expert System."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_SERVER_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ValidationException(AppException):
    """Raised when input validation fails."""

    def __init__(self, message: str = "Validation failed", details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class ExpertNotFoundException(AppException):
    """Raised when a requested expert is not found."""

    def __init__(self, expert_name: str, available_experts: list | None = None):
        message = f"Expert '{expert_name}' not found"
        details = {"expert_name": expert_name}
        if available_experts:
            details["available_experts"] = available_experts

        super().__init__(
            message=message,
            code=ErrorCode.EXPERT_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class ExpertInitializationException(AppException):
    """Raised when expert initialization fails."""

    def __init__(self, expert_name: str, reason: str):
        message = f"Failed to initialize expert '{expert_name}': {reason}"
        super().__init__(
            message=message,
            code=ErrorCode.EXPERT_INITIALIZATION_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"expert_name": expert_name, "reason": reason},
        )


class ExpertGenerationException(AppException):
    """Raised when expert generation fails."""

    def __init__(self, expert_name: str, reason: str, retry_count: int = 0):
        message = f"Expert '{expert_name}' failed to generate response: {reason}"
        super().__init__(
            message=message,
            code=ErrorCode.EXPERT_GENERATION_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"expert_name": expert_name, "reason": reason, "retry_count": retry_count},
        )


class ExpertTimeoutException(AppException):
    """Raised when expert operation times out."""

    def __init__(self, expert_name: str, timeout_seconds: int):
        message = f"Expert '{expert_name}' timed out after {timeout_seconds} seconds"
        super().__init__(
            message=message,
            code=ErrorCode.EXPERT_TIMEOUT,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details={"expert_name": expert_name, "timeout_seconds": timeout_seconds},
        )


class ModelException(AppException):
    """Base class for model-related exceptions."""

    pass


class ModelNotAvailableException(ModelException):
    """Raised when a model is not available."""

    def __init__(self, model_name: str, provider: str):
        message = f"Model '{model_name}' from provider '{provider}' is not available"
        super().__init__(
            message=message,
            code=ErrorCode.MODEL_NOT_AVAILABLE,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"model_name": model_name, "provider": provider},
        )


class ModelQuotaExceededException(ModelException):
    """Raised when model quota is exceeded."""

    def __init__(self, model_name: str, provider: str):
        message = f"Quota exceeded for model '{model_name}' from provider '{provider}'"
        super().__init__(
            message=message,
            code=ErrorCode.MODEL_QUOTA_EXCEEDED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"model_name": model_name, "provider": provider},
        )


class ModelAuthenticationException(ModelException):
    """Raised when model authentication fails."""

    def __init__(self, provider: str):
        message = f"Authentication failed for provider '{provider}'"
        super().__init__(
            message=message,
            code=ErrorCode.MODEL_AUTHENTICATION_ERROR,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={"provider": provider},
        )


class DatabaseException(AppException):
    """Raised when database operations fail."""

    def __init__(self, message: str = "Database operation failed", operation: str | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.DATABASE_OPERATION_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"operation": operation} if operation else {},
        )


class CacheException(AppException):
    """Raised when cache operations fail."""

    def __init__(self, message: str = "Cache operation failed", operation: str | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.CACHE_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"operation": operation} if operation else {},
        )


class FineTuningException(AppException):
    """Raised when fine-tuning operations fail."""

    def __init__(self, message: str = "Fine-tuning operation failed", phase: str | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.FINE_TUNING_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"phase": phase} if phase else {},
        )


class RateLimitException(AppException):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int | None = None):
        super().__init__(
            message=message,
            code=ErrorCode.RATE_LIMITED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after": retry_after} if retry_after else {},
        )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle BitNet SME custom exceptions."""

    logger.error(
        f"BitNet SME Exception: {exc.code.value} - {exc.message}",
        extra={
            "error_code": exc.code.value,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    error_response = {
        "error": True,
        "code": exc.code.value,
        "message": exc.message,
        "details": exc.details,
        "path": request.url.path,
        "method": request.method,
    }

    return JSONResponse(status_code=exc.status_code, content=error_response)


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic validation exceptions."""

    logger.error(
        f"Validation error: {exc!s}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error_type": type(exc).__name__,
        },
    )

    # Extract validation details if available
    details = {}
    if hasattr(exc, "errors"):
        details = {"validation_errors": exc.errors()}

    error_response = {
        "error": True,
        "code": ErrorCode.VALIDATION_ERROR.value,
        "message": "Request validation failed",
        "details": details,
        "path": request.url.path,
        "method": request.method,
    }

    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_response)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""

    logger.error(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={"status_code": exc.status_code, "path": request.url.path, "method": request.method},
    )

    error_response = {
        "error": True,
        "code": f"HTTP_{exc.status_code}",
        "message": exc.detail,
        "details": {},
        "path": request.url.path,
        "method": request.method,
    }

    return JSONResponse(status_code=exc.status_code, content=error_response)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""

    # Stable, collision-free correlation id (the old hash(str(exc)) could collide
    # and isn't process-stable). The exception type is logged server-side but not
    # returned to the client — it leaks internal implementation detail.
    error_id = f"err_{uuid.uuid4().hex[:16]}"

    logger.error(
        f"Unexpected error [{error_id}]: {exc!s}",
        extra={
            "error_id": error_id,
            "error_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
    )

    error_response = {
        "error": True,
        "code": ErrorCode.INTERNAL_SERVER_ERROR.value,
        "message": "An unexpected error occurred",
        "details": {"error_id": error_id},
        "path": request.url.path,
        "method": request.method,
    }

    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response)

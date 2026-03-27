"""
Error handling middleware for comprehensive exception management.
"""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from ..core.exceptions import (
    BitNetSMEException,
    bitnet_sme_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)


def setup_error_handling(app: FastAPI) -> None:
    """Set up comprehensive error handling for the FastAPI application."""

    # Custom BitNet SME exceptions
    app.add_exception_handler(BitNetSMEException, bitnet_sme_exception_handler)

    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)

    # HTTP exceptions
    from fastapi import HTTPException
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Catch-all for unexpected errors
    app.add_exception_handler(Exception, general_exception_handler)
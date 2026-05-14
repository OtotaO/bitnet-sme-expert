"""
CORS (Cross-Origin Resource Sharing) middleware configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware for the FastAPI application."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=settings.ALLOW_CREDENTIALS,
        allow_methods=settings.ALLOWED_METHODS,
        allow_headers=settings.ALLOWED_HEADERS,
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )

"""Logging middleware for FastAPI with request IDs and metrics."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response

from ..observability import (
    get_request_id,
    observe_http_request,
    reset_request_id,
    set_request_id,
)

logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """Middleware for structured request/response logging and metrics."""

    async def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log request lifecycle and capture request metrics."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = set_request_id(request_id)

        start_time = time.perf_counter()
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        logger.info(
            "request.received",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_host": client_host,
                "query": request.url.query,
            },
        )

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            elapsed_s = time.perf_counter() - start_time
            observe_http_request(method=method, path=path, status_code=500, elapsed_s=elapsed_s)
            logger.exception(
                "request.failed",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "elapsed_ms": round(elapsed_s * 1000, 2),
                },
            )
            raise
        finally:
            if "response" not in locals():
                reset_request_id(token)

        elapsed_s = time.perf_counter() - start_time
        observe_http_request(
            method=method,
            path=path,
            status_code=status_code,
            elapsed_s=elapsed_s,
        )

        response.headers["X-Request-ID"] = get_request_id()
        logger.info(
            "request.completed",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "elapsed_ms": round(elapsed_s * 1000, 2),
            },
        )
        reset_request_id(token)
        return response

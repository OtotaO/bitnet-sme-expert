"""Observability primitives: structured logging, request context, and Prometheus metrics."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import UTC, datetime
from typing import Any, ClassVar

from prometheus_client import Counter, Histogram

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_ERRORS_TOTAL = Counter(
    "http_request_errors_total",
    "Total HTTP requests resulting in server errors",
    ["method", "path", "status_code"],
)

DOMAIN_REQUESTS_TOTAL = Counter(
    "domain_requests_total",
    "Total domain-level expert requests",
    ["domain", "status"],
)


class JsonLogFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    RESERVED_FIELDS: ClassVar[set[str]] = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }

        for key, value in record.__dict__.items():
            if key not in self.RESERVED_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a JSON formatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def set_request_id(request_id: str) -> contextvars.Token:
    """Set request id in context and return token for reset."""
    return _request_id_ctx.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    """Reset request id context variable."""
    _request_id_ctx.reset(token)


def get_request_id() -> str:
    """Return current request id from context."""
    return _request_id_ctx.get()


def normalized_path(raw_path: str) -> str:
    """Reduce high-cardinality path labels for metrics."""
    if raw_path.startswith("/api/v1"):
        segments = raw_path.split("/")
        if len(segments) >= 4:
            return "/".join(segments[:4])
    return raw_path


def observe_http_request(method: str, path: str, status_code: int, elapsed_s: float) -> None:
    """Record HTTP request metrics."""
    status_label = str(status_code)
    metric_path = normalized_path(path)
    REQUEST_LATENCY_SECONDS.labels(
        method=method, path=metric_path, status_code=status_label
    ).observe(elapsed_s)
    HTTP_REQUEST_TOTAL.labels(method=method, path=metric_path, status_code=status_label).inc()
    if status_code >= 500:
        HTTP_REQUEST_ERRORS_TOTAL.labels(
            method=method, path=metric_path, status_code=status_label
        ).inc()


def record_domain_outcome(domain: str, success: bool) -> None:
    """Record per-domain request outcomes."""
    DOMAIN_REQUESTS_TOTAL.labels(domain=domain, status="success" if success else "error").inc()

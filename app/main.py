"""DSPy SME Expert API — FastAPI entrypoint.

Wires up the lifespan-managed ``ExpertService``, configures DSPy globals (LM,
async workers, optional MLflow autolog), and mounts the API routers behind
standard middleware (CORS, structured logging) with rate limiting and per-route
authorization dependencies (see ``app/auth.py``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.endpoints import api_router
from app.auth import require_role
from app.bootstrap import register_experts
from app.config import settings
from app.database import Base, engine, init_db
from app.limiter import limiter
from app.llm import configure_dspy
from app.middleware import LoggingMiddleware, setup_cors, setup_error_handling
from app.models import feedback as _feedback_model  # noqa: F401 — register table for create_all
from app.observability import configure_logging
from app.schemas.response import (
    CacheClearResponse,
    CacheStatsResponse,
    HealthCheckResponse,
    LivenessResponse,
    ReadinessResponse,
    RootInfoResponse,
)
from app.services.expert_service import ExpertService

load_dotenv()
configure_logging(level=getattr(logging, settings.LOG_LEVEL.value, logging.INFO))
logger = logging.getLogger(__name__)

# DB bootstrap. There is no migration tool wired in yet, so this create_all is
# the schema source for dev / test; add Alembic before relying on it in prod.
Base.metadata.create_all(bind=engine)
init_db()

expert_service: ExpertService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize / tear down the expert service and DSPy."""
    global expert_service  # noqa: PLW0603 — module-level singleton wired in lifespan
    logger.info("app.startup")

    configure_dspy()

    expert_service = ExpertService()
    await register_experts(expert_service)
    await expert_service.initialize()
    logger.info("app.startup.completed")

    try:
        yield
    finally:
        logger.info("app.shutdown")
        if expert_service is not None:
            await expert_service.cleanup()
            expert_service = None


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.DESCRIPTION,
    version=settings.API_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

setup_cors(app)
setup_error_handling(app)
# Authorization is enforced per-route via Depends(require_role(...)), not here —
# see app/auth.py. (No AuthzMiddleware: path-prefix matching is bypass-prone.)
app.middleware("http")(LoggingMiddleware())

app.include_router(api_router, prefix=settings.API_PREFIX)


# ---------------------------------------------------------------------------
# Health & ops endpoints
# ---------------------------------------------------------------------------


def _check_database() -> dict[str, Any]:
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


def _check_redis() -> dict[str, Any]:
    if not settings.REDIS_URL:
        return {"status": "skipped", "reason": "REDIS_URL not configured"}
    import redis

    client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
    client.ping()
    return {"status": "ok"}


def _check_experts() -> dict[str, Any]:
    if expert_service is None:
        return {"status": "error", "reason": "service not initialized"}
    return {"status": "ok", "count": len(expert_service._experts)}


@app.get("/", include_in_schema=False, response_model=RootInfoResponse)
async def root() -> RootInfoResponse:
    return RootInfoResponse(
        name=settings.APP_NAME,
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT.value,
        docs=settings.DOCS_URL,
    )


@app.get("/health", response_model=HealthCheckResponse)
@limiter.limit("60/minute")
async def health(request: Request) -> HealthCheckResponse:
    return HealthCheckResponse(version=settings.API_VERSION)


@app.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    return LivenessResponse(service=settings.APP_NAME)


@app.get("/health/ready", response_model=ReadinessResponse)
async def readiness() -> JSONResponse:
    checks: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name, fn in (
        ("database", _check_database),
        ("redis", _check_redis),
        ("experts", _check_experts),
    ):
        try:
            checks[name] = fn()
            if checks[name].get("status") == "error":
                failures[name] = checks[name].get("reason", "unknown")
        except Exception as exc:
            failures[name] = str(exc)
            checks[name] = {"status": "error", "error": str(exc)}

    body = ReadinessResponse(
        status="ok" if not failures else "degraded",
        checks=checks,
    )
    code = status.HTTP_200_OK if not failures else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body.model_dump(mode="json"), status_code=code)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/cache/clear",
    response_model=CacheClearResponse,
    dependencies=[Depends(require_role("admin"))],
)
@limiter.limit("5/minute")
async def cache_clear(request: Request) -> CacheClearResponse:
    from app.utils.cache import cache

    cache.clear()
    return CacheClearResponse()


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    from app.utils.cache import cache

    return CacheStatsResponse(**cache.stats())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD and not settings.is_production,
        log_level=settings.LOG_LEVEL.value.lower(),
        workers=settings.API_WORKERS,
    )

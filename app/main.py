"""DSPy SME Expert API — FastAPI entrypoint.

Wires up the lifespan-managed ``ExpertService``, configures DSPy globals (LM,
async workers, optional MLflow autolog), and mounts the API routers behind
standard middleware (CORS, auth, structured logging, rate limiting).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.endpoints import api_router
from app.config import settings
from app.database import Base, engine, init_db
from app.llm import configure_dspy
from app.middleware import AuthzMiddleware, LoggingMiddleware, setup_cors, setup_error_handling
from app.observability import configure_logging
from app.services.expert_service import ExpertService

load_dotenv()
configure_logging(level=getattr(logging, settings.LOG_LEVEL.value, logging.INFO))
logger = logging.getLogger(__name__)

# DB bootstrap. Async migrations live in Alembic; the engine here covers
# bootstrap for dev / test where Alembic hasn't been run.
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
    await _register_experts(expert_service)
    await expert_service.initialize()
    logger.info("app.startup.completed")

    try:
        yield
    finally:
        logger.info("app.shutdown")
        if expert_service is not None:
            await expert_service.cleanup()
            expert_service = None


async def _register_experts(service: ExpertService) -> None:
    """Register all bundled experts. The actual LM per expert lives in ``app/llm.py``."""
    from app.experts.code_expert import CodeExpert
    from app.experts.general_expert import GeneralExpert
    from app.experts.math_expert import MathExpert
    from app.schemas.base import ExpertDomain

    service.register_expert_class(
        domain=ExpertDomain.MATH,
        expert_class=MathExpert,
        config={
            "name": "Math Expert",
            "description": "ReAct over sympy tools, with a deterministic fast-path for trivial expressions.",
            "domain": ExpertDomain.MATH,
        },
    )
    service.register_expert_class(
        domain=ExpertDomain.CODE,
        expert_class=CodeExpert,
        config={
            "name": "Code Expert",
            "description": "ChainOfThought for code generation, debugging, refactoring, and review.",
            "domain": ExpertDomain.CODE,
        },
    )
    service.register_expert_class(
        domain=ExpertDomain.GENERAL,
        expert_class=GeneralExpert,
        config={
            "name": "General Expert",
            "description": "ChainOfThought for open-ended general knowledge questions.",
            "domain": ExpertDomain.GENERAL,
        },
    )

    # Materialize one instance per domain so the API can resolve experts by domain.
    for domain in (ExpertDomain.MATH, ExpertDomain.CODE, ExpertDomain.GENERAL):
        await service.create_expert(domain)

    logger.info("experts.registered", extra={"count": len(service._experts)})


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

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

setup_cors(app)
setup_error_handling(app)
app.add_middleware(AuthzMiddleware)
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


@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "docs": settings.DOCS_URL,
    }


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": settings.API_VERSION,
    }


@app.get("/health/live")
async def liveness() -> dict[str, Any]:
    return {"status": "ok", "service": settings.APP_NAME}


@app.get("/health/ready")
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

    body = {
        "status": "ok" if not failures else "degraded",
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
    }
    code = status.HTTP_200_OK if not failures else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body, status_code=code)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/cache/clear")
@limiter.limit("5/minute")
async def cache_clear(request: Request) -> dict[str, Any]:
    from app.utils.cache import cache

    cache.clear()
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    from app.utils.cache import cache

    return cache.stats()


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

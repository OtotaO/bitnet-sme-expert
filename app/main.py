"""BitNet SME Expert API application entrypoint."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

import redis
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from .api.endpoints import router as api_router
from .api.endpoints.fine_tuning import router as fine_tuning_router
from .config import settings
from .database import Base, engine, init_db
from .middleware.logging_middleware import LoggingMiddleware
from .observability import configure_logging
from .services.expert_service import ExpertService

load_dotenv()

configure_logging(level=getattr(logging, settings.LOG_LEVEL.value, logging.INFO))
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
init_db()

expert_service: ExpertService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup/shutdown resources."""
    global expert_service
    logger.info("app.startup")

    try:
        expert_service = ExpertService()
        await register_experts(expert_service)
        await expert_service.initialize()
        logger.info("app.startup.completed")
        yield
    except Exception:
        logger.exception("app.startup.failed")
        raise
    finally:
        logger.info("app.shutdown")
        if expert_service:
            await expert_service.cleanup()


async def register_experts(service: ExpertService) -> None:
    """Register all expert implementations."""
    from .experts.code_expert import CodeExpert
    from .experts.general_expert import GeneralExpert
    from .experts.math_expert import MathExpert

    service.register_expert_class(
        domain="math",
        expert_class=MathExpert,
        config={
            "name": "Math Expert",
            "description": "Specialized in mathematical problems and calculations",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.3,
        },
    )
    service.register_expert_class(
        domain="code",
        expert_class=CodeExpert,
        config={
            "name": "Code Expert",
            "description": "Specialized in programming and software development",
            "model_name": "gpt-4",
            "temperature": 0.5,
        },
    )
    service.register_expert_class(
        domain="general",
        expert_class=GeneralExpert,
        config={
            "name": "General Expert",
            "description": "General knowledge and broad expertise",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
        },
    )
    logger.info("experts.registered", extra={"count": len(service._expert_classes)})


app = FastAPI(
    title="BitNet SME Expert API",
    description="API for interacting with specialized AI experts.",
    version=settings.API_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)
app.middleware("http")(LoggingMiddleware())

app.include_router(api_router, prefix=settings.API_PREFIX)
app.include_router(fine_tuning_router, prefix=settings.API_PREFIX)


def _check_database() -> Dict[str, Any]:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok"}


def _check_redis() -> Dict[str, Any]:
    if not settings.REDIS_URL:
        return {"status": "skipped", "reason": "REDIS_URL not configured"}
    client = redis.from_url(settings.REDIS_URL, decode_responses=settings.REDIS_DECODE_RESPONSES)
    try:
        ping_ok = client.ping()
        return {"status": "ok" if ping_ok else "error"}
    finally:
        client.close()


def _check_providers() -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}

    if settings.OPENAI_API_KEY:
        from openai import OpenAI

        OpenAI(api_key=settings.OPENAI_API_KEY)
        results["openai"] = {"status": "ok"}
    else:
        results["openai"] = {"status": "skipped", "reason": "OPENAI_API_KEY not configured"}

    if settings.ANTHROPIC_API_KEY:
        from anthropic import Anthropic

        Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        results["anthropic"] = {"status": "ok"}
    else:
        results["anthropic"] = {
            "status": "skipped",
            "reason": "ANTHROPIC_API_KEY not configured",
        }

    if settings.GOOGLE_API_KEY:
        import google.generativeai as genai

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        results["google"] = {"status": "ok"}
    else:
        results["google"] = {"status": "skipped", "reason": "GOOGLE_API_KEY not configured"}

    return results


@app.get("/health/live")
async def liveness() -> Dict[str, Any]:
    """Liveness probe endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "service": settings.APP_NAME}


@app.get("/health/ready")
async def readiness() -> Dict[str, Any]:
    """Readiness probe endpoint with dependency verification."""
    checks: Dict[str, Any] = {}
    failures: Dict[str, str] = {}

    try:
        checks["database"] = _check_database()
    except Exception as exc:
        failures["database"] = str(exc)
        checks["database"] = {"status": "error", "error": str(exc)}

    try:
        checks["redis"] = _check_redis()
    except Exception as exc:
        failures["redis"] = str(exc)
        checks["redis"] = {"status": "error", "error": str(exc)}

    try:
        checks["providers"] = _check_providers()
    except Exception as exc:
        failures["providers"] = str(exc)
        checks["providers"] = {"status": "error", "error": str(exc)}

    payload = {
        "status": "ok" if not failures else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }

    if failures:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=payload)

    return payload


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for scraping."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/alerts/thresholds")
async def alert_thresholds() -> Dict[str, Any]:
    """Expose configured SLO-based alert thresholds."""
    return {
        "error_rate_ratio": settings.SLO_ERROR_RATE_THRESHOLD,
        "p95_latency_ms": settings.SLO_P95_LATENCY_MS_THRESHOLD,
        "window_minutes": settings.SLO_ALERT_WINDOW_MINUTES,
    }


@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request) -> Dict[str, Any]:
    """Backwards-compatible health endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.API_VERSION,
        "rate_limit": {
            "limit": request.scope.get("rate_limit", "").split("/")[0],
            "remaining": request.scope.get("remaining", 0),
        },
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": settings.DOCS_URL,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.value.lower(),
        workers=settings.API_WORKERS,
    )

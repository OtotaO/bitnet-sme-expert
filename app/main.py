"""BitNet SME Expert API application entrypoint."""

from __future__ import annotations

A high-performance API for interacting with specialized AI experts
in various domains including math, coding, and general knowledge.
"""
import logging
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

import redis
import uvicorn
import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

# Load environment variables
load_dotenv()

# Initialize database
from app.database import init_db, engine, Base
Base.metadata.create_all(bind=engine)
init_db()

from app.config import settings
from app.api.endpoints import api_router
from app.services.expert_service import ExpertService

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

async def register_experts(service: ExpertService):
    """Register all available expert implementations."""
    from app.experts.math_expert import MathExpert
    from app.experts.code_expert import CodeExpert
    from app.experts.general_expert import GeneralExpert
    
    # Register expert classes
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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# Authentication/authorization for sensitive endpoints
app.add_middleware(AuthzMiddleware)

# Add logging middleware
from app.middleware.logging_middleware import LoggingMiddleware
app.middleware("http")(LoggingMiddleware())

# Dependency to get the expert service
async def get_expert_service() -> ExpertService:
    """Dependency to get the expert service instance."""
    if expert_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Expert service not initialized",
        )
    return expert_service

# Include API routers
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "rate_limit": {
            "limit": request.scope.get("rate_limit", "").split("/")[0],
            "remaining": request.scope.get("remaining", 0)
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

# Cache statistics endpoint
@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    from app.utils.cache import cache
    return {
        "error_rate_ratio": settings.SLO_ERROR_RATE_THRESHOLD,
        "p95_latency_ms": settings.SLO_P95_LATENCY_MS_THRESHOLD,
        "window_minutes": settings.SLO_ALERT_WINDOW_MINUTES,
    }

# Clear cache endpoint (protected by rate limiting)
@app.post("/cache/clear")
@limiter.limit("1/minute")
async def clear_cache(request: Request):
    """Clear the cache."""
    from app.utils.cache import cache
    cache.clear()
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

"""
BitNet SME Expert API

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
from typing import AsyncGenerator, List, Optional
import uvicorn
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize database
from app.database import init_db, engine, Base
Base.metadata.create_all(bind=engine)
init_db()

from app.config import settings
from app.api.endpoints import api_router
from app.services.expert_service import ExpertService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Global expert service instance
expert_service = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage the application lifecycle."""
    global expert_service
    
    # Startup: Initialize services
    logger.info("Starting up...")
    
    try:
        # Initialize the expert service
        expert_service = ExpertService()
        
        # Register expert implementations
        await register_experts(expert_service)
        
        # Initialize the expert service
        await expert_service.initialize()
        
        logger.info("Startup complete")
        
        yield
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}", exc_info=True)
        raise
        
    finally:
        # Shutdown: Clean up resources
        logger.info("Shutting down...")
        if expert_service:
            await expert_service.cleanup()
        logger.info("Shutdown complete")

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
        }
    )
    
    service.register_expert_class(
        domain="code",
        expert_class=CodeExpert,
        config={
            "name": "Code Expert",
            "description": "Specialized in programming and software development",
            "model_name": "gpt-4",
            "temperature": 0.5,
        }
    )
    
    service.register_expert_class(
        domain="general",
        expert_class=GeneralExpert,
        config={
            "name": "General Expert",
            "description": "General knowledge and broad expertise",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
        }
    )
    
    logger.info(f"Registered {len(service._expert_classes)} expert classes")

# Create the FastAPI application
app = FastAPI(
    title="BitNet SME Expert API",
    description=(
        "API for interacting with specialized AI experts in various domains "
        "including math, coding, and general knowledge."
    ),
    version="0.1.0",
    contact={
        "name": "BitNet Team",
        "email": "support@bitnet.ai",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Rate limiting configuration
from slowapi import Limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[os.getenv("RATE_LIMIT", "100/minute")]
)
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
    }

# Cache statistics endpoint
@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    from app.utils.cache import cache
    return {
        "status": "ok",
        "cache_stats": cache.stats(),
        "timestamp": datetime.utcnow().isoformat()
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
        "message": "Cache cleared",
        "timestamp": datetime.utcnow().isoformat()
    }

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "BitNet SME Expert API",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs",
        "redoc": "/redoc",
    }

if __name__ == "__main__":
    # Run the application using uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        workers=1,
    )

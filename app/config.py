"""Application configuration via Pydantic Settings v2.

Provider-key validation is intentionally minimal here: DSPy / LiteLLM handle
provider auth at request time, and missing keys surface as 4xx errors from the
provider rather than as startup failures. Production still requires explicit
``ALLOWED_ORIGINS`` and non-default secret keys.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from typing import Any

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    SIMPLE = "simple"
    JSON = "json"
    DETAILED = "detailed"


_DEFAULT_SECRET_KEY = "change-this-to-a-secure-random-string-in-production"
_DEFAULT_JWT_SECRET_KEY = "jwt-secret-change-in-production"


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    # Application identity
    APP_NAME: str = "DSPy SME Expert"
    API_VERSION: str = "3.0.0"
    API_PREFIX: str = "/api/v1"
    DESCRIPTION: str = "DSPy-powered multi-expert SME system with FastAPI, MLflow, and LiteLLM."

    # Environment
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False

    # Server
    API_HOST: str = Field(default="0.0.0.0", alias="HOST")
    API_PORT: int = Field(default=8000, alias="PORT")
    API_WORKERS: int = Field(default=1, alias="WORKERS")
    API_RELOAD: bool = Field(default=True, alias="RELOAD")

    # Security
    SECRET_KEY: str = _DEFAULT_SECRET_KEY
    JWT_SECRET_KEY: str = _DEFAULT_JWT_SECRET_KEY
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ENABLE_AUTHENTICATION: bool = False

    # CORS
    ALLOWED_ORIGINS: list[str] = Field(default_factory=list)
    ALLOW_CREDENTIALS: bool = False
    ALLOWED_METHODS: list[str] = Field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    ALLOWED_HEADERS: list[str] = Field(default_factory=lambda: ["Authorization", "Content-Type"])

    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: LogFormat = LogFormat.JSON
    ENABLE_STRUCTURED_LOGGING: bool = True

    # OpenAPI
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"

    # Expert defaults
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # Rate limiting
    RATE_LIMIT: str = "100/minute"
    ENABLE_RATE_LIMITING: bool = True

    # Cache
    CACHE_TTL: int = 300
    ENABLE_CACHING: bool = True

    # Database / Redis
    DATABASE_URL: str | None = None
    DATABASE_ECHO: bool = False
    REDIS_URL: str | None = None

    # Performance
    REQUEST_TIMEOUT: int = 30
    MAX_CONCURRENT_REQUESTS: int = 8

    # Observability
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    ENABLE_TRACING: bool = False
    SLO_ERROR_RATE_THRESHOLD: float = 0.01
    SLO_P95_LATENCY_MS_THRESHOLD: int = 750
    SLO_ALERT_WINDOW_MINUTES: int = 5

    # MLflow (opt-in)
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT_NAME: str = "dspy-sme-expert"

    @field_validator("ALLOWED_ORIGINS", "ALLOWED_METHODS", "ALLOWED_HEADERS", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def _coerce_env(cls, v: Any) -> Environment:
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v or Environment.DEVELOPMENT

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _db_default(cls, v: Any) -> str:
        if isinstance(v, str) and v:
            return v
        return "sqlite:///./dspy_sme.db"

    @model_validator(mode="after")
    def _validate_production(self) -> Settings:
        if not self.is_production:
            return self
        if self.SECRET_KEY == _DEFAULT_SECRET_KEY:
            raise ValueError("SECRET_KEY must be changed from the default in production")
        if self.JWT_SECRET_KEY == _DEFAULT_JWT_SECRET_KEY:
            raise ValueError("JWT_SECRET_KEY must be changed from the default in production")
        if not self.ALLOWED_ORIGINS or "*" in self.ALLOWED_ORIGINS:
            raise ValueError("ALLOWED_ORIGINS must be an explicit allowlist (no '*') in production")
        return self

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    @computed_field  # type: ignore[misc]
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == Environment.DEVELOPMENT

    @computed_field  # type: ignore[misc]
    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == Environment.TESTING

    @computed_field  # type: ignore[misc]
    @property
    def api_info(self) -> dict[str, Any]:
        return {
            "title": self.APP_NAME,
            "description": self.DESCRIPTION,
            "version": self.API_VERSION,
        }


settings = Settings()

# Basic root logging — full structured config lives in observability.configure_logging.
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.value), force=False)
for noisy in ("httpx", "httpcore", "litellm", "urllib3", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Surface MLflow tracking URI as an env var so DSPy/MLflow pick it up downstream.
if settings.MLFLOW_TRACKING_URI and "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI

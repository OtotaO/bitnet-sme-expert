"""
Application configuration settings.

This module provides a centralized configuration system using environment variables
with sensible defaults. It uses pydantic-settings for validation and type conversion.
"""
import os
import logging
from typing import List, Optional, Any, Dict
from enum import Enum

from pydantic import Field, field_validator, computed_field, ConfigDict
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format options."""
    SIMPLE = "simple"
    JSON = "json"
    DETAILED = "detailed"


class Settings(BaseSettings):
    """Application settings with modern Pydantic v2 configuration."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    # Application settings
    APP_NAME: str = "BitNet SME Expert System"
    API_VERSION: str = "2.0.0"
    API_PREFIX: str = "/api/v1"
    DESCRIPTION: str = """
    Advanced BitNet SME Expert System with multi-model AI integration.
    Provides specialized AI experts for mathematics, code generation, and general knowledge.
    """

    # Environment
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    DEBUG: bool = Field(default=False)

    # Security
    SECRET_KEY: str = Field(
        default="change-this-to-a-secure-random-string-in-production",
        description="Secret key for cryptographic operations"
    )

    # Server settings
    API_HOST: str = Field(default="0.0.0.0", alias="HOST")
    API_PORT: int = Field(default=8000, alias="PORT")
    API_WORKERS: int = Field(default=1, alias="WORKERS")
    API_RELOAD: bool = Field(default=True, alias="RELOAD")

    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    ALLOW_CREDENTIALS: bool = True
    ALLOWED_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    ALLOWED_HEADERS: List[str] = Field(default=["*"])

    # Logging configuration
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: LogFormat = LogFormat.JSON
    ENABLE_STRUCTURED_LOGGING: bool = True

    # API Documentation
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"

    # Expert system settings
    DEFAULT_MODEL: str = "gpt-4o-mini"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0

    # Rate limiting
    RATE_LIMIT: str = "100/minute"
    ENABLE_RATE_LIMITING: bool = True

    # Caching
    CACHE_TTL: int = 300  # 5 minutes
    ENABLE_CACHING: bool = True

    # Database settings
    DATABASE_URL: Optional[str] = Field(default=None)
    DATABASE_ECHO: bool = Field(default=False)
    DATABASE_POOL_SIZE: int = Field(default=5)
    DATABASE_MAX_OVERFLOW: int = Field(default=10)
    DATABASE_POOL_PRE_PING: bool = Field(default=True)

    # Redis settings
    REDIS_URL: Optional[str] = Field(default=None)
    REDIS_DECODE_RESPONSES: bool = True
    REDIS_MAX_CONNECTIONS: int = 10

    # AI Service API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google AI API key")

    # Model configuration per expert
    MATH_EXPERT_MODEL: str = "gpt-4o-mini"
    CODE_EXPERT_MODEL: str = "claude-3-5-haiku-20241022"
    GENERAL_EXPERT_MODEL: str = "gpt-4o-mini"

    # Performance settings
    REQUEST_TIMEOUT: int = 30
    MAX_CONCURRENT_REQUESTS: int = 10

    # Monitoring and observability
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    ENABLE_TRACING: bool = False
    SLO_ERROR_RATE_THRESHOLD: float = 0.01
    SLO_P95_LATENCY_MS_THRESHOLD: int = 750
    SLO_ALERT_WINDOW_MINUTES: int = 5

    # Security settings
    ENABLE_AUTHENTICATION: bool = False
    JWT_SECRET_KEY: str = Field(default="jwt-secret-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Health check settings
    HEALTH_CHECK_INTERVAL: int = 30

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate and convert environment value."""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v or Environment.DEVELOPMENT

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Any) -> Optional[str]:
        """Build database URL from components if not provided directly."""
        if isinstance(v, str) and v:
            return v

        # Build from environment variables
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "bitnet_sme")

        if all([user, password, host, port, database]):
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

        # Default to SQLite for development
        return "sqlite:///./bitnet_sme.db"

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT

    @computed_field
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == Environment.TESTING

    @computed_field
    @property
    def api_info(self) -> Dict[str, Any]:
        """Get API metadata for OpenAPI specification."""
        return {
            "title": self.APP_NAME,
            "description": self.DESCRIPTION,
            "version": self.API_VERSION,
            "contact": {
                "name": "SUM Equities",
                "email": "hi@sumequities.com",
                "url": "https://sumequities.com"
            },
            "license_info": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        }

    @computed_field
    @property
    def log_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        if self.LOG_FORMAT == LogFormat.JSON:
            return {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "json": {
                        "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                        "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "formatter": "json",
                        "stream": "ext://sys.stdout"
                    }
                },
                "root": {
                    "level": self.LOG_LEVEL.value,
                    "handlers": ["console"]
                }
            }
        else:
            return {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "formatter": "default",
                        "stream": "ext://sys.stdout"
                    }
                },
                "root": {
                    "level": self.LOG_LEVEL.value,
                    "handlers": ["console"]
                }
            }


# Create global settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.value),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" if not settings.ENABLE_STRUCTURED_LOGGING else None
)

# Suppress noisy third-party loggers
noisy_loggers = [
    "httpx", "httpcore", "openai", "anthropic", "google.generativeai",
    "asyncio", "urllib3", "requests", "transformers"
]

for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Create application logger
logger = logging.getLogger("bitnet_sme")
logger.info(f"Settings loaded for {settings.ENVIRONMENT.value} environment")

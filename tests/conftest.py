"""
Pytest configuration and fixtures for BitNet SME Expert System tests.
"""
import asyncio
import os
import pytest
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set test environment before importing app
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.main import app
from app.config import settings
from app.database import get_db


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI application."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_db():
    """Mock database session."""
    return MagicMock()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock = AsyncMock()
    mock.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="Mocked response"))]
    ))
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    mock = AsyncMock()
    mock.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text="Mocked response")]
    ))
    return mock


@pytest.fixture
def mock_google_client():
    """Mock Google AI client."""
    mock = AsyncMock()
    mock.generate_content = AsyncMock(return_value=MagicMock(
        text="Mocked response"
    ))
    return mock


@pytest.fixture
def sample_math_query():
    """Sample math query for testing."""
    return {
        "question": "What is the derivative of x^2 + 3x + 2?",
        "domain": "math",
        "context": {}
    }


@pytest.fixture
def sample_code_query():
    """Sample code query for testing."""
    return {
        "question": "Write a Python function to calculate fibonacci numbers",
        "domain": "code",
        "context": {"language": "python"}
    }


@pytest.fixture
def sample_general_query():
    """Sample general query for testing."""
    return {
        "question": "What is the capital of France?",
        "domain": "general",
        "context": {}
    }


@pytest.fixture
def mock_expert_config():
    """Mock expert configuration."""
    return {
        "name": "Test Expert",
        "description": "Expert for testing",
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        "ENVIRONMENT": "testing",
        "DATABASE_URL": "sqlite:///:memory:",
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "GOOGLE_API_KEY": "test-key",
        "ENABLE_RATE_LIMITING": "false",
        "ENABLE_CACHING": "false"
    }

    # Store original values
    original_values = {}
    for key, value in test_env.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original values
    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.exists = AsyncMock(return_value=False)
    return mock


class MockResponse:
    """Mock response object for testing."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    @property
    def text(self):
        return str(self.json_data)


@pytest.fixture
def mock_response():
    """Factory for creating mock responses."""
    return MockResponse


# Test data fixtures
@pytest.fixture
def test_expert_data():
    """Test data for experts."""
    return {
        "math": {
            "name": "Math Expert",
            "description": "Expert in mathematics",
            "model_name": "gpt-4o-mini",
            "temperature": 0.3
        },
        "code": {
            "name": "Code Expert",
            "description": "Expert in programming",
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.2
        },
        "general": {
            "name": "General Expert",
            "description": "General knowledge expert",
            "model_name": "gpt-4o-mini",
            "temperature": 0.7
        }
    }


@pytest.fixture
def test_queries():
    """Test queries for different domains."""
    return {
        "math": [
            "What is 2 + 2?",
            "Solve for x: 2x + 5 = 15",
            "What is the derivative of x^3?",
            "Calculate the area of a circle with radius 5"
        ],
        "code": [
            "Write a hello world program in Python",
            "Create a function to reverse a string",
            "Implement a binary search algorithm",
            "Write a REST API endpoint in FastAPI"
        ],
        "general": [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is photosynthesis?",
            "Explain quantum computing in simple terms"
        ]
    }
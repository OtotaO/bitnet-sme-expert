"""Tests for expert implementations."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from app.experts.math_expert import MathExpert
from app.experts.code_expert import CodeExpert
from app.experts.general_expert import GeneralExpert
from app.schemas.base import ExpertDomain

# Test data
MATH_QUESTIONS = [
    "What is 2 + 2?",
    "Calculate the derivative of x^2",
    "What is the square root of 144?",
]

CODE_QUESTIONS = [
    "Write a Python function to calculate factorial",
    "How do I sort a list in JavaScript?",
    "Explain this code: def add(a, b): return a + b",
]

GENERAL_QUESTIONS = [
    "What is the capital of France?",
    "Tell me about quantum computing",
    "Who is the president of the United States?",
]

@pytest.fixture
def math_expert():
    """Create a MathExpert instance for testing."""
    return MathExpert({
        "name": "Test Math Expert",
        "description": "Test math expert",
        "domain": ExpertDomain.MATH,
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.3,
    })

@pytest.fixture
def code_expert():
    """Create a CodeExpert instance for testing."""
    return CodeExpert({
        "name": "Test Code Expert",
        "description": "Test code expert",
        "domain": ExpertDomain.CODE,
        "model_name": "gpt-4",
        "temperature": 0.5,
    })

@pytest.fixture
def general_expert():
    """Create a GeneralExpert instance for testing."""
    return GeneralExpert({
        "name": "Test General Expert",
        "description": "Test general expert",
        "domain": ExpertDomain.GENERAL,
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
    })

@pytest.mark.asyncio
async def test_math_expert_initialization(math_expert):
    """Test that the math expert initializes correctly."""
    assert math_expert.name == "Test Math Expert"
    assert math_expert.domain == ExpertDomain.MATH
    assert not math_expert.initialized
    
    # Initialize the expert
    await math_expert.initialize()
    assert math_expert.initialized

@pytest.mark.asyncio
async def test_code_expert_initialization(code_expert):
    """Test that the code expert initializes correctly."""
    assert code_expert.name == "Test Code Expert"
    assert code_expert.domain == ExpertDomain.CODE
    assert not code_expert.initialized
    
    # Initialize the expert
    await code_expert.initialize()
    assert code_expert.initialized

@pytest.mark.asyncio
async def test_general_expert_initialization(general_expert):
    """Test that the general expert initializes correctly."""
    assert general_expert.name == "Test General Expert"
    assert general_expert.domain == ExpertDomain.GENERAL
    assert not general_expert.initialized
    
    # Initialize the expert
    await general_expert.initialize()
    assert general_expert.initialized

@pytest.mark.asyncio
async def test_math_expert_queries(math_expert):
    """Test that the math expert can handle various math questions."""
    await math_expert.initialize()
    
    for question in MATH_QUESTIONS:
        response = await math_expert.generate(question, {})
        assert "response" in response
        assert "metadata" in response
        assert response["metadata"]["domain"] == ExpertDomain.MATH.value
        assert response["metadata"]["expert"] == "Test Math Expert"

@pytest.mark.asyncio
async def test_code_expert_queries(code_expert):
    """Test that the code expert can handle various code-related questions."""
    await code_expert.initialize()
    
    for question in CODE_QUESTIONS:
        response = await code_expert.generate(question, {})
        assert "response" in response
        assert "metadata" in response
        assert response["metadata"]["domain"] == ExpertDomain.CODE.value
        assert response["metadata"]["expert"] == "Test Code Expert"

@pytest.mark.asyncio
async def test_general_expert_queries(general_expert):
    """Test that the general expert can handle various general knowledge questions."""
    await general_expert.initialize()
    
    for question in GENERAL_QUESTIONS:
        response = await general_expert.generate(question, {})
        assert "response" in response
        assert "metadata" in response
        assert response["metadata"]["domain"] == ExpertDomain.GENERAL.value
        assert response["metadata"]["expert"] == "Test General Expert"

@pytest.mark.asyncio
async def test_math_expert_specific_operations(math_expert):
    """Test specific math operations with the math expert."""
    await math_expert.initialize()
    
    # Test arithmetic
    response = await math_expert.generate("What is 5 * 8?", {})
    assert "40" in response["response"]
    
    # Test equation solving
    response = await math_expert.generate("Solve for x: 2x + 3 = 7", {})
    assert "x = 2" in response["response"]
    
    # Test calculus
    response = await math_expert.generate("What is the derivative of x^2?", {})
    assert "2x" in response["response"]

@pytest.mark.asyncio
async def test_code_expert_specific_operations(code_expert):
    """Test specific code operations with the code expert."""
    await code_expert.initialize()
    
    # Test code generation
    response = await code_expert.generate("Generate a Python function to calculate factorial", {})
    assert "def factorial" in response["response"]
    
    # Test code explanation
    response = await code_expert.generate("Explain this code: def add(a, b): return a + b", {})
    assert "add" in response["response"]
    assert "function" in response["response"]
    
    # Test code debugging
    response = await code_expert.generate("Fix this code: for i in range(5) print(i)", {})
    assert ":" in response["response"]  # Should add the missing colon

@pytest.mark.asyncio
async def test_general_expert_knowledge(general_expert):
    """Test the general knowledge of the general expert."""
    await general_expert.initialize()
    
    # Test factual knowledge
    response = await general_expert.generate("What is the capital of France?", {})
    assert "Paris" in response["response"]
    
    # Test explanation
    response = await general_expert.generate("Explain quantum computing", {})
    assert "quantum" in response["response"]
    assert "comput" in response["response"]
    
    # Test opinion
    response = await general_expert.generate("What do you think about artificial intelligence?", {})
    assert "AI" in response["response"] or "artificial intelligence" in response["response"]

if __name__ == "__main__":
    # Run the tests
    import sys
    import pytest
    sys.exit(pytest.main(["-v", __file__]))

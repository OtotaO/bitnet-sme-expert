"""Example script demonstrating how to use the expert implementations."""

import asyncio
import json
from datetime import datetime

# Add the project root to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.experts.math_expert import MathExpert
from app.experts.code_expert import CodeExpert
from app.experts.general_expert import GeneralExpert

async def main():
    """Run the example script."""
    print("BitNet SME Expert System Demo")
    print("=" * 50)
    
    # Initialize the experts
    print("\nInitializing experts...")
    math_expert = MathExpert({
        "name": "Math Expert",
        "description": "Expert in mathematical problem solving",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.3,
    })
    
    code_expert = CodeExpert({
        "name": "Code Expert",
        "description": "Expert in programming and software development",
        "model_name": "gpt-4",
        "temperature": 0.5,
    })
    
    general_expert = GeneralExpert({
        "name": "General Expert",
        "description": "Expert in general knowledge and information",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
    })
    
    # Initialize the experts
    await asyncio.gather(
        math_expert.initialize(),
        code_expert.initialize(),
        general_expert.initialize()
    )
    
    print("\nExperts initialized and ready!")
    
    # Example questions
    math_questions = [
        "What is 2 + 2?",
        "Calculate the derivative of x^2",
        "What is the square root of 144?",
        "Solve for x: 3x + 5 = 20",
        "What is the integral of 1/x?"
    ]
    
    code_questions = [
        "Write a Python function to calculate factorial",
        "How do I sort a list in JavaScript?",
        "Explain this code: def add(a, b): return a + b",
        "What's wrong with this code: for i in range(5) print(i)",
        "Convert this Python function to JavaScript: def greet(name): return f'Hello, {name}!"
    ]
    
    general_questions = [
        "What is the capital of France?",
        "Tell me about quantum computing",
        "Who is the president of the United States?",
        "What are the main causes of climate change?",
        "Explain how a blockchain works"
    ]
    
    # Test the math expert
    print("\nTesting Math Expert:")
    print("-" * 50)
    for question in math_questions:
        print(f"\nQuestion: {question}")
        response = await math_expert.generate(question, {})
        print(f"Response: {response['response']}")
        print(f"(Confidence: {response['metadata'].get('confidence', 'N/A')})")
    
    # Test the code expert
    print("\nTesting Code Expert:")
    print("-" * 50)
    for question in code_questions:
        print(f"\nQuestion: {question}")
        response = await code_expert.generate(question, {})
        print(f"Response: {response['response']}")
        print(f"(Confidence: {response['metadata'].get('confidence', 'N/A')})")
    
    # Test the general expert
    print("\nTesting General Expert:")
    print("-" * 50)
    for question in general_questions:
        print(f"\nQuestion: {question}")
        response = await general_expert.generate(question, {})
        print(f"Response: {response['response']}")
        print(f"(Confidence: {response['metadata'].get('confidence', 'N/A')})")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    asyncio.run(main())

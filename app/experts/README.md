# Expert Implementations

This directory contains the implementation of various expert modules for the BitNet SME system. Each expert is designed to handle a specific domain of knowledge or type of query.

## Available Experts

### 1. Math Expert (`math_expert.py`)
- **Domain**: Mathematical problem solving and analysis
- **Capabilities**:
  - Solve equations and expressions
  - Perform calculus operations (derivatives, integrals, limits)
  - Simplify and factor expressions
  - Calculate series expansions
  - Solve linear algebra problems
  - Handle statistical calculations
  - Generate step-by-step solutions

### 2. Code Expert (`code_expert.py`)
- **Domain**: Programming and software development
- **Capabilities**:
  - Generate code in multiple programming languages
  - Explain and document code
  - Debug and fix issues
  - Optimize code for performance
  - Refactor code for better quality
  - Write unit tests
  - Convert code between languages
  - Provide programming best practices

### 3. General Expert (`general_expert.py`)
- **Domain**: General knowledge and information
- **Capabilities**:
  - Answer factual questions
  - Provide explanations on various topics
  - Handle general knowledge queries
  - Offer insights and summaries
  - Respond to conversational prompts

## Base Implementation

The `base_expert.py` file contains the `BaseExpertImpl` class that all expert implementations inherit from. It provides common functionality including:

- Configuration management
- Logging
- Error handling
- Response formatting
- Common utility methods

## Adding a New Expert

To add a new expert:

1. Create a new Python file in this directory
2. Create a class that inherits from `BaseExpertImpl`
3. Implement the required methods, especially `_generate_impl`
4. Define the `DOMAIN` class variable with the appropriate `ExpertDomain`
5. Add any domain-specific methods and logic
6. Update this README to document the new expert

## Usage Example

```python
from .math_expert import MathExpert

# Initialize the expert
expert = MathExpert()

# Generate a response
response = await expert.generate(
    "What is the derivative of x^2?",
    context={"user_id": "123"}
)

print(response["response"])
```

## Dependencies

- Python 3.8+
- sympy (for MathExpert)
- Other dependencies as specified in the project's requirements.txt

## Testing

Run the test suite to verify all experts are functioning correctly:

```bash
pytest tests/experts/
```

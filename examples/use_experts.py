"""Smoke-test the DSPy-backed experts from the command line.

Requires at least one provider API key (e.g. ``OPENAI_API_KEY``) — or override
``DSPY_LM_*`` to point at a local server such as ``bitnet.cpp``.

Run with::

    uv run python examples/use_experts.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.experts import CodeExpert, GeneralExpert, MathExpert
from app.llm import configure_dspy


async def main() -> None:
    print("dspy-sme-expert demo")
    print("=" * 50)
    configure_dspy(enable_mlflow=False)

    math_expert = MathExpert({"name": "Math Expert"})
    code_expert = CodeExpert({"name": "Code Expert"})
    general_expert = GeneralExpert({"name": "General Expert"})

    await asyncio.gather(
        math_expert.initialize(),
        code_expert.initialize(),
        general_expert.initialize(),
    )

    queries: list[tuple[str, object, dict]] = [
        ("What is 5 * 8?", math_expert, {}),
        (
            "Write a Python function that returns the n-th Fibonacci number.",
            code_expert,
            {"language": "python"},
        ),
        ("What is the capital of France?", general_expert, {}),
    ]

    for question, expert, context in queries:
        print(f"\n[{expert.config.name}] {question}")
        result = await expert.generate(question, context=context)
        print(f"  -> {result['response']}")
        print(
            f"     ({result['metadata'].get('model')}, {result['metadata'].get('processing_time', 0.0):.2f}s)"
        )


if __name__ == "__main__":
    asyncio.run(main())

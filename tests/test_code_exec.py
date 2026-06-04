"""Tests for the execution-based code grader (tests/eval/exec.py).

These run *known-safe* code, so they execute unconditionally (no LM, no network)
and prove the harness grades behavior correctly: correct code passes, wrong code
fails, and a runaway loop is killed by the timeout rather than hanging CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tests.eval.exec import code_exec_metric, run_code_test

_FACTORIAL = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"


def test_correct_code_passes() -> None:
    assert run_code_test(_FACTORIAL, "assert factorial(5) == 120\nassert factorial(0) == 1")


def test_wrong_code_fails() -> None:
    assert not run_code_test("def factorial(n):\n    return 0", "assert factorial(5) == 120")


def test_exception_fails() -> None:
    assert not run_code_test("def factorial(n):\n    raise ValueError('boom')", "factorial(3)")


def test_timeout_is_killed() -> None:
    # An infinite loop must be bounded by the timeout, not hang the test run.
    assert not run_code_test("def f():\n    while True:\n        pass", "f()", timeout=2.0)


def test_metric_uses_code_and_test_fields() -> None:
    example = SimpleNamespace(test="assert factorial(4) == 24")
    good = SimpleNamespace(code=_FACTORIAL)
    bad = SimpleNamespace(code="def factorial(n):\n    return n")
    assert code_exec_metric(example, good) == 1.0
    assert code_exec_metric(example, bad) == 0.0
    # Missing fields grade as 0, never raise.
    assert code_exec_metric(SimpleNamespace(), SimpleNamespace()) == 0.0


def test_seed_dataset_is_self_consistent() -> None:
    # Every committed code_exec item must have a runnable test, and a correct
    # reference solution must pass it — guards against a poisoned gold test.
    path = Path(__file__).parent / "eval" / "datasets" / "code_exec.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 10
    references = {
        "factorial": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
        "is_prime": "def is_prime(n):\n    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))",
        "fib": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "reverse_string": "def reverse_string(s):\n    return s[::-1]",
        "gcd": "import math\ndef gcd(a, b):\n    return math.gcd(a, b)",
        "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]",
        "sum_list": "def sum_list(xs):\n    return sum(xs)",
        "count_vowels": "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s)",
        "flatten": "def flatten(nested):\n    return [x for sub in nested for x in sub]",
        "to_snake_case": "import re\ndef to_snake_case(s):\n    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()",
        "merge_sorted": "import heapq\ndef merge_sorted(a, b):\n    return list(heapq.merge(a, b))",
        "binary_search": (
            "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n"
            "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
            "        if xs[mid] == target:\n            return mid\n"
            "        if xs[mid] < target:\n            lo = mid + 1\n"
            "        else:\n            hi = mid - 1\n    return -1"
        ),
    }
    for row in rows:
        assert row["test"].strip(), row["request"]
        fn = next((name for name in references if name + "(" in row["test"]), None)
        assert fn is not None, f"no reference for: {row['request']}"
        assert run_code_test(references[fn], row["test"]), f"reference failed its own test: {fn}"

#!/usr/bin/env python
"""Compile a DSPy program with MIPROv2 or GEPA against a gold dataset.

Usage::

    python scripts/optimize.py --domain math --optimizer miprov2 --auto light
    python scripts/optimize.py --domain code --optimizer gepa --reflection-lm openai/gpt-5

Optimized programs are written to ``compiled/<domain>.json`` and loaded at app
startup by ``app/experts/base_expert.py``.

The metric callables and gold sets are reused from ``tests/eval``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dspy

from app.dspy_modules import CodeProgram, GeneralProgram, MathProgram
from app.llm import configure_dspy, get_lm
from tests.eval.test_eval import code_metric, general_metric, math_metric

COMPILED_DIR = ROOT / "compiled"
COMPILED_DIR.mkdir(exist_ok=True)

DATA_DIR = ROOT / "tests" / "eval" / "datasets"


def _load(name: str, *, inputs: tuple[str, ...]) -> list[dspy.Example]:
    rows = [json.loads(line) for line in (DATA_DIR / name).read_text().splitlines() if line.strip()]
    return [dspy.Example(**row).with_inputs(*inputs) for row in rows]


DOMAINS: dict[str, dict] = {
    "math": {
        "program": MathProgram,
        "metric": math_metric,
        "dataset": ("math.jsonl", ("question",)),
    },
    "code": {
        "program": CodeProgram,
        "metric": code_metric,
        "dataset": ("code.jsonl", ("request", "language")),
    },
    "general": {
        "program": GeneralProgram,
        "metric": general_metric,
        "dataset": ("general.jsonl", ("question",)),
    },
}


def _baseline(program: dspy.Module, devset: list[dspy.Example], metric) -> float:
    evaluator = dspy.Evaluate(devset=devset, num_threads=4, display_progress=True)
    return evaluator(program, metric=metric)


def _optimize_miprov2(program, devset, metric, auto: str):
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(metric=metric, auto=auto, num_threads=4)
    return optimizer.compile(program, trainset=devset)


def _optimize_gepa(program, devset, metric, reflection_lm: str):
    from dspy.teleprompt import GEPA

    optimizer = GEPA(
        metric=metric,
        reflection_lm=dspy.LM(reflection_lm, max_tokens=2048),
        auto="light",
    )
    return optimizer.compile(program, trainset=devset)


def main() -> int:
    parser = argparse.ArgumentParser(description="DSPy program optimizer")
    parser.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--optimizer", choices=("miprov2", "gepa"), default="miprov2")
    parser.add_argument("--auto", choices=("light", "medium", "heavy"), default="light")
    parser.add_argument(
        "--reflection-lm",
        default="openai/gpt-4o",
        help="Stronger model used by GEPA for reflection",
    )
    args = parser.parse_args()

    configure_dspy(enable_mlflow=False)
    spec = DOMAINS[args.domain]

    program_cls = spec["program"]
    metric = spec["metric"]
    name, inputs = spec["dataset"]
    devset = _load(name, inputs=inputs)

    print(f"\n=== {args.domain} / {args.optimizer} ===")
    print(f"Dataset size: {len(devset)}")
    print(f"Default LM: {getattr(get_lm(args.domain), 'model', 'unknown')}")

    baseline_program = program_cls()
    baseline_score = _baseline(baseline_program, devset, metric)
    print(f"Baseline score: {baseline_score:.4f}")

    if args.optimizer == "miprov2":
        compiled = _optimize_miprov2(baseline_program, devset, metric, args.auto)
    else:
        compiled = _optimize_gepa(baseline_program, devset, metric, args.reflection_lm)

    optimized_score = _baseline(compiled, devset, metric)
    print(f"Optimized score: {optimized_score:.4f}")
    print(f"Delta: {optimized_score - baseline_score:+.4f}")

    output = COMPILED_DIR / f"{args.domain}.json"
    compiled.save(str(output))
    print(f"Saved compiled program to {output}")

    summary = {
        "domain": args.domain,
        "optimizer": args.optimizer,
        "auto": args.auto,
        "baseline": baseline_score,
        "optimized": optimized_score,
        "delta": optimized_score - baseline_score,
        "n": len(devset),
    }
    (COMPILED_DIR / f"{args.domain}_metrics.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

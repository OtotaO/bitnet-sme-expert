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
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dspy

from app.dspy_modules import CodeProgram, GeneralProgram, MathProgram
from app.llm import configure_dspy, get_lm
from tests.eval.loader import load_split
from tests.eval.test_eval import code_metric, general_metric, math_metric, to_fraction

COMPILED_DIR = ROOT / "compiled"
COMPILED_DIR.mkdir(exist_ok=True)

# Compiled programs live in gitignored compiled/, but the metrics *receipt* is a
# committed proof artifact — a dated, held-out baseline-vs-compiled record.
RECEIPTS_DIR = ROOT / "eval" / "receipts"


DOMAINS: dict[str, dict] = {
    "math": {"program": MathProgram, "metric": math_metric},
    "code": {"program": CodeProgram, "metric": code_metric},
    "general": {"program": GeneralProgram, "metric": general_metric},
}


def _baseline(program: dspy.Module, devset: list[dspy.Example], metric) -> float:
    evaluator = dspy.Evaluate(devset=devset, num_threads=4, display_progress=True)
    return to_fraction(evaluator(program, metric=metric))


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

    # Pin temperature=0 so the baseline-vs-compiled delta is reproducible and a
    # win/loss isn't sampling noise. Honored by app/llm.py:_resolve_spec.
    for role in ("router", "math", "code", "general"):
        os.environ.setdefault(f"DSPY_LM_{role.upper()}_TEMPERATURE", "0.0")

    configure_dspy(enable_mlflow=False)
    spec = DOMAINS[args.domain]

    program_cls = spec["program"]
    metric = spec["metric"]

    # Train on `train`, measure on `holdout`. Scoring on data the optimizer
    # never saw is what makes the reported delta an honest generalization
    # number rather than a memorization artifact.
    trainset = load_split(args.domain, "train")
    holdout = load_split(args.domain, "holdout")

    print(f"\n=== {args.domain} / {args.optimizer} ===")
    print(f"Train size: {len(trainset)}  |  Holdout size: {len(holdout)}")
    print(f"Default LM: {getattr(get_lm(args.domain), 'model', 'unknown')}")

    baseline_program = program_cls()
    baseline_score = _baseline(baseline_program, holdout, metric)
    print(f"Baseline holdout score: {baseline_score:.4f}")

    if args.optimizer == "miprov2":
        compiled = _optimize_miprov2(baseline_program, trainset, metric, args.auto)
    else:
        compiled = _optimize_gepa(baseline_program, trainset, metric, args.reflection_lm)

    optimized_score = _baseline(compiled, holdout, metric)
    print(f"Optimized holdout score: {optimized_score:.4f}")
    print(f"Holdout delta: {optimized_score - baseline_score:+.4f}")

    output = COMPILED_DIR / f"{args.domain}.json"
    compiled.save(str(output))
    print(f"Saved compiled program to {output}")

    summary = {
        "domain": args.domain,
        "optimizer": args.optimizer,
        "auto": args.auto,
        "lm": getattr(get_lm(args.domain), "model", "unknown"),
        "split": "train->holdout",
        "baseline": baseline_score,
        "optimized": optimized_score,
        "delta": optimized_score - baseline_score,
        "n_train": len(trainset),
        "n_holdout": len(holdout),
    }
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt = RECEIPTS_DIR / f"{args.domain}-{args.optimizer}.json"
    receipt.write_text(json.dumps(summary, indent=2))
    print(f"Wrote receipt to {receipt.relative_to(ROOT)}")

    _log_to_mlflow(summary, receipt)
    return 0


def _log_to_mlflow(summary: dict, receipt: Path) -> None:
    """Log this optimizer run to MLflow when MLFLOW_TRACKING_URI is set (else no-op).

    Records params + baseline/optimized/delta metrics and attaches the committed
    receipt, so runs are A/B-comparable in the MLflow UI. Best-effort: a logging
    failure must not fail the compile.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "dspy-sme-expert-optimize"))
        with mlflow.start_run(run_name=f"{summary['domain']}-{summary['optimizer']}"):
            mlflow.log_params(
                {
                    k: summary[k]
                    for k in ("domain", "optimizer", "auto", "lm", "n_train", "n_holdout")
                }
            )
            mlflow.log_metrics({k: float(summary[k]) for k in ("baseline", "optimized", "delta")})
            mlflow.log_artifact(str(receipt))
        print("Logged optimizer run to MLflow.")
    except Exception as exc:
        print(f"MLflow logging skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    raise SystemExit(main())

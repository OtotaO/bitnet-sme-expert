"""Run ``dspy.Evaluate`` against the gold datasets and emit JSON results.

Designed to be invoked from CI (``.github/workflows/eval.yml``) with a domain
selector and a LiteLLM model string. Each domain's result is printed as one
JSON object on its own line on stdout, so the workflow can parse the output
without re-running pytest or scraping logs:

    {"domain": "math", "n": 12, "score": 0.83, "passed": true, "threshold": 0.6}

Exit status:
* 0 — every selected domain met its threshold
* 1 — one or more domains were below threshold (regression)
* 2 — invocation / setup error (bad domain name, missing dataset, LM error)

The script reuses the metrics defined in ``tests/eval/test_eval.py`` so the
in-CI score uses the same yardstick as the pytest gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import dspy

# Make `app.*` and `tests.eval.*` importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.dspy_modules import CodeProgram, GeneralProgram, MathProgram
from app.llm import configure_dspy
from tests.eval.test_eval import code_metric, general_metric, math_metric

logger = logging.getLogger("eval_runner")
DATA_DIR = REPO_ROOT / "tests" / "eval" / "datasets"
DEFAULT_THRESHOLDS = {"math": 0.6, "code": 0.6, "general": 0.7}


def _load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / name).read_text().splitlines() if line.strip()]


def _build(domain: str) -> tuple[dspy.Module, list[dspy.Example], Callable[..., float]]:
    if domain == "math":
        examples = [
            dspy.Example(**row).with_inputs("question") for row in _load_jsonl("math.jsonl")
        ]
        return MathProgram(), examples, math_metric
    if domain == "code":
        examples = [
            dspy.Example(**row).with_inputs("request", "language")
            for row in _load_jsonl("code.jsonl")
        ]
        return CodeProgram(), examples, code_metric
    if domain == "general":
        examples = [
            dspy.Example(**row).with_inputs("question") for row in _load_jsonl("general.jsonl")
        ]
        return GeneralProgram(), examples, general_metric
    raise ValueError(f"unknown domain: {domain}")


def _set_per_role_lm(domain: str, lm: str | None) -> None:
    """Override the role-specific LM via the same env vars ``app/llm.py`` reads."""
    if not lm:
        return
    role = domain.upper()
    os.environ[f"DSPY_LM_{role}"] = lm
    # Inherit base_url / api_key envs from the caller if set.


def run_domain(domain: str, threshold: float, num_threads: int) -> dict[str, object]:
    program, examples, metric = _build(domain)
    evaluator = dspy.Evaluate(devset=examples, num_threads=num_threads, display_progress=False)
    score = float(evaluator(program, metric=metric))
    return {
        "domain": domain,
        "n": len(examples),
        "score": round(score, 4),
        "threshold": threshold,
        "passed": score >= threshold,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=["math", "code", "general", "all"],
        default="all",
        help="Which domain to evaluate. 'all' runs every gold dataset.",
    )
    parser.add_argument(
        "--lm",
        default=None,
        help="LiteLLM model string applied to the selected domain(s) via DSPY_LM_<ROLE>.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the per-domain pass threshold (defaults: math/code=0.6, general=0.7).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to also write the full results array as JSON.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    domains = ["math", "code", "general"] if args.domain == "all" else [args.domain]
    for d in domains:
        _set_per_role_lm(d, args.lm)
    configure_dspy(enable_mlflow=bool(os.environ.get("MLFLOW_TRACKING_URI")))

    results: list[dict[str, object]] = []
    setup_failed = False
    for domain in domains:
        try:
            threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS[domain]
            result = run_domain(domain, threshold=threshold, num_threads=args.num_threads)
        except Exception as exc:
            result = {
                "domain": domain,
                "n": 0,
                "score": 0.0,
                "threshold": DEFAULT_THRESHOLDS[domain],
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            setup_failed = True
        results.append(result)
        print(json.dumps(result), flush=True)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))

    if setup_failed:
        return 2
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

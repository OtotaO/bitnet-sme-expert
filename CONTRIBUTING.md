# Contributing

Thanks for considering a contribution. This project's value is its measurement
discipline (see [`docs/strategy.md`](./docs/strategy.md)), so the contribution
rules are mostly about keeping that honest.

## Setup

```bash
git clone https://github.com/OtotaO/bitnet-sme-expert.git
cd bitnet-sme-expert
make install        # uv sync --all-extras + pre-commit hooks
```

Requires `uv` and Python 3.12+.

## The loop

```bash
make test           # unit tests (no LM calls)
make lint           # ruff check + ruff format --check + mypy
make format         # auto-fix + format
make check          # lint + test
```

All of `lint`, `test`, `build-container`, `security`, and `eval` run in CI and
are required to merge.

## Changing model behavior → re-run the eval

If you touch a `dspy.Module`, a signature, the router, or anything that changes
what an expert produces, you must check it against the gold sets:

```bash
RUN_EVAL=1 OPENAI_API_KEY=... uv run pytest tests/eval -v -s
# or, the CI-equivalent runner that scores on the holdout split:
uv run python scripts/run_eval.py --domain all --lm openai/gpt-4o-mini
```

The eval is scored on the **holdout** split (`tests/eval/datasets/*.holdout.jsonl`);
the optimizer only ever trains on `*.train.jsonl`. This split is committed on
purpose so results are reproducible — don't shuffle it at runtime.

## Optimizing a program → commit the receipt

```bash
make optimize-math         # MIPROv2 light  (trains on train, scores on holdout)
make optimize-math-gepa    # GEPA
```

This writes `eval/receipts/<domain>-<optimizer>.json`. **Commit it as-is —
including a non-positive delta.** A recorded "no improvement" is a real result;
a massaged positive one is not. See [`eval/receipts/README.md`](./eval/receipts/README.md).

## Adding a domain expert

See [`app/experts/README.md`](./app/experts/README.md) for the end-to-end
recipe (signature → program → expert → LM spec → registration → gold split +
metric). A new capability lands with a gold split and a metric, or it doesn't
land.

## Truthfulness in docs

Public claims must match the code. If a feature is designed-for but not yet
wired (e.g. a reserved config field), say so explicitly — use the
"current state + future" framing the README already uses. Don't assert a
capability the code doesn't back.

## Commits & PRs

- Keep PRs scoped; explain *why*, not just *what*.
- Don't commit generated artifacts except committed eval receipts.
- CI must be green before merge.

# AGENTS.md — orientation for AI/agent sessions

Read this first, then [`docs/strategy.md`](docs/strategy.md). It exists so a fresh
session can continue this project's work without re-deriving its context.

> Repo is `bitnet-sme-expert` (historical name); the package / CLI / import path
> are `dspy-sme-expert` / `dspy_sme`. Same project.

## What this is

A FastAPI service that routes questions to per-domain `dspy.Module` experts
(math = ReAct + sympy, code/general = ChainOfThought) over LiteLLM, regression-
gated in CI against a committed gold set, with an optional 1-bit `bitnet.cpp`
local provider and an MCP server. The thesis and roadmap are in
[`docs/strategy.md`](docs/strategy.md); the moat is **corpus + held-out eval +
receipts**, not any single model.

## Where things live (read in this order)

| To understand… | Read |
| --- | --- |
| Strategy, goals, definition-of-done, honest current state | `docs/strategy.md` |
| Dev loop, the change→re-eval→commit-receipt cycle | `CONTRIBUTING.md` |
| How to add a new domain expert end-to-end | `app/experts/README.md` |
| The eval gate, split, and receipts policy | `README.md` (“Eval gate & receipts”), `eval/receipts/README.md` |
| BitNet + DSPy specialization playbook | `docs/specialization.md` |
| Open follow-up work (the live TODO) | **GitHub issue #19** |

## Non-negotiable invariants (don't violate these)

1. **Truthfulness.** Every public claim (README/docs/comments) must match the
   code. If a feature is designed-for but not wired, say so explicitly. No
   aspirational claims stated as current fact.
2. **Receipts are committed as-is — including losses.** `eval/receipts/*.json`
   records optimizer results; a non-positive delta gets committed, never
   massaged. (The math receipt was −0.13 on a tiny set before the honest +0.18
   on the real split — that history is the point.)
3. **A new capability lands with a gold split + a metric, or it doesn't land.**
   No feature outruns its eval.
4. **The eval is a statistical estimate, not an exact number.** Hosted-LLM output
   isn't bitwise-reproducible; eval is pinned to `temperature=0` and reports a
   95% Wilson interval. Don't trust a single run as ground truth.
5. **Authorization is per-route dependencies** (`app/auth.py`), never path-string
   middleware. **JWT** stays algorithm-pinned with `require=["exp"]`.
6. **Never run untrusted code unsandboxed.** Serving uses the opt-in Deno/Pyodide
   sandbox (`app/dspy_modules/code_module.py`); eval uses the timeout-bounded
   subprocess runner (`tests/eval/exec.py`). The sympy math tools are DoS-guarded.
7. **Commit/PR conventions:** branch off `main` (never commit to it directly);
   keep CI green (`lint`, `test`, `build-container`, `security`, `eval`); end
   commits with the `Co-Authored-By` trailer.

## Dev loop

```bash
make install            # uv sync --all-extras + pre-commit
make check              # lint (ruff + mypy) + unit tests (no LM)
make test-eval          # RUN_EVAL=1 — needs an OPENAI_API_KEY
uv run python scripts/run_eval.py --domain all --lm openai/gpt-4o-mini   # the CI gate, scored on holdout
make optimize-math      # MIPROv2 light → writes eval/receipts/math-miprov2.json
uv run dspy-sme-mcp     # MCP server over stdio ([mcp] extra)
```

Python 3.12+, `uv`-managed, single `pyproject.toml` + committed `uv.lock`.
DSPy is pinned `>=3.2.1` (current stable; 3.3 is beta — don't adopt in CI).

## Current state (2026-06-04) & next work

Done and on `main`: the eval gate (50-item holdouts, temp-0, Wilson CIs); a real
+0.18 MIPROv2 math win; per-route auth + JWT hardening; persisted feedback;
MLflow optimizer logging; rate-limited paid endpoints; an MCP server; and an
execution-based code-eval harness (foundation).

Open (tracked in **issue #19**), in rough priority:
- Flip the **code gate onto execution grading** (`tests/eval/exec.py`) — needs an
  opt-in CI exec job + expanding `tests/eval/datasets/code_exec.jsonl`.
- **LLM-as-judge for `general`** (the substring metric still saturates at 1.00).
- Gate on the **Wilson lower bound** rather than the point estimate (needs larger N).
- Deeper **MCP** (HTTP/streamable transport, mounting), `make bitnet-demo` on
  real hardware, `LMSpec.fallbacks` wiring.
- Prefer **GEPA** over MIPROv2 as gold sets grow (ICLR-2026 SOTA).

To enforce the `eval` required check, a maintainer runs `branch-protection.yml`
(it mutates repo settings + needs `BRANCH_PROTECTION_TOKEN`).

## Key file map

```
app/main.py            FastAPI app + lifespan        app/bootstrap.py   shared expert-service construction
app/auth.py            per-route auth (require_role) app/limiter.py     shared SlowAPI limiter
app/llm.py             per-role LM routing (LiteLLM) app/mcp_server.py  MCP server (dspy-sme-mcp)
app/dspy_modules/      signatures + Modules          app/experts/       API-contract wrappers
scripts/run_eval.py    the CI eval gate (holdout)    scripts/optimize.py MIPROv2/GEPA + receipts
tests/eval/loader.py   committed train/holdout split tests/eval/exec.py  execution-based code grader
eval/receipts/         committed optimizer receipts
```

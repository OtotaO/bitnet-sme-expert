# dspy-sme-expert

**DSPy-powered multi-expert SME system.** A FastAPI service that routes
questions to domain-specialist `dspy.Module`s (math, code, general), with
LiteLLM provider routing, MLflow tracing, optional local inference via
`bitnet.cpp`, and a `dspy.Evaluate` harness driving MIPROv2 / GEPA
optimization.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![DSPy](https://img.shields.io/badge/DSPy-3.2-orange)](https://dspy.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Previously branded "BitNet SME Expert v2.0". Renamed in v3.0 because the
> previous incarnation was a multi-provider FastAPI shell with hardcoded
> string stubs in place of every expert, and zero DSPy or BitNet code. v3
> rebuilds the expert layer on DSPy 3.2 and adds `bitnet.cpp` as an
> _optional_ local provider rather than a marketing centerpiece.

---

## What changed in v3.0

| Layer | v2 | v3 |
| --- | --- | --- |
| Expert layer | Hardcoded canned-string stubs | `dspy.Module`s with typed signatures |
| Math | Regex routing + sympy fallback | `dspy.ReAct(SolveMathProblem, tools=[sympy_*])` + deterministic fast-path |
| Code | Pattern matching + template strings | `dspy.ChainOfThought(GenerateCode)` |
| General | Random pick from canned responses | `dspy.ChainOfThought(AnswerGeneralQuestion)` |
| Routing | Hardcoded keyword `if/elif` | `RouterProgram` (`dspy.ChainOfThought`) — optimizable |
| Multi-provider | Three SDKs imported, none called | `dspy.LM` over LiteLLM, per-role fallback chains |
| Optimization | n/a | `MIPROv2` / `GEPA` via `scripts/optimize.py`, persisted to `compiled/` |
| Observability | structlog + Prometheus | + `mlflow.dspy.autolog()` (OTel traces, optimizer runs) |
| Package mgmt | `requirements.txt` + `pyproject.toml` (drift) | `uv` + single `pyproject.toml` + `uv.lock` |
| Python | 3.9+ | 3.12+ |
| Pydantic | mixed v1/v2 | v2 throughout |
| JWT | `python-jose` (unmaintained) | `PyJWT` |
| BitNet | name only | optional `bitnet.cpp` provider via OpenAI-compatible llama-server |

## Architecture

```mermaid
graph TB
    A[Client] --> B[FastAPI<br/>app.main]
    B --> C[Auth + Logging + Rate-limit<br/>middleware]
    C --> D[ExpertService]
    D --> E[RouterProgram<br/>dspy.ChainOfThought]
    E --> F{domain}
    F -->|math| G[MathExpert<br/>dspy.ReAct + sympy tools]
    F -->|code| H[CodeExpert<br/>dspy.ChainOfThought]
    F -->|general| I[GeneralExpert<br/>dspy.ChainOfThought]
    G --> J[dspy.LM via LiteLLM]
    H --> J
    I --> J
    J -->|OpenAI/Anthropic/Gemini| K[Provider APIs]
    J -->|local bitnet.cpp| L[llama-server :8080]
    J -.->|autolog| M[MLflow<br/>traces + optimizer runs]
```

## Quickstart

```bash
# 1. Get uv if you don't already have it.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install everything.
uv sync --all-extras

# 3. Configure providers.
cp .env.example .env
$EDITOR .env   # at minimum set OPENAI_API_KEY (or override DSPY_LM_*)

# 4. Run.
uv run uvicorn app.main:app --reload
# -> http://localhost:8000/docs
```

One-shot query without spinning up the server:

```bash
uv run dspy-sme query "What is the derivative of x^3 + 2x?" --domain math
```

## Optimization loop

```bash
# Baseline + MIPROv2 light compile (saves to compiled/math.json)
make optimize-math

# Or use GEPA for reflection-based prompt evolution
make optimize-math-gepa
```

Compiled programs are loaded automatically at startup. To re-run the eval
harness against the compiled programs:

```bash
RUN_EVAL=1 uv run pytest tests/eval -v -s
```

## Eval gate & receipts

The gold sets ship as a committed **train/holdout split** so a reported score
can't be an overfit-to-the-eval-set artifact (`tests/eval/loader.py`):

| Domain | Train | Holdout | Pass threshold (holdout) |
| --- | ---: | ---: | ---: |
| math | 45 | 15 | 0.60 |
| code | 40 | 15 | 0.60 |
| general | 40 | 15 | 0.70 |

`scripts/optimize.py` compiles on `train` and scores on `holdout`, writing a
dated receipt to `eval/receipts/<domain>-<optimizer>.json` (baseline, compiled,
delta, LM, split sizes). The `Eval` workflow runs `scripts/run_eval.py` against
the holdout set on every PR to `main` as a required check (see
`.github/workflows/eval.yml`).

### Committed receipts (`openai/gpt-4o-mini`, 2026-06-04)

Baseline holdout scores (no compiled program loaded):

| Domain | N (holdout) | Score | Threshold | Pass |
| --- | ---: | ---: | ---: | :---: |
| math | 15 | 0.73 | 0.60 | ✅ |
| code | 15 | 1.00 | 0.60 | ✅ |
| general | 15 | 1.00 | 0.70 | ✅ |

MIPROv2 (`auto=light`) compile — math (`eval/receipts/math-miprov2.json`):

| Domain | Baseline | Compiled | Delta |
| --- | ---: | ---: | ---: |
| math | 0.73 | 0.60 | **−0.13** |

**Honest result: this compile did not beat the baseline.** MIPROv2 light on a
45-example train set with `gpt-4o-mini` regressed the held-out math score — the
deterministic arithmetic fast-path in `MathProgram` is already a strong
baseline, and the chosen instructions/demos hurt more than they helped on the
small holdout. The receipt is committed as-is (project policy: never massage a
non-positive delta). A win likely needs a larger gold set, `auto=medium`/GEPA,
or a stronger reflection LM. `code` and `general` were not compiled: their
baselines already saturate the holdout at 1.00, leaving no measurable headroom.

## Optional: local inference with `bitnet.cpp`

`bitnet.cpp` ships an OpenAI-compatible `llama-server` (built during its
`setup_env.py` cmake step). Wire it up as any other `dspy.LM`:

```bash
# One-time: clone, build, download the b1.58 2B 4T model (~3 GB)
make bitnet-setup

# Serve it locally on :8080
make bitnet-serve

# Tell DSPy to use it for the math expert
export DSPY_LM_MATH="openai/bitnet"
export DSPY_LM_MATH_API_BASE="http://localhost:8080/v1"
export DSPY_LM_MATH_API_KEY_ENV="BITNET_DUMMY_KEY"
export BITNET_DUMMY_KEY="local"
```

The 2B-4T BitNet model is a useful cheap fallback for PII-sensitive or offline
workloads. Throughput is hardware-dependent — measure it on your own box rather
than trusting a headline number:

```bash
make bitnet-demo   # sends a fixed prompt through the DSPy path, prints
                   # measured tok/s, writes eval/receipts/bitnet-<date>.txt
```

(Microsoft's published bitnet.cpp benchmarks put the b1.58-2B-4T model in the
~5-7 tok/s range on a typical x86 CPU core; this repo ships no measured figure
of its own until `make bitnet-demo` produces one.)

## Observability

Set `MLFLOW_TRACKING_URI` (or run `docker compose up mlflow`) to enable
`mlflow.dspy.autolog()`. Every module call produces an OpenTelemetry span;
every optimizer run is logged as an MLflow run with the baseline and
optimized scores, so you can A/B compiled artifacts in the UI.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET  | `/health`, `/health/live`, `/health/ready` | Probes |
| GET  | `/metrics` | Prometheus scrape endpoint |
| GET  | `/api/v1/experts` | List registered experts |
| POST | `/api/v1/query` | Route a question (auto or `domain=...`) |
| POST | `/api/v1/collaborate` | Fan out to several experts in parallel |
| POST | `/api/v1/feedback` | Submit feedback on a previous query |

Interactive docs at `/docs` (Swagger) and `/redoc`.

## Project layout

```
app/
├── main.py                  FastAPI app + lifespan
├── cli.py                   `dspy-sme` console script
├── llm.py                   DSPy LM configuration (LiteLLM + MLflow)
├── config.py                Settings (Pydantic v2)
├── observability.py         JSON logging + Prometheus metrics
├── dspy_modules/            Signatures + Modules (the model code)
│   ├── signatures.py
│   ├── router.py
│   ├── math_module.py
│   ├── code_module.py
│   └── general_module.py
├── experts/                 Thin wrappers exposing the API contract
├── services/expert_service.py
├── api/endpoints/           FastAPI routes
├── middleware/              CORS, auth (PyJWT), logging, errors
├── schemas/                 Pydantic v2 request/response models
└── ...
scripts/
├── optimize.py              MIPROv2 / GEPA compile pipeline
├── bitnet_setup.sh          Build bitnet.cpp + pull the model
└── bitnet_serve.sh          Run the llama-server
tests/
├── test_experts.py          Wiring smoke tests (no LM)
└── eval/                    dspy.Evaluate gold sets + metrics (RUN_EVAL=1)
compiled/                    Optimized programs land here (gitignored)
```

## Development

```bash
make install      # uv sync --all-extras
make test         # unit tests (no LM)
make test-eval    # eval tests (requires provider keys, RUN_EVAL=1)
make lint         # ruff + mypy
make format       # ruff format + fix
make build        # docker build (production target)
```

## License

MIT. See `LICENSE`.

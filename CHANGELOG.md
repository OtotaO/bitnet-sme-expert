# Changelog

All notable changes to this project are documented in this file. The format
is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Agentic Hybrid RAG for the GeneralExpert** (opt-in, behind `RAG_ENABLED=1`).
  When the env var is set, `GeneralProgram` switches from `dspy.ChainOfThought`
  to `dspy.ReAct` with a `retrieve(query)` tool backed by
  `LanceDBRetriever`. Stack:
    - **LanceDB** as the embedded vector + FTS store. Native hybrid search
      (dense ANN + BM25) with built-in RRF fusion, first-class multivector
      support for the ColBERT upgrade path.
    - **BGE-M3** as the embedder.
    - **BGE-Reranker-v2-m3** as the cross-encoder rerank stage.
  Pulled via the new `[rag]` optional extra. See `docs/rag.md` for the full
  recipe, env knobs, and the LanceDB-vs-DuckDB tradeoff table.
- **`scripts/build_rag_index.py`** — ingest a directory of `.md`/`.txt`/`.rst`
  files into the LanceDB store with paragraph-based chunking and BGE-M3
  embeddings.
- **Eval-on-demand CI workflow** (`.github/workflows/eval.yml`). Triggered via
  `gh workflow run eval.yml --field domain=<math|code|general|all> --field lm=<...>`,
  it runs `dspy.Evaluate` against the gold JSONL datasets, uploads
  `eval-results.json` + stdout as an artifact, and optionally posts a markdown
  results table to a PR via `--field pr_number=<n>`. Exit code 0 = all
  thresholds met, 1 = regression, 2 = setup error.
- **`scripts/run_eval.py`** — the runner the workflow invokes. Importable so
  the same metric definitions work in pytest (`tests/eval/test_eval.py`) and
  in CI without drift. Emits JSON-per-line per domain to stdout, plus a
  consolidated array via `--output`.

## [3.0.0] - 2026-05-11

### Changed (breaking)
- **Rename**: project is now `dspy-sme-expert`. The previous `bitnet-sme-expert`
  name implied a BitNet-centric system that did not exist.
- **Python**: minimum bumped to 3.12 (was 3.9). `datetime.utcnow()` removed
  throughout — all timestamps are now `datetime.now(UTC)`.
- **Pydantic v2** throughout. `@validator` -> `@field_validator`, `.dict()` ->
  `.model_dump()`, `class Config:` -> `model_config = ConfigDict(...)`.
- **JWT**: `python-jose` (unmaintained, multiple CVEs) replaced with `PyJWT`.
- **Package manager**: single source of truth in `pyproject.toml` with
  `uv.lock`. `requirements.txt` removed.
- **Expert layer**: every expert (`MathExpert`, `CodeExpert`, `GeneralExpert`)
  is now a thin wrapper around a `dspy.Module`. The previous canned-string
  stubs are gone.
- **API routing**: `/api/v1/query` now uses `RouterProgram`
  (`dspy.ChainOfThought("question -> domain, confidence")`) for auto-routing
  instead of hardcoded keyword matching. Explicit `domain=` still wins.

### Added
- **DSPy 3.2** stack: signatures, modules, optimizers.
- **LiteLLM routing** with per-role LM selection via `DSPY_LM_<ROLE>` env
  overrides. OpenAI / Anthropic / Gemini / local OpenAI-compatible servers
  all reachable through the same `dspy.LM` interface.
- **MathProgram** = `dspy.ReAct(SolveMathProblem, tools=[sympy_*])` with a
  deterministic sympy fast-path for trivial expressions (avoids LM round-trips
  on plain arithmetic).
- **CodeProgram** = `dspy.ChainOfThought(GenerateCode)` with language inference.
- **GeneralProgram** = `dspy.ChainOfThought(AnswerGeneralQuestion)`.
- **MLflow autolog** opt-in via `MLFLOW_TRACKING_URI` — traces every module
  call, logs optimizer runs.
- **Eval harness** under `tests/eval/` with gold JSONL datasets for each
  domain. Run with `RUN_EVAL=1`.
- **Optimizer script** at `scripts/optimize.py` — runs baseline -> MIPROv2 or
  GEPA -> persisted to `compiled/<domain>.json`.
- **bitnet.cpp integration** as an optional local provider. `scripts/
  bitnet_setup.sh` builds the included llama-server; configure DSPy to point
  at it with `DSPY_LM_*_API_BASE=http://localhost:8080/v1`.
- **`dspy-sme` CLI** with `serve`, `query`, `optimize` subcommands.
- **uv-based Dockerfile** (multi-stage) and CI workflow.

### Removed
- Hardcoded `BitNetSMEException` references in user-visible places (class
  retained for backwards-compatible imports).
- Broken `pipeline/integrated_training_pipeline` references that prevented
  the previous `app/main.py` from importing.
- `requirements.txt`, black, isort, safety, hadolint pre-commit hooks (all
  superseded by `ruff format` and `pip-audit`).

### Fixed
- `app/main.py` previously did not parse: stray docstring fragment at lines
  5-7, duplicated imports at 10-34, an unbounded `health_check` body that
  bled into orphan Google AI code. The file is now a clean ~150 lines.
- `app/middleware/__init__.py` imported three modules that did not exist.

## [2.0.0] - 2026-04-22

### Added
- Initial v2.0 baseline documented for multi-expert architecture and
  deployment workflows.

[Unreleased]: https://github.com/ototao/bitnet-sme-expert/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/ototao/bitnet-sme-expert/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/ototao/bitnet-sme-expert/releases/tag/v2.0.0
